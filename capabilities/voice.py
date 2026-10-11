"""Optional speech input and cancellable queued playback; imports start no devices."""
import asyncio
import codecs
import os
import queue
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave

from config import config
from core.execution import ExecutionContext, current
from logger import error, measure
from services.runtime import worker

_asr = worker("asr", config.ASR_PYTHON, timeout=120)
_asr_context = ExecutionContext()
_enabled = config.ENABLE_VOICE
_loaded = False
_loading = False
_start_lock = threading.Lock()
_generation = 0
_speech = queue.Queue()
_speech_lock = threading.Lock()
_speech_thread = None
_playback_cancel = threading.Event()
_speech_stop = threading.Event()


def _load(generation):
    global _loading, _loaded, _asr_context
    with _start_lock:
        if generation != _generation or not _enabled:
            return "语音已关闭"
        if ready():
            return "语音服务已就绪"
        if _loading:
            return "语音模型加载中"
        _loading = True
        _asr_context = ExecutionContext()
        context = _asr_context
    token = current.set(context)
    try:
        result = _asr.request("warmup")
        with _start_lock:
            if generation == _generation:
                _loaded = result.ok and _enabled
        return result.message
    finally:
        with _start_lock:
            if generation == _generation:
                _loading = False
        current.reset(token)


def _enable():
    global _enabled
    with _start_lock:
        _enabled = True
        return _generation


def start():
    return _load(_enable())


def start_async():
    generation = _enable()
    threading.Thread(target=lambda: _load(generation), daemon=True).start()


def ready():
    return _enabled and _loaded and _asr.ready()


def status():
    if not _enabled:
        return "语音已关闭"
    return "语音就绪" if ready() else "语音模型加载中" if _loading else "语音未就绪（可继续文字输入）"


def stop_speaking():
    with _speech_lock:
        _playback_cancel.set()
        while True:
            try:
                _speech.get_nowait()
            except queue.Empty:
                break


def stop():
    global _enabled, _loaded, _loading, _generation
    with _start_lock:
        _enabled = _loaded = _loading = False
        _generation += 1
        _asr_context.cancelled.set()
    stop_speaking()
    _asr.stop()


def capture_vad():
    """Press Tab to record, finish after 0.9s of silence. No microphone starts on import."""
    if not ready():
        return ""
    import numpy as np
    stop_speaking()
    proc = subprocess.Popen(["arecord", "-D", config.MIC_DEVICE, "-f", "S16_LE", "-r", "16000",
                             "-c", "1", "-t", "raw", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frames, speech_time, last_heard = [], 0, None
    start_time = time.monotonic()
    baseline = 150
    try:
        while time.monotonic()-start_time < 15:
            if not select.select([proc.stdout], [], [], 0.1)[0]:
                continue
            data = os.read(proc.stdout.fileno(), 4096)
            if not data:
                break
            now = time.monotonic()
            if now-start_time < 0.5:
                continue
            frames.append(data)
            samples = np.frombuffer(data, np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(samples**2))) if len(samples) else 0
            if rms < baseline:
                baseline = baseline*0.96 + rms*0.04
            if rms > max(baseline*2.2, 600):
                speech_time += len(samples)/16000
                last_heard = now
            if speech_time > 0.15 and last_heard is not None and now-last_heard > 0.9:
                break
            if speech_time <= 0.15 and now-start_time > 6:
                break
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        proc.stdout.close()
    if speech_time < 0.2:
        return ""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        with wave.open(path, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"".join(frames))
        result = _asr.request("transcribe", {"audio_path": path})
        return result.data.get("text", "") if result.ok else ""
    finally:
        os.unlink(path)


capture_text = capture_vad


def term_input(prompt="  你 › "):
    if not sys.stdin.isatty():
        return input(prompt).strip()
    import termios
    import tty
    import ui
    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = []
    escape = False
    sys.stdout.write(prompt)
    sys.stdout.flush()
    try:
        tty.setraw(fd)
        while True:
            if not select.select([fd], [], [], 0.05)[0]:
                continue
            data = os.read(fd, 1024)
            if not data:
                return None
            for ch in decoder.decode(data):
                if escape:
                    if ch.isalpha() or ch == "~":
                        escape = False
                    continue
                if ch == "\x1b":
                    escape = True
                    continue
                if ch == "\t":
                    if not ready():
                        sys.stdout.write("\r\n  " + status() + "\r\n" + prompt + "".join(buffer))
                        sys.stdout.flush()
                        continue
                    sys.stdout.write("\r\n  说话中 · 停顿后自动发送\r\n")
                    sys.stdout.flush()
                    termios.tcsetattr(fd, termios.TCSADRAIN, original)
                    try:
                        text = capture_vad()
                    except Exception as exc:
                        error("录音失败", exc)
                        text = ""
                    finally:
                        tty.setraw(fd)
                    if text:
                        sys.stdout.write("  你 › " + text + "\r\n")
                        sys.stdout.flush()
                        return text
                    sys.stdout.write("  未识别到语音\r\n" + prompt + "".join(buffer))
                    sys.stdout.flush()
                    continue
                if ch in ("\r", "\n"):
                    sys.stdout.write("\r\n")
                    sys.stdout.flush()
                    return "".join(buffer)
                if ch == "\x03":
                    sys.stdout.write("\r\n")
                    raise KeyboardInterrupt
                if ch == "\x04":
                    sys.stdout.write("\r\n")
                    return None
                if ch in ("\x7f", "\x08"):
                    if buffer:
                        n = ui._disp_width(buffer.pop())
                        sys.stdout.write("\b"*n + " "*n + "\b"*n)
                elif ch.isprintable():
                    stop_speaking()
                    buffer.append(ch)
                    sys.stdout.write(ch)
                sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)


async def _synth(text, path):
    import edge_tts
    await asyncio.wait_for(edge_tts.Communicate(text, config.TTS_VOICE).save(path), timeout=20)


def speak(text, cancel=None):
    cancel = cancel or threading.Event()
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        with measure("voice.synthesis"):
            for attempt in range(3):
                if cancel.is_set():
                    return
                try:
                    asyncio.run(_synth(text, path))
                    break
                except Exception as exc:
                    if attempt == 2:
                        error("语音合成失败", exc)
                        return
                    if cancel.wait(0.5):
                        return
        if cancel.is_set():
            return
        with measure("voice.playback"):
            proc = subprocess.Popen(["ffplay", "-nodisp", "-loglevel", "quiet", "-autoexit", path],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            while proc.poll() is None:
                if cancel.wait(0.05):
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    break
    except Exception as exc:
        error("语音播放失败", exc)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _play_queue():
    while not _speech_stop.is_set():
        try:
            text, cancel = _speech.get(timeout=0.1)
        except queue.Empty:
            continue
        if not cancel.is_set():
            speak(text, cancel)


def speak_async(text):
    global _speech_thread, _playback_cancel
    if not _enabled or not text.strip():
        return
    stop_speaking()
    if _speech_thread is None or not _speech_thread.is_alive():
        _speech_thread = threading.Thread(target=_play_queue, daemon=True)
        _speech_thread.start()
    with _speech_lock:
        _playback_cancel = threading.Event()
        _speech.put((text, _playback_cancel))


def close():
    stop()
    _speech_stop.set()
    if _speech_thread:
        _speech_thread.join(timeout=1)
