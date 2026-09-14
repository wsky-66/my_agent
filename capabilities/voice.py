"""语音模块：对讲式(push-to-talk)语音输入 + 语音播报。

- 输入: FunASR SenseVoiceSmall(CPU) 常驻服务进程
- 输出: edge-tts(zh-CN-YunxiNeural) 阻塞播报, 用 ffplay 播放 mp3
- 录音: arecord(板载 ALC257); Enter 开始/结束; 不用安装 PortAudio
"""

import asyncio
import collections
import json
import os
import queue
import signal
import subprocess
import tempfile
import threading
import time
import wave

import numpy as np

from logger import error as log_error, warning as log_warning

_PY = "/home/wsky/miniconda3/envs/my_agent/bin/python"
_SERVER = "/home/wsky/my_agent/capabilities/funasr_server.py"

_SR = 16000
_MIC_DEVICE = "plughw:1,0"      # 板载 ALC257 录音(plughw 自动重采样到 16k)
_TTS_VOICE = "zh-CN-YunxiNeural"
_ASR_TIMEOUT = 120               # 首次模型加载(含下载)放宽
_RSP_TIMEOUT = 30                # 单次识别结果等待
_MAX_REC = 60                    # 单次录音上限(秒)

# ------------------------------------------------------------------ ASR 服务器
_ss = None   # 常驻 ASR 服务


class _AsrServer:
    def __init__(self):
        self.proc = None
        self._q = queue.Queue()
        self._err = collections.deque(maxlen=200)   # 保留 stderr 尾部便于诊断
        self.ready = threading.Event()

    def start(self):
        try:
            self.proc = subprocess.Popen(
                [_PY, _SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1)
        except Exception as e:
            log_error("ASR 服务启动失败", e)
            return False
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._read_err, daemon=True).start()
        t0 = time.time()
        while time.time() - t0 < _ASR_TIMEOUT:
            if self.ready.is_set():
                return True
            if self.proc.poll() is not None:   # 模型加载失败退出
                log_error("ASR 服务进程退出", Exception("".join(list(self._err))[-500:]))
                return False
            time.sleep(0.2)
        log_error("ASR 服务启动超时", Exception(">120s 未就绪"))
        return False

    def _read(self):
        for line in self.proc.stdout:
            line = line.strip()
            if line == "VOICE_READY":
                self.ready.set()
            elif line:
                self._q.put(line)

    def _read_err(self):
        for line in self.proc.stderr:
            self._err.append(line)

    def request(self, audio_path):
        self.proc.stdin.write(json.dumps({"audio_path": audio_path}) + "\n")
        self.proc.stdin.flush()
        t0 = time.time()
        while time.time() - t0 < _RSP_TIMEOUT:
            try:
                line = self._q.get_nowait()
            except queue.Empty:
                time.sleep(0.05)
                continue
            if line.startswith("RESULT:"):
                try:
                    return json.loads(line[len("RESULT:"):])
                except json.JSONDecodeError:
                    return {"text": ""}
        return {"text": "", "error": "asr timeout"}

    def stop(self):
        if self.proc:
            try:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            except Exception:
                pass
            self.proc = None


def start() -> str:
    """启动语音服务(ASR)。成功返回信息, 失败返回错误说明。"""
    global _ss
    if _ss and _ss.ready.is_set():
        return "语音服务已在运行"
    _ss = _AsrServer()
    if not _ss.start():
        _ss = None
        return "语音服务启动失败(ASR 模型/依赖不可用)，已回退文本模式"
    return "语音服务就绪"


def stop() -> None:
    global _ss
    if _ss:
        _ss.stop()
        _ss = None


def ready() -> bool:
    return _ss is not None and _ss.ready.is_set() and _ss.proc is not None and _ss.proc.poll() is None


def status() -> str:
    return "语音服务就绪" if ready() else "语音服务未运行"


# ------------------------------------------------------------------ 录音 + 识别
def _record() -> str:
    """对讲式录音: 按 Enter 开始, 再按 Enter 结束。返回归一化 wav 路径; 失败返回 None。"""
    print("  按 Enter 开始说话…", end="", flush=True)
    # 开始: 若 Ctrl-C 在此处, 让 KeyboardInterrupt 冒泡到上层以退出语音模式
    input()
    fd, raw = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    proc = subprocess.Popen(["arecord", "-D", _MIC_DEVICE, "-f", "S16_LE",
                             "-r", str(_SR), "-c", "1", "-d", str(_MAX_REC), raw],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  🎙 录音中… 再按 Enter 结束", end="", flush=True)
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("  ✔ 已停止")
    # ffmpeg 归一化为干净 16k wav(应对 arecord 中断可能不完整的头部)
    clean = raw + ".clean.wav"
    r = subprocess.run(["ffmpeg", "-y", "-i", raw, "-ar", str(_SR), "-ac", "1",
                        "-f", "wav", clean], capture_output=True)
    os.unlink(raw)
    if r.returncode != 0 or not os.path.exists(clean):
        return None
    return clean


def capture_text() -> str:
    """对讲式录一句 -> ASR 识别, 返回文字(无内容/失败返回 "")。"""
    if not ready():
        return ""
    wav = _record()
    if not wav:
        print("  未能采集到音频")
        return ""
    res = _ss.request(wav)
    try:
        os.unlink(wav)
    except OSError:
        pass
    text = (res.get("text") or "").strip()
    if not text and res.get("error"):
        print(f"  识别失败: {res['error']}")
    return text


# 静音自动停的录音(带 VAD)。用于终端"按一下开始、说完自动结束"。
_VAD_SILENCE = 0.9       # 停口静音多少秒判定结束
_VAD_START_TIMEOUT = 6   # 开始后多久没说话就放弃
_VAD_MIN_SPEECH = 0.2    # 至少说这么多秒才算有效
_VAD_HEARD_AT = 0.15     # 累计语音达到此值才认定"开始说话"(抗单帧爆音)
_VAD_WARMUP = 0.5        # 跳过起始爆音/瞬态
_VAD_CAP = 15            # 硬上限, 防止一直挂起


def capture_vad() -> str:
    """按一下开始录音, 实时能量检测: 听到底噪->说话, 停口_VAD_SILENCE秒自动停。

    返回识别文字(未说话/失败返回 "")。
    """
    if not ready():
        return ""
    proc = subprocess.Popen(["arecord", "-D", _MIC_DEVICE, "-f", "S16_LE", "-r", str(_SR),
                             "-c", "1", "-t", "raw", "-"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
    frames = []
    baseline = 150.0
    speech_time = 0.0
    last_hear = None
    heard_any = False
    start = time.time()
    warmup_end = start + _VAD_WARMUP
    chunk = 2048  # 采样点/次
    try:
        while True:
            now0 = time.time()
            if (now0 - start) > _VAD_CAP:
                break
            data = os.read(proc.stdout.fileno(), chunk * 2)
            if not data:
                break
            now = time.time()
            if now < warmup_end:
                continue   # 丢弃起始爆音
            frames.append(data)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
            if rms < baseline:
                baseline = baseline * 0.96 + rms * 0.04
            is_speech = rms > max(baseline * 2.2, 600)
            if is_speech:
                speech_time += len(samples) / _SR
                last_hear = now
            heard_any = speech_time > _VAD_HEARD_AT
            # 说完: 已认作说话且静音足够久
            if heard_any and last_hear is not None and (now - last_hear) > _VAD_SILENCE:
                break
            # 一直没认作说话超时
            if not heard_any and (now - start) > _VAD_START_TIMEOUT:
                break
    finally:
        try:
            proc.kill()
        except Exception:
            pass

    pcm = b"".join(frames)
    if not pcm or speech_time < _VAD_MIN_SPEECH:
        return ""
    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_SR)
        w.writeframes(pcm)
    res = _ss.request(wav)
    try:
        os.unlink(wav)
    except OSError:
        pass
    return (res.get("text") or "").strip()


def record_hold(stop_event: threading.Event) -> str:
    """按住录音: 直到 stop_event 置位才停止。返回归一化 wav 路径; 失败 None。"""
    if not ready():
        return None
    fd, raw = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    proc = subprocess.Popen(["arecord", "-D", _MIC_DEVICE, "-f", "S16_LE",
                             "-r", str(_SR), "-c", "1", "-d", str(_MAX_REC), raw],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while not stop_event.wait(0.1):
        if proc.poll() is not None:
            break
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
    clean = raw + ".clean.wav"
    r = subprocess.run(["ffmpeg", "-y", "-i", raw, "-ar", str(_SR), "-ac", "1",
                        "-f", "wav", clean], capture_output=True)
    os.unlink(raw)
    if r.returncode != 0 or not os.path.exists(clean):
        return None
    return clean


def capture_hold(stop_event: threading.Event) -> str:
    """按住录音到置位 -> ASR, 返回文字(无内容/失败 "")。"""
    wav = record_hold(stop_event)
    if not wav:
        return ""
    res = _ss.request(wav)
    try:
        os.unlink(wav)
    except OSError:
        pass
    return (res.get("text") or "").strip()


# ------------------------------------------------------------------ 终端 PTT 输入
PTT_KEY = "\t"          # 按住 Tab 说话
_PTT_GAP = 0.35         # 松开判定: 超过该秒数无 PTT_KEY 重复视为松开


def term_input(prompt: str = "你 › ") -> str:
    """终端输入: 打字(回车发送) 或 按 Tab 说话(说完了静音自动结束识别)。

    返回字符串; Ctrl-C 抛 KeyboardInterrupt; Ctrl-D(EOF) 返回 None; Esc 清空本行。
    非 tty 环境回退到普通 input()。
    """
    import codecs
    import select
    import sys as _sys
    import termios
    import tty

    if not _sys.stdin.isatty():
        return input(prompt).strip()

    fd = _sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    buf = []
    _sys.stdout.write(prompt + "  [Tab=说] "); _sys.stdout.flush()
    dec = codecs.getincrementaldecoder("utf-8")()
    try:
        tty.setraw(fd)
        while True:
            r, _, _ = select.select([fd], [], [], 0.04)
            if not r:
                continue
            data = os.read(fd, 1024)
            if not data:
                return None
            for ch in dec.decode(data):
                if ch == "\t":
                    # 按 Tab: 开始录音(静音自动停)
                    _sys.stdout.write("\r\n  🔴 说话中…(说完稍停自动结束)\r\n"); _sys.stdout.flush()
                    text = capture_vad()
                    if text:
                        _sys.stdout.write("  ✅ 你：%s\r\n" % text); _sys.stdout.flush()
                        return text
                    _sys.stdout.write("  (未识别到语音，可重按 Tab 或继续打字)\r\n"); _sys.stdout.flush()
                    _sys.stdout.write(prompt + "  [Tab=说] "); _sys.stdout.flush()
                    continue
                if ch in ("\r", "\n"):
                    _sys.stdout.write("\r\n"); _sys.stdout.flush()
                    return "".join(buf)
                if ch in ("\x7f", "\x08"):     # Backspace
                    if buf:
                        buf.pop()
                        _sys.stdout.write("\b \b"); _sys.stdout.flush()
                elif ch == "\x03":              # Ctrl-C
                    _sys.stdout.write("\r\n"); _sys.stdout.flush()
                    raise KeyboardInterrupt
                elif ch == "\x04":              # Ctrl-D (EOF)
                    _sys.stdout.write("\r\n"); _sys.stdout.flush()
                    return None
                elif ch == "\x1b":              # Esc: 清空本行
                    _sys.stdout.write("\b" * len(buf) + " " * len(buf) + "\b" * len(buf))
                    _sys.stdout.flush()
                    buf = []
                elif ch.isprintable():
                    buf.append(ch)
                    _sys.stdout.write(ch); _sys.stdout.flush()
    except (KeyboardInterrupt, EOFError):
        raise
    finally:
        dec.reset()
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ------------------------------------------------------------------ TTS 播报
async def _synth(text: str, out: str):
    import edge_tts
    await edge_tts.Communicate(text, _TTS_VOICE).save(out)


def _tts_to_mp3(text: str, out: str) -> bool:
    """edge-tts 合成(偶发连接重置, 重试)。返回是否成功。"""
    last = None
    for _ in range(3):
        try:
            asyncio.run(_synth(text, out))
            return True
        except Exception as e:
            last = e
            time.sleep(1)
    log_error("edge-tts 合成失败", last)
    return False


def speak(text: str) -> None:
    """TTS 播报(阻塞到播完)。"""
    text = (text or "").strip()
    if not text:
        return
    fd, mp3 = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        if not _tts_to_mp3(text, mp3):
            return
        subprocess.run(["ffplay", "-nodisp", "-loglevel", "quiet", "-autoexit", mp3],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    finally:
        try:
            os.unlink(mp3)
        except OSError:
            pass


# 注意：不再在此注册 atexit 清理，避免 import 即毁掉语音服务。清理统一由 main.py 负责。
