"""SAM2 分割服务进程生命周期：常驻加载模型一次，反复处理分割请求。

仿相机/语音服务：启动时拉起后台进程加载模型(约4.5s)，此后每次 segment_mask
只发一条请求经 stdin/stdout 通道推理，避免每程冷启动(重载 torch+hydra+checkpoint)。
退出时自动关停；未就绪/超时会给出可读错误。
"""

import collections
import json
import os
import queue
import subprocess
import threading
import time

from logger import error as log_error, warning as log_warning, is_error

_GRASPNET_PY = "/home/wsky/miniconda3/envs/graspnet/bin/python"
_SERVER = "/home/wsky/msk_tool/sam2_server.py"
_START_TIMEOUT = 120       # 首次加载模型 + 拉起的等待秒数
_RSP_TIMEOUT = 600         # 单次分割请求等待(含交互弹窗，需等人点选，放宽)

_ss = None   # 常驻 SAM2 服务


class _Sam2Server:
    def __init__(self):
        self.proc = None
        self._q = queue.Queue()
        self._err = collections.deque(maxlen=200)
        self.ready = threading.Event()

    def start(self) -> bool:
        try:
            self.proc = subprocess.Popen(
                [_GRASPNET_PY, _SERVER],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1,
            )
        except Exception as e:  # noqa: BLE001
            log_error("SAM2 服务启动失败", e)
            return False
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._read_err, daemon=True).start()
        t0 = time.time()
        while time.time() - t0 < _START_TIMEOUT:
            if self.ready.is_set():
                return True
            if self.proc.poll() is not None:   # 模型加载失败退出
                log_error("SAM2 服务进程退出", Exception("".join(list(self._err))[-500:]))
                return False
            time.sleep(0.2)
        log_error("SAM2 服务启动超时", Exception(f">{_START_TIMEOUT}s 未就绪"))
        return False

    def _read(self):
        for line in self.proc.stdout:
            line = line.strip()
            if line == "SAM2_READY":
                self.ready.set()
            elif line:
                self._q.put(line)

    def _read_err(self):
        for line in self.proc.stderr:
            line = line.strip()
            self._err.append(line)
            # 模型加载/推理的异常 stderr 落盘，便于排查
            if is_error(line) or "Traceback" in line:
                log_warning(f"SAM2 服务 stderr: {line[:300]}")

    def request(self, req: dict) -> dict:
        self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
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
                    return {"ok": False, "msg": "bad result"}
        log_error("SAM2 请求超时", Exception(f">{_RSP_TIMEOUT}s 无响应"))
        return {"ok": False, "msg": "sam2 请求超时(>600s)"}

    def stop(self):
        if self.proc:
            try:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            except Exception:  # noqa: BLE001
                pass
            self.proc = None


def start() -> str:
    """启动 SAM2 常驻服务，返回状态文字。已就绪则跳过。"""
    global _ss
    if _ss and _ss.ready.is_set() and _ss.proc is not None and _ss.proc.poll() is None:
        return "SAM2 服务已在运行"
    if _ss:
        _ss.stop()
    _ss = _Sam2Server()
    if not _ss.start():
        err = "".join(list(_ss._err))[-300:]
        _ss = None
        return f"SAM2 服务启动失败(模型加载失败)，请检查状态：{err}".strip()
    return "SAM2 服务就绪"


def stop() -> None:
    global _ss
    if _ss:
        _ss.stop()
        _ss = None


def ready() -> bool:
    return _ss is not None and _ss.ready.is_set() and _ss.proc is not None and _ss.proc.poll() is None


def status() -> str:
    return "SAM2 服务就绪" if ready() else "SAM2 服务未运行"


def run(req: dict) -> dict:
    """确保服务就绪后发送一次分割请求，返回结果 dict。"""
    # 未运行则尝试拉起
    s = start()
    if not ready():
        log_error("SAM2 服务未就绪，无法分割", Exception(s or "unknown"))
        return {"ok": False, "msg": s}
    try:
        result = _ss.request(req)
        if isinstance(result, dict) and not result.get("ok"):
            # 分割失败也记录(如无 DISPLAY/无 points、推理报错)，便于排查
            log_error(f"SAM2 分割失败 | {json.dumps(req, ensure_ascii=False)[:200]} -> {result.get('msg', '')[:200]}")
        return result
    except Exception as e:  # noqa: BLE001
        log_error("SAM2 请求异常", e)
        return {"ok": False, "msg": f"sam2 请求出错: {e}"}


# 注意：不再在此注册 atexit 清理，避免 import 即毁掉 SAM2 服务。清理统一由 main.py 负责。
