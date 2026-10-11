"""Lifecycle ownership: only terminate process groups launched by this application."""
from pathlib import Path
import subprocess
import threading
import time

from core.execution import current, is_cancelled
from core.results import ToolResult
from core.runtime import terminate


class ManagedProcess:
    def __init__(self, name, command, log_path, probe, *, preflight=None, diagnose=None):
        self.name, self.command, self.probe = name, command, probe
        self.preflight, self.diagnose = preflight, diagnose
        self.log_path = Path(log_path)
        self.proc = None
        self._lock = threading.RLock()
        self._log_offset = 0

    def running(self):
        proc = self.proc
        return proc is not None and proc.poll() is None

    def _probe(self):
        try:
            result = self.probe()
            if isinstance(result, ToolResult):
                return result
            if result:
                return ToolResult.success(f"{self.name}已就绪", {"ready": True})
            return ToolResult.failure("DEVICE_NOT_READY", f"{self.name}未就绪", {"ready": False})
        except Exception as exc:
            return ToolResult.failure("HEALTH_CHECK_FAILED", f"{self.name}检测失败：{exc}", {"ready": False})

    def status(self):
        result = self._probe()
        result.data = {**result.data, "ready": result.ok, "owned": self.running()}
        return result

    def _diagnostic(self):
        if self.diagnose is None or not self.log_path.exists():
            return None
        with self.log_path.open("rb") as log:
            log.seek(max(self._log_offset, self.log_path.stat().st_size - 32768))
            return self.diagnose(log.read().decode("utf-8", errors="replace"))

    def _failed(self, result):
        proc, self.proc = self.proc, None
        terminate(proc)
        result.data = {**result.data, "ready": False, "owned": False, "log": str(self.log_path)}
        return result

    def start(self, timeout=30):
        with self._lock:
            try:
                if is_cancelled():
                    return self._failed(ToolResult.failure("TASK_CANCELLED", f"{self.name}启动已中断"))
                result = self.status()
                if result.ok:
                    return result
                if is_cancelled():
                    return self._failed(ToolResult.failure("TASK_CANCELLED", f"{self.name}启动已中断"))
                if self.preflight:
                    preflight = self.preflight()
                    if not preflight.ok:
                        return self._failed(preflight)
                if not self.running():
                    terminate(self.proc)
                    self.proc = None
                    self.log_path.parent.mkdir(parents=True, exist_ok=True)
                    self._log_offset = self.log_path.stat().st_size if self.log_path.exists() else 0
                    with self.log_path.open("ab") as log:
                        self.proc = subprocess.Popen(self.command, stdout=log, stderr=log,
                                                     start_new_session=True)
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    if is_cancelled():
                        return self._failed(ToolResult.failure("TASK_CANCELLED", f"{self.name}启动已中断"))
                    diagnosis = self._diagnostic()
                    if diagnosis:
                        return self._failed(diagnosis)
                    if self.proc.poll() is not None:
                        return self._failed(ToolResult.failure("STARTUP_FAILED", f"{self.name}启动进程退出，日志 {self.log_path}"))
                    result = self.status()
                    # A critical child can fail while the ROS launch parent remains alive.
                    diagnosis = self._diagnostic()
                    if diagnosis:
                        return self._failed(diagnosis)
                    if is_cancelled():
                        continue
                    if result.ok:
                        return result
                    ctx = current.get()
                    if ctx:
                        ctx.cancelled.wait(0.1)
                    else:
                        time.sleep(0.1)
                return self._failed(ToolResult.failure("STARTUP_TIMEOUT",
                    f"{result.message}；启动超时，日志 {self.log_path}", result.data))
            except (KeyboardInterrupt, GeneratorExit):
                self._failed(ToolResult.failure("TASK_CANCELLED", f"{self.name}启动已中断"))
                raise
            except Exception as exc:
                return self._failed(ToolResult.failure("STARTUP_FAILED", f"{self.name}启动失败：{exc}"))

    def stop(self):
        with self._lock:
            owned = self.proc is not None
            proc, self.proc = self.proc, None
            terminate(proc)
            return ToolResult.success(f"{self.name}已停止" if owned else f"{self.name}无本程序启动的进程，外部服务保留")
