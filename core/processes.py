"""Lifecycle ownership: only terminate process groups launched by this application."""
from pathlib import Path
import subprocess
import time

from core.results import ToolResult
from core.runtime import terminate


class ManagedProcess:
    def __init__(self, name, command, log_path, probe):
        self.name, self.command, self.probe = name, command, probe
        self.log_path = Path(log_path)
        self.proc = None

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def status(self):
        ready = self.probe()
        return ToolResult.success(f"{self.name}{'就绪' if ready else '未就绪'}",
                                  {"ready": ready, "owned": self.running()})

    def start(self, timeout=30):
        if self.probe():
            return ToolResult.success(f"{self.name}已就绪", {"owned": self.running()})
        if not self.running():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("ab") as log:
                self.proc = subprocess.Popen(self.command, stdout=log, stderr=log,
                                             start_new_session=True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                break
            if self.probe():
                return ToolResult.success(f"{self.name}已就绪", {"pid": self.proc.pid, "owned": True})
            time.sleep(0.3)
        return ToolResult.failure("STARTUP_FAILED", f"{self.name}未就绪，日志 {self.log_path}")

    def stop(self):
        owned = self.proc is not None
        proc, self.proc = self.proc, None
        terminate(proc)
        return ToolResult.success(f"{self.name}已停止" if owned else f"{self.name}无本程序启动的进程，外部服务保留")
