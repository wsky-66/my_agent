"""Serialized JSON workers with owned process groups and timeout isolation."""
import collections
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
import uuid

from core.results import ToolResult
from core.execution import is_cancelled
from logger import record_timing


def terminate(proc):
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        deadline = time.monotonic() + 3
        # The group may outlive its leader (ROS launch can lose a critical child).
        while time.monotonic() < deadline:
            proc.poll()  # reap the leader before checking for remaining descendants
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=3)
    except ProcessLookupError:
        proc.wait()


class JsonWorker:
    def __init__(self, name, command, *, cwd=None, timeout=90, startup_timeout=120,
                 ready_marker="READY", legacy=False):
        self.name, self.command = name, command
        self.cwd = cwd or str(Path(__file__).resolve().parents[1])
        self.timeout, self.startup_timeout = timeout, startup_timeout
        self.ready_marker, self.legacy = ready_marker, legacy
        self.proc = None
        self._ready = threading.Event()
        self._lock = threading.RLock()
        self._queue = queue.Queue()
        self._errors = collections.deque(maxlen=30)
        self._threads = []

    def ready(self):
        return self.proc is not None and self.proc.poll() is None and self._ready.is_set()

    def _start(self):
        if self.ready():
            return
        self._stop()
        self._queue = queue.Queue()
        self._ready = threading.Event()
        self._errors.clear()
        start = time.monotonic()
        proc = subprocess.Popen(self.command, cwd=self.cwd, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, bufsize=1, start_new_session=True)
        self.proc = proc
        responses, ready, errors = self._queue, self._ready, self._errors

        def read():
            try:
                for line in proc.stdout:
                    line = line.strip()
                    if line == self.ready_marker:
                        ready.set()
                    elif line.startswith("RESULT:"):
                        try:
                            responses.put(json.loads(line[7:]))
                        except ValueError:
                            responses.put({"protocol_error": "无效的 JSON 响应"})
            finally:
                responses.put({"protocol_error": "服务进程退出"})

        def read_errors():
            for line in proc.stderr:
                errors.append(line.rstrip())

        self._threads = [threading.Thread(target=read, daemon=True), threading.Thread(target=read_errors, daemon=True)]
        for thread in self._threads:
            thread.start()
        deadline = time.monotonic() + self.startup_timeout
        while not ready.wait(0.05):
            if is_cancelled():
                self._stop()
                raise RuntimeError("任务已中断")
            if proc.poll() is not None or time.monotonic() >= deadline:
                detail = "\n".join(errors)[-600:]
                self._stop()
                raise RuntimeError(f"{self.name} 启动失败或超时: {detail}")
        record_timing(f"worker.{self.name}.startup", time.monotonic() - start)

    def request(self, action, params=None):
        with self._lock:
            if is_cancelled():
                return ToolResult.failure("TASK_CANCELLED", "任务已中断")
            start = time.monotonic()
            result = None
            try:
                self._start()
                request_id = uuid.uuid4().hex
                req = params or {} if self.legacy else {"id": request_id, "action": action, "params": params or {}}
                self.proc.stdin.write(json.dumps(req, ensure_ascii=False, allow_nan=False) + "\n")
                self.proc.stdin.flush()
                deadline = time.monotonic() + self.timeout
                while True:
                    if is_cancelled():
                        self._stop()
                        result = ToolResult.failure("TASK_CANCELLED", "任务中断，停止后续步骤；已发送的物理动作需要确认状态")
                        return result
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        raise queue.Empty
                    try:
                        reply = self._queue.get(timeout=min(0.1, remaining))
                        break
                    except queue.Empty:
                        continue
                if "protocol_error" in reply:
                    raise RuntimeError(reply["protocol_error"])
                if self.legacy:
                    data = {k: v for k, v in reply.items() if k not in ("ok", "msg")}
                    result = (ToolResult.success(reply.get("msg", "分割完成"), data) if reply.get("ok")
                              else ToolResult.failure("SEGMENT_FAILED", reply.get("msg", "分割失败"), data))
                else:
                    if reply.get("id") != request_id:
                        raise RuntimeError("服务响应编号不匹配")
                    result = ToolResult.from_dict(reply["result"])
            except queue.Empty:
                self._stop()  # discard late replies; never retry a physical command automatically
                result = ToolResult.failure("WORKER_TIMEOUT", f"{self.name} 请求超时")
            except (KeyboardInterrupt, GeneratorExit):
                self._stop()
                raise
            except Exception as exc:
                self._stop()
                result = ToolResult.failure("WORKER_UNAVAILABLE", str(exc))
            finally:
                record_timing(f"worker.{self.name}.{action}", time.monotonic() - start,
                              result.ok if result else False)
            return result

    def _stop(self):
        proc, self.proc = self.proc, None
        self._ready.clear()
        terminate(proc)
        for thread in self._threads:
            thread.join(timeout=1)
        self._threads = []
        if proc:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream:
                    stream.close()

    def stop(self):
        with self._lock:
            self._stop()
