"""Model-facing built-in tools and compatibility exports for custom tools."""
from pathlib import Path
import subprocess
import time
from config import config
from core.execution import is_cancelled
from core.registry import ToolRegistry, registry
from core.results import ToolResult
from core.runtime import terminate


@registry.register(description="用于用户明确要求的命令或项目维护，返回 stdout、stderr 和退出码。常规视觉、抓放、目标澄清不使用此工具，不能用它寻找旧场景或绕过工作流失败。")
def run_shell(command: str) -> ToolResult:
    if is_cancelled():
        return ToolResult.failure("TASK_CANCELLED", "任务已中断")
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd=config.ROOT, start_new_session=True)
    completed = False
    deadline = time.monotonic()+config.SHELL_TIMEOUT
    try:
        while True:
            if is_cancelled():
                return ToolResult.failure("TASK_CANCELLED", "任务中断，Shell 进程组已停止")
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                return ToolResult.failure("SHELL_TIMEOUT", "命令执行超时，Shell 进程组已停止")
            try:
                stdout, stderr = proc.communicate(timeout=min(.1, remaining))
                completed = True
                break
            except subprocess.TimeoutExpired:
                continue
    finally:
        if not completed:
            terminate(proc)
        proc.stdout.close()
        proc.stderr.close()
    data = {"stdout": stdout[-4000:], "stderr": stderr[-2000:], "returncode": proc.returncode}
    return ToolResult.success("命令完成", data) if proc.returncode == 0 else ToolResult.failure("SHELL_FAILED", f"命令退出码 {proc.returncode}", data)


def _path(path):
    value = Path(path).expanduser()
    return value if value.is_absolute() else config.ROOT / value


@registry.register(description="读取文本文件，返回内容。相对路径以项目目录为基准。")
def read_file(path: str) -> ToolResult:
    value = _path(path)
    return ToolResult.success("文件已读取", {"path": str(value), "content": value.read_text(encoding="utf-8")[:8000]})


@registry.register(description="创建或覆盖文本文件。相对路径以项目目录为基准。")
def write_file(path: str, content: str) -> ToolResult:
    value = _path(path)
    value.parent.mkdir(parents=True, exist_ok=True)
    value.write_text(content, encoding="utf-8")
    return ToolResult.success("文件已保存", {"path": str(value), "characters": len(content)})
