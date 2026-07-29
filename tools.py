import inspect
import os
import subprocess
from typing import Callable, Any, get_type_hints

from config import config

_PYTHON_TYPE_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


def _python_type_to_json_schema(annotation) -> dict:
    if annotation in _PYTHON_TYPE_TO_JSON:
        return {"type": _PYTHON_TYPE_TO_JSON[annotation]}
    origin = getattr(annotation, "__origin__", None)
    if origin is list:
        inner = getattr(annotation, "__args__", (str,))[0]
        return {"type": "array", "items": _python_type_to_json_schema(inner)}
    return {"type": "string"}


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Callable] = {}

    def register(self, func: Callable = None, *, name: str = None, description: str = None):
        """装饰器：注册一个 Python 函数为可调用工具。"""
        if func is None:
            return lambda f: self.register(f, name=name, description=description)

        tool_name = name or func.__name__
        self._tools[tool_name] = func
        func._tool_name = tool_name
        func._tool_description = description or func.__doc__ or ""
        return func

    def get_schemas(self) -> list[dict]:
        """生成 OpenAI 兼容的 tool schemas。"""
        schemas = []
        for name, func in self._tools.items():
            hints = get_type_hints(func)
            sig = inspect.signature(func)
            params = {}
            required = []

            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue
                param_type = hints.get(param_name, str)
                param_schema = _python_type_to_json_schema(param_type)

                param_doc = _parse_param_doc(func._tool_description, param_name) or \
                            _parse_param_doc(func.__doc__ or "", param_name)

                param_def = {
                    "type": param_schema["type"],
                    "description": param_doc,
                }
                if "items" in param_schema:
                    param_def["items"] = param_schema["items"]

                if param.default is inspect.Parameter.empty:
                    required.append(param_name)

                params[param_name] = param_def

            schema = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": _parse_func_description(func._tool_description),
                    "parameters": {
                        "type": "object",
                        "properties": params,
                        "required": required,
                    },
                },
            }
            schemas.append(schema)

        return schemas

    def execute(self, name: str, arguments: dict) -> str:
        """执行一个工具调用并返回结果。"""
        if name not in self._tools:
            return f"错误：未找到工具 '{name}'"

        func = self._tools[name]
        try:
            result = func(**arguments)
            return str(result)
        except Exception as e:
            return f"工具执行出错: {e}"

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())


def _parse_func_description(doc: str) -> str:
    """从 docstring 提取函数描述（第一行）。"""
    if not doc:
        return "No description"
    return doc.strip().split("\n")[0].strip()


def _parse_param_doc(doc: str, param_name: str) -> str:
    """从 docstring 提取参数描述。"""
    if not doc:
        return ""
    for line in doc.split("\n"):
        line = line.strip()
        if line.startswith(f":param {param_name}:"):
            return line.split(":", 2)[-1].strip()
        if line.startswith(f"{param_name}:"):
            parts = line.split(":", 1)
            if len(parts) > 1:
                return parts[1].strip()
    return ""


registry = ToolRegistry()


# ---- 内置 Shell 工具 ----

@registry.register(name="run_shell", description="执行一条 Shell 命令并返回输出。可用于文件操作、系统信息查询、运行脚本等。")
def run_shell(command: str) -> str:
    """
    执行一条 Shell 命令并返回输出。

    :param command: 要执行的 Shell 命令
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=config.SHELL_TIMEOUT,
            cwd="/home/wsky/my_agent",
        )
        output = result.stdout.strip()
        if result.stderr.strip():
            output += "\n[stderr]\n" + result.stderr.strip()
        if not output:
            output = f"(退出码: {result.returncode})"
        return output[:4000]
    except subprocess.TimeoutExpired:
        return f"命令超时 ({config.SHELL_TIMEOUT}秒)"
    except Exception as e:
        return f"执行命令时出错: {e}"


# ---- 内置文件工具 ----

@registry.register(name="read_file", description="读取指定文件的内容。用于查看代码、配置、日志等文件。")
def read_file(path: str) -> str:
    """
    读取指定文件的内容。

    :param path: 文件路径（绝对路径或相对于工作目录的路径）
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()[:8000]
    except FileNotFoundError:
        return f"文件不存在: {path}"
    except Exception as e:
        return f"读取文件出错: {e}"


@registry.register(name="write_file", description="将内容写入指定文件。用于创建或覆盖文件。")
def write_file(path: str, content: str) -> str:
    """
    将内容写入指定文件。

    :param path: 文件路径（绝对路径或相对于工作目录的路径）
    :param content: 要写入的内容
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"已写入 {len(content)} 字节到 {path}"
    except Exception as e:
        return f"写入文件出错: {e}"
