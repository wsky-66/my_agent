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


# ---- YOLO-World 工具 ----

_YOLO_PYTHON = "/home/wsky/miniconda3/envs/yoloworld/bin/python"
_YOLO_MODEL_PATH = "/home/wsky/yolo/yolov8s-world.pt"


@registry.register(
    name="detect_objects",
    description="使用 YOLO-World 检测图片中的目标物体。适合图片分析、物体识别、目标检测等任务。"
)
def detect_objects(image_path: str, classes: list) -> str:
    """
    使用 YOLO-World 检测图片中的指定类别物体。\n\n    :param image_path: 图片的绝对路径
    :param classes: 要检测的目标类别列表，例如 ["person", "car", "dog"]
    """
    try:
        script = f"""
from ultralytics import YOLOWorld
model = YOLOWorld({_YOLO_MODEL_PATH!r})
model.set_classes({classes!r})
results = model({image_path!r})
boxes = results[0].boxes
if boxes is None or len(boxes) == 0:
    print("EMPTY")
else:
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        label = {classes!r}[cls_id] if cls_id < len({classes!r}) else f"class_{{cls_id}}"
        print(f"{{label}}|{{conf:.4f}}|{{xyxy[0]:.0f}}|{{xyxy[1]:.0f}}|{{xyxy[2]:.0f}}|{{xyxy[3]:.0f}}")
"""
        result = subprocess.run(
            [_YOLO_PYTHON, "-c", script],
            capture_output=True, text=True, timeout=config.SHELL_TIMEOUT + 120,
            cwd="/home/wsky/yolo",
        )
        output = result.stdout.strip()
        if result.stderr.strip():
            output += "\n[stderr]\n" + result.stderr.strip()

        if not output or output == "EMPTY":
            return f"在 {image_path} 中未检测到任何 {classes} 类别物体。"

        lines = [f"检测结果 ({image_path}):"]
        for line in output.strip().split("\n"):
            if not line or line == "EMPTY":
                continue
            parts = line.split("|", 5)
            if len(parts) == 6:
                label, conf, x1, y1, x2, y2 = parts
                lines.append(
                    f"  - {label}: 置信度={float(conf):.2f}, 坐标=[{x1}, {y1}, {x2}, {y2}]"
                )
            else:
                lines.append(f"  {line}")
        return "\n".join(lines)
    except Exception as e:
        return f"YOLO-World 检测出错: {e}"


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
