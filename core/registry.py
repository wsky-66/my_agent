"""Typed tool registration, schema generation and validated dispatch."""
import inspect
import math
import time
import types
from typing import Any, Union, get_args, get_origin, get_type_hints

from core.results import ToolResult
from logger import error, record_timing


def schema(annotation):
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, types.UnionType):
        return {"anyOf": [schema(a) for a in args]}
    if annotation in (Any, inspect.Parameter.empty):
        return {}
    if annotation is type(None):
        return {"type": "null"}
    if origin is list or annotation is list:
        return {"type": "array", "items": schema(args[0]) if args else {}}
    if origin is dict or annotation is dict:
        return {"type": "object", "additionalProperties": schema(args[1]) if args else {}}
    return {"type": {str: "string", int: "integer", float: "number", bool: "boolean"}[annotation]}


def validate(value, spec, path):
    if "anyOf" in spec:
        for part in spec["anyOf"]:
            try:
                validate(value, part, path)
                return
            except ValueError:
                pass
        raise ValueError(f"{path} 类型不匹配")
    kind = spec.get("type")
    valid = {
        "null": value is None, "string": isinstance(value, str),
        "integer": type(value) is int, "number": type(value) in (int, float),
        "boolean": type(value) is bool, "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }
    if kind and not valid[kind]:
        raise ValueError(f"{path} 必须是 {kind}")
    if kind == "number" and not math.isfinite(value):
        raise ValueError(f"{path} 必须为有限数字")
    if kind == "array":
        for i, item in enumerate(value):
            validate(item, spec.get("items", {}), f"{path}[{i}]")
    if kind == "object":
        for key, item in value.items():
            validate(item, spec.get("additionalProperties", {}), f"{path}.{key}")


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, func=None, *, name=None, description=None):
        if func is None:
            return lambda f: self.register(f, name=name, description=description)
        key = name or func.__name__
        if key in self._tools and self._tools[key] is not func:
            raise ValueError(f"重复注册工具: {key}")
        func._tool_name = key
        func._tool_description = description or inspect.getdoc(func) or ""
        self._tools[key] = func
        return func

    def _parameters(self, func):
        hints = get_type_hints(func)
        params, required = {}, []
        for name, p in inspect.signature(func).parameters.items():
            spec = schema(hints.get(name, str))
            if p.default is None and "anyOf" not in spec:
                spec = {"anyOf": [spec, {"type": "null"}]}
            for line in func._tool_description.splitlines():
                prefix = f":param {name}:"
                if line.strip().startswith(prefix):
                    spec["description"] = line.strip()[len(prefix):].strip()
            params[name] = spec
            if p.default is inspect.Parameter.empty:
                required.append(name)
        return params, required

    def get_schemas(self):
        result = []
        for name, func in self._tools.items():
            params, required = self._parameters(func)
            description = func._tool_description.split(":param", 1)[0].strip()
            result.append({"type": "function", "function": {
                "name": name, "description": description,
                "parameters": {"type": "object", "properties": params,
                               "required": required, "additionalProperties": False},
            }})
        return result

    def execute_result(self, name, arguments):
        start = time.monotonic()
        try:
            func = self._tools.get(name)
            if func is None:
                result = ToolResult.failure("UNKNOWN_TOOL", f"未找到工具 {name}")
            else:
                if not isinstance(arguments, dict):
                    raise ValueError("工具参数必须是 JSON 对象")
                try:
                    inspect.signature(func).bind(**arguments)
                except TypeError as exc:
                    raise ValueError(str(exc)) from exc
                specs, _ = self._parameters(func)
                for key, value in arguments.items():
                    validate(value, specs[key], key)
                value = func(**arguments)
                result = value if isinstance(value, ToolResult) else ToolResult.success(str(value))
        except ValueError as exc:
            result = ToolResult.failure("INVALID_ARGUMENT", str(exc))
        except Exception as exc:
            error(f"工具执行异常: {name}", exc)
            result = ToolResult.failure("TOOL_ERROR", str(exc))
        result.elapsed = round(time.monotonic() - start, 4)
        record_timing(f"tool.{name}", result.elapsed, result.ok)
        if not result.ok:
            error(f"工具失败: {name} [{result.error_code}] {result.message}")
        return result

    def execute(self, name, arguments):
        return self.execute_result(name, arguments).to_json()

    def list_tools(self):
        return list(self._tools)

    def describe_tools(self):
        return [(k, f._tool_description.splitlines()[0]) for k, f in self._tools.items()]


registry = ToolRegistry()
