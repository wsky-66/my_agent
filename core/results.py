"""One result contract for services, workflows, tools and UI."""
from dataclasses import asdict, dataclass, field
from typing import Any
import json


@dataclass
class ToolResult:
    ok: bool
    message: str = ""
    data: Any = field(default_factory=dict)
    error_code: str = ""
    elapsed: float = 0.0

    @classmethod
    def success(cls, message="", data=None):
        return cls(True, message, {} if data is None else data)

    @classmethod
    def failure(cls, code, message, data=None):
        return cls(False, message, {} if data is None else data, code)

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or not isinstance(value.get("ok"), bool):
            raise ValueError("无效的服务结果")
        return cls(**{k: value[k] for k in cls.__dataclass_fields__ if k in value})

    def __str__(self):
        return self.message or self.to_json()
