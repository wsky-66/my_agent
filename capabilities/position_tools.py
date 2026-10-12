"""Model interface to scene geometry; no subprocess or ROS initialization."""
from tools import registry
from services.perception import position
from core.results import ToolResult


@registry.register(description="用当前 RGB-D 快照的深度、内参和标定，把像素或分割 mask 转换成 base_link 坐标（米）。")
def object_position(px: float = None, py: float = None, mask_path: str = None, radius: int = 12) -> ToolResult:
    return position(px, py, mask_path, radius)


def object_position_raw(*args, **kwargs):
    result = position(*args, **kwargs)
    return (result.data, None) if result.ok else (None, result.message)
