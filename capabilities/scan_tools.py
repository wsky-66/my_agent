"""Scene preparation model tool."""
from tools import registry
from core.results import ToolResult
from services.scene import scan


@registry.register(description="预扫一组英文物体名称，共用同步 RGB-D 帧定位。grasp_labels 仅列要抓的物体；放置目标只定位，省去抓取推理。返回 scene_id、每物体坐标、抓取候选及明确失败状态。扫描前应收拢机械臂。")
def scan_objects(labels: list[str], grasp_labels: list[str] = None) -> ToolResult:
    return scan(labels, grasp_labels)
