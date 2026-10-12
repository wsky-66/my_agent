"""Scene preparation model tool."""
from tools import registry
from core.results import ToolResult
from services.scene import scan


@registry.register(description="预扫一组英文物体名称，共用同步 RGB-D 帧定位。grasp_labels 仅列要抓的物体；放置目标只定位，省去抓取推理。detection_hints 可补充按常识或用户信息确定的颜色描述，例如 {\"banana\":[\"yellow banana\",\"yellow fruit\",\"yellow object\"]}，颜色不确定时先问用户。返回 scene_id、每物体坐标、抓取候选及明确失败状态。扫描前应收拢机械臂。")
def scan_objects(labels: list[str], grasp_labels: list[str] = None,
                 detection_hints: dict[str, list[str]] = None) -> ToolResult:
    return scan(labels, grasp_labels, detection_hints)
