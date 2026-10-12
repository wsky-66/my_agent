"""Checked motion workflows exposed as high-level model tools."""
from tools import registry
from core.results import ToolResult
from workflows.manipulation import execute


@registry.register(description="执行一个或多个抓取/放置任务，优先用于常规抓放，避免逐步请求模型。actions 例子：[{\"object\":\"apple\",\"target\":\"bowl\"},{\"object\":\"banana\",\"target_position\":[0.3,0.1,0.05]}]。纯抓取只指定 object。所有标签用英文。detection_hints 可补充按常识或用户信息确定的颜色描述，例如 {\"banana\":[\"yellow banana\",\"yellow fruit\",\"yellow object\"]}；不确定颜色先问用户，再执行任务。无 scene_id 时先收拢、统一预扫；提供 scene_id 可复用本轮场景，detection_hints 仅用于新扫描。有配置后端返回的 pose 时处理 roll 范围后由 MoveIt 求解；无 pose 时默认用定位坐标和固定姿态兜底。strategies 标明 graspnet、grcnn 或 fallback。仅需要禁用兜底时设 allow_fallback=false。运动失败立即停止，不将 IK 失败当作无 pose 兜底。")
def execute_pick_place(actions: list[dict], scene_id: str = None,
                       return_to_sleep: bool = True, allow_fallback: bool = True,
                       detection_hints: dict[str, list[str]] = None) -> ToolResult:
    return execute(actions, scene_id, return_to_sleep, allow_fallback, detection_hints)
