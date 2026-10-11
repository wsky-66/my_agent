"""Checked motion workflows exposed as high-level model tools."""
from tools import registry
from core.results import ToolResult
from workflows.manipulation import execute


@registry.register(description="执行一个或多个抓取/放置任务，优先用于常规抓放，避免逐步请求模型。actions 例子：[{\"object\":\"apple\",\"target\":\"bowl\"},{\"object\":\"banana\",\"target_position\":[0.3,0.1,0.05]}]。纯抓取只指定 object。所有标签用英文。无 scene_id 时先收拢、统一预扫；提供 scene_id 可复用本轮场景。程序检查每步成功，失败立即停止；位置用米、朝向用度。allow_fallback 允许无候选时沿用原有坐标+50度标准姿态兜底。")
def execute_pick_place(actions: list[dict], scene_id: str = None,
                       return_to_sleep: bool = True, allow_fallback: bool = True) -> ToolResult:
    return execute(actions, scene_id, return_to_sleep, allow_fallback)
