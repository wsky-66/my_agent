"""Model adapter to the persistent GraspNet service."""
from pathlib import Path
from config import config
from core.results import ToolResult
from tools import registry
from services import grasp


@registry.register(description="基于当前 RGB-D 和严格物体 mask 搜索多方向/角度/深度抓取候选。返回接触位置 position、完整 rotation、depth、width、转换后的 ee_position 和夹爪开合值。检查实测桌面、完整场景、进近和闭合路径。no_collision=true 的结果仅供查看，不能用于抓放。模型常驻复用。")
def detect_grasp(top: int = 5, no_collision: bool = False, score_min: float = 0,
                 use_mask: bool = True, mask_path: str = None) -> ToolResult:
    path = mask_path or config.WORKSPACE_MASK
    if use_mask and not Path(path).is_file():
        return ToolResult.failure("MISSING_MASK", "请先 segment_mask 或提供有效 mask_path")
    result = grasp.infer([{"label": "object", "mask_path": path if use_mask else None}],
                         top=top, no_collision=no_collision, score_min=score_min)
    if result.ok:
        diagnostics=result.data.get("_diagnostics",{}).get("object",{})
        result.data = result.data["object"]
        if not result.data:
            return ToolResult.failure("NO_GRASP",diagnostics.get("reason","没有可用抓取候选"),
                                      {"diagnostics":diagnostics})
    return result


def detect_grasp_raw(*args, **kwargs):
    result = detect_grasp(*args, **kwargs)
    return (result.data, None) if result.ok else (None, result.message)


def detect_grasp_batch(items, top=3, no_collision=False, score_min=0):
    result = grasp.infer([{"label": lb, "mask_path": path} for lb, path in items], top=top,
                         no_collision=no_collision, score_min=score_min)
    return (result.data, None) if result.ok else (None, result.message)
