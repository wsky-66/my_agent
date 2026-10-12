"""Model adapter to the configured persistent grasp service."""
from pathlib import Path
from config import config
from core.results import ToolResult
from tools import registry
from services import grasp


@registry.register(description="基于当前 RGB-D 和物体 mask 运行配置的 GraspNet 或 GR-CNN 推理与模型碰撞过滤。返回 pipeline 表示实际后端，及 position、rotation、rpy、depth、width 和 ee_position；发布分割点云和抓取框供 RViz 查看。no_collision=true 的结果仅供查看，不能用于抓放。模型常驻复用。")
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
