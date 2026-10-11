"""Model adapter to the persistent GraspNet service."""
from pathlib import Path
from config import config
from core.results import ToolResult
from tools import registry
from services import grasp


@registry.register(description="基于当前 RGB-D 场景和 mask 推理抓取候选，返回 base_link 位置（米）、rpy（度）、score 和 width。模型常驻复用。no_collision 默认 false。")
def detect_grasp(top: int = 5, no_collision: bool = False, score_min: float = 0,
                 use_mask: bool = True, mask_path: str = None) -> ToolResult:
    path = mask_path or config.WORKSPACE_MASK
    if use_mask and not Path(path).is_file():
        return ToolResult.failure("MISSING_MASK", "请先 segment_mask 或提供有效 mask_path")
    result = grasp.infer([{"label": "object", "mask_path": path if use_mask else None}],
                         top=top, no_collision=no_collision, score_min=score_min)
    if result.ok:
        result.data = result.data["object"]
        if not result.data:
            return ToolResult.failure("NO_GRASP", "没有可用抓取候选")
    return result


def detect_grasp_raw(*args, **kwargs):
    result = detect_grasp(*args, **kwargs)
    return (result.data, None) if result.ok else (None, result.message)


def detect_grasp_batch(items, top=3, no_collision=False, score_min=0):
    result = grasp.infer([{"label": lb, "mask_path": path} for lb, path in items], top=top,
                         no_collision=no_collision, score_min=score_min)
    return (result.data, None) if result.ok else (None, result.message)
