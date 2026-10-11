"""Persistent grasp inference adapter; all objects share one RGB-D snapshot."""
from config import config
from core.results import ToolResult
from pathlib import Path
import time
from services import camera, segmentation
from services.perception import calibrate
from services.runtime import worker

client = worker("grasp", config.VISION_PYTHON, {"controller_dir": config.CONTROLLER_DIR}, ros=True, timeout=300)


def infer(items, snapshot=None, top=3, no_collision=False, score_min=0):
    if not items:
        return ToolResult.success("无需抓取推理", {})
    snapshot = snapshot or camera.latest()
    if snapshot is None:
        result = camera.capture()
        if not result.ok:
            return result
        snapshot = result.data
    if time.time()-snapshot["created"] > config.SCENE_MAX_AGE:
        return ToolResult.failure("SCENE_EXPIRED", "场景已过期，请重新扫描")
    for item in items:
        mask = item.get("mask_path")
        if mask and not Path(mask).is_file():
            return ToolResult.failure("MISSING_MASK", f"mask 不存在: {mask}")
        if mask and not segmentation.mask_matches(mask, snapshot["image_path"]):
            return ToolResult.failure("MASK_SCENE_MISMATCH", "mask 与当前 RGB-D 场景不一致，请在当前场景重新分割")
    return client.request("infer", {"items": items, "snapshot": calibrate(snapshot), "top": top,
                                   "no_collision": no_collision, "score_min": score_min})


def stop():
    client.stop()
