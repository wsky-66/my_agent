"""Segmentation service; one call can segment multiple boxes in one encoded image."""
from config import config
from pathlib import Path
import math
import time
from core.results import ToolResult
from services import camera
from services.runtime import worker

client = worker("sam", config.VISION_PYTHON, {"server_path": config.SAM2_SERVER}, timeout=600)
_mask_sources = {}


def batch(image_path, items):
    result = client.request("segment", {"image_path": image_path, "items": items})
    if result.ok:
        for value in result.data.values():
            path = Path(value["mask_path"]).resolve()
            _mask_sources[str(path)] = (str(Path(image_path).resolve()), path.stat().st_mtime_ns)
    return result


def mask_matches(mask_path, image_path):
    path = Path(mask_path).resolve()
    source = _mask_sources.get(str(path))
    return path.is_file() and source == (str(Path(image_path).resolve()), path.stat().st_mtime_ns)


def segment(pos_points=None, neg_points=None, box=None, image_path=None, out=None, dilate_cm=None):
    dilation = 1.0 if dilate_cm is None else dilate_cm
    if not math.isfinite(dilation) or dilation < 0:
        return ToolResult.failure("INVALID_ARGUMENT", "膨胀距离必须为非负有限数字")
    if image_path is None:
        snapshot = camera.latest()
        if snapshot is None or time.time()-snapshot["created"] > config.SCENE_MAX_AGE:
            result = camera.capture()
            if not result.ok:
                return result
            snapshot = result.data
        image_path = snapshot["image_path"]
    result = batch(image_path, [{"label": "object", "box": box, "points": pos_points,
                                "neg_points": neg_points, "out": out or config.WORKSPACE_MASK,
                                "dilate_cm": dilation}])
    if result.ok:
        result.data = result.data["object"]
        result.data["image_path"] = image_path
    return result


def stop():
    client.stop()
