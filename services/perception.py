"""Pure RGB-D geometry, independent of ROS subscriptions and model tool wrappers."""
import hashlib
import json
from pathlib import Path
import time

from config import config
from core.results import ToolResult
from logger import measure
from services import camera, segmentation


def calibrate(snapshot):
    if snapshot.get("extrinsics") is not None:
        return snapshot
    raw = Path(config.CALIBRATION).read_bytes()
    value = json.loads(raw)["method2_nonlinear_ax_zb"]["matrix"]
    snapshot = {**snapshot, "extrinsics": value, "calibration_id": hashlib.sha256(raw).hexdigest(),
                "base_frame": f"{config.ROBOT_NAME}/base_link"}
    path = Path(snapshot["image_path"]).parent / "snapshot.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    return snapshot


def position(px=None, py=None, mask_path=None, radius=12, snapshot=None):
    import cv2
    import numpy as np
    implicit = snapshot is None
    if implicit:
        snapshot = camera.latest()
        if snapshot is None:
            result = camera.capture()
            if not result.ok:
                return result
            snapshot = result.data
    if implicit and mask_path and not segmentation.mask_matches(mask_path, snapshot["image_path"]):
        return ToolResult.failure("MASK_SCENE_MISMATCH", "mask 与当前 RGB-D 场景不一致，请重新分割")
    if time.time() - snapshot["created"] > config.SCENE_MAX_AGE:
        return ToolResult.failure("SCENE_EXPIRED", "场景已过期，请重新扫描")
    with measure("vision.position"):
        snapshot = calibrate(snapshot)
        dm = np.load(snapshot["depth_path"], allow_pickle=False)
        height, width = dm.shape
        if mask_path:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != dm.shape:
                return ToolResult.failure("MASK_MISMATCH", "mask 不存在或尺寸与场景不一致")
            ys, xs = np.where(mask > 0)
        elif px is not None and py is not None:
            px, py, radius = int(px), int(py), max(1, min(int(radius), 40))
            if not 0 <= px < width or not 0 <= py < height:
                return ToolResult.failure("PIXEL_OUT_OF_RANGE", "像素超出图像范围")
            ys, xs = np.mgrid[max(0, py-radius):min(height, py+radius+1),
                              max(0, px-radius):min(width, px+radius+1)]
            ys, xs = ys.ravel(), xs.ravel()
        else:
            return ToolResult.failure("INVALID_ARGUMENT", "请提供像素位置或 mask")
        z = dm[ys, xs]
        valid = np.isfinite(z) & (z >= 0.2) & (z <= 1)
        xs, ys, z = xs[valid], ys[valid], z[valid]
        if not len(z):
            return ToolResult.failure("INVALID_DEPTH", "物体区域没有有效深度")
        k = snapshot["intrinsics"]
        if len(k) != 9 or not np.isfinite(k).all() or k[0] <= 0 or k[4] <= 0:
            return ToolResult.failure("INVALID_INTRINSICS", "相机内参无效")
        points = np.stack([(xs-k[2])/k[0]*z, (ys-k[5])/k[4]*z, z, np.ones_like(z)], -1)
        transform = np.asarray(snapshot["extrinsics"], dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            return ToolResult.failure("INVALID_CALIBRATION", "标定矩阵无效")
        base = (points @ transform.T)[:, :3]
        xyz = base.mean(0).tolist()
        return ToolResult.success("已定位物体", {"position": xyz, "x": xyz[0], "y": xyz[1], "z": xyz[2],
                                  "size": (base.max(0)-base.min(0)).tolist(), "npix": len(z),
                                  "frame": snapshot["base_frame"], "timestamp": snapshot["timestamp"]})
