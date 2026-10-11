"""Load GraspNet once; infer all masks using the supplied immutable RGB-D scene."""
from types import SimpleNamespace
import sys

from core.results import ToolResult
from workers.protocol import options, serve


class GraspBackend:
    def __init__(self, opts):
        sys.path.insert(0, opts["controller_dir"])
        self.model = None

    def __call__(self, action, params):
        import cv2
        import numpy as np
        import torch
        import graspnet_client as backend
        if action != "infer":
            raise ValueError("未知 GraspNet 动作")
        snapshot = params["snapshot"]
        color = cv2.imread(snapshot["image_path"])
        if color is None:
            raise ValueError("场景图像不存在")
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth = np.load(snapshot["depth_path"], allow_pickle=False)
        if color.shape[:2] != depth.shape:
            raise ValueError("彩色与深度尺寸不一致")
        k = snapshot["intrinsics"]
        transform = np.asarray(snapshot["extrinsics"], dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("标定矩阵无效")
        if self.model is None:
            self.model = backend.load_model()
        args = SimpleNamespace(top=max(1, min(int(params.get("top", 3)), 50)),
                               no_collision=params.get("no_collision", False), no_z=False,
                               score_min=float(params.get("score_min", 0)), grip_z_min=backend.GRIP_Z_MIN)
        result = {}
        for item in params["items"]:
            label, mask_path = item["label"], item.get("mask_path")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None
            if mask_path and (mask is None or mask.shape != depth.shape):
                raise ValueError(f"{label} mask 尺寸不匹配")
            if mask is not None:
                mask = mask > 0
            np.random.seed(0)
            torch.manual_seed(0)
            gg, _ = backend.run_pipeline(*self.model, color, depth, k[0], k[4], k[2], k[5], transform, mask, args)
            grasps = []
            from scipy.spatial.transform import Rotation
            for i in range(len(gg)):
                rpy = Rotation.from_matrix(gg.rotation_matrices[i]).as_euler("xyz", degrees=True)
                rpy = backend.normalize_rpy(*rpy)
                grasps.append({"rank": i, "score": float(gg.scores[i]), "position": gg.translations[i].tolist(),
                               "rpy": list(map(float, rpy)), "width": float(gg.widths[i])})
            result[label] = grasps
        return ToolResult.success("抓取推理完成", result)


if __name__ == "__main__":
    serve(GraspBackend(options()))
