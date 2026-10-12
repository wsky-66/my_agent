"""Load one grasp model; infer all masks using the supplied immutable RGB-D scene."""
import sys

from core.results import ToolResult
from workers.protocol import options, serve


class GraspBackend:
    def __init__(self, opts):
        sys.path.insert(0, opts["controller_dir"])
        self.opts = opts
        self.model = None
        self.pipeline = None
        self.visualizer = None

    def __call__(self, action, params):
        import cv2
        import numpy as np
        if action != "infer":
            raise ValueError("未知抓取推理动作")
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
        from core.grasp_geometry import rigid_transform
        transform = rigid_transform(transform)
        if len(k)!=9 or not np.isfinite(k).all() or k[0]<=0 or k[4]<=0:
            raise ValueError("相机内参无效")
        score_min = float(params.get("score_min", 0))
        if not np.isfinite(score_min) or not 0 <= score_min <= 1:
            raise ValueError("score_min 必须在 0~1 之间")
        if self.pipeline is None:
            name = self.opts.get("grasp_backend", "graspnet")
            if name == "grcnn":
                from workers.grcnn_pipeline import GrCNNPipeline
                self.pipeline = GrCNNPipeline(self.opts)
            elif name == "graspnet":
                import graspnet_client as backend
                from workers.grasp_pipeline import GraspPipeline
                if self.model is None:
                    self.model = backend.load_model()
                self.pipeline = GraspPipeline(backend, self.model)
            else:
                raise ValueError(f"未知抓取后端 {name}")
        masks = []
        for item in params["items"]:
            label, mask_path = item["label"], item.get("mask_path")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None
            if mask_path and (mask is None or mask.shape != depth.shape):
                raise ValueError(f"{label} mask 尺寸不匹配")
            if mask is not None:
                mask = mask > 0
            masks.append(mask)
        self.pipeline.prepare(color, depth, k, transform)
        result = {"_diagnostics": {}}
        for item, mask in zip(params["items"], masks):
            grasps, diagnostics = self.pipeline.infer(mask,
                top=max(1, min(int(params.get("top", 3)), 50)), score_min=score_min,
                no_collision=bool(params.get("no_collision", False)))
            for g in grasps:
                g.update(frame=snapshot["base_frame"], calibration_id=snapshot["calibration_id"],
                         snapshot_timestamp=snapshot["timestamp"], ee_frame=f"{self.opts['robot_name']}/ee_gripper_link")
            result[item["label"]] = grasps
            result["_diagnostics"][item["label"]] = diagnostics
        try:
            from workers.grasp_visualization import GraspVisualizer
            if self.visualizer is None:
                self.visualizer = GraspVisualizer()
            candidates = [dict(g, label=item["label"]) for item in params["items"] for g in result[item["label"]]]
            points, colors = self.pipeline.segmented_cloud(masks)
            self.visualizer.publish_scene(points, colors, candidates, snapshot["base_frame"])
        except Exception as exc:
            from logger import error
            error("RViz 抓取结果发布失败", exc)
            result["_visualization_error"] = str(exc)
        return ToolResult.success("抓取推理完成" + ("，RViz 发布失败" if "_visualization_error" in result else "，点云和抓取 pose 已发布到 RViz"), result)

    def close(self):
        if self.visualizer is not None:
            self.visualizer.close()


if __name__ == "__main__":
    backend = GraspBackend(options())
    serve(backend, backend.close)
