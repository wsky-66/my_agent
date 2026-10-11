"""SAM2 model and image embedding reuse, including multi-box segmentation."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import math

from core.results import ToolResult
from workers.protocol import options, serve


class SamBackend:
    def __init__(self, opts):
        self.opts = opts
        self.backend = self.predictor = self.device = None
        self.image_key = None

    def load(self):
        if self.predictor is None:
            path = self.opts["server_path"]
            sys.path.insert(0, str(Path(path).parent))
            spec = importlib.util.spec_from_file_location("my_agent_sam_backend", path)
            self.backend = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.backend)
            self.predictor, self.device = self.backend.build_predictor()

    def __call__(self, action, params):
        self.load()
        if action == "warmup":
            return ToolResult.success("SAM2 模型就绪")
        if action != "segment":
            raise ValueError("未知 SAM2 动作")
        import cv2
        import numpy as np
        import torch
        path = Path(params["image_path"])
        key = (str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size)
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("无法读取分割图像")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        context = lambda: torch.autocast("cuda", dtype=torch.float16) if str(self.device).startswith("cuda") else nullcontext()
        if key != self.image_key:
            with torch.inference_mode(), context():
                self.predictor.set_image(image)
            self.image_key = key
        results = {}
        for item in params["items"]:
            out = item["out"]
            dilation = float(item.get("dilate_cm", 1.0))
            if not math.isfinite(dilation) or dilation < 0:
                raise ValueError("膨胀距离必须为非负有限数字")
            px = self.backend.dilate_px(dilation)
            box = item.get("box")
            pos, neg = item.get("points") or [], item.get("neg_points") or []
            if box is None and not pos and not neg:
                if len(params["items"]) != 1:
                    raise ValueError("批量分割必须提供 bbox 或点")
                Path(out).parent.mkdir(parents=True, exist_ok=True)
                value = self.backend._interactive(self.predictor, image, px, out)
                self.image_key = None
                if not value.get("ok"):
                    return ToolResult.failure("SEGMENT_FAILED", value.get("msg", "分割取消"))
                results[item["label"]] = {k: v for k, v in value.items() if k != "ok"}
                continue
            kwargs = {}
            if box is not None:
                if len(box) != 4 or not np.isfinite(box).all() or box[0] >= box[2] or box[1] >= box[3]:
                    raise ValueError("bbox 必须为有效的 [x1,y1,x2,y2]")
                kwargs["box"] = np.asarray(box, dtype=float)
            if pos or neg:
                points = np.asarray(pos + neg, dtype=float)
                if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                    raise ValueError("分割点必须为 [[x,y], ...]")
                kwargs.update(point_coords=points, point_labels=np.asarray([1]*len(pos)+[0]*len(neg)))
            with torch.inference_mode(), context():
                masks, scores, _ = self.predictor.predict(multimask_output=False, **kwargs)
            mask = self.backend.dilate_mask(masks[0], px)
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(out, mask.astype(np.uint8)*255):
                raise RuntimeError("无法保存 mask")
            stats = self.backend.mask_stats(mask)
            results[item["label"]] = {"mask_path": out, "score": float(scores[0]),
                    "coverage": round(float(mask.mean()*100), 1), "center": [stats["cx"], stats["cy"]],
                    "bbox": [stats["x1"], stats["y1"], stats["x2"], stats["y2"]], "area": stats["area"]}
        return ToolResult.success("分割完成", results)


if __name__ == "__main__":
    serve(SamBackend(options()))
