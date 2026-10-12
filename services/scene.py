"""Scene preparation and explicit cache lifetime. No dependency on model tool modules."""
from dataclasses import dataclass, field
from pathlib import Path
import json
import math
import re
import time
import uuid

from config import config
from core.results import ToolResult
from core.execution import is_cancelled
from logger import measure
from services import camera, grounding, grasp, segmentation
from services.perception import calibrate, position


def normalize(label):
    return " ".join(label.lower().replace("_", " ").replace("-", " ").split())


def detection_candidates(objects, label):
    """Count distinct, plausible instances instead of treating all hits equally."""
    words = label.split()
    hits = []
    for obj in objects:
        if normalize(str(obj.get("category", ""))).split()[-len(words):]!=words:
            continue
        box = obj.get("bbox")
        if (not isinstance(box,list) or len(box)!=4 or
                any(type(v) not in (int,float) or not math.isfinite(v) for v in box) or
                box[0]>=box[2] or box[1]>=box[3]):
            continue
        score = obj.get("score",0.)
        if type(score) not in (int,float) or not math.isfinite(score):continue
        hits.append(dict(category=obj["category"],bbox=box,score=float(score)))
    hits.sort(key=lambda o:o["score"],reverse=True)
    distinct = []
    for hit in hits:
        a = hit["bbox"]
        duplicate = False
        for other in distinct:
            b = other["bbox"]
            intersection = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
            if intersection/union>=.75:duplicate=True;break
        if not duplicate:distinct.append(hit)
    if len(distinct)>1 and distinct[0]["score"]>=.5 and distinct[1]["score"]<distinct[0]["score"]*.5:
        return distinct[:1],hits,"confidence_dominant"
    return distinct,hits,"single" if len(distinct)==1 else "ambiguous"


@dataclass
class ScenePlan:
    scene_id: str
    snapshot: dict
    objects: dict
    created: float = field(default_factory=time.time)
    moved: set = field(default_factory=set)
    invalid: bool = False

    def public(self):
        return {"scene_id": self.scene_id, "created": self.created, "frame": self.snapshot["base_frame"],
                "image_path":self.snapshot["image_path"],
                "objects": self.objects, "plan_path": str(Path(self.snapshot["image_path"]).parent / "plan.json")}


class SceneStore:
    def __init__(self):
        self.plan = None

    def clear(self):
        self.plan = None

    def get(self, scene_id):
        plan = self.plan
        if plan is None or plan.scene_id != scene_id:
            raise ValueError("场景不存在，请重新扫描")
        if plan.invalid or time.time()-plan.snapshot["created"] > config.SCENE_MAX_AGE:
            raise ValueError("场景失效或已过期，请重新扫描")
        return plan


store = SceneStore()


def scan(labels, grasp_labels=None):
    if not labels or any(not isinstance(s, str) or not s.strip() for s in labels):
        return ToolResult.failure("INVALID_ARGUMENT", "labels 必须为非空物体名称列表")
    labels = list(dict.fromkeys(normalize(s) for s in labels))
    pick_labels = set(labels if grasp_labels is None else map(normalize, grasp_labels))
    if not pick_labels.issubset(labels):
        return ToolResult.failure("INVALID_ARGUMENT", "grasp_labels 必须包含在 labels 中")
    store.clear()
    captured = camera.capture()
    if not captured.ok:
        return captured
    snapshot = calibrate(captured.data)
    detected = grounding.detect(".".join(labels), snapshot["image_path"])
    if not detected.ok:
        return detected
    records, items = {}, []
    folder = Path(snapshot["image_path"]).parent
    (folder/"grounding.json").write_text(json.dumps(detected.data,ensure_ascii=False,indent=2),encoding="utf-8")
    for i, label in enumerate(labels):
        hits,raw,selection = detection_candidates(detected.data["objects"],label)
        rec = {"object_id": f"object_{i}", "label": label, "ok": False, "grasps": [], "instances": len(hits),
               "raw_instances":len(raw),"detections":raw,"selection":selection}
        records[label] = rec
        if not hits:
            rec.update(error_code="OBJECT_NOT_FOUND", error="未检测到目标")
            continue
        if len(hits) > 1:
            rec.update(error_code="AMBIGUOUS_OBJECT", error="检测到多个同名物体，需要明确目标")
            continue
        best = hits[0]
        rec.update(bbox=best["bbox"], detection_score=best.get("score", 0))
        items.append({"label": label, "box": best["bbox"], "dilate_cm": 0,
                      "out": str(folder / "masks" / f"object_{i}.png")})
    if items:
        with measure("vision.segmentation"):
            segmented = segmentation.batch(snapshot["image_path"], items)
        if not segmented.ok:
            return segmented
        for item in items:
            label = item["label"]
            rec = records[label]
            seg = segmented.data[label]
            rec.update(mask_path=seg["mask_path"], center=seg.get("center"))
            located = position(mask_path=seg["mask_path"], snapshot=snapshot)
            if located.ok:
                rec.update(ok=True, base_xyz=located.data["position"], size=located.data["size"])
            else:
                rec.update(error_code=located.error_code, error=located.message)
        picks = [{"label": lb, "mask_path": r["mask_path"], "box": r["bbox"]} for lb, r in records.items()
                 if lb in pick_labels and r["ok"]]
        with measure("vision.grasp"):
            inferred = grasp.infer(picks, snapshot, top=3, score_min=config.GRASP_SCORE_MIN)
        if not inferred.ok:
            return ToolResult.failure(inferred.error_code,"抓取模型推理失败："+inferred.message,inferred.data)
        for item in picks:
            rec = records[item["label"]]
            rec["grasps"] = inferred.data.get(item["label"], [])
            rec["grasp_diagnostics"] = inferred.data.get("_diagnostics", {}).get(item["label"], {})
            if inferred.data.get("_visualization_error"):
                rec["grasp_visualization_error"] = inferred.data["_visualization_error"]
    if is_cancelled():
        return ToolResult.failure("TASK_CANCELLED", "任务已中断，场景不再缓存")
    plan = ScenePlan(uuid.uuid4().hex, snapshot, records)
    store.plan = plan
    public = plan.public()
    (folder / "plan.json").write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")
    missing = [lb for lb, rec in records.items() if not rec["ok"]]
    if missing:
        return ToolResult.failure("SCENE_INCOMPLETE", "部分物体未能定位: " + ", ".join(missing), public)
    return ToolResult.success(f"预扫完成：{len(records)} 个物体，共用同一 RGB-D 场景", public)
