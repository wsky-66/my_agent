"""长程任务预扫工具：在机械臂运动前，一次性定位并预计算任务中所有物体。

把「要抓的物体 + 放置目标」合并成一次开放集检测，逐物体分割/定位，再用一次 GraspNet
批量推理得到抓取位姿，结果按语义名(label)绑定，供后续执行阶段直接复用，
避免抓取过程中因机械臂遮挡相机而反复调用视觉工具。

产物落盘到 /tmp/agent_scene_<时间戳>/：
  color 帧(检测/分割同帧)、masks/<label>.png、plan.json
"""

import json
import os
import re
from datetime import datetime

from tools import registry
from . import camera
from . import sam2 as sam2_svc
from .grounding_dino_tools import run_detection
from .sam2_tools import segment_object
from .position_tools import object_position_raw
from .grasp_tools import detect_grasp_batch

_SCENE_ROOT = "/tmp/agent_scene"
_GRASP_TOP = 3

# 最近一次预扫计划，供执行阶段/调试查询
_LAST_PLAN: dict = {}


def _slug(name: str) -> str:
    """把语义名转成安全的文件名片段。"""
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "_", str(name).strip().lower())
    return s.strip("_") or "obj"


def _norm(s: str) -> str:
    """归一化用于匹配：去空白/下划线/连字符，转小写。"""
    return re.sub(r"[\s_\-]+", "", str(s).strip().lower())


def _pick_bbox(objects: list, label: str):
    """从检测结果里给 label 挑 bbox(置信度最高)，返回 (bbox, 命中数)。"""
    n = _norm(label)
    hits = []
    for o in objects:
        c = _norm(o.get("category", ""))
        if c and (c == n or n in c or c in n):
            b = o.get("bbox")
            if b and len(b) == 4:
                hits.append((float(o.get("score", 0)), [int(v) for v in b]))
    if not hits:
        return None, 0
    hits.sort(key=lambda x: x[0], reverse=True)
    return hits[0][1], len(hits)


def _fmt_xyz(v):
    return None if not v else f"({v[0]:.3f},{v[1]:.3f},{v[2]:.3f})"


def _format_plan(records: list, scene: str, missing: list) -> str:
    hit = sum(1 for r in records if r.get("ok"))
    lines = [f"预扫完成（同一帧，base_link 米）：{hit}/{len(records)} 命中"]
    for r in records:
        lb = r["label"]
        if not r.get("ok"):
            why = r.get("error", "未检测到")
            lines.append(f"- {lb}: 失败({why})")
            continue
        xyz = _fmt_xyz(r.get("base_xyz"))
        sz = r.get("size")
        sz_str = f" 尺寸=({sz[0]:.2f},{sz[1]:.2f},{sz[2]:.2f})" if sz else ""
        grasps = r.get("grasps") or []
        if grasps:
            g = grasps[0]
            p = g["position"]
            ry = g["rpy"]
            g_str = (f" 抓取#0: pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
                     f"rpy=({ry[0]:.1f},{ry[1]:.1f},{ry[2]:.1f}) w={g['width']:.3f}")
        else:
            g_str = " (无抓取候选/放置目标)"
        inst = f" [{r['instances']}个实例]" if r.get("instances", 1) > 1 else ""
        lines.append(f"- {lb}{inst}: xyz={xyz}{sz_str}{g_str}")
    if missing:
        lines.append(f"漏检: {', '.join(missing)}（可 recognize_camera 确认后补扫）")
    else:
        lines.append("漏检: 无")
    lines.append(f"场景: {scene}")
    lines.append("执行阶段直接复用以上坐标与抓取位姿，不要再调用视觉工具。")
    return "\n".join(lines)


@registry.register(
    name="scan_objects",
    description="长程任务预扫：机械臂运动前一次性定位并预计算所有物体(要抓的+放置目标)的 "
               "bbox/mask/base_link 坐标/抓取位姿，按语义名绑定返回。之后执行阶段直接复用这些"
               "坐标，避免抓取过程中因机械臂遮挡相机反复调用视觉工具。labels 传物体英文名列表"
               "（如 ['apple','bowl','cup','plate']）。",
)
def scan_objects(labels: list) -> str:
    """
    一次性预扫任务所需的全部物体。

    :param labels: 物体英文名列表，含要抓的物体与放置目标
    """
    if not isinstance(labels, list) or not labels:
        return "失败: 请提供物体英文名列表 labels"
    labels = [str(x).strip() for x in labels if str(x).strip()]
    if not labels:
        return "失败: labels 不能为空"
    if not camera.ready():
        return "相机未运行，请先 start_camera 再预扫"
    if not sam2_svc.ready():
        sam2_svc.start()

    scene = os.path.join(_SCENE_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(os.path.join(scene, "masks"), exist_ok=True)

    from .camera_tools import _grab_frame
    frame = _grab_frame()
    if not frame:
        return "抓帧失败，请确认相机已启动"

    # 1) 一次多类别开放集检测（同一帧）
    objects, err = run_detection(".".join(labels), image_path=frame)
    if err:
        return f"检测失败: {err}"

    records, missing = [], []
    grasp_items = []
    for lb in labels:
        bbox, count = _pick_bbox(objects, lb)
        if bbox is None:
            missing.append(lb)
            records.append({"label": lb, "ok": False})
            continue
        rec = {"label": lb, "ok": True, "bbox": bbox, "instances": count,
               "mask_path": None, "center": None, "base_xyz": None,
               "size": None, "grasps": []}
        # 2) 分割出 per-object mask
        seg, serr = segment_object(box=bbox, image_path=frame,
                                   out=os.path.join(scene, "masks", f"{_slug(lb)}.png"))
        if serr:
            rec["ok"] = False
            rec["error"] = serr
            records.append(rec)
            continue
        rec["mask_path"] = seg.get("mask_path")
        rec["center"] = seg.get("center")
        # 3) 反投影到 base_link
        pos, perr = object_position_raw(mask_path=rec["mask_path"])
        if not perr:
            rec["base_xyz"] = [pos.get("x"), pos.get("y"), pos.get("z")]
            rec["size"] = pos.get("size")
        grasp_items.append((_slug(lb), rec["mask_path"], rec))
        records.append(rec)

    # 4) 一次 GraspNet 批量推理（模型只加载一次、共用一帧），按 label 绑定
    if grasp_items:
        grasps_by_label, gerr = detect_grasp_batch(
            [(label, mp) for label, mp, _ in grasp_items], top=_GRASP_TOP)
        if not gerr:
            for label, _, rec in grasp_items:
                rec["grasps"] = grasps_by_label.get(label, [])
        else:
            for _, _, rec in grasp_items:
                rec["grasp_error"] = gerr

    plan = {
        "scene": scene,
        "frame": frame,
        "created": datetime.now().isoformat(),
        "labels": labels,
        "objects": records,
    }
    try:
        with open(os.path.join(scene, "plan.json"), "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
    except OSError:
        pass
    global _LAST_PLAN
    _LAST_PLAN = plan
    return _format_plan(records, scene, missing)
