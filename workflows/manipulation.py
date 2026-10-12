"""Validated pick/place workflow: one model request, checked sequential actions."""
import math
import time

from config import config
from core.execution import current, is_cancelled, report
from core.results import ToolResult
from services import arm, scene


def _grasp(value, snapshot):
    """Use one model pose, normalising the parallel-jaw roll before MoveIt."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from core.grasp_geometry import preferred_grasp_rotation, rotation_rpy, rigid_transform
    if value.get("execution_ready") is False:
        raise ValueError("候选仅供查看，请重新扫描")
    for key, expected in (("frame",snapshot.get("base_frame")),
                          ("calibration_id",snapshot.get("calibration_id")),
                          ("snapshot_timestamp",snapshot.get("timestamp"))):
        if key in value and value[key]!=expected:
            raise ValueError("抓取 pose 与当前场景不一致，请重新扫描")
    position = _xyz(value["ee_position"] if "ee_position" in value else value["position"])
    rotation = (np.asarray(value["rotation"],dtype=float) if "rotation" in value else
                Rotation.from_euler("xyz",_xyz(value["rpy"]),degrees=True).as_matrix())
    t = np.eye(4); t[:3,:3] = rotation; rigid_transform(t)
    rotation = preferred_grasp_rotation(rotation)
    mode = "grcnn" if value.get("pipeline")=="grcnn" else "graspnet"
    return position, rotation_rpy(rotation), dict(value, rotation=rotation.tolist(),mode=mode)


def _xyz(value):
    if not isinstance(value, list) or len(value) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
        raise ValueError("坐标必须为三个有限数字 [x,y,z]，单位米")
    return list(value)


def _actions(actions):
    if not isinstance(actions, list) or not actions:
        raise ValueError("actions 不能为空")
    result, sources = [], set()
    for value in actions:
        if not isinstance(value, dict) or set(value) - {"object", "target", "target_position"}:
            raise ValueError("每个动作只支持 object、target 或 target_position")
        source = value.get("object")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("每个动作必须指定 object 英文名")
        source = scene.normalize(source)
        if source in sources:
            raise ValueError("同一物体不能重复使用旧坐标抓取，请拆分任务并重新扫描")
        sources.add(source)
        target = value.get("target")
        xyz = value.get("target_position")
        if target is not None and xyz is not None:
            raise ValueError("target 与 target_position 只能选一个")
        if target is not None:
            if not isinstance(target, str) or not target.strip():
                raise ValueError("target 必须是非空物体名称")
            target = scene.normalize(target)
            if target in sources:
                raise ValueError("放置目标包含本任务已移动的物体，请重新扫描后使用")
        result.append({"object": source, "target": target,
                       "target_position": _xyz(xyz) if xyz is not None else None})
    if len(result) > 1 and any(not a["target"] and a["target_position"] is None for a in result[:-1]):
        raise ValueError("多物体任务中，继续抓下一个物体前必须明确前一个物体的放置目标")
    return result


def execute(actions, scene_id=None, return_to_sleep=True, allow_fallback=True, detection_hints=None):
    actions = _actions(actions)  # validate the entire request before any physical command
    labels = list(dict.fromkeys([a["object"] for a in actions] + [a["target"] for a in actions if a["target"]]))
    if detection_hints is not None:
        scene.build_prompt(labels, detection_hints)
    completed, steps = [], []

    def step(name, pause=0., **args):
        if is_cancelled():
            return ToolResult.failure("TASK_CANCELLED", "任务已中断，停止后续步骤")
        report(name, args)
        result = arm.call(name, **args)
        if result.ok and pause:
            started, elapsed = time.monotonic(), result.elapsed
            context = current.get()
            if context is not None:
                cancelled = context.cancelled.wait(pause)
            else:
                time.sleep(pause)
                cancelled = is_cancelled()
            if cancelled:
                result = ToolResult.failure("TASK_CANCELLED", "松爪后等待已中断，停止后续步骤", result.data)
            result.elapsed = round(elapsed + time.monotonic() - started, 4)
        steps.append({"name": name, "ok": result.ok, "elapsed": result.elapsed})
        report(name, args, result)
        return result

    if scene_id is None:
        settled = step("go_sleep")
        if not settled.ok:
            return settled
        if is_cancelled():
            return ToolResult.failure("TASK_CANCELLED", "任务已中断")
        report("scan_objects", {"labels": labels})
        options = {"detection_hints": detection_hints} if detection_hints is not None else {}
        prepared = scene.scan(labels, [a["object"] for a in actions], **options)
        report("scan_objects", {"labels": labels}, prepared)
        if not prepared.ok:
            return prepared
        scene_id = prepared.data["scene_id"]
    plan = scene.store.get(scene_id)
    prepared_actions = []
    for action in actions:
        labels = [action["object"]] + ([action["target"]] if action["target"] else [])
        for label in labels:
            rec = plan.objects.get(label)
            if not rec or not rec["ok"] or label in plan.moved:
                return ToolResult.failure("INVALID_SCENE", f"{label} 坐标不可用或物体已移动，请重新扫描")
        obj = plan.objects[action["object"]]
        candidates = [g for g in obj["grasps"] if g["score"] >= config.GRASP_SCORE_MIN]
        if candidates:
            best = max(candidates, key=lambda g: g["score"])
            try:
                position, rpy, debug = _grasp(best, plan.snapshot)
            except (ValueError, TypeError, KeyError) as exc:
                return ToolResult.failure("INVALID_GRASP", str(exc))
            grasp_z = position[2]
        elif allow_fallback:
            position, rpy = _xyz(obj["base_xyz"]), [0., 50., 0.]
            grasp_z = position[2]-.03
            debug = dict(mode="fallback",position=[position[0],position[1],grasp_z],rpy=rpy)
        else:
            return ToolResult.failure("NO_GRASP", f"{action['object']} 无可用抓取候选")
        target = _xyz(plan.objects[action["target"]]["base_xyz"]) if action["target"] else action["target_position"]
        debug["label"] = action["object"]
        prepared_actions.append((action, position, grasp_z, rpy, target, debug))
    strategies = []
    for action, xyz, grasp_z, rpy, target, debug in prepared_actions:
        try:
            scene.store.get(scene_id)  # cache lifetime is also checked between objects
        except ValueError as exc:
            return ToolResult.failure("SCENE_EXPIRED", str(exc), {"completed": completed})
        x, y, z = xyz
        orientation = dict(roll=rpy[0],pitch=rpy[1],yaw=rpy[2])
        strategy = {"object":action["object"],"mode":debug["mode"],"position":[x,y,grasp_z],"rpy":rpy}
        strategies.append(strategy)
        message = {"graspnet":"使用 GraspNet pose", "grcnn":"使用 GR-CNN pose",
                   "fallback":"无抓取 pose，使用原版定位坐标和固定姿态兜底"}[debug["mode"]]
        if debug["mode"] != "fallback":
            from core.grasp_geometry import STOCK_TIP_FROM_EE
            tip = [xyz[i]+debug["rotation"][i][0]*STOCK_TIP_FROM_EE for i in range(3)]
            strategy["tip_position"] = tip
            message += f" · 末端目标 z={grasp_z*100:.2f} cm，指尖目标 z={tip[2]*100:.2f} cm"
        report("grasp_strategy",strategy,ToolResult.success(message))
        sequence = [
            ("gripper_set", {"position": 0.037}, 0.),
            ("move_arm", dict(x=x, y=y, z=z+0.10, **orientation, grasp_debug=debug), 0.),
            ("move_arm", dict(x=x, y=y, z=grasp_z, cartesian=True), 0.),
            ("gripper_set", {"position": 0.015}, 0.),
            ("move_arm", dict(x=x, y=y, z=z+0.08, cartesian=True, allow_partial=True), 0.),
        ]
        if target is not None:
            tx, ty, tz = target
            sequence += [
                ("move_arm", dict(x=tx, y=ty, z=tz+0.10, planned=True), 0.),
                ("move_arm", dict(x=tx, y=ty, z=tz+0.04, cartesian=True), 0.),
                # Allow the fingers to finish releasing before retreating.
                ("gripper_set", {"position": 0.037}, 0.5),
                ("move_arm", dict(x=tx, y=ty, z=tz+0.10, cartesian=True, allow_partial=True), 0.),
            ]
        for index, (name, args, pause) in enumerate(sequence):
            try:
                result = step(name, pause=pause, **args)
            except BaseException:
                plan.invalid = True
                raise
            if not result.ok:
                plan.invalid = True
                return ToolResult.failure(result.error_code, f"{action['object']} 在 {name} 步骤停止：{result.message}",
                        {"scene_id": scene_id, "completed": completed, "failed_step": index+1, "steps": steps, "failure":result.data,
                         "physical_state": "已执行动作不回滚，需要确认当前机械臂和物体状态"})
        plan.moved.add(action["object"])
        completed.append(action["object"])
    if return_to_sleep:
        result = step("go_sleep")
        if not result.ok:
            plan.invalid = True
            return ToolResult.failure(result.error_code, "抓放步骤完成，但收拢失败：" + result.message,
                                      {"completed": completed, "steps": steps})
    return ToolResult.success("抓放动作序列完成", {"completed": completed, "scene_id": scene_id, "strategies":strategies, "steps": steps,
                            "execution_completed":True,"grasp_verified":False,"placement_verified":False})
