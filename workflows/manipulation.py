"""Validated pick/place workflow: one model request, checked sequential actions."""
import math

from config import config
from core.execution import is_cancelled, report
from core.results import ToolResult
from services import arm, scene


def _grasp(value, snapshot):
    """Validate the complete inference/execution contract before any command."""
    import numpy as np
    from core.grasp_geometry import PIPELINE_VERSION, ee_position, rigid_transform,rotation_rpy
    if (value.get("pipeline_version") != PIPELINE_VERSION or value.get("collision_checked") is not True or
            value.get("execution_ready") is not True):
        raise ValueError("候选没有通过当前版本的完整碰撞检查，请重新扫描")
    if not snapshot.get("base_frame") or not snapshot.get("calibration_id") or "timestamp" not in snapshot:
        raise ValueError("场景缺少坐标系、标定或时间戳")
    if (value.get("frame") != snapshot.get("base_frame") or
            value.get("calibration_id") != snapshot.get("calibration_id") or
            value.get("snapshot_timestamp") != snapshot.get("timestamp") or
            value.get("ee_frame") != f"{config.ROBOT_NAME}/ee_gripper_link"):
        raise ValueError("抓取候选与当前场景、标定或末端坐标系不一致")
    rotation = np.asarray(value["rotation"], dtype=float)
    transform = np.eye(4); transform[:3,:3] = rotation
    rigid_transform(transform)
    position = _xyz(value["position"]); ee = _xyz(value["ee_position"])
    depth, tip = float(value["depth"]), float(value["tip_from_ee"])
    width, offset = float(value["width"]), float(value["gripper_gap_offset"])
    opening, closing = float(value["gripper_open"]), float(value["gripper_close"])
    approach = float(value["approach_distance"])
    values = (depth, tip, width, offset, opening, closing, approach, value["score"], value["table_clearance"])
    if not all(math.isfinite(v) for v in values):
        raise ValueError("抓取候选包含无效数字")
    if (not any(abs(depth-v)<1e-6 for v in (.01,.02,.03,.04)) or not .01<tip<.02 or
            not .005<offset<.015 or not .015<=closing<=opening<=.037 or
            abs(opening-(width/2+offset))>1e-6 or abs(closing-max(.015,offset+width/2.4-.0005))>1e-6 or
            not .05<=approach<=.15 or value["table_clearance"]<.003):
        raise ValueError("抓取深度、开口、进近距离或桌面净空无效")
    if not np.allclose(ee, ee_position(position, rotation, depth, tip), atol=1e-6):
        raise ValueError("抓取接触位置没有正确转换到 ee_gripper_link")
    rpy = rotation_rpy(rotation)
    pre = (np.asarray(ee)-approach*rotation[:,0]).tolist()
    return ee, pre, rpy, opening, closing


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


def execute(actions, scene_id=None, return_to_sleep=True, allow_fallback=False):
    actions = _actions(actions)  # validate the entire request before any physical command
    if allow_fallback:
        return ToolResult.failure("FALLBACK_DISABLED", "已移除无候选时的重心盲抓；请使用经过检查的抓取候选")
    completed, steps = [], []

    def step(name, **args):
        if is_cancelled():
            return ToolResult.failure("TASK_CANCELLED", "任务已中断，停止后续步骤")
        report(name, args)
        result = arm.call(name, **args)
        steps.append({"name": name, "ok": result.ok, "elapsed": result.elapsed})
        report(name, args, result)
        return result

    if scene_id is None:
        settled = step("go_sleep")
        if not settled.ok:
            return settled
        if is_cancelled():
            return ToolResult.failure("TASK_CANCELLED", "任务已中断")
        labels = list(dict.fromkeys([a["object"] for a in actions] + [a["target"] for a in actions if a["target"]]))
        report("scan_objects", {"labels": labels})
        prepared = scene.scan(labels, [a["object"] for a in actions])
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
        candidates = [g for g in obj["grasps"] if g["score"] >= config.GRASP_SCORE_MIN
                      and g.get("execution_ready") is True and g.get("collision_checked") is True]
        if candidates:
            try:
                grasps = [_grasp(g, plan.snapshot) for g in sorted(candidates,key=lambda g: -g["score"])[:12]]
            except (ValueError, TypeError, KeyError) as exc:
                return ToolResult.failure("INVALID_GRASP", str(exc))
        else:
            return ToolResult.failure("NO_GRASP", f"{action['object']} 无经过完整检查的抓取候选",
                                      {"diagnostics": obj.get("grasp_diagnostics", {})})
        target = _xyz(plan.objects[action["target"]]["base_xyz"]) if action["target"] else action["target_position"]
        prepared_actions.append((action, grasps, target))
    for action, grasps, target in prepared_actions:
        try:
            scene.store.get(scene_id)  # cache lifetime is also checked between objects
        except ValueError as exc:
            return ToolResult.failure("SCENE_EXPIRED", str(exc), {"completed": completed})
        checked = step("inspect_grasps", candidates=[dict(position=g[0],pre_position=g[1],rpy=g[2],opening=g[3])
                                                     for g in grasps])
        if not checked.ok:
            if is_cancelled() or completed:
                plan.invalid = True
            return ToolResult.failure(checked.error_code, checked.message,
                                      {"scene_id": scene_id, "completed": completed, "preflight": checked.data})
        selected = checked.data.get("selected_index")
        if type(selected) is not int or not 0<=selected<len(grasps):
            return ToolResult.failure("INVALID_PREFLIGHT", "预检未返回有效候选编号，停止动作")
        # Read-only planning can take time; do not execute an expired snapshot.
        try:
            scene.store.get(scene_id)
        except ValueError as exc:
            return ToolResult.failure("SCENE_EXPIRED", str(exc), {"completed": completed})
        xyz, pre, rpy, opening, closing = grasps[selected]
        x, y, z = xyz
        orientation = dict(roll=rpy[0],pitch=rpy[1],yaw=rpy[2])
        sequence = [
            ("gripper_set", {"position": opening}),
            ("move_arm", dict(x=pre[0], y=pre[1], z=pre[2], **orientation)),
            ("move_arm", dict(x=x, y=y, z=z, cartesian=True, **orientation)),
            ("gripper_set", {"position": closing}),
            ("move_arm", dict(x=pre[0], y=pre[1], z=pre[2], cartesian=True, **orientation)),
        ]
        if target is not None:
            tx, ty, tz = target
            sequence += [
                ("move_arm", dict(x=tx, y=ty, z=tz+0.20, **orientation)),
                ("move_arm", dict(x=tx, y=ty, z=tz+0.04, cartesian=True, **orientation)),
                ("gripper_set", {"position": 0.037}),
                ("move_arm", dict(x=tx, y=ty, z=tz+0.20, cartesian=True, **orientation)),
            ]
        for index, (name, args) in enumerate(sequence):
            try:
                result = step(name, **args)
            except BaseException:
                plan.invalid = True
                raise
            if not result.ok:
                plan.invalid = True
                return ToolResult.failure(result.error_code, f"{action['object']} 在 {name} 步骤停止：{result.message}",
                        {"scene_id": scene_id, "completed": completed, "failed_step": index+1, "steps": steps,
                         "failure": result.data,
                         "physical_state": "已执行动作不回滚，需要确认当前机械臂和物体状态"})
        plan.moved.add(action["object"])
        completed.append(action["object"])
    if return_to_sleep:
        result = step("go_sleep")
        if not result.ok:
            plan.invalid = True
            return ToolResult.failure(result.error_code, "抓放步骤完成，但收拢失败：" + result.message,
                                      {"completed": completed, "steps": steps})
    return ToolResult.success("抓放动作序列完成", {"completed": completed, "scene_id": scene_id, "steps": steps})
