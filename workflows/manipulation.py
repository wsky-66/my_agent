"""Validated pick/place workflow: one model request, checked sequential actions."""
import math

from config import config
from core.execution import is_cancelled, report
from core.results import ToolResult
from services import arm, scene


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


def execute(actions, scene_id=None, return_to_sleep=True, allow_fallback=True):
    actions = _actions(actions)  # validate the entire request before any physical command
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
        candidates = [g for g in obj["grasps"] if g["score"] >= config.GRASP_SCORE_MIN]
        if candidates:
            best = max(candidates, key=lambda g: g["score"])
            position, rpy = _xyz(best["position"]), _xyz(best["rpy"])
        elif allow_fallback:
            position, rpy = _xyz(obj["base_xyz"]), [0, 50, 0]
        else:
            return ToolResult.failure("NO_GRASP", f"{action['object']} 无可用抓取候选")
        target = _xyz(plan.objects[action["target"]]["base_xyz"]) if action["target"] else action["target_position"]
        prepared_actions.append((action, position, rpy, target))
    for action, xyz, rpy, target in prepared_actions:
        try:
            scene.store.get(scene_id)  # cache lifetime is also checked between objects
        except ValueError as exc:
            return ToolResult.failure("SCENE_EXPIRED", str(exc), {"completed": completed})
        x, y, z = xyz
        sequence = [
            ("gripper_set", {"position": 0.037}),
            ("move_arm", dict(x=x, y=y, z=z+0.10, roll=rpy[0], pitch=rpy[1], yaw=rpy[2])),
            ("move_arm", dict(x=x, y=y, z=z-0.03)),
            ("gripper_set", {"position": 0.024}),
            ("move_arm", dict(x=x, y=y, z=z+0.15)),
        ]
        if target is not None:
            tx, ty, tz = target
            sequence += [
                ("move_arm", dict(x=tx, y=ty, z=tz+0.20)),
                ("move_arm", dict(x=tx, y=ty, z=tz+0.04)),
                ("gripper_set", {"position": 0.037}),
                ("move_arm", dict(x=tx, y=ty, z=tz+0.20)),
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
