"""Bounded joint motion checks. Angles are absolute radians, never wrapped."""
from dataclasses import dataclass
import math

from core.health import ARM_JOINTS
from core.results import ToolResult


class MotionRejected(RuntimeError):
    def __init__(self, code, message, data=None):
        super().__init__(message)
        self.code, self.data = code, data or {}


@dataclass(frozen=True)
class MotionPolicy:
    max_joint_change: float = 2.6
    velocity_scaling: float = .25
    timeout: float = 45.

    def __post_init__(self):
        if not math.isfinite(self.max_joint_change) or not 0 < self.max_joint_change <= math.pi:
            raise ValueError("ARM_MAX_JOINT_CHANGE 必须在 0~pi 弧度之间")
        if not math.isfinite(self.velocity_scaling) or not 0 < self.velocity_scaling <= 1:
            raise ValueError("ARM_VELOCITY_SCALING 必须在 0~1 之间")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("ARM_MOTION_TIMEOUT 必须为正数")


def driver_limits(info):
    if info.mode != "position":
        raise MotionRejected("MOTION_MODE_MISMATCH", "驱动不是 position 模式，停止自动位姿移动，请检查电机模式")
    names = list(info.joint_names)
    arrays = (info.joint_lower_limits, info.joint_upper_limits, info.joint_velocity_limits)
    if len(set(names)) != len(names) or any(len(a) != len(names) for a in arrays):
        raise MotionRejected("MOTION_LIMITS_MISSING", "驱动没有提供完整关节限位")
    limits = {n: tuple(float(a[i]) for a in arrays) for i, n in enumerate(names)}
    if any(n not in limits or not all(math.isfinite(v) for v in limits[n]) or
           limits[n][0] >= limits[n][1] or limits[n][2] <= 0 for n in ARM_JOINTS):
        raise MotionRejected("MOTION_LIMITS_MISSING", "驱动关节限位或速度限制无效")
    return {n: limits[n] for n in ARM_JOINTS}


def check_joints(values, limits, *, feedback=False):
    code = "MOTION_FEEDBACK_INVALID" if feedback else "MOTION_JOINT_INVALID"
    if not isinstance(values, dict) or any(n not in values for n in ARM_JOINTS):
        raise MotionRejected(code, "关节数据不完整")
    result = {}
    for n in ARM_JOINTS:
        v = float(values[n])
        lo, hi, _ = limits[n]
        if not math.isfinite(v) or not lo-1e-5 <= v <= hi+1e-5:
            raise MotionRejected(code, f"{n} 的绝对角度超出限位或无效，停止移动", {"joint": n})
        result[n] = v
    return result


def nearest_solution(candidates, start, limits, policy):
    accepted = []
    for candidate in candidates:
        try:
            target = check_joints(candidate, limits)
        except (MotionRejected, TypeError, ValueError):
            continue
        delta = {n: target[n]-start[n] for n in ARM_JOINTS}
        if max(abs(v) for v in delta.values()) > policy.max_joint_change:
            continue
        # Prefer small wrist changes without changing the user's requested pose.
        score = sum((2. if n in ("forearm_roll", "wrist_rotate") else 1.)*v*v for n, v in delta.items())
        accepted.append((score, target))
    if not accepted:
        raise MotionRejected("IK_BRANCH_REJECTED", "没有找到关节变化足够小的有效逆解，停止移动；请拆分位置或朝向变化")
    return min(accepted, key=lambda item: item[0])[1]


def validate_trajectory(trajectory, start, target, limits, policy):
    names = list(trajectory.joint_names)
    if len(names) != len(ARM_JOINTS) or set(names) != set(ARM_JOINTS) or len(trajectory.points) < 2:
        raise MotionRejected("MOTION_PLAN_INVALID", "规划轨迹必须包含完整关节、起点和终点")
    previous, previous_time = start, None
    travel = {n: 0. for n in ARM_JOINTS}
    for index, point in enumerate(trajectory.points):
        if len(point.positions) != len(names):
            raise MotionRejected("MOTION_PLAN_INVALID", "规划轨迹关节数量错误")
        positions = check_joints(dict(zip(names, point.positions)), limits)
        t = point.time_from_start.sec + point.time_from_start.nanosec*1e-9
        if point.time_from_start.sec < 0 or not 0 <= point.time_from_start.nanosec < 1_000_000_000:
            raise MotionRejected("MOTION_PLAN_INVALID", "规划轨迹时间无效")
        if previous_time is not None and t <= previous_time:
            raise MotionRejected("MOTION_PLAN_INVALID", "规划轨迹时间必须递增")
        if index == 0 and (t > .001 or max(abs(positions[n]-start[n]) for n in names) > .03):
            raise MotionRejected("MOTION_START_MISMATCH", "规划轨迹起点与当前关节不一致")
        for field in ("velocities", "accelerations"):
            values = getattr(point, field, ())
            if values and (len(values) != len(names) or not all(math.isfinite(v) for v in values)):
                raise MotionRejected("MOTION_PLAN_INVALID", "规划轨迹速度或加速度数据无效")
        for j, n in enumerate(names):
            travel[n] += abs(positions[n]-previous[n])
            if abs(positions[n]-start[n]) > policy.max_joint_change+1e-5:
                raise MotionRejected("MOTION_EXCESS_ROTATION", f"{n} 规划转角过大，停止执行", {"joint": n})
            speed_limit = limits[n][2]*policy.velocity_scaling*1.05 + .01
            if previous_time is not None and abs(positions[n]-previous[n])/(t-previous_time) > speed_limit:
                raise MotionRejected("MOTION_EXCESS_SPEED", f"{n} 规划速度超出限制，停止执行", {"joint": n})
            if point.velocities and abs(point.velocities[j]) > speed_limit:
                raise MotionRejected("MOTION_EXCESS_SPEED", f"{n} 规划速度超出限制，停止执行", {"joint": n})
        previous, previous_time = positions, t
    if max(abs(previous[n]-target[n]) for n in names) > .02:
        raise MotionRejected("MOTION_END_MISMATCH", "规划轨迹终点与选定逆解不一致")
    for n in names:
        allowed = min(policy.max_joint_change, 1.5*abs(target[n]-start[n])+.35)
        if travel[n] > allowed+1e-5:
            raise MotionRejected("MOTION_EXCESS_ROTATION", f"{n} 累计转角或绕路过大，停止执行",
                                 {"joint": n, "travel_rad": travel[n], "allowed_rad": allowed})
    if previous_time <= 0 or previous_time > policy.timeout-5:
        raise MotionRejected("MOTION_PLAN_DURATION", "规划轨迹时长超出本次运动预算")
    return {"travel_rad": travel, "duration": previous_time, "points": len(trajectory.points)}


def move_pose(backend, xyz, rpy, policy, audit=lambda event: None):
    data = {"position": list(xyz), "rpy": list(rpy)}
    try:
        limits = backend.limits()
        start = check_joints(backend.joints(), limits, feedback=True)
        data.update(start=start, limits=limits)
        audit({"stage": "start", **data})
        candidates = backend.inverse_solutions(xyz, rpy, start, limits, policy)
        target = nearest_solution(candidates, start, limits, policy)
        data.update(target=target, delta_rad={n: target[n]-start[n] for n in ARM_JOINTS}, candidates=len(candidates))
        audit({"stage": "ik_selected", **data})
        trajectory = backend.plan(start, target, limits, policy)
        data.update(validate_trajectory(trajectory, start, target, limits, policy))
        current = check_joints(backend.joints(), limits, feedback=True)
        if max(abs(current[n]-start[n]) for n in ARM_JOINTS) > .03:
            raise MotionRejected("MOTION_ROBOT_MOVED", "规划期间关节状态变化，停止发送轨迹")
        audit({"stage": "planned", **data})
        backend.execute(trajectory, data["duration"])
        actual = check_joints(backend.verify(target), limits, feedback=True)
        if max(abs(actual[n]-target[n]) for n in ARM_JOINTS) > .03:
            raise MotionRejected("MOTION_NOT_CONFIRMED", "执行后关节未到选定逆解，停止后续动作")
        data["actual"] = actual
        audit({"stage": "completed", **data})
        return ToolResult.success("规划轨迹已执行并确认关节到位", data)
    except MotionRejected as exc:
        audit({"stage": "rejected", "error_code": exc.code, **data, **exc.data})
        return ToolResult.failure(exc.code, str(exc), {**data, **exc.data})
