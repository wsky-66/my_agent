"""Driver bounds and timed trajectories. Angles are absolute, never wrapped."""
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
    velocity_scaling: float = .25
    timeout: float = 45.

    def __post_init__(self):
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


def nearest_solution(candidates, start, limits):
    if not candidates:
        raise MotionRejected("IK_NO_SOLUTION", "逆解接口未返回有效解；需检查目标可达性、碰撞及求解器错误码",
                             {"ik_solutions": 0})
    accepted = []
    for candidate in candidates:
        try:
            target = check_joints(candidate, limits)
        except (MotionRejected, TypeError, ValueError):
            continue
        delta = {n: target[n]-start[n] for n in ARM_JOINTS}
        # Prefer small wrist changes without changing the user's requested pose.
        score = sum((2. if n in ("forearm_roll", "wrist_rotate") else 1.)*v*v for n, v in delta.items())
        accepted.append((score, target))
    if not accepted:
        raise MotionRejected("IK_BRANCH_REJECTED", "逆解均超出驱动关节限位或数据无效，停止移动",
                             {"ik_solutions": len(candidates)})
    return min(accepted, key=lambda item: item[0])[1]


def validate_trajectory(trajectory, start, target, limits, policy):
    names = list(trajectory.joint_names)
    if len(names) != len(ARM_JOINTS) or set(names) != set(ARM_JOINTS) or len(trajectory.points) < 2:
        raise MotionRejected("MOTION_PLAN_INVALID", "规划轨迹必须包含完整关节、起点和终点")
    previous, previous_time = start, None
    travel = {n: 0. for n in ARM_JOINTS}
    peak_speeds = {n: 0. for n in ARM_JOINTS}
    speed_failure = None
    required_time_scale = 1.
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
            speed_limit = limits[n][2]*policy.velocity_scaling*1.05 + .01
            speeds = []
            if previous_time is not None:
                speeds.append(("segment", abs(positions[n]-previous[n])/(t-previous_time)))
            if point.velocities:
                speeds.append(("reported", abs(point.velocities[j])))
            for source, speed in speeds:
                peak_speeds[n] = max(peak_speeds[n], speed)
                required_time_scale = max(required_time_scale, speed/(limits[n][2]*policy.velocity_scaling))
                if speed > speed_limit and (speed_failure is None or
                        speed/speed_limit > speed_failure["speed_rad_s"]/speed_failure["allowed_rad_s"]):
                    speed_failure = {"joint": n, "point_index": index, "speed_source": source,
                                     "speed_rad_s": speed, "allowed_rad_s": speed_limit}
        previous, previous_time = positions, t
    if max(abs(previous[n]-target[n]) for n in names) > .02:
        raise MotionRejected("MOTION_END_MISMATCH", "规划轨迹终点与选定逆解不一致")
    if previous_time <= 0 or previous_time > policy.timeout-5:
        raise MotionRejected("MOTION_PLAN_DURATION", "规划轨迹时长超出本次运动预算")
    # Check bounds, endpoints and timing before reconciling driver speed limits.
    if speed_failure is not None:
        raise MotionRejected("MOTION_EXCESS_SPEED", f"{speed_failure['joint']} 规划速度超出限制，停止执行",
                             {**speed_failure, "required_time_scale": required_time_scale,
                              "duration": previous_time})
    return {"travel_rad": travel, "delta_rad": {n: target[n]-start[n] for n in names},
            "duration": previous_time, "points": len(trajectory.points),
            "peak_speed_rad_s": peak_speeds}


def prepare_trajectory(trajectory, start, target, limits, policy):
    """Reconcile timed plans with driver speed limits without changing the path."""
    factor, original_duration = 1., None
    try:
        metrics = validate_trajectory(trajectory, start, target, limits, policy)
    except MotionRejected as exc:
        if exc.code != "MOTION_EXCESS_SPEED":
            raise
        # Some MoveIt configs disable velocity bounds, so their timing can
        # ignore MotionPlanRequest's scaling. Stretch the entire timeline to
        # the driver's nominal scaled limit; reduce derivatives consistently.
        factor = exc.data["required_time_scale"]*(1.+1e-6)
        original_duration = exc.data["duration"]
        slow_trajectory(trajectory, 1./factor)
        metrics = validate_trajectory(trajectory, start, target, limits, policy)
    return {**metrics, "time_scale": factor,
            "original_duration": metrics["duration"] if original_duration is None else original_duration}


def move_pose(backend, xyz, rpy, policy, audit=lambda event: None):
    data = {"position": list(xyz), "rpy": list(rpy)}
    try:
        limits = backend.limits()
        start = check_joints(backend.joints(), limits, feedback=True)
        data.update(start=start, limits=limits)
        audit({"stage": "start", **data})
        candidates = backend.inverse_solutions(xyz, rpy, start, limits, policy)
        target = nearest_solution(candidates, start, limits)
        data.update(target=target, delta_rad={n: target[n]-start[n] for n in ARM_JOINTS}, candidates=len(candidates))
        audit({"stage": "ik_selected", **data})
        trajectory = backend.plan(start, target, limits, policy)
        data.update(prepare_trajectory(trajectory, start, target, limits, policy))
        audit({"stage": "planned", **data})
        backend.execute(trajectory, data["duration"])
        data["controller_completed"] = True
        audit({"stage": "completed", **data})
        return ToolResult.success("控制器报告规划轨迹执行完成", data)
    except MotionRejected as exc:
        audit({"stage": "rejected", "error_code": exc.code, **data, **exc.data})
        return ToolResult.failure(exc.code, str(exc), {**data, **exc.data})


def slow_trajectory(trajectory, scaling):
    """Uniformly slow a timed plan; positions and interpolation stay unchanged."""
    if not math.isfinite(scaling) or not 0 < scaling <= 1:
        raise MotionRejected("MOTION_PLAN_INVALID", "轨迹减速比例必须在 0~1 之间")
    for point in trajectory.points:
        stamp = point.time_from_start
        if stamp.sec<0 or not 0<=stamp.nanosec<1_000_000_000:
            raise MotionRejected("MOTION_PLAN_INVALID", "笛卡尔轨迹时间无效")
        ns = round((stamp.sec*1_000_000_000+stamp.nanosec)/scaling)
        stamp.sec, stamp.nanosec = divmod(ns, 1_000_000_000)
        point.velocities = [v*scaling for v in point.velocities]
        point.accelerations = [v*scaling*scaling for v in point.accelerations]
    return trajectory


def move_cartesian(backend, xyz, rpy, policy, audit=lambda event: None):
    """Keep the current orientation unless one was explicitly requested."""
    data = {"position":list(xyz), "rpy":list(rpy) if rpy is not None else None, "cartesian":True}
    try:
        limits = backend.limits()
        start = check_joints(backend.joints(), limits, feedback=True)
        data.update(start=start,limits=limits)
        path = backend.cartesian_plan(xyz, rpy, start, limits, policy)
        if not path.points:
            raise MotionRejected("MOTION_PLAN_INVALID", "笛卡尔轨迹为空")
        target = check_joints(dict(zip(path.joint_names,path.points[-1].positions)),limits)
        data.update(target=target,delta_rad={n:target[n]-start[n] for n in ARM_JOINTS})
        # MoveIt already returns a timed Cartesian path. Only retime when its
        # speed exceeds the driver target; do not multiply all times again.
        data.update(prepare_trajectory(path,start,target,limits,policy))
        audit({"stage":"planned",**data})
        backend.execute(path,data["duration"])
        data["controller_completed"] = True
        audit({"stage":"completed",**data})
        return ToolResult.success("控制器报告完整笛卡尔轨迹执行完成",data)
    except MotionRejected as exc:
        audit({"stage":"rejected","error_code":exc.code,**data,**exc.data})
        return ToolResult.failure(exc.code,str(exc),{**data,**exc.data})
