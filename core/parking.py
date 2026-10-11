"""Confirmed Sleep parking, independent of ROS and normal task cancellation."""
import math

from core.health import ARM_JOINTS
from core.results import ToolResult


def sleep_error(joints, target):
    if any(n not in joints or n not in target or not math.isfinite(joints[n]) or not math.isfinite(target[n])
           for n in ARM_JOINTS):
        raise ValueError("Sleep 检查需要完整、有限的六关节数据")
    return max(abs(joints[n] - target[n]) for n in ARM_JOINTS)


def park(backend, tolerance):
    stopped = backend.cancel_motion()
    if not stopped.ok:
        return stopped
    target = backend.sleep_target()
    if not target.ok:
        return target
    joints = backend.joints()
    if not joints.ok:
        return joints
    already = sleep_error(joints.data, target.data) <= tolerance
    if not already:
        moved = backend.move_to_sleep(target.data)
        if not moved.ok:
            return moved
    verified = backend.verify_sleep(target.data)
    verified.data = {**verified.data, "already_sleep": already, "sleep_verified": verified.ok}
    if verified.ok:
        verified.message = "机械臂已在 Sleep，确认收拢" if already else "机械臂已收拢到 Sleep 并确认到位"
    return verified
