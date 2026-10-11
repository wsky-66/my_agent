"""Readiness evidence shared by probes and hardware adapters, without ROS imports."""
import math

from core.results import ToolResult


ARM_JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
MAX_FEEDBACK_AGE = 1.0


def fresh_joints(sample, now):
    if sample is None:
        return False
    received, state = sample
    if not 0 <= now - received <= MAX_FEEDBACK_AGE:
        return False
    if len(state.name) != len(state.position):
        return False
    positions = dict(zip(state.name, state.position))
    return all(name in positions and math.isfinite(positions[name]) for name in ARM_JOINTS)


def readiness(name, checks, labels, **details):
    missing = [labels[key] for key, passed in checks.items() if not passed]
    data = {"ready": not missing, "checks": checks, **details}
    if missing:
        return ToolResult.failure("DEVICE_NOT_READY", f"{name}未就绪：" + "、".join(missing), data)
    return ToolResult.success(f"{name}已就绪", data)


def robot_readiness(sample, now, driver_joints, controllers, actions, planning):
    checks = {
        "joint_feedback": fresh_joints(sample, now),
        "driver_response": set(ARM_JOINTS).issubset(driver_joints),
        "active_controllers": all(controllers.get(n) == "active" for n in ("arm_controller", "gripper_controller")),
        "trajectory_actions": all(actions.get(n, False) for n in ("arm_controller", "gripper_controller")),
        "planning_services": all(planning.get(n, False) for n in ("ik", "cartesian")),
    }
    return readiness("机械臂", checks, {
        "joint_feedback": "缺少近期完整关节反馈", "driver_response": "驱动未响应",
        "active_controllers": "控制器未激活", "trajectory_actions": "轨迹接口不可用",
        "planning_services": "规划服务不可用",
    }, controllers=controllers)


def camera_readiness(pair, info_sample, now):
    fresh = pair is not None and 0 <= now - pair[0] <= MAX_FEEDBACK_AGE
    info_fresh = info_sample is not None and 0 <= now - info_sample[0] <= MAX_FEEDBACK_AGE
    images = False
    intrinsics = False
    if pair is not None:
        _, rgb, depth = pair
        shapes = rgb.width > 0 and rgb.height > 0 and (rgb.width, rgb.height) == (depth.width, depth.height)
        channels = 3 if rgb.encoding in ("rgb8", "bgr8") else 0
        depth_bytes = {"16UC1": 2, "mono16": 2, "32FC1": 4}.get(depth.encoding, 0)
        images = bool(shapes and channels and depth_bytes and
                      rgb.step >= rgb.width * channels and depth.step >= depth.width * depth_bytes and
                      len(rgb.data) >= rgb.step * rgb.height and len(depth.data) >= depth.step * depth.height)
        if info_sample is not None:
            info = info_sample[1]
            intrinsics = bool((info.width, info.height) == (rgb.width, rgb.height) and len(info.k) == 9 and
                              all(math.isfinite(v) for v in info.k) and info.k[0] > 0 and info.k[4] > 0)
    return readiness("相机", {"synchronized_frames": fresh, "image_data": images,
                              "camera_info": info_fresh and intrinsics}, {
        "synchronized_frames": "未收到近期同步彩色与对齐深度帧",
        "image_data": "图像数据缺失或格式无效", "camera_info": "缺少近期有效内参",
    })
