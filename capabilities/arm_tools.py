"""Thin model tool adapters; motion and connection ownership live in services.arm."""
from tools import registry
from services import arm
from core.results import ToolResult


@registry.register(description="末端移动：x/y/z 为 base_link 下的米；roll/pitch/yaw 为度，需同时给出。无朝向保持当前朝向走笛卡尔直线，有朝向走 IK。")
def move_arm(x: float, y: float, z: float, roll: float = None, pitch: float = None, yaw: float = None) -> ToolResult:
    if any(v is not None for v in (roll, pitch, yaw)) and any(v is None for v in (roll, pitch, yaw)):
        return ToolResult.failure("INVALID_ARGUMENT", "roll/pitch/yaw 需同时给出或都不给")
    return arm.call("move_arm", x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)


@registry.register(description="读取当前末端 position（米）和 rpy（度）。")
def get_arm_pose() -> ToolResult:
    return arm.call("get_arm_pose")


@registry.register(description="设置夹爪张开位置，范围 0.015~0.037 米。")
def gripper_set(position: float) -> ToolResult:
    if not 0.015 <= position <= 0.037:
        return ToolResult.failure("INVALID_ARGUMENT", "夹爪位置范围为 0.015~0.037 米")
    return arm.call("gripper_set", position=position)


@registry.register(description="张开夹爪。")
def gripper_open() -> ToolResult:
    return gripper_set(0.037)


@registry.register(description="闭合夹爪。")
def gripper_close() -> ToolResult:
    return gripper_set(0.015)


@registry.register(description="按 0~100 闭合百分比控制夹爪。")
def gripper_grasp(close_percent: float = 100) -> ToolResult:
    if not 0 <= close_percent <= 100:
        return ToolResult.failure("INVALID_ARGUMENT", "闭合百分比范围为 0~100")
    return gripper_set(0.037-(0.037-0.015)*close_percent/100)


@registry.register(description="设置多个关节的绝对角度（弧度）。关节名：waist、shoulder、elbow、forearm_roll、wrist_angle、wrist_rotate。")
def set_joints(joints: dict[str, float]) -> ToolResult:
    return arm.call("set_joints", joints=joints)


@registry.register(description="设置单个关节的绝对角度（弧度）。")
def set_joint(joint_name: str, value: float) -> ToolResult:
    return set_joints({joint_name: value})


@registry.register(description="读取所有关节角度（弧度）。")
def get_joints() -> ToolResult:
    return arm.call("get_joints")


@registry.register(description="机械臂回到 Home 位姿。")
def go_home() -> ToolResult:
    return arm.call("go_home")


@registry.register(description="机械臂收拢到 Sleep 位姿，给相机让出视野。")
def go_sleep() -> ToolResult:
    return arm.call("go_sleep")
