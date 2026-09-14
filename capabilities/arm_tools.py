"""机械臂控制工具：通过 controller 模块控制 MoveIt 与夹爪。

不直接 import controller（其依赖 ROS/scipy），而是用系统 Python 在 ROS 环境里
跑子进程，避免污染 agent 进程。每个工具一次调用、一个连接、用完即释。
"""

import json
import os
import subprocess
import tempfile

from tools import registry

_CONTROLLER = "/home/wsky/interbotix_ws/controller"
_PYTHON = "/usr/bin/python3"
_ROS_SETUP = "/opt/ros/humble/setup.bash"
_WS_SETUP = "/home/wsky/interbotix_ws/install/setup.bash"
_TIMEOUT = 90


class _ArmError(Exception):
    pass


def _run(body: str) -> dict:
    """在 ROS 环境用系统 Python 执行机械臂逻辑，返回脚本打出的 RESULT: JSON。"""
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(f"import sys, json\nsys.path.insert(0, {_CONTROLLER!r})\n{body}\n")
    try:
        proc = subprocess.run(
            ["/bin/bash", "-lc",
             f"source {_ROS_SETUP} && source {_WS_SETUP} && {_PYTHON} {path}"],
            capture_output=True, text=True, timeout=_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise _ArmError("机械臂命令超时")
    finally:
        os.unlink(path)

    for line in proc.stdout.splitlines():
        if line.startswith("RESULT:"):
            return json.loads(line[len("RESULT:"):])
    raise _ArmError((proc.stderr or proc.stdout).strip() or "无输出")


def _fmt(result: dict, ok_msg: str) -> str:
    msg = result.get("msg") or ""
    return ok_msg if result.get("ok") else f"失败: {msg or '未知错误'}"


def _f(value, name: str):
    """把参数强转为 float，失败返回带说明的字符串（由调用方判断类型）。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return name


@registry.register(
    name="move_arm",
    description="把机械臂末端移动到指定位置。坐标单位米（相对 base_link），roll/pitch/yaw 单位为度（相对末端朝向）。"
               "未给朝向时为笛卡尔直线移动并保持当前朝向；给了朝向则为位姿规划。适合让机械臂移动、摆位等任务。",
)
def move_arm(x: float, y: float, z: float,
             roll: float = None, pitch: float = None, yaw: float = None) -> str:
    try:
        x, y, z = _f(x, "x"), _f(y, "y"), _f(z, "z")
        ori = {k: (None if v is None else _f(v, k)) for k, v in
               (("roll", roll), ("pitch", pitch), ("yaw", yaw))}
    except (TypeError, ValueError):
        return f"失败: 坐标/朝向必须是数字，收到 x={x!r}, y={y!r}, z={z!r}"
    for name, v in ori.items():
        if isinstance(v, str):
            return f"失败: 参数 {v} 必须是数字"
    given = [k for k, v in ori.items() if v is not None]
    if given and len(given) != 3:
        return "失败: roll/pitch/yaw 需同时给出或都不给"

    roll, pitch, yaw = (ori["roll"], ori["pitch"], ori["yaw"])
    body = f"""
from moveit_client import MoveItClient
try:
    arm = MoveItClient()
except Exception as e:
    print("RESULT:" + json.dumps({{"ok": False, "msg": str(e)}})); raise SystemExit
try:
    ok, msg = arm.move_to({x}, {y}, {z}, roll={roll!r}, pitch={pitch!r}, yaw={yaw!r})
    print("RESULT:" + json.dumps({{"ok": bool(ok), "msg": msg}}))
finally:
    arm.shutdown()
"""
    d = _run(body)
    return _fmt(d, f"移动成功，末端已到 ({x}m, {y}m, {z}m)")


@registry.register(
    name="get_arm_pose",
    description="读取机械臂末端当前位姿。返回末端的 (x, y, z) 坐标（米，base_link）和 (roll, pitch, yaw) 朝向（度）。"
               "适合在执行移动前确认位置，或查询机械臂当前姿态。",
)
def get_arm_pose() -> str:
    body = """
from moveit_client import MoveItClient
try:
    arm = MoveItClient()
except Exception as e:
    print("RESULT:" + json.dumps({"ok": False, "msg": str(e)})); raise SystemExit
try:
    p = arm.get_current_ee_pose()
    if p is None:
        print("RESULT:" + json.dumps({"ok": False, "msg": "无法读取当前末端位姿"}))
    else:
        from scipy.spatial.transform import Rotation as Rot
        e = Rot.from_quat([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]) \\
            .as_euler("xyz", degrees=True)
        print("RESULT:" + json.dumps({"ok": True, "x": p.position.x, "y": p.position.y, "z": p.position.z,
                                      "roll": round(e[0],2), "pitch": round(e[1],2), "yaw": round(e[2],2)}))
finally:
    arm.shutdown()
"""
    d = _run(body)
    if not d.get("ok"):
        return _fmt(d, "")
    return (f"末端位置: x={d['x']:.3f}m, y={d['y']:.3f}m, z={d['z']:.3f}m\n"
            f"末端朝向: roll={d['roll']}°, pitch={d['pitch']}°, yaw={d['yaw']}°")


def _gripper(action: str, args: str, ok_msg: str) -> str:
    body = f"""
from gripper_client import GripperClient
try:
    g = GripperClient()
except Exception as e:
    print("RESULT:" + json.dumps({{"ok": False, "msg": str(e)}})); raise SystemExit
try:
    ok = g.{action}({args})
    print("RESULT:" + json.dumps({{"ok": bool(ok), "msg": "夹爪控制失败"}}))
finally:
    g.shutdown()
"""
    return _fmt(_run(body), ok_msg)


@registry.register(
    name="gripper_open",
    description="张开机械臂夹爪。用在抓取前松开、放置物体、释放等场景。",
)
def gripper_open() -> str:
    return _gripper("open", "", "夹爪已张开")


@registry.register(
    name="gripper_close",
    description="完全闭合机械臂夹爪（抓取）。用在抓取物体时。",
)
def gripper_close() -> str:
    return _gripper("close", "", "夹爪已闭合")


@registry.register(
    name="gripper_set",
    description="把夹爪设置为指定张开位置。position 范围 0.015~0.037（米，0.015 闭合抓取，0.037 完全张开）。"
               "用于夹爪精细控制。",
)
def gripper_set(position: float) -> str:
    position = _f(position, "position")
    if isinstance(position, str):
        return f"失败: 参数 {position} 必须是数字"
    return _gripper("set", f"{position}", f"夹爪已设置为 {position}")


@registry.register(
    name="gripper_grasp",
    description="按闭合度抓取物体。close_percent 为 0~100，0 张开、100 完全闭合（越接近 100 夹得越紧）。",
)
def gripper_grasp(close_percent: float = 100.0) -> str:
    close_percent = _f(close_percent, "close_percent")
    if isinstance(close_percent, str):
        return f"失败: 参数 {close_percent} 必须是数字"
    return _gripper("grasp", f"{close_percent}", f"已按 {close_percent}% 闭合度抓取")


# ---- 关节级控制（与 MoveIt 末端控制不同，直接指定关节角）----

_ARM_JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]


def _joint(action: str, args: str, ok_msg: str) -> str:
    body = f"""
from joint_client import JointClient
try:
    arm = JointClient()
except Exception as e:
    print("RESULT:" + json.dumps({{"ok": False, "msg": str(e)}})); raise SystemExit
try:
    ok = arm.{action}({args})
    print("RESULT:" + json.dumps({{"ok": bool(ok), "msg": "关节控制失败"}}))
finally:
    arm.shutdown()
"""
    return _fmt(_run(body), ok_msg)


def _check_joints(*names) -> str:
    bad = [n for n in names if n not in _ARM_JOINTS]
    if bad:
        return f"失败: 未知关节 {', '.join(bad)} (可用: {', '.join(_ARM_JOINTS)})"
    return None


@registry.register(
    name="set_joint",
    description="把指定单个关节设为绝对角度（单位弧度 rad）。可用于调整机械臂某个关节。"
               "关节名: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate。",
)
def set_joint(joint_name: str, value: float) -> str:
    err = _check_joints(joint_name)
    if err:
        return err
    value = _f(value, "value")
    if isinstance(value, str):
        return f"失败: 参数 {value} 必须是数字"
    return _joint("set_joint", f"{joint_name!r}, {value}", f"已设置关节 {joint_name} 为 {value} rad")


@registry.register(
    name="set_joints",
    description="把多个关节同时设为绝对角度（弧度）。传入 {关节名: 角度} 字典，只控制传入的关节，其余保持当前值。",
)
def set_joints(joints: dict) -> str:
    if not isinstance(joints, dict) or not joints:
        return "失败: 请提供 {关节名: 角度} 字典"
    err = _check_joints(*joints)
    if err:
        return err
    try:
        cleaned = {k: _f(v, k) for k, v in joints.items()}
    except (TypeError, ValueError):
        return "失败: 关节角度必须是数字"
    for k, v in cleaned.items():
        if isinstance(v, str):
            return f"失败: 关节 {v} 的值必须是数字"
    desc = json.dumps(cleaned, ensure_ascii=False)
    return _joint("set_joints", f"{cleaned!r}", f"已设置关节 {desc} (rad)")


@registry.register(
    name="get_joints",
    description="读取机械臂所有关节的当前角度（弧度 rad）。返回各关节角度，便于确认当前位姿。",
)
def get_joints() -> str:
    body = """
from joint_client import JointClient
try:
    arm = JointClient()
except Exception as e:
    print("RESULT:" + json.dumps({"ok": False, "msg": str(e)})); raise SystemExit
try:
    j = arm.get_joints()
    if j is None:
        print("RESULT:" + json.dumps({"ok": False, "msg": "未收到关节状态"}))
    else:
        print("RESULT:" + json.dumps({"ok": True, "joints": j}))
finally:
    arm.shutdown()
"""
    d = _run(body)
    if not d.get("ok"):
        return _fmt(d, "")
    return "关节当前角度 (rad)\n" + "\n".join(
        f"  {k}: {round(v, 3)}" for k, v in d["joints"].items()
    )


@registry.register(
    name="go_home",
    description="让机械臂回到 Home 位姿（所有关节归零）。用于复位机械臂。",
)
def go_home() -> str:
    return _joint("go_home", "", "已回到 Home 位姿")


@registry.register(
    name="go_sleep",
    description="让机械臂回到 Sleep 位姿（收拢折叠）。用于机械臂收拢或长时间待机。",
)
def go_sleep() -> str:
    return _joint("go_sleep", "", "已回到 Sleep 位姿")

