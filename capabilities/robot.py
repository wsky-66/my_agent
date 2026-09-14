"""机器人进程生命周期：start_robot 用 shell 拉起，stop_robot 杀死对应进程。

判定「是否运行」以进程存在为准（可靠），/moveit_plan 服务仅作就绪判据（带重试）。
退出或 SIGTERM 时自动关停。
"""

import os
import signal
import subprocess
import time

from logger import error as log_error, warning as log_warning

_ROS_SETUP = "/opt/ros/humble/setup.bash"
_WS_SETUP = "/home/wsky/interbotix_ws/install/setup.bash"
_CMD = (
    "ros2 launch interbotix_xsarm_moveit_interface "
    "xsarm_moveit_interface.launch.py robot_model:=wx250s "
    "use_moveit_rviz:=true use_moveit_interface_gui:=false"
)
_LOG = "/tmp/arm_launch.log"
_GRACE = 6      # SIGTERM 后升级 SIGKILL 的等待秒数
_READY = 30     # 拉起后等待 /moveit_plan 就绪的秒数
_NODE_NAMES = (
    "ros2 launch", "move_group", "moveit_interface", "ros2_control_node",
    "xs_sdk", "robot_state_publisher", "rviz2", "moveit_interface_gui",
)

_proc: subprocess.Popen | None = None


def _pids() -> list:
    """扫描进程，返回所有机器人类进程 PID。"""
    try:
        out = subprocess.run(["ps", "-eo", "pid,args"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    result = []
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and any(n in parts[1] for n in _NODE_NAMES):
            result.append(int(parts[0]))
    return result


def _service_up() -> bool:
    """/moveit_plan 服务是否可用（失败重试，避免 ROS2 daemon 结果抖动）。"""
    for _ in range(3):
        try:
            r = subprocess.run(
                ["bash", "-lc",
                 f"source {_ROS_SETUP} && source {_WS_SETUP} && "
                 f"timeout 6 ros2 service list 2>/dev/null | grep -q moveit_plan"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def ready() -> bool:
    """MoveIt 服务是否就绪。"""
    return _service_up()


def robot_running() -> bool:
    return _proc is not None and _proc.poll() is None


def _running() -> bool:
    """是否已有机器人进程在跑。"""
    return bool(_pids())


def _kill_all() -> list:
    """按名杀掉所有机器人进程（应对进程组/孤儿化），返回杀掉的 PID 列表。"""
    killed = []
    for _ in range(3):
        pids = _pids()
        if not pids:
            break
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(1)
    return killed


def start_robot() -> str:
    """拉起机器人：先清理残留进程再干净启动，避免降级/残留实例持续存在。"""
    global _proc
    if _running():
        stale = _kill_all()
        _proc = None
        log_warning(f"start_robot：检测到残留机器人进程 {stale}，已清理后重启")
    cmd = f"source {_ROS_SETUP} && source {_WS_SETUP} && {_CMD} > {_LOG} 2>&1"
    _proc = subprocess.Popen(["bash", "-c", cmd], start_new_session=True)

    deadline = time.time() + _READY
    while time.time() < deadline:
        if _service_up():
            return f"机器人已就绪 (PID {_proc.pid})"
        time.sleep(2)
    log_error(f"机器人启动失败：{_READY}s 内 /moveit_plan 未就绪，日志 {_LOG}")
    return f"机器人已拉起 (PID {_proc.pid})，但服务尚未就绪，日志 {_LOG}"


def stop_robot() -> None:
    """停止机器人：关闭自己拉起的进程组，并按名逐个杀掉所有机器人进程。"""
    global _proc
    proc, _proc = _proc, None
    if proc is not None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=_GRACE)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    # 按名杀直到确认没有残留（应对进程组/孤儿化）
    killed = _kill_all()
    if killed:
        log_warning(f"stop_robot：已清理机器人进程 {killed}")


def robot_status() -> str:
    if _running():
        return ("机器人运行中，MoveIt 服务就绪" if _service_up()
                else "机器人进程运行中，但 MoveIt 服务未就绪（可能崩溃），建议 restart")
    return "机器人未运行"


# 注意：不再在此注册 atexit 清理，避免 import 即毁掉机器人。清理统一由 main.py 负责。
