"""相机进程生命周期：realsense2_camera 用 shell 拉起，stop_camera 杀死对应进程。

判定「是否运行」以进程存在为准（可靠），彩色话题仅作就绪判据（带重试）。
退出或 SIGTERM 时自动关停。
"""

import os
import signal
import subprocess
import time

from logger import error as log_error, warning as log_warning

_ROS_SETUP = "/opt/ros/humble/setup.bash"
_CMD = ("ros2 launch realsense2_camera rs_launch.py camera_name:=camera "
        "align_depth.enable:=true depth_module.depth_profile:=640x480x30 "
        "rgb_camera.color_profile:=640x480x30 pointcloud.enable:=false "
        "spatial_filter.enable:=true spatial_filter.holes_fill:=2 "
        "temporal_filter.enable:=true")
_LOG = "/tmp/camera_launch.log"
_GRACE = 6      # SIGTERM 后升级 SIGKILL 的等待秒数
_READY = 30     # 拉起后等待彩色话题的秒数
_NODE_NAMES = ("realsense2_camera", "rs_launch", "camera_node", "dynamic_tf")

_proc: subprocess.Popen | None = None


def _pids() -> list:
    """返回所有相机类进程 PID。"""
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


def _color_up() -> bool:
    """彩色图像话题是否可用（失败重试，避免 ROS2 daemon 抖动）。"""
    for _ in range(3):
        try:
            r = subprocess.run(
                ["bash", "-lc",
                 f"source {_ROS_SETUP} && timeout 6 ros2 topic list 2>/dev/null "
                 f"| grep -q color/image_raw"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def ready() -> bool:
    return _color_up()


def running() -> bool:
    return bool(_pids())


def start_camera() -> str:
    """拉起相机：运行 shell 启动命令。已在运行则跳过。"""
    global _proc
    if running():
        log_warning("start_camera：检测到已有相机进程（可能多实例/残留），未重复拉起")
        return "相机已在运行"
    cmd = f"source {_ROS_SETUP} && {_CMD} > {_LOG} 2>&1"
    _proc = subprocess.Popen(["bash", "-c", cmd], start_new_session=True)

    deadline = time.time() + _READY
    while time.time() < deadline:
        if _color_up():
            return f"相机已就绪 (PID {_proc.pid})"
        time.sleep(2)
    log_error(f"相机启动失败：{_READY}s 内彩色话题未就绪，日志 {_LOG}")
    return f"相机已拉起 (PID {_proc.pid})，但彩色话题未就绪，日志 {_LOG}"


def stop_camera() -> None:
    """停止相机：关闭自己拉起的进程组，并按名清理所有相机进程。"""
    global _proc
    try:
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
        if killed:
            log_warning(f"stop_camera：已清理相机进程 {killed}")
    except KeyboardInterrupt:
        pass


def camera_status() -> str:
    if running():
        return "相机运行中" if _color_up() else "相机进程运行中，但彩色话题未就绪"
    return "相机未运行"


# 注意：不再在此注册 atexit 清理。否则任何 import capabilities 的脚本(如 test/infer.py)
# 一销毁就会连带停掉相机节点。相机由 main.py(agent 主进程)负责退出清理。

