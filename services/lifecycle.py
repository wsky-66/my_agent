"""Application-owned launch processes; existing external ROS services are left running."""
from pathlib import Path
import re

from config import config
from core.processes import ManagedProcess
from core.results import ToolResult
from services import health, camera as frames
from services.runtime import ros_command


def arm_diagnostic(log):
    if "Failed to open port" in log or "Error opening serial port" in log:
        return ToolResult.failure("SERIAL_OPEN_FAILED", "机械臂驱动无法打开串口，请检查 USB 连接和串口权限")
    if "InterbotixDriverXS initialization failed" in log or re.search(r"\[xs_sdk-\d+\].*process has died", log):
        return ToolResult.failure("ARM_DRIVER_FAILED", "机械臂驱动已退出，请查看 logs/arm_launch.log")
    if "Could not get robot_info service within timeout" in log or re.search(r"\[ros2_control_node-\d+\].*process has died", log):
        return ToolResult.failure("ARM_CONTROLLER_FAILED", "机械臂控制器初始化失败，未收到驱动响应")
    return None


def camera_diagnostic(log):
    if "UVCIOC_CTRL_QUERY" in log and "Connection timed out" in log:
        return ToolResult.failure("CAMERA_USB_TIMEOUT", "相机 USB 通信超时，请检查扩展坞、数据线和供电，日志 logs/camera_launch.log")
    if re.search(r"\[realsense2_camera_node-\d+\].*process has died", log):
        return ToolResult.failure("CAMERA_DRIVER_FAILED", "相机驱动已退出，请查看 logs/camera_launch.log")
    return None


robot = ManagedProcess("机械臂", ros_command([
    "ros2", "launch", "interbotix_xsarm_moveit_interface", "xsarm_moveit_interface.launch.py",
    f"robot_model:={config.ROBOT_NAME}", "use_moveit_rviz:=true", "use_moveit_interface_gui:=false"], True),
    Path(config.ROOT) / "logs/arm_launch.log",
    health.robot_status, preflight=health.serial_preflight, diagnose=arm_diagnostic)

_camera_parts = config.CAMERA_PREFIX.strip("/").split("/")
camera = ManagedProcess("相机", ros_command([
    "ros2", "launch", "realsense2_camera", "rs_launch.py",
    f"camera_namespace:={'/'.join(_camera_parts[:-1])}", f"camera_name:={_camera_parts[-1]}",
    "align_depth.enable:=true", "depth_module.depth_profile:=640x480x30", "rgb_camera.color_profile:=640x480x30",
    "pointcloud.enable:=false", "spatial_filter.enable:=true", "temporal_filter.enable:=true"]),
    Path(config.ROOT) / "logs/camera_launch.log",
    frames.status, diagnose=camera_diagnostic)


def stop_robot():
    from services import arm
    from services.scene import store
    arm.stop()
    health.stop()
    store.clear()
    return robot.stop()


def stop_camera():
    from services import camera as frames
    from services.scene import store
    frames.stop()
    store.clear()
    return camera.stop()


def close_all():
    from services import arm, camera as frames, grasp, grounding, segmentation
    from logger import error
    # Close workers before their ROS launch processes. Every hook gets a chance to run.
    for close in (arm.stop, health.stop, frames.stop, grasp.stop, segmentation.stop, grounding.stop, robot.stop, camera.stop):
        try:
            close()
        except Exception as exc:
            error("服务退出清理失败", exc)
