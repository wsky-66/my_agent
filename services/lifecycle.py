"""Application-owned launch processes; existing external ROS services are left running."""
from pathlib import Path
import subprocess

from config import config
from core.processes import ManagedProcess
from services.runtime import ros_command


def _has(kind, names, workspace=False):
    try:
        result = subprocess.run(ros_command(["ros2", kind, "list"], workspace),
                                capture_output=True, text=True, timeout=3)
        return result.returncode == 0 and set(names).issubset(result.stdout.split())
    except (subprocess.TimeoutExpired, OSError):
        return False


robot = ManagedProcess("机械臂", ros_command([
    "ros2", "launch", "interbotix_xsarm_moveit_interface", "xsarm_moveit_interface.launch.py",
    f"robot_model:={config.ROBOT_NAME}", "use_moveit_rviz:=true", "use_moveit_interface_gui:=false"], True),
    Path(config.ROOT) / "logs/arm_launch.log",
    lambda: _has("service", ["/compute_ik", "/compute_cartesian_path"], True))

camera = ManagedProcess("相机", ros_command([
    "ros2", "launch", "realsense2_camera", "rs_launch.py", "camera_name:=camera",
    "align_depth.enable:=true", "depth_module.depth_profile:=640x480x30", "rgb_camera.color_profile:=640x480x30",
    "pointcloud.enable:=false", "spatial_filter.enable:=true", "spatial_filter.holes_fill:=2", "temporal_filter.enable:=true"]),
    Path(config.ROOT) / "logs/camera_launch.log",
    lambda: _has("topic", [config.CAMERA_PREFIX + "/color/image_raw", config.CAMERA_PREFIX + "/aligned_depth_to_color/image_raw"]))


def stop_robot():
    from services import arm
    from services.scene import store
    arm.stop()
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
    for close in (arm.stop, frames.stop, grasp.stop, segmentation.stop, grounding.stop, robot.stop, camera.stop):
        try:
            close()
        except Exception as exc:
            error("服务退出清理失败", exc)
