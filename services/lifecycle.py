"""Hardware session: park before teardown, sweep reused and orphaned device processes."""
from pathlib import Path
import re
import threading

from config import config
from core.execution import ExecutionContext, current, cancel_tasks, open_session
from core.hardware_processes import HardwareSweep, SessionLock
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

_session = None
_shutdown_result = None
_shutdown_context = None
_abort_parking = threading.Event()
_close_lock = threading.RLock()


def begin_session():
    global _session, _shutdown_result
    with _close_lock:
        if _session is not None:
            raise RuntimeError("硬件会话已启动")
        lock = SessionLock(config.ROOT / "logs/session.lock") if config.ENABLE_ARM or config.ENABLE_CAMERA else None
        try:
            open_session()
        except Exception:
            if lock:
                lock.close()
            raise
        _session = {"lock": lock, "arm": config.ENABLE_ARM, "camera": config.ENABLE_CAMERA}
        _shutdown_result = None
        _abort_parking.clear()


def abort_parking():
    _abort_parking.set()
    if _shutdown_context is not None:
        _shutdown_context.cancelled.set()


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


def close_all(notify=lambda message, ok=True: None):
    global _session, _shutdown_result, _shutdown_context
    from services import arm, camera as frames, grasp, grounding, segmentation
    from services.shutdown import park_robot
    from logger import error
    callback = notify

    def notify(message, ok=True):
        try:
            callback(message, ok)
        except Exception as exc:
            error("退出提示显示失败", exc)

    with _close_lock:
        if _shutdown_result is not None:
            return _shutdown_result
        session = _session
        failures = []
        parked = ToolResult.success("本次未启用机械臂，跳过 Sleep", {"skipped": True})
        swept = ToolResult.success("本次未启用硬件，无硬件进程清理")
        sweep, captured = None, {}
        _shutdown_context = ExecutionContext()
        if _abort_parking.is_set():
            _shutdown_context.cancelled.set()
        token = current.set(_shutdown_context)
        try:
            if session:
                notify("正在退出：取消任务，停止接收新指令")
                try:
                    if not cancel_tasks():
                        failures.append("任务取消等待超时")
                except Exception as exc:
                    failures.append(f"任务取消失败：{exc}")
                owned = []
                for resource in (robot, camera, arm.client, frames.client, health.client):
                    proc = resource.proc
                    if proc is not None and proc.poll() is None:
                        owned.append(proc.pid)
                if session["arm"] or session["camera"]:
                    try:
                        port = None
                        if session["arm"]:
                            serial = health.serial_preflight(allow_shutdown=True)
                            port = serial.data.get("port")
                        sweep = HardwareSweep(config.ROBOT_NAME, config.CAMERA_PREFIX, config.ROOT,
                            arm=session["arm"], camera=session["camera"], serial_port=port)
                        captured = sweep.targets(owned)
                    except Exception as exc:
                        failures.append(f"进程检查失败：{exc}")
                        error("退出进程检查失败", exc)
                if session["arm"]:
                    notify("正在检查 Sleep；需要时规划收拢并确认到位")
                    try:
                        parked = park_robot()
                    except (Exception, KeyboardInterrupt) as exc:
                        parked = ToolResult.failure("SHUTDOWN_SLEEP_FAILED", f"无法确认 Sleep：{exc}")
                    notify(parked.message, parked.ok)
            if session:
                notify("正在释放连接并结束相关进程")
            # Parking must finish before any controller or launch process is stopped.
            # Every hook runs even if a worker fails to close.
            for close in (arm.stop, health.stop, frames.stop, grasp.stop, segmentation.stop, grounding.stop, robot.stop, camera.stop):
                try:
                    close()
                except (Exception, KeyboardInterrupt) as exc:
                    failures.append(f"{close.__module__}：{exc}")
                    error("服务退出清理失败", exc)
            if sweep:
                notify("正在检查并清理旧会话、外部驱动和残留设备占用")
                try:
                    swept = sweep.clean(captured)
                except (Exception, KeyboardInterrupt) as exc:
                    swept = ToolResult.failure("SHUTDOWN_SWEEP_FAILED", str(exc))
                notify(swept.message, swept.ok)
            data = {"sleep": {"ok": parked.ok, "message": parked.message, **parked.data},
                    "cleanup": {"ok": swept.ok, **swept.data}, "failures": failures}
            if failures or not swept.ok:
                result = ToolResult.failure("SHUTDOWN_INCOMPLETE", "退出清理未完成，请查看日志和残留进程", data)
            elif not parked.ok:
                result = ToolResult.failure("SLEEP_NOT_CONFIRMED", "进程清理已完成，但未确认机械臂已到 Sleep", data)
            else:
                result = ToolResult.success("退出清理完成", data)
            if session:
                _shutdown_result = result
            return result
        finally:
            current.reset(token)
            _shutdown_context = None
            if session:
                if session["lock"]:
                    session["lock"].close()
                _session = None
