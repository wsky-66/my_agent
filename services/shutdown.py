"""Separate shutdown connection so cancellation of a task cannot cancel parking."""
from config import config
from core.results import ToolResult
from services.runtime import worker


def park_robot():
    from services.health import robot_status
    health = robot_status(allow_shutdown=True)
    checks = health.data.get("checks", {})
    if not checks.get("joint_feedback") or not checks.get("driver_response"):
        return ToolResult.failure("SHUTDOWN_NO_FEEDBACK", "驱动或近期关节反馈不可用，无法确认 Sleep，跳过收拢动作",
                                  {"sleep_verified": False, "health_error": health.error_code})
    client = worker("shutdown", config.ROS_PYTHON, {"robot_name": config.ROBOT_NAME,
                    "timeout": config.SHUTDOWN_SLEEP_TIMEOUT, "tolerance": config.SLEEP_JOINT_TOLERANCE},
                    ros=True, workspace=True, timeout=config.SHUTDOWN_SLEEP_TIMEOUT+2)
    client.startup_timeout = 8
    try:
        return client.request("park", allow_shutdown=True)
    finally:
        client.stop()
