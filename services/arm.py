"""Reusable arm connection. Services return data and never register model tools."""
from config import config
from services.runtime import worker

client = worker("arm", config.ROS_PYTHON, {"controller_dir": config.CONTROLLER_DIR,
                "robot_name": config.ROBOT_NAME, "max_joint_change": config.ARM_MAX_JOINT_CHANGE,
                "velocity_scaling": config.ARM_VELOCITY_SCALING,
                "motion_timeout": config.ARM_MOTION_TIMEOUT}, ros=True, workspace=True,
                timeout=max(90, config.ARM_MOTION_TIMEOUT+10))


def call(action, **params):
    from services.health import robot_status
    ready = robot_status()
    if not ready.ok:
        return ready
    return client.request(action, params)


def stop():
    client.stop()
