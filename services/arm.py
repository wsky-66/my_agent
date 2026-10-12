"""Reusable arm connection. Services return data and never register model tools."""
from config import config
from services.runtime import worker

client = worker("arm", config.ROS_PYTHON, {"controller_dir": config.CONTROLLER_DIR,
                "robot_name": config.ROBOT_NAME,
                "velocity_scaling": config.ARM_VELOCITY_SCALING,
                "motion_timeout": config.ARM_MOTION_TIMEOUT}, ros=True, workspace=True,
                timeout=max(90, config.ARM_MOTION_TIMEOUT+10))


def call(action, **params):
    from services.health import robot_status
    ready = robot_status()
    if not ready.ok:
        return ready
    if action in {"move_arm","set_joints","go_home","go_sleep","gripper_set"}:
        try:
            return client.request(action, params)
        finally:
            from services import camera
            camera.clear_latest()
    return client.request(action, params)


def stop():
    client.stop()
