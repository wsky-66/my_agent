"""Reusable arm connection. Services return data and never register model tools."""
from config import config
from services.runtime import worker

client = worker("arm", config.ROS_PYTHON, {"controller_dir": config.CONTROLLER_DIR,
                "robot_name": config.ROBOT_NAME}, ros=True, workspace=True)


def call(action, **params):
    from services.health import robot_status
    ready = robot_status()
    if not ready.ok:
        return ready
    return client.request(action, params)


def stop():
    client.stop()
