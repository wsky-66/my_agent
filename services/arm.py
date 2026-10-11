"""Reusable arm connection. Services return data and never register model tools."""
from config import config
from services.runtime import worker

client = worker("arm", config.ROS_PYTHON, {"controller_dir": config.CONTROLLER_DIR,
                "robot_name": config.ROBOT_NAME}, ros=True, workspace=True)


def call(action, **params):
    return client.request(action, params)


def stop():
    client.stop()
