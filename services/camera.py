"""Immutable RGB-D artifacts shared by segmentation, position and grasp services."""
from pathlib import Path
import uuid
from config import config
from services.runtime import worker

client = worker("camera", config.ROS_PYTHON, {"prefix": config.CAMERA_PREFIX}, ros=True, timeout=20)
client.startup_timeout = 10
_last = None


def status():
    return client.request("status")


def capture():
    global _last
    folder = Path(config.SCENE_DIR) / uuid.uuid4().hex
    result = client.request("capture", {"folder": str(folder)})
    if result.ok:
        _last = result.data
    return result


def latest():
    return _last


def clear_latest():
    global _last
    _last = None


def stop():
    global _last
    _last = None
    client.stop()
