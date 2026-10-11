"""Compatibility facade for camera lifecycle callers."""
from services.lifecycle import camera, stop_camera


def start_camera():
    return camera.start()


def camera_status():
    return camera.status()


def ready():
    return camera.probe()


def running():
    return camera.running()
