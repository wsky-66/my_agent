"""Compatibility facade for robot lifecycle callers."""
from services.lifecycle import robot, stop_robot


def start_robot():
    return robot.start()


def robot_status():
    return robot.status()


def ready():
    return robot.probe()


def robot_running():
    return robot.running()
