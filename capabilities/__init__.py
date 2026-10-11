"""Explicit capability loading; importing the package never registers or starts services."""
import importlib
from config import config


def load_tools():
    modules = []
    if config.ENABLE_ARM:
        modules += ["arm_tools", "robot_tools"]
    if config.ENABLE_CAMERA:
        modules += ["camera_tools"]
    if config.ENABLE_VISION and config.ENABLE_CAMERA:
        modules += ["grounding_dino_tools", "sam2_tools", "position_tools", "grasp_tools", "scan_tools"]
        if config.ENABLE_ARM:
            modules += ["workflow_tools"]
    for name in modules:
        importlib.import_module(f"capabilities.{name}")
