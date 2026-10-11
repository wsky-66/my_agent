"""Central configuration; workers receive explicit options and need no dotenv dependency."""
import os
import math
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def normalize_proxy_environment(environ=None):
    """Accept the legacy SOCKS alias without removing or bypassing the proxy."""
    environ = os.environ if environ is None else environ
    for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        value = environ.get(key, "").strip()
        if value.lower().startswith("socks://"):
            environ[key] = "socks5://" + value[len("socks://"):]


# Normalize after dotenv loading, before any HTTP client or worker is created.
# This affects only this process and its children; the caller's shell is unchanged.
normalize_proxy_environment()


def _path(key, default):
    return str(Path(os.path.expandvars(os.getenv(key, str(default)))).expanduser().resolve())


def _flag(key, default=True):
    return os.getenv(key, "1" if default else "0").lower() not in {"0", "false", "no", "off"}


class Config:
    def __init__(self):
        home = Path.home()
        self.ROOT = ROOT
        self.API_KEY = os.getenv("LLM_API_KEY", "")
        self.BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.MODEL = os.getenv("LLM_MODEL", "gpt-4o")
        self.MAX_TOOL_ROUNDS = int(os.getenv("LLM_MAX_TOOL_ROUNDS", "50"))
        self.LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))
        self.SHELL_TIMEOUT = float(os.getenv("SHELL_TIMEOUT", "30"))
        self.DDS_API_TOKEN = os.getenv("DDS_API_TOKEN", "")
        self.DDS_BASE_URL = os.getenv("DDS_BASE_URL", "https://api.deepdataspace.com")
        self.ROS_SETUP = _path("ROS_SETUP", "/opt/ros/humble/setup.bash")
        self.WS_SETUP = _path("WS_SETUP", home / "interbotix_ws/install/setup.bash")
        self.CONTROLLER_DIR = _path("CONTROLLER_DIR", home / "interbotix_ws/controller")
        self.ROS_PYTHON = _path("ROS_PYTHON", "/usr/bin/python3")
        self.VISION_PYTHON = _path("VISION_PYTHON", home / "miniconda3/envs/graspnet/bin/python")
        self.ASR_PYTHON = _path("ASR_PYTHON", sys.executable)
        self.SAM2_SERVER = _path("SAM2_SERVER", home / "msk_tool/sam2_server.py")
        self.CALIBRATION = _path("CALIBRATION", home / "interbotix_ws/biaoding/calibration_result.json")
        self.WORKSPACE_MASK = _path("WORKSPACE_MASK", home / "msk_tool/masks/current_workspace.png")
        self.SCENE_DIR = _path("SCENE_DIR", "/tmp/my_agent_scenes")
        self.ROBOT_NAME = os.getenv("ROBOT_NAME", "wx250s")
        self.CAMERA_PREFIX = os.getenv("CAMERA_PREFIX", "/camera/camera").rstrip("/")
        self.MIC_DEVICE = os.getenv("MIC_DEVICE", "plughw:1,0")
        self.TTS_VOICE = os.getenv("TTS_VOICE", "zh-CN-YunxiNeural")
        self.SCENE_MAX_AGE = float(os.getenv("SCENE_MAX_AGE", "300"))
        self.GRASP_SCORE_MIN = float(os.getenv("GRASP_SCORE_MIN", "0"))
        self.SHUTDOWN_SLEEP_TIMEOUT = float(os.getenv("SHUTDOWN_SLEEP_TIMEOUT", "30"))
        self.SLEEP_JOINT_TOLERANCE = float(os.getenv("SLEEP_JOINT_TOLERANCE", "0.03"))
        for key in ("ARM_AUTO_START", "CAM_AUTO_START", "ENABLE_ARM", "ENABLE_CAMERA", "ENABLE_VISION", "ENABLE_VOICE"):
            setattr(self, key, _flag(key))

    def validate(self):
        if not self.API_KEY:
            raise ValueError("LLM_API_KEY 未设置，请在项目 .env 中配置。")
        times = (self.LLM_TIMEOUT, self.SHELL_TIMEOUT, self.SCENE_MAX_AGE, self.SHUTDOWN_SLEEP_TIMEOUT)
        if self.MAX_TOOL_ROUNDS < 1 or any(not math.isfinite(v) or v <= 0 for v in times):
            raise ValueError("工具轮数和超时必须为正数。")
        if not math.isfinite(self.GRASP_SCORE_MIN) or not 0 <= self.GRASP_SCORE_MIN <= 1:
            raise ValueError("GRASP_SCORE_MIN 必须在 0~1 之间。")
        if not math.isfinite(self.SLEEP_JOINT_TOLERANCE) or not 0 < self.SLEEP_JOINT_TOLERANCE <= .1:
            raise ValueError("SLEEP_JOINT_TOLERANCE 必须在 0~0.1 弧度之间。")


config = Config()
