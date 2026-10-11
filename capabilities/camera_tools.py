"""Camera and description model tools; frame capture is a shared service."""
from tools import registry
from core.results import ToolResult
from services import camera, lifecycle
from services.vision import describe


@registry.register(description="启动或连接 RealSense 相机 ROS 服务。")
def start_camera() -> ToolResult:
    return lifecycle.camera.start()


@registry.register(description="停止本程序启动的相机与订阅连接，外部相机服务保留。")
def stop_camera() -> ToolResult:
    return lifecycle.stop_camera()


@registry.register(description="查询相机彩色和对齐深度话题就绪情况。")
def camera_status() -> ToolResult:
    return lifecycle.camera.status()


@registry.register(description="捕获同步 RGB-D 快照并返回彩色图、深度文件及内参。")
def capture_frame() -> ToolResult:
    return camera.capture()


@registry.register(description="用多模态模型描述相机画面；用于理解画面，不提供精确抓取坐标。")
def recognize_camera(prompt: str = "请描述画面中的物体和大概位置。") -> ToolResult:
    captured = camera.capture()
    if not captured.ok:
        return captured
    return describe(captured.data["image_path"], prompt)


def _grab_frame():
    result = camera.capture()
    return result.data["image_path"] if result.ok else None


def _ask_vision(image_path, prompt):
    return str(describe(image_path, prompt))
