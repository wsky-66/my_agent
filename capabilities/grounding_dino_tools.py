"""Detection model tool adapter and legacy helper."""
from tools import registry
from services.grounding import detect
from core.results import ToolResult


@registry.register(description="Grounding DINO 文本检测，prompt 为英文类别，以 '.' 分隔。返回像素 bbox、类别、置信度和本次图像路径。未传 image_path 时抓取同步 RGB-D 场景；后续分割应使用返回的图像路径。")
def detect_grounding(prompt: str, image_path: str = None, camera_take: bool = True,
                     model: str = "GroundingDino-1.6-Pro", bbox_threshold: float = 0.25,
                     iou_threshold: float = 0.8) -> ToolResult:
    return detect(prompt, image_path, camera_take, model, bbox_threshold, iou_threshold)


def run_detection(*args, **kwargs):
    result = detect(*args, **kwargs)
    return (result.data["objects"], None) if result.ok else (None, result.message)
