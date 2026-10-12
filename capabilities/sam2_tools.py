"""SAM2 model tool adapters."""
from tools import registry
from core.results import ToolResult
from services import segmentation


@registry.register(description="启动并预加载 SAM2 常驻模型。")
def start_sam2() -> ToolResult:
    return segmentation.client.request("warmup")


@registry.register(description="停止本程序的 SAM2 服务，释放模型内存。")
def stop_sam2() -> ToolResult:
    segmentation.stop()
    return ToolResult.success("SAM2 已停止")


@registry.register(description="查询本程序 SAM2 服务状态。")
def sam2_status() -> ToolResult:
    return ToolResult.success("SAM2 运行中" if segmentation.client.ready() else "SAM2 未运行")


@registry.register(description="用像素框 box 或正负点生成严格物体 mask。image_path 应使用检测返回的同一张图；out 指定保存路径。无框和点时支持原有交互点击。dilate_cm 默认为 0，抓取上下文由推理端另行构造。")
def segment_mask(pos_points: list[list[float]] = None, neg_points: list[list[float]] = None,
                 box: list[float] = None, image_path: str = None, dilate_cm: float = None,
                 out: str = None) -> ToolResult:
    return segmentation.segment(pos_points, neg_points, box, image_path, out, dilate_cm)


def segment_object(pos_points=None, neg_points=None, box=None, image_path=None, out=None, dilate_cm=None):
    result = segmentation.segment(pos_points, neg_points, box, image_path, out, dilate_cm)
    return (result.data, None) if result.ok else (None, result.message)
