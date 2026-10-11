"""DDS detection adapter, with reusable HTTP connection and a bounded polling deadline."""
import base64
import mimetypes
from pathlib import Path
import time

from config import config
from core.results import ToolResult
from core.execution import is_cancelled
from logger import measure
from services import camera

_session = None


def detect(prompt, image_path=None, camera_take=True, model="GroundingDino-1.6-Pro",
           bbox_threshold=0.25, iou_threshold=0.8):
    global _session
    if not config.DDS_API_TOKEN:
        return ToolResult.failure("MISSING_DDS_TOKEN", "未配置 DDS_API_TOKEN")
    if not prompt.strip():
        return ToolResult.failure("INVALID_ARGUMENT", "检测提示词不能为空")
    if not 0 <= bbox_threshold <= 1 or not 0 <= iou_threshold <= 1:
        return ToolResult.failure("INVALID_ARGUMENT", "检测阈值范围为 0~1")
    if image_path is None:
        if not camera_take:
            return ToolResult.failure("MISSING_IMAGE", "请提供图片或启用相机抓帧")
        result = camera.capture()
        if not result.ok:
            return result
        image_path = result.data["image_path"]
    path = Path(image_path)
    if not path.is_file():
        return ToolResult.failure("MISSING_IMAGE", f"图片不存在: {path}")
    import requests
    if _session is None:
        _session = requests.Session()
    image = "data:" + (mimetypes.guess_type(str(path))[0] or "image/png") + ";base64," + base64.b64encode(path.read_bytes()).decode()
    headers = {"Token": config.DDS_API_TOKEN}
    endpoint = config.DDS_BASE_URL.rstrip("/")
    body = {"model": model, "image": image, "prompt": {"type": "text", "text": prompt},
            "targets": ["bbox"], "bbox_threshold": bbox_threshold, "iou_threshold": iou_threshold}
    try:
        if is_cancelled():
            return ToolResult.failure("TASK_CANCELLED", "任务已中断")
        with measure("vision.grounding.create"):
            response = _session.post(endpoint + "/v2/task/grounding_dino/detection", json=body, headers=headers, timeout=(10, 90))
            response.raise_for_status()
            data = response.json()
        if data.get("code") != 0:
            return ToolResult.failure("DETECTION_FAILED", f"创建检测任务失败: {data.get('msg')}")
        task = (data.get("data") or {}).get("task_uuid")
        if not task:
            return ToolResult.failure("INVALID_RESPONSE", "检测服务未返回 task_uuid")
        deadline = time.monotonic() + 90
        with measure("vision.grounding.poll"):
            while time.monotonic() < deadline:
                if is_cancelled():
                    return ToolResult.failure("TASK_CANCELLED", "任务已中断")
                time.sleep(min(1, max(0, deadline-time.monotonic())))
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    break
                response = _session.get(endpoint + "/v2/task_status/" + task, headers=headers,
                                        timeout=(min(10, remaining), min(30, remaining)))
                response.raise_for_status()
                data = response.json()
                if data.get("code") != 0:
                    return ToolResult.failure("DETECTION_FAILED", f"查询检测任务失败: {data.get('msg')}")
                value = data.get("data") or {}
                if value.get("status") == "success":
                    objects = (value.get("result") or {}).get("objects") or []
                    return ToolResult.success(f"检测到 {len(objects)} 个目标", {"objects": objects, "image_path": str(path)})
                if value.get("status") == "failed":
                    return ToolResult.failure("DETECTION_FAILED", str(value.get("error")))
        return ToolResult.failure("DETECTION_TIMEOUT", "检测任务超时")
    except (requests.RequestException, ValueError) as exc:
        return ToolResult.failure("DETECTION_REQUEST_FAILED", str(exc))


def stop():
    global _session
    if _session:
        _session.close()
        _session = None
