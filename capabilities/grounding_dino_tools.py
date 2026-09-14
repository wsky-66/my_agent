"""Grounding DINO 文本开放集检测工具：调 DDS 算法平台云端 API，按文本 prompt 检测物体。

依赖 DDS 平台 Token（.env 的 DDS_API_TOKEN），异步任务式调用：创建任务 → 轮询结果。
"""

import base64
import json
import mimetypes
import os
import time

import requests

from tools import registry
from config import config
from . import camera

_CREATE = f"{config.DDS_BASE_URL}/v2/task/grounding_dino/detection"
_POLL = f"{config.DDS_BASE_URL}/v2/task_status/{{task_uuid}}"
_TIMEOUT = 90    # 创建请求超时(秒)
_POLL_SECONDS = 90  # 轮询总时长
_DEFAULT_MODEL = "GroundingDino-1.6-Pro"


def _image_to_base64_data(image_path: str) -> str:
    """读本地图片为 base64 data URI。"""
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def _format_result(data: dict) -> str:
    """把 API 返回的 objects 转为可读文本。"""
    result = data.get("result", {})
    objects = result.get("objects") or []
    if not objects:
        return "未检测到任何目标物体。"
    lines = [f"检测到 {len(objects)} 个目标："]
    for obj in objects:
        score = float(obj.get("score", 0))
        category = obj.get("category", "?")
        bbox = obj.get("bbox")
        if bbox and len(bbox) == 4:
            bbox_str = f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]"
        else:
            bbox_str = "None"
        lines.append(f"  - {category}: 置信度={score:.3f}, bbox={bbox_str}")
    return "\n".join(lines)


@registry.register(
    name="detect_grounding",
    description="用 Grounding DINO 开放集检测模型，按文本提示在图像/相机画面中定位物体，"
               "返回检测框 bbox、类别与置信度。prompt 用英文单词、多个类别以 '.' 分隔（如 "
               "'cat.dog.person'）。可选传 image_path 检测指定图片；不传则抓取相机当前画面。"
               "适合按任意文本找物体，比预置类别灵活。",
)
def detect_grounding(prompt: str, image_path: str = None,
                     camera_take: bool = True,
                     model: str = _DEFAULT_MODEL,
                     bbox_threshold: float = 0.25,
                     iou_threshold: float = 0.8) -> str:
    if not config.DDS_API_TOKEN:
        return "错误：未配置 DDS_API_TOKEN（请在 .env 中设置 DDS 平台 Token）"

    image = None
    if image_path and os.path.isfile(image_path):
        image = _image_to_base64_data(image_path)
    elif camera_take:
        if not camera.ready():
            return "相机未运行或无彩色话题，请先 start_camera 再检测，或传 image_path 指定图片"
        from .camera_tools import _grab_frame
        frame = _grab_frame()
        if frame is None or not os.path.isfile(frame):
            return "相机抓帧失败，请传 image_path 指定图片"
        image = _image_to_base64_data(frame)

    if not image:
        return "未找到图片：请传有效的 image_path，或先 start_camera"

    body = {
        "model": model,
        "image": image,
        "prompt": {"type": "text", "text": prompt},
        "targets": ["bbox"],
        "bbox_threshold": float(bbox_threshold),
        "iou_threshold": float(iou_threshold),
    }
    headers = {"Content-Type": "application/json", "Token": config.DDS_API_TOKEN}

    try:
        resp = requests.post(_CREATE, json=body, headers=headers, timeout=_TIMEOUT)
        data = resp.json()
    except requests.RequestException as e:
        return f"检测请求出错: {e}"
    except json.JSONDecodeError:
        return f"检测返回非 JSON: {resp.text[:200]}"

    if data.get("code") != 0:
        return f"创建任务失败: {data.get('code')} {data.get('msg')}"

    task_uuid = (data.get("data") or {}).get("task_uuid")
    if not task_uuid:
        return f"创建任务未返回 task_uuid: {data}"

    deadline = time.time() + _POLL_SECONDS
    while time.time() < deadline:
        time.sleep(1)
        try:
            poll = requests.get(_POLL.format(task_uuid=task_uuid),
                                headers=headers, timeout=_TIMEOUT)
            pdata = poll.json()
        except requests.RequestException as e:
            return f"查询任务出错: {e}"
        except json.JSONDecodeError:
            continue

        if pdata.get("code") != 0:
            return f"查询任务失败: {pdata.get('code')} {pdata.get('msg')}"

        pdata = pdata.get("data") or {}
        status = pdata.get("status")
        if status == "success":
            return _format_result(pdata)
        if status == "failed":
            return f"任务失败: {pdata.get('error')}"

    return f"检测任务超时（>{_POLL_SECONDS}s）"
