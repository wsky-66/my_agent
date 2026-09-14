"""相机工具：进程生命周期 + 抓帧送入多模态大模型识别。"""

import base64
import json
import mimetypes
import os
import subprocess
import sys
import tempfile

from openai import OpenAI

from tools import registry
from . import camera
from config import config

_ROS_SETUP = "/opt/ros/humble/setup.bash"
_PYTHON = sys.executable
_client = None


def _run_ros(body: str, timeout: int = 30) -> dict:
    """在 ROS 环境用本解释器跑一段脚本，取 RESULT: JSON。"""
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(body)
    try:
        r = subprocess.run(
            ["/bin/bash", "-lc", f"source {_ROS_SETUP} && {_PYTHON} {path}"],
            capture_output=True, text=True, timeout=timeout,
        )
    finally:
        os.unlink(path)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT:"):
            return json.loads(line[len("RESULT:"):])
    raise RuntimeError((r.stderr or r.stdout).strip() or "无输出")


def _grab_frame() -> str:
    """订阅相机彩色话题抓取一帧，保存为临时 JPG，返回路径；失败返回 None。"""
    body = '''
import json, subprocess, sys, time
from datetime import datetime
import numpy as np, cv2, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

tops = subprocess.run(
    ["/bin/bash", "-lc", "source /opt/ros/humble/setup.bash && timeout 5 ros2 topic list 2>/dev/null"],
    capture_output=True, text=True).stdout.split()
ct = next((t for t in tops if "/color/image_raw" in t), None)
if ct is None:
    print("RESULT:" + json.dumps({"ok": False, "msg": "未找到相机彩色话题"})); sys.exit(0)

rclpy.init(); node = Node("camera_grab"); got = []
def cb(msg):
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
    if a.shape[2] == 3 and msg.encoding == "rgb8":
        a = a[:, :, ::-1]  # RGB -> BGR
    got.append(a)
node.create_subscription(Image, ct, cb, 10)
t0 = time.time()
while time.time() - t0 < 5 and not got:
    rclpy.spin_once(node, timeout_sec=0.1)
node.destroy_node(); rclpy.shutdown()
if not got:
    print("RESULT:" + json.dumps({"ok": False, "msg": "未收到相机画面"})); sys.exit(0)

p = "/tmp/camera_frame_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
cv2.imwrite(p, got[-1])
print("RESULT:" + json.dumps({"ok": True, "path": p}))
'''
    try:
        d = _run_ros(body)
        return d.get("path") if d.get("ok") else None
    except Exception:
        return None


def _ask_vision(image_path: str, prompt: str) -> str:
    """把图片交给多模态模型识别，返回文字描述。"""
    global _client
    if _client is None:
        config.validate()
        _client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    url = f"data:{mime};base64,{b64}"
    content = [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": prompt},
    ]
    # max_tokens 要足够大：DeepSeek 会先输出长 reasoning_content，预算耗尽会导致 content 为空
    resp = _client.chat.completions.create(
        model=config.MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=2000,
    )
    msg = resp.choices[0].message
    text = (msg.content or "").strip()
    if not text:
        text = (getattr(msg, "reasoning_content", "") or "").strip()
    return text or "（模型未能识别画面内容）"


@registry.register(
    name="start_camera",
    description="拉起并启动 RealSense 相机 ROS 节点。适合在需要视觉检测前调用。已运行会跳过，话题就绪后返回。",
)
def start_camera() -> str:
    return camera.start_camera()


@registry.register(
    name="stop_camera",
    description="关闭 RealSense 相机 ROS 节点并清理相关进程。适合视觉任务结束、待机或退出前调用。",
)
def stop_camera() -> str:
    camera.stop_camera()
    return "相机已停止。"


@registry.register(
    name="camera_status",
    description="查询相机 ROS 节点是否在运行、彩色话题是否就绪。",
)
def camera_status() -> str:
    return camera.camera_status()


@registry.register(
    name="capture_frame",
    description="订阅相机抓取一帧并保存为图片文件，返回其绝对路径。可用于先抓帧，再查看/传给视觉模型。需要相机已启动。",
)
def capture_frame() -> str:
    """订阅相机抓一帧存盘。相机未开则提示。"""
    path = _grab_frame()
    if path is None or not os.path.exists(path):
        return "相机未运行或无彩色话题，请先 start_camera 再抓帧"
    return f"已保存单帧画面: {path}"


@registry.register(
    name="recognize_camera",
    description="订阅相机抓取一帧，用多模态大模型识别画面，返回文字描述(可含物体大概方位)。"
               "只用于看图『大概在哪』『镜头全景/画面里有什么/描述看到的东西』；不能提供精确坐标，"
               "精确定位/抓取/验证一律不要用本工具——用 segment_mask+object_position 或 detect_grasp。",
)
def recognize_camera(prompt: str = "请描述这张图片里的内容。") -> str:
    """订阅相机抓帧后交给多模态模型识别。相机未开则提示。"""
    path = _grab_frame()
    if path is None or not os.path.exists(path):
        return "相机未运行或无彩色话题，请先 start_camera 再识别"
    try:
        return _ask_vision(path, prompt)
    except Exception as e:
        return f"识别出错: {e}"
