"""SAM2 工作区分割工具：得到二值 workspace mask，供 GraspNet 限定抓取范围。

与 grasp_tools.detect_grasp 联动：本工具把 mask 写到固定的"当前工作区"路径，
detect_grasp(use_mask=True) 默认读取同一路径，从而实现"用 SAM2 圈工作区 -> GraspNet 只在工作区内抓"。

拆分采用常驻服务进程(sam2.py)，模型仅加载一次，后续请求经 stdin/stdout 通道快速推理。
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

from . import camera
from . import sam2 as sam2_svc
from tools import registry

_ROS_SETUP = "/opt/ros/humble/setup.bash"
_PYTHON = sys.executable
# 与 graspnet_client 的 MASK_PATH(默认工作区)保持一致, detect_grasp 默认读取此处
_WORKSPACE_MASK = "/home/wsky/msk_tool/masks/current_workspace.png"
_DILATE_CM = 1.0   # 固定膨胀距离(cm)


def _grab_frame() -> str:
    """订阅相机彩色话题抓一帧保存为临时 PNG，返回路径；失败返回 None。"""
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

rclpy.init(); node = Node("sam2_grab"); got = []
def cb(msg):
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
    if a.shape[2] == 3:
        a = a[:, :, ::-1]  # RGB -> BGR
    got.append(a)
node.create_subscription(Image, ct, cb, 10)
t0 = time.time()
while time.time() - t0 < 5 and not got:
    rclpy.spin_once(node, timeout_sec=0.1)
node.destroy_node(); rclpy.shutdown()
if not got:
    print("RESULT:" + json.dumps({"ok": False, "msg": "未收到相机画面"})); sys.exit(0)

p = "/tmp/sam2_frame_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
cv2.imwrite(p, got[-1])
print("RESULT:" + json.dumps({"ok": True, "path": p}))
'''
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(body)
    try:
        r = subprocess.run(
            ["/bin/bash", "-lc", f"source {_ROS_SETUP} && {_PYTHON} {path}"],
            capture_output=True, text=True, timeout=30,
        )
    finally:
        os.unlink(path)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT:"):
            try:
                d = json.loads(line[len("RESULT:"):])
            except json.JSONDecodeError:
                return None
            return d.get("path") if d.get("ok") else None
    return None


@registry.register(
    name="start_sam2",
    description="拉起并启动 SAM2 分割常驻服务（加载模型一次）。首次约几秒，之后 segment_mask 即走常驻进程快速推理。已就绪会跳过。适合预加载模型避免首次分割时久等。",
)
def start_sam2() -> str:
    sam2_svc.start()
    return sam2_svc.status()


@registry.register(
    name="stop_sam2",
    description="关闭 SAM2 分割常驻服务进程并释放其占用内存。适合长期不使用分割、待机或退出前调用。建议在 segment_mask 调用后长期空闲时再关，以省去下次加载时间。",
)
def stop_sam2() -> str:
    sam2_svc.stop()
    return "SAM2 服务已停止。"


@registry.register(
    name="sam2_status",
    description="查询 SAM2 分割常驻服务进程是否在运行、模型是否已就绪。",
)
def sam2_status() -> str:
    return sam2_svc.status()


@registry.register(
    name="segment_mask",
    description="圈定抓取工作区并生成二值 workspace mask，供 detect_grasp(use_mask=True) 限定抓取范围。"
               "分割注入优先级：①box 像素框(有框直接用，最高优先，可由 detect_grounding 得到) ②弹窗(有显示时人左键点目标、右键点背景、Enter 保存) ③points 像素点。"
               "默认抓当前相机帧；image_path 可指定已保存的图片，此时 box/points 必须与该图同坐标系。"
               "适合抓取前先圈出物体所在区域；用 detect_grounding 得目标 bbox 后传 box 即可免人工点击。",
)
def segment_mask(pos_points: list = None, neg_points: list = None,
                 box: list = None, image_path: str = None,
                 dilate_cm: float = None) -> str:
    """
    用 SAM2 圈定抓取工作区。默认弹窗由鼠标点击注入分割信息。

    :param pos_points: (一般不传)正点像素坐标列表 [[x,y]]，仅无 DISPLAY 时兜底用
    :param neg_points: (一般不传)负点像素坐标列表 [[x,y]]
    :param box: (优先)像素框 [x1,y1,x2,y2]，有框即用它分割，免弹窗；可由 detect_grounding 的 bbox 注入
    :param image_path: 图片路径，不传则抓当前相机帧
    :param dilate_cm: mask 向外膨胀的物理距离(cm)，固定为 1.0，忽略外部传入
    """
    dc = _DILATE_CM

    img = image_path
    if not img:
        if not camera.ready():
            return "相机未运行或无彩色话题，请先 start_camera 或传入 image_path"
        img = _grab_frame()
        if not img:
            return "相机未运行或无彩色话题，请先 start_camera 或传入 image_path"
    elif not os.path.isfile(img):
        return f"图片文件不存在: {img}"

    req = {
        "image_path": img,
        "points": pos_points,
        "neg_points": neg_points,
        "box": box,
        "dilate_cm": dc,
        "out": _WORKSPACE_MASK,
    }
    result = sam2_svc.run(req)
    if not result.get("ok"):
        return result.get("msg", "未知原因")

    path = result.get("mask_path", _WORKSPACE_MASK)
    lines = [f"工作区 mask 已生成: {path}",
             f"覆盖画面 {result.get('coverage', 0)}%（已膨胀 {dc}cm）"]
    if result.get("score") is not None:
        lines.append(f"分割 score={result['score']}")
    c = result.get("center")
    if c:
        lines.append(f"中心像素=({c[0]}, {c[1]})")
    b = result.get("bbox")
    if b:
        lines.append(f"包围框=({b[0]},{b[1]})~({b[2]},{b[3]}), 面积={result.get('area', 0)}px")
    lines.append("后续用 detect_grasp(use_mask=True) 即在此工作区内出抓取候选；"
                 "或用 object_position 根据中心像素求该物体相对 base_link 的坐标。")
    return "\n".join(lines)
