"""分割物体 3D 定位工具：结合 SAM mask 的中心像素(或整个 mask) + 相机内参/外参/深度，
求该物体在 base_link 系下的坐标。

与 sam2_tools.segment_mask 联动：segment_mask 返回中心像素后，调本工具得到 base_link 3D 位姿。
"""

import json
import os
import subprocess
import sys
import tempfile

from . import camera
from tools import registry

_ROS_SETUP = "/opt/ros/humble/setup.bash"
_PYTHON = sys.executable
_CALIB = "~/interbotix_ws/biaoding/calibration_result.json"


def _run(body: str, timeout: int = 30) -> dict:
    """在 ROS 环境用本解释器跑 body, 取 RESULT: JSON。"""
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(body)
    try:
        r = subprocess.run(["bash", "-lc", f"source {_ROS_SETUP} && {_PYTHON} {path}"],
                           capture_output=True, text=True, timeout=timeout)
    finally:
        os.unlink(path)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT:"):
            return json.loads(line[len("RESULT:"):])
    raise RuntimeError((r.stderr or r.stdout).strip() or "无输出")


_DEPROJ = '''
import json, os, sys, time, subprocess
import numpy as np, cv2, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo

PX, PY, RAD = %d, %d, %d
MASK = %r
CALIB = %r

tops = subprocess.run(["bash","-lc","source /opt/ros/humble/setup.bash && timeout 6 ros2 topic list 2>/dev/null"],
                      capture_output=True, text=True).stdout.split()
dt = next((t for t in tops if "aligned_depth_to_color/image_raw" in t), None)
it = next((t for t in tops if "color/camera_info" in t), None) or next((t for t in tops if "camera_info" in t), None)
if dt is None or it is None:
    print("RESULT:"+json.dumps({"ok":False,"msg":"未找到深度/内参话题"})); sys.exit(0)

rclpy.init(); node=Node("obj_pos"); got={}
def cb_im(m): got["d"]=np.frombuffer(m.data,np.uint16).reshape(m.height,m.width).copy()
def cb_inf(m): got["info"]=m
node.create_subscription(Image, dt, cb_im, 10)
node.create_subscription(CameraInfo, it, cb_inf, 10)
t0=time.time()
while time.time()-t0<6 and ("d" not in got or "info" not in got):
    rclpy.spin_once(node, timeout_sec=0.1)
node.destroy_node(); rclpy.shutdown()
if "d" not in got or "info" not in got:
    print("RESULT:"+json.dumps({"ok":False,"msg":"未收到深度/内参"})); sys.exit(0)

k = got["info"].k; fx, fy, cx, cy = k[0], k[4], k[2], k[5]
dm = got["d"].astype(np.float32)*0.001
H, W = dm.shape

ys, xs = [], []
if MASK and os.path.isfile(MASK):
    m = cv2.imread(MASK, cv2.IMREAD_GRAYSCALE)
    if m is not None:
        if m.shape[:2] != (H, W): m = cv2.resize(m, (W, H))
        yy, xx = np.where(m > 0); xs, ys = xx, yy
if len(xs) == 0 and PX is not None and PY is not None:
    y0, y1 = max(0, PY-RAD), min(H, PY+RAD+1)
    x0, x1 = max(0, PX-RAD), min(W, PX+RAD+1)
    yy, xx = np.mgrid[y0:y1, x0:x1]; ys, xs = yy.ravel(), xx.ravel()
if len(xs) == 0:
    print("RESULT:"+json.dumps({"ok":False,"msg":"没有可用像素"})); sys.exit(0)

z = dm[ys, xs]
valid = (z >= 0.20) & (z <= 1.0)
if valid.sum() == 0:
    print("RESULT:"+json.dumps({"ok":False,"msg":"该像素深度无效(可能超出范围)"})); sys.exit(0)
xs, ys, z = xs[valid], ys[valid], z[valid]
cam = np.stack([(xs-cx)/fx*z, (ys-cy)/fy*z, z], -1)

with open(CALIB) as f:
    d = json.load(f)
X = np.array(d["method2_nonlinear_ax_zb"]["matrix"], dtype=np.float64)
base = (X @ np.hstack([cam, np.ones((len(cam),1))]).T)[:3,:].T
mean = base.mean(0); mn = base.min(0); mx = base.max(0)
print("RESULT:"+json.dumps({"ok":True,
    "x": round(float(mean[0]),3), "y": round(float(mean[1]),3), "z": round(float(mean[2]),3),
    "size": [round(float(mx[i]-mn[i]),3) for i in range(3)], "npix": int(valid.sum())}))
'''


@registry.register(
    name="object_position",
    description="把 SAM 分割出的物体定位到 base_link 系：结合相机内参、外参标定、对齐深度，"
               "反投影得到物体中心的 (x,y,z) 米及尺寸。传 center 像素(来自 segment_mask 的中心像素)"
               "或 mask_path(直接用分割 mask)。返回 base_link 系坐标，适合接着 move_arm 抓取。",
)
def object_position(px: int = None, py: int = None, mask_path: str = None,
                    radius: int = 12) -> str:
    if not camera.ready():
        return "相机未运行或无彩色话题，请先 start_camera 再定位"
    uses_center = px is not None and py is not None
    if not uses_center:
        if not mask_path or not os.path.isfile(mask_path):
            return "失败: 请传入中心像素 (px, py) 或有效的 mask_path"
        px = py = 0   # 占位, 用 mask
    else:
        # SAM 中心像素是小数, 兼容字符串/小数, 取整到像素
        try:
            px = int(float(px))
            py = int(float(py))
        except (TypeError, ValueError):
            return "失败: px/py 必须是数字(中心像素, 可为小数)"
    if mask_path and not os.path.isfile(mask_path):
        return f"mask 文件不存在: {mask_path}"
    try:
        radius = max(8, min(int(float(radius)), 40))
    except (TypeError, ValueError):
        radius = 12
    body = _DEPROJ % (px, py, radius, mask_path or "", os.path.expanduser(_CALIB))
    try:
        d = _run(body)
    except subprocess.TimeoutExpired:
        return "定位超时（>30s），相机/深度可能异常"
    except Exception as e:
        return f"定位出错: {e}"
    if not d.get("ok"):
        return f"定位失败: {d.get('msg', '未知原因')}"
    sz = d.get("size")
    line = (f"物体相对 base_link 坐标: x={d['x']}m, y={d['y']}m, z={d['z']}m"
            + (f"\n尺寸(长宽高)≈{sz[0]}×{sz[1]}×{sz[2]}m" if sz else "")
            + f"\n参与像素: {d.get('npix')}")
    return line
