#!/usr/bin/env python3
"""Grounding DINO 推理 + 可视化脚本。

用法:
  python3 test/infer.py "banana"                        # 一次性: 抓相机帧检测 banana
  python3 test/infer.py "cat.dog.person" --image a.jpg  # 一次性: 检测指定图
  python3 test/infer.py                                 # 交互模式: 反复输入提示词检测
  python3 test/infer.py --image a.jpg                   # 交互模式 + 固定用某张图

交互模式:
  输入英文提示词(多类别用 '.' 分隔, 如 cat.dog.person)即检测一次并保存可视化图;
  输入 q / quit / exit 退出; 输入 image=路径 切换检测图片; 输入 camera 切回抓相机帧。

流程: 取图(相机帧或 --image) -> DDS Grounding DINO 推理得 bbox -> 在原图上画框 ->
保存可视化图到 test/ 并打印结果。依赖 .env 的 DDS_API_TOKEN。
"""

import argparse
import base64
import json
import mimetypes
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config  # noqa: E402
from capabilities.camera_tools import _grab_frame  # noqa: E402

DETECT_URL = f"{config.DDS_BASE_URL}/v2/task/grounding_dino/detection"
POLL_URL = f"{config.DDS_BASE_URL}/v2/task_status/{{task_uuid}}"
CREATE_TIMEOUT = 90
POLL_SECONDS = 90
DEFAULT_MODEL = "GroundingDino-1.6-Pro"
TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def to_base64_data(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def run_detection(image: str, prompt: str, model: str,
                  bbox_threshold: float, iou_threshold: float) -> dict:
    """创建任务并轮询，返回 data.result dict 或 {'error': ...}。"""
    headers = {"Content-Type": "application/json", "Token": config.DDS_API_TOKEN}
    body = {
        "model": model,
        "image": image,
        "prompt": {"type": "text", "text": prompt},
        "targets": ["bbox"],
        "bbox_threshold": float(bbox_threshold),
        "iou_threshold": float(iou_threshold),
    }
    resp = requests.post(DETECT_URL, json=body, headers=headers, timeout=CREATE_TIMEOUT)
    data = resp.json()
    if data.get("code") != 0:
        return {"error": f"创建任务失败: {data.get('code')} {data.get('msg')}"}
    uuid = (data.get("data") or {}).get("task_uuid")
    if not uuid:
        return {"error": f"未返回 task_uuid: {data}"}

    deadline = time.time() + POLL_SECONDS
    while time.time() < deadline:
        time.sleep(1)
        try:
            p = requests.get(POLL_URL.format(task_uuid=uuid), headers=headers,
                             timeout=CREATE_TIMEOUT).json()
        except (requests.RequestException, json.JSONDecodeError):
            continue
        if p.get("code") != 0:
            return {"error": f"查询失败: {p.get('code')} {p.get('msg')}"}
        d = p.get("data") or {}
        status = d.get("status")
        if status == "success":
            return d.get("result") or {}
        if status == "failed":
            return {"error": f"任务失败: {d.get('error')}"}
    return {"error": f"检测超时(>{POLL_SECONDS}s)"}


def draw_boxes(image_path: str, objects: list, out_path: str):
    """在原图上画检测框并保存可视化图。"""
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"无法读取图: {image_path}")
    for obj in objects:
        bbox = obj.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        label = obj.get("category", "?")
        score = float(obj.get("score", 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {score:.2f}"
        cv2.putText(img, text, (x1, max(10, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cv2.imwrite(out_path, img)


def acquire_image(image: str):
    """返回可用图片路径；失败返回 None。指定 image 则校验存在，否则抓相机帧。"""
    if image:
        if not os.path.isfile(image):
            return None
        return image
    return _grab_frame()


def detect_once(src: str, prompt: str, args):
    """对 src 图检测 prompt，保存可视化图，打印结果。"""
    if not src or not os.path.isfile(src):
        print(f"  [失败] 图片不可用: {src}")
        return False
    t0 = time.time()
    image = to_base64_data(src)
    print(f"  [推理中] prompt='{prompt}' model={args.model} ...")
    result = run_detection(image, prompt, args.model,
                           args.bbox_threshold, args.iou_threshold)
    elapsed = time.time() - t0
    if "error" in result:
        print(f"  [错误] ({elapsed:.1f}s) {result['error']}")
        return False
    objects = result.get("objects") or []
    if not objects:
        print(f"  [未检测到目标] ({elapsed:.1f}s)")
        return False
    out = os.path.join(TEST_DIR, f"vis_{time.strftime('%Y%m%d_%H%M%S')}.png")
    draw_boxes(src, objects, out)
    print(f"  [检测到 {len(objects)} 个目标] ({elapsed:.1f}s):")
    for obj in objects:
        print(f"    - {obj.get('category')}: {float(obj.get('score',0)):.3f}, "
              f"bbox={obj.get('bbox')}")
    print(f"  [可视化已保存] {out}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Grounding DINO 推理 + 可视化")
    ap.add_argument("prompt", nargs="?", default=None,
                    help="英文提示词，多类别用 '.' 分隔；不传则进入交互模式")
    ap.add_argument("--image", default=None, help="指定图片；不传则抓相机帧")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--bbox-threshold", type=float, default=0.25)
    ap.add_argument("--iou-threshold", type=float, default=0.8)
    args = ap.parse_args()

    if not config.DDS_API_TOKEN:
        print("错误: 未配置 DDS_API_TOKEN"); return 1

    # 一次性模式: 给了 prompt 就只跑一次
    if args.prompt:
        src = acquire_image(args.image)
        if src is None:
            print(f"错误: 图片不可用: {args.image or '(抓帧失败)'}")
            return 1
        detect_once(src, args.prompt, args)
        return 0

    # 交互模式
    print("进入交互模式(输入英文提示词检测, q 退出, image=路径 换图, camera 切回抓帧):")
    image = args.image
    try:
        while True:
            line = input("prompt> ").strip()
            if not line:
                continue
            low = line.lower()
            if low in ("q", "quit", "exit"):
                print("退出。")
                break
            if low.startswith("image="):
                image = line.split("=", 1)[1].strip()
                print(f"  [切换图片] {image}")
                continue
            if low == "camera":
                image = None
                print("  [切回抓相机帧]")
                continue
            src = acquire_image(image)
            if src is None:
                print(f"  [失败] 图片不可用: {image or '(抓帧失败)'}")
                continue
            detect_once(src, line, args)
    except (KeyboardInterrupt, EOFError):
        print("\n退出。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
