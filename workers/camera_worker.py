"""Persistent RGB-D subscription and synchronized scene capture."""
import json
from pathlib import Path
import threading
import time

from core.results import ToolResult
from workers.protocol import options, serve


class CameraBackend:
    def __init__(self, opts):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, CameraInfo
        import message_filters
        rclpy.init()
        self.node = Node("my_agent_camera")
        self._condition = threading.Condition()
        self._pair = None
        self._info = None
        prefix = opts["prefix"]
        self._rgb = message_filters.Subscriber(self.node, Image, prefix + "/color/image_raw", qos_profile_sensor_data)
        self._depth = message_filters.Subscriber(self.node, Image, prefix + "/aligned_depth_to_color/image_raw", qos_profile_sensor_data)
        self._sync = message_filters.ApproximateTimeSynchronizer([self._rgb, self._depth], 10, 0.05)
        self._sync.registerCallback(self._on_pair)
        self.node.create_subscription(CameraInfo, prefix + "/color/camera_info", self._on_info, qos_profile_sensor_data)
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _on_pair(self, rgb, depth):
        with self._condition:
            self._pair = (time.monotonic(), rgb, depth)
            self._condition.notify_all()

    def _on_info(self, info):
        with self._condition:
            self._info = info
            self._condition.notify_all()

    def _spin(self):
        import rclpy
        while self._running and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def capture(self, folder):
        import cv2
        import numpy as np
        deadline = time.monotonic() + 6
        with self._condition:
            while not self._info or not self._pair or time.monotonic()-self._pair[0] > 1:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    return ToolResult.failure("CAMERA_NO_FRAME", "未收到近期的同步彩色、深度和内参")
                self._condition.wait(remaining)
            _, rgb, depth = self._pair
            info = self._info
        if rgb.encoding not in ("rgb8", "bgr8") or depth.encoding not in ("16UC1", "mono16", "32FC1"):
            return ToolResult.failure("IMAGE_ENCODING", "不支持的相机图像编码")
        color = np.ndarray((rgb.height, rgb.width, 3), np.uint8, buffer=rgb.data,
                           strides=(rgb.step, 3, 1)).copy()
        if rgb.encoding == "rgb8":
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        dtype = np.dtype(">u2" if depth.is_bigendian else "<u2") if depth.encoding != "32FC1" else np.dtype(">f4" if depth.is_bigendian else "<f4")
        dm = np.ndarray((depth.height, depth.width), dtype, buffer=depth.data,
                        strides=(depth.step, dtype.itemsize)).astype(np.float32)
        if depth.encoding != "32FC1":
            dm *= 0.001
        if dm.shape != color.shape[:2]:
            return ToolResult.failure("IMAGE_SHAPE", "彩色与对齐深度尺寸不匹配")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        image, depth_path = folder / "color.png", folder / "depth.npy"
        if not cv2.imwrite(str(image), color):
            raise RuntimeError("无法保存彩色帧")
        np.save(depth_path, dm)
        timestamp = rgb.header.stamp.sec + rgb.header.stamp.nanosec * 1e-9
        snapshot = {"image_path": str(image), "depth_path": str(depth_path), "intrinsics": list(info.k),
                    "timestamp": timestamp, "created": time.time(), "frame": rgb.header.frame_id,
                    "width": rgb.width, "height": rgb.height}
        (folder / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
        return ToolResult.success("已捕获同步 RGB-D 场景", snapshot)

    def __call__(self, action, params):
        if action == "capture":
            return self.capture(params["folder"])
        raise ValueError("未知相机动作")

    def close(self):
        import rclpy
        self._running = False
        self._thread.join(timeout=1)
        self.node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    backend = CameraBackend(options())
    serve(backend, backend.close)
