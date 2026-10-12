"""DDS 目标检测与画框诊断；图片会发送到配置的 DDS 云服务。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services import camera, grounding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="英文类别，多类别用 . 分隔")
    parser.add_argument("--image", help="已有图片；省略时捕获相机 RGB-D 帧")
    parser.add_argument("--output", type=Path, default=Path("scratch/detection.png"))
    parser.add_argument("--model", default="GroundingDino-1.6-Pro")
    parser.add_argument("--bbox-threshold", type=float, default=.25)
    parser.add_argument("--iou-threshold", type=float, default=.8)
    args = parser.parse_args()
    try:
        result = grounding.detect(args.prompt, image_path=args.image, model=args.model,
                                  bbox_threshold=args.bbox_threshold, iou_threshold=args.iou_threshold)
        print(result.to_json())
        if not result.ok:
            return 1
        import cv2
        image = cv2.imread(result.data["image_path"])
        if image is None:
            raise ValueError("检测图片无法读取")
        for obj in result.data["objects"]:
            bbox = obj.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (40, 220, 80), 2)
            text = f"{obj.get('category', '?')} {float(obj.get('score', 0)):.3f}"
            cv2.putText(image, text, (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, (40, 220, 80), 1, cv2.LINE_AA)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), image):
            raise ValueError("检测预览保存失败")
        print(f"预览：{args.output.resolve()}")
        return 0
    finally:
        grounding.stop()
        camera.stop()


if __name__ == "__main__":
    raise SystemExit(main())
