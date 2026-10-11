"""Compatibility facade; lifecycle and transport live in services.segmentation."""
from services import segmentation


def start():
    return str(segmentation.client.request("warmup"))


def stop():
    segmentation.stop()


def ready():
    return segmentation.client.ready()


def status():
    return "SAM2 服务就绪" if ready() else "SAM2 服务未运行"


def run(req):
    result = segmentation.segment(req.get("points"), req.get("neg_points"), req.get("box"),
                                  req.get("image_path"), req.get("out"))
    return {"ok": result.ok, "msg": result.message, **result.data}
