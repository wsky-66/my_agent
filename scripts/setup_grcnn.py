"""Deploy the author's pinned GR-CNN code and RGB-D weights; no device access."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import config

COMMIT = "183c6f68c44c1c7ff0f07707e2db6fcfd6840d2d"
CHECKPOINT = "trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98"
CHECKPOINT_SHA256 = "e6c03afc0f8266d29ec92141088cc6a557215fabc2681d4ca5a162f451853d28"


def runtime_source(relative):
    """Keep inference code, its licence and the one deployed checkpoint."""
    return (relative in {"LICENSE", "README.md", "inference/__init__.py", CHECKPOINT} or
            relative.startswith("inference/models/") and relative.endswith(".py"))


def main():
    folder = config.ROOT/".runtime/grcnn"
    folder.mkdir(parents=True,exist_ok=True)
    root = folder/"upstream"
    source = root/CHECKPOINT
    required = (CHECKPOINT, "LICENSE", "inference/__init__.py", "inference/models/__init__.py",
                "inference/models/grasp_model.py", "inference/models/grconvnet3.py")
    if any(not (root/relative).is_file() for relative in required):
        with tempfile.TemporaryDirectory(dir=folder) as temporary:
            archive = Path(temporary)/"upstream.zip"
            url = f"https://codeload.github.com/skumra/robotic-grasping/zip/{COMMIT}"
            subprocess.run(["curl","--fail","--location","--silent","--show-error","--connect-timeout","10",
                            "--max-time","180",url,"--output",str(archive)],check=True)
            with zipfile.ZipFile(archive) as z:
                if z.comment.decode()!=COMMIT:
                    raise ValueError("GR-CNN 压缩包版本不匹配")
                prefix = z.namelist()[0].split("/")[0]+"/"
                for info in z.infolist():
                    relative = info.filename[len(prefix):]
                    if info.is_dir() or not runtime_source(relative):
                        continue
                    destination = root/relative
                    if not destination.resolve().is_relative_to(root.resolve()):
                        raise ValueError("压缩包路径无效")
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    destination.write_bytes(z.read(info))
    if hashlib.sha256(source.read_bytes()).hexdigest()!=CHECKPOINT_SHA256:
        raise ValueError("作者的预训练权重校验失败")
    # Convert this verified official full-module pickle once. Production loads
    # only tensor state with weights_only=True, across current PyTorch versions.
    converter = """
import pathlib,sys,torch
root,source,output=map(pathlib.Path,sys.argv[1:])
sys.path.insert(0,str(root))
net=torch.load(source,map_location='cpu',weights_only=False)
assert net.conv1.in_channels==4 and net.conv1.out_channels==32
torch.save({'state_dict':net.state_dict(),'input_channels':4,'channel_size':32},output)
print('GR-CNN RGB-D weights ready')
"""
    subprocess.run([config.VISION_PYTHON,"-c",converter,str(root),str(source),str(folder/"model.pt")],check=True)
    manifest = dict(repository="https://github.com/skumra/robotic-grasping",commit=COMMIT,
                    checkpoint=CHECKPOINT,checkpoint_sha256=CHECKPOINT_SHA256,python=config.VISION_PYTHON)
    (folder/"source.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("部署完成。测试：GRASP_BACKEND=grcnn python3 main.py")


if __name__=="__main__":
    main()
