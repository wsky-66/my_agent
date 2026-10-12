import hashlib
import importlib.util
from pathlib import Path
import zipfile

spec = importlib.util.spec_from_file_location("setup_grcnn", Path(__file__).resolve().parents[1]/"scripts/setup_grcnn.py")
setup_grcnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup_grcnn)


def test_deployment_keeps_only_runtime_sources_and_reuses_verified_checkpoint(monkeypatch, tmp_path):
    checkpoint = b"synthetic checkpoint"
    monkeypatch.setattr(setup_grcnn.config, "ROOT", tmp_path)
    monkeypatch.setattr(setup_grcnn, "CHECKPOINT_SHA256", hashlib.sha256(checkpoint).hexdigest())
    downloads, conversions = [], []
    files = {"LICENSE": b"licence", "README.md": b"source",
             "inference/__init__.py": b"", "inference/models/__init__.py": b"",
             "inference/models/grconvnet3.py": b"network", "inference/models/grasp_model.py": b"base",
             setup_grcnn.CHECKPOINT: checkpoint,
             "train_network.py": b"unused", "trained-models/unused/epoch": b"unused"}
    def run(command, check):
        if command[0] == "curl":
            downloads.append(command)
            with zipfile.ZipFile(command[-1], "w") as archive:
                archive.writestr("upstream/", b"")
                archive.comment = setup_grcnn.COMMIT.encode()
                for name, content in files.items():
                    archive.writestr("upstream/"+name, content)
        else:
            conversions.append(command)
            Path(command[-1]).write_bytes(b"converted")
    monkeypatch.setattr(setup_grcnn.subprocess, "run", run)
    setup_grcnn.main()
    folder = tmp_path/".runtime/grcnn"
    assert (folder/"model.pt").read_bytes() == b"converted"
    assert (folder/"source.json").is_file()
    kept = {str(p.relative_to(folder/"upstream")) for p in (folder/"upstream").rglob("*") if p.is_file()}
    assert kept == set(files)-{"train_network.py", "trained-models/unused/epoch"}
    assert not list(folder.rglob("*.zip"))
    setup_grcnn.main()
    assert len(downloads) == 1 and len(conversions) == 2
