"""Synthetic geometry and execution checks for the common model compensation."""
import copy
import sys
import time
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from core.grasp_geometry import ee_position, offset_grasp, visual_vertices
from core.results import ToolResult
from services import arm, grasp
from services.scene import ScenePlan, store
from workers.grasp_worker import GraspBackend
from workflows.manipulation import execute


def raw_grasp():
    rotation = Rotation.from_euler("xyz", [30, 65, -20], degrees=True).as_matrix()
    position = [.3, .05, .08]
    return dict(position=position, rotation=rotation.tolist(), rpy=[30, 65, -20],
                ee_position=ee_position(position, rotation, .01, .015575).tolist(),
                score=.8, width=.04, depth=.01, execution_ready=True)


def test_base_translation_preserves_tip_geometry_and_moves_the_entire_rviz_gripper():
    original = raw_grasp()
    unchanged = copy.deepcopy(original)
    offset = [-.015, 0., -.010]
    adjusted = offset_grasp(original, offset)
    assert original == unchanged
    assert adjusted["position"] == pytest.approx(np.asarray(original["position"])+offset)
    assert adjusted["ee_position"] == pytest.approx(np.asarray(original["ee_position"])+offset)
    assert adjusted["rotation"] == original["rotation"] and adjusted["rpy"] == original["rpy"]
    assert ee_position(adjusted["position"], adjusted["rotation"], adjusted["depth"], .015575) == pytest.approx(adjusted["ee_position"])
    def vertices(g):
        row = [g["score"], g["width"], .02, g["depth"], *np.ravel(g["rotation"]), *g["position"], -1.]
        return visual_vertices([row])
    assert vertices(adjusted) == pytest.approx(vertices(original)+offset)


@pytest.mark.parametrize("backend_name", ["graspnet", "grcnn"])
def test_model_offset_reaches_rviz_and_moveit_once_without_changing_place_target(monkeypatch, tmp_path, backend_name):
    import workers.grasp_pipeline as graspnet
    import workers.grcnn_pipeline as grcnn
    import workers.grasp_visualization as visualization
    original = raw_grasp()
    unchanged = copy.deepcopy(original)
    original["pipeline"] = backend_name
    unchanged["pipeline"] = backend_name
    loaded, displayed, calls = [], [], []
    pipeline = NS(prepare=lambda *args: None, infer=lambda *args, **kw: ([original], {}),
                  segmented_cloud=lambda masks: (np.empty((0, 3)), np.empty((0, 3))))
    def load(*args):
        loaded.append(args)
        return pipeline
    monkeypatch.setattr(graspnet, "GraspPipeline", load)
    monkeypatch.setattr(grcnn, "GrCNNPipeline", load)
    monkeypatch.setitem(sys.modules, "graspnet_client", NS(load_model=lambda: (None, None)))
    monkeypatch.setattr(visualization, "GraspVisualizer", lambda: NS(publish_scene=lambda *args: displayed.append(args)))
    monkeypatch.setattr(arm, "call", lambda name, **args: calls.append((name, args)) or ToolResult.success("done"))
    image, depth = tmp_path/"rgb.png", tmp_path/"depth.npy"
    cv2.imwrite(str(image), np.full((2, 2, 3), 128, np.uint8))
    np.save(depth, np.full((2, 2), .5))
    snapshot = dict(image_path=str(image), depth_path=str(depth), intrinsics=[100, 0, 1, 0, 100, 1, 0, 0, 1],
                    extrinsics=np.eye(4).tolist(), base_frame="base", calibration_id="synthetic",
                    timestamp=12., created=time.time())
    # Use the real service's configured worker options, including the compensation.
    import json
    import shlex
    command = shlex.split(grasp.client.command[-1])
    opts = json.loads(command[command.index("--options")+1])
    opts.update(controller_dir="/tmp", grasp_backend=backend_name)
    backend = GraspBackend(opts)
    try:
        for _ in range(2):
            result = backend("infer", dict(snapshot=snapshot, items=[dict(label="apple")]))
            candidate = result.data["apple"][0]
            offset = [-.015, 0., -.010]
            assert result.ok and candidate["base_offset"] == offset
            assert candidate["ee_position"] == pytest.approx(np.asarray(original["ee_position"])+offset)
            assert displayed[-1][2][0]["position"] == candidate["position"]
            cached = copy.deepcopy(candidate)
            store.plan = ScenePlan("test", snapshot, {"apple": dict(ok=True, grasps=[candidate])})
            calls.clear()
            executed = execute([dict(object="apple", target_position=[.25, -.1, .03])], "test", return_to_sleep=False)
            assert executed.ok and len(calls) == 9
            approach, descent, lift = calls[1][1], calls[2][1], calls[4][1]
            x, y, z = candidate["ee_position"]
            for args, height in [(approach, z+.1), (descent, z), (lift, z+.08)]:
                assert [args["x"], args["y"], args["z"]] == pytest.approx([x, y, height])
            assert approach["grasp_debug"]["position"] == candidate["position"]
            assert executed.data["strategies"][0]["position"] == candidate["ee_position"]
            assert [calls[6][1][k] for k in ("x", "y", "z")] == pytest.approx([.25, -.1, .07])
            assert candidate == cached and original == unchanged
        assert len(loaded) == 1
    finally:
        store.clear()
