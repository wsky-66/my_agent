from types import SimpleNamespace as NS
import threading
import time
import sys
import cv2
import numpy as np
import pytest

from workers.arm_worker import ArmBackend
from workers.camera_worker import CameraBackend
from core.health import ARM_JOINTS


def test_missing_joint_state_never_zeros_unknown_joints():
    backend = ArmBackend({'controller_dir': '/tmp', 'robot_name': 'wx250s'})
    names = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']
    called = []
    backend.clients['arm'] = NS(ARM_JOINTS=names)
    backend.clients['joint'] = NS(get_joints=lambda: {'waist': 1.0}, set_joints=called.append)
    result = backend('set_joints', {'joints': {'waist': .5}})
    assert result.error_code == 'JOINT_STATE_UNAVAILABLE' and not called


def test_partial_joint_target_preserves_all_other_current_angles():
    backend = ArmBackend({'controller_dir': '/tmp', 'robot_name': 'wx250s'})
    state = {'waist': 1.0, 'shoulder': .2, 'elbow': .3, 'forearm_roll': .4, 'wrist_angle': .5, 'wrist_rotate': .6}
    called = []
    backend.clients['arm'] = NS(ARM_JOINTS=list(state))
    backend.clients['joint'] = NS(get_joints=lambda: state,
                                 set_joints=lambda target: called.append(target) or True)
    assert backend('set_joints', {'joints': {'waist': .5}}).ok
    assert called == [{**state, 'waist': .5}]


def test_joint_client_rejects_stale_cache_before_legacy_zero_fallback(monkeypatch):
    sent = []
    state = NS(name=ARM_JOINTS, position=[.2] * 6)

    class LegacyJoint:
        def __init__(self, name):
            self._on_state(state)

        def _on_state(self, msg):
            self._state = msg

        def set_joints(self, joints):
            current = self.get_joints(timeout=.01) or dict.fromkeys(ARM_JOINTS, 0.)
            sent.append(current)
            return True

    monkeypatch.setitem(sys.modules, 'joint_client', NS(JointClient=LegacyJoint, ARM_JOINTS=ARM_JOINTS))
    backend = ArmBackend({'controller_dir': '/tmp', 'robot_name': 'wx250s'})
    joint = backend.joint()
    assert joint.get_joints(timeout=.01) == dict.fromkeys(ARM_JOINTS, .2)
    joint._feedback = time.monotonic()-5, state
    assert joint.get_joints(timeout=.01) is None
    with pytest.raises(RuntimeError, match='反馈丢失'):
        joint.set_joints({'waist': .5})
    assert not sent


@pytest.mark.parametrize('encoding,big_endian', [('16UC1', False), ('16UC1', True), ('32FC1', False)])
def test_rgb_depth_capture_handles_row_padding_units_and_endianness(tmp_path, encoding, big_endian):
    backend = CameraBackend.__new__(CameraBackend)
    backend._condition = threading.Condition()
    # RGB rows contain two padding bytes that must not become pixels.
    rgb = NS(height=2, width=2, step=8, encoding='rgb8', data=bytes([255,0,0, 0,255,0, 9,9]*2),
             header=NS(stamp=NS(sec=12, nanosec=500000000), frame_id='camera_color_optical_frame'))
    dtype = np.dtype('>u2' if big_endian else '<u2') if encoding == '16UC1' else np.dtype('<f4')
    values = np.full((2, 2), 500 if encoding == '16UC1' else .5, dtype=dtype)
    data = b''.join(row.tobytes()+b'\0\0' for row in values)
    depth = NS(height=2, width=2, step=2*dtype.itemsize+2, encoding=encoding,
               data=data, is_bigendian=big_endian)
    backend._pair = (time.monotonic(), rgb, depth)
    backend._info = (time.monotonic(), NS(width=2, height=2, k=[100,0,1, 0,100,1, 0,0,1]))
    result = backend.capture(tmp_path)
    assert result.ok and result.data['timestamp'] == 12.5
    assert np.load(result.data['depth_path']) == pytest.approx(np.full((2,2), .5))
    image = cv2.imread(result.data['image_path'])
    assert image[0,0].tolist() == [0,0,255] and image[0,1].tolist() == [0,255,0]
