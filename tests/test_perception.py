import time
import cv2
import numpy as np
import pytest

from config import config
from services import perception, segmentation, camera


@pytest.fixture
def snapshot(tmp_path):
    depth = tmp_path / 'depth.npy'
    np.save(depth, np.full((5, 5), .5, dtype=np.float32))
    transform = np.eye(4)
    transform[:3, 3] = [.1, .2, .3]
    return {'image_path': str(tmp_path / 'color.png'), 'depth_path': str(depth),
            'intrinsics': [100, 0, 2, 0, 100, 2, 0, 0, 1], 'extrinsics': transform.tolist(),
            'base_frame': 'wx250s/base_link', 'timestamp': 123, 'created': time.time()}


def test_pixel_geometry_transforms_depth_to_robot_frame(snapshot):
    result = perception.position(px=2, py=2, radius=1, snapshot=snapshot)
    assert result.ok and result.data['npix'] == 9
    assert result.data['position'] == pytest.approx([.1, .2, .8])
    assert result.data['frame'] == 'wx250s/base_link'


def test_empty_depth_and_mask_shape_are_rejected(snapshot, tmp_path):
    mask = tmp_path / 'mask.png'
    cv2.imwrite(str(mask), np.ones((3, 3), np.uint8)*255)
    assert perception.position(mask_path=str(mask), snapshot=snapshot).error_code == 'MASK_MISMATCH'
    np.save(snapshot['depth_path'], np.zeros((5, 5)))
    assert perception.position(px=2, py=2, snapshot=snapshot).error_code == 'INVALID_DEPTH'


def test_old_or_invalid_calibration_does_not_produce_coordinates(snapshot):
    snapshot['extrinsics'][0][0] = float('nan')
    assert perception.position(px=2, py=2, snapshot=snapshot).error_code == 'INVALID_CALIBRATION'
    snapshot['created'] = time.time()-config.SCENE_MAX_AGE-1
    assert perception.position(px=2, py=2, snapshot=snapshot).error_code == 'SCENE_EXPIRED'


def test_mask_from_another_image_is_rejected_without_recapturing(monkeypatch, snapshot, tmp_path):
    mask = tmp_path / 'mask.png'
    cv2.imwrite(str(mask), np.ones((5, 5), np.uint8)*255)
    monkeypatch.setattr(camera, 'latest', lambda: snapshot)
    monkeypatch.setattr(camera, 'capture', lambda: pytest.fail('must reuse snapshot'))
    monkeypatch.setattr(segmentation, '_mask_sources', {str(mask.resolve()): ('other.png', mask.stat().st_mtime_ns)})
    assert perception.position(mask_path=str(mask)).error_code == 'MASK_SCENE_MISMATCH'
