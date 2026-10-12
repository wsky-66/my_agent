"""GraspNet geometry and the installed stock wx250s gripper, in metres.

GraspNet X is approach, Y is opening, and its leading tips are at X=depth.
The robot pose targets ee_gripper_link, whose tips are further along local X.
"""
import itertools
import warnings

import numpy as np

STOCK_TIP_FROM_EE = .015575


def rigid_transform(value):
    t = np.asarray(value, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all() or
            not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6) or
            not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5) or
            not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-5)):
        raise ValueError("标定必须是有限、右手正交的刚体变换")
    return t


def visual_vertices(grasps):
    """Match graspnetAPI's lower-corner box construction, not centred boxes."""
    a = np.asarray(grasps).reshape(-1, 17)
    n = len(a); w = a[:, 1]; d = a[:, 3]
    fw = .004; db = .02; tail = .04; h = .004
    sizes = np.stack([np.stack([d+db+fw, d+db+fw, np.full(n, fw), np.full(n, tail)], 1),
                      np.stack([np.full(n, fw), np.full(n, fw), w, np.full(n, fw)], 1),
                      np.full((n, 4), h)], 2)
    shifts = np.stack([np.tile([-db-fw, -db-fw, -db-fw, -tail-fw-db], (n, 1)),
                       np.stack([-w/2-fw, w/2, -w/2, np.full(n, -fw/2)], 1),
                       np.full((n, 4), -h/2)], 2)
    corners = np.asarray(list(itertools.product((0, 1), repeat=3)))
    local = shifts[:, :, None] + sizes[:, :, None]*corners
    return np.einsum('nij,nbcj->nbci', a[:, 4:13].reshape(-1, 3, 3), local)+a[:, None, None, 13:16]


def ee_position(position, rotation, depth, tip_from_ee):
    return np.asarray(position)+np.asarray(rotation)[:, 0]*(depth-tip_from_ee)


def offset_grasp(grasp, offset):
    """Translate a model result in base_link, keeping its jaw/EE geometry intact.

    Apply once at the worker output boundary, after native model filtering.
    A new dictionary leaves cached/raw predictions untouched on repeated scans.
    """
    return {**grasp,
            "position": (np.asarray(grasp["position"])+offset).tolist(),
            "ee_position": (np.asarray(grasp["ee_position"])+offset).tolist(),
            "base_offset": list(offset)}


def rotation_rpy(rotation):
    from scipy.spatial.transform import Rotation
    # Euler angles are non-unique at pitch=90 degrees, but represent the same
    # checked matrix. Never fold roll by 180 degrees or alter the orientation.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Gimbal lock detected.*',category=UserWarning)
        return Rotation.from_matrix(rotation).as_euler('xyz',degrees=True).tolist()


def preferred_grasp_rotation(rotation):
    """Swap parallel jaws about approach X, before checking the physical pose.

    R @ Rx(pi) preserves approach/contact position and exchanges the jaws.
    This is a different tool pose, so its collision checks must run afterward.
    Explicit user-supplied move_arm orientations never use this operation.
    """
    r = np.asarray(rotation, dtype=float)
    roll = np.asarray(rotation_rpy(r))[..., 0]
    flip = np.abs(roll)>90.+1e-6
    return np.where(flip[...,None,None], r @ np.diag([1., -1., -1.]), r)
