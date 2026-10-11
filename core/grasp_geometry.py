"""GraspNet geometry and the installed stock wx250s gripper, in metres.

GraspNet X is approach, Y is opening, and its leading tips are at X=depth.
The robot pose targets ee_gripper_link, whose tips are further along local X.
"""
import itertools
import struct
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

PIPELINE_VERSION = 3


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


def read_stl(path):
    raw = Path(path).read_bytes()
    if len(raw) < 84:
        raise ValueError("夹爪 STL 文件无效")
    count = struct.unpack_from('<I', raw, 80)[0]
    if count == 0 or len(raw) != 84+50*count:
        raise ValueError("夹爪需要有效的二进制 STL 网格")
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attr', '<u2')])
    vertices = np.frombuffer(raw, dtype, count=count, offset=84)['vertices'].astype(float)*.001
    if not np.isfinite(vertices).all():
        raise ValueError("夹爪网格包含无效坐标")
    return vertices


class StockGripper:
    """Stock wx250s URDF geometry. Individual convex covers are conservative.

    Covers avoid relying on watertightness: the installed finger/bar meshes
    contain topology defects. Swept covers check the entire straight approach.
    """
    lower = .015
    upper = .037
    ee_from_fingers = .027575

    def __init__(self, mesh_dir, robot_name='wx250s', approach=.10):
        if robot_name != 'wx250s':
            raise ValueError("当前抓取几何只支持 wx250s 原装夹指")
        self.approach = approach
        folder = Path(mesh_dir)
        finger = read_stl(folder/'gripper_finger.stl')
        left = finger.reshape(-1, 3) @ np.diag([-1, -1, 1])
        right = finger.reshape(-1, 3) @ np.diag([-1, 1, -1])
        self.front = float(max(left[:, 0].max(), right[:, 0].max()))
        self.tip_from_ee = self.front-self.ee_from_fingers
        # Validate contact sections for all four network depth bins.
        offsets = []
        for depth in (.01, .02, .03, .04):
            x = depth-self.front
            sections = []
            for i, j in ((0, 1), (1, 2), (2, 0)):
                p, q = finger[:, i], finger[:, j]
                dx = q[:, 0]-p[:, 0]
                keep = (abs(dx)>1e-9) & ((p[:, 0]-x)*(q[:, 0]-x)<=0)
                f = (x-p[keep, 0])/dx[keep]
                sections.extend((p[keep]+f[:, None]*(q[keep]-p[keep]))[:, 1])
            if not sections:
                raise ValueError("夹指网格无法建立抓取接触截面")
            offsets.append(max(sections)-.005)
        if not np.allclose(offsets, offsets[0], atol=1e-5):
            raise ValueError("夹指接触截面与原装模型不符，需重新标定开口映射")
        self.gap_offset = float(offsets[0])
        self.gap_min = 2*(self.lower-self.gap_offset)
        self.gap_max = 2*(self.upper-self.gap_offset)
        # The installed stock URDF: finger mesh origins +/-5 mm; each body
        # mesh's net offset from fingers_link is -86 mm and Rz(90 degrees).
        rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        parts = [(left, 1), (right, -1)]
        for name in ('gripper', 'gripper_bar', 'gripper_prop'):
            vertices=read_stl(folder/(name+'.stl')).reshape(-1,3) @ rz.T+[-.086,0,0]
            if name=='gripper_prop':
                # This motor link rotates about X while the fingers close.
                # Bound every rotor angle rather than assume zero feedback.
                radius=float(np.linalg.norm(vertices[:,1:3],axis=1).max())
                vertices=np.asarray(list(itertools.product((vertices[:,0].min(),vertices[:,0].max()),
                                                           (-radius,radius),(-radius,radius))))
            parts.append((vertices,0))
        self.parts = []
        self.closing_hulls = []
        for vertices, sign in parts:
            vertices = np.unique(vertices, axis=0)
            hull = ConvexHull(vertices)
            vertices = vertices[hull.vertices]
            swept = ConvexHull(np.concatenate([vertices, vertices+[-approach, 0, 0]]))
            self.parts.append((vertices, sign, swept.equations))
            self.closing_hulls.append(ConvexHull(vertices).equations)
        if not .01 < self.tip_from_ee < .02 or not .04 < self.gap_max < .07:
            raise ValueError("安装的夹爪尺寸与原装 wx250s 几何不符")

    def opening(self, widths):
        return np.asarray(widths)/2+self.gap_offset

    def minimum_clearance(self, grasps, plane):
        a = np.asarray(grasps).reshape(-1, 17)
        normal = a[:, 4:13].reshape(-1, 3, 3).transpose(0, 2, 1) @ np.asarray(plane[:3])
        q = self.opening(a[:, 1])
        origin = a[:, 13:16] @ plane[:3]+plane[3]+normal[:, 0]*(a[:, 3]-self.front)
        minimum = np.full(len(a), np.inf)
        for vertices, sign, _ in self.parts:
            height = (normal @ vertices.T).min(axis=1)
            if sign:
                height += normal[:, 1]*sign*(q+.005)
            minimum = np.minimum(minimum, origin+height)
        return minimum

    def collides(self, grasp, points, tree, margin=-.001):
        return self.collision_details(grasp, points, tree, margin) is not None

    def collision_details(self, grasp, points, tree, margin=-.001):
        """Reject penetration beyond a 1 mm depth-surface uncertainty band.

        All scene points still participate. The table has its own 3 mm margin;
        a near-surface point alone is not evidence of gripper interpenetration.
        """
        r = grasp[4:13].reshape(3, 3); t = grasp[13:16]
        q = float(self.opening(grasp[1])); shift = grasp[3]-self.front
        radius = 0.
        for vertices, sign, _ in self.parts:
            offset = np.array([shift, sign*(q+.005) if sign else 0., 0.])
            radius = max(radius, np.linalg.norm(vertices+offset,axis=1).max(),
                         np.linalg.norm(vertices+offset+[-self.approach,0,0],axis=1).max())
        indices = tree.query_ball_point(t, float(radius)+max(0.,margin)*2)
        if not indices:
            return None
        local = (points[indices]-t) @ r
        local[:, 0] -= shift
        names = ('left_finger', 'right_finger', 'gripper', 'gripper_bar', 'gripper_prop')
        for name, (vertices, sign, equations) in zip(names, self.parts):
            p = local.copy()
            if sign:
                p[:, 1] -= sign*(q+.005)
            lo = vertices.min(axis=0)-[self.approach, 0, 0]-margin
            hi = vertices.max(axis=0)+margin
            nearby = ((p>=lo) & (p<=hi)).all(axis=1)
            if not nearby.any():
                continue
            distances = (p[nearby] @ equations[:, :3].T+equations[:, 3]).max(axis=1)
            inside = distances<=margin
            if inside.any():
                hit_indices = np.asarray(indices)[nearby][inside]
                return {"part": name, "points": int(inside.sum()),
                        "max_penetration_mm": float(-distances[inside].min()*1000),
                        "sample_points_base": points[hit_indices[:3]].tolist()}
        return None

    def commands(self, grasp):
        open_q = float(self.opening(grasp[1]))
        # GraspNet's decoder includes 1.2x opening margin. A small inset is
        # position control, not a claim of force control or grasp confirmation.
        close_q = max(self.lower, float(self.opening(grasp[1]/1.2))-.0005)
        if not self.lower-1e-7 <= close_q <= open_q <= self.upper+1e-7:
            raise ValueError("抓取开口无法映射到夹爪关节限位")
        return min(self.upper,max(self.lower,open_q)),min(self.upper,max(self.lower,close_q))

    def closing_collides(self, grasp, obstacles, tree, margin=-.001):
        """Exclude only the segmented target when checking the closing sweep."""
        r=grasp[4:13].reshape(3,3);t=grasp[13:16]
        opening,closing=self.commands(grasp);delta=opening-closing
        indices=tree.query_ball_point(t,.15)
        if not indices:
            return False
        local=(obstacles[indices]-t) @ r
        local[:,0]-=grasp[3]-self.front
        for (vertices,sign,_),equations in zip(self.parts,self.closing_hulls):
            p=local.copy()
            if sign:
                p[:,1]-=sign*(opening+.005)
            shift=np.array([0.,-sign*delta,0.])
            lo=np.minimum(vertices.min(0),vertices.min(0)+shift)-margin
            hi=np.maximum(vertices.max(0),vertices.max(0)+shift)+margin
            p=p[((p>=lo)&(p<=hi)).all(1)]
            extension=np.maximum(0.,-sign*equations[:,1]*delta)
            if len(p) and np.any(np.all(p @ equations[:,:3].T+equations[:,3]<=margin+extension,axis=1)):
                return True
        return False
