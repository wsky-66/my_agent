"""The original GraspNet pipeline, reused with one loaded model and RGB-D scene."""
import time
from types import SimpleNamespace
import numpy as np
from core.grasp_geometry import STOCK_TIP_FROM_EE, ee_position, preferred_grasp_rotation, rotation_rpy


class GraspPipeline:
    def __init__(self, backend, model):
        self.backend, self.model = backend, model

    def prepare(self, color, depth, k, transform):
        self.color, self.depth, self.k, self.transform = color, depth, k, transform
        b = self.backend
        b.IMAGE_ROWS, b.IMAGE_COLS = depth.shape
        cam = b.CamInfo(depth.shape[1], depth.shape[0], k[0], k[4], k[2], k[5], scale=1.)
        cloud = b.create_point_cloud_from_depth_image(depth, cam, organized=True)
        self.valid = np.isfinite(depth) & (depth>b.DEPTH_MIN) & (depth<b.DEPTH_MAX)
        self.points = cloud[self.valid] @ transform[:3,:3].T + transform[:3,3]
        self.colors = color[self.valid]/255.

    def segmented_cloud(self, masks):
        """Display only the union of masks, keeping inference's full scene intact."""
        selected = np.zeros_like(self.valid, dtype=bool)
        for mask in masks:
            if mask is not None:
                selected |= mask
        selected = selected[self.valid]
        return self.points[selected], self.colors[selected]

    def infer(self, mask, *, top, score_min, no_collision=False):
        import torch
        start = time.monotonic()
        b = self.backend
        valid = self.valid & (mask if mask is not None else True)
        if not valid.any():
            return [], {'returned':0, 'reason':'目标没有有效深度点', 'pipeline':'original'}
        np.random.seed(0); torch.manual_seed(0)
        args = SimpleNamespace(top=top, score_min=score_min, no_collision=no_collision,
                               no_z=False, grip_z_min=b.GRIP_Z_MIN)
        gg, _ = b.run_pipeline(*self.model, self.color, self.depth,
                              self.k[0], self.k[4], self.k[2], self.k[5], self.transform, mask, args)
        grasps = []
        for i in range(len(gg)):
            r = preferred_grasp_rotation(gg.rotation_matrices[i])
            position, depth = gg.translations[i], float(gg.depths[i])
            grasps.append({'rank':i, 'score':float(gg.scores[i]), 'position':position.tolist(),
                'rotation':r.tolist(), 'rpy':rotation_rpy(r), 'width':float(gg.widths[i]), 'depth':depth,
                'ee_position':ee_position(position,r,depth,STOCK_TIP_FROM_EE).tolist(),
                'execution_ready':not no_collision, 'pipeline':'original'})
        return grasps, {'returned':len(grasps), 'pipeline':'original',
                        'elapsed_s':round(time.monotonic()-start,3),
                        'reason':'' if grasps else '原版 GraspNet 未返回抓取 pose'}
