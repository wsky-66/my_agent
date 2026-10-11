"""Tabletop search: decode alternatives, then check the whole visible scene."""
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree

from core.grasp_geometry import (PIPELINE_VERSION, StockGripper, ee_position, visual_vertices,
                                 rotation_rpy, preferred_grasp_rotation)


class GraspRejected(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def fit_table(base_cloud, valid, masks):
    import open3d as o3d
    keep = valid.copy()
    for mask in masks:
        if mask is not None:
            keep &= ~cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    points = base_cloud[keep]
    points = points[(points[:, 2]>-.06) & (points[:, 2]<.15)]
    if len(points)<500:
        raise GraspRejected('TABLE_UNAVAILABLE', '没有足够桌面深度点，停止抓取推理')
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc = pc.voxel_down_sample(.005)
    o3d.utility.random.seed(0)
    plane, indices = pc.segment_plane(distance_threshold=.003, ransac_n=3, num_iterations=1200)
    plane = np.asarray(plane); plane /= np.linalg.norm(plane[:3])
    if plane[2]<0:
        plane *= -1
    inliers = np.asarray(pc.points)[indices]
    if (len(indices)<300 or len(indices)<.30*len(pc.points) or
            plane[2]<np.cos(np.radians(12)) or
            np.any(np.ptp(inliers[:, :2], axis=0)<.20)):
        raise GraspRejected('TABLE_UNAVAILABLE', '未找到可靠、覆盖足够范围的桌面平面，停止抓取推理')
    return plane, {'plane_base': plane.tolist(), 'inliers': len(indices), 'fit_points': len(pc.points),
                   'residual_p95_mm': float(np.quantile(abs(inliers @ plane[:3]+plane[3]), .95)*1000)}


def decode_alternatives(ep, backend, gripper, score_min):
    """Vectorised decoding of original scores/widths for all 12x4 bins."""
    import torch
    from loss_utils import batch_viewpoint_params_to_matrix, GRASP_MAX_TOLERANCE, GRASP_MAX_WIDTH
    objectness = ep['objectness_score'][0].argmax(0)==1
    scores = (ep['grasp_score_pred'][0]*ep['grasp_tolerance_pred'][0]/GRASP_MAX_TOLERANCE).permute(1, 0, 2)
    widths = torch.clamp(1.2*ep['grasp_width_pred'][0], min=0, max=GRASP_MAX_WIDTH).permute(1, 0, 2)
    keep = objectness[:, None, None] & torch.isfinite(scores) & (scores>score_min)
    keep &= torch.isfinite(widths) & (widths>=gripper.gap_min) & (widths<=gripper.gap_max)
    seed, angle, d = keep.nonzero(as_tuple=True)
    if len(seed)==0:
        return np.empty((0, 17))
    r = batch_viewpoint_params_to_matrix(-ep['grasp_top_view_xyz'][0][seed], angle.float()*np.pi/12)
    centers = ep['fp2_xyz'][0][seed]
    a = torch.cat([scores[seed, angle, d, None], widths[seed, angle, d, None],
                   torch.full((len(seed), 1), .02, device=scores.device),
                   (d[:, None].float()+1)*.01, r.reshape(-1, 9), centers,
                   torch.full((len(seed), 1), -1., device=scores.device)], 1)
    return a.detach().cpu().numpy()


class TabletopPipeline:
    def __init__(self, backend, model, opts):
        self.backend, (self.net, self.device), self.opts = backend, model, opts
        self.gripper = StockGripper(opts['gripper_mesh_dir'], opts['robot_name'])

    def prepare(self, color, depth, k, transform, masks):
        import open3d as o3d
        camera = self.backend.CamInfo(depth.shape[1], depth.shape[0], k[0], k[4], k[2], k[5], scale=1.)
        self.cloud = self.backend.create_point_cloud_from_depth_image(depth, camera, organized=True)
        self.base = self.cloud @ transform[:3, :3].T+transform[:3, 3]
        self.valid = np.isfinite(depth) & (depth>.2) & (depth<1.)
        self.plane, self.table = fit_table(self.base, self.valid, masks)
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.base[self.valid]))
        self.scene_points = np.asarray(pc.voxel_down_sample(.003).points)
        self.tree = cKDTree(self.scene_points)
        self.color, self.depth, self.k, self.transform = color, depth, k, transform

    def infer(self, mask, box, *, top, score_min, no_collision=False):
        import torch
        from loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix
        start = time.monotonic()
        b = self.backend; k = self.k; t = self.transform; gripper = self.gripper
        heights = self.base @ self.plane[:3]+self.plane[3]
        target = self.valid & (mask if mask is not None else heights>.005)
        target_points = self.base[target]
        # Apply source-pixel labels BEFORE downsampling: projecting voxel
        # centroids back to pixels can incorrectly relabel target edge points.
        import open3d as o3d
        obstacle_cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.base[self.valid & ~target]))
        obstacles=np.asarray(obstacle_cloud.voxel_down_sample(.003).points)
        obstacle_tree=cKDTree(obstacles)
        diagnostics = {'table': self.table, 'target_points': len(target_points), 'regions': [],
                       'collision_voxel_mm': 3, 'table_margin_mm': 3, 'pipeline_version': PIPELINE_VERSION,
                       'surface_tolerance_mm': 1, 'collision_by_part': {},
                       'collision_model': 'individual convex covers of stock URDF meshes with approach and closing sweeps'}
        if len(target_points)<30:
            diagnostics['reason'] = '目标有效深度点不足'
            return [], diagnostics
        if mask is None:
            rois = [('scene', target)]
        else:
            y, x = np.where(mask)
            if not len(x):
                diagnostics['reason'] = '目标掩码为空'
                return [], diagnostics
            if box is None:
                box = [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)]
            x1, y1, x2, y2 = map(int, box)
            margin = self.opts.get('context_px', 35)
            context = np.zeros_like(mask)
            context[max(0,y1-margin):min(mask.shape[0],y2+margin), max(0,x1-margin):min(mask.shape[1],x2+margin)] = True
            rois = [('object', mask), ('local_support', cv2.dilate(mask.astype(np.uint8), np.ones((25,25),np.uint8))>0),
                    ('context', context)]
        views = generate_grasp_views(b.NUM_VIEW).to(self.device)
        normal = torch.as_tensor(t[:3, :3].T @ self.plane[:3], dtype=views.dtype, device=self.device)
        allowed = views @ normal >= np.cos(np.radians(30))
        all_candidates = []
        for roi_name, roi in rois:
            if (roi & self.valid).sum()<30:
                continue
            np.random.seed(0); torch.manual_seed(0)
            endpoint, _ = b.build_endpoint(self.color, self.depth, k[0], k[4], k[2], k[5], t, roi)
            endpoint['point_clouds'] = endpoint['point_clouds'].to(self.device)
            with torch.no_grad():
                stage1 = self.net.view_estimator(endpoint)
                scores = stage1['view_score'].clone(); scores[:, :, ~allowed] = -float('inf')
                ranked_scores, ranked_views = scores.topk(3, dim=-1)
                decoded = []
                for rank in range(3):
                    ep = dict(stage1); inds = ranked_views[:, :, rank]; vp = views[inds]
                    ep.update(grasp_top_view_inds=inds, grasp_top_view_score=ranked_scores[:, :, rank], grasp_top_view_xyz=vp,
                        grasp_top_view_rot=batch_viewpoint_params_to_matrix(-vp.reshape(-1,3),
                            torch.zeros(vp.shape[0]*vp.shape[1],device=self.device)).reshape(vp.shape[0],vp.shape[1],3,3))
                    decoded.append(decode_alternatives(self.net.grasp_generator(ep), b, gripper, score_min))
            a = np.concatenate(decoded)
            a=a[np.isfinite(a).all(axis=1)]
            region = {'name': roi_name, 'positive_score_and_physical_width': len(a)}
            if len(a):
                u = np.rint(a[:,13]/a[:,15]*k[0]+k[2]).astype(int)
                v = np.rint(a[:,14]/a[:,15]*k[4]+k[5]).astype(int)
                inside = (u>=0)&(u<target.shape[1])&(v>=0)&(v<target.shape[0])
                keep = inside.copy(); keep[inside] = target[v[inside],u[inside]]
                a = a[keep]
                gg = b.GraspGroup(a); gg.transform(t); a = gg.grasp_group_array
                if len(a):
                    # Apply the stock parallel-jaw symmetry BEFORE every
                    # physical check; execution receives this same matrix.
                    a[:,4:13] = preferred_grasp_rotation(a[:,4:13].reshape(-1,3,3)).reshape(-1,9)
                    mesh_height = (visual_vertices(a) @ self.plane[:3]+self.plane[3]).min(axis=(1,2))
                    closed=a.copy()
                    closing_q=np.maximum(gripper.lower,gripper.opening(a[:,1]/1.2)-.0005)
                    closed[:,1]=2*(closing_q-gripper.gap_offset)
                    a = a[(mesh_height>=.003) & (gripper.minimum_clearance(a, self.plane)>=.003)
                          & (gripper.minimum_clearance(closed, self.plane)>=.003)]
            region['after_target_and_table'] = len(a)
            diagnostics['regions'].append(region)
            all_candidates.append(a)
        a = np.concatenate(all_candidates) if all_candidates else np.empty((0,17))
        a = a[np.argsort(-a[:, 0], kind='stable')]
        passing = []; tested = 0; collision_rejections = 0; empty_rejections = 0;closing_rejections=0
        budget = self.opts.get('collision_budget', 2000)
        for g in a[:budget]:
            tested += 1
            local = (target_points-g[13:16]) @ g[4:13].reshape(3,3)
            interior = ((local[:,0]>=-.02)&(local[:,0]<=g[3]) &
                        (abs(local[:,1])<g[1]/2)&(abs(local[:,2])<.0145))
            if interior.sum()<3:
                empty_rejections += 1
                continue
            collision = None if no_collision else gripper.collision_details(g, self.scene_points, self.tree)
            if collision is not None:
                collision_rejections += 1
                counts=diagnostics['collision_by_part'];part=collision['part']
                counts[part]=counts.get(part,0)+1
                diagnostics.setdefault('sample_collision',collision)
                continue
            if not no_collision and gripper.closing_collides(g,obstacles,obstacle_tree):
                closing_rejections+=1
                continue
            passing.append(g)
            gg = b.GraspGroup(np.asarray(passing)); gg.nms(translation_thresh=.01,rotation_thresh=.3); gg.sort_by_score()
            if len(gg)>=top:
                break
        diagnostics.update(candidates_before_scene_collision=len(a), collision_tested=tested,
                           collision_rejections=collision_rejections, empty_rejections=empty_rejections,
                           closing_collision_rejections=closing_rejections,
                           budget_exhausted=tested==budget and len(a)>budget,
                           elapsed_s=round(time.monotonic()-start,3))
        grasps = []
        if passing:
            gg = b.GraspGroup(np.asarray(passing)); gg.nms(translation_thresh=.01,rotation_thresh=.3); gg.sort_by_score()
            for g in gg.grasp_group_array[:top]:
                r = g[4:13].reshape(3,3); open_q, close_q = gripper.commands(g)
                ee = ee_position(g[13:16], r, g[3], gripper.tip_from_ee)
                grasps.append({'rank':len(grasps), 'score':float(g[0]), 'position':g[13:16].tolist(),
                    'rotation':r.tolist(), 'rpy':rotation_rpy(r),
                    'width':float(g[1]), 'depth':float(g[3]), 'ee_position':ee.tolist(),
                    'tip_from_ee':gripper.tip_from_ee, 'gripper_gap_offset':gripper.gap_offset,
                    'gripper_open':open_q, 'gripper_close':close_q, 'approach_distance':gripper.approach,
                    'table_clearance':float(gripper.minimum_clearance(g[None],self.plane)[0]),
                    'pipeline_version':PIPELINE_VERSION, 'collision_checked':not no_collision,
                    'execution_ready':not no_collision and mask is not None, 'table_plane':self.plane.tolist()})
        diagnostics['returned'] = len(grasps)
        if not grasps:
            diagnostics['reason'] = ('候选检查达到计算预算，请查看逐级数量' if diagnostics['budget_exhausted'] else
                                     '没有候选同时通过开口、桌面、目标内部点和全场景夹爪碰撞检查')
        return grasps, diagnostics
