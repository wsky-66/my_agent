"""Official GR-CNN RGB-D prediction, adapted to the calibrated stock gripper."""
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from core.grasp_geometry import ee_position, preferred_grasp_rotation, rotation_rpy
from workers.grasp_pipeline import GraspPipeline

INPUT_SIZE = 224


def grasp_peaks(quality, threshold, limit):
    maxima = (quality>threshold)&(quality==maximum_filter(quality,size=41,mode="constant"))
    points = np.argwhere(maxima)
    order = np.argsort(-quality[maxima],kind="stable")
    result = []
    for point in points[order]:
        if all(np.max(np.abs(point-other))>=20 for other in result):
            result.append(point)
            if len(result)==limit:break
    return result


def crop_target(color, depth, mask):
    """Square crop with context, preserving pixel coordinates even at image edges."""
    ys, xs = np.where(mask)
    # Cornell zoom augmentation keeps 0.5–1.0 of its native 224 px crop.
    side = max(INPUT_SIZE//2, int(np.ceil(max(np.ptp(xs)+1, np.ptp(ys)+1)*1.5)))
    left = int(np.floor((xs.min()+xs.max()+1-side)/2))
    top = int(np.floor((ys.min()+ys.max()+1-side)/2))
    h, w = depth.shape
    pad = ((max(0,-top),max(0,top+side-h)), (max(0,-left),max(0,left+side-w)))
    y, x = top+pad[0][0], left+pad[1][0]
    def crop(array, padding, mode):
        return np.pad(array,padding,mode=mode)[y:y+side,x:x+side]
    rgb = cv2.resize(crop(color,(*pad,(0,0)),"edge"),(INPUT_SIZE,INPUT_SIZE))
    dm = cv2.resize(crop(depth,pad,"edge"),(INPUT_SIZE,INPUT_SIZE),interpolation=cv2.INTER_NEAREST)
    target = cv2.resize(crop(mask.astype(np.uint8),pad,"constant"),(INPUT_SIZE,INPUT_SIZE),
                        interpolation=cv2.INTER_NEAREST)>0
    return rgb, dm, target, (left,top,side)


def planar_pose(pixel, angle, width_pixels, optical_depth, k, transform):
    """Map the image opening axis onto the horizontal grasp plane in base_link."""
    u,v = pixel
    origin = transform[:3,3]
    def ray(x,y):
        return transform[:3,:3] @ np.array([(x-k[2])/k[0],(y-k[5])/k[4],1.])
    position = origin + ray(u,v)*optical_depth
    axis = np.array([np.cos(angle),-np.sin(angle)])*width_pixels/2
    contacts = []
    for sign in (-1,1):
        direction = ray(u+sign*axis[0],v+sign*axis[1])
        if abs(direction[2])<1e-8:
            raise ValueError("抓取平面与相机射线平行")
        distance = (position[2]-origin[2])/direction[2]
        if distance<=0:
            raise ValueError("抓取接触点位于相机后方")
        contacts.append(origin+direction*distance)
    opening = contacts[1]-contacts[0]
    opening[2] = 0.
    width = float(np.linalg.norm(opening))
    if not np.isfinite(width) or width<=0:
        raise ValueError("抓取宽度无效")
    approach = np.array([0.,0.,-1.])
    rotation = preferred_grasp_rotation(np.column_stack((approach,opening/width,np.cross(approach,opening/width))))
    return position,rotation,width


class GrCNNPipeline(GraspPipeline):
    def __init__(self, opts):
        import torch
        root, checkpoint = Path(opts["grcnn_root"]), Path(opts["grcnn_checkpoint"])
        if not (root/"inference/models/grconvnet3.py").is_file() or not checkpoint.is_file():
            raise ValueError("GR-CNN 尚未部署，请运行 python3 scripts/setup_grcnn.py")
        sys.path.insert(0,str(root))
        from inference.models.grconvnet3 import GenerativeResnet
        device = opts.get("grcnn_device","auto")
        self.device = "cuda" if device=="auto" and torch.cuda.is_available() else ("cpu" if device=="auto" else device)
        state = torch.load(checkpoint,map_location="cpu",weights_only=True)
        self.model = GenerativeResnet(input_channels=state["input_channels"],channel_size=state["channel_size"])
        self.model.load_state_dict(state["state_dict"])
        self.model.eval().to(self.device)
        torch.set_num_threads(1)
        self.checkpoint = str(checkpoint)

    def prepare(self, color, depth, k, transform):
        self.color,self.depth,self.k,self.transform = color,depth,k,transform
        self.valid = np.isfinite(depth)&(depth>.2)&(depth<1.)
        ys,xs = np.where(self.valid)
        z = depth[self.valid]
        camera = np.column_stack(((xs-k[2])/k[0]*z,(ys-k[5])/k[4]*z,z))
        self.points = camera @ transform[:3,:3].T + transform[:3,3]
        self.colors = color[self.valid]/255.

    def predict_maps(self, rgb, dm):
        import torch
        # Fill only network input holes; execution coordinates use raw valid depth.
        valid = np.isfinite(dm)&(dm>.2)&(dm<1.)
        filled = np.where(valid,dm,0.).astype(np.float32)
        filled = cv2.inpaint(filled,(~valid).astype(np.uint8),1,cv2.INPAINT_NS)
        filled = np.clip(filled-filled.mean(),-1.,1.)
        rgb = rgb.astype(np.float32)/255.
        rgb -= rgb.mean()
        tensor = torch.from_numpy(np.concatenate((filled[None],rgb.transpose(2,0,1)))[None]).to(self.device)
        with torch.inference_mode():
            pred = self.model.predict(tensor)
        arrays = {name:value.cpu().numpy().squeeze() for name,value in pred.items()}
        # Training labels divide pixel opening by output_size/2. The upstream
        # post-process hardcodes 150 for a 300 px image; this network uses 224.
        return (gaussian_filter(arrays["pos"],2),
                np.arctan2(gaussian_filter(arrays["sin"],2),gaussian_filter(arrays["cos"],2))/2,
                gaussian_filter(arrays["width"]*(INPUT_SIZE/2),1))

    def collision_filter(self, grasps):
        # Reuse the existing model-free detector, without loading the GraspNet net
        # or applying its absolute 3 cm minimum-height filter to planar grasps.
        import graspnet_client
        from graspnetAPI import GraspGroup
        rows = [[g["score"],g["width"],.02,g["depth"],*np.ravel(g["rotation"]),*g["position"],i]
                for i,g in enumerate(grasps)]
        group = GraspGroup(np.asarray(rows,dtype=float))
        detector = graspnet_client.filter_collision(group,self.points)
        # The last column stores the source index only for round-tripping poses.
        # filter_collision keeps input ordering and all fields intact.
        return [grasps[int(row[-1])] for row in detector.grasp_group_array]

    def infer(self, mask, *, top, score_min, no_collision=False):
        started = time.monotonic()
        target = self.valid & (mask if mask is not None else True)
        diagnostics = dict(pipeline="grcnn",device=self.device,target_points=int(target.sum()))
        if not target.any():
            return [],dict(diagnostics,returned=0,reason="目标没有有效深度点")
        rgb,dm,target_crop,roi = crop_target(self.color,self.depth,target)
        q,angles,widths = self.predict_maps(rgb,dm)
        q = np.where(target_crop&np.isfinite(q),q,0.)
        peaks = grasp_peaks(q,max(.2,score_min),max(top,12))
        left,up,side = roi
        grasps = []
        rejected_width = rejected_geometry = 0
        for y,x in peaks:
            u,v = left+(x+.5)*side/INPUT_SIZE-.5, up+(y+.5)*side/INPUT_SIZE-.5
            px,py = int(round(u)),int(round(v))
            if not (0<=px<self.depth.shape[1] and 0<=py<self.depth.shape[0] and target[py,px]):
                continue
            # Median in a small raw-depth target neighbourhood, never inpainted.
            ys,xs = slice(max(0,py-2),py+3),slice(max(0,px-2),px+3)
            local = self.depth[ys,xs][target[ys,xs]]
            try:
                position,rotation,width = planar_pose([u,v],float(angles[y,x]),float(widths[y,x])*side/INPUT_SIZE,
                    float(np.median(local)),self.k,self.transform)
            except ValueError:
                rejected_geometry += 1
                continue
            if not np.isfinite(rotation).all() or not 0<width<=.074:
                rejected_width += 1
                continue
            depth = .01
            grasps.append(dict(score=float(np.clip(q[y,x],0,1)),position=position.tolist(),rotation=rotation.tolist(),
                rpy=rotation_rpy(rotation),width=width,depth=depth,
                ee_position=ee_position(position,rotation,depth,.015575).tolist(),
                execution_ready=not no_collision,pipeline="grcnn",pixel=[u,v],image_angle=float(angles[y,x])))
        diagnostics.update(peaks=len(peaks),before_collision=len(grasps),roi=list(roi),max_quality=float(np.max(q)),
                           rejected_width=rejected_width,rejected_geometry=rejected_geometry)
        if grasps and not no_collision:
            grasps = self.collision_filter(grasps)
        diagnostics["rejected_collision"] = diagnostics["before_collision"]-len(grasps)
        grasps = sorted(grasps,key=lambda g:g["score"],reverse=True)[:top]
        for i,g in enumerate(grasps):g["rank"]=i
        reason = ("没有超过分数下限的目标候选" if not peaks else
                  "候选均被夹爪开口或几何转换排除" if not diagnostics["before_collision"] else
                  "候选均被现有点云碰撞或空抓检测排除")
        diagnostics.update(returned=len(grasps),elapsed_s=round(time.monotonic()-started,3),
                           reason="" if grasps else reason)
        return grasps,diagnostics
