import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from workers.grcnn_pipeline import GrCNNPipeline, crop_target, planar_pose, INPUT_SIZE


def scene():
    pipeline=GrCNNPipeline.__new__(GrCNNPipeline)
    pipeline.device='test'
    transform=np.eye(4)
    transform[:3,:3]=np.diag([1.,-1.,-1.])
    transform[2,3]=.55
    pipeline.prepare(np.full((120,160,3),128,np.uint8),np.full((120,160),.5),
                     [500,0,80,0,500,60,0,0,1],transform)
    mask=np.zeros((120,160),bool);mask[45:76,60:101]=True
    def maps(*args):
        q=np.zeros((INPUT_SIZE,INPUT_SIZE));q[112,112]=.7;q[10,10]=.99
        return q,np.zeros_like(q),np.full_like(q,60.)
    pipeline.predict_maps=maps
    return pipeline,mask


@pytest.mark.parametrize('angle',[0.,.4,-.8])
def test_oblique_camera_opening_axis_is_projected_onto_horizontal_grasp_plane(angle):
    transform=np.eye(4);transform[:3,:3]=Rotation.from_euler('xyz',[175,25,-15],degrees=True).as_matrix()
    transform[:3,3]=[.1,.2,.8]
    k=[600,0,160,0,400,120,0,0,1]
    pixel,width=[200,130],40.
    p,r,opening=planar_pose(pixel,angle,width,.55,k,transform)
    assert r[:,0]==pytest.approx([0,0,-1])
    assert r[:,1][2]==pytest.approx(0.) and np.linalg.det(r)==pytest.approx(1.)
    camera=transform[:3,:3].T@(p-transform[:3,3])
    assert camera==pytest.approx([.55*(pixel[0]-k[2])/k[0],.55*(pixel[1]-k[5])/k[4],.55])
    contacts=[]
    for sign in (-1,1):
        u=pixel[0]+sign*width/2*np.cos(angle)
        v=pixel[1]-sign*width/2*np.sin(angle)
        direction=transform[:3,:3]@np.array([(u-k[2])/k[0],(v-k[5])/k[4],1.])
        contacts.append(transform[:3,3]+direction*(p[2]-transform[2,3])/direction[2])
    axis=contacts[1]-contacts[0]
    assert opening==pytest.approx(np.linalg.norm(axis))
    # Parallel jaws may exchange sides when canonicalising roll. Perspective
    # also makes the two ray intersections asymmetric about the centre pixel.
    assert abs(np.dot(axis/opening,r[:,1]))==pytest.approx(1.)


def test_edge_crop_preserves_target_without_stretching_or_losing_original_coordinates():
    depth=np.full((120,160),.5);rgb=np.full((120,160,3),128,np.uint8)
    mask=np.zeros_like(depth,bool);mask[:10,:8]=True
    color,dm,selected,(left,top,side)=crop_target(rgb,depth,mask)
    assert color.shape==(224,224,3) and dm.shape==selected.shape==(224,224)
    assert left<0 and top<0 and side==INPUT_SIZE//2
    ys,xs=np.where(selected)
    u=left+(xs+.5)*side/224-.5;v=top+(ys+.5)*side/224-.5
    assert u.min()>=-1 and u.max()<9 and v.min()>=-1 and v.max()<11


def test_planar_prediction_uses_raw_depth_target_only_and_tip_offset():
    pipeline,mask=scene()
    poses,diag=pipeline.infer(mask,top=3,score_min=0,no_collision=True)
    assert len(poses)==1 and diag['pipeline']=='grcnn'
    g=poses[0]
    u,v=g['pixel']
    assert mask[round(v),round(u)] and g['score']==pytest.approx(.7)
    assert g['position'][2]==pytest.approx(.05)
    assert g['ee_position'][2]==pytest.approx(.055575)
    assert g['execution_ready'] is False
    assert np.array(g['rotation'])[:,0]==pytest.approx([0,0,-1])


def test_collision_failure_does_not_mark_a_candidate_executable():
    pipeline,mask=scene();seen=[]
    pipeline.collision_filter=lambda values:seen.extend(values) or []
    poses,diag=pipeline.infer(mask,top=3,score_min=0)
    assert seen and not poses and diag['before_collision']==1 and diag['returned']==0


def test_empty_depth_never_calls_the_network():
    pipeline,mask=scene();pipeline.valid[:]=False
    pipeline.predict_maps=lambda *args:pytest.fail('missing raw depth must not produce motion pose')
    poses,diag=pipeline.infer(mask,top=3,score_min=0)
    assert not poses and diag['target_points']==0


def test_network_input_units_width_scale_and_parallel_jaw_angle_wrap(monkeypatch):
    pipeline,_=scene();inputs=[]
    class Tensor:
        def __init__(self,value):self.value=value
        def to(self,device):inputs.append(self.value);return self
        def cpu(self):return self
        def numpy(self):return self.value
    monkeypatch.setitem(sys.modules,'torch',NS(from_numpy=Tensor,inference_mode=nullcontext))
    angles=np.full((224,224),np.deg2rad(89.));angles[:,112:]=np.deg2rad(-89.)
    arrays=dict(pos=np.ones_like(angles),cos=np.cos(2*angles),sin=np.sin(2*angles),width=np.full_like(angles,.5))
    pipeline.model=NS(predict=lambda value:{key:Tensor(v) for key,v in arrays.items()})
    depth=np.full((224,224),.5,np.float32);depth[:,112:]=.52
    q,angle,width=GrCNNPipeline.predict_maps(pipeline,np.full((224,224,3),128,np.uint8),depth)
    assert inputs[0].shape==(1,4,224,224) and inputs[0].dtype==np.float32
    assert inputs[0][0,0,100,50]==pytest.approx(-.01,abs=1e-6)
    assert inputs[0][0,1:].max()==pytest.approx(0.)
    assert width[100,100]==pytest.approx(56.)
    assert abs(angle[100,111])>np.deg2rad(88.)


def test_worker_selects_grcnn_once_and_reuses_model_for_next_scene(monkeypatch,tmp_path):
    import cv2
    import workers.grcnn_pipeline as grcnn
    import workers.grasp_visualization as visualization
    from workers.grasp_worker import GraspBackend
    image,depth,mask=tmp_path/'image.png',tmp_path/'depth.npy',tmp_path/'mask.png'
    cv2.imwrite(str(image),np.full((2,2,3),128,np.uint8));np.save(depth,np.full((2,2),.5))
    cv2.imwrite(str(mask),np.full((2,2),255,np.uint8))
    loaded,prepared=[],[]
    pipeline=NS(prepare=lambda *args:prepared.append(args),infer=lambda *a,**kw:([],{'pipeline':'grcnn'}),
                segmented_cloud=lambda masks:(np.empty((0,3)),np.empty((0,3))))
    def load(opts):loaded.append(opts);return pipeline
    monkeypatch.setattr(grcnn,'GrCNNPipeline',load)
    monkeypatch.setattr(visualization,'GraspVisualizer',lambda:NS(publish_scene=lambda *args:None))
    backend=GraspBackend(dict(controller_dir='/tmp',robot_name='test',grasp_backend='grcnn'))
    snapshot=dict(image_path=str(image),depth_path=str(depth),intrinsics=[100,0,1,0,100,1,0,0,1],
                  extrinsics=np.eye(4).tolist(),base_frame='base')
    for _ in range(2):
        result=backend('infer',dict(snapshot=snapshot,items=[dict(label='pepper',mask_path=str(mask))]))
        assert result.ok and result.data['_diagnostics']['pepper']['pipeline']=='grcnn'
    assert len(loaded)==1 and len(prepared)==2 and backend.model is None
import sys
from contextlib import nullcontext
from types import SimpleNamespace as NS
