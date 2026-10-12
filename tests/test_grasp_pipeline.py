import sys
from types import SimpleNamespace as NS
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from workers.grasp_pipeline import GraspPipeline


def test_original_pipeline_runs_once_and_normalizes_the_returned_pose(monkeypatch):
    calls=[];seeds=[]
    monkeypatch.setitem(sys.modules,'torch',NS(manual_seed=seeds.append))
    rotation=Rotation.from_euler('xyz',[140,65,20],degrees=True).as_matrix()
    group=NS(rotation_matrices=np.array([rotation]),translations=np.array([[.3,.1,.05]]),
             depths=np.array([.02]),widths=np.array([.04]),scores=np.array([.7]))
    class Group:
        def __len__(self):return 1
        def __getattr__(self,key):return getattr(group,key)
    def run(*args):
        calls.append(args)
        return Group(),None
    backend=NS(DEPTH_MIN=.2,DEPTH_MAX=1.,GRIP_Z_MIN=.03,run_pipeline=run,
               CamInfo=lambda *args,**kwargs:None,
               create_point_cloud_from_depth_image=lambda *args,**kwargs:np.zeros((2,2,3)))
    pipeline=GraspPipeline(backend,('net','device'))
    pipeline.prepare(np.full((2,2,3),255,np.uint8),np.full((2,2),.5),[100,0,1,0,100,1,0,0,1],np.eye(4))
    mask=np.ones((2,2),bool)
    poses,diagnostics=pipeline.infer(mask,top=3,score_min=0)
    assert len(calls)==1 and calls[0][-2] is mask and len(poses)==1
    assert calls[0][-1].no_collision is False
    assert poses[0]['rpy']==pytest.approx([-40,65,20])
    assert poses[0]['rotation'][0][0]==pytest.approx(rotation[0,0])
    assert poses[0]['execution_ready'] is True
    assert poses[0]['ee_position']==pytest.approx(group.translations[0]+rotation[:,0]*(.02-.015575))
    assert diagnostics['pipeline']=='original' and seeds==[0]


def test_missing_depth_returns_no_pose_before_loading_more_inference(monkeypatch):
    monkeypatch.setitem(sys.modules,'torch',NS())
    pipeline=GraspPipeline(NS(run_pipeline=lambda *args:pytest.fail('no valid input')),('net','device'))
    pipeline.valid=np.zeros((2,2),bool)
    poses,diagnostics=pipeline.infer(np.ones((2,2),bool),top=3,score_min=0)
    assert poses==[] and diagnostics['returned']==0


def test_segmented_cloud_combines_masks_without_background_duplicates_or_bad_depth():
    pipeline=GraspPipeline(None,None)
    pipeline.valid=np.array([[True,False,True],[True,True,True]])
    pipeline.points=np.arange(15).reshape(5,3).astype(float)
    pipeline.colors=np.arange(15).reshape(5,3)/15.
    first=np.array([[True,True,False],[False,True,False]])
    second=np.array([[False,False,True],[False,True,False]])
    original=pipeline.points.copy()
    points,colors=pipeline.segmented_cloud([first,second,None])
    assert np.array_equal(points,pipeline.points[[0,1,3]])
    assert np.array_equal(colors,pipeline.colors[[0,1,3]])
    assert np.array_equal(pipeline.points,original)


@pytest.mark.parametrize('masks',[[],[None],[np.zeros((2,3),bool)]])
def test_without_a_segmentation_cloud_is_empty_instead_of_showing_the_full_scene(masks):
    pipeline=GraspPipeline(None,None)
    pipeline.valid=np.ones((2,3),bool)
    pipeline.points=np.ones((6,3))
    pipeline.colors=np.ones((6,3))
    points,colors=pipeline.segmented_cloud(masks)
    assert points.shape==colors.shape==(0,3)


def test_grasp_worker_publishes_the_segmented_cloud_even_without_grasp_poses(monkeypatch,tmp_path):
    import cv2
    from workers.grasp_worker import GraspBackend
    import workers.grasp_visualization as visualization
    image,depth,mask=tmp_path/'image.png',tmp_path/'depth.npy',tmp_path/'mask.png'
    cv2.imwrite(str(image),np.full((2,2,3),255,np.uint8))
    np.save(depth,np.full((2,2),.5))
    cv2.imwrite(str(mask),np.array([[255,0],[0,0]],np.uint8))
    clouds=[]
    pipeline=GraspPipeline(NS(DEPTH_MIN=.2,DEPTH_MAX=1.,CamInfo=lambda *a,**kw:None,
        create_point_cloud_from_depth_image=lambda *a,**kw:np.arange(12).reshape(2,2,3)),None)
    pipeline.infer=lambda *a,**kw:([],{'returned':0})
    monkeypatch.setitem(sys.modules,'graspnet_client',NS())
    monkeypatch.setattr(visualization,'GraspVisualizer',lambda:NS(publish_scene=lambda *args:clouds.append(args)))
    backend=GraspBackend({'controller_dir':'/tmp','robot_name':'test'})
    backend.model,backend.pipeline=object(),pipeline
    snapshot=dict(image_path=str(image),depth_path=str(depth),intrinsics=[100,0,1,0,100,1,0,0,1],
                  extrinsics=np.eye(4).tolist(),base_frame='base')
    result=backend('infer',{'snapshot':snapshot,'items':[{'label':'pepper','mask_path':str(mask)}]})
    assert result.ok and result.data['pepper']==[] and len(clouds)==1
    assert clouds[0][0].tolist()==[[0.,1.,2.]]
    assert clouds[0][1].tolist()==[[1.,1.,1.]]
    assert clouds[0][2:]==([], 'base')
