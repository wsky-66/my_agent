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
