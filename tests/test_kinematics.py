from types import SimpleNamespace as NS
import copy
import numpy as np
import pytest

from core.kinematics import LinkChain, sample_trajectory


MODEL = '''<robot name="test">
<joint name="world" type="fixed"><parent link="world"/><child link="base"/><origin xyz="10 0 0"/></joint>
<joint name="rotation" type="revolute"><parent link="base"/><child link="link"/><axis xyz="0 0 1"/></joint>
<joint name="extension" type="prismatic"><parent link="link"/><child link="slider"/><axis xyz="1 0 0"/></joint>
<joint name="tip_offset" type="fixed"><parent link="slider"/><child link="tip"/><origin xyz="1 0 0"/></joint>
</robot>'''


def path(*positions, velocity=None, acceleration=None):
    return NS(joint_names=['rotation'],points=[NS(positions=[p],velocities=[] if velocity is None else [velocity],
        accelerations=[] if acceleration is None else [acceleration],time_from_start=NS(sec=i,nanosec=0))
        for i,p in enumerate(positions)])


def test_joint_order_base_frame_fixed_tip_and_prismatic_displacement():
    model=LinkChain(MODEL,'base','tip')
    actual=model.positions(['extension','rotation'],np.array([[0,0],[.5,np.pi/2]]))
    assert actual==pytest.approx(np.array([[1,0,0],[0,1.5,0]]),abs=1e-12)


def test_dense_end_effector_points_follow_fk_arc_instead_of_joining_endpoints():
    traj=path(0,np.pi/2);original=copy.deepcopy(traj)
    sampled=sample_trajectory(traj)
    sampled=np.column_stack((sampled[:,0],np.zeros(len(sampled))))
    xyz=LinkChain(MODEL,'base','tip').positions(['rotation','extension'],sampled)
    assert len(xyz)>100 and np.linalg.norm(np.diff(xyz,axis=0),axis=1).max()<.011
    assert np.linalg.norm(xyz[:,:2],axis=1)==pytest.approx(np.ones(len(xyz)))
    assert xyz[len(xyz)//2,0]>.7 and xyz[len(xyz)//2,1]>.7
    assert traj==original


@pytest.mark.parametrize('velocity,acceleration,quarter',[(None,None,.25),(0.,None,.15625),(0.,0.,.103515625)])
def test_controller_interpolation_uses_available_derivatives(velocity,acceleration,quarter):
    actual=sample_trajectory(path(0,1,velocity=velocity,acceleration=acceleration),time_step=.25,angle_step=1.)
    assert actual[:,0]==pytest.approx([0,quarter,.5,1-quarter,1])


def test_sampling_preserves_original_waypoints_and_absolute_angles():
    samples=sample_trajectory(path(3.,-3.,0.))
    assert samples[0,0]==3. and samples[-1,0]==0.
    assert np.min(samples[:,0])==-3. and np.any(np.isclose(samples[1:-1,0],0.))
    assert np.abs(np.diff(samples[:,0])).max()<=.01000001


def test_unknown_tip_does_not_generate_an_unrelated_path():
    with pytest.raises(ValueError):LinkChain(MODEL,'base','missing')
