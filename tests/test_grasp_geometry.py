import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from core.grasp_geometry import ee_position,rigid_transform,visual_vertices,rotation_rpy


def grasp(rotation=None):
    a=np.zeros(17);a[0]=.3;a[1]=.04;a[2]=.02;a[3]=.02
    a[4:13]=(np.eye(3) if rotation is None else rotation).ravel()
    a[13:16]=[.1,.2,.03];a[16]=-1
    return a


def test_lower_corner_gripper_matches_unrotated_and_pitched_minimum():
    a=grasp()
    assert visual_vertices(a[None])[...,2].min()==pytest.approx(.028)
    a=grasp(Rotation.from_euler('y',90,degrees=True).as_matrix())
    assert visual_vertices(a[None])[...,2].min()==pytest.approx(.01)


def test_depth_changes_ee_along_approach_without_changing_grasp_center():
    r=Rotation.from_euler('xyz',[23,67,-31],degrees=True).as_matrix();p=np.array([.3,.1,.05])
    a=ee_position(p,r,.01,.015575);b=ee_position(p,r,.02,.015575)
    assert b-a==pytest.approx(.01*r[:,0])
    assert a+r[:,0]*.015575==pytest.approx(p+r[:,0]*.01)


@pytest.mark.parametrize('rotation',[np.diag([1,1,-1]),np.diag([2,1,1]),np.full((3,3),np.nan)])
def test_bad_calibration_is_not_used(rotation):
    t=np.eye(4);t[:3,:3]=rotation
    with pytest.raises(ValueError):rigid_transform(t)


def test_vertical_euler_conversion_preserves_the_checked_matrix():
    r=Rotation.from_euler('xyz',[120,90,-65],degrees=True).as_matrix()
    assert Rotation.from_euler('xyz',rotation_rpy(r),degrees=True).as_matrix()==pytest.approx(r,abs=1e-7)


@pytest.mark.parametrize('roll',[137.,162.,-170.,-101.,45.])
def test_grasp_symmetry_preserves_approach_and_swaps_jaws_before_checks(roll):
    from core.grasp_geometry import preferred_grasp_rotation
    r=Rotation.from_euler('xyz',[roll,69,-21],degrees=True).as_matrix()
    chosen=preferred_grasp_rotation(r)
    assert abs(rotation_rpy(chosen)[0])<=90+1e-6
    assert chosen[:,0]==pytest.approx(r[:,0])
    assert np.linalg.det(chosen)==pytest.approx(1)
    assert chosen[:,1]==pytest.approx(-r[:,1] if abs(roll)>90 else r[:,1])
