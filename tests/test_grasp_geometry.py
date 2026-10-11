import itertools
import numpy as np
import pytest
from scipy.spatial import ConvexHull,cKDTree
from scipy.spatial.transform import Rotation

from core.grasp_geometry import StockGripper,ee_position,rigid_transform,visual_vertices,rotation_rpy


def grasp(rotation=None):
    a=np.zeros(17);a[0]=.3;a[1]=.04;a[2]=.02;a[3]=.02
    a[4:13]=(np.eye(3) if rotation is None else rotation).ravel()
    a[13:16]=[.1,.2,.03];a[16]=-1
    return a


def cover(lo,hi):
    return np.asarray(list(itertools.product(*zip(lo,hi))))


def gripper():
    g=StockGripper.__new__(StockGripper)
    g.approach=.1;g.front=.04315;g.tip_from_ee=.015575;g.gap_offset=.009
    g.parts=[]
    g.closing_hulls=[]
    for v,sign in [(cover([-.01,-.014,-.014],[.04315,.002,.014]),1),
                   (cover([-.01,-.002,-.014],[.04315,.014,.014]),-1),
                   (cover([-.06,-.035,-.01],[-.023,.035,.01]),0)]:
        swept=ConvexHull(np.concatenate([v,v+[-g.approach,0,0]]))
        g.parts.append((v,sign,swept.equations))
        g.closing_hulls.append(ConvexHull(v).equations)
    return g


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


def test_real_mesh_clearance_matches_explicit_vertices_on_tilted_table():
    g=gripper();a=grasp(Rotation.from_euler('xyz',[13,54,72],degrees=True).as_matrix())
    normal=np.array([.02,-.03,1.]);normal/=np.linalg.norm(normal);plane=np.r_[normal,-.01]
    r=a[4:13].reshape(3,3);q=g.opening(a[1]);vertices=[]
    for v,sign,_ in g.parts:
        local=v+[a[3]-g.front,sign*(q+.005),0]
        vertices.extend(local@r.T+a[13:16])
    expected=(np.asarray(vertices)@plane[:3]+plane[3]).min()
    assert g.minimum_clearance(a[None],plane)[0]==pytest.approx(expected)


def test_object_between_fingers_is_allowed_but_scene_obstacle_is_not():
    g=gripper();a=grasp();a[3]=.01;a[13:16]=0
    interior=np.array([[.005,0,0]])
    assert not g.collides(a,interior,cKDTree(interior))
    scene=np.r_[interior,[[.005,.025,0]]]
    assert g.collides(a,scene,cKDTree(scene))


def test_approach_sweep_rejects_obstacle_outside_final_gripper():
    g=gripper();a=grasp();a[3]=.01;a[13:16]=0
    obstacle=np.array([[-.08,.025,0]])
    assert g.collides(a,obstacle,cKDTree(obstacle))


def test_gap_maps_to_joint_position_and_never_clips_oversized_opening():
    g=gripper();a=grasp();opening,closing=g.commands(a)
    assert opening==pytest.approx(a[1]/2+g.gap_offset)
    assert .015<=closing<opening<=.037
    a[1]=.10
    with pytest.raises(ValueError):g.commands(a)


def test_other_object_in_closing_sweep_is_rejected():
    g=gripper();a=grasp();a[3]=.01;a[13:16]=0
    obstacle=np.array([[.005,.018,0]])
    assert not g.collides(a,obstacle,cKDTree(obstacle))
    assert g.closing_collides(a,obstacle,cKDTree(obstacle))
    empty=np.empty((0,3))
    assert not g.closing_collides(a,empty,cKDTree(empty))


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


def test_surface_uncertainty_band_allows_shallow_contact_but_blocks_penetration():
    g=gripper();a=grasp();a[3]=.01;a[13:16]=0
    # Inner face is at y=20 mm. A sub-mm depth error is not deep collision.
    shallow=np.array([[.005,.0205,0]])
    deep=np.array([[.005,.024,0]])
    assert not g.collides(a,shallow,cKDTree(shallow))
    details=g.collision_details(a,deep,cKDTree(deep))
    assert details['part']=='left_finger' and details['max_penetration_mm']>3.9
    assert g.closing_collides(a,np.array([[.005,.018,0]]),cKDTree([[.005,.018,0]]))


def test_jaw_symmetry_must_be_checked_again_for_asymmetric_geometry():
    from core.grasp_geometry import preferred_grasp_rotation
    g=gripper();v=cover([-.06,-.03,-.008],[-.02,.03,-.004])
    g.parts=[(v,0,ConvexHull(np.r_[v,v+[-g.approach,0,0]]).equations)]
    a=grasp(Rotation.from_euler('x',180,degrees=True).as_matrix());a[13:16]=[0,0,.003]
    assert g.minimum_clearance(a[None],[0,0,1,0])[0]>.003
    a[4:13]=preferred_grasp_rotation(a[4:13].reshape(3,3)).ravel()
    assert g.minimum_clearance(a[None],[0,0,1,0])[0]<0


def test_batched_symmetry_matches_individual_candidates():
    from core.grasp_geometry import preferred_grasp_rotation
    matrices=Rotation.from_euler('xyz',[[137,69,-21],[-162,60,35],[20,90,10],[90,50,0]],degrees=True).as_matrix()
    expected=np.asarray([preferred_grasp_rotation(r) for r in matrices])
    assert preferred_grasp_rotation(matrices)==pytest.approx(expected)
