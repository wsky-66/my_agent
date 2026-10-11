import time
import pytest
from core.execution import ExecutionContext, current
from core.results import ToolResult
from services.scene import ScenePlan, store
from workflows.manipulation import execute
from services import arm


def plan():
    objects = {}
    for i, label in enumerate(('apple', 'banana', 'bowl', 'plate')):
        objects[label] = {'ok': True, 'base_xyz': [0.2+i*0.02, 0.05, 0.06], 'grasps': []}
    snapshot = {'created':time.time(),'base_frame':'wx250s/base_link','calibration_id':'test','timestamp':12.}
    from scipy.spatial.transform import Rotation
    from core.grasp_geometry import ee_position
    r = Rotation.from_euler('y',90,degrees=True).as_matrix()
    for label in ('apple','banana'):
        xyz=objects[label]['base_xyz']
        objects[label]['grasps']=[dict(score=.2,position=xyz,rotation=r.tolist(),depth=.01,width=.04,
            ee_position=ee_position(xyz,r,.01,.015575).tolist(),tip_from_ee=.015575,gripper_gap_offset=.009,
            gripper_open=.029,gripper_close=.025166666666666667,approach_distance=.10,table_clearance=.004,
            pipeline_version=3,execution_ready=True,collision_checked=True,
            frame=snapshot['base_frame'],calibration_id='test',snapshot_timestamp=12.,ee_frame='wx250s/ee_gripper_link')]
    value = ScenePlan('scene', snapshot, objects)
    store.plan = value
    return value


@pytest.fixture
def recorded(monkeypatch):
    calls = []
    monkeypatch.setattr(arm, 'call', lambda name, **args: calls.append((name, args)) or
                        ToolResult.success('done', {'selected_index':0} if name=='inspect_grasps' else {}))
    yield calls
    store.clear()


def test_multi_object_release_lift_precedes_next_approach(recorded):
    value = plan()
    result = execute([{'object': 'apple', 'target': 'bowl'}, {'object': 'banana', 'target': 'plate'}], 'scene')
    assert result.ok
    assert len(recorded) == 21
    assert recorded[8] == ('gripper_set', {'position': 0.037})
    assert recorded[9][0] == 'move_arm' and recorded[9][1]['z'] == pytest.approx(0.26)
    assert recorded[12][1]['pitch'] == pytest.approx(90)
    assert all(args['pitch']==pytest.approx(90) for name,args in recorded if name=='move_arm')
    assert recorded[3][1]['z']==pytest.approx(.065575)
    assert recorded[3][1]['cartesian'] is True
    assert recorded[0][0]=='inspect_grasps'
    assert recorded[1][1]['position']==.029
    assert recorded[-1][0] == 'go_sleep'
    assert sum(name == 'go_sleep' for name, _ in recorded) == 1
    assert value.moved == {'apple', 'banana'}


def test_failed_descent_does_not_close_gripper(monkeypatch):
    value = plan()
    calls = []
    def call(name, **args):
        calls.append((name,args))
        return ToolResult.failure('COLLISION','blocked') if len(calls) == 4 else ToolResult.success('done',{'selected_index':0})
    monkeypatch.setattr(arm, 'call', call)
    result = execute([{'object':'apple','target':'bowl'}], 'scene')
    assert not result.ok and result.error_code == 'COLLISION'
    assert len(calls) == 4 and value.invalid
    assert not any(args.get('position') == 0.024 for _, args in calls)


@pytest.mark.parametrize('actions', [
    [{'object':'apple','target_position':[float('nan'),0,0]}],
    [{'object':'apple','target':'bowl'}, {'object':'apple','target':'plate'}],
    [{'object':'apple','target':'bowl'}, {'object':'banana','target':'apple'}],
    [{'object':'apple'}, {'object':'banana','target':'bowl'}],
])
def test_bad_plan_rejected_before_motion(recorded, actions):
    plan()
    with pytest.raises(ValueError):
        execute(actions, 'scene')
    assert not recorded


def test_cancellation_prevents_motion(recorded):
    value = plan()
    context = ExecutionContext()
    context.cancelled.set()
    token = current.set(context)
    try:
        result = execute([{'object':'apple','target':'bowl'}], 'scene')
        assert result.error_code == 'TASK_CANCELLED'
        assert not recorded and value.invalid
    finally:
        current.reset(token)


def test_missing_target_prevents_any_motion(recorded):
    value=plan()
    value.objects['bowl']['ok']=False
    result=execute([{'object':'apple','target':'bowl'}], 'scene')
    assert not result.ok and not recorded


def test_expired_scene_rejected(recorded):
    value=plan()
    value.snapshot['created'] -= 1000
    with pytest.raises(ValueError):
        execute([{'object':'apple','target':'bowl'}], 'scene')
    assert not recorded


def test_missing_grasp_never_uses_centroid(recorded):
    value=plan(); value.objects['apple']['grasps']=[]
    result=execute([{'object':'apple'}],'scene')
    assert result.error_code=='NO_GRASP' and not recorded


def test_explicit_legacy_fallback_is_rejected_before_any_command(recorded):
    plan()
    assert execute([{'object':'apple'}],'scene',allow_fallback=True).error_code=='FALLBACK_DISABLED'
    assert not recorded


@pytest.mark.parametrize('change',[
    {'execution_ready':False}, {'collision_checked':False}, {'pipeline_version':1},
    {'snapshot_timestamp':11.}, {'calibration_id':'other'}, {'ee_frame':'wrong'},
    {'ee_position':[.2,.05,.06]}, {'depth':.08}, {'gripper_open':.04},
    {'rotation':[[1,0,0],[0,1,0],[0,0,-1]]}, {'table_clearance':-.01},
])
def test_invalid_or_inspection_only_candidate_cannot_move_arm(recorded,change):
    value=plan();value.objects['apple']['grasps'][0].update(change)
    result=execute([{'object':'apple'}],'scene')
    assert not result.ok and not recorded


def test_preflight_failure_leaves_gripper_untouched(monkeypatch):
    plan();calls=[]
    def call(name,**args):
        calls.append(name)
        return ToolResult.failure('NO_REACHABLE_GRASP','no path',{'attempts':[{'error_code':'IK_NO_SOLUTION'}]})
    monkeypatch.setattr(arm,'call',call)
    result=execute([{'object':'apple'}],'scene')
    assert result.error_code=='NO_REACHABLE_GRASP'
    assert calls==['inspect_grasps']
    assert result.data['preflight']['attempts'][0]['error_code']=='IK_NO_SOLUTION'


def test_workflow_executes_selected_candidate_instead_of_highest_score(monkeypatch):
    import copy
    from core.grasp_geometry import ee_position
    value=plan();g=copy.deepcopy(value.objects['apple']['grasps'][0]);g['score']=.1
    g['position'][0]+=.02;g['ee_position']=ee_position(g['position'],g['rotation'],g['depth'],g['tip_from_ee']).tolist()
    value.objects['apple']['grasps'].append(g);calls=[]
    def call(name,**args):
        calls.append((name,args));return ToolResult.success('ok',{'selected_index':1} if name=='inspect_grasps' else {})
    monkeypatch.setattr(arm,'call',call)
    assert execute([{'object':'apple'}],'scene',return_to_sleep=False).ok
    assert calls[2][1]['x']==pytest.approx(g['ee_position'][0])
    assert calls[0][1]['candidates'][1]['position']==g['ee_position']


def test_scene_expiry_during_read_only_preflight_prevents_opening(monkeypatch):
    value=plan();calls=[]
    def call(name,**args):
        calls.append(name);value.snapshot['created']-=1000
        return ToolResult.success('ok',{'selected_index':0})
    monkeypatch.setattr(arm,'call',call)
    result=execute([{'object':'apple'}],'scene')
    assert result.error_code=='SCENE_EXPIRED' and calls==['inspect_grasps']
