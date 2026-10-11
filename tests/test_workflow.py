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
        objects[label]['grasps']=[dict(score=.2,position=xyz,rotation=r.tolist(),rpy=[0,90,0],depth=.01,width=.04,
            ee_position=ee_position(xyz,r,.01,.015575).tolist(),execution_ready=True,
            frame=snapshot['base_frame'],calibration_id='test',snapshot_timestamp=12.)]

    value = ScenePlan('scene', snapshot, objects)
    store.plan = value
    return value


@pytest.fixture
def recorded(monkeypatch):
    calls = []
    monkeypatch.setattr(arm, 'call', lambda name, **args: calls.append((name, args)) or ToolResult.success('done'))
    yield calls
    store.clear()


def test_multi_object_release_lift_precedes_next_approach(recorded):
    value = plan()
    result = execute([{'object': 'apple', 'target': 'bowl'}, {'object': 'banana', 'target': 'plate'}], 'scene')
    assert result.ok
    assert len(recorded) == 19
    assert recorded[7] == ('gripper_set', {'position': 0.037})
    assert recorded[8][0] == 'move_arm' and recorded[8][1]['z'] == pytest.approx(0.26)
    assert recorded[10][1]['pitch'] == pytest.approx(90)
    approaches = [args for name,args in recorded if name=='move_arm' and not args.get('cartesian')]
    assert len(approaches) == 2
    assert all(args['pitch']==pytest.approx(90) for args in approaches)
    assert all(not {'roll','pitch','yaw'}.intersection(args) for name,args in recorded
               if name=='move_arm' and args.get('cartesian'))
    assert recorded[2][1]['z']==pytest.approx(.065575)
    assert recorded[2][1]['cartesian'] is True
    assert recorded[0] == ('gripper_set', {'position':.037})
    assert not any(name=='inspect_grasps' for name,args in recorded)
    assert all(v['mode']=='graspnet' for v in result.data['strategies'])
    assert recorded[-1][0] == 'go_sleep'
    assert sum(name == 'go_sleep' for name, _ in recorded) == 1
    assert value.moved == {'apple', 'banana'}


def test_failed_descent_does_not_close_gripper(monkeypatch):
    value = plan()
    calls = []
    def call(name, **args):
        calls.append((name,args))
        return ToolResult.failure('COLLISION','blocked') if len(calls) == 3 else ToolResult.success('done',{'selected_index':0})
    monkeypatch.setattr(arm, 'call', call)
    result = execute([{'object':'apple','target':'bowl'}], 'scene')
    assert not result.ok and result.error_code == 'COLLISION'
    assert len(calls) == 3 and value.invalid
    assert not any(args.get('position') == 0.024 for _, args in calls)


def test_controller_failure_details_are_preserved_and_stop_followup_actions(monkeypatch):
    value = plan(); calls = []
    details = {'action_status':6, 'controller_error_code':-4, 'controller_error':'trajectory aborted'}
    def call(name, **args):
        calls.append(name)
        return (ToolResult.failure('MOTION_EXECUTION_FAILED', 'controller failure', details) if name=='move_arm'
                else ToolResult.success('done', {'selected_index':0}))
    monkeypatch.setattr(arm, 'call', call)
    result = execute([{'object':'apple','target':'bowl'}], 'scene')
    assert result.error_code == 'MOTION_EXECUTION_FAILED'
    assert result.data['failure'] == details
    assert calls == ['gripper_set','move_arm']
    assert value.invalid


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
    result=execute([{'object':'apple'}],'scene',allow_fallback=False)
    assert result.error_code=='NO_GRASP' and not recorded


def test_no_model_pose_uses_original_fallback_with_fixed_orientation(recorded):
    value=plan();value.objects['apple']['grasps']=[]
    result=execute([{'object':'apple'}],'scene',return_to_sleep=False)
    assert result.ok and result.data['strategies'][0]['mode']=='fallback'
    assert len(recorded)==5
    assert recorded[1][1]['z']==pytest.approx(.16)
    assert recorded[1][1]['grasp_debug']['mode']=='fallback'
    assert recorded[2][1]['z']==pytest.approx(.03)
    assert recorded[4][1]['z']==pytest.approx(.21)
    approach = recorded[1][1]
    assert [approach['roll'],approach['pitch'],approach['yaw']]==[0,50,0]
    assert all(not {'roll','pitch','yaw'}.intersection(args) for name,args in recorded
               if name=='move_arm' and args.get('cartesian'))
    assert recorded[2][1]['cartesian'] and recorded[4][1]['cartesian']



@pytest.mark.parametrize('change',[
    {'execution_ready':False}, {'snapshot_timestamp':11.}, {'calibration_id':'other'},
    {'frame':'other/base_link'}, {'ee_position':[float('nan'),.05,.06]},
    {'rotation':[[1,0,0],[0,1,0],[0,0,-1]]},
])
def test_invalid_or_inspection_only_pose_cannot_move_arm(recorded,change):
    value=plan();value.objects['apple']['grasps'][0].update(change)
    result=execute([{'object':'apple'}],'scene')
    assert not result.ok and not recorded


def test_ik_failure_with_a_model_pose_does_not_start_fallback(monkeypatch):
    value=plan();calls=[]
    def call(name,**args):
        calls.append((name,args))
        return ToolResult.failure('IK_NO_SOLUTION','no pose solution') if name=='move_arm' else ToolResult.success('ok')
    monkeypatch.setattr(arm,'call',call)
    result=execute([{'object':'apple'}],'scene')
    assert result.error_code=='IK_NO_SOLUTION' and value.invalid
    assert [n for n,a in calls]==['gripper_set','move_arm']
    assert calls[-1][1]['pitch']==pytest.approx(90)


def test_workflow_selects_highest_score_once_without_preflight(recorded):
    import copy
    from core.grasp_geometry import ee_position
    value=plan();g=copy.deepcopy(value.objects['apple']['grasps'][0]);g['score']=.8
    g['position'][0]+=.02;g['ee_position']=ee_position(g['position'],g['rotation'],g['depth'],.015575).tolist()
    value.objects['apple']['grasps'].append(g)
    result=execute([{'object':'apple'}],'scene',return_to_sleep=False)
    assert result.ok and recorded[1][1]['x']==pytest.approx(g['ee_position'][0])
    assert len(recorded)==5
    assert recorded[1][1]['grasp_debug']['score']==.8


def test_roll_is_normalized_before_the_pose_is_sent_to_moveit(recorded):
    from scipy.spatial.transform import Rotation
    value=plan();g=value.objects['apple']['grasps'][0]
    g['rotation']=Rotation.from_euler('xyz',[140,65,20],degrees=True).as_matrix().tolist()
    result=execute([{'object':'apple'}],'scene',return_to_sleep=False)
    assert result.ok
    assert recorded[1][1]['roll']==pytest.approx(-40)
    assert recorded[1][1]['pitch']==pytest.approx(65)
    assert recorded[1][1]['yaw']==pytest.approx(20)


def test_expiry_before_first_move_prevents_any_command(recorded,monkeypatch):
    value=plan();original=store.get;count=[]
    def get(scene_id):
        if count: value.snapshot['created']-=1000
        count.append(1)
        return original(scene_id)
    monkeypatch.setattr(store,'get',get)
    result=execute([{'object':'apple'}],'scene')
    assert result.error_code=='SCENE_EXPIRED' and not recorded
