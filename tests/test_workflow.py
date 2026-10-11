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
    value = ScenePlan('scene', {'created': time.time()}, objects)
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
    assert recorded[10][1]['pitch'] == 50
    assert all('pitch' not in args for i, (name, args) in enumerate(recorded) if name == 'move_arm' and i not in (1,10))
    assert recorded[-1][0] == 'go_sleep'
    assert sum(name == 'go_sleep' for name, _ in recorded) == 1
    assert value.moved == {'apple', 'banana'}


def test_failed_descent_does_not_close_gripper(monkeypatch):
    value = plan()
    calls = []
    def call(name, **args):
        calls.append((name,args))
        return ToolResult.failure('COLLISION','blocked') if len(calls) == 3 else ToolResult.success('done')
    monkeypatch.setattr(arm, 'call', call)
    result = execute([{'object':'apple','target':'bowl'}], 'scene')
    assert not result.ok and result.error_code == 'COLLISION'
    assert len(calls) == 3 and value.invalid
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
