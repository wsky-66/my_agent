from types import SimpleNamespace as NS
import threading
import time

import pytest

from core.execution import ExecutionContext, current, begin_task, end_task, cancel_tasks, open_session
from core.health import ARM_JOINTS
from core.parking import park, sleep_error
from core.results import ToolResult
from services import lifecycle, arm, camera, health, grasp, grounding, segmentation, shutdown


TARGET = dict(zip(ARM_JOINTS, [0, -1.8, 1.55, 0, .8, 0]))


class Parking:
    def __init__(self, already=False, failure=None):
        self.events = []
        self.already, self.failure = already, failure

    def step(self, name, data=None):
        self.events.append(name)
        if name == self.failure:
            return ToolResult.failure('PARK_FAILED', name)
        return ToolResult.success(name, data)

    def cancel_motion(self):
        return self.step('cancel')

    def sleep_target(self):
        return self.step('target', TARGET.copy())

    def joints(self):
        return self.step('feedback', TARGET.copy() if self.already else dict.fromkeys(ARM_JOINTS, 0.))

    def move_to_sleep(self, target):
        assert target == TARGET
        return self.step('move')

    def verify_sleep(self, target):
        return self.step('verify', {'max_joint_error': .01})


def test_shutdown_already_at_sleep_does_not_move():
    backend = Parking(already=True)
    result = park(backend, .03)
    assert result.ok and result.data['already_sleep'] and result.data['sleep_verified']
    assert backend.events == ['cancel', 'target', 'feedback', 'verify']


def test_shutdown_moves_once_then_verifies_actual_feedback():
    backend = Parking()
    result = park(backend, .03)
    assert result.ok and not result.data['already_sleep']
    assert backend.events == ['cancel', 'target', 'feedback', 'move', 'verify']


@pytest.mark.parametrize('step', ['cancel', 'target', 'feedback', 'move', 'verify'])
def test_shutdown_failure_never_claims_sleep_or_retries(step):
    backend = Parking(failure=step)
    result = park(backend, .03)
    assert not result.ok and not result.data.get('sleep_verified', False)
    assert backend.events[-1] == step and backend.events.count('move') <= 1


def test_sleep_check_rejects_missing_or_nonfinite_positions():
    with pytest.raises(ValueError):
        sleep_error({'waist': 0}, TARGET)
    with pytest.raises(ValueError):
        sleep_error({**TARGET, 'waist': float('nan')}, TARGET)


@pytest.fixture
def shutdown_session(monkeypatch, tmp_path):
    from config import config
    (tmp_path/'logs').mkdir()
    monkeypatch.setattr(config,'ROOT',tmp_path)
    events = []
    monkeypatch.setattr(lifecycle, '_shutdown_result', None)
    monkeypatch.setattr(lifecycle, '_session', {'lock': NS(close=lambda: events.append('unlock')), 'arm': True, 'camera': True})
    monkeypatch.setattr(lifecycle, '_shutdown_context', None)
    monkeypatch.setattr(lifecycle, '_abort_parking', threading.Event())
    monkeypatch.setattr(lifecycle, 'cancel_tasks', lambda: events.append('cancel_tasks') or True)
    monkeypatch.setattr(health, 'serial_preflight', lambda **kwargs: ToolResult.success('port', {'port': str(tmp_path/'port')}))
    monkeypatch.setattr(shutdown, 'park_robot', lambda: events.append('park') or ToolResult.success('sleep', {'sleep_verified': True}))
    for name, module in [('arm',arm),('health',health),('camera',camera),('grasp',grasp),('segmentation',segmentation),('grounding',grounding)]:
        monkeypatch.setattr(module, 'stop', lambda name=name: events.append('stop_'+name))
    monkeypatch.setattr(lifecycle.robot, 'stop', lambda: events.append('stop_robot'))
    monkeypatch.setattr(lifecycle.camera, 'stop', lambda: events.append('stop_camera_launch'))

    class Sweep:
        def __init__(self, *args, **kwargs):
            pass

        def targets(self, owned):
            events.append('capture')
            return {7: 'identity'}

        def clean(self, captured):
            assert captured == {7: 'identity'}
            events.append('sweep')
            return ToolResult.success('clean', {'remaining_pids': []})

    monkeypatch.setattr(lifecycle, 'HardwareSweep', Sweep)
    yield events
    open_session()


def test_exit_parks_before_closing_controllers_then_checks_residue(shutdown_session):
    result = lifecycle.close_all()
    assert result.ok
    assert shutdown_session.index('park') < shutdown_session.index('stop_arm') < shutdown_session.index('stop_robot') < shutdown_session.index('sweep')
    assert shutdown_session[-1] == 'unlock'
    previous = list(shutdown_session)
    assert lifecycle.close_all() is result and shutdown_session == previous


def test_sleep_failure_still_kills_and_releases_every_resource(shutdown_session, monkeypatch):
    monkeypatch.setattr(shutdown, 'park_robot', lambda: ToolResult.failure('NO_STATE', 'feedback missing'))
    result = lifecycle.close_all()
    assert result.error_code == 'SLEEP_NOT_CONFIRMED'
    assert 'stop_robot' in shutdown_session and 'stop_camera_launch' in shutdown_session and 'sweep' in shutdown_session
    assert shutdown_session[-1] == 'unlock'


def test_shutdown_saves_specific_sleep_failure_without_changing_cleanup(shutdown_session,monkeypatch,tmp_path):
    import json
    from config import config
    (tmp_path/'logs').mkdir(exist_ok=True)
    monkeypatch.setattr(config,'ROOT',tmp_path)
    monkeypatch.setattr(shutdown,'park_robot',lambda:ToolResult.failure('SHUTDOWN_SLEEP_UNCONFIRMED','无法确认 Sleep',
        {'max_joint_error':.041,'sleep_verified':False}))
    result=lifecycle.close_all()
    saved=json.loads((tmp_path/'logs/shutdown_result.json').read_text())
    assert saved['error_code']==result.error_code=='SLEEP_NOT_CONFIRMED'
    assert saved['data']['sleep']['max_joint_error']==.041 and 'sweep' in shutdown_session
    assert saved['data']['sleep']['error_code']=='SHUTDOWN_SLEEP_UNCONFIRMED'


def test_worker_close_failure_does_not_skip_other_cleanup(shutdown_session, monkeypatch):
    monkeypatch.setattr(grasp, 'stop', lambda: (_ for _ in ()).throw(RuntimeError('failed close')))
    result = lifecycle.close_all()
    assert result.error_code == 'SHUTDOWN_INCOMPLETE'
    assert 'stop_grounding' in shutdown_session and 'stop_robot' in shutdown_session and 'sweep' in shutdown_session


def test_cancellation_error_still_releases_and_sweeps_hardware(shutdown_session, monkeypatch):
    monkeypatch.setattr(lifecycle, 'cancel_tasks', lambda: (_ for _ in ()).throw(RuntimeError('cancel error')))
    result = lifecycle.close_all()
    assert result.error_code == 'SHUTDOWN_INCOMPLETE' and 'sweep' in shutdown_session


def test_terminal_output_failure_does_not_skip_hardware_cleanup(shutdown_session):
    result = lifecycle.close_all(notify=lambda *a: (_ for _ in ()).throw(BrokenPipeError('terminal closed')))
    assert result.ok and 'sweep' in shutdown_session


def test_shutdown_uses_fresh_context_even_after_task_cancelled(shutdown_session, monkeypatch):
    context = ExecutionContext()
    context.cancelled.set()
    token = current.set(context)

    def parking():
        assert current.get() is not context and not current.get().cancelled.is_set()
        return ToolResult.success('sleep')

    monkeypatch.setattr(shutdown, 'park_robot', parking)
    try:
        assert lifecycle.close_all().ok
        assert current.get() is context
    finally:
        current.reset(token)


def test_second_interrupt_aborts_parking_but_not_cleanup(shutdown_session, monkeypatch):
    def parking():
        lifecycle.abort_parking()
        assert current.get().cancelled.is_set()
        return ToolResult.failure('TASK_CANCELLED', 'parking aborted')

    monkeypatch.setattr(shutdown, 'park_robot', parking)
    assert lifecycle.close_all().error_code == 'SLEEP_NOT_CONFIRMED'
    assert 'sweep' in shutdown_session and shutdown_session[-1] == 'unlock'


def test_cancel_pending_tasks_before_shutdown_and_reject_new_tasks():
    open_session()
    context = ExecutionContext()
    assert begin_task(context)
    timer = threading.Timer(.1, lambda: end_task(context))
    try:
        timer.start()
        assert cancel_tasks(timeout=1)
        assert context.cancelled.is_set()
        assert not begin_task(ExecutionContext())
    finally:
        timer.join()
        end_task(context)
        open_session()


def test_shutdown_with_missing_feedback_never_creates_motion_worker(monkeypatch):
    monkeypatch.setattr(health, 'robot_status', lambda **kwargs: ToolResult.failure('NO_STATE', 'missing', {'checks': {'joint_feedback':False,'driver_response':True}}))
    monkeypatch.setattr(shutdown, 'worker', lambda *a, **k: pytest.fail('must not initialize motion'))
    assert shutdown.park_robot().error_code == 'SHUTDOWN_NO_FEEDBACK'


def test_real_session_lock_is_released_after_exit(shutdown_session, monkeypatch, tmp_path):
    from config import config
    from core.hardware_processes import SessionLock
    monkeypatch.setattr(config, 'ROOT', tmp_path)
    monkeypatch.setattr(config, 'ENABLE_ARM', True)
    monkeypatch.setattr(config, 'ENABLE_CAMERA', False)
    monkeypatch.setattr(lifecycle, '_session', None)
    lifecycle.begin_session()
    with pytest.raises(RuntimeError):
        SessionLock(tmp_path/'logs/session.lock')
    assert lifecycle.close_all().ok
    lock = SessionLock(tmp_path/'logs/session.lock')
    lock.close()
