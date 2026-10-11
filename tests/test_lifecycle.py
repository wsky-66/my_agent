import subprocess
import sys
import threading
import time

import pytest

from core.processes import ManagedProcess
from core.results import ToolResult
from core.execution import ExecutionContext, current


def test_external_services_are_reused_and_not_killed(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: (_ for _ in ()).throw(AssertionError('must not spawn')))
    process = ManagedProcess('external', [], tmp_path/'launch.log', lambda: True)
    assert process.start().ok
    assert process.status().data == {'ready': True, 'owned': False}
    assert process.stop().ok and process.proc is None


def test_stop_terminates_only_the_owned_process_group(tmp_path):
    process = ManagedProcess('owned', [sys.executable, '-c', 'import time; time.sleep(60)'],
                             tmp_path/'launch.log', lambda: process.running())
    # First probe sees no process; subsequent probe sees the launched process.
    assert process.start().ok
    proc = process.proc
    assert proc.poll() is None and process.status().data['owned']
    assert process.stop().ok and proc.poll() is not None


def test_unready_status_is_failure():
    process = ManagedProcess('device', [], '/tmp/unused-health.log', lambda: False)
    assert not process.status().ok and not process.status().data['ready']


def test_missing_serial_preflight_does_not_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('must not spawn'))
    process = ManagedProcess('arm', [], tmp_path/'launch.log', lambda: False,
                             preflight=lambda: ToolResult.failure('SERIAL_MISSING', 'no port'))
    assert process.start().error_code == 'SERIAL_MISSING' and process.proc is None


def test_start_timeout_terminates_owned_launch(tmp_path):
    launched = []
    process = ManagedProcess('device', [sys.executable, '-c', 'import time; time.sleep(60)'],
                             tmp_path/'launch.log', lambda: launched.append(process.proc) or False)
    assert process.start(timeout=.15).error_code == 'STARTUP_TIMEOUT'
    assert process.proc is None
    assert launched[-1].poll() is not None


def test_cancelled_start_cleans_owned_launch(tmp_path):
    process = ManagedProcess('device', [sys.executable, '-c', 'import time; time.sleep(60)'],
                             tmp_path/'launch.log', lambda: False)
    context = ExecutionContext()
    token = current.set(context)
    timer = threading.Timer(.15, context.cancelled.set)
    try:
        timer.start()
        assert process.start(timeout=10).error_code == 'TASK_CANCELLED'
        assert process.proc is None
    finally:
        timer.join()
        process.stop()
        current.reset(token)


def test_fatal_child_error_stops_launch_before_timeout(tmp_path):
    from services.lifecycle import arm_diagnostic
    script = "import time; print('Failed to open port',flush=True); time.sleep(60)"
    process = ManagedProcess('arm', [sys.executable, '-u', '-c', script], tmp_path/'launch.log',
                             lambda: False, diagnose=arm_diagnostic)
    started = time.monotonic()
    assert process.start(timeout=10).error_code == 'SERIAL_OPEN_FAILED'
    assert time.monotonic()-started < 3 and process.proc is None


def test_historical_fatal_log_does_not_break_new_launch(tmp_path):
    from services.lifecycle import arm_diagnostic
    log = tmp_path/'launch.log'
    log.write_text('Failed to open port\n')
    process = ManagedProcess('arm', [sys.executable, '-c', 'import time; time.sleep(60)'], log,
                             lambda: process.running(), diagnose=arm_diagnostic)
    try:
        assert process.start().ok
    finally:
        process.stop()


def test_keyboard_interrupt_during_start_cleans_owned_process(tmp_path):
    def probe():
        if process.running():
            raise KeyboardInterrupt
        return False
    process = ManagedProcess('arm', [sys.executable, '-c', 'import time; time.sleep(60)'],
                             tmp_path/'launch.log', probe)
    with pytest.raises(KeyboardInterrupt):
        process.start()
    assert process.proc is None
