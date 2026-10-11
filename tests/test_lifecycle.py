import subprocess
import sys

from core.processes import ManagedProcess


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
