import os
import shlex
import sys
import time
import pytest

from config import config
from core.execution import ExecutionContext, current
from tools import run_shell
from services import segmentation
from core.results import ToolResult


def test_shell_exit_status_and_output_are_preserved():
    command = shlex.join([sys.executable, '-c', 'import sys; print("output"); print("error",file=sys.stderr); sys.exit(7)'])
    result = run_shell(command)
    assert result.error_code == 'SHELL_FAILED'
    assert result.data == {'stdout': 'output\n', 'stderr': 'error\n', 'returncode': 7}


def test_shell_timeout_stops_child_process(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'SHELL_TIMEOUT', .3)
    pid_file = tmp_path/'child.pid'
    script = f'from pathlib import Path; import os,time; Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(30)'
    result = run_shell(shlex.join([sys.executable, '-c', script]))
    assert result.error_code == 'SHELL_TIMEOUT'
    pid = int(pid_file.read_text())
    # A killed orphan may briefly remain a zombie before init reaps it.
    state = None
    stat = __import__('pathlib').Path(f'/proc/{pid}/stat')
    for _ in range(20):
        if not stat.exists():
            break
        state = stat.read_text().split()[2]
        if state == 'Z':
            break
        time.sleep(.01)
    assert not stat.exists() or state == 'Z'


def test_cancelled_shell_does_not_launch_command(tmp_path):
    context = ExecutionContext()
    context.cancelled.set()
    token = current.set(context)
    output = tmp_path/'output'
    try:
        result = run_shell(shlex.join([sys.executable, '-c', f'open({str(output)!r},"w").close()']))
        assert result.error_code == 'TASK_CANCELLED' and not output.exists()
    finally:
        current.reset(token)


def test_segmentation_honors_dilation_parameter(monkeypatch):
    calls = []
    monkeypatch.setattr(segmentation, 'batch', lambda image, items: calls.append(items) or
                        ToolResult.success('done', {'object': {'mask_path': 'mask.png'}}))
    assert segmentation.segment(box=[0,0,10,10], image_path='image.png', dilate_cm=2.5).ok
    assert calls[0][0]['dilate_cm'] == 2.5
    assert not segmentation.segment(image_path='image.png', dilate_cm=-1).ok
    assert len(calls) == 1
