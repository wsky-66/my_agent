import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from core.hardware_processes import ProcessRecord, SessionLock, HardwareSweep, related, read_process
from core.runtime import terminate


def record(pid, args, ppid=1, started=100, uid=None, domain='0'):
    return ProcessRecord(pid, ppid, os.getuid() if uid is None else uid, started, 'S', tuple(args), '/project', domain)


def match(args, **kwargs):
    return related(record(20, args, **kwargs), 'wx250s', '/camera/camera', '/project')


def test_matches_only_current_hardware_launches_and_nodes():
    assert match(['/usr/bin/python3','/opt/ros/humble/bin/ros2','launch','interbotix_xsarm_moveit_interface','xsarm_moveit_interface.launch.py','robot_model:=wx250s'])
    assert match(['/opt/ros/humble/lib/controller_manager/ros2_control_node','--ros-args','-r','__ns:=/wx250s'])
    assert match(['/opt/ros/humble/lib/moveit_ros_move_group/move_group','-r','/joint_states:=/wx250s/joint_states'])
    assert match(['ros2','launch','realsense2_camera','rs_launch.py'])
    assert match(['/opt/ros/humble/lib/realsense2_camera/realsense2_camera_node','-r','__ns:=/camera','-r','__node:=camera'])
    assert not match(['ros2','launch','interbotix_xsarm_control','xsarm_control.launch.py','robot_model:=vx300s'])
    assert not match(['ros2','launch','realsense2_camera','rs_launch.py','camera_name:=other'])
    assert not match(['/opt/ros/humble/lib/controller_manager/ros2_control_node','-r','__ns:=/other'])
    assert not match(['python3','-c','print("ros2 launch robot_model:=wx250s")'])
    assert not match(['ros2','launch','interbotix_xsarm_control','xsarm_control.launch.py','robot_model:=wx250s'], domain='23')


def test_selects_related_descendants_even_when_they_change_process_groups(monkeypatch, tmp_path):
    root = record(20, ['ros2','launch','realsense2_camera','rs_launch.py'])
    child = record(21, ['driver'], ppid=20)
    unrelated = record(30, ['ros2','launch','realsense2_camera','rs_launch.py','camera_name:=other'])
    sweep = HardwareSweep('wx250s','/camera/camera','/project',proc_root=tmp_path,sys_root=tmp_path)
    monkeypatch.setattr(sweep, 'snapshot', lambda: {20:root,21:child,30:unrelated})
    assert set(sweep.targets()) == {20,21}


def test_pid_reuse_does_not_signal_an_unrelated_process(monkeypatch, tmp_path):
    sweep = HardwareSweep('wx250s','/camera/camera','/project',proc_root=tmp_path,sys_root=tmp_path)
    old = record(20, ['driver'], started=100)
    new = record(20, ['unrelated'], started=200)
    monkeypatch.setattr('core.hardware_processes.read_process', lambda *args: new)
    monkeypatch.setattr(os, 'kill', lambda *args: pytest.fail('must not signal reused PID'))
    monkeypatch.setattr(os, 'pidfd_open', lambda *args: pytest.fail('must not open reused PID'))
    assert not sweep._signal(old, signal.SIGTERM)


def test_session_lock_blocks_a_second_instance_and_releases(tmp_path):
    first = SessionLock(tmp_path/'session.lock')
    try:
        with pytest.raises(RuntimeError, match='另一实例'):
            SessionLock(tmp_path/'session.lock')
    finally:
        first.close()
    second = SessionLock(tmp_path/'session.lock')
    second.close()


def test_cleanup_catches_escaped_child_and_actual_device_holder(tmp_path):
    device = tmp_path/'fake-device'
    device.touch()
    ready, pidfile = tmp_path/'ready', tmp_path/'pid'
    child = f"import time; from pathlib import Path; stream=open({str(device)!r}); Path({str(ready)!r}).touch(); time.sleep(60)"
    script = f"import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c',{child!r}],start_new_session=True); Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(60)"
    leader = subprocess.Popen([sys.executable,'-c',script], start_new_session=True)
    sweep = HardwareSweep('wx250s','/camera/camera',tmp_path,camera=False,serial_port=str(device))
    try:
        until=time.monotonic()+3
        while not ready.exists() and time.monotonic() < until:
            time.sleep(.02)
        assert ready.exists()
        child_pid=int(pidfile.read_text())
        captured=sweep.targets([leader.pid])
        assert leader.pid in captured and child_pid in captured
        terminate(leader)
        assert sweep.alive(captured[child_pid])
        result=sweep.clean(captured,grace=.2)
        assert result.ok and not result.data['remaining_pids']
        assert not sweep.alive(captured[child_pid])
        assert not sweep.targets()
    finally:
        terminate(leader)
        sweep.clean(grace=.1)


def test_parse_real_process_identity_matches_kernel_stat():
    process=read_process(os.getpid())
    assert process.pid == os.getpid() and process.uid == os.getuid() and process.started > 0


def test_protected_device_holder_is_reported_without_being_killed(monkeypatch, tmp_path):
    sweep = HardwareSweep('wx250s','/camera/camera',tmp_path,proc_root=tmp_path,sys_root=tmp_path,protected={20})
    holder = record(20, ['parent-shell'])
    monkeypatch.setattr(sweep, 'targets', lambda *args: {})
    monkeypatch.setattr(sweep, 'device_holders', lambda: {20:holder})
    monkeypatch.setattr(os, 'kill', lambda *args: pytest.fail('must not kill ancestor'))
    result = sweep.clean(grace=0)
    assert not result.ok and result.data['protected_holders'] == [20]
