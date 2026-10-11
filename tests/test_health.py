from types import SimpleNamespace as NS
import threading
import time
import sys

import pytest

from core.health import ARM_JOINTS, camera_readiness, robot_readiness
from core.results import ToolResult
from services import arm, health
from services.lifecycle import arm_diagnostic, camera_diagnostic
from workers.camera_worker import CameraBackend
from workers.health_worker import HealthBackend


def robot_evidence():
    return dict(sample=(10, NS(name=ARM_JOINTS, position=[0.] * 6)), now=10.1,
                driver_joints=ARM_JOINTS,
                controllers={"arm_controller": "active", "gripper_controller": "active"},
                actions={"arm_controller": True, "gripper_controller": True},
                planning={"ik": True, "cartesian": True, "joint_path": True})


def test_planning_services_alone_do_not_mean_connected():
    result = robot_readiness(None, 10, [], {}, {}, {"ik": True, "cartesian": True, "joint_path": True})
    assert not result.ok and not result.data["ready"]
    assert result.data["checks"]["planning_services"]
    assert not result.data["checks"]["joint_feedback"]


@pytest.mark.parametrize("change", [
    {"sample": None}, {"now": 11.1},
    {"sample": (10, NS(name=ARM_JOINTS, position=[0.] * 5))},
    {"sample": (10, NS(name=ARM_JOINTS, position=[0.] * 5 + [float("nan")]))},
    {"driver_joints": []}, {"controllers": {"arm_controller": "inactive", "gripper_controller": "active"}},
    {"actions": {"arm_controller": False, "gripper_controller": True}},
    {"planning": {"ik": False, "cartesian": True}},
    {"planning": {"ik": True, "cartesian": True}},
])
def test_robot_requires_live_complete_feedback_and_control(change):
    assert not robot_readiness(**{**robot_evidence(), **change}).ok


def test_robot_is_ready_when_all_evidence_is_present():
    assert robot_readiness(**robot_evidence()).ok


def camera_evidence():
    rgb = NS(width=2, height=2, encoding="rgb8", step=6, data=bytes(12))
    depth = NS(width=2, height=2, encoding="16UC1", step=4, data=bytes(8))
    info = NS(width=2, height=2, k=[100,0,1, 0,100,1, 0,0,1])
    return (10, rgb, depth), (10, info)


def test_camera_without_frames_is_not_ready():
    assert not camera_readiness(None, None, 10).ok


def test_camera_requires_recent_matching_frames_and_valid_intrinsics():
    pair, info = camera_evidence()
    assert camera_readiness(pair, info, 10.1).ok
    assert not camera_readiness(pair, info, 11.1).ok
    assert not camera_readiness(pair, (8, info[1]), 10.1).ok
    info[1].k[0] = 0
    assert not camera_readiness(pair, info, 10.1).ok
    info[1].k[0] = 100
    pair[2].width = 3
    assert not camera_readiness(pair, info, 10.1).ok


def test_camera_empty_buffer_is_not_ready():
    pair, info = camera_evidence()
    pair[1].data = b""
    assert not camera_readiness(pair, info, 10.1).ok


def test_camera_status_reuses_subscription_without_writing_artifacts(tmp_path):
    backend = CameraBackend.__new__(CameraBackend)
    backend._condition = threading.Condition()
    pair, info = camera_evidence()
    backend._pair, backend._info = (time.monotonic(), *pair[1:]), (time.monotonic(), info[1])
    assert backend("status", {}).ok
    assert not list(tmp_path.iterdir())


def test_unready_robot_never_initializes_or_dispatches_motion(monkeypatch):
    failure = ToolResult.failure("SERIAL_MISSING", "port missing")
    monkeypatch.setattr(health, "robot_status", lambda: failure)
    monkeypatch.setattr(arm.client, "request", lambda *a: pytest.fail("must not dispatch"))
    assert arm.call("go_home") is failure


def test_fatal_driver_diagnostics_are_explicit():
    assert arm_diagnostic("[xs_sdk-2] Failed to open port at '/dev/ttyDXL'").error_code == "SERIAL_OPEN_FAILED"
    assert arm_diagnostic("[ERROR] [xs_sdk-2]: process has died [pid 5, exit code 1]").error_code == "ARM_DRIVER_FAILED"
    assert camera_diagnostic("xioctl(UVCIOC_CTRL_QUERY) failed Last Error: Connection timed out").error_code == "CAMERA_USB_TIMEOUT"
    assert camera_diagnostic("device not found, waiting") is None


def test_health_worker_queries_driver_and_controller_without_motion(monkeypatch):
    subscriptions, calls, clients = [], [], {}

    class RequestType:
        Request = NS

    class Client:
        def __init__(self, name):
            self.name = name

        def service_is_ready(self):
            return True

        def call_async(self, request):
            calls.append((self.name, request))
            if self.name.endswith('/get_robot_info'):
                response = NS(joint_names=ARM_JOINTS)
            elif self.name.endswith('/controller_manager/list_controllers'):
                response = NS(controller=[NS(name=n, state='active') for n in ('arm_controller', 'gripper_controller')])
            else:
                pytest.fail('readiness must not request motion or planning')
            return NS(done=lambda: True, result=lambda: response)

    class Node:
        def __init__(self, name):
            pass

        def create_subscription(self, msg, topic, callback, qos):
            subscriptions.append((topic, callback))

        def create_client(self, msg, name):
            clients[name] = Client(name)
            return clients[name]

    def spin(node, timeout_sec):
        for topic, callback in subscriptions:
            callback(NS(name=ARM_JOINTS, position=[.1] * 6))

    class Action:
        def __init__(self, node, message, name):
            pass

        def server_is_ready(self):
            return True

    modules = {
        'rclpy': NS(init=lambda: None, ok=lambda: True, spin_once=spin),
        'rclpy.node': NS(Node=Node), 'rclpy.qos': NS(qos_profile_sensor_data=object()),
        'rclpy.action': NS(ActionClient=Action), 'sensor_msgs.msg': NS(JointState=object()),
        'controller_manager_msgs.srv': NS(ListControllers=RequestType),
        'interbotix_xs_msgs.srv': NS(RobotInfo=RequestType),
        'moveit_msgs.srv': NS(GetPositionIK=RequestType, GetCartesianPath=RequestType, GetMotionPlan=RequestType),
        'control_msgs.action': NS(FollowJointTrajectory=object()),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    backend = HealthBackend({'robot_name': 'wx250s'})
    assert backend.robot(timeout=.2).ok
    assert {name for name, request in calls} == {
        '/wx250s/get_robot_info', '/wx250s/controller_manager/list_controllers'}
    assert calls[0][1].cmd_type == 'group' and calls[0][1].name == 'arm'
    assert subscriptions[0][0] == '/wx250s/joint_states'
