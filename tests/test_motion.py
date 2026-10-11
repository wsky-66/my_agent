import math
from types import SimpleNamespace as NS

import pytest

from core.health import ARM_JOINTS
from core.motion import (MotionPolicy, MotionRejected, check_joints, driver_limits,
                         move_pose, move_cartesian, nearest_solution, validate_trajectory,
                         prepare_trajectory)
from core.results import ToolResult
from workers.arm_worker import ArmBackend
from workers.pose_motion import PoseMotion


def joints(value=0.):
    return {n: value for n in ARM_JOINTS}


def limits():
    return {n: (-math.pi+1e-5, math.pi-1e-5, 3.14) for n in ARM_JOINTS}


def trajectory(*positions, times=None):
    times = times or list(range(len(positions)))
    return NS(joint_names=list(ARM_JOINTS), points=[NS(positions=[p[n] for n in ARM_JOINTS],
        velocities=[], accelerations=[], time_from_start=NS(sec=int(t), nanosec=round((t%1)*1e9)))
        for p, t in zip(positions, times)])


class Backend:
    def __init__(self):
        self.start, self.target = joints(), joints(.2)
        self.candidates = [joints(.8), self.target]
        self.events = []
        self.path = None
        self.current = None

    def limits(self):
        return limits()

    def joints(self):
        self.events.append("joints")
        return self.current if self.current is not None and "plan" in self.events else self.start

    def inverse_solutions(self, *args):
        self.events.append("ik")
        return self.candidates

    def plan(self, start, target, *args):
        self.events.append("plan")
        assert target == self.target
        return self.path or trajectory(start, target)

    def execute(self, *args):
        self.events.append("execute")

    def verify(self, *args):
        pytest.fail("controller success must not trigger an arrival check")


def test_pose_is_planned_before_controller_completion_is_reported():
    backend, events = Backend(), []
    result = move_pose(backend, [0.3, 0., .2], [0., 50., 0.], MotionPolicy(), events.append)
    assert result.ok
    assert backend.events == ["joints", "ik", "plan", "joints", "execute"]
    assert result.data["target"] == backend.target
    assert [e["stage"] for e in events] == ["start", "ik_selected", "planned", "completed"]


def test_wrist_solution_is_ranked_by_absolute_distance_without_a_rotation_cutoff():
    start, target = joints(), joints()
    start["wrist_rotate"], target["wrist_rotate"] = map(math.radians, (170, -170))
    near = dict(target, wrist_rotate=math.radians(160))
    assert nearest_solution([target, near], start, limits()) == near
    assert nearest_solution([target], start, limits()) == target


def test_seeded_candidates_choose_nearby_wrist_branch():
    far, near = joints(.1), joints(.2)
    far["wrist_rotate"] = 2.
    assert nearest_solution([far, near], joints(), limits()) == near


@pytest.mark.parametrize("value", [2*math.pi, -2*math.pi, float("nan"), float("inf")])
def test_feedback_outside_model_is_never_normalized_and_sent(value):
    backend = Backend()
    backend.start["forearm_roll"] = value
    result = move_pose(backend, [.3,0,.2], [0,50,0], MotionPolicy())
    assert result.error_code == "MOTION_FEEDBACK_INVALID"
    assert "ik" not in backend.events and "execute" not in backend.events


@pytest.mark.parametrize("kind", ["start", "end", "limit", "duration", "nan", "time", "duplicate", "single"])
def test_bad_planned_path_never_reaches_controller(kind):
    backend = Backend()
    start, target = backend.start, backend.target
    if kind == "start":
        path = trajectory(joints(.1), target)
    elif kind == "end":
        path = trajectory(start, joints(.1))
    elif kind == "limit":
        path = trajectory(start, joints(3.2), target, times=[0,5,10])
    elif kind == "duration":
        path = trajectory(start, target, times=[0,.1])
        path.points[1].velocities = [1000.]*6  # slowing would exceed the execution budget
    elif kind == "nan":
        path = trajectory(start, target)
        path.points[1].velocities = [float("nan")]*6
    elif kind == "time":
        path = trajectory(start, target, times=[0,0])
    elif kind == "duplicate":
        path = trajectory(start, target)
        path.joint_names[-1] = path.joint_names[0]
    else:
        path = trajectory(target)
    backend.path = path
    result = move_pose(backend, [.3,0,.2], [0,50,0], MotionPolicy())
    assert not result.ok and "execute" not in backend.events


def test_speed_check_reports_actual_limit_source_and_required_slowdown():
    path = trajectory(joints(), joints(.2), times=[0,.2])
    with pytest.raises(MotionRejected) as exc:
        validate_trajectory(path, joints(), joints(.2), limits(), MotionPolicy())
    assert exc.value.code == "MOTION_EXCESS_SPEED"
    assert exc.value.data["speed_rad_s"] == 1.
    assert exc.value.data["allowed_rad_s"] == pytest.approx(.83425)
    assert exc.value.data["speed_source"] == "segment"
    assert exc.value.data["point_index"] == 1


def test_overspeed_plan_is_slowed_to_driver_limit_with_geometry_preserved():
    from copy import deepcopy
    start, target = joints(), joints(.4)
    path = trajectory(start, joints(.2), target, times=[0,.2,.4])
    for p in path.points:
        p.velocities = [1.2]*6
        p.accelerations = [2.4]*6
    before = deepcopy(path)
    metrics = prepare_trajectory(path, start, target, limits(), MotionPolicy())
    factor = metrics["time_scale"]
    assert factor == pytest.approx(1.2/(3.14*.25)*(1.+1e-6))
    assert metrics["original_duration"] == .4
    assert metrics["duration"] == pytest.approx(.4*factor, abs=1e-9)
    assert max(metrics["peak_speed_rad_s"].values()) <= 3.14*.25
    for a, b in zip(before.points, path.points):
        assert a.positions == b.positions
        assert b.velocities == pytest.approx([v/factor for v in a.velocities])
        assert b.accelerations == pytest.approx([v/factor**2 for v in a.accelerations])


def test_speed_repair_cannot_hide_driver_bound_or_endpoint_errors():
    for path in (trajectory(joints(), joints(3.2), joints(.2), times=[0,.1,.2]),
                 trajectory(joints(), joints(.1), times=[0,.01])):
        with pytest.raises(MotionRejected) as exc:
            prepare_trajectory(path, joints(), joints(.2), limits(), MotionPolicy())
        assert exc.value.code in {"MOTION_JOINT_INVALID", "MOTION_END_MISMATCH"}


def test_pose_execution_uses_the_slowed_trajectory_and_duration():
    backend = Backend()
    backend.path = trajectory(backend.start, backend.target, times=[0,.1])
    sent = []
    backend.execute = lambda path, duration: sent.append((path, duration))
    result = move_pose(backend, [.3,0,.2], [0,50,0], MotionPolicy())
    assert result.ok and len(sent) == 1
    assert sent[0][1] == result.data["duration"] > .1
    assert validate_trajectory(sent[0][0], backend.start, backend.target, limits(), MotionPolicy())


def test_robot_change_during_planning_aborts_dispatch():
    backend = Backend()
    backend.current = joints(.1)
    result = move_pose(backend, [.3,0,.2], [0,50,0], MotionPolicy())
    assert result.error_code == "MOTION_ROBOT_MOVED" and "execute" not in backend.events


def test_execution_failure_never_retries_or_claims_success():
    backend = Backend()
    def fail(*args):
        backend.events.append("execute")
        raise MotionRejected("MOTION_TIMEOUT", "timeout")
    backend.execute = fail
    result = move_pose(backend, [.3,0,.2], [0,50,0], MotionPolicy())
    assert result.error_code == "MOTION_TIMEOUT"
    assert backend.events.count("execute") == 1 and "verify" not in backend.events


@pytest.mark.parametrize("move", [move_pose, move_cartesian])
def test_controller_success_does_not_wait_for_new_joint_feedback(move):
    backend = Backend()
    backend.cartesian_plan = lambda *args: trajectory(backend.start, backend.target)
    def execute(*args):
        backend.events.append("execute")
        backend.joints = lambda: pytest.fail("no feedback read after controller completion")
    backend.execute = execute
    result = move(backend, [.3,0,.2], [0,50,0], MotionPolicy())
    assert result.ok and result.data["controller_completed"] is True
    assert "actual" not in result.data
    assert backend.events.count("execute") == 1


@pytest.mark.parametrize("change", [dict(mode="ext_position"), dict(joint_upper_limits=[]),
    dict(joint_velocity_limits=[0.]*6), dict(joint_lower_limits=[float("nan")]*6)])
def test_driver_modes_and_limits_checked(change):
    info = NS(mode="position", joint_names=list(ARM_JOINTS), joint_lower_limits=[-3.]*6,
        joint_upper_limits=[3.]*6, joint_velocity_limits=[3.14]*6)
    for key, value in change.items():
        setattr(info, key, value)
    with pytest.raises(MotionRejected):
        driver_limits(info)


def test_valid_driver_limits_are_used_instead_of_hardcoded_angles():
    info = NS(mode="position", joint_names=list(ARM_JOINTS), joint_lower_limits=[-1.]*6,
        joint_upper_limits=[1.]*6, joint_velocity_limits=[.4]*6)
    actual = driver_limits(info)
    with pytest.raises(MotionRejected):
        check_joints(joints(1.1), actual)


@pytest.mark.parametrize("options", [dict(velocity_scaling=0), dict(velocity_scaling=2),
    dict(timeout=float("nan"))])
def test_invalid_motion_policy(options):
    with pytest.raises(ValueError):
        MotionPolicy(**options)


def test_pose_rejection_does_not_fall_back_to_legacy_joint_execution(monkeypatch):
    backend = ArmBackend({"controller_dir": "/tmp", "robot_name": "test"})
    failure = ToolResult.failure("IK_BRANCH_REJECTED", "large rotation")
    monkeypatch.setattr(backend, "ik_move", lambda *args: failure)
    monkeypatch.setattr(backend, "arm", lambda: pytest.fail("no fallback"))
    assert backend("move_arm", dict(x=.3,y=0,z=.2,roll=0,pitch=50,yaw=0)) is failure


@pytest.mark.parametrize("cartesian", [False, True])
def test_arm_does_not_require_extra_pose_feedback_after_controller_success(monkeypatch, cartesian):
    backend = ArmBackend({"controller_dir": "/tmp", "robot_name": "test"})
    details = {"controller_completed": True}
    monkeypatch.setattr(backend, "ik_move", lambda *args, **kwargs: ToolResult.success("controller completed", details))
    monkeypatch.setattr(backend, "pose", lambda: pytest.fail("no second pose verification"))
    result = backend("move_arm", dict(x=.3,y=0,z=.2,roll=0,pitch=50,yaw=0,cartesian=cartesian))
    assert result.ok and result.data == {"motion": details}
    assert result.message == "controller completed"


@pytest.mark.parametrize("ok", [True, False])
def test_position_only_move_uses_controller_result_without_pose_query(monkeypatch, ok):
    backend = ArmBackend({"controller_dir": "/tmp", "robot_name": "test"})
    calls = []
    def move(*xyz):
        calls.append(xyz)
        return ok, "controller error"
    monkeypatch.setattr(backend, "arm", lambda: NS(_move_group_move=move))
    monkeypatch.setattr(backend, "pose", lambda: pytest.fail("no second pose verification"))
    result = backend("move_arm", dict(x=.3,y=0,z=.2))
    assert result.ok == ok and calls == [(.3,0.,.2)]
    if ok:
        assert result.data["controller_completed"] is True
    else:
        assert result.error_code == "MOTION_FAILED"


@pytest.mark.parametrize("ok", [True, False])
def test_position_only_cartesian_move_uses_the_same_planner_without_rpy(monkeypatch, ok):
    backend = ArmBackend({"controller_dir":"/tmp", "robot_name":"test"})
    expected = ToolResult.success("done") if ok else ToolResult.failure("MOTION_PLAN_FAILED","failed")
    calls = []
    def move(xyz,rpy,cartesian=False):
        calls.append((xyz,rpy,cartesian))
        return expected
    monkeypatch.setattr(backend,"ik_move",move)
    monkeypatch.setattr(backend,"arm",lambda:pytest.fail("position-only Cartesian must use PoseMotion"))
    result=backend("move_arm",dict(x=.3,y=0,z=.2,cartesian=True))
    assert result.ok==ok and calls==[([.3,0.,.2],None,True)]
    if not ok:
        assert result is expected


def test_late_accepted_goal_is_cancelled_after_send_timeout():
    calls = []
    PoseMotion._cancel_late(NS(result=lambda: NS(accepted=True, cancel_goal_async=lambda: calls.append("cancel"))))
    assert calls == ["cancel"]


def test_result_timeout_cancels_goal_and_never_resends(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "control_msgs.action", NS(FollowJointTrajectory=NS(Goal=NS)))
    monkeypatch.setitem(sys.modules, "builtin_interfaces.msg", NS(Duration=NS))
    motion, calls = PoseMotion.__new__(PoseMotion), []
    motion.deadline, motion.goal = 1e12, None
    handle = NS(accepted=True, get_result_async=lambda: "result", cancel_goal_async=lambda: calls.append("cancel"))
    motion.action = NS(wait_for_server=lambda **kwargs: True, send_goal_async=lambda goal: "send")
    def wait(future, timeout):
        calls.append(future)
        if future == "send":
            return handle
        raise MotionRejected("MOTION_TIMEOUT", "timeout")
    motion._wait = wait
    with pytest.raises(MotionRejected):
        motion.execute(trajectory(joints(), joints(.1)), 1)
    assert calls == ["send", "result", "cancel"] and motion.goal is None


@pytest.mark.parametrize("status,code", [(6,-5), (5,0), (4,-4)])
def test_controller_failure_is_reported_without_retry(monkeypatch, status, code):
    import sys
    monkeypatch.setitem(sys.modules, "control_msgs.action", NS(FollowJointTrajectory=NS(Goal=NS)))
    monkeypatch.setitem(sys.modules, "builtin_interfaces.msg", NS(Duration=NS))
    motion, sent = PoseMotion.__new__(PoseMotion), []
    motion.deadline, motion.goal = 1e12, None
    handle = NS(accepted=True, get_result_async=lambda: "result",
                cancel_goal_async=lambda: pytest.fail("terminal goal is already finished"))
    def send(goal):
        assert not getattr(goal, "goal_tolerance", [])
        sent.append(goal)
        return "send"
    motion.action = NS(wait_for_server=lambda **kwargs: True, send_goal_async=send)
    response = NS(status=status, result=NS(error_code=code, error_string="controller failure"))
    motion._wait = lambda future, timeout: handle if future == "send" else response
    with pytest.raises(MotionRejected) as exc:
        motion.execute(trajectory(joints(), joints(.1)), 1)
    assert exc.value.code == "MOTION_EXECUTION_FAILED" and len(sent) == 1
    assert exc.value.data == {"action_status":status, "controller_error_code":code,
                              "controller_error":"controller failure"}
    assert motion.goal is None


def test_remaining_execution_budget_is_checked_before_sending(monkeypatch):
    import sys
    import time
    monkeypatch.setitem(sys.modules, "control_msgs.action", NS(FollowJointTrajectory=NS(Goal=NS)))
    monkeypatch.setitem(sys.modules, "builtin_interfaces.msg", NS(Duration=NS))
    motion = PoseMotion.__new__(PoseMotion)
    motion.deadline = time.monotonic()+3
    motion.action = NS(wait_for_server=lambda **kwargs: True,
        send_goal_async=lambda goal: pytest.fail("budget exhausted; must not send"))
    with pytest.raises(MotionRejected) as exc:
        motion.execute(trajectory(joints(), joints(.1)), 1)
    assert exc.value.code == "MOTION_PLAN_DURATION"


def test_cartesian_path_keeps_valid_moveit_timing():
    backend=Backend()
    backend.cartesian_plan=lambda *args: trajectory(backend.start,backend.target,times=[0,1])
    result=move_cartesian(backend,[.3,0,.2],[0,90,0],MotionPolicy())
    assert result.ok and result.data['duration']==result.data['original_duration']==1
    assert result.data['time_scale']==1
    assert backend.events==['joints','joints','execute']


def test_cartesian_overspeed_is_retimed_once_against_the_driver_target():
    backend=Backend()
    path=trajectory(backend.start,joints(.1),backend.target,times=[0,.1,.2])
    path.points[1].velocities=[1.]*6
    path.points[1].accelerations=[2.]*6
    backend.cartesian_plan=lambda *args: path
    sent=[]
    backend.execute=lambda p,duration:sent.append((p,duration))
    result=move_cartesian(backend,[.3,0,.2],None,MotionPolicy())
    factor=1./(3.14*.25)*(1.+1e-6)
    assert result.ok and result.data['original_duration']==.2
    assert result.data['time_scale']==pytest.approx(factor)
    assert result.data['duration']==pytest.approx(.2*factor,abs=1e-9)
    assert len(sent)==1 and sent[0][1]==result.data['duration']
    assert [p.positions for p in path.points]==[[0.]*6,[.1]*6,[.2]*6]
    assert path.points[1].velocities==pytest.approx([1./factor]*6)
    assert path.points[1].accelerations==pytest.approx([2./factor**2]*6)


def test_cartesian_travel_is_reported_without_the_old_net_displacement_cutoff():
    backend=Backend()
    # Match the reported pepper failure: 5.62 rad travel, only .424 rad net,
    # with every position in the driver's range. The old allowed .986 rad.
    start, end, middle = joints(), joints(), joints()
    start['forearm_roll'] = .109
    end['forearm_roll'] = start['forearm_roll'] + (.9860633508750463-.35)/1.5
    middle['forearm_roll'] = start['forearm_roll'] - (5.622836129247646-(end['forearm_roll']-start['forearm_roll']))/2
    backend.start, backend.target = start, end
    backend.cartesian_plan=lambda *args: trajectory(start,middle,end,times=[0,1.5,3])
    result=move_cartesian(backend,[.348668,-.017547,.003533],None,MotionPolicy())
    assert result.ok and backend.events.count('execute')==1
    assert result.data['rpy'] is None
    assert result.data['travel_rad']['forearm_roll']==pytest.approx(5.622836129247646)
    assert result.data['delta_rad']['forearm_roll']==pytest.approx(.4240422339166976)


def test_cartesian_driver_bound_failure_keeps_start_in_diagnostics():
    backend=Backend()
    backend.cartesian_plan=lambda *args: trajectory(backend.start,joints(3.2),backend.target)
    result=move_cartesian(backend,[.3,0,.2],None,MotionPolicy())
    assert result.error_code=='MOTION_JOINT_INVALID' and 'execute' not in backend.events
    assert result.data['start']==backend.start and result.data['target']==backend.target


def test_cartesian_plan_without_time_parameterisation_stops():
    backend=Backend()
    backend.cartesian_plan=lambda *args: trajectory(backend.start,backend.target,times=[0,0])
    result=move_cartesian(backend,[.3,0,.2],[0,90,0],MotionPolicy())
    assert result.error_code=='MOTION_PLAN_INVALID' and 'execute' not in backend.events


def test_empty_ik_does_not_claim_excessive_joint_rotation():
    with pytest.raises(MotionRejected) as exc:
        nearest_solution([], joints(), limits())
    assert exc.value.code == 'IK_NO_SOLUTION'
    assert exc.value.data['ik_solutions'] == 0


def test_start_state_assigns_array_backed_positions_once():
    from array import array
    class State:
        @property
        def position(self):return self._position
        @position.setter
        def position(self,value):self._position=array('d',value)
    motion=PoseMotion.__new__(PoseMotion);state=NS(joint_state=State())
    motion._start_state(state,joints(.2))
    assert state.is_diff and state.joint_state.name==list(ARM_JOINTS)
    assert list(state.joint_state.position)==[.2]*6


@pytest.mark.parametrize('error,expected', [(-31,'IK_NO_SOLUTION'),(-18,'IK_REQUEST_INVALID'),(1,'IK_RESPONSE_INVALID')])
def test_ik_solver_error_codes_are_preserved(monkeypatch,error,expected):
    import sys
    def request():
        return NS(ik_request=NS(robot_state=NS(joint_state=NS()),
            pose_stamped=NS(header=NS(),pose=NS(position=NS(),orientation=NS()))))
    monkeypatch.setitem(sys.modules,'moveit_msgs.srv',NS(GetPositionIK=NS(Request=request)))
    monkeypatch.setitem(sys.modules,'builtin_interfaces.msg',NS(Duration=NS))
    motion=PoseMotion.__new__(PoseMotion);motion.arm=NS(ee_frame='ee',parent_frame='base');motion.ik='ik'
    motion._service=lambda *args: NS(error_code=NS(val=error),solution=NS(joint_state=NS(name=[],position=[])))
    with pytest.raises(MotionRejected) as exc:
        motion.inverse_solutions([.3,0,.2],[0,90,0],joints(),limits(),MotionPolicy())
    assert exc.value.code==expected and exc.value.data['ik_error_codes']==[error]*5


def test_real_ros_planning_states_serialize_complete_joint_positions():
    from pathlib import Path
    import subprocess
    setup=Path('/opt/ros/humble/setup.bash')
    if not setup.is_file():
        pytest.skip('ROS Humble runtime unavailable')
    script='''from moveit_msgs.srv import GetPositionIK,GetMotionPlan,GetCartesianPath
from rclpy.serialization import serialize_message,deserialize_message
from workers.pose_motion import PoseMotion
from core.health import ARM_JOINTS
motion=PoseMotion.__new__(PoseMotion)
for message,state in [(GetPositionIK.Request(),lambda m:m.ik_request.robot_state),
                      (GetMotionPlan.Request(),lambda m:m.motion_plan_request.start_state),
                      (GetCartesianPath.Request(),lambda m:m.start_state)]:
    motion._start_state(state(message),{n:.2 for n in ARM_JOINTS})
    decoded=deserialize_message(serialize_message(message),type(message))
    assert state(decoded).is_diff
    assert state(decoded).joint_state.name==list(ARM_JOINTS)
    assert list(state(decoded).joint_state.position)==[.2]*6
print('IK, motion-plan and Cartesian requests serialized successfully')
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation
from types import SimpleNamespace as NS
actual=Pose();q=Rotation.from_euler('xyz',[.2,50.3,-.1],degrees=True).as_quat()
actual.orientation.x,actual.orientation.y,actual.orientation.z,actual.orientation.w=map(float,q)
motion.arm=NS(ee_frame='ee',parent_frame='base',get_current_ee_pose=lambda:actual)
motion.cartesian='cartesian'
for desired in (None,[0.,50.,0.]):
    def service(client,request,timeout):
        decoded=deserialize_message(serialize_message(request),type(request))
        pose=decoded.waypoints[0];expected=q if desired is None else Rotation.from_euler('xyz',desired,degrees=True).as_quat()
        assert max(abs(a-b) for a,b in zip([pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w],expected))<1e-12
        assert not decoded.path_constraints.joint_constraints
        return NS(error_code=NS(val=1),fraction=1.,solution=NS(joint_trajectory='path'))
    motion._service=service
    assert motion.cartesian_plan([.3,0,.2],desired,{n:0. for n in ARM_JOINTS},None,None)=='path'
print('Cartesian requests preserve actual orientation or explicit RPY')
'''
    result=subprocess.run(['bash','-c','source /opt/ros/humble/setup.bash && exec /usr/bin/python3 -'],
                          input=script,text=True,capture_output=True,timeout=15,
                          cwd=Path(__file__).resolve().parents[1])
    assert result.returncode==0,result.stdout+result.stderr


def test_ik_can_try_a_neutral_guess_without_changing_the_requested_pose(monkeypatch):
    import sys
    def request():
        return NS(ik_request=NS(robot_state=NS(joint_state=NS()),
            pose_stamped=NS(header=NS(),pose=NS(position=NS(),orientation=NS()))))
    monkeypatch.setitem(sys.modules,'moveit_msgs.srv',NS(GetPositionIK=NS(Request=request)))
    monkeypatch.setitem(sys.modules,'builtin_interfaces.msg',NS(Duration=NS))
    motion=PoseMotion.__new__(PoseMotion);motion.arm=NS(ee_frame='ee',parent_frame='base');motion.ik='ik'
    requests=[]
    def service(client,req,timeout):
        requests.append(req.ik_request)
        if len(requests)<4:return NS(error_code=NS(val=-31))
        return NS(error_code=NS(val=1),solution=NS(joint_state=NS(name=list(ARM_JOINTS),position=[.2]*6)))
    motion._service=service
    solutions=motion.inverse_solutions([.3,0,.2],[0,50,0],joints(.1),limits(),MotionPolicy())
    assert solutions==[joints(.2)] and len(requests)==4
    assert requests[-1].robot_state.joint_state.position==[0.]*6
    assert all(not hasattr(req,'constraints') and req.avoid_collisions for req in requests)
    assert all([req.pose_stamped.pose.position.x,req.pose_stamped.pose.position.y,req.pose_stamped.pose.position.z]==[.3,0,.2] for req in requests)
    q=requests[0].pose_stamped.pose.orientation
    assert all(vars(req.pose_stamped.pose.orientation)==vars(q) for req in requests)


def test_rviz_failure_does_not_turn_controller_success_into_motion_failure(monkeypatch):
    import workers.grasp_visualization as visual
    backend=ArmBackend({"controller_dir":"/tmp","robot_name":"test"})
    monkeypatch.setattr(backend,"arm",lambda:NS(node='node'))
    monkeypatch.setattr(visual,"GraspVisualizer",lambda *args:(_ for _ in ()).throw(RuntimeError('rviz offline')))
    monkeypatch.setattr(backend,"ik_move",lambda *args,**kwargs:ToolResult.success('controller completed'))
    result=backend('move_arm',dict(x=.3,y=0,z=.2,roll=0,pitch=50,yaw=0,grasp_debug={'mode':'fallback'}))
    assert result.ok
