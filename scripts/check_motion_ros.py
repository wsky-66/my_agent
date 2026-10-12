"""Synthetic pose-motion checks in isolated ROS domain 218; no device access."""
import os
os.environ["ROS_DOMAIN_ID"] = "218"
os.environ["ROS_LOCALHOST_ONLY"] = "1"
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import threading
import time
from types import SimpleNamespace as NS
import rclpy
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from interbotix_xs_msgs.srv import RobotInfo
from moveit_msgs.srv import GetPositionIK, GetMotionPlan, GetCartesianPath
from control_msgs.action import FollowJointTrajectory
from core.health import ARM_JOINTS, fresh_joints
from workers.pose_motion import PoseMotion

PREFIX = "/my_agent_motion_test"


class Simulator(Node):
    def __init__(self):
        super().__init__("pose_motion_simulator")
        self.positions = [0.]*6
        self.reported_velocity = 0.
        self.pending = None
        self.scenario, self.iks, self.plans, self.goals = "normal", 0, 0, 0
        self.publisher = self.create_publisher(JointState, PREFIX+"/joint_states", qos_profile_sensor_data)
        self.create_timer(.02, self.publish)
        self.create_service(RobotInfo, PREFIX+"/get_robot_info", self.info)
        self.create_service(GetPositionIK, "/compute_ik", self.ik)
        self.create_service(GetMotionPlan, "/plan_kinematic_path", self.plan)
        self.create_service(GetCartesianPath, "/compute_cartesian_path", self.cartesian)
        self.action = ActionServer(self, FollowJointTrajectory, PREFIX+"/arm_controller/follow_joint_trajectory", self.execute)

    def publish(self):
        if self.pending is not None and time.monotonic() >= self.pending[0]:
            self.positions, self.pending = self.pending[1], None
        msg = JointState()
        msg.name, msg.position, msg.velocity = list(ARM_JOINTS), self.positions, [self.reported_velocity]*6
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(msg)

    def info(self, request, response):
        assert request.cmd_type == "group" and request.name == "arm"
        response.mode, response.joint_names = "position", list(ARM_JOINTS)
        response.joint_lower_limits, response.joint_upper_limits = [-math.pi+1e-5]*6, [math.pi-1e-5]*6
        response.joint_velocity_limits = [3.14]*6
        return response

    def ik(self, request, response):
        ir = request.ik_request
        assert ir.avoid_collisions and ir.robot_state.is_diff and not ir.constraints.joint_constraints
        assert ir.pose_stamped.header.frame_id == PREFIX[1:]+"/base_link"
        assert ir.timeout.nanosec == 350_000_000
        values = [0.]*6
        values[-1] = 3.2 if self.scenario == "limit" else ((1.8, .2, .8)[self.iks%3])
        self.iks += 1
        response.error_code.val = 1
        response.solution.joint_state.name, response.solution.joint_state.position = list(ARM_JOINTS), values
        return response

    def cartesian(self, request, response):
        assert self.scenario in ("straight", "travel")
        assert request.avoid_collisions and request.start_state.is_diff
        assert not request.path_constraints.joint_constraints
        # A straight move without RPY keeps the measured orientation, including
        # small differences from the original requested 50-degree pitch.
        from scipy.spatial.transform import Rotation
        q=Rotation.from_euler('xyz',[.2,50.3,-.1],degrees=True).as_quat()
        p=request.waypoints[0].orientation
        assert max(abs(a-b) for a,b in zip([p.x,p.y,p.z,p.w],q))<1e-12
        start=dict(zip(request.start_state.joint_state.name,request.start_state.joint_state.position))
        values=[start[n] for n in ARM_JOINTS]
        end=list(values);end[-1]+=.05
        response.error_code.val=1
        response.fraction=1.
        response.solution.joint_trajectory.joint_names=list(ARM_JOINTS)
        path=[(values,0),(end,1)]
        if self.scenario == "travel":
            middle=list(values);middle[3]=-2.490396947665474
            end[3]=.5330422339166976
            path=[(values,0),(middle,1.1),(end,2.2)]
        for positions,seconds in path:
            point=JointTrajectoryPoint()
            point.positions=positions
            point.time_from_start.sec,point.time_from_start.nanosec=divmod(round(seconds*1e9),1_000_000_000)
            response.solution.joint_trajectory.points.append(point)
        return response

    def plan(self, request, response):
        self.plans += 1
        motion = request.motion_plan_request
        assert motion.num_planning_attempts == 1 and motion.start_state.is_diff
        assert motion.max_velocity_scaling_factor == motion.max_acceleration_scaling_factor == .25
        assert not motion.path_constraints.joint_constraints
        target = [j.position for j in motion.goal_constraints[0].joint_constraints]
        assert abs(target[-1]-.2) < 1e-6, target
        planned = response.motion_plan_response
        planned.error_code.val = -1 if self.scenario == "fail" else 1
        planned.trajectory.joint_trajectory.joint_names = list(ARM_JOINTS)
        values = [self.positions, target]
        times = [0, .2] if self.scenario == "speed" else [0, 2]
        for positions, seconds in zip(values, times):
            point = JointTrajectoryPoint()
            point.positions = list(positions)
            point.time_from_start.sec, point.time_from_start.nanosec = divmod(round(seconds*1e9), 1_000_000_000)
            planned.trajectory.joint_trajectory.points.append(point)
        return response

    def execute(self, handle):
        self.goals += 1
        goal = handle.request
        self.last_trajectory = goal.trajectory
        assert goal.goal_time_tolerance.sec == 2
        assert not goal.goal_tolerance and not goal.path_tolerance
        points = handle.request.trajectory.points
        assert len(points) >= 2
        for a, b in zip(points, points[1:]):
            dt = b.time_from_start.sec+b.time_from_start.nanosec*1e-9-a.time_from_start.sec-a.time_from_start.nanosec*1e-9
            assert max(abs(y-x)/dt for x,y in zip(a.positions,b.positions)) <= 3.14*.25+1e-8
        target = list(handle.request.trajectory.points[-1].positions)
        if self.scenario == "lag":
            # Delay simulated feedback to prove that controller completion does
            # not trigger a second arrival wait or position-error check.
            self.pending = time.monotonic()+2.4, target
        else:
            self.positions = target
        self.reported_velocity = .12
        handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = 0
        return result


class Client(Node):
    def __init__(self):
        super().__init__("pose_motion_test_client")
        self._feedback = None
        self.opts = {"robot_name": PREFIX[1:], "motion_timeout": 15.}
        self.create_subscription(JointState, PREFIX+"/joint_states", self.on_state, qos_profile_sensor_data)

    def on_state(self, message):
        self._feedback = time.monotonic(), message

    def arm(self):
        return NS(node=self, ee_frame=PREFIX[1:]+"/ee_gripper_link", parent_frame=PREFIX[1:]+"/base_link",
                  get_current_ee_pose=self.current_pose)

    def current_pose(self):
        from geometry_msgs.msg import Pose
        from scipy.spatial.transform import Rotation
        pose=Pose();q=Rotation.from_euler('xyz',[.2,50.3,-.1],degrees=True).as_quat()
        pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w=map(float,q)
        return pose

    def joint(self):
        return self

    def get_joints(self, timeout=1.):
        until = time.monotonic()+timeout
        while time.monotonic() < until:
            if fresh_joints(self._feedback, time.monotonic()):
                return dict(zip(self._feedback[1].name, self._feedback[1].position))
            time.sleep(.02)
        return None


rclpy.init()
simulator, client = Simulator(), Client()
executor = MultiThreadedExecutor(num_threads=4)
executor.add_node(simulator)
executor.add_node(client)
thread = threading.Thread(target=executor.spin, daemon=True)
thread.start()
motion = PoseMotion(client)
try:
    first = motion.move([.3,0.,.2], [0.,50.,0.])
    assert first.ok and abs(first.data["target"]["wrist_rotate"]-.2) < 1e-6, first.to_json()
    assert simulator.goals == simulator.plans == 1 and simulator.iks == 3
    # Let the subscriber receive residual velocity after controller success.
    # Production motion must neither wait for it to settle nor reject it.
    time.sleep(.05)
    assert list(client._feedback[1].velocity)==[.12]*6
    simulator.scenario = "straight"
    before = time.monotonic()
    continued = motion.move_cartesian([.3,0.,.1],None)
    assert continued.ok and continued.data["controller_completed"] and simulator.goals==2, continued.to_json()
    assert time.monotonic()-before<2., "unexpected residual-velocity stability wait"
    assert simulator.reported_velocity==.12
    simulator.scenario, simulator.positions = "limit", [0.]*6
    time.sleep(.1)
    invalid = motion.move([.3,0.,.2], [0.,50.,0.])
    assert invalid.error_code == "IK_BRANCH_REJECTED", invalid.to_json()
    assert simulator.goals == 2 and simulator.plans == 1
    simulator.scenario = "fail"
    failed = motion.move([.3,0.,.2], [0.,50.,0.])
    assert failed.error_code == "MOTION_PLAN_FAILED", failed.to_json()
    assert simulator.goals == 2
    simulator.scenario = "speed"
    slowed = motion.move([.3,0.,.2], [0.,50.,0.])
    assert slowed.ok and slowed.data["time_scale"] > 1, slowed.to_json()
    assert max(slowed.data["peak_speed_rad_s"].values()) <= 3.14*.25
    assert simulator.goals == 3
    simulator.positions = [0.]*6
    time.sleep(.1)
    simulator.scenario = "lag"
    before = time.monotonic()
    lagged = motion.move([.3,0.,.2], [0.,50.,0.])
    assert lagged.ok and lagged.data["controller_completed"] is True, lagged.to_json()
    assert time.monotonic()-before < 2., "unexpected post-controller feedback wait"
    assert simulator.pending is not None and simulator.positions == [0.]*6
    assert simulator.goals == 4
    simulator.pending = None
    simulator.positions = [0.]*6
    time.sleep(.1)
    simulator.scenario="straight"
    straight=motion.move_cartesian([.3,0.,.1],None)
    assert straight.ok and straight.data["cartesian"] and simulator.goals==5, straight.to_json()
    assert straight.data['duration']==straight.data['original_duration']==1.
    assert straight.data['time_scale']==1., 'valid MoveIt timing must not be slowed again'
    simulator.scenario="travel";simulator.positions=[0.,0.,0.,.109,0.,0.]
    time.sleep(.1)
    travelled=motion.move_cartesian([.348668,-.017547,.003533],None)
    assert travelled.ok and simulator.goals==6, travelled.to_json()
    assert abs(travelled.data['travel_rad']['forearm_roll']-5.622836129247646)<1e-9
    assert abs(travelled.data['delta_rad']['forearm_roll']-.4240422339166976)<1e-9
    assert travelled.data['original_duration']==2.2 and travelled.data['time_scale']>1.
    assert travelled.data['duration']<8., 'slow only to the driver target, without a separate 4x stretch'
    from workers.grasp_visualization import GraspVisualizer
    from workers.grasp_pipeline import GraspPipeline
    from visualization_msgs.msg import MarkerArray
    from sensor_msgs.msg import PointCloud2
    from moveit_msgs.msg import DisplayTrajectory
    from rclpy.qos import QoSProfile,DurabilityPolicy
    import numpy as np
    visual=GraspVisualizer();selected=GraspVisualizer(client)
    executor.add_node(visual.node)
    pipeline=GraspPipeline(None,None)
    pipeline.valid=np.ones((2,3),bool)
    pipeline.points=np.array([[.3,0,.05],[.9,.9,.9],[.31,0,.05],[.32,0,.05],[.9,.9,.9],[.33,0,.05]])
    pipeline.colors=np.array([[1.,0.,0.],[0.,1.,0.],[1.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,0.,0.]])
    points,colors=pipeline.segmented_cloud([np.array([[True,False,True],[True,False,True]])])
    visual.publish_scene(points,colors,
        [dict(label="pepper",position=[.3,0,.05],rpy=[0,50,0],width=.04,depth=.02,score=.3)],"base")
    selected.publish_selection(dict(label="pepper",mode="fallback",position=[.3,0,.03],rpy=[0,50,0]),"base")
    late=Node("rviz_late_subscriber");executor.add_node(late);received={}
    qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
    for topic,kind in [("/graspnet/pointcloud",PointCloud2),("/graspnet/grasps",MarkerArray),
                       ("/graspnet/selected_grasp",MarkerArray),("/move_group/display_planned_path",DisplayTrajectory)]:
        late.create_subscription(kind,topic,lambda msg,topic=topic:received.update({topic:msg}),qos)
    until=time.monotonic()+4
    while len(received)<4 and time.monotonic()<until:time.sleep(.02)
    assert len(received)==4, list(received)
    pc=received["/graspnet/pointcloud"]
    assert pc.header.frame_id=="base" and pc.width==4 and len(pc.data)==64
    assert np.all(np.frombuffer(pc.data,dtype='<u4').reshape(-1,4)[:,3]==0xff0000)
    assert np.max(np.frombuffer(pc.data,dtype='<f4').reshape(-1,4)[:,:3])<.4
    markers=received["/graspnet/grasps"].markers
    assert markers[0].action==markers[0].DELETEALL
    assert len(next(m for m in markers if m.ns=='gripper').points)==96
    arrow=next(m for m in markers if m.ns=='approach')
    assert arrow.points[1].x>arrow.points[0].x and arrow.points[1].z<arrow.points[0].z
    assert next(m for m in received["/graspnet/selected_grasp"].markers if m.ns=='label').text.endswith('fallback')
    displayed=received["/move_group/display_planned_path"]
    assert displayed.trajectory[0].joint_trajectory==simulator.last_trajectory
    assert displayed.trajectory_start.joint_state.position==simulator.last_trajectory.points[0].positions
    # The installed MoveIt RViz plugin uses volatile QoS, so test a viewer
    # joining after execution with that exact durability too.
    volatile=[]
    late.create_subscription(DisplayTrajectory,"/move_group/display_planned_path",volatile.append,10)
    until=time.monotonic()+3
    while not volatile and time.monotonic()<until:time.sleep(.02)
    assert volatile and volatile[-1].trajectory[0].joint_trajectory==simulator.last_trajectory
    count=len(volatile)
    time.sleep(.7)
    assert len(volatile)==count, 'do not continuously restart RViz animation'
    executor.remove_node(late);late.destroy_node()
    executor.remove_node(visual.node);visual.close();selected.close()
    print("ROS 模拟验证通过：邻近逆解与物理限位；控制器成功后残余速度不阻止下一动作；直线运动保持实际朝向和有效计时；5.62 rad 累计变化不被拦截；过快路径仅按驱动目标减速一次；无额外到位等待；RViz 显示实际下发轨迹，支持 Volatile 后加入订阅者且不持续重播；点云只含分割目标。")

finally:
    motion.close()
    executor.shutdown(timeout_sec=2)
    thread.join(timeout=2)
    simulator.action.destroy()
    client.destroy_node()
    simulator.destroy_node()
    rclpy.shutdown()
