"""Optional ROS smoke check using synthetic services in domain 217; no hardware access."""
import os
os.environ['ROS_DOMAIN_ID'] = '217'
os.environ['ROS_LOCALHOST_ONLY'] = '1'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import threading
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionServer
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from interbotix_xs_msgs.srv import RobotInfo
from moveit_msgs.srv import GetMotionPlan
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from core.health import ARM_JOINTS
from services.runtime import worker

TARGET = [0., -1.8, 1.55, 0., .8, 0.]


class Simulator(Node):
    def __init__(self):
        super().__init__('shutdown_simulator')
        self.positions = [0.] * 6
        self.plans = self.goals = 0
        self.plan_ok = True
        self.publisher = self.create_publisher(JointState, '/my_agent_shutdown_test/joint_states', qos_profile_sensor_data)
        self.timer = self.create_timer(.02, self.publish)
        self.create_service(RobotInfo, '/my_agent_shutdown_test/get_robot_info', self.info)
        self.create_service(GetMotionPlan, '/plan_kinematic_path', self.plan)
        self.action = ActionServer(self, FollowJointTrajectory, '/my_agent_shutdown_test/arm_controller/follow_joint_trajectory', self.execute)

    def publish(self):
        message = JointState()
        message.name, message.position, message.velocity = list(ARM_JOINTS), self.positions, [0.] * 6
        message.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(message)

    def info(self, request, response):
        assert request.cmd_type == 'group' and request.name == 'arm'
        response.joint_names, response.joint_sleep_positions = list(ARM_JOINTS), TARGET
        response.joint_lower_limits, response.joint_upper_limits = [-3.] * 6, [3.] * 6
        return response

    def plan(self, request, response):
        self.plans += 1
        motion = request.motion_plan_request
        assert motion.group_name == 'interbotix_arm'
        assert motion.max_velocity_scaling_factor == .25 and motion.max_acceleration_scaling_factor == .25
        assert all(abs(j.position-v) < 1e-6 for j,v in zip(motion.goal_constraints[0].joint_constraints, TARGET))
        planned = response.motion_plan_response
        planned.error_code.val = 1 if self.plan_ok else -1
        planned.trajectory.joint_trajectory.joint_names = list(ARM_JOINTS)
        point = JointTrajectoryPoint()
        point.positions = TARGET
        point.time_from_start.sec = 1
        planned.trajectory.joint_trajectory.points = [point]
        return response

    def execute(self, handle):
        self.goals += 1
        self.positions = list(handle.request.trajectory.points[-1].positions)
        handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = 0
        return result


rclpy.init()
simulator = Simulator()
executor = MultiThreadedExecutor(num_threads=3)
executor.add_node(simulator)
thread = threading.Thread(target=executor.spin, daemon=True)
thread.start()
client = worker('shutdown', '/usr/bin/python3', {'robot_name':'my_agent_shutdown_test', 'timeout':15, 'tolerance':.03}, ros=True, workspace=True, timeout=18)
client.startup_timeout = 8
try:
    first = client.request('park', allow_shutdown=True)
    assert first.ok and first.data['sleep_verified'] and not first.data['already_sleep'], first.to_json()
    assert simulator.goals == simulator.plans == 1
    second = client.request('park', allow_shutdown=True)
    assert second.ok and second.data['already_sleep'], second.to_json()
    assert simulator.goals == simulator.plans == 1
    simulator.positions = [0.] * 6
    simulator.plan_ok = False
    failed = client.request('park', allow_shutdown=True)
    assert failed.error_code == 'SHUTDOWN_PLAN_FAILED', failed.to_json()
    assert simulator.goals == 1
    print('隔离 ROS 域模拟验证通过：规划一次并确认 Sleep；已在 Sleep 不发目标；规划失败不发送动作。')
finally:
    client.stop()
    executor.shutdown(timeout_sec=2)
    thread.join(timeout=2)
    simulator.action.destroy()
    simulator.destroy_node()
    rclpy.shutdown()
