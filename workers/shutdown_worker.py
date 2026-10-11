"""Bounded shutdown: cancel goals, plan to driver-defined Sleep, verify live feedback."""
import time
import math

from core.health import ARM_JOINTS, fresh_joints
from core.parking import park, sleep_error
from core.results import ToolResult
from workers.protocol import options, serve


class ShutdownBackend:
    def __init__(self, opts):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data, qos_profile_action_status_default
        from rclpy.action import ActionClient
        from sensor_msgs.msg import JointState
        from action_msgs.msg import GoalStatusArray
        from action_msgs.srv import CancelGoal
        from interbotix_xs_msgs.srv import RobotInfo
        from moveit_msgs.srv import GetMotionPlan
        from control_msgs.action import FollowJointTrajectory
        rclpy.init()
        self.node = Node("my_agent_shutdown")
        self.opts = opts
        self.deadline = time.monotonic() + opts["timeout"]
        self._sample = None
        self._statuses = {}
        self._goal = None
        prefix = "/" + opts["robot_name"]
        self.node.create_subscription(JointState, prefix + "/joint_states", self._on_state, qos_profile_sensor_data)
        self.driver = self.node.create_client(RobotInfo, prefix + "/get_robot_info")
        self.planner = self.node.create_client(GetMotionPlan, "/plan_kinematic_path")
        self.action = ActionClient(self.node, FollowJointTrajectory, prefix + "/arm_controller/follow_joint_trajectory")
        self.cancellers = {}
        for name in ("arm_controller", "gripper_controller"):
            endpoint = prefix + f"/{name}/follow_joint_trajectory/_action"
            self.cancellers[name] = self.node.create_client(CancelGoal, endpoint + "/cancel_goal")
            self.node.create_subscription(GoalStatusArray, endpoint + "/status",
                lambda msg, name=name: self._statuses.__setitem__(name, msg), qos_profile_action_status_default)

    def _on_state(self, msg):
        self._sample = time.monotonic(), msg

    def _spin(self):
        import rclpy
        if time.monotonic() >= self.deadline or not rclpy.ok():
            raise TimeoutError("退出收拢超时或 ROS 已关闭")
        rclpy.spin_once(self.node, timeout_sec=min(.05, self.deadline-time.monotonic()))

    def _wait(self, future, timeout):
        until = min(self.deadline, time.monotonic()+timeout)
        while not future.done() and time.monotonic() < until:
            self._spin()
        if not future.done():
            raise TimeoutError("退出收拢请求超时")
        result = future.result()
        if result is None:
            raise RuntimeError("退出收拢收到空响应")
        return result

    def _service(self, client, request, timeout=2):
        until = min(self.deadline, time.monotonic()+timeout)
        while not client.service_is_ready() and time.monotonic() < until:
            self._spin()
        if not client.service_is_ready():
            raise RuntimeError("退出所需 ROS 服务不可用")
        future = client.call_async(request)
        try:
            return self._wait(future, max(.05, until-time.monotonic()))
        finally:
            if not future.done():
                client.remove_pending_request(future)

    def cancel_motion(self):
        from action_msgs.srv import CancelGoal
        # Allow discovery, then cancel all controller goals, including orphaned goals.
        until = min(self.deadline, time.monotonic()+1)
        while time.monotonic() < until:
            self._spin()
        cancelling = {}
        for name, client in self.cancellers.items():
            if name == "arm_controller" and not client.service_is_ready():
                return ToolResult.failure("SHUTDOWN_CANCEL_MISSING", "机械臂取消接口不可用，跳过自动收拢")
            if client.service_is_ready():
                reply = self._service(client, CancelGoal.Request())
                if reply.return_code != 0:
                    return ToolResult.failure("SHUTDOWN_CANCEL_FAILED", "控制器拒绝取消动作，跳过自动收拢")
                cancelling[name] = {tuple(g.goal_id.uuid) for g in reply.goals_canceling}
        until = min(self.deadline, time.monotonic()+2)
        while time.monotonic() < until:
            active = any(s.status in (1, 2, 3) for msg in self._statuses.values() for s in msg.status_list)
            for name, ids in cancelling.items():
                msg = self._statuses.get(name)
                states = {tuple(s.goal_info.goal_id.uuid): s.status for s in msg.status_list} if msg else {}
                active = active or any(states.get(goal) not in (4, 5, 6) for goal in ids)
            if not active:
                return ToolResult.success("已取消控制器任务")
            self._spin()
        return ToolResult.failure("SHUTDOWN_ROBOT_BUSY", "控制器仍有进行中的动作，跳过自动收拢")

    def sleep_target(self):
        from interbotix_xs_msgs.srv import RobotInfo
        request = RobotInfo.Request()
        request.cmd_type, request.name = "group", "arm"
        info = self._service(self.driver, request)
        target = dict(zip(info.joint_names, info.joint_sleep_positions))
        if len(info.joint_names) != len(info.joint_sleep_positions) or any(n not in target for n in ARM_JOINTS):
            return ToolResult.failure("SHUTDOWN_SLEEP_CONFIG", "驱动没有提供完整 Sleep 姿态")
        lower = dict(zip(info.joint_names, info.joint_lower_limits))
        upper = dict(zip(info.joint_names, info.joint_upper_limits))
        if any(not math.isfinite(target[n]) or n not in lower or n not in upper or
               not lower[n] <= target[n] <= upper[n] for n in ARM_JOINTS):
            return ToolResult.failure("SHUTDOWN_SLEEP_CONFIG", "驱动的 Sleep 姿态无效或超出限位")
        return ToolResult.success("已读取驱动 Sleep 姿态", {n: float(target[n]) for n in ARM_JOINTS})

    def joints(self):
        until = min(self.deadline, time.monotonic()+1.5)
        while time.monotonic() < until:
            self._spin()
            if fresh_joints(self._sample, time.monotonic()):
                values = dict(zip(self._sample[1].name, self._sample[1].position))
                return ToolResult.success("已读取近期关节反馈", {n: float(values[n]) for n in ARM_JOINTS})
        return ToolResult.failure("SHUTDOWN_NO_FEEDBACK", "未收到近期完整关节反馈，无法确认或调整 Sleep")

    def move_to_sleep(self, target):
        from moveit_msgs.srv import GetMotionPlan
        from moveit_msgs.msg import Constraints, JointConstraint
        from control_msgs.action import FollowJointTrajectory
        from builtin_interfaces.msg import Duration
        initial = self.joints()
        if not initial.ok:
            return initial
        request = GetMotionPlan.Request()
        motion = request.motion_plan_request
        motion.group_name = "interbotix_arm"
        motion.num_planning_attempts, motion.allowed_planning_time = 1, 5.0
        motion.max_velocity_scaling_factor = motion.max_acceleration_scaling_factor = .25
        motion.start_state.joint_state = self._sample[1]
        constraints = Constraints()
        for name in ARM_JOINTS:
            constraints.joint_constraints.append(JointConstraint(joint_name=name, position=target[name],
                tolerance_above=.005, tolerance_below=.005, weight=1.0))
        motion.goal_constraints = [constraints]
        response = self._service(self.planner, request, timeout=6).motion_plan_response
        if response.error_code.val != 1:
            return ToolResult.failure("SHUTDOWN_PLAN_FAILED", "Sleep 路径规划失败，跳过收拢并继续清理")
        trajectory = response.trajectory.joint_trajectory
        if not trajectory.points or set(trajectory.joint_names) != set(ARM_JOINTS):
            return ToolResult.failure("SHUTDOWN_PLAN_INVALID", "Sleep 规划轨迹为空或关节不完整")
        previous = -1
        for point in trajectory.points:
            timestamp = point.time_from_start.sec + point.time_from_start.nanosec*1e-9
            if len(point.positions) != len(ARM_JOINTS) or not all(math.isfinite(v) for v in point.positions) or timestamp <= previous:
                return ToolResult.failure("SHUTDOWN_PLAN_INVALID", "Sleep 规划轨迹数据无效")
            previous = timestamp
        end = dict(zip(trajectory.joint_names, trajectory.points[-1].positions))
        if sleep_error(end, target) > self.opts["tolerance"]:
            return ToolResult.failure("SHUTDOWN_PLAN_INVALID", "规划终点不在 Sleep 允许误差内")
        current = self.joints()
        if not current.ok or sleep_error(current.data, initial.data) > .03:
            return ToolResult.failure("SHUTDOWN_ROBOT_MOVED", "规划期间机械臂状态变化，停止发送收拢轨迹")
        if not self.action.server_is_ready():
            return ToolResult.failure("SHUTDOWN_ACTION_MISSING", "机械臂轨迹控制器不可用")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        goal.goal_time_tolerance = Duration(sec=2)
        self._goal = self._wait(self.action.send_goal_async(goal), 2)
        if not self._goal.accepted:
            self._goal = None
            return ToolResult.failure("SHUTDOWN_GOAL_REJECTED", "控制器拒绝 Sleep 轨迹")
        try:
            reply = self._wait(self._goal.get_result_async(), max(1, previous+3))
            self._goal = None
            if reply.status != 4 or reply.result.error_code != 0:
                return ToolResult.failure("SHUTDOWN_MOTION_FAILED", "Sleep 轨迹执行失败")
        finally:
            self._cancel_goal()
        return ToolResult.success("Sleep 轨迹执行完成，正在验证关节反馈")

    def verify_sleep(self, target):
        until = min(self.deadline, time.monotonic()+2)
        settled = None
        last_received = None
        error = None
        while time.monotonic() < until:
            self._spin()
            if not fresh_joints(self._sample, time.monotonic()):
                settled = None
                continue
            received, msg = self._sample
            if received == last_received:
                continue
            last_received = received
            error = sleep_error(dict(zip(msg.name, msg.position)), target)
            velocities = dict(zip(msg.name, msg.velocity))
            quiet = not msg.velocity or all(n in velocities and math.isfinite(velocities[n]) and abs(velocities[n]) <= .05
                                           for n in ARM_JOINTS)
            if error <= self.opts["tolerance"] and quiet:
                settled = received if settled is None else settled
                if received-settled >= .25:
                    return ToolResult.success("Sleep 到位确认", {"max_joint_error": error})
            else:
                settled = None
        return ToolResult.failure("SHUTDOWN_SLEEP_UNCONFIRMED", "无法确认机械臂已到 Sleep；继续释放进程", {"max_joint_error": error})

    def _cancel_goal(self):
        if self._goal is not None:
            self._goal.cancel_goal_async()
            self._goal = None

    def __call__(self, action, params):
        if action != "park":
            raise ValueError("未知退出动作")
        try:
            return park(self, self.opts["tolerance"])
        except (TimeoutError, RuntimeError) as exc:
            return ToolResult.failure("SHUTDOWN_SLEEP_FAILED", str(exc), {"sleep_verified": False})
        finally:
            self._cancel_goal()

    def close(self):
        import rclpy
        self._cancel_goal()
        self.action.destroy()
        self.node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    backend = ShutdownBackend(options())
    serve(backend, backend.close)
