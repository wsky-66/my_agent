"""ROS adapter for guarded pose moves, sharing the arm worker's spinning node."""
import time
import math

from core.health import ARM_JOINTS, fresh_joints
from core.motion import (MotionPolicy, MotionRejected, driver_limits, move_pose, move_cartesian,
                         inspect_grasp_candidates)


class PoseMotion:
    def __init__(self, backend):
        from moveit_msgs.srv import GetPositionIK, GetMotionPlan
        from interbotix_xs_msgs.srv import RobotInfo
        from control_msgs.action import FollowJointTrajectory
        from rclpy.action import ActionClient
        self.backend, self.arm = backend, backend.arm()
        self.policy = MotionPolicy(backend.opts.get("max_joint_change", 2.6),
                                   backend.opts.get("velocity_scaling", .25),
                                   backend.opts.get("motion_timeout", 45.))
        self.driver = self.arm.node.create_client(RobotInfo, f"/{backend.opts['robot_name']}/get_robot_info")
        self.ik = self.arm.node.create_client(GetPositionIK, "/compute_ik")
        self.planner = self.arm.node.create_client(GetMotionPlan, "/plan_kinematic_path")
        self.action = ActionClient(self.arm.node, FollowJointTrajectory,
            f"/{backend.opts['robot_name']}/arm_controller/follow_joint_trajectory")
        self.goal = None

    def _wait(self, future, timeout):
        import rclpy
        until = min(self.deadline, time.monotonic()+timeout)
        while not future.done() and rclpy.ok() and time.monotonic() < until:
            time.sleep(.01)
        if not future.done() or not rclpy.ok():
            raise MotionRejected("MOTION_TIMEOUT", "位姿运动请求超时或 ROS 已关闭")
        result = future.result()
        if result is None:
            raise MotionRejected("MOTION_RESPONSE_EMPTY", "位姿运动接口返回空响应")
        return result

    def _service(self, client, request, timeout):
        if time.monotonic() >= self.deadline:
            raise MotionRejected("MOTION_TIMEOUT", "候选预检或运动请求超过总时间预算")
        available = min(2., max(0., self.deadline-time.monotonic()))
        if not client.wait_for_service(timeout_sec=available):
            raise MotionRejected("MOTION_SERVICE_MISSING", "位姿运动所需的驱动、逆解或路径规划接口不可用")
        future = client.call_async(request)
        try:
            return self._wait(future, timeout)
        finally:
            if not future.done():
                client.remove_pending_request(future)

    def limits(self):
        from interbotix_xs_msgs.srv import RobotInfo
        request = RobotInfo.Request()
        request.cmd_type, request.name = "group", "arm"
        return driver_limits(self._service(self.driver, request, 2))

    def joints(self):
        client = self.backend.joint()
        positions = client.get_joints(timeout=min(1., max(.01, self.deadline-time.monotonic())))
        if not positions:
            raise MotionRejected("MOTION_NO_FEEDBACK", "没有近期完整关节反馈，停止移动")
        sample = getattr(client, "_feedback", None)
        if sample and sample[1].velocity:
            velocities = dict(zip(sample[1].name, sample[1].velocity))
            if any(n not in velocities or not math.isfinite(velocities[n]) or abs(velocities[n]) > .05 for n in ARM_JOINTS):
                raise MotionRejected("MOTION_ROBOT_BUSY", "机械臂仍在运动，停止发送新的位姿轨迹")
        return positions

    @staticmethod
    def _constraints(start, limits, max_change):
        from moveit_msgs.msg import Constraints, JointConstraint
        result = Constraints()
        for n in ARM_JOINTS:
            lo, hi, _ = limits[n]
            result.joint_constraints.append(JointConstraint(joint_name=n, position=start[n], weight=1.,
                tolerance_above=max(0., min(max_change, hi-start[n])),
                tolerance_below=max(0., min(max_change, start[n]-lo))))
        return result

    def inverse_solutions(self, xyz, rpy, start, limits, policy):
        from builtin_interfaces.msg import Duration
        from moveit_msgs.srv import GetPositionIK
        from scipy.spatial.transform import Rotation
        quaternion = Rotation.from_euler("xyz", rpy, degrees=True).as_quat()
        # A small, deterministic seed neighbourhood; all solutions obey the
        # same absolute-angle budget around the real state, including roll joints.
        seeds = [start]
        for sign in (1., -1.):
            seed = dict(start)
            for n in ("forearm_roll", "wrist_rotate"):
                lo, hi, _ = limits[n]
                seed[n] = max(lo, min(hi, start[n]+sign*.45))
            seeds.append(seed)
        solutions = []
        errors = []
        for seed in seeds:
            request = GetPositionIK.Request()
            ir = request.ik_request
            ir.group_name, ir.ik_link_name = "interbotix_arm", self.arm.ee_frame
            self._start_state(ir.robot_state, seed)
            ir.constraints = self._constraints(start, limits, policy.max_joint_change)
            ir.avoid_collisions = True
            ir.timeout = Duration(nanosec=350_000_000)
            ir.pose_stamped.header.frame_id = self.arm.parent_frame
            pose = ir.pose_stamped.pose
            pose.position.x, pose.position.y, pose.position.z = map(float, xyz)
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, quaternion)
            response = self._service(self.ik, request, 1.5)
            errors.append(int(response.error_code.val))
            if response.error_code.val == 1:
                state = response.solution.joint_state
                if (len(state.name) == len(state.position) and len(set(state.name)) == len(state.name)
                        and set(ARM_JOINTS).issubset(state.name)):
                    solutions.append(dict(zip(state.name, state.position)))
        if not solutions:
            if 1 in errors:
                raise MotionRejected("IK_RESPONSE_INVALID", "逆解接口报告成功但未返回完整关节解，停止预检",
                                     {"ik_error_codes": errors, "ik_solutions": 0})
            if any(code in (-15,-16,-17,-18,-21,-22,-23,-24,-25,-26) for code in errors):
                raise MotionRejected("IK_REQUEST_INVALID", "逆解接口报告请求、坐标系或机器人状态错误，停止预检",
                                     {"ik_error_codes": errors, "ik_solutions": 0})
            raise MotionRejected("IK_NO_SOLUTION", "逆解求解器未返回有效解，尚不能归因于关节转角限制",
                                 {"ik_error_codes": errors, "ik_solutions": 0})
        return solutions

    def preview_opening(self, value):
        # Hypothetical finger state is sent only to planning services.
        # It never calls the gripper controller.
        self._preview_opening = value

    def _start_state(self, state, joints):
        state.is_diff = True
        state.joint_state.name = list(ARM_JOINTS)
        state.joint_state.position = [joints[n] for n in ARM_JOINTS]
        opening = getattr(self, "_preview_opening", None)
        if opening is not None:
            state.joint_state.name += ["left_finger", "right_finger"]
            state.joint_state.position += [opening, -opening]

    def inspect_grasps(self, candidates):
        from logger import record_motion
        self.deadline = time.monotonic()+self.policy.timeout
        return inspect_grasp_candidates(self, candidates, self.policy, audit=record_motion)

    def plan(self, start, target, limits, policy):
        from moveit_msgs.srv import GetMotionPlan
        from moveit_msgs.msg import Constraints, JointConstraint
        request = GetMotionPlan.Request()
        motion = request.motion_plan_request
        motion.group_name = "interbotix_arm"
        motion.num_planning_attempts, motion.allowed_planning_time = 1, 4.
        motion.max_velocity_scaling_factor = motion.max_acceleration_scaling_factor = policy.velocity_scaling
        self._start_state(motion.start_state, start)
        constraints = Constraints()
        constraints.joint_constraints = [JointConstraint(joint_name=n, position=target[n],
            tolerance_above=.005, tolerance_below=.005, weight=1.) for n in ARM_JOINTS]
        motion.goal_constraints = [constraints]
        motion.path_constraints = self._constraints(start, limits, policy.max_joint_change)
        response = self._service(self.planner, request, 6).motion_plan_response
        if response.error_code.val != 1:
            raise MotionRejected("MOTION_PLAN_FAILED", "完整路径规划失败，停止移动；请拆分位置或朝向变化")
        return response.trajectory.joint_trajectory

    def cartesian_plan(self, xyz, rpy, start, limits, policy):
        from moveit_msgs.srv import GetCartesianPath
        from geometry_msgs.msg import Pose
        from scipy.spatial.transform import Rotation
        if not hasattr(self,"cartesian"):
            self.cartesian = self.arm.node.create_client(GetCartesianPath,"/compute_cartesian_path")
        request = GetCartesianPath.Request()
        request.header.frame_id = self.arm.parent_frame
        request.group_name,request.link_name = "interbotix_arm",self.arm.ee_frame
        self._start_state(request.start_state, start)
        pose = Pose(); pose.position.x,pose.position.y,pose.position.z = map(float,xyz)
        q = Rotation.from_euler("xyz",rpy,degrees=True).as_quat()
        pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w = map(float,q)
        request.waypoints = [pose]
        request.max_step,request.jump_threshold,request.avoid_collisions = .003,0.,True
        request.path_constraints = self._constraints(start,limits,policy.max_joint_change)
        response = self._service(self.cartesian,request,6)
        if response.error_code.val!=1 or not math.isfinite(response.fraction) or response.fraction<.999:
            raise MotionRejected("CARTESIAN_PATH_INCOMPLETE","进近/撤离的笛卡尔路径不完整，停止执行")
        return response.solution.joint_trajectory

    @staticmethod
    def _cancel_late(future):
        try:
            handle = future.result()
            if handle is not None and handle.accepted:
                handle.cancel_goal_async()
        except Exception:
            pass

    def execute(self, trajectory, duration):
        from control_msgs.action import FollowJointTrajectory
        from builtin_interfaces.msg import Duration
        if not self.action.wait_for_server(timeout_sec=min(2., max(0., self.deadline-time.monotonic()))):
            raise MotionRejected("MOTION_CONTROLLER_MISSING", "机械臂轨迹控制器不可用")
        if time.monotonic()+duration+4 > self.deadline:
            raise MotionRejected("MOTION_PLAN_DURATION", "剩余时间不足以执行轨迹并确认到位，停止发送动作")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        goal.goal_time_tolerance = Duration(sec=2)
        future = self.action.send_goal_async(goal)
        try:
            self.goal = self._wait(future, 2)
        except MotionRejected:
            future.add_done_callback(self._cancel_late)
            raise
        if not self.goal.accepted:
            self.goal = None
            raise MotionRejected("MOTION_GOAL_REJECTED", "控制器拒绝规划轨迹")
        try:
            response = self._wait(self.goal.get_result_async(), duration+3)
            self.goal = None
            if response.status != 4 or response.result.error_code != 0:
                raise MotionRejected("MOTION_EXECUTION_FAILED", "规划轨迹执行失败，停止后续步骤")
        finally:
            if self.goal is not None:
                self.goal.cancel_goal_async()
                self.goal = None

    def verify(self, target):
        client = self.backend.joint()
        since = time.monotonic()
        settled = None
        until = min(self.deadline, since+2)
        while time.monotonic() < until:
            sample = getattr(client, "_feedback", None)
            if fresh_joints(sample, time.monotonic()) and sample[0] > since:
                state = sample[1]
                positions = dict(zip(state.name, state.position))
                velocities = dict(zip(state.name, state.velocity))
                quiet = not state.velocity or all(n in velocities and abs(velocities[n]) <= .05 for n in ARM_JOINTS)
                if quiet and max(abs(positions[n]-target[n]) for n in ARM_JOINTS) <= .03:
                    settled = sample[0] if settled is None else settled
                    if sample[0]-settled >= .15:
                        return positions
                else:
                    settled = None
            else:
                settled = None
            time.sleep(.02)
        raise MotionRejected("MOTION_NOT_CONFIRMED", "没有收到稳定到位的近期关节反馈，停止后续动作")

    def move(self, xyz, rpy):
        from logger import record_motion
        self.deadline = time.monotonic()+self.policy.timeout
        return move_pose(self, xyz, rpy, self.policy, audit=record_motion)

    def move_cartesian(self, xyz, rpy):
        from logger import record_motion
        self.deadline = time.monotonic()+self.policy.timeout
        return move_cartesian(self,xyz,rpy,self.policy,audit=record_motion)

    def close(self):
        if self.goal is not None:
            self.goal.cancel_goal_async()
            self.goal = None
        self.action.destroy()
