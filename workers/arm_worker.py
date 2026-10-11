"""Persistent ROS clients, with one ROS context owned by this worker."""
import math
import sys
import time

from core.results import ToolResult
from workers.protocol import options, serve


class ArmBackend:
    def __init__(self, opts):
        self.opts = opts
        self.clients = {}
        sys.path.insert(0, opts["controller_dir"])

    def joint(self):
        if "joint" not in self.clients:
            import joint_client
            # The legacy module uses a global topic despite accepting robot_name.
            joint_client.JOINT_TOPIC = f"/{self.opts['robot_name']}/joint_states"
            self.clients["joint"] = joint_client.JointClient(self.opts["robot_name"])
        return self.clients["joint"]

    def arm(self):
        if "arm" not in self.clients:
            from moveit_client import MoveItClient
            backend = self

            class PersistentMoveIt(MoveItClient):
                def _get_current_joint_positions(self, timeout=3):
                    positions = backend.joint().get_joints(timeout)
                    return positions if positions and all(n in positions for n in self.ARM_JOINTS) else None

            self.clients["arm"] = PersistentMoveIt(self.opts["robot_name"])
        return self.clients["arm"]

    def gripper(self):
        if "gripper" not in self.clients:
            from gripper_client import GripperClient
            self.clients["gripper"] = GripperClient(self.opts["robot_name"])
        return self.clients["gripper"]

    def pose(self):
        from scipy.spatial.transform import Rotation
        pose = self.arm().get_current_ee_pose()
        if pose is None:
            return ToolResult.failure("POSE_UNAVAILABLE", "无法读取末端位姿")
        q = pose.orientation
        rpy = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz", degrees=True).tolist()
        p = pose.position
        return ToolResult.success("已读取末端位姿", {"position": [p.x, p.y, p.z], "rpy": rpy,
                                                     "frame": self.arm().parent_frame})

    def ik_move(self, xyz, rpy):
        """Reuse the joint client; legacy clients shut down the entire ROS context on exit."""
        import rclpy
        from builtin_interfaces.msg import Duration
        from moveit_msgs.srv import GetPositionIK
        from scipy.spatial.transform import Rotation
        arm = self.arm()
        if not hasattr(arm, "_ik"):
            arm._ik = arm.node.create_client(GetPositionIK, "/compute_ik")
        if not arm._ik.wait_for_service(timeout_sec=5):
            return False, "compute_ik 服务未就绪"
        req = GetPositionIK.Request()
        ir = req.ik_request
        ir.group_name, ir.ik_link_name = "interbotix_arm", arm.ee_frame
        ir.pose_stamped.header.frame_id = arm.parent_frame
        p = ir.pose_stamped.pose
        p.position.x, p.position.y, p.position.z = xyz
        q = Rotation.from_euler("xyz", rpy, degrees=True).as_quat()
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = map(float, q)
        ir.timeout = Duration(sec=2)
        ir.avoid_collisions = True
        current = self.joint().get_joints()
        if not current or any(n not in current for n in arm.ARM_JOINTS):
            return False, "无法读取关节状态"
        ir.robot_state.joint_state.name = list(current)
        ir.robot_state.joint_state.position = list(current.values())
        future = arm._ik.call_async(req)
        deadline = time.monotonic() + 25
        while not future.done() and rclpy.ok() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done():
            return False, "compute_ik 超时"
        res = future.result()
        if res is None or res.error_code.val != 1:
            return False, "compute_ik 无解，目标位姿不可达"
        js = res.solution.joint_state
        joints = {n: float(p) for n, p in zip(js.name, js.position) if n in arm.ARM_JOINTS}
        if len(joints) != len(arm.ARM_JOINTS):
            return False, "compute_ik 关节结果不完整"
        ok = self.joint().set_joints(joints)
        return ok, "" if ok else "关节执行失败"

    def __call__(self, action, args):
        if action == "get_arm_pose":
            return self.pose()
        if action == "get_joints":
            data = self.joint().get_joints()
            return ToolResult.success("已读取关节角度（弧度）", data) if data else ToolResult.failure("JOINT_STATE_UNAVAILABLE", "未收到关节状态")
        if action == "move_arm":
            xyz = [float(args[k]) for k in ("x", "y", "z")]
            rpy = [args.get(k) for k in ("roll", "pitch", "yaw")]
            if not all(math.isfinite(x) for x in xyz):
                raise ValueError("坐标必须为有限数字")
            given = [v is not None for v in rpy]
            if any(given) and not all(given):
                raise ValueError("roll/pitch/yaw 需同时给出或都不给")
            if all(given):
                rpy = [float(v) for v in rpy]
                if not all(math.isfinite(v) for v in rpy):
                    raise ValueError("朝向必须为有限数字")
                ok, msg = self.ik_move(xyz, rpy)
            else:
                ok, msg = self.arm()._move_group_move(*xyz)
            if not ok:
                return ToolResult.failure("MOTION_FAILED", msg)
            actual = self.pose()
            if not actual.ok:
                return actual
            dev = max(abs(a-b) for a, b in zip(actual.data["position"], xyz))
            if dev > 0.07:
                return ToolResult.failure("POSITION_DEVIATION", f"实际位置偏差 {dev:.3f}m，未确认到位", actual.data)
            return ToolResult.success("末端已到目标位置", actual.data)
        if action == "gripper_set":
            value = float(args["position"])
            if not math.isfinite(value) or not 0.015 <= value <= 0.037:
                raise ValueError("夹爪位置范围为 0.015~0.037 米")
            ok = self.gripper().set(value)
        elif action in ("go_home", "go_sleep"):
            ok = getattr(self.joint(), action)()
        elif action == "set_joints":
            joints = args["joints"]
            if not joints or any(k not in self.arm().ARM_JOINTS for k in joints):
                raise ValueError("关节名称无效或列表为空")
            if not all(math.isfinite(float(v)) for v in joints.values()):
                raise ValueError("关节角必须为有限数字")
            current = self.joint().get_joints()
            if not current or any(n not in current for n in self.arm().ARM_JOINTS):
                return ToolResult.failure("JOINT_STATE_UNAVAILABLE", "关节状态不完整，停止关节移动")
            # Pass a full target so legacy defaults cannot turn missing joints into zero.
            target = {n: float(joints.get(n, current[n])) for n in self.arm().ARM_JOINTS}
            ok = self.joint().set_joints(target)
        else:
            raise ValueError(f"未知机械臂动作 {action}")
        return ToolResult.success(f"{action} 已完成") if ok else ToolResult.failure("MOTION_FAILED", f"{action} 执行失败")

    def close(self):
        # Release each node without an individual client shutting down other clients' context.
        for client in self.clients.values():
            client._running = False
            client._spin.join(timeout=1)
            client.executor.shutdown(timeout_sec=1)
            client.node.destroy_node()
        if self.clients:
            import rclpy
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    backend = ArmBackend(options())
    serve(backend, backend.close)
