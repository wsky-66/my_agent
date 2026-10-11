"""Persistent ROS clients, with one ROS context owned by this worker."""
import math
import sys
import time

from core.results import ToolResult
from core.health import fresh_joints
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
            class FreshJointClient(joint_client.JointClient):
                def _on_state(self, msg):
                    self._feedback = time.monotonic(), msg
                    super()._on_state(msg)

                def get_joints(self, timeout=3):
                    deadline = time.monotonic() + timeout
                    while time.monotonic() < deadline:
                        sample = getattr(self, "_feedback", None)
                        if fresh_joints(sample, time.monotonic()):
                            positions = dict(zip(sample[1].name, sample[1].position))
                            return {n: positions[n] for n in joint_client.ARM_JOINTS}
                        time.sleep(0.05)
                    if getattr(self, "_require_feedback", False):
                        raise RuntimeError("近期完整关节反馈丢失，停止发送轨迹")
                    return None

                def set_joints(self, joints):
                    # Legacy code substitutes zero when feedback is missing.
                    self._require_feedback = True
                    try:
                        return super().set_joints(joints)
                    finally:
                        self._require_feedback = False

            self.clients["joint"] = FreshJointClient(self.opts["robot_name"])
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

    def ik_move(self, xyz, rpy, cartesian=False):
        """Choose a nearby bounded IK branch, then plan and validate a full path."""
        if not hasattr(self, "_pose_motion"):
            from workers.pose_motion import PoseMotion
            self._pose_motion = PoseMotion(self)
        return (self._pose_motion.move_cartesian(xyz,rpy) if cartesian else self._pose_motion.move(xyz, rpy))

    def __call__(self, action, args):
        if action == "inspect_grasps":
            candidates = args.get("candidates")
            if not isinstance(candidates, list) or not 1<=len(candidates)<=12:
                raise ValueError("预检候选需为 1~12 个")
            for candidate in candidates:
                for key in ("pre_position", "position", "rpy"):
                    v = candidate.get(key)
                    if not isinstance(v,list) or len(v)!=3 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in v):
                        raise ValueError("预检位姿需为三个有限数字")
                if not .015<=candidate.get("opening",0)<=.037:
                    raise ValueError("预检夹爪位置无效")
            if not hasattr(self, "_pose_motion"):
                from workers.pose_motion import PoseMotion
                self._pose_motion = PoseMotion(self)
            return self._pose_motion.inspect_grasps(candidates)
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
                motion = self.ik_move(xyz, rpy, cartesian=True) if args.get("cartesian") else self.ik_move(xyz, rpy)
                if not motion.ok:
                    return motion
                return ToolResult.success(motion.message, {"motion": motion.data})
            else:
                if args.get("cartesian"):
                    raise ValueError("抓放的受检笛卡尔移动必须给出完整朝向")
                ok, msg = self.arm()._move_group_move(*xyz)
            if not ok:
                return ToolResult.failure("MOTION_FAILED", msg)
            return ToolResult.success("控制器报告移动执行完成", {"requested_position": xyz, "controller_completed": True})
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
        if hasattr(self, "_pose_motion"):
            self._pose_motion.close()
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
