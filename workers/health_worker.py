"""Read-only ROS health checks. Never construct motion clients or send goals."""
import os
from pathlib import Path
import time

from core.health import robot_readiness
from core.results import ToolResult
from workers.protocol import options, serve


class HealthBackend:
    def __init__(self, opts):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.action import ActionClient
        from sensor_msgs.msg import JointState
        from controller_manager_msgs.srv import ListControllers
        from interbotix_xs_msgs.srv import RobotInfo
        from moveit_msgs.srv import GetPositionIK, GetCartesianPath, GetMotionPlan
        from control_msgs.action import FollowJointTrajectory
        rclpy.init()
        self.node = Node("my_agent_health")
        self.robot_name = opts["robot_name"]
        prefix = "/" + self.robot_name
        self._sample = None
        self.node.create_subscription(JointState, prefix + "/joint_states", self._on_state, qos_profile_sensor_data)
        self.driver = self.node.create_client(RobotInfo, prefix + "/get_robot_info")
        self.controllers = self.node.create_client(ListControllers, prefix + "/controller_manager/list_controllers")
        self.planning = {"ik": self.node.create_client(GetPositionIK, "/compute_ik"),
                         "cartesian": self.node.create_client(GetCartesianPath, "/compute_cartesian_path"),
                         "joint_path": self.node.create_client(GetMotionPlan, "/plan_kinematic_path")}
        self.actions = {name: ActionClient(self.node, FollowJointTrajectory, prefix + f"/{name}/follow_joint_trajectory")
                        for name in ("arm_controller", "gripper_controller")}

    def _on_state(self, state):
        self._sample = time.monotonic(), state

    def serial(self):
        from ament_index_python.packages import get_package_share_directory
        import yaml
        path = Path(get_package_share_directory("interbotix_xsarm_control")) / "config" / f"{self.robot_name}.yaml"
        port = yaml.safe_load(path.read_text())["port"]
        data = {"ready": False, "port": port, "motor_config": str(path)}
        if not Path(port).exists():
            return ToolResult.failure("SERIAL_MISSING", f"机械臂串口 {port} 不存在，请检查电源、USB 连接及设备规则", data)
        if not os.access(port, os.R_OK | os.W_OK):
            return ToolResult.failure("SERIAL_PERMISSION", f"机械臂串口 {port} 无读写权限", data)
        return ToolResult.success("机械臂串口存在；仍需验证驱动与关节反馈", data)

    def robot(self, timeout=1.2):
        import rclpy
        from controller_manager_msgs.srv import ListControllers
        from interbotix_xs_msgs.srv import RobotInfo
        pending = {}
        responses = {}
        deadline = time.monotonic() + timeout
        result = robot_readiness(None, time.monotonic(), [], {}, {}, {})
        try:
            while time.monotonic() < deadline and rclpy.ok():
                for key, client in (("driver", self.driver), ("controllers", self.controllers)):
                    if key not in pending and client.service_is_ready():
                        if key == "driver":
                            request = RobotInfo.Request()
                            request.cmd_type, request.name = "group", "arm"
                        else:
                            request = ListControllers.Request()
                        pending[key] = client, client.call_async(request)
                    if key in pending and key not in responses and pending[key][1].done():
                        responses[key] = pending[key][1].result()
                result = robot_readiness(
                    self._sample, time.monotonic(),
                    list(responses["driver"].joint_names) if responses.get("driver") else [],
                    {c.name: c.state for c in responses["controllers"].controller} if responses.get("controllers") else {},
                    {n: a.server_is_ready() for n, a in self.actions.items()},
                    {n: c.service_is_ready() for n, c in self.planning.items()})
                if result.ok:
                    return result
                rclpy.spin_once(self.node, timeout_sec=min(0.05, max(0, deadline-time.monotonic())))
            return result
        finally:
            for client, future in pending.values():
                if not future.done():
                    client.remove_pending_request(future)

    def __call__(self, action, params):
        if action == "robot":
            return self.robot()
        if action == "serial":
            return self.serial()
        raise ValueError("未知健康检查")

    def close(self):
        import rclpy
        for action in self.actions.values():
            action.destroy()
        self.node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    backend = HealthBackend(options())
    serve(backend, backend.close)
