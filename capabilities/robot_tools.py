"""Robot lifecycle tools."""
from tools import registry
from services import lifecycle
from core.results import ToolResult


@registry.register(description="启动机械臂 ROS 服务；已有服务可复用，不杀死外部进程。")
def start_robot() -> ToolResult:
    return lifecycle.robot.start()


@registry.register(description="停止本程序拥有的机械臂服务和连接，外部启动的服务保留。")
def stop_robot() -> ToolResult:
    return lifecycle.stop_robot()


@registry.register(description="查询机械臂真实使用的 IK 和笛卡尔服务是否就绪。")
def robot_status() -> ToolResult:
    return lifecycle.robot.status()
