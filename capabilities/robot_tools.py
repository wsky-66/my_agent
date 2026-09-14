"""机器人进程生命周期工具：让 LLM 也能自行拉起/关闭机器人。"""

from tools import registry
from . import robot


@registry.register(
    name="start_robot",
    description="拉起并启动机械臂 ROS 进程（含 RViz）。适合在开始机械臂任务前调用。会先清理残留进程再干净启动，服务就绪后返回。",
)
def start_robot() -> str:
    return robot.start_robot()


@registry.register(
    name="stop_robot",
    description="关闭机械臂 ROS 进程并清理相关节点。适合在机械臂任务结束、待机或退出前调用。",
)
def stop_robot() -> str:
    robot.stop_robot()
    return "机器人已停止。"


@registry.register(
    name="robot_status",
    description="查询机械臂 ROS 进程是否在运行。",
)
def robot_status() -> str:
    return robot.robot_status()
