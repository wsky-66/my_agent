# 抓取流程与 RViz

抓取沿用重构前的两条路径，共用一次 RGB-D 预扫和常驻 GraspNet 模型。

1. 检测、分割、定位目标和放置位置，在同一场景运行原版 GraspNet 推理。仍用原版的模型碰撞、NMS、分数和高度过滤；取消后来增加的桌面拟合、STL 凸包、进近/闭合扫掠和多区域角度深度搜索。
2. **有抓取 pose**：取最高分候选，用平行夹爪对称性将 roll 调到 −90°~90°，保持进近轴不变。GraspNet 接触参考点按原装夹指的指尖偏移 15.575 mm 和模型 depth 转为 ee_gripper_link 目标。MoveIt 先将末端移到目标上方 10 cm，再直线降到该目标，闭合后抬到目标上方 15 cm。
3. **无抓取 pose**：默认恢复原版兜底，使用物体定位坐标和固定姿态 `[0°,50°,0°]`。先到定位高度上方 10 cm，直线降到定位高度下方 3 cm，闭合后抬到定位高度上方 15 cm。夹爪沿用原版张开 0.037 m、闭合 0.024 m。
4. 下降、抬起和放置时保持当前实际末端朝向，不重复发送初始目标 RPY。放置时直线移到目标上方 20 cm，降到目标上方 4 cm，张开，控制器报告松爪成功后等待 0.5 秒，再抬到上方 20 cm。等待可以取消；松爪失败或取消后不再上升。抓取前张爪和抓取闭爪不增加等待，也不增加夹爪开口容差或稳定反馈检查。多物体只在全部完成后收拢。

工作流不再预检八个候选，也不重复计算下降路径。有 pose 但 IK/运动失败时停止；只在没有 pose 时兜底。`allow_fallback` 默认 true，明确禁用时可设为 false。结果中的 `strategies` 提供实际使用的 graspnet / fallback。

## 邻近逆解与运动

旧版有朝向的移动请求没有传入实际关节初值，拿到单个逆解后直接发给关节客户端。当前使用近期实际反馈、邻近初值求解；附近无解时额外尝试中立和伸展初值。初值只用于求解，不是机械臂命令。IK 接口不提前添加转角约束，返回后选择限位内、与实际关节变化较小的分支，再由 MoveIt 规划。

执行前检查驱动物理限位、轨迹起终点、时间和速度。取消 2.6 rad 单次关节转角限制，以及 `1.5 × |终点−起点| + 0.35` 累计绕路阈值和规划附加关节约束；累计变化只记录，不拦截。累计转角是逐点绝对变化总和，不等于关节净转角或旋转圈数。不会直接将绝对关节反馈取模；有朝向的移动仍先选较近的逆解，再通过 MoveIt 规划和控制器执行。有效路径速度过高时统一延长时间；控制器报告成功就继续，下一动作只读一次当前关节位置用于规划；取消残余速度判忙及规划后关节变化复查，无额外关节到位容差、多帧稳定等待或末端误差复查。退出时的 Sleep 确认沿用退出流程。

安装的 MoveIt 2.5.9 已为笛卡尔路径完成计时（[官方服务源码](https://raw.githubusercontent.com/moveit/moveit2/2.5.9/moveit_ros/move_group/src/default_capabilities/cartesian_path_service_capability.cpp)）。下降、提起和放置直接沿用这份计时，仅在超过驱动速度目标时统一减速，不再无条件把时长乘 4。`ARM_VELOCITY_SCALING` 默认 0.25，表示驱动速度目标比例。日志的 `original_duration` 是 MoveIt 原始时长，`time_scale` 是实际减速倍数。按已保存的同一路径估算，12.31 秒下降和 13.35 秒提起可分别缩短至约 3.9 秒和 4.25 秒；这是同轨迹的计时估算，尚未重新实机执行。

## RViz

```bash
source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/setup.bash
rviz2 -d /home/wsky/my_agent/configs/grasp.rviz
```

仓库配置使用 `wx250s/base_link` 为 Fixed Frame，包含机械臂和这些显示：

| 话题 | 内容 |
| --- | --- |
| `/graspnet/pointcloud` | 本次 RGB-D 转换后的基座坐标系彩色点云 |
| `/graspnet/grasps` | 原版 GraspNet 返回的候选，绿色夹爪轮廓、进近箭头、标签与分数 |
| `/graspnet/selected_grasp` | 金色所选 GraspNet pose；蓝色固定姿态兜底目标 |
| `/move_group/display_planned_path` | MoveIt 返回的规划轨迹 |

发布者随对应 worker 常驻，使用 Transient Local 保留最后一条消息；后打开 RViz 也能看到最近的结果。进近箭头沿 GraspNet 的局部 X 轴，夹爪轮廓使用与 GraspNet 展示一致的四块几何，避免把夹持内部画成实心碰撞盒。新场景替换候选 marker，新选择替换所选目标 marker；所选目标会保留到下次选择。退出 worker 后销毁发布者和节点。

## 离线复现

```bash
python3 scripts/replay_grasps.py /tmp/my_agent_scenes/<scene>/plan.json --output scratch/replay.json
```

回放只启动推理和可视化，不启动设备或发送运动。输出候选标记为仅供查看；应重新扫描后执行。模拟验证使用 `scripts/check_motion_ros.py` 的隔离 ROS 域，不连接实际机械臂。
