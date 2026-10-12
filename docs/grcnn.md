# GR-CNN 部署与测试

使用作者 [robotic-grasping](https://github.com/skumra/robotic-grasping) 仓库的 GR-ConvNet v3，RGB-D 四通道输入，Cornell 权重 `cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98`。固定源码版本 `183c6f68c44c1c7ff0f07707e2db6fcfd6840d2d`，下载后校验权重 SHA256。文件名中的 0.98 是作者的数据集指标，不代表本机抓取成功率。

## 部署

```bash
python3 scripts/setup_grcnn.py
GRASP_BACKEND=grcnn python3 main.py
```

模型和作者源码保存在 `.runtime/grcnn/`，保留上游许可证，不加入 Git。现有 `VISION_PYTHON` 环境已包含推理依赖；其他机器可在该环境安装 `requirements-grcnn.txt`，并补齐现有 ROS 和碰撞检测依赖。首次转换仅加载已校验的作者权重，后续 worker 用 `weights_only=True` 读取张量。`source.json` 记录来源、校验值及 Python 路径。

在 `.env` 中设置 `GRASP_BACKEND=grcnn` 可作为本机默认；`GRASP_BACKEND=graspnet python3 main.py` 可临时切回原模型。`GRCNN_DEVICE=auto` 有 CUDA 就使用 GPU，否则使用 CPU；也可指定 `cpu` 或 `cuda`。源码和权重路径可用 `GRCNN_ROOT`、`GRCNN_CHECKPOINT` 覆盖。启动会显示后端名称，每次实际抓取会显示“使用 GR-CNN pose”或兜底。

## 预测与执行

模型预测目标附近裁剪图像的中心像素、图像平面角度和夹爪宽度。输入为 224×224，裁剪保留目标周围背景，最小边长 112 像素，对应作者 Cornell 训练的缩放范围；记录裁剪原点和比例，正确还原原图坐标。RGB 使用作者的均值归一化，深度以米为单位去均值，网络输入的缺失深度可以插值；执行坐标始终来自目标掩膜内的原始有效深度。宽度按训练标签的 `224/2` 反归一化，修正上游示例硬编码 150 的尺寸不一致。角度在双角正余弦上平滑，避免 ±90° 接缝变成 0°。

相机射线与目标高度的水平面相交，得到基座下夹爪开合方向及米制宽度，进近方向为基座竖直向下。平行夹爪对称归一化 roll，按原装指尖偏移转换到 `ee_gripper_link`。抓取插入深度固定 1 cm；这是本机平面抓取适配参数，仍需实机验证。它不是任意方向的六自由度模型，不承诺在所有低矮物体上优于 GraspNet。

沿用作者的目标分数下限 0.2 和局部峰值去重；`GRASP_SCORE_MIN` 可进一步提高下限。候选宽度不能超过当前张爪配置的 0.074 m；复用原有点云碰撞及空抓检测，输入仍是完整有效场景。没有额外桌面拟合、STL 扫掠、八候选 MoveIt 预检或关节到位等待，也没有 GraspNet 的绝对 3 cm 高度门槛。诊断包含峰值数、开口/几何排除数、碰撞排除数和耗时。

最终 pose 进入现有工作流：上方进近 → 直线下降 → 闭爪 → 提到参考高度上方 8 cm → 完整路径规划搬运到目标上方 10 cm → 直线下降到上方 4 cm → 松爪成功 → 等待 0.5 秒 → 直线上升到上方 10 cm。模型成功但零候选时沿用原兜底；加载或推理异常直接报错。IK 或执行失败不切换到兜底。

## 测试

启动后可先输入“扫描辣椒和胡萝卜，显示抓取框，先不要抓放”。预扫前会收拢让出视野，随后 RViz 显示分割点云和候选；不会执行抓放。确认框的位置后再输入抓放任务，重新扫描当前场景。

已有场景可只做回放，不启动设备或发运动命令：

```bash
GRASP_BACKEND=grcnn python3 scripts/replay_grasps.py \
  /tmp/my_agent_scenes/<scene>/plan.json \
  --output scratch/grcnn.json --preview scratch/grcnn.png
```

回放输出标记为仅供查看。两种后端沿用 `/graspnet/pointcloud`、`/graspnet/grasps`、`/graspnet/selected_grasp`，已有 RViz 配置直接可用。

可用本机保存的场景比较不同模型，诊断结果与场景数据留在本机。首次推理包含权重和模块加载；检测、分割、规划及机械臂运动也各自占用时间，应分别测量。回放不是实机夹持验证，最终效果需用当前物体和硬件测试。
