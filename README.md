# my_agent

基于 LLM 的命令行 Agent，支持工具调用与多模态视觉理解，可操控 Interbotix 机械臂与 RealSense 相机。

## 环境要求

- Python 3.10+（Conda 环境 `my_agent`）
- ROS 2 Humble（机械臂 / 相机子进程需要）
- Interbotix `wx250s` 机械臂 + RealSense D435 相机（可选）

## 快速开始

```bash
cd my_agent

# 配置 API Key
cp .env.example .env
# 编辑 .env 填入 LLM_API_KEY（deepseek / openai 均可）

pip install -r requirements.txt

# 运行（自动拉起机器人 + 相机，退出时一并关闭）
python3 main.py
```

## 配置（.env）

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_API_KEY` | API 密钥 | - |
| `LLM_BASE_URL` | API 端点 | `https://api.openai.com/v1` |
| `LLM_MODEL` | 模型名称 | `gpt-4o` |
| `LLM_MAX_TOOL_ROUNDS` | 最大工具调用轮数 | `10` |
| `SHELL_TIMEOUT` | Shell 命令超时(秒) | `30` |
| `DDS_API_TOKEN` | DDS 算法平台 Token（Grounding DINO 开放集检测用） | - |
| `DDS_BASE_URL` | DDS 平台 API 端点 | `https://api.deepdataspace.com` |
| `ARM_AUTO_START` | 启动时自动拉起机器人 | `1` |
| `CAM_AUTO_START` | 启动时自动拉起相机 | `1` |

## 工具概览

| 类别 | 工具 |
|------|------|
| 基础 | `run_shell` `read_file` `write_file` |
| 机械臂 | `move_arm` `get_arm_pose` `set_joint(s)` `get_joints` `go_home` `go_sleep` |
| 夹爪 | `gripper_open` `gripper_close` `gripper_set` `gripper_grasp` |
| 视觉理解 (多模态) | `recognize_camera` |
| 开放集检测 | `detect_grounding` |
| 生命周期 | `start_robot` `stop_robot` `robot_status` `start_camera` `stop_camera` `camera_status` |

- `recognize_camera` 订阅相机话题抓帧后交给 `LLM_MODEL`（视觉模型）描述画面。
- `detect_grounding` 调 DDS 算法平台（`api.deepdataspace.com`）的 Grounding DINO 云端 API，按英文文本 prompt（多类别以 `.` 分隔）做开放集物体定位，需在 `.env` 配置 `DDS_API_TOKEN`。可传 `image_path` 检测指定图，不传则抓相机当前帧。
- 机器人/相机进程由 shell 拉起、按名 kill 关闭，agent 退出或 `SIGTERM` 时自动清理。
- 相机默认常驻、由你手动开关；agent 的检测工具只**订阅**话题。

## 对话命令

| 命令 | 说明 |
|------|------|
| `/help` | 显示帮助 |
| `/tools` | 列出所有工具 |
| `/image <路径> [问题]` | 发送图片给视觉模型 |
| `/robot start\|stop\|status` | 控制机器人进程 |
| `/camera start\|stop\|status` | 控制相机进程 |
| `/reset` | 重置对话历史 |
| `/exit` | 退出 |

## 项目结构

```
my_agent/
├── main.py             # 入口：交互式对话、命令、生命周期
├── agent.py            # LLM 交互与工具编排（事件流）
├── config.py           # 配置管理
├── tools.py            # 工具注册中心 + 内置 shell/file 工具
├── ui.py               # 终端界面渲染
├── capabilities/       # 能力模块（导入即注册全部工具）
│   ├── arm_tools.py    # 机械臂 + 夹爪控制
│   ├── robot.py        # 机器人进程生命周期
│   ├── robot_tools.py  # 机器人启停工具
│   ├── camera.py       # 相机进程生命周期
│   ├── camera_tools.py # 相机启停 + 多模态识别工具
│   └── grounding_dino_tools.py  # Grounding DINO 云端开放集检测
├── scratch/            # 临时草稿脚本（不参与主流程）
├── weights/            # 模型权重
├── requirements.txt
└── .env
```

## 扩展自定义工具

在 `tools.py`（或任意模块）用 `@registry.register` 注册：

```python
from tools import registry

@registry.register(description="计算两个数的和")
def add(a: int, b: int) -> str:
    return str(a + b)
```

然后在启动时导入该模块（`main.py` 或 `capabilities/__init__.py`）。
