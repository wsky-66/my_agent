# my_agent

基于 LLM 的命令行 Agent，支持工具调用（Shell 执行、文件读写、YOLO-World 目标检测）。

## 环境要求

- Python 3.10+
- Conda（YOLO-World 工具需要 `yoloworld` 环境）

## 快速开始

```bash
git clone https://github.com/wsky-66/my_agent.git
cd my_agent

# 配置 API Key
cp .env.example .env
# 编辑 .env 填入你的 LLM_API_KEY

# 安装依赖
pip install -r requirements.txt

# 运行
python3 main.py
```

## 配置

`.env` 文件：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_API_KEY` | API 密钥 | - |
| `LLM_BASE_URL` | API 端点 | `https://api.openai.com/v1` |
| `LLM_MODEL` | 模型名称 | `gpt-4o` |
| `LLM_MAX_TOOL_ROUNDS` | 最大工具调用轮数 | `10` |
| `SHELL_TIMEOUT` | Shell 命令超时(秒) | `30` |

## 内置工具

| 工具 | 功能 |
|------|------|
| `run_shell` | 执行 Shell 命令 |
| `read_file` | 读取文件内容 |
| `write_file` | 写入文件 |
| `detect_objects` | YOLO-World 目标检测 |

## YOLO-World 目标检测

Agent 会根据任务自动判断是否调用检测工具。

**要求**：需要 Conda 环境名为 `yoloworld`，路径为 `/home/wsky/miniconda3/envs/yoloworld`，并安装 `ultralytics`。

**权重文件**：需下载以下权重放入对应目录：

```
/home/wsky/yolo/
├── yolov8s-world.pt           # YOLO-World 模型权重
└── weights/clip/
    └── ViT-B-32.pt            # CLIP 文本编码器权重
```

CLIP 权重也可放在 `~/.cache/clip/ViT-B-32.pt`。

**使用示例**：

```
你: 帮我看一下 /path/to/image.jpg 里有什么
Agent: [调用 detect_objects 检测...]
  检测结果: car(0.63), bicycle(0.94), person(0.87)
```

## 扩展自定义工具

参考 `examples.py`，使用 `@registry.register` 装饰器即可注册新工具：

```python
from tools import registry

@registry.register(description="计算两个数的和")
def add(a: int, b: int) -> str:
    return str(a + b)
```

然后在 `main.py` 中 `import` 你的工具模块即可。

## 对话命令

| 命令 | 说明 |
|------|------|
| `/help` | 显示帮助 |
| `/tools` | 列出所有工具 |
| `/reset` | 重置对话历史 |
| `/exit` | 退出 |

## 项目结构

```
my_agent/
├── main.py          # 入口，交互式对话
├── agent.py         # LLM Agent 核心逻辑
├── config.py        # 配置管理
├── tools.py         # 工具注册与实现
├── examples.py      # 自定义工具示例
├── requirements.txt # Python 依赖
├── .env.example     # 环境变量模板
└── weights/         # 权重文件目录（需自行下载）
```
