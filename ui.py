"""终端界面渲染：颜色、横幅、加载动画、工具调用展示等。"""

import re
import sys
import threading
import time
import unicodedata

# ---- ANSI 颜色 ----
RESET = "\033[0m"
BOLD = 1
DIM = 2
RED = 91
GREEN = 92
YELLOW = 93
BLUE = 94
MAGENTA = 95
CYAN = 96
WHITE = 97

WIDTH = 62


def c(text: str, *codes) -> str:
    """给文本上色。codes 为 ANSI 码，例如 ui.c('hi', BOLD, BLUE)。"""
    if not codes:
        return text
    return f"\033[{';'.join(str(x) for x in codes)}m{text}{RESET}"


def dim(text: str) -> str:
    return c(text, DIM)


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _disp_width(text: str) -> int:
    """计算字符串显示的宽度（忽略颜色码，中日韩字符算 2 个）。"""
    width = 0
    for ch in _ANSI.sub("", text):
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            width += 2
        else:
            width += 1
    return width


def _pad(text: str, width: int) -> str:
    """按显示宽度向左补齐到 width。"""
    return text + " " * max(0, width - _disp_width(text))


def banner(model: str, base_url: str, tool_count: int) -> None:
    """打印程序启动横幅。"""
    inner = WIDTH - 2
    title = "  LLM Agent · 工具调用助手  "

    print()
    print(c("╔" + "═" * inner + "╗", CYAN, BOLD))
    print(c("║", CYAN, BOLD) + _pad(title, inner) + c("║", CYAN, BOLD))
    print(c("╠" + "═" * inner + "╣", CYAN, BOLD))
    info = [
        ("模型", model),
        ("服务", base_url),
        ("工具", f"{tool_count} 个"),
    ]
    for label, value in info:
        line = c(f"  {label}  ", BOLD) + value
        print(c("║", CYAN, BOLD) + _pad(line, inner) + c("║", CYAN, BOLD))
    print(c("╚" + "═" * inner + "╝", CYAN, BOLD))
    print()
    print(dim("  输入 /help 查看帮助，直接输入文字开始对话。"))
    print()


_cursor_open = False


def _ensure_newline() -> None:
    """若上一段输出还停留在行尾（未换行），先补一个换行。"""
    global _cursor_open
    if _cursor_open:
        sys.stdout.write("\n")
        _cursor_open = False


def prompt() -> str:
    """显示用户输入提示符，返回输入行。"""
    try:
        return input(c("\n你 › ", CYAN, BOLD)).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise


def turn_rule() -> None:
    """每回合结束的分隔线。"""
    _ensure_newline()
    print(dim("─") * WIDTH)


def content(text: str) -> None:
    """逐字流式输出模型回复内容。"""
    global _cursor_open
    sys.stdout.write(text)
    sys.stdout.flush()
    _cursor_open = text and not text.endswith("\n")


def response(text: str) -> None:
    """回合最终回复（非流式时用）。"""
    _ensure_newline()
    print(text)
    _cursor_open = False


def tool_call(name: str, args: dict, step: int = None):
    """展示工具调用（单行紧凑）。"""
    import json
    _ensure_newline()
    s = json.dumps(args, ensure_ascii=False)
    if WIDTH and len(s) > 56:
        s = s[:53] + "…"
    head = f"[{step}] " if step else ""
    print(c("  ⟩ ", MAGENTA) + c(head + name, WHITE, BOLD) + c("  " + s, DIM))
    _cursor_open = False


FAIL_WORDS = ("出错", "错误", "失败", "超时", "异常", "无法", "未找到", "解析失败")
MAX_RESULT_LINES = 60


def tool_result(result: str, elapsed: float) -> None:
    """展示工具执行结果（首行 + 若干行，超长截断）。"""
    _ensure_newline()
    lines = result.split("\n")
    first, *rest = lines
    ok = not any(w in result for w in FAIL_WORDS)
    col = GREEN if ok else RED
    print(c("  " + ("✓" if ok else "✗"), col, BOLD)
          + c(f" ({elapsed:.1f}s)", DIM) + c("  " + first, WHITE))
    for line in rest[:MAX_RESULT_LINES]:
        print(c("      ", DIM) + line)
    if len(rest) > MAX_RESULT_LINES:
        print(c("      ", DIM) + c(f"… 共 {len(rest)} 行", DIM))
    _cursor_open = False


class Spinner:
    """在工具执行期间显示加载动画。"""

    CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, text: str):
        self.text = text
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        i = 0
        while not self._stop.is_set():
            ch = c(self.CHARS[i % len(self.CHARS)], CYAN)
            sys.stdout.write("\r\x1b[K" + f"  {ch} {self.text}")
            sys.stdout.flush()
            i += 1
            time.sleep(0.08)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        sys.stdout.write("\r\x1b[K")
        sys.stdout.flush()


def error(message: str) -> None:
    print()
    print(c("  ✗ ", RED, BOLD) + c(message, RED))


def help_text() -> None:
    print()
    print(c("可用命令", CYAN, BOLD))
    print()
    rules = [
        ("/help", "显示此帮助信息"),
        ("/tools", "列出所有可用工具"),
        ("/image", "发送图片给视觉模型：/image <路径> [问题]"),
        ("/robot", "拉起的机器人：/robot start|stop|status"),
        ("/camera", "拉起的相机：/camera start|stop|status"),
        ("/reset", "重置对话历史"),
        ("/exit", "退出程序"),
    ]
    for cmd, desc in rules:
        print(c("  " + _pad(cmd, 10), YELLOW) + dim(desc))
    print()
    print(dim("  直接输入自然语言即可与 Agent 对话，Agent 会自动调用工具完成任务。"))
    print()


def tools_list(tool_items) -> None:
    print()
    print(c("可用工具", CYAN, BOLD))
    print()
    for name, desc in tool_items:
        print(c("  " + _pad("▸ " + name, 22), WHITE) + dim(desc))
    print()
    print(dim(f"  共 {len(tool_items)} 个工具"))
    print()
