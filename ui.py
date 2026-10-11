"""Responsive terminal presentation with clear task steps and structured results."""
import json
import os
import re
import shutil
import sys
import threading
import time
import unicodedata
from urllib.parse import urlsplit

from core.results import ToolResult

RESET = "\033[0m"
BOLD, DIM = 1, 2
RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = 91, 92, 93, 94, 95, 96, 97
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_cursor_open = False
_reply_open = False
_LOCK = threading.RLock()
LABELS = {"execute_pick_place": "执行抓放任务", "scan_objects": "预扫场景", "move_arm": "移动末端",
          "gripper_set": "调整夹爪", "go_sleep": "收拢机械臂", "go_home": "机械臂复位",
          "recognize_camera": "理解画面", "detect_grounding": "定位物体", "segment_mask": "分割物体",
          "detect_grasp": "推理抓取姿态", "object_position": "计算三维位置"}


def color_enabled():
    return not os.getenv("NO_COLOR") and os.getenv("TERM") != "dumb" and (sys.stdout.isatty() or os.getenv("FORCE_COLOR") == "1")


def c(text, *codes):
    return f"\033[{';'.join(map(str, codes))}m{text}{RESET}" if codes and color_enabled() else str(text)


def dim(text):
    return c(text, DIM)


def _disp_width(text):
    return sum(0 if unicodedata.combining(ch) else 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
               for ch in _ANSI.sub("", text))


def width():
    return max(12, min(88, shutil.get_terminal_size((92, 24)).columns-4))


def _truncate(text, limit):
    plain = _ANSI.sub("", str(text))
    if _disp_width(plain) <= limit:
        return plain
    result = ""
    for ch in plain:
        if _disp_width(result+ch) > max(0, limit-1):
            break
        result += ch
    return result + "…"


def _pad(text, limit):
    return text + " "*max(0, limit-_disp_width(text))


def _wrap(text, limit):
    for line in str(text).splitlines() or [""]:
        chunk = ""
        for ch in line:
            if _disp_width(chunk+ch) > limit and chunk:
                yield chunk
                chunk = ""
            chunk += ch
        yield chunk


def _ensure_newline():
    global _cursor_open
    if _cursor_open:
        print()
        _cursor_open = False


def _panel(title, rows, accent=CYAN):
    _ensure_newline()
    size = width()
    inner = size-4
    heading = _truncate(" " + title + " ", size-2)
    print("  " + c("╭" + heading + "─"*max(0, size-2-_disp_width(heading)) + "╮", accent))
    for row in rows:
        for line in _wrap(row, inner):
            print("  " + c("│", accent) + " " + _pad(line, inner) + " " + c("│", accent))
    print("  " + c("╰" + "─"*(size-2) + "╯", accent))


def banner(model, base_url, tool_count):
    print()
    _panel("ww  /  ROBOT WORKSPACE", ["自然语言 · 视觉理解 · 机械臂",
            "", "模型  " + model, "服务  " + (urlsplit(base_url).hostname or base_url),
            f"工具  {tool_count} 项能力", "", "输入文字开始任务  ·  Tab 说话  ·  /help 查看帮助"])
    print()


def input_prompt():
    return c("  你 › ", CYAN, BOLD)


def content(text):
    global _cursor_open, _reply_open
    with _LOCK:
        if not _reply_open:
            _ensure_newline()
            sys.stdout.write(c("  ww › ", CYAN, BOLD))
            _reply_open = True
        sys.stdout.write(text)
        sys.stdout.flush()
        _cursor_open = not text.endswith("\n")


def response(text):
    content(text)
    _ensure_newline()


def _args(args):
    if not args:
        return ""
    return _truncate(json.dumps(args, ensure_ascii=False, separators=(",", ":")), max(8, width()-8))


def tool_call(name, args, step=None):
    global _reply_open
    _ensure_newline()
    _reply_open = False
    head = f"{step:02d}" if step else "•"
    label = _truncate(LABELS.get(name, name), max(1, width()-_disp_width(head)-2))
    remaining = width()-_disp_width(head)-2-_disp_width(label)
    detail = _truncate("  / " + name, remaining) if remaining > 4 else ""
    print("  " + c(head, CYAN, BOLD) + "  " + c(label, WHITE, BOLD) + dim(detail))
    if args:
        print(dim("      " + _args(args)))


def tool_result(result, elapsed=None):
    _ensure_newline()
    if not isinstance(result, ToolResult):
        result = ToolResult.success(str(result))
    elapsed = result.elapsed if elapsed is None else elapsed
    status = "完成" if result.ok else "停止"
    print("      " + c("✓ " if result.ok else "× ", GREEN if result.ok else RED)
          + c(status, GREEN if result.ok else RED) + dim(f" · {elapsed:.2f}s")
          + "  " + _truncate(result.message, max(8, width()-22)))
    if not result.ok:
        print(dim("      " + result.error_code))
    data = result.data
    if isinstance(data, dict):
        if "objects" in data and isinstance(data["objects"], dict):
            for label, value in data["objects"].items():
                xyz = value.get("base_xyz")
                detail = "(" + ", ".join(f"{v:.3f}" for v in xyz) + ") m" if xyz else value.get("error", "未定位")
                print(dim("      " + _truncate(label + "  " + detail, width()-6)))
        elif "position" in data and isinstance(data["position"], list):
            print(dim("      位置  " + ", ".join(f"{v:.3f}" for v in data["position"]) + " m"))
        elif "stdout" in data or "content" in data:
            text = data.get("stdout", data.get("content", ""))
            for line in list(_wrap(text, max(8, width()-8)))[:12]:
                print(dim("      " + line))
        elif "path" in data or "image_path" in data:
            print(dim("      " + _truncate(data.get("path", data.get("image_path", "")), width()-8)))


def workflow_step(name, args, result=None):
    _ensure_newline()
    if result is None:
        title = LABELS.get(name, name)
        remaining = max(0, width()-9-_disp_width(title))
        detail = _truncate(json.dumps(args, ensure_ascii=False, separators=(",", ":")), remaining) if args and remaining else ""
        print(dim("      ├ ") + c(title, WHITE) + dim("  " + detail))
    else:
        print(dim("      │ ") + c("✓" if result.ok else "×", GREEN if result.ok else RED)
              + dim(f" {result.elapsed:.2f}s  ") + _truncate(result.message, width()-20))


def turn_rule(elapsed=None, tool_count=None):
    global _reply_open
    _ensure_newline()
    _reply_open = False
    suffix = _truncate(f"  {elapsed:.1f}s · {tool_count} 次工具调用", width()) if elapsed is not None else ""
    print("  " + dim("─"*max(0, width()-_disp_width(suffix)) + suffix))
    print()


def notice(message, ok=True):
    _ensure_newline()
    print("  " + c("•", GREEN if ok else YELLOW) + " " + str(message))


def error(message):
    _ensure_newline()
    _panel("任务停止", [str(message)], RED)


def help_text():
    _panel("命令与快捷键", [
        "/help      查看帮助", "/tools     查看模型可用工具", "/status    查看服务状态",
        "/metrics   查看最近任务的阶段耗时", "/image 路径 [问题]   分析图片",
        "/robot start|stop|status", "/camera start|stop|status",
        "/voice on|off|status", "/reset     清除对话和场景缓存", "/exit      退出",
        "", "Tab 开始说话，停顿自动结束；Ctrl-C 停止后续步骤。"])


def tools_list(tool_items):
    items = list(tool_items)
    _panel(f"可用工具 · {len(items)}", [f"{name}  {desc}" for name, desc in items])


class Spinner:
    CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, text):
        self.text, self._stop = text, threading.Event()
        self._thread = None

    def start(self):
        if not sys.stdout.isatty():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        started, i = time.monotonic(), 0
        while not self._stop.wait(0.08):
            with _LOCK:
                sys.stdout.write("\r\033[2K  " + c(self.CHARS[i % len(self.CHARS)], CYAN)
                                 + " " + _truncate(self.text, width()-12) + dim(f"  {time.monotonic()-started:.1f}s"))
                sys.stdout.flush()
            i += 1

    def stop(self):
        self._stop.set()
        with _LOCK:
            thread, self._thread = self._thread, None
        if thread:
            thread.join(timeout=1)
            with _LOCK:
                # Repeated stops must not erase text streamed after the spinner.
                sys.stdout.write("\r\033[2K")
                sys.stdout.flush()
