#!/usr/bin/env python3
"""CLI composition root: configuration, capabilities, presentation and owned cleanup."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

from config import config
from core.results import ToolResult
from tools import registry
import ui


def _split_image_arg(rest):
    rest = rest.strip()
    if not rest:
        raise ValueError("用法：/image 图片路径 [问题]")
    if rest[0] in ('\"', "'"):
        quote = rest[0]
        end = rest.find(quote, 1)
        if end < 0:
            raise ValueError("图片路径的引号未闭合")
        return rest[1:end], rest[end+1:].strip()
    for i in range(len(rest), 0, -1):
        if (i == len(rest) or rest[i] == " ") and Path(rest[:i]).expanduser().is_file():
            return rest[:i], rest[i:].strip()
    parts = rest.split(maxsplit=1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def run_turn(agent, user_input, image_path=None, voice=None):
    started, count = time.monotonic(), 0
    spinner = ui.Spinner("理解任务")
    spinner.start()
    stream = agent.stream(user_input, image_path)
    streamed = False
    try:
        for event in stream:
            if event.type == "content":
                spinner.stop()
                ui.content(event.text)
                streamed = True
            elif event.type == "tool_call":
                spinner.stop()
                count += 1
                ui.tool_call(event.name, event.args, count)
                spinner = ui.Spinner(ui.LABELS.get(event.name, event.name))
                spinner.start()
                streamed = False
            elif event.type == "workflow_step":
                spinner.stop()
                ui.workflow_step(event.name, event.args, event.result)
                spinner = ui.Spinner(ui.LABELS.get(event.name, event.name) if event.phase == "start" else "继续任务")
                spinner.start()
            elif event.type == "tool_result":
                spinner.stop()
                ui.tool_result(event.result)
                spinner = ui.Spinner("整理结果")
                spinner.start()
            elif event.type == "response":
                spinner.stop()
                if not streamed:
                    ui.response(event.text)
                if voice:
                    voice.speak_async(event.text)
    except KeyboardInterrupt:
        spinner.stop()
        ui.notice("任务中断，停止后续步骤；已发送的机械臂动作需要确认当前状态。", False)
    except Exception as exc:
        spinner.stop()
        ui.error(str(exc))
        from logger import error
        error("对话回合异常", exc)
    finally:
        stream.close()
        spinner.stop()
        ui.turn_rule(time.monotonic()-started, count)


def _metrics():
    path = config.ROOT / "logs/timings.jsonl"
    if not path.exists():
        ui.notice("还没有任务耗时记录")
        return
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            values.append(json.loads(line))
        except ValueError:
            pass
    latest = next((v["turn_id"] for v in reversed(values) if v.get("turn_id")), None)
    totals = {}
    for value in values:
        if value.get("turn_id") == latest and latest:
            stage = value["stage"]
            totals[stage] = totals.get(stage, 0)+value["elapsed"]
    ui._panel("最近任务 · 阶段耗时（部分阶段包含子阶段）", [f"{stage}  {elapsed:.2f}s" for stage, elapsed in totals.items()] or ["还没有完整任务记录"])


def demo():
    ui.banner("Vision + Tools", "https://api.example.invalid/v1", 30)
    ui.notice("界面演示")
    print(ui.input_prompt() + "把苹果放进碗里")
    ui.tool_call("execute_pick_place", {"actions": [{"object": "apple", "target": "bowl"}]}, 1)
    ui.workflow_step("scan_objects", {"labels": ["apple", "bowl"]})
    ui.workflow_step("scan_objects", {}, ToolResult.success("2 个物体已定位；1 个物体需要抓取推理"))
    ui.workflow_step("move_arm", {"x": 0.25, "y": 0.08, "z": 0.16})
    ui.workflow_step("move_arm", {}, ToolResult.success("末端已到目标位置"))
    ui.workflow_step("gripper_set", {"position": 0.037})
    ui.workflow_step("gripper_set", {}, ToolResult.success("夹爪已张开"))
    ui.tool_result(ToolResult.success("抓放动作序列完成", {"completed": ["apple"]}, ), 8.42)
    ui.response("动作完成，苹果已放到碗的位置。")
    ui.turn_rule(9.6, 1)


def main():
    parser = argparse.ArgumentParser(description="ww · 自然语言机械臂助手")
    parser.add_argument("--demo", action="store_true", help="仅展示终端界面示例")
    parser.add_argument("--no-hardware", action="store_true", help="关闭机械臂、相机及相关工具")
    parser.add_argument("--text-only", action="store_true", help="关闭录音和语音播报")
    args = parser.parse_args()
    if args.demo:
        demo()
        return
    if args.no_hardware:
        config.ENABLE_ARM = config.ENABLE_CAMERA = config.ENABLE_VISION = False
    if args.text_only:
        config.ENABLE_VOICE = False
    try:
        config.validate()
    except ValueError as exc:
        ui.error(str(exc))
        return 1
    import capabilities
    capabilities.load_tools()
    from agent import Agent
    from services import lifecycle
    from services.scene import store
    voice = None
    if config.ENABLE_VOICE:
        from capabilities import voice
    try:
        agent = Agent()
    except (ValueError, ImportError) as exc:
        ui.error("模型客户端初始化失败：" + str(exc))
        if isinstance(exc, ImportError):
            ui.notice("请在当前环境运行 python3 -m pip install -r requirements.txt 补齐依赖", False)
        return 1
    try:
        lifecycle.begin_session()
    except RuntimeError as exc:
        ui.error(str(exc))
        return 1
    ui.banner(config.MODEL, config.BASE_URL, len(registry.list_tools()))
    if config.ENABLE_VISION:
        ui.notice("抓取后端：" + {"graspnet":"GraspNet", "grcnn":"GR-CNN"}[config.GRASP_BACKEND])
    exit_requested = False

    def on_term(signum, frame):
        nonlocal exit_requested
        exit_requested = True
        raise KeyboardInterrupt

    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, on_term)
    try:
        if config.ENABLE_ARM and config.ARM_AUTO_START:
            ui.notice("正在连接机械臂…")
            result = lifecycle.robot.start()
            ui.notice(result.message, result.ok)
        if config.ENABLE_CAMERA and config.CAM_AUTO_START:
            ui.notice("正在连接相机…")
            result = lifecycle.camera.start()
            ui.notice(result.message, result.ok)
        if voice:
            voice.start_async()
            ui.notice("语音后台加载中 · Tab 开始说话")
        while not exit_requested:
            try:
                text = voice.term_input(ui.input_prompt()) if voice else input(ui.input_prompt()).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text is None:
                break
            if not text.strip():
                continue
            text = text.strip()
            if not text.startswith("/"):
                run_turn(agent, text, voice=voice)
                continue
            parts = text.split(maxsplit=1)
            command, rest = parts[0].lower(), parts[1] if len(parts) > 1 else ""
            try:
                if command == "/exit":
                    break
                if command == "/help":
                    ui.help_text()
                elif command == "/tools":
                    ui.tools_list(registry.describe_tools())
                elif command == "/metrics":
                    _metrics()
                elif command == "/reset":
                    agent.reset()
                    store.clear()
                    ui.notice("对话和场景缓存已清除")
                elif command == "/image":
                    path, question = _split_image_arg(rest)
                    run_turn(agent, question or "请描述这张图片。", str(Path(path).expanduser()), voice)
                elif command in ("/robot", "/camera"):
                    enabled = config.ENABLE_ARM if command == "/robot" else config.ENABLE_CAMERA
                    if not enabled:
                        ui.notice("该能力已关闭", False)
                        continue
                    device = lifecycle.robot if command == "/robot" else lifecycle.camera
                    if rest in ("", "status"):
                        result = device.status()
                    elif rest == "start":
                        result = device.start()
                    elif rest == "stop":
                        result = lifecycle.stop_robot() if command == "/robot" else lifecycle.stop_camera()
                    else:
                        raise ValueError("用法：" + command + " start|stop|status")
                    ui.tool_result(result)
                elif command == "/status":
                    if config.ENABLE_ARM:
                        ui.tool_result(lifecycle.robot.status())
                    if config.ENABLE_CAMERA:
                        ui.tool_result(lifecycle.camera.status())
                    ui.notice(voice.status() if voice else "文字模式")
                elif command == "/voice":
                    if voice is None:
                        ui.notice("语音能力已关闭；启用 ENABLE_VOICE 并取消 --text-only 后可用", False)
                    elif rest == "on":
                        voice.start_async()
                        ui.notice("语音后台加载中")
                    elif rest == "off":
                        voice.stop()
                        ui.notice(voice.status())
                    elif rest in ("", "status"):
                        ui.notice(voice.status())
                    else:
                        raise ValueError("用法：/voice on|off|status")
                else:
                    ui.notice("未知命令，输入 /help 查看帮助", False)
            except Exception as exc:
                ui.error(str(exc))
    except KeyboardInterrupt:
        pass
    finally:
        # A second interrupt aborts parking but still allows process cleanup to finish.
        def interrupt_cleanup(signum, frame):
            lifecycle.abort_parking()

        signal.signal(signal.SIGINT, interrupt_cleanup)
        signal.signal(signal.SIGTERM, interrupt_cleanup)
        signal.signal(signal.SIGHUP, interrupt_cleanup)
        try:
            if voice:
                try:
                    voice.close()
                except (Exception, KeyboardInterrupt) as exc:
                    ui.notice(f"语音退出失败：{exc}，继续清理硬件", False)
        finally:
            result = lifecycle.close_all(notify=ui.notice)
            ui.notice(result.message, result.ok)
            ui.notice("会话结束")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
