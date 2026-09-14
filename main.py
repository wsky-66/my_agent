#!/usr/bin/env python3
import atexit
import os
import signal
import sys
import threading
import time

import readline

from agent import Agent
from config import config
from tools import registry
import capabilities  # noqa: F811 - 注册全部能力工具
from capabilities import robot, camera, voice
from logger import error as log_error
import ui


def _on_term(signum, frame):
    """SIGTERM：关闭机器人、相机、语音后退出。"""
    robot.stop_robot()
    camera.stop_camera()
    voice.stop()
    sys.exit(0)


def stop_spinner(spinner: "ui.Spinner | None"):
    if spinner:
        spinner.stop()
    return None


def _split_image_arg(line: str):
    """把 '/image <路径> [问题]' 拆分为 (路径, 问题)。"""
    rest = line.strip()
    for i in range(len(rest), 0, -1):
        if i < len(rest) and rest[i] == " ":
            cand = rest[:i]
            if os.path.isfile(cand):
                return cand, rest[i + 1 :].strip()
    if os.path.isfile(rest):
        return rest, ""
    return rest.split()[0], " ".join(rest.split()[1:]).strip()


def run_turn(agent: Agent, user_input: str, image_path: str = None, tts: bool = False) -> None:
    """渲染一轮完整对话：思考 → 调用工具 → 结果 → 答复。tts=True 时用语音播报最终答复。"""
    spinner = ui.Spinner("思考中…")
    spinner.start()
    tool_start = 0.0
    step = 0
    streamed = False
    try:
        for event in agent.stream(user_input, image_path):
            if event.type == "content":
                spinner = stop_spinner(spinner)
                streamed = True
                ui.content(event.text)
            elif event.type == "tool_call":
                step += 1
                spinner = stop_spinner(spinner)
                ui.tool_call(event.name, event.args, step)
                spinner = ui.Spinner(f"调用 {event.name}…")
                spinner.start()
                tool_start = time.time()
            elif event.type == "tool_result":
                spinner = stop_spinner(spinner)
                ui.tool_result(event.result, elapsed=time.time() - tool_start)
            elif event.type == "response":
                spinner = stop_spinner(spinner)
                if not streamed:
                    ui.response(event.text)
                if tts and event.text.strip():
                    voice.speak(event.text)
    except (KeyboardInterrupt, EOFError):
        spinner = stop_spinner(spinner)
        print()
    except Exception as e:
        spinner = stop_spinner(spinner)
        ui.error(f"发生错误: {e}")
        log_error("对话回合异常", e)
    finally:
        ui.turn_rule()


def main() -> None:
    try:
        config.validate()
    except ValueError as e:
        print(f"配置错误: {e}")
        sys.exit(1)

    ui.banner(config.MODEL, config.BASE_URL, len(registry.list_tools()))
    if os.environ.get("ARM_AUTO_START", "1") != "0":
        print(ui.c("  ⟩ 正在拉起机器人…", ui.BLUE))
        print(ui.c("  ⟩ " + robot.start_robot(), ui.BLUE))
    if os.environ.get("CAM_AUTO_START", "1") != "0":
        print(ui.c("  ⟩ 正在拉起相机…", ui.BLUE))
        print(ui.c("  ⟩ " + camera.start_camera(), ui.BLUE))
    signal.signal(signal.SIGTERM, _on_term)
    # 仅 agent 主进程注册退出清理(正常退出也关机器人/相机/语音)。
    # 不要加在 capabilities 生命周期模块里，否则任何 import 该包的脚本(如 test/infer.py)
    # 一退出就会连带杀掉相机/机器人节点。
    atexit.register(robot.stop_robot)
    atexit.register(camera.stop_camera)
    atexit.register(voice.stop)
    agent = Agent()
    # 语音始终可用: 后台加载 ASR 服务(不阻塞终端), 回复始终语音播报
    threading.Thread(target=voice.start, daemon=True).start()
    print(ui.c("  ⟩ 语音服务加载中…(" + voice.status() + ")", ui.BLUE))

    while True:
        try:
            user_input = voice.term_input()
            if user_input is None:
                continue
        except (EOFError, KeyboardInterrupt):
            print("再见！")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.split()[0].lower()
            if cmd == "/exit":
                print("\n再见！")
                break
            elif cmd == "/help":
                ui.help_text()
            elif cmd == "/tools":
                ui.tools_list(
                    (name, (registry._tools[name]._tool_description or "").split("\n")[0])
                    for name in registry.list_tools()
                )
            elif cmd == "/reset":
                agent.reset()
                print(ui.c("  对话历史已重置。", ui.GREEN))
                ui.turn_rule()
            elif cmd == "/image":
                path, question = _split_image_arg(user_input[6:])
                if not os.path.isfile(path):
                    print(ui.c(f"  找不到图片: {path}", ui.RED))
                    ui.turn_rule()
                    continue
                print(ui.c(f"  🖼  图片 {path}", ui.BLUE)
                      + (ui.c(f"   {question}", ui.DIM) if question else ""))
                run_turn(agent, question or "请详细描述这张图片的内容。", image_path=path)
            elif cmd == "/robot":
                sub = (user_input.split(None, 1) + ["status"])[1].lower()
                if sub == "start":
                    print(ui.c("  ⟩ " + robot.start_robot(), ui.BLUE))
                elif sub == "stop":
                    robot.stop_robot()
                    print(ui.c("  机器人已停止。", ui.GREEN))
                else:
                    print(ui.c("  ⟩ " + robot.robot_status(), ui.BLUE))
                ui.turn_rule()
            elif cmd == "/camera":
                sub = (user_input.split(None, 1) + ["status"])[1].lower()
                if sub == "start":
                    print(ui.c("  ⟩ " + camera.start_camera(), ui.BLUE))
                elif sub == "stop":
                    camera.stop_camera()
                    print(ui.c("  相机已停止。", ui.GREEN))
                else:
                    print(ui.c("  ⟩ " + camera.camera_status(), ui.BLUE))
                ui.turn_rule()
            elif cmd == "/voice":
                print(ui.c("  ⟩ " + voice.status() + "（语音始终可用：按住 Tab 说话）", ui.BLUE))
                ui.turn_rule()
            else:
                print(ui.c(f"  未知命令: {cmd}，输入 /help 查看帮助", ui.RED))
            continue

        run_turn(agent, user_input, tts=True)


if __name__ == "__main__":
    main()
