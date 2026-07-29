#!/usr/bin/env python3
import readline
import sys

from agent import Agent
from config import config
from tools import registry


HELP_TEXT = """
可用命令:
  /help       - 显示此帮助信息
  /tools      - 列出所有可用工具
  /reset      - 重置对话历史
  /exit       - 退出程序

直接输入自然语言即可与 Agent 对话，Agent 会自动调用工具完成任务。
""".strip()


def print_banner():
    print(f"""
╔══════════════════════════════════════╗
║        LLM Agent - 工具调用助手        ║
╠══════════════════════════════════════╣
║  模型: {config.MODEL:<28}║
║  服务: {config.BASE_URL:<28}║
║  工具: {len(registry.list_tools()):<28}║
╚══════════════════════════════════════╝
输入 /help 查看帮助，直接输入文字开始对话。
""")


def main():
    try:
        config.validate()
    except ValueError as e:
        print(f"配置错误: {e}")
        sys.exit(1)

    print_banner()
    agent = Agent()

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.split()[0].lower()
            if cmd == "/exit":
                print("再见！")
                break
            elif cmd == "/help":
                print(HELP_TEXT)
            elif cmd == "/tools":
                for name in registry.list_tools():
                    func = registry._tools[name]
                    desc = func._tool_description or ""
                    desc_line = desc.split("\n")[0]
                    print(f"  • {name}: {desc_line}")
                print(f"\n共 {len(registry.list_tools())} 个工具")
            elif cmd == "/reset":
                agent.reset()
                print("对话历史已重置。")
            else:
                print(f"未知命令: {cmd}，输入 /help 查看帮助")
            continue

        try:
            print("Agent: ", end="", flush=True)
            agent.chat(user_input)
        except Exception as e:
            print(f"\n错误: {e}")


if __name__ == "__main__":
    main()
