"""LLM event stream; deterministic motion belongs to workflows, hardware to services."""
import base64
from contextvars import copy_context
from dataclasses import dataclass
import json
import mimetypes
from pathlib import Path
import queue
import threading
import time
from typing import Iterator
import uuid

from openai import OpenAI
from config import config
from core.execution import ExecutionContext, current, begin_task, end_task
from core.results import ToolResult
from logger import error, record_timing, turn_id
from tools import registry

SYSTEM_PROMPT = (Path(__file__).resolve().parent / "prompts/system.md").read_text(encoding="utf-8")
STOP_ON_FAILURE = {"execute_pick_place", "scan_objects", "move_arm", "set_joints", "set_joint", "go_home", "go_sleep",
                   "gripper_set", "gripper_open", "gripper_close", "gripper_grasp"}


@dataclass
class AgentEvent:
    type: str
    text: str = ""
    name: str = ""
    args: dict | None = None
    result: ToolResult | None = None
    phase: str = ""
    elapsed: float = 0


class Agent:
    def __init__(self, client=None, tool_registry=None):
        if client is None:
            config.validate()
        self.client = client or OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL,
                                       timeout=config.LLM_TIMEOUT, max_retries=1)
        self.registry = tool_registry or registry
        self.tool_schemas = self.registry.get_schemas()
        self.reset()

    def reset(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    @staticmethod
    def _build_content(text, image_path=None):
        if image_path is None:
            return text
        path = Path(image_path)
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        uri = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
        return [{"type": "image_url", "image_url": {"url": uri}}, {"type": "text", "text": text}]

    def _compact(self):
        user = next((m for m in reversed(self.messages) if m["role"] == "user"), None)
        assistant = next((m for m in reversed(self.messages) if m["role"] == "assistant" and not m.get("tool_calls")), None)
        self.messages[:] = [self.messages[0]] + ([user] if user else []) + ([assistant] if assistant else [])

    def _dispatch(self, name, args, context):
        result = []
        inherited = copy_context()

        def run():
            if not begin_task(context):
                result.append(ToolResult.failure("SESSION_CLOSING", "程序正在退出，拒绝新任务"))
                return
            try:
                result.append(self.registry.execute_result(name, args))
            except BaseException as exc:
                result.append(ToolResult.failure("TASK_CANCELLED", str(exc) or "任务中断"))
            finally:
                end_task(context)

        thread = threading.Thread(target=lambda: inherited.run(run), daemon=True)
        thread.start()
        try:
            while thread.is_alive() or not context.progress.empty():
                try:
                    step, params, completed = context.progress.get(timeout=0.05)
                    yield AgentEvent(type="workflow_step", name=step, args=params, result=completed,
                                     phase="done" if completed is not None else "start")
                except queue.Empty:
                    pass
            return result[0]
        finally:
            if thread.is_alive():
                context.cancelled.set()

    def stream(self, user_input: str, image_path: str = None) -> Iterator[AgentEvent]:
        from services.scene import store
        from services import camera
        store.clear()
        camera.clear_latest()
        mark = len(self.messages)
        context = ExecutionContext()
        execution_token = current.set(context)
        timing_token = turn_id.set(uuid.uuid4().hex)
        started = time.monotonic()
        finished = False
        stopped = None
        try:
            self.messages.append({"role": "user", "content": self._build_content(user_input, image_path)})
            for _ in range(config.MAX_TOOL_ROUNDS):
                request_start = time.monotonic()
                response = None
                request_ok = False
                parts, calls = [], {}
                first_token = False
                try:
                    messages = self.messages
                    if stopped is not None:
                        messages = self.messages + [{"role":"system","content":
                            "本轮任务因工具失败已经停止。仅用现有结果说明失败、已执行和未完成部分；目标仍有歧义时询问目标。"
                            "不要调用工具、不要重排或跳过原动作，也不要编造画面或宣称抓放已被视觉确认。"}]
                    response = self.client.chat.completions.create(
                        model=config.MODEL, messages=messages, tools=(self.tool_schemas or None) if stopped is None else None,
                        stream=True, parallel_tool_calls=False)
                    for chunk in response:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        if not first_token and (delta.content or delta.tool_calls):
                            record_timing("llm.first_token", time.monotonic()-request_start)
                            first_token = True
                        if delta.content:
                            parts.append(delta.content)
                            if stopped is None:
                                yield AgentEvent(type="content", text=delta.content)
                        for call in delta.tool_calls or []:
                            entry = calls.setdefault(call.index, {"id": "", "name": "", "arguments": ""})
                            if call.id:
                                entry["id"] = call.id
                            if call.function:
                                entry["name"] += call.function.name or ""
                                entry["arguments"] += call.function.arguments or ""
                    request_ok = True
                finally:
                    if response is not None and hasattr(response, "close"):
                        response.close()
                    record_timing("llm.request", time.monotonic()-request_start, request_ok)
                text = "".join(parts)
                if calls and stopped is not None:
                    text = stopped.message+"。本轮已停止后续操作。"
                    calls = {}
                if stopped is not None and not calls:
                    text = text or stopped.message+"。本轮已停止后续操作。"
                    yield AgentEvent(type="content",text=text)
                if not calls:
                    self.messages.append({"role": "assistant", "content": text})
                    self._compact()
                    finished = True
                    yield AgentEvent(type="response", text=text)
                    return
                ordered = [calls[i] for i in sorted(calls)]
                self.messages.append({"role": "assistant", "content": text, "tool_calls": [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in ordered]})
                failed = False
                for call in ordered:
                    args = {}
                    try:
                        args = json.loads(call["arguments"] or "{}")
                        if not isinstance(args, dict):
                            raise ValueError("工具参数必须是对象")
                        if failed:
                            result = ToolResult.failure("PREVIOUS_TOOL_FAILED", "前一步失败，跳过本批次后续工具")
                        else:
                            yield AgentEvent(type="tool_call", name=call["name"], args=args)
                            result = yield from self._dispatch(call["name"], args, context)
                    except ValueError as exc:
                        result = ToolResult.failure("INVALID_ARGUMENT", str(exc))
                    failed = failed or not result.ok
                    if not result.ok and call["name"] in STOP_ON_FAILURE and stopped is None:
                        stopped = result
                    self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result.to_json()})
                    yield AgentEvent(type="tool_result", name=call["name"], result=result, elapsed=result.elapsed)
            text = "已达到工具调用轮数上限，停止后续操作。"
            self.messages.append({"role": "assistant", "content": text})
            self._compact()
            finished = True
            yield AgentEvent(type="response", text=text)
        except BaseException:
            if not finished:
                context.cancelled.set()
                del self.messages[mark:]
                store.clear()
            raise
        finally:
            record_timing("turn.total", time.monotonic()-started, finished)
            current.reset(execution_token)
            turn_id.reset(timing_token)
