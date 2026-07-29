import json
from openai import OpenAI

from config import config
from tools import registry


class Agent:
    def __init__(self):
        config.validate()
        self.client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
        self.messages: list[dict] = []
        self.tool_schemas = registry.get_schemas()

    def chat(self, user_input: str, stream: bool = True) -> str:
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(config.MAX_TOOL_ROUNDS):
            response = self.client.chat.completions.create(
                model=config.MODEL,
                messages=self.messages,
                tools=self.tool_schemas if self.tool_schemas else None,
                stream=stream,
            )

            if stream:
                content, tool_calls = self._handle_stream(response)
            else:
                msg = response.choices[0].message
                content = msg.content or ""
                tool_calls = msg.tool_calls or []

            if tool_calls:
                self.messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for tc in tool_calls
                    ],
                })

                for tc in tool_calls:
                    name = tc["name"]
                    try:
                        args = json.loads(tc["arguments"])
                    except json.JSONDecodeError:
                        args = {}
                    print(f"\n  [调用工具: {name}({json.dumps(args, ensure_ascii=False)})]")

                    result = registry.execute(name, args)
                    print(f"  [结果]: {result[:200]}{'...' if len(result) > 200 else ''}")

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })
            else:
                self.messages.append({"role": "assistant", "content": content})
                return content

        return "已达到最大工具调用轮数，对话终止。"

    def _handle_stream(self, response):
        content_parts = []
        tool_calls_map: dict[int, dict] = {}

        for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            if delta.content:
                print(delta.content, end="", flush=True)
                content_parts.append(delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {"id": tc.id or "", "name": "", "arguments": ""}
                    if tc.id:
                        tool_calls_map[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        tool_calls_map[idx]["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        tool_calls_map[idx]["arguments"] += tc.function.arguments

        content = "".join(content_parts)
        tool_calls = list(tool_calls_map.values()) if tool_calls_map else []

        if content and not tool_calls:
            print()

        return content, tool_calls

    def reset(self):
        self.messages = []
