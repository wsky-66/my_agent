from copy import deepcopy
from types import SimpleNamespace as NS
import pytest

from agent import Agent
from config import config
from core.execution import current
from core.registry import ToolRegistry
from core.results import ToolResult


def chunk(content=None, calls=None):
    return NS(choices=[NS(delta=NS(content=content, tool_calls=calls))])


def call(index, name, arguments, call_id=None):
    return NS(index=index, id=call_id, function=NS(name=name, arguments=arguments))


class Response:
    def __init__(self, chunks):
        self.chunks, self.closed = chunks, False

    def __iter__(self):
        yield from self.chunks

    def close(self):
        self.closed = True


class Client:
    def __init__(self, responses):
        self.responses, self.requests = responses, []
        self.chat = NS(completions=self)

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        return self.responses[len(self.requests)-1]


def test_fragmented_tool_call_and_completed_history_survive_generator_close():
    registry = ToolRegistry()
    seen = []
    @registry.register
    def add(a: int, b: int):
        seen.append((a, b))
        return ToolResult.success('done', {'sum': a+b})
    replies = [Response([chunk(calls=[call(0, 'ad', '{"a":', 'c1')]),
                         chunk(calls=[call(0, 'd', '2,"b":3}')])]), Response([chunk('完成')])]
    client = Client(replies)
    agent = Agent(client, registry)
    stream = agent.stream('算一下')
    for event in stream:
        if event.type == 'response':
            stream.close()
            break
    assert seen == [(2, 3)]
    assert [m['role'] for m in agent.messages] == ['system', 'user', 'assistant']
    assert agent.messages[-1]['content'] == '完成'
    assert all(r.closed for r in replies)
    assert current.get() is None
    assert client.requests[1]['messages'][-1]['tool_call_id'] == 'c1'


def test_failed_tool_skips_remaining_actions_and_completes_protocol():
    registry = ToolRegistry()
    invoked = []
    @registry.register
    def fail():
        return ToolResult.failure('NO_REACH', '不可达')
    @registry.register
    def move():
        invoked.append(True)
        return ToolResult.success('done')
    client = Client([Response([chunk(calls=[call(0, 'fail', '{}', 'c1'), call(1, 'move', '{}', 'c2')])]),
                     Response([chunk('已停止')])])
    events = list(Agent(client, registry).stream('移动'))
    results = [e.result for e in events if e.type == 'tool_result']
    assert [r.error_code for r in results] == ['NO_REACH', 'PREVIOUS_TOOL_FAILED']
    assert not invoked
    assert [m['tool_call_id'] for m in client.requests[1]['messages'] if m['role'] == 'tool'] == ['c1', 'c2']


def test_invalid_json_never_calls_tool():
    registry = ToolRegistry()
    @registry.register
    def move():
        raise AssertionError('must not execute')
    client = Client([Response([chunk(calls=[call(0, 'move', '{', 'c1')])]), Response([chunk('停止')])])
    events = list(Agent(client, registry).stream('移动'))
    assert next(e.result for e in events if e.type == 'tool_result').error_code == 'INVALID_ARGUMENT'


def test_interrupted_stream_rolls_back_partial_turn_and_closes_connection():
    response = Response([chunk('一部分'), chunk('剩余部分')])
    agent = Agent(Client([response]), ToolRegistry())
    before = deepcopy(agent.messages)
    stream = agent.stream('新任务')
    assert next(stream).text == '一部分'
    stream.close()
    assert response.closed and agent.messages == before and current.get() is None


def test_tool_round_limit_keeps_valid_conversation(monkeypatch):
    monkeypatch.setattr(config, 'MAX_TOOL_ROUNDS', 1)
    registry = ToolRegistry()
    @registry.register
    def ping():
        return ToolResult.success('pong')
    agent = Agent(Client([Response([chunk(calls=[call(0, 'ping', '{}', 'c1')])])]), registry)
    events = list(agent.stream('循环'))
    assert events[-1].type == 'response' and '上限' in events[-1].text
    assert [m['role'] for m in agent.messages] == ['system', 'user', 'assistant']


def test_api_exception_rolls_back_history():
    class Broken(Client):
        def create(self, **kwargs):
            raise RuntimeError('network unavailable')
    agent = Agent(Broken([]), ToolRegistry())
    with pytest.raises(RuntimeError):
        list(agent.stream('新任务'))
    assert len(agent.messages) == 1 and current.get() is None


def test_failed_workflow_disables_tools_in_followup_round_and_cannot_reorder_or_use_shell():
    registry=ToolRegistry();invoked=[]
    @registry.register
    def execute_pick_place():
        invoked.append('workflow')
        return ToolResult.failure('SCENE_INCOMPLETE','红辣椒目标不明确')
    @registry.register
    def run_shell():
        invoked.append('shell')
        return ToolResult.success('shell')
    client=Client([Response([chunk(calls=[call(0,'execute_pick_place','{}','c1')])]),
                   Response([chunk('我先抓香蕉',calls=[call(0,'run_shell','{}','c2'),call(1,'execute_pick_place','{}','c3')])])])
    events=list(Agent(client,registry).stream('辣椒放箱子，香蕉放碗'))
    assert invoked==['workflow'] and len(client.requests)==2
    assert client.requests[1]['tools'] is None
    assert events[-1].type=='response' and '已停止' in events[-1].text
    assert sum(e.type=='tool_call' for e in events)==1 and all(r.closed for r in client.responses)
    assert not any('先抓香蕉' in e.text for e in events if e.type=='content')


def test_user_clarification_starts_new_turn_with_tools_enabled_after_failure():
    registry=ToolRegistry();invoked=[]
    @registry.register
    def execute_pick_place():
        invoked.append('workflow')
        return ToolResult.failure('SCENE_INCOMPLETE','目标不明确') if len(invoked)==1 else ToolResult.success('done')
    client=Client([Response([chunk(calls=[call(0,'execute_pick_place','{}','c1')])]),Response([chunk('请明确目标')]),
                   Response([chunk(calls=[call(0,'execute_pick_place','{}','c2')])]),Response([chunk('动作序列完成')])])
    agent=Agent(client,registry)
    list(agent.stream('抓辣椒放箱子，香蕉放碗'))
    list(agent.stream('红色的辣椒'))
    assert len(invoked)==2 and client.requests[1]['tools'] is None and client.requests[2]['tools']
