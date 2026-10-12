import math
import pytest
from core.registry import ToolRegistry
from core.results import ToolResult


def test_typed_nested_parameters_and_nullable_schema():
    r = ToolRegistry()
    calls = []
    @r.register
    def move(points: list[list[float]], pitch: float = None):
        calls.append(points)
        return ToolResult.success('done')
    params = r.get_schemas()[0]['function']['parameters']
    assert params['properties']['points']['items']['items']['type'] == 'number'
    assert params['properties']['pitch']['anyOf'][1]['type'] == 'null'
    assert r.execute_result('move', {'points': [[1, 2.0]], 'pitch': None}).ok
    assert not r.execute_result('move', {'points': [[True, 2]]}).ok
    assert not r.execute_result('move', {'points': [[float('nan'), 2]]}).ok
    assert not r.execute_result('move', {'points': [[1, 2]], 'surprise': 1}).ok
    assert len(calls) == 1


def test_explicit_status_does_not_depend_on_words_in_message():
    r = ToolRegistry()
    @r.register
    def good():
        return ToolResult.success('未发现错误')
    @r.register
    def bad():
        return ToolResult.failure('NO_REACH', '目标偏差')
    assert r.execute_result('good', {}).ok
    assert not r.execute_result('bad', {}).ok
    assert r.execute_result('missing', {}).error_code == 'UNKNOWN_TOOL'


def test_missing_argument_never_executes():
    r = ToolRegistry()
    @r.register
    def danger(x: float):
        raise AssertionError('must not execute')
    assert r.execute_result('danger', {}).error_code == 'INVALID_ARGUMENT'
