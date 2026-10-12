from types import SimpleNamespace as NS
from capabilities import voice
from core.results import ToolResult


def test_disabled_voice_cannot_be_reenabled_by_an_old_loader(monkeypatch):
    calls = []
    monkeypatch.setattr(voice, '_asr', NS(request=lambda *a: calls.append(a) or ToolResult.success('ready'),
                                        ready=lambda: False, stop=lambda: None))
    generation = voice._enable()
    voice.stop()
    assert voice._load(generation) == '语音已关闭'
    assert not calls and not voice._enabled
