import os
from pathlib import Path
import subprocess
import sys
import signal
import select
import time
import pytest
from types import SimpleNamespace as NS

from main import _split_image_arg, run_turn
from agent import AgentEvent
import ui

ROOT = Path(__file__).resolve().parents[1]


def test_image_path_with_spaces_and_question(tmp_path):
    image = tmp_path/'a picture.png'
    image.touch()
    assert _split_image_arg(f'"{image}" 这是什么') == (str(image), '这是什么')
    assert _split_image_arg(f'{image} 这是什么') == (str(image), '这是什么')


def test_no_hardware_cli_commands_work_without_api_requests():
    env = {**os.environ, 'LLM_API_KEY': 'test-placeholder', 'NO_COLOR': '1'}
    result = subprocess.run([sys.executable, 'main.py', '--no-hardware', '--text-only'],
                            input='/tools\n/status\n/help\n/exit\n', capture_output=True, text=True,
                            cwd=ROOT, env=env, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'read_file' in result.stdout and '文字模式' in result.stdout and '会话结束' in result.stdout
    assert '启动机械臂' not in result.stdout and '正在连接' not in result.stdout


def test_demo_wraps_panels_in_a_narrow_terminal():
    env = {**os.environ, 'COLUMNS': '48', 'NO_COLOR': '1'}
    result = subprocess.run([sys.executable, 'main.py', '--demo'], capture_output=True, text=True,
                            cwd=ROOT, env=env, timeout=10)
    assert result.returncode == 0, result.stderr
    from ui import _disp_width
    assert all(_disp_width(line) <= 48 for line in result.stdout.splitlines())


def test_tty_reply_keeps_streamed_text_and_speaks_same_response(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdout, 'isatty', lambda: True)
    spoken = []

    def stream(*args):
        yield AgentEvent('content', text='你好，')
        yield AgentEvent('content', text='小王。')
        yield AgentEvent('response', text='你好，小王。')

    run_turn(NS(stream=stream), '你好', voice=NS(speak_async=spoken.append))
    output = capsys.readouterr().out
    assert output.count('你好，小王。') == 1
    assert '你好，小王。' in output.split('\r\033[2K')[-1]
    assert spoken == ['你好，小王。']


def test_tty_streamed_text_remains_visible_without_voice(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdout, 'isatty', lambda: True)

    def stream(*args):
        yield AgentEvent('content', text='文字')
        yield AgentEvent('content', text='回复。')
        yield AgentEvent('response', text='文字回复。')

    run_turn(NS(stream=stream), '你好')
    output = capsys.readouterr().out
    assert output.count('文字回复。') == 1
    assert '文字回复。' in output.split('\r\033[2K')[-1]


def test_reply_without_content_chunks_displays_text_and_speaks(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdout, 'isatty', lambda: True)
    spoken = []

    def stream(*args):
        yield AgentEvent('response', text='动作已完成。')

    run_turn(NS(stream=stream), '查询任务', voice=NS(speak_async=spoken.append))
    assert capsys.readouterr().out.count('动作已完成。') == 1
    assert spoken == ['动作已完成。']


def test_sigterm_runs_clean_exit_without_hardware(tmp_path):
    env = {**os.environ, 'LLM_API_KEY':'test-placeholder', 'NO_COLOR':'1'}
    # Isolate lock assertions from an actual robot session running in ROOT.
    script = """import sys
from pathlib import Path
from config import config
config.ROOT=Path(sys.argv.pop(1))
from main import main
sys.exit(main() or 0)
"""
    proc = subprocess.Popen([sys.executable, '-u', '-c', script, str(tmp_path), '--no-hardware', '--text-only'],
        cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = b''
    try:
        until = time.monotonic()+10
        while '你 ›'.encode() not in output and time.monotonic() < until:
            if select.select([proc.stdout], [], [], .1)[0]:
                data = os.read(proc.stdout.fileno(), 4096)
                if not data:
                    break
                output += data
        assert '你 ›'.encode() in output, output.decode()
        proc.send_signal(signal.SIGTERM)
        remaining, errors = proc.communicate(timeout=10)
        output += remaining
        assert proc.returncode == 0, errors.decode()
        assert '退出清理完成' in output.decode() and '会话结束' in output.decode()
        from core.hardware_processes import SessionLock
        lock = SessionLock(tmp_path/'logs/session.lock')
        lock.close()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()


@pytest.mark.parametrize('signum', [signal.SIGTERM, signal.SIGHUP])
def test_termination_during_a_task_exits_without_waiting_for_more_input(signum):
    script = """import sys,time
import agent
from agent import AgentEvent
from main import main
class FakeAgent:
    def stream(self,*args):
        print('BUSY_READY',flush=True)
        time.sleep(60)
        yield AgentEvent('response', text='done')
agent.Agent=FakeAgent
sys.argv=['main.py','--no-hardware','--text-only']
sys.exit(main() or 0)
"""
    env = {**os.environ, 'LLM_API_KEY':'test-placeholder', 'NO_COLOR':'1'}
    proc = subprocess.Popen([sys.executable,'-u','-c',script], cwd=ROOT, env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = b''
    try:
        proc.stdin.write('开始任务\n'.encode())
        proc.stdin.flush()
        until = time.monotonic()+10
        while b'BUSY_READY' not in output and time.monotonic() < until:
            if select.select([proc.stdout],[],[],.1)[0]:
                data = os.read(proc.stdout.fileno(),4096)
                if not data:
                    break
                output += data
        assert b'BUSY_READY' in output, output.decode()
        proc.send_signal(signum)
        # Keep stdin open: a swallowed signal must not be hidden by an EOF exit.
        proc.wait(timeout=5)
        tail, errors = proc.communicate()
        output += tail
        assert proc.returncode == 0, errors.decode()
        assert '退出清理完成' in output.decode() and '会话结束' in output.decode()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
