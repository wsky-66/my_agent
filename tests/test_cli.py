import os
from pathlib import Path
import subprocess
import sys

from main import _split_image_arg

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
