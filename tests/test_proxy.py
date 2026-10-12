import os
from pathlib import Path
import subprocess
import sys
import pytest

from config import normalize_proxy_environment

ROOT = Path(__file__).resolve().parents[1]
PROXY_KEYS = ('ALL_PROXY', 'all_proxy', 'HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy')


def test_legacy_socks_urls_keep_address_port_and_auth():
    env = {key: 'socks://user:password@127.0.0.1:10808/' for key in PROXY_KEYS}
    env['NO_PROXY'] = 'localhost,127.0.0.1'
    normalize_proxy_environment(env)
    assert all(env[key] == 'socks5://user:password@127.0.0.1:10808/' for key in PROXY_KEYS)
    assert env['NO_PROXY'] == 'localhost,127.0.0.1'


def test_standard_proxy_urls_and_other_environment_values_are_preserved():
    env = {'ALL_PROXY': 'socks5h://localhost:10808', 'HTTP_PROXY': 'http://localhost:10809',
           'https_proxy': 'https://localhost:10809', 'NO_PROXY': '*', 'MY_SETTING': 'value'}
    before = env.copy()
    normalize_proxy_environment(env)
    assert env == before
    assert normalize_proxy_environment({}) is None


@pytest.mark.parametrize('key', PROXY_KEYS)
def test_cli_initializes_real_sdk_with_legacy_socks_proxy(key):
    env = {name: value for name, value in os.environ.items() if name not in PROXY_KEYS}
    env.update({key: 'socks://127.0.0.1:10808/', 'NO_PROXY': '', 'no_proxy': '',
                'LLM_API_KEY': 'test-placeholder', 'LLM_BASE_URL': 'https://api.example.invalid/v1',
                'NO_COLOR': '1'})
    result = subprocess.run([sys.executable, 'main.py', '--no-hardware', '--text-only'],
                            input='/exit\n', capture_output=True, text=True, cwd=ROOT, env=env, timeout=10)
    assert result.returncode == 0, result.stderr + result.stdout
    assert 'ROBOT WORKSPACE' in result.stdout and '会话结束' in result.stdout
    assert 'Traceback' not in result.stderr
