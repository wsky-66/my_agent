"""Construct worker commands with explicit runtime paths and shell quoting."""
import json
import shlex
import subprocess
from config import config
from core.runtime import JsonWorker


def ros_command(args, workspace=False):
    setups = [config.ROS_SETUP] + ([config.WS_SETUP] if workspace else [])
    body = " && ".join([f"source {shlex.quote(p)}" for p in setups] + ["exec " + shlex.join(args)])
    return ["bash", "-c", body]


def worker(name, python, options=None, *, ros=False, workspace=False, timeout=90,env=None):
    args = [python, "-u", "-m", f"workers.{name}_worker", "--options", json.dumps(options or {})]
    if env:
        args=["env",*[f"{key}={value}" for key,value in env.items()],*args]
    return JsonWorker(name, ros_command(args, workspace) if ros else args, timeout=timeout)


def probe(args, workspace=False):
    try:
        return subprocess.run(ros_command(args, workspace), capture_output=True,
                              timeout=3).returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False
