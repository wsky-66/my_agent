"""Lazy rotating error and timing logs; never record keys or full tool arguments."""
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar

turn_id = ContextVar("turn_id", default="")
_lock = threading.Lock()


def _get(name, filename):
    log = logging.getLogger(name)
    with _lock:
        if not log.handlers:
            folder = Path(__file__).resolve().parent / "logs"
            folder.mkdir(exist_ok=True)
            handler = RotatingFileHandler(folder / filename, maxBytes=2_000_000,
                                          backupCount=3, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s" if name == "agent.metrics"
                                                  else "%(asctime)s [%(levelname)s] %(message)s"))
            log.addHandler(handler)
            log.setLevel(logging.INFO)
            log.propagate = False
    return log


def error(msg, exc=None):
    _get("agent.errors", "agent_error.log").error("%s%s", msg, f" | {exc!r}" if exc else "")


def warning(msg):
    _get("agent.errors", "agent_error.log").warning(msg)


def record_timing(stage, elapsed, ok=True):
    _get("agent.metrics", "timings.jsonl").info(json.dumps({
        "timestamp": time.time(), "turn_id": turn_id.get(), "stage": stage,
        "elapsed": round(elapsed, 4), "ok": ok,
    }, ensure_ascii=False))


@contextmanager
def measure(stage):
    start = time.monotonic()
    ok = False
    try:
        yield
        ok = True
    finally:
        record_timing(stage, time.monotonic() - start, ok)
