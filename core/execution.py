"""Per-turn cancellation and workflow progress, separate from chat history."""
from contextvars import ContextVar
from dataclasses import dataclass, field
import queue
import threading


@dataclass
class ExecutionContext:
    cancelled: threading.Event = field(default_factory=threading.Event)
    progress: queue.Queue = field(default_factory=queue.Queue)


current = ContextVar("execution_context", default=None)
shutdown_requested = threading.Event()
_tasks = {}
_tasks_changed = threading.Condition()


def begin_task(context):
    with _tasks_changed:
        if shutdown_requested.is_set():
            return False
        _tasks[id(context)] = context
        return True


def end_task(context):
    with _tasks_changed:
        _tasks.pop(id(context), None)
        _tasks_changed.notify_all()


def cancel_tasks(timeout=5):
    shutdown_requested.set()
    with _tasks_changed:
        for context in _tasks.values():
            context.cancelled.set()
        return _tasks_changed.wait_for(lambda: not _tasks, timeout=timeout)


def open_session():
    with _tasks_changed:
        if _tasks:
            raise RuntimeError("旧任务尚未退出")
        shutdown_requested.clear()


def is_cancelled():
    ctx = current.get()
    return ctx is not None and ctx.cancelled.is_set()


def report(name, args, result=None):
    ctx = current.get()
    if ctx:
        ctx.progress.put((name, args, result))
