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


def is_cancelled():
    ctx = current.get()
    return ctx is not None and ctx.cancelled.is_set()


def report(name, args, result=None):
    ctx = current.get()
    if ctx:
        ctx.progress.put((name, args, result))
