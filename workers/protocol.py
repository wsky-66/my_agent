"""Newline JSON protocol; stdout belongs exclusively to the transport."""
import argparse
from contextlib import redirect_stdout
import json
import sys
import time
from core.results import ToolResult


def options():
    p = argparse.ArgumentParser()
    p.add_argument("--options", default="{}")
    return json.loads(p.parse_args().options)


def serve(handler, close=lambda: None):
    output = sys.stdout
    print("READY", file=output, flush=True)
    try:
        for line in sys.stdin:
            request_id = None
            start = time.monotonic()
            try:
                req = json.loads(line)
                request_id = req["id"]
                with redirect_stdout(sys.stderr):
                    result = handler(req["action"], req.get("params", {}))
                if not isinstance(result, ToolResult):
                    raise TypeError("worker 必须返回 ToolResult")
            except ValueError as exc:
                result = ToolResult.failure("INVALID_ARGUMENT", str(exc))
            except Exception as exc:
                result = ToolResult.failure("BACKEND_ERROR", str(exc))
            result.elapsed = round(time.monotonic() - start, 4)
            try:
                payload = json.loads(result.to_json())
            except (TypeError, ValueError) as exc:
                payload = json.loads(ToolResult.failure("INVALID_RESULT", str(exc)).to_json())
            print("RESULT:" + json.dumps({"id": request_id, "result": payload},
                                         ensure_ascii=False), file=output, flush=True)
    finally:
        with redirect_stdout(sys.stderr):
            close()
