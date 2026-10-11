import json
import os
import sys
import time
import threading
from core.execution import ExecutionContext, current
from core.runtime import JsonWorker

SCRIPT = r'''
import json, os, sys, time
print('READY',flush=True)
for line in sys.stdin:
    req=json.loads(line)
    if req['action']=='slow': time.sleep(1)
    print('RESULT:'+json.dumps({'id':req['id'],'result':{'ok':True,'message':'done','data':{'pid':os.getpid()}}}),flush=True)
'''


def test_worker_reuses_process_and_restarts_after_timeout():
    worker = JsonWorker('test', [sys.executable,'-u','-c',SCRIPT], timeout=0.2)
    try:
        first=worker.request('fast')
        second=worker.request('fast')
        assert first.ok and second.ok and first.data['pid']==second.data['pid']
        assert worker.request('slow').error_code == 'WORKER_TIMEOUT'
        assert worker.proc is None
        third=worker.request('fast')
        assert third.ok and third.data['pid'] != first.data['pid']
    finally:
        worker.stop()
    assert worker.proc is None


def test_mismatched_response_is_rejected_and_worker_stopped():
    script=SCRIPT.replace("'id':req['id']", "'id':'wrong'")
    worker=JsonWorker('test',[sys.executable,'-u','-c',script])
    try:
        result=worker.request('fast')
        assert not result.ok and worker.proc is None
    finally:
        worker.stop()


def test_crashed_worker_does_not_report_success():
    script="import sys; print('READY',flush=True); sys.stdin.readline(); sys.exit(2)"
    worker=JsonWorker('test',[sys.executable,'-u','-c',script])
    try:
        assert not worker.request('fast').ok
    finally:
        worker.stop()


def test_cancellation_terminates_worker_without_retry():
    worker = JsonWorker('test', [sys.executable, '-u', '-c', SCRIPT], timeout=10)
    context = ExecutionContext()
    token = current.set(context)
    timer = threading.Timer(.15, context.cancelled.set)
    try:
        timer.start()
        result = worker.request('slow')
        assert result.error_code == 'TASK_CANCELLED' and worker.proc is None
        assert worker.request('fast').error_code == 'TASK_CANCELLED'
    finally:
        timer.join()
        worker.stop()
        current.reset(token)


def test_invalid_worker_result_remains_a_structured_response():
    script = """from core.results import ToolResult
from workers.protocol import serve
serve(lambda action, args: ToolResult.success('invalid', {'x': float('nan')}) if action == 'bad' else ToolResult.success('done'))
"""
    worker = JsonWorker('test', [sys.executable, '-u', '-c', script])
    try:
        assert worker.request('bad').error_code == 'INVALID_RESULT'
        assert worker.request('good').ok
    finally:
        worker.stop()
