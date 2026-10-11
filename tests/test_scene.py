import time
from core.results import ToolResult
from core.execution import ExecutionContext, current
from services import scene


def setup_scene(monkeypatch, tmp_path, failed=False):
    frame=str(tmp_path/'color.png')
    snapshot={'image_path':frame,'created':time.time(),'base_frame':'base_link'}
    monkeypatch.setattr(scene.camera,'capture',lambda:ToolResult.success('frame',snapshot))
    monkeypatch.setattr(scene,'calibrate',lambda value:value)
    monkeypatch.setattr(scene.grounding,'detect',lambda *a:ToolResult.success('detected',{'objects':[
        {'category':'apple','bbox':[0,0,2,2]}, {'category':'bowl','bbox':[2,0,4,2]}]}))
    batches=[]
    def segment(image,items):
        batches.append((image,items))
        return ToolResult.success('seg',{v['label']:{'mask_path':v['out']} for v in items})
    monkeypatch.setattr(scene.segmentation,'batch',segment)
    monkeypatch.setattr(scene,'position',lambda **args: ToolResult.failure('DEPTH','invalid depth') if failed else
                        ToolResult.success('pos',{'position':[0.2,0.1,0.05],'size':[0.1,0.1,0.1]}))
    inferred=[]
    def infer(items, snapshot, **kwargs):
        inferred.extend(items)
        return ToolResult.success('grasp',{})
    monkeypatch.setattr(scene.grasp,'infer',infer)
    return batches,inferred


def test_scene_uses_one_segmentation_batch_and_only_grasps_sources(monkeypatch,tmp_path):
    batches,inferred=setup_scene(monkeypatch,tmp_path)
    result=scene.scan(['apple','bowl'],['apple'])
    assert result.ok
    assert len(batches)==1 and len(batches[0][1])==2
    assert all(item['dilate_cm']==0 for item in batches[0][1])
    assert [v['label'] for v in inferred]==['apple']
    assert result.data['scene_id']==scene.store.plan.scene_id


def test_position_failure_cannot_be_marked_ok(monkeypatch,tmp_path):
    setup_scene(monkeypatch,tmp_path,failed=True)
    result=scene.scan(['apple','bowl'],['apple'])
    assert not result.ok and result.error_code=='SCENE_INCOMPLETE'
    assert not result.data['objects']['apple']['ok']


def test_pineapple_does_not_match_apple(monkeypatch,tmp_path):
    setup_scene(monkeypatch,tmp_path)
    monkeypatch.setattr(scene.grounding,'detect',lambda *a:ToolResult.success('det',{'objects':[
        {'category':'pineapple','bbox':[0,0,2,2]}]}))
    assert not scene.scan(['apple'],['apple']).ok


def test_cancelled_scan_cannot_restore_old_scene(monkeypatch, tmp_path):
    setup_scene(monkeypatch, tmp_path)
    context = ExecutionContext()
    def infer(*args, **kwargs):
        context.cancelled.set()
        return ToolResult.success('done', {})
    monkeypatch.setattr(scene.grasp, 'infer', infer)
    token = current.set(context)
    try:
        result = scene.scan(['apple'], ['apple'])
        assert result.error_code == 'TASK_CANCELLED' and scene.store.plan is None
    finally:
        current.reset(token)
