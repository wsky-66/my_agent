import time
import json
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


def test_model_load_failure_is_not_reported_as_a_valid_scene_for_fallback(monkeypatch,tmp_path):
    setup_scene(monkeypatch,tmp_path)
    monkeypatch.setattr(scene.grasp,'infer',lambda *a,**kw:ToolResult.failure('BACKEND_ERROR','checkpoint missing'))
    result=scene.scan(['apple'],['apple'])
    assert result.error_code=='BACKEND_ERROR' and scene.store.plan is None


def test_dominant_detection_is_not_blocked_by_weak_secondary_hit(monkeypatch,tmp_path):
    _,inferred=setup_scene(monkeypatch,tmp_path)
    objects=[
        {'category':'apple','bbox':[0,0,40,30],'score':.84},
        {'category':'apple','bbox':[60,0,90,30],'score':.3},
        {'category':'bowl','bbox':[0,50,80,100],'score':.88}]
    monkeypatch.setattr(scene.grounding,'detect',lambda *a:ToolResult.success('detected',{'objects':objects}))
    result=scene.scan(['apple','bowl'],['apple'])
    assert result.ok
    selected=result.data['objects']['apple']
    assert selected['instances']==1 and selected['raw_instances']==2 and selected['selection']=='confidence_dominant'
    assert selected['bbox']==objects[0]['bbox'] and inferred[0]['box']==objects[0]['bbox']
    assert result.data['image_path']==str(tmp_path/'color.png')
    assert json.loads((tmp_path/'grounding.json').read_text())['objects']==objects


def test_two_plausible_peppers_still_require_disambiguation():
    hits,raw,reason=scene.detection_candidates([
        {'category':'red chili pepper','bbox':[0,0,30,30],'score':.78},
        {'category':'red chili pepper','bbox':[50,0,80,30],'score':.6}], 'red chili pepper')
    assert len(hits)==len(raw)==2 and reason=='ambiguous'


def test_overlapping_duplicate_boxes_are_one_instance_and_single_weak_hit_is_kept():
    objects=[{'category':'apple','bbox':[0,0,30,30],'score':.4},
             {'category':'apple','bbox':[1,1,31,31],'score':.28}]
    hits,raw,reason=scene.detection_candidates(objects,'apple')
    assert len(hits)==1 and len(raw)==2 and hits[0]==objects[0] and reason=='single'
    assert len(scene.detection_candidates(objects[1:],'apple')[0])==1
