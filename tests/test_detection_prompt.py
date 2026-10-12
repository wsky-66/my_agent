import json
import time

import pytest

from core.registry import ToolRegistry
from core.results import ToolResult
from services import scene, vision
from services.detection_prompt import build_prompt
from workflows import manipulation


def test_agent_phrases_keep_original_names_and_form_one_prompt():
    prompt, aliases = build_prompt(['banana', 'bowl'],
        {' Banana ': ['yellow banana', 'yellow fruit', 'yellow object', 'yellow banana']})
    assert prompt=='banana.bowl.yellow banana.yellow fruit.yellow object'
    assert aliases['banana']=={'names':['banana','yellow banana'], 'generic':['yellow fruit','yellow object']}
    assert aliases['bowl']=={'names':['bowl'], 'generic':[]}
    assert build_prompt(['grape', 'bowl'])[0]=='grape.bowl'


def test_same_colour_targets_keep_specific_names_instead_of_shared_generic_phrases():
    prompt, aliases = build_prompt(['apple', 'chili'],
        {'apple':['red apple','red fruit'], 'chili':['red chili','red object']})
    assert 'red apple' in prompt and 'red chili' in prompt
    assert 'red fruit' not in prompt and 'red object' not in prompt
    assert not aliases['apple']['generic'] and not aliases['chili']['generic']


def test_named_detection_wins_over_a_higher_scoring_colour_only_object():
    _, aliases = build_prompt(['banana'], {'banana':['yellow banana','yellow object']})
    named={'category':'yellow banana','bbox':[0,0,30,30],'score':.4}
    other={'category':'yellow object','bbox':[60,0,90,30],'score':.9}
    hits, raw, reason = scene.detection_candidates([other,named],'banana',aliases['banana'])
    assert hits==raw==[named] and reason=='single'


def test_generic_fruit_and_object_duplicates_are_mapped_to_one_target():
    _, aliases = build_prompt(['grape'], {'grape':['grapes','purple grapes','purple fruit','purple object']})
    fruit={'category':'purple fruit','bbox':[0,0,30,30],'score':.8}
    obj={'category':'purple object','bbox':[1,1,31,31],'score':.7}
    hits, raw, reason = scene.detection_candidates([fruit,obj],'grape',aliases['grape'])
    assert hits==[fruit] and len(raw)==2 and reason=='single'
    grape={'category':'grapes','bbox':[0,0,30,30],'score':.5}
    assert scene.detection_candidates([grape],'grape',aliases['grape'])[0]==[grape]


@pytest.mark.parametrize('hints', [['yellow banana'], {'unknown':['yellow object']},
                                  {'banana':'yellow object'}, {'banana':['']},
                                  {'banana':['yellow object.banana']}])
def test_invalid_hints_are_rejected_before_physical_commands(monkeypatch, hints):
    commands=[]
    monkeypatch.setattr(manipulation.arm,'call',lambda *a,**kw:commands.append(a))
    with pytest.raises(ValueError):
        manipulation.execute([{'object':'banana'}], detection_hints=hints)
    assert not commands


def test_colour_alias_scene_uses_one_detection_and_no_additional_llm(monkeypatch, tmp_path):
    frame=str(tmp_path/'color.png')
    snapshot={'image_path':frame,'created':time.time(),'base_frame':'base'}
    detected, segmented, inferred=[],[],[]
    objects=[{'category':'purple fruit','bbox':[0,0,30,30],'score':.8},
             {'category':'purple object','bbox':[1,1,31,31],'score':.7},
             {'category':'bowl','bbox':[60,0,90,30],'score':.8}]
    monkeypatch.setattr(scene.camera,'capture',lambda:ToolResult.success('frame',snapshot))
    monkeypatch.setattr(scene,'calibrate',lambda value:value)
    monkeypatch.setattr(vision,'describe',lambda *a,**kw:pytest.fail('do not request a colour vision pass'))
    def detect(prompt,image):
        detected.append((prompt,image))
        return ToolResult.success('detected',{'objects':objects})
    def segment(image,items):
        segmented.extend(items)
        return ToolResult.success('segmented',{v['label']:{'mask_path':v['out']} for v in items})
    def infer(items,snapshot,**kwargs):
        inferred.extend(items)
        return ToolResult.success('predicted',{})
    monkeypatch.setattr(scene.grounding,'detect',detect)
    monkeypatch.setattr(scene.segmentation,'batch',segment)
    monkeypatch.setattr(scene,'position',lambda **kw:ToolResult.success('located',{'position':[.2,.1,.05],'size':[.03]*3}))
    monkeypatch.setattr(scene.grasp,'infer',infer)
    result=scene.scan(['grape','bowl'],['grape'], {'grape':['grapes','purple grapes','purple fruit','purple object']})
    try:
        assert result.ok and len(detected)==1
        assert detected[0][0]=='grape.bowl.grapes.purple grapes.purple fruit.purple object'
        assert detected[0][1]==frame
        assert [v['label'] for v in segmented]==['grape','bowl']
        assert [v['label'] for v in inferred]==['grape'] and inferred[0]['box']==objects[0]['bbox']
        assert result.data['objects']['grape']['instances']==1
        assert json.loads((tmp_path/'grounding.json').read_text())['prompt']==detected[0][0]
    finally:
        scene.store.clear()


def test_workflow_and_tool_schema_forward_agent_hints_without_changing_task_order(monkeypatch):
    from capabilities.workflow_tools import execute_pick_place
    registry=ToolRegistry()
    registry.register(execute_pick_place,description=execute_pick_place._tool_description)
    props=registry.get_schemas()[0]['function']['parameters']['properties']
    mapping=props['detection_hints']['anyOf'][0]
    assert mapping['additionalProperties']=={'type':'array','items':{'type':'string'}}
    scans, commands=[],[]
    monkeypatch.setattr(manipulation.arm,'call',lambda name,**kw:commands.append(name) or ToolResult.success('done'))
    def scan(labels,grasp_labels,**kwargs):
        scans.append((labels,grasp_labels,kwargs))
        return ToolResult.failure('SCENE_INCOMPLETE','synthetic stop before picking')
    monkeypatch.setattr(scene,'scan',scan)
    hints={'banana':['yellow banana','yellow fruit'], 'grape':['purple grapes','purple fruit']}
    result=registry.execute_result('execute_pick_place',{'actions':[{'object':'banana','target':'plate'},
        {'object':'grape','target':'bowl'}],'detection_hints':hints})
    assert result.error_code=='SCENE_INCOMPLETE'
    assert scans==[(['banana','grape','plate','bowl'],['banana','grape'],{'detection_hints':hints})]
    assert commands==['go_sleep']
