"""Replay saved RGB-D through the production worker, without hardware actions."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from services import grasp
from services.perception import calibrate
from config import config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=json.loads(args.manifest.read_text())
    snapshot=data.get('snapshot') or json.loads((args.manifest.parent/'snapshot.json').read_text())
    snapshot=calibrate(snapshot)
    if 'boxes' in data:
        items=[{'label':label,'box':box,'mask_path':data['objects'][label+'_dilate0']['mask_path']}
               for label,box in data['boxes'].items()]
    else:
        items=[{'label':label,'box':obj['bbox'],'mask_path':obj['mask_path']}
               for label,obj in data['objects'].items() if obj.get('ok')]
    try:
        result=grasp.client.request('infer',{'snapshot':snapshot,'items':items,'top':3,'score_min':config.GRASP_SCORE_MIN})
        if result.ok:
            for item in items:
                label=item['label'];d=result.data['_diagnostics'][label]
                for candidate in result.data[label]:
                    candidate['execution_ready']=False
                print(f"{label}: {len(result.data[label])} candidates; {d.get('pipeline','original')}; "
                      f"{d.get('reason','')}",flush=True)
        else:
            print(f'{result.error_code}: {result.message}',file=sys.stderr)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(result.to_json(),encoding='utf-8')
        return 0 if result.ok else 1
    finally:
        grasp.stop()


if __name__=='__main__':
    raise SystemExit(main())
