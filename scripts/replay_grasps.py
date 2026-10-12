"""Replay saved RGB-D through the production worker, without hardware actions."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from services import grasp
from services.perception import calibrate
from config import config


def preview(snapshot, items, results, output):
    """Project the same 3D candidate wireframes as RViz onto the saved RGB."""
    import cv2
    import numpy as np
    from core.grasp_geometry import visual_vertices
    image=cv2.imread(snapshot['image_path'])
    k=np.asarray(snapshot['intrinsics']).reshape(3,3)
    transform=np.asarray(snapshot['extrinsics'])
    edges=[(i,j) for i in range(8) for j in range(i+1,8) if (i^j) in (1,2,4)]
    for item in items:
        mask=cv2.imread(item['mask_path'],0)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image,contours,-1,(210,180,20),1)
        grasps=results[item['label']]
        for g in grasps:
            row=np.array([g['score'],g['width'],.02,g['depth'],*np.ravel(g['rotation']),*g['position'],-1.])
            boxes=visual_vertices(row[None])[0]
            camera=(boxes-transform[:3,3])@transform[:3,:3]
            projected=camera@k.T
            if np.any(camera[...,2]<=0):continue
            pixels=np.rint(projected[...,:2]/projected[...,2:]).astype(int)
            for box in pixels:
                for a,b in edges:cv2.line(image,tuple(box[a]),tuple(box[b]),(50,220,60),1,cv2.LINE_AA)
        x,y,w,h=cv2.boundingRect(mask)
        cv2.putText(image,f"{item['label']}: {len(grasps)}",(x,max(15,y-6)),cv2.FONT_HERSHEY_SIMPLEX,.45,
                    (50,220,60) if grasps else (40,160,240),1,cv2.LINE_AA)
    output.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(output),image):raise ValueError('预览图片保存失败')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preview',type=Path,help='将通过检测的 3D 抓取框投影到保存图像')
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
            if args.preview:
                preview(snapshot,items,result.data,args.preview)
        else:
            print(f'{result.error_code}: {result.message}',file=sys.stderr)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(result.to_json(),encoding='utf-8')
        return 0 if result.ok else 1
    finally:
        grasp.stop()


if __name__=='__main__':
    raise SystemExit(main())
