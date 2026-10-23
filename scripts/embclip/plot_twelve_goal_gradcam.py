"""Render two models by twelve target categories, with overlay panels only."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    summary=json.loads((args.result/'summary.json').read_text())
    if summary['status']!='complete' or len(summary['records'])!=12:
        raise RuntimeError('Expected twelve completed categories')
    if json.loads((args.result/'exit.json').read_text())['return_code']!=0:
        raise RuntimeError('Analysis did not exit successfully')
    names={'AlarmClock':'Alarm\nClock','BaseballBat':'Baseball\nBat','BasketBall':'Basket\nBall',
           'GarbageCan':'Garbage\nCan','HousePlant':'House\nPlant','SprayBottle':'Spray\nBottle'}
    records=sorted(summary['records'],key=lambda r:r['target'])
    fig=plt.figure(figsize=(24,4.8),facecolor='white')
    grid=fig.add_gridspec(2,13,width_ratios=[1.7]+[1]*12,wspace=.035,hspace=.055)
    checks=[]
    for row,model in enumerate(('gru1','gawf1')):
        label=fig.add_subplot(grid[row,0]);label.axis('off')
        label.text(.98,.5,'EmbCLIP' if row==0 else 'EmbCLIP-\nGaWF',ha='right',va='center',fontsize=24,color='black')
        for column,record in enumerate(records,1):
            target=record['target'];data=record['models'][model]
            cam_path=args.result/(target+'_'+model+'_cam.npy');rgb_path=args.result/(target+'_'+model+'_rgb.npy')
            for path,key in [(cam_path,'cam_sha256'),(rgb_path,'rgb_sha256')]:
                if hashlib.sha256(path.read_bytes()).hexdigest()!=data[key]:
                    raise RuntimeError('Attribution source changed: '+str(path))
            cam=np.load(cam_path);rgb=np.load(rgb_path)
            if cam.shape!=(7,7) or rgb.shape!=(224,224,3) or not np.isfinite(cam).all():
                raise RuntimeError('Unexpected attribution shape')
            ax=fig.add_subplot(grid[row,column]);ax.imshow(rgb)
            ax.imshow(cam,cmap='jet',vmin=0,vmax=max(float(cam.max()),1e-12),alpha=.48,
                      extent=(-.5,223.5,223.5,-.5),interpolation='bilinear')
            ax.axis('off')
            if row==0:ax.set_title(names.get(target,target),fontsize=22,color='black',pad=13)
            checks.append(dict(target=target,model=model,positive_max=float(cam.max()),episode_id=data['episode_id']))
    fig.subplots_adjust(left=.005,right=.995,top=.79,bottom=.02)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180,facecolor='white');plt.close(fig)
    receipt=dict(rows=2,columns=12,panels=24,layout='overlays only; no separate raw RGB panels',
                 targets=[r['target'] for r in records],comparison_mode=summary['comparison_mode'],
                 source_summary_sha256=hashlib.sha256((args.result/'summary.json').read_bytes()).hexdigest(),
                 image_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),panels_verified=checks)
    args.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Rendered 2 x 12 Grad-CAM overlays:',args.output)


if __name__=='__main__':main()
