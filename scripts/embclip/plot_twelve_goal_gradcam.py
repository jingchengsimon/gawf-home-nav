"""Render RGB/Grad-CAM pairs: three goals per model row, six goals per page."""
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
    parser.add_argument('--page',type=int,choices=(1,2),default=1)
    args=parser.parse_args()
    summary=json.loads((args.result/'summary.json').read_text())
    if summary['status']!='complete' or len(summary['records'])!=12:
        raise RuntimeError('Expected twelve completed categories')
    if json.loads((args.result/'exit.json').read_text())['return_code']!=0:
        raise RuntimeError('Analysis did not exit successfully')
    names={'AlarmClock':'Alarm Clock','BaseballBat':'Baseball Bat','BasketBall':'Basketball',
           'GarbageCan':'Garbage Can','HousePlant':'House Plant','SprayBottle':'Spray Bottle'}
    records=sorted(summary['records'],key=lambda r:r['target'])[(args.page-1)*6:args.page*6]
    fig=plt.figure(figsize=(14,9),facecolor='white')
    fig.subplots_adjust(left=.002,right=.998,top=.94,bottom=.005)
    groups=fig.add_gridspec(2,1,hspace=.23)
    checks=[]
    for group in range(2):
        grid=groups[group].subgridspec(2,7,width_ratios=[.78]+[1]*6,wspace=.025,hspace=.06)
        for row,model in enumerate(('gru1','gawf1')):
            label=fig.add_subplot(grid[row,0]);label.axis('off')
            label.text(0,.5,'EmbCLIP' if row==0 else 'EmbCLIP-\nGaWF',ha='left',va='center',
                       fontsize=18,fontstyle='normal',fontweight='bold',color='black')
            for pair,record in enumerate(records[group*3:(group+1)*3]):
                target=record['target'];data=record['models'][model]
                cam_path=args.result/(target+'_'+model+'_cam.npy');rgb_path=args.result/(target+'_'+model+'_rgb.npy')
                for path,key in [(cam_path,'cam_sha256'),(rgb_path,'rgb_sha256')]:
                    if hashlib.sha256(path.read_bytes()).hexdigest()!=data[key]:
                        raise RuntimeError('Attribution source changed: '+str(path))
                cam=np.load(cam_path);rgb=np.load(rgb_path)
                if cam.shape!=(7,7) or rgb.shape!=(224,224,3) or not np.isfinite(cam).all():
                    raise RuntimeError('Unexpected attribution shape')
                column=1+pair*2
                original=fig.add_subplot(grid[row,column]);original.imshow(rgb);original.axis('off')
                overlay=fig.add_subplot(grid[row,column+1]);overlay.imshow(rgb)
                overlay.imshow(cam,cmap='jet',vmin=0,vmax=max(float(cam.max()),1e-12),alpha=.48,
                               extent=(-.5,223.5,223.5,-.5),interpolation='bilinear')
                overlay.axis('off')
                if row==0:
                    a,b=original.get_position(),overlay.get_position()
                    fig.text((a.x0+b.x1)/2,max(a.y1,b.y1)+.012,names.get(target,target),
                             ha='center',va='bottom',fontsize=18,color='black')
                checks.append(dict(target=target,model=model,panels=['rgb','gradient'],
                                   positive_max=float(cam.max()),episode_id=data['episode_id']))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180,facecolor='white');plt.close(fig)
    receipt=dict(page=args.page,rows=4,columns=6,panels=24,rgb_panels=12,gradient_panels=12,
                 goal_labels=6,model_labels='bold, upright, left aligned',
                 layout='three adjacent RGB/gradient pairs per row; one shared goal heading per pair',
                 targets=[r['target'] for r in records],comparison_mode=summary['comparison_mode'],
                 source_summary_sha256=hashlib.sha256((args.result/'summary.json').read_bytes()).hexdigest(),
                 image_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),panels_verified=checks)
    args.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Rendered page',args.page,'with 4 x 6 RGB/gradient panels:',args.output)


if __name__=='__main__':main()
