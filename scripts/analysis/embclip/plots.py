"""One plotting entry point for validation, training, Grad-CAM and the teaser."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
DIST = ROOT / "project-page/dist"
SOURCE = ROOT / "results/nav-robothor-embclip-visible-gradcam-selection-local-v2-20261023"
COLORS = {"gru1": "#1f77b4", "gawf1": "#ff7f0e"}

from scripts.analysis.plotting import finish, clean, plotting



def validation(source, output):
    np, plt = plotting()
    d=json.loads(Path(source).read_text())
    if d["selected_seeds"] != {model: list(range(5)) for model in COLORS}:
        raise RuntimeError("Expected all five seeds for both models")
    for model in COLORS:
        if sorted(row["seed"] for row in d["runs"] if row["model"] == model) != list(range(5)):
            raise RuntimeError("Duplicate or missing seed")
    fig, axes=plt.subplots(1,2,figsize=(9.2,4.1))
    stats={}
    for ax,metric,title,ylabel in zip(axes,("common_sr","common_spl"),("Validation success","Validation path efficiency"),("SR (%)","SPL (%)")):
        for x,model in enumerate(("gru1","gawf1")):
            rows=[r for r in d["runs"] if r["model"]==model]
            vals=np.array([r[metric]*100 for r in rows])
            stats.setdefault(model,{})[metric]={"mean":float(vals.mean()),"seeds":[r["seed"] for r in rows],"values":vals.tolist()}
            sem=vals.std(ddof=1)/np.sqrt(len(vals))
            ax.bar(x,vals.mean(),width=.72,color=COLORS[model],yerr=sem,capsize=2.5,
                   error_kw={"elinewidth":1,"ecolor":"#333333"},zorder=2)
            jitter=np.random.default_rng(0).uniform(-.1584,.1584,len(vals))
            ax.scatter(x+jitter,vals,s=12,color="#333333",alpha=.58,zorder=3,linewidths=0)
        ax.set_xticks([0,1],["EmbCLIP","EmbCLIP-GaWF"],rotation=40,ha="right")
        ax.set_ylabel(ylabel);ax.set_title(title,pad=12);ax.set_xlim(-.55,1.55)
        ax.set_ylim(0,max(r[metric]*100 for r in d["runs"])+5);clean(ax)
    fig.suptitle("EmbCLIP · RoboTHOR · light 5M",fontsize=14,y=1.02)
    fig.tight_layout(w_pad=2.5)
    finish(fig,output)
    return stats



def training(source, output):
    np, plt = plotting()
    d=json.loads(Path(source).read_text())
    fig,axes=plt.subplots(1,2,figsize=(10.6,3.9))
    start=max(r["points"][0]["step"] for r in d["runs"])
    end=min(r["points"][-1]["step"] for r in d["runs"])
    grid=np.arange(start,end+1,10240)
    for ax,metric,ylabel in zip(axes,("success","spl"),("Training SR (%)","Training SPL (%)")):
        for model in ("gru1","gawf1"):
            arr=[]
            for row in d["runs"]:
                if row["model"]!=model:continue
                points=row["points"]
                values=np.interp(grid,[p["step"] for p in points],[p[metric] for p in points])*100
                arr.append(np.convolve(np.pad(values,(5,5),mode="edge"),np.ones(11)/11,mode="valid"))
            arr=np.array(arr);mean=arr.mean(0);std=arr.std(0)
            ax.plot(grid/1e6,mean,color=COLORS[model],lw=2,label={"gru1":"EmbCLIP","gawf1":"EmbCLIP-GaWF"}[model])
            ax.fill_between(grid/1e6,mean-std,mean+std,color=COLORS[model],alpha=.14,linewidth=0)
        ax.set_xlabel("Global training frames (millions)");ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0);ax.set_xlim(0,5.01);clean(ax);ax.legend(frameon=False,fontsize=10)
    fig.suptitle("RoboTHOR training · mean ± population SD",fontsize=14)
    fig.tight_layout(w_pad=2)
    finish(fig,output)



def gradcam(result, output, page=1):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args = SimpleNamespace(result=Path(result), output=Path(output), page=page)
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



def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()



def teaser():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D

    summary_path = SOURCE / 'summary.json'
    summary = json.loads(summary_path.read_text())
    assert summary['status'] == 'complete'
    assert json.loads((SOURCE / 'exit.json').read_text())['return_code'] == 0
    project = json.loads((DIST / 'assets/project-data.js').read_text().split(' = ', 1)[1].rstrip(';\n'))
    plant = next(row for row in project['demos'] if row['target'] == 'HousePlant')
    records = {row['target']: row for row in summary['records']}
    font = FontProperties(fname=DIST / 'assets/fonts/noto-sans-400.ttf', size=16)
    emphasis = FontProperties(fname=DIST / 'assets/fonts/noto-sans-700.ttf', size=22)
    fig = plt.figure(figsize=(14.8, 7.2), facecolor='white')
    grid = fig.add_gridspec(2, 5, width_ratios=[.60, 1.72, 1, 1, 1],
                           left=.003, right=.997, top=.84, bottom=.015,
                           wspace=.065, hspace=.08)
    checks = []
    first_row = []
    for row, model in enumerate(('gru1', 'gawf1')):
        label = fig.add_subplot(grid[row, 0]); label.axis('off')
        label.text(1, .5, 'EmbCLIP' if row == 0 else 'EmbCLIP-\nGaWF',
                   ha='right', va='center', fontproperties=emphasis, color='black')
        path = DIST / next(r for r in plant['models'] if r['model'] == model)['final_path']
        ax = fig.add_subplot(grid[row, 1]); ax.imshow(plt.imread(path)); ax.axis('off')
        ax.set_anchor('W')
        if row == 0:
            first_row.append(ax)
        checks.append(dict(model=model, kind='full_trajectory', episode_id=plant['episode_id'],
                           source=str(path.relative_to(ROOT)), sha256=digest(path)))
        for column, (target, title) in enumerate((('AlarmClock', 'Alarm Clock'),
                                                ('Mug', 'Mug'),
                                                ('SprayBottle', 'Spray Bottle')), start=2):
            record = records[target]['models'][model]
            cam_path = SOURCE / f'{target}_{model}_cam.npy'
            rgb_path = SOURCE / f'{target}_{model}_rgb.npy'
            assert digest(cam_path) == record['cam_sha256']
            assert digest(rgb_path) == record['rgb_sha256']
            cam, rgb = np.load(cam_path), np.load(rgb_path)
            assert cam.shape == (7, 7) and rgb.shape == (224, 224, 3)
            assert np.isfinite(cam).all() and np.isfinite(rgb).all()
            ax = fig.add_subplot(grid[row, column]); ax.imshow(rgb)
            ax.imshow(cam, cmap='jet', vmin=0, vmax=max(float(cam.max()), 1e-12),
                      alpha=.48, extent=(-.5, 223.5, 223.5, -.5), interpolation='bilinear')
            ax.axis('off')
            if row == 0:
                first_row.append(ax)
            checks.append(dict(model=model, kind='gradcam_overlay', target=target,
                               episode_id=record['episode_id'], cam_sha256=digest(cam_path),
                               rgb_sha256=digest(rgb_path)))
    # Align headings above panels despite their different aspect ratios.
    for ax, title in zip(first_row, ('House Plant', 'Alarm Clock', 'Mug', 'Spray Bottle')):
        bounds = ax.get_position()
        fig.text((bounds.x0+bounds.x1)/2, .86, title, ha='center', va='bottom', fontproperties=font)
    trajectory, first_cam, _, last_cam = [ax.get_position() for ax in first_row]
    fig.text((trajectory.x0+trajectory.x1)/2, .945, 'Trajectory',
             ha='center', va='bottom', fontproperties=emphasis)
    fig.text((first_cam.x0+last_cam.x1)/2, .945, 'Grad-CAM',
             ha='center', va='bottom', fontproperties=emphasis)
    divider = (trajectory.x1+first_cam.x0)/2
    fig.add_artist(Line2D([divider, divider], [.015, .985], transform=fig.transFigure,
                          color='#666', linewidth=1, linestyle=(0, (3, 3))))
    output = DIST / 'assets/teaser-summary.png'
    fig.savefig(output, dpi=180, facecolor='white'); plt.close(fig)
    receipt = dict(rows=2, visual_columns=4, label_columns=1, source_summary_sha256=digest(summary_path),
                   raw_rgb_columns=0, image_sha256=digest(output), panels=checks,
                   layout=dict(width_ratios=[.60, 1.72, 1, 1, 1],
                               group_titles=['Trajectory', 'Grad-CAM'], dashed_separator=True),
                   emphasis='Group headings and model labels: Noto Sans Bold, 22pt.',
                   normalization='Same as full figures: per-map positive maximum, jet overlay alpha 0.48.',
                   comparison='Paired HousePlant trajectory episode; Grad-CAM uses separate successful episodes and each model\'s own view.')
    (DIST / 'assets/sources/teaser-summary.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(output)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("validation", "training", "gradcam", "teaser"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--page", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.operation == "teaser":
        teaser()
        return
    if args.source is None or args.output is None:
        parser.error("--source and --output required")
    if args.operation == "gradcam":
        gradcam(args.source, args.output, args.page)
    else:
        result = {"validation": validation, "training": training}[args.operation](args.source, args.output)
        if result is not None:
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
