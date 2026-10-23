"""Compare action-log-probability gradients at each model's successful stop view."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/embclip-allenact')]

def main():
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from allenact.base_abstractions.misc import Memory
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-id',required=True)
    args=parser.parse_args()
    mp=ROOT/'experiments/monitoring/jobs'/(args.experiment_id+'.json')
    m=json.loads(mp.read_text());out=ROOT/m['result_path']
    if m['status']!='prepared' or out!=Path(__file__).resolve().parent or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=m['runner_sha256']:
        raise RuntimeError('Analysis identity mismatch')
    suite=ROOT/'results'/m['source_suite'];sm=json.loads((ROOT/'experiments/monitoring/jobs'/(m['source_suite']+'.json')).read_text())
    if sm['status']!='complete':raise RuntimeError('Source suite incomplete')
    if hashlib.sha256((suite/'runner.py').read_bytes()).hexdigest()!=sm['runner_sha256']:
        raise RuntimeError('Source recording runner changed')
    if hashlib.sha256((suite/'summary.json').read_bytes()).hexdigest()!=sm['completion_evidence']['summary_sha256'] or json.loads((suite/'exit.json').read_text())['return_code']!=0:
        raise RuntimeError('Source suite has no verified successful exit')
    spec=importlib.util.spec_from_file_location('terminal_rollouts',suite/'runner.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    m['status']='running';module.write(mp,m);torch.set_num_threads(1)
    bundles={r['model']:module.load_bundle(r) for r in sm['policies']}
    records=[]
    try:
        for case in m['cases']:
            eid=case['episode_id'];maps={}
            for name,(model,graph) in bundles.items():
                receipt=json.loads((suite/eid/(name+'_rollout0.json')).read_text())
                policy=next(r for r in sm['policies'] if r['model']==name)
                if receipt['model']!=name or receipt['training_seed']!=policy['seed'] or receipt['checkpoint_sha256']!=policy['checkpoint_sha256'] or receipt['episode']['id']!=eid:
                    raise RuntimeError('Probe policy or episode identity differs')
                probe=suite/eid/(name+'_rollout0_terminal_probe.pt')
                if module.sha(probe)!=receipt['terminal_probe_sha256'] or not receipt['success']:
                    raise RuntimeError('Expected an actual successful terminal observation')
                p=torch.load(probe,map_location='cpu')
                if not p['success'] or not p['target_visible'] or int(p['action'])!=3:
                    raise RuntimeError('Terminal probe is not a successful visible-target End decision')
                def memory():
                    return Memory({key:(p['memory'][key].cuda().clone(),[d[0] for d in dims].index('sampler')) for key,(dims,_) in model.recurrent_memory_specification.items()})
                obs={'rgb_lowres':torch.as_tensor(p['rgb'],device='cuda:0').unsqueeze(0),'goal_object_type_ind':torch.as_tensor(p['goal'],device='cuda:0').unsqueeze(0)}
                with torch.no_grad():processed=graph.get_observations(obs)
                feature=processed['rgb_clip_resnet'].detach().requires_grad_();processed['rgb_clip_resnet']=feature
                prev,mask=p['prev_actions'].cuda(),p['masks'].cuda()
                with torch.no_grad():ref,_=model({k:v.unsqueeze(0) for k,v in processed.items()},memory(),prev,mask)
                with torch.backends.cudnn.flags(enabled=False):
                    result,_=model({k:v.unsqueeze(0) for k,v in processed.items()},memory(),prev,mask)
                    delta=float((ref.distributions.logits-result.distributions.logits.detach()).abs().max())
                    if delta>1e-3:raise RuntimeError('Diagnostic native forward differs from ordinary inference')
                    score=result.distributions.logits[...,3].sum();grad=torch.autograd.grad(score,feature)[0]
                    cam=(feature*grad.mean((-2,-1),keepdim=True)).sum(1).relu()[0].detach().cpu().numpy()
                means=np.array((.48145466,.4578275,.40821073));std=np.array((.26862954,.26130258,.27577711))
                rgb=np.clip(p['rgb']*std+means,0,1)
                maps[name]=dict(rgb=rgb,cam=cam,probe_sha256=module.sha(probe),decision_step=p['decision_step'],gradient_l2=float(grad.norm()),native_forward_delta=delta)
                np.save(out/(eid+'_'+name+'.npy'),cam)
                np.save(out/(eid+'_'+name+'_rgb.npy'),rgb)
            vmax=max(float(r['cam'].max()) for r in maps.values())
            fig,axes=plt.subplots(2,2,figsize=(7.2,7.4))
            for row,name in enumerate(('gru1','gawf1')):
                r=maps[name];axes[row,0].imshow(r['rgb']);axes[row,1].imshow(r['rgb'])
                axes[row,1].imshow(r['cam'],cmap='inferno',vmin=0,vmax=max(float(r['cam'].max()),1e-12),alpha=.58,extent=(-.5,223.5,223.5,-.5),interpolation='bilinear')
                axes[row,0].set_title('EmbCLIP' if name=='gru1' else 'EmbCLIP-GaWF',fontsize=12,color='black',loc='left')
                for ax in axes[row]:ax.axis('off')
            fig.subplots_adjust(left=.01,right=.99,top=.96,bottom=.01,wspace=.02,hspace=.12)
            filename=eid+'.png';fig.savefig(out/filename,dpi=170,facecolor='white');plt.close(fig)
            records.append(dict(episode_id=eid,target=case['target'],plot=filename,raw_positive_max_by_model={name:float(r['cam'].max()) for name,r in maps.items()},models={name:{k:v for k,v in r.items() if k not in ('rgb','cam')} for name,r in maps.items()}))
        summary=dict(status='complete',records=records,method='End-action log-probability Grad-CAM at each model\'s own successful final decision view',comparison='same episode and initial pose; each model uses its own terminal RGB, prior action and actual recurrent state; visual inputs differ',normalization='each positive map divided by its own positive maximum for spatial visualization; brightness is not compared across models; zero maps remain zero',limits='local gradient sensitivity, not intrinsic attention or a causal comparison under identical inputs',weights_updated=False)
        module.write(out/'summary.json',summary);m['status']='complete';m['units'][0]['status']='complete';m['completion_evidence']={'return_code':0,'summary_sha256':module.sha(out/'summary.json')};module.write(mp,m)
    except Exception as e:
        m['status']='failed';m['failure']=repr(e);module.write(mp,m);raise

if __name__=='__main__':main()
