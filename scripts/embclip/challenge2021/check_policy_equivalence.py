"""Compare the legacy loading adapter to original 5M constructors on SJC."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--manifest',required=True,type=Path)
    a=p.parse_args();root=a.root.resolve();m=json.loads(a.manifest.read_text());out=root/m['result_path']
    sys.path[:0]=[str(root),str(root/'vendor/embclip-allenact')]
    if hashlib.sha256((out/'policy_adapter.py').read_bytes()).hexdigest()!=m['adapter_sha256']:
        raise RuntimeError('Adapter source changed')
    from policy_adapter import model_bundle
    import numpy as np
    import torch
    from allenact.base_abstractions.misc import Memory
    from scripts.embclip.light5m_gru1_config import Light5MGRU1Config
    from scripts.embclip.light5m_gawf1_config import Light5MGaWF1Config
    torch.set_num_threads(1);torch.manual_seed(128)
    checks=[]
    def memory_for(model):
        rows={}
        for name,(dims,dtype) in model.recurrent_memory_specification.items():
            rows[name]=(torch.zeros(tuple(1 if n is None else n for _,n in dims),dtype=dtype,device='cuda:0'),[x[0] for x in dims].index('sampler'))
        return Memory(rows)
    for spec in m['policies']:
        adapter,graph,model=model_bundle(root,spec)
        config={'gru1':Light5MGRU1Config,'gawf1':Light5MGaWF1Config}[spec['model']](headless=True,test_on_validation=True)
        reference=config.create_model(sensor_preprocessor_graph=graph).cuda().eval()
        reference.load_state_dict(torch.load(root/spec['checkpoint_path'],map_location='cpu')['model_state_dict'],strict=True)
        if set(model.state_dict())!=set(reference.state_dict()):raise RuntimeError('Different state keys')
        for k,v in model.state_dict().items():
            if not torch.equal(v,reference.state_dict()[k]):raise RuntimeError('Different parameter '+k)
        for shape in [(300,400,3),(480,640,3)]:
            env=types.SimpleNamespace(current_frame=np.random.RandomState(17).randint(0,256,shape,dtype=np.uint8))
            for target in config.TARGET_TYPES:
                task=types.SimpleNamespace(task_info={'object_type':target})
                for x,y in zip(adapter.SENSORS,config.SENSORS):
                    if not np.array_equal(x.get_observation(env,task),y.get_observation(env,task)):
                        raise RuntimeError('Sensor preprocessing or label mapping differs')
        left,right=memory_for(model),memory_for(reference)
        prev=torch.zeros((1,1),dtype=torch.long,device='cuda:0');delta=0.0
        for step in range(4):
            obs={'rgb_clip_resnet':torch.randn(1,1,2048,7,7,device='cuda:0'),
                'goal_object_type_ind':torch.tensor([[step*3]],device='cuda:0')}
            mask=torch.full((1,1,1),0.0 if step in (0,3) else 1.0,device='cuda:0')
            with torch.no_grad():
                x,left=model(dict(obs),left,prev,mask);y,right=reference(dict(obs),right,prev,mask)
            for v,w in [(x.distributions.probs,y.distributions.probs),(x.values,y.values)]+[(left.tensor(k),right.tensor(k)) for k in model.recurrent_memory_specification]:
                delta=max(delta,float((v-w).abs().max()))
            prev.fill_(step)
        if delta>1e-6:raise RuntimeError('Policy numerical equivalence failed')
        checks.append(dict(model=spec['model'],seed=spec['seed'],checkpoint_sha256=spec['checkpoint_sha256'],
            state_parameters_identical=True,sensors_and_goal_mapping_identical=True,
            sequence_steps=4,reset_steps=[0,3],max_output_and_state_difference=delta))
        del model,reference,graph
        torch.cuda.empty_cache()
    receipt=dict(status='pass',adapter_sha256=m['adapter_sha256'],checks=checks,
        method='Original constructors versus architecture-only adapter; identical state dictionaries, RGB/category observations, action probabilities, critic outputs and all recurrent memories.',weights_updated=False)
    (out/'equivalence.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
