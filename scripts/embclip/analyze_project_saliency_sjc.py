"""Offline action log-probability Grad-CAM on identical RGB/action histories.

This is an independent diagnostic, not Codebook's unreleased analysis code.
The 7x7 frozen CLIP feature map is the attribution layer. No weights are updated.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "vendor/embclip-allenact")]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from allenact.base_abstractions.misc import Memory
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--suite-id", required=True)
    args = parser.parse_args()
    manifest_path = ROOT / "experiments/monitoring/jobs" / (args.experiment_id + ".json")
    manifest = json.loads(manifest_path.read_text())
    out = Path(__file__).resolve().parent
    source = ROOT / "results" / args.suite_id
    suite_manifest = json.loads((ROOT / "experiments/monitoring/jobs" / (args.suite_id + ".json")).read_text())
    if (manifest["status"] != "prepared" or suite_manifest["status"] != "complete"
            or manifest["runner_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            or out != ROOT / manifest["result_path"]):
        raise RuntimeError("Analysis identity mismatch")
    spec = importlib.util.spec_from_file_location("project_rollouts", source / "runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest["status"] = "running"
    module.write(manifest_path, manifest)
    torch.set_num_threads(1)
    bundles = {r["model"]: module.load_bundle(r) for r in suite_manifest["policies"]}
    records = []
    def detached(model, memory):
        return Memory({name:(memory.tensor(name).detach().clone(), [d[0] for d in dims].index("sampler"))
                       for name,(dims,_) in model.recurrent_memory_specification.items()})
    try:
        for e in suite_manifest["episodes"]:
            history_path = source / e["episode_id"] / "shared_history.npz"
            history = np.load(history_path)
            chosen_steps = sorted(set([0, min(16, len(history["rgb"])-1)]))
            cams = {t:{} for t in chosen_steps}
            for model_name, (model, graph) in bundles.items():
                memory = module.memory_for(model)
                prev = torch.zeros((1,1), dtype=torch.long, device="cuda:0")
                mask = torch.zeros((1,1,1), device="cuda:0")
                for t in range(max(chosen_steps)+1):
                    obs = {"rgb_lowres": torch.as_tensor(history["rgb"][t], device="cuda:0").unsqueeze(0),
                           "goal_object_type_ind": torch.as_tensor(history["goal"][t], device="cuda:0").unsqueeze(0)}
                    with torch.no_grad():
                        processed = graph.get_observations(obs)
                    feature = processed["rgb_clip_resnet"].detach().requires_grad_(t in chosen_steps)
                    processed["rgb_clip_resnet"] = feature
                    if t in chosen_steps:
                        with torch.no_grad():
                            reference, _ = model({k:v.unsqueeze(0) for k,v in processed.items()}, detached(model,memory), prev, mask)
                            reference_logits = reference.distributions.logits.detach()
                    # cuDNN's inference RNN does not retain backward reserve data.
                    # Use the equivalent native GRU only in this offline diagnostic.
                    with torch.backends.cudnn.flags(enabled=False), torch.set_grad_enabled(t in chosen_steps):
                        result, memory = model({k:v.unsqueeze(0) for k,v in processed.items()}, memory, prev, mask)
                        action = int(history["actions"][t])
                        if t in chosen_steps:
                            forward_delta = float((reference_logits-result.distributions.logits.detach()).abs().max().item())
                            if forward_delta > 1e-3:
                                raise RuntimeError("Native diagnostic forward differs from inference backend")
                            score = result.distributions.logits[..., action].sum()
                            gradient = torch.autograd.grad(score, feature)[0]
                            cam = (feature * gradient.mean(dim=(-2,-1), keepdim=True)).sum(dim=1).relu()[0]
                            cams[t][model_name] = {"map": cam.detach().cpu().numpy(),
                                                 "gradient_l2": float(gradient.norm().item()),
                                                 "native_vs_inference_logprob_max_delta": forward_delta,
                                                 "action_log_probability": float(score.item())}
                    memory = detached(model, memory)
                    prev.fill_(int(history["actions"][t]))
                    mask.fill_(1)
            for t in chosen_steps:
                rgb = history["rgb"][t].copy()
                means = np.array((0.48145466,0.4578275,0.40821073))
                stds = np.array((0.26862954,0.26130258,0.27577711))
                rgb = np.clip(rgb*stds+means,0,1)
                vmax = max(float(cams[t][m]["map"].max()) for m in bundles)
                fig, axes = plt.subplots(1,3,figsize=(9.6,3.65))
                axes[0].imshow(rgb); axes[0].set_title("Same policy RGB input",fontsize=11)
                for ax,m,title in zip(axes[1:],("gru1","gawf1"),("GRU1 · seed 1","GaWF1 · seed 3")):
                    ax.imshow(rgb)
                    cam = cams[t][m]["map"]
                    ax.imshow(cam, cmap="inferno", vmin=0, vmax=max(vmax,1e-12), alpha=.58,
                              extent=(-.5,223.5,223.5,-.5), interpolation="bilinear")
                    ax.set_title(title,fontsize=11,color={"gru1":"#1f77b4","gawf1":"#ff7f0e"}[m])
                for ax in axes:ax.axis("off")
                action = int(history["actions"][t])
                fig.suptitle(e["episode_id"]+f" · step {t} · action {action}",fontsize=10,y=.97)
                fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.86,wspace=.02)
                filename=e["episode_id"]+f"_step{t}.png"
                fig.savefig(out/filename,dpi=170,facecolor="white");plt.close(fig)
                for m in bundles:np.save(out/(e["episode_id"]+f"_step{t}_{m}.npy"),cams[t][m]["map"])
                records.append({"episode_id":e["episode_id"],"step":t,"action_index":action,"plot":filename,
                    "shared_history_sha256":module.sha(history_path),"shared_positive_cam_max":vmax,
                    "models":{m:{k:v for k,v in cams[t][m].items() if k!="map"} for m in bundles}})
        summary={"status":"complete","method":"Grad-CAM of selected action log probability w.r.t. frozen CLIP RN50 final 2048x7x7 feature map",
                 "target_action_source":"action actually sampled by GRU1 seed1 rollout seed0; both models receive the same action target",
                 "history":"both models teacher-forced through the same GRU1 RGB, goal and prior-action history; separate recurrent states reset to zero",
                 "normalization":"one shared positive CAM scale per frame across both models; zero maps remain zero",
                 "causal_limit":"local gradient sensitivity, not attention weights or evidence of causal mechanism",
                 "weights_updated":False,"records":records}
        module.write(out/"summary.json",summary)
        manifest["status"]="complete";manifest["units"][0]["status"]="complete"
        manifest["completion_evidence"]={"return_code":0,"summary_sha256":module.sha(out/"summary.json")}
        module.write(manifest_path,manifest)
    except Exception as error:
        manifest["status"]="failed";manifest["failure"]=repr(error);module.write(manifest_path,manifest);raise


if __name__ == "__main__":main()
