"""Register the six independently schedulable 10M-frame PointNav jobs."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = [
    "scripts/train_pointnav2020.py",
    "scripts/pointnav2020_runtime.py",
    "scripts/pointnav2020_compat.py",
    "scripts/pointnav2020_longrun.py",
    "scripts/pointnav2020_gawf.py",
    "scripts/run_pointnav2020_formal.py",
    "scripts/pointnav2020_formal_job.sh",
    "scripts/embclip/canonical_core/__init__.py",
    "scripts/embclip/canonical_core/gawf.py",
    "scripts/embclip/canonical_core/rnn.py",
    "configs/pointnav2020/task.yaml",
    "configs/pointnav2020/ddppo.yaml",
    "vendor/habitat-lab-2020/habitat_baselines/rl/ddppo/algo/ddppo_trainer.py",
    "vendor/habitat-lab-2020/habitat_baselines/rl/ddppo/algo/ddppo.py",
    "vendor/habitat-lab-2020/habitat_baselines/rl/ddppo/policy/resnet_policy.py",
    "vendor/habitat-lab-2020/habitat_baselines/rl/models/rnn_state_encoder.py",
]


def main():
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
               for name in SOURCE_FILES}
    folder = ROOT / "experiments/monitoring/jobs"
    for model in ("lstm2", "gawf2"):
        for seed in range(3):
            experiment_id = ("nav-pointnav2020-gibsonv2-ddppo-%s-formal10m-4gpu-v1-seed%d-20261020"
                             % (model, seed))
            path = folder / (experiment_id + ".json")
            if path.exists():
                raise FileExistsError(path)
            remote = "~/projects/gawf-home-nav"
            manifest = {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "host": "amarel", "mode": "formal_train", "status": "prepared",
                "model": model, "seed": seed,
                "purpose": "Gibson v2 PointNav 2020 RGB-D fixed-g0 DD-PPO 10M global frames",
                "remote_root": remote,
                "environment": {"python": "python3", "training_python": remote + "/.envs/pointnav2020/bin/python"},
                "execution": {"scheduler": "slurm", "job_id": None,
                              "batch_script": "scripts/pointnav2020_formal_job.sh",
                              "target_steps": 10000000, "dependency": None,
                              "resume": "SIGUSR1 600s before 72h -> full state -> scontrol requeue"},
                "resources": {"partition": "gpu", "nodes": 1, "gpus": 4,
                              "cpus": 24, "memory_gb": 120, "wall_time": "3-00:00:00"},
                "budget": {"target_global_frames": 10000000,
                           "checkpoint_global_frames": 10000,
                           "updates_ceiling": 50000,
                           "rollout_steps_per_env": 64,
                           "environments_per_gpu": 4},
                "expected_units": 10000000,
                "units": [{"name": "training", "status": "prepared",
                           "result_path": remote + "/results/" + experiment_id,
                           "log_path": remote + "/logs/" + experiment_id,
                           "training_log_path": remote + "/logs/" + experiment_id + "/train.log"}],
                "source_files": sources,
                "datasets": {"episodes": "Gibson PointNav v2", "scenes": "licensed Gibson trainval"},
                "prerequisite": {"lstm_smoke": "nav-pointnav2020-gibsonv2-ddppo-lstm2-smoke2gpu-v3-seed0-20261020",
                                 "gawf_smoke": "SJC formal-path train and resume checks"},
            }
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(experiment_id)


if __name__ == "__main__":
    main()
