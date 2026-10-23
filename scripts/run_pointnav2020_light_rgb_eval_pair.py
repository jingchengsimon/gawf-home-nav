"""Run the two registered RGB-only PointNav seed0 evaluations in GPU0 order."""

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path("/G/MIMOlab/Codes/gawf-home-nav")
IDS = [
    "nav-pointnav2020-gibsonv2-rgb-light-lstm2-evalval284-sjc-v1-seed0-20261023",
    "nav-pointnav2020-gibsonv2-rgb-light-gawf2-evalval284-sjc-v1-seed0-20261023",
]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def gpu0_free_mib():
    output = subprocess.check_output([
        "nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"
    ], text=True)
    return int(output.strip())


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    for experiment_id in IDS:
        manifest_path = ROOT / "experiments/monitoring/jobs" / (experiment_id + ".json")
        manifest = json.loads(manifest_path.read_text())
        result = ROOT / "results" / experiment_id
        assert manifest["status"] == "queued" and manifest["units"][0]["status"] == "queued"
        assert result.is_dir() and not (result / "summary.json").exists()
        assert not (result / "exit.attempt001.json").exists()
        # Leave a queued job untouched if another workload has occupied GPU0.
        free = gpu0_free_mib()
        if free < 11000:
            print(f"GPU0 has only {free} MiB free; {experiment_id} remains queued", flush=True)
            break
        env = dict(os.environ, PYTHONPATH=str(ROOT / "scripts") + os.pathsep +
                   str(ROOT / "vendor/habitat-lab-2020"),
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MAGNUM_LOG="quiet",
                   HABITAT_SIM_LOG="quiet", GLOG_minloglevel="2")
        command = [str(ROOT / ".envs/pointnav2020/bin/python"), "-u",
                   str(ROOT / "scripts/eval_pointnav2020_light_rgb.py"), experiment_id]
        with (result / "worker.log").open("x") as log:
            completed = subprocess.run(command, cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
        receipt = {"return_code": completed.returncode,
                   "completed_at": datetime.now().astimezone().isoformat()}
        write_json(result / "exit.attempt001.json", receipt)
        if completed.returncode != 0:
            manifest = json.loads(manifest_path.read_text())
            manifest["status"] = "failed"
            manifest["units"][0]["status"] = "failed"
            write_json(manifest_path, manifest)
            subprocess.run([sys.executable, "-B", str(ROOT / "scripts/rebuild_index.py")],
                           cwd=ROOT, check=True)
            print(f"Evaluation failed: {experiment_id}; see {result / 'worker.log'}", flush=True)


if __name__ == "__main__":
    main()
