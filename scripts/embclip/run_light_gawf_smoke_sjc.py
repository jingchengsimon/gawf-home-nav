"""Run one registered SJC GaWF smoke and record its full PPO checkpoint."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess


ROOT = Path(__file__).resolve().parents[2]
MODELS = {"gawf1", "gawf3"}
STEP_RE = re.compile(r"__steps_(\d+)\.pt$")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    args = parser.parse_args()
    path = ROOT / "experiments/monitoring/jobs" / (args.experiment_id + ".json")
    run = json.loads(path.read_text())
    model = run["model"]["family"]
    execution = run["execution"]
    assert run["experiment_id"] == args.experiment_id and run["host"] == "sjc-remote"
    assert run["mode"] == "train-smoke" and model in MODELS
    assert execution["steps_budget"] == 2048 and execution["global_train_envs"] == 16
    assert execution["checkpoint_interval_steps"] == 2048
    assert execution["physical_gpu_index"] in (0, 1)
    for relative, expected in run["source"]["source_files"].items():
        assert sha256(ROOT / relative) == expected, relative
    output = ROOT / run["units"][0]["result_path"]
    assert output.is_dir() and not (output / "exit.json").exists()
    for relative, expected in run["source"]["source_files"].items():
        assert sha256(output / "config_snapshot" / relative) == expected, relative
    command = [str(ROOT / ".envs/embclip-robothor/bin/python"), "-u",
               "vendor/embclip-allenact/allenact/main.py",
               f"scripts/embclip/light_{model}_smoke_sjc_config.py", "-b", ".",
               "--seed", "0", "--config_kwargs", json.dumps(dict(
                   headless=True, train_gpu_ids=[0], num_train_processes=16,
                   target_steps=2048, checkpoint_interval=2048)),
               "-o", str(output / "allenact")]
    env = dict(os.environ)
    env.update(HOME=str(ROOT / "assets/embclip-robothor/home"),
               PATH=str(ROOT / "assets/embclip-robothor/vulkan-tools/package/usr/bin") + ":" + env["PATH"],
               PYTHONPATH=str(ROOT / "vendor/embclip-allenact") + ":" + str(ROOT),
               CUDA_VISIBLE_DEVICES=str(execution["physical_gpu_index"]), PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("DISPLAY", None)
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (output / "train.log").open("wb") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return_code = process.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return_code = 124
    checkpoints = [p for p in (output / "allenact/checkpoints").rglob("*.pt")
                   if STEP_RE.search(p.name)]
    latest = max(checkpoints, key=lambda p:int(STEP_RE.search(p.name).group(1))) if checkpoints else None
    steps = int(STEP_RE.search(latest.name).group(1)) if latest else 0
    finished = datetime.now().astimezone().isoformat()
    success = return_code == 0 and steps >= 2048 and b"Training finished successfully." in (
        output / "train.log").read_bytes()
    receipt = dict(return_code=return_code, checkpoint_steps=steps,
                   checkpoint_path=str(latest.relative_to(ROOT)) if latest else None,
                   checkpoint_sha256=sha256(latest) if latest else None,
                   finished_at=finished, status="complete" if success else "failed")
    (output / "exit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    run["status"] = receipt["status"]
    run["units"][0]["status"] = receipt["status"]
    run["completion_evidence"] = dict(global_frames=steps, completed_updates=int(success),
                                      exit_code=return_code,
                                      checkpoint_sha256=receipt["checkpoint_sha256"])
    path.write_text(json.dumps(run, indent=2) + "\n")
    subprocess.run([str(ROOT / ".envs/embclip-robothor/bin/python"),
                    "scripts/rebuild_index.py"], cwd=ROOT, check=True)
    print(json.dumps(receipt), flush=True)
    if not success:
        raise RuntimeError(f"GaWF smoke failed: {args.experiment_id}")


if __name__ == "__main__":
    main()
