"""Start a registered GaWF seed-0 5M continuation after its 1M source stops."""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
STEP = re.compile(r"__steps_(\d+)\.pt$")


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def source_ready(source_id):
    manifest = json.loads((ROOT / "experiments/monitoring/jobs" / (source_id + ".json")).read_text())
    session = manifest["execution"]["session"]
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        return None
    processes = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True,
                               check=True).stdout
    if any(source_id in line for line in processes.splitlines()):
        return None
    result = ROOT / manifest["units"][0]["result_path"]
    checkpoints = []
    for path in (result / "allenact/checkpoints").rglob("*.pt"):
        match = STEP.search(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    if not checkpoints:
        return None
    steps, path = max(checkpoints)
    if steps < 1_000_000:
        return None
    import torch
    state = torch.load(path, map_location="cpu")
    required = {"model_state_dict", "optimizer_state_dict",
                "training_pipeline_state_dict", "trainer_seed", "scheduler_state"}
    if required - state.keys() or int(state["total_steps"]) != steps or int(state["trainer_seed"]) != 0:
        raise RuntimeError("Source checkpoint is incomplete or has the wrong seed")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(source_experiment_id=source_id, checkpoint_path=str(path.relative_to(ROOT)),
                sha256=digest, global_frames=steps)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: wait_light5m_sjc.py <experiment-id>")
    experiment_id = sys.argv[1]
    path = ROOT / "experiments/monitoring/jobs" / (experiment_id + ".json")
    manifest = json.loads(path.read_text())
    if (manifest["experiment_id"] != experiment_id or manifest["host"] != "sjc-remote"
            or manifest["status"] != "queued" or manifest["model"]["family"] not in ("gawf1", "gawf3")
            or manifest["execution"]["seed"] != 0
            or manifest["execution"]["steps_budget"] != 5_000_000):
        raise RuntimeError("Unexpected queued continuation")
    source_id = manifest["resume_after"]["source_experiment_id"]
    while True:
        prior = source_ready(source_id)
        if prior:
            manifest = json.loads(path.read_text())
            if manifest["status"] != "queued":
                raise RuntimeError("Queued status changed while waiting")
            manifest["resume_from"] = prior
            manifest["status"] = "running"
            manifest["units"][0]["status"] = "running"
            manifest["execution"]["session"] = manifest["execution"].pop("queue_session")
            manifest["execution"]["activated_at"] = datetime.now().astimezone().isoformat()
            manifest["completion_evidence"] = dict(global_frames=prior["global_frames"],
                                                    checkpoint_steps=prior["global_frames"])
            write_json(path, manifest)
            subprocess.run([sys.executable, "scripts/rebuild_index.py"], cwd=ROOT, check=True)
            os.execv(sys.executable, [sys.executable,
                     str(ROOT / "scripts/embclip/run_light_5m_sjc.py"), experiment_id])
        time.sleep(30)


if __name__ == "__main__":
    main()
