"""Launch or execute one frozen SJC media job with an actual exit receipt."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    mp = root / "experiments/monitoring/jobs" / (args.experiment_id + ".json")
    m = json.loads(mp.read_text())
    leaf = root / m["result_path"]
    inputs = m.get("input_files", {})
    if any(Path(name).name != name for name in inputs):
        raise RuntimeError("Input files must be leaf filenames")
    if m["status"] != "prepared" or {p.name for p in leaf.iterdir()} != {"runner.py", *inputs}:
        raise RuntimeError("Job or result leaf is not fresh")
    for name, digest in inputs.items():
        if hashlib.sha256((leaf / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Input file hash mismatch: " + name)
    if hashlib.sha256((leaf / "runner.py").read_bytes()).hexdigest() != m["runner_sha256"]:
        raise RuntimeError("Frozen runner hash mismatch")
    for relative, digest in m.get("source_files", {}).items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Source changed: " + relative)
    if not args.worker:
        session = m["execution"]["session"]
        if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
            raise RuntimeError("Session already exists")
        command = ["python3", str(Path(__file__).resolve()), "--root", str(root),
                   "--experiment-id", args.experiment_id, "--worker"]
        subprocess.run(["tmux", "new-session", "-d", "-s", session, shlex.join(command)], check=True)
        print("STARTED", session, flush=True)
        return
    gpu = int(m["execution"]["physical_gpu_index"])
    uuid, free = subprocess.check_output(["nvidia-smi", "--id=" + str(gpu),
        "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"], text=True).strip().split(",")
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                    "--format=csv,noheader"], text=True)
    if int(free) < 11000 or any(line.split(",")[0].strip() == uuid.strip() for line in apps.splitlines()):
        raise RuntimeError("Target GPU is occupied")
    if shutil.disk_usage(leaf).free < 5 * 1024**3:
        raise RuntimeError("Insufficient result space")
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), HOME=str(root / "assets/embclip-robothor/home"),
               PYTHONPATH=str(root / "vendor/embclip-allenact") + ":" + str(root),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PATH=str(root / "assets/embclip-robothor/vulkan-tools/package/usr/bin") + ":" + env["PATH"])
    env.pop("DISPLAY", None)
    with (leaf / "run.log").open("x") as log:
        completed = subprocess.run([str(root / ".envs/embclip-robothor/bin/python"), "-u",
                                   str(leaf / "runner.py"), "--experiment-id", args.experiment_id],
                                  cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    receipt = dict(return_code=completed.returncode, finished_at=datetime.now().astimezone().isoformat())
    (leaf / "exit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if completed.returncode:
        m = json.loads(mp.read_text())
        m["status"] = "failed"
        for unit in m.get("units", []):
            unit["status"] = "failed"
        mp.write_text(json.dumps(m, indent=2) + "\n")
    subprocess.run(["python3", "scripts/rebuild_index.py"], cwd=root, check=True)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
