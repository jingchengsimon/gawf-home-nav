"""Run one registered RGB-only PointNav seed from scratch or its full checkpoint."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from run_pointnav2020_light_train import ports, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    args = parser.parse_args()
    manifest_path = ROOT / "experiments/monitoring/jobs" / (args.experiment_id + ".json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["experiment_id"] == args.experiment_id
    assert manifest["host"] == "sjc-remote" and manifest["mode"] == "formal_train"
    assert manifest["protocol_id"] == "pointnav2020-gibsonv2-light-rgb-fixedgoal-resnet18-v1"
    assert manifest["model"] in {"lstm2", "gawf2"} and manifest["seed"] == 0
    assert manifest["resources"]["gpus"] == 1
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(manifest["execution"]["physical_gpu_index"])
    assert manifest.get("resume_from") is None
    for relative, digest in manifest["source_files"].items():
        assert sha256(ROOT / relative) == digest, "Source mismatch: " + relative

    results = ROOT / "results" / args.experiment_id
    logs = ROOT / "logs" / args.experiment_id
    spec = {key: manifest[key] for key in ("experiment_id", "model", "seed", "budget", "protocol_id", "source_files")}
    if results.exists():
        assert json.loads((results / "run_spec.json").read_text()) == spec
        assert not (results / "summary.json").exists()
        assert (results / "checkpoints/interrupted.pth").is_file()
    else:
        assert not logs.exists()
        (results / "checkpoints").mkdir(parents=True)
        logs.mkdir()
        (results / "run_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
        snapshot = results / "config_snapshot"
        snapshot.mkdir()
        for name in ("task.yaml", "ddppo.yaml", "protocol.json"):
            shutil.copy2(ROOT / "configs/pointnav2020/light_rgb" / name, snapshot / name)

    attempt = len(list(logs.glob("worker.attempt*.log"))) + 1
    log_path = logs / ("worker.attempt%03d.log" % attempt)
    assert not log_path.exists()
    launch_port, trainer_port = ports()
    env = dict(os.environ, PYTHONPATH=str(ROOT / "vendor/habitat-lab-2020") + os.pathsep + str(ROOT / "scripts"),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", GLOO_SOCKET_IFNAME="lo", NCCL_SOCKET_IFNAME="lo",
               MAGNUM_LOG="quiet", HABITAT_SIM_LOG="quiet", GLOG_minloglevel="2",
               MASTER_ADDR="127.0.0.1", MASTER_PORT=str(launch_port),
               POINTNAV_DDPPO_MASTER_PORT=str(trainer_port))
    budget = manifest["budget"]
    command = [str(ROOT / ".envs/pointnav2020/bin/python"), "-u", "-m", "torch.distributed.launch",
               "--use_env", "--nproc_per_node", "1", "--master_addr", "127.0.0.1",
               "--master_port", str(launch_port),
               str(ROOT / "scripts/train_pointnav2020_light_rgb_longrun.py"), args.experiment_id,
               "--formal", "--model", manifest["model"], "--seed", "0",
               "--target-steps", str(budget["target_global_frames"]),
               "--checkpoint-steps", str(budget["checkpoint_global_frames"])]
    (results / ("command.attempt%03d.json" % attempt)).write_text(json.dumps(command, indent=2) + "\n")
    started = time.monotonic()
    with log_path.open("w") as output:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    progress_path = results / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    checkpoint = results / "checkpoints/interrupted.pth"
    target = budget["target_global_frames"]
    success = (completed.returncode == 0 and progress.get("global_frames", 0) >= target
               and progress.get("saved_global_frames", 0) >= target
               and (results / "rank_0.complete.json").is_file() and checkpoint.is_file())
    receipt = {"return_code": completed.returncode, "complete": success,
               "global_frames": progress.get("global_frames"),
               "saved_global_frames": progress.get("saved_global_frames"),
               "elapsed_s": time.monotonic() - started}
    (results / ("exit.attempt%03d.json" % attempt)).write_text(json.dumps(receipt, indent=2) + "\n")
    if not success:
        if checkpoint.is_file():
            manifest["status"] = "recovering"
            manifest["completion_evidence"] = {"saved_global_frames": progress.get("saved_global_frames", 0),
                                               "last_attempt": attempt}
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise RuntimeError("RGB-only run stopped before target; inspect its full checkpoint")
    summary = {"status": "complete", "experiment_id": args.experiment_id,
               "global_frames": progress["global_frames"],
               "updates_completed": progress["updates_completed"],
               "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
               "elapsed_s": receipt["elapsed_s"]}
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest["status"] = "complete"
    manifest["units"][0]["status"] = "complete"
    manifest["completion_evidence"] = {"global_frames": summary["global_frames"],
                                       "checkpoint_sha256": summary["checkpoint_sha256"],
                                       "summary_path": str(results / "summary.json")}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run([str(ROOT / ".envs/pointnav2020/bin/python"), "-B",
                    str(ROOT / "scripts/rebuild_index.py")], cwd=ROOT, check=True)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
