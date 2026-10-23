"""Read the two registered SJC EmbCLIP training runs without changing them."""

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path


RUN_IDS = (
    "nav-robothor-embclip-gru3-formal-200m-sjc-v1-seed0-20261020",
    "nav-robothor-embclip-gawf3-formal-200m-sjc-v1-seed0-20261020",
)
STEP_RE = re.compile(rb"TRAIN:\s+([0-9]+) rollout steps")
CHECKPOINT_RE = re.compile(r"__steps_([0-9]+)\.pt$")


def latest_step(log_path):
    try:
        with log_path.open("rb") as stream:
            stream.seek(0, 2)
            end = stream.tell()
            stream.seek(max(0, end - 262144))
            tail = stream.read()
    except OSError:
        return None
    if end > len(tail):
        tail = tail.partition(b"\n")[2]
    if not tail.endswith(b"\n"):
        tail = tail.rpartition(b"\n")[0]
    for line in reversed(tail.splitlines()):
        match = STEP_RE.search(line)
        if match:
            return int(match.group(1))
    return None


def latest_checkpoint(result_path):
    checkpoint_root = result_path / "allenact" / "checkpoints"
    steps = (
        int(match.group(1))
        for path in checkpoint_root.rglob("*.pt")
        if (match := CHECKPOINT_RE.search(path.name))
    )
    return max(steps, default=None)


def tmux_sessions():
    check = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return set(check.stdout.splitlines()) if check.returncode == 0 else set()


def snapshot(root):
    sessions = tmux_sessions()
    jobs = []
    for run_id in RUN_IDS:
        manifest_path = root / "experiments" / "monitoring" / "jobs" / f"{run_id}.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["experiment_id"] == run_id and manifest["host"] == "sjc-remote"
        assert manifest["mode"] == "formal_train"
        unit = manifest["units"][0]
        result_path = Path(unit["result_path"])
        log_path = Path(unit["log_path"])
        session = manifest["execution"]["session"]
        target = manifest["execution"]["target_steps"]
        step = latest_step(log_path)
        checkpoint = latest_checkpoint(result_path)
        exit_path = result_path / "exit.json"
        exit_code = json.loads(exit_path.read_text())["return_code"] if exit_path.is_file() else None
        if manifest.get("status") == "paused":
            state = "paused"
        elif exit_code is not None:
            state = "complete" if exit_code == 0 and checkpoint is not None and checkpoint >= target else "exited"
        else:
            state = "running" if session in sessions else "inactive"
        jobs.append({
            "experiment_id": run_id,
            "model": manifest["model"]["family"] if "model" in manifest else run_id.split("-embclip-")[1].split("-")[0],
            "session": session,
            "state": state,
            "steps": step,
            "target_steps": target,
            "checkpoint_steps": checkpoint,
            "exit_code": exit_code,
            "log_path": str(log_path),
            "log_mtime": datetime.fromtimestamp(log_path.stat().st_mtime).astimezone().isoformat() if log_path.is_file() else None,
        })
    return jobs


def render(jobs):
    lines = [f"EmbCLIP RoboTHOR on sjc-remote · {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}"]
    for job in jobs:
        step, target = job["steps"], job["target_steps"]
        fraction = min(step / target, 1) if step is not None else 0
        filled = max(1, int(16 * fraction)) if step else 0
        bar = "#" * filled + "." * (16 - filled)
        progress = f"{step:,}/{target:,} ({fraction:.2%})" if step is not None else f"?/{target:,}"
        checkpoint = f" · checkpoint {job['checkpoint_steps']:,}" if job["checkpoint_steps"] is not None else " · no checkpoint"
        lines.append(f"  {job['model'].upper()} seed0 [{bar}] {progress} steps · {job['state']}{checkpoint}")
        lines.append(f"    id: {job['experiment_id']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    jobs = snapshot(args.root)
    print(json.dumps(jobs, indent=2) if args.json else render(jobs))


if __name__ == "__main__":
    main()
