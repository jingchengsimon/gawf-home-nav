"""Run one registered inference unit with a bounded process group and an exit receipt."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id")
    parser.add_argument("unit", choices=["pointnav2020", "imagenav", "robothor"])
    parser.add_argument("--seconds", type=int, default=1200)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "experiments/monitoring/jobs" / (args.experiment_id + ".json")).read_text())
    assert manifest["experiment_id"] == args.experiment_id
    unit = next(u for u in manifest["units"] if u["name"] == args.unit)
    output = Path(unit["result_path"])
    assert not output.exists(), output
    output.parent.mkdir(parents=True, exist_ok=True)
    log = Path(unit["log_path"])
    log.parent.mkdir(parents=True, exist_ok=True)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    assert command and 1 <= args.seconds <= 1800
    state_path = output.parent / (output.name + ".process.json")
    receipt_path = output.parent / (output.name + ".exit.json")
    state = {"experiment_id": args.experiment_id, "unit": args.unit, "command": command,
             "controller_pid": os.getpid(), "started_unix": time.time(), "timeout_seconds": args.seconds}
    with log.open("x") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        state["worker_pid"] = process.pid
        state_path.write_text(json.dumps(state, indent=2))
        timeout = False
        try:
            code = process.wait(timeout=args.seconds)
        except subprocess.TimeoutExpired:
            timeout = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
    state.update({"exit_code": code, "timed_out": timeout, "finished_unix": time.time()})
    receipt_path.write_text(json.dumps(state, indent=2))
    raise SystemExit(code if code else int(timeout))


if __name__ == "__main__":
    main()
