"""Start the registered GaWF1 seed2 job when SJC GPU1 has room."""

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
MIN_FREE_MIB = 11_000


def gpu1_free_mib():
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True)
    for line in query.stdout.splitlines():
        index, free = (part.strip() for part in line.split(",", 1))
        if int(index) == 1:
            return int(free)
    raise RuntimeError("GPU1 not found")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: wait_light5m_seed2_gpu1.py <experiment-id>")
    experiment_id = sys.argv[1]
    path = ROOT / "experiments/monitoring/jobs" / (experiment_id + ".json")
    while True:
        run = json.loads(path.read_text())
        execution = run["execution"]
        if (run["experiment_id"] != experiment_id or run["host"] != "sjc-remote"
                or run["status"] != "queued" or run["model"]["family"] != "gawf1"
                or execution["seed"] != 2 or execution["physical_gpu_index"] != 1
                or execution["steps_budget"] != 5_000_000):
            raise RuntimeError("Unexpected queued GaWF1 seed2 job")
        try:
            free = gpu1_free_mib()
        except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
            print(f"GPU1 capacity check unavailable: {exc}", flush=True)
            time.sleep(60)
            continue
        if free >= MIN_FREE_MIB:
            run["status"] = "running"
            run["units"][0]["status"] = "running"
            execution["activated_at"] = datetime.now().astimezone().isoformat()
            execution["gpu1_free_mib_at_activation"] = free
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(run, indent=2) + "\n")
            temporary.replace(path)
            subprocess.run([sys.executable, "scripts/rebuild_index.py"], cwd=ROOT, check=True)
            python = ROOT / ".envs/embclip-robothor/bin/python"
            os.execv(str(python), [str(python), "-u",
                                   str(ROOT / "scripts/embclip/run_light_5m_sjc.py"),
                                   experiment_id])
        time.sleep(60)


if __name__ == "__main__":
    main()
