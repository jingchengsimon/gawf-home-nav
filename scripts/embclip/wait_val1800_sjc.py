"""Finalize the seed0/1 EmbCLIP comparison when all eight val runs finish."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "experiments/monitoring/jobs"
OUT = ROOT / "results/embclip-val1800-seed01-20261022"


def main():
    paths = sorted(JOBS.glob("nav-robothor-embclip-light-*-eval-val1800-sjc-v1-seed[01]-20261022.json"))
    if len(paths) != 8:
        raise RuntimeError(f"Expected eight evaluation manifests, found {len(paths)}")
    previous = None
    deadline = time.monotonic() + 8 * 3600
    while time.monotonic() < deadline:
        states = {path.stem: json.loads(path.read_text())["status"] for path in paths}
        current = tuple(states.items())
        if current != previous:
            print(datetime.now().astimezone().isoformat(), states, flush=True)
            previous = current
        if any(status == "failed" for status in states.values()):
            raise RuntimeError("Evaluation failed; inspect the corresponding manifest and log")
        if all(status in ("complete", "partial") for status in states.values()):
            subprocess.run([sys.executable, "-B", "scripts/embclip/compare_val1800_sjc.py"],
                           cwd=ROOT, check=True)
            comparison = OUT / "comparison.json"
            report = json.loads(comparison.read_text())
            receipt = dict(status="complete_with_partial_coverage"
                           if any(status == "partial" for status in states.values()) else "complete",
                           finished_at=datetime.now().astimezone().isoformat(),
                           evaluation_count=8, common_episode_count=report["common_episode_count"],
                           comparison_sha256=hashlib.sha256(comparison.read_bytes()).hexdigest())
            (OUT / "completion.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            print(receipt, flush=True)
            return
        time.sleep(60)
    raise TimeoutError("Eight evaluations did not finish within eight hours")


if __name__ == "__main__":
    main()
