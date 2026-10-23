"""Read completed light GRU1/GaWF1 logs on SJC; emit a source-hashed snapshot."""
import hashlib
import json
from pathlib import Path
import re
from datetime import datetime

ROOT = Path.cwd()
rows = []
for model, seeds in {"gru1": (0, 1, 2, 3, 4), "gawf1": (0, 1, 2, 3, 4)}.items():
    for seed in seeds:
        date = "20261021" if seed < 2 else "20261022"
        eid = f"nav-robothor-embclip-light-{model}-formal5m-sjc-v1-seed{seed}-{date}"
        ids = [eid]
        if seed == 0:
            version = "v2" if model == "gru1" else "v1"
            ids.insert(0, f"nav-robothor-embclip-light-{model}-formal1m-sjc-{version}-seed0-20261020")
        points, sources = {}, []
        for identity in ids:
            for p in sorted((ROOT / "results" / identity).glob("**/train.log")):
                content = p.read_text(errors="replace")
                found = 0
                for line in content.splitlines():
                    match = re.search(r"TRAIN:\s+(\d+) rollout steps", line)
                    if match is None:
                        continue
                    step = int(match.group(1))
                    values = {}
                    for metric in ("success", "spl"):
                        m = re.search(r"\b" + metric + r"\s+([0-9.eE+-]+)", line)
                        if m is None:
                            raise RuntimeError(f"Missing {metric} in {p}")
                        values[metric] = float(m.group(1))
                    points[step] = {"step": step, **values}
                    found += 1
                if found:
                    sources.append({"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
        if not points or max(points) != 5001216:
            raise RuntimeError(f"Incomplete source curve {model} {seed}")
        rows.append({"model": model, "seed": seed, "experiment_ids": ids,
                     "sources": sources, "points": [points[s] for s in sorted(points)]})
print(json.dumps({"observed_at": datetime.now().astimezone().isoformat(),
                  "scope": "training completed episode means from AllenAct TRAIN records; not validation",
                  "runs": rows}, indent=2, sort_keys=True))
