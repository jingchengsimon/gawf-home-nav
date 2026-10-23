"""Create a deterministic, separately recorded RoboTHOR validation smoke subset."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    assert args.episodes > 0
    source_files = sorted((args.source / "episodes").glob("FloorPlan_Val*.json.gz"))
    assert len(source_files) == 15, len(source_files)
    assert not args.output.exists(), args.output
    (args.output / "episodes").mkdir(parents=True)
    base, extra = divmod(args.episodes, len(source_files))
    receipt = {"source": str(args.source), "output": str(args.output),
               "requested_episodes": args.episodes, "scenes": []}
    for index, source in enumerate(source_files):
        with gzip.open(source, "rt") as handle:
            original = json.load(handle)
        selected = sorted(original, key=lambda ep: ep["id"])[:base + (index < extra)]
        assert len(selected) == base + (index < extra)
        scene = source.name[:-8]
        assert all(ep["scene"] == scene for ep in selected)
        target = args.output / "episodes" / source.name
        with gzip.open(target, "wt") as handle:
            json.dump(selected, handle)
        receipt["scenes"].append({
            "scene": scene,
            "count": len(selected),
            "source_sha256": digest(source),
            "subset_sha256": digest(target),
            "episode_ids": [ep["id"] for ep in selected],
        })
    assert sum(row["count"] for row in receipt["scenes"]) == args.episodes
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"scenes": len(source_files), "episodes": args.episodes,
                      "receipt": str(args.output / "receipt.json")}))


if __name__ == "__main__":
    main()
