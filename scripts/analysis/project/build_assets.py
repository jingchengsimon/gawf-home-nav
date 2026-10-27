"""Build project figures through the shared EmbCLIP and PointNav plotting backends."""
import argparse
import json
from pathlib import Path
from scripts.analysis.embclip import plots as embclip
from scripts.analysis.pointnav import plots as pointnav
ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "project-page/sources")
    parser.add_argument("--output", type=Path, default=ROOT / "project-page/dist/assets")
    args = parser.parse_args()
    statistics = embclip.validation(args.source_root / "embclip_validation.json", args.output / "embclip-validation")
    embclip.training(args.source_root / "embclip_training.json", args.output / "embclip-training")
    pointnav.validation(args.source_root / "pointnav_validation.json", args.output / "pointnav-validation")
    pointnav.training(args.source_root / "pointnav_training.json", args.output / "pointnav-training")
    statistics_path = (ROOT / "project-page/sources/display_statistics.json"
                       if args.output.resolve() == (ROOT / "project-page/dist/assets").resolve()
                       else args.output / "display_statistics.json")
    statistics_path.write_text(json.dumps(statistics, indent=2) + "\n")


if __name__ == "__main__":
    main()
