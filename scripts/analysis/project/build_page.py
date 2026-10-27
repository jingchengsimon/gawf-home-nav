"""Prepare label-free video assets and the minimal project-page data."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
PAGE = ROOT / "project-page"
DIST = PAGE / "dist"


def prepare(selection):
    demos = []
    for item in selection["demos"]:
        models = []
        for row in item["models"]:
            source = ROOT / row["source_video"]
            source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if row.get("source_sha256") and source_digest != row["source_sha256"]:
                raise RuntimeError(f"Recorded source video changed: {source}")
            asset_key = hashlib.sha256((source_digest + row.get("crop", "")).encode()).hexdigest()[:12]
            dest = DIST / "assets/minimal-videos" / item["episode_id"] / (row["model"] + "-" + asset_key + ".mp4")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(source)]
                if row.get("crop"):
                    command += ["-vf", row["crop"], "-c:v", "libx264", "-crf", "18", "-preset", "fast"]
                else:
                    command += ["-c:v", "copy"]
                command += ["-an", "-movflags", "+faststart", str(dest)]
                subprocess.run(command, check=True)
            poster = dest.with_suffix(".png")
            if not poster.exists():
                subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(dest),
                                "-vf", "select=eq(n\\,0)", "-frames:v", "1", str(poster)], check=True)
            final = dest.with_name(dest.stem + "-path.png")
            if not final.exists():
                subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(dest),
                                "-vf", f"select=eq(n\\,{row['steps']}),crop=500:374:0:0", "-frames:v", "1", str(final)], check=True)
            models.append({"model": row["model"], "steps": row["steps"],
                           "asset": str(dest.relative_to(DIST)), "poster": str(poster.relative_to(DIST)),
                           "final_path": str(final.relative_to(DIST))})
        demos.append({"episode_id": item["episode_id"], "target": item["target"], "models": models})
    attribution = []
    for row in selection.get("attribution", []):
        asset = row["asset"].split("?")[0]
        fingerprint = hashlib.sha256((DIST / asset).read_bytes()).hexdigest()[:12]
        attribution.append({**row, "asset": asset + "?v=" + fingerprint})
    data = {"demos": demos, "attribution": attribution}
    (DIST / "assets/project-data.js").write_text("window.PROJECT = " + json.dumps(data, separators=(",", ":")) + ";\n")
    # Full experimental detail stays inspectable without appearing in the layout.
    (DIST / "assets/sources/protocol.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    print(f"Prepared {len(demos)} paired demos")


if __name__ == "__main__":
    prepare(json.loads((PAGE / "sources/minimal_selection.json").read_text()))
