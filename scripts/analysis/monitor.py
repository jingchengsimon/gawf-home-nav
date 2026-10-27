"""Show running evaluation and analysis jobs, keeping training jobs separate."""
import json
from pathlib import Path
import re
from scripts.training import monitor
ROOT = Path(__file__).resolve().parents[2]
EVAL_COMPLETE = re.compile(rb"\bnum_test_tasks_complete\s+([0-9]+)")

def last_embclip_eval_episodes(path, max_episodes=None):
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        stream.seek(0, 2)
        end = stream.tell()
        stream.seek(max(0, end - 262144))
        tail = stream.read()
    if end > len(tail):
        tail = tail.partition(b"\n")[2]
    if not tail.endswith(b"\n"):
        tail = tail.rpartition(b"\n")[0]
    values = (int(match.group(1)) for line in tail.splitlines()
              if (match := EVAL_COMPLETE.search(line)))
    return max((value for value in values
                if max_episodes is None or value <= max_episodes), default=None)




def row(run, slurm, tmux):
    evidence = run.get("completion_evidence") or {}
    execution = run.get("execution") or {}
    result = Path(run["units"][0]["result_path"]).expanduser()
    if not result.is_absolute():
        result = ROOT / result
    progress_path = result / "progress.json"
    summary_path = result / "summary.json"
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    evaluation = run.get("mode") == "checkpoint_eval"
    target = (run.get("expected_episodes") or progress.get("expected_cases")
              or progress.get("expected_attempts") or run.get("expected_units"))
    frames = max(summary.get("episode_count") or summary.get("completed_episodes") or 0,
                 progress.get("completed_episodes") or progress.get("completed_cases")
                 or progress.get("completed_attempts") or 0, evidence.get("episode_count") or 0,
                 len(summary.get("records", summary.get("pairs", []))))
    logged = last_embclip_eval_episodes(result / "eval.log", target)
    frames = max(frames, logged or 0)
    saved = 0
    status = run["status"]
    detail = ""
    job = execution.get("job_id")
    session = execution.get("session")
    if summary.get("status") == "complete" and target and frames >= target:
        status = "complete"
        detail = "summary verified"
    elif status == "paused":
        detail = "paused by user"
    elif job:
        scheduler = slurm.get(str(job))
        status = scheduler["state"] if scheduler else "NOT_IN_SQUEUE"
        detail = scheduler["reason"] if scheduler else "check result/Slurm accounting"
    elif session and run["status"] == "queued":
        status = "queued"
        detail = "waiting in tmux " + session
    elif session:
        status = tmux.get(session, "inactive")
        detail = "tmux " + session
    model = run.get("model")
    if isinstance(model, dict):
        model = model.get("family")
    if not model:
        match = re.search(r"-embclip-([a-z0-9]+)-", run["experiment_id"])
        model = match.group(1) if match else "model?"
    return dict(experiment_id=run["experiment_id"], model=model,
                seed=run.get("seed", execution.get("seed", 0)),
                status=status, detail=detail, global_frames=int(frames),
                target_global_frames=target, saved_global_frames=int(saved),
                job_id=job, session=session,
                unit="episodes" if evaluation else "cases", evaluation=True)



def main():
    monitor.main(analysis=True, row_reader=row)


if __name__ == "__main__":
    main()
