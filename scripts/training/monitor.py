"""Show running navigation jobs on the current host without changing their state."""

import argparse
from datetime import datetime
import getpass
import json
import re
from pathlib import Path
import socket
import subprocess


ROOT = Path(__file__).resolve().parents[2]
ACTIVE = {"submitted", "pending", "queued", "running", "recovering", "paused"}
TRAIN_STEP = re.compile(rb"TRAIN:\s+([0-9]+) rollout steps")
CHECKPOINT_STEP = re.compile(r"__steps_(\d+)\.pt$")


def current_host():
    name = socket.gethostname().lower()
    if "sjc" in name:
        return "sjc-remote"
    if "amarel" in name:
        return "amarel"
    raise RuntimeError("Unknown host; pass --host sjc-remote or --host amarel")


def manifests(host):
    folder = ROOT / "experiments/monitoring/jobs"
    rows = []
    for path in sorted(folder.glob("*.json")):
        if path.name.startswith("._"):
            continue
        run = json.loads(path.read_text())
        if run.get("host") != host:
            continue
        execution = run.get("execution") or {}
        registered = execution.get("job_id") or execution.get("session")
        if run.get("status") not in ACTIVE and not (run.get("status") == "prepared" and registered):
            continue
        if run.get("experiment_id") != path.stem:
            raise ValueError("Manifest identity mismatch: " + str(path))
        rows.append(run)
    return rows


def scheduler_states(runs):
    ids = sorted({str(run.get("execution", {}).get("job_id"))
                  for run in runs if run.get("execution", {}).get("job_id")})
    try:
        query = subprocess.run(["squeue", "-h", "-u", getpass.getuser(),
                                "-o", "%i|%j|%T|%R"],
                               capture_output=True, text=True, timeout=12, check=True)
    except FileNotFoundError:
        return ({job: dict(state="UNKNOWN", reason="squeue unavailable on this host")
                 for job in ids}, [])
    states = {}
    unregistered = []
    for line in query.stdout.splitlines():
        fields = line.split("|", 3)
        if len(fields) == 4:
            job, name, state, reason = fields
            if job in ids:
                states[job] = dict(state=state, reason=reason)
            elif state.upper() == "RUNNING" and name.lower().startswith(("embclip-", "pointnav-", "nav-")):
                unregistered.append("Slurm %s %s %s" % (job, name, state))
    return states, unregistered


def tmux_states(runs):
    try:
        query = subprocess.run(["tmux", "list-panes", "-a", "-F",
                                "#{session_name}|#{pane_dead}"],
                               capture_output=True, text=True, timeout=10, check=False)
    except FileNotFoundError:
        return ({run["execution"]["session"]: "unknown (tmux unavailable)"
                 for run in runs if run.get("execution", {}).get("session")}, [])
    states = {}
    unregistered = []
    known = {run.get("execution", {}).get("session") for run in runs}
    if query.returncode == 0:
        for line in query.stdout.splitlines():
            fields = line.split("|", 1)
            if len(fields) == 2:
                states[fields[0]] = "running" if fields[1] == "0" else "exited"
                if (fields[1] == "0" and fields[0] not in known
                        and fields[0].startswith(("embclip-", "pointnav-", "nav-"))):
                    unregistered.append("tmux " + fields[0])
    return states, unregistered


def last_embclip_step(path):
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
    for line in reversed(tail.splitlines()):
        match = TRAIN_STEP.search(line)
        if match:
            return int(match.group(1))
    return None


def last_embclip_checkpoint(result):
    checkpoint_root = result / "allenact" / "checkpoints"
    if not checkpoint_root.is_dir():
        return 0
    return max((int(match.group(1)) for path in checkpoint_root.rglob("*.pt")
                if (match := CHECKPOINT_STEP.search(path.name))), default=0)


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
    target = ((run.get("budget") or {}).get("target_global_frames")
              or execution.get("target_steps") or execution.get("steps_budget"))
    frames = (summary.get("global_frames") or progress.get("global_frames")
              or evidence.get("global_frames") or 0)
    if not progress and not summary and run["experiment_id"].startswith("nav-robothor-embclip-"):
        logged = last_embclip_step(result / "train.log")
        if logged is not None:
            frames = logged
    saved = (progress.get("saved_global_frames") or evidence.get("checkpoint_steps")
             or evidence.get("saved_global_frames") or 0)
    if run["experiment_id"].startswith("nav-robothor-embclip-light-"):
        saved = max(saved, last_embclip_checkpoint(result))
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
                unit="frames", evaluation=False)



def sjc_display_groups(runs, rows):
    by_id = {item["experiment_id"]: item for item in rows}
    superseded = set()
    # A completed continuation is no longer in `runs`; still hide its old source.
    for path in (ROOT / "experiments/monitoring/jobs").glob("*.json"):
        if path.name.startswith("._"):
            continue
        run = json.loads(path.read_text())
        if run.get("host") != "sjc-remote":
            continue
        source = run.get("resume_from") or run.get("resume_after") or {}
        source_id = source.get("source_experiment_id")
        prior = by_id.get(source_id)
        execution = run.get("execution") or {}
        target = ((run.get("budget") or {}).get("target_global_frames")
                  or execution.get("target_steps") or execution.get("steps_budget"))
        if (prior and target and prior["target_global_frames"]
                and target > prior["target_global_frames"]):
            superseded.add(source_id)

    groups = {}
    for run, item in zip(runs, rows):
        if item["experiment_id"] in superseded:
            continue
        item = item.copy()
        source = run.get("resume_after") or {}
        prior = by_id.get(source.get("source_experiment_id"))
        if item["status"] == "queued" and prior:
            item["global_frames"] = prior["global_frames"]
            item["saved_global_frames"] = prior["saved_global_frames"]
            prior_target = prior["target_global_frames"]
            phase = f"{prior_target // 1_000_000}M" if prior_target and prior_target % 1_000_000 == 0 else "source"
            item["detail"] = phase + " phase " + prior["status"]
        experiment_id = item["experiment_id"]
        if experiment_id.startswith("nav-pointnav2020-"):
            task, name = 0, "PointNav Gibson v2"
        elif experiment_id.startswith("nav-robothor-embclip-"):
            task, name = 1, "EmbCLIP RoboTHOR"
        else:
            task, name = 2, "Navigation"
        target = item["target_global_frames"]
        budget = f"{target // 1_000_000}M" if target and target % 1_000_000 == 0 else f"{target:,}" if target else "?"
        date = experiment_id.rsplit("-", 1)[-1]
        batch = "%s-%s-%s" % (date[:4], date[4:6], date[6:]) if re.fullmatch(r"\d{8}", date) else date
        key = (task, batch, target, item["evaluation"])
        variant = "light " if "-light-" in experiment_id else ""
        description = (f"{variant}validation {budget} episodes" if item["evaluation"]
                       else f"{variant}{budget}")
        groups.setdefault(key, {"title": f"{name} · {description} · batch {batch}", "rows": []})["rows"].append(item)
    order = {"lstm2": 0, "gawf2": 1, "gru1": 0, "gawf1": 1, "gru3": 2, "gawf3": 3}
    for key in sorted(groups):
        group = groups[key]
        group["rows"].sort(key=lambda item: (order.get(item["model"], 99), item["seed"]))
        yield group



def print_progress(item, show_detail=True):
    current, target = item["global_frames"], item["target_global_frames"]
    fraction = min(current / target, 1) if target else 0
    filled = max(1, int(16 * fraction)) if current else 0
    bar = "#" * filled + "." * (16 - filled)
    percentage = "%.2f%%" % (100 * fraction) if target else "?"
    progress = "%s/%s %s (%s)" % (f"{current:,}", f"{target:,}" if target else "?",
                                     item.get("unit", "frames"), percentage)
    saved = " · checkpoint %s" % f"{item['saved_global_frames']:,}" if item["saved_global_frames"] else ""
    job = " · job %s" % item["job_id"] if item["job_id"] else ""
    detail = " · " + item["detail"] if show_detail and item["detail"] else ""
    print("  %s seed%s [%s] %s · %s%s%s%s" % (
        str(item["model"]).upper(), item["seed"], bar, progress,
        item["status"], saved, job, detail))



def main(analysis=False, row_reader=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("sjc-remote", "amarel"))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.rebuild:
        rebuild_index()
        return
    host = args.host or current_host()
    all_runs = manifests(host)
    runs = [run for run in all_runs
            if (run.get("mode") not in {"formal_train", "train_smoke", "train-smoke"}) == analysis]
    row_reader = row_reader or row
    slurm, slurm_missing = scheduler_states(all_runs) if host == "amarel" else ({}, [])
    tmux, tmux_missing = tmux_states(all_runs) if host == "sjc-remote" else ({}, [])
    running = [(run, item) for run in runs
               if (item := row_reader(run, slurm, tmux))["status"].lower() == "running"]
    runs = [run for run, _ in running]
    rows = [item for _, item in running]
    unregistered = slurm_missing + tmux_missing
    if args.json:
        print(json.dumps(dict(host=host, checked_at=datetime.now().astimezone().isoformat(),
                              jobs=rows, unregistered=unregistered), indent=2))
        return
    print("Navigation on %s · %s" % (host, datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")))
    if not rows:
        print("  No navigation jobs are running on this host.")
    if host == "sjc-remote":
        for group in sjc_display_groups(runs, rows):
            print(group["title"])
            for item in group["rows"]:
                print_progress(item, show_detail=item["status"] == "queued")
    else:
        for item in rows:
            print_progress(item)
            print("    id: " + item["experiment_id"])
    for item in unregistered:
        print("  UNREGISTERED navigation job: " + item)



def rebuild_index():
    root = Path(__file__).resolve().parents[2] / 'experiments/monitoring'
    rows, active = [], []
    for path in sorted((root / 'jobs').glob('*.json')):
        if path.name.startswith('._'):
            continue  # macOS tar metadata is not a manifest.
        run = json.loads(path.read_text())
        status = run['status']
        evidence = run.get('completion_evidence') or {}
        if run.get('mode') == 'train-smoke':
            progress = '%s/%s updates' % (evidence.get('completed_updates', 0), run['budget']['updates'])
        elif run.get('mode') == 'train_smoke':
            progress = '%s/%s stages' % (evidence.get('completed_stages', 0), run['expected_units'])
        elif run.get('mode') == 'checkpoint_eval_smoke':
            progress = '%s/%s episodes' % (evidence.get('episode_count', 0), run['expected_units'])
        elif run.get('mode') == 'preflight':
            progress = '%s/1 checks' % int(status == 'complete')
        elif run.get('mode') == 'formal_train':
            target = run['execution'].get('steps_budget') or run['execution'].get('target_steps') or run['expected_units']
            progress = '%s/%s steps' % (evidence.get('global_frames') or 0, target)
        else:
            done = sum(unit.get('metrics', {}).get('completed_episodes', 0) for unit in run.get('units', []))
            progress = '%s/%s episodes' % (done, run.get('expected_episodes', '?'))
        rows.append('| %s | %s | %s | %s |' % (run['experiment_id'], status, run['host'], progress))
        if status in ('submitted', 'running', 'recovering', 'paused', 'prepared'):
            active.append({'experiment_id': run['experiment_id'], 'host': run['host'], 'status': status,
                           'job_id': run.get('execution', {}).get('job_id')})
    (root / 'JOBS.md').write_text('# Navigation experiment history\n\n| Experiment ID | Status | Host | Progress |\n| --- | --- | --- | --- |\n' + '\n'.join(rows) + '\n')
    (root / 'active_jobs.json').write_text(json.dumps(active, indent=2) + '\n')
    print('Indexed %d experiments; %d active' % (len(rows), len(active)))


if __name__ == "__main__":
    main()
