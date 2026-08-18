#!/usr/bin/env python3
"""External runtime-budget watchdog for theo_search SLURM jobs, covering
the gap the in-process check in theo_pipeline_new.py can't: a job that was
ALREADY RUNNING (had already loaded the old script into its Python process
memory) when that check was added can't pick up the code change -- there's
no way to hot-patch a running interpreter. This replicates the exact same
"project from the first 2 folds, compare to budget" logic from OUTSIDE,
by polling squeue + each task's own SLURM log (slurm/<ArrayJobID>_<TaskID>/
logs/search_<TaskID>.log -- see run_theo_pipeline_raven.sh's -o/-e), and
scancel's + appends a note to that log for any job whose projected total
exceeds budget.

Also works as a redundant safety net going forward -- harmless for jobs
that already have the in-process check (theo_pipeline_new.py should always
catch those first), but covers the case where a job is running a stale
cached script for any other reason.

Usage:
    nohup python monitor_runtime_budget.py > monitor_runtime_budget.log 2>&1 &
    disown

    python monitor_runtime_budget.py --dry-run --once   # detect-only, single pass, no scancel
    python monitor_runtime_budget.py --once             # single real pass
"""

import argparse
import re
import subprocess
import time
from pathlib import Path

THEO_DECODING_DIR = Path("/raven/u/gongt/scripts/theo_decoding")
JOB_NAME = "theo_search"
DEFAULT_BUDGET_HOURS = 23.0
DEFAULT_POLL_INTERVAL_SEC = 300

FOLD_RE = re.compile(r"Model (\S+): finished fold (\d+)/(\d+) in ([\d.]+)s")


def whoami():
    return subprocess.check_output(["whoami"], text=True).strip()


def parse_elapsed(s):
    """squeue's %M format: [D-]HH:MM:SS or MM:SS."""
    days = 0
    if "-" in s:
        d, s = s.split("-")
        days = int(d)
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = parts
    else:
        h = "0"
        m, sec = parts
    return days * 86400 + int(h) * 3600 + int(m) * 60 + float(sec)


def squeue_running(user):
    """[(array_job_id, task_id, elapsed_sec)] for every RUNNING theo_search task."""
    out = subprocess.run(
        ["squeue", "-u", user, "-r", "-h", "-t", "RUNNING", "-n", JOB_NAME, "-o", "%i|%M"],
        capture_output=True, text=True, check=True,
    ).stdout
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if not line or "_" not in line:
            continue
        jobid, elapsed = line.split("|")
        arr, task = jobid.split("_", 1)
        rows.append((arr, task, parse_elapsed(elapsed)))
    return rows


def check_task(arr, task, elapsed_sec, budget_hours):
    """None if there's not yet enough data (fewer than 2 completed folds for
    the task's own current model); otherwise a dict with the projection."""
    log_path = THEO_DECODING_DIR / "slurm" / f"{arr}_{task}" / "logs" / f"search_{task}.log"
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="replace")
    folds = FOLD_RE.findall(text)
    if len(folds) < 2:
        return None
    # Only the most recent model's own fold sequence matters -- in practice
    # every manifest row here runs exactly one model, but this stays correct
    # if a row ever selects more than one.
    current_model = folds[-1][0]
    this_model_folds = [f for f in folds if f[0] == current_model]
    if len(this_model_folds) < 2:
        return None
    f1, f2 = this_model_folds[0], this_model_folds[1]
    n_splits = int(f1[2])
    avg_fold_time = (float(f1[3]) + float(f2[3])) / 2
    folds_done = len(this_model_folds)
    remaining = n_splits - folds_done
    projected_sec = elapsed_sec + avg_fold_time * remaining
    return {
        "arr": arr, "task": task, "model": current_model, "n_splits": n_splits,
        "folds_done": folds_done, "avg_fold_time": avg_fold_time,
        "elapsed_sec": elapsed_sec, "projected_sec": projected_sec,
        "exceeds": projected_sec > budget_hours * 3600, "log_path": log_path,
    }


def abort_job(info, dry_run):
    jobid = f"{info['arr']}_{info['task']}"
    msg = (
        f"ABORTED BY EXTERNAL MONITOR (monitor_runtime_budget.py): projected total runtime "
        f"{info['projected_sec'] / 3600:.2f}h (model={info['model']}, "
        f"{info['folds_done']} folds done so far @ {info['avg_fold_time']:.1f}s/fold "
        f"avg from its first 2 folds, n_splits={info['n_splits']}, "
        f"{info['elapsed_sec'] / 3600:.2f}h elapsed) exceeds budget -- cancelling {jobid}."
    )
    print(msg, flush=True)
    if dry_run:
        print(f"  (dry-run: would scancel {jobid} and append this note to {info['log_path']})", flush=True)
        return
    subprocess.run(["scancel", jobid])
    with open(info["log_path"], "a") as f:
        f.write("\n" + msg + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="Detect and print only -- never scancel.")
    p.add_argument("--once", action="store_true", help="Single pass, then exit (default: loop forever).")
    p.add_argument("--budget-hours", type=float, default=DEFAULT_BUDGET_HOURS,
        help=f"Must match (or be more conservative than) theo_pipeline_new.py's own "
        f"--runtime-abort-budget-hours default ({DEFAULT_BUDGET_HOURS}) to behave consistently.")
    p.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL_SEC)
    args = p.parse_args()

    user = whoami()
    seen_aborted = set()
    while True:
        running = squeue_running(user)
        for arr, task, elapsed_sec in running:
            key = (arr, task)
            if key in seen_aborted:
                continue
            info = check_task(arr, task, elapsed_sec, args.budget_hours)
            if info is None:
                continue
            status = "*** EXCEEDS BUDGET ***" if info["exceeds"] else "ok"
            print(
                f"{arr}_{task}: model={info['model']} folds_done={info['folds_done']}/{info['n_splits']} "
                f"avg_fold={info['avg_fold_time']:.1f}s elapsed={info['elapsed_sec'] / 3600:.2f}h "
                f"projected={info['projected_sec'] / 3600:.2f}h {status}",
                flush=True,
            )
            if info["exceeds"]:
                abort_job(info, args.dry_run)
                seen_aborted.add(key)
        if args.once:
            break
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
