#!/bin/bash -l
# Submits every row of a search manifest as a sequence of SLURM array jobs,
# sized to stay under this account's MaxSubmit=300 cap, without needing a
# human to babysit or a one-off watcher script per round (see git history --
# this used to be run_theo_pipeline_search_array_raven_{cpu,gpu}.sh plus
# wait_and_submit_gpu_array.sh; one script now covers what those three did).
#
# Submits BATCH_SIZE rows at a time (as one array job, throttled to
# CONCURRENCY running at once); once this script's own outstanding
# (PENDING+RUNNING) task count drops below RESUBMIT_THRESHOLD, it submits
# the next batch. Runs until the whole manifest is submitted, then exits --
# it does not wait for the last batch to finish.
#
# Usage: nohup ./submit_search.sh <manifest.tsv> > submit_search.log 2>&1 &
#        disown
#
# Safe to re-run on the same manifest: re-generate it with build_search_grid.py
# first (which skips anything already in results/), so a rerun only ever
# resubmits genuinely new rows. Don't regenerate the manifest while a batch
# from a previous call is still outstanding -- rows queued but not yet
# started have no results/ entry yet, so build_search_grid.py can't see them
# and would duplicate them into the new manifest.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARRAY_SCRIPT="${SCRIPT_DIR}/run_theo_pipeline_raven.sh"
# sbatch sets SLURM_SUBMIT_DIR to the cwd it's invoked from -- run_theo_pipeline_raven.sh
# relies on that to find its own directory once spooled to /var/spool/slurmd/, so pin it
# here rather than trusting the caller's cwd.
cd "${SCRIPT_DIR}"

MANIFEST="${1:?Usage: submit_search.sh <manifest.tsv>}"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: manifest not found: ${MANIFEST}" >&2
  exit 1
fi
N_TOTAL=$(wc -l < "${MANIFEST}")
if (( N_TOTAL == 0 )); then
  echo "Manifest is empty -- nothing to submit."
  exit 0
fi

BATCH_SIZE="${BATCH_SIZE:-100}"
CONCURRENCY="${CONCURRENCY:-25}"
RESUBMIT_THRESHOLD="${RESUBMIT_THRESHOLD:-30}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-120}"
JOB_NAME="theo_search"

echo "$(date -Is) Submitting ${N_TOTAL} rows from ${MANIFEST} in batches of ${BATCH_SIZE}" \
     "(concurrency %${CONCURRENCY}, next batch once <${RESUBMIT_THRESHOLD} outstanding)."

start=0
while (( start < N_TOTAL )); do
  end=$(( start + BATCH_SIZE - 1 ))
  if (( end >= N_TOTAL )); then
    end=$(( N_TOTAL - 1 ))
  fi

  while true; do
    # -r expands array jobs to one line per task -- squeue's default
    # collapses a whole pending array range into a single display line,
    # which would undercount outstanding tasks by ~100x here.
    outstanding=$(squeue -u "${USER}" -n "${JOB_NAME}" -r -h | wc -l)
    echo "$(date -Is) Outstanding ${JOB_NAME} tasks: ${outstanding} (need < ${RESUBMIT_THRESHOLD} to submit rows ${start}-${end})"
    if (( outstanding < RESUBMIT_THRESHOLD )); then
      break
    fi
    sleep "${POLL_INTERVAL_SEC}"
  done

  echo "$(date -Is) Submitting rows ${start}-${end}."
  sbatch --array="${start}-${end}%${CONCURRENCY}" "${ARRAY_SCRIPT}" "${MANIFEST}"
  start=$(( end + 1 ))
done

echo "$(date -Is) All ${N_TOTAL} rows submitted. Exiting (not waiting for completion)."
