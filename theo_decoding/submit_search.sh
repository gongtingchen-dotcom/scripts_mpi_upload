#!/bin/bash -l
# 
# runs as a long-lived background process; 
# this script submits run_theo_pipeline_raven.sh 
# 
# Submits every row of a search manifest as a sequence of SLURM array jobs,
# sized to stay under this account's MaxSubmit=300 cap. 
# 
# Usage: nohup ./submit_search.sh <manifest.tsv> > submit_search.log 2>&1 &
#        disown
#


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
  ### key line 
  sbatch --array="${start}-${end}%${CONCURRENCY}" "${ARRAY_SCRIPT}" "${MANIFEST}"
  start=$(( end + 1 ))
done

echo "$(date -Is) All ${N_TOTAL} rows submitted. Exiting (not waiting for completion)."
