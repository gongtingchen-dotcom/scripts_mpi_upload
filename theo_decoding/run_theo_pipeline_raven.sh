#!/bin/bash -l
# Initial working directory:
#SBATCH -D ./
# Output / error logs per array task:
#SBATCH -o ./slurm/%A_%a/logs/search_%a.log
#SBATCH -e ./slurm/%A_%a/logs/search_%a.log
# Job name:
#SBATCH -J theo_search
#
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100000
#SBATCH --mail-user=gong.tingchen@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=23:59:59
# Raised from 5h: the 5h cap was killing large-window/large-n_splits SVR/XGB
# runs right before they finished (confirmed via several logs stopped mid-fold
# at ~4.9h), burning the compute for nothing since a killed run leaves
# run_params.json but no cv_summary.json -- build_search_grid.py's dedup then
# treats it as permanently "still running" and never retries it. 23:59:59
# (not 24:00:00 -- SLURM's --time format rejects a bare "24:00:00" as the
# next day) gives the slowest custom-followup-sweep XGB/SVR combos room to
# actually finish.
#
# One script, two ways to call it -- previously two separate files
# (run_theo_pipeline_search_array_raven.sh calling run_theo_pipeline_new_raven.sh)
# doing this same job/logging split; merged since the array half had shrunk
# to "look up a line, forward it" and the job half to "set up logging/env,
# forward again".
#
#  1. Manual/interactive single run -- pass full theo_pipeline_new.py args
#     directly, plain bash, no SLURM/sbatch involved:
#       bash run_theo_pipeline_raven.sh --models WC --wc-degree 2 \
#         --recording-path ... (see build_search_grid.py's FIXED_ARGS for
#         the usual dataset flags -- there's no default dataset path here,
#         everything must be passed explicitly in this mode)
#
#  2. SLURM array batch -- $1 is a manifest file (one theo_pipeline_new.py
#     args-string per line, built by build_search_grid.py); detected via
#     $SLURM_ARRAY_TASK_ID being set, in which case this reads its own line
#     out of the manifest instead of using "$@" directly. Always requests a
#     GPU node regardless of which model the row is for -- --device
#     defaults to cuda in theo_pipeline_new.py and is explicitly ignored by
#     WF/WC/SVR/XGB, so this is harmless for the CPU-only models and means
#     one script/queue covers every model instead of a CPU/GPU split.
#
#     Don't call sbatch on this directly -- use submit_search.sh, which
#     submits it in batches sized to this account's MaxSubmit cap and
#     resubmits the next batch automatically as the queue drains:
#
#       python build_search_grid.py               # (re)builds search_manifest.tsv
#       nohup ./submit_search.sh search_manifest.tsv > submit_search.log 2>&1 &
#       disown

set -euo pipefail

# Under sbatch, the submitted script is spooled to /var/spool/slurmd/jobNNNNN/
# and executed from there, so BASH_SOURCE[0] no longer points at this file's
# real location -- SLURM_SUBMIT_DIR (the cwd sbatch was invoked from, which is
# always this directory -- see submit_search.sh) is the reliable source of
# truth in that case. Manual/interactive mode has no SLURM_SUBMIT_DIR, so
# BASH_SOURCE[0] resolution (unaffected by spooling there) is used instead.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  # --- Array mode: this task's row comes from the manifest, not "$@" ---
  MANIFEST="${1:?Usage under sbatch --array=...: run_theo_pipeline_raven.sh <manifest.tsv>}"
  if [[ ! -f "${MANIFEST}" ]]; then
    echo "ERROR: manifest not found: ${MANIFEST}" >&2
    exit 1
  fi
  # Manifest is 0-indexed (row 0 = first line) to match --array=0-N-1.
  LINE_NO=$((SLURM_ARRAY_TASK_ID + 1))
  FORWARD_ARGS="$(sed -n "${LINE_NO}p" "${MANIFEST}")"
  if [[ -z "${FORWARD_ARGS}" ]]; then
    echo "ERROR: manifest line ${LINE_NO} (array task ${SLURM_ARRAY_TASK_ID}) is empty -- " \
         "array range doesn't match manifest length." >&2
    exit 1
  fi
  echo "Array task ${SLURM_ARRAY_TASK_ID}: line ${LINE_NO} of ${MANIFEST}"
  echo "Args: ${FORWARD_ARGS}"

  # All search runs share the manifest's own directory as --output-root, so
  # every run + the manifest + this task's SLURM log all live under one tree.
  export OUTPUT_ROOT="$(cd "$(dirname "${MANIFEST}")" && pwd)"

  # Keep BLAS/OMP threading inside the cpus SLURM actually gave this task --
  # unset/default thread counts would otherwise let numpy/xgboost/torch each
  # try to use all cores on the node, oversubscribing when several array
  # tasks share it.
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
else
  # --- Manual mode: forward whatever was passed on the command line ---
  FORWARD_ARGS="$*"
fi

# raven variant: activates the decoding conda env by absolute interpreter
# path (see below) and logs to OUTPUT_ROOT/logs instead of assuming
# /gpfs/laur (not mounted on raven at all). Does NOT set any dataset/
# experiment defaults -- those live in build_search_grid.py's FIXED_ARGS,
# baked into every manifest row already for array mode, and must be passed
# explicitly for a manual run.
#
# In array mode OUTPUT_ROOT is overridden to the manifest's own directory
# (above). The default below is only used for a bare manual run, and points
# at the ORIGINAL (pre-leakage-fix) search folder -- for correct numbers,
# pass OUTPUT_ROOT=.../theo_pipeline_params_search_lagfixed  (see
# build_search_grid.py's SEARCH_ROOT note).
OUTPUT_ROOT="${OUTPUT_ROOT:-/nexus/posix0/MBR-neuralsystems/ephys/GLC7189/GLC7189_c_g0/theo_pipeline_params_search}"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
RUN_TS="$(date +%Y%m%d_%H%M%S_%N)"
LOG_FILE="${LOG_DIR}/theo_pipeline_new_${RUN_TS}.log"

# Optional email notifications for local runs.
ENABLE_EMAIL="${ENABLE_EMAIL:-0}"
MAIL_TO="${MAIL_TO:-gong.tingchen@gmail.com}"

send_mail_msg() {
  local subject="$1"
  local body="$2"
  if [[ "${ENABLE_EMAIL}" != "1" ]]; then
    return 0
  fi
  if command -v mail >/dev/null 2>&1; then
    printf '%s\n' "${body}" | mail -s "${subject}" "${MAIL_TO}" || true
  elif command -v mailx >/dev/null 2>&1; then
    printf '%s\n' "${body}" | mailx -s "${subject}" "${MAIL_TO}" || true
  else
    echo "WARN: ENABLE_EMAIL=1 but no 'mail' or 'mailx' command found; skipping email." >&2
  fi
}

on_exit() {
  local exit_code=$?
  if [[ ${exit_code} -eq 0 ]]; then
    send_mail_msg "[theo_pipeline_raven] END (success)" "Run finished successfully on $(hostname).\nLog: ${LOG_FILE}"
  else
    send_mail_msg "[theo_pipeline_raven] FAIL (exit=${exit_code})" "Run failed on $(hostname).\nLog: ${LOG_FILE}"
  fi
}
trap on_exit EXIT

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Log file: ${LOG_FILE}"
echo "Start time: $(date -Is)"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"

# Activate the "decoding" conda env (/u/gongt/conda-envs/decoding) mainly for
# diagnostics/env vars -- best-effort, not fatal if it doesn't fully "take".
# We do NOT rely on this to put the right `python` first on PATH: in this
# raven login environment (in particular under VSCode Remote-SSH), PATH can
# already have a different env's bin/ (e.g. conda-envs/spikeinterface)
# pinned ahead of everything from session startup, so `conda activate` here
# reorders PATH but doesn't win the race. PYTHON below is instead pointed at
# the decoding env's interpreter by absolute path, which sidesteps PATH
# ordering entirely (conda envs are self-contained; verified importable with
# a bare PATH, no activation needed).
if command -v conda >/dev/null 2>&1; then
  conda activate decoding 2>/dev/null || true
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate decoding 2>/dev/null || true
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate decoding 2>/dev/null || true
fi

DECODING_ENV_PYTHON="/u/gongt/conda-envs/decoding/bin/python"
if [[ ! -x "${DECODING_ENV_PYTHON}" ]]; then
  echo "ERROR: expected decoding env python not found at ${DECODING_ENV_PYTHON}" >&2
  exit 1
fi

echo "Host: $(hostname)"
echo "Python: ${DECODING_ENV_PYTHON}"

send_mail_msg "[theo_pipeline_raven] BEGIN" "Run started on $(hostname).\nLog: ${LOG_FILE}"

PYTHON=${PYTHON_PATH:-${DECODING_ENV_PYTHON}}

# shellcheck disable=SC2086
$PYTHON "${SCRIPT_DIR}/theo_pipeline_new.py" ${FORWARD_ARGS}

echo "End time: $(date -Is)"
