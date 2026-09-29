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

# Can run using the following methods: 
#
#  1. pass full theo_pipeline_new.py args directly: 
#       sbatch run_theo_pipeline_raven.sh --models WC --wc-degree 2 \
#         --recording-path ... 
# OR 
#  2. SLURM array batch -- $1 is a manifest file (one theo_pipeline_new.py
#     args-string per line, built by build_search_grid.py); detected via
#     $SLURM_ARRAY_TASK_ID being set, in which case this reads its own line
#     out of the manifest instead of using "$@" directly. Always requests a
#     GPU node regardless of which model the row is for -- --device
#     defaults to cuda in theo_pipeline_new.py and is explicitly ignored by
#     WF/WC/SVR/XGB, so this is harmless for the CPU-only models and means
#     one script/queue covers every model instead of a CPU/GPU split.
#
#     use submit_search.sh, which submits it in batches sized to this account's MaxSubmit cap and
#     resubmits the next batch automatically as the queue drains:
#
#       python build_search_grid.py               # (re)builds search_manifest.tsv
#       nohup ./submit_search.sh search_manifest.tsv > submit_search.log 2>&1 &
#       disown

set -euo pipefail

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
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
else
  # --- Manual mode: forward whatever was passed on the command line ---
  FORWARD_ARGS="$*"
fi


OUTPUT_ROOT="${OUTPUT_ROOT:-/nexus/posix0/MBR-neuralsystems/ephys/GLC7189/GLC7189_c_g0/theo_pipeline_params_search_aligned}"
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

# Activate the "decoding" conda env (/u/gongt/conda-envs/decoding) 

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
