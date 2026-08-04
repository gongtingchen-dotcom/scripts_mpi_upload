#!/bin/bash -l
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parent directory for ALL runs (each invocation creates its own
# timestamped+tagged subfolder underneath -- see build_output_tag() in
# theo_pipeline_new.py). Override with OUTPUT_ROOT=... for a different
# search, or pass --output-root explicitly as an extra arg below.
OUTPUT_ROOT="${OUTPUT_ROOT:-/gpfs/laur/data/0_collaboration/ceph_phys/GLC6962/GLC6962_g_g0/theo_pipeline_new_search}"
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
    send_mail_msg "[theo_pipeline_new] END (success)" "Run finished successfully on $(hostname).\nLog: ${LOG_FILE}"
  else
    send_mail_msg "[theo_pipeline_new] FAIL (exit=${exit_code})" "Run failed on $(hostname).\nLog: ${LOG_FILE}"
  fi
}
trap on_exit EXIT

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Log file: ${LOG_FILE}"
echo "Start time: $(date -Is)"

# Activate environment (prefer micromamba, fallback to conda). Tries, in
# order: an already-active "decoding" env, micromamba on PATH, then a few
# common micromamba install locations, then conda.
activated=0

if [[ "${CONDA_DEFAULT_ENV:-}" == "decoding" || "${MAMBA_DEFAULT_ENV:-}" == "decoding" ]]; then
  activated=1
  echo "Using already-active environment: decoding"
fi

if [[ ${activated} -eq 0 ]]; then
  for candidate in micromamba "${MAMBA_EXE:-}" "$HOME/.local/bin/micromamba" "$HOME/micromamba/bin/micromamba"; do
    [[ -z "${candidate}" ]] && continue
    mamba_bin="$(command -v "${candidate}" 2>/dev/null || true)"
    if [[ -n "${mamba_bin}" ]]; then
      export MAMBA_EXE="${mamba_bin}"
      export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
      eval "$("${MAMBA_EXE}" shell hook -s bash)"
      micromamba activate decoding
      activated=1
      break
    fi
  done
fi

if [[ ${activated} -eq 0 ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
      source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
      source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
  fi

  if command -v conda >/dev/null 2>&1; then
    conda activate decoding
    activated=1
  fi
fi

if [[ ${activated} -eq 0 ]]; then
  echo "ERROR: neither micromamba nor conda environment activation is available."
  echo "Expected env name: decoding"
  echo "Debug: CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<unset>} MAMBA_DEFAULT_ENV=${MAMBA_DEFAULT_ENV:-<unset>}"
  echo "Debug: MAMBA_EXE=${MAMBA_EXE:-<unset>}"
  exit 1
fi

echo "Host: $(hostname)"
echo "Python: $(which python)"
echo "Running local shell execution."

send_mail_msg "[theo_pipeline_new] BEGIN" "Run started on $(hostname).\nLog: ${LOG_FILE}"

PYTHON=${PYTHON_PATH:-$(which python)}

# Fixed dataset/experiment defaults, matching run_theo_decode_cv.sh. Any of
# these -- and --models/hyperparameter flags for a search -- can be
# overridden or added via extra arguments to this script: whatever's passed
# on the command line ("$@") is appended last, so it wins over the defaults
# below (argparse takes the last value given for a repeated flag).
$PYTHON "${SCRIPT_DIR}/theo_pipeline_new.py" \
  --recording-path /gpfs/laur/data/0_collaboration/ceph_phys/GLC6962/GLC6962_g_g0/preprocessed \
  --peaks-path /gpfs/laur/data/0_collaboration/ceph_phys/GLC6962/GLC6962_g_g0/preprocessed/thresholding_peaks_3.5/peaks_combined2_sorted.npy \
  --behavior-df-path /gpfs/laur/data/0_collaboration/sepia_filming/sepia_processed/ceph_analysis_dfs/GLC6633-6644-6811-6962-20260223/texture_pca_df_preproc.pkl \
  --metadata-df-path /gpfs/laur/data/0_collaboration/sepia_filming/sepia_processed/ceph_analysis_dfs/GLC6633-6644-6811-6962-20260223/metadata_df.pkl \
  --output-root "${OUTPUT_ROOT}" \
  --animal GLC6962 \
  --background eink \
  --pc-cols PC1,PC2,PC3,PC4,PC5 \
  --dt 0.2 \
  --bin-before 6 \
  --bin-current 1 \
  --bin-after 6 \
  --n-splits 5 \
  --models SVR,XGB \
  "$@"

echo "End time: $(date -Is)"
