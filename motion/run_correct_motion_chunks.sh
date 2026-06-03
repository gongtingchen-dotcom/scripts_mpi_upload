#!/bin/bash -l
# Usage (to submit):  ./run_correct_motion_chunks.sh [run_name] [amp_pct]
#   run_name : suffix after "chunks_"  (default: n25)
#   amp_pct  : amplitude percentile    (default: 20)
#
# When run directly (not under SLURM) it counts the chunks and re-submits
# itself as an array job — no separate submit script needed.
# --array is set dynamically by the self-submit block below.
#
# Initial working directory:
#SBATCH -D ./
# Output / error logs per chunk:
#SBATCH -o /u/gongt/slurm/%j/logs/correct_motion_%a.log
#SBATCH -e /u/gongt/slurm/%j/logs/correct_motion_%a.log
# Job name:
#SBATCH -J correct_motion_chunk
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=4

#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:1

#SBATCH --mem=100000

#SBATCH --mail-user=gong.tingchen@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --time=02:00:00

# --- Self-submit when called from the shell (not inside a SLURM job) --------
if [ -z "$SLURM_JOB_ID" ]; then
    RUN_NAME=${1:-1}
    # AMP_PCT=${2:-20}
    MOTION_BASE="/ptmp/gongt/GLC6962_l_g1/preprocessed/thresholding_peaks/motion_estimates"
    N=$(ls -d "$MOTION_BASE/chunks_${RUN_NAME}"/motion_dredge_*/ 2>/dev/null | wc -l)
    if [ "$N" -eq 0 ]; then
        echo "ERROR: No motion_dredge_* directories found in chunks_${RUN_NAME}"
        exit 1
    fi
    echo "Found $N chunks in chunks_${RUN_NAME} — submitting array job 0-$((N-1))"
    sbatch --array=0-$((N-1)) \
           --export=ALL,RUN_NAME="$RUN_NAME",AMP_PCT="$AMP_PCT" \
           "$0"
    exit $?
fi
# ----------------------------------------------------------------------------

conda activate spikeinterface

PYTHON=${PYTHON_PATH:-$(which python)}

RUN_NAME=${RUN_NAME:-1}
# AMP_PCT=${AMP_PCT:-20}

MOTION_BASE="/ptmp/gongt/GLC6962_l_g1/preprocessed/thresholding_peaks/motion_estimates"
CHUNKS_DIR="chunks_${RUN_NAME}"

# Build sorted list of motion directories for this chunks_dir
mapfile -t MOTION_NAMES < <(ls -d "$MOTION_BASE/$CHUNKS_DIR"/motion_dredge_*/ 2>/dev/null \
    | xargs -n1 basename | sort)

if [ ${#MOTION_NAMES[@]} -eq 0 ]; then
    echo "ERROR: No motion_dredge_* directories found in $MOTION_BASE/$CHUNKS_DIR"
    exit 1
fi

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Run this script with ./run_correct_motion_chunks.sh, not sbatch directly."
    exit 1
fi

MOTION_NAME="${MOTION_NAMES[$SLURM_ARRAY_TASK_ID]}"

if [ -z "$MOTION_NAME" ]; then
    echo "ERROR: Array index $SLURM_ARRAY_TASK_ID out of range (only ${#MOTION_NAMES[@]} chunks)"
    exit 1
fi

mkdir -p /u/gongt/slurm/$SLURM_JOB_ID/logs

echo "Job array index : $SLURM_ARRAY_TASK_ID"
echo "Chunks dir      : $CHUNKS_DIR"
echo "Motion name     : $MOTION_NAME"
echo "Amp percentile  : $AMP_PCT"

$PYTHON /u/gongt/scripts/correct_motion.py \
    --chunks_dir "$CHUNKS_DIR" \
    --motion_name "$MOTION_NAME" \
    --amp_pct "$AMP_PCT"
