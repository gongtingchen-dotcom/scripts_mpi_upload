#!/bin/bash -l
#SBATCH -D ./
#SBATCH -o ./slurm/%A_%a/logs/shuffle_%a.log
#SBATCH -e ./slurm/%A_%a/logs/shuffle_%a.log
#SBATCH -J density_shuffle
#
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=100000
#SBATCH --mail-user=gong.tingchen@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=23:00:00
#
# 100 shuffles, max 10 running at once.
# When one finishes the next starts automatically.
# Submit with:  sbatch run_shuffle_array.sh

#SBATCH --array=0-99%10

conda activate density_decoding
PYTHON=${PYTHON_PATH:-$(which python)}

# ─────────────────────────────────────────────────────────────────────────────

mkdir -p ./slurm/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/logs

echo "=== SHUFFLE ${SLURM_ARRAY_TASK_ID} / $((N_SHUFFLES - 1)) ==="

python /u/gongt/scripts/density_decoding/density_decoding_unified.py \
    --animal GLC6962 \
    --exp _g_g0 \
    --behavior_var PC1 \
    --chunk_size 10 \
    --context_before_bins 2 \
    --context_after_bins 2 \
    --smooth_beh_window_sec 1 \
    --shuffle shift \
    --shuffle_mode neural \
    --selected_trials /ptmp/gongt/behaviour/GLC6962/GLC6962_g_g0/similar_trials_PC1_peak_rms0.5.csv \
    --shuffle_index   "$SLURM_ARRAY_TASK_ID" \