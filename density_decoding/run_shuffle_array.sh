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

# ── SETTINGS ── edit these ────────────────────────────────────────────────────
ANIMAL="GLC6962"
EXP="_g_g0"
BEHAVIOR_VAR="PC9"
BIN_SIZE_MS=100
SELECTED_TRIALS="/ptmp/gongt/behaviour/GLC6962/GLC6962_g_g0/similar_trials_PC9_peak_dur5s.csv"
CHUNK_STYLE="interleaved"
N_SHUFFLES=100
SEED=42
SCRAMBLE_MODE="neural" 
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p ./slurm/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/logs

echo "=== SHUFFLE ${SLURM_ARRAY_TASK_ID} / $((N_SHUFFLES - 1)) ==="

$PYTHON /u/gongt/scripts/density_decoding/density_decoding_trial_selection_shuffle_all.py \
    --animal          "$ANIMAL" \
    --exp             "$EXP" \
    --behavior_var    "$BEHAVIOR_VAR" \
    --bin_size_ms     "$BIN_SIZE_MS" \
    --selected_trials "$SELECTED_TRIALS" \
    --chunk_style     "$CHUNK_STYLE" \
    --n_shuffles      "$N_SHUFFLES" \
    --shuffle_index   "$SLURM_ARRAY_TASK_ID" \
    --seed            "$SEED" \
    --shuffle_mode   "$SCRAMBLE_MODE"