#!/bin/bash -l
# THE active pipeline (real run) — runs density_decoding_unified.py with
# --shuffle none. Requires a --selected_trials CSV from trial_structure.py.
# For the null-distribution runs, see run_shuffle_array.sh; for comparing the
# two afterwards, see shuffle_analysis.py.
# Submit with: sbatch run_density_decoding_unified.sh
# Initial working directory:
#SBATCH -D ./
# Output directory:
#SBATCH -o ./slurm/%j/logs/density.log
# Error directory:
#SBATCH -e ./slurm/%j/logs/density.log
# Job Name:
#SBATCH -J density_decoding
#
# Number of nodes and MPI tasks per node:
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:1

#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1

#SBATCH --cpus-per-task=18
#
#SBATCH --mem=100000  # Memory per node (in MB)
#
#SBATCH --mail-user=gong.tingchen@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --time=23:00:00

# Load modules in the correct order for PyTorch distributed training
# module purge
# module load matlab

# unset DISPLAY

# Activate the conda environment. 
#source /mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/etc/profile.d/conda.sh
conda activate density_decoding

# Python path should be passed as environment variable when submitting job
PYTHON=${PYTHON_PATH:-$(which python)}

python /u/gongt/scripts/density_decoding/density_decoding_unified.py \
    --animal GLC6962 \
    --exp _g_g0 \
    --behavior_var PC1 \
    --chunk_size 10 \
    --context_before_bins 2 \
    --context_after_bins 2 \
    --smooth_beh_window_sec 1 \
    --selected_trials /ptmp/gongt/behaviour/GLC6962/GLC6962_g_g0/similar_trials_PC1_peak_rms0.5.csv \
    