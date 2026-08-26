#!/bin/bash -l
# THE active auto-merging pipeline — runs try_merge_selfdbuild_cleaned.py
# (step 1). Edit paths/RUN_* flags directly in that script before submitting.
# For the optional second merge pass (try_merge_step_2.py), run it manually
# in the same `slay` environment — no separate .sh wrapper exists for it yet.
# Submit with: sbatch run_try_merge_selfbuild.sh

# Initial working directory:
#SBATCH -D ./
# Output directory:
#SBATCH -o ./slurm/%j/logs/tomato_merge.log
# Error directory:
#SBATCH -e ./slurm/%j/logs/tomato_merge.log
# Job Name:
#SBATCH -J tomato_merge

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4

#SBATCH --cpus-per-task=64
#
#SBATCH --mem=500000  # Memory per node (in MB)
#
#SBATCH --mail-user=gong.tingchen@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --time=23:59:55

mkdir -p ./slurm/${SLURM_JOB_ID}/logs

# Activate the conda environment.
#source /mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/etc/profile.d/conda.sh
conda activate slay

# Python path should be passed as environment variable when submitting job
PYTHON=${PYTHON_PATH:-$(which python)}

python /u/gongt/scripts/tomato/try_merge_selfdbuild_cleaned.py