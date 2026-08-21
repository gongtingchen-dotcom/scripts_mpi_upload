#!/bin/bash -l
# SLURM submit script for the preprocessing pipeline.
# Runs preprocess_artifact_removal_by_sqr_deviation.py on one recording session
# (bad-channel removal, common referencing, eInk-artifact removal, TTL extraction,
# scaling) and saves the result to a "preprocessed" folder next to the raw data.
# See the top of that script for what it does and why.
#
# Usage: edit --animal_tag / --exp_tag below for the session you want to process,
# then submit with: sbatch run_preprocess.sh
# (--exp_tag accepts multiple tags, e.g. "a_g0 a_g1", to concatenate several
# recordings from the same session before preprocessing — see the script.)
# 
# Change slurm setup based on data size




# Initial working directory:
#SBATCH -D ./
# Output directory:
#SBATCH -o ./slurm/%j/logs/preprocess.log
# Error directory:
#SBATCH -e ./slurm/%j/logs/preprocess.log
# Job Name:
#SBATCH -J preprocess

#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:2

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=36
#
#SBATCH --mem=250000  # Memory per node (in MB)
#
#SBATCH --mail-user=gong.tingchen@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=23:00:00

# (leftover from an unrelated job template, not needed here — left commented out)
# module purge
# module load matlab

# unset DISPLAY

# Activate the conda environment (has spikeinterface / kilosort4 and their deps).
#source /mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/etc/profile.d/conda.sh
conda activate phy2

# Python path should be passed as environment variable when submitting job
PYTHON=${PYTHON_PATH:-$(which python)}

python /u/gongt/scripts/preprocess/preprocess_artifact_removal_by_sqr_deviation.py --animal_tag GLC7248 --exp_tag a_g1

# (not part of this pipeline — leftover commented-out line, spike sorting is a separate step)
# python /u/gongt/scripts/ironclust.py