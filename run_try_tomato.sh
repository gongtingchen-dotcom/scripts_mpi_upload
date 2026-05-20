#!/bin/bash -l
# Initial working directory:
#SBATCH -D ./
# Output directory:
#SBATCH -o ./slurm/%j/logs/training.log
# Error directory:
#SBATCH -e ./slurm/%j/logs/training.log
# Job Name:
#SBATCH -J tomato
#
# Number of nodes and MPI tasks per node:
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4

#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1

#SBATCH --cpus-per-task=18
#
#SBATCH --mem=500000  # Memory per node (in MB)
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
conda activate spikeinterface

# Python path should be passed as environment variable when submitting job
PYTHON=${PYTHON_PATH:-$(which python)}

# Set your sample range here, or override via sbatch --export:
#   sbatch --export=ALL,SAMPLE_START=0,SAMPLE_END=520000000 run_try_tomato.sh

$PYTHON /u/gongt/scripts/try_tomato.py --sample_start 540000000