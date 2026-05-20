#!/bin/bash -l
# Initial working directory:
#SBATCH -D ./
# Output directory:
#SBATCH -o ./slurm/%j/logs/training.log
# Error directory:
#SBATCH -e ./slurm/%j/logs/training.log
# Job Name:
#SBATCH -J compute
#
# Number of nodes and MPI tasks per node:
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=64
#
#SBATCH --mem=200000  # Memory per node (in MB)
#
#SBATCH --mail-user=tingchen.gong@brain.mpg.de

#SBATCH --time=23:00:00

# Load modules in the correct order for PyTorch distributed training
# module purge
# module load matlab

# unset DISPLAY

# Activate the conda environment. 
#source /mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/etc/profile.d/conda.sh
conda activate phy2

# Python path should be passed as environment variable when submitting job
PYTHON=${PYTHON_PATH:-$(which python)}

python /u/gongt/scripts/compute_in_si.py 
    
