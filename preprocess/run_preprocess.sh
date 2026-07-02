#!/bin/bash -l
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

# Load modules in the correct order for PyTorch distributed training
# module purge
# module load matlab

# unset DISPLAY

# Activate the conda environment. 
#source /mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/etc/profile.d/conda.sh
conda activate phy2

# Python path should be passed as environment variable when submitting job
PYTHON=${PYTHON_PATH:-$(which python)}

python /u/gongt/scripts/preprocess/preprocess_artifact_removal_by_sqr_deviation.py --animal_tag GLC7248 --exp_tag a_g1

# python /u/gongt/scripts/ironclust.py