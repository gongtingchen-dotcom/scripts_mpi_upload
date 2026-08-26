#!/bin/bash -l
# THE active pipeline — use this one. Runs try_tomato.py: ToMATo clustering
# directly on drift-corrected localized peaks (no spike sorting). Edit
# DATA_PATH/FOLDER_NAME below for the session, and the sample range if
# splitting a long recording into chunks (see try_tomato.py's header for
# why/how, and merge_tomato_results.py for stitching chunks back together).
# Submit with: sbatch run_try_tomato.sh
#
# Initial working directory:
#SBATCH -D ./
# Output directory:
#SBATCH -o ./slurm/%j/logs/tomato.log
# Error directory:
#SBATCH -e ./slurm/%j/logs/tomato.log
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

#SBATCH --time=2:20:00

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

DATA_PATH="/nexus/posix0/MBR-neuralsystems/ephys/GLC7189/GLC7189_c_g0/preprocessed"
FOLDER_NAME="thresholding_peaks_3.5"
MOTION_PATH="${DATA_PATH}/${FOLDER_NAME}/motion_estimates/motion_dredge"
LOCALIZED_PEAKS_PATH="${DATA_PATH}/${FOLDER_NAME}/localized_peaks_mt_raw.npy"

$PYTHON /u/gongt/scripts/tomato/try_tomato.py \
    --data_path "$DATA_PATH" \
    --motion_path "$MOTION_PATH" \
    --localized_peaks_path "$LOCALIZED_PEAKS_PATH"