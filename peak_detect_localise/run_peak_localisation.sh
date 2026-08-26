#!/bin/bash -l
# THE active pipeline — use this one. Runs peak_localisation.py: spike
# detection + monopolar-triangulation localization. In practice run in
# DETECTION + LOCALIZATION MODE ONLY — motion estimation/correction stays
# disabled here (see the RUN_*/APPLY_MOTION_CORRECTION flags and their
# comments in peak_localisation.py's config section) and is done afterwards
# as separate steps:
#   1. this script                              -> peaks + locations
#   2. slope_artifact_filter_analysis.ipynb      -> filter narrow-waveform artifacts
#   3. motion/run_dredge_chunks.sh + run_dredge_stitch.sh + run_correct_motion_chunks.sh
#                                                 -> motion estimation (chunked) + correction
# See peak_localisation.py's own header for why this is done as separate
# steps instead of end-to-end in this one script.
#
# Edit --base_path and the other CLI args below for the session to process
# (see peak_localisation.py's header for what each one does), then submit
# with: sbatch run_peak_localisation.sh
#
# Initial working directory:
#SBATCH -D ./
# Output directory:
#SBATCH -o ./slurm/%j/logs/training.log
# Error directory:
#SBATCH -e ./slurm/%j/logs/training.log
# Job Name:
#SBATCH -J peak_localisation
#
# Number of nodes and MPI tasks per node:
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4

#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1

#SBATCH --cpus-per-task=36
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

export PYTHONPATH="/u/gongt/scripts:$PYTHONPATH"

# Python path should be passed as environment variable when submitting job
PYTHON=${PYTHON_PATH:-$(which python)}

python /u/gongt/scripts/peak_detect_localise/peak_localisation.py \
    --mua_detect_threshold 3.3 \
    --base_path /ptmp/gongt/GLC7189/GLC7189_g_g1/preprocessed \
    --folder_name thresholding_peaks_3.3_new \
    --support_radius 4 \
    --support_fraction 0.8 \
    --support_n_required 5 \
    --matched_filter_detect_threshold 3.3 \
    --matched_amplitude_threshold 3.0
