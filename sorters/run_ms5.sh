#!/bin/bash -l
# Initial working directory:
#SBATCH -D ./
# Output directory:
#SBATCH -o ./slurm/%j/logs/training.log
# Error directory:
#SBATCH -e ./slurm/%j/logs/training.log
# Job Name:
#SBATCH -J ms5
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

python /u/gongt/scripts/ms5.py \
  --root_folder /nexus/posix0/MBR-neuralsystems/ephys/GLC6811/GLC6811__g2/snippet \
  --data_folder /nexus/posix0/MBR-neuralsystems/ephys/GLC6811/GLC6811__g2/snippet/preprocessed \
  --phase1_detect_channel_radius None \
  --detect_channel_radius None \
  --phase1_detect_threshold 4.5 \
  --phase1_npca_per_channel 6 \
  --detect_sign 0 \
  --detect_threshold 4.5 \
  --snippet_T1 40 \
  --snippet_T2 40 \
  --training_recording_sampling_mode 'uniform'


#default Scheme2SortingParameters(
# phase1_detect_channel_radius=200, 
# detect_channel_radius=100, 
# phase1_detect_threshold=5.5, 
# phase1_detect_time_radius_msec=1.5, 
# detect_time_radius_msec=0.5, 
# phase1_npca_per_channel=3, 
# phase1_npca_per_subdivision=10, 
# phase1_pairwise_merge_step=False, 
# detect_sign=-1, 
# detect_threshold=5.5, 
# snippet_T1=20, 
# snippet_T2=20, 
# snippet_mask_radius=None, 
# max_num_snippets_per_training_batch=200, 
# classifier_npca=None, 
# training_duration_sec=None, 
# training_recording_sampling_mode='initial', 
# classification_chunk_sec=None)