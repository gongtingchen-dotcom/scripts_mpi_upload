#!/bin/bash -l
# Initial working directory:
#SBATCH -D ./
# Output directory:
#SBATCH -o ./slurm/%j/logs/ks4.log
# Error directory:
#SBATCH -e ./slurm/%j/logs/ks4.log
# Job Name:
#SBATCH -J kilosort
#
# Number of nodes and MPI tasks per node:
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=64
#
#SBATCH --mem=500000 # Memory per node (in MB)
#
#SBATCH --mail-user=gong.tingchen@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --time=23:59:59

# Load modules in the correct order for PyTorch distributed training
# module purge
# module load matlab

# unset DISPLAY

# Activate the conda environment. 
#source /mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/etc/profile.d/conda.sh
conda activate phy2 

# Python path should be passed as environment variable when submitting job
PYTHON=${PYTHON_PATH:-$(which python)}

python /u/gongt/scripts/KS4.py \
    --root_folder /ptmp/gongt/GLC7248/GLC7248_a_g1 \
    --data_folder /ptmp/gongt/GLC7248/GLC7248_a_g1/preprocessed \
    --nblocks 1 \
    --Th_universal 9 \
    --Th_learned 8 \
    --whitening_range 32 \
    --nearest_chans 16 \
    --min_template_size 12 \
    --template_sizes 8 \
    --templates_from_data True \
    --n_templates 6 \
    --n_pcs 32 \
    --Th_single_ch 6 \
    --acg_threshold 0.1 \
    --ccg_threshold 0.05 \
    --cluster_downsampling 10 \
    --cluster_neighbors 10 \
    --do_CAR False \
    --n_jobs 36 \
    --save_preprocessed_copy False \
    --max_threads_per_worker 8 \
    --clear_cache True \
    --skip_kilosort_preprocessing False \
    --delete_recording_dat False \

# setting and preprocessing: 
    # "batch_size": 60000, 
    # "nblocks": 1, 
    # "Th_universal": 9, 
    # "Th_learned": 8, 
    # "nt": 61, 
    # "shift": None, 
    # "scale": None, 
    # "artifact_threshold": np.inf, 
    # "nskip": 25, 
    # "whitening_range": 32, 
    # "highpass_cutoff": 300
    # "binning_depth": 5, 
    # "sig_interp": 20, 
    # "drift_smoothing": [0.5, 0.5, 0.5]
# spike detection 
    # "nt0min": int(20*nt/61)
    # "dmin": None
    # "dminx": 32
    # "min_template_size": 10, 
    # "template_sizes": 5, 
    # "nearest_chans": 10, 
    # "nearest_templates": 100, 
    # "max_channel_distance": 32, 
    # "max_peels": 100, 
    # "templates_from_data": True, 
    # "n_templates": 6, 
    # "n_pcs": 6, 
    # "Th_single_ch": 6, 
# clustering: 
    # "acg_threshold": 0.2
    # "ccg_threshold": 0.25
    # "cluster_neighbors": 10
    # "cluster_downsampling": 20, 
    # "max_cluster_subset": 250000, 
    # "x_centers": None, 
# postprocessing: 
    # "duplicate_spike_ms": 0.25
    # "position_limit": 100, 
# spikeinterface ones 
    # "do_CAR": "car", 
    # "invert_sign": "inv",
    # "save_extra_vars": "evars", 
    # "save_preprocessed_copy": "spc", 
    # "torch_device": "gpu",
    # "bad_channels": "bad", 
    # "clear_cache": "clr", 
    # "do_correction": "dc",
    # "skip_kilosort_preprocessing": "skip", 
    # "keep_good_only": "good",
    # "use_binary_file": "bin", 
    # "delete_recording_dat": "del", 
    # "pool_engine": "pool",
    # "n_jobs": "jobs", 
    # "chunk_duration": "chk", 
    # "max_threads_per_worker": "thrd"