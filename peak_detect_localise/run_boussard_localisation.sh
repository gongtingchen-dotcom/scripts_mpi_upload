#!/bin/bash -l
# Initial working directory:
#SBATCH -D ./
# Output directory:
#SBATCH -o ./slurm/%j/logs/boussard_loc.log
# Error directory:
#SBATCH -e ./slurm/%j/logs/boussard_loc.log
# Job Name:
#SBATCH -J boussard_loc
#
# Number of nodes and MPI tasks per node:
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:2

#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1

#SBATCH --cpus-per-task=18
#
#SBATCH --mem=200000  # Memory per node (in MB)
#
#SBATCH --mail-user=gong.tingchen@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --time=23:00:00
#SBATCH --signal=B:USR1@120   # send SIGUSR1 to batch script 2 min before time limit

# Load modules in the correct order for PyTorch distributed training
# module purge
# module load matlab

# unset DISPLAY

# Activate the conda environment. 
#source /mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/etc/profile.d/conda.sh
conda activate spikeinterface

# Python path should be passed as environment variable when submitting job
PYTHON=${PYTHON_PATH:-$(which python)}

# Auto-resubmit when approaching the time limit:
# SLURM sends SIGUSR1 120 s before the wall-clock limit (see --signal above).
# We trap it: resubmit the same script, then kill Python cleanly and exit.
trap '
    echo "[$(date)] SIGUSR1: time limit in ~2 min — resubmitting job and exiting."
    sbatch /u/gongt/scripts/run_boussard_localisation.sh
    kill "$PYTHON_PID" 2>/dev/null
    wait "$PYTHON_PID" 2>/dev/null
    exit 0
' USR1

python /u/gongt/scripts/boussard_localization.py &
PYTHON_PID=$!
wait "$PYTHON_PID"