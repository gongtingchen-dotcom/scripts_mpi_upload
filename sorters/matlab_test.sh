#!/bin/bash
# Not part of the spike-sorting pipeline. One-off diagnostic: submit this to
# a GPU node to check that MATLAB can see and initialize the GPU
# (gpuDevice()) in this cluster environment — useful if a MATLAB+GPU job
# elsewhere is failing and you need to check whether it's an environment
# problem rather than a problem with that job itself. Submit with:
# sbatch matlab_test.sh, then check the log for the GPU device info (success)
# or an error message (environment problem).
#SBATCH -D ./
#SBATCH -o ./slurm/%j/logs/training.log
#SBATCH -e ./slurm/%j/logs/training.log
#SBATCH -J matlab
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --constraint=gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10000 # Increased to 10GB, 5GB is very tight for MATLAB + GPU

module purge
module load matlab
module load cuda/12.6

# Critical Apptainer/Singularity variables for Raven
export APPTAINER_NV=1
export APPTAINER_BIND="/usr/lib64/nvidia:/usr/lib64/nvidia"
export LD_LIBRARY_PATH="/usr/lib64/nvidia:$LD_LIBRARY_PATH"

unset DISPLAY

# Diagnostic: Verify the file exists on the host node before starting MATLAB
ls -l /usr/lib64/nvidia/libcuda.so.1

# Run MATLAB
matlab -nodisplay -nosplash -r "try, d = gpuDevice(); disp(d); catch ME, disp(ME.message); end; exit"