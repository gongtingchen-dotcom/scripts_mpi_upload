#!/bin/bash -l
# Step 2 of motion estimation (preferred pipeline) — run after all chunks
# from run_dredge_chunks.sh have finished. Combines them into one motion
# estimate via run_dredge_chunked.py's own --stitch mode (aligns overlapping
# chunks, not the separate stitch_motion.py — see that file's header).
# Submit with: sbatch run_dredge_stitch.sh
#
# REQUIRES the chunk estimates to already exist under
# motion_estimates/chunks_<run_name>/ (from run_dredge_chunks.sh). If none
# are found, --stitch fails immediately and cleanly with
# FileNotFoundError("No chunks found in ..."), it does not run and wait.
# Run these two steps separately: submit run_dredge_chunks.sh first, confirm
# (e.g. via `sacct`) that every array task in it finished successfully, THEN
# submit this script — submitting it too early will just error out.
#
#SBATCH -D ./
#SBATCH -o ./slurm/%j/logs/dredge_stitch.log
#SBATCH -e ./slurm/%j/logs/dredge_stitch.log
#SBATCH -J dredge_stitch

#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4

#SBATCH --mem=100000     # stitching is just array operations, very light

#SBATCH --mail-user=gong.tingchen@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --time=5:00:00

conda activate spikeinterface

PYTHON=${PYTHON_PATH:-$(which python)}

mkdir -p ./slurm/$SLURM_JOB_ID/logs

$PYTHON /u/gongt/scripts/motion/run_dredge_chunked.py --stitch --run_name 2
