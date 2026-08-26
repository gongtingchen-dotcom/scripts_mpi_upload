#!/bin/bash -l
# Step 1 of motion estimation (preferred pipeline). SLURM array job — one
# task per time chunk of the recording. Edit BASE_PATH/FOLDER_NAME/chunk
# duration in run_dredge_chunked.py, and the --array range below to match
# the number of chunks your recording needs, then: sbatch run_dredge_chunks.sh
# Wait for the WHOLE array to finish, then separately submit
# run_dredge_stitch.sh to combine the chunks — these are two deliberately
# separate, manually-run steps (see NOTE below), not one job.
#
# NOTE: don't fold the --stitch step into this script. This is a SLURM
# ARRAY job — tasks run independently and in parallel, with no built-in way
# for one task to know whether the others have finished. If a task called
# --stitch right after its own --chunk_idx, the first task to finish would
# stitch whatever chunks happen to exist yet — silently producing an
# incomplete/wrong result, not a clean error (stitch_chunks() just globs
# whatever's on disk; it doesn't check that every expected chunk is there).
# Multiple tasks would also race to write the same output file. Submit
# run_dredge_stitch.sh yourself once you've confirmed (e.g. via `sacct`)
# that every array task completed successfully.
#
# Initial working directory:
#SBATCH -D ./
# Output / error logs per chunk:
#SBATCH -o ./slurm/%j/logs/dredge_chunk_%a.log
#SBATCH -e ./slurm/%j/logs/dredge_chunk_%a.log
# Job name:
#SBATCH -J dredge_chunk
#
# Array: one job per chunk (0–10 for a 10h recording; adjust upper bound if needed)
#SBATCH --array=0-10
#
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4

#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=64

#SBATCH --mem=500000     # 500 GB — each 70min chunk needs ~10–30 GB

#SBATCH --mail-user=gong.tingchen@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --time=23:59:59

conda activate spikeinterface

PYTHON=${PYTHON_PATH:-$(which python)}

mkdir -p ./slurm/$SLURM_JOB_ID/logs

echo "Running DREDge for chunk index: $SLURM_ARRAY_TASK_ID  depth_args=${DEPTH_ARGS:-none}"
$PYTHON /u/gongt/scripts/motion/run_dredge_chunked.py --chunk_idx $SLURM_ARRAY_TASK_ID --run_name 2
