#!/bin/bash -l
# Step 1 of motion estimation (preferred pipeline). SLURM array job — one
# task per time chunk of the recording. Edit BASE_PATH/FOLDER_NAME/chunk
# duration in run_dredge_chunked.py, and the --array range below to match
# the number of chunks your recording needs, then: sbatch run_dredge_chunks.sh
# Wait for the WHOLE array to finish (confirm via `sacct`), then separately
# submit run_dredge_stitch.sh — never fold --stitch into this array job
# itself (race condition, see run_dredge_chunked.py's --stitch docs for why).
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
