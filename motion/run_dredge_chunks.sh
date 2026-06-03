#!/bin/bash -l
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
#SBATCH --gres=gpu:a100:1

#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=18

#SBATCH --mem=100000     # 100 GB — each 70min chunk needs ~10–30 GB

#SBATCH --mail-user=gong.tingchen@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --time=2:00:00

conda activate spikeinterface

PYTHON=${PYTHON_PATH:-$(which python)}

mkdir -p ./slurm/$SLURM_JOB_ID/logs

echo "Running DREDge for chunk index: $SLURM_ARRAY_TASK_ID  depth_args=${DEPTH_ARGS:-none}"
$PYTHON /u/gongt/scripts/motion/run_dredge_chunked.py --chunk_idx $SLURM_ARRAY_TASK_ID --run_name 1
