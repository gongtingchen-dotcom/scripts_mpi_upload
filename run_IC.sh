#!/bin/bash -l
#SBATCH -D ./
#SBATCH -o ./slurm/%j/logs/training.log
#SBATCH -e ./slurm/%j/logs/training.log
#SBATCH -J IC
#SBATCH --constraint=gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1   # Changed to 1 GPU for faster queue times
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18  # 18 cores is usually plenty for 1 GPU
#SBATCH --mem=64000         # 64GB is plenty for this recording size
#SBATCH --time=23:00:00
#SBATCH --mail-user=tingchen.gong@brain.mpg.de
#SBATCH --mail-type=ALL

# 1. Load Environment
module purge
module load cuda/12.6
module load anaconda/3/2023.03

# 2. Container (Apptainer) Configuration
export APPTAINER_NV=1
# Bind nexus, u, and ptmp so the container can see data, scripts, and cache
export APPTAINER_BIND="/nexus:/nexus,/u:/u,/ptmp:/ptmp,/tmp:/tmp"
export APPTAINER_CACHEDIR="/ptmp/gongt/.apptainer_cache"
mkdir -p $APPTAINER_CACHEDIR

# 3. Environment Paths
export QT_QPA_PLATFORM=offscreen
unset DISPLAY
# Ensure system binaries (apptainer) are found
export PATH=$PATH:/usr/bin:/usr/local/bin

# 4. Activate Conda by absolute path
source /mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/etc/profile.d/conda.sh
conda activate /u/gongt/conda-envs/phy2

# Debug Info
echo "Checking environment details:"
which python
python -c "import spython; print('spython version:', spython.__version__)"
which apptainer || which singularity

# 5. Run IronClust
echo "--- Starting IronClust via Container ---"
nvidia-smi

python /u/gongt/scripts/ironclust.py