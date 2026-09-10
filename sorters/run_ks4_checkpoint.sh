#!/bin/bash -l
# Kilosort 4 with checkpointing + SLURM auto-requeue.
#
# USE THIS ONE for kilosort4 runs. Absolutely necessary for recordings long enough that
# sorting won't finish inside one 24h SLURM allocation (the job checkpoints
# itself and SLURM requeues it automatically instead of the job dying at
# walltime) — but it works fine for short recordings too, so there's no real
# reason to prefer plain run_ks4.sh over this.
#
# Differences vs run_ks4.sh:
#   --requeue                : SLURM puts the job back in the queue on signal.
#   --signal=B:USR1@900      : SLURM sends SIGUSR1 to the batch script 900s
#                              before walltime. ks4_checkpoint.py traps it,
#                              writes a checkpoint, and exits 0.
#   --open-mode=append       : log file appended across restarts.
#
# On restart, ks4_checkpoint.py reads _checkpoints/progress.json in the output
# folder and resumes from the last completed stage. The first time you submit
# this job, the output folder must not contain an outdated _checkpoints/
# subdirectory from a previous (different) parameter set; the run name encodes
# the parameters so this normally takes care of itself.

#SBATCH -D ./
#SBATCH -o ./slurm/%j/logs/ks4.log
#SBATCH -e ./slurm/%j/logs/ks4.log
#SBATCH -J kilosort_ckpt
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500000
#SBATCH --mail-user=gong.tingchen@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=23:59:55

# --- Checkpoint requeue plumbing -------------------------------------------
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
#SBATCH --open-mode=append
# ---------------------------------------------------------------------------

# Trap USR1 in the batch script as well so the trap is registered before
# Python starts. Python's own handler (installed by ks4_checkpoint.py) is
# what actually flushes state; this shell trap just makes sure the signal
# doesn't terminate the script via the default disposition.
USR1_RECEIVED=0
trap 'echo "[sbatch] USR1 received at $(date), forwarding to python (PID $PYPID)"; kill -USR1 $PYPID 2>/dev/null; USR1_RECEIVED=1' SIGUSR1
trap 'echo "[sbatch] TERM received at $(date), forwarding to python (PID $PYPID)"; kill -TERM $PYPID 2>/dev/null' SIGTERM

echo "[sbatch] job $SLURM_JOB_ID restart $SLURM_RESTART_COUNT starting at $(date)"

conda activate phy2

# (Used to override PYTHONPATH to a local "Kilosort-main" checkout here in
# preference to the conda env's kilosort. That directory no longer exists and
# isn't referenced anywhere else — the phy2 env's own kilosort (4.1.1) is
# already patched in place and satisfies the KS_MIN check in
# ks4_checkpoint.py, so this was already a silent no-op. Removed.)

PYTHON=${PYTHON_PATH:-$(which python)}

# Sanity-print which kilosort is being used (one line in the log).
"$PYTHON" -c "import kilosort, inspect; print('[ks4] using', inspect.getfile(kilosort))"

# Run python in the background so the shell can react to USR1 immediately.
# Wait on it; if it exits 0 with a partial-progress sentinel file present,
# SLURM's --requeue will put the job back in the queue when we exit.
"$PYTHON" /u/gongt/scripts/sorters/ks4_checkpoint.py \
    --root_folder /nexus/posix0/MBR-neuralsystems/ephys/GLC6962/GLC6962_q_g0 \
    --data_folder /nexus/posix0/MBR-neuralsystems/ephys/GLC6962/GLC6962_q_g0/preprocessed \
    --nblocks 1 \
    --Th_universal 9 \
    --Th_learned 8 \
    --whitening_range 32 \
    --nt0min 20 \
    --nearest_chans 64 \
    --min_template_size 10 \
    --template_sizes 5 \
    --nearest_templates 100 \
    --max_channel_distance 32 \
    --templates_from_data True \
    --n_templates 6 \
    --n_pcs 6 \
    --Th_single_ch 6 \
    --acg_threshold 1.0 \
    --ccg_threshold 1.5 \
    --cluster_downsampling 1 \
    --cluster_neighbors 10 \
    --do_CAR False \
    --n_jobs -1 \
    --save_preprocessed_copy True \
    --max_threads_per_worker 4 \
    --clear_cache False \
    --skip_kilosort_preprocessing False \
    --delete_recording_dat True &

PYPID=$!
# Loop on wait: in bash, wait returns early (with 128+signum) when a signal
# arrives and a trap fires, even if the child is still alive.  Without the
# loop the shell would exit immediately on USR1, killing Python before it can
# checkpoint.  We keep waiting until Python has actually exited.
RC=0
while true; do
    wait $PYPID
    RC=$?
    # If Python is still alive the wait was interrupted by a signal; keep waiting.
    kill -0 $PYPID 2>/dev/null || break
done
echo "[sbatch] python exited with code $RC at $(date)"

# --- Requeue logic ---------------------------------------------------------
# Case 1: Python checkpointed cleanly and exited with REQUEUE_EXIT_CODE (75).
#         Explicitly requeue so the next allocation resumes from the checkpoint.
# Case 2: USR1 was received (SLURM time-limit warning) but Python was stuck
#         inside a non-interruptible stage (e.g. drift) and was killed by
#         SIGTERM (RC=143) or SIGKILL (RC=137) at walltime.  Requeue anyway
#         so the next run resumes from the last completed stage checkpoint.
# Any other non-zero RC is a genuine error; do not requeue.
if [[ $RC -eq 75 ]]; then
    echo "[sbatch] checkpoint exit, requeuing job $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
elif [[ $USR1_RECEIVED -eq 1 ]] && [[ $RC -eq 143 || $RC -eq 137 ]]; then
    echo "[sbatch] killed at walltime after USR1; requeuing from last stage checkpoint"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
# ---------------------------------------------------------------------------

exit $RC
