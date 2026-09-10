#!/usr/bin/env python3
"""Kilosort 4 driver with stage-level checkpointing + SLURM USR1 requeue.

A good default. Run via run_ks4_checkpoint.sh. Same pipeline and output as
plain KS4.py; the addition is that the job checkpoints itself and gets
SLURM-requeued automatically, resuming from its last completed stage instead
of dying at walltime. That matters for recordings long enough that sorting
won't finish inside a single 24h SLURM allocation; on shorter recordings the
checkpointing is just a no-op safety net, so it's fine to use for everything.
KS4.py is equally fine when you don't need the resume behaviour.

Drop-in replacement for KS4.py. Same CLI. Differences:
  * Output folder is NOT wiped if a checkpoint marker is present; the run
    resumes from the last completed stage.
  * Each kilosort stage (preprocessing, drift, detect, cluster, save) writes
    a checkpoint to <output>/_checkpoints/.
  * clustering_qr.run (the largest single stage) writes intra-stage
    checkpoints every N x-y centers via the patched kilosort/clustering_qr.py.
  * SIGUSR1 (sent by SLURM ~900s before walltime via --signal=B:USR1@900)
    flips an interrupt flag. The flag is polled at every (jj, kk) pair inside
    clustering_qr.run and at every stage boundary. When set, partial state is
    flushed and the process exits 0; SLURM then requeues the job.

Resume invariant: rerunning the exact same command on an existing output
folder picks up where the previous run stopped.

ENVIRONMENT NOTE — kilosort4's own auto-merge step is disabled: the `phy2`
conda env's installed kilosort package has the `template_matching.merging_function`
call inside `cluster_spikes()` (kilosort/run_kilosort.py) commented out by hand
("skipped merging..." printed instead). This is a hand-edit to the installed
package itself, not something this script does — it is NOT reproducible by
`pip install kilosort==<version>` alone; redo that edit (or just delete the
merging_function call) after installing kilosort into a fresh `phy2` env. See
environments/README.md. Effect: clusters coming out of Stage 4 (cluster_spikes)
are kilosort4's raw per-template clusters, un-merged — whatever downstream
merging/curation you do on this output needs to account for that (it hasn't
already been through kilosort4's own merge step).

USAGE / EXAMPLE (via run_ks4_checkpoint.sh [recommended], or otherwise directly):
  python ks4_checkpoint.py \
      --root_folder /nexus/posix0/MBR-neuralsystems/ephys/GLC7189/GLC7189_g_g1 \
      --data_folder /nexus/posix0/MBR-neuralsystems/ephys/GLC7189/GLC7189_g_g1/preprocessed \
      --nblocks 1 --Th_universal 9 --Th_learned 8

PARAMETERS: every kilosort4 parameter from si.get_default_sorter_params
('kilosort4') is exposed as a CLI flag automatically (see the argparse loop
below) — pass any of them the same way. Plus:
  --root_folder / --data_folder   required. Session dir and its preprocessed/
                        folder (the recording this sort runs on).
  --time_start / --time_end        optional, seconds. Restrict sorting to a
                        sub-window of the recording.

What differs from kilosort4's own defaults gets encoded into the output
run-folder name (see build_run_name below), so it's safe to pass many flags.

OUTPUTS, all under <root_folder>/KS4/<run_name>/ (run_name auto-built from
whichever params you changed from default, e.g. "nb1_thu9_thl8"):
  sorter_output/            — the actual sort result, Phy-compatible:
                        spike_times.npy, spike_clusters.npy, templates.npy,
                        amplitudes.npy, channel_map.npy, cluster_*.tsv, plus
                        ops.npy (full run state — also what
                        kilosort_diagnostic_plots-style notebooks read).
  sorter_output/_checkpoints/   — this script's own checkpoint state (safe to
                        delete once a run has finished successfully). Two kinds:
    progress.json               — which named stages have completed
                        (preprocessing/drift/detect_univ/detect_clu0/detect/
                        cluster/save) + SLURM job id/restart count. Read this
                        first to see how far a run got.
    ops_after_<stage>.npy       — one per completed stage: the full KS4 ops
                        dict (whitening matrix, drift estimate, etc.) as it
                        stood right after that stage finished.
    st0.npy / st0_univ.npy / st_after_detect.npy / st_after_cluster.npy,
    tF_univ.pt / tF_after_detect.pt / tF_after_cluster.pt,
    clu_after_clu0.npy / clu_after_cluster.npy,
    Wall_after_clu0.pt / Wall3_after_clu0.pt / Wall_after_cluster.pt
                                — spike times / template features / cluster
                        assignments / cluster weight matrices at each stage
                        boundary — this is what a restart actually resumes from.
    clustering_qr/<mode>/       — INTRA-stage checkpoints for clustering_qr
                        (the single largest stage) — saved every SAVE_EVERY
                        pair-iterations, not just at the stage boundary, so a
                        walltime kill doesn't lose an entire long clustering
                        run. `_complete` marks a mode as finished.
    extract_partial_*.json/.npy, extract_tF_memmap.dat
                                — INTRA-stage checkpoints for feature
                        extraction (the other slow step), saved per-batch.
                        tF is kept as a disk-backed memmap rather than an
                        in-memory array specifically to avoid OOM on large
                        recordings (documented inline where this is set up).
  kilosort_params.txt        — every parameter this run actually used, for
                        reproducibility.
"""

import argparse
import json
import os
import shutil
import signal
import sys
import time
import traceback
from datetime import datetime
from importlib.metadata import version as _importlib_version
from pathlib import Path

import numpy as np
import spikeinterface.full as si
from spikeinterface.core.core_tools import check_json

# Populated inside main() as soon as the relevant values exist, so the
# top-level error handler (see bottom of file) can still write a
# spikeinterface_log.json with error=True if a stage raises — mirrors what
# spikeinterface.sorters.basesorter.BaseSorter.run_from_folder does on
# failure, without having to wrap the whole multi-stage body in one giant
# try/except.
_SI_BOOKKEEPING_CTX = {}


# --------------------------------------------------------------------------- #
# CLI: identical to KS4.py so the sbatch wrapper can be swapped in.
# --------------------------------------------------------------------------- #

PARAM_ABBREV = {
    "batch_size": "bs", "nblocks": "nb", "Th_universal": "thu",
    "Th_learned": "thl", "nt": "nt", "shift": "shft", "scale": "scl",
    "artifact_threshold": "art", "nskip": "nskp", "whitening_range": "wr",
    "highpass_cutoff": "hp", "binning_depth": "bd", "sig_interp": "si",
    "drift_smoothing": "dsm", "nt0min": "nt0m", "dmin": "dmin",
    "dminx": "dmix", "min_template_size": "mts", "template_sizes": "tsz",
    "nearest_chans": "nc", "nearest_templates": "ntmp",
    "max_channel_distance": "mcd", "max_peels": "mp",
    "templates_from_data": "tfd", "n_templates": "n_t", "n_pcs": "pcs",
    "Th_single_ch": "thsc", "acg_threshold": "acg", "ccg_threshold": "ccg",
    "cluster_neighbors": "cn", "cluster_downsampling": "cd",
    "max_cluster_subset": "mcs", "x_centers": "xc", "duplicate_spike_ms": "dup",
    "position_limit": "plim", "do_CAR": "car", "invert_sign": "inv",
    "save_extra_vars": "evars", "save_preprocessed_copy": "spc",
    "torch_device": "gpu", "bad_channels": "bad", "clear_cache": "clr",
    "do_correction": "dc", "skip_kilosort_preprocessing": "skip",
    "keep_good_only": "good", "use_binary_file": "bin",
    "delete_recording_dat": "del", "pool_engine": "pool", "n_jobs": "jobs",
    "chunk_duration": "chk", "max_threads_per_worker": "thrd",
    "time_end": "te", "time_start": "ts",
}

INT_PARAMS = [
    "nt0min", "nt", "batch_size", "nblocks", "whitening_range",
    "nearest_chans", "n_templates", "n_pcs", "max_threads_per_worker",
    "time_end", "time_start",
]


def build_run_name(defaults, current, extra_params=None):
    tags = []
    for k, v in current.items():
        if k in defaults and str(defaults[k]) != str(v):
            abbr = PARAM_ABBREV.get(k, k)
            clean_v = int(v) if isinstance(v, (float, int)) and v == int(v) else v
            tags.append(f"{abbr}{clean_v}")
    if extra_params:
        for k, v in extra_params.items():
            if v is not None:
                abbr = PARAM_ABBREV.get(k, k)
                clean_v = int(v) if isinstance(v, (float, int)) and v == int(v) else v
                tags.append(f"{abbr}{clean_v}")
    return "_".join(tags) if tags else "default"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def _write_si_log_json(output_folder, sorter_output, sorter_version,
                        error, error_trace, run_time):
    """spikeinterface_log.json, matching BaseSorter.run_from_folder.

    Lets si.read_sorter_result() / si.get_result_from_folder() work on
    ks4_checkpoint.py's output the same as it does on plain KS4.py's
    (si.run_sorter) output.
    """
    runtime_trace = []
    runtime_trace_path = Path(sorter_output) / "kilosort4.log"
    if runtime_trace_path.is_file():
        with open(runtime_trace_path, "r") as fp:
            runtime_trace = [line.strip() for line in fp]
    log = {
        "sorter_name": "kilosort4",
        "sorter_version": str(sorter_version),
        "datetime": datetime.now(),
        "runtime_trace": runtime_trace,
        "error": bool(error),
        "run_time": run_time,
    }
    if error_trace is not None:
        log["error_trace"] = error_trace
    with (Path(output_folder) / "spikeinterface_log.json").open(
            "w", encoding="utf8") as f:
        json.dump(check_json(log), f, indent=4)


# --------------------------------------------------------------------------- #
# Checkpointing primitives.
# --------------------------------------------------------------------------- #

# Set by SIGUSR1 handler. Polled at safe boundaries.
INTERRUPT = {"flag": False, "received_at": None, "signum": None}

# Exit code reserved for "checkpointed and exiting cleanly for requeue".
# Must match the RC -eq check in run_ks4_checkpoint.sh.
REQUEUE_EXIT_CODE = 75

# Batches between intra-extract periodic checkpoint saves (~25 min apart).
EXTRACT_SAVE_EVERY = 2000


def _signal_handler(signum, frame):
    INTERRUPT["flag"] = True
    INTERRUPT["received_at"] = time.time()
    INTERRUPT["signum"] = signum


def install_signal_handlers():
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        try:
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            # E.g. not the main thread, or not supported on platform.
            pass


def atomic_save_npy(path, arr):
    path = Path(path)
    # Use stem+'.tmp'+suffix so the temp name ends in '.npy'.
    # np.save silently appends '.npy' to paths that don't already end in it,
    # which would make os.replace fail with FileNotFoundError.
    tmp = path.parent / (path.stem + '.tmp' + path.suffix)
    np.save(tmp, arr, allow_pickle=True)
    os.replace(tmp, path)


def atomic_save_torch(path, obj):
    import torch
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_save_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


class StageLog:
    """Tracks which top-level stages have completed."""

    def __init__(self, ckpt_dir):
        self.path = Path(ckpt_dir) / 'progress.json'
        self.state = {
            "completed_stages": [],
            "kilosort_version": None,
            "started_at": datetime.now().isoformat(),
            "last_updated": None,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
        }
        if self.path.exists():
            try:
                with open(self.path) as f:
                    saved = json.load(f)
                # Preserve completed list; refresh runtime metadata.
                self.state["completed_stages"] = saved.get(
                    "completed_stages", []
                )
                self.state["started_at"] = saved.get(
                    "started_at", self.state["started_at"]
                )
            except json.JSONDecodeError:
                print(f"[checkpoint] progress.json unreadable, starting fresh")

    def is_done(self, stage):
        return stage in self.state["completed_stages"]

    def mark(self, stage, **meta):
        if stage not in self.state["completed_stages"]:
            self.state["completed_stages"].append(stage)
        self.state["last_updated"] = datetime.now().isoformat()
        self.state.update(meta)
        atomic_save_json(self.path, self.state)


class ClusteringCheckpoint:
    """Intra-stage checkpoint hook for kilosort.clustering_qr.run().

    Saves partial (clu, Wall, counters, next_pair_idx) every SAVE_EVERY pairs
    and whenever the global INTERRUPT flag flips. On resume, load() returns
    the saved state so the loop continues from the right pair.

    There are two invocations of clustering_qr.run per pipeline:
      mode='spikes'    -> inside detect_spikes (first clustering)
      mode='template'  -> inside cluster_spikes (final clustering)
    They have separate checkpoint files.
    """

    SAVE_EVERY = 10  # pair-iterations between checkpoint writes

    def __init__(self, ckpt_dir, interrupt_dict):
        self.dir = Path(ckpt_dir) / 'clustering_qr'
        self.dir.mkdir(parents=True, exist_ok=True)
        self.interrupt = interrupt_dict
        # Track last-saved index per mode to avoid double-writes.
        self._last_saved = {}

    def _paths(self, mode):
        d = self.dir / mode
        return {
            "dir": d,
            "meta": d / "state.json",
            "clu": d / "clu.npy",
            "wall": d / "Wall.pt",
            "complete": d / "_complete",
        }

    def load(self, mode, expected_nsp, expected_total):
        import torch
        p = self._paths(mode)
        if p["complete"].exists():
            # Stage already finished — caller should detect and skip entirely.
            return None
        if not p["meta"].exists():
            return None
        try:
            with open(p["meta"]) as f:
                meta = json.load(f)
            if meta.get("nsp") != int(expected_nsp):
                print(f"[checkpoint] nsp mismatch in clustering_qr {mode} "
                      f"({meta.get('nsp')} vs {expected_nsp}); ignoring.")
                return None
            clu = np.load(p["clu"])
            Wall = torch.load(p["wall"], map_location='cpu', weights_only=False)
            return (
                clu, Wall,
                int(meta["nmax"]),
                int(meta["t"]),
                int(meta["nearby_chans_empty"]),
                int(meta["next_pair_idx"]),
            )
        except Exception as e:
            print(f"[checkpoint] failed to load clustering_qr {mode}: {e}")
            return None

    def should_save(self, pair_idx, mode=None):
        last = self._last_saved.get(mode, -1)
        if pair_idx == 0 or pair_idx == last:
            return False
        return (pair_idx - last) >= self.SAVE_EVERY

    def should_stop(self):
        return bool(self.interrupt.get("flag"))

    def save(self, mode, clu, Wall, nmax, t, nearby_chans_empty,
             next_pair_idx, total_pairs, nsp, total_centers):
        p = self._paths(mode)
        p["dir"].mkdir(parents=True, exist_ok=True)
        atomic_save_npy(p["clu"], np.ascontiguousarray(clu))
        atomic_save_torch(p["wall"], Wall.cpu())
        atomic_save_json(p["meta"], {
            "mode": mode,
            "nmax": int(nmax),
            "t": int(t),
            "nearby_chans_empty": int(nearby_chans_empty),
            "next_pair_idx": int(next_pair_idx),
            "total_pairs": int(total_pairs),
            "nsp": int(nsp),
            "total_centers": int(total_centers),
            "saved_at": datetime.now().isoformat(),
        })
        self._last_saved[mode] = int(next_pair_idx)

    def mark_complete(self, mode):
        p = self._paths(mode)
        p["dir"].mkdir(parents=True, exist_ok=True)
        p["complete"].touch()


# --------------------------------------------------------------------------- #
# Main driver: replicates spikeinterface.sorters.external.kilosort4 internals,
# splitting around our checkpoint boundaries.
# --------------------------------------------------------------------------- #

def main():
    install_signal_handlers()

    # Import kilosort lazily so import errors surface inside our log.
    import torch
    import kilosort
    from kilosort import io as ks_io
    from kilosort.run_kilosort import (
        set_files, initialize_ops, compute_preprocessing,
        compute_drift_correction, cluster_spikes,
        save_sorting, setup_logger,
    )
    from kilosort.parameters import DEFAULT_SETTINGS
    # clustering_qr: called directly in Stage 3 and monkey-patched for Stage 4
    # (cluster_spikes calls it internally without a checkpoint kwarg).
    from kilosort import clustering_qr, spikedetect, template_matching
    import kilosort.plots as kplots
    from torch.nn.functional import conv1d

    # --- Patch prepare_matching to avoid 77 GB UtU intermediate ----------- #
    # With ~4000 templates the original einsum materialises (N,N,K,K) float32
    # which is ~77 GB and OOMs on a 40 GB A100.  We chunk over the first
    # template index so the largest intermediate is (chunk, N, K, K) ~ 2 GB.
    def _chunked_prepare_matching(ops, U, chunk_size=128):
        nt  = ops['nt']
        W   = ops['wPCA'].contiguous()
        WtW = conv1d(W.reshape(-1, 1, nt), W.reshape(-1, 1, nt), padding=nt)
        WtW = torch.flip(WtW, [2,])
        N   = U.shape[0]
        T   = WtW.shape[-1]
        ctc = torch.zeros(N, N, T, dtype=U.dtype, device=U.device)
        for i in range(0, N, chunk_size):
            Ui             = U[i:i + chunk_size]                         # (B, K, L)
            UtU_chunk      = torch.einsum('bkl, jml -> bjkm', Ui, U)    # (B, N, K, K)
            ctc[i:i + chunk_size] = torch.einsum('bjkm, kmt -> bjt', UtU_chunk, WtW)
        return ctc

    template_matching.prepare_matching = _chunked_prepare_matching
    print("[checkpoint] patched template_matching.prepare_matching with chunked version")

    # --- Patch extract to pre-allocate tF at the right size --------------- #
    # The original extract starts tF at 10^6 and doubles it each time it fills
    # up.  With ~43 M spikes the array doubles 6-7 times; the final doubling
    # (32 M → 64 M) temporarily allocates 32+32+64 = 128 M entries = 262 GB.
    # Combined with the 88 GB tF_univ still in memory that pushes past 512 GB.
    # Fix: pre-allocate once at 1.5 × the detect_univ spike count.
    # _extract_n_pre is set just before the extract() call below.
    _extract_n_pre_holder  = [None]  # mutable cell so the closure can read it
    _extract_ckpt_dir_holder = [None]  # set to ckpt_dir just before extract()

    _orig_extract = template_matching.extract
    from kilosort.template_matching import prepare_extract, run_matching
    from tqdm import tqdm as _tqdm
    import logging as _logging
    _tm_logger = _logging.getLogger('kilosort.template_matching')

    def _extract_preallocated(ops, bfile, U, device=torch.device('cuda'),
                              progress_bar=None):
        _ckpt  = _extract_ckpt_dir_holder[0]   # ckpt_dir Path or None
        n_pre  = _extract_n_pre_holder[0]
        nC     = ops['settings']['nearest_chans']
        n_pcs  = ops['settings']['n_pcs']
        nt     = ops['nt']

        if _ckpt is None or n_pre is None:
            print("[checkpoint] extract: no ckpt_dir or n_pre, falling back to original")
            return _orig_extract(ops, bfile, U, device=device,
                                 progress_bar=progress_bar)

        # Paths for partial-checkpoint files.
        # tF is stored as a numpy memmap (file-backed) so it does NOT occupy
        # anonymous RAM during the loop — the OS can evict its pages under
        # memory pressure, just like the recording mmap.  This removes the
        # ~133 GB anonymous pressure that caused the 508 GB OOM.
        _pstate = _ckpt / "extract_partial_state.json"
        _pst    = _ckpt / "extract_partial_st.npy"
        _ptF    = _ckpt / "extract_partial_tF.npy"   # legacy; migrated below
        _pmm    = _ckpt / "extract_tF_memmap.dat"    # primary tF storage

        ibatch_start = 0
        k = 0

        if _pstate.exists():
            with open(_pstate) as _f:
                _ps = json.load(_f)
            ibatch_start = int(_ps['ibatch']) + 1
            k             = int(_ps['k'])
            n_pre         = int(_ps['n_pre'])
            print(f"[checkpoint] extract: resuming from batch {ibatch_start}, "
                  f"k={k:,} spikes")
            _st_saved = np.load(_pst)
            st = np.zeros((n_pre, 3), 'float64')
            st[:k] = _st_saved
            del _st_saved

            if _pmm.exists():
                # Primary path: reopen existing memmap (data already in file).
                tF = np.memmap(_pmm, dtype='float32', mode='r+',
                               shape=(n_pre, nC, n_pcs))
                print(f"[checkpoint] extract: reopened tF memmap "
                      f"({n_pre * nC * n_pcs * 4 / 1e9:.1f} GB on disk)")
            elif _ptF.exists():
                # Legacy .npy checkpoint: migrate to memmap without loading
                # the full 100+ GB into RAM (use mmap_mode + chunk copy).
                print(f"[checkpoint] extract: migrating legacy tF.npy → memmap ")
                tF = np.memmap(_pmm, dtype='float32', mode='w+',
                               shape=(n_pre, nC, n_pcs))
                _tF_leg = np.load(_ptF, mmap_mode='r')  # file-backed, no RAM spike
                _CHUNK  = 500_000
                for _i in range(0, k, _CHUNK):
                    _end = min(_i + _CHUNK, k)
                    tF[_i:_end] = _tF_leg[_i:_end]
                tF.flush()
                del _tF_leg
                _ptF.unlink()   # remove legacy file to free /ptmp space
                print(f"[checkpoint] extract: migration done")
            else:
                # No tF data found: restart extraction from batch 0.
                print(f"[checkpoint] extract: no tF data found, restarting")
                tF = np.memmap(_pmm, dtype='float32', mode='w+',
                               shape=(n_pre, nC, n_pcs))
                k = 0
                ibatch_start = 0
        else:
            print(f"[checkpoint] extract: pre-allocating memmap for {n_pre:,} spikes "
                  f"({n_pre * nC * n_pcs * 4 / 1e9:.1f} GB on disk, ~0 GB anonymous)")
            st = np.zeros((n_pre, 3), 'float64')
            tF = np.memmap(_pmm, dtype='float32', mode='w+',
                           shape=(n_pre, nC, n_pcs))

        # Always recompute — deterministic and cheap compared to I/O
        iCC, iCC_mask, iU, Ucc = prepare_extract(
            ops['xc'], ops['yc'], U, nC, ops['settings']['position_limit'],
            device=device)
        ops['iCC']      = iCC
        ops['iCC_mask'] = iCC_mask
        ops['iU']       = iU
        tiwave = torch.arange(-(nt // 2), nt // 2 + 1, device=device)
        ctc    = template_matching.prepare_matching(ops, U)  # already chunked

        def _save_partial(ibatch_done):
            """Flush memmap + st to disk and write state file."""
            tF.flush()   # msync: ensure writes reach the filesystem
            atomic_save_npy(_pst, st[:k])
            atomic_save_json(_pstate,
                             {'ibatch': ibatch_done, 'k': k,
                              'n_pre': int(tF.shape[0])})
            print(f"[checkpoint] extract: saved at batch {ibatch_done}, k={k:,}")

        prog = _tqdm(
            np.arange(ibatch_start, bfile.n_batches, dtype=np.int64),
            miniters=200 if progress_bar else None,
            mininterval=60 if progress_bar else None,
        )
        try:
            for ibatch in prog:
                X = bfile.padded_batch_to_torch(ibatch, ops)
                stt, amps, th_amps, Xres = run_matching(
                    ops, X, U, ctc, device=device)
                xfeat = (Xres[iCC[:, iU[stt[:, 1:2]]], stt[:, :1] + tiwave]
                         @ ops['wPCA'].T)
                xfeat += amps * Ucc[:, stt[:, 1]]
                if ibatch == 0:
                    neg     = (stt[:, 0] - nt - nt // 2 + ops['nt0min']) < 0
                    stt     = stt[~neg, :]
                    xfeat   = xfeat[:, ~neg, :]
                    amps    = amps[~neg, :]
                    th_amps = th_amps[~neg, :]
                nsp = len(stt)
                if k + nsp > tF.shape[0]:
                    # Extend the memmap file cheaply — truncate the backing
                    # file to the new size (POSIX fills extension with zeros)
                    # then remap.  No large anonymous allocation.
                    extra = max(nsp, 10_000_000)
                    new_n = tF.shape[0] + extra
                    print(f"[checkpoint] extract: extending tF memmap from "
                          f"{tF.shape[0]:,} to {new_n:,} at batch {ibatch}")
                    tF.flush()
                    del tF
                    # Must truncate first; mode='r+' cannot grow the file.
                    os.truncate(_pmm, new_n * nC * n_pcs * 4)
                    tF = np.memmap(_pmm, dtype='float32', mode='r+',
                                   shape=(new_n, nC, n_pcs))
                    st_new     = np.zeros((new_n, 3), 'float64')
                    st_new[:k] = st[:k]
                    st = st_new
                stt = stt.double()
                st[k:k + nsp, 0] = (
                    (stt[:, 0] - nt) + ibatch * ops['batch_size']
                ).cpu().numpy() - nt // 2 + ops['nt0min']
                st[k:k + nsp, 1] = stt[:, 1].cpu().numpy()
                st[k:k + nsp, 2] = th_amps.cpu().numpy().squeeze()
                tF[k:k + nsp]    = xfeat.transpose(0, 1).contiguous().cpu().numpy()
                k += nsp

                if (ibatch + 1) % EXTRACT_SAVE_EVERY == 0:
                    _save_partial(ibatch)
                if INTERRUPT['flag']:
                    _save_partial(ibatch)
                    print("[checkpoint] extract: interrupted, "
                          "exiting for SLURM requeue.")
                    sys.exit(REQUEUE_EXIT_CODE)

                if progress_bar is not None:
                    progress_bar.emit(
                        int((ibatch + 1) / bfile.n_batches * 100))
        except SystemExit:
            raise
        except Exception:
            _tm_logger.exception(
                f'Error in _extract_preallocated on batch {ibatch}')
            raise

        # Sort by spike time.  tF is a memmap; read in 500 K-spike chunks to
        # bound peak anonymous RAM at ~1 GB per chunk regardless of total k.
        isort  = np.argsort(st[:k, 0])
        st_out = st[:k][isort]
        _CHUNK = 500_000
        tF_out = torch.empty(k, nC, n_pcs)
        for _i in range(0, k, _CHUNK):
            _end = min(_i + _CHUNK, k)
            _idx = isort[_i:_end]
            tF_out[_i:_end] = torch.from_numpy(
                np.ascontiguousarray(tF[_idx], dtype='float32'))
        tF.flush()
        del tF

        # Clean up memmap file and partial checkpoint on success.
        for _p in (_pmm, _pstate, _pst, _ptF):
            if _p is not None and _p.exists():
                _p.unlink()
        print(f"[checkpoint] extract: done, {k:,} spikes; cleared memmap")
        return st_out, tF_out, ops

    template_matching.extract = _extract_preallocated
    print("[checkpoint] patched template_matching.extract with pre-allocated version")
    # ----------------------------------------------------------------------- #
    from packaging import version as _pkgver

    # Fail fast if the installed kilosort predates the API this script targets.
    # 4.1.0 is the floor because we rely on:
    #   - cluster_spikes returning (clu, Wall, st, tF)  (4.0.31+)
    #   - detect_spikes returning  (st, tF, Wall0, clu0) (4.0.33+)
    #   - set_files accepting shank_idx=                  (4.0.34+)
    #   - kilosort.plots.*                                (4.0.33+)
    KS_MIN = "4.1.0"
    _ks_ver = getattr(kilosort, "__version__", "0")
    if _pkgver.parse(_ks_ver) < _pkgver.parse(KS_MIN):
        raise RuntimeError(
            f"ks4_checkpoint.py requires kilosort >= {KS_MIN}, "
            f"got {_ks_ver}. The PYTHONPATH override in "
            f"run_ks4_checkpoint.sh may be pointing at the wrong copy."
        )
    print(f"[ks4] kilosort {_ks_ver}")

    ks4_defaults = si.get_default_sorter_params('kilosort4')
    parser = argparse.ArgumentParser(description="set Kilosort4 params")
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--data_folder", required=True)
    parser.add_argument("--time_end", type=int, default=None)
    parser.add_argument("--time_start", type=int, default=0)
    for param, default_val in ks4_defaults.items():
        if isinstance(default_val, bool):
            parser.add_argument(f"--{param}", type=str2bool, default=default_val)
        elif isinstance(default_val, (list, tuple)):
            parser.add_argument(f"--{param}", nargs='+', type=float,
                                default=default_val)
        else:
            parser.add_argument(
                f"--{param}",
                type=type(default_val) if default_val is not None else float,
                default=default_val,
            )
    args = parser.parse_args()
    args_dict = vars(args)

    current_ks_params = {}
    for k in ks4_defaults.keys():
        val = args_dict[k]
        if k in INT_PARAMS and val is not None:
            current_ks_params[k] = int(val)
        else:
            current_ks_params[k] = val

    time_end = args_dict.get('time_end', None)
    time_start = args_dict.get('time_start', 0) or 0
    extra_params = {}
    if time_end is not None:
        extra_params["time_end"] = time_end
    if time_start:
        extra_params["time_start"] = time_start

    run_name = build_run_name(ks4_defaults, current_ks_params, extra_params)
    output_folder = Path(args.root_folder) / "KS4" / run_name
    data_folder = args.data_folder

    output_folder.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_folder / "_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sorter_output = output_folder / "sorter_output"
    sorter_output.mkdir(parents=True, exist_ok=True)
    _SI_BOOKKEEPING_CTX["output_folder"] = output_folder
    _SI_BOOKKEEPING_CTX["sorter_output"] = sorter_output

    progress = StageLog(ckpt_dir)
    progress.state["kilosort_version"] = getattr(kilosort, "__version__", "unknown")
    restart = progress.state["slurm_restart_count"]
    print(f"[checkpoint] output: {output_folder}")
    print(f"[checkpoint] SLURM restart count: {restart}")
    print(f"[checkpoint] completed stages so far: "
          f"{progress.state['completed_stages']}")

    # ------------------------------------------------------------------ #
    # 1. Build the SpikeInterface recording (matches KS4.py).
    # ------------------------------------------------------------------ #
    print(f"Data folder: {data_folder}")
    if "imec" in data_folder:
        spikeglx_folder = data_folder
        for meta_filename in Path(spikeglx_folder).glob("*ap.meta"):
            print(meta_filename)
        with open(meta_filename, 'r') as meta_info:
            lines = meta_info.readlines()
            for i, line in enumerate(lines):
                if line.find('imSampRate') != -1:
                    fs = float(line[11:])
                    print(fs)
                    break
        rec0 = si.read_spikeglx(spikeglx_folder, stream_name='imec0.ap',
                                load_sync_channel=False)
        outliers_path = os.path.join(args.root_folder,
                                     "outliers_list_sqr_overview.txt")
        outliers_list = np.loadtxt(outliers_path, delimiter=',')
        list_triggers = outliers_list
        list_triggers_samp = list_triggers * fs
        print(f"number of outliers: {len(list_triggers)}")
        if len(list_triggers) > 0:
            rec = si.remove_artifacts(
                recording=rec0, list_triggers=list_triggers_samp,
                mode='zeros', ms_before=150, ms_after=150,
            )
        else:
            rec = rec0
    else:
        rec = si.load(data_folder)
        fs = rec.get_sampling_frequency()
    print(rec)

    if time_start or time_end is not None:
        start_frame = int(time_start * fs) if time_start else 0
        end_frame = (int(time_end * fs) if time_end is not None
                     else rec.get_num_samples())
        print(f"Cleaving data {time_start}s..{time_end}s "
              f"({start_frame}-{end_frame})")
        rec = rec.frame_slice(start_frame=start_frame, end_frame=end_frame)

    # Write spikeinterface_recording.json (or .pickle fallback), matching
    # BaseSorter.initialize_folder — lets si.read_sorter_result() / other SI
    # tooling that expects this file work on ks4_checkpoint.py output too.
    rec_json_file = output_folder / "spikeinterface_recording.json"
    if rec.check_serializability("json"):
        rec.dump(rec_json_file)
    elif rec.check_serializability("pickle"):
        rec.dump(output_folder / "spikeinterface_recording.pickle",
                  relative_to=output_folder)
    else:
        print("[checkpoint] warning: recording not json/pickle serializable, "
              "skipping spikeinterface_recording.json")

    # ------------------------------------------------------------------ #
    # 2. Probe + binary handling, matching SI's kilosort4 wrapper.
    # ------------------------------------------------------------------ #
    probe_json = sorter_output / "chanMap.json"
    # Always (re)write the probe JSON; cheap, and ensures it matches the rec.
    # Mirrors spikeinterface.sorters.external.kilosort4._setup_json_probe_map.
    from kilosort.io import save_probe
    groups = rec.get_channel_groups()
    positions = np.array(rec.get_channel_locations())
    if positions.shape[1] != 2:
        raise RuntimeError("3D 'location' not supported; set 2D channel locations.")
    n_chan = rec.get_num_channels()
    unique_groups = set(groups)
    group_map = {group: idx for idx, group in enumerate(unique_groups)}
    kcoords = np.array([group_map[g] for g in groups], dtype=int)
    probe_dict = {
        "chanMap": np.arange(n_chan),
        "xc": positions[:, 0],
        "yc": positions[:, 1],
        "kcoords": kcoords,
        "n_chan": n_chan,
    }
    save_probe(probe_dict, str(probe_json))
    probe = ks_io.load_probe(str(probe_json))

    # Pick binary file path (SI convention). For SpikeGLX, the file is binary
    # compatible, so we point kilosort at it directly.
    if rec.binary_compatible_with(time_axis=0, file_paths_length=1):
        bin_desc = rec.get_binary_description()
        filename = str(bin_desc["file_paths"][0])
        file_object = None
    else:
        # Write a local copy once (cached across restarts).
        from spikeinterface.core import write_binary_recording
        binary_file_path = sorter_output / "recording.dat"
        if not binary_file_path.exists():
            print("[checkpoint] writing binary copy of recording...")
            write_binary_recording(
                rec, file_paths=[binary_file_path],
                n_jobs=current_ks_params.get('n_jobs', 1),
                chunk_duration=current_ks_params.get('chunk_duration', '1s'),
            )
        filename = str(binary_file_path)
        file_object = None

    # ------------------------------------------------------------------ #
    # 3. Build settings & ops, set up logger.
    # ------------------------------------------------------------------ #
    setup_logger(sorter_output, verbose_console=False)

    torch_device = current_ks_params.get('torch_device', 'auto')
    if torch_device == 'auto' or torch_device is None:
        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(torch_device)

    do_CAR = current_ks_params.get('do_CAR', True)
    invert_sign = current_ks_params.get('invert_sign', False)
    save_extra_vars = current_ks_params.get('save_extra_vars', False)
    save_preprocessed_copy = current_ks_params.get('save_preprocessed_copy', False)
    clear_cache = current_ks_params.get('clear_cache', False)
    bad_channels = current_ks_params.get('bad_channels', None)
    do_correction = current_ks_params.get('do_correction', True)
    skip_kilosort_preprocessing = current_ks_params.get(
        'skip_kilosort_preprocessing', False
    )

    settings_ks = {k: v for k, v in current_ks_params.items()
                   if k in DEFAULT_SETTINGS}
    settings_ks["n_chan_bin"] = rec.get_num_channels()
    settings_ks["fs"] = rec.get_sampling_frequency()
    settings = {**DEFAULT_SETTINGS, **settings_ks}
    if settings["nt0min"] is None:
        settings["nt0min"] = int(20 * settings["nt"] / 61)
    if settings["artifact_threshold"] is None:
        settings["artifact_threshold"] = np.inf
    # KS 4.1.1 clustering_qr.run reads cluster_init_seed from ops['settings'],
    # but it was never added to DEFAULT_SETTINGS or RECOGNIZED_SETTINGS, so
    # initialize_ops() will reject it if present in 'settings'.
    # We keep it in a side-dict and inject it into ops['settings'] via fixup_ops().
    extra_ops_settings = {}
    extra_ops_settings["cluster_init_seed"] = settings.pop("cluster_init_seed", 42)

    filename_, data_dir_, results_dir_, probe = set_files(
        settings, filename=filename, probe=probe, probe_name="",
        data_dir="", results_dir=sorter_output, bad_channels=bad_channels,
        shank_idx=None,
    )

    data_dtype = rec.get_dtype()

    # ------------------------------------------------------------------ #
    # 4. Stage execution with checkpoints.
    # ------------------------------------------------------------------ #
    params_file = output_folder / "kilosort_params.txt"
    with open(params_file, "w") as f:
        f.write(f"Kilosort parameters saved on {datetime.now()}\n")
        f.write(f"SpikeInterface version: {si.__version__}\n\n")
        f.write("Parameters:\n")
        for key, value in current_ks_params.items():
            f.write(f"{key}: {value}\n")

    # spikeinterface_params.json, matching BaseSorter._dump_params.
    _SI_BOOKKEEPING_CTX["sorter_params"] = current_ks_params
    all_params = {"sorter_name": "kilosort4", "sorter_params": current_ks_params}
    with (output_folder / "spikeinterface_params.json").open(
            mode="w", encoding="utf8") as f:
        json.dump(check_json(all_params), f, indent=4)

    tic0 = time.time()

    def check_interrupt_between_stages(next_stage):
        if INTERRUPT["flag"]:
            print(f"[checkpoint] interrupt received before stage '{next_stage}'; "
                  f"exiting for SLURM requeue.")
            sys.exit(REQUEUE_EXIT_CODE)

    def fixup_ops(ops):
        """Fix ops fields that get corrupted by kilosort's save_ops / load_ops
        round-trip.  Call after every ks_io.load_ops().
        - ops['filename']: saved as str([PosixPath(...)]) by save_ops.
        - ops['settings']: may be missing keys added to clustering_qr after
          the checkpoint was written (e.g. cluster_init_seed in KS 4.1.1).
        """
        ops['filename'] = filename
        ops['settings']['filename'] = [Path(filename)]
        for k, v in settings.items():
            ops['settings'].setdefault(k, v)
        # inject extra keys that initialize_ops rejects but clustering_qr needs
        for k, v in extra_ops_settings.items():
            ops['settings'].setdefault(k, v)
        return ops

    # ---------- Stage 1: preprocessing (whiten + hp filter) ------------ #
    if progress.is_done("preprocessing"):
        ops = fixup_ops(ks_io.load_ops(ckpt_dir / "ops_after_preprocessing.npy",
                                        device=device))
        print("[checkpoint] skipped: preprocessing")
    else:
        check_interrupt_between_stages("preprocessing")
        ops_tuple = initialize_ops(
            settings=settings, probe=probe, data_dtype=data_dtype,
            do_CAR=do_CAR, invert_sign=invert_sign, device=device,
            save_preprocessed_copy=save_preprocessed_copy,
        )
        ops = ops_tuple[0] if isinstance(ops_tuple, tuple) else ops_tuple
        if not skip_kilosort_preprocessing:
            ops = compute_preprocessing(ops=ops, device=device, tic0=tic0,
                                        file_object=file_object)
        else:
            # Match SI behaviour for skip_kilosort_preprocessing.
            from kilosort.run_kilosort import get_run_parameters
            from kilosort.io import BinaryFiltered
            (n_chan_bin, fs_, NT, nt, twav_min, chan_map, dtype, do_CAR_,
             invert_, _, _, tmin, tmax, artifact, *_) = get_run_parameters(ops)
            bfile_skip = BinaryFiltered(
                filename=ops["filename"], n_chan_bin=n_chan_bin, fs=fs_,
                NT=NT, nt=nt, nt0min=twav_min, chan_map=chan_map,
                hp_filter=None, device=device, do_CAR=do_CAR_,
                invert_sign=invert_, dtype=dtype, tmin=tmin, tmax=tmax,
                artifact_threshold=artifact, file_object=file_object,
            )
            ops["preprocessing"] = dict(hp_filter=None, whiten_mat=None)
            ops["Wrot"] = torch.as_tensor(np.eye(rec.get_num_channels()))
            ops["Nbatches"] = bfile_skip.n_batches
        ks_io.save_ops(ops, ckpt_dir)  # writes ops.npy
        os.replace(ckpt_dir / "ops.npy", ckpt_dir / "ops_after_preprocessing.npy")
        progress.mark("preprocessing")
        print("[checkpoint] done: preprocessing")

    # ---------- Stage 2: drift correction ------------------------------ #
    if progress.is_done("drift"):
        ops = fixup_ops(ks_io.load_ops(ckpt_dir / "ops_after_drift.npy", device=device))
        bfile = ks_io.bfile_from_ops(ops=ops, filename=filename, device=device)
        print("[checkpoint] skipped: drift")
    else:
        check_interrupt_between_stages("drift")
        np.random.seed(1)
        torch.cuda.manual_seed_all(1)
        torch.random.manual_seed(1)
        if not do_correction:
            ops["nblocks"] = 0
        ops, bfile, st0 = compute_drift_correction(
            ops=ops, device=device, tic0=tic0, progress_bar=None,
            file_object=file_object, clear_cache=clear_cache,
        )
        ops = fixup_ops(ops)  # inject cluster_init_seed etc. after drift modifies ops
        if save_preprocessed_copy:
            ks_io.save_preprocessing(sorter_output / "temp_wh.dat", ops, bfile)

        # Diagnostic plots — matches SI 0.103.1's _run_from_folder.
        if ops.get("dshift") is not None and st0 is not None:
            try:
                kplots.plot_drift_amount(ops, sorter_output)
                kplots.plot_drift_scatter(st0, sorter_output)
            except Exception as e:
                # Plotting failures must not abort the sort.
                print(f"[checkpoint] drift plot failed (non-fatal): {e}")

        ks_io.save_ops(ops, ckpt_dir)
        os.replace(ckpt_dir / "ops.npy", ckpt_dir / "ops_after_drift.npy")
        if st0 is not None:
            atomic_save_npy(ckpt_dir / "st0.npy", st0)
        progress.mark("drift")
        print("[checkpoint] done: drift")

    # ---------- Stage 3: detect (3 explicit sub-stages) ----------------- #
    # Breaking detect_spikes() apart lets us save st0 *before* first
    # clustering so that intra-clustering checkpoints can resume correctly
    # on SLURM restart (same st0 → same nsp → checkpoint nsp always matches).
    #   3a. detect_univ  – spikedetect.run() → st0, tF0
    #   3b. detect_clu0  – clustering_qr.run(mode='spikes') → Wall3
    #   3c. detect       – template_matching.extract() → st, tF
    ckpt_clu = ClusteringCheckpoint(ckpt_dir, INTERRUPT)
    _orig_clu_run = clustering_qr.run

    def _patched_clu_run(*a, **kw):
        # Only Stage 4 (cluster_spikes) reaches here via monkey-patch;
        # Stage 3b calls clustering_qr.run directly with checkpoint= set.
        kw.setdefault("checkpoint", ckpt_clu)
        return _orig_clu_run(*a, **kw)

    clustering_qr.run = _patched_clu_run

    try:
        if progress.is_done("detect"):
            ops = fixup_ops(ks_io.load_ops(ckpt_dir / "ops_after_detect.npy", device=device))
            bfile = ks_io.bfile_from_ops(ops=ops, filename=filename, device=device)
            st = np.load(ckpt_dir / "st_after_detect.npy")
            tF = torch.load(ckpt_dir / "tF_after_detect.pt",
                            map_location='cpu', weights_only=False)
            print("[checkpoint] skipped: detect")
        else:
            # --- 3a: universal spike detection --- #
            if progress.is_done("detect_univ"):
                st0 = np.load(ckpt_dir / "st0_univ.npy")
                tF = torch.load(ckpt_dir / "tF_univ.pt",
                                map_location='cpu', weights_only=False)
                ops = fixup_ops(ks_io.load_ops(
                    ckpt_dir / "ops_after_detect_univ.npy", device=device))
                bfile = ks_io.bfile_from_ops(ops=ops, filename=filename,
                                             device=device)
                print("[checkpoint] skipped: detect_univ")
            else:
                check_interrupt_between_stages("detect_univ")
                st0, tF_np, ops = spikedetect.run(
                    ops, bfile, device=device, progress_bar=None,
                    clear_cache=clear_cache,
                )
                ops = fixup_ops(ops)  # inject cluster_init_seed etc. into ops['settings']
                tF = torch.from_numpy(tF_np)
                if len(st0) == 0:
                    raise ValueError('No spikes detected, cannot continue sorting.')
                ks_io.save_ops(ops, ckpt_dir)
                os.replace(ckpt_dir / "ops.npy",
                           ckpt_dir / "ops_after_detect_univ.npy")
                atomic_save_npy(ckpt_dir / "st0_univ.npy", st0)
                atomic_save_torch(ckpt_dir / "tF_univ.pt", tF)
                progress.mark("detect_univ")
                print(f"[checkpoint] done: detect_univ ({len(st0)} spikes)")

            # --- 3b: first clustering --- #
            if progress.is_done("detect_clu0"):
                Wall3 = torch.load(ckpt_dir / "Wall3_after_clu0.pt",
                                   map_location='cpu', weights_only=False).to(device)
                print("[checkpoint] skipped: detect_clu0 (first clustering)")
            else:
                check_interrupt_between_stages("detect_clu0")
                try:
                    clu0, Wall0 = clustering_qr.run(
                        ops, st0, tF, mode='spikes', device=device,
                        progress_bar=None, clear_cache=clear_cache,
                        checkpoint=ckpt_clu,
                    )
                except clustering_qr._CheckpointInterrupt as e:
                    print(f"[checkpoint] {e}")
                    progress.state["last_updated"] = datetime.now().isoformat()
                    atomic_save_json(progress.path, progress.state)
                    print("[checkpoint] exiting for SLURM requeue.")
                    sys.exit(REQUEUE_EXIT_CODE)

                Wall3 = template_matching.postprocess_templates(
                    Wall0, ops, clu0, st0, tF, device=device)
                ckpt_clu.mark_complete('spikes')

                # Diagnostic plot.
                try:
                    kplots.plot_diagnostics(Wall0, clu0, ops, sorter_output)
                except Exception as e:
                    print(f"[checkpoint] diagnostics plot failed (non-fatal): {e}")

                atomic_save_torch(ckpt_dir / "Wall3_after_clu0.pt", Wall3)
                atomic_save_npy(ckpt_dir / "clu_after_clu0.npy", clu0)
                atomic_save_torch(ckpt_dir / "Wall_after_clu0.pt", Wall0)
                progress.mark("detect_clu0")
                print(f"[checkpoint] done: detect_clu0 ({clu0.max()+1} clusters)")

            # --- 3c: template-based spike re-extraction --- #
            # Free tF_univ and st0 — no longer needed; keeps 90 GB off the heap
            # during extract (tF_univ alone is ~88 GB).
            import gc as _gc
            try:
                _n_univ = len(st0)
                del tF, st0
            except NameError:
                # st0/tF not in scope: detect_clu0 was already done on a
                # prior run and we skipped loading them.  Read spike count
                # from the saved .npy without materialising the 88 GB tensor.
                _n_univ = int(
                    np.load(ckpt_dir / "st0_univ.npy", mmap_mode='r').shape[0])
            _gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Set pre-allocation size: 2.5 × universal detection spike count.
            # Template matching with per-template thresholds finds considerably
            # MORE spikes than universal detection (~2× empirically for this
            # recording). Use 2.5× as a safe margin to avoid file extensions.
            # At ~43 M universal spikes: 2.5 × 43 M × 16 × 32 × 4 B = 222 GB on
            # disk (file-backed, not anonymous RAM) — fine for /ptmp (1.5 PB free).
            if _n_univ > 0:
                _extract_n_pre_holder[0] = int(_n_univ * 2.5)
            _extract_ckpt_dir_holder[0] = ckpt_dir
            check_interrupt_between_stages("detect")
            st, tF, ops = template_matching.extract(
                ops, bfile, Wall3, device=device, progress_bar=None)
            ks_io.save_ops(ops, ckpt_dir)
            os.replace(ckpt_dir / "ops.npy",
                       ckpt_dir / "ops_after_detect.npy")
            atomic_save_npy(ckpt_dir / "st_after_detect.npy", st)
            atomic_save_torch(ckpt_dir / "tF_after_detect.pt", tF)
            progress.mark("detect")
            print(f"[checkpoint] done: detect ({len(st)} spikes extracted)")

        # ---------- Stage 4: final clustering + merging ---------------- #
        if progress.is_done("cluster"):
            ops = fixup_ops(ks_io.load_ops(ckpt_dir / "ops_after_cluster.npy",
                                            device=device))
            bfile = ks_io.bfile_from_ops(ops=ops, filename=filename,
                                          device=device)
            st = np.load(ckpt_dir / "st_after_cluster.npy")
            tF = torch.load(ckpt_dir / "tF_after_cluster.pt",
                            map_location='cpu', weights_only=False)
            clu = np.load(ckpt_dir / "clu_after_cluster.npy")
            Wall = torch.load(ckpt_dir / "Wall_after_cluster.pt",
                              map_location='cpu', weights_only=False)
            print("[checkpoint] skipped: cluster")
        else:
            check_interrupt_between_stages("cluster")
            try:
                result = cluster_spikes(
                    st=st, tF=tF, ops=ops, device=device, bfile=bfile,
                    tic0=tic0, progress_bar=None, clear_cache=clear_cache,
                )
            except clustering_qr._CheckpointInterrupt as e:
                print(f"[checkpoint] {e}")
                progress.state["last_updated"] = datetime.now().isoformat()
                atomic_save_json(progress.path, progress.state)
                print("[checkpoint] exiting 0 for SLURM requeue.")
                sys.exit(REQUEUE_EXIT_CODE)

            if len(result) == 4:
                clu, Wall, st, tF = result
            else:
                clu, Wall = result
            ckpt_clu.mark_complete('template')

            ks_io.save_ops(ops, ckpt_dir)
            os.replace(ckpt_dir / "ops.npy",
                       ckpt_dir / "ops_after_cluster.npy")
            atomic_save_npy(ckpt_dir / "st_after_cluster.npy", st)
            atomic_save_torch(ckpt_dir / "tF_after_cluster.pt", tF)
            atomic_save_npy(ckpt_dir / "clu_after_cluster.npy", clu)
            atomic_save_torch(ckpt_dir / "Wall_after_cluster.pt", Wall)
            progress.mark("cluster")
            print("[checkpoint] done: cluster")

        # ---------- Stage 5: save to phy ------------------------------- #
        if progress.is_done("save"):
            print("[checkpoint] skipped: save")
        else:
            check_interrupt_between_stages("save")
            if skip_kilosort_preprocessing:
                ops["preprocessing"] = dict(
                    hp_filter=torch.as_tensor(np.zeros(1)),
                    whiten_mat=torch.as_tensor(np.eye(rec.get_num_channels())),
                )
            save_sorting(
                ops=ops, results_dir=sorter_output, st=st, clu=clu, tF=tF,
                Wall=Wall, imin=bfile.imin, tic0=tic0,
                save_extra_vars=save_extra_vars,
                save_preprocessed_copy=save_preprocessed_copy,
            )
            progress.mark("save")
            print("[checkpoint] done: save")
    finally:
        clustering_qr.run = _orig_clu_run

    runtime = time.time() - tic0
    print(f"[checkpoint] this segment took {runtime:.0f}s "
          f"({runtime/3600:.2f}h) on SLURM restart "
          f"{progress.state['slurm_restart_count']}")
    print(f"[checkpoint] sorter output: {sorter_output}")

    # spikeinterface_log.json, matching BaseSorter.run_from_folder's success
    # path (only reached here if no stage exited early for SLURM requeue —
    # see the module-level error handler below for the failure path).
    _write_si_log_json(
        output_folder=output_folder, sorter_output=sorter_output,
        sorter_version=_importlib_version("kilosort"), error=False,
        error_trace=None, run_time=runtime,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Genuine failure (not the SIGUSR1/checkpoint SystemExit(75) requeue
        # path, which is a BaseException and so isn't caught here). Write
        # spikeinterface_log.json with error=True before re-raising, matching
        # BaseSorter.run_from_folder's failure path, so SI tooling that reads
        # that file to check for errors sees a real (not missing) log.
        _output_folder = _SI_BOOKKEEPING_CTX.get("output_folder")
        _sorter_output = _SI_BOOKKEEPING_CTX.get("sorter_output")
        if _output_folder is not None and _sorter_output is not None:
            _error_trace_lines = traceback.format_exc().strip().split("\n")
            _error_to_json = ["Traceback (most recent call last):"] + [
                f"  {l}" if not l.startswith(" ") else l
                for l in _error_trace_lines[1:]
            ]
            try:
                _sv = _importlib_version("kilosort")
            except Exception:
                _sv = "unknown"
            _write_si_log_json(
                output_folder=_output_folder, sorter_output=_sorter_output,
                sorter_version=_sv, error=True,
                error_trace=_error_to_json, run_time=None,
            )
        raise
