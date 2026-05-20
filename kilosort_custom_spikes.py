#!/usr/bin/env python3
"""
kilosort_custom_spikes.py
=========================
Feed YOUR pre-detected spikes into Kilosort 4's clustering engine.

What this script does
---------------------
1.  Loads a saved KS4 ops.npy (from a previous KS4 run on the same data).
    That gives us the whitening matrix, high-pass filter, PCA basis (wPCA),
    learned template positions (iU / iCC) — all for free, with no re-training.
2.  Opens the raw/preprocessed binary through KS4's BinaryFiltered pipeline
    (same hp-filter + whitening + drift-correction as the original run).
3.  For each of YOUR detected spikes:
       • finds the nearest KS4 learned template
       • extracts the filtered/whitened waveform snippet
       • projects it through wPCA → produces the tF feature tensor
4.  Runs KS4's graph-based clustering (clustering_qr) on those features.
5.  Saves results in Phy format.

Run with the phy2 conda environment:
    conda run -n phy2 python /u/gongt/scripts/kilosort_custom_spikes.py

=======================================================================
CONFIGURE EVERYTHING IN THE "USER CONFIGURATION" SECTION BELOW.
=======================================================================
"""

import sys
import time
import warnings
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ── point at phy2 site-packages if running outside conda ───────────────────
PHY2_SITE = "/u/gongt/conda-envs/phy2/lib/python3.11/site-packages"
if PHY2_SITE not in sys.path:
    sys.path.insert(0, PHY2_SITE)

from kilosort import io
from kilosort.run_kilosort import (
    cluster_spikes, save_sorting, setup_logger, close_logger,
)


# =====================================================================
#  USER CONFIGURATION
# =====================================================================

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

# ops.npy from a previous KS4 run on this data.
# This gives us the PCA basis, whitening matrix, template positions etc.
OPS_NPY_PATH = (
    "/nexus/posix0/MBR-neuralsystems/ephys/GLC6962/GLC6962_g_g0"
    "/KS4/nc64_wr64_thu9_thl8_short/sorter_output/ops.npy"
)

# Raw / SI-preprocessed binary for the recording.
# KS4 reads this via its own hp-filter + whitening pipeline.
# For a SpikeInterface BinaryFolderRecording the file is traces_cached_seg0.raw
BINARY_FILE = (
    "/nexus/posix0/MBR-neuralsystems/ephys/GLC6962/GLC6962_g_g0"
    "/preprocessed_slice_550_1270s/traces_cached_seg0.raw"
)

# Your detected peaks (structured numpy array with at least
# 'sample_index' and 'channel_index' fields, as produced by SI).
PEAKS_PATH = (
    "/nexus/posix0/MBR-neuralsystems/ephys/GLC6962/GLC6962_g_g0"
    "/preprocessed_slice_550_1270s/thresholding_peaks/peaks_combined2.npy"
)

# Where to save the Phy-compatible output.
RESULTS_DIR = (
    "/nexus/posix0/MBR-neuralsystems/ephys/GLC6962/GLC6962_g_g0"
    "/kilosort4_custom_spikes"
)

# ------------------------------------------------------------------
# Runtime options
# ------------------------------------------------------------------

# PyTorch device. Auto-selects GPU → CPU.
DEVICE = None   # None = auto   or   torch.device('cuda')   / torch.device('cpu')

# Number of spikes to process at a time when finding nearest templates
# (only affects RAM usage, not results).
TEMPLATE_ASSIGN_CHUNK = 100_000

# Set True to also save tF / Wall / full_st / full_clu files.
SAVE_EXTRA_VARS = False

# =====================================================================
#  END OF USER CONFIGURATION
# =====================================================================


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device(device=None):
    if device is not None:
        return device
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _tensor(x, device):
    """Convert numpy array to torch tensor on device."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.from_numpy(np.asarray(x)).to(device)


def assign_templates(peaks, ops):
    """
    Assign each spike to the nearest *learned* KS4 template.

    Returns
    -------
    template_idx : np.ndarray, shape (n_spikes,), dtype int32
        Index into ops['iU'] (0-based) for each spike.
    """
    iU = ops["iU"]
    if isinstance(iU, torch.Tensor):
        iU = iU.cpu().numpy()

    xc = np.asarray(ops["xc"], dtype=np.float32)
    yc = np.asarray(ops["yc"], dtype=np.float32)

    # Learned template x/y positions (one per entry in iU)
    xt = xc[iU]   # (n_templates,)
    yt = yc[iU]   # (n_templates,)

    spike_chans  = peaks["channel_index"].astype(int)
    x_spk = xc[spike_chans]   # (n_spikes,)
    y_spk = yc[spike_chans]

    n_spikes    = len(peaks)
    n_templates = len(iU)
    template_idx = np.empty(n_spikes, dtype=np.int32)

    # Chunked distance computation (avoids allocating an (n_spikes, n_templates) matrix)
    chunk = TEMPLATE_ASSIGN_CHUNK
    for i in range(0, n_spikes, chunk):
        xc_chunk = x_spk[i : i + chunk, np.newaxis]   # (C, 1)
        yc_chunk = y_spk[i : i + chunk, np.newaxis]
        dists    = (xc_chunk - xt) ** 2 + (yc_chunk - yt) ** 2  # (C, T)
        template_idx[i : i + chunk] = dists.argmin(axis=1)

    return template_idx


def extract_features(bfile, ops, peaks, template_idx, device):
    """
    Extract KS4 PCA features at user spike times.

    The algorithm mirrors spikedetect.run():
        xsub  = whitened_batch[chan_idx, local_time + tarange]   (nC, m, nt)
        xfeat = xsub @ wPCA.T                                    (nC, m, n_pcs)
        tF    = xfeat.permute(1, 0, 2)                           (m, nC, n_pcs)

    Parameters
    ----------
    bfile : BinaryFiltered
    ops : dict
    peaks : structured np.ndarray  (sample_index, channel_index, …)
    template_idx : np.ndarray (n_spikes,)  — as returned by assign_templates()
    device : torch.device

    Returns
    -------
    st_valid : np.ndarray (n_valid, 3)
        [ sample_relative_to_bfile_imin, template_idx, approx_amplitude ]
    tF_valid : torch.Tensor (n_valid, nC, n_pcs)  on CPU
    valid_mask : np.ndarray (n_spikes,) bool
    """
    iU  = ops["iU"]
    if isinstance(iU, torch.Tensor):
        iU = iU.cpu().numpy()

    iCC = ops["iCC"]
    if isinstance(iCC, torch.Tensor):
        iCC = iCC.cpu().numpy()

    wPCA = ops["wPCA"].to(device)          # (n_pcs, nt)
    wPCA = wPCA.float()

    nC    = int(ops["settings"]["nearest_chans"])
    n_pcs = int(ops["settings"]["n_pcs"])
    nt    = int(ops["nt"])
    NT    = int(ops["batch_size"])
    imin  = int(bfile.imin)

    tarange = torch.arange(-(nt // 2), nt // 2 + 1, device=device, dtype=torch.long)

    spike_samples = peaks["sample_index"].astype(np.int64)
    n_spikes      = len(peaks)

    # Relative sample index and which batch each spike lives in
    rel_samples   = spike_samples - imin           # relative to bfile start
    batch_of      = rel_samples // NT              # which batch
    # padded offset: batches are padded by +nt on each side
    # local position within that padded batch:
    #   batch 0 starts at imin (no leading padding), other batches start at imin-nt
    # BinaryRWFile.get_batch_edges handles this; we just need the sample offset
    # within the padded batch array that padded_batch_to_torch returns.
    # For batch > 0: bstart = imin + ibatch*NT - nt  → local = rel_samples - ibatch*NT + nt
    # For batch 0:   bstart = imin                   → local = rel_samples + nt  (nt padding added at front)
    local_samples = rel_samples - batch_of * NT + nt  # same formula works for batch 0 too

    # ── Validity mask ─────────────────────────────────────────────────────
    valid_mask = (
        (batch_of >= 0)
        & (batch_of < bfile.n_batches)
        & (local_samples >= nt)           # not in leading padding
        & (local_samples < NT + nt)       # not in trailing padding
    )
    n_invalid = int((~valid_mask).sum())
    if n_invalid:
        print(f"  [extract] dropping {n_invalid} edge spikes out of valid range")

    # ── Pre-channel lookup: iCC[:, iU[t]] for each template t ────────────
    # shape (nC, n_templates)  — cached so we don't repeat the indexing
    n_templates = len(iU)
    chan_by_template = iCC[:nC, iU]   # (nC, n_templates)

    # ── Allocate output ───────────────────────────────────────────────────
    tF_np = np.zeros((n_spikes, nC, n_pcs), dtype=np.float32)
    st    = np.zeros((n_spikes, 3),          dtype=np.float64)
    st[:, 0] = rel_samples          # time (relative to bfile.imin)
    st[:, 1] = template_idx         # learned template index

    # ── Batch loop ────────────────────────────────────────────────────────
    batches_used = np.unique(batch_of[valid_mask])
    print(f"  [extract] {n_spikes:,} spikes across {len(batches_used)} batches …")

    for ibatch in tqdm(batches_used, desc="Extracting features"):
        mask = (batch_of == ibatch) & valid_mask
        idx  = np.where(mask)[0]           # indices into full peaks array
        m    = len(idx)

        # Load whitened, hp-filtered, drift-corrected batch
        X = bfile.padded_batch_to_torch(int(ibatch), ops=ops)  # (n_chan, NT+2nt)
        X = X.float()

        t_arr = template_idx[idx]          # (m,) template indices
        l_arr = local_samples[idx]         # (m,) local sample positions

        # Channel indices for each spike: (nC, m)
        chan_idx = torch.from_numpy(chan_by_template[:, t_arr]).to(device).long()

        # Time windows for each spike: (m, nt)
        l_t      = torch.from_numpy(l_arr).to(device).long()
        time_idx = l_t[:, None] + tarange[None, :]   # (m, nt)

        # Vectorised extraction:
        #   chan_idx : (nC, m) → (nC, m, 1) broadcast with time (1, m, nt) → (nC, m, nt)
        xsub  = X[chan_idx.unsqueeze(-1).expand(-1, -1, nt),
                   time_idx.unsqueeze(0).expand(nC, -1, -1)]   # (nC, m, nt)

        # Project: (nC, m, nt) @ (nt, n_pcs) → (nC, m, n_pcs)
        xfeat = xsub @ wPCA.T

        tF_np[idx] = xfeat.permute(1, 0, 2).cpu().numpy()         # (m, nC, n_pcs)

    # Approximate amplitude (L2 norm of feature vector per spike)
    amplitudes   = np.linalg.norm(tF_np.reshape(n_spikes, -1), axis=1)
    st[:, 2]     = amplitudes

    tF_tensor    = torch.from_numpy(tF_np[valid_mask])   # CPU, float32
    st_valid     = st[valid_mask]

    return st_valid, tF_tensor, valid_mask


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    tic0 = time.time()
    device = _get_device(DEVICE)
    print(f"[device] {device}")

    # 1. Load ops ----------------------------------------------------------
    print(f"\n[ops] loading {OPS_NPY_PATH}")
    ops = io.load_ops(OPS_NPY_PATH, device=device)
    print(f"  wPCA shape   : {ops['wPCA'].shape}")
    iU_np = ops["iU"].cpu().numpy() if isinstance(ops["iU"], torch.Tensor) else ops["iU"]
    print(f"  n templates  : {len(iU_np)}")
    print(f"  nearest_chans: {ops['settings']['nearest_chans']}")
    print(f"  n_pcs        : {ops['settings']['n_pcs']}")

    # 2. Open binary -------------------------------------------------------
    print(f"\n[bfile] opening {BINARY_FILE}")
    bfile = io.bfile_from_ops(ops, filename=BINARY_FILE, device=device)
    print(f"  n_samples: {bfile.n_samples:,}   n_batches: {bfile.n_batches}")
    print(f"  imin={bfile.imin}  dtype={ops['data_dtype']}")

    # 3. Load & prepare spikes --------------------------------------------
    print(f"\n[peaks] loading {PEAKS_PATH}")
    peaks = np.load(PEAKS_PATH)
    peaks = np.sort(peaks, order="sample_index")
    print(f"  {len(peaks):,} peaks loaded")

    # 4. Assign each spike to nearest learned template --------------------
    print("\n[assign] finding nearest KS4 template for each spike …")
    template_idx = assign_templates(peaks, ops)
    uniq, counts = np.unique(template_idx, return_counts=True)
    print(f"  {len(uniq)} distinct templates used  "
          f"(avg {counts.mean():.0f} spikes/template)")

    # 5. Extract PCA features at user spike times -------------------------
    print("\n[features] extracting waveform features …")
    st, tF, valid_mask = extract_features(bfile, ops, peaks, template_idx, device)
    print(f"  valid spikes : {len(st):,}  "
          f"(dropped {(~valid_mask).sum():,} edge spikes)")

    # st[:,0] must be int for save_to_phy (it adds bfile.imin to get absolute time)
    # keep as float64 for now — the downstream functions cast internally

    # 6. Cluster ----------------------------------------------------------
    print(f"\n[cluster] running KS4 graph clustering …")
    clu, Wall, st, tF = cluster_spikes(
        st, tF, ops, device, bfile,
        tic0=tic0,
        verbose=False,
    )
    n_units = int(clu.max()) + 1
    print(f"  {n_units} clusters found")

    # 7. Save to Phy format -----------------------------------------------
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(results_dir)
    print(f"\n[save] writing Phy output to {results_dir} …")

    ops, similar_templates, is_ref, est_contam_rate, kept_spikes = save_sorting(
        ops,
        results_dir,
        st,
        clu,
        tF,
        Wall,
        bfile.imin,         # absolute sample offset
        tic0=tic0,
        save_extra_vars=SAVE_EXTRA_VARS,
    )

    n_good = int(is_ref.sum())
    runtime = time.time() - tic0
    print(f"\n✓  Done in {runtime/60:.1f} min")
    print(f"   {n_units} units  ({n_good} refractory / 'good')")
    print(f"\n   Open in Phy:\n"
          f"   phy template-gui {results_dir / 'params.py'}")

    close_logger()


if __name__ == "__main__":
    main()
