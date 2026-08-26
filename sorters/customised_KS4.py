#!/usr/bin/env python3
"""
LEGACY / REFERENCE ONLY — not part of the active pipeline.

This was an attempt to replace Kilosort4's own spike detection with our own
detected spikes (from the peak_detect_localise pipeline / mua_pipeline.py),
feeding those into KS4 only for clustering. The resulting sorting quality
was still poor — this approach may not be worth pursuing further. Kept here
for reference in case someone wants to pick the idea back up.

===================
Kilosort 4 clustering with mua_pipeline spike detection replacing
KS4's built-in template detector.

Assumes input recording is already preprocessed (hp-filtered, CAR applied).
Only performs drift correction before spike detection.

All recording metadata (fs, n_channels, dtype, probe geometry) is
read directly from the recording file — nothing is hard-coded.

Pipeline
--------
 1. Load preprocessed recording with load_data() — probe, fs, dtype, n_channels
    all come from the file. Recording should already be hp-filtered and CAR'd.
 2. Initialize KS4 ops (no additional CAR applied to data via do_CAR=False).
 3. Compute whitening matrix (required for drift correction), then replace with
    identity since we don't want whitening on already-preprocessed data.
 4. KS4 drift correction → drift plots saved to RESULTS_DIR.
 5. Save drift-corrected data as SpikeInterface binary to KS4_corrected folder
    (for mua_pipeline compatibility).
 6. mua_pipeline.run_mua_pipeline() on KS4_corrected SI recording → peaks_combined2.
 7. Translate peak sample indices to absolute KS4 coordinates (if needed).
 8. Learn wPCA from drift-corrected data (spikedetect.run — detections discarded).
 9. Assign each spike to nearest universal template → channel neighbourhood.
10. Extract wPCA features at spike times.
11. KS4 cluster_spikes() (clustering_qr; merging step is disabled in this
    conda env — see ks4_checkpoint.py's ENVIRONMENT NOTE).
12. Save Phy-compatible output to RESULTS_DIR.

=====================================================================
CONFIGURE EVERYTHING IN THE "USER CONFIGURATION" SECTION BELOW
=====================================================================
"""

import sys
import time
import platform
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── scripts directory (where mua_pipeline.py lives) ───────────────────────
SCRIPTS_DIR = "/u/gongt/scripts"
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import kilosort
from kilosort import io, datashift, spikedetect
from kilosort.parameters import DEFAULT_SETTINGS
import kilosort.plots as kplots
from kilosort.run_kilosort import (
    initialize_ops, compute_preprocessing, compute_drift_correction,
    cluster_spikes, save_sorting, setup_logger, close_logger,
)

from scripts.peak_detect_localise.mua_pipeline import run_mua_pipeline, load_data


# =====================================================================
#  USER CONFIGURATION
# =====================================================================

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

# Path passed to load_data() — SpikeGLX dir (contains 'imec') or
# a SpikeInterface-saved folder.
RECORDING_PATH = (
    "/nexus/posix0/MBR-neuralsystems/ephys/GLC6962/GLC6962_g_g0/preprocessed_slice_550_1270s"
)

# All output: Phy files, drift plots, log, temp_wh.dat
RESULTS_DIR = (
    "/nexus/posix0/MBR-neuralsystems/ephys/GLC6962/GLC6962_g_g0/preprocessed_slice_550_1270s/kilosort4_detected_spikes_integrated"
)

# ------------------------------------------------------------------
# KS4 settings
# Start with kilosort's DEFAULT_SETTINGS and override only what you want to customize.
# fs, n_chan_bin, and data_dtype are filled automatically from the recording.
# ------------------------------------------------------------------

# Customizations only (merged with DEFAULT_SETTINGS in main())
KS4_CUSTOM_SETTINGS = {
    # batch / timing
    "nblocks":         5,       # drift blocks (default=1); set to 0 to skip drift correction
    "batch_size":      60000,   # samples per batch

    # preprocessing
    "whitening_range": 32,      # (NOT used — whitening replaced with identity below)

    # templates / PCA
    "n_templates":     6,       # number of universal templates (default=6)
    "n_pcs":           6,       # number of PCs per channel (default=6)
    "nearest_chans":   32,      # channels per template (default=10)
    "nearest_templates": 100,   # templates to search for matches (default=100)
}

# ------------------------------------------------------------------
# mua_pipeline settings
# ------------------------------------------------------------------

MUA_SETTINGS = dict(
    detect_threshold   = 3.3,
    exclude_sweep_ms   = 0.3,
    spatial_radius     = 10,
    chunk_size_s       = 60.0,
    n_jobs             = -1,
    proto_n_jobs       = -1,
    load_computed_data = False,  # True = reload saved peaks if present
)

# ------------------------------------------------------------------
# Runtime
# ------------------------------------------------------------------
DEVICE                = None     # None = auto (GPU → CPU)
TEMPLATE_ASSIGN_CHUNK = 100_000
SAVE_EXTRA_VARS       = False

# =====================================================================
#  END OF USER CONFIGURATION
# =====================================================================


# ---------------------------------------------------------------------------
# Probe extraction from SI recording
# ---------------------------------------------------------------------------

def probe_from_recording(rec):
    """
    Build a KS4 probe dict directly from the SI recording's embedded probe.

    KS4 needs:
        chanMap  : (n_contacts,)  int32   — 0-based index into the binary file
        xc       : (n_contacts,)  float32 — contact x position (µm)
        yc       : (n_contacts,)  float32 — contact y position (µm)
        kcoords  : (n_contacts,)  float32 — shank id (used to group contacts)
        n_chan   : int            — number of contacts

    The SI contact_vector structured array has fields 'x', 'y', 'shank_ids',
    and 'device_channel_indices', which map directly onto the above.
    """
    probe_si = rec.get_probe()
    contact_vector = rec.get_property("contact_vector")

    if contact_vector is None:
        raise ValueError(
            "No probe information found in the recording. "
            "The recording must have a probe attached "
            "(contact_vector property must be present). "
            "For SpikeGLX recordings this is set automatically; "
            "for SI-saved folders a probe.json must be present."
        )

    xc      = contact_vector["x"].astype(np.float32)
    yc      = contact_vector["y"].astype(np.float32)
    chan_map = contact_vector["device_channel_indices"].astype(np.int32)

    # shank_ids are strings in probeinterface ('0', '1', …); convert to float
    shank_ids = contact_vector["shank_ids"]
    try:
        kcoords = shank_ids.astype(np.float32)
    except (ValueError, TypeError):
        # non-numeric shank labels → encode as integer rank
        unique_shanks = {s: i for i, s in enumerate(np.unique(shank_ids))}
        kcoords = np.array([unique_shanks[s] for s in shank_ids], dtype=np.float32)

    probe = {
        "chanMap": chan_map,
        "xc":      xc,
        "yc":      yc,
        "kcoords": kcoords,
        "n_chan":  len(chan_map),
    }

    logger.info(f"  Probe: {len(chan_map)} contacts  "
                f"x=[{xc.min():.0f}, {xc.max():.0f}] µm  "
                f"y=[{yc.min():.0f}, {yc.max():.0f}] µm  "
                f"shanks={np.unique(kcoords)}")

    return probe


def get_merged_settings(rec, custom_settings):
    """
    Merge custom settings with kilosort defaults.

    Starts with DEFAULT_SETTINGS, applies custom overrides, then injects
    recording metadata (fs, n_chan_bin).
    """
    # Start with kilosort defaults, then override with custom settings
    settings = {**DEFAULT_SETTINGS, **custom_settings}

    # Inject recording metadata
    settings["fs"]          = float(rec.get_sampling_frequency())
    settings["n_chan_bin"]  = int(rec.get_num_channels())

    return settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device(device=None):
    if device is not None:
        return device
    if torch.cuda.is_available():
        dev   = torch.device("cuda")
        props = torch.cuda.get_device_properties(dev)
        logger.info(f"GPU: {props.name}  ({props.total_memory/1024**3:.1f} GB)")
        return dev
    logger.info("No GPU found — using CPU")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _assign_templates(peaks, ops):
    """
    Match each spike to the nearest universal template grid position.
    Determines which nC channels to read per spike.
    peaks['sample_index'] must be in absolute KS4 sample coordinates.
    """
    xt = ops["xcup"].astype(np.float32)
    yt = ops["ycup"].astype(np.float32)
    xc = ops["xc"].astype(np.float32)
    yc = ops["yc"].astype(np.float32)

    spike_chans  = peaks["channel_index"].astype(int)
    x_spk        = xc[spike_chans]
    y_spk        = yc[spike_chans]
    n_spikes     = len(peaks)
    template_idx = np.empty(n_spikes, dtype=np.int32)

    for i in range(0, n_spikes, TEMPLATE_ASSIGN_CHUNK):
        sl = slice(i, i + TEMPLATE_ASSIGN_CHUNK)
        dx = x_spk[sl, np.newaxis] - xt
        dy = y_spk[sl, np.newaxis] - yt
        template_idx[sl] = (dx**2 + dy**2).argmin(axis=1)

    return template_idx


def _extract_features(bfile, ops, peaks, template_idx, device):
    """
    Extract wPCA features for each spike.
    peaks['sample_index'] must be absolute sample indices.
    st[:,0] is stored as relative (absolute - imin); save_to_phy adds imin back.
    """
    iC = ops["iC"]
    if not isinstance(iC, torch.Tensor):
        iC = torch.from_numpy(np.asarray(iC)).to(device)
    iC = iC.long()

    wPCA  = ops["wPCA"].to(device).float()
    nC    = int(ops["settings"]["nearest_chans"])
    n_pcs = int(ops["settings"]["n_pcs"])
    nt    = int(ops["nt"])
    NT    = int(ops["batch_size"])
    imin  = int(bfile.imin)

    tarange = torch.arange(-(nt // 2), nt // 2 + 1, device=device, dtype=torch.long)
    assert len(tarange) == nt

    spike_samples = peaks["sample_index"].astype(np.int64)
    n_spikes      = len(peaks)

    rel_samples   = spike_samples - imin
    batch_of      = rel_samples // NT
    local_samples = rel_samples - batch_of * NT + nt

    half = nt // 2
    valid_mask = (
        (batch_of >= 0)
        & (batch_of < bfile.n_batches)
        & (local_samples - half >= 0)
        & (local_samples + half < NT + 2 * nt)
    )
    n_dropped = int((~valid_mask).sum())
    if n_dropped:
        logger.warning(f"Dropping {n_dropped} edge spikes outside valid range")

    tF_np = np.zeros((n_spikes, nC, n_pcs), dtype=np.float32)
    st    = np.zeros((n_spikes, 3),         dtype=np.float64)
    st[:, 0] = rel_samples
    st[:, 1] = template_idx

    batches_used = np.unique(batch_of[valid_mask])
    logger.info(f"  {n_spikes:,} spikes across {len(batches_used)} batches")

    for ibatch in tqdm(batches_used, desc="Extracting features"):
        mask = (batch_of == ibatch) & valid_mask
        idx  = np.where(mask)[0]

        X = bfile.padded_batch_to_torch(int(ibatch), ops=ops).float()

        t_arr    = template_idx[idx]
        l_arr    = local_samples[idx]
        chan_idx  = iC[:nC, t_arr].long()
        l_t       = torch.from_numpy(l_arr).to(device).long()
        time_idx  = l_t[:, None] + tarange[None, :]

        xsub  = X[chan_idx.unsqueeze(-1).expand(-1, -1, nt),
                   time_idx.unsqueeze(0).expand(nC, -1, -1)]
        xfeat = xsub @ wPCA.T
        tF_np[idx] = xfeat.permute(1, 0, 2).cpu().numpy()

    st[:, 2]  = np.linalg.norm(tF_np.reshape(n_spikes, -1), axis=1)
    tF_tensor = torch.from_numpy(tF_np[valid_mask]).float()
    st_valid  = st[valid_mask]

    assert st_valid.dtype == np.float64
    assert tF_tensor.dtype == torch.float32
    assert tF_tensor.shape == (len(st_valid), nC, n_pcs)
    assert (st_valid[:, 0] >= 0).all(), "Negative relative spike times"

    return st_valid, tF_tensor, valid_mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tic0 = time.time()
    device = _get_device(DEVICE)

    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(results_dir)

    logger.info(f"Kilosort {kilosort.__version__}  |  "
                f"Python {platform.python_version()}  |  "
                f"PyTorch {torch.__version__}  |  device {device}")

    print("loading recording...")
    # ── 1. Load recording ─────────────────────────────────────────────────
    logger.info(f"\n[1/10] Loading recording from {RECORDING_PATH} …")
    rec = load_data(RECORDING_PATH)
    logger.info(f"  fs={rec.get_sampling_frequency()} Hz  "
                f"n_channels={rec.get_num_channels()}  "
                f"dtype={rec.get_dtype()}  "
                f"duration={rec.get_num_samples() / rec.get_sampling_frequency():.1f} s")

    print("building ops...")
    # ── 2. Build ops from recording metadata ──────────────────────────────
    logger.info("\n[2/10] Building ops from recording …")
    settings = get_merged_settings(rec, KS4_CUSTOM_SETTINGS)
    probe    = probe_from_recording(rec)

    logger.info(f"  Using {len(settings)} total settings "
                f"({len(KS4_CUSTOM_SETTINGS)} customized, "
                f"{len(DEFAULT_SETTINGS) - len(KS4_CUSTOM_SETTINGS)} from defaults)")

    # KS4 needs the path to the raw binary. For a SpikeGLX recording this
    # is the .bin file; for an SI BinaryFolderRecording it's traces_cached_seg0.raw.
    binary_file = rec.get_binary_description()["file_paths"][0] \
                  if hasattr(rec, "get_binary_description") \
                  else Path(RECORDING_PATH) / "traces_cached_seg0.raw"

    ops, _ = initialize_ops(
        settings, probe,
        do_CAR=False,  # Recording is already preprocessed (CAR applied); skip double-CAR
        invert_sign=False,
        device=device,
        data_dtype=rec.get_dtype(),
        save_preprocessed_copy=False,  # We'll save as SI binary, not KS4 binary
    )
    ops["filename"] = [Path(binary_file)]
    ops["data_dir"] = Path(binary_file).parent

    logger.info(f"  Nchan={ops['Nchan']}  n_chan_bin={ops['n_chan_bin']}  "
                f"fs={ops['fs']}  nt={ops['nt']}  batch_size={ops['batch_size']}")

    print("preprocessing...")
    # ── 3. Preprocessing ──────────────────────────────────────────────────
    logger.info("\n[3/10] Computing hp-filter and whitening matrix …")
    ops = compute_preprocessing(ops, device, tic0=tic0)

    # Replace whitening matrix with identity.
    # datashift needed the real whitening matrix to compute iKxx — that is
    # now done. Going forward every BinaryFiltered applies only
    # hp-filter + CAR + drift correction; no whitening ever touches the data.
    n_probe_chans = ops["Nchan"]
    identity = torch.eye(n_probe_chans, device=device)
    ops["Wrot"]                        = identity
    ops["preprocessing"]["whiten_mat"] = identity
    logger.info("  Whitening matrix replaced with identity (whitening disabled)")

    print("drift correction...")
    # ── 4. Drift correction + drift plots ─────────────────────────────────
    logger.info(f"\n[4/10] Drift correction (nblocks={ops['nblocks']}) …")
    ops, bfile, st0_drift = compute_drift_correction(ops, device, tic0=tic0)

    if st0_drift is not None:
        logger.info("  Saving drift plots …")
        kplots.plot_drift_amount(ops, results_dir)
        kplots.plot_drift_scatter(st0_drift, results_dir)

    logger.info(f"  n_batches={bfile.n_batches}  "
                f"imin={bfile.imin}  imax={bfile.imax}")

    print("saving drift-corrected binary...")
    # ── 5. Save drift-corrected binary as SpikeInterface format ───────────────────
    # Save directly to SI binary format (more portable for mua_pipeline).
    results_dir_corrected = Path(results_dir) / "KS4_corrected"
    results_dir_corrected.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n[5/10] Saving drift-corrected data → {results_dir_corrected} (SI binary)")

    # Write drift-corrected data directly from bfile (no KS4 binary intermediate)
    n_samples = bfile.imax - bfile.imin
    n_channels = ops['n_chan_bin']
    fs = ops['fs']
    
    output_path = results_dir_corrected / "traces_cached_seg0.raw"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"  Writing {n_samples:,} samples × {n_channels} channels …")
    with open(output_path, 'wb') as f:
        for ibatch in tqdm(range(bfile.n_batches), desc="Writing SI binary"):
            batch_data = bfile.padded_batch_to_torch(ibatch, ops=ops)
            batch_data = batch_data[:, ops['nt']:-ops['nt']]  # remove padding
            f.write(batch_data.cpu().numpy().astype(np.float32).tobytes())
    
    # Create SI binary recording object with metadata
    rec_si = si.read_binary(
        file_paths=[str(output_path)],
        sampling_frequency=float(fs),
        num_channels=int(n_channels),
        dtype=np.float32,
        is_filtered=True,
    )
    # Attach probe information (contact locations)
    rec_si.set_channel_locations(np.column_stack([ops['xc'], ops['yc']]))
    # Save with SI metadata (binary.json, probe.json, etc.)
    rec_si = rec_si.save(folder=str(results_dir_corrected), format='binary', verbose=False)
    logger.info(f"  ✓ Saved drift-corrected data to {results_dir_corrected}")

    imin = int(bfile.imin)

    print("running mua_pipeline...")
    # ── 6. mua_pipeline spike detection ───────────────────────────────────────────
    logger.info(f"\n[6/10] Running mua_pipeline on drift-corrected data …")
    logger.info(f"  Input: {results_dir_corrected} (SI binary)")
    logger.info(f"  Settings: {MUA_SETTINGS}")
    (_, peaks_combined2, *_) = run_mua_pipeline(
        bin_path=str(results_dir_corrected),  # SI-saved folder path
        probe=probe,  # Contains xc, yc, chanMap, kcoords for channel locations
        **MUA_SETTINGS,
    )
    peaks = peaks_combined2
    logger.info(f"  peaks_combined2: {len(peaks):,} spikes")

    print("translating sample indices...")
    # ── 7. Translate sample indices to KS4 absolute coordinates ──────────
    # mua_pipeline loads SI recording starting at frame 0.
    # Add imin to convert from SI frame to absolute KS4 sample index.
    peaks = peaks.copy()
    peaks["sample_index"] += imin
    peaks = np.sort(peaks, order="sample_index")
    logger.info(f"  Sample range after +imin={imin}: "
                f"[{peaks['sample_index'].min()}, {peaks['sample_index'].max()}]")

    print("learning wPCA...")
    # ── 8. Learn wPCA and universal template grid ─────────────────────────
    logger.info("\n[7/10] Learning wPCA + template grid (KS4 detections discarded) …")
    _st_discard, _tF_discard, ops = spikedetect.run(ops, bfile, device=device)
    del _st_discard, _tF_discard
    logger.info(f"  wPCA: {ops['wPCA'].shape}  "
                f"iC: {ops['iC'].shape}")

    print("assigning spikes to template positions...")
    # ── 9. Assign spikes → template positions + extract features ──────────
    logger.info("\n[8/10] Assigning spikes to template positions …")
    template_idx = _assign_templates(peaks, ops)
    uniq, counts = np.unique(template_idx, return_counts=True)
    logger.info(f"  {len(uniq)} templates used  "
                f"(avg {counts.mean():.0f} / max {counts.max():,} per template)")

    logger.info("\n[9/10] Extracting wPCA features …")
    st, tF, valid_mask = _extract_features(
        bfile, ops, peaks, template_idx, device
    )
    logger.info(f"  valid: {len(st):,}  dropped: {(~valid_mask).sum():,}")

    print("clustering...")
    # ── 10. Cluster + save ────────────────────────────────────────────────
    logger.info("\n[10/10] Clustering (clustering_qr + merging) …")
    clu, Wall, st, tF = cluster_spikes(
        st, tF, ops, device, bfile, tic0=tic0, verbose=False,
    )
    n_units = int(clu.max()) + 1
    logger.info(f"  {n_units} units found")

    logger.info(f"\nSaving Phy output → {results_dir} …")
    ops, _, is_ref, _, _ = save_sorting(
        ops, results_dir, st, clu, tF, Wall,
        bfile.imin, tic0=tic0,
        save_extra_vars=SAVE_EXTRA_VARS,
        save_preprocessed_copy=True,
    )

    runtime = time.time() - tic0
    logger.info(f"\n✓  Done in {runtime / 60:.1f} min  |  "
                f"{n_units} units  ({int(is_ref.sum())} refractory)")
    logger.info(f"   phy template-gui {results_dir / 'params.py'}")
    close_logger()


if __name__ == "__main__":
    main()