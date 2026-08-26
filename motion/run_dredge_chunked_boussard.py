"""
Only needed for the Boussard-localization comparison track, not the default
pipeline (which uses run_dredge_chunked.py directly on
peak_localisation.py's monopolar-triangulation output). Use this only if
you're specifically working with boussard_localization.py's output — see
peak_detect_localise/boussard_localization.py's header for why that's not
the default.

DREDge motion estimation from Boussard localisation outputs, with time-chunking.

Unlike run_dredge_chunked.py (which uses SpikeInterface peaks + localized_peaks_*.npy),
this script reads peak times and locations directly from the Boussard localisation
output directory (boussard_localization_merged/).

Boussard output files used
--------------------------
  boussard_spike_index_30min.npy   — (N, 2) int64: [sample_index, channel_index]
                                     for every peak fed to the localiser
  boussard_localization_merged/
    times_read.npy                 — binary validity mask (1 = successfully localised)
    results_z_merged.npy           — depth  y  (µm)
    results_x_merged.npy           — lateral x  (µm)
    results_max_ptp_merged.npy     — max PTP amplitude (used as peak amplitude)
    results_alpha_merged.npy       — alpha amplitude (alternative; see AMPLITUDE_FIELD)

Only peaks where times_read == 1 are passed to DREDge.  All other Boussard arrays
(spread, width, z_mean, etc.) are ignored by this script.

Effective coverage: the Boussard data covers roughly the first 30 min of the recording.
Chunking is therefore sized relative to that coverage, not the full recording length.

Two modes
---------
  --chunk_idx N    Estimate motion for chunk N, save to motion_estimates/chunks[_<run_name>]/
  --stitch         Align and merge all saved chunks into motion_estimates/motion_dredge_boussard[_<run_name>]

Usage
-----
  python run_dredge_chunked_boussard.py --list_chunks
  python run_dredge_chunked_boussard.py --chunk_idx 0
  python run_dredge_chunked_boussard.py --stitch
"""

import argparse
import json
import numpy as np
from pathlib import Path
import spikeinterface.full as si
from spikeinterface.sortingcomponents.motion import estimate_motion

# =============================================================================
# CONFIGURATION
# =============================================================================

# SpikeInterface recording (needed for fs, sub-recording slices, channel locations)
BASE_PATH = r"/ptmp/gongt/GLC7189/GLC7189_g_g1/preprocessed"

# Parent folder that contains both the spike-index file and the localization directory
PEAKS_FOLDER = r"/ptmp/gongt/GLC7189/GLC7189_g_g1/preprocessed/thresholding_peaks_3.3"

# spike_index file: (N, 2) int64 columns [sample_index, channel_index]
# Row order matches the N entries in every boussard_localization_merged array.
SPIKE_INDEX_FILENAME = "boussard_spike_index_30min.npy"

# Boussard localisation output directory
BOUSSARD_DIRNAME = "boussard_localization_merged"

# Which amplitude array to use as peak amplitude for DREDge:
#   "ptp"   → results_max_ptp_merged.npy  (peak-to-peak voltage, always positive)
#   "alpha" → results_alpha_merged.npy    (dipole amplitude proxy)
AMPLITUDE_FIELD = "ptp"

# Chunk / overlap settings (tuned for ~30 min of Boussard coverage)
CHUNK_DURATION_S = 35 * 60   # 30 min per chunk
OVERLAP_S        = 5  * 60   # 5 min overlap for reliable stitching
STEP_S           = CHUNK_DURATION_S - OVERLAP_S  # 25 min advance

# Minimum amplitude percentile for peaks passed to DREDge (0 = no filter).
AMP_PERCENTILE = 0

# Spatial depth range filter (µm).  0 / None = no cutoff.
DEPTH_MIN = 0       # inclusive lower bound
DEPTH_MAX = None    # inclusive upper bound (None = no upper cutoff)

DREDGE_PARAMS = {
    'time_horizon_s': 300,
    'win_step_um': 150,
    'win_scale_um': 300,
    'max_disp_um': 200,
    'bin_s': 1,
    'mincorr': 0.5,
    'thomas_kw': {
        'lambda_s': 0.5,
        'lambda_t': 1,
    },
    'device': 'cuda',
    'win_margin_um': 0,
}

# =============================================================================
# PATHS  (derived; do not edit unless you know what you're doing)
# =============================================================================

peaks_folder   = Path(PEAKS_FOLDER)
boussard_dir   = peaks_folder / BOUSSARD_DIRNAME
spike_idx_path = peaks_folder / SPIKE_INDEX_FILENAME
motion_dir     = boussard_dir / "motion_estimates"
chunks_dir     = motion_dir / "chunks"
OUTPUT_NAME    = "motion_dredge_boussard_30min"

# =============================================================================
# DATA LOADING  (done once per process; called inside estimate_chunk)
# =============================================================================

def load_boussard_peaks(rec_fs):
    """
    Load all valid Boussard peaks and return (peaks_sa, locs_sa).

    peaks_sa  — structured array dtype [sample_index i8, channel_index i8,
                                         amplitude f8, segment_index i8]
    locs_sa   — structured array dtype [x f8, y f8]

    Only entries where times_read == 1 are included.
    """
    spike_index = np.load(spike_idx_path)            # (N, 2): [sample_index, channel_index]
    valid_mask  = np.load(boussard_dir / "times_read.npy") == 1

    z   = np.load(boussard_dir / "results_z_merged.npy")[valid_mask]
    x   = np.load(boussard_dir / "results_x_merged.npy")[valid_mask]

    if AMPLITUDE_FIELD == "ptp":
        amp = np.load(boussard_dir / "results_max_ptp_merged.npy")[valid_mask]
    else:
        amp = np.load(boussard_dir / "results_alpha_merged.npy")[valid_mask]

    sample_idx  = spike_index[valid_mask, 0].astype(np.int64)
    channel_idx = spike_index[valid_mask, 1].astype(np.int64)

    n = valid_mask.sum()
    peaks_sa = np.zeros(n, dtype=[('sample_index',  '<i8'),
                                   ('channel_index', '<i8'),
                                   ('amplitude',     '<f8'),
                                   ('segment_index', '<i8')])
    peaks_sa['sample_index']  = sample_idx
    peaks_sa['channel_index'] = channel_idx
    peaks_sa['amplitude']     = amp
    peaks_sa['segment_index'] = 0

    locs_sa = np.zeros(n, dtype=[('x', '<f8'), ('y', '<f8')])
    locs_sa['x'] = x
    locs_sa['y'] = z

    # Sort by sample_index so chunk slicing with searchsorted works reliably
    order = np.argsort(peaks_sa['sample_index'], kind='stable')
    peaks_sa = peaks_sa[order]
    locs_sa  = locs_sa[order]

    print(f"  Boussard valid peaks: {n:,}  "
          f"(sample range {sample_idx.min():,} – {sample_idx.max():,},"
          f"  {sample_idx.max() / rec_fs / 60:.1f} min)")
    return peaks_sa, locs_sa


def boussard_max_sample():
    """Return the maximum sample_index in the Boussard spike-index file."""
    spike_index = np.load(spike_idx_path)
    return int(spike_index[:, 0].max())

# =============================================================================
# CHUNKING
# =============================================================================

def get_chunk_windows(total_duration_s):
    """Return list of (start_s, end_s) tuples covering total_duration_s."""
    windows = []
    start = 0.0
    while start < total_duration_s:
        end = min(start + CHUNK_DURATION_S, total_duration_s)
        windows.append((start, end))
        if end >= total_duration_s:
            break
        start += STEP_S
    return windows

# =============================================================================
# CHUNK ESTIMATION
# =============================================================================

def estimate_chunk(chunk_idx, run_name=None, amp_percentile=None,
                   depth_min=None, depth_max=None):
    """Run DREDge on a single Boussard chunk and save the result."""
    _chunks_dir = (motion_dir / f"chunks_{run_name}") if run_name else chunks_dir
    _chunks_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading recording from: {BASE_PATH}")
    rec = si.load(BASE_PATH)
    fs  = rec.get_sampling_frequency()

    # Use the Boussard coverage to define total duration (not the full recording)
    boussard_end_sample = boussard_max_sample()
    total_duration_s    = boussard_end_sample / fs
    print(f"  Boussard coverage: {total_duration_s / 60:.1f} min  ({boussard_end_sample:,} samples)")

    windows = get_chunk_windows(total_duration_s)
    if chunk_idx >= len(windows):
        raise ValueError(f"chunk_idx {chunk_idx} out of range (0–{len(windows)-1})")

    start_s, end_s = windows[chunk_idx]
    start_frame = int(start_s * fs)
    end_frame   = min(int(end_s * fs), boussard_end_sample + 1)

    print(f"\nChunk {chunk_idx}/{len(windows)-1}:  {start_s/60:.1f} – {end_s/60:.1f} min"
          f"  ({end_frame - start_frame} frames)")

    # Sub-recording sets the correct absolute time offset for DREDge histogram
    sub_rec = rec.frame_slice(start_frame, end_frame)

    # Load and filter Boussard peaks
    print("  Loading Boussard localisation data...")
    all_peaks, all_locs = load_boussard_peaks(fs)

    # Restrict to this chunk via searchsorted (peaks are sorted by sample_index)
    lo = np.searchsorted(all_peaks['sample_index'], start_frame,  side='left')
    hi = np.searchsorted(all_peaks['sample_index'], end_frame,    side='left')
    chunk_peaks = all_peaks[lo:hi].copy()
    chunk_locs  = all_locs[lo:hi].copy()

    # Amplitude filter (optional)
    amp_pct = amp_percentile if amp_percentile is not None else AMP_PERCENTILE
    if amp_pct > 0:
        amp_threshold = np.percentile(np.abs(chunk_peaks['amplitude']), amp_pct)
        amp_mask = np.abs(chunk_peaks['amplitude']) >= amp_threshold
        chunk_peaks = chunk_peaks[amp_mask]
        chunk_locs  = chunk_locs[amp_mask]
        print(f"  Amplitude filter: top {100 - amp_pct}%  "
              f"(threshold={amp_threshold:.4f}, kept {amp_mask.sum():,}/{hi - lo:,})")

    # Spatial (depth) filter
    _dmin = depth_min if depth_min is not None else DEPTH_MIN
    _dmax = depth_max if depth_max is not None else DEPTH_MAX
    if _dmin > 0 or _dmax is not None:
        depth_mask = chunk_locs['y'] >= _dmin
        if _dmax is not None:
            depth_mask &= chunk_locs['y'] <= _dmax
        n_before = len(chunk_peaks)
        chunk_peaks = chunk_peaks[depth_mask]
        chunk_locs  = chunk_locs[depth_mask]
        _dmax_str = f'{_dmax:.0f}' if _dmax is not None else 'max'
        print(f"  Depth filter: {_dmin:.0f} – {_dmax_str} µm  "
              f"(kept {len(chunk_peaks):,}/{n_before:,})")

    # Make sample indices relative to chunk start
    chunk_peaks['sample_index'] -= start_frame
    print(f"  Peaks in chunk: {len(chunk_peaks):,}")

    # Build output path
    start_min = int(round(start_s / 60))
    end_min   = int(round(end_s   / 60))
    amp_suffix = f"_amp{amp_pct}pct" if amp_pct > 0 else ""
    _dmin_eff = _dmin
    _dmax_eff = _dmax
    if _dmin_eff > 0 and _dmax_eff is None:
        depth_suffix = f"_d{int(_dmin_eff)}um+"
    elif _dmin_eff == 0 and _dmax_eff is not None:
        depth_suffix = f"_d0-{int(_dmax_eff)}um"
    elif _dmin_eff > 0 and _dmax_eff is not None:
        depth_suffix = f"_d{int(_dmin_eff)}-{int(_dmax_eff)}um"
    else:
        depth_suffix = ""
    save_path = _chunks_dir / f"motion_dredge{amp_suffix}{depth_suffix}_{start_min:04d}min_{end_min:04d}min"

    if save_path.exists():
        print(f"  Already exists, skipping: {save_path}")
        return

    print("  Running DREDge...")
    motion = estimate_motion(
        recording=sub_rec,
        peaks=chunk_peaks,
        peak_locations=chunk_locs,
        method="dredge_ap",
        progress_bar=True,
        **DREDGE_PARAMS,
    )
    motion.save(save_path)
    print(f"  Saved to: {save_path}")

    # Persist run parameters alongside chunks
    params_record = {
        "method": "dredge_ap",
        "dredge_params": DREDGE_PARAMS,
        "chunk_duration_s": CHUNK_DURATION_S,
        "overlap_s": OVERLAP_S,
        "step_s": STEP_S,
        "base_path": BASE_PATH,
        "boussard_dir": str(boussard_dir),
        "spike_index_file": str(spike_idx_path),
        "amplitude_field": AMPLITUDE_FIELD,
        "amp_percentile": amp_pct,
        "depth_min_um": _dmin_eff,
        "depth_max_um": _dmax_eff,
    }
    params_path = _chunks_dir / "motion_params.json"
    with open(params_path, "w") as f:
        json.dump(params_record, f, indent=2)
    print(f"  Motion params saved to: {params_path}")


# =============================================================================
# STITCH
# =============================================================================

def stitch_chunks(run_name=None):
    """Load all chunk motions, align via overlap, save a combined si.Motion."""
    _chunks_dir  = (motion_dir / f"chunks_{run_name}") if run_name else chunks_dir
    _output_name = f"motion_dredge_boussard_{run_name}" if run_name else OUTPUT_NAME

    chunk_paths = sorted(_chunks_dir.glob("motion_dredge_*min_*min"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunks found in {_chunks_dir}")

    print(f"Found {len(chunk_paths)} chunks to stitch:")
    motions = []
    for p in chunk_paths:
        m = si.Motion.load(p)
        motions.append(m)
        print(f"  {p.name}  t=[{m.temporal_bins_s[0].min():.1f}, {m.temporal_bins_s[0].max():.1f}]s"
              f"  shape={m.displacement[0].shape}")

    # Verify all chunks share the same spatial bins
    ref_spatial = motions[0].spatial_bins_um
    for i, m in enumerate(motions[1:], 1):
        if not np.allclose(m.spatial_bins_um, ref_spatial, atol=0.1):
            raise ValueError(f"Chunk {i} has different spatial_bins_um — cannot stitch")

    n_spatial = len(ref_spatial)

    # ---------- Step 1: align chunks via overlap offset ----------
    corrections = [np.zeros(n_spatial)]

    for i in range(len(motions) - 1):
        t_i   = motions[i].temporal_bins_s[0]
        t_i1  = motions[i + 1].temporal_bins_s[0]
        d_i   = motions[i].displacement[0]
        d_i1  = motions[i + 1].displacement[0]

        t_overlap_start = t_i1[0]
        t_overlap_end   = t_i[-1]

        mask_i   = (t_i  >= t_overlap_start) & (t_i  <= t_overlap_end)
        mask_i1  = (t_i1 >= t_overlap_start) & (t_i1 <= t_overlap_end)

        if mask_i.sum() < 5 or mask_i1.sum() < 5:
            print(f"  WARNING: chunk {i}↔{i+1} overlap too short — keeping previous correction")
            corrections.append(corrections[-1].copy())
            continue

        mean_i   = d_i[mask_i].mean(axis=0)
        mean_i1  = d_i1[mask_i1].mean(axis=0)
        new_corr = corrections[-1] + (mean_i - mean_i1)
        corrections.append(new_corr)
        print(f"  Chunk {i}↔{i+1} alignment offset: mean={np.mean(np.abs(mean_i - mean_i1)):.2f} µm")

    # Apply corrections
    corrected_disps = [motions[i].displacement[0] + corrections[i] for i in range(len(motions))]

    # ---------- Step 2: merge (average in overlap, concatenate elsewhere) ----------
    all_t = np.concatenate([motions[i].temporal_bins_s[0] for i in range(len(motions))])
    all_d = np.concatenate(corrected_disps, axis=0)

    unique_t, _ = np.unique(all_t, return_index=True)
    _, inv = np.unique(all_t, return_inverse=True)
    merged_disp  = np.zeros((len(unique_t), n_spatial), dtype=np.float64)
    merged_count = np.zeros(len(unique_t),               dtype=np.float64)

    for j in range(len(all_t)):
        merged_disp[inv[j]]  += all_d[j]
        merged_count[inv[j]] += 1

    merged_disp /= merged_count[:, np.newaxis]

    # ---------- Step 3: save ----------
    combined = si.Motion(
        displacement=merged_disp,
        temporal_bins_s=unique_t,
        spatial_bins_um=ref_spatial,
    )
    save_path = motion_dir / _output_name
    combined.save(save_path)
    print(f"\nCombined motion saved to: {save_path}")
    print(f"  Time range: {unique_t.min():.1f} – {unique_t.max():.1f} s")
    print(f"  Shape: {merged_disp.shape}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DREDge chunked motion estimation from Boussard localisation outputs"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--chunk_idx", type=int, metavar="N",
                       help="Run DREDge for chunk N (0-indexed)")
    group.add_argument("--stitch", action="store_true",
                       help="Stitch all saved chunks into combined motion")
    group.add_argument("--list_chunks", action="store_true",
                       help="List all chunks without running anything")

    parser.add_argument("--run_name", type=str, default=None,
                        help="Name suffix for this run (e.g. 'v2'). "
                             "Chunks go to chunks_<run_name>/, output to motion_dredge_boussard_<run_name>.")
    parser.add_argument("--amp_pct", type=int, default=None,
                        help="Minimum amplitude percentile filter (e.g. 50 keeps top 50%%). "
                             "0 or omit = no filter.")
    parser.add_argument("--depth_min", type=float, default=None,
                        help="Keep peaks at or above this depth (µm).")
    parser.add_argument("--depth_max", type=float, default=None,
                        help="Keep peaks at or below this depth (µm).")
    args = parser.parse_args()

    if args.list_chunks:
        rec = si.load(BASE_PATH)
        fs  = rec.get_sampling_frequency()
        total_s = boussard_max_sample() / fs
        windows = get_chunk_windows(total_s)
        print(f"Boussard coverage: {total_s / 60:.1f} min")
        print(f"Chunk: {CHUNK_DURATION_S/60:.0f} min, overlap: {OVERLAP_S/60:.0f} min, step: {STEP_S/60:.0f} min")
        print(f"Number of chunks: {len(windows)}")
        for i, (s, e) in enumerate(windows):
            print(f"  Chunk {i:3d}:  {s/60:6.2f} – {e/60:6.2f} min")

    elif args.stitch:
        stitch_chunks(run_name=args.run_name)

    else:
        estimate_chunk(
            args.chunk_idx,
            run_name=args.run_name,
            amp_percentile=args.amp_pct,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
        )
