"""
DREDge motion estimation with time-chunking to avoid OOM.

Two modes:
  --chunk_idx N    Estimate motion for chunk N and save to chunks/motion_dredge_{start}min_{end}min/
  --stitch         Load all saved chunks, align via overlap, save combined motion_dredge_all

Usage:
  # estimate chunk 3
  python run_dredge_chunked.py --chunk_idx 3

  # stitch all chunks into one motion object
  python run_dredge_chunked.py --stitch
"""

import argparse
import json
import numpy as np
from pathlib import Path
import spikeinterface.full as si
from spikeinterface.sortingcomponents.motion import estimate_motion

# =============================================================================
# CONFIGURATION — mirror peak_localisation.py settings
# =============================================================================

BASE_PATH       = r"/ptmp/gongt/GLC7189/GLC7189_g_g1/preprocessed"
FOLDER_NAME     = "thresholding_peaks_3.3"
PEAKS_FILENAME  = "peaks_combined2_sorted.npy"
# LOC_FILENAME    = None
LOC_FILENAME    = "localized_peaks_mt_raw.npy"  # set to None to derive locations from max channel depth

CHUNK_DURATION_S = 70 * 60   # 70 min per chunk
OVERLAP_S        = 10 * 60   # 10 min overlap (>= time_horizon_s for reliable alignment)
STEP_S           = CHUNK_DURATION_S - OVERLAP_S   # 60 min advance per chunk

# Minimum amplitude percentile for peaks passed to DREDge (0 = no filter).
# Filtering to high-amplitude peaks reduces noise in the spatial histogram
# and can substantially improve motion estimates on noisy datasets.
AMP_PERCENTILE   = 20

DREDGE_PARAMS = {
    'time_horizon_s': 600,
    'win_step_um': 150,
    'win_scale_um': 500,        # reduced further for tighter spatial locality
    'max_disp_um': 200,
    'bin_s': 1,
    'mincorr': 0.5,            # (default 0.1 is too permissive); drops noisy pairs before solve
    'thomas_kw': {
        'lambda_s': 0.5,
        'lambda_t': 1},     
    'device': 'cuda',
    'win_shape': 'rect'
}

# =============================================================================
# PATHS
# =============================================================================

base_dir    = Path(BASE_PATH) / FOLDER_NAME
motion_dir  = base_dir / "motion_estimates"
# chunks_dir and output_name are set at runtime from --run_name (see main below)
chunks_dir  = motion_dir / "chunks"
OUTPUT_NAME = "motion_dredge_all"

# =============================================================================
# HELPERS
# =============================================================================

def get_chunk_windows(total_duration_s):
    """Return list of (start_s, end_s) for all chunks."""
    windows = []
    start = 0.0
    while start < total_duration_s:
        end = min(start + CHUNK_DURATION_S, total_duration_s)
        windows.append((start, end))
        if end >= total_duration_s:
            break
        start += STEP_S
    return windows


def estimate_chunk(chunk_idx, run_name=None, amp_percentile=None):
    """Run DREDge on a single chunk and save the result."""
    _chunks_dir = (motion_dir / f"chunks_{run_name}") if run_name else chunks_dir
    _chunks_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading recording from: {BASE_PATH}")
    rec = si.load(BASE_PATH)
    fs  = rec.get_sampling_frequency()
    total_duration = rec.get_total_duration()

    windows = get_chunk_windows(total_duration)
    if chunk_idx >= len(windows):
        raise ValueError(f"chunk_idx {chunk_idx} out of range (0–{len(windows)-1})")

    start_s, end_s = windows[chunk_idx]
    start_frame = int(start_s * fs)
    end_frame   = min(int(end_s * fs), rec.get_num_samples())

    print(f"\nChunk {chunk_idx}/{len(windows)-1}:  {start_s/60:.1f} – {end_s/60:.1f} min"
          f"  ({end_frame - start_frame} frames)")

    # Sub-recording — sets correct absolute time offset for DREDge histogram
    sub_rec = rec.frame_slice(start_frame, end_frame)

    # Load peaks & locations
    peaks_path = base_dir / PEAKS_FILENAME
    peaks      = np.load(peaks_path)

    if LOC_FILENAME is not None:
        print(f"  Loading peak locations from: {LOC_FILENAME}")
        locs = np.load(base_dir / LOC_FILENAME)
    else:
        # Derive (x, y) from the peak's max channel position
        print("  No peak locations provided, deriving from max channel depths...")
        ch_locs = rec.get_channel_locations()  # (n_channels, 2): columns are [x, depth]
        locs = np.zeros(len(peaks), dtype=[('x', 'f8'), ('y', 'f8')])
        locs['x'] = ch_locs[peaks['channel_index'], 0]
        locs['y'] = ch_locs[peaks['channel_index'], 1]
        print("  Using max-channel depths as peak locations")

    # Filter to this chunk
    mask = (peaks["sample_index"] >= start_frame) & (peaks["sample_index"] < end_frame)
    chunk_peaks = peaks[mask].copy()
    chunk_locs  = locs[mask].copy()

    # Amplitude filter (optional)
    amp_pct = amp_percentile if amp_percentile is not None else AMP_PERCENTILE
    if amp_pct > 0:
        amp_threshold = np.percentile(np.abs(chunk_peaks['amplitude']), amp_pct)
        amp_mask = np.abs(chunk_peaks['amplitude']) >= amp_threshold
        chunk_peaks = chunk_peaks[amp_mask]
        chunk_locs  = chunk_locs[amp_mask]
        print(f"  Amplitude filter: top {100 - amp_pct}%  (threshold={amp_threshold:.4f}, kept {amp_mask.sum():,}/{mask.sum():,})")

    # Make sample indices relative to chunk start so DREDge histogram is chunk-sized
    chunk_peaks["sample_index"] -= start_frame

    print(f"  Peaks in chunk: {len(chunk_peaks):,}")

    start_min = int(round(start_s / 60))
    end_min   = int(round(end_s   / 60))
    amp_suffix = f"_amp{amp_pct}pct" if amp_pct > 0 else ""
    save_path = _chunks_dir / f"motion_dredge{amp_suffix}_{start_min:04d}min_{end_min:04d}min"
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

    # Save motion parameters alongside chunks
    params_record = {
        "method": "dredge_ap",
        "dredge_params": DREDGE_PARAMS,
        "chunk_duration_s": CHUNK_DURATION_S,
        "overlap_s": OVERLAP_S,
        "step_s": STEP_S,
        "base_path": BASE_PATH,
        "folder_name": FOLDER_NAME,
        "peaks_filename": PEAKS_FILENAME,
        "loc_filename": LOC_FILENAME,
        "loc_mode": "localized" if LOC_FILENAME is not None else "max_channel",
        "amp_percentile": amp_pct,
    }
    params_path = _chunks_dir / "motion_params.json"
    with open(params_path, "w") as f:
        json.dump(params_record, f, indent=2)
    print(f"  Motion params saved to: {params_path}")


def stitch_chunks(run_name=None):
    """Load all chunk motions, align via overlap, save a combined si.Motion."""
    import glob

    # Discover chunks in order
    _chunks_dir = (motion_dir / f"chunks_{run_name}") if run_name else chunks_dir
    _output_name = f"motion_dredge_{run_name}" if run_name else OUTPUT_NAME
    chunk_paths = sorted(_chunks_dir.glob("motion_dredge_*min_*min"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunks found in {chunks_dir}")

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
    # corrections[i] = cumulative offset (shape: n_spatial_windows) to add to chunk i
    corrections = [np.zeros(n_spatial)]

    for i in range(len(motions) - 1):
        t_i   = motions[i].temporal_bins_s[0]
        t_i1  = motions[i + 1].temporal_bins_s[0]
        d_i   = motions[i].displacement[0]     # (T_i, S)
        d_i1  = motions[i + 1].displacement[0] # (T_i1, S)

        t_overlap_start = t_i1[0]
        t_overlap_end   = t_i[-1]

        mask_i   = (t_i  >= t_overlap_start) & (t_i  <= t_overlap_end)
        mask_i1  = (t_i1 >= t_overlap_start) & (t_i1 <= t_overlap_end)

        if mask_i.sum() < 5 or mask_i1.sum() < 5:
            print(f"  WARNING: chunk {i}↔{i+1} overlap too short for alignment — keeping previous correction")
            corrections.append(corrections[-1].copy())
            continue

        mean_i   = d_i[mask_i].mean(axis=0)    # (S,)
        mean_i1  = d_i1[mask_i1].mean(axis=0)  # (S,)

        # offset to add to chunk i+1 so its overlap mean matches chunk i's
        new_corr = corrections[-1] + (mean_i - mean_i1)
        corrections.append(new_corr)
        print(f"  Chunk {i}↔{i+1} alignment offset: mean={np.mean(np.abs(mean_i - mean_i1)):.2f} µm")

    # Apply corrections
    corrected_disps = [motions[i].displacement[0] + corrections[i] for i in range(len(motions))]

    # ---------- Step 2: merge (average in overlap, concatenate elsewhere) ----------
    # Collect all unique time bins and accumulate displacements
    all_t   = np.concatenate([motions[i].temporal_bins_s[0] for i in range(len(motions))])
    all_d   = np.concatenate(corrected_disps, axis=0)  # (total_bins, S)

    unique_t, first_idx = np.unique(all_t, return_index=True)
    # For overlapping bins, average: use np.unique inverse to accumulate then divide
    _, inv = np.unique(all_t, return_inverse=True)
    merged_disp  = np.zeros((len(unique_t), n_spatial), dtype=np.float64)
    merged_count = np.zeros(len(unique_t), dtype=np.float64)

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
    parser = argparse.ArgumentParser(description="DREDge chunked motion estimation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--chunk_idx", type=int, metavar="N",
                       help="Run DREDge for chunk N (0-indexed)")
    group.add_argument("--stitch", action="store_true",
                       help="Stitch all saved chunks into combined motion")
    group.add_argument("--list_chunks", action="store_true",
                       help="List all chunks without running anything")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name suffix for this run (e.g. 'v2'). "
                             "Chunks go to chunks_<run_name>/, output to motion_dredge_<run_name>.")
    parser.add_argument("--amp_pct", type=int, default=None,
                        help="Minimum amplitude percentile for DREDge peaks (e.g. 50 keeps top 50%%). "
                             "Overrides AMP_PERCENTILE in script. 0 or omit = no filter.")
    args = parser.parse_args()

    if args.list_chunks:
        rec = si.load(BASE_PATH)
        windows = get_chunk_windows(rec.get_total_duration())
        print(f"Total duration: {rec.get_total_duration()/60:.1f} min")
        print(f"Chunk duration: {CHUNK_DURATION_S/60:.0f} min, overlap: {OVERLAP_S/60:.0f} min, step: {STEP_S/60:.0f} min")
        print(f"Number of chunks: {len(windows)}")
        for i, (s, e) in enumerate(windows):
            print(f"  Chunk {i:3d}:  {s/60:7.2f} – {e/60:7.2f} min")
    elif args.stitch:
        stitch_chunks(run_name=args.run_name)
    else:
        estimate_chunk(args.chunk_idx, run_name=args.run_name, amp_percentile=args.amp_pct)
