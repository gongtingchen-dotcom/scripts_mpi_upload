"""Full data-loading + preprocessing pipeline for the GLC7189/c_g0 texture
decoding search: raw recording/peaks -> lagged feature matrix
(X_lagged_search / y_search) ready for a decoder to train on. This is
everything theo_pipeline_new.py's main() does up to (not including)
build_models()/the actual model-fit loop.

Hardcoded to the exact paths + settings currently used for this search (see
build_search_grid.py's FIXED_ARGS), and drops branches that don't fire for
this specific run (AP-channel-number remapping, spike-count thresholding,
channel-pooling/smoothing, multi-row dataset-match disambiguation) --
see theo_pipeline_new.py for the general, CLI-configurable version with all
of those options.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import spikeinterface.full as si

from Neural_Decoding.preprocessing_funcs import LagMat, bin_spikes


# --- Drift/motion correction helpers -----------------------------------------
# Each peak's depth is corrected by the DREDge motion estimate at its own
# time, then snapped to whichever channel is now physically nearest -- that
# becomes its corrected channel_index for MUA binning.

def get_corrected_positions(peak_times_sec, positions, motion_obj):
    """positions: (n_peaks, 2) [x, y]. Shifts depth by the motion estimate at
    each peak's own time (peak_times_sec must be in the recording's absolute
    time frame, matching motion_obj.temporal_bins_s)."""
    depths = positions[:, 1]
    shifts = motion_obj.get_displacement_at_time_and_depth(peak_times_sec, depths)
    corrected = positions.copy()
    corrected[:, 1] = depths - shifts
    return corrected


def snap_to_nearest_channel(positions, channel_positions, chunk_size=10000):
    """Snap each (x, y) position to its nearest channel. Chunked so the
    (n_peaks, n_channels) distance matrix never fully materializes."""
    snapped_ch_idx = np.empty(len(positions), dtype=int)
    for start in range(0, len(positions), chunk_size):
        stop = min(start + chunk_size, len(positions))
        chunk = positions[start:stop]
        diff = chunk[:, None, :] - channel_positions[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        snapped_ch_idx[start:stop] = np.argmin(dist2, axis=1)
    return snapped_ch_idx


def bin_output_safe(outputs, output_times, dt, t_start, t_end):
    """Mean-pool `outputs` into dt-wide bins over [t_start, t_end); a bin
    with no rows in it is left NaN (dropped later via valid_mask)."""
    edges = np.arange(t_start, t_end, dt)
    num_bins = edges.shape[0] - 1
    out = np.full((num_bins, outputs.shape[1]), np.nan, dtype=float)
    for i in range(num_bins):
        m = (output_times >= edges[i]) & (output_times < edges[i + 1])
        if np.any(m):
            out[i, :] = np.nanmean(outputs[m, :], axis=0)
    return out, edges


# ==============================================================================
# 1. Paths + settings for this run (GLC7189/c_g0, matches build_search_grid.py)
# ==============================================================================
recording_path = "/nexus/posix0/MBR-neuralsystems/ephys/GLC7189/GLC7189_c_g0/preprocessed"
peaks_path = "/nexus/posix0/MBR-neuralsystems/ephys/GLC7189/GLC7189_c_g0/preprocessed/thresholding_peaks_3.5/peaks_combined2_sorted.npy"
motion_path = Path(peaks_path).parent / "motion_estimates" / "motion_dredge"
localized_peaks_path = Path(peaks_path).parent / "localized_peaks_mt_raw.npy"
behavior_df_path = "/ptmp/gongt/behaviour/20260603_texture_pca_df.pkl"
metadata_df_path = "/ptmp/gongt/behaviour/20260603_metadata_df.pkl"

animal = "GLC7189"
neural_prefix = "c_g0"          # (animal, neural_prefix) -> metadata_df's "dataset" id
pc_cols = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10", "PC11", "PC12", "PC13", "PC14", "PC15"]  # behavior outputs to decode

dt = 0.2                                   # bin width, seconds
window_before, window_after = 2, 2  # lag window: seconds before/after
bin_current = 1                            # number of bins to include for the "current" time point
n_splits = 20                              # CV fold count used by the real search (recorded below, not applied here)
holdout_frac = 0                         # fraction of the session's tail reserved, excluded from the search

# ==============================================================================
# 2. Load the recording + detected peaks
# ==============================================================================
rec = si.load(recording_path)
peaks = np.load(peaks_path, allow_pickle=True)

channel_ids = rec.get_channel_ids()
channel_positions = rec.get_channel_locations()
fs = float(rec.get_sampling_frequency())
time_offset = float(rec.get_times()[0])
n_channels = channel_ids.shape[0]

peak_channel_index = peaks["channel_index"].astype(int) # peak channel for each spike
peak_times_sec = peaks["sample_index"].astype(float) / fs + time_offset
print(f"Loaded {len(peaks):,} peaks across {n_channels} channels.")

assert peak_channel_index.min() >= 0 and peak_channel_index.max() < n_channels
peak_row_index = peak_channel_index

# ==============================================================================
# 3. Drift/motion correction inputs
# ==============================================================================
motion = si.Motion.load(str(motion_path))
localized_peaks = np.load(str(localized_peaks_path), allow_pickle=True)
localized_y = localized_peaks["y"].astype(float)
print(f"Loaded motion estimate: {motion_path}")
print(f"Loaded localized peaks: {localized_peaks_path}")

# ==============================================================================
# 4. Load behavior (PC traces) + match this recording's session
# ==============================================================================
texture_pca_df = pd.read_pickle(behavior_df_path)
metadata_df = pd.read_pickle(metadata_df_path)

meta_match = metadata_df[(metadata_df["animal"] == animal) & (metadata_df["neural_prefix"] == neural_prefix)]
assert len(meta_match) == 1, (
    f"expected exactly one metadata_df row for animal={animal!r}, neural_prefix={neural_prefix!r}, "
    f"got {len(meta_match)} -- theo_pipeline_new.py disambiguates multiple matches by checking which "
    "candidate dataset id actually has rows in texture_pca_df."
)
dataset_id = meta_match["dataset"].iloc[0]
beh = texture_pca_df[texture_pca_df["dataset"] == dataset_id].copy()
print(f"Behavior session: animal={animal}, neural_prefix={neural_prefix} -> dataset={dataset_id!r}, {len(beh):,} frames.")

beh_t = beh["time"].dt.total_seconds().to_numpy(dtype=float)  # beh["time"] is a Timedelta relative to session start
beh_outputs = beh[pc_cols].to_numpy(dtype=float)

# ==============================================================================
# 5. Trim to the neural/behavior time overlap, apply drift correction
# ==============================================================================
neural_start_raw, neural_end_raw = float(np.min(peak_times_sec)), float(np.max(peak_times_sec))
beh_start_raw, beh_end_raw = float(np.min(beh_t)), float(np.max(beh_t))
t_start = max(neural_start_raw, beh_start_raw)
t_end = min(neural_end_raw, beh_end_raw)

peak_keep = (peak_times_sec >= t_start) & (peak_times_sec < t_end)
peak_times_trim = peak_times_sec[peak_keep]
peak_rows_trim = peak_row_index[peak_keep]

# Correct each trimmed peak's depth via the motion estimate, then snap to the
# nearest channel -- that becomes its corrected channel_index.
localized_y_trim = localized_y[peak_keep]
raw_positions_trim = np.column_stack([channel_positions[peak_rows_trim, 0], localized_y_trim])
corrected_positions_trim = get_corrected_positions(peak_times_trim, raw_positions_trim, motion)
peak_rows_trim_corrected = snap_to_nearest_channel(corrected_positions_trim, channel_positions)
n_reassigned = int(np.sum(peak_rows_trim_corrected != peak_rows_trim))
print(f"Drift correction: {n_reassigned:,}/{len(peak_rows_trim):,} peaks reassigned to a different channel.")
peak_rows_trim = peak_rows_trim_corrected

# ==============================================================================
# 6. Bin peaks per channel, drop channels with zero spikes in the window
# ==============================================================================
spike_times_trim = np.asarray(
    [peak_times_trim[peak_rows_trim == ch] for ch in range(n_channels)], dtype=object
)
channel_spike_counts = np.array([len(a) for a in spike_times_trim], dtype=int)
active_channel_idx = np.flatnonzero(channel_spike_counts > 0)
spike_times_trim = spike_times_trim[active_channel_idx]
print(f"Channels: {active_channel_idx.size}/{n_channels} have at least one spike in window.")

# ==============================================================================
# 7. Bin spikes and behavior onto the same dt grid, keep only complete rows
# ==============================================================================
neural_data = bin_spikes(spike_times_trim, dt, t_start, t_end)

beh_keep = (beh_t >= t_start) & (beh_t < t_end)
y_binned, t_edges = bin_output_safe(beh_outputs[beh_keep], beh_t[beh_keep], dt, t_start, t_end)

valid_mask = np.all(np.isfinite(y_binned), axis=1)  # drop bins with no behavior frame
X = neural_data[valid_mask].astype("float32")
y = y_binned[valid_mask].astype("float32")
t_centers = ((t_edges[:-1] + t_edges[1:]) / 2.0)[valid_mask]
print(f"Binned to {X.shape[0]:,} samples x {active_channel_idx.size} channels at dt={dt}s (output dim = {len(pc_cols)}).")


# ==============================================================================
# 8. Reserve a chronological holdout, then build the lagged feature matrix
# ==============================================================================
bin_before = int(round(window_before / dt))
bin_after = int(round(window_after / dt))
print(f"Lag window: {bin_before + bin_current + bin_after} bins/sample x {active_channel_idx.size} channels "
      f"= {active_channel_idx.size * (bin_before + bin_current + bin_after):,} input features per sample.")

n_holdout = int(round(X.shape[0] * holdout_frac)) if holdout_frac > 0 else 0 #if choosng to hold out some parts of the recording
holdout_start_idx = X.shape[0] - n_holdout

X_search_raw = X[:holdout_start_idx]
y_search = y[:holdout_start_idx]
X_lagged_search = LagMat(bin_before=bin_before, bin_current=bin_current, bin_after=bin_after, flat=True).fit_transform(X_search_raw)

X_lagged_holdout, y_holdout, holdout_start_time = None, None, None
if n_holdout > 0:
    X_lagged_holdout = LagMat(bin_before=bin_before, bin_current=bin_current, bin_after=bin_after, flat=True).fit_transform(X[holdout_start_idx:])
    y_holdout = y[holdout_start_idx:]
    holdout_start_time = float(t_centers[holdout_start_idx])
    print(f"Holdout: reserved last {n_holdout:,}/{X.shape[0]:,} samples starting at t={holdout_start_time:.1f}s, excluded from the search.")

print(f"Ready for decoder training -- X_lagged_search: {X_lagged_search.shape}, y_search: {y_search.shape}")

# ==============================================================================
# 9. Config summary -- everything needed to reproduce this run
# ==============================================================================
config = {
    "paths": {
        "recording_path": recording_path,
        "peaks_path": peaks_path,
        "motion_path": str(motion_path),
        "localized_peaks_path": str(localized_peaks_path),
        "behavior_df_path": behavior_df_path,
        "metadata_df_path": metadata_df_path,
    },
    "session": {"animal": animal, "neural_prefix": neural_prefix, "dataset": dataset_id},
    "pc_cols": pc_cols,
    "dt": dt, "window_before": window_before, "bin_current": bin_current, "window_after": window_after,
    "bin_before": bin_before, "bin_after": bin_after,
    "n_splits": n_splits,
    "holdout_frac": holdout_frac, "holdout_n_samples": n_holdout, "holdout_start_time": holdout_start_time,
    "raw_ranges": {
        "neural": [neural_start_raw, neural_end_raw],
        "behavior": [beh_start_raw, beh_end_raw],
        "trimmed_overlap": [t_start, t_end],
    },
    "channels": {"total": n_channels, "active": int(active_channel_idx.size)},
    "shapes": {
        "X": list(X.shape), "y": list(y.shape),
        "X_lagged_search": list(X_lagged_search.shape),
        "X_lagged_holdout": list(X_lagged_holdout.shape) if X_lagged_holdout is not None else None,
    },
}
print(json.dumps(config, indent=2))

