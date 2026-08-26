# %%
"""
Peak Localization and Drift Correction Pipeline
================================================
THE active pipeline — use this one. Run via run_peak_localisation.sh.

WHAT IT DOES: detects spikes with mua_pipeline.run_mua_pipeline() (thresholding
+ matched filtering, see mua_pipeline.py), localizes each one in space with
SpikeInterface's 'monopolar_triangulation' method, estimates motion/drift from
those locations, corrects for it, then re-detects and re-localizes on the
corrected data as a sanity check.

**NOTE: the recommended workflow is to run this script for detection + localization only, 
then run the separate /motion/ pipeline for motion estimation and correction. This is to 
allow more control on fine tuning the parameters of each step. See PREFERRED SETTINGS section below.**

OUTPUTS, all written to <base_path>/<folder_name>/ (folder_name defaults to
"thresholding_peaks_<detect_threshold>", see --folder_name above):
  peaks0.npy, peaks1.npy, peaks2.npy, peaks_matched.npy, peaks_combined1.npy,
  peaks_combined2.npy   — from mua_pipeline.run_mua_pipeline(); two detection
                          branches that get merged:
    Branch A — plain threshold-crossing detection:
      peaks0   raw threshold crossings (every sample/channel crossing
               detect_threshold x noise; not deduplicated — one spike
               triggers many nearby crossings in time and across channels).
      peaks1   peaks0 after spatial_and_temporal_filter groups crossings
               close in time+channel into one representative peak per spike
               (the "same spike -> one entry" dedup step).
      peaks2   peaks1 after filter_peaks_by_spatial_support: keeps only
               peaks where a neighboring channel also shows a supporting
               crossing in the same window (rejects single-channel noise
               blips that don't spread across the probe like a real spike).
               Final output of the threshold-only branch, on its own.
    Branch B — matched-filter detection (separate, runs in parallel):
      peaks_matched   a different detection method: build a "prototype"
               spike waveform (learned from peaks2's waveforms, 
               slide it across the recording as a matched filter,
               keep points whose similarity score clears a threshold. Catches real
               spikes too small/atypical to cross the plain amplitude
               threshold, as long as they look like the prototype shape.
    Merge — the actual recommended output:
      peaks_combined1   peaks1 + peaks_matched concatenated, then run
               through the same grouping step again — mainly resolves
               duplicates *between* the two branches (same spike caught by
               both methods collapses to one entry).
      peaks_combined2   peaks_combined1 after the same spatial-support check
               as peaks2. This is the real final output — threshold +
               matched-filter merged. Everything downstream should use this
               (equivalently, the sorted copy peaks_combined2_sorted.npy
               saved below).
    Every array carries a matched_filter_score column (NaN for pure
    threshold-crossing peaks, a real value for matched-filter-derived ones,
    preserved through the merge since the winning representative peak at
    each step keeps its own score). Full derivation in mua_pipeline.py's
    run_mua_pipeline() docstring/body.
  peaks_combined2_sorted.npy   — peaks_combined2, sorted by sample_index —
                          this is the actual detected-spike output of this
                          pipeline (sample_index, channel_index, amplitude,
                          etc. per detected spike).
  localized_peaks_mt_raw.npy   — one (x, y, z, alpha) location per peak in
                          peaks_combined2_sorted.npy, from monopolar
                          triangulation — this is the localization output.
  plots/   — QC plots (raw/localization/drift scatter, method-comparison,
             etc.) — visual sanity checks, not consumed by anything else.
  motion_estimates/   — only populated if RUN_DECENTRALIZED_MOTION/
             RUN_DREDGE_MOTION are enabled (see PREFERRED SETTINGS below);
             not used by the recommended workflow, which does motion
             separately via motion/.

WHY MONOPOLAR TRIANGULATION, not boussard_localization.py's full IBL pipeline:
compared head-to-head in compare_localisation.py, the extra GPU-only denoising
step there wasn't meaningfully better for the time cost. Full reasoning in
peak_detect_localise/README.md.

USAGE: edit --base_path (and other CLI args, or the CONFIG section below) for
the session to process, then `sbatch run_peak_localisation.sh`. Key settings
to change per-session/animal:
  --base_path                      the preprocessed/ folder for this session
  --mua_detect_threshold            spike-detection threshold (units of noise)
  --support_radius/--support_fraction/--support_n_required   how strict the
                                     spatial-support artifact check is (see
                                     mua_pipeline.py)
  --time_start_s/--time_end_s       restrict to a sub-window (see below)
  --folder_name                     output subfolder name, so reruns with
                                     different settings don't collide

TIME WINDOW SUPPORT
--------------------
Set TIME_START_S / TIME_END_S below (or via --time_start_s / --time_end_s) to
restrict the ENTIRE pipeline — detection, localization, motion estimation,
motion correction — to a sub-window of the recording.

If left as None, the full recording is used (unchanged behaviour).

When a window is set:
  - The recording is sliced to [TIME_START_S, TIME_END_S) before anything
    else happens. All detection/localization/motion runs only see this slice.
  - Every peak's 'sample_index' is immediately shifted back to be relative to
    the ORIGINAL (unsliced) recording start, so it lines up with full-recording
    time axes in plots and is comparable across runs with different windows.
  - The results folder name gets a '_ts<start>_te<end>' suffix so different
    windows don't collide on disk.
"""

import json
import numpy as np
import os
import shutil
from pathlib import Path
import argparse
import spikeinterface.full as si
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
import matplotlib.pyplot as plt
from spikeinterface.sortingcomponents.motion import estimate_motion, interpolate_motion
from peak_detect_localise.mua_pipeline import run_mua_pipeline, load_data as load_data_mua

# %%
# =============================================================================
# PARSE COMMAND-LINE ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser(description='Peak Localization and Drift Correction Pipeline')
parser.add_argument('--mua_detect_threshold', type=float, default=None,
                    help='MUA detection threshold (overrides config value)')
parser.add_argument('--base_path', type=str, default=None)
parser.add_argument('--time_start_s', type=float, default=None,
                    help='Restrict pipeline to recording time >= this value (seconds). '
                         'Overrides TIME_START_S in config.')
parser.add_argument('--time_end_s', type=float, default=None,
                    help='Restrict pipeline to recording time < this value (seconds). '
                         'Overrides TIME_END_S in config.')
parser.add_argument('--support_radius', type=int, default=None,
                    help='Number of channels to check on each side of a peak channel '
                         'for the spatial-support check (overrides config value).')
parser.add_argument('--support_fraction', type=float, default=None,
                    help='Lowered-threshold fraction (of detect_threshold) a neighbor channel '
                         'must cross to count as supporting a peak (overrides config value).')
parser.add_argument('--support_n_required', type=int, default=None,
                    help='How many of the checked neighbor channels must pass the lowered '
                         'threshold for a peak to be kept (overrides config value).')
parser.add_argument('--matched_filter_detect_threshold', type=float, default=None,
                    help='Detection threshold (in units of noise of the matched-filter/convolution '
                         'score) used during matched-filtering detection. Lower = more permissive, '
                         'i.e. a peak does not need to look as similar to the learned prototype to '
                         'be detected (overrides config value).')
parser.add_argument('--matched_amplitude_threshold', type=float, default=None,
                    help='Post-hoc amplitude threshold (in units of noise) applied to peaks_matched '
                         'after matched-filtering detection. If not set, no extra amplitude '
                         'filtering is applied to peaks_matched (overrides config value).')
parser.add_argument('--folder_name', type=str, default=None,
                    help='Overrides the auto-generated output folder name '
                         '(default: thresholding_peaks_<detect_threshold>). Useful for keeping '
                         'a re-run from colliding with an existing results folder.')
args = parser.parse_args()

# %%
# =============================================================================
# CONFIGURATION SECTION - SET ALL PARAMETERS HERE
# =============================================================================

# --- Data Paths ---
BASE_PATH = args.base_path
PEAKS_FILENAME = "peaks_combined2.npy"  # In BASE_PATH/folder_name/

# --- Time window (None, None = full recording) ---
TIME_START_S = None   # e.g. 0.0
TIME_END_S   = None   # e.g. 3600.0

# Command-line overrides for time window
if args.time_start_s is not None:
    TIME_START_S = args.time_start_s
    print(f"Command-line override: time_start_s = {args.time_start_s}")
if args.time_end_s is not None:
    TIME_END_S = args.time_end_s
    print(f"Command-line override: time_end_s = {args.time_end_s}")

# --- Data Source: Choose one mode ---
USE_TEMP_WH = False  # If True: load from temp_wh.dat (same directory as BASE_PATH)
                     # If False: load from preprocessed folder (BASE_PATH)
# When USE_TEMP_WH=True, BIN_PATH will be automatically set to BASE_PATH/'temp_wh.dat'
# When USE_TEMP_WH=False, BIN_PATH will be BASE_PATH

# --- mua_pipeline Settings (for spike detection) ---
RUN_MUA_PIPELINE = True  # Set to True to run mua_pipeline on raw recording

# --- Output Paths ---
PLOTS_DIR = None   # Set below, after FOLDER_NAME is determined
MOTION_DIR = None  # Set below, after FOLDER_NAME is determined

# --- Processing Flags ---
LOAD_COMPUTED_DATA = True  # Set to False to recompute everything

# =============================================================================
# RECOMMENDED WORKFLOW / PREFERRED SETTINGS
# =============================================================================
# In practice it is recommended to run this script for DETECTION + LOCALIZATION ONLY
# (defaults below reflect that), then slope_artifact_filter_analysis.ipynb, then
# motion/ for estimation+correction. Full 3-step workflow and why:
# peak_detect_localise/README.md.

# --- Localization Methods to Run ---
RUN_MONOPOLAR_TRIANGULATION = True   # preferred method — keep True
RUN_CENTER_OF_MASS = False           # not used, a much simpler method but much less accurate

# --- Motion Estimation Methods to Run ---
# it is preferred to, instead of running motion estimation here, run the separate /motion/ 
# pipeline (step 3 above) which allows chunking on long data (much faster), and more control over parameters. 
# 
# DREDge is the preferred motion estimation method (also what
# motion/run_dredge_chunked.py uses for the real, chunked estimation) —
# "decentralized" is not preferred. Note: with APPLY_MOTION_CORRECTION=False
# below, this only computes an estimate for the drift-overlay plots; it is
# NOT what's actually used to correct anything for long recordings — that's
# the separate /motion/ pipeline (step 3 above).
RUN_DECENTRALIZED_MOTION = False     # not preferred, much simpler and less accurate
RUN_DREDGE_MOTION = False            # preferred motion estimation method if have to 
                                     # use this script; remember that the best practice 
                                     # is to run the separate /motion/ pipeline for 
                                     # motion estimation and correction.

# --- Motion Correction (Interpolation) ---
# Keep this False. This applies motion correction by interpolating the
# recording itself (si.interpolate_motion) — this is the legacy path and
# isn't well tested/trusted; not what's actually used. The preferred approach
# is to correct motion using motion/correct_motion.py, which instead subtracts the estimated motion
# directly from peak depths / shifts channel assignments — no recording
# interpolation. See motion/correct_motion.py's header for why.
APPLY_MOTION_CORRECTION = False
CORRECTION_METHOD = "dredge"  # "decentralized" or "dredge" — only matters if APPLY_MOTION_CORRECTION=True

# --- Re-detect Peaks After Correction (using mua_pipeline) ---
# Keep this False. Legacy section, doesn't test well — only runs at all if
# APPLY_MOTION_CORRECTION=True anyway; recording gets messed up after motion interpolation, 
# so re-detecting peaks on it is really not recommended. The preferred approach is to
# just correct motion on the existing peaks and locations (see motion/correct_motion.py). 
REDETECT_PEAKS_AFTER_CORRECTION = False

# --- Rolling Noise (optional, time-varying noise levels) ---
# If provided, uses rolling_noise_levels.npz from rolling_snr.py/ instead of si.get_noise_levels();
# it is not recommended to use rolling noise levels as they do not appear better than using a 
# single noise level for the whole recording, and they are slower to compute.

# but if need to use, need to run rolling_snr.py first to generate the rolling noise levels, then set ROLLING_NOISE_DIR to the path of the rolling_snr output folder.
# Example: ROLLING_NOISE_DIR = os.path.join(BASE_PATH, "rolling_snr")
# ROLLING_NOISE_DIR = os.path.join(BASE_PATH, "rolling_snr")  # Set to path/to/rolling_snr/ to use rolling noise levels
# ROLLING_NOISE_DIR = r'/ptmp/gongt/GLC7189/GLC7189_g_g1/preprocessed/rolling_snr'
ROLLING_NOISE_DIR = None

# --- mua_pipeline Settings (for re-detection after motion correction) ---
# Passed straight through to mua_pipeline.run_mua_pipeline() — see that
# function's own docstring in mua_pipeline.py for the authoritative version
# of every explanation below, and for the full list/meaning of what it
# returns (peaks_combined2, peaks_combined1, peaks_matched, peaks0, peaks1,
# peaks2 — peaks_combined2 is the one downstream steps use).
MUA_SETTINGS = {
    "detect_threshold":   3.3,   # threshold-crossing detection cutoff, in units of noise (sigma) on each channel
    "exclude_sweep_ms":   0.3,   # temporal exclusion window (ms) around a detected peak, so the same spike isn't counted twice
    "spatial_radius":     10,    # channels to group together (on each side) when merging threshold-crossings from the same spike across nearby channels
    "chunk_size_s":       60.0,  # seconds of recording processed per chunk (memory/parallelism knob, doesn't change results)
    "n_jobs":             6,     # parallel workers for the main thresholding/filtering pass
    "proto_n_jobs":       6,     # parallel workers for building the matched-filter prototype waveform
    "load_computed_data": True,   # reload saved peaks if present
    "rolling_noise_dir":  ROLLING_NOISE_DIR,  # time-varying noise levels (optional)

    # --- Spatial support filter (neighbor-channel check) ---
    # A peak is kept if at least support_n_required of the support_radius
    # channels on each side of the peak channel also cross a lowered
    # threshold of support_fraction * detect_threshold within the same window.
    "support_radius":     2,     # how many channels to check on each side; if 2, 5 channels in total are checked
    "support_fraction":   0.7,   # lowered-threshold fraction of detect_threshold
    "support_n_required": 2,     # how many of the checked neighbors must pass

    # --- Matched-filter detection (peaks_matched) ---
    # matched_filter_detect_threshold controls how similar a candidate waveform
    # must be to the learned prototype (in units of noise of the matched-filter
    # convolution score) to be detected at all. None -> defaults to
    # detect_threshold - 0.2 (same as spikeinterface default).
    "matched_filter_detect_threshold": None,
    # matched_amplitude_threshold is an additional, separate post-hoc filter:
    # peaks_matched are kept only if their raw amplitude exceeds
    # matched_amplitude_threshold * noise level on their channel.
    # None -> no extra amplitude filtering (default).
    "matched_amplitude_threshold": None,

    # --- Matched-filter detection: other knobs (rarely need changing) ---
    "matched_filter_peak_sign":        "neg",  # which deflection direction counts as a spike: 'neg', 'pos', or 'both'
    "matched_filter_exclude_sweep_ms": 0.1,    # temporal exclusion window (ms) for the matched-filter pass specifically (separate from exclude_sweep_ms above)
    "matched_filter_radius_um":        50,     # spatial radius (µm, physical distance not channel count) for suppressing duplicate matched-filter detections nearby
    "matched_filter_n_jobs":           8,      # parallel workers for the matched-filter detection job
    "matched_filter_chunk_duration":   "1s",   # chunk duration (SpikeInterface job_kwargs format) for the matched-filter detection job

    # --- Prototype/waveform extraction (feeds matched filtering) ---
    "proto_ms_before":     1.0,   # ms of waveform kept before the peak sample when building the matched-filter prototype
    "proto_ms_after":      1.0,   # ms of waveform kept after the peak sample when building the matched-filter prototype
    "proto_chunk_duration": "1s", # chunk duration (SpikeInterface job_kwargs format) for prototype/waveform extraction
}

# Override detect_threshold from command-line if provided
if args.mua_detect_threshold is not None:
    MUA_SETTINGS["detect_threshold"] = args.mua_detect_threshold
    print(f"Command-line override: mua_detect_threshold = {args.mua_detect_threshold}")

# Override spatial-support and matched-filter settings from command-line if provided
if args.support_radius is not None:
    MUA_SETTINGS["support_radius"] = args.support_radius
    print(f"Command-line override: support_radius = {args.support_radius}")
if args.support_fraction is not None:
    MUA_SETTINGS["support_fraction"] = args.support_fraction
    print(f"Command-line override: support_fraction = {args.support_fraction}")
if args.support_n_required is not None:
    MUA_SETTINGS["support_n_required"] = args.support_n_required
    print(f"Command-line override: support_n_required = {args.support_n_required}")
if args.matched_filter_detect_threshold is not None:
    MUA_SETTINGS["matched_filter_detect_threshold"] = args.matched_filter_detect_threshold
    print(f"Command-line override: matched_filter_detect_threshold = {args.matched_filter_detect_threshold}")
if args.matched_amplitude_threshold is not None:
    MUA_SETTINGS["matched_amplitude_threshold"] = args.matched_amplitude_threshold
    print(f"Command-line override: matched_amplitude_threshold = {args.matched_amplitude_threshold}")

FOLDER_NAME = f"thresholding_peaks_{MUA_SETTINGS['detect_threshold']}"
if ROLLING_NOISE_DIR is not None:
    FOLDER_NAME += "_rolling"
# Encode time window in folder name so different windows don't collide on disk
if TIME_START_S is not None or TIME_END_S is not None:
    _ts_tag = f"{TIME_START_S:.0f}" if TIME_START_S is not None else "start"
    _te_tag = f"{TIME_END_S:.0f}"   if TIME_END_S   is not None else "end"
    FOLDER_NAME += f"_ts{_ts_tag}_te{_te_tag}"
if args.folder_name is not None:
    FOLDER_NAME = args.folder_name
    print(f"Command-line override: folder_name = {args.folder_name}")
if BASE_PATH is not None:
    PLOTS_DIR = os.path.join(BASE_PATH, FOLDER_NAME, "plots")
    MOTION_DIR = os.path.join(BASE_PATH, FOLDER_NAME, "motion_estimates")
    print(f"Peaks will be saved to folder: {FOLDER_NAME}")

# --- Plotting Parameters ---
SCATTER_WINDOW_S = 720  # seconds: each scatter shows first N s and last N s

WAVEFORM_PLOT_TIME = {
    't_start': 500,    # Start time for waveform plot (seconds)
    't_end': 500.05,   # End time for waveform plot (seconds)
}

# Channel range for waveform plot
WAVEFORM_PLOT_CHANNELS = {
    'chan_start': 100,
    'chan_end': 200,
}

# --- Localization Parameters ---
LOCALIZATION_PARAMS = {
    'monopolar_triangulation': {
        'radius_um': 100,
        'max_distance_um': 1000,
        'enforce_decrease': False,
        'feature': 'ptp',  # 'ptp', 'energy', or 'peak_voltage'
    },
    'center_of_mass': {
        'radius_um': 100,
        'feature': 'ptp',
    }
}

# --- Motion Estimation Parameters ---
MOTION_PARAMS = {
    'decentralized': {
        'conv_engine': 'torch',
        'torch_device': 'cuda',  # 'cpu' or 'cuda'
        'batch_size': 10,
        'time_horizon_s': 5 * 60,
    },
    'dredge_ap': {
        'time_horizon_s': 5 * 60,
    }
}

# --- Interpolation Parameters ---
INTERPOLATION_PARAMS = {
    'border_mode': 'remove_channels',  # or 'force_extrapolate', 'force_zeros'
    'spatial_interpolation_method': 'kriging',
    'sigma_um': 30.0,
}

# --- Job Parameters ---
JOB_KWARGS = {
    'chunk_duration': '1s',
    'n_jobs': 10,
    'progress_bar': True
}

# --- Output Settings ---
SAVE_N_JOBS = 8  # For saving corrected recordings
PLOT_DPI = 300  # DPI for saved plots

# =============================================================================
# END CONFIGURATION
# =============================================================================

# %%
# =============================================================================
# COMPUTE DATA PATHS BASED ON CONFIGURATION
# =============================================================================

# Determine which file to load based on USE_TEMP_WH flag (without printing - avoids worker process spam)
# Guard against None BASE_PATH (worker processes re-import this module without CLI args)
if BASE_PATH is not None:
    if USE_TEMP_WH:
        BIN_PATH = os.path.join(BASE_PATH, 'temp_wh.dat')
        LOAD_SOURCE = "temp_wh.dat"
    else:
        BIN_PATH = BASE_PATH
        LOAD_SOURCE = "preprocessed folder"
else:
    BIN_PATH = None
    LOAD_SOURCE = None

# %%
# Create output directories (only if main process)
if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MOTION_DIR, exist_ok=True)

# %%
# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_or_compute(save_path, compute_fn, load_fn=None, compute_msg="Computing", save_msg="Saving"):
    """Generic load-or-compute pattern."""
    if LOAD_COMPUTED_DATA and os.path.exists(save_path):
        print(f"Loading precomputed data from: {save_path}")
        if load_fn is None:
            data = np.load(save_path)
        else:
            data = load_fn(save_path)
        return data, True
    else:
        print(f"{compute_msg}...")
        data = compute_fn()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if isinstance(data, np.ndarray):
            np.save(save_path, data)
        else:
            data.save(save_path)  # For SpikeInterface objects like motion
        print(f"{save_msg} to: {save_path}")
        return data, False

def plot_spikes_scatter(peaks, peak_locations, sampling_frequency, ax=None, 
                       t_start=None, t_end=None, title="Peak Depths Over Time", **scatter_kws):
    """Common scatter plot for peaks over time (used by multiple plot functions).

    NOTE: peaks['sample_index'] is expected to already be relative to the
    ORIGINAL (unsliced) recording, so t_start/t_end here are in original-
    recording time, matching SCATTER_WINDOW_S windows etc.
    """
    if ax is None:
        ax = plt.gca()
    
    # Apply time mask if specified
    if t_start is not None or t_end is not None:
        times = peaks['sample_index'] / sampling_frequency
        mask = np.ones(len(peaks), dtype=bool)
        if t_start is not None:
            mask &= (times >= t_start)
        if t_end is not None:
            mask &= (times <= t_end)
        peaks_plot = peaks[mask]
        locations_plot = peak_locations[mask]
    else:
        peaks_plot = peaks
        locations_plot = peak_locations
    
    scatter_defaults = {'s': 1, 'color': 'k', 'alpha': 0.01}
    scatter_defaults.update(scatter_kws)
    
    ax.scatter(peaks_plot['sample_index'] / sampling_frequency, 
               locations_plot['y'], 
               **scatter_defaults)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Depth (µm)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.2)
    return ax

def load_data(file_path):
    """Load SpikeInterface recording from path.
    
    Supports:
    - Preprocessed folders (with binary.json)
    - temp_wh.dat files (drift-corrected binary with binary.json in same directory)
    - SpikeGLX .bin files
    """
    return load_data_mua(file_path, n_jobs=JOB_KWARGS['n_jobs'])


def slice_recording_to_window(rec, time_start_s, time_end_s):
    """Slice a recording to [time_start_s, time_end_s) in seconds.

    Returns (rec_sliced, start_frame). start_frame is the frame offset of the
    slice within the ORIGINAL recording — add it back onto any sample_index
    computed on the sliced recording to recover original-recording sample
    indices.

    If both time_start_s and time_end_s are None, returns (rec, 0) unchanged.
    """
    if time_start_s is None and time_end_s is None:
        return rec, 0

    fs = rec.sampling_frequency
    total_duration = rec.get_total_duration()

    start_frame = int(round((time_start_s or 0.0) * fs))
    end_frame = (int(round(time_end_s * fs))
                if time_end_s is not None
                else rec.get_num_frames())

    start_frame = max(0, min(start_frame, rec.get_num_frames()))
    end_frame = max(start_frame, min(end_frame, rec.get_num_frames()))

    print(f"  Slicing recording to frames [{start_frame}, {end_frame})  "
          f"=  time [{start_frame/fs:.2f}s, {end_frame/fs:.2f}s)  "
          f"out of {total_duration:.2f}s total")

    rec_sliced = rec.frame_slice(start_frame=start_frame, end_frame=end_frame)
    return rec_sliced, start_frame


def shift_peaks_to_original_time(peaks, start_frame):
    """Shift peaks['sample_index'] by start_frame so it is relative to the
    ORIGINAL (unsliced) recording rather than the slice.

    No-op (returns peaks unchanged) if start_frame == 0.
    Operates on a copy; does not mutate the input array.
    """
    if start_frame == 0:
        return peaks
    peaks = peaks.copy()
    peaks['sample_index'] = peaks['sample_index'] + start_frame
    return peaks


def materialize_sliced_recording_for_mua(rec_sliced, base_path, folder_name, n_jobs):
    """run_mua_pipeline takes a bin_path (string) and reloads from disk
    internally, so an in-memory sliced recording can't be passed directly.
    When a time window is active, save the slice to a temp folder under
    base_path/folder_name/_sliced_recording so run_mua_pipeline can load it,
    and return that path as the bin_path to use.
    """
    sliced_path = os.path.join(base_path, folder_name, "_sliced_recording")
    if os.path.exists(sliced_path):
        print(f"  Using existing sliced recording at: {sliced_path}")
        return sliced_path
    print(f"  Saving sliced recording to: {sliced_path}")
    os.makedirs(os.path.dirname(sliced_path), exist_ok=True)
    rec_sliced.save_to_folder(sliced_path, overwrite=False, n_jobs=n_jobs)
    print("  Sliced recording saved.")
    return sliced_path


def save_scatter_pair(peaks, locations, sampling_frequency, total_duration,
                      title_base, save_prefix):
    """Save two drift scatter plots: first and last SCATTER_WINDOW_S seconds."""
    windows = [
        ("start", 0.0,                                       min(SCATTER_WINDOW_S, total_duration)),
        ("end",   max(0.0, total_duration - SCATTER_WINDOW_S), total_duration),
    ]
    for label, t_start, t_end in windows:
        fig, ax = plt.subplots(figsize=(15, 8))
        plot_spikes_scatter(peaks, locations, sampling_frequency, ax=ax,
                            t_start=t_start, t_end=t_end,
                            title=f"{title_base} ({t_start:.0f}\u2013{t_end:.0f}s)")
        path = f"{save_prefix}_{label}.png"
        plt.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"  Plot saved to: {path}")
        plt.close()


if __name__ == "__main__":
    # %%
    # =============================================================================
    # PRINT CONFIGURATION
    # =============================================================================
    
    print("="*80)
    print("CONFIGURATION")
    print("="*80)
    
    if USE_TEMP_WH:
        print(f"Using temp_wh.dat mode")
        print(f"  Recording: {BIN_PATH}")
    else:
        print(f"Using preprocessed folder mode")
        print(f"  Recording: {BIN_PATH}")
    print(f"All outputs will be saved to: {BASE_PATH}\n")
    if TIME_START_S is not None or TIME_END_S is not None:
        print(f"TIME WINDOW ACTIVE: [{TIME_START_S}, {TIME_END_S})  (None = recording start/end)\n")
    else:
        print("TIME WINDOW: full recording\n")
    
    # %%
    # =============================================================================
    # LOAD BASE RECORDING AND PEAKS
    # =============================================================================
    
    print("="*80)
    print("LOADING BASE DATA")
    print("="*80)
    
    # Load base recording from BIN_PATH (either preprocessed folder or temp_wh.dat)
    print(f"\nLoading recording from: {BIN_PATH}")
    rec_full = load_data(BIN_PATH)
    print(f"Recording loaded: {rec_full}")
    print(f"  Duration: {rec_full.get_total_duration():.2f} s")
    print(f"  Sampling rate: {rec_full.sampling_frequency} Hz")
    print(f"  Channels: {rec_full.get_num_channels()}")

    # --- Apply time window slice (if set) ---
    # rec_original is what everything downstream uses. TIME_WINDOW_START_FRAME
    # is the frame offset of this slice within the original, full recording —
    # add it back onto any sample_index computed from rec_original to recover
    # original-recording-relative sample indices.
    rec_original, TIME_WINDOW_START_FRAME = slice_recording_to_window(
        rec_full, TIME_START_S, TIME_END_S
    )
    if TIME_WINDOW_START_FRAME != 0 or TIME_START_S is not None or TIME_END_S is not None:
        print(f"  Sliced recording duration: {rec_original.get_total_duration():.2f} s")
        print(f"  Time window start frame (offset into original recording): {TIME_WINDOW_START_FRAME}")
    
    # Load or generate peaks
    if RUN_MUA_PIPELINE:
        print(f"\nRunning mua_pipeline on original recording to detect peaks...")
        if TIME_WINDOW_START_FRAME != 0 or TIME_START_S is not None or TIME_END_S is not None:
            # run_mua_pipeline takes a bin_path and reloads from disk, so we
            # need to materialize the sliced recording to disk first.
            _mua_bin_path = materialize_sliced_recording_for_mua(
                rec_original, BASE_PATH, FOLDER_NAME, n_jobs=MUA_SETTINGS.get('n_jobs', 6)
            )
        else:
            _mua_bin_path = BIN_PATH
        _, peaks_original, *_ = run_mua_pipeline(bin_path=_mua_bin_path, folder_name=FOLDER_NAME, **MUA_SETTINGS)
        # Shift sample_index back to be relative to the ORIGINAL (unsliced) recording
        peaks_original = shift_peaks_to_original_time(peaks_original, TIME_WINDOW_START_FRAME)
    else:
        peaks_path = Path(BASE_PATH) / FOLDER_NAME / PEAKS_FILENAME
        print(f"\nLoading cached peaks from: {peaks_path}")
        peaks_original = np.load(peaks_path)
        print(f"  Loaded {len(peaks_original):,} peaks")
        # NOTE: cached peaks are assumed to already be in original-recording
        # time (i.e. were saved after the shift above on a previous run).
    _sorted_path = Path(BASE_PATH) / FOLDER_NAME / (Path(PEAKS_FILENAME).stem + "_sorted.npy")
    if not _sorted_path.exists():
        print(f"  Sorting peaks and saving to {_sorted_path} ...")
        np.save(_sorted_path, np.sort(peaks_original, order='sample_index'))
        print(f"  Saved.")
    else:
        print(f"  Loading pre-sorted peaks from {_sorted_path}")
    peaks_original = np.load(_sorted_path)
    
    # %%
    # =============================================================================
    # PEAK LOCALIZATION ON ORIGINAL DATA
    # =============================================================================
    
    print("\n" + "="*80)
    print("PEAK LOCALIZATION (ORIGINAL DATA)")
    print("="*80)
    
    # --- Monopolar Triangulation ---
    if RUN_MONOPOLAR_TRIANGULATION:
        save_path = os.path.join(BASE_PATH, FOLDER_NAME, "localized_peaks_mt_raw.npy")

        # localize_peaks needs peaks['sample_index'] relative to rec_original
        # (the sliced recording), but peaks_original is in ORIGINAL-recording
        # time. Shift back to slice-relative time for this call only.
        _peaks_for_localization = peaks_original.copy()
        _peaks_for_localization['sample_index'] = (
            _peaks_for_localization['sample_index'] - TIME_WINDOW_START_FRAME
        )

        print("\nComputing Monopolar Triangulation localization...")
        peak_locations_mt_original, was_loaded = load_or_compute(
            save_path,
            compute_fn=lambda: localize_peaks(
                recording=rec_original,
                peaks=_peaks_for_localization,
                method='monopolar_triangulation',
                **LOCALIZATION_PARAMS['monopolar_triangulation'],
                **JOB_KWARGS
            ),
            compute_msg="Computing Monopolar Triangulation localization",
            save_msg="Saved to"
        )
        
        print(f"  Localized peaks: {len(peak_locations_mt_original)}")
        print(f"  Depth range: {peak_locations_mt_original['y'].min():.1f} - {peak_locations_mt_original['y'].max():.1f} µm")
    else:
        # MT computation disabled — try loading a previously saved file
        _mt_candidates = [
            os.path.join(BASE_PATH, FOLDER_NAME, "localized_peaks_mt_raw.npy"),
            os.path.join(BASE_PATH, FOLDER_NAME, "localized_peaks_mt.npy"),
        ]
        _mt_loaded = next((p for p in _mt_candidates if os.path.exists(p)), None)
        if _mt_loaded:
            print(f"\nSkipping Monopolar Triangulation (disabled in config)")
            print(f"  Loading existing MT localization from: {_mt_loaded}")
            peak_locations_mt_original = np.load(_mt_loaded)
            if len(peak_locations_mt_original) != len(peaks_original):
                print(f"  WARNING: MT file has {len(peak_locations_mt_original):,} entries "
                      f"but peaks has {len(peaks_original):,} — ignoring.")
                peak_locations_mt_original = None
            else:
                print(f"  Loaded {len(peak_locations_mt_original):,} MT locations  "
                      f"depth {peak_locations_mt_original['y'].min():.1f}–"
                      f"{peak_locations_mt_original['y'].max():.1f} µm")
        else:
            peak_locations_mt_original = None
            print("\nSkipping Monopolar Triangulation (disabled in config)")

    # --- Center of Mass ---
    if RUN_CENTER_OF_MASS:
        save_path = os.path.join(BASE_PATH, FOLDER_NAME, "localized_peaks_cm_raw.npy")

        _peaks_for_localization = peaks_original.copy()
        _peaks_for_localization['sample_index'] = (
            _peaks_for_localization['sample_index'] - TIME_WINDOW_START_FRAME
        )

        print("\nComputing Center of Mass localization...")
        peak_locations_cm_original, was_loaded = load_or_compute(
            save_path,
            compute_fn=lambda: localize_peaks(
                recording=rec_original,
                peaks=_peaks_for_localization,
                method='center_of_mass',
                **LOCALIZATION_PARAMS['center_of_mass'],
                **JOB_KWARGS
            ),
            compute_msg="Computing Center of Mass localization",
            save_msg="Saved to"
        )
        
        print(f"  Localized peaks: {len(peak_locations_cm_original)}")
        print(f"  Depth range: {peak_locations_cm_original['y'].min():.1f} - {peak_locations_cm_original['y'].max():.1f} µm")
    else:
        peak_locations_cm_original = None
        print("\nSkipping Center of Mass (disabled in config)")
    
    # %%
    # =============================================================================
    # VISUALIZATION: LOCALIZATION COMPARISON
    # =============================================================================
    
    def plot_localized_peaks_on_depth(
        rec,
        peaks,
        locations_sets,
        t_start,
        t_end,
        chan_start,
        chan_end,
        labels=None,
        save_path=None,
        time_offset_frames=0
    ):
        """Plot localized peaks overlaid on raw traces with depth labels.

        peaks['sample_index'] is assumed to be relative to the ORIGINAL
        recording. time_offset_frames is subtracted to recover the frame
        index within `rec` (which may itself be a slice), and t_start/t_end
        are interpreted in ORIGINAL-recording time for consistency with the
        rest of the pipeline's plots.
        """
        
        COLORS = [
            '#e6194b', '#4363d8', '#3cb44b', '#f58231', '#911eb4',
            '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
        ]
    
        if labels is None:
            labels = [f"Method {i+1}" for i in range(len(locations_sets))]
    
        fs = rec.sampling_frequency
        chans = rec.get_channel_ids()
        chan_ids_to_plot = chans[chan_start:chan_end]
        
        # Get physical micron depths for the Y-Axis
        chan_locs = rec.get_channel_locations(channel_ids=chan_ids_to_plot)
        y_depths = chan_locs[:, 1] 
    
        # Extract Data — t_start/t_end are in ORIGINAL time; convert to frames
        # within `rec` (subtract the slice offset).
        start_frame_orig = int(t_start * fs)
        end_frame_orig = int(t_end * fs)
        start_frame = start_frame_orig - time_offset_frames
        end_frame = end_frame_orig - time_offset_frames
        traces = rec.get_traces(start_frame=start_frame, end_frame=end_frame, channel_ids=chan_ids_to_plot)
        time_axis = np.linspace(t_start, t_end, traces.shape[0])
    
        # Spacing & figsize
        num_chans = len(chan_ids_to_plot)
        duration = t_end - t_start
        max_amp = np.max(np.abs(traces))
        spacing = max_amp * 1.5 if max_amp > 0 else 100
        offsets = np.arange(num_chans) * -spacing 
    
        plt.figure(figsize=(duration * 180, num_chans * 0.4))
    
        # Plot traces
        for i in range(traces.shape[1]):
            plt.plot(time_axis, traces[:, i] + offsets[i], color='black', lw=0.5)
    
        # Map Localization Dots onto the Stacked Plot
        # peaks['sample_index'] is in ORIGINAL time, matching t_start/t_end here.
        p_times = peaks['sample_index'] / fs
        time_mask = (p_times >= t_start) & (p_times <= t_end)
        chan_mask = (peaks['channel_index'] >= chan_start) & (peaks['channel_index'] < chan_end)
        subset_idx = np.where(time_mask & chan_mask)[0]
    
        for set_idx, locations in enumerate(locations_sets):
            color = COLORS[set_idx % len(COLORS)]
            label = labels[set_idx]
            y_shift = set_idx * (spacing * 0.15)
            
            if len(subset_idx) > 0:
                p_chan_local = peaks['channel_index'][subset_idx] - chan_start
                p_y_est = locations['y'][subset_idx]
    
                p_pos = np.zeros(len(subset_idx))
                for k in range(len(subset_idx)):
                    c_idx = p_chan_local[k]
                    local_actual_y = y_depths[c_idx]
                    local_est_y = p_y_est[k]
    
                    p_pos[k] = offsets[c_idx] - (local_est_y - local_actual_y) * (spacing / 20.0) - y_shift
    
                plt.scatter(peaks['sample_index'][subset_idx] / fs, 
                            p_pos, color=color, s=10, zorder=3 + set_idx, label=label)
    
        # Y-axis labels = depth
        depth_labels = [f"{d:.0f}µm" for d in y_depths]
        plt.yticks(offsets, depth_labels)
        
        plt.xlim(t_start, t_end)
        plt.xlabel("Time (s)")
        plt.ylabel("Probe Depth (µm)")
        plt.title(f"Localized Spikes on Depth-Labeled Traces ({t_start:.2f}-{t_end:.2f}s)")
        plt.grid(axis='x', alpha=0.1)
        plt.legend(loc='upper right', markerscale=2)
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
            print(f"  Plot saved to: {save_path}")
        plt.close()
    
    # %%
    # Plot comparison if both methods were run
    if RUN_MONOPOLAR_TRIANGULATION and RUN_CENTER_OF_MASS:
        print("\nPlotting localization method comparison on traces...")
        save_path = os.path.join(PLOTS_DIR, "01_waveform_localization_comparison_original.png")
        plot_localized_peaks_on_depth(
            rec_original, 
            peaks=peaks_original,
            locations_sets=[peak_locations_cm_original, peak_locations_mt_original],
            labels=['Center of Mass', 'Monopolar Triangulation'],
            t_start=WAVEFORM_PLOT_TIME['t_start'],
            t_end=WAVEFORM_PLOT_TIME['t_end'],
            chan_start=WAVEFORM_PLOT_CHANNELS['chan_start'],
            chan_end=WAVEFORM_PLOT_CHANNELS['chan_end'],
            save_path=save_path,
            time_offset_frames=TIME_WINDOW_START_FRAME
        )
    
    # %%
    # Method comparison scatter plot
    if RUN_MONOPOLAR_TRIANGULATION and RUN_CENTER_OF_MASS:
        print("\nPlotting method comparison scatter...")
        plt.figure(figsize=(8, 8))
        plt.scatter(peak_locations_cm_original['y'], peak_locations_mt_original['y'], 
                    s=1, alpha=0.1, c='black')
        
        # Add identity line
        min_val = min(peak_locations_cm_original['y'].min(), peak_locations_mt_original['y'].min())
        max_val = max(peak_locations_cm_original['y'].max(), peak_locations_mt_original['y'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Identity', linewidth=2)
        
        plt.xlabel('Center of Mass Depth (µm)', fontsize=12)
        plt.ylabel('Monopolar Triangulation Depth (µm)', fontsize=12)
        plt.title('Localization Method Comparison (Original Data)', fontsize=14)
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        
        save_path = os.path.join(PLOTS_DIR, "02_method_comparison_scatter.png")
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"  Plot saved to: {save_path}")
        plt.close()
    
    # %%
    # =============================================================================
    # DRIFT VISUALIZATION
    # =============================================================================
    
    # Plot drift for each localization method
    if RUN_MONOPOLAR_TRIANGULATION:
        print("\nPlotting drift scatter (Monopolar Triangulation - Original)...")
        save_scatter_pair(
            peaks_original, peak_locations_mt_original,
            rec_original.sampling_frequency, rec_full.get_total_duration(),
            title_base="Monopolar Triangulation: Peak Depths Over Time (Original)",
            save_prefix=os.path.join(PLOTS_DIR, "03_drift_scatter_mt_original"),
        )
    
    if RUN_CENTER_OF_MASS:
        print("\nPlotting drift scatter (Center of Mass - Original)...")
        save_scatter_pair(
            peaks_original, peak_locations_cm_original,
            rec_original.sampling_frequency, rec_full.get_total_duration(),
            title_base="Center of Mass: Peak Depths Over Time (Original)",
            save_prefix=os.path.join(PLOTS_DIR, "04_drift_scatter_cm_original"),
        )
    
    # %%
    # =============================================================================
    # MOTION ESTIMATION
    # =============================================================================
    
    print("\n" + "="*80)
    print("MOTION ESTIMATION")
    print("="*80)

    # estimate_motion needs peaks relative to rec_original (the slice), so
    # build a slice-relative copy for this section.
    _peaks_for_motion = peaks_original.copy()
    _peaks_for_motion['sample_index'] = (
        _peaks_for_motion['sample_index'] - TIME_WINDOW_START_FRAME
    )
    
    # --- Decentralized Motion (requires MT peak locations) ---
    if RUN_DECENTRALIZED_MOTION:
        if peak_locations_mt_original is None:
            print("\nSkipping Decentralized motion (requires Monopolar Triangulation locations).")
            motion_decentralized = None
        else:
            save_path = os.path.join(MOTION_DIR, "motion_decentralized")
            print("\nEstimating motion with Decentralized method...")
            motion_decentralized, was_loaded = load_or_compute(
                save_path,
                compute_fn=lambda: estimate_motion(
                    recording=rec_original,
                    peaks=_peaks_for_motion,
                    peak_locations=peak_locations_mt_original,
                    method="decentralized",
                    progress_bar=True,
                    **MOTION_PARAMS['decentralized']
                ),
                load_fn=si.Motion.load,
                compute_msg="Estimating motion with Decentralized method",
                save_msg="✓ Decentralized motion estimation complete. Saved to"
            )
            params_record = {"method": "decentralized", **MOTION_PARAMS['decentralized']}
            with open(save_path + "_params.json", "w") as _f:
                json.dump(params_record, _f, indent=2)
    else:
        motion_decentralized = None
        print("\nSkipping Decentralized motion (disabled in config)")

    # --- DREDge Motion (uses MT locations — loaded from file if available) ---
    if RUN_DREDGE_MOTION:
        if peak_locations_mt_original is None:
            print("\nSkipping DREDge motion (no MT peak locations available; "
                  "ensure localized_peaks_mt.npy or localized_peaks_mt_raw.npy exists "
                  "in the peaks folder, or enable RUN_MONOPOLAR_TRIANGULATION).")
            motion_dredge = None
        else:
            save_path = os.path.join(MOTION_DIR, "motion_dredge")
            print("\nEstimating motion with DREDge method...")
            motion_dredge, was_loaded = load_or_compute(
                save_path,
                compute_fn=lambda: estimate_motion(
                    recording=rec_original,
                    peaks=_peaks_for_motion,
                    peak_locations=peak_locations_mt_original,
                    method="dredge_ap",
                    progress_bar=True,
                    **MOTION_PARAMS['dredge_ap']
                ),
                load_fn=si.Motion.load,
                compute_msg="Estimating motion with DREDge method",
                save_msg="✓ DREDge motion estimation complete. Saved to"
            )
            params_record = {"method": "dredge_ap", **MOTION_PARAMS['dredge_ap']}
            with open(save_path + "_params.json", "w") as _f:
                json.dump(params_record, _f, indent=2)
    else:
        motion_dredge = None
        print("\nSkipping DREDge motion (disabled in config)")

    
    # %%
    # =============================================================================
    # MOTION VISUALIZATION
    # =============================================================================
    
    _MOTION_PALETTE = [
        '#e6194b', '#4363d8', '#3cb44b', '#f58231', '#911eb4',
        '#42d4f4', '#f032e6', '#e6ac00', '#469990', '#dcbeff',
    ]

    def plot_motion_overlay_on_scatter(peaks, peak_locations, motion_list, labels,
                                       sampling_frequency, colors=None, time_offset=0,
                                       t_start=None, t_end=None, save_path=None):
        """Plot motion estimates overlaid on spike scatter.

        colors is optional: if None, colours are auto-assigned from a palette.
        Title adapts: single method shows the method name, multiple shows 'comparison'.

        NOTE: peaks['sample_index'] is in ORIGINAL-recording time, and
        motion_obj.temporal_bins_s is in slice-relative time, so time_offset
        (the original recording's get_times()[0] PLUS the slice start time)
        is used to align them — same convention as the non-windowed pipeline.
        """
        active = [(i, m) for i, m in enumerate(motion_list) if m is not None]
        n_active = len(active)

        # Auto-assign colors
        if colors is not None:
            _colors = colors
        elif n_active == 1:
            _colors = ['#e6194b']          # vivid red stands out on black scatter
        else:
            _colors = [_MOTION_PALETTE[i % len(_MOTION_PALETTE)] for i in range(len(motion_list))]

        # Dynamic title
        if n_active == 1:
            title = f"Motion Overlay: {labels[active[0][0]]}"
        else:
            title = "Motion Estimation Comparison"

        fig, ax = plt.subplots(figsize=(16, 9))

        # Background: raw spike data
        plot_spikes_scatter(peaks, peak_locations, sampling_frequency,
                            ax=ax, t_start=t_start, t_end=t_end, title=title)

        # Overlay motion estimates
        for m_idx, motion_obj in enumerate(motion_list):
            if motion_obj is None:
                continue

            color = _colors[m_idx]
            label = labels[m_idx]
            times = motion_obj.temporal_bins_s[0] - time_offset
            depths = motion_obj.spatial_bins_um
            displacements = motion_obj.displacement[0]  # Shape: (n_time_bins, n_spatial_bins)

            # Clip to the requested time window
            t_mask = np.ones(len(times), dtype=bool)
            if t_start is not None:
                t_mask &= times >= t_start
            if t_end is not None:
                t_mask &= times <= t_end
            times_w = times[t_mask]
            displacements_w = displacements[t_mask, :]

            for i, start_depth in enumerate(depths):
                drift_line = start_depth + displacements_w[:, i]
                ax.plot(times_w, drift_line, color=color, linewidth=1.2, alpha=0.6, label=label if i == 0 else "")
        
        ax.legend(loc='upper right', markerscale=10)
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
            print(f"  Plot saved to: {save_path}")
        plt.close()
    
    # %%
    # Plot motion comparison if multiple methods were run
    if peak_locations_mt_original is not None:
        motion_list = []
        motion_labels = []

        if motion_decentralized is not None:
            motion_list.append(motion_decentralized)
            motion_labels.append('Decentralized')

        if motion_dredge is not None:
            motion_list.append(motion_dredge)
            motion_labels.append('DREDge')

        if len(motion_list) > 0:
            print(f"\nPlotting motion {'overlay' if len(motion_list) == 1 else 'comparison'} ({', '.join(motion_labels)})...")
            # rec_original.get_times()[0] gives the slice-relative t0 of the
            # sliced recording (typically 0); add the slice's start time in
            # seconds (TIME_WINDOW_START_FRAME / fs) to express motion bins
            # in ORIGINAL-recording time, matching peaks_original.
            time_offset = rec_original.get_times()[0] - (TIME_WINDOW_START_FRAME / rec_original.sampling_frequency)
            t_total = rec_full.get_total_duration()
            for _label, _t0, _t1 in [
                ("start", 0.0,                             min(SCATTER_WINDOW_S, t_total)),
                ("end",   max(0.0, t_total - SCATTER_WINDOW_S), t_total),
            ]:
                save_path = os.path.join(PLOTS_DIR, f"05_motion_overlay_{_label}.png")
                plot_motion_overlay_on_scatter(
                    peaks_original,
                    peak_locations_mt_original,
                    motion_list=motion_list,
                    labels=motion_labels,
                    sampling_frequency=rec_original.sampling_frequency,
                    time_offset=time_offset,
                    t_start=_t0,
                    t_end=_t1,
                    save_path=save_path,
                )
    
    # %%
    # =============================================================================
    # MOTION CORRECTION (INTERPOLATION)
    # =============================================================================
    # LEGACY, not well tested — not what's actually used. Preferred: leave
    # APPLY_MOTION_CORRECTION=False and use motion/correct_motion.py instead
    # (direct subtraction of estimated motion from peak depths, no recording
    # interpolation). See the config section above for why.

    print("\n" + "="*80)
    print("MOTION CORRECTION")
    print("="*80)
    
    if not APPLY_MOTION_CORRECTION:
        print("\nMotion correction disabled in config. Skipping interpolation.")
        rec_corrected = None
    else:
        # Select motion estimate to use
        if CORRECTION_METHOD == "decentralized":
            motion_to_use = motion_decentralized
            method_name = "Decentralized"
        elif CORRECTION_METHOD == "dredge":
            motion_to_use = motion_dredge
            method_name = "DREDge"
        else:
            raise ValueError(f"Unknown correction method: {CORRECTION_METHOD}")
        
        if motion_to_use is None:
            print(f"\nERROR: {method_name} motion not available.")
            print(f"Enable RUN_{CORRECTION_METHOD.upper()}_MOTION in config.")
            rec_corrected = None
        else:
            save_path = os.path.join(BASE_PATH, FOLDER_NAME, f"rec_corrected_{CORRECTION_METHOD}")
            
            if LOAD_COMPUTED_DATA and os.path.exists(save_path):
                print(f"\nLoading precomputed corrected recording ({method_name})...")
                rec_corrected = load_data(save_path)
                print(f"  Loaded from: {save_path}")
            else:
                print(f"\nApplying motion correction using {method_name} estimate...")
                rec_corrected = interpolate_motion(
                    recording=rec_original,
                    motion=motion_to_use,
                    **INTERPOLATION_PARAMS
                )
                print(f"  Saving corrected recording to: {save_path}")
                rec_corrected.save_to_folder(save_path, overwrite=False, n_jobs=SAVE_N_JOBS)
                print("  Save complete")
            
            print(f"\nCorrected recording: {rec_corrected}")
    
    # %%
    # =============================================================================
    # RE-DETECT PEAKS ON CORRECTED DATA
    # =============================================================================
    # LEGACY, doesn't test well — only runs if motion correction above ran
    # (which should stay disabled). Not part of the recommended workflow.
    
    if rec_corrected is not None and REDETECT_PEAKS_AFTER_CORRECTION:
        print("\n" + "="*80)
        print("RE-DETECTING PEAKS ON CORRECTED DATA (mua_pipeline)")
        print("="*80)
    
        # mua_pipeline saves peaks into <bin_path>/thresholding_peaks/ — use the corrected recording folder
        corrected_rec_path = os.path.join(BASE_PATH, FOLDER_NAME, f"rec_corrected_{CORRECTION_METHOD}")
        peaks_corrected_path = os.path.join(corrected_rec_path, FOLDER_NAME, "peaks_combined2.npy")
    
        if LOAD_COMPUTED_DATA and os.path.exists(peaks_corrected_path):
            print(f"\nLoading precomputed re-detected peaks (mua_pipeline)...")
            peaks_corrected = np.load(peaks_corrected_path)
            peaks_corrected = np.sort(peaks_corrected, order='sample_index')
            print(f"  Loaded from: {peaks_corrected_path}")
        else:
            print(f"\nRe-detecting peaks with mua_pipeline on corrected recording...")
            _, peaks_corrected, *_ = run_mua_pipeline(
                bin_path=corrected_rec_path,
                folder_name=FOLDER_NAME,
                **MUA_SETTINGS,
            )
            peaks_corrected = np.sort(peaks_corrected, order='sample_index')
            print(f"  Peaks saved by mua_pipeline to: {peaks_corrected_path}")

        # Shift re-detected peaks back to ORIGINAL-recording time as well
        peaks_corrected = shift_peaks_to_original_time(peaks_corrected, TIME_WINDOW_START_FRAME)
    
        print(f"  Re-detected peaks: {len(peaks_corrected):,}")
        print(f"  Original peaks:    {len(peaks_original):,}")
        print(f"  Difference:        {len(peaks_corrected) - len(peaks_original):+,}")
    
    elif rec_corrected is not None and not REDETECT_PEAKS_AFTER_CORRECTION:
        print("\n" + "="*80)
        print("USING ORIGINAL PEAKS ON CORRECTED DATA")
        print("="*80)
        print("\nRe-detection disabled. Using original peaks for verification.")
        peaks_corrected = peaks_original
    
    else:
        peaks_corrected = None
        print("\nSkipping peak re-detection (no corrected recording)")
    
    # %%
    # =============================================================================
    # RE-LOCALIZE ON CORRECTED DATA
    # =============================================================================
    # LEGACY, doesn't test well — same as the two sections above, only runs
    # if motion correction ran (which should stay disabled).

    if rec_corrected is not None and peaks_corrected is not None:
        print("\n" + "="*80)
        print("RE-LOCALIZING PEAKS ON CORRECTED DATA")
        print("="*80)
        
        # Re-run monopolar triangulation on corrected data
        save_path = os.path.join(BASE_PATH, FOLDER_NAME, f"rec_corrected_{CORRECTION_METHOD}", 
                                "localized_peaks_mt_corrected.npy")

        # peaks_corrected is in ORIGINAL-recording time; shift to slice-relative
        # time for the localize_peaks call against rec_corrected (which is the
        # same length/offset as rec_original, the slice).
        _peaks_corrected_for_localization = peaks_corrected.copy()
        _peaks_corrected_for_localization['sample_index'] = (
            _peaks_corrected_for_localization['sample_index'] - TIME_WINDOW_START_FRAME
        )
        
        if LOAD_COMPUTED_DATA and os.path.exists(save_path):
            print(f"\nLoading precomputed localization on corrected data...")
            peak_locations_mt_corrected = np.load(save_path)
            print(f"  Loaded from: {save_path}")
        else:
            print(f"\nRe-localizing peaks on corrected recording...")
            peak_locations_mt_corrected = localize_peaks(
                recording=rec_corrected,
                peaks=_peaks_corrected_for_localization,
                method='monopolar_triangulation',
                **LOCALIZATION_PARAMS['monopolar_triangulation'],
                **JOB_KWARGS
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, peak_locations_mt_corrected)
            print(f"  Saved to: {save_path}")
    
        print(f"  Localized peaks: {len(peak_locations_mt_corrected):,}")
        print(f"  Depth range: {peak_locations_mt_corrected['y'].min():.1f} - {peak_locations_mt_corrected['y'].max():.1f} µm")
    
    else:
        peak_locations_mt_corrected = None
        print("\nSkipping re-localization (no corrected recording or peaks)")
    
    # %%
    # =============================================================================
    # VERIFICATION PLOTS
    # =============================================================================
    
    if peak_locations_mt_corrected is not None:
        print("\n" + "="*80)
        print("VERIFICATION PLOTS")
        print("="*80)
        
        # Drift scatter after correction
        print(f"\nPlotting drift after {CORRECTION_METHOD} correction...")
        save_scatter_pair(
            peaks_corrected, peak_locations_mt_corrected,
            rec_corrected.sampling_frequency, rec_full.get_total_duration(),
            title_base=f"After {CORRECTION_METHOD.capitalize()} Correction: Peak Depths Over Time",
            save_prefix=os.path.join(PLOTS_DIR, f"06_drift_scatter_corrected_{CORRECTION_METHOD}"),
        )
        
        # Waveform plot with localized peaks on corrected data
        print(f"\nPlotting waveforms with localized peaks (corrected data)...")
        save_path = os.path.join(PLOTS_DIR, f"07_waveform_localization_corrected_{CORRECTION_METHOD}.png")
        plot_localized_peaks_on_depth(
            rec_corrected, 
            peaks=peaks_corrected,
            locations_sets=[peak_locations_mt_corrected],
            labels=['Monopolar Triangulation (Corrected)'],
            t_start=WAVEFORM_PLOT_TIME['t_start'],
            t_end=WAVEFORM_PLOT_TIME['t_end'],
            chan_start=WAVEFORM_PLOT_CHANNELS['chan_start'],
            chan_end=WAVEFORM_PLOT_CHANNELS['chan_end'],
            save_path=save_path,
            time_offset_frames=TIME_WINDOW_START_FRAME
        )
        
        # Comparison: Before vs After correction
        if peak_locations_mt_original is not None and REDETECT_PEAKS_AFTER_CORRECTION:
            print("\nPlotting before/after comparison...")
            t_total = rec_full.get_total_duration()
            for _label, _t0, _t1 in [
                ("start", 0.0,                             min(SCATTER_WINDOW_S, t_total)),
                ("end",   max(0.0, t_total - SCATTER_WINDOW_S), t_total),
            ]:
                fig, axes = plt.subplots(1, 2, figsize=(20, 8))
                plot_spikes_scatter(
                    peaks_original, peak_locations_mt_original,
                    rec_original.sampling_frequency, ax=axes[0],
                    t_start=_t0, t_end=_t1,
                    title=f"Before Correction ({_t0:.0f}\u2013{_t1:.0f}s)"
                )
                plot_spikes_scatter(
                    peaks_corrected, peak_locations_mt_corrected,
                    rec_original.sampling_frequency, ax=axes[1],
                    t_start=_t0, t_end=_t1,
                    title=f"After {CORRECTION_METHOD.capitalize()} ({_t0:.0f}\u2013{_t1:.0f}s)"
                )
                plt.tight_layout()
                save_path = os.path.join(PLOTS_DIR, f"08_before_after_comparison_{CORRECTION_METHOD}_{_label}.png")
                plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
                print(f"  Plot saved to: {save_path}")
                plt.close()
    
    # %%
    # =============================================================================
    # COMPREHENSIVE SUMMARY & SAVED RESULTS
    # =============================================================================
    
    print("\n" + "="*80)
    print("PIPELINE SUMMARY — ALL SAVED RESULTS")
    print("="*80)
    
    print("\n[INPUT DATA]")
    print(f"  Recording: {BASE_PATH}")
    print(f"  Sampling rate: {rec_original.sampling_frequency} Hz")
    print(f"  Full recording duration: {rec_full.get_total_duration():.1f} s")
    if TIME_START_S is not None or TIME_END_S is not None:
        print(f"  Time window applied: [{TIME_START_S}, {TIME_END_S})  "
              f"(sliced duration: {rec_original.get_total_duration():.1f} s)")
    else:
        print(f"  Time window: full recording")
    
    print("\n[SPIKE DETECTION]")
    print(f"  Original peaks: {len(peaks_original):,}")
    if RUN_MUA_PIPELINE:
        print(f"  Source: mua_pipeline (generated in this run)")
    else:
        print(f"  Source: Cached from {peaks_path}")
    
    print("\n[LOCALIZATION (Original Data) — SAVED]")
    if peak_locations_mt_original is not None:
        mt_save = os.path.join(BASE_PATH, FOLDER_NAME, "localized_peaks_mt_raw.npy")
        print(f"  ✓ Monopolar Triangulation: {mt_save}")
        print(f"    {len(peak_locations_mt_original):,} peaks, "
              f"depth range: {peak_locations_mt_original['y'].min():.1f}–{peak_locations_mt_original['y'].max():.1f} µm")
    if peak_locations_cm_original is not None:
        cm_save = os.path.join(BASE_PATH, FOLDER_NAME, "localized_peaks_cm_raw.npy")
        print(f"  ✓ Center of Mass: {cm_save}")
        print(f"    {len(peak_locations_cm_original):,} peaks, "
              f"depth range: {peak_locations_cm_original['y'].min():.1f}–{peak_locations_cm_original['y'].max():.1f} µm")
    
    print("\n[MOTION ESTIMATION — SAVED]")
    if motion_decentralized is not None:
        dec_save = os.path.join(MOTION_DIR, "motion_decentralized")
        print(f"  ✓ Decentralized: {dec_save}")
        print(f"    Temporal bins: {len(motion_decentralized.temporal_bins_s[0])}, "
              f"Spatial bins: {len(motion_decentralized.spatial_bins_um)}")
    if motion_dredge is not None:
        dredge_save = os.path.join(MOTION_DIR, "motion_dredge")
        print(f"  ✓ DREDge: {dredge_save}")
        print(f"    Temporal bins: {len(motion_dredge.temporal_bins_s[0])}, "
              f"Spatial bins: {len(motion_dredge.spatial_bins_um)}")
    if motion_decentralized is None and motion_dredge is None:
        print("  (None computed; enable in config)")
    
    print("\n[MOTION CORRECTION — SAVED]")
    if rec_corrected is not None:
        corr_dir = os.path.join(BASE_PATH, FOLDER_NAME, f"rec_corrected_{CORRECTION_METHOD}")
        print(f"  ✓ Method: {CORRECTION_METHOD.capitalize()}")
        print(f"  ✓ Recording: {corr_dir}")
        
        if peaks_corrected is not None and REDETECT_PEAKS_AFTER_CORRECTION:
            peaks_save = os.path.join(corr_dir, FOLDER_NAME, "peaks_combined2.npy")
            print(f"  ✓ Re-detected peaks (mua_pipeline): {peaks_save}")
            print(f"    {len(peaks_corrected):,} peaks (Δ {len(peaks_corrected)-len(peaks_original):+d} vs original)")
        
        if peak_locations_mt_corrected is not None:
            locs_save = os.path.join(corr_dir, "localized_peaks_mt_corrected.npy")
            print(f"  ✓ Re-localized peaks: {locs_save}")
            print(f"    depth range: {peak_locations_mt_corrected['y'].min():.1f}–{peak_locations_mt_corrected['y'].max():.1f} µm")
    else:
        print("  (Not applied; set APPLY_MOTION_CORRECTION = True in config)")
    
    print("\n[PLOTS — SAVED]")
    plot_list = [
        ("01_waveform_localization_comparison_original.png", "Localization methods on raw data"),
        ("02_method_comparison_scatter.png", "MT vs CM comparison"),
        ("03_drift_scatter_mt_original_{start,end}.png", "Drift (MT) on original"),
        ("04_drift_scatter_cm_original_{start,end}.png", "Drift (CM) on original"),
        ("05_motion_overlay_{start,end}.png", "Motion overlay (1 or 2 methods)"),
        ("06_drift_scatter_corrected_*_{start,end}.png", f"Drift after {CORRECTION_METHOD} correction"),
        ("07_waveform_localization_corrected_*.png", f"Waveforms with corrected localization"),
        ("08_before_after_comparison_*_{start,end}.png", f"Before/after correction comparison"),
    ]
    print(f"  Directory: {PLOTS_DIR}")
    for fname, desc in plot_list:
        print(f"    • {fname:45} ({desc})")
    
    print(f"\n[MOTION ESTIMATES — SAVED]")
    print(f"  Directory: {MOTION_DIR}")
    if motion_decentralized is not None or motion_dredge is not None:
        print("    • motion_decentralized/")
        print("    • motion_dredge/")
    else:
        print("    (None; enable in config)")
    
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETE")
    print("="*80 + "\n")