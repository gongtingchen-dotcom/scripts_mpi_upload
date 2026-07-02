"""
spike_density.py
================
Spike density along the probe vector.

Loads peaks_combined2_sorted.npy and (optionally) localized_peaks_mt_raw.npy
(outputs of peak_localisation.py) together with an optional SpikeInterface
motion object, then produces up to four plot groups:

  1.  Waveform snippets  — raw traces with detected peaks indicated
      (always produced). Shows three peak sets:
        - peaks_combined2_sorted  (red)
        - peaks2                  (green)
        - peaks_matched           (blue)
      Plus per-channel threshold lines (dotted red) derived from
      detect_threshold in params.json and si.get_noise_levels().

  If PEAK_LOCS_PATH is set (or default file found):
  2.  Raw scatter + motion overlay  — 3 user-chosen windows
  3.  Motion-corrected scatter  — same 3 windows
  4.  Snapped scatter + channel density histograms (depth-ordered & id-ordered)
      + 4x96 regularized grid map with deleted channels marked black with red lines.

If MOTION_DIR is None, no motion loading or correction is performed; all
motion-related sections treat displacement as zero (i.e. corrected = raw).

All plots are saved to PLOTS_DIR with numbered, descriptive names.
"""

# =============================================================================
# ── USER CONFIGURATION  (edit only this block) ───────────────────────────────
# =============================================================================

from pathlib import Path
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_PATH      = r"/nexus/posix0/MBR-neuralsystems/ephys/GLC7248/GLC7248_a_g3/preprocessed"
PEAKS_DIR      = r"/nexus/posix0/MBR-neuralsystems/ephys/GLC7248/GLC7248_a_g3/preprocessed/thresholding_peaks_3.5"
                 # if None → BASE_PATH / "thresholding_peaks"

PEAK_LOCS_PATH = r"/nexus/posix0/MBR-neuralsystems/ephys/GLC7248/GLC7248_a_g3/preprocessed/thresholding_peaks_3.5/localized_peaks_mt_raw.npy"
                 # Path to localized_peaks_mt_raw.npy.
                 # If None → PEAKS_DIR / "localized_peaks_mt_raw.npy" is tried;
                 # if that also doesn't exist, sections 2–4 are skipped entirely.

MOTION_DIR     = r"/nexus/posix0/MBR-neuralsystems/ephys/GLC7248/GLC7248_a_g3/preprocessed/thresholding_peaks_3.5/motion_estimates/motion_dredge"
                 # Path to SpikeInterface motion object folder.
                 # Set to None to skip motion loading/correction (displacement = 0).

PLOTS_DIR      = os.path.join(PEAKS_DIR, "spike_density_plots_3.5motion")   # if None → PEAKS_DIR / "spike_density_plots"

# ── Section 1: Waveform snippet windows ──────────────────────────────────────
WAVEFORM_WINDOWS = [
    {"plot_start_s": 100, "channel_start": 0, "plot_duration": 0.05, "channel_end": 384},
    {"plot_start_s": 1100,  "channel_start": 0, "plot_duration": 0.05, "channel_end": 384},
    {"plot_start_s": 2100,  "channel_start": 0, "plot_duration": 0.05, "channel_end": 384},
]

# ── Section 2 & 3: Scatter / motion-overlay windows ──────────────────────────
SCATTER_WINDOWS    = None
SCATTER_DURATION_S = 720.0

# ── Analysis time range ───────────────────────────────────────────────────────
ANALYSIS_START_S = None
ANALYSIS_END_S   = None

# ── Normalisation ─────────────────────────────────────────────────────────────
NORMALISE_BY_DURATION = True   # True → spikes / s;  False → raw counts

# ── Misc ──────────────────────────────────────────────────────────────────────
PLOT_DPI          = 150
RNG_SEED          = 42
CORRECTION_METHOD = "dredge"
RMS_DURATION_S    = 30 * 60
RMS_START_S       = 0.0
# =============================================================================
# ── END CONFIGURATION ─────────────────────────────────────────────────────────
# =============================================================================

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import spikeinterface as si

def _time_mask(arr_times, t0, t1):
    return (arr_times >= t0) & (arr_times < t1)

# ── Resolve paths ─────────────────────────────────────────────────────────────
BASE_PATH  = Path(BASE_PATH)
PEAKS_DIR  = Path(PEAKS_DIR) if PEAKS_DIR  else BASE_PATH / "thresholding_peaks"
MOTION_DIR = Path(MOTION_DIR) if MOTION_DIR is not None else None
PLOTS_DIR  = Path(PLOTS_DIR)  if PLOTS_DIR  else PEAKS_DIR / "spike_density_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

if PEAK_LOCS_PATH is not None:
    _locs_path = Path(PEAK_LOCS_PATH)
else:
    _locs_path = PEAKS_DIR / "localized_peaks_mt_raw.npy"

HAS_LOCS   = _locs_path.exists()
HAS_MOTION = MOTION_DIR is not None

print("="*70)
print("SPIKE DENSITY PIPELINE")
print("="*70)
print(f"  Base path  : {BASE_PATH}")
print(f"  Peaks dir  : {PEAKS_DIR}")
print(f"  Locs file  : {_locs_path}  {'✓' if HAS_LOCS else '✗ NOT FOUND — falling back to channel-map positions'}")
print(f"  Motion dir : {MOTION_DIR if HAS_MOTION else '(none — displacement = 0)'}")
print(f"  Plots dir  : {PLOTS_DIR}")

# ── Load recording ────────────────────────────────────────────────────────────
print("\n[1/6] Loading recording …")
rec = si.load(str(BASE_PATH))
fs          = rec.sampling_frequency
t_total     = rec.get_total_duration()
time_offset = float(rec.get_times()[0])
n_channels  = rec.get_num_channels()
channel_positions = rec.get_channel_locations()
channel_depths    = channel_positions[:, 1]
sort_idx_ch       = np.argsort(channel_depths)
sorted_depths_ch  = channel_depths[sort_idx_ch]
print(f"  Duration : {t_total:.1f} s   |  fs = {fs:.0f} Hz  |  channels = {n_channels}")

# ── Load threshold params + noise levels ──────────────────────────────────────
print("\n[1b/6] Loading threshold params and noise levels …")
_params_path = PEAKS_DIR / "params.json"
if _params_path.exists():
    with open(_params_path) as _f:
        _params = json.load(_f)
    detect_threshold = float(_params.get("detect_threshold", 3.3))
else:
    detect_threshold = 3.3
    print(f"  params.json not found — using default detect_threshold={detect_threshold}")
print(f"  detect_threshold = {detect_threshold}")

noise_levels = si.get_noise_levels(rec, return_scaled=False)
abs_thresholds = noise_levels * detect_threshold   # (n_channels,)
print(f"  Noise levels range: {noise_levels.min():.2f} – {noise_levels.max():.2f}")
print(f"  Threshold range  : {abs_thresholds.min():.2f} – {abs_thresholds.max():.2f}")

# ── Load peaks ────────────────────────────────────────────────────────────────
print("\n[2/6] Loading peaks …")
peaks_combined = np.load(PEAKS_DIR / "peaks_combined2_sorted.npy")

_peaks2_path   = PEAKS_DIR / "peaks2.npy"
_matched_path  = PEAKS_DIR / "peaks_matched.npy"
peaks2         = np.load(_peaks2_path)   if _peaks2_path.exists()  else None
peaks_matched  = np.load(_matched_path) if _matched_path.exists() else None

print(f"  peaks_combined2_sorted : {len(peaks_combined):,}")
print(f"  peaks2                 : {len(peaks2):,}"   if peaks2        is not None else "  peaks2         : not found")
print(f"  peaks_matched          : {len(peaks_matched):,}" if peaks_matched is not None else "  peaks_matched  : not found")

peak_times_combined = peaks_combined["sample_index"] / fs
peak_times2         = peaks2["sample_index"] / fs         if peaks2        is not None else None
peak_times_matched  = peaks_matched["sample_index"] / fs  if peaks_matched is not None else None

# ── Restrict to analysis window ───────────────────────────────────────────────
_analysis_start   = float(np.clip(ANALYSIS_START_S if ANALYSIS_START_S is not None else 0.0,     0.0, t_total))
_analysis_end     = float(np.clip(ANALYSIS_END_S   if ANALYSIS_END_S   is not None else t_total, 0.0, t_total))
assert _analysis_end > _analysis_start, "ANALYSIS_END_S must be greater than ANALYSIS_START_S"
analysis_duration = _analysis_end - _analysis_start

def _trim(pk_arr, pk_times):
    if pk_arr is None:
        return None, None
    m = _time_mask(pk_times, _analysis_start, _analysis_end)
    return pk_arr[m], pk_times[m]

peaks_combined, peak_times_combined = _trim(peaks_combined, peak_times_combined)
peaks2,         peak_times2         = _trim(peaks2,         peak_times2)
peaks_matched,  peak_times_matched  = _trim(peaks_matched,  peak_times_matched)

peaks      = peaks_combined
peak_times = peak_times_combined
n_peaks    = len(peaks)

# ── Resolve peak positions ────────────────────────────────────────────────────
print("\n[2b/6] Resolving peak positions …")
'''
if HAS_LOCS:
    _all_locs = np.load(_locs_path)
    if isinstance(_all_locs, np.lib.npyio.NpzFile):
        _all_locs = _all_locs["locs"]
    if hasattr(_all_locs, "dtype") and _all_locs.dtype.names is not None:
        locs_x = _all_locs["x"] if "x" in _all_locs.dtype.names else np.zeros(len(_all_locs), dtype=float)
        locs_y = _all_locs["y"] if "y" in _all_locs.dtype.names else np.zeros(len(_all_locs), dtype=float)
    elif np.ndim(_all_locs) == 2 and _all_locs.shape[1] >= 2:
        locs_x = _all_locs[:, 0]
        locs_y = _all_locs[:, 1]
    else:
        locs_x = np.zeros(len(_all_locs), dtype=float)
        locs_y = np.zeros(len(_all_locs), dtype=float)
    peak_positions_raw = np.column_stack([locs_x, locs_y]).astype(float)
    print("  Using external localization file for peak positions.")
else:
    peak_positions_raw = channel_positions[peaks_combined["channel_index"]].astype(float)
    print("  No localization file found — using peak channel locations from the channel map.")
'''

if HAS_LOCS:
    _all_locs = np.load(_locs_path)
    if isinstance(_all_locs, np.lib.npyio.NpzFile):
        _all_locs = _all_locs["locs"]
    if hasattr(_all_locs, "dtype") and _all_locs.dtype.names is not None:
        locs_y = _all_locs["y"] if "y" in _all_locs.dtype.names else np.zeros(len(_all_locs), dtype=float)
    elif np.ndim(_all_locs) == 2 and _all_locs.shape[1] >= 2:
        locs_y = _all_locs[:, 1]
    else:
        locs_y = np.zeros(len(_all_locs), dtype=float)
    # x is deliberately NOT taken from the localization file — the localization x
    # estimate is known to be noisy/unreliable. Instead we use the x position of
    # each peak's max (detection) channel, taken directly from the channel map.
    locs_x = channel_positions[peaks_combined["channel_index"], 0].astype(float)
    peak_positions_raw = np.column_stack([locs_x, locs_y]).astype(float)
    print("  Using external localization file for peak depth (y); x taken from max-channel position.")
else:
    peak_positions_raw = channel_positions[peaks_combined["channel_index"]].astype(float)
    print("  No localization file found — using peak channel locations from the channel map.")

peak_times_for_locs = peaks_combined["sample_index"] / fs
analysis_mask = _time_mask(peak_times_for_locs, _analysis_start, _analysis_end)
peak_positions_raw = peak_positions_raw[analysis_mask]
peak_times = peak_times_for_locs[analysis_mask]
peaks = peaks_combined[analysis_mask]
peak_times_combined = peak_times
peaks_combined = peaks
n_peaks = len(peaks)

depths_raw = peak_positions_raw[:, 1]
print(f"  Peak positions resolved — depth range : {depths_raw.min():.1f} – {depths_raw.max():.1f} µm")
print(f"  Analysis window : {_analysis_start:.1f} – {_analysis_end:.1f} s  "
      f"({analysis_duration:.1f} s)  →  {n_peaks:,} combined peaks retained")

# ── Load motion (optional) ────────────────────────────────────────────────────
if HAS_MOTION:
    print("\n[3/6] Loading motion estimate …")
    try:
        motion = si.Motion.load(str(MOTION_DIR))
        print(f"  Temporal bins : {len(motion.temporal_bins_s[0])}  |  "
              f"Spatial bins : {len(motion.spatial_bins_um)}")
    except Exception as exc:
        motion = None
        print(f"  Motion object unavailable ({exc}) — displacement = 0.")
else:
    motion = None
    print("\n[3/6] No MOTION_DIR set — displacement = 0.")

rng = np.random.default_rng(RNG_SEED)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _resolve_waveform_windows(windows_cfg, t_start, t_end, n_channels, rng):
    cfg = windows_cfg if windows_cfg is not None else [None, None, None]
    resolved = []
    for w in cfg:
        w   = w or {}
        dur = w.get("plot_duration") or 0.05
        t0  = w.get("plot_start_s")
        if t0 is None:
            t0 = float(rng.uniform(t_start, max(t_start, t_end - dur)))
        ch0 = w.get("channel_start")
        if ch0 is None:
            ch0 = int(rng.integers(0, max(1, n_channels - 70)))
        ch1 = w.get("channel_end")
        if ch1 is None:
            ch1 = min(ch0 + 70, n_channels)
        resolved.append({"plot_start_s": t0, "plot_duration": dur,
                          "channel_start": ch0, "channel_end": ch1})
    return resolved


def _resolve_scatter_windows(windows_cfg, t_start, t_end, duration, rng):
    cfg = windows_cfg if windows_cfg is not None else [None, None, None]
    resolved = []
    for w in cfg:
        w  = w or {}
        t0 = w.get("start_s")
        if t0 is None:
            t0 = float(rng.uniform(t_start, max(t_start, t_end - duration)))
        resolved.append({"start_s": t0, "end_s": min(t0 + duration, t_end)})
    return resolved


def get_corrected_positions(peaks, positions, motion_obj, fs, time_offset=0):
    if motion_obj is None:
        return positions.copy()
    pt = peaks["sample_index"] / fs
    depths = positions[:, 1]
    shifts = motion_obj.get_displacement_at_time_and_depth(pt + time_offset, depths)
    corrected = positions.copy()
    corrected[:, 1] = depths - shifts
    return corrected


def snap_to_nearest_channel(positions, channel_positions):
    if len(positions) == 0:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=int)

    snapped_ch_idx = np.empty(len(positions), dtype=int)
    for start in range(0, len(positions), 10000):
        stop = min(start + 10000, len(positions))
        chunk = positions[start:stop]
        diff = chunk[:, None, :] - channel_positions[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        snapped_ch_idx[start:stop] = np.argmin(dist2, axis=1)

    snapped_positions = channel_positions[snapped_ch_idx]
    return snapped_positions, snapped_ch_idx


def compute_channel_rms(recording, channel_ids, start_s, duration_s, fs, chunk_size_s=60.0):
    frame_start = int(max(0.0, start_s) * fs)
    frame_end = int(min(float(recording.get_total_duration()), max(0.0, start_s + duration_s)) * fs)
    if frame_end <= frame_start:
        return np.zeros(len(channel_ids), dtype=float)

    chunk_frames = max(1000, int(chunk_size_s * fs))
    sum_sq = np.zeros(len(channel_ids), dtype=np.float64)
    n_samples = 0
    for s0 in range(frame_start, frame_end, chunk_frames):
        s1 = min(frame_end, s0 + chunk_frames)
        traces = recording.get_traces(start_frame=s0, end_frame=s1, channel_ids=channel_ids)
        if traces.size == 0:
            continue
        sum_sq += np.sum(traces.astype(np.float64) * traces.astype(np.float64), axis=0)
        n_samples += traces.shape[0]

    if n_samples == 0:
        return np.zeros(len(channel_ids), dtype=float)
    return np.sqrt(sum_sq / float(n_samples))


def save_fig(fig, path, dpi=PLOT_DPI):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path.name}")


# =============================================================================
# SECTION 1 — WAVEFORM SNIPPETS WITH DETECTED PEAKS  (always runs)
# =============================================================================
print("\n" + "="*70)
print("SECTION 1 — Waveform snippets")
print("="*70)

waveform_windows = _resolve_waveform_windows(
    WAVEFORM_WINDOWS, _analysis_start, _analysis_end, n_channels, rng
)

_DOT_OFFSETS = {
    "combined": -0.1,
    "peaks2":   -0.2,   
    "matched":  -0.3,   
}
_DOT_COLORS  = {"combined": "#e6194b", "peaks2": "#3cb44b", "matched": "#4363d8"}
_DOT_LABELS  = {"combined": "peaks_combined2_sorted", "peaks2": "peaks2", "matched": "peaks_matched"}

def _overlay_peaks(ax, pk_arr, pk_times_arr, t0, t1, ch0, ch1,
                   offsets_arr, spacing, traces, fs, tag):
    if pk_arr is None:
        return
    pm  = _time_mask(pk_times_arr, t0, t1)
    pm &= (pk_arr["channel_index"] >= ch0) & (pk_arr["channel_index"] < ch1)
    pk_t      = pk_times_arr[pm]
    pk_ci_loc = pk_arr["channel_index"][pm] - ch0
    dot_offset_frac = _DOT_OFFSETS[tag]
    color = _DOT_COLORS[tag]

    for k in range(len(pk_t)):
        ci_loc = int(pk_ci_loc[k])
        t_samp = np.clip(int((pk_t[k] - t0) * fs), 0, traces.shape[0] - 1)
        trace_val = traces[t_samp, ci_loc]
        dot_y = offsets_arr[ci_loc] + trace_val + dot_offset_frac * spacing
        ax.scatter(pk_t[k], dot_y, color=color, s=10, zorder=4,
                   label=_DOT_LABELS[tag] if k == 0 else "")

for i, w in enumerate(waveform_windows):
    t0, t1 = w["plot_start_s"], w["plot_start_s"] + w["plot_duration"]
    ch0, ch1 = w["channel_start"], w["channel_end"]

    print(f"\n  Snippet {i+1}: t=[{t0:.3f}, {t1:.3f}] s  ch=[{ch0}, {ch1})")

    chan_ids      = rec.get_channel_ids()[ch0:ch1]
    chan_locs_sub = rec.get_channel_locations(channel_ids=chan_ids)
    y_depths_sub  = chan_locs_sub[:, 1]
    all_chan_ids  = rec.get_channel_ids()
    global_ch_indices = np.array([np.where(all_chan_ids == c)[0][0] for c in chan_ids])

    traces = rec.get_traces(start_frame=int(t0 * fs), end_frame=int(t1 * fs),
                            channel_ids=chan_ids)
    t_axis = np.linspace(t0, t1, traces.shape[0])

    num_ch  = len(chan_ids)
    max_amp = np.max(np.abs(traces)) if traces.size else 1.0
    spacing = max_amp * 1.8 if max_amp > 0 else 100
    offsets = np.arange(num_ch) * -spacing

    fig, ax = plt.subplots(figsize=(max(8, w["plot_duration"] * 180), max(4, num_ch * 0.35)))

    for ci in range(num_ch):
        ax.plot(t_axis, traces[:, ci] + offsets[ci], color="black", lw=0.4)

    for ci, global_ci in enumerate(global_ch_indices):
        thr_val = abs_thresholds[global_ci]
        ax.axhline(offsets[ci] - thr_val, color="#e6194b", lw=0.6, ls=":", alpha=0.7,
                   label="threshold" if ci == 0 else "")

    _overlay_peaks(ax, peaks_combined, peak_times_combined,
                   t0, t1, ch0, ch1, offsets, spacing, traces, fs, "combined")
    _overlay_peaks(ax, peaks2,         peak_times2,
                   t0, t1, ch0, ch1, offsets, spacing, traces, fs, "peaks2")
    _overlay_peaks(ax, peaks_matched,  peak_times_matched,
                   t0, t1, ch0, ch1, offsets, spacing, traces, fs, "matched")

    ax.set_yticks(offsets)
    ax.set_yticklabels([f"{d:.0f} µm" for d in y_depths_sub], fontsize=5)
    ax.set_xlim(t0, t1)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Probe depth (µm)", fontsize=9)
    ax.set_title(f"Snippet {i+1}:  t=[{t0:.3f}–{t1:.3f}] s  |  ch=[{ch0}–{ch1})  |  thr={detect_threshold}", fontsize=9)
    ax.grid(axis="x", alpha=0.1)

    _handles, _labels = ax.get_legend_handles_labels()
    _seen = {}
    for h, l in zip(_handles, _labels):
        if l not in _seen: _seen[l] = h
    ax.legend(_seen.values(), _seen.keys(), fontsize=6, loc="upper right", markerscale=2, framealpha=0.7)

    plt.tight_layout()
    save_fig(fig, PLOTS_DIR / f"01_waveform_snippet_{i+1:02d}_t{t0:.2f}s_ch{ch0}-{ch1}.png")


# =============================================================================
# SECTIONS 2–4
# =============================================================================
print("\n" + "="*70)
print("SECTION 2 — Raw scatter" + (" with motion overlay" if HAS_MOTION else " (no motion)"))
print("="*70)

scatter_windows = _resolve_scatter_windows(
    SCATTER_WINDOWS, _analysis_start, _analysis_end, SCATTER_DURATION_S, rng
)

DEPTH_COLORS_MOTION = ["red","royalblue","darkgreen","darkorange","deeppink","saddlebrown","teal"]

for i, w in enumerate(scatter_windows):
    t0, t1 = w["start_s"], w["end_s"]
    print(f"\n  Scatter {i+1}: t=[{t0:.0f}, {t1:.0f}] s")

    m   = _time_mask(peak_times, t0, t1)
    pt  = peak_times[m]
    dep = peak_positions_raw[:, 1][m]

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.scatter(pt, dep, s=1, color="k", alpha=0.02, rasterized=True)

    if HAS_MOTION:
        t_bins = motion.temporal_bins_s[0] - time_offset
        d_bins = motion.spatial_bins_um
        displ  = motion.displacement[0]
        tmask  = _time_mask(t_bins, t0, t1)
        for di, start_d in enumerate(d_bins):
            drift_line = start_d + displ[:, di]
            ax.plot(t_bins[tmask], drift_line[tmask],
                    color=DEPTH_COLORS_MOTION[di % len(DEPTH_COLORS_MOTION)],
                    lw=1.0, alpha=0.7, label=f"DREDge {di}" if di == 0 else "")
        ax.legend(loc="upper right", markerscale=6, fontsize=8)
        title = f"Raw scatter + motion overlay  [{t0:.0f}–{t1:.0f} s]"
    else:
        title = f"Raw scatter (no motion)  [{t0:.0f}–{t1:.0f} s]"

    ax.set_xlim(t0, t1)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Depth-like coordinate (µm)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    save_fig(fig, PLOTS_DIR / f"02_raw_scatter_{i+1:02d}_t{t0:.0f}-{t1:.0f}s.png")

# =========================================================================
# SECTION 3 — MOTION-CORRECTED SCATTER
# =========================================================================
print("\n" + "="*70)
print("SECTION 3 — " + ("Motion-corrected scatter" if HAS_MOTION else "Scatter (no motion correction)"))
print("="*70)

print("  Computing corrected positions …")
corrected_positions = get_corrected_positions(peaks, peak_positions_raw, motion, fs, time_offset)
if HAS_MOTION:
    print(f"  Corrected depth range: {corrected_positions[:, 1].min():.1f} – {corrected_positions[:, 1].max():.1f} µm")
else:
    print("  No motion — corrected positions equal raw positions.")

for i, w in enumerate(scatter_windows):
    t0, t1 = w["start_s"], w["end_s"]
    print(f"\n  Corrected scatter {i+1}: t=[{t0:.0f}, {t1:.0f}] s")

    m   = _time_mask(peak_times, t0, t1)
    pt  = peak_times[m]
    dep = corrected_positions[:, 1][m]

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.scatter(pt, dep, s=1, color="k", alpha=0.02, rasterized=True)
    ax.set_xlim(t0, t1)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Motion-corrected depth (µm)" if HAS_MOTION else "Depth-like coordinate (µm)", fontsize=11)
    ax.set_title(
        (f"Motion-corrected scatter ({CORRECTION_METHOD})  [{t0:.0f}–{t1:.0f} s]"
         if HAS_MOTION else f"Scatter (no correction)  [{t0:.0f}–{t1:.0f} s]"),
        fontsize=12
    )
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    save_fig(fig, PLOTS_DIR / f"03_corrected_scatter_{i+1:02d}_t{t0:.0f}-{t1:.0f}s.png")

# =========================================================================
# SECTION 4 — REGULARIZED MATRIX GRID MAP (4 COLUMNS × 96 ROWS)
# =========================================================================
print("\n" + "="*70)
print("SECTION 4 — Spike density and RMS Maps (Regular Matrix Layout)")
print("="*70)

print("  Snapping corrected peak positions to nearest channel …")
snapped_xy, snapped_ch_idx = snap_to_nearest_channel(corrected_positions, channel_positions)

print(f"  Counting spikes per channel [{_analysis_start:.1f}–{_analysis_end:.0f} s] …")
counts_all_raw = np.bincount(snapped_ch_idx.astype(int), minlength=n_channels).astype(float)
counts_by_id_rate = counts_all_raw / analysis_duration

ch_x_by_id = channel_positions[:, 0]
ch_y_by_id = channel_positions[:, 1]

if NORMALISE_BY_DURATION:
    density_by_channel = counts_by_id_rate
    count_label = "Spikes / s"
else:
    density_by_channel = counts_all_raw
    count_label = "Spike count"

_tag = f"_t{_analysis_start:.0f}-{_analysis_end:.0f}s"
_npy_result = PLOTS_DIR / f"channel_density_rms{_tag}.npy"
if _npy_result.exists():
    result_table = np.load(_npy_result)
    density_by_channel = result_table[:, 0].astype(float)
    rms_by_channel = result_table[:, 1].astype(float)
    ch_x_by_id = result_table[:, 2].astype(float)
    ch_y_by_id = result_table[:, 3].astype(float)
    print(f"  Loaded result table → {_npy_result.name}")
else:
    print(f"  Computing RMS per channel over {RMS_DURATION_S/60:.0f} min of recording …")
    rms_by_channel = compute_channel_rms(rec, rec.get_channel_ids(), RMS_START_S, RMS_DURATION_S, fs)
    result_table = np.column_stack([density_by_channel, rms_by_channel, ch_x_by_id, ch_y_by_id])
    np.save(_npy_result, result_table)
    print(f"  Saved result table → {_npy_result.name}")

corr_label = "corrected" if HAS_MOTION else "raw (no correction)"
# ──────────────────────────────────────────────────────────────────────────────
# CONSTRUCT 96 ROW × 4 COLUMN IDEALIZED GRID MATRIX
# ──────────────────────────────────────────────────────────────────────────────
print(f"  Mapping active channels to un-staggered 96x4 matrix grid ...")

grid_active = np.zeros((96, 4), dtype=bool)
grid_density = np.zeros((96, 4))
grid_rms = np.zeros((96, 4))

# Dynamically extract geometric anchors directly from YOUR data
unique_x = np.sort(np.unique(ch_x_by_id))
unique_y = np.sort(np.unique(ch_y_by_id))

x_min = np.min(unique_x)
y_min = np.min(unique_y)

# Use median differences to find your probe's true column and row spacing (pitch)
diff_x = np.diff(unique_x)
x_pitch = np.median(diff_x[diff_x > 2.0]) if len(diff_x[diff_x > 2.0]) > 0 else 16.0

diff_y = np.diff(unique_y)
y_pitch = np.median(diff_y[diff_y > 2.0]) if len(diff_y[diff_y > 2.0]) > 0 else 20.0
y_anchor = int(np.round(y_min / y_pitch) * y_pitch)
y_tick_step = max(1, int(round(2 * y_pitch)))

# Populate matrices by quantizing physical coordinates using the dynamic pitch
for idx in range(n_channels):
    cx = ch_x_by_id[idx]
    cy = ch_y_by_id[idx]
    
    # Accurately map X coordinate to column 0, 1, 2, or 3
    col = np.clip(int(np.round((cx - x_min) / x_pitch)), 0, 3)
    
    # For a 96-row grid, 2 physical depth levels (e.g. y=20 and y=40) form 1 logical row
    depth_idx = int(np.round((cy - y_min) / y_pitch))
    row = np.clip(depth_idx // 2, 0, 95)
    
    grid_active[row, col] = True
    grid_density[row, col] = density_by_channel[idx]
    grid_rms[row, col] = rms_by_channel[idx]

# ──────────────────────────────────────────────────────────────────────────────
# PLOT MATRIX VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 16), constrained_layout=True)
cm = plt.get_cmap("plasma")  # Standard high-contrast Purple-Yellow scheme

for ax_idx, ax in enumerate(axes):
    if ax_idx == 0:
        data_grid = grid_density
        v_active = grid_density[grid_active]
        vmin, vmax = (v_active.min(), v_active.max()) if len(v_active) > 0 else (0, 1)
        ax_title = f"Channel Map: {count_label}\n[{_analysis_start:.0f}–{_analysis_end:.0f} s] | {corr_label}"
        cbar_label = count_label
    else:
        data_grid = grid_rms
        v_active = grid_rms[grid_active]
        vmin, vmax = (v_active.min(), v_active.max()) if len(v_active) > 0 else (0, 1)
        ax_title = f"Channel Map: Noise RMS\n[Over {RMS_DURATION_S/60:.0f} min] | {corr_label}"
        cbar_label = "RMS Amplitude"

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    # Render every cell in the 96x4 matrix
    for r in range(96):
        for c in range(4):
            if grid_active[r, c]:
                val = data_grid[r, c]
                color = cm(norm(val))
                # Active cells: drawn flush against each other with matching thin edges to prevent cracks
                rect = Rectangle((c, r), 1.0, 1.0, facecolor=color, edgecolor=color, linewidth=0.05)
                ax.add_patch(rect)
            else:
                # Deleted channels: colored solid black with a sharp red diagonal line from corner to corner
                rect = Rectangle((c, r), 1.0, 1.0, facecolor='black', edgecolor='black', linewidth=0.05)
                ax.add_patch(rect)
                ax.plot([c, c + 1.0], [r, r + 1.0], color='red', linewidth=0.75, zorder=4)

    # Grid configurations
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 96)
    ax.set_aspect('equal')
    
    # Structural axes descriptors
    ax.set_xticks(np.arange(4) + 0.5)
    ax.set_xticklabels([f"Col {col_num}" for col_num in range(4)], fontsize=9)
    
    # Y-ticks map out the physical recording depths corresponding to the grid rows
    y_ticks = np.arange(0, 97, 10)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{int(y_anchor + y_idx * y_tick_step)} µm" for y_idx in y_ticks], fontsize=9)
    
    ax.set_xlabel("Columns", fontsize=11)
    ax.set_ylabel("Calculated Probe Depth (µm)", fontsize=11)
    ax.set_title(ax_title, fontsize=12, pad=12)
    
    # Append the calibrated colorbar mapping
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=cbar_label, fraction=0.046, pad=0.05)

save_fig(fig, PLOTS_DIR / f"04_channel_density_rms_matrix_grid_t{_analysis_start:.0f}-{_analysis_end:.0f}s.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("DONE — all plots saved to:")
print(f"  {PLOTS_DIR}")
print("\nFiles generated:")
for p in sorted(PLOTS_DIR.glob("0*.png")):
    print(f"  {p.name}")
npy_files = list(PLOTS_DIR.glob("*.npy"))
if npy_files:
    print("NPY arrays:")
    for p in npy_files:
        print(f"  {p.name}")
print("="*70)