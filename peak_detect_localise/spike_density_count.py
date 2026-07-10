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
  4.  Channel density / RMS / sorted-unit-rate grid map (2 columns x
      exact-depth rows) + a slim per-depth spike-rate strip. Deleted
      channels (missing from the recording, cross-checked against
      probe.json) are only marked black/red in the noise-RMS panel, since
      that's the only quantity directly tied to a channel's own signal.
      Density and sorted-unit snapping are free to land on any nearest
      channel regardless of deletion status.

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
BASE_PATH      = r"/nexus/posix0/MBR-neuralsystems/ephys/GLC7248/GLC7248_a_g1/preprocessed"
PEAKS_DIR      = r"/nexus/posix0/MBR-neuralsystems/ephys/GLC7248/GLC7248_a_g1/preprocessed/thresholding_peaks_3.5_ts5600_teend/_sliced_recording/thresholding_peaks_3.5_ts5600_teend"
# PEAKS_DIR      = os.path.join(BASE_PATH, "thresholding_peaks_3.5_ts5600_teend")
                 # if None → BASE_PATH / "thresholding_peaks"

PEAK_LOCS_PATH = None
# PEAK_LOCS_PATH = os.path.join(PEAKS_DIR, "localized_peaks_mt_raw.npy")
                 # Path to localized_peaks_mt_raw.npy.
                 # If None → PEAKS_DIR / "localized_peaks_mt_raw.npy" is tried;
                 # if that also doesn't exist, sections 2–4 are skipped entirely.

# MOTION_DIR     = os.path.join(PEAKS_DIR, "motion_estimates/motion_dredge")
#                  # Path to SpikeInterface motion object folder.
#                  # Set to None to skip motion loading/correction (displacement = 0).

MOTION_DIR     = r"/nexus/posix0/MBR-neuralsystems/ephys/GLC7248/GLC7248_a_g1/preprocessed/thresholding_peaks_3.5_ts5600_teend/motion_estimates/motion_dredge"

PLOTS_DIR      = os.path.join(PEAKS_DIR, "spike_density_plots_3.5motion_halfwidthsfiltered")   # if None → PEAKS_DIR / "spike_density_plots"

CLUSTERS_PATH  = r"/nexus/posix0/MBR-neuralsystems/ephys/GLC7248/GLC7248_a_g1/GLC7248_a_g1_imec0/20260625_clusters.npz"

# ── Section 1: Waveform snippet windows ──────────────────────────────────────
WAVEFORM_WINDOWS = [
    {"plot_start_s": 5700, "channel_start": 0, "plot_duration": 0.05, "channel_end": 384},
    {"plot_start_s": 6200,  "channel_start": 0, "plot_duration": 0.05, "channel_end": 384},
    {"plot_start_s": 6900,  "channel_start": 0, "plot_duration": 0.05, "channel_end": 384},
]

# ── Section 2 & 3: Scatter / motion-overlay windows ──────────────────────────
SCATTER_WINDOWS    = None
SCATTER_DURATION_S = 720.0

# ── Analysis time range ───────────────────────────────────────────────────────
ANALYSIS_START_S = 5600
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
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
import spikeinterface as si

def _time_mask(arr_times, t0, t1):
    return (arr_times >= t0) & (arr_times < t1)


def _find_clusters_file(base_path):
    """
    Search base_path and its ancestors for a directory whose name contains
    "imec" (case-insensitive), then look inside that directory for either
    "clusters.npz" or "<YYYYMMDD>_clusters.npz". If several dated files
    exist, the most recent date wins. Returns a Path or None.
    """
    import re
    candidate_dirs = [base_path] + list(base_path.parents)
    imec_dir = None
    for d in candidate_dirs:
        if "imec" in d.name.lower():
            imec_dir = d
            break
    if imec_dir is None or not imec_dir.exists():
        return None

    pattern = re.compile(r"^(?:(\d{8})_)?clusters\.npz$")
    matches = []
    for f in imec_dir.glob("*clusters.npz"):
        m = pattern.match(f.name)
        if m:
            date_str = m.group(1) or "00000000"
            matches.append((date_str, f))
    if not matches:
        return None
    matches.sort(key=lambda x: x[0])
    return matches[-1][1]


# ── Resolve paths ─────────────────────────────────────────────────────────────
BASE_PATH  = Path(BASE_PATH)
PEAKS_DIR  = Path(PEAKS_DIR)
MOTION_DIR = Path(MOTION_DIR)
PLOTS_DIR  = Path(PLOTS_DIR)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

if PEAK_LOCS_PATH is not None:
    _locs_path = Path(PEAK_LOCS_PATH)
else:
    _locs_path = None

HAS_LOCS = _locs_path is not None and _locs_path.exists()
HAS_MOTION = MOTION_DIR is not None

if CLUSTERS_PATH is not None:
    _clusters_path = Path(CLUSTERS_PATH)
    HAS_CLUSTERS = _clusters_path.exists()
else:
    _found = _find_clusters_file(BASE_PATH)
    _clusters_path = _found
    HAS_CLUSTERS = _found is not None

print("="*70)
print("SPIKE DENSITY PIPELINE")
print("="*70)
print(f"  Base path  : {BASE_PATH}")
print(f"  Peaks dir  : {PEAKS_DIR}")
print(f"  Locs file  : {_locs_path}  {'✓' if HAS_LOCS else '✗ NOT FOUND — falling back to channel-map positions'}")
print(f"  Motion dir : {MOTION_DIR if HAS_MOTION else '(none — displacement = 0)'}")
print(f"  Clusters   : {_clusters_path if HAS_CLUSTERS else '✗ NOT FOUND — sorted-units panel will be left blank'}")
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

# ── Parse channel numbers from channel_ids (e.g. "imec0.ap#AP0" → 0) ─────────
# SpikeInterface channel_ids for this rig are strings like "imec0.ap#AP<n>".
# The trailing integer <n> is the device channel number; gaps in that integer
# sequence (e.g. AP10 missing) are exactly the deleted channels. Consecutive
# channel numbers are paired two-per-row on this staggered probe, so the row
# and left/right column of ANY channel — present or deleted — follow directly
# from its number: row = n // 2, col = n % 2 (verified against the raw
# channel_locations pattern: col 0 is always the smaller-x contact in its row,
# col 1 the larger-x one, even though the actual x values alternate between
# {16,48} and {0,32} depending on row parity). No probe.json or geometry
# inference needed — this is exact and works for any missing channel.
print("\n[1c/6] Parsing channel numbers from channel_ids for grid placement …")
import re as _re
_chan_id_strs = [str(c) for c in rec.get_channel_ids()]

def _parse_channel_number(s):
    m = _re.search(r"(\d+)$", s)
    if m is None:
        raise ValueError(f"Could not parse a trailing channel number from channel_id '{s}'")
    return int(m.group(1))

chan_num_local = np.array([_parse_channel_number(s) for s in _chan_id_strs], dtype=int)
n_total_channels = int(chan_num_local.max()) + 1   # assumes numbering starts at 0
present_set = set(chan_num_local.tolist())
deleted_channel_numbers = sorted(set(range(n_total_channels)) - present_set)
print(f"  Parsed {n_channels} channel numbers, range 0–{n_total_channels - 1} expected.")
print(f"  Deleted/missing channel numbers ({len(deleted_channel_numbers)}): {deleted_channel_numbers}")

# ── Load peaks ────────────────────────────────────────────────────────────────
print("\n[2/6] Loading peaks …")
peaks_combined = np.load(PEAKS_DIR / "peaks_combined2_sorted_halfwidthsfiltered_0.075ms.npy")
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
    """
    Snap each (x, y) position to the nearest channel — no exclusions;
    every channel currently present in `channel_positions` is eligible.
    """
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
# SECTION 4 — REGULARIZED MATRIX GRID MAP (2 COLUMNS × exact-depth ROWS)
# =========================================================================
print("\n" + "="*70)
print("SECTION 4 — Spike density and RMS Maps (Regular Matrix Layout)")
print("="*70)

print("  Snapping corrected peak positions to nearest channel (no exclusions) …")
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
# BUILD THE PROBE MAP DIRECTLY FROM REAL (x, y) POSITIONS — no channel-number
# arithmetic, no renumbering. Every surviving channel is drawn as a rectangle
# of fixed size (cell_width × cell_height) centered on its own real (x, y) in
# µm, exactly as get_channel_locations() reports it. Multiple shanks (or any
# other x-gaps) fall out automatically, since we plot true x/y coordinates
# rather than artificial column/row indices.
#
# Deleted channels have no entry in channel_positions at all (they were fully
# removed from `rec`), so their position can't be read directly. To still draw
# them, we reconstruct the *local* staggering pattern purely from the spatial
# data itself (which two x-offsets recur every other row, on which shank) and
# use that to interpolate the missing (x, y) — never from channel numbering.
# ──────────────────────────────────────────────────────────────────────────────
print("  Building probe map directly from channel_locations (x, y) …")

all_x = ch_x_by_id
all_y = ch_y_by_id

# cell height = smallest real spacing between distinct y-values (row pitch)
_unique_y_all = np.sort(np.unique(all_y))
_y_diffs = np.diff(_unique_y_all)
_y_diffs = _y_diffs[_y_diffs > 1e-6]
cell_height = float(np.min(_y_diffs)) if len(_y_diffs) > 0 else 1.0

# ── Segment channels into shanks by clustering distinct x-values: a gap much
# larger than the typical local x-spacing marks a new shank. This is what
# "automatically maps out the distance between different shanks" — real x
# gaps are preserved as-is on the plot, no artificial column collapsing.
_unique_x_all = np.sort(np.unique(all_x))
_x_diffs = np.diff(_unique_x_all)
_typical_x_step = float(np.median(_x_diffs)) if len(_x_diffs) > 0 else 1.0
_shank_break = _x_diffs > (_typical_x_step * 4 if _typical_x_step > 0 else 1e9)

_x_groups = []
_cur = [_unique_x_all[0]] if len(_unique_x_all) > 0 else []
for i in range(1, len(_unique_x_all)):
    if _shank_break[i - 1]:
        _x_groups.append(_cur)
        _cur = []
    _cur.append(_unique_x_all[i])
if _cur:
    _x_groups.append(_cur)
n_shanks = len(_x_groups)
print(f"  cell_height (row pitch) = {cell_height:.2f} µm  |  {n_shanks} shank(s) detected "
      f"from x-gaps > {_typical_x_step * 4:.1f} µm.")

def _shank_of_x(x):
    for gi, grp in enumerate(_x_groups):
        if min(grp) - 1e-6 <= x <= max(grp) + 1e-6:
            return gi
    # fallback: nearest group
    return int(np.argmin([min(abs(x - v) for v in grp) for grp in _x_groups]))

shank_of_local = np.array([_shank_of_x(x) for x in all_x])

# position -> local channel index, for exact real-channel lookup
_pos_key = lambda x, y: (round(float(x), 2), round(float(y), 2))
pos_to_local_idx = {_pos_key(all_x[i], all_y[i]): i for i in range(n_channels)}

# ── Per shank: reconstruct the 2-per-row stagger pattern from real data,
# then build every (x, y) slot the lattice implies — filled where a real
# channel matches exactly, otherwise deleted (interpolated position). ──────
slots = []   # each: dict(x, y, shank, local_idx or None)
for gi in range(n_shanks):
    members = np.where(shank_of_local == gi)[0]
    if len(members) == 0:
        continue
    xs_g = all_x[members]
    ys_g = all_y[members]
    y_min_g, y_max_g = ys_g.min(), ys_g.max()
    n_rows_g = int(round((y_max_g - y_min_g) / cell_height)) + 1

    rows_by_y_g = defaultdict(list)
    for m in members:
        row = int(round((all_y[m] - y_min_g) / cell_height))
        rows_by_y_g[row].append(all_x[m])

    # Determine the x-pair pattern separately for even and odd rows (mode of
    # observed full pairs), and the channel cell width (median pair spacing).
    even_pairs, odd_pairs, widths = [], [], []
    for row, xs_here in rows_by_y_g.items():
        if len(xs_here) >= 2:
            xs_sorted = tuple(sorted(set(round(x, 2) for x in xs_here))[:2])
            widths.append(xs_sorted[-1] - xs_sorted[0])
            (even_pairs if row % 2 == 0 else odd_pairs).append(xs_sorted)
    cell_width = float(np.median(widths)) if widths else _typical_x_step * 2

    def _mode_pair(pairs):
        if not pairs:
            return None
        vals, counts = np.unique(np.array(pairs), axis=0, return_counts=True)
        return tuple(vals[np.argmax(counts)])

    pattern_even = _mode_pair(even_pairs)
    pattern_odd = _mode_pair(odd_pairs)

    # De-stagger for VISUALIZATION only: shift the odd-row pattern to align
    # with the even-row pattern, so both parities render at the same two x
    # positions instead of the true alternating offset. This never changes
    # which real channel a given (x, y) maps to — that lookup already used
    # the TRUE x above (pos_to_local_idx) — it only changes where the
    # rectangle is drawn. Shanks are unaffected: this offset is tiny (the
    # within-shank stagger amount) relative to the shank-detection gap
    # threshold, so it can't move a channel into the wrong shank.
    align_offset = (pattern_even[0] - pattern_odd[0]
                     if (pattern_even is not None and pattern_odd is not None) else 0.0)

    for row in range(n_rows_g):
        y_row = y_min_g + row * cell_height
        is_odd = row % 2 == 1
        pattern = pattern_odd if is_odd else pattern_even
        xs_present_here = sorted(set(round(x, 2) for x in rows_by_y_g.get(row, [])))

        if pattern is not None:
            expected_xs = list(pattern)
        elif len(xs_present_here) >= 1:
            # No fully-paired row of this parity anywhere on the shank to
            # learn the pattern from — fall back to just the x(s) we actually
            # observed at this row (can't safely guess the partner's x).
            expected_xs = xs_present_here
        else:
            continue   # nothing known about this row at all — skip silently

        vis_shift = align_offset if (is_odd and pattern is pattern_odd and align_offset != 0.0) else 0.0

        for x_slot in expected_xs:
            local_idx = pos_to_local_idx.get(_pos_key(x_slot, y_row))
            slots.append({"x": x_slot + vis_shift, "y": y_row, "shank": gi,
                           "local_idx": local_idx, "cell_width": cell_width})

n_active = sum(1 for s in slots if s["local_idx"] is not None)
n_deleted = sum(1 for s in slots if s["local_idx"] is None)
print(f"  Reconstructed {len(slots)} probe slots across {n_shanks} shank(s)  |  "
      f"{n_active} active  |  {n_deleted} deleted (interpolated position).")
if n_deleted != len(deleted_channel_numbers):
    print(f"  NOTE: spatial reconstruction found {n_deleted} deleted slots, but channel_id "
          f"gaps indicated {len(deleted_channel_numbers)} deleted channels — worth a manual "
          f"check if these don't match (could mean a row's pattern couldn't be inferred).")

# Per-slot values, for the plotting loop below
slot_density = np.array([density_by_channel[s["local_idx"]] if s["local_idx"] is not None else 0.0
                          for s in slots])
slot_rms = np.array([rms_by_channel[s["local_idx"]] if s["local_idx"] is not None else 0.0
                      for s in slots])
slot_active = np.array([s["local_idx"] is not None for s in slots])
slot_x = np.array([s["x"] for s in slots])
slot_y = np.array([s["y"] for s in slots])
slot_w = np.array([s["cell_width"] for s in slots])

# local channel index -> index into `slots`, for cluster-unit snapping below
local_idx_to_slot = {}
for si, s in enumerate(slots):
    if s["local_idx"] is not None:
        local_idx_to_slot[s["local_idx"]] = si

# Per-depth (real y) summed spike rate, across every slot at that exact y —
# generalizes "sum of both channels sharing a depth" to any number of shanks.
_unique_y_slots = np.sort(np.unique(slot_y))
row_spike_density = np.array([slot_density[slot_y == y].sum() for y in _unique_y_slots])

# ──────────────────────────────────────────────────────────────────────────────
# SORTED-UNIT SPIKE RATES — from clusters.npz
# Each unit's (posX, posY) is snapped to the nearest surviving channel (no
# exclusions — deleted channels aren't in channel_positions, so this can't
# select them), and its rate is summed into that channel's slot.
# ──────────────────────────────────────────────────────────────────────────────
slot_cluster_rate = np.zeros(len(slots))
n_units_plotted = 0

if HAS_CLUSTERS:
    print(f"  Loading sorted units → {_clusters_path.name}")
    _clusters = np.load(_clusters_path)
    cl_x = np.asarray(_clusters["posX"], dtype=float)
    cl_y = np.asarray(_clusters["posY"], dtype=float)
    cl_rate = np.asarray(_clusters["rate"], dtype=float)
    cluster_positions = np.column_stack([cl_x, cl_y])
    _, cluster_ch_idx = snap_to_nearest_channel(cluster_positions, channel_positions)
    n_occupied = 0
    for u_idx, ch_idx in enumerate(cluster_ch_idx):
        si = local_idx_to_slot.get(int(ch_idx))
        if si is None:
            continue
        if slot_cluster_rate[si] == 0:
            n_occupied += 1
        slot_cluster_rate[si] += cl_rate[u_idx]
    n_units_plotted = len(cl_rate)
    print(f"  {n_units_plotted} sorted units snapped onto the channel grid "
          f"({n_occupied} distinct channels occupied).")
else:
    print("  No clusters.npz file found — sorted-units panel will be left blank.")

# ──────────────────────────────────────────────────────────────────────────────
# PLOT — True Physical Scale (32:20 aspect) with dynamic tight bounding box
# ──────────────────────────────────────────────────────────────────────────────
x_min_all, x_max_all = slot_x.min() - slot_w.max() * 0.5, slot_x.max() + slot_w.max() * 0.5
y_min_all, y_max_all = slot_y.min() - cell_height * 0.5, slot_y.max() + cell_height * 0.5
x_range = x_max_all - x_min_all
y_range = y_max_all - y_min_all

# ######### CHANGED ########
# # 1. Anchor the physical HEIGHT. This ensures the figure is always tall enough 
# # to fit the text, whether you are plotting a short tetrode or a tall single-shank.
# FIG_HEIGHT = 8.0  

# # 2. Calculate panel width dynamically to perfectly preserve the true physical aspect ratio.
# PANEL_W_IN = FIG_HEIGHT * (x_range / y_range)

# # 3. Size the total figure width (lowered the +10 padding to +5 to remove empty whitespace)
# FIG_WIDTH = (PANEL_W_IN * 3.35) + 5



# 1. Anchor the exact physical width you want for each map panel (in inches)
PANEL_W_IN = 0.5 ###THIS IS WHAT CHANGES THE SCALE! 

# 2. Allow the figure to be exactly as tall as physically necessary to maintain 
# the 1:1 aspect ratio. This is what preserves your 32x20 block geometry perfectly.
FIG_HEIGHT = PANEL_W_IN * (y_range / x_range)

# 3. Size the total figure width perfectly to fit the panels + colorbars + spacing
# (3 main panels + 0.35 ratio for the bar strip)
FIG_WIDTH = (PANEL_W_IN * 6) + 10

fig, axes = plt.subplots(
    1, 4, figsize=(FIG_WIDTH, FIG_HEIGHT),
    constrained_layout=True,
    gridspec_kw={"width_ratios": [1, 0.35, 1, 1]}, sharey=True
)
# fig.set_constrained_layout_pads(wspace=0.15, hspace=0.05, w_pad=0.2, h_pad=0.1)
cm = plt.get_cmap("plasma")  # Standard high-contrast Purple-Yellow scheme
ax_map_idx = {0: 0, 1: 2, 2: 3}   # panel index -> axes slot

for panel in range(3):
    ax = axes[ax_map_idx[panel]]

    if panel == 0:
        data_vals = slot_density
        v_active = slot_density[slot_active]
        vmin, vmax = (v_active.min(), v_active.max()) if len(v_active) > 0 else (0, 1)
        ax_title = f"Channel Map: {count_label}\n[{_analysis_start:.0f}–{_analysis_end:.0f} s] | {corr_label}"
        cbar_label = count_label
    elif panel == 1:
        data_vals = slot_rms
        v_active = slot_rms[slot_active]
        vmin, vmax = (v_active.min(), v_active.max()) if len(v_active) > 0 else (0, 1)
        ax_title = f"Channel Map: Noise RMS\n[Over {RMS_DURATION_S/60:.0f} min] | {corr_label}"
        cbar_label = "RMS Amplitude"
    else:
        data_vals = slot_cluster_rate
        v_active = slot_cluster_rate[slot_active]
        vmin, vmax = (v_active.min(), v_active.max()) if len(v_active) > 0 else (0, 1)
        ax_title = (f"Sorted Units: Spike Rate (Hz)\n{n_units_plotted} units"
                    if HAS_CLUSTERS else "Sorted Units\n(no clusters.npz found)")
        cbar_label = "Rate (Hz)"

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for si, s in enumerate(slots):
        x, y, w = s["x"], s["y"], s["cell_width"]
        if slot_active[si]:
            color = cm(norm(data_vals[si]))
            rect = Rectangle((x - w / 2, y - cell_height / 2), w, cell_height,
                              facecolor=color, edgecolor=color, linewidth=0.05)
            ax.add_patch(rect)
        else:
            rect = Rectangle((x - w / 2, y - cell_height / 2), w, cell_height,
                              facecolor='black', edgecolor='black', linewidth=0.05)
            ax.add_patch(rect)
            ax.plot([x - w / 2, x + w / 2], [y - cell_height / 2, y + cell_height / 2],
                    color='red', linewidth=0.75, zorder=4)

    ax.set_xlim(x_min_all, x_max_all)
    ax.set_ylim(y_min_all, y_max_all)
    
    # REVERT TO EQUAL: 1 µm width exactly equals 1 µm height visually.
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel("x position (µm)", fontsize=16)
    if panel == 0:
        ax.set_ylabel("Probe depth, y (µm)", fontsize=18)
    ax.set_title(ax_title, fontsize=18, pad=20)
    ax.tick_params(axis="both", labelsize=14)

    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label=cbar_label, fraction=0.09, pad=0.08, aspect=30)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(cbar_label, fontsize=16)

# ── Slim per-depth rate strip, sitting directly beside the density panel ──────
ax_bar = axes[1]
ax_bar.barh(_unique_y_slots, row_spike_density, height=cell_height * 0.95,
            color="steelblue", edgecolor="none")
ax_bar.set_ylim(y_min_all, y_max_all)
ax_bar.set_xlabel(count_label, fontsize=14)
ax_bar.set_title("Per-depth\nsum", fontsize=18, pad=20)
for spine in ("top", "right", "left"):
    ax_bar.spines[spine].set_visible(False)
ax_bar.tick_params(axis="y", left=False, labelleft=False)
ax_bar.tick_params(axis="x", labelsize=12)

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