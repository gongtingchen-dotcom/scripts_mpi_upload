"""
spike_density.py
================
Spike density along the probe vector.

Loads peaks_combined2_sorted.npy and localized_peaks_mt_raw.npy (outputs of
peak_localisation.py) together with a SpikeInterface motion object, then
produces four plot groups:

  1.  Waveform snippets  — raw traces with detected peaks indicated
  2.  Raw scatter + motion overlay  — 3 user-chosen windows
  3.  Motion-corrected scatter  — same 3 windows
  4a. (SNAP_CHANNEL=True)  Corrected scatter snapped to nearest channel,
      + per-window spike-count bar chart aligned to depth axis,
      + whole-recording channel histogram  →  saved as .npy
  4b. (SNAP_CHANNEL=False) Continuous corrected depth histogram per window
      + whole-recording depth histogram  →  saved as .npy

All plots are saved to PLOTS_DIR with numbered, descriptive names.
"""

# =============================================================================
# ── USER CONFIGURATION  (edit only this block) ───────────────────────────────
# =============================================================================

import os, sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_PATH  = r"/nexus/posix0/MBR-neuralsystems/ephys/GLC6962/GLC6962_g_g0/preprocessed"   # preprocessed recording folder
PEAKS_DIR  = None   # folder containing peaks_combined2_sorted.npy & localized_peaks_mt_raw.npy
                     # if None, defaults to BASE_PATH / "thresholding_peaks"
MOTION_DIR = r"/nexus/posix0/MBR-neuralsystems/ephys/GLC6962/GLC6962_g_g0/preprocessed/thresholding_peaks/motion_estimates/motion_dredge"   # folder containing the SpikeInterface motion object
                     # if None, defaults to PEAKS_DIR / "motion_estimates" / "chunks_1" / "motion_dredge_0000min_0070min"
PLOTS_DIR  = None   # where to save all output plots
                     # if None, defaults to BASE_PATH / "spike_density_plots"

# ── Section 1: Waveform snippet windows ──────────────────────────────────────
# Set as a list of dicts, one per snippet.  Any entry may be None to pick randomly.
# plot_duration defaults to 0.05 s; channel_end defaults to channel_start + 70.
# Leave WAVEFORM_WINDOWS = None to pick all three randomly.
WAVEFORM_WINDOWS = None
# Example of manual specification (mix of set / random):
# WAVEFORM_WINDOWS = [
#     {"plot_start_s": 500.0,  "channel_start": 100, "plot_duration": 0.05, "channel_end": 170},
#     {"plot_start_s": 1200.0, "channel_start": None, "plot_duration": 0.05, "channel_end": None},
#     {"plot_start_s": None,   "channel_start": 50,  "plot_duration": 0.05, "channel_end": None},
# ]

# ── Section 2 & 3: Scatter / motion-overlay windows ──────────────────────────
# List of {"start_s": float_or_None}.  duration is always 720 s.
# Leave SCATTER_WINDOWS = None to pick all three randomly.
SCATTER_WINDOWS = None
# Example:
# SCATTER_WINDOWS = [
#     {"start_s": 0.0},
#     {"start_s": None},
#     {"start_s": None},
# ]
SCATTER_DURATION_S = 720.0   # seconds shown per scatter window

# ── Section 4: Channel snapping ───────────────────────────────────────────────
SNAP_CHANNEL = True   # True → snap corrected depths to nearest channel

# ── Analysis time range ───────────────────────────────────────────────────────
# Restrict ALL sections (scatter windows, density counts) to this interval.
# Set to None to use the whole recording.
ANALYSIS_START_S = None   # e.g. 0.0
ANALYSIS_END_S   = None   # e.g. 3600.0

# ── Normalisation ─────────────────────────────────────────────────────────────
# If True, divide spike counts by the analysis duration (s) → spikes / second.
NORMALISE_BY_DURATION = True

# ── Misc ──────────────────────────────────────────────────────────────────────
PLOT_DPI  = 150
RNG_SEED  = 42          # for reproducible random window picks
CORRECTION_METHOD = "dredge"   # used only for labels / filenames

# =============================================================================
# ── END CONFIGURATION ─────────────────────────────────────────────────────────
# =============================================================================

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import spikeinterface as si

def _time_mask(arr_times, t0, t1):
    return (arr_times >= t0) & (arr_times < t1)

# ── Resolve paths ──────────────────────────────────────────────────────────────
BASE_PATH  = Path(BASE_PATH)
PEAKS_DIR  = Path(PEAKS_DIR) if PEAKS_DIR  else BASE_PATH / "thresholding_peaks"
MOTION_DIR = (Path(MOTION_DIR) if MOTION_DIR
              else PEAKS_DIR / "motion_estimates" / "chunks_1" / "motion_dredge_0000min_0070min")
PLOTS_DIR  = Path(PLOTS_DIR) if PLOTS_DIR else BASE_PATH / "spike_density_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("SPIKE DENSITY PIPELINE")
print("="*70)
print(f"  Base path  : {BASE_PATH}")
print(f"  Peaks dir  : {PEAKS_DIR}")
print(f"  Motion dir : {MOTION_DIR}")
print(f"  Plots dir  : {PLOTS_DIR}")
print(f"  SNAP_CHANNEL = {SNAP_CHANNEL}")

# ── Load recording ─────────────────────────────────────────────────────────────
print("\n[1/6] Loading recording …")
rec = si.load(str(BASE_PATH))
fs  = rec.sampling_frequency
t_total = rec.get_total_duration()
time_offset = float(rec.get_times()[0])
n_channels  = rec.get_num_channels()
channel_positions = rec.get_channel_locations()   # (n_ch, 2)  col0=x, col1=y
channel_depths    = channel_positions[:, 1]
sort_idx_ch       = np.argsort(channel_depths)
sorted_depths_ch  = channel_depths[sort_idx_ch]
print(f"  Duration : {t_total:.1f} s   |  fs = {fs:.0f} Hz  |  channels = {n_channels}")

# ── Load peaks & localisation ─────────────────────────────────────────────────
print("\n[2/6] Loading peaks and localisation …")
peaks_path  = PEAKS_DIR / "peaks_combined2_sorted.npy"
locs_path   = PEAKS_DIR / "localized_peaks_mt_raw.npy"
peaks       = np.load(peaks_path)
peak_locs   = np.load(locs_path)
depths_raw  = peak_locs["y"]
peak_times  = peaks["sample_index"] / fs
n_peaks     = len(peaks)
print(f"  Peaks : {n_peaks:,}   |  depth range : {depths_raw.min():.1f} – {depths_raw.max():.1f} µm")

# ── Restrict to analysis window ────────────────────────────────────────────────
_analysis_start = ANALYSIS_START_S if ANALYSIS_START_S is not None else 0.0
_analysis_end   = ANALYSIS_END_S   if ANALYSIS_END_S   is not None else t_total
_analysis_start = float(np.clip(_analysis_start, 0.0, t_total))
_analysis_end   = float(np.clip(_analysis_end,   0.0, t_total))
assert _analysis_end > _analysis_start, "ANALYSIS_END_S must be greater than ANALYSIS_START_S"
analysis_duration = _analysis_end - _analysis_start

_analysis_mask = _time_mask(peak_times, _analysis_start, _analysis_end)
peaks      = peaks[_analysis_mask]
depths_raw = depths_raw[_analysis_mask]
peak_times = peak_times[_analysis_mask]
n_peaks    = len(peaks)

print(f"  Analysis window : {_analysis_start:.1f} – {_analysis_end:.1f} s  "
      f"({analysis_duration:.1f} s)  →  {n_peaks:,} peaks retained")
print(f"  NORMALISE_BY_DURATION = {NORMALISE_BY_DURATION}  "
      f"({'spikes / s' if NORMALISE_BY_DURATION else 'raw counts'})")

# ── Load motion ────────────────────────────────────────────────────────────────
print("\n[3/6] Loading motion estimate …")
motion = si.Motion.load(str(MOTION_DIR))
print(f"  Temporal bins : {len(motion.temporal_bins_s[0])}  |  Spatial bins : {len(motion.spatial_bins_um)}")

# ── Amplitude colour scale ─────────────────────────────────────────────────────
_amp_vmin, _amp_vmax = np.percentile(np.abs(peaks["amplitude"]), [5, 95])

# ── RNG for random windows ─────────────────────────────────────────────────────
rng = np.random.default_rng(RNG_SEED)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _resolve_waveform_windows(windows_cfg, t_start, t_end, n_channels, rng):
    """Return list of 3 dicts with all fields filled in."""
    cfg = windows_cfg if windows_cfg is not None else [None, None, None]
    resolved = []
    for w in cfg:
        w = w or {}
        dur   = w.get("plot_duration") or 0.05
        t0    = w.get("plot_start_s")
        if t0 is None:
            t0 = float(rng.uniform(t_start, max(t_start, t_end - dur)))
        ch0   = w.get("channel_start")
        if ch0 is None:
            ch0 = int(rng.integers(0, max(1, n_channels - 70)))
        ch1   = w.get("channel_end")
        if ch1 is None:
            ch1 = min(ch0 + 70, n_channels)
        resolved.append({"plot_start_s": t0, "plot_duration": dur,
                          "channel_start": ch0, "channel_end": ch1})
    return resolved


def _resolve_scatter_windows(windows_cfg, t_start, t_end, duration, rng):
    """Return list of 3 dicts with start_s filled in."""
    cfg = windows_cfg if windows_cfg is not None else [None, None, None]
    resolved = []
    for w in cfg:
        w = w or {}
        t0 = w.get("start_s")
        if t0 is None:
            t0 = float(rng.uniform(t_start, max(t_start, t_end - duration)))
        resolved.append({"start_s": t0, "end_s": min(t0 + duration, t_end)})
    return resolved


def get_corrected_depths(peaks, depths, motion_obj, fs, time_offset=0):
    """Subtract motion displacement from raw depths (vectorised via si.Motion)."""
    pt = peaks["sample_index"] / fs
    shifts = motion_obj.get_displacement_at_time_and_depth(
        pt + time_offset, depths
    )
    return depths - shifts


def snap_to_nearest_channel(depths_corrected, channel_depths):
    """Return (snapped_depths, channel_indices) arrays."""
    sort_idx = np.argsort(channel_depths)
    sd = channel_depths[sort_idx]
    ins   = np.clip(np.searchsorted(sd, depths_corrected), 0, len(sd) - 1)
    left  = np.clip(ins - 1, 0, len(sd) - 1)
    closer_left = np.abs(depths_corrected - sd[left]) < np.abs(depths_corrected - sd[ins])
    best  = np.where(closer_left, left, ins)
    ch_idx = sort_idx[best]
    return channel_depths[ch_idx], ch_idx


def save_fig(fig, path, dpi=PLOT_DPI):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path.name}")


# =============================================================================
# SECTION 1 — WAVEFORM SNIPPETS WITH DETECTED PEAKS
# =============================================================================
print("\n" + "="*70)
print("SECTION 1 — Waveform snippets")
print("="*70)

waveform_windows = _resolve_waveform_windows(
    WAVEFORM_WINDOWS, _analysis_start, _analysis_end, n_channels, rng
)

COLORS_LOC = ["#e6194b", "#4363d8", "#3cb44b"]

for i, w in enumerate(waveform_windows):
    t0    = w["plot_start_s"]
    t1    = t0 + w["plot_duration"]
    ch0   = w["channel_start"]
    ch1   = w["channel_end"]

    print(f"\n  Snippet {i+1}: t=[{t0:.3f}, {t1:.3f}] s  ch=[{ch0}, {ch1})")

    chan_ids = rec.get_channel_ids()[ch0:ch1]
    chan_locs_sub = rec.get_channel_locations(channel_ids=chan_ids)
    y_depths_sub  = chan_locs_sub[:, 1]

    sf  = int(t0 * fs)
    ef  = int(t1 * fs)
    traces = rec.get_traces(start_frame=sf, end_frame=ef, channel_ids=chan_ids)
    t_axis = np.linspace(t0, t1, traces.shape[0])

    num_ch   = len(chan_ids)
    max_amp  = np.max(np.abs(traces)) if traces.size else 1.0
    spacing  = max_amp * 1.8 if max_amp > 0 else 100
    offsets  = np.arange(num_ch) * -spacing

    fig_w = max(8, w["plot_duration"] * 180)
    fig_h = max(4, num_ch * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for ci in range(num_ch):
        ax.plot(t_axis, traces[:, ci] + offsets[ci], color="black", lw=0.4)

    # Overlay detected peaks in this window & channel range
    pm   = _time_mask(peak_times, t0, t1)
    pm  &= (peaks["channel_index"] >= ch0) & (peaks["channel_index"] < ch1)
    pk_t = peak_times[pm]
    pk_ci_local = peaks["channel_index"][pm] - ch0

    for k in range(len(pk_t)):
        ci_loc = int(pk_ci_local[k])
        # Place dot just above the trace trough at this peak's time sample,
        # nudged upward by a small fixed fraction of the inter-channel spacing.
        t_samp = int((pk_t[k] - t0) * fs)
        t_samp = np.clip(t_samp, 0, traces.shape[0] - 1)
        trace_val = traces[t_samp, ci_loc]
        dot_y = offsets[ci_loc] + trace_val - spacing * 0.08
        ax.scatter(pk_t[k], dot_y, color=COLORS_LOC[0], s=12, zorder=4)

    depth_labels = [f"{d:.0f} µm" for d in y_depths_sub]
    ax.set_yticks(offsets)
    ax.set_yticklabels(depth_labels, fontsize=5)
    ax.set_xlim(t0, t1)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Probe depth (µm)", fontsize=9)
    ax.set_title(
        f"Snippet {i+1}:  t=[{t0:.3f}–{t1:.3f}] s  |  ch=[{ch0}–{ch1})  |  "
        f"{pm.sum()} peaks detected",
        fontsize=9
    )
    ax.grid(axis="x", alpha=0.1)
    plt.tight_layout()

    fname = PLOTS_DIR / f"01_waveform_snippet_{i+1:02d}_t{t0:.2f}s_ch{ch0}-{ch1}.png"
    save_fig(fig, fname)


# =============================================================================
# SECTION 2 — RAW SCATTER + MOTION OVERLAY
# =============================================================================
print("\n" + "="*70)
print("SECTION 2 — Raw scatter with motion overlay")
print("="*70)

scatter_windows = _resolve_scatter_windows(
    SCATTER_WINDOWS, _analysis_start, _analysis_end, SCATTER_DURATION_S, rng
)

DEPTH_COLORS_MOTION = [
    "red", "royalblue", "darkgreen", "darkorange", "deeppink", "saddlebrown", "teal"
]

for i, w in enumerate(scatter_windows):
    t0, t1 = w["start_s"], w["end_s"]
    print(f"\n  Scatter {i+1}: t=[{t0:.0f}, {t1:.0f}] s")

    m    = _time_mask(peak_times, t0, t1)
    pt   = peak_times[m]
    dep  = depths_raw[m]

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.scatter(pt, dep, s=1, color="k", alpha=0.02, rasterized=True)

    # Overlay motion lines
    t_bins = motion.temporal_bins_s[0] - time_offset
    d_bins = motion.spatial_bins_um
    displ  = motion.displacement[0]
    tmask  = _time_mask(t_bins, t0, t1)
    for di, start_d in enumerate(d_bins):
        drift_line = start_d + displ[:, di]
        ax.plot(t_bins[tmask], drift_line[tmask],
                color=DEPTH_COLORS_MOTION[di % len(DEPTH_COLORS_MOTION)],
                lw=1.0, alpha=0.7, label=f"DREDge {di}" if di == 0 else "")

    ax.set_xlim(t0, t1)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Depth (µm)", fontsize=11)
    ax.set_title(f"Raw scatter + motion overlay  [{t0:.0f}–{t1:.0f} s]", fontsize=12)
    ax.legend(loc="upper right", markerscale=6, fontsize=8)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()

    fname = PLOTS_DIR / f"02_raw_scatter_motion_{i+1:02d}_t{t0:.0f}-{t1:.0f}s.png"
    save_fig(fig, fname)


# =============================================================================
# SECTION 3 — MOTION-CORRECTED SCATTER (same windows as section 2)
# =============================================================================
print("\n" + "="*70)
print("SECTION 3 — Motion-corrected scatter")
print("="*70)

print("  Computing motion-corrected depths for all peaks …")
corrected_depths = get_corrected_depths(peaks, depths_raw, motion, fs, time_offset)
print(f"  Corrected depth range: {corrected_depths.min():.1f} – {corrected_depths.max():.1f} µm")

for i, w in enumerate(scatter_windows):
    t0, t1 = w["start_s"], w["end_s"]
    print(f"\n  Corrected scatter {i+1}: t=[{t0:.0f}, {t1:.0f}] s")

    m   = _time_mask(peak_times, t0, t1)
    pt  = peak_times[m]
    dep = corrected_depths[m]

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.scatter(pt, dep, s=1, color="k", alpha=0.02, rasterized=True)
    ax.set_xlim(t0, t1)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Motion-corrected depth (µm)", fontsize=11)
    ax.set_title(
        f"Motion-corrected scatter ({CORRECTION_METHOD})  [{t0:.0f}–{t1:.0f} s]",
        fontsize=12
    )
    ax.grid(True, alpha=0.15)
    plt.tight_layout()

    fname = PLOTS_DIR / f"03_corrected_scatter_{i+1:02d}_t{t0:.0f}-{t1:.0f}s.png"
    save_fig(fig, fname)


# =============================================================================
# SECTION 4 — CHANNEL DENSITY
# Branches on SNAP_CHANNEL
# =============================================================================
print("\n" + "="*70)
print(f"SECTION 4 — Spike density  (SNAP_CHANNEL = {SNAP_CHANNEL})")
print("="*70)

if SNAP_CHANNEL:
    # ── 4a: snap to nearest channel ───────────────────────────────────────────
    print("  Snapping corrected depths to nearest channel …")
    snapped_depths, snapped_ch_idx = snap_to_nearest_channel(corrected_depths, channel_depths)

    # unique channel depths in probe order (for axis)
    unique_ch_depths = sorted_depths_ch                     # sorted by depth
    unique_ch_idx    = sort_idx_ch                          # corresponding channel indices

    # ── per-window scatter + side histogram ──────────────────────────────────
    for i, w in enumerate(scatter_windows):
        t0, t1 = w["start_s"], w["end_s"]
        print(f"\n  Snapped scatter {i+1}: t=[{t0:.0f}, {t1:.0f}] s")

        m   = _time_mask(peak_times, t0, t1)
        pt  = peak_times[m]
        dep = snapped_depths[m]
        ch  = snapped_ch_idx[m]

        # count per channel within this window
        counts_win = np.bincount(ch.astype(int), minlength=n_channels).astype(float)
        win_dur = t1 - t0
        if NORMALISE_BY_DURATION:
            counts_win = counts_win / win_dur
            win_count_label = "Spikes / s"
        else:
            win_count_label = "Spike count"

        # reorder by depth for the bar chart
        counts_by_depth = counts_win[sort_idx_ch]

        fig = plt.figure(figsize=(22, 8))
        gs  = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.04)
        ax_sc  = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1])

        ax_sc.scatter(pt, dep, s=1, color="k", alpha=0.03, rasterized=True)
        ax_sc.set_xlim(t0, t1)
        ax_sc.set_ylim(unique_ch_depths.min() - 20, unique_ch_depths.max() + 20)
        ax_sc.set_xlabel("Time (s)", fontsize=11)
        ax_sc.set_ylabel("Snapped depth (µm)", fontsize=11)
        ax_sc.set_title(
            f"Snapped corrected scatter  [{t0:.0f}–{t1:.0f} s]", fontsize=12
        )
        ax_sc.grid(True, alpha=0.12)

        # horizontal bar chart — depth axis shared with scatter
        ax_bar.barh(unique_ch_depths, counts_by_depth,
                    height=np.diff(np.concatenate([[unique_ch_depths[0] - 10],
                                                   unique_ch_depths,
                                                   [unique_ch_depths[-1] + 10]]))[:-1] * 0.8,
                    color="steelblue", alpha=0.75, edgecolor="none")
        ax_bar.set_ylim(ax_sc.get_ylim())
        ax_bar.set_xlabel(win_count_label, fontsize=9)
        ax_bar.set_yticklabels([])
        ax_bar.yaxis.set_tick_params(left=False)
        ax_bar.set_title(win_count_label, fontsize=9)
        ax_bar.grid(True, alpha=0.15, axis="x")

        ax_sc.text(0.01, 0.99, f"N = {m.sum():,} peaks",
                   transform=ax_sc.transAxes, va="top", fontsize=8)

        fname = PLOTS_DIR / f"04a_snapped_scatter_{i+1:02d}_t{t0:.0f}-{t1:.0f}s.png"
        save_fig(fig, fname)

    # ── whole-recording channel histogram ────────────────────────────────────
    print(f"\n  Computing channel spike counts over analysis window "
          f"[{_analysis_start:.1f}–{_analysis_end:.1f} s] …")
    counts_all = np.bincount(snapped_ch_idx.astype(int), minlength=n_channels)
    counts_all_by_depth = counts_all[sort_idx_ch].astype(float)

    if NORMALISE_BY_DURATION:
        counts_all_by_depth = counts_all_by_depth / analysis_duration
        count_label = "Spikes / s"
        npy_stem    = "spike_rate_per_channel"
    else:
        count_label = "Spike count"
        npy_stem    = "spike_counts_per_channel"

    # save
    save_npy = PLOTS_DIR / f"{npy_stem}_t{_analysis_start:.0f}-{_analysis_end:.0f}s.npy"
    np.save(save_npy, np.stack(
        [unique_ch_idx.astype(float), unique_ch_depths, counts_all_by_depth], axis=1
    ))
    print(f"  Saved channel counts → {save_npy.name}")

    bar_h = np.diff(np.concatenate([[unique_ch_depths[0] - 10],
                                     unique_ch_depths,
                                     [unique_ch_depths[-1] + 10]]))[:-1] * 0.8

    fig, ax = plt.subplots(figsize=(7, 9))
    ax.barh(unique_ch_depths, counts_all_by_depth, height=bar_h,
            color="steelblue", alpha=0.8, edgecolor="none")
    ax.set_xlabel(f"{count_label}  [{_analysis_start:.0f}–{_analysis_end:.0f} s]", fontsize=11)
    ax.set_ylabel("Probe depth (µm)", fontsize=11)
    ax.set_ylim(unique_ch_depths.min() - 20, unique_ch_depths.max() + 20)
    ax.set_title("Spike density per channel", fontsize=12)
    ax.grid(True, alpha=0.15, axis="x")
    plt.tight_layout()
    fname = PLOTS_DIR / f"04a_channel_density_t{_analysis_start:.0f}-{_analysis_end:.0f}s.png"
    save_fig(fig, fname)

else:
    # ── 4b: continuous depth histogram ────────────────────────────────────────
    # Bin depth into fine bins for histograms
    depth_min = corrected_depths.min()
    depth_max = corrected_depths.max()
    n_depth_bins = max(50, int((depth_max - depth_min) / 10))  # ~10 µm per bin
    depth_edges  = np.linspace(depth_min, depth_max, n_depth_bins + 1)
    depth_ctrs   = 0.5 * (depth_edges[:-1] + depth_edges[1:])

    # ── per-window scatter + side histogram ──────────────────────────────────
    for i, w in enumerate(scatter_windows):
        t0, t1 = w["start_s"], w["end_s"]
        print(f"\n  Corrected density scatter {i+1}: t=[{t0:.0f}, {t1:.0f}] s")

        m   = _time_mask(peak_times, t0, t1)
        pt  = peak_times[m]
        dep = corrected_depths[m]

        counts_win, _ = np.histogram(dep, bins=depth_edges)
        counts_win = counts_win.astype(float)
        win_dur = t1 - t0
        if NORMALISE_BY_DURATION:
            counts_win = counts_win / win_dur
            win_count_label = "Spikes / s"
        else:
            win_count_label = "Spike count"

        fig = plt.figure(figsize=(22, 8))
        gs  = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.04)
        ax_sc  = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1])

        ax_sc.scatter(pt, dep, s=1, color="k", alpha=0.03, rasterized=True)
        ax_sc.set_xlim(t0, t1)
        ax_sc.set_ylim(depth_min - 20, depth_max + 20)
        ax_sc.set_xlabel("Time (s)", fontsize=11)
        ax_sc.set_ylabel("Motion-corrected depth (µm)", fontsize=11)
        ax_sc.set_title(
            f"Motion-corrected scatter + depth histogram  [{t0:.0f}–{t1:.0f} s]",
            fontsize=12
        )
        ax_sc.grid(True, alpha=0.12)

        bin_h = np.diff(depth_edges) * 0.85
        ax_bar.barh(depth_ctrs, counts_win, height=bin_h,
                    color="steelblue", alpha=0.75, edgecolor="none")
        ax_bar.set_ylim(ax_sc.get_ylim())
        ax_bar.set_xlabel(win_count_label, fontsize=9)
        ax_bar.set_yticklabels([])
        ax_bar.yaxis.set_tick_params(left=False)
        ax_bar.set_title(win_count_label, fontsize=9)
        ax_bar.grid(True, alpha=0.15, axis="x")

        ax_sc.text(0.01, 0.99, f"N = {m.sum():,} peaks",
                   transform=ax_sc.transAxes, va="top", fontsize=8)

        fname = PLOTS_DIR / f"04b_corrected_density_{i+1:02d}_t{t0:.0f}-{t1:.0f}s.png"
        save_fig(fig, fname)

    # ── whole-window depth histogram ──────────────────────────────────────────
    print(f"\n  Computing depth histogram over analysis window "
          f"[{_analysis_start:.1f}–{_analysis_end:.1f} s] …")
    counts_all, _ = np.histogram(corrected_depths, bins=depth_edges)
    counts_all = counts_all.astype(float)

    if NORMALISE_BY_DURATION:
        counts_all = counts_all / analysis_duration
        count_label = "Spikes / s"
        npy_stem    = "spike_rate_per_depth"
    else:
        count_label = "Spike count"
        npy_stem    = "spike_density_per_depth"

    save_npy = PLOTS_DIR / f"{npy_stem}_t{_analysis_start:.0f}-{_analysis_end:.0f}s.npy"
    np.save(save_npy, np.stack([depth_ctrs, counts_all], axis=1))
    print(f"  Saved depth density → {save_npy.name}")

    bar_h = np.diff(depth_edges) * 0.85
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.barh(depth_ctrs, counts_all, height=bar_h,
            color="steelblue", alpha=0.8, edgecolor="none")
    ax.set_xlabel(f"{count_label}  [{_analysis_start:.0f}–{_analysis_end:.0f} s]", fontsize=11)
    ax.set_ylabel("Motion-corrected depth (µm)", fontsize=11)
    ax.set_ylim(depth_min - 20, depth_max + 20)
    ax.set_title("Spike density along probe", fontsize=12)
    ax.grid(True, alpha=0.15, axis="x")
    plt.tight_layout()
    fname = PLOTS_DIR / f"04b_depth_density_t{_analysis_start:.0f}-{_analysis_end:.0f}s.png"
    save_fig(fig, fname)


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