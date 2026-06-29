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

If MOTION_DIR is None, no motion loading or correction is performed; all
motion-related sections treat displacement as zero (i.e. corrected = raw).

All plots are saved to PLOTS_DIR with numbered, descriptive names.
"""

# =============================================================================
# ── USER CONFIGURATION  (edit only this block) ───────────────────────────────
# =============================================================================

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_PATH      = r"/nexus/posix0/MBR-neuralsystems/ephys/GLC7248/GLC7248_a_g1/preprocessed"
PEAKS_DIR      = r"/nexus/posix0/MBR-neuralsystems/ephys/GLC7248/GLC7248_a_g1/preprocessed/thresholding_peaks_3.3"
                 # if None → BASE_PATH / "thresholding_peaks"

PEAK_LOCS_PATH = None
                 # Path to localized_peaks_mt_raw.npy.
                 # If None → PEAKS_DIR / "localized_peaks_mt_raw.npy" is tried;
                 # if that also doesn't exist, sections 2–4 are skipped entirely.

MOTION_DIR     = None
                 # Path to SpikeInterface motion object folder.
                 # Set to None to skip motion loading/correction (displacement = 0).

PLOTS_DIR      = None   # if None → BASE_PATH / "spike_density_plots"

# ── Section 1: Waveform snippet windows ──────────────────────────────────────
# WAVEFORM_WINDOWS = None
WAVEFORM_WINDOWS = [
    {"plot_start_s": 500, "channel_start": 100, "plot_duration": 0.05, "channel_end": 170},
    {"plot_start_s": 1500,  "channel_start": 100, "plot_duration": 0.05, "channel_end": 170},
    {"plot_start_s": 2500,  "channel_start": 100, "plot_duration": 0.05, "channel_end": 170},
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

# =============================================================================
# ── END CONFIGURATION ─────────────────────────────────────────────────────────
# =============================================================================

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import spikeinterface as si

def _time_mask(arr_times, t0, t1):
    return (arr_times >= t0) & (arr_times < t1)

# ── Resolve paths ─────────────────────────────────────────────────────────────
BASE_PATH  = Path(BASE_PATH)
PEAKS_DIR  = Path(PEAKS_DIR) if PEAKS_DIR  else BASE_PATH / "thresholding_peaks"
MOTION_DIR = Path(MOTION_DIR) if MOTION_DIR is not None else None
PLOTS_DIR  = Path(PLOTS_DIR)  if PLOTS_DIR  else BASE_PATH / "spike_density_plots"
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
print(f"  Locs file  : {_locs_path}  {'✓' if HAS_LOCS else '✗ NOT FOUND — sections 2–4 skipped'}")
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

# ── Load threshold params + noise levels (for section 1 threshold lines) ─────
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
abs_thresholds = noise_levels * detect_threshold   # (n_channels,) — negative threshold is -abs_thresholds
print(f"  Noise levels range: {noise_levels.min():.2f} – {noise_levels.max():.2f}")
print(f"  Threshold range  : {abs_thresholds.min():.2f} – {abs_thresholds.max():.2f}")

# ── Load peaks ────────────────────────────────────────────────────────────────
print("\n[2/6] Loading peaks …")
peaks_combined = np.load(PEAKS_DIR / "peaks_combined2_sorted.npy")

# peaks2 and peaks_matched — load if present; gracefully skip if absent
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

# 'peaks' alias used by sections 2–4 (combined set)
peaks      = peaks_combined
peak_times = peak_times_combined
n_peaks    = len(peaks)

# Load localisation
if HAS_LOCS:
    _all_locs   = np.load(_locs_path)
    _all_ptimes = np.load(PEAKS_DIR / "peaks_combined2_sorted.npy")["sample_index"] / fs
    _lm         = _time_mask(_all_ptimes, _analysis_start, _analysis_end)
    depths_raw  = _all_locs["y"][_lm]
    print(f"  Locs loaded — depth range : {depths_raw.min():.1f} – {depths_raw.max():.1f} µm")
else:
    depths_raw = None
    print("  No locs file — skipping sections 2–4.")

print(f"  Analysis window : {_analysis_start:.1f} – {_analysis_end:.1f} s  "
      f"({analysis_duration:.1f} s)  →  {n_peaks:,} combined peaks retained")

# ── Load motion (optional) ────────────────────────────────────────────────────
if HAS_MOTION:
    print("\n[3/6] Loading motion estimate …")
    motion = si.Motion.load(str(MOTION_DIR))
    print(f"  Temporal bins : {len(motion.temporal_bins_s[0])}  |  "
          f"Spatial bins : {len(motion.spatial_bins_um)}")
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


def get_corrected_depths(peaks, depths, motion_obj, fs, time_offset=0):
    if motion_obj is None:
        return depths.copy()
    pt     = peaks["sample_index"] / fs
    shifts = motion_obj.get_displacement_at_time_and_depth(pt + time_offset, depths)
    return depths - shifts


def snap_to_nearest_channel(depths_corrected, channel_depths):
    sort_idx    = np.argsort(channel_depths)
    sd          = channel_depths[sort_idx]
    ins         = np.clip(np.searchsorted(sd, depths_corrected), 0, len(sd) - 1)
    left        = np.clip(ins - 1, 0, len(sd) - 1)
    closer_left = np.abs(depths_corrected - sd[left]) < np.abs(depths_corrected - sd[ins])
    best        = np.where(closer_left, left, ins)
    ch_idx      = sort_idx[best]
    return channel_depths[ch_idx], ch_idx


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

# Dot vertical offsets so the three peak sets don't overlap
# combined = at trough, peaks2 = slightly above, peaks_matched = slightly below
_DOT_OFFSETS = {
    "combined": -0.1,
    "peaks2":   -0.2,   # fraction of spacing, above trough
    "matched":  -0.3,   # fraction of spacing, below trough
}
_DOT_COLORS  = {"combined": "#e6194b", "peaks2": "#3cb44b", "matched": "#4363d8"}
_DOT_LABELS  = {"combined": "peaks_combined2_sorted", "peaks2": "peaks2", "matched": "peaks_matched"}

def _overlay_peaks(ax, pk_arr, pk_times_arr, t0, t1, ch0, ch1,
                   offsets_arr, spacing, traces, fs, tag):
    """Scatter detected peaks from one peaks array onto the waveform plot."""
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
    t0  = w["plot_start_s"]
    t1  = t0 + w["plot_duration"]
    ch0 = w["channel_start"]
    ch1 = w["channel_end"]

    print(f"\n  Snippet {i+1}: t=[{t0:.3f}, {t1:.3f}] s  ch=[{ch0}, {ch1})")

    chan_ids      = rec.get_channel_ids()[ch0:ch1]
    chan_locs_sub = rec.get_channel_locations(channel_ids=chan_ids)
    y_depths_sub  = chan_locs_sub[:, 1]
    # global channel indices for this sub-range
    all_chan_ids  = rec.get_channel_ids()
    global_ch_indices = np.array([np.where(all_chan_ids == c)[0][0] for c in chan_ids])

    traces = rec.get_traces(start_frame=int(t0 * fs), end_frame=int(t1 * fs),
                            channel_ids=chan_ids)
    t_axis = np.linspace(t0, t1, traces.shape[0])

    num_ch  = len(chan_ids)
    max_amp = np.max(np.abs(traces)) if traces.size else 1.0
    spacing = max_amp * 1.8 if max_amp > 0 else 100
    offsets = np.arange(num_ch) * -spacing

    fig, ax = plt.subplots(figsize=(max(8, w["plot_duration"] * 180),
                                    max(4, num_ch * 0.35)))

    # ── draw traces ──────────────────────────────────────────────────────────
    for ci in range(num_ch):
        ax.plot(t_axis, traces[:, ci] + offsets[ci], color="black", lw=0.4)

    # ── draw threshold lines (one per channel in the subplot) ─────────────
    for ci, global_ci in enumerate(global_ch_indices):
        thr_val = abs_thresholds[global_ci]
        # threshold is the negative crossing level; draw as horizontal dotted line
        ax.axhline(offsets[ci] - thr_val,
                   color="#e6194b", lw=0.6, ls=":", alpha=0.7,
                   label="threshold" if ci == 0 else "")

    # ── overlay peak dots ─────────────────────────────────────────────────
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
    ax.set_title(f"Snippet {i+1}:  t=[{t0:.3f}–{t1:.3f}] s  |  ch=[{ch0}–{ch1})  |  "
                 f"thr={detect_threshold}", fontsize=9)
    ax.grid(axis="x", alpha=0.1)

    # deduplicate legend entries
    _handles, _labels = ax.get_legend_handles_labels()
    _seen = {}
    for h, l in zip(_handles, _labels):
        if l not in _seen:
            _seen[l] = h
    ax.legend(_seen.values(), _seen.keys(), fontsize=6, loc="upper right",
              markerscale=2, framealpha=0.7)

    plt.tight_layout()
    save_fig(fig, PLOTS_DIR / f"01_waveform_snippet_{i+1:02d}_t{t0:.2f}s_ch{ch0}-{ch1}.png")


# =============================================================================
# SECTIONS 2–4  (only if localisation file is available)
# =============================================================================
if not HAS_LOCS:
    print("\nNo localisation file — stopping after section 1.")
else:

    # =========================================================================
    # SECTION 2 — RAW SCATTER + MOTION OVERLAY
    # =========================================================================
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
        dep = depths_raw[m]

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
        ax.set_ylabel("Depth (µm)", fontsize=11)
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

    print("  Computing corrected depths …")
    corrected_depths = get_corrected_depths(peaks, depths_raw, motion, fs, time_offset)
    if HAS_MOTION:
        print(f"  Corrected depth range: {corrected_depths.min():.1f} – {corrected_depths.max():.1f} µm")
    else:
        print("  No motion — corrected depths equal raw depths.")

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
        ax.set_ylabel("Motion-corrected depth (µm)" if HAS_MOTION else "Depth (µm)", fontsize=11)
        ax.set_title(
            (f"Motion-corrected scatter ({CORRECTION_METHOD})  [{t0:.0f}–{t1:.0f} s]"
             if HAS_MOTION else f"Scatter (no correction)  [{t0:.0f}–{t1:.0f} s]"),
            fontsize=12
        )
        ax.grid(True, alpha=0.15)
        plt.tight_layout()
        save_fig(fig, PLOTS_DIR / f"03_corrected_scatter_{i+1:02d}_t{t0:.0f}-{t1:.0f}s.png")

    # =========================================================================
    # SECTION 4 — CHANNEL DENSITY
    # =========================================================================
    print("\n" + "="*70)
    print("SECTION 4 — Spike density")
    print("="*70)

    print("  Snapping depths to nearest channel …")
    snapped_depths, snapped_ch_idx = snap_to_nearest_channel(corrected_depths, channel_depths)

    unique_ch_depths = sorted_depths_ch
    unique_ch_x      = channel_positions[sort_idx_ch, 0]

    print(f"  Counting spikes per channel [{_analysis_start:.1f}–{_analysis_end:.1f} s] …")
    counts_all_raw = np.bincount(snapped_ch_idx.astype(int), minlength=n_channels).astype(float)

    counts_by_id_raw   = counts_all_raw
    counts_by_id_rate  = counts_by_id_raw / analysis_duration
    ch_x_by_id         = channel_positions[:, 0]
    ch_y_by_id         = channel_positions[:, 1]

    counts_by_depth_raw  = counts_all_raw[sort_idx_ch]
    counts_by_depth_rate = counts_by_depth_raw / analysis_duration

    _tag = f"_t{_analysis_start:.0f}-{_analysis_end:.0f}s"

    _npy_counts = PLOTS_DIR / f"spike_count_per_channel{_tag}.npy"
    np.save(_npy_counts, np.stack([counts_by_id_raw, ch_x_by_id, ch_y_by_id], axis=1))
    print(f"  Saved raw counts → {_npy_counts.name}  (cols: count, x, y; row = channel id)")

    _npy_rate = PLOTS_DIR / f"spike_rate_per_channel{_tag}.npy"
    np.save(_npy_rate, np.stack([counts_by_id_rate, ch_x_by_id, ch_y_by_id], axis=1))
    print(f"  Saved spike rate → {_npy_rate.name}  (cols: spikes/s, x, y; row = channel id)")

    if NORMALISE_BY_DURATION:
        counts_for_plot_depth = counts_by_depth_rate
        counts_for_plot_id    = counts_by_id_rate
        count_label           = "Spikes / s"
    else:
        counts_for_plot_depth = counts_by_depth_raw
        counts_for_plot_id    = counts_by_id_raw
        count_label           = "Spike count"

    _depth_ylim  = (unique_ch_depths.min() - 20, unique_ch_depths.max() + 20)
    _bar_h_depth = np.diff(np.concatenate([[unique_ch_depths[0] - 10],
                                            unique_ch_depths,
                                            [unique_ch_depths[-1] + 10]]))[:-1] * 0.8
    _ch_ids   = np.arange(n_channels)
    _id_ylim  = (-1, n_channels)
    _bar_h_id = 0.7
    corr_label = "corrected" if HAS_MOTION else "raw (no correction)"

    for i, w in enumerate(scatter_windows):
        t0, t1 = w["start_s"], w["end_s"]
        print(f"\n  Section 4 scatter {i+1}: t=[{t0:.0f}, {t1:.0f}] s")

        m   = _time_mask(peak_times, t0, t1)
        pt  = peak_times[m]
        dep = snapped_depths[m]
        ch  = snapped_ch_idx[m]

        win_counts_raw = np.bincount(ch.astype(int), minlength=n_channels).astype(float)
        win_dur        = t1 - t0
        if NORMALISE_BY_DURATION:
            win_counts_depth = win_counts_raw[sort_idx_ch] / win_dur
            win_counts_id    = win_counts_raw               / win_dur
            win_count_label  = "Spikes / s"
        else:
            win_counts_depth = win_counts_raw[sort_idx_ch]
            win_counts_id    = win_counts_raw
            win_count_label  = "Spike count"

        # depth-ordered
        fig = plt.figure(figsize=(22, 8))
        gs  = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.04)
        ax_sc  = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1])

        ax_sc.scatter(pt, dep, s=1, color="k", alpha=0.03, rasterized=True)
        ax_sc.set_xlim(t0, t1)
        ax_sc.set_ylim(*_depth_ylim)
        ax_sc.set_xlabel("Time (s)", fontsize=11)
        ax_sc.set_ylabel("Snapped depth (µm)", fontsize=11)
        ax_sc.set_title(f"Snapped ({corr_label}), depth-ordered  [{t0:.0f}–{t1:.0f} s]", fontsize=12)
        ax_sc.grid(True, alpha=0.12)
        ax_sc.text(0.01, 0.99, f"N = {m.sum():,} peaks",
                   transform=ax_sc.transAxes, va="top", fontsize=8)
        ax_bar.barh(unique_ch_depths, win_counts_depth, height=_bar_h_depth,
                    color="steelblue", alpha=0.75, edgecolor="none")
        ax_bar.set_ylim(*_depth_ylim)
        ax_bar.set_xlabel(win_count_label, fontsize=9)
        ax_bar.set_yticklabels([])
        ax_bar.yaxis.set_tick_params(left=False)
        ax_bar.set_title(win_count_label, fontsize=9)
        ax_bar.grid(True, alpha=0.15, axis="x")
        save_fig(fig, PLOTS_DIR / f"04_snapped_scatter_depth_{i+1:02d}_t{t0:.0f}-{t1:.0f}s.png")

        # channel-id-ordered
        fig = plt.figure(figsize=(22, 8))
        gs  = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.04)
        ax_sc  = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1])

        ax_sc.scatter(pt, snapped_ch_idx[m], s=1, color="k", alpha=0.03, rasterized=True)
        ax_sc.set_xlim(t0, t1)
        ax_sc.set_ylim(*_id_ylim)
        ax_sc.set_xlabel("Time (s)", fontsize=11)
        ax_sc.set_ylabel("Channel ID", fontsize=11)
        ax_sc.set_title(f"Snapped ({corr_label}), channel-id-ordered  [{t0:.0f}–{t1:.0f} s]", fontsize=12)
        ax_sc.grid(True, alpha=0.12)
        ax_sc.text(0.01, 0.99, f"N = {m.sum():,} peaks",
                   transform=ax_sc.transAxes, va="top", fontsize=8)
        ax_bar.barh(_ch_ids, win_counts_id, height=_bar_h_id,
                    color="steelblue", alpha=0.75, edgecolor="none")
        ax_bar.set_ylim(*_id_ylim)
        ax_bar.set_xlabel(win_count_label, fontsize=9)
        ax_bar.set_yticklabels([])
        ax_bar.yaxis.set_tick_params(left=False)
        ax_bar.set_title(win_count_label, fontsize=9)
        ax_bar.grid(True, alpha=0.15, axis="x")
        save_fig(fig, PLOTS_DIR / f"04_snapped_scatter_chid_{i+1:02d}_t{t0:.0f}-{t1:.0f}s.png")

    print(f"\n  Plotting channel density histograms …")
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.barh(unique_ch_depths, counts_for_plot_depth, height=_bar_h_depth,
            color="steelblue", alpha=0.8, edgecolor="none")
    ax.set_xlabel(f"{count_label}  [{_analysis_start:.0f}–{_analysis_end:.0f} s]", fontsize=11)
    ax.set_ylabel("Probe depth (µm)", fontsize=11)
    ax.set_ylim(*_depth_ylim)
    ax.set_title(f"Spike density per channel (depth-ordered)\n{corr_label}", fontsize=12)
    ax.grid(True, alpha=0.15, axis="x")
    plt.tight_layout()
    save_fig(fig, PLOTS_DIR / f"04_channel_density_depth_t{_analysis_start:.0f}-{_analysis_end:.0f}s.png")

    fig, ax = plt.subplots(figsize=(7, 9))
    ax.barh(_ch_ids, counts_for_plot_id, height=_bar_h_id,
            color="steelblue", alpha=0.8, edgecolor="none")
    ax.set_xlabel(f"{count_label}  [{_analysis_start:.0f}–{_analysis_end:.0f} s]", fontsize=11)
    ax.set_ylabel("Channel ID", fontsize=11)
    ax.set_ylim(*_id_ylim)
    ax.set_title(f"Spike density per channel (channel-id-ordered)\n{corr_label}", fontsize=12)
    ax.grid(True, alpha=0.15, axis="x")
    plt.tight_layout()
    save_fig(fig, PLOTS_DIR / f"04_channel_density_chid_t{_analysis_start:.0f}-{_analysis_end:.0f}s.png")


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