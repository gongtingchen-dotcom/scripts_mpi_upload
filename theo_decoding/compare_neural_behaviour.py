# %%

"""SUPPLEMENTARY analysis tool -- not run as part of a search.

Same load + preprocess pipeline as theo_pipeline_new.py / theo_data_prep_demo.py
(GLC7189/c_g0, with the full set of diagnostic QC plots restored), plus two
extra analysis steps:

  1. PCA on the binned per-channel firing rates (X) -> top 50 neural PCs,
     saved to disk.
  2. A single time-aligned comparison figure with 4 stacked panels: the
     drift-corrected neural scatter, the first 3 texture (behavior) PCs, the
     binned firing-rate heatmap, and the first 3 neural PCs from step 1 --
     all sharing one time axis, with the CV fold boundaries marked on every
     panel.

See theo_pipeline_new.py for the general, CLI-configurable version.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import spikeinterface.full as si
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from Neural_Decoding.preprocessing_funcs import LagMat, bin_spikes
# %%

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


# --- Diagnostic plots (verbatim from theo_pipeline_new.py, except
# plot_density_before_after which drops its "after smoothing" panel since
# channel smoothing isn't used in this run) -----------------------------------

def plot_spike_count_histogram(out_path, channel_spike_counts, n_channels, spike_thresh):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(channel_spike_counts, bins=40, color="#3B6FA0", edgecolor="white")
    axes[0].set_xlabel("spike count (whole window)")
    axes[0].set_ylabel("# channels")
    axes[0].set_title("Linear scale")

    axes[1].hist(
        channel_spike_counts,
        bins=np.logspace(np.log10(max(channel_spike_counts.min(), 1)), np.log10(channel_spike_counts.max()), 40),
        color="#3B6FA0", edgecolor="white",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("spike count (whole window, log scale)")
    axes[1].set_title("Log scale -- easier to see the low-count tail")

    if spike_thresh > 0:
        n_kept = int((channel_spike_counts >= spike_thresh).sum())
        pct_lost = 100 * channel_spike_counts[channel_spike_counts < spike_thresh].sum() / channel_spike_counts.sum()
        label = f"spike_thresh={spike_thresh:,} -> keeps {n_kept}/{n_channels} channels, loses {pct_lost:.1f}% of spikes"
        for ax in axes:
            ax.axvline(spike_thresh, color="red", linestyle="--", linewidth=1.5, label=label)
            ax.legend(fontsize=7.5, loc="upper right")

    fig.suptitle(f"Per-channel spike counts, {n_channels} channels")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_density_before_after(out_path, active_channel_idx, channel_positions, active_channel_counts, duration_s):
    active_y = channel_positions[active_channel_idx, 1]
    unique_y = np.sort(np.unique(active_y))
    row_pitch = float(np.min(np.diff(unique_y))) if unique_y.size > 1 else 20.0
    row_rate = np.array([active_channel_counts[active_y == y].sum() for y in unique_y]) / duration_s

    fig, ax = plt.subplots(figsize=(5.5, 8))
    ax.barh(unique_y, row_rate, height=row_pitch * 0.9, color="steelblue", edgecolor="none")
    ax.set_xlabel("Spike rate (spikes/s)")
    ax.set_ylabel("Depth (um)")
    ax.set_title(f"Spike density along probe depth\n({unique_y.size} depth rows, ~{row_pitch:.0f}um pitch)")
    ax.grid(alpha=0.2, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_drift_scatter(out_path, peak_times_abs, depths, title, fold_boundary_times):
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.scatter(peak_times_abs, depths, s=1, color="k", alpha=0.02, rasterized=True)
    for bt in fold_boundary_times:
        ax.axvline(bt, color="tab:red", lw=0.8, ls=":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Depth (um)")
    ax.set_title(title)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_motion_overlay(out_path, peak_times_abs, depths, motion_obj, fold_boundary_times):
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.scatter(peak_times_abs, depths, s=1, color="k", alpha=0.02, rasterized=True, label="Peaks (localized depth)")

    colors = ["red", "royalblue", "darkgreen", "darkorange", "deeppink", "saddlebrown", "teal"]
    t_bins_abs = motion_obj.temporal_bins_s[0]
    d_bins = motion_obj.spatial_bins_um
    displ = motion_obj.displacement[0]
    for di, start_d in enumerate(d_bins):
        drift_line = start_d + displ[:, di]
        ax.plot(
            t_bins_abs, drift_line,
            color=colors[di % len(colors)], lw=1.0, alpha=0.8,
            label="DREDge motion" if di == 0 else "",
        )
    for bt in fold_boundary_times:
        ax.axvline(bt, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Depth (um)")
    ax.set_title("Raw peak depths + motion (DREDge) overlay")
    ax.legend(loc="upper right", markerscale=8, fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_alignment_check(
    out_path, peak_times_sec, depths, beh_t, beh_outputs, pc_labels,
    neural_start_raw, neural_end_raw, beh_start_raw, beh_end_raw, t_start, t_end,
    seconds_per_inch, plot_dpi,
):
    """Neural scatter + texture PCs, each over its FULL raw time range (not
    trimmed to the overlap) -- lets you visually confirm t_start/t_end were
    picked correctly, by checking that the shaded/excluded edges really are
    the parts that don't line up between the two recordings."""
    t_min = min(neural_start_raw, beh_start_raw)
    t_max = max(neural_end_raw, beh_end_raw)
    fig_width = max(40.0, (t_max - t_min) / seconds_per_inch)

    fig, axes = plt.subplots(
        2, 1, figsize=(fig_width, 9), sharex=True,
        gridspec_kw={"hspace": 0.15, "height_ratios": [1.2, 1]},
    )
    ax_scatter, ax_beh = axes
    TITLE_KW = dict(loc="left", fontsize=13, fontweight="bold")

    ax_scatter.scatter(peak_times_sec, depths, s=1, color="k", alpha=0.02, rasterized=True)
    ax_scatter.set_ylabel("Depth (um)")
    ax_scatter.set_title("1. Neural scatter (raw, uncorrected depth, full recording)", **TITLE_KW)
    ax_scatter.grid(alpha=0.15)

    beh_colors = ["tab:blue", "tab:orange", "tab:green"]
    for i, col in enumerate(pc_labels):
        ax_beh.plot(beh_t, beh_outputs[:, i], color=beh_colors[i], lw=0.6, label=col)
    ax_beh.set_ylabel("Texture PC value")
    ax_beh.set_title("2. Texture (behavior) PCs 1-3 (full session)", **TITLE_KW)
    ax_beh.legend(loc="upper right", fontsize=8, ncol=3)
    ax_beh.grid(alpha=0.15)

    # All boundary events to mark: (time, label, color, linestyle)
    markers = [
        (0.0, "t=0", "black", "-"),
        (neural_start_raw, "neural start", "tab:cyan", ":"),
        (neural_end_raw, "neural end", "tab:cyan", "-."),
        (beh_start_raw, "beh start", "tab:purple", ":"),
        (beh_end_raw, "beh end", "tab:purple", "-."),
        (t_start, "t_start (trim)", "tab:red", "--"),
        (t_end, "t_end (trim)", "tab:red", "--"),
    ]

    for ax in axes:
        ax.axvspan(t_min, t_start, color="gray", alpha=0.15, zorder=0)
        ax.axvspan(t_end, t_max, color="gray", alpha=0.15, zorder=0)
        for time_val, _, color, ls in markers:
            ax.axvline(time_val, color=color, lw=1.5, ls=ls, zorder=3)

    # Text labels directly above the top panel, at each marker's own x
    # position -- a legend is useless on a 40+ inch-wide figure since the
    # line you're looking at could be anywhere along it. Labels are
    # greedily assigned to height levels: a marker reuses the lowest level
    # whose last-placed label is far enough away (in data time) not to
    # collide, so closely spaced markers (e.g. t=0 vs. a trim boundary a
    # few seconds later) fan out instead of overlapping.
    height_levels = [1.10, 1.22, 1.34, 1.46]
    min_gap = (t_max - t_min) * 0.035
    level_last_x = [-np.inf] * len(height_levels)
    for time_val, label, color, _ in sorted(markers, key=lambda m: m[0]):
        level = next((lvl for lvl, last_x in enumerate(level_last_x) if time_val - last_x > min_gap), None)
        if level is None:
            level = int(np.argmin(level_last_x))  # all crowded -- reuse the level idle longest
        level_last_x[level] = time_val
        ax_scatter.annotate(
            f"{label}\n{time_val:.2f}s",
            xy=(time_val, 1.0), xycoords=("data", "axes fraction"),
            xytext=(time_val, height_levels[level]), textcoords=("data", "axes fraction"),
            ha="center", va="bottom", fontsize=7.5, color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8, shrinkA=0, shrinkB=0),
        )

    # x-axis spans from absolute t=0 (not just t_min) so the plot always
    # shows where the session clock's zero point sits relative to the data.
    # Tick labels, though, are relative to t_start (the trim boundary) --
    # every 10s, with t_start itself landing exactly on a "0" tick -- so you
    # can read off "how many seconds before/after the trim point" directly.
    tick_step = 10.0
    xlim_left = min(0.0, t_min)
    k_min = np.floor((xlim_left - t_start) / tick_step)
    k_max = np.ceil((t_max - t_start) / tick_step)
    ticks_abs = t_start + tick_step * np.arange(k_min, k_max + 1)
    tick_labels = [f"{k * tick_step:.0f}" for k in np.arange(k_min, k_max + 1)]
    for ax in axes:
        ax.set_xticks(ticks_abs)
        ax.set_xticklabels(tick_labels, fontsize=6, rotation=90)
    ax_beh.set_xlabel("Time relative to t_start (s)")

    axes[0].set_xlim(xlim_left, t_max)
    fig.suptitle(
        f"Neural/behavior alignment check -- full raw traces, shaded = trimmed off, "
        f"dashed red = t_start/t_end (kept: {t_start:.1f}s to {t_end:.1f}s)",
        y=1.12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=plot_dpi, bbox_inches="tight")
    plt.close(fig)


def plot_firing_rate_heatmap(out_path, X, t_centers, fold_boundary_times):
    # vmax capped at the 99th percentile, not the true max -- a handful of
    # rare outlier bins (e.g. artifact bursts) otherwise wash out everything
    # else on a linear 0-to-max scale (this dataset: median=0, p99=3, but
    # max=48). Outlier bins simply clip to full black.
    vmax = np.percentile(X, 99)
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(
        X.T, aspect="auto", origin="lower", cmap="Greys", vmin=0, vmax=vmax,
        extent=[t_centers[0], t_centers[-1], 0, X.shape[1]],
    )
    for bt in fold_boundary_times:
        ax.axvline(bt, color="tab:red", lw=1, ls=":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title(
        f"MUA spike counts per bin, per channel (white=0 -> black={vmax:.0f}, 99th pct; clipped above)\n"
        "(dotted lines = CV fold boundaries)",
        loc="left",
    )
    fig.colorbar(im, ax=ax, label="Spikes per bin")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

# %%

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
window_before, window_after = 2, 2         # lag window: seconds before/after
bin_current = 1                            # number of bins to include for the "current" time point
n_splits = 20                              # CV fold count -- also drawn as boundary lines on every plot below
holdout_frac = 0                           # fraction of the session's tail reserved, excluded from the search

n_neural_pcs = 50                          # top PCs to keep from the neural firing-rate PCA (step 11)

# --- Comparison-figure size knobs -- change these to change the zoom/size ---
# matplotlib/Agg refuses to rasterize a canvas wider than 2**16 px in either
# dimension, so SECONDS_PER_INCH * PLOT_DPI must stay comfortably above
# duration_s / 65536 for whatever this session's duration turns out to be
# (~2900s here -> ~287in wide at these settings, ~28700px).
SECONDS_PER_INCH = 10   # smaller = more zoomed in (wider figure for the same duration)
FIG_HEIGHT_INCHES = 16  # total height, split across all 4 panels
PLOT_DPI = 100

out_dir = Path("/nexus/posix0/MBR-neuralsystems/ephys/GLC7189/GLC7189_c_g0/theo_pipeline_params_search_lagfixed/compare_neural_behaviour_output")
figs_dir = out_dir / "figures"
figs_dir.mkdir(parents=True, exist_ok=True)
# %%

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

peak_channel_index = peaks["channel_index"].astype(int)
# Two time conventions, deliberately kept separate:
#   peak_times_sec    -- neural clock zeroed at its own first sample (drops
#                        the SpikeGLX `firstSample` offset baked into
#                        rec.get_times()). Used for alignment against
#                        behavior: "behavior starts beh_start_raw seconds
#                        after neural start", with neural start = 0.
#   peak_times_motion -- rec.get_times()'s own clock (i.e. WITH time_offset)
#                        -- the same clock the saved DREDge motion estimate
#                        was computed in (confirmed: motion.temporal_bins_s
#                        starts at ~4.8s, matching rec.get_times()[0]=4.324,
#                        not 0). Drift correction below must keep using
#                        this one, unchanged.
peak_times_sec = peaks["sample_index"].astype(float) / fs
peak_times_motion = peak_times_sec + time_offset
print(f"Loaded {len(peaks):,} peaks across {n_channels} channels.")

# For this dataset peaks["channel_index"] is already a direct row index into
# channel_positions (0 <= index < n_channels) -- theo_pipeline_new.py also
# handles recordings where it's a raw AP channel number instead, remapped
# through channel_ids.
assert peak_channel_index.min() >= 0 and peak_channel_index.max() < n_channels
peak_row_index = peak_channel_index

# %%
# ==============================================================================
# 3. Drift/motion correction inputs
# ==============================================================================
motion = si.Motion.load(str(motion_path))
localized_peaks = np.load(str(localized_peaks_path), allow_pickle=True)
localized_y = localized_peaks["y"].astype(float)
print(f"Loaded motion estimate: {motion_path}")
print(f"Loaded localized peaks: {localized_peaks_path}")

# %%
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

print(beh)

beh_t = beh["time"].dt.total_seconds().to_numpy(dtype=float)  # beh["time"] is a Timedelta relative to session start
beh_outputs = beh[pc_cols].to_numpy(dtype=float)

# %%
# ==============================================================================
# 5. Trim to the neural/behavior time overlap, apply drift correction
# ==============================================================================
neural_start_raw, neural_end_raw = 0.0, float(rec.get_times()[-1] - time_offset)
print(neural_start_raw, neural_end_raw)
beh_start_raw, beh_end_raw = float(np.min(beh_t)), float(np.max(beh_t))
print(beh_start_raw, beh_end_raw)
t_start = max(neural_start_raw, beh_start_raw)
t_end = min(neural_end_raw, beh_end_raw)

plot_alignment_check(
    figs_dir / "alignment_check.png",
    peak_times_sec, localized_y, beh_t, beh_outputs, pc_cols[:3],
    neural_start_raw, neural_end_raw, beh_start_raw, beh_end_raw, t_start, t_end,
    SECONDS_PER_INCH, PLOT_DPI,
)
print(f"Saved alignment check figure -> {figs_dir / 'alignment_check.png'}")

peak_keep = (peak_times_sec >= t_start) & (peak_times_sec < t_end)
peak_times_trim = peak_times_sec[peak_keep]            # alignment clock (neural start = 0) -- feeds binning/plots against behavior
peak_times_motion_trim = peak_times_motion[peak_keep]  # motion clock (rec.get_times()) -- feeds drift correction below
peak_rows_trim = peak_row_index[peak_keep]

# Correct each trimmed peak's depth via the motion estimate, then snap to the
# nearest channel -- that becomes its corrected channel_index. Must use
# peak_times_motion_trim (NOT peak_times_trim) here: the saved motion
# estimate was computed in rec.get_times()'s clock (confirmed:
# motion.temporal_bins_s starts at ~4.8s, matching rec.get_times()[0]=
# 4.324s -- not 0), so looking it up with the offset-free alignment clock
# would silently shift every displacement lookup by time_offset (~4.3s
# here), corrupting the drift correction.
localized_y_trim = localized_y[peak_keep]
raw_positions_trim = np.column_stack([channel_positions[peak_rows_trim, 0], localized_y_trim])
corrected_positions_trim = get_corrected_positions(peak_times_motion_trim, raw_positions_trim, motion)
corrected_depths_trim = corrected_positions_trim[:, 1]
peak_rows_trim_corrected = snap_to_nearest_channel(corrected_positions_trim, channel_positions)
n_reassigned = int(np.sum(peak_rows_trim_corrected != peak_rows_trim))
print(f"Drift correction: {n_reassigned:,}/{len(peak_rows_trim):,} peaks reassigned to a different channel.")
peak_rows_trim = peak_rows_trim_corrected
raw_depths_trim = localized_y_trim  # pre-correction depth, kept for the "raw" plot below

# %%
'''
# NOTE (when re-enabling this block): section 9 below plots peak_times_trim
# against motion_obj / fold_boundary_times. peak_times_trim is now the
# offset-free alignment clock (neural start = 0), but the motion object's
# own clock still has the rec.get_times() offset baked in (see the note by
# get_corrected_positions() above) -- so plot_drift_scatter/plot_motion_overlay
# calls there need peak_times_motion_trim instead, or their x-axis will be
# off by time_offset relative to motion_obj.temporal_bins_s.
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
# (theo_pipeline_new.py also supports an optional --spike-thresh to drop
# further low-firing channels, and an optional --channel-smooth-window to
# pool neighboring channels into overlapping depth windows -- neither is
# used for this run, so both steps are omitted here.)


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
# bin_before/bin_after convert the SECOND-valued window_before/window_after
# into bin counts at this dt. LagMat is applied separately to the search
# block and the holdout block (never to the whole session before splitting),
# so no real neighboring sample can leak across that boundary through its
# own lag window.
bin_before = int(round(window_before / dt))
bin_after = int(round(window_after / dt))
print(f"Lag window: {bin_before + bin_current + bin_after} bins/sample x {active_channel_idx.size} channels "
      f"= {active_channel_idx.size * (bin_before + bin_current + bin_after):,} input features per sample.")

n_holdout = int(round(X.shape[0] * holdout_frac)) if holdout_frac > 0 else 0
holdout_start_idx = X.shape[0] - n_holdout

X_search_raw = X[:holdout_start_idx]
y_search = y[:holdout_start_idx]
t_centers_search = t_centers[:holdout_start_idx]
X_lagged_search = LagMat(bin_before=bin_before, bin_current=bin_current, bin_after=bin_after, flat=True).fit_transform(X_search_raw)

X_lagged_holdout, y_holdout, holdout_start_time = None, None, None
if n_holdout > 0:
    X_lagged_holdout = LagMat(bin_before=bin_before, bin_current=bin_current, bin_after=bin_after, flat=True).fit_transform(X[holdout_start_idx:])
    y_holdout = y[holdout_start_idx:]
    holdout_start_time = float(t_centers[holdout_start_idx])
    print(f"Holdout: reserved last {n_holdout:,}/{X.shape[0]:,} samples starting at t={holdout_start_time:.1f}s, excluded from the search.")

print(f"Ready for decoder training -- X_lagged_search: {X_lagged_search.shape}, y_search: {y_search.shape}")

kf = KFold(n_splits=n_splits, shuffle=False)
fold_starts = [test_idx[0] for _, test_idx in kf.split(X_search_raw)]
fold_boundary_times = t_centers_search[fold_starts[1:]]  # internal boundaries only


# ==============================================================================
# 9. Diagnostic plots -- QC only, don't feed into X/y above
# ==============================================================================
plot_spike_count_histogram(figs_dir / "channel_spike_count_histogram.png", channel_spike_counts, n_channels, spike_thresh=0)
plot_density_before_after(figs_dir / "channel_density.png", active_channel_idx, channel_positions, channel_spike_counts[active_channel_idx], t_end - t_start)
plot_firing_rate_heatmap(figs_dir / "firing_rate_heatmap.png", X, t_centers, fold_boundary_times)
plot_drift_scatter(figs_dir / "drift_01_raw_scatter.png", peak_times_trim, raw_depths_trim, "Raw peak depths (localized, uncorrected)", fold_boundary_times)
plot_motion_overlay(figs_dir / "drift_02_motion_overlay.png", peak_times_trim, raw_depths_trim, motion, fold_boundary_times)
plot_drift_scatter(figs_dir / "drift_03_corrected_scatter.png", peak_times_trim, corrected_depths_trim, "Motion-corrected peak depths (before channel-snap)", fold_boundary_times)
plot_drift_scatter(figs_dir / "drift_04_corrected_snapped_scatter.png", peak_times_trim, channel_positions[peak_rows_trim, 1], "Motion-corrected peak depths (snapped to nearest channel)", fold_boundary_times)
print(f"Saved figures -> {figs_dir}")


# ==============================================================================
# 10. Config summary -- everything needed to reproduce this run
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
with open(out_dir / "config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
print(f"Saved config -> {out_dir / 'config.json'}")


# ==============================================================================
# 11. PCA on the binned firing rates (X): top n_neural_pcs neural PCs
# ==============================================================================
# PCA centers X internally -- no per-channel unit-variance scaling here, so
# channels that fire more (and so carry more absolute spike-count variance)
# contribute more to the top components. (Z-scoring first would instead
# equalize every channel's noise floor to unit variance regardless of firing
# rate, which pulls the top PCs toward cross-channel co-variation patterns
# rather than firing-rate magnitude -- tried both; this dataset's per-channel
# rates span ~0.0-2.3 spikes/bin, so that choice changes top-50 variance
# explained from 79.4% (raw) down to 37.7% (z-scored).)
neural_pca = PCA(n_components=n_neural_pcs, random_state=0)
neural_pc_scores = neural_pca.fit_transform(X)  # (n_samples, n_neural_pcs)
print(
    f"Neural PCA: top {n_neural_pcs} PCs explain "
    f"{100 * neural_pca.explained_variance_ratio_.sum():.1f}% of variance "
    f"(PC1={100 * neural_pca.explained_variance_ratio_[0]:.1f}%, "
    f"PC2={100 * neural_pca.explained_variance_ratio_[1]:.1f}%, "
    f"PC3={100 * neural_pca.explained_variance_ratio_[2]:.1f}%)."
)
neural_pca_path = out_dir / "neural_pca_top50.npz"
np.savez_compressed(
    neural_pca_path,
    scores=neural_pc_scores,               # (n_samples, n_neural_pcs) -- the actual "top 50 PCs" over time
    components=neural_pca.components_,     # (n_neural_pcs, n_channels) loadings
    explained_variance_ratio=neural_pca.explained_variance_ratio_,
    t_centers=t_centers,                   # time (s) of each row in `scores`, for reloading aligned to the neural clock
    active_channel_idx=active_channel_idx,  # which physical channels the loadings' columns correspond to
)
print(f"Saved neural PCA -> {neural_pca_path}")


# ==============================================================================
# 12. Comparison figure: neural scatter / texture PCs / firing rates / neural PCs
# ==============================================================================
fig_width = max(40.0, (t_end - t_start) / SECONDS_PER_INCH)
fig, axes = plt.subplots(
    4, 1, figsize=(fig_width, FIG_HEIGHT_INCHES), sharex=True,
    gridspec_kw={"hspace": 0.12, "height_ratios": [1.2, 1, 1.4, 1]},
)
ax_scatter, ax_beh_pcs, ax_heatmap, ax_neural_pcs = axes

# Titles are left-aligned (loc="left"), not centered -- at ~287in wide, a
# centered title sits far off-screen from wherever you're actually scrolled
# to, so it may as well not exist. Left-aligned keeps it pinned near the
# y-axis, visible no matter how far right you've scrolled.
TITLE_KW = dict(loc="left", fontsize=13, fontweight="bold")

# --- 1. Neural scatter, after drift correction + channel-snap ---
ax_scatter.scatter(peak_times_trim, channel_positions[peak_rows_trim, 1], s=1, color="k", alpha=0.02, rasterized=True)
ax_scatter.set_ylabel("Depth (um)")
ax_scatter.set_title("1. Neural scatter (drift-corrected, snapped to channel)", **TITLE_KW)
ax_scatter.grid(alpha=0.15)

# --- 2. First 3 texture (behavior) PCs, dt-binned (same grid as X/y) ---
beh_colors = ["tab:blue", "tab:orange", "tab:green"]
for i, col in enumerate(pc_cols[:3]):
    ax_beh_pcs.plot(t_centers, y[:, i], color=beh_colors[i], lw=0.6, label=col)
ax_beh_pcs.set_ylabel("Texture PC value")
ax_beh_pcs.set_title("2. Texture (behavior) PCs 1-3", **TITLE_KW)
ax_beh_pcs.legend(loc="upper right", fontsize=8, ncol=3)
ax_beh_pcs.grid(alpha=0.15)

# --- 3. Neural data after binning (per-channel firing rate heatmap) ---
# White (0 spikes) -> black, not viridis -- with most bins near-zero
# (median 0, 99th pct = 3 spikes/bin), viridis's perceptual color range
# made almost the whole panel look like one flat dark blob. vmax is capped
# at the 99th percentile rather than the true max (48, a rare outlier
# burst) so the rest of the panel isn't washed out to near-white; outlier
# bins just clip to full black.
heatmap_vmax = np.percentile(X, 99)
im = ax_heatmap.imshow(
    X.T, aspect="auto", origin="lower", cmap="Greys", vmin=0, vmax=heatmap_vmax,
    extent=[t_centers[0], t_centers[-1], 0, X.shape[1]],
)
ax_heatmap.set_ylabel("Channel")
ax_heatmap.set_title(
    f"3. Neural data after binning (spikes/bin, per channel; white=0 -> black={heatmap_vmax:.0f}, 99th pct)",
    **TITLE_KW,
)

# --- 4. First 3 neural PCs (from step 11) ---
neural_colors = ["tab:red", "tab:purple", "tab:brown"]
for i in range(3):
    ax_neural_pcs.plot(t_centers, neural_pc_scores[:, i], color=neural_colors[i], lw=0.6, label=f"neural PC{i + 1}")
ax_neural_pcs.set_ylabel("Neural PC value")
ax_neural_pcs.set_xlabel("Time (s)")
ax_neural_pcs.set_title("4. Neural PCs 1-3 (from PCA on binned firing rates)", **TITLE_KW)
ax_neural_pcs.legend(loc="upper right", fontsize=8, ncol=3)
ax_neural_pcs.grid(alpha=0.15)

# --- CV fold boundaries, marked on every panel ---
for ax in axes:
    for bt in fold_boundary_times:
        ax.axvline(bt, color="tab:red", lw=0.8, ls=":", alpha=0.7)

axes[0].set_xlim(t_start, t_end)
fig.suptitle(
    f"{animal}/{neural_prefix} ({dataset_id}) -- neural vs. behavior, time-aligned "
    f"(dotted red lines = {n_splits}-fold CV boundaries)",
    y=1.0,
)
fig.tight_layout()
comparison_path = figs_dir / "neural_behaviour_comparison.png"
fig.savefig(comparison_path, dpi=PLOT_DPI, bbox_inches="tight")
plt.close(fig)
print(f"Saved comparison figure -> {comparison_path} ({fig_width:.0f}in wide @ {PLOT_DPI}dpi)")
'''
# %%