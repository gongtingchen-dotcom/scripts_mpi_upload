"""
WORTH CHECKING — possibly superseded. run_dredge_chunked.py (the preferred
pipeline, see its header) has its own self-contained stitch_chunks()
function used by its --stitch mode / run_dredge_stitch.sh, independent from
this file — this script isn't called by anything else here. Not certain
whether this predates that and was folded in, or is still meant to be used
standalone for something the built-in stitching doesn't cover; check before
relying on either one silently doing the same thing.

stitch_motion_chunks.py
=======================
Stitch two overlapping DREDge motion estimations into a single si.Motion object.

Setup:
    chunk0  :  motion_dredge_0000min_0350min   (0 – 21000 s)
    chunk1  :  motion_dredge_0340min_0483min   (20400 – 28980 s)
    overlap :  20400 – 21000 s  (600 s / 10 min)

Stitching logic
---------------
DREDge estimates *relative* displacement; any constant offset between the two
runs is a free parameter.  The correct assumption is:

    "The true corrected depth is the same whether we use chunk0 or chunk1 in
    the overlap window."  That is:
        depth - disp0(t, d) ≈ depth - disp1(t, d)   for t in overlap
    =>  disp1(t, d) - disp0(t, d) ≈ Δ(d)             (constant in time, per spatial bin)

We:
1. Evaluate disp0 and disp1 on their shared spatial grid in the overlap.
2. Compute Δ(d) = mean over overlap time of [disp1 - disp0], per spatial bin.
3. Build a stitched displacement array on a merged time axis:
    - t in chunk0 region (< overlap_start)        → disp0(t, d)
    - t in overlap (chunk0_end – chunk1_start)     → disp0(t, d)   [chunk0 is reference]
    - t beyond overlap (> chunk0_end)              → disp1(t, d) - Δ(d)   [re-anchored]

The result is saved as an si.Motion to:
    chunks_2/motion_dredge_stitched/

Diagnostics
-----------
• Δ(d) profile plotted per spatial bin.
• motion-overlay scatter + after-correction scatter in the stitching window
  (STITCH_VIZ_MARGIN_S before overlap_start  →  STITCH_VIZ_MARGIN_S after overlap_end).

Usage
-----
    python stitch_motion_chunks.py [--viz_margin_s 600] [--no_save]
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import spikeinterface as si
import shutil

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR = Path("/ptmp/gongt/GLC6962_l_g1/preprocessed")
CHUNKS_DIR = BASE_DIR / "thresholding_peaks/motion_estimates/chunks_2"

CHUNK0_NAME = "motion_dredge_0000min_0350min"
CHUNK1_NAME = "motion_dredge_0340min_0483min"
OUTPUT_NAME = "motion_dredge_stitched"

# Peaks / depths for scatter diagnostics (same files correct_motion.py uses)
PEAKS_PATH  = BASE_DIR / "thresholding_peaks/peaks_combined2_sorted.npy"
LOCS_PATH   = BASE_DIR / "thresholding_peaks/localized_peaks_mt_raw.npy"  # or localized_peaks_mt_raw.npy

# How many seconds before/after the overlap window to show in diagnostic plots
STITCH_VIZ_MARGIN_S = 600   # 10 min on each side

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--viz_margin_s', type=float, default=STITCH_VIZ_MARGIN_S)
parser.add_argument('--no_save', action='store_true', help='Skip saving the stitched motion')
args = parser.parse_args()
VIZ_MARGIN = args.viz_margin_s

# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
print("Loading motion chunks …")
m0 = si.Motion.load(CHUNKS_DIR / CHUNK0_NAME)
m1 = si.Motion.load(CHUNKS_DIR / CHUNK1_NAME)

# We assume a single segment (seg 0) and that both estimations share the same
# spatial bins (same probe, same dredge parameters).
t0 = m0.temporal_bins_s[0]   # absolute times, seconds
t1 = m1.temporal_bins_s[0]
d0 = m0.displacement[0]      # (n_time0, n_spatial)
d1 = m1.displacement[0]      # (n_time1, n_spatial)
sp0 = m0.spatial_bins_um     # (n_spatial,)
sp1 = m1.spatial_bins_um

print(f"  chunk0: {t0[0]:.1f} – {t0[-1]:.1f} s  |  shape {d0.shape}")
print(f"  chunk1: {t1[0]:.1f} – {t1[-1]:.1f} s  |  shape {d1.shape}")
print(f"  spatial bins chunk0: {sp0.shape}  chunk1: {sp1.shape}")

# Verify / reconcile spatial grids
if not np.allclose(sp0, sp1, atol=1.0):
    raise ValueError(
        "Spatial bins differ between chunks — check that both were estimated "
        "on the same probe with the same spatial parameters.\n"
        f"  chunk0: {sp0}\n  chunk1: {sp1}"
    )
spatial_bins = sp0   # shared

# ---------------------------------------------------------------------------
# IDENTIFY OVERLAP IN TIME
# ---------------------------------------------------------------------------
overlap_start = t1[0]          # chunk1 begins here
overlap_end   = t0[-1]         # chunk0 ends here

if overlap_start >= overlap_end:
    raise ValueError(
        f"No overlap found: chunk1 starts at {overlap_start:.1f} s but chunk0 "
        f"ends at {overlap_end:.1f} s.  Adjust CHUNK0_NAME / CHUNK1_NAME."
    )

print(f"\nOverlap window: {overlap_start:.1f} – {overlap_end:.1f} s  "
      f"({(overlap_end - overlap_start)/60:.1f} min)")

# Time-bin masks for the overlap region
ov_mask0 = (t0 >= overlap_start) & (t0 <= overlap_end)   # chunk0 bins in overlap
ov_mask1 = (t1 >= overlap_start) & (t1 <= overlap_end)   # chunk1 bins in overlap

print(f"  Overlap bins: chunk0 has {ov_mask0.sum()}, chunk1 has {ov_mask1.sum()}")

# ---------------------------------------------------------------------------
# COMPUTE OFFSET Δ(d)  =  mean_over_overlap [ disp1(t,d) - disp0(t,d) ]
#
# Because the two time-bin grids may not be perfectly aligned, we interpolate
# each chunk onto the overlap bins of the OTHER chunk, then average.
# ---------------------------------------------------------------------------

def interp_disp(t_src, d_src, t_query):
    """Linearly interpolate displacement matrix d_src (n_time, n_spatial)
    from time axis t_src to query times t_query.
    Returns (len(t_query), n_spatial).  Clamps to edge values outside range."""
    idx = np.searchsorted(t_src, t_query)
    idx = np.clip(idx, 1, len(t_src) - 1)
    lo, hi = idx - 1, idx
    t_lo, t_hi = t_src[lo], t_src[hi]
    dt = np.where(t_hi > t_lo, t_hi - t_lo, 1.0)          # avoid /0
    w_hi = ((t_query - t_lo) / dt)[:, None]               # (n_q, 1)
    w_lo = 1.0 - w_hi
    return w_lo * d_src[lo] + w_hi * d_src[hi]            # (n_q, n_spatial)

# Use chunk0's overlap bins as the reference time axis for Δ
t_ov = t0[ov_mask0]
d0_ov = d0[ov_mask0]                          # (n_ov, n_spatial)  — exact
d1_ov = interp_disp(t1, d1, t_ov)             # (n_ov, n_spatial)  — interpolated

delta = np.mean(d1_ov - d0_ov, axis=0)        # (n_spatial,)  — the offset to subtract

print(f"\nΔ(d) per spatial bin (µm) — offset to subtract from chunk1:")
for si_i, (sp, dv) in enumerate(zip(spatial_bins, delta)):
    print(f"  spatial bin {si_i:2d}  ({sp:6.1f} µm) :  Δ = {dv:+.3f} µm")

# Optional: check how much Δ varies in time across the overlap (should be small)
delta_std = np.std(d1_ov - d0_ov, axis=0)
print(f"\nStd of (disp1 - disp0) over overlap time, per spatial bin:")
for si_i, (sp, sv) in enumerate(zip(spatial_bins, delta_std)):
    print(f"  spatial bin {si_i:2d}  ({sp:6.1f} µm) :  std = {sv:.3f} µm")

# ---------------------------------------------------------------------------
# BUILD STITCHED DISPLACEMENT ON A MERGED TIME AXIS
#
# Strategy:
#   • Keep ALL chunk0 time bins (they define the absolute reference).
#   • Append chunk1 time bins that are STRICTLY after chunk0 ends.
#   • For those chunk1 bins, subtract Δ(d) so the corrected trace is continuous.
# ---------------------------------------------------------------------------

# chunk1 bins beyond the overlap (the new territory we need)
post_mask1 = t1 > overlap_end
t_post = t1[post_mask1]
d_post = d1[post_mask1] - delta[None, :]     # subtract per-spatial-bin offset

t_stitched = np.concatenate([t0, t_post])
d_stitched = np.concatenate([d0, d_post], axis=0)

print(f"\nStitched time axis: {t_stitched[0]:.1f} – {t_stitched[-1]:.1f} s  "
      f"({len(t_stitched)} bins, dt ≈ {np.median(np.diff(t_stitched)):.2f} s)")
print(f"Stitched displacement shape: {d_stitched.shape}")

# ---------------------------------------------------------------------------
# SAVE STITCHED MOTION  (si.Motion format)
# ---------------------------------------------------------------------------
out_dir = CHUNKS_DIR / OUTPUT_NAME
if out_dir.exists():
    print(f"Removing existing directory: {out_dir}")
    shutil.rmtree(out_dir)

if not args.no_save:
    motion_stitched = si.Motion(
        displacement=[d_stitched],
        temporal_bins_s=[t_stitched],
        spatial_bins_um=spatial_bins,
        direction=m0.direction,
        interpolation_method=m0.interpolation_method,
    )
    motion_stitched.save(out_dir)
    print(f"\nStitched motion saved → {out_dir}")
else:
    motion_stitched = None
    print("\n[--no_save] skipping save.")

# ---------------------------------------------------------------------------
# DIAGNOSTIC PLOTS
# ---------------------------------------------------------------------------
print("\nGenerating diagnostic plots …")

plots_dir = out_dir / "plots"
plots_dir.mkdir(exist_ok=True)

# ── 1. Δ(d) profile ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.barh(spatial_bins, delta, height=np.diff(spatial_bins).mean() * 0.8, color='steelblue')
ax.axvline(0, color='k', linewidth=0.8)
ax.set_xlabel('Δ displacement (µm)')
ax.set_ylabel('Spatial bin depth (µm)')
ax.set_title('Per-bin offset  Δ(d) = mean[disp1 – disp0]  over overlap')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.barh(spatial_bins, delta_std, height=np.diff(spatial_bins).mean() * 0.8, color='tomato')
ax.axvline(0, color='k', linewidth=0.8)
ax.set_xlabel('Std of (disp1 – disp0) over overlap (µm)')
ax.set_ylabel('Spatial bin depth (µm)')
ax.set_title('Temporal variability of offset in overlap\n(ideally small)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(plots_dir / 'delta_profile.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {plots_dir / 'delta_profile.png'}")

# ── 2. Displacement trace comparison in overlap + surroundings ────────────────
viz_t_start = overlap_start - VIZ_MARGIN
viz_t_end   = overlap_end   + VIZ_MARGIN

DEPTH_COLORS = ['red', 'royalblue', 'darkgreen', 'darkorange', 'deeppink', 'saddlebrown', 'teal']

def _plot_displacement_lines(ax, t_bins, disp, label_prefix, linestyle='-', alpha=0.85):
    """Plot one displacement line per spatial bin on ax."""
    t_mask = (t_bins >= viz_t_start) & (t_bins <= viz_t_end)
    for i, depth in enumerate(spatial_bins):
        color = DEPTH_COLORS[i % len(DEPTH_COLORS)]
        lbl = f'{label_prefix} {depth:.0f}µm' if i == 0 else ''
        ax.plot(t_bins[t_mask], disp[t_mask, i] + depth,
                color=color, linewidth=1.1, alpha=alpha, linestyle=linestyle, label=lbl)

fig, ax = plt.subplots(figsize=(22, 7))
_plot_displacement_lines(ax, t0, d0, 'chunk0', linestyle='-')
_plot_displacement_lines(ax, t1, d1 - delta[None, :], 'chunk1 (re-anchored)', linestyle='--')
ax.axvspan(overlap_start, overlap_end, alpha=0.12, color='gold', label='Overlap window')
ax.axvline(overlap_start, color='goldenrod', linewidth=1.5, linestyle=':')
ax.axvline(overlap_end,   color='goldenrod', linewidth=1.5, linestyle=':')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Depth + displacement (µm)')
ax.set_title(f'Drift lines: chunk0 vs chunk1 re-anchored\n'
             f'Viz window {viz_t_start/60:.1f} – {viz_t_end/60:.1f} min')
ax.set_xlim(viz_t_start, viz_t_end)
ax.legend(loc='upper right', fontsize=7)
ax.grid(True, alpha=0.2)
plt.tight_layout()
fig.savefig(plots_dir / 'displacement_comparison_overlap.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {plots_dir / 'displacement_comparison_overlap.png'}")

# ── 3 & 4. Spike scatter + motion overlay  (raw depths  and  corrected depths)
# These require peaks and localizations — skip gracefully if files not found.
if PEAKS_PATH.exists() and LOCS_PATH.exists():
    print("  Loading peaks for scatter diagnostics …")
    peaks  = np.load(PEAKS_PATH)
    locs   = np.load(LOCS_PATH)
    depths = locs['y']
    fs = si.load(str(BASE_DIR)).sampling_frequency
    time_offset = si.load(str(BASE_DIR)).get_times()[0]

    peak_times_abs = peaks['sample_index'] / fs + time_offset   # absolute seconds

    # Window mask for viz
    vt_mask = (peak_times_abs >= viz_t_start) & (peak_times_abs <= viz_t_end)
    pt_viz  = peak_times_abs[vt_mask] - time_offset   # back to relative for plotting
    dep_viz = depths[vt_mask]

    # ── 3. Raw scatter + overlay of chunk0, chunk1 (re-anchored), stitched ──
    fig, ax = plt.subplots(figsize=(22, 8))
    ax.scatter(pt_viz, dep_viz, s=1, color='k', alpha=0.015, label='Spikes (raw depths)')

    for label, t_bins, disp_arr, ls in [
        ('chunk0',              t0,          d0,                     '-'),
        ('chunk1 (re-anchored)', t1,         d1 - delta[None, :],   '--'),
        ('stitched',             t_stitched, d_stitched,             ':'),
    ]:
        t_rel  = t_bins - time_offset
        t_mask = (t_rel >= pt_viz.min() - 60) & (t_rel <= pt_viz.max() + 60)
        for i, depth in enumerate(spatial_bins):
            color = DEPTH_COLORS[i % len(DEPTH_COLORS)]
            lbl = label if i == 0 else ''
            ax.plot(t_rel[t_mask], depth + disp_arr[t_mask, i],
                    color=color, linewidth=1.0, alpha=0.75, linestyle=ls, label=lbl)

    ax.axvspan(overlap_start - time_offset, overlap_end - time_offset,
               alpha=0.10, color='gold', label='Overlap window')
    ax.set_xlabel('Time (s, relative)')
    ax.set_ylabel('Depth (µm)')
    ax.set_title('Motion overlay: raw spike depths + drift lines\n'
                 f'Window {(viz_t_start-time_offset)/60:.1f}–{(viz_t_end-time_offset)/60:.1f} min (relative)')
    ax.set_xlim(pt_viz.min(), pt_viz.max())
    ax.legend(loc='upper right', fontsize=7, markerscale=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(plots_dir / 'motion_overlay_scatter_stitch_window.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {plots_dir / 'motion_overlay_scatter_stitch_window.png'}")

    # ── 4. After-correction scatter using stitched motion ────────────────────
    # We need motion in absolute time for get_displacement_at_time_and_depth
    if motion_stitched is not None:
        shifts = motion_stitched.get_displacement_at_time_and_depth(
            peaks['sample_index'] / fs + time_offset,
            depths
        )
        corrected_depths = depths - shifts

        cor_dep_viz = corrected_depths[vt_mask]

        fig, axes = plt.subplots(1, 2, figsize=(28, 7))

        axes[0].scatter(pt_viz, dep_viz,      s=1, color='k', alpha=0.015)
        axes[0].axvspan(overlap_start - time_offset, overlap_end - time_offset,
                        alpha=0.10, color='gold')
        axes[0].set_xlabel('Time (s, relative)')
        axes[0].set_ylabel('Depth (µm)')
        axes[0].set_title('Before correction (raw depths)')
        axes[0].set_xlim(pt_viz.min(), pt_viz.max())

        axes[1].scatter(pt_viz, cor_dep_viz,  s=1, color='k', alpha=0.015)
        axes[1].axvspan(overlap_start - time_offset, overlap_end - time_offset,
                        alpha=0.10, color='gold')
        axes[1].set_xlabel('Time (s, relative)')
        axes[1].set_ylabel('Corrected depth (µm)')
        axes[1].set_title('After stitched correction')
        axes[1].set_xlim(pt_viz.min(), pt_viz.max())

        plt.suptitle(f'Stitching diagnostic — stitch window '
                     f'{(viz_t_start-time_offset)/60:.1f}–{(viz_t_end-time_offset)/60:.1f} min')
        plt.tight_layout()
        fig.savefig(plots_dir / 'before_after_correction_stitch_window.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {plots_dir / 'before_after_correction_stitch_window.png'}")
else:
    print(f"  Peaks / locs not found — skipping scatter diagnostics.\n"
          f"    peaks : {PEAKS_PATH}\n    locs  : {LOCS_PATH}")

# ── 5. Full-recording stitched displacement overview ─────────────────────────
fig, ax = plt.subplots(figsize=(28, 7))
for i, depth in enumerate(spatial_bins):
    color = DEPTH_COLORS[i % len(DEPTH_COLORS)]
    ax.plot(t_stitched, depth + d_stitched[:, i],
            color=color, linewidth=0.8, alpha=0.85,
            label=f'{depth:.0f} µm' if i == 0 else '')
ax.axvspan(overlap_start, overlap_end, alpha=0.12, color='gold', label='Overlap window')
ax.axvline(overlap_start, color='goldenrod', linewidth=1.2, linestyle=':')
ax.axvline(overlap_end,   color='goldenrod', linewidth=1.2, linestyle=':')
ax.set_xlabel('Time (s, absolute)')
ax.set_ylabel('Depth + displacement (µm)')
ax.set_title('Stitched motion — full recording')
ax.legend(loc='upper right', fontsize=7)
ax.grid(True, alpha=0.2)
plt.tight_layout()
fig.savefig(plots_dir / 'stitched_motion_full_recording.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {plots_dir / 'stitched_motion_full_recording.png'}")

print("\n✓ Done.  All outputs in:")
print(f"    {out_dir}")