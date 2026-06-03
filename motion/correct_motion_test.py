# %% [markdown]
# #### **IMPORTS & SETUP**

# %%
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import spikeinterface as si
import numpy as np
from pathlib import Path

# Fixed target paths
rec_dir     = '/ptmp/gongt/GLC6962_l_g1/preprocessed'
motion_path = Path(rec_dir) / 'thresholding_peaks/motion_estimates/chunks_1/motion_dredge_0000min_0070min'
peaks_path  = Path(rec_dir) / 'thresholding_peaks/peaks_combined2_sorted.npy'
locs_path   = Path(rec_dir) / 'thresholding_peaks/localized_peaks_mt_raw.npy'

rec    = si.load(rec_dir)
print(rec)
motion = si.Motion.load(motion_path)
peaks  = np.load(peaks_path)
estimated_peak_locations = np.load(locs_path)
depths = estimated_peak_locations['y']

print(f"Peaks: {len(peaks):,}  |  fields: {peaks.dtype.names}")
print(f"spatial_bins_um: {motion.spatial_bins_um}")
print(f"n_spatial_windows: {len(motion.spatial_bins_um)}")
print(f"displacement shape: {motion.displacement[0].shape}")

# Time offset (recording absolute start)
time_offset = rec.get_times()[0]
print(f"t_start (time_offset): {time_offset:.3f} s")

plot_start_time = float(motion.temporal_bins_s[0][0])  - time_offset
plot_end_time   = float(motion.temporal_bins_s[0][-1]) - time_offset
print(f"Plot window: {plot_start_time:.1f} s – {plot_end_time:.1f} s  (relative)")

# Amplitude filter: keep top 80% by amplitude (same 20th-pct threshold as motion estimation)
_amp_threshold = np.percentile(np.abs(peaks['amplitude']), 20)
_amp_mask      = np.abs(peaks['amplitude']) >= _amp_threshold
print(f"Amplitude filter: top 80%  (threshold={_amp_threshold:.4f}, kept {_amp_mask.sum():,}/{len(peaks):,})")
peaks_filt  = peaks[_amp_mask]
depths_filt = depths[_amp_mask]

_amp_vmin, _amp_vmax = np.percentile(np.abs(peaks['amplitude']), [5, 95])
print(f"Global amplitude scale: vmin={_amp_vmin:.4f}, vmax={_amp_vmax:.4f}")

# %% [markdown]
# #### **HELPER FUNCTIONS**

# %%
def _peak_times(peaks, fs):
    return peaks['sample_index'] / fs


def _draw_spatial_windows(ax, d_bins, x_start, x_end):
    """Draw alternating shaded bands and centre-depth labels for each spatial window.

    Bands span each window's depth range (centre ± half-spacing).
    Odd-indexed windows get a faint blue tint; even-indexed are plain white.
    Each band is labelled on the right y-axis with its centre depth.
    """
    spacing = float(np.median(np.diff(d_bins)))
    half    = spacing / 2.0
    ax2     = ax.twinx()         # right-axis carries window labels
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(d_bins)
    ax2.set_yticklabels([f'{d:.0f} µm' for d in d_bins], fontsize=7)
    ax2.tick_params(axis='y', length=0, pad=2)
    ax2.set_ylabel('Spatial window centres', fontsize=8)

    for i, center in enumerate(d_bins):
        lo, hi = center - half, center + half
        color  = '#e8f0ff' if i % 2 == 1 else 'white'
        ax.axhspan(lo, hi, facecolor=color, alpha=0.35, zorder=0)
        ax.axhline(lo, color='#aaaaaa', linewidth=0.4, linestyle='--', zorder=1)
    # top edge of last window
    ax.axhline(d_bins[-1] + half, color='#aaaaaa', linewidth=0.4, linestyle='--', zorder=1)


def plot_motion_overlay_scatter(peaks, depths, motion, sampling_frequency,
                                time_offset, start_time, end_time,
                                save_name=None, title=None):
    """Basic motion overlay scatter (black dots + cycling-colour depth traces)."""
    if save_name is not None and Path(save_name).exists():
        print(f"  [skip] {Path(save_name).name} already exists")
        return

    DEPTH_COLORS = ['red', 'royalblue', 'darkgreen', 'darkorange',
                    'deeppink', 'saddlebrown', 'teal']

    t_bins  = motion.temporal_bins_s[0] - time_offset
    d_bins  = motion.spatial_bins_um
    dshift_ = motion.displacement[0]

    pt = _peak_times(peaks, sampling_frequency)
    t_mask = (pt >= start_time) & (pt <= end_time)

    fig, ax = plt.subplots(figsize=(50, 15))
    ax.scatter(pt[t_mask], depths[t_mask], s=1, color='k', alpha=0.01, label='peaks')

    t_mask_lines = (t_bins >= start_time) & (t_bins <= end_time)
    for i, d in enumerate(d_bins):
        color = DEPTH_COLORS[i % len(DEPTH_COLORS)]
        ax.plot(t_bins[t_mask_lines], d + dshift_[t_mask_lines, i],
                color=color, linewidth=1.2, alpha=0.8,
                label=f'{d:.0f} µm' if i == 0 else '')

    _draw_spatial_windows(ax, d_bins, start_time, end_time)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Depth (µm)')
    ax.set_xlim(start_time, end_time)
    ax.set_title(title or f'Motion overlay scatter  [{start_time:.0f}s – {end_time:.0f}s]')
    ax.legend(loc='upper right', markerscale=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    if save_name is not None:
        fig.savefig(save_name, dpi=150, bbox_inches='tight')
        print(f"  saved: {Path(save_name).name}")
    plt.close(fig)


def compute_window_stats(peaks, depths, motion, sampling_frequency,
                         start_time, end_time, amp_pct_filter=20):
    """For each spatial window compute mean spike |amplitude| and spike count.

    ``peaks['amplitude']`` is the spike peak amplitude (in units of std)
    detected by SpikeInterface – NOT a displacement amplitude.

    Returns
    -------
    window_mean_amp : float array (n_windows,)  -- mean spike |amplitude| per window
    window_n_spikes : int   array (n_windows,)  -- number of spikes per window
    """
    d_bins   = motion.spatial_bins_um
    n_win    = len(d_bins)
    spacing  = float(np.median(np.diff(d_bins)))
    half     = spacing / 2.0

    pt = _peak_times(peaks, sampling_frequency)
    t_mask = (pt >= start_time) & (pt <= end_time)
    amps_all   = np.abs(peaks['amplitude'][t_mask])
    depths_sub = depths[t_mask]

    window_mean_amp = np.full(n_win, np.nan)
    window_n_spikes = np.zeros(n_win, dtype=int)

    for i, center in enumerate(d_bins):
        w_mask = np.abs(depths_sub - center) <= half
        window_n_spikes[i] = w_mask.sum()
        if w_mask.any():
            window_mean_amp[i] = amps_all[w_mask].mean()

    return window_mean_amp, window_n_spikes


def plot_window_stats_scatter(window_mean_amp, window_n_spikes, d_bins,
                               amp_thresh=None, nspikes_thresh=None,
                               locally_only_pass_mask=None,
                               title=None,
                               save_name=None):
    """Scatter: x = n_spikes per window, y = mean spike |amplitude| per window.
    Points are coloured by depth and labelled with depth.
    If amp_thresh / nspikes_thresh are given, threshold lines are drawn and the
    region that fails BOTH thresholds (filtered-off under OR logic) is shaded.
    locally_only_pass_mask : bool array – windows that pass THIS chunk's filter
    but fail in at least one other chunk; these are marked with a red triangle.
    """
    if save_name is not None and Path(save_name).exists():
        print(f"  [skip] {Path(save_name).name} already exists")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Shaded region: fails BOTH thresholds (filtered-off under OR logic: need amp>=thresh OR n>=thresh to pass)
    if amp_thresh is not None and nspikes_thresh is not None:
        x_min = 0
        x_max = window_n_spikes.max() * 1.15
        y_min = window_mean_amp.min() * 0.97
        y_max = window_mean_amp.max() * 1.03
        # filtered-off zone: amp < thresh AND n_spikes < thresh (bottom-left rectangle)
        ax.fill_between([x_min, nspikes_thresh], y_min, amp_thresh,
                        color='salmon', alpha=0.25, zorder=0,
                        label='filtered off (amp < thresh AND n_spikes < thresh)')
        ax.axvline(nspikes_thresh, color='tomato', linewidth=1.2,
                   linestyle='--', label=f'n_spikes = {nspikes_thresh:,}')
        ax.axhline(amp_thresh, color='darkorange', linewidth=1.2,
                   linestyle='--', label=f'mean amp = {amp_thresh}')

    sc = ax.scatter(window_n_spikes, window_mean_amp, c=d_bins,
                    cmap='viridis', s=80, zorder=3)
    plt.colorbar(sc, ax=ax, label='Spatial window centre (µm)')

    # Red triangles: passes locally but is filtered off in some other chunk
    if locally_only_pass_mask is not None and locally_only_pass_mask.any():
        ax.scatter(window_n_spikes[locally_only_pass_mask],
                   window_mean_amp[locally_only_pass_mask],
                   marker='^', s=120, facecolors='none', edgecolors='red',
                   linewidths=1.5, zorder=4,
                   label='passes here, filtered in another chunk')

    for i, (ns, ma, d) in enumerate(zip(window_n_spikes, window_mean_amp, d_bins)):
        ax.annotate(f'{d:.0f}', (ns, ma), fontsize=7, ha='left', va='bottom',
                    xytext=(3, 3), textcoords='offset points')

    ax.set_xlabel('Spike count in window')
    ax.set_ylabel('Mean spike |amplitude| (a.u., units of std)')
    ax.set_title(title or ('Per-spatial-window: spike count vs mean spike amplitude\n'
                           '(each point = one spatial window; colour = depth; label = window centre in µm)'))
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_name is not None:
        fig.savefig(save_name, dpi=150, bbox_inches='tight')
        print(f"  saved: {Path(save_name).name}")
    plt.close(fig)


def plot_motion_overlay_reliability(peaks, depths, motion, sampling_frequency,
                                     time_offset, start_time, end_time,
                                     window_values, cmap_name='RdYlGn',
                                     cbar_label='reliability score',
                                     window_mask=None,
                                     title=None, save_name=None):
    """Motion overlay scatter where each depth-window trace is coloured by
    a per-window reliability value (e.g. mean amplitude, n_spikes, or combined).

    Parameters
    ----------
    window_values : array (n_windows,)
        One value per spatial window.  Will be normalised linearly to [0, 1]
        for the colormap so that the lowest value = bottom of cmap and the
        highest = top.
    window_mask : bool array (n_windows,) or None
        If provided, only windows where mask is True will have their trace
        plotted.  Masked-out windows are silently skipped.
    """
    if save_name is not None and Path(save_name).exists():
        print(f"  [skip] {Path(save_name).name} already exists")
        return

    t_bins  = motion.temporal_bins_s[0] - time_offset
    d_bins  = motion.spatial_bins_um
    dshift_ = motion.displacement[0]

    pt = _peak_times(peaks, sampling_frequency)
    t_mask = (pt >= start_time) & (pt <= end_time)
    t_mask_lines = (t_bins >= start_time) & (t_bins <= end_time)

    # Normalise window_values for colormap (using only unmasked windows)
    v = np.array(window_values, dtype=float)
    active = window_mask if window_mask is not None else np.ones(len(v), dtype=bool)
    v_valid = v[active & np.isfinite(v)]
    vmin, vmax = v_valid.min(), v_valid.max()
    norm    = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap    = plt.get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=(50, 15))
    ax.scatter(pt[t_mask], depths[t_mask], s=1, color='k', alpha=0.01, label='peaks')

    for i, d in enumerate(d_bins):
        if not active[i]:
            continue
        val   = v[i] if np.isfinite(v[i]) else vmin
        color = cmap(norm(val))
        ax.plot(t_bins[t_mask_lines], d + dshift_[t_mask_lines, i],
                color=color, linewidth=2.0, alpha=0.9,
                label=f'{d:.0f} µm (val={v[i]:.3g})')

    # Colorbar for the trace colours
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=cbar_label, fraction=0.015, pad=0.01)

    _draw_spatial_windows(ax, d_bins, start_time, end_time)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Depth (µm)')
    ax.set_xlim(start_time, end_time)
    ax.set_title(title or f'Motion overlay  [{start_time:.0f}s – {end_time:.0f}s]  traces coloured by {cbar_label}')
    ax.legend(loc='upper right', fontsize=7, markerscale=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    if save_name is not None:
        fig.savefig(save_name, dpi=150, bbox_inches='tight')
        print(f"  saved: {Path(save_name).name}")
    plt.close(fig)


# %% [markdown]
# #### **GLOBAL PRE-ANALYSIS**

# ── Plot selection: set True to generate, False to skip ───────────────────────────────────
PLOT_MOTION_OVERLAY_SCATTER  = True   # spike scatter + all drift traces
PLOT_WINDOW_STATS_SCATTER    = False  # per-window spike count vs mean amplitude
PLOT_RELIABILITY_AMP         = False  # drift traces coloured by mean amplitude
PLOT_RELIABILITY_NSPIKES     = False  # drift traces coloured by spike count
PLOT_RELIABILITY_COMBINED    = False  # drift traces coloured by combined reliability
# ──────────────────────────────────────────────────────────────────────────────

# Quality thresholds (OR logic: pass if amp >= thresh OR n_spikes >= thresh)
_AMP_THRESH     = 3.4
_NSPIKES_THRESH = 500_000
_NEED_STATS = (PLOT_WINDOW_STATS_SCATTER or PLOT_RELIABILITY_AMP
               or PLOT_RELIABILITY_NSPIKES or PLOT_RELIABILITY_COMBINED)

# %%
print("\n=== Global pre-analysis across all chunks ===")
chunks_dir  = Path(rec_dir) / 'thresholding_peaks/motion_estimates/chunks_1'
motion_dirs = sorted(p for p in chunks_dir.iterdir() if p.is_dir())

fs = rec.sampling_frequency
all_motions    = {}
all_stats      = {}   # name -> (w_amp, w_n, t0, t1)
all_pass_masks = {}   # name -> bool array

for mdir in motion_dirs:
    m  = si.Motion.load(mdir)
    t0 = float(m.temporal_bins_s[0][0])  - time_offset
    t1 = float(m.temporal_bins_s[0][-1]) - time_offset
    all_motions[mdir.name] = m
    all_stats[mdir.name]   = (None, None, t0, t1)
    if _NEED_STATS:
        w_amp, w_n = compute_window_stats(peaks_filt, depths_filt, m, fs,
                                           start_time=t0, end_time=t1)
        all_stats[mdir.name]      = (w_amp, w_n, t0, t1)
        all_pass_masks[mdir.name] = (w_amp >= _AMP_THRESH) | (w_n >= _NSPIKES_THRESH)
        print(f"  {mdir.name}:  {all_pass_masks[mdir.name].sum()} pass / "
              f"{(~all_pass_masks[mdir.name]).sum()} fail")
    else:
        print(f"  {mdir.name}:  loaded")

_ref_d_bins = list(all_motions.values())[0].spatial_bins_um
if _NEED_STATS:
    # Global pass mask: windows that pass the filter in EVERY chunk
    global_pass_mask = np.ones(len(_ref_d_bins), dtype=bool)
    for mask in all_pass_masks.values():
        global_pass_mask &= mask
    print(f"\nGlobally reliable ({global_pass_mask.sum()}/{len(global_pass_mask)} windows): "
          + ', '.join(f'{d:.0f}' for d in _ref_d_bins[global_pass_mask]))
    print("Always filtered out: "
          + ', '.join(f'{d:.0f} µm' for d in _ref_d_bins[~global_pass_mask]))
else:
    global_pass_mask = None

# %% [markdown]
# #### **PER-CHUNK PLOTS**

# %%
for mdir in motion_dirs:
    m                    = all_motions[mdir.name]
    _, _, t0, t1         = all_stats[mdir.name]
    d_bins               = m.spatial_bins_um

    print(f"\n--- {mdir.name}  ({t0:.0f}s \u2013 {t1:.0f}s) ---")

    # 1. Basic overlay scatter
    if PLOT_MOTION_OVERLAY_SCATTER:
        plot_motion_overlay_scatter(
            peaks_filt, depths_filt, m, fs,
            time_offset=time_offset,
            start_time=t0, end_time=t1,
            title=f'{mdir.name}  [{t0:.0f}s \u2013 {t1:.0f}s]  motion overlay',
            save_name=mdir / f'test_motion_overlay_scatter_{t0:.0f}s_{t1:.0f}s.png'
        )

    if _NEED_STATS:
        w_amp, w_n       = all_stats[mdir.name][:2]
        local_pass_mask  = all_pass_masks[mdir.name]
        _norm_amp = (w_amp - np.nanmin(w_amp)) / (np.nanmax(w_amp) - np.nanmin(w_amp) + 1e-12)
        _norm_n   = (w_n   - w_n.min())        / (w_n.max()   - w_n.min()   + 1e-12)
        reliability_score = np.sqrt(_norm_amp * _norm_n)
        locally_only_pass = local_pass_mask & ~global_pass_mask

        # 2. Window stats scatter
        if PLOT_WINDOW_STATS_SCATTER:
            plot_window_stats_scatter(
                w_amp, w_n, d_bins,
                amp_thresh=_AMP_THRESH,
                nspikes_thresh=_NSPIKES_THRESH,
                locally_only_pass_mask=locally_only_pass,
                title=f'{mdir.name}  [{t0:.0f}s \u2013 {t1:.0f}s]\nspike count vs mean spike amplitude',
                save_name=mdir / f'test_window_stats_scatter_{t0:.0f}s_{t1:.0f}s.png'
            )

        # 3a. Coloured by mean amplitude (local pass mask)
        if PLOT_RELIABILITY_AMP:
            plot_motion_overlay_reliability(
                peaks_filt, depths_filt, m, fs,
                time_offset=time_offset, start_time=t0, end_time=t1,
                window_values=w_amp, window_mask=local_pass_mask,
                cmap_name='RdYlGn', cbar_label='mean spike |amplitude|',
                title=f'{mdir.name} \u2013 traces coloured by mean spike amplitude  [{t0:.0f}s \u2013 {t1:.0f}s]',
                save_name=mdir / f'test_motion_overlay_reliability_amp_{t0:.0f}s_{t1:.0f}s.png'
            )

        # 3b. Coloured by spike count (local pass mask)
        if PLOT_RELIABILITY_NSPIKES:
            plot_motion_overlay_reliability(
                peaks_filt, depths_filt, m, fs,
                time_offset=time_offset, start_time=t0, end_time=t1,
                window_values=w_n.astype(float), window_mask=local_pass_mask,
                cmap_name='RdYlGn', cbar_label='spike count',
                title=f'{mdir.name} \u2013 traces coloured by spike count  [{t0:.0f}s \u2013 {t1:.0f}s]',
                save_name=mdir / f'test_motion_overlay_reliability_nspikes_{t0:.0f}s_{t1:.0f}s.png'
            )

        # 3c. Combined reliability – ONLY windows passing in ALL chunks
        if PLOT_RELIABILITY_COMBINED:
            filtered_str = ', '.join(f'{d:.0f}' for d in d_bins[~global_pass_mask])
            plot_motion_overlay_reliability(
                peaks_filt, depths_filt, m, fs,
                time_offset=time_offset, start_time=t0, end_time=t1,
                window_values=reliability_score, window_mask=global_pass_mask,
                cmap_name='RdYlGn',
                cbar_label='reliability score (\u221a(norm_amp \u00d7 norm_n))',
                title=(f'{mdir.name} \u2013 combined reliability, globally reliable windows only '
                       f'({global_pass_mask.sum()}/{len(d_bins)})  [{t0:.0f}s \u2013 {t1:.0f}s]\n'
                       f'excluded: {filtered_str} \u00b5m'),
                save_name=mdir / f'test_motion_overlay_reliability_combined_{t0:.0f}s_{t1:.0f}s.png'
    )

print("\n=== All chunks done ===")

# %% [markdown]
# #### **CROSS-CHUNK SUMMARY**

# %%
if _NEED_STATS:
    print("\n=== Cross-chunk window filter summary ===")
    chunk_failed = {name: set(_ref_d_bins[~mask].tolist())
                    for name, mask in all_pass_masks.items()}

    print(f"\n{'Chunk':<40}  {'t_start':>8}  {'t_end':>8}  "
          f"{'n_pass':>6}  {'n_fail':>6}  filtered windows (depth \u00b5m)")
    print('-' * 120)
    for mdir in motion_dirs:
        name = mdir.name
        _, _, t0, t1 = all_stats[name]
        mask = all_pass_masks[name]
        failed_str = ', '.join(f'{d:.0f}' for d in sorted(chunk_failed[name])) or '(none)'
        print(f"{name:<40}  {t0:>8.0f}  {t1:>8.0f}  "
              f"{mask.sum():>6}  {(~mask).sum():>6}  {failed_str}")

    all_failed_union = sorted(set().union(*chunk_failed.values()))
    print(f"\nUnion of filtered-off: {', '.join(f'{d:.0f} \u00b5m' for d in all_failed_union)}")
print(f"Globally reliable:    {', '.join(f'{d:.0f} \u00b5m' for d in _ref_d_bins[global_pass_mask])}")

print(f"\n{'Depth (\u00b5m)':>12}  {'N chunks filtered':>17}  chunk names")
for d in _ref_d_bins:
    failed_in = [name for name, fs_set in chunk_failed.items() if d in fs_set]
    if failed_in:
        print(f"{d:>12.0f}  {len(failed_in):>17}  {', '.join(failed_in)}")

print("\nDone.")

