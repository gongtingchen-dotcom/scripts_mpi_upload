# %% [markdown]
# #### **IMPORTS**

# %%
import argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument('--chunks_dir', default=None)
_parser.add_argument('--motion_name', default='motion_dredge_0000min_0070min')
_parser.add_argument('--amp_pct', type=int, default=0,
                    help='Minimum amplitude percentile (same as used in run_dredge_chunked.py). 0 = no filter.')
_parser.add_argument('--boussard', action='store_true',
                    help='Load peaks and depths from Boussard localisation outputs '
                         '(boussard_localization_merged/) instead of the standard '
                         'peaks_combined2_sorted + localized_peaks_mt_raw files.')
_args, _ = _parser.parse_known_args()

import spikeinterface as si
import numpy as np
import spikeinterface as si
from pathlib import Path
import os

# Use non-interactive backend when running from the command line with --chunks_dir
if _args.chunks_dir is not None:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Suppress interactive display when running in batch mode (--chunks_dir provided)
if _args.chunks_dir is not None:
    plt.show = lambda: None

# %%
rec_dir = r'/ptmp/gongt/GLC6962_l_g1/preprocessed'
rec = si.load(rec_dir)
print(rec)

_peaks_folder = Path(rec_dir) / "thresholding_peaks"
_boussard_dir = _peaks_folder / "boussard_localization_merged"

if _args.boussard:
    # ── Boussard localisation mode ──────────────────────────────────────────
    _default_chunks = "chunks_b1"
    motion_path = _boussard_dir / "motion_estimates" / (_args.chunks_dir or _default_chunks) / _args.motion_name
    print(f"[correct_motion] Boussard mode — motion: {motion_path}")

    _spike_index = np.load(_peaks_folder / "boussard_spike_index_30min.npy")  # (N, 2): [sample_index, channel_index]
    _valid       = np.load(_boussard_dir / "times_read.npy") == 1
    _order       = np.argsort(_spike_index[_valid, 0], kind='stable')  # ensure sorted by sample_index
    _n = int(_valid.sum())
    peaks = np.zeros(_n, dtype=[('sample_index',  '<i8'), ('channel_index', '<i8'),
                                 ('amplitude',     '<f8'), ('segment_index', '<i8')])
    peaks['sample_index']  = _spike_index[_valid, 0][_order]
    peaks['channel_index'] = _spike_index[_valid, 1][_order]
    peaks['amplitude']     = np.load(_boussard_dir / "results_max_ptp_merged.npy")[_valid][_order]
    peaks['segment_index'] = 0
    depths = np.load(_boussard_dir / "results_z_merged.npy")[_valid][_order]
    print(f"Boussard valid peaks: {_n:,}")
else:
    # ── Standard mode (SpikeInterface peaks + localized_peaks_mt_raw) ──────
    base_dir        = _peaks_folder
    _default_chunks = "chunks_1"
    motion_path = base_dir / "motion_estimates" / (_args.chunks_dir or _default_chunks) / _args.motion_name
    print(f"[correct_motion] Standard mode — motion: {motion_path}")

    peaks                    = np.load(base_dir / "peaks_combined2_sorted.npy")
    estimated_peak_locations = np.load(base_dir / "localized_peaks_mt_raw.npy")
    print(f"Estimated peak locations shape: {estimated_peak_locations.shape}")
    depths = estimated_peak_locations['y']

motion = si.Motion.load(motion_path)
print(f"Peaks: {len(peaks):,}  |  fields: {peaks.dtype.names}")

# Global amplitude colour scale – computed from ALL peaks before any filtering so
# every amplitude-coloured plot shares the same colour axis.
_amp_vmin, _amp_vmax = np.percentile(np.abs(peaks['amplitude']), [5, 95])
print(f"Global amplitude scale: vmin={_amp_vmin:.4f}, vmax={_amp_vmax:.4f}")

# Keep unfiltered copies for the "all spikes" overlay plots
peaks_all  = peaks
depths_all = depths

# Apply the same amplitude filter used during motion estimation (if any)
if _args.amp_pct > 0:
    _amp_threshold = np.percentile(np.abs(peaks['amplitude']), _args.amp_pct)
    _amp_mask = np.abs(peaks['amplitude']) >= _amp_threshold
    print(f"Amplitude filter: top {100 - _args.amp_pct}%  "
          f"(threshold={_amp_threshold:.4f}, kept {_amp_mask.sum():,}/{len(peaks):,})")
    peaks  = peaks[_amp_mask]
    depths = depths[_amp_mask]

# %%
# time_offset: recording's absolute start time (t_start).
# peak_times = sample_index / fs  →  RELATIVE (0 … duration)
# motion.temporal_bins_s[0]       →  ABSOLUTE (t_start … t_start+duration)
# Everything that compares against peak_times must be in relative coords.
time_offset = rec.get_times()[0]
print(f"t_start (time_offset): {time_offset:.3f} s")
print(f"t_end  : {rec.get_times()[-1]:.3f} s")

# Derive plot time range from the motion's temporal bins, converted to relative time.
plot_start_time = float(motion.temporal_bins_s[0][0])  - time_offset
plot_end_time   = float(motion.temporal_bins_s[0][-1]) - time_offset
print(f"Plot window: {plot_start_time:.1f} s – {plot_end_time:.1f} s  (relative)")

# %% [markdown]
# #### **FUNCTIONS**

# %%
def plot_drift_scatter(peaks, depths, sampling_frequency, t_min=None, t_max=None,
                       save_name=None, color_by_amplitude=False, title='Estimated Peak Depths Over Time',
                       amp_vmin=None, amp_vmax=None):
    if save_name is not None and Path(save_name).exists():
        print(f"  [skip] {Path(save_name).name} already exists")
        return
    times = peaks['sample_index'] / sampling_frequency
    mask = np.ones(len(times), dtype=bool)
    if t_min is not None:
        mask &= times >= t_min
    if t_max is not None:
        mask &= times <= t_max

    times  = times[mask]
    depths = depths[mask]

    plt.figure(figsize=(50, 15))
    if color_by_amplitude:
        amps = np.abs(peaks['amplitude'][mask])
        vmin = amp_vmin if amp_vmin is not None else np.percentile(amps, 5)
        vmax = amp_vmax if amp_vmax is not None else np.percentile(amps, 95)
        _ord = np.argsort(amps)  # low-amp first so high-amp drawn on top
        sc = plt.scatter(times[_ord], depths[_ord], s=1, c=amps[_ord], cmap='plasma',
                         vmin=vmin, vmax=vmax, alpha=0.05)
        plt.colorbar(sc, label='|amplitude|')
    else:
        plt.scatter(times, depths, s=1, color='k', alpha=0.01)
    plt.xlabel('Time (s)')
    plt.ylabel('Depth (um)')
    plt.title(title)
    if save_name is not None:
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
    # plt.show()


# %%
def plot_motion_overlay(peaks, depths, motion_list, labels, colors, sampling_frequency,
                        start_time=None, end_time=None, time_offset=0, save_name=None,
                        color_by_amplitude=False, amp_vmin=None, amp_vmax=None):
    """Scatter of peak depths with drift lines overlaid.
    Each item in motion_list can be an si.Motion object or a dict with
    keys 'time_bins', 'depth_bins', 'dshift'.
    Drift lines cycle through distinct colours (each neighbouring window gets
    a different colour; colours repeat after one cycle).
    Multiple motion sources in motion_list are distinguished by linestyle.
    """
    if save_name is not None and Path(save_name).exists():
        print(f"  [skip] {Path(save_name).name} already exists")
        return
    DEPTH_COLORS = [
        'red', 'royalblue', 'darkgreen', 'darkorange', 'deeppink', 'saddlebrown', 'teal'
    ]

    plt.figure(figsize=(50, 15))

    peak_times = peaks['sample_index'] / sampling_frequency

    if start_time is None:
        start_time = peak_times.min()
    if end_time is None:
        end_time = peak_times.max()

    time_mask = (peak_times >= start_time) & (peak_times <= end_time)
    if color_by_amplitude:
        amps = np.abs(peaks['amplitude'][time_mask])
        vmin = amp_vmin if amp_vmin is not None else np.percentile(amps, 5)
        vmax = amp_vmax if amp_vmax is not None else np.percentile(amps, 95)
        _ord = np.argsort(amps)  # low-amp first so high-amp drawn on top
        _pt  = peak_times[time_mask][_ord]
        _dep = depths[time_mask][_ord]
        sc = plt.scatter(_pt, _dep,
                         s=1, c=amps[_ord], cmap='plasma', vmin=vmin, vmax=vmax, alpha=0.05,
                         label='Individual Spikes')
        plt.colorbar(sc, label='|amplitude|')
    else:
        plt.scatter(peak_times[time_mask], depths[time_mask],
                    s=1, color='k', alpha=0.01, label='Individual Spikes')

    linestyles = ['-', '--', ':', '-.']

    for m_idx, motion_src in enumerate(motion_list):
        label_base = labels[m_idx]
        ls = linestyles[m_idx % len(linestyles)]

        if isinstance(motion_src, dict):
            t_bins  = motion_src['time_bins']
            d_bins  = motion_src['depth_bins']
            dshift_ = motion_src['dshift']
        else:  # si.Motion
            t_bins  = motion_src.temporal_bins_s[0] - time_offset
            d_bins  = motion_src.spatial_bins_um
            dshift_ = motion_src.displacement[0]

        t_mask = (t_bins >= start_time) & (t_bins <= end_time)
        for i, start_depth in enumerate(d_bins):
            color = DEPTH_COLORS[i % len(DEPTH_COLORS)]  # cycle: each neighbour gets a different colour
            label = label_base if i == 0 else ""
            drift_line = start_depth + dshift_[:, i]
            plt.plot(t_bins[t_mask], drift_line[t_mask],
                     color=color, linewidth=1.2, alpha=0.8, label=label, linestyle=ls)

    plt.xlabel('Time (s)')
    plt.ylabel('Depth (um)')
    plt.title(f'Motion Estimation  [{start_time:.1f}s - {end_time:.1f}s]')
    plt.xlim(start_time, end_time)
    plt.legend(loc='upper right', markerscale=10)
    plt.grid(True, alpha=0.2)
    if save_name is not None:
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
    # plt.show()


# %%
def get_corrected_depths(peaks, depths, motion_src, sampling_frequency, time_offset=0):
    """Subtract motion shift from depths.
    motion_src can be an si.Motion object or a dict with keys 'time_bins', 'depth_bins', 'dshift'.
    """
    peak_times = peaks['sample_index'] / sampling_frequency
    if isinstance(motion_src, dict):
        t_bins  = motion_src['time_bins']
        d_bins  = motion_src['depth_bins']
        dshift_ = motion_src['dshift']
        # searchsorted: no large broadcast matrix, O(n log m)
        t_idx = np.clip(np.searchsorted(t_bins, peak_times) - 1, 0, len(t_bins) - 1)
        d_idx = np.clip(np.searchsorted(d_bins, depths)     - 1, 0, len(d_bins) - 1)
        shifts = dshift_[t_idx, d_idx]
    else:  # si.Motion
        shifts = motion_src.get_displacement_at_time_and_depth(
            peak_times + time_offset, depths
        )
    return depths - shifts

# %%
def get_channel_corrected_binned(peaks, channel_depths, motion_src, sampling_frequency,
                                  time_offset=0, bin_duration=10.0, smooth_sigma=0.0):
    """Non-rigid, time-binned channel correction.

    Non-rigid: each channel gets its own depth-dependent shift per time bin,
    matching DREDge's spatial_bins_um. Drift is evaluated at every channel's
    actual depth rather than at a single median depth.

    Parameters
    ----------
    bin_duration : float
        Width of time bins in seconds (default 10 s).
    smooth_sigma : float
        Sigma (in bins) for optional Gaussian smoothing of the drift matrix
        over the time axis before rounding. 0 = no smoothing (default).

    Returns
    -------
    corrected_channel : int array (n_peaks,)  -- corrected channel index
    bin_shifts_ref    : int array (n_bins,)   -- integer shift per bin at median depth (diagnostics)
    bin_centers       : float array (n_bins,) -- centre time of each bin
    """
    from scipy.ndimage import gaussian_filter1d

    peak_times = peaks['sample_index'] / sampling_frequency
    n_chan = len(channel_depths)

    sort_idx        = np.argsort(channel_depths)
    sorted_depths   = channel_depths[sort_idx]
    channel_spacing = float(np.median(np.diff(sorted_depths)))

    bin_edges   = np.arange(peak_times.min(), peak_times.max() + bin_duration, bin_duration)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_bins = len(bin_centers)

    # Build drift matrix: shape (n_chan, n_bins)
    # drift_matrix[ch, t] = displacement (um) at channel ch's depth at time bin t
    if isinstance(motion_src, dict):
        t_b     = motion_src['time_bins']
        d_b     = motion_src['depth_bins']
        dshift_ = motion_src['dshift']
        t_idx = np.clip(np.searchsorted(t_b, bin_centers)    - 1, 0, len(t_b) - 1)
        d_idx = np.clip(np.searchsorted(d_b, channel_depths) - 1, 0, len(d_b) - 1)
        drift_matrix = dshift_[np.ix_(t_idx, d_idx)].T   # (n_chan, n_bins)
    else:  # si.Motion -- vectorised over all (channel, bin) pairs
        ch_rep = np.repeat(channel_depths, n_bins)          # n_chan * n_bins
        t_rep  = np.tile(bin_centers + time_offset, n_chan)
        drift_matrix = motion_src.get_displacement_at_time_and_depth(
            t_rep, ch_rep
        ).reshape(n_chan, n_bins)

    if smooth_sigma > 0:
        drift_matrix = gaussian_filter1d(drift_matrix, sigma=smooth_sigma, axis=1)

    # Round drift to nearest integer number of channels  ->  (n_chan, n_bins)
    shifts_matrix = np.round(drift_matrix / channel_spacing).astype(int)

    # Each spike: look up its channel's shift at its time bin
    spike_bin       = np.clip(np.digitize(peak_times, bin_edges) - 1, 0, n_bins - 1)
    ch_idx          = peaks['channel_index']
    shift_per_spike = shifts_matrix[ch_idx, spike_bin]
    corrected_ch    = np.clip(ch_idx - shift_per_spike, 0, n_chan - 1)

    # Return median-channel bin shifts for diagnostics
    median_ch_idx  = np.argmin(np.abs(channel_depths - np.median(channel_depths)))
    bin_shifts_ref = shifts_matrix[median_ch_idx, :]

    return corrected_ch, bin_shifts_ref, bin_centers


def get_channel_corrected_floor_2d(peaks, channel_positions, motion_src, sampling_frequency,
                                    time_offset=0):
    """Per-spike floor-rounded channel correction with 2D nearest-channel reassignment.

    For each spike the motion shift is evaluated at its exact time and channel
    depth (no time-binning).  The shift is truncated toward zero so a spike
    only moves when drift exceeds a FULL channel spacing.  The target channel
    is found via 2D Euclidean distance (dx² + dy²), naturally penalising
    cross-column lateral jumps.

    Returns
    -------
    corrected_ch  : int array (n_peaks,)
    lateral_moved : bool array (n_peaks,) -- True where the assigned column changed
    """
    channel_depths = channel_positions[:, 1]
    channel_xs     = channel_positions[:, 0]
    n_chan = len(channel_depths)

    peak_times  = peaks['sample_index'] / sampling_frequency
    ch_idx      = peaks['channel_index']
    spike_depths = channel_depths[ch_idx]   # depth of each spike's home channel

    sort_idx        = np.argsort(channel_depths)
    sorted_depths   = channel_depths[sort_idx]
    channel_spacing = float(np.median(np.diff(sorted_depths)))

    # Per-spike shift at the spike's own (time, depth)
    if isinstance(motion_src, dict):
        t_b     = motion_src['time_bins']
        d_b     = motion_src['depth_bins']
        dshift_ = motion_src['dshift']
        t_idx = np.clip(np.searchsorted(t_b, peak_times)   - 1, 0, len(t_b) - 1)
        d_idx = np.clip(np.searchsorted(d_b, spike_depths) - 1, 0, len(d_b) - 1)
        shifts_per_spike = dshift_[t_idx, d_idx]
    else:  # si.Motion
        shifts_per_spike = motion_src.get_displacement_at_time_and_depth(
            peak_times + time_offset, spike_depths
        )

    # Truncate toward zero: symmetric dead-zone of ±1 channel spacing.
    # 0.99 channels → 0 steps (no move).  1.01 channels → 1 step.
    floor_steps = np.trunc(shifts_per_spike / channel_spacing).astype(int)  # (n_peaks,)

    # Pre-compute 2D nearest-channel lookup for each unique integer step.
    # reassign_lut[n][c] = nearest channel to (channel_xs[c], channel_depths[c] - n*spacing)
    unique_steps = np.unique(floor_steps)
    reassign_lut = {}
    for n_step in unique_steps:
        target_ys = channel_depths - n_step * channel_spacing   # (n_chan,)
        dy2 = (target_ys[:, None] - channel_depths[None, :]) ** 2  # (n_chan, n_chan)
        dx2 = (channel_xs[:, None] - channel_xs[None, :])  ** 2   # (n_chan, n_chan)
        reassign_lut[n_step] = np.argmin(dy2 + dx2, axis=1)        # (n_chan,)

    # Apply per-spike via disjoint masks over unique step values
    corrected_ch = np.empty(len(peaks), dtype=int)
    for n_step, targets in reassign_lut.items():
        mask = floor_steps == n_step
        corrected_ch[mask] = targets[ch_idx[mask]]

    lateral_moved = channel_xs[corrected_ch] != channel_xs[ch_idx]
    return corrected_ch, lateral_moved


def plot_channel_scatter(peaks, channel_idx, sampling_frequency, t_min=None, t_max=None,
                         save_name=None, title='Channel Index Over Time', color_by_amplitude=False,
                         amp_vmin=None, amp_vmax=None):
    """Scatter plot of channel index (integer) vs time."""
    if save_name is not None and Path(save_name).exists():
        print(f"  [skip] {Path(save_name).name} already exists")
        return
    times = peaks['sample_index'] / sampling_frequency
    mask  = np.ones(len(times), dtype=bool)
    if t_min is not None:
        mask &= times >= t_min
    if t_max is not None:
        mask &= times <= t_max

    plt.figure(figsize=(50, 15))
    if color_by_amplitude:
        amps = np.abs(peaks['amplitude'][mask])
        vmin = amp_vmin if amp_vmin is not None else np.percentile(amps, 5)
        vmax = amp_vmax if amp_vmax is not None else np.percentile(amps, 95)
        _ord = np.argsort(amps)  # low-amp first so high-amp drawn on top
        sc = plt.scatter(times[mask][_ord], channel_idx[mask][_ord], s=1, c=amps[_ord],
                         cmap='plasma', vmin=vmin, vmax=vmax, alpha=0.05)
        plt.colorbar(sc, label='|amplitude|')
    else:
        plt.scatter(times[mask], channel_idx[mask], s=1, color='k', alpha=0.01)
    plt.xlabel('Time (s)')
    plt.ylabel('Channel index')
    plt.title(title)
    if save_name is not None:
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
    # plt.show()


# %% [markdown]
# #### **PLOT SELECTION** — set to True to generate, False to skip

# %%
PLOT_MOTION_OVERLAY_SCATTER         = False   # filtered peaks + drift lines
PLOT_MOTION_OVERLAY_SCATTER_AMP     = True   # filtered peaks + drift lines, coloured by amplitude
PLOT_MOTION_OVERLAY_SCATTER_ALL     = True   # ALL (unfiltered) peaks + drift lines
PLOT_DRIFT_SCATTER                  = True   # filtered peaks, no drift lines
PLOT_DRIFT_SCATTER_AMP              = True   # filtered peaks, coloured by amplitude
PLOT_CORRECTED_SCATTER              = False   # corrected filtered peaks
PLOT_CORRECTED_SCATTER_AMP         = True   # corrected filtered peaks, coloured by amplitude
PLOT_CORRECTED_SCATTER_ALL          = True   # corrected ALL peaks
PLOT_MOTION_OVERLAY_CHANNELS        = False   # channel-based drift overlay
PLOT_MOTION_OVERLAY_CHANNELS_AMP    = False   # channel-based drift overlay, amplitude coloured
PLOT_CORRECTED_CHANNEL              = True   # corrected channel depths
PLOT_CORRECTED_CHANNEL_AMP         = False   # corrected channel depths, amplitude coloured
PLOT_DRIFT_CHANNEL                  = True   # original channel index scatter
PLOT_DRIFT_CHANNEL_AMP             = True   # original channel index scatter, amplitude coloured
PLOT_CORRECTED_CHANNEL_BINNED       = False   # binned-corrected channel index
PLOT_CORRECTED_CHANNEL_BINNED_AMP  = False   # binned-corrected channel index, amplitude coloured
PLOT_CORRECTED_CHANNEL_FLOOR_2D     = False   # floor+2D nearest-channel correction
PLOT_CORRECTED_DRIFT_CHANNEL        = True   # localized corrected depths snapped to nearest channel (all peaks)
PLOT_CORRECTED_DRIFT_CHANNEL_AMP    = True   # same, coloured by amplitude
PLOT_CHANNEL_SUMMARY                = True   # drift-channel vs binned-channel comparison summary

# Derived flags — skip expensive computations when not needed
_NEED_CORRECTED   = (PLOT_CORRECTED_SCATTER or PLOT_CORRECTED_SCATTER_AMP
                     or PLOT_CORRECTED_CHANNEL or PLOT_CORRECTED_CHANNEL_AMP)
_NEED_CORRECTED_ALL = (PLOT_CORRECTED_SCATTER_ALL
                       or PLOT_CORRECTED_DRIFT_CHANNEL or PLOT_CORRECTED_DRIFT_CHANNEL_AMP
                       or PLOT_CHANNEL_SUMMARY)
_NEED_CHANNELS    = (PLOT_MOTION_OVERLAY_CHANNELS or PLOT_MOTION_OVERLAY_CHANNELS_AMP
                     or PLOT_CORRECTED_CHANNEL or PLOT_CORRECTED_CHANNEL_AMP
                     or PLOT_DRIFT_CHANNEL or PLOT_DRIFT_CHANNEL_AMP
                     or PLOT_CORRECTED_CHANNEL_BINNED or PLOT_CORRECTED_CHANNEL_BINNED_AMP
                     or PLOT_CORRECTED_CHANNEL_FLOOR_2D
                     or PLOT_CORRECTED_DRIFT_CHANNEL or PLOT_CORRECTED_DRIFT_CHANNEL_AMP
                     or PLOT_CHANNEL_SUMMARY)
_NEED_BINNED      = (PLOT_CORRECTED_CHANNEL_BINNED or PLOT_CORRECTED_CHANNEL_BINNED_AMP
                     or PLOT_CHANNEL_SUMMARY)

# %% [markdown]
# #### **LOCATIONS**

# %%
if PLOT_MOTION_OVERLAY_SCATTER:
    print("plotting motion_overlay_scatter...")
    plot_motion_overlay(
        peaks,
        depths,
        motion_list=[motion],
        labels=['DREDge'],
        colors=['red'],
        sampling_frequency=rec.sampling_frequency,
        time_offset=time_offset,
        start_time=plot_start_time,
        end_time=plot_end_time,
        save_name=motion_path / f'motion_overlay_scatter_{plot_start_time:.0f}s_{plot_end_time:.0f}s.png'
    )

if PLOT_MOTION_OVERLAY_SCATTER_AMP:
    print("plotting motion_overlay_scatter_amp...")
    plot_motion_overlay(
        peaks,
        depths,
        motion_list=[motion],
        labels=['DREDge'],
        colors=['red'],
        sampling_frequency=rec.sampling_frequency,
        time_offset=time_offset,
        start_time=plot_start_time,
        end_time=plot_end_time,
        color_by_amplitude=True,
        amp_vmin=_amp_vmin, amp_vmax=_amp_vmax,
        save_name=motion_path / f'motion_overlay_scatter_amp_{plot_start_time:.0f}s_{plot_end_time:.0f}s.png'
    )

if PLOT_MOTION_OVERLAY_SCATTER_ALL:
    print("plotting motion_overlay_scatter_all (unfiltered peaks)...")
    plot_motion_overlay(
        peaks_all,
        depths_all,
        motion_list=[motion],
        labels=['DREDge'],
        colors=['red'],
        sampling_frequency=rec.sampling_frequency,
        time_offset=time_offset,
        start_time=plot_start_time,
        end_time=plot_end_time,
        save_name=motion_path / f'motion_overlay_scatter_all_{plot_start_time:.0f}s_{plot_end_time:.0f}s.png'
    )

if _NEED_CORRECTED:
    print("computing corrected depths...")
    corrected_depths = get_corrected_depths(
        peaks,
        depths,
        motion,
        rec.sampling_frequency,
        time_offset=time_offset
    )

if PLOT_DRIFT_SCATTER:
    print("plotting drift_scatter...")
    plot_drift_scatter(peaks, depths, rec.sampling_frequency, t_min=plot_start_time, t_max=plot_end_time,
                       save_name=motion_path / f'drift_scatter_{plot_start_time:.0f}s_{plot_end_time:.0f}s.png')

if PLOT_DRIFT_SCATTER_AMP:
    print("plotting drift_scatter_amp...")
    plot_drift_scatter(peaks, depths, rec.sampling_frequency, t_min=plot_start_time, t_max=plot_end_time,
                       color_by_amplitude=True, title='Estimated Peak Depths Over Time (coloured by amplitude)',
                       amp_vmin=_amp_vmin, amp_vmax=_amp_vmax,
                       save_name=motion_path / f'drift_scatter_amp_{plot_start_time:.0f}s_{plot_end_time:.0f}s.png')

if PLOT_CORRECTED_SCATTER:
    print("plotting corrected_scatter...")
    plot_drift_scatter(peaks, corrected_depths, rec.sampling_frequency, t_min=plot_start_time, t_max=plot_end_time,
                       save_name=motion_path / f'corrected_scatter_{plot_start_time:.0f}s_{plot_end_time:.0f}s.png')

if PLOT_CORRECTED_SCATTER_AMP:
    print("plotting corrected_scatter_amp...")
    plot_drift_scatter(peaks, corrected_depths, rec.sampling_frequency, t_min=plot_start_time, t_max=plot_end_time,
                       color_by_amplitude=True, title='DREDge-Corrected Peak Depths (coloured by amplitude)',
                       amp_vmin=_amp_vmin, amp_vmax=_amp_vmax,
                       save_name=motion_path / f'corrected_scatter_amp_{plot_start_time:.0f}s_{plot_end_time:.0f}s.png')

if _NEED_CORRECTED_ALL:
    print("computing corrected depths (all peaks)...")
    corrected_depths_all = get_corrected_depths(
        peaks_all,
        depths_all,
        motion,
        rec.sampling_frequency,
        time_offset=time_offset
    )

if PLOT_CORRECTED_SCATTER_ALL:
    print("plotting corrected_scatter_all (unfiltered peaks)...")
    plot_drift_scatter(peaks_all, corrected_depths_all, rec.sampling_frequency, t_min=plot_start_time, t_max=plot_end_time,
                       save_name=motion_path / f'corrected_scatter_all_{plot_start_time:.0f}s_{plot_end_time:.0f}s.png')


# %% [markdown]
# #### **CHANNELS**

# %%
if _NEED_CHANNELS:
    channel_positions = rec.get_channel_locations()   # (n_channels, 2)  col 0=x, col 1=y
    channel_depths    = channel_positions[:, 1]        # y-coords in µm
    depths2 = channel_depths[peaks["channel_index"]]   # depth of each peak's max channel
    sort_idx      = np.argsort(channel_depths)
    sorted_depths = channel_depths[sort_idx]

if PLOT_MOTION_OVERLAY_CHANNELS:
    print("plotting motion_overlay_channels...")
    plot_motion_overlay(
        peaks,
        depths2,
        motion_list=[motion],
        labels=['DREDge'],
        colors=['red'],
        sampling_frequency=rec.sampling_frequency,
        time_offset=time_offset,
        start_time=plot_start_time,
        end_time=plot_end_time,
        save_name=motion_path / f'motion_overlay_channels_{plot_start_time:.0f}s_{plot_end_time:.0f}s.png'
    )

if PLOT_MOTION_OVERLAY_CHANNELS_AMP:
    print("plotting motion_overlay_channels_amp...")
    plot_motion_overlay(
        peaks,
        depths2,
        motion_list=[motion],
        labels=['DREDge'],
        colors=['red'],
        sampling_frequency=rec.sampling_frequency,
        time_offset=time_offset,
        start_time=plot_start_time,
        end_time=plot_end_time,
        color_by_amplitude=True,
        amp_vmin=_amp_vmin, amp_vmax=_amp_vmax,
        save_name=motion_path / f'motion_overlay_channels_amp_{plot_start_time:.0f}s_{plot_end_time:.0f}s.png'
    )


# %%
if _NEED_CHANNELS and (PLOT_CORRECTED_CHANNEL or PLOT_CORRECTED_CHANNEL_AMP
                       or PLOT_CORRECTED_CHANNEL_BINNED or PLOT_CORRECTED_CHANNEL_BINNED_AMP):
    print("computing corrected depths at channel locations...")
    corrected_depths2 = get_corrected_depths(
        peaks,
        depths2,
        motion,
        rec.sampling_frequency,
        time_offset=time_offset
    )

# %%
if PLOT_CORRECTED_CHANNEL or PLOT_CORRECTED_CHANNEL_AMP:
    print("reassigning channels based on corrected depths...")
    ins  = np.searchsorted(sorted_depths, corrected_depths2)
    ins  = np.clip(ins, 0, len(sorted_depths) - 1)
    left = np.clip(ins - 1, 0, len(sorted_depths) - 1)
    closer_left     = np.abs(corrected_depths2 - sorted_depths[left]) < np.abs(corrected_depths2 - sorted_depths[ins])
    best_sorted_idx = np.where(closer_left, left, ins)

    corrected_channel2        = sort_idx[best_sorted_idx]
    corrected_depths2_snapped = channel_depths[corrected_channel2]

    changed = corrected_channel2 != peaks['channel_index']
    print(f'Peaks reassigned: {changed.sum():,} / {len(peaks):,}  ({100*changed.mean():.1f}%)')

if PLOT_CORRECTED_CHANNEL:
    print("plotting corrected_channel_scatter...")
    plot_drift_scatter(peaks, corrected_depths2_snapped, rec.sampling_frequency, t_min=plot_start_time, t_max=plot_end_time,
                       save_name=motion_path / f'corrected_channel_{plot_start_time:.0f}s_{plot_end_time:.0f}s.png')

if PLOT_CORRECTED_CHANNEL_AMP:
    print("plotting corrected_channel_scatter_amp...")
    plot_drift_scatter(peaks, corrected_depths2_snapped, rec.sampling_frequency, t_min=plot_start_time, t_max=plot_end_time,
                       color_by_amplitude=True, title='DREDge-Corrected Channel Depths (coloured by amplitude)',
                       amp_vmin=_amp_vmin, amp_vmax=_amp_vmax,
                       save_name=motion_path / f'corrected_channel_amp_{plot_start_time:.0f}s_{plot_end_time:.0f}s.png')

# %%
if _NEED_BINNED:
    # Binned channel correction: all spikes in each 10s window get the same integer
    # channel shift, avoiding the per-spike boundary-flipping problem.
    print("computing binned channel correction...")
    corrected_channel_binned, bin_shifts, bin_centers = get_channel_corrected_binned(
        peaks, channel_depths, motion, rec.sampling_frequency,
        time_offset=time_offset, bin_duration=10
    )

    changed_binned = corrected_channel_binned != peaks['channel_index']
    ch_spacing = float(np.median(np.diff(np.sort(channel_depths))))
    print(f'Channel spacing:          {ch_spacing:.1f} um')
    print(f'Max integer shift:        {np.abs(bin_shifts).max()} channels')
    print(f'Peaks reassigned (binned): {changed_binned.sum():,} / {len(peaks):,}  ({100*changed_binned.mean():.1f}%)')

# %%
# --- channel index scatter: shows discrete assignments directly ---
if PLOT_DRIFT_CHANNEL:
    print("plotting channel index scatter...")
    plot_channel_scatter(peaks, peaks['channel_index'], rec.sampling_frequency,
                         t_min=plot_start_time, t_max=plot_end_time,
                         save_name=motion_path / f'drift_channel_{plot_start_time:.0f}s_{plot_end_time:.0f}s.png',
                         title='Original Channel Index Over Time')

if PLOT_DRIFT_CHANNEL_AMP:
    print("plotting channel index scatter (amplitude)...")
    plot_channel_scatter(peaks, peaks['channel_index'], rec.sampling_frequency,
                         t_min=plot_start_time, t_max=plot_end_time,
                         color_by_amplitude=True, amp_vmin=_amp_vmin, amp_vmax=_amp_vmax,
                         save_name=motion_path / f'drift_channel_amp_{plot_start_time:.0f}s_{plot_end_time:.0f}s.png',
                         title='Original Channel Index Over Time (coloured by amplitude)')

if PLOT_CORRECTED_CHANNEL_BINNED:
    print("plotting corrected channel index scatter...")
    plot_channel_scatter(peaks, corrected_channel_binned, rec.sampling_frequency,
                         t_min=plot_start_time, t_max=plot_end_time,
                         save_name=motion_path / f'corrected_channel_binned_{plot_start_time:.0f}s_{plot_end_time:.0f}s.png',
                         title='Corrected Channel Index Over Time (10 s bins)')

if PLOT_CORRECTED_CHANNEL_BINNED_AMP:
    print("plotting corrected channel index scatter (amplitude)...")
    plot_channel_scatter(peaks, corrected_channel_binned, rec.sampling_frequency,
                         t_min=plot_start_time, t_max=plot_end_time,
                         color_by_amplitude=True, amp_vmin=_amp_vmin, amp_vmax=_amp_vmax,
                     save_name=motion_path / f'corrected_channel_binned_amp_{plot_start_time:.0f}s_{plot_end_time:.0f}s.png',
                     title='Corrected Channel Index Over Time, 10 s bins (coloured by amplitude)')


# %%
# Floor-rounded + 2D nearest-channel correction
if PLOT_CORRECTED_CHANNEL_FLOOR_2D:
    print("computing floor+2D channel correction...")
    corrected_channel_floor_2d, lateral_moved_floor_2d = get_channel_corrected_floor_2d(
        peaks, channel_positions, motion, rec.sampling_frequency,
        time_offset=time_offset
    )
    changed_floor_2d = corrected_channel_floor_2d != peaks['channel_index']
    print(f'Peaks reassigned (floor+2D):    {changed_floor_2d.sum():,} / {len(peaks):,}  ({100*changed_floor_2d.mean():.1f}%)')
    print(f'  of which laterally shifted:   {lateral_moved_floor_2d.sum():,} / {len(peaks):,}  ({100*lateral_moved_floor_2d.mean():.1f}%)')
    print(f'  lateral as share of reassigned: {100 * lateral_moved_floor_2d.sum() / max(changed_floor_2d.sum(), 1):.1f}%')
    print("plotting corrected channel index scatter (floor+2D)...")
    plot_channel_scatter(peaks, corrected_channel_floor_2d, rec.sampling_frequency,
                         t_min=plot_start_time, t_max=plot_end_time,
                         save_name=motion_path / f'corrected_channel_floor_column_{plot_start_time:.0f}s_{plot_end_time:.0f}s.png',
                         title='Corrected Channel Index Over Time (floor+2D, 10 s bins)')


# %%
# ── CORRECTED DRIFT CHANNEL ───────────────────────────────────────────────────
# Snap corrected_depths_all (localized depths corrected for ALL peaks) to the
# nearest probe channel and plot a channel-index scatter.
if PLOT_CORRECTED_DRIFT_CHANNEL or PLOT_CORRECTED_DRIFT_CHANNEL_AMP or PLOT_CHANNEL_SUMMARY:
    print("computing corrected_drift_channel (snap corrected_depths_all → nearest channel)...")
    _ins  = np.searchsorted(sorted_depths, corrected_depths_all)
    _ins  = np.clip(_ins, 0, len(sorted_depths) - 1)
    _left = np.clip(_ins - 1, 0, len(sorted_depths) - 1)
    _closer_left = (np.abs(corrected_depths_all - sorted_depths[_left]) <
                    np.abs(corrected_depths_all - sorted_depths[_ins]))
    _best = np.where(_closer_left, _left, _ins)
    corrected_drift_channel_all = sort_idx[_best]   # (n_peaks_all,) channel indices

if PLOT_CORRECTED_DRIFT_CHANNEL:
    print("plotting corrected_drift_channel...")
    plot_channel_scatter(peaks_all, corrected_drift_channel_all, rec.sampling_frequency,
                         t_min=plot_start_time, t_max=plot_end_time,
                         save_name=motion_path / f'corrected_drift_channel_{plot_start_time:.0f}s_{plot_end_time:.0f}s.png',
                         title='Corrected Drift Channel (localized depth → nearest channel, all peaks)')

if PLOT_CORRECTED_DRIFT_CHANNEL_AMP:
    print("plotting corrected_drift_channel_amp...")
    plot_channel_scatter(peaks_all, corrected_drift_channel_all, rec.sampling_frequency,
                         t_min=plot_start_time, t_max=plot_end_time,
                         color_by_amplitude=True, amp_vmin=_amp_vmin, amp_vmax=_amp_vmax,
                         save_name=motion_path / f'corrected_drift_channel_amp_{plot_start_time:.0f}s_{plot_end_time:.0f}s.png',
                         title='Corrected Drift Channel (coloured by amplitude, all peaks)')

# ── Channel assignment comparison summary ─────────────────────────────────────
if PLOT_CHANNEL_SUMMARY:
    print("computing channel assignment comparison summary...")
    _orig_ch  = peaks_all['channel_index'].astype(int)
    _new_ch   = corrected_drift_channel_all.astype(int)
    _delta    = _new_ch - _orig_ch
    _changed  = _delta != 0

    def _fmt(arr, changed):
        if not changed.any():
            return 0, 0, 0.0
        a = np.abs(arr[changed])
        return int(a.max()), int(a.min()), float(a.mean())

    _max1, _min1, _mean1 = _fmt(_delta, _changed)

    if _args.amp_pct > 0:
        _filt_mask = np.abs(peaks_all['amplitude']) >= np.percentile(
            np.abs(peaks_all['amplitude']), _args.amp_pct)
    else:
        _filt_mask = np.ones(len(peaks_all), dtype=bool)

    _delta2   = corrected_drift_channel_all[_filt_mask].astype(int) - corrected_channel_binned.astype(int)
    _changed2 = _delta2 != 0
    _max2, _min2, _mean2 = _fmt(_delta2, _changed2)

    _summary_lines = [
        "=" * 62,
        "CHANNEL ASSIGNMENT COMPARISON SUMMARY",
        "=" * 62,
        f"  Total peaks (all):           {len(peaks_all):>12,}",
        f"  Peaks used for binned corr:  {_filt_mask.sum():>12,}",
        "",
        "── corrected_drift_channel  vs  ORIGINAL (peaks_combined2) ──",
        f"  Changed:              {_changed.sum():>10,}  ({100*_changed.mean():.2f} %)",
        f"  Largest  shift (ch):  {_max1:>10}",
        f"  Smallest shift (ch):  {_min1:>10}",
        f"  Mean |shift|   (ch):  {_mean1:>10.2f}",
        "",
        "── corrected_drift_channel  vs  CORRECTED_CHANNEL_BINNED ──",
        f"  (filtered-peaks subset, n={_filt_mask.sum():,})",
        f"  Differ:               {_changed2.sum():>10,}  ({100*_changed2.mean():.2f} %)",
        f"  Largest  diff  (ch):  {_max2:>10}",
        f"  Smallest diff  (ch):  {_min2:>10}",
        f"  Mean |diff|    (ch):  {_mean2:>10.2f}",
        "=" * 62,
    ]
    _summary_text = "\n".join(_summary_lines)
    print(_summary_text)

    _summary_path = motion_path / f'corrected_drift_channel_summary_{plot_start_time:.0f}s_{plot_end_time:.0f}s.txt'
    with open(_summary_path, 'w') as _f:
        _f.write(_summary_text + "\n")
    print(f"Summary saved → {_summary_path}")


# %%
# # Export for neural decoding
# out_dir = base_dir
# spike_times    = peaks["sample_index"].astype(np.int64)
# spike_clusters = corrected_channel.astype(np.int32)   # cluster id = motion-corrected channel

# np.save(out_dir / "spike_times.npy",    spike_times)
# np.save(out_dir / "spike_clusters.npy", spike_clusters)
# print(f"Saved spike_times ({spike_times.shape}) and spike_clusters ({spike_clusters.shape})")
# print(f"Unique clusters (channels): {np.unique(spike_clusters).size}")


# %% [markdown]
# #### **KS4 CORRECTION**

# %%
# kilosort_output_dir = r'/ptmp/gongt/GLC7189/GLC7189_g_g1/KS4/nb5_nc32_n_t9_pcs10_car0_clr1_jobs36_thrd8_te360/sorter_output'
# ops = np.load(os.path.join(kilosort_output_dir, 'ops.npy'), allow_pickle=True).item()
# dshift = ops['dshift']                        # (n_batches, n_depth_bins)  shift in um
# depth_bins_ks4 = ops['yblk']                  # (n_depth_bins,) depth in um
# time_bins_ks4 = (np.arange(ops['Nbatches']) + 0.5) * ops['batch_size'] / ops['fs']  # batch centre times in s
# print(f"dshift shape: {dshift.shape}")
# print(f"time_bins range: {time_bins_ks4.min():.1f} – {time_bins_ks4.max():.1f} s")
# print(f"depth_bins range: {depth_bins_ks4.min():.1f} – {depth_bins_ks4.max():.1f} um")

# %%
# plot_motion_overlay(
#     peaks,
#     depths,
#     motion_list=[{"time_bins": time_bins_ks4, "depth_bins": depth_bins_ks4, "dshift": -dshift}],
#     labels=['KS4'],
#     colors=['red'],
#     sampling_frequency=rec.sampling_frequency,
#     start_time=0*60,
#     end_time=0*60 + 15*60,
#     save_name=motion_path / 'motion_overlay_ks4.png'
# )


# %%
# ks4_corrected_depths = get_corrected_depths(
#     peaks,
#     depths,
#     {"time_bins": time_bins_ks4, "depth_bins": depth_bins_ks4, "dshift": -dshift},
#     rec.sampling_frequency
# )

# %%
# plot_drift_scatter(peaks, ks4_corrected_depths, rec.sampling_frequency, t_min=0*60, t_max=15*60,
#                    save_name=motion_path / 'drift_scatter_ks4_corrected.png')



