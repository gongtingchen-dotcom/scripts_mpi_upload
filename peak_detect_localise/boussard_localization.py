#!/usr/bin/env python3
"""
Comparison method, not the default — see peak_localisation.py for the active
pipeline (monopolar triangulation, a simpler/faster version of this same
method). Runs the full IBL localization pipeline: a neural-network denoiser
on each waveform, then localization from the denoised result. Needs a GPU,
much slower than monopolar triangulation. This is the method Yizi Zhang's
density decoding paper uses. Compared head-to-head against monopolar
triangulation in compare_localisation.py — not meaningfully better for the
extra cost, hence not the default; see peak_detect_localise/README.md for
the full comparison. Kept as the standing comparison, or for cases where the
denoiser might matter more (e.g. noisier data).

No CLI args — edit the hardcoded paths/config below (REC_DIR, PEAKS_DIR, the
RUN_* flags) for the session to process, then submit via
run_boussard_localisation.sh.

OUTPUTS:
  PEAKS_DIR/boussard_localization_all/            — raw per-batch localization
                          results (times_read_*.npy etc., one set per 1s batch)
  PEAKS_DIR/boussard_localization_all_merged/      — merged across all batches
    results_x_merged.npy, results_y_merged.npy, results_z_merged.npy   — each
                          peak's (x, y, z) location in µm (z = depth along probe)
    results_alpha_merged.npy, results_max_ptp_merged.npy   — two alternative
                          amplitude estimates per peak
    times_read.npy       — 1 if that peak's localization passed the denoiser
                          quality threshold, 0 otherwise (used to exclude
                          unreliable localizations downstream)
  PEAKS_DIR/boussard_plots_all/   — QC plots: trough-alignment check,
                          raw-vs-denoised waveforms, drift scatter, feature
                          distributions — visual sanity checks only.

Boussard spike localization pipeline — GLC7189_g_g1  (FULL RECORDING)
Adapted from: https://github.com/int-brain-lab/spikes_localization_registration

Steps:
  1. Load SpikeInterface-detected peaks (full recording)
  2. Verify trough alignment (offset_detector_denoiser = 18)
  3. Run LOCALIZER for N × 1-second batches (N = ceil(recording_duration_s))
  4. Denoiser quality check (batch 0)
  5. Merge per-batch result files
  6. Save localization quality plots

Usage:
  conda activate spikeinterface
  python run_boussard_localization.py
"""

import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive; must come before pyplot import
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec as gridspec
from tqdm import tqdm

sys.path.insert(0, '/u/gongt/spikes_localization_registration')
from localization_pipeline.localizer import LOCALIZER
from localization_pipeline.merge_results import get_merged_arrays


# ── RUN CONTROL ───────────────────────────────────────────────────────────────
# Set any of these to False to skip that stage (e.g. when resuming after a
# partial run or to just redo visualization).

RUN_ALIGN_CHECK     = True   # trough-alignment sanity check (fast, batch 0 only)
RUN_LOCALIZATION    = True   # main 1800-batch localization loop
RUN_DENOISER_CHECK  = True   # raw vs denoised waveform plots (batch 0)
RUN_MERGE           = True   # merge per-batch .npy files into final arrays
RUN_VIZ             = True   # localization quality plots


# ── CONFIG ────────────────────────────────────────────────────────────────────

REPO       = '/u/gongt/spikes_localization_registration'
REC_DIR    = '/ptmp/gongt/GLC7189/GLC7189_g_g1/preprocessed'
PEAKS_DIR  = REC_DIR + '/thresholding_peaks_3.3'
OUT_DIR    = PEAKS_DIR + '/boussard_localization_all'
MERGED_DIR = OUT_DIR + '_merged'
PLOTS_DIR  = PEAKS_DIR + '/boussard_plots_all'

bin_file      = REC_DIR + '/traces_cached_seg0.raw'
dtype_input   = 'float16'
n_channels    = 368
sampling_rate = 30000

denoiser_weights     = REPO + '/pretrained_denoiser/denoise.pt'
fname_templates      = None
n_filters_denoise    = [16, 8, 4]
filter_sizes_denoise = [5, 11, 21]

# geometry: 368-channel post-preprocessed map
# Must use 368-ch: binary has 368 ch; read_waveforms byte offsets =
#   sample_index * n_channels * itemsize — n_channels must match the binary.
# probe.json stores the correct (x,y) for all 368 kept channels.
with open(REC_DIR + '/probe.json') as f:
    _probe = json.load(f)
geom_array = np.array(_probe['probes'][0]['contact_positions'])  # (368, 2)
print(f"geom_array: {geom_array.shape}  "
      f"x={np.unique(geom_array[:,0])}  "
      f"y={geom_array[:,1].min():.0f}–{geom_array[:,1].max():.0f} µm")

# spike index: [sample_index, channel_index], sorted by sample
_peaks = np.load(PEAKS_DIR + '/peaks_combined2_sorted.npy')
spike_index_all = np.stack([_peaks['sample_index'], _peaks['channel_index']], axis=1)
spike_index_all = spike_index_all[spike_index_all[:, 0].argsort()]

# full recording — derive length from binary file size
_filesize     = os.path.getsize(bin_file)
_n_samples    = _filesize // (n_channels * np.dtype(dtype_input).itemsize)
_rec_secs     = _n_samples / sampling_rate
len_recording = int(np.ceil(_rec_secs))   # seconds, rounded up to cover last partial second
n_batches     = len_recording
print(f"Full recording: {_n_samples:,} samples = {_rec_secs:.2f} s  →  {n_batches} batches")

spike_index = spike_index_all   # all peaks, full recording
print(f"All peaks: {len(spike_index):,}")

fname_spike_train = PEAKS_DIR + '/boussard_spike_index_all.npy'
np.save(fname_spike_train, spike_index)
print(f"Saved spike index → {fname_spike_train}")

# denoiser alignment offset
# read_waveforms(center) reads [center-60, center+60] (121 samples).
# Denoiser expects trough at position 42 from window start.
# Our sample_index IS the trough sample:
#   center = trough + offset  →  trough_pos = 60 - offset = 42  →  offset = 18
denoiser_min = 42
detector_min = 60   # trough at window center when passing trough sample as center
offset_detector_denoiser = detector_min - denoiser_min   # = 18
print(f"offset_detector_denoiser = {offset_detector_denoiser}  "
      f"(trough will land at position {60 - offset_detector_denoiser} in the window)")

os.makedirs(PLOTS_DIR, exist_ok=True)

# batch-0 spike times (shared by alignment check and denoiser check)
_b0_mask  = spike_index[:, 0] < sampling_rate
_b0_times = spike_index[_b0_mask, 0]
_b0_chans = spike_index[_b0_mask, 1]
print(f"Batch 0: {len(_b0_times)} spikes")


# ── LOCALIZER ─────────────────────────────────────────────────────────────────

localizer_obj = LOCALIZER(
    bin_file,
    dtype_input,
    fname_spike_train,
    fname_templates,
    geom_array,           # array accepted directly (no need to save to disk)
    denoiser_weights,
    offset_detector_denoiser,
    n_filters=n_filters_denoise,
    filter_sizes=filter_sizes_denoise,
    sampling_rate=sampling_rate,
)
localizer_obj.load_denoiser()


# ── ALIGNMENT CHECK ───────────────────────────────────────────────────────────
# Overlay 20 random batch-0 spike traces at their main channel.
# If offset is correct, every minimum lands at position 42 (dashed line).

if RUN_ALIGN_CHECK:
    N_ALIGN = 20
    _rng    = np.random.default_rng(42)
    _sel    = _rng.choice(len(_b0_times), size=min(N_ALIGN, len(_b0_times)), replace=False)

    _traces, _ptps, _trough_at = [], [], []
    for i in _sel:
        center = int(_b0_times[i] + offset_detector_denoiser)
        wf, skip = localizer_obj.read_waveforms(np.array([center]))
        if len(skip) > 0:
            continue
        mc = int(_b0_chans[i])
        trace = wf[0, :, mc].copy()
        _traces.append(trace)
        _ptps.append(float(np.ptp(trace)))
        _trough_at.append(int(trace.argmin()))

    _tr   = np.array(_traces)
    _ptp  = np.array(_ptps)
    _pn   = (_ptp - _ptp.min()) / (_ptp.max() - _ptp.min() + 1e-12)
    _cmap = plt.colormaps['plasma']
    _tax  = np.arange(121)

    fig, ax = plt.subplots(figsize=(10, 5))

    for trace, pn, ptp_val in zip(_tr, _pn, _ptp):
        c = _cmap(pn)
        ax.plot(_tax, trace, color=c, lw=0.9, alpha=0.85)
        ax.text(122, trace[-1], f'{ptp_val:.3f}', color=c, fontsize=6, va='center')
    ax.axvline(42, color='k', lw=1.5, ls='--', label='expected trough (pos 42)')
    ax.set_xlabel('Sample in window (0–120)', fontsize=12)
    ax.set_ylabel('Amplitude (raw)', fontsize=12)
    ax.set_title(f'Trough alignment check  (offset_detector_denoiser={offset_detector_denoiser})\n'
                 f'Raw waveforms at main channel  (N={len(_tr)})  |  PTP labels on right',
                 fontsize=11)
    ax.legend(fontsize=9)
    _sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(_ptp.min(), _ptp.max()))
    _sm.set_array([])
    plt.colorbar(_sm, ax=ax, label='PTP amplitude')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR + '/alignment_check.png', dpi=150, bbox_inches='tight')
    plt.close()

    _ta = np.array(_trough_at)
    print(f"Trough positions: {sorted(_ta.tolist())}")
    print(f"Mean ± std: {_ta.mean():.1f} ± {_ta.std():.1f}  (target = 42)")
    print(f"All at 42?  {'YES ✓' if (_ta == 42).all() else 'NO ✗ — adjust offset_detector_denoiser'}")
    print(f"Saved → {PLOTS_DIR}/alignment_check.png")


# ── RUN LOCALIZATION ──────────────────────────────────────────────────────────

if RUN_LOCALIZATION:
    os.makedirs(OUT_DIR, exist_ok=True)
    # Resume support: skip batches whose sentinel file already exists
    _done = [i for i in range(n_batches)
             if os.path.exists(os.path.join(OUT_DIR, f'times_read_{i:06d}.npy'))]
    _todo = [i for i in range(n_batches) if i not in set(_done)]
    print(f"Localization: {len(_done)} batches already done, {len(_todo)} remaining.")
    for i in tqdm(_todo, total=n_batches, initial=len(_done)):
        localizer_obj.get_estimate(i, threshold=6, output_directory=OUT_DIR)


# ── DENOISER QUALITY CHECK (batch 0) ─────────────────────────────────────────
# Raw vs denoised waveforms for 5 random spikes × 8 nearest channels.
# Separate scatter + histogram of raw vs denoised PTP across all batch-0 spikes.

if RUN_DENOISER_CHECK:
    N_SHOW    = 5
    N_CH_SHOW = 8
    rng       = np.random.default_rng(42)
    _sel5     = rng.choice(len(_b0_times), size=min(N_SHOW, len(_b0_times)), replace=False)

    fig, axes = plt.subplots(N_SHOW, N_CH_SHOW,
                             figsize=(N_CH_SHOW * 2, N_SHOW * 2.5), sharey='row')
    fig.suptitle("Raw (blue) vs Denoised (red) waveforms\n"
                 "Vertical dashed line = trough (position 42)", fontsize=11)

    for row, (t, mc) in enumerate(zip(_b0_times[_sel5], _b0_chans[_sel5])):
        center = int(t + offset_detector_denoiser)
        raw_wf, skip = localizer_obj.read_waveforms(np.array([center]))
        if skip:
            axes[row, 0].set_title("skipped", fontsize=7)
            continue
        denoised_wf = localizer_obj.denoise_wf_nn_tmp(raw_wf)[0]   # (121, 368)
        raw_wf = raw_wf[0]
        if mc <= N_CH_SHOW // 2:
            chs = np.arange(0, N_CH_SHOW)
        elif mc >= n_channels - N_CH_SHOW // 2:
            chs = np.arange(n_channels - N_CH_SHOW, n_channels)
        else:
            chs = np.arange(mc - N_CH_SHOW // 2, mc + N_CH_SHOW // 2)
        for col, ch in enumerate(chs):
            ax = axes[row, col]
            t_ax = np.arange(121)
            ax.plot(t_ax, raw_wf[:, ch],      color='steelblue', lw=0.8, alpha=0.9,
                    label='raw' if col == 0 else None)
            ax.plot(t_ax, denoised_wf[:, ch], color='crimson',   lw=0.8, alpha=0.9,
                    label='denoised' if col == 0 else None)
            ax.axvline(42, color='gray', lw=0.8, ls='--')
            ax.set_title(f'ch {ch}' + (' ★' if ch == mc else ''), fontsize=7)
            ax.set_xticks([])
            if col > 0:
                ax.set_yticks([])
        axes[row, 0].set_ylabel(f't={t}', fontsize=7)
        axes[row, 0].legend(fontsize=6, loc='upper left')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR + '/denoiser_waveforms.png', dpi=150, bbox_inches='tight')
    plt.close()

    raw_ptp_list, den_ptp_list = [], []
    for t, mc in zip(_b0_times, _b0_chans):
        center = int(t + offset_detector_denoiser)
        wf, skip = localizer_obj.read_waveforms(np.array([center]))
        if skip:
            continue
        den = localizer_obj.denoise_wf_nn_tmp(wf)[0]
        raw_ptp_list.append(np.ptp(wf[0, :, mc]))
        den_ptp_list.append(np.ptp(den[:, mc]))

    raw_ptp = np.array(raw_ptp_list)
    den_ptp = np.array(den_ptp_list)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(raw_ptp, den_ptp, s=3, alpha=0.3, color='steelblue')
    lim = max(raw_ptp.max(), den_ptp.max()) * 1.05
    axes[0].plot([0, lim], [0, lim], 'k--', lw=0.8)
    axes[0].set_xlabel('Raw PTP (main channel)')
    axes[0].set_ylabel('Denoised PTP')
    axes[0].set_title('Denoiser: PTP scatter (batch 0)')
    axes[1].hist(raw_ptp, bins=60, alpha=0.6, label='raw',      color='steelblue', density=True)
    axes[1].hist(den_ptp, bins=60, alpha=0.6, label='denoised', color='crimson',   density=True)
    axes[1].set_xlabel('PTP (main channel)')
    axes[1].set_title('PTP distribution (batch 0)')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR + '/denoiser_ptp.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Batch-0: {len(raw_ptp)} spikes  |  "
          f"mean raw ptp={raw_ptp.mean():.4f}  denoised={den_ptp.mean():.4f}")
    print(f"Saved → {PLOTS_DIR}/denoiser_waveforms.png  denoiser_ptp.png")


# ── MERGE RESULTS ─────────────────────────────────────────────────────────────

if RUN_MERGE:
    # Guard: only merge if all batches completed (merge_results.py crashes on missing files)
    _done_for_merge = [i for i in range(n_batches)
                       if os.path.exists(os.path.join(OUT_DIR, f'times_read_{i:06d}.npy'))]
    if len(_done_for_merge) < n_batches:
        print(f"WARNING: Only {len(_done_for_merge)}/{n_batches} batches done — skipping merge and viz.")
        RUN_MERGE = False
        RUN_VIZ = False
    else:
        os.makedirs(MERGED_DIR, exist_ok=True)
        get_merged_arrays(OUT_DIR, MERGED_DIR, n_batches)
        print(f"Merged → {MERGED_DIR}")


# ── LOCALISATION QUALITY VISUALISATION ────────────────────────────────────────

if RUN_VIZ:
    _merged_sentinel = MERGED_DIR + '/results_max_ptp_merged.npy'
    if not os.path.exists(_merged_sentinel):
        print(f"WARNING: Merged files not found at {MERGED_DIR} — skipping viz.")
        RUN_VIZ = False

if RUN_VIZ:
    ptp_all    = np.load(MERGED_DIR + '/results_max_ptp_merged.npy')
    x_all      = np.load(MERGED_DIR + '/results_x_merged.npy')
    z_all      = np.load(MERGED_DIR + '/results_z_merged.npy')
    y_all      = np.load(MERGED_DIR + '/results_y_merged.npy')
    alpha_all  = np.load(MERGED_DIR + '/results_alpha_merged.npy')
    times_flag = np.load(MERGED_DIR + '/times_read.npy')   # 1 = passed threshold

    good  = (times_flag == 1) & (ptp_all > 0)
    ptp   = ptp_all[good]
    x     = x_all[good]
    z     = z_all[good]
    y     = y_all[good]
    alpha = alpha_all[good]
    spike_times_s = spike_index[good, 0].astype(float) / sampling_rate

    print(f"Good localisations: {good.sum():,} / {len(good):,}")
    print(f"z: {z.min():.0f}–{z.max():.0f} µm  |  y: {y.min():.1f}–{y.max():.1f} µm")

    # 1. Drift scatter: localised z vs time
    ptp_clip = np.clip(ptp, *np.percentile(ptp, [5, 95]))
    ptp_norm = (ptp_clip - ptp_clip.min()) / (ptp_clip.max() - ptp_clip.min())

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    sc = axes[0].scatter(spike_times_s, z, c=ptp_norm, cmap='viridis', s=1, alpha=0.15)
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('Localised depth z (µm)', fontsize=12)
    axes[0].set_title('Drift scatter — localised z', fontsize=12)
    axes[0].set_ylim(-100, geom_array[:, 1].max() + 200)
    plt.colorbar(sc, ax=axes[0], label='PTP (normalised)', shrink=0.7)

    raw_ch_depths = geom_array[spike_index[good, 1], 1]
    sc2 = axes[1].scatter(spike_times_s, raw_ch_depths, c=ptp_norm,
                          cmap='viridis', s=1, alpha=0.15)
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_title('Raw channel depth (for comparison)', fontsize=12)
    plt.colorbar(sc2, ax=axes[1], label='PTP (normalised)', shrink=0.7)

    plt.suptitle('Localised z vs raw channel depth  (first 30 min)', fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR + '/drift_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {PLOTS_DIR}/drift_scatter.png")

    # 2. Feature distributions (y, x, alpha, ptp)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    axes[0, 0].hist(y[y < 500], bins=80, color='steelblue', density=True)
    axes[0, 0].set_xlabel('y — distance from probe (µm)', fontsize=10)
    axes[0, 0].set_title('y distribution (< 500 µm shown)', fontsize=10)
    axes[0, 0].axvline(100, color='r', ls='--', lw=0.8, label='100 µm')
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].hist(x, bins=80, color='steelblue', density=True)
    for xc in np.unique(geom_array[:, 0]):
        axes[0, 1].axvline(xc, color='red', ls='--', lw=0.8)
    axes[0, 1].set_xlabel('x (µm)', fontsize=10)
    axes[0, 1].set_title('x distribution (red = probe columns)', fontsize=10)

    axes[0, 2].hist(np.log10(alpha[alpha > 0] + 1), bins=80, color='steelblue', density=True)
    axes[0, 2].set_xlabel('log10(alpha + 1)', fontsize=10)
    axes[0, 2].set_title('Alpha distribution (log)', fontsize=10)

    axes[1, 0].hist(ptp, bins=80, color='steelblue', density=True)
    axes[1, 0].set_xlabel('Denoised PTP (main channel)', fontsize=10)
    axes[1, 0].set_title('PTP distribution', fontsize=10)

    axes[1, 1].scatter(ptp, alpha, s=1, alpha=0.1, color='steelblue')
    axes[1, 1].set_xlabel('PTP', fontsize=10)
    axes[1, 1].set_ylabel('Alpha', fontsize=10)
    axes[1, 1].set_title('Alpha vs PTP', fontsize=10)

    axes[1, 2].hist(np.clip(x, -20, 70), bins=100, color='steelblue', density=True)
    for xc in np.unique(geom_array[:, 0]):
        axes[1, 2].axvline(xc, color='red', ls='--', lw=0.8)
    axes[1, 2].set_xlabel('x (µm, clipped)', fontsize=10)
    axes[1, 2].set_title('x near probe columns (should peak at 0,16,32,48)', fontsize=10)

    plt.suptitle('Localisation feature distributions (first 30 min)', fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR + '/feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {PLOTS_DIR}/feature_distributions.png")

    # 3. Rescaled-PTP visualisations (ptp clipped at 0.4 for colour range)
    ptp_array   = np.load(MERGED_DIR + '/results_max_ptp_merged.npy')
    idx_good    = np.where(ptp_array != 0)[0]
    x_results   = np.load(MERGED_DIR + '/results_x_merged.npy')[idx_good]
    y_results   = np.load(MERGED_DIR + '/results_y_merged.npy')[idx_good]
    z_results   = np.load(MERGED_DIR + '/results_z_merged.npy')[idx_good]
    times_array = np.load(MERGED_DIR + '/times_read.npy')[idx_good]

    ptp_rescaled = ptp_array[idx_good] - ptp_array[idx_good].min()
    ptp_rescaled = ptp_rescaled / ptp_rescaled.max()
    ptp_rescaled[ptp_rescaled >= 0.4] = 0.4

    plt.hist(ptp_array[idx_good], bins=50)
    plt.title('PTP histogram (raw)')
    plt.savefig(PLOTS_DIR + '/ptp_hist.png', dpi=150, bbox_inches='tight')
    plt.close()

    plt.hist(ptp_rescaled, bins=50)
    plt.title('PTP histogram (rescaled, clipped at 0.4)')
    plt.savefig(PLOTS_DIR + '/ptp_rescaled_hist.png', dpi=150, bbox_inches='tight')
    plt.close()

    vir = plt.colormaps['viridis']
    plt.figure(figsize=(20, 10))
    plt.scatter(times_array, z_results, color=vir(ptp_rescaled), s=1)
    plt.xlabel('Time (batch index)')
    plt.ylabel('z (µm)')
    plt.title('z vs time (rescaled PTP colouring)')
    plt.savefig(PLOTS_DIR + '/drift_scatter_rescaled.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(26, 20))
    spec = gridspec(ncols=8, nrows=1, figure=fig)
    f_ax = fig.add_subplot(spec[0, 1])
    f_ax.set_xlim((-10, 60))
    f_ax.scatter(x_results, z_results, s=2, color=vir(ptp_rescaled), alpha=0.05)
    f_ax.scatter(geom_array[:, 0], geom_array[:, 1],
                 c='orange', label='channels', marker='s', s=10)
    f_ax.set_yticks([])
    f_ax.set_ylim(0, geom_array[:, 1].max() + 200)
    f_ax.set_xlabel('X (µm)', fontsize=22)
    f_ax.set_title('Localised position\nby max-ptp', fontsize=18)
    plt.savefig(PLOTS_DIR + '/xz_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {PLOTS_DIR}/ptp_hist.png  ptp_rescaled_hist.png")
    print(f"Saved → {PLOTS_DIR}/drift_scatter_rescaled.png  xz_scatter.png")

print("Done.")
