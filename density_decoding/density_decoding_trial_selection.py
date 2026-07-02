# %% [markdown]
# #### **IMPORTS**

# %%
import sys
import os
import json
import pickle
import datetime

_this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path[:] = [p for p in sys.path if os.path.realpath(p) != _this_dir]

import numpy as np
import pandas as pd
import spikeinterface as si
import matplotlib
matplotlib.use('Agg')    # headless / SLURM: must be called before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter1d
from scipy.stats import zscore
from pandas.core.arrays.categorical import Categorical
from pandas._libs.arrays import NDArrayBacked

# %% [markdown]
# #### **LOAD**

# %%
### for pandas pickle compatibility ###
_orig_categorical_setstate = Categorical.__setstate__

def _patched_categorical_setstate(self, state):
    if not isinstance(state, dict):
        dtype, codes = state
        NDArrayBacked.__init__(self, codes, dtype)
    else:
        _orig_categorical_setstate(self, state)

Categorical.__setstate__ = _patched_categorical_setstate

# %%
### SETTINGS ###

import argparse as _argparse
_ap = _argparse.ArgumentParser(description='Density decoding pipeline')
_ap.add_argument('--animal',        default='GLC6962',
                 help='Animal ID (e.g. GLC6962)')
_ap.add_argument('--exp',           default='_g_g0',
                 help='Experiment suffix with leading underscore (e.g. _g_g0)')
_ap.add_argument('--behavior_var',  default='PC1',
                 help='Behavioral variable to decode: PC1, PC2, texture_speed, …')
_ap.add_argument('--t_trial',       type=float, default=5,
                 help='Trial window length in seconds (default 1.5). '
                      'Ignored when --selected_trials is provided (window length is '
                      'read from the CSV win_duration_sec column).')
_ap.add_argument('--bin_size_ms',   type=float, default=100.0,
                 help='Size of each time bin in milliseconds (default 100 ms). '
                      'N_T_BINS is derived as round(T_TRIAL / bin_size_ms * 1000).')
_ap.add_argument('--learning_rate', type=float, default=1e-3,
                 help='ADVI learning rate (default 1e-3)')
_ap.add_argument('--n_workers',     type=int,   default=16,
                 help='Number of parallel workers (default 16)')
_ap.add_argument('--chunk_style',   choices=['contiguous', 'interleaved'], default='interleaved',
                 help='Train/test split style: contiguous (80-20 at end) or interleaved (distributed, default)')
_ap.add_argument('--test_percentage', type=float, default=20,
                 help='Percentage of data to use for testing (default 20, range 1-99)')
_ap.add_argument('--chunk_size',    type=int,   default=10,
                 help='Trials per chunk for interleaved train/test split (default 50); '
                      'only used when chunk_style=interleaved')
_ap.add_argument('--early_stopping', action=_argparse.BooleanOptionalAction, default=True,
                 help='Enable EMA-patience early stopping in ADVI training (--no-early-stopping to disable)')
_ap.add_argument('--es_patience',   type=int,   default=500,
                 help='Early stopping patience in iterations (default 500)')
_ap.add_argument('--batch_size',    type=int,   default=10,
                 help='ADVI batch size in units of trials (default 10). '
                      'Auto-shrunk to n_train_trials if smaller.')
_ap.add_argument('--zscore_spike_features', action=_argparse.BooleanOptionalAction, default=True,
                 help='Z-score x, z, and ptp independently before GMM/ADVI fitting. '
                      'Recommended when features have very different scales (e.g. z in µm vs ptp in µV). '
                      'Use --no-zscore-spike-features to disable. '
                      'If a spike-bins checkpoint already exists, delete it before toggling this flag.')
_ap.add_argument('--embargo_trials',  type=int,   default=0,
                 help='Number of trials to drop from training on each side of every CV test '
                      'block boundary (default 1; set 0 to disable).')
_ap.add_argument('--animal_speed_gate', type=float, default=0.0,
                 help='If > 0, drop trial windows where the animal mean speed exceeds this '
                      'threshold (units match the "speed" column in texture_pca_df, '
                      'typically px/frame or normalised). '
                      'Keeps only stationary periods. 0 = off (default).')
# ── SELECTED TRIALS (from trial_structure.py) ─────────────────────────────────
_ap.add_argument('--selected_trials', default=None,
                 help='Path to a similar_trials CSV produced by trial_structure.py. '
                      'When provided, win_start_sec / win_end_sec from the CSV are used '
                      'as the trial windows instead of the regular sliding grid. '
                      'T_TRIAL and N_T_BINS are still respected (bin resolution), but '
                      'T_TRIAL is overridden by the window duration in the CSV.')
_ap.add_argument('--demean_beh', action=_argparse.BooleanOptionalAction, default=False,
                 help='If set, subtract each trial\'s own mean from its binned behavioural '
                      'trace before passing to the decoder. Removes slow within-trial drift '
                      'so the decoder learns shape rather than absolute level. '
                      'Applied after global z-scoring. Use --no-demean-beh to disable.')
_args = _ap.parse_args()

DF_DIR = r"/ptmp/gongt/behaviour"

metadata_df    = pd.read_pickle(os.path.join(DF_DIR, "metadata_df.pkl"))
texture_pca_df = pd.read_pickle(os.path.join(DF_DIR, "texture_pca_df_preproc.pkl"))

print(f"Texture PCA: {texture_pca_df.shape}")

animal  = _args.animal
exp     = _args.exp
SESSION = f"{animal}{exp}"

BASE_DIR    = rf"/nexus/posix0/MBR-neuralsystems/ephys/{animal}/{SESSION}"
PREPROC_DIR = os.path.join(BASE_DIR, "preprocessed")
PEAKS_FILE  = os.path.join(PREPROC_DIR, "thresholding_peaks", "peaks_combined2_sorted.npy")
LOC_FILE    = os.path.join(PREPROC_DIR, "thresholding_peaks", "localized_peaks_mt_raw.npy")
MOTION_DIR  = os.path.join(PREPROC_DIR, "thresholding_peaks", "motion_estimates", "motion_dredge")

_beh_key_dir   = _args.behavior_var.lower().replace(' ', '_')
_speedgate_tag = f"_gate{_args.animal_speed_gate:.3g}" if _args.animal_speed_gate > 0 else ""

# ── tag for results directory: append _selected if using hand-picked trials ───
_selected_tag = ""
if _args.selected_trials is not None:
    _sel_stem = os.path.splitext(os.path.basename(_args.selected_trials))[0]
    _selected_tag = f"_selected-{_sel_stem}"

_demean_tag = "_demeanbeh" if _args.demean_beh else ""

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "density_decoding",
    f"{_beh_key_dir}_{_args.bin_size_ms:.3g}ms_{_args.t_trial:.3g}s_{_args.chunk_style}_"
    f"{'zscoreboth' if _args.zscore_spike_features else 'nozscore'}_embargo{_args.embargo_trials}"
    f"{_speedgate_tag}{_selected_tag}{_demean_tag}_chunksize{_args.chunk_size}"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

# %%
sub = texture_pca_df[(texture_pca_df['animal'] == animal) & (texture_pca_df['background'] == 'eink')]

print(f"=== {animal} eink ===")
print(f"Number of ROWS (= video frames, not individual trials): {len(sub):,}")
print()

tt    = sub['trial_time'].dt.total_seconds()
abs_t = sub['time'].dt.total_seconds()
print(f"Session duration (trial_time): {tt.min():.4f} s  →  {tt.max():.4f} s")
print(f"Session duration (absolute)  : {abs_t.min():.4f} s  →  {abs_t.max():.4f} s")
print(f"  = {(tt.max() - tt.min()) / 60:.2f} min, {len(sub):,} frames at "
      f"{len(sub)/(tt.max()-tt.min()):.2f} fps effective")
print()

disp_cols = ['dataset', 'trial_idx', 'frame', 'trial_time', 'time', _args.behavior_var]
disp_cols = [c for c in disp_cols if c in sub.columns]
print(f"First 10 rows (each row = one video frame):")
print(sub[disp_cols].head(10).to_string())

# %% [markdown]
# #### NEURAL DATA

# %%
BEHAVIOR_VAR  = _args.behavior_var
Z_SCORE_BEH   = _args.zscore_spike_features
T_TRIAL       = _args.t_trial
BIN_SIZE_MS   = _args.bin_size_ms
N_T_BINS      = max(1, round(T_TRIAL * 1000 / BIN_SIZE_MS))   # derived; overridden below if selected trials
LEARNING_RATE = _args.learning_rate
N_WORKERS     = _args.n_workers

from spikeinterface import NumpyRecording
rec_si      = NumpyRecording.load(PREPROC_DIR)
ch_locs     = rec_si.get_channel_locations()
ap_fs       = rec_si.get_sampling_frequency()
ap_t_offset = float(rec_si._recording_segments[0].t_start)
n_ch        = rec_si.get_num_channels()
n_samp      = rec_si.get_num_samples()
ap_dtype    = rec_si.get_dtype()

beh     = texture_pca_df[(texture_pca_df['animal'] == animal) & (texture_pca_df['background'] == 'eink')].copy()
beh_t   = beh['time'].dt.total_seconds().values
beh_var = beh[BEHAVIOR_VAR].values
beh_spd = beh['speed'].values if 'speed' in beh.columns else None

OUT_PATH = os.path.join(RESULTS_DIR, f'spike_features_{SESSION}.npz')

if os.path.exists(OUT_PATH):
    print(f"\n[CACHE] Loading cached spike features -> {OUT_PATH}")
    data             = np.load(OUT_PATH, allow_pickle=False)
    spike_times_samp = data['times_samples']
    spike_channels   = data['channels']
    spike_features   = data['point_source_localizations']

    spike_t_gate = spike_times_samp / ap_fs + ap_t_offset
    spike_x      = spike_features[:, 0]
    z_corr       = spike_features[:, 1]
    spike_ptp    = spike_features[:, 2]
    spike_amp    = spike_ptp

    motion        = si.Motion.load(MOTION_DIR)
    temporal_bins = motion.temporal_bins_s[0]
    spatial_bins  = motion.spatial_bins_um
    z_raw         = None

    print(f"Loaded {len(spike_t_gate):,} spikes from cache.")
else:
    print("\n[COMPUTE] Computing spike features from raw data...")

    peaks = np.load(PEAKS_FILE)
    loc   = np.load(LOC_FILE)

    spike_times_samp = peaks['sample_index']
    spike_amp        = np.abs(peaks['amplitude']).astype(np.float32)
    spike_x          = loc['x'].astype(np.float32)
    spike_z          = loc['y'].astype(np.float32)

    spike_t_gate = spike_times_samp / ap_fs + ap_t_offset

    print(f"AP sampling rate    : {ap_fs:.4f} Hz  (from binary.json)")
    print(f"Gate offset         : {ap_t_offset:.3f} s  (t_starts from binary.json)")
    print(f"Total peaks         : {len(spike_t_gate):,}")
    print(f"spike_t_gate range  : {spike_t_gate.min():.2f} – {spike_t_gate.max():.2f} s")
    print(f"x range (lateral)   : {spike_x.min():.1f} – {spike_x.max():.1f} µm")
    print(f"z range (depth)     : {spike_z.min():.1f} – {spike_z.max():.1f} µm")
    print(f"amplitude range     : {spike_amp.min():.2f} – {spike_amp.max():.2f} µV")
    print(f"\nBehavioral rows     : {len(beh_t):,}")
    print(f"beh_t range         : {beh_t.min():.3f} – {beh_t.max():.3f} s")

    N_BEFORE_PTP  = round(0.5 * ap_fs / 1000)
    N_AFTER_PTP   = round(0.5 * ap_fs / 1000)
    wf_len_ptp    = N_BEFORE_PTP + N_AFTER_PTP
    CHUNK_S_PTP   = 60.0
    chunk_samp_ptp = int(CHUNK_S_PTP * ap_fs)
    n_spikes      = len(peaks)
    spike_ptp     = np.full(n_spikes, np.nan, dtype=np.float32)

    print(f"Computing raw PTP  window=±{N_BEFORE_PTP} samples (±0.5 ms)  "
          f"chunks={int(np.ceil(n_samp / chunk_samp_ptp))} × {CHUNK_S_PTP:.0f} s ...")
    chunk_start = 0
    ci = 0
    while chunk_start < n_samp:
        chunk_end = min(chunk_start + chunk_samp_ptp, n_samp)
        lo = int(np.searchsorted(spike_times_samp, chunk_start))
        hi = int(np.searchsorted(spike_times_samp, chunk_end))
        if lo < hi:
            tr_start = max(0, chunk_start - N_BEFORE_PTP)
            tr_end   = min(n_samp, chunk_end + N_AFTER_PTP)
            traces   = rec_si.get_traces(start_frame=tr_start, end_frame=tr_end)
            idxs  = np.arange(lo, hi)
            s_rel = spike_times_samp[idxs] - tr_start
            ch_loc = peaks['channel_index'][idxs].astype(int)
            full_mask = (s_rel >= N_BEFORE_PTP) & (s_rel + N_AFTER_PTP <= len(traces))
            if full_mask.any():
                t_off  = s_rel[full_mask][:, None] + np.arange(-N_BEFORE_PTP, N_AFTER_PTP)[None, :]
                ch_rep = np.broadcast_to(ch_loc[full_mask][:, None], t_off.shape)
                wfs    = traces[t_off, ch_rep].astype(np.float32)
                spike_ptp[idxs[full_mask]] = np.ptp(wfs, axis=1)
            for _ei in np.where(~full_mask)[0]:
                _ws = max(0, s_rel[_ei] - N_BEFORE_PTP)
                _we = min(len(traces), s_rel[_ei] + N_AFTER_PTP)
                _wf = traces[_ws:_we, ch_loc[_ei]].astype(np.float32)
                if len(_wf) > 0:
                    spike_ptp[idxs[_ei]] = np.ptp(_wf)
        chunk_start = chunk_end
        ci += 1
        if ci % 10 == 0 or chunk_start >= n_samp:
            print(f"  chunk {ci}  ({chunk_start / ap_fs / 60:.1f} min processed)")

    n_nan_ptp = int(np.isnan(spike_ptp).sum())
    p999 = float(np.nanpercentile(spike_ptp, 99.9))
    print(f"spike_ptp: min={np.nanmin(spike_ptp):.1f}  median={np.nanmedian(spike_ptp):.1f}  "
          f"99.9th={p999:.1f}  max={np.nanmax(spike_ptp):.1f}  µV")
    print(f"  NaN remaining: {n_nan_ptp} ({n_nan_ptp/n_spikes*100:.2f}%)")

    motion = si.Motion.load(MOTION_DIR)
    temporal_bins = motion.temporal_bins_s[0]
    spatial_bins  = motion.spatial_bins_um
    print(f"Motion loaded from  : {MOTION_DIR}")
    print(f"Temporal range      : {temporal_bins[0]:.2f} – {temporal_bins[-1]:.2f} s")
    print(f"Spatial range       : {spatial_bins[0]:.0f} – {spatial_bins[-1]:.0f} µm")

    z_raw = loc['y'].astype(np.float32)
    disp_at_spike = motion.get_displacement_at_time_and_depth(
        spike_t_gate, z_raw.astype(np.float64)
    ).astype(np.float32)
    z_corr = z_raw - disp_at_spike

    print(f"z_raw   range  : {z_raw.min():.1f} – {z_raw.max():.1f} µm")
    print(f"displacement   : {disp_at_spike.min():.1f} – {disp_at_spike.max():.1f} µm")
    print(f"z_corr  range  : {z_corr.min():.1f} – {z_corr.max():.1f} µm")
    print(f"Median abs correction : {np.median(np.abs(disp_at_spike)):.2f} µm")

    spike_channels = peaks['channel_index'].astype(np.int32)
    spike_features = np.column_stack([
        loc['x'].astype(np.float32),
        z_corr,
        spike_ptp,
    ])

    np.savez(
        OUT_PATH,
        times_samples              = peaks['sample_index'].astype(np.int64),
        channels                   = spike_channels,
        point_source_localizations = spike_features,
        t_start_gate_s             = np.float64(ap_t_offset),
        sampling_rate              = np.float64(ap_fs),
    )
    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"Saved → {OUT_PATH}  ({size_mb:.0f} MB)")
    print(f"  spike_features shape: {spike_features.shape}  dtype: {spike_features.dtype}")

# %% [markdown]
# #### ALIGNMENT CHECK

# %%
_align_save = os.path.join(RESULTS_DIR, f'alignment_check_{SESSION}.png')
if not os.path.exists(_align_save):
    print("[PLOT] Generating alignment check ...")
    BIN_ALIGN  = 1.0
    t_lo = min(spike_t_gate.min(), beh_t.min()) - 2
    t_hi = max(spike_t_gate.max(), beh_t.max()) + 2
    bins_1s  = np.arange(t_lo, t_hi + BIN_ALIGN, BIN_ALIGN)
    t_ctr    = (bins_1s[:-1] + bins_1s[1:]) / 2

    mua_counts, _ = np.histogram(spike_t_gate, bins=bins_1s)
    mua_hz = uniform_filter1d(mua_counts.astype(float) / BIN_ALIGN, size=5)

    def bin_signal(t_arr, sig, bins):
        out = np.full(len(bins) - 1, np.nan)
        for i in range(len(bins) - 1):
            m = (t_arr >= bins[i]) & (t_arr < bins[i + 1])
            if m.any():
                out[i] = np.nanmean(sig[m])
        return out

    beh_var_1s = bin_signal(beh_t, beh_var, bins_1s)
    spd_1s = bin_signal(beh_t, beh_spd, bins_1s) if beh_spd is not None else None

    def norm01(x):
        mn, mx = np.nanmin(x), np.nanmax(x)
        return (x - mn) / (mx - mn + 1e-9)

    n_rows = 2 + (spd_1s is not None)
    fig, axes = plt.subplots(n_rows + 1, 1, figsize=(22, 4 * (n_rows + 1)), sharex=True)
    fig.suptitle(f"{SESSION}  ·  Alignment check  ·  Common clock: SpikeGLX gate-open",
                 fontsize=11, y=1.002)

    axes[0].fill_between(t_ctr, mua_hz, alpha=0.7, color='steelblue')
    axes[0].set_ylabel("MUA\n(sp / s)", fontsize=9)
    axes[0].set_title("All KS4 spikes — 1 s bins, 5 s smoothed", fontsize=8, loc='left', pad=2)
    axes[1].plot(t_ctr, beh_var_1s, color='darkorange', lw=0.8)
    axes[1].set_ylabel(f"{BEHAVIOR_VAR}\n(1 s mean)", fontsize=9)

    row = 2
    if spd_1s is not None:
        axes[row].plot(t_ctr, spd_1s, color='forestgreen', lw=0.8)
        axes[row].set_ylabel("Speed\n(1 s mean)", fontsize=9)
        row += 1

    ax_ov = axes[row]
    ax_ov.plot(t_ctr, norm01(mua_hz),     color='steelblue',   lw=1.0, alpha=0.85, label='MUA (norm)')
    ax_ov.plot(t_ctr, norm01(beh_var_1s), color='darkorange',  lw=0.8, alpha=0.85, label=f'{BEHAVIOR_VAR} (norm)')
    if spd_1s is not None:
        ax_ov.plot(t_ctr, norm01(spd_1s), color='forestgreen', lw=0.8, alpha=0.85, label='Speed (norm)')
    ax_ov.set_ylabel("Normalised\n[0 – 1]", fontsize=9)
    ax_ov.legend(fontsize=8, loc='upper right')
    ax_ov.set_title("Overlay — use this to judge temporal alignment", fontsize=8, loc='left', pad=2)

    beh_patch = mpatches.Patch(color='gold',   alpha=0.35, label=f"Behavioral  {beh_t.min():.0f}–{beh_t.max():.0f} s")
    neu_patch = mpatches.Patch(color='violet', alpha=0.35, label=f"Neural      {ap_t_offset:.0f}–{spike_t_gate.max():.0f} s")
    for ax in axes:
        ax.axvspan(beh_t.min(),   beh_t.max(),        color='gold',   alpha=0.10, zorder=0)
        ax.axvspan(ap_t_offset,   spike_t_gate.max(), color='violet', alpha=0.08, zorder=0)
        ax.grid(True, alpha=0.25, lw=0.5)
    axes[0].legend(handles=[beh_patch, neu_patch], fontsize=8, loc='upper right')
    axes[-1].set_xlabel("Time from SpikeGLX gate open (s)", fontsize=9)
    plt.tight_layout()
    plt.savefig(_align_save, dpi=130, bbox_inches='tight')
    print(f"Saved → {_align_save}")
    plt.close()

    beh_dur = beh_t.max() - beh_t.min()
    neu_dur = spike_t_gate.max() - ap_t_offset
    ov_lo   = max(beh_t.min(), ap_t_offset)
    ov_hi   = min(beh_t.max(), spike_t_gate.max())
    print(f"Behavioral window : {beh_t.min():.3f} – {beh_t.max():.3f} s  ({beh_dur/60:.2f} min)")
    print(f"Neural window     : {ap_t_offset:.3f} – {spike_t_gate.max():.3f} s  ({neu_dur/60:.2f} min)")
    print(f"Overlap           : {ov_lo:.3f} – {ov_hi:.3f} s  ({(ov_hi-ov_lo)/60:.2f} min)")
    print(f"Neural starts {ap_t_offset - beh_t.min():.3f} s after video (gate-relative offset)")

# %% [markdown]
# #### SPIKE POSITION VISUALISATION

# %%
_pos_save = os.path.join(RESULTS_DIR, 'spike_positions.png')
if not os.path.exists(_pos_save):
    print("[PLOT] Generating spike positions ...")
    probe_x_min  = ch_locs[:, 0].min()
    probe_x_max  = ch_locs[:, 0].max()
    print(f"Channels: {n_ch}   probe x: {probe_x_min:.1f} – {probe_x_max:.1f} µm   "
          f"depth: {ch_locs[:,1].min():.0f} – {ch_locs[:,1].max():.0f} µm")

    T_START = 0
    T_WIN   = 5
    mask    = (spike_t_gate >= T_START + ap_t_offset) & (spike_t_gate < T_START + T_WIN + ap_t_offset)
    sx, sz, sa = spike_x[mask], z_corr[mask], spike_amp[mask]
    outside = (sx < probe_x_min) | (sx > probe_x_max)
    print(f"Spikes in {T_WIN*1e3:.0f} ms window : {mask.sum()}")
    print(f"  outside probe x range           : {outside.sum()}  ({outside.mean()*100:.0f}%)")

    fig, ax = plt.subplots(figsize=(6, 14))
    fig.suptitle(f"Spike positions  ·  t = {T_START}–{T_START+T_WIN:.3f} s  ({T_WIN*1e3:.0f} ms window)",
                 fontsize=10)
    ax.scatter(ch_locs[:, 0], ch_locs[:, 1], s=18, c='lightgray',
               edgecolors='gray', linewidths=0.5, zorder=2, label='Electrodes')
    ax.axvspan(probe_x_min, probe_x_max, color='dodgerblue', alpha=0.06, zorder=0)
    ax.axvline(probe_x_min, color='dodgerblue', lw=1, alpha=0.6, ls='--', label='Probe x extent')
    ax.axvline(probe_x_max, color='dodgerblue', lw=1, alpha=0.6, ls='--')
    if mask.any():
        amp_vmax = np.percentile(spike_amp, 95)
        sc_h = ax.scatter(sx, sz, c=sa, s=50, cmap='hot_r',
                          vmin=0, vmax=amp_vmax, edgecolors='k', linewidths=0.2, zorder=5,
                          label=f'Spikes (n={mask.sum()}, {outside.sum()} outside probe)')
        plt.colorbar(sc_h, ax=ax, label='Amplitude (µV)', shrink=0.55)
    ax.set_xlabel('Lateral x (µm)')
    ax.set_ylabel('Depth z (µm)')
    ax.set_title('Probe map  (blue = probe x extent)', fontsize=8, loc='left')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(_pos_save, dpi=130, bbox_inches='tight')
    print(f"Saved → {_pos_save}")
    plt.close()

# %% [markdown]
# #### EXTRACT SPIKE FEATURES FOR DENSITY DECODING

# %%
print(f"Recording : {n_ch} ch  |  {ap_fs:.2f} Hz  |  "
      f"{n_samp:,} samples  ({n_samp / ap_fs / 60:.1f} min)  |  dtype {ap_dtype}")
print(f"t_start   : {ap_t_offset:.3f} s (gate-relative)")
print(f"Output    : {OUT_PATH}")

data = np.load(OUT_PATH, allow_pickle=False)

spike_times    = data['times_samples']
spike_channels = data['channels']
spike_features = data['point_source_localizations']

print("Loaded spike features:")
print(f"  spike_times    shape {spike_times.shape},    dtype {spike_times.dtype}")
print(f"  spike_channels shape {spike_channels.shape}, dtype {spike_channels.dtype}")
print(f"  spike_features shape {spike_features.shape}, dtype {spike_features.dtype}")
print(f"  columns: x (µm) | z (µm, DREDge-corrected) | PTP amplitude (µV)")
print(f"  raw ranges: x=[{spike_features[:,0].min():.1f}, {spike_features[:,0].max():.1f}]  "
      f"z=[{spike_features[:,1].min():.1f}, {spike_features[:,1].max():.1f}]  "
      f"ptp=[{spike_features[:,2].min():.2f}, {spike_features[:,2].max():.2f}]")
print()

p999 = float(np.nanpercentile(spike_ptp, 99.9))

_loc_save = os.path.join(RESULTS_DIR, f'spike_localisations_{SESSION}.png')
if not os.path.exists(_loc_save):
    print("[PLOT] Generating localisation scatter ...")
    rng_sf = np.random.default_rng(0)
    n_show = min(200_000, len(spike_features))
    isf    = rng_sf.choice(len(spike_features), n_show, replace=False)
    isf    = isf[np.argsort(spike_ptp[isf])]

    fig, ax = plt.subplots(figsize=(4, 10))
    sc = ax.scatter(spike_features[isf, 0], spike_features[isf, 1],
                    c=np.clip(spike_ptp[isf], 0, p999),
                    s=0.3, alpha=0.3, cmap='hot', rasterized=True)
    plt.colorbar(sc, ax=ax, label='PTP (µV)')
    ax.set_xlabel('x (µm)'); ax.set_ylabel('depth z (µm)')
    ax.set_title(f'{SESSION} — localisations  ({n_show:,} random, DREDge-corrected depth)')
    plt.tight_layout()
    plt.savefig(_loc_save, dpi=130, bbox_inches='tight')
    print(f"Saved → {_loc_save}")
    plt.close()

# %% [markdown]
# #### WAVEFORM VISUALISATIONS (plot-guarded)

# %%
_wf_val_save = os.path.join(RESULTS_DIR, f'ptp_waveforms_{SESSION}.png')
if not os.path.exists(_wf_val_save) and 'peaks' in dir():
    print("[PLOT] Generating waveform grid ...")
    N_BEFORE_PTP_GRID = round(0.5 * ap_fs / 1000)
    N_AFTER_PTP_GRID  = round(0.5 * ap_fs / 1000)
    wf_len_val        = N_BEFORE_PTP_GRID + N_AFTER_PTP_GRID

    rng_val   = np.random.default_rng(42)
    val_idx   = rng_val.choice(len(peaks), 100, replace=False)
    wf_val    = np.full((100, wf_len_val), np.nan, dtype=np.float32)
    ptp_val   = spike_ptp[val_idx].copy()

    for k, idx in enumerate(val_idx):
        pk_ch = int(peaks['channel_index'][idx])
        samp  = int(spike_times_samp[idx])
        t0 = max(0, samp - N_BEFORE_PTP_GRID)
        t1 = min(n_samp,  samp + N_AFTER_PTP_GRID)
        tr = rec_si.get_traces(start_frame=t0, end_frame=t1)[:, pk_ch].astype(np.float32)
        if tr.shape[0] < wf_len_val:
            tr = np.pad(tr, (0, wf_len_val - tr.shape[0]))
        wf_val[k] = tr

    order_val = np.argsort(ptp_val)[::-1]
    t_ax_val  = np.arange(wf_len_val) - N_BEFORE_PTP_GRID
    y_half    = np.nanpercentile(ptp_val, 99) * 0.7
    ylim_val  = (-y_half, y_half)

    ncols_val, nrows_val = 10, 10
    fig2, axes2 = plt.subplots(nrows_val, ncols_val,
                                figsize=(ncols_val * 1.8, nrows_val * 1.8),
                                sharex=True, sharey=True)
    fig2.suptitle(
        'Waveforms on detection channel — 100 random spikes, ordered by descending PTP\n'
        'blue ▲▼ = max/min in window  |  orange dashed = ±PTP/2\n'
        f'window: ±{N_BEFORE_PTP_GRID} samples = ±0.5 ms',
        fontsize=10
    )

    for pi, k in enumerate(order_val):
        r, c  = divmod(pi, ncols_val)
        ax2   = axes2[r, c]
        wf    = wf_val[k]
        ax2.plot(t_ax_val, wf, color='k', lw=0.8)
        i_max = int(np.argmax(wf));  i_min = int(np.argmin(wf))
        t_max, v_max = t_ax_val[i_max], float(wf[i_max])
        t_min, v_min = t_ax_val[i_min], float(wf[i_min])
        arrow_offset = y_half * 0.28
        ax2.annotate('', xy=(t_max, v_max), xytext=(t_max, v_max + arrow_offset),
                     arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.0))
        ax2.annotate('', xy=(t_min, v_min), xytext=(t_min, v_min - arrow_offset),
                     arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.0))
        ax2.axhline( ptp_val[k] / 2, color='darkorange', lw=0.8, ls='--', alpha=0.85)
        ax2.axhline(-ptp_val[k] / 2, color='darkorange', lw=0.8, ls='--', alpha=0.85)
        ax2.axvline(0, color='gray', lw=0.5, ls=':')
        ax2.set_title(f'ptp={ptp_val[k]:.0f} µV', fontsize=5.5, pad=1)
        ax2.tick_params(labelsize=5)

    for ax2 in axes2.flat:
        ax2.set_ylim(*ylim_val)
    axes2[0, 0].plot([], [], color='darkorange', ls='--', lw=1, label='±PTP/2')
    axes2[0, 0].legend(fontsize=5, loc='upper right', framealpha=0.6)
    plt.tight_layout()
    plt.savefig(_wf_val_save, dpi=130, bbox_inches='tight')
    print(f"Saved → {_wf_val_save}")
    plt.close()

_scatter_save = os.path.join(RESULTS_DIR, f'drift_scatter_before_after_{SESSION}.png')
if not os.path.exists(_scatter_save) and 'z_raw' in dir() and z_raw is not None:
    print("[PLOT] Generating drift scatter before/after ...")
    T_VIZ_S = 12 * 60
    MAX_PTS  = 200_000

    t_rec_start = spike_t_gate.min()
    t_rec_end   = spike_t_gate.max()
    windows = [
        (f'First 12 min  ({t_rec_start:.0f}–{t_rec_start+T_VIZ_S:.0f} s)',
         t_rec_start, t_rec_start + T_VIZ_S),
        (f'Last 12 min   ({t_rec_end-T_VIZ_S:.0f}–{t_rec_end:.0f} s)',
         t_rec_end - T_VIZ_S, t_rec_end),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharey=True, sharex='row')
    fig.suptitle(f'{SESSION} — Spike depth vs time\nLeft: raw  |  Right: DREDge corrected',
                 fontsize=10)

    rng_sc    = np.random.default_rng(7)
    disp_cmap = plt.cm.plasma
    disp_norm = plt.Normalize(spatial_bins[0], spatial_bins[-1])

    for row_i, (win_label, t0, t1) in enumerate(windows):
        sel = np.where((spike_t_gate >= t0) & (spike_t_gate < t1))[0]
        if len(sel) > MAX_PTS:
            sel = rng_sc.choice(sel, MAX_PTS, replace=False)
            sel = sel[np.argsort(spike_t_gate[sel])]

        t_sel  = spike_t_gate[sel]
        t_disp = (temporal_bins >= t0) & (temporal_bins <= t1)
        t_line = temporal_bins[t_disp]

        for col_i, (z_arr, panel_title) in enumerate([
            (z_raw[sel],  'Before correction'),
            (z_corr[sel], 'After DREDge correction'),
        ]):
            ax = axes[row_i, col_i]
            ax.scatter(t_sel, z_arr, c='gray', s=0.3, alpha=0.25, rasterized=True, linewidths=0)
            if col_i == 0:
                for k, z_bin in enumerate(spatial_bins):
                    apparent = z_bin + motion.displacement[0][t_disp, k]
                    ax.plot(t_line, apparent, color=disp_cmap(disp_norm(z_bin)),
                            lw=1.5, alpha=0.85, zorder=5)
                sm = plt.cm.ScalarMappable(cmap=disp_cmap, norm=disp_norm)
                sm.set_array([])
                cb = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
                cb.set_label('Spatial bin depth (µm)', fontsize=7)
                cb.ax.tick_params(labelsize=6)
            ax.set_title(f'{win_label}  —  {panel_title}', fontsize=9)
            ax.set_xlabel('Time from gate open (s)', fontsize=8)
            if col_i == 0:
                ax.set_ylabel('Depth (µm)', fontsize=8)
            ax.grid(True, alpha=0.12, lw=0.4)

    plt.tight_layout()
    plt.savefig(_scatter_save, dpi=130, bbox_inches='tight')
    print(f"Saved → {_scatter_save}")
    plt.close()

_wf_save = os.path.join(RESULTS_DIR, f'waveform_qc_{SESSION}.png')
if not os.path.exists(_wf_save) and 'peaks' in dir():
    print("[PLOT] Generating waveform QC probe-layout ...")
    T_POOL_START = 100.0
    T_POOL_END   = 110.0
    N_SHOW       = 5
    N_BEFORE     = 20
    N_AFTER      = 62
    N_NBR        = 32

    pool_mask = (spike_t_gate >= T_POOL_START) & (spike_t_gate < T_POOL_END)
    pool_idx  = np.where(pool_mask)[0]
    if len(pool_idx) < N_SHOW:
        print(f"[WARN] Not enough spikes in pooling window: found {len(pool_idx)} — skipping waveform QC")
    else:
        rng = np.random.default_rng(2026)
        sel = rng.choice(pool_idx, N_SHOW, replace=False)

        wf_len       = N_BEFORE + N_AFTER
        n_total_samp = rec_si.get_num_samples()

        sort_depths  = np.sort(np.unique(ch_locs[:, 1]))
        elec_spacing = float(np.median(np.diff(sort_depths)))
        col_xs       = np.sort(np.unique(ch_locs[:, 0]))
        col_gap      = float(np.min(np.diff(col_xs))) if len(col_xs) > 1 else elec_spacing
        t_um         = np.linspace(-col_gap * 0.45, col_gap * 0.45, wf_len)

        all_wfs = [];  all_nbr = []
        for _spi in sel:
            pk_ch = int(peaks['channel_index'][_spi])
            dists = np.linalg.norm(ch_locs - ch_locs[pk_ch], axis=1)
            nbr   = np.argsort(dists)[:N_NBR]
            samp  = int(spike_times_samp[_spi])
            t0    = max(0, samp - N_BEFORE)
            t1    = min(n_total_samp, samp + N_AFTER)
            tr    = rec_si.get_traces(start_frame=t0, end_frame=t1)[:, nbr].astype(np.float32)
            if tr.shape[0] < wf_len:
                tr = np.pad(tr, ((0, wf_len - tr.shape[0]), (0, 0)))
            all_wfs.append(tr);  all_nbr.append(nbr)

        all_ptp    = np.concatenate([np.ptp(w, axis=0) for w in all_wfs])
        global_ptp = float(np.percentile(all_ptp, 99))
        AMP_SCALE  = (0.7 * elec_spacing) / max(global_ptp, 1.0)

        ncols = N_SHOW
        fig, axes = plt.subplots(1, ncols, figsize=(ncols * 5, 12), sharey=True, sharex=True)
        if ncols == 1:
            axes = [axes]
        fig.suptitle(f"Waveform QC (post-DREDge) — {SESSION}", fontsize=10)

        for pi, (_spi, wf, nbr) in enumerate(zip(sel, all_wfs, all_nbr)):
            ax    = axes[pi]
            pk_ch = int(peaks['channel_index'][_spi])
            px    = float(spike_x[_spi])
            pz    = float(z_corr[_spi])
            ax.scatter(ch_locs[:, 0], ch_locs[:, 1], s=8, c='lightgray',
                       edgecolors='none', zorder=1, rasterized=True)
            for j, ch_j in enumerate(nbr):
                xc, yc = ch_locs[ch_j, 0], ch_locs[ch_j, 1]
                ax.plot(xc + t_um, yc + wf[:, j] * AMP_SCALE,
                        lw=1.2, color='royalblue', alpha=0.95, zorder=3)
            ax.scatter(ch_locs[pk_ch, 0], ch_locs[pk_ch, 1],
                       s=60, c='darkorange', marker='D', zorder=5,
                       label='Peak ch' if pi == 0 else None)
            ax.axhline(pz, color='red', lw=1.1, ls='--', alpha=0.7, zorder=4)
            ax.axvline(px, color='red', lw=1.1, ls='--', alpha=0.7, zorder=4)
            ax.scatter(px, pz, s=150, c='red', marker='*', zorder=6,
                       label='Triangulated' if pi == 0 else None)
            ax.set_title(f"spike {pi+1}  det_ch={pk_ch}\n"
                         f"{spike_amp[_spi]:.0f} µV  x={px:.0f}  z={pz:.0f} µm (corr)", fontsize=8)
            ax.tick_params(labelsize=7);  ax.grid(True, alpha=0.18)
            if pi == 0:
                ax.legend(fontsize=8, loc='upper right')

        axes[0].set_ylabel('Depth (µm)', fontsize=9)
        for ax in axes:
            ax.set_xlabel('Lateral x (µm)', fontsize=9)

        plt.tight_layout()
        plt.savefig(_wf_save, dpi=130, bbox_inches='tight')
        print(f"Saved → {_wf_save}")
        plt.close()

# %% [markdown]
# #### FEATURE Z-SCORING

# %%
ZSCORE_SPIKE_FEATURES = _args.zscore_spike_features
if ZSCORE_SPIKE_FEATURES:
    _sf_mean = spike_features.mean(axis=0)
    _sf_std  = spike_features.std(axis=0)
    _sf_std[_sf_std < 1e-9] = 1.0
    print(f"[ZSCORE] Spike features BEFORE z-score:")
    print(f"  means : x={_sf_mean[0]:.3f}  z={_sf_mean[1]:.3f}  ptp={_sf_mean[2]:.3f}")
    print(f"  stds  : x={_sf_std[0]:.3f}  z={_sf_std[1]:.3f}  ptp={_sf_std[2]:.3f}")
    spike_features = ((spike_features - _sf_mean) / _sf_std).astype(np.float32)
    print(f"[ZSCORE] Spike features AFTER z-score:")
    print(f"  means : x={spike_features[:,0].mean():.4f}  z={spike_features[:,1].mean():.4f}  ptp={spike_features[:,2].mean():.4f}")
    print(f"  stds  : x={spike_features[:,0].std():.4f}  z={spike_features[:,1].std():.4f}  ptp={spike_features[:,2].std():.4f}")
    _zscore_params_sf = {
        'x_mean': float(_sf_mean[0]),  'x_std': float(_sf_std[0]),
        'z_mean': float(_sf_mean[1]),  'z_std': float(_sf_std[1]),
        'ptp_mean': float(_sf_mean[2]),'ptp_std': float(_sf_std[2]),
    }
else:
    print("[ZSCORE] Spike features NOT z-scored.")
    _zscore_params_sf = None

# %% [markdown]
# #### BEHAVIOURAL DATA FOR DENSITY DECODING

# %%
_dt        = np.diff(beh_t)
_dt_median = float(np.median(_dt))
_dt_tol    = 0.01
_irregular = np.where(np.abs(_dt - _dt_median) > _dt_tol)[0]
print(f"Frame interval  : median={_dt_median*1000:.3f} ms  "
      f"min={_dt.min()*1000:.3f} ms  max={_dt.max()*1000:.3f} ms")
print(f"Irregular gaps  : {len(_irregular)} / {len(_dt)} intervals")

beh_raw = beh[BEHAVIOR_VAR].values.astype(np.float32)

if Z_SCORE_BEH:
    beh_z = zscore(beh_raw, nan_policy='omit').astype(np.float32)
    print(f"Original {BEHAVIOR_VAR}:")
    print(f"mean={np.nanmean(beh_raw):.6f} std={np.nanstd(beh_raw):.6f}")
    print(f"[ZSCORE] {BEHAVIOR_VAR} AFTER z-score:")
    print(f"  mean={np.nanmean(beh_z):.6f}  std={np.nanstd(beh_z):.6f}  "
          f"min={np.nanmin(beh_z):.4f}  max={np.nanmax(beh_z):.4f}")
else:
    beh_z = beh_raw.copy()
    print(f"[ZSCORE] {BEHAVIOR_VAR} NOT z-scored (using raw values).")

T_OV_START = max(beh_t.min(), ap_t_offset)
T_OV_END   = min(beh_t.max(), spike_t_gate.max())
print(f"\nOverlap window: {T_OV_START:.2f} – {T_OV_END:.2f} s "
      f"({(T_OV_END - T_OV_START)/60:.2f} min)")

# %% [markdown]
# #### TRIAL WINDOWS
# If --selected_trials is provided: use the pre-selected, similarity-filtered windows
# (win_start_sec / win_end_sec from trial_structure.py CSV).
# Otherwise: build the regular sliding grid over the full overlap window.

# %%
USING_SELECTED_TRIALS = _args.selected_trials is not None

if USING_SELECTED_TRIALS:
    # ── load the CSV produced by trial_structure.py ───────────────────────────
    _sel_df = pd.read_csv(_args.selected_trials)

    # sanity check: expected columns
    _required_cols = {'win_start_sec', 'win_end_sec', 'win_duration_sec'}
    _missing = _required_cols - set(_sel_df.columns)
    if _missing:
        raise ValueError(f"--selected_trials CSV is missing columns: {_missing}. "
                         f"Make sure this file was produced by trial_structure.py.")

    trial_starts = _sel_df['win_start_sec'].values
    trial_ends   = _sel_df['win_end_sec'].values

    # override T_TRIAL from the CSV (all windows should be the same length)
    _win_durations = trial_ends - trial_starts
    _win_dur_unique = np.unique(np.round(_win_durations, 4))
    if len(_win_dur_unique) > 1:
        print(f"[WARN] Selected trials have varying window lengths: {_win_dur_unique} s. "
              f"Using median = {np.median(_win_durations):.4f} s as T_TRIAL.")
    T_TRIAL  = float(np.median(_win_durations))
    N_T_BINS = max(1, round(T_TRIAL * 1000 / BIN_SIZE_MS))   # re-derive with fixed bin size

    # clip to overlap window (drop any selected trials outside neural coverage)
    _in_overlap = (trial_starts >= T_OV_START) & (trial_ends <= T_OV_END)
    if (~_in_overlap).any():
        print(f"[WARN] {(~_in_overlap).sum()} selected trials fall outside the "
              f"neural–behavioral overlap window and will be dropped.")
        trial_starts = trial_starts[_in_overlap]
        trial_ends   = trial_ends[_in_overlap]
        _sel_df      = _sel_df[_in_overlap].reset_index(drop=True)

    n_trials_beh = len(trial_starts)

    # log metadata from the CSV for traceability
    _sel_meta = {
        'source_csv':   _args.selected_trials,
        'n_trials':     int(n_trials_beh),
        'align_mode':   str(_sel_df['align_mode'].iloc[0]) if 'align_mode' in _sel_df.columns else 'unknown',
        'rms_threshold': float(_sel_df['rms_threshold'].iloc[0]) if 'rms_threshold' in _sel_df.columns else None,
        'pc':           str(_sel_df['pc'].iloc[0]) if 'pc' in _sel_df.columns else 'unknown',
        'T_TRIAL_s':    float(T_TRIAL),
    }
    print(f"\n[SELECTED TRIALS] Using {n_trials_beh} pre-selected similarity-filtered windows")
    print(f"  Source : {_args.selected_trials}")
    print(f"  PC     : {_sel_meta['pc']}   align={_sel_meta['align_mode']}   "
          f"RMS threshold={_sel_meta['rms_threshold']}")
    print(f"  T_TRIAL overridden → {T_TRIAL:.4f} s  (from win_duration_sec in CSV)")
    print(f"  Window range: {trial_starts.min():.2f} – {trial_ends.max():.2f} s")

else:
    # ── regular sliding grid ──────────────────────────────────────────────────
    trial_starts = np.arange(T_OV_START, T_OV_END - T_TRIAL, T_TRIAL)
    trial_ends   = trial_starts + T_TRIAL
    n_trials_beh = len(trial_starts)
    _sel_meta    = None
    print(f"\n[REGULAR GRID] {n_trials_beh} sliding windows of {T_TRIAL} s")

CHUNK_STYLE     = _args.chunk_style
CHUNK_SIZE      = _args.chunk_size
TEST_PERCENTAGE = _args.test_percentage
DEMEAN_BEH      = _args.demean_beh
TEST_FRAC       = TEST_PERCENTAGE / 100.0
TRAIN_FRAC      = 1.0 - TEST_FRAC
print(f"Window length : {T_TRIAL} s  |  bin size {BIN_SIZE_MS:.3g} ms  →  {N_T_BINS} bins  "
      f"(actual bin = {T_TRIAL/N_T_BINS*1000:.2f} ms)")
print(f"Number of windows : {n_trials_beh}  "
      f"(~{n_trials_beh*TRAIN_FRAC:.0f} train, ~{n_trials_beh*TEST_FRAC:.0f} test)")

# ── params JSON (saved before heavy compute) ──────────────────────────────────
_VAR_KEY     = BEHAVIOR_VAR.lower().replace(' ', '_')
_params_json = os.path.join(RESULTS_DIR, f'decode_params_{_VAR_KEY}_{SESSION}.json')
_params_ts   = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
_params = {
    "session":  SESSION,
    "started":  _params_ts,
    "decode": {
        "BEHAVIOR_VAR":  BEHAVIOR_VAR,
        "Z_SCORE_BEH":   Z_SCORE_BEH,
        "T_TRIAL_s":     float(T_TRIAL),
        "bin_size_ms":   float(BIN_SIZE_MS),
        "N_T_BINS":      int(N_T_BINS),
        "actual_ms_per_bin": float(T_TRIAL / N_T_BINS * 1000),
        "LEARNING_RATE": float(LEARNING_RATE),
        "N_WORKERS":     int(N_WORKERS),
        "chunk_style":   CHUNK_STYLE,
        "test_percentage": float(TEST_PERCENTAGE),
        "train_frac":    float(TRAIN_FRAC),
        "test_frac":     float(TEST_FRAC),
        "embargo_trials": int(_args.embargo_trials),
        "animal_speed_gate": float(_args.animal_speed_gate),
        "zscore_spike_features": ZSCORE_SPIKE_FEATURES,
        "spike_feature_zscore_params": _zscore_params_sf,
        "using_selected_trials": USING_SELECTED_TRIALS,
        "selected_trials_meta": _sel_meta,
        "demean_beh":    DEMEAN_BEH,
    },
    "data": {
        "PREPROC_DIR":      PREPROC_DIR,
        "PEAKS_FILE":       PEAKS_FILE,
        "LOC_FILE":         LOC_FILE,
        "ap_fs_hz":         float(ap_fs),
        "ap_t_offset_s":    float(ap_t_offset),
        "n_total_peaks":    int(len(spike_t_gate)),
        "drift_correction": "DREDge",
        "MOTION_DIR":       MOTION_DIR,
    },
    "behavior": {
        "animal":            animal,
        "background":        "eink",
        "T_OV_START_s":      float(T_OV_START),
        "T_OV_END_s":        float(T_OV_END),
        "n_windows_total":   int(n_trials_beh),
    },
    "gmm_init": {
        "method":               "isosplit",
        "K_init":               20,
        "min_cluster_size":     10,
        "whiten_cluster_pairs": 1,
        "refine_clusters":      1,
    },
}
with open(_params_json, 'w') as _f:
    json.dump(_params, _f, indent=2)
print(f"\n[PARAMS] Saved -> {_params_json}")

# ── behaviour overview plot (plot-guarded) ────────────────────────────────────
_beh_plot_save = os.path.join(RESULTS_DIR, f'behaviour_overview_{_VAR_KEY}_{SESSION}.png')
# Always regenerate — demean flag or selected trials may have changed
print(f"[PLOT] Generating behaviour overview -> {_beh_plot_save}")
fig, ax = plt.subplots(figsize=(18, 2.5))
ax.plot(beh_t, beh_z, lw=0.4, color='tab:blue', alpha=0.7,
        label=f"{BEHAVIOR_VAR}{'  (z-scored)' if Z_SCORE_BEH else ''} [session-wide]")
ax.axvspan(T_OV_START, T_OV_END, color='violet', alpha=0.08, label='Neural-behavioral overlap')
if USING_SELECTED_TRIALS:
    for _ts_mark, _te_mark in zip(trial_starts, trial_ends):
        ax.axvspan(_ts_mark, _te_mark, color='gold', alpha=0.25, zorder=0)
    ax.axvspan(0, 0, color='gold', alpha=0.5, label='Selected trial windows')  # legend proxy
ax.set_xlabel("Time from SpikeGLX gate open (s)", fontsize=9)
ax.set_ylabel(BEHAVIOR_VAR, fontsize=9)
_title = (f"Behaviour: {BEHAVIOR_VAR} ({animal} eink)"
          + ("  [selected trials highlighted]" if USING_SELECTED_TRIALS else "")
          + ("  [per-trial mean will be subtracted before decoding]" if DEMEAN_BEH else ""))
ax.set_title(_title, fontsize=9)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(_beh_plot_save, dpi=130, bbox_inches='tight')
print(f"Saved -> {_beh_plot_save}")
plt.close()

# %%
sys.path.insert(0, '/u/gongt/density_decoding')
from density_decoding.utils.data_utils import BaseDataLoader

data_loader = BaseDataLoader(
    t_before=0.0,
    t_after=T_TRIAL,
    n_t_bins=N_T_BINS,
)

# ── spike bins checkpoint — NOTE: when using selected trials the window grid is
# different, so the checkpoint is NOT shared with the regular-grid run. The
# results dir already has the _selected tag, so the path is naturally separate.
SPIKE_CKPT = os.path.join(RESULTS_DIR, f'ckpt_spikebins_{SESSION}.pkl')

print(f"Processing behaviour ({BEHAVIOR_VAR}) ...")
bin_beh_all = data_loader.process_behaviors(
    time_points       = beh_t,
    raw_behaviors     = beh_z,
    trial_start_times = trial_starts,
    trial_end_times   = trial_ends,
)

valid_mask_beh  = ~np.isnan(bin_beh_all).any(axis=1)
valid_trials_beh = np.where(valid_mask_beh)[0]
print(f"Valid windows (no NaN in {BEHAVIOR_VAR}): {valid_mask_beh.sum()} / {n_trials_beh}")

ANIMAL_SPEED_GATE = _args.animal_speed_gate
_speed_gate_info  = dict(used=False)
if ANIMAL_SPEED_GATE > 0:
    if beh_spd is None:
        print("[SPEED GATE] WARNING: 'speed' column not found — gate DISABLED.")
        ANIMAL_SPEED_GATE = 0.0
    else:
        _win_spd = np.array([
            np.nanmean(beh_spd[(beh_t >= trial_starts[k]) & (beh_t < trial_ends[k])])
            for k in range(n_trials_beh)
        ])
        _speed_gate_mask = _win_spd < ANIMAL_SPEED_GATE
        valid_mask_beh   = valid_mask_beh & _speed_gate_mask
        valid_trials_beh = np.where(valid_mask_beh)[0]
        _n_dropped_spd   = int((~_speed_gate_mask).sum())
        _speed_gate_info = dict(
            used=True, threshold=float(ANIMAL_SPEED_GATE),
            n_total=int(n_trials_beh), n_kept=int(valid_mask_beh.sum()),
            n_dropped_speed=_n_dropped_spd,
            pct_stationary=100.0 * valid_mask_beh.sum() / max(n_trials_beh, 1)
        )
        print(f"[SPEED GATE] threshold={ANIMAL_SPEED_GATE:.4g} → "
              f"{valid_mask_beh.sum()} / {n_trials_beh} windows kept "
              f"({_speed_gate_info['pct_stationary']:.1f}% stationary); "
              f"{_n_dropped_spd} dropped for animal motion")
print(f"Valid windows after all filters: {valid_mask_beh.sum()} / {n_trials_beh}")

if os.path.exists(SPIKE_CKPT):
    print(f"\n[CHECKPOINT] Loading spike bins -> {SPIKE_CKPT}")
    with open(SPIKE_CKPT, 'rb') as _f:
        _ckpt = pickle.load(_f)
    bin_spike_features_all = _ckpt['bin_spike_features_all']
    bin_trial_idxs_all     = _ckpt['bin_trial_idxs_all']
    bin_time_idxs_all      = _ckpt['bin_time_idxs_all']
    n_windows_cached       = len(bin_spike_features_all)
    print(f"  {n_windows_cached} cached windows")
    if n_windows_cached != n_trials_beh:
        raise RuntimeError(
            f"Cached {n_windows_cached} windows != current {n_trials_beh} windows. "
            f"Delete {SPIKE_CKPT} and rerun."
        )
else:
    print("Processing spike features (ALL temporal windows) ...")
    bin_spike_features_all, bin_trial_idxs_all, bin_time_idxs_all = \
        data_loader.process_spike_features(
            spike_times       = spike_t_gate,
            spike_channels    = spike_channels,
            spike_features    = spike_features,
            trial_start_times = trial_starts,
            trial_end_times   = trial_ends,
            valid_trials      = np.arange(n_trials_beh),
        )
    print(f"  Done — {len(bin_spike_features_all)} windows binned")
    with open(SPIKE_CKPT, 'wb') as _f:
        pickle.dump({
            'bin_spike_features_all': bin_spike_features_all,
            'bin_trial_idxs_all':     bin_trial_idxs_all,
            'bin_time_idxs_all':      bin_time_idxs_all,
        }, _f, protocol=4)
    print(f"\n[CHECKPOINT] Saved -> {SPIKE_CKPT}")

bin_beh_v = bin_beh_all[valid_mask_beh]   # (n_valid, N_T_BINS)

# ── per-trial demeaning of behaviour ─────────────────────────────────────────
# Subtracts each trial's own mean across time bins so the decoder sees
# shape/dynamics rather than absolute PC level.
# Applied AFTER global z-scoring; the global scale is preserved.
if DEMEAN_BEH:
    _beh_trial_means = bin_beh_v.mean(axis=1, keepdims=True)   # (n_valid, 1)
    bin_beh_v = bin_beh_v - _beh_trial_means
    print(f"[DEMEAN_BEH] Subtracted per-trial mean from binned behaviour.")
    print(f"  Post-demean: mean={bin_beh_v.mean():.6f}  std={bin_beh_v.std():.6f}  "
          f"min={bin_beh_v.min():.4f}  max={bin_beh_v.max():.4f}")

bin_sf_v  = [bin_spike_features_all[k] for k in valid_trials_beh]
bin_ti_v, bin_tmi_v = [], []
for new_k, k in enumerate(valid_trials_beh):
    for t in range(N_T_BINS):
        old = bin_trial_idxs_all[k * N_T_BINS + t]
        bin_ti_v.append(np.full_like(old, new_k))
        bin_tmi_v.append(bin_time_idxs_all[k * N_T_BINS + t])

n_valid = len(valid_trials_beh)

if TEST_PERCENTAGE <= 0 or TEST_PERCENTAGE >= 100:
    raise ValueError(f"test_percentage must be in range (0, 100), got {TEST_PERCENTAGE}")

# auto-fallback: interleaved requires at least CHUNK_SIZE valid trials
if CHUNK_STYLE == 'interleaved' and n_valid < CHUNK_SIZE:
    print(f"\n[WARN] n_valid={n_valid} < CHUNK_SIZE={CHUNK_SIZE} — "
          f"falling back to contiguous split (not enough trials for interleaved).")
    CHUNK_STYLE = 'contiguous'

if CHUNK_STYLE == 'interleaved' and n_valid >= CHUNK_SIZE:
    _n_test_per_chunk  = max(1, int(round(CHUNK_SIZE * TEST_FRAC)))
    _n_train_per_chunk = CHUNK_SIZE - _n_test_per_chunk
    train_idx = np.concatenate([
        np.arange(c * CHUNK_SIZE, c * CHUNK_SIZE + _n_train_per_chunk)
        for c in range(n_valid // CHUNK_SIZE)
        if c * CHUNK_SIZE + _n_train_per_chunk <= n_valid
    ])
    test_idx = np.concatenate([
        np.arange(c * CHUNK_SIZE + _n_train_per_chunk,
                  min((c + 1) * CHUNK_SIZE, n_valid))
        for c in range(n_valid // CHUNK_SIZE)
    ])
    _n_chunks = n_valid // CHUNK_SIZE
    print(f"\nTrain/test split (style='{CHUNK_STYLE}', test={TEST_PERCENTAGE:.0f}%, "
          f"{_n_chunks} chunks x {CHUNK_SIZE} trials, "
          f"{_n_train_per_chunk} train + {_n_test_per_chunk} test per chunk)")
    print(f"  train: {len(train_idx)} windows  |  test: {len(test_idx)} windows")
else:
    _n_train  = int(n_valid * TRAIN_FRAC)
    train_idx = np.arange(_n_train)
    test_idx  = np.arange(_n_train, n_valid)
    print(f"\nTrain/test split (style='{CHUNK_STYLE}', test={TEST_PERCENTAGE:.0f}%)")
    print(f"  train: {len(train_idx)} windows  |  test: {len(test_idx)} windows")

# ── interleaved split explanation ────────────────────────────────────────────
# Within each chunk of CHUNK_SIZE trials, the LAST _n_test_per_chunk trials
# are test, the rest are train. This distributes test trials evenly across
# the session timeline rather than bunching them at the end.
# Example with CHUNK_SIZE=50, test=20%: trials 0-39 train, 40-49 test,
#   50-89 train, 90-99 test, etc.
# Note: trial ORDER within valid_trials_beh is chronological (earliest trial
# first), so test trials are temporally spread across the recording.
if CHUNK_STYLE == 'interleaved':
    print(f"  [interleaved detail] test trials are the last {_n_test_per_chunk} of each "
          f"{CHUNK_SIZE}-trial chunk → distributed across session timeline")

print(f"\nbin_beh_v shape : {bin_beh_v.shape}")

beh_save = os.path.join(RESULTS_DIR, f'beh_windows_{_VAR_KEY}_{SESSION}.npz')
np.savez_compressed(
    beh_save,
    bin_beh      = bin_beh_v,
    trial_starts = trial_starts[valid_trials_beh],
    trial_ends   = trial_ends[valid_trials_beh],
    train_idx    = train_idx,
    test_idx     = test_idx,
    T_TRIAL      = np.float32(T_TRIAL),
    N_T_BINS     = np.int32(N_T_BINS),
)
print(f"Saved -> {beh_save}")

# %%
_insp_path = os.path.join(RESULTS_DIR, f'trial_inspection_{_VAR_KEY}_{SESSION}.png')
if not os.path.exists(_insp_path):
    print("[PLOT] Generating trial inspection ...")
    _bin_size_s       = T_TRIAL / N_T_BINS
    _n_trials_per_set = max(3, min(10, n_valid // 3))  # adaptive: 3–10 per set for short trial lists
    _n_sets           = min(3, n_valid // _n_trials_per_set)

    if _n_sets >= 1 and n_valid >= _n_trials_per_set:
        _rng_inspect  = np.random.default_rng(seed=42)
        _pool = list(range(n_valid - _n_trials_per_set + 1))
        _rng_inspect.shuffle(_pool)
        _set_starts = []
        for _s in _pool:
            if all(abs(_s - _s2) >= _n_trials_per_set for _s2 in _set_starts):
                _set_starts.append(_s)
            if len(_set_starts) == _n_sets:
                break
        _set_starts.sort()

        _shade_colors = ['#e8eaf6', '#e8f5e9']

        fig_insp, axes_insp = plt.subplots(_n_sets, 1, figsize=(22, 3.5 * _n_sets))
        if _n_sets == 1:
            axes_insp = [axes_insp]

        for _row, _start in enumerate(_set_starts):
            _vis = valid_trials_beh[_start : _start + _n_trials_per_set]
            _t_row_start = trial_starts[_vis[0]]
            _t_row_end   = trial_ends[_vis[-1]]

            _all_bin_edges  = [];  _all_bin_ctrs   = []
            _all_scounts    = [];  _all_binned_beh = []

            for _ti, _k in enumerate(_vis):
                _t0 = trial_starts[_k];  _t1 = trial_ends[_k]
                _be = _t0 + np.arange(N_T_BINS + 1) * _bin_size_s
                _all_bin_edges.append(_be)
                _all_bin_ctrs.append(0.5 * (_be[:-1] + _be[1:]))
                _smask = (spike_t_gate >= _t0) & (spike_t_gate < _t1)
                _sc, _ = np.histogram(spike_t_gate[_smask], bins=_be)
                _all_scounts.append(_sc)
                _all_binned_beh.append(bin_beh_v[_start + _ti])

            _bin_ctrs_all = np.concatenate(_all_bin_ctrs)
            _scounts_all  = np.concatenate(_all_scounts)
            _bmask = (beh_t >= _t_row_start) & (beh_t <= _t_row_end)

            ax_l = axes_insp[_row]
            ax_r = ax_l.twinx()

            for _ti, _k in enumerate(_vis):
                ax_l.axvspan(trial_starts[_k], trial_ends[_k],
                             facecolor=_shade_colors[_ti % 2], alpha=0.55, zorder=0)

            _all_bin_edge_flat = np.unique(np.concatenate(_all_bin_edges))
            for _be_val in _all_bin_edge_flat:
                ax_l.axvline(_be_val, color='silver', lw=0.35, zorder=1)

            for _k in _vis:
                ax_l.axvline(trial_starts[_k], color='#333333', lw=1.3,
                             linestyle='--', zorder=2, alpha=0.75)
            ax_l.axvline(trial_ends[_vis[-1]], color='#333333', lw=1.3,
                         linestyle='--', zorder=2, alpha=0.75)

            ax_l.bar(_bin_ctrs_all, _scounts_all, width=_bin_size_s * 0.88,
                     color='steelblue', alpha=0.55, edgecolor='none', zorder=3,
                     label='spike count (MUA)')
            ax_l.set_ylabel('spikes / bin', fontsize=8, color='steelblue')
            ax_l.tick_params(axis='y', labelcolor='steelblue', labelsize=7)
            ax_l.set_xlim(_t_row_start - _bin_size_s * 0.1,
                          _t_row_end   + _bin_size_s * 0.1)

            # raw continuous trace — show beh_z (pre-demean) as context,
            # labelled clearly; binned stairs show the actual decoder input
            ax_r.plot(beh_t[_bmask], beh_z[_bmask],
                      color='darkgray', lw=0.7, alpha=0.9, zorder=4,
                      label=f'{BEHAVIOR_VAR} raw (pre-demean)' if DEMEAN_BEH else f'{BEHAVIOR_VAR} raw')
            for _ti, _be in enumerate(_all_bin_edges):
                _lbl = (f'{BEHAVIOR_VAR} binned (demeaned)' if DEMEAN_BEH else f'{BEHAVIOR_VAR} binned') if _ti == 0 else None
                ax_r.stairs(_all_binned_beh[_ti], _be, color='tab:orange', lw=1.8, zorder=5,
                            baseline=None, label=_lbl)
            ax_r.set_ylabel(f'{BEHAVIOR_VAR} ({"demeaned" if DEMEAN_BEH else "z"})', fontsize=8, color='tab:orange')
            ax_r.tick_params(axis='y', labelcolor='tab:orange', labelsize=7)

            ax_top = ax_l.twiny()
            ax_top.set_xlim(ax_l.get_xlim())
            _trial_ctrs = [(trial_starts[_k] + trial_ends[_k]) / 2 for _k in _vis]
            ax_top.set_xticks(_trial_ctrs)
            ax_top.set_xticklabels(
                [str(valid_trials_beh[_start + _ti]) for _ti in range(_n_trials_per_set)],
                fontsize=5, rotation=90)
            ax_top.set_xlabel('trial index', fontsize=7)

            _h1, _l1 = ax_l.get_legend_handles_labels()
            _h2, _l2 = ax_r.get_legend_handles_labels()
            ax_l.legend(_h1 + _h2, _l1 + _l2, fontsize=7, loc='upper right', framealpha=0.7)
            ax_l.set_title(
                f"Set {_row+1}  |  t = {_t_row_start:.1f} – {_t_row_end:.1f} s  "
                f"|  {_n_trials_per_set} trials × {T_TRIAL*1e3:.0f} ms  "
                f"(bin = {_bin_size_s*1e3:.0f} ms)"
                + ("  [selected]" if USING_SELECTED_TRIALS else ""),
                fontsize=8, loc='left')
            if _row == _n_sets - 1:
                ax_l.set_xlabel('Time from SpikeGLX gate open (s)', fontsize=8)

        fig_insp.suptitle(
            f"Trial inspection — {SESSION} — {BEHAVIOR_VAR} vs spike count"
            + (f"\n[selected trials: {_sel_meta['pc']} {_sel_meta['align_mode']}-aligned, "
               f"RMS ≤ {_sel_meta['rms_threshold']}]" if USING_SELECTED_TRIALS else ""),
            fontsize=9)
        plt.tight_layout()
        fig_insp.savefig(_insp_path, dpi=130, bbox_inches='tight')
        print(f"Saved → {_insp_path}")
        plt.close(fig_insp)
    else:
        print(f"[WARN] Not enough valid trials ({n_valid}) for trial inspection plot — skipping.")

# %%
import torch
from density_decoding.decode_pipeline import decode_pipeline
from density_decoding.decoders.behavior_decoder import sliding_window_decoder

print("=" * 60)
print("GPU / DEVICE CHECK")
print("=" * 60)
print(f"  torch version      : {torch.__version__}")
print(f"  CUDA available     : {torch.cuda.is_available()}")
print(f"  GPU count          : {torch.cuda.device_count()}")
for _i in range(torch.cuda.device_count()):
    print(f"  GPU {_i}              : {torch.cuda.get_device_name(_i)}")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"  -> using device     : {device}")
print("=" * 60)

print("=" * 60)
print(f"DECODE  {BEHAVIOR_VAR}{'  (z-scored)' if Z_SCORE_BEH else ''}  —  temporal train/test split")
if USING_SELECTED_TRIALS:
    print(f"  [selected trials mode: {_sel_meta['pc']}, {_sel_meta['align_mode']}-aligned, "
          f"RMS ≤ {_sel_meta['rms_threshold']}, n={n_trials_beh}]")
print("=" * 60)

WEIGHT_CKPT = os.path.join(RESULTS_DIR, f'ckpt_weight_matrix_{_VAR_KEY}_{SESSION}.npy')
if os.path.exists(WEIGHT_CKPT):
    print(f"\n[CHECKPOINT] Loading weight matrix -> {WEIGHT_CKPT}")
    weight_matrix = np.load(WEIGHT_CKPT)
    print(f"  shape: {weight_matrix.shape}")
else:
    _sf_all = np.concatenate([
        np.concatenate([_b for _b in _w if len(_b) > 0])
        for _w in bin_sf_v if any(len(_b) > 0 for _b in _w)
    ])
    _sf_chans = _sf_all[:, 0].astype(int)
    _sf_feats = _sf_all[:, 1:]
    _unique_chans_diag = np.unique(_sf_chans)
    print("\n[GMM DIAGNOSTIC] Feature (x, z, ptp) mean ± std — all channels")
    print(f"  total spikes: {len(_sf_all):,}   unique channels: {len(_unique_chans_diag)}")
    del _sf_all, _sf_chans, _sf_feats

    import threading as _threading
    _advi_epoch_ckpt = os.path.join(RESULTS_DIR, f'ckpt_{_VAR_KEY}_{SESSION}_advi_epoch.pt')
    _advi_live_plot  = os.path.join(RESULTS_DIR, f'advi_loss_live_{_VAR_KEY}_{SESSION}.png')
    _glm_losses_ckpt = os.path.join(RESULTS_DIR, f'ckpt_{_VAR_KEY}_{SESSION}_glm_losses.npz')
    _glm_live_plot   = os.path.join(RESULTS_DIR, f'glm_loss_live_{_VAR_KEY}_{SESSION}.png')

    def _advi_loss_monitor(ckpt_path, plot_path, stop_evt, interval=60):
        import matplotlib;  matplotlib.use('Agg')
        import matplotlib.pyplot as _plt
        _last_mtime = 0
        while not stop_evt.wait(interval):
            if not os.path.exists(ckpt_path):
                continue
            try:
                _mtime = os.path.getmtime(ckpt_path)
                if _mtime <= _last_mtime:
                    continue
                _last_mtime = _mtime
                _ec = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                _tr = _ec.get('elbos', []);  _vl = _ec.get('val_elbos', [])
                if not _tr:
                    continue
                _fig, _ax = _plt.subplots(figsize=(12, 4))
                _ax.plot(np.arange(len(_tr)), _tr, lw=0.8, color='steelblue', alpha=0.85, label='Train ELBO')
                if _vl:
                    _ax.plot([v[0] for v in _vl], [v[1] for v in _vl],
                             lw=1.2, color='darkorange', marker='o', markersize=3, label='Val ELBO')
                _ax.set_xlabel('Iteration');  _ax.set_ylabel('ELBO')
                _ax.set_title(f'ADVI training (live) — {BEHAVIOR_VAR}  {SESSION}', fontsize=9)
                _ax.legend(fontsize=8);  _ax.grid(True, alpha=0.25)
                _plt.tight_layout()
                _plt.savefig(plot_path, dpi=130, bbox_inches='tight')
                _plt.close(_fig)
            except Exception:
                pass

    def _glm_loss_monitor(ckpt_path, plot_path, stop_evt, interval=30):
        import matplotlib;  matplotlib.use('Agg')
        import matplotlib.pyplot as _plt
        _last_mtime = 0
        while not stop_evt.wait(interval):
            if not os.path.exists(ckpt_path):
                continue
            try:
                _mtime = os.path.getmtime(ckpt_path)
                if _mtime <= _last_mtime:
                    continue
                _last_mtime = _mtime
                _gl = np.load(ckpt_path)
                _tr = _gl['train_losses']
                if len(_tr) == 0:
                    continue
                _fig, axes_glm = _plt.subplots(1, 2, figsize=(14, 4))
                axes_glm[0].plot(np.arange(len(_tr)), _tr, lw=0.8, color='steelblue', alpha=0.85, label='Train')
                axes_glm[0].set_xlabel('Epoch');  axes_glm[0].set_ylabel('Poisson NLL')
                axes_glm[0].set_title(f'GLM loss (live) — {BEHAVIOR_VAR}  {SESSION}', fontsize=9)
                axes_glm[0].legend(fontsize=8);  axes_glm[0].grid(True, alpha=0.25)
                _tail = max(1, len(_tr) * 4 // 5)
                axes_glm[1].plot(np.arange(_tail, len(_tr)), _tr[_tail:], lw=0.8, color='steelblue')
                axes_glm[1].set_title('Last 20% (convergence zoom)', fontsize=9)
                axes_glm[1].set_xlabel('Epoch');  axes_glm[1].grid(True, alpha=0.25)
                _plt.tight_layout()
                _plt.savefig(plot_path, dpi=130, bbox_inches='tight')
                _plt.close(_fig)
            except Exception:
                pass

    _stop_monitor = _threading.Event()
    _threading.Thread(target=_advi_loss_monitor,
                      args=(_advi_epoch_ckpt, _advi_live_plot, _stop_monitor),
                      daemon=True).start()
    _threading.Thread(target=_glm_loss_monitor,
                      args=(_glm_losses_ckpt, _glm_live_plot, _stop_monitor),
                      daemon=True).start()
    print(f"  Live ADVI loss plot → {_advi_live_plot}  (updated every 60 s)")
    print(f"  Live GLM  loss plot → {_glm_live_plot}  (updated every 30 s)")

    # ── safe batch size: decode_pipeline batches over TRIALS (not spikes).
    # list(zip(*(iter(train),)*batch_size)) returns [] when len(train) < batch_size
    # → empty batch_idxs → crash in advi at batch_idxs[0].
    _n_train_trials = len(train_idx)
    _batch_size = min(_args.batch_size, max(1, _n_train_trials))
    if _batch_size < _args.batch_size:
        print(f"[WARN] Only {_n_train_trials} training trials — "
              f"reducing batch_size from {_args.batch_size} to {_batch_size}")
    else:
        print(f"[INFO] batch_size={_batch_size}  ({_n_train_trials} training trials)")

    weight_matrix = decode_pipeline(
        data_loader,
        bin_sf_v,
        bin_ti_v,
        bin_tmi_v,
        bin_behaviors    = bin_beh_v,
        behavior_type    = "continuous",
        train            = train_idx,
        test             = test_idx,
        learning_rate    = LEARNING_RATE,
        device           = device,
        n_workers        = N_WORKERS,
        checkpoint_dir   = RESULTS_DIR,
        max_iter         = 7000,
        checkpoint_name  = f"{_VAR_KEY}_{SESSION}",
        early_stopping   = _args.early_stopping,
        es_patience      = _args.es_patience,
        batch_size       = _batch_size,
    )
    _stop_monitor.set()
    np.save(WEIGHT_CKPT, weight_matrix)
    print(f"\n[CHECKPOINT] Saved weight matrix -> {WEIGHT_CKPT}")

# ── ADVI loss curve (plot-guarded) ────────────────────────────────────────────
_elbos_npz  = os.path.join(RESULTS_DIR, f'ckpt_{_VAR_KEY}_{SESSION}_elbos.npz')
_loss_plot  = os.path.join(RESULTS_DIR, f'advi_loss_{_VAR_KEY}_{SESSION}.png')
if os.path.exists(_elbos_npz) and not os.path.exists(_loss_plot):
    _ed = np.load(_elbos_npz)
    fig_l, ax_l = plt.subplots(figsize=(12, 4))
    ax_l.plot(np.arange(len(_ed['train_elbos'])), _ed['train_elbos'],
              lw=0.8, color='steelblue', alpha=0.85, label='Train ELBO')
    if len(_ed['val_iters']) > 0:
        ax_l.plot(_ed['val_iters'], _ed['val_elbos'],
                  lw=1.2, color='darkorange', marker='o', markersize=2, label='Val ELBO')
    ax_l.set_xlabel('Iteration');  ax_l.set_ylabel('ELBO')
    ax_l.set_title(f'ADVI training — {BEHAVIOR_VAR}  {SESSION}', fontsize=9)
    ax_l.legend(fontsize=8);  ax_l.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(_loss_plot, dpi=130, bbox_inches='tight')
    print(f"Saved → {_loss_plot}")
    plt.close()

# %%
_post_params_path = os.path.join(RESULTS_DIR, f'ckpt_{_VAR_KEY}_{SESSION}_post_params.npz')
_pn_save          = os.path.join(RESULTS_DIR, f'putative_neurons_{_VAR_KEY}_{SESSION}.png')
_diag_save        = os.path.join(RESULTS_DIR, f'trial_diagnostics_{_VAR_KEY}_{SESSION}.png')
if os.path.exists(_post_params_path) and not os.path.exists(_pn_save):
    print("\n[PUTATIVE NEURONS] Loading post_params ...")
    _pp          = np.load(_post_params_path)
    _pn_means    = _pp['means']
    _pn_beta     = _pp['U'] @ _pp['V']
    _pn_activity = weight_matrix.sum(axis=(0, 2))
    n_pn = len(_pn_means)

    _act_99   = np.percentile(_pn_activity, 99)
    _act_norm = np.clip(_pn_activity / max(_act_99, 1e-9), 0, 1)
    _pn_sizes = 10 + 190 * _act_norm
    _beta_mean = _pn_beta.mean(axis=1)
    _t_bins_ms = np.arange(N_T_BINS) * (T_TRIAL / N_T_BINS) * 1000

    fig_pn, axes_pn = plt.subplots(1, 3, figsize=(18, 12),
                                    gridspec_kw={'width_ratios': [1, 1, 2]})
    fig_pn.suptitle(f"Putative neurons — {SESSION} — {BEHAVIOR_VAR}\n"
                    f"{n_pn} components  |  marker size ∝ total posterior weight",
                    fontsize=10)

    _bm_abs = max(np.abs(_beta_mean).max(), 1e-9)
    sc_pn0 = axes_pn[0].scatter(_pn_means[:, 0], _pn_means[:, 1], s=_pn_sizes, c=_beta_mean,
                                  cmap='RdBu_r', vmin=-_bm_abs, vmax=_bm_abs,
                                  edgecolors='k', linewidths=0.3, alpha=0.85)
    plt.colorbar(sc_pn0, ax=axes_pn[0], label=f'Mean β ({BEHAVIOR_VAR})', shrink=0.6)
    axes_pn[0].set_xlabel('x (µm)');  axes_pn[0].set_ylabel('Depth z (µm)')
    axes_pn[0].set_title('Behavioural tuning (mean β)', fontsize=9)
    axes_pn[0].grid(True, alpha=0.2)

    _ptp99 = np.percentile(_pn_means[:, 2], 99)
    sc_pn1 = axes_pn[1].scatter(_pn_means[:, 0], _pn_means[:, 1], s=_pn_sizes,
                                  c=np.clip(_pn_means[:, 2], 0, _ptp99),
                                  cmap='hot_r', vmin=0, vmax=_ptp99,
                                  edgecolors='k', linewidths=0.3, alpha=0.85)
    plt.colorbar(sc_pn1, ax=axes_pn[1], label='Mean PTP amplitude (µV)', shrink=0.6)
    axes_pn[1].set_xlabel('x (µm)');  axes_pn[1].set_ylabel('Depth z (µm)')
    axes_pn[1].set_title('Amplitude (mean PTP µV)', fontsize=9)
    axes_pn[1].grid(True, alpha=0.2)

    _top20  = np.argsort(_pn_activity)[-20:][::-1]
    _cmap20 = plt.cm.tab20(np.linspace(0, 1, len(_top20)))
    for _ci, _idx in enumerate(_top20):
        axes_pn[2].plot(_t_bins_ms, _pn_beta[_idx], color=_cmap20[_ci], lw=1.2, alpha=0.85,
                        label=f'#{_idx}  act={_pn_activity[_idx]:.0f}')
    axes_pn[2].axhline(0, color='k', lw=0.6, ls='--', alpha=0.4)
    axes_pn[2].set_xlabel(f'Time within trial (ms, bin={T_TRIAL/N_T_BINS*1000:.0f} ms)')
    axes_pn[2].set_ylabel(f'β ({BEHAVIOR_VAR} tuning)')
    axes_pn[2].set_title('β(t) tuning curves — top-20 most active', fontsize=9)
    axes_pn[2].legend(fontsize=5, loc='upper right', ncol=2, framealpha=0.7)
    axes_pn[2].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(_pn_save, dpi=130, bbox_inches='tight')
    print(f"Saved → {_pn_save}")
    plt.close(fig_pn)

# ── TRIAL DIAGNOSTIC: beta heatmap + mean firing rate + mean behaviour ─────────
# Three panels sharing the x-axis (time within trial in ms):
#   Row 0: U @ V (GLM tuning coefficients, from post_params) — has negative values,
#          matches advi_diagnostics. Falls back to weight_matrix mean if not available.
#   Row 1: mean spike count per bin across training trials (individual traces + mean±SD)
#   Row 2: mean binned behaviour per bin across training trials (individual traces + mean±SD)
#
# Colorbar is placed in a dedicated axes to the right so it does NOT steal width
# from the imshow axes — all three rows stay x-aligned.
# Always regenerate — bin_beh_v (and demean flag) may have changed
print("[PLOT] Generating trial diagnostics ...")
if True:
    import matplotlib.gridspec as _gs
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    _t_bins_ms_diag = np.arange(N_T_BINS) * (T_TRIAL / N_T_BINS * 1000)  # ms

    # ── beta matrix: use U @ V (GLM tuning) from post_params if available ─────
    # This matches advi_diagnostics and shows negative (inhibitory) tuning.
    # weight_matrix is always positive (mixture weights) — wrong for this plot.
    _post_params_path_diag = os.path.join(RESULTS_DIR,
                                          f'ckpt_{_VAR_KEY}_{SESSION}_post_params.npz')
    if os.path.exists(_post_params_path_diag):
        _pp_diag   = np.load(_post_params_path_diag)
        _beta_hmap = _pp_diag['U'] @ _pp_diag['V']   # (n_c, n_t) — GLM tuning coefficients
        _beta_label = 'GLM tuning coefficient (U @ V)'
        _beta_cmap  = 'RdBu_r'
    else:
        # fallback: mean posterior weight across training trials
        _wm_train  = weight_matrix[train_idx]           # (n_train, n_c, n_t)
        _beta_hmap = _wm_train.mean(axis=0)             # (n_c, n_t)
        _beta_label = 'Mean posterior weight (weight_matrix)'
        _beta_cmap  = 'hot_r'
        print("  [WARN] post_params not found — using weight_matrix mean (no negative values)")
    _bmax = float(np.percentile(np.abs(_beta_hmap), 99))

    # ── spike counts per bin for training trials ──────────────────────────────
    _spike_counts = np.zeros((len(train_idx), N_T_BINS))
    for _new_k, _k in enumerate(train_idx):
        for _tb in range(N_T_BINS):
            _spike_counts[_new_k, _tb] = len(bin_sf_v[_k][_tb])
    _sc_mean = _spike_counts.mean(axis=0)
    _sc_std  = _spike_counts.std(axis=0)

    # ── binned behaviour for training trials ──────────────────────────────────
    _beh_train = bin_beh_v[train_idx]
    _beh_mean  = _beh_train.mean(axis=0)
    _beh_std   = _beh_train.std(axis=0)

    # ── layout: 2-column gridspec — col 0 = plots (wide), col 1 = colorbar (thin)
    # All three plot rows share x-axis. Colorbar only sits next to row 0.
    fig_diag = plt.figure(figsize=(15, 11))
    _spec = _gs.GridSpec(
        3, 2, figure=fig_diag,
        hspace=0.08, wspace=0.03,
        height_ratios=[2.5, 1.5, 1.5],
        width_ratios=[1, 0.03],
    )
    ax0  = fig_diag.add_subplot(_spec[0, 0])
    ax1  = fig_diag.add_subplot(_spec[1, 0], sharex=ax0)
    ax2  = fig_diag.add_subplot(_spec[2, 0], sharex=ax0)
    ax_cb = fig_diag.add_subplot(_spec[0, 1])   # colorbar lives here

    # make rows 1 and 2 colorbar columns invisible so layout is clean
    fig_diag.add_subplot(_spec[1, 1]).set_visible(False)
    fig_diag.add_subplot(_spec[2, 1]).set_visible(False)

    fig_diag.suptitle(
        f"{SESSION} - {BEHAVIOR_VAR} trial diagnostics\n"
        f"training trials (n={len(train_idx)})  |  "
        f"bin={T_TRIAL/N_T_BINS*1000:.0f} ms",
        fontsize=11,
    )

    # row 0: beta heatmap
    _im = ax0.imshow(
        _beta_hmap, aspect='auto', cmap=_beta_cmap,
        vmin=-_bmax, vmax=_bmax,
        extent=[_t_bins_ms_diag[0], _t_bins_ms_diag[-1],
                _beta_hmap.shape[0] - 0.5, -0.5],
        interpolation='nearest',
    )
    plt.colorbar(_im, cax=ax_cb, label=_beta_label)
    ax_cb.yaxis.set_label_position('right')
    ax_cb.yaxis.tick_right()
    ax0.set_ylabel('GMM component', fontsize=9)
    ax0.set_title('Beta heatmap — ' + _beta_label, fontsize=9, loc='left')
    plt.setp(ax0.get_xticklabels(), visible=False)

    # row 1: spike counts — bar chart of mean per bin, error bars = 1 SD
    _bar_width = (T_TRIAL / N_T_BINS * 1000) * 0.75   # 75% of bin width in ms
    ax1.bar(_t_bins_ms_diag, _sc_mean, width=_bar_width,
            color='steelblue', alpha=0.75, label='mean', zorder=3)
    ax1.errorbar(_t_bins_ms_diag, _sc_mean, yerr=_sc_std,
                 fmt='none', color='steelblue', elinewidth=1.2,
                 capsize=2.5, alpha=0.9, zorder=4, label='± 1SD')
    ax1.set_ylabel('Spikes / bin', fontsize=9)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.15, axis='y')
    plt.setp(ax1.get_xticklabels(), visible=False)

    # row 2: binned behaviour — individual trials (faint) + mean ± SD
    for _ti in range(len(train_idx)):
        ax2.plot(_t_bins_ms_diag, _beh_train[_ti], color='darkorange',
                 alpha=0.15, lw=0.6)
    ax2.fill_between(_t_bins_ms_diag, _beh_mean - _beh_std, _beh_mean + _beh_std,
                     color='darkorange', alpha=0.25)
    ax2.plot(_t_bins_ms_diag, _beh_mean, color='darkorange', lw=1.8, label='mean ± 1SD')
    ax2.set_xlabel('Time within trial (ms)', fontsize=9)
    ax2.set_ylabel(f'{BEHAVIOR_VAR} ({"demeaned" if DEMEAN_BEH else "z"})', fontsize=9)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.15)

    plt.savefig(_diag_save, dpi=130, bbox_inches='tight')
    print(f"Saved → {_diag_save}")
    plt.close(fig_diag)

_n_wm = weight_matrix.shape[0]
_bin_beh_for_decode = bin_beh_v[:_n_wm]

print(f"\n[DECODE] Running decoder on held-out test windows ...")
_, obs_beh, pred_beh, metrics_beh = sliding_window_decoder(
    weight_matrix, _bin_beh_for_decode, train_idx, test_idx,
    behavior_type="continuous", verbose=False,
)
print(f"\n{'='*52}")
print(f"  {BEHAVIOR_VAR}  R² = {metrics_beh['r2']:+.4f}   r = {metrics_beh['corr']:.4f}")
print(f"  (n_test={len(test_idx)} windows, held out from ADVI + ridge)")
if USING_SELECTED_TRIALS:
    print(f"  [selected trials: {_sel_meta['pc']} {_sel_meta['align_mode']}-aligned, "
          f"RMS ≤ {_sel_meta['rms_threshold']}, n={n_trials_beh}]")
print(f"{'='*52}\n")

# %%
_fig_path = os.path.join(RESULTS_DIR, f'decode_traces_{_VAR_KEY}_{SESSION}.png')
# Always regenerate — decoder inputs may have changed
print(f"[PLOT] Generating decoded vs observed traces -> {_fig_path}")
if True:

    # sliding_window_decoder uses a window of size 7 (half=3) so the first and
    # last few bins of each trial have no prediction.
    _half_win    = 7 // 2
    _n_win       = N_T_BINS - _half_win + 1
    _n_pred_bins = _n_win - _half_win          # predicted bins per trial
    _bin_dur_s   = T_TRIAL / N_T_BINS
    # bin centres relative to trial start for the predicted bins
    _bin_offsets = np.arange(_half_win, _n_win) * _bin_dur_s   # (n_pred_bins,) in seconds

    _n_test = len(test_idx)
    _pred_by_trial = pred_beh.reshape(_n_test, _n_pred_bins)   # (n_test, n_pred_bins)
    _obs_by_trial  = obs_beh.reshape(_n_test, _n_pred_bins)

    # ── one panel per test trial ──────────────────────────────────────────────
    # Panels are arranged in a grid: max 4 columns, rows as needed.
    _ncols = min(4, _n_test)
    _nrows = int(np.ceil(_n_test / _ncols))
    _panel_w = max(3.5, T_TRIAL * 0.8)   # width scales with trial length

    fig, axes = plt.subplots(
        _nrows, _ncols,
        figsize=(_ncols * _panel_w, _nrows * 3.0),
        squeeze=False,
        sharey=True,
    )
    fig.suptitle(
        f"Density decoding — {SESSION} — {BEHAVIOR_VAR}\n"
        f"R² = {metrics_beh['r2']:+.4f}   r = {metrics_beh['corr']:.4f}"
        f"  ({_n_test} test trials, one panel each)"
        + (f"\n[selected: {_sel_meta['pc']} {_sel_meta['align_mode']}-aligned, "
           f"RMS ≤ {_sel_meta['rms_threshold']}, n={n_trials_beh}]"
           if USING_SELECTED_TRIALS else ""),
        fontsize=10,
    )

    for _j in range(_n_test):
        _row, _col = divmod(_j, _ncols)
        _ax = axes[_row, _col]

        # x-axis: time within trial in seconds (0 → T_TRIAL)
        _t_within = _bin_offsets   # seconds from trial start

        _ax.plot(_t_within, _obs_by_trial[_j],  color='darkgray', lw=1.2, label='observed')
        _ax.plot(_t_within, _pred_by_trial[_j], color='tab:blue', lw=1.2,
                 alpha=0.85, label='decoded')

        _orig_trial = valid_trials_beh[test_idx[_j]]
        _t_abs = trial_starts[_orig_trial]
        _ax.set_title(
            f"test trial {_j}  (t={_t_abs:.1f} s)",
            fontsize=7, pad=2,
        )
        _ax.set_xlabel('Time within trial (s)', fontsize=7)
        if _col == 0:
            _ax.set_ylabel(f'{BEHAVIOR_VAR} ({"demeaned" if DEMEAN_BEH else "z"})', fontsize=7)
        _ax.tick_params(labelsize=6)
        _ax.grid(True, alpha=0.18)
        _ax.set_xlim(0, T_TRIAL)
        if _j == 0:
            _ax.legend(fontsize=6, loc='upper right')

    # hide unused panels
    for _j in range(_n_test, _nrows * _ncols):
        _row, _col = divmod(_j, _ncols)
        axes[_row, _col].set_visible(False)

    plt.tight_layout()
    fig.savefig(_fig_path, dpi=130, bbox_inches='tight')
    print(f"Figure   -> {_fig_path}")
    plt.close(fig)

# %%
_decode_npz = os.path.join(RESULTS_DIR, f'decode_results_{_VAR_KEY}_{SESSION}.npz')
_decode_txt = os.path.join(RESULTS_DIR, f'decode_results_{_VAR_KEY}_{SESSION}.txt')

np.savez_compressed(
    _decode_npz,
    bin_beh          = _bin_beh_for_decode,
    obs_test         = obs_beh,
    pred_test        = pred_beh,
    train_idx        = train_idx,
    test_idx         = test_idx,
    weight_matrix    = weight_matrix,
    trial_starts     = trial_starts[valid_trials_beh],
    trial_ends       = trial_ends[valid_trials_beh],
    T_TRIAL          = np.float32(T_TRIAL),
    N_T_BINS         = np.int32(N_T_BINS),
)
print(f"Arrays   -> {_decode_npz}")

_ts_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
summary_lines = [
    f"Density decoding results — {SESSION}",
    f"Saved: {_ts_now}",
    f"",
    f"--- Design ---",
    f"  BEHAVIOR_VAR       : {BEHAVIOR_VAR}",
    f"  Z_SCORE_BEH        : {Z_SCORE_BEH}",
    f"  Window length      : {T_TRIAL:.4f} s",
    f"  N_T_BINS           : {N_T_BINS}  (bin size {BIN_SIZE_MS:.3g} ms → actual {T_TRIAL/N_T_BINS*1000:.2f} ms/bin)",
    f"  Valid windows      : {n_valid}  (train={len(train_idx)}, test={len(test_idx)})",
    f"  Embargo trials     : {_args.embargo_trials}",
    f"  Speed gate         : {'threshold=%.4g (%.1f%% kept)' % (ANIMAL_SPEED_GATE, _speed_gate_info.get('pct_stationary', 100.0)) if ANIMAL_SPEED_GATE > 0 else 'OFF'}",
    f"  Drift correction   : DREDge",
    f"  Demean behaviour   : {'YES (per-trial mean subtracted)' if DEMEAN_BEH else 'NO'}",
    f"  Selected trials    : {'YES — ' + _args.selected_trials if USING_SELECTED_TRIALS else 'NO (regular grid)'}",
]
if USING_SELECTED_TRIALS:
    summary_lines += [
        f"    PC              : {_sel_meta['pc']}",
        f"    Alignment       : {_sel_meta['align_mode']}",
        f"    RMS threshold   : {_sel_meta['rms_threshold']}",
        f"    n trials used   : {n_trials_beh}",
    ]
summary_lines += [
    f"",
    f"--- {BEHAVIOR_VAR} decoding ---",
    f"  R²      : {metrics_beh['r2']:+.4f}",
    f"  r       : {metrics_beh['corr']:.4f}",
    f"  n_test  : {len(test_idx)} windows",
]
with open(_decode_txt, 'w') as _f:
    _f.write('\n'.join(summary_lines) + '\n')
print(f"Summary  -> {_decode_txt}")

_params['decode']['animal_speed_gate_info'] = _speed_gate_info
_params['results'] = {
    BEHAVIOR_VAR: {
        'r2':    float(metrics_beh['r2']),
        'corr':  float(metrics_beh['corr']),
        'mse':   float(metrics_beh['mse']),
        'n_test_windows': int(len(test_idx)),
        'note':  'metrics computed on test_idx only',
    }
}
_params['completed'] = _ts_now
with open(_params_json, 'w') as _f:
    json.dump(_params, _f, indent=2)
print(f"Params   -> {_params_json}")

print()
print('\n'.join(summary_lines))