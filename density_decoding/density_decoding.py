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
_ap.add_argument('--t_trial',       type=float, default=1.5,
                 help='Trial window length in seconds (default 1.5)')
_ap.add_argument('--n_t_bins',      type=int,   default=30,
                 help='Number of time bins per window (default 30)')
_ap.add_argument('--learning_rate', type=float, default=1e-3,
                 help='ADVI learning rate (default 1e-3)') 
_ap.add_argument('--n_workers',     type=int,   default=16,
                 help='Number of parallel workers (default 16)')
_ap.add_argument('--chunk_style',   choices=['contiguous', 'interleaved'], default='interleaved',
                 help='Train/test split style: contiguous (80-20 at end) or interleaved (distributed, default)')
_ap.add_argument('--test_percentage', type=float, default=20,
                 help='Percentage of data to use for testing (default 20, range 1-99)')
_ap.add_argument('--chunk_size',    type=int,   default=50,         
                 help='Trials per chunk for interleaved train/test split (default 50); '
                      'only used when chunk_style=interleaved')
_ap.add_argument('--early_stopping', action=_argparse.BooleanOptionalAction, default=True,
                 help='Enable EMA-patience early stopping in ADVI training (--no-early-stopping to disable)')
_ap.add_argument('--es_patience',   type=int,   default=800,
                 help='Early stopping patience in iterations (default 800)')
_ap.add_argument('--zscore_spike_features', action=_argparse.BooleanOptionalAction, default=True,
                 help='Z-score x, z, and ptp independently before GMM/ADVI fitting.')
_ap.add_argument('--embargo_trials',  type=int,   default=1,
                 help='Number of trials to drop from training on each side of every CV test '
                      'block boundary (default 1; set 0 to disable).')
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

_beh_key_dir = _args.behavior_var.lower().replace(' ', '_')
RESULTS_DIR  = os.path.join(
    BASE_DIR,
    "density_decoding",
    f"{_beh_key_dir}_{_args.n_t_bins}bins_{_args.t_trial:.3g}s_{_args.chunk_style}_{'zscoreboth' if _args.zscore_spike_features else 'nozscore'}"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

# %%
# Extract behaviour subset for this session
beh = texture_pca_df[(texture_pca_df['animal'] == animal) & (texture_pca_df['background'] == 'eink')].copy()
beh_t = beh['time'].dt.total_seconds().values   

print(f"=== {animal} eink ===")
print(f"Number of ROWS (= video frames): {len(beh):,}\n")

tt = beh['trial_time'].dt.total_seconds()
abs_t = beh['time'].dt.total_seconds()
print(f"Session duration (trial_time): {tt.min():.4f} s  →  {tt.max():.4f} s")
print(f"Session duration (absolute)  : {abs_t.min():.4f} s  →  {abs_t.max():.4f} s")
print(f"  = {(tt.max() - tt.min()) / 60:.2f} min, {len(beh):,} frames at {len(beh)/(tt.max()-tt.min()):.2f} fps effective\n")

disp_cols = ['dataset', 'trial_idx', 'frame', 'trial_time', 'time', _args.behavior_var]
disp_cols = [c for c in disp_cols if c in beh.columns]
print(f"First 10 rows:")
print(beh[disp_cols].head(10).to_string())

# %% [markdown]
# #### NEURAL DATA

# %%
BEHAVIOR_VAR  = _args.behavior_var
Z_SCORE_BEH   = _args.zscore_spike_features
T_TRIAL       = _args.t_trial
N_T_BINS      = _args.n_t_bins
LEARNING_RATE = _args.learning_rate
N_WORKERS     = _args.n_workers

# Load SI recording parameters
from spikeinterface import NumpyRecording
rec_si      = NumpyRecording.load(PREPROC_DIR)
ch_locs     = rec_si.get_channel_locations()              
ap_fs       = rec_si.get_sampling_frequency()                 
ap_t_offset = float(rec_si._recording_segments[0].t_start)   
n_ch        = rec_si.get_num_channels()
n_samp      = rec_si.get_num_samples()
ap_dtype    = rec_si.get_dtype()

beh_var = beh[BEHAVIOR_VAR].values                
beh_spd = beh['speed'].values if 'speed' in beh.columns else None

OUT_PATH = os.path.join(RESULTS_DIR, f'spike_features_{SESSION}.npz')

if os.path.exists(OUT_PATH):
    print(f"\n[CACHE] Loading cached spike features -> {OUT_PATH}")
    data = np.load(OUT_PATH, allow_pickle=False)
    spike_times_samp = data['times_samples']
    spike_channels   = data['channels']
    spike_features   = data['point_source_localizations']
    
    spike_t_gate = spike_times_samp / ap_fs + ap_t_offset
    spike_x      = spike_features[:, 0]
    z_corr       = spike_features[:, 1]
    spike_ptp    = spike_features[:, 2]
    print(f"Loaded {len(spike_t_gate):,} spikes directly.")
else:
    print("\n[COMPUTE] Computing spike features from raw data...")
    peaks = np.load(PEAKS_FILE)
    loc   = np.load(LOC_FILE)

    spike_times_samp = peaks['sample_index']
    spike_amp        = np.abs(peaks['amplitude']).astype(np.float32)
    spike_x          = loc['x'].astype(np.float32)
    spike_z          = loc['y'].astype(np.float32)
    spike_t_gate     = spike_times_samp / ap_fs + ap_t_offset

    # Compute raw PTP amplitude
    N_BEFORE_PTP  = round(0.5 * ap_fs / 1000)
    N_AFTER_PTP   = round(0.5 * ap_fs / 1000)
    wf_len_ptp    = N_BEFORE_PTP + N_AFTER_PTP
    CHUNK_S_PTP   = 60.0
    chunk_samp_ptp = int(CHUNK_S_PTP * ap_fs)
    n_spikes      = len(peaks)
    spike_ptp     = np.full(n_spikes, np.nan, dtype=np.float32)

    print(f"Computing raw PTP window=±{N_BEFORE_PTP} samples... Chunks={int(np.ceil(n_samp / chunk_samp_ptp))}")
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
            ch_loc_idx = peaks['channel_index'][idxs].astype(int)
            
            full_mask = (s_rel >= N_BEFORE_PTP) & (s_rel + N_AFTER_PTP <= len(traces))
            if full_mask.any():
                t_off  = s_rel[full_mask][:, None] + np.arange(-N_BEFORE_PTP, N_AFTER_PTP)[None, :]
                ch_rep = np.broadcast_to(ch_loc_idx[full_mask][:, None], t_off.shape)
                wfs    = traces[t_off, ch_rep].astype(np.float32)
                spike_ptp[idxs[full_mask]] = np.ptp(wfs, axis=1)
                
            for _ei in np.where(~full_mask)[0]:
                _ws = max(0, s_rel[_ei] - N_BEFORE_PTP)
                _we = min(len(traces), s_rel[_ei] + N_AFTER_PTP)
                _wf = traces[_ws:_we, ch_loc_idx[_ei]].astype(np.float32)
                if len(_wf) > 0:
                    spike_ptp[idxs[_ei]] = np.ptp(_wf)
        chunk_start = chunk_end
        ci += 1

    # DREDge Drift Correction
    motion = si.Motion.load(MOTION_DIR)
    temporal_bins = motion.temporal_bins_s[0]
    spatial_bins  = motion.spatial_bins_um
    z_raw = loc['y'].astype(np.float32)
    disp_at_spike = motion.get_displacement_at_time_and_depth(spike_t_gate, z_raw.astype(np.float64)).astype(np.float32)
    z_corr = z_raw - disp_at_spike

    spike_channels = peaks['channel_index'].astype(np.int32)
    spike_features = np.column_stack([spike_x, z_corr, spike_ptp])
    np.savez(OUT_PATH, times_samples=spike_times_samp.astype(np.int64), channels=spike_channels, point_source_localizations=spike_features, t_start_gate_s=np.float64(ap_t_offset), sampling_rate=np.float64(ap_fs))
    print(f"Saved completed features -> {OUT_PATH}")

# %% [markdown]
# #### PLOT GUARD BLOCKS (DECOUPLED VISUALIZATIONS)

# %%
p999 = float(np.nanpercentile(spike_ptp, 99.9))

# 1. Localization Scatter
_loc_save = os.path.join(RESULTS_DIR, f'spike_localisations_{SESSION}.png')
if not os.path.exists(_loc_save):
    print("[PLOT] Generating localization plot...")
    rng_sf = np.random.default_rng(0)
    n_show = min(200_000, len(spike_x))
    isf    = rng_sf.choice(len(spike_x), n_show, replace=False)
    isf    = isf[np.argsort(spike_ptp[isf])]
    fig, ax = plt.subplots(figsize=(4, 10))
    sc = ax.scatter(spike_x[isf], z_corr[isf], c=np.clip(spike_ptp[isf], 0, p999), s=0.3, alpha=0.3, cmap='hot', rasterized=True)
    plt.colorbar(sc, ax=ax, label='PTP (µV)')
    ax.set_xlabel('x (µm)'); ax.set_ylabel('corrected depth z (µm)')
    plt.tight_layout()
    plt.savefig(_loc_save, dpi=130, bbox_inches='tight')
    plt.close()

# 2. Waveform Grid Visualizer
_wf_val_save = os.path.join(RESULTS_DIR, f'ptp_waveforms_{SESSION}.png')
if not os.path.exists(_wf_val_save) and 'peaks' in locals():
    print("[PLOT] Generating waveform evaluation grid...")
    rng_val = np.random.default_rng(42)
    val_idx = rng_val.choice(len(peaks), 100, replace=False)
    wf_val = np.full((100, wf_len_ptp), np.nan, dtype=np.float32)
    ptp_val = spike_ptp[val_idx].copy()
    for k, idx in enumerate(val_idx):
        pk_ch = int(peaks['channel_index'][idx])
        samp  = int(spike_times_samp[idx])
        t0, t1 = max(0, samp - N_BEFORE_PTP), min(n_samp, samp + N_AFTER_PTP)
        tr = rec_si.get_traces(start_frame=t0, end_frame=t1)[:, pk_ch].astype(np.float32)
        if tr.shape[0] < wf_len_ptp: tr = np.pad(tr, (0, wf_len_ptp - tr.shape[0]))
        wf_val[k] = tr
    order_val = np.argsort(ptp_val)[::-1]
    t_ax_val  = np.arange(wf_len_ptp) - N_BEFORE_PTP
    y_half    = np.nanpercentile(ptp_val, 99) * 0.7
    fig2, axes2 = plt.subplots(10, 10, figsize=(18, 18), sharex=True, sharey=True)
    for pi, k in enumerate(order_val):
        r, c  = divmod(pi, 10)
        ax2   = axes2[r, c]
        ax2.plot(t_ax_val, wf_val[k], color='k', lw=0.8)
        ax2.set_ylim(-y_half, y_half)
    plt.tight_layout()
    plt.savefig(_wf_val_save, dpi=130, bbox_inches='tight')
    plt.close()

# 3. Drift Plot Before / After
_scatter_save = os.path.join(RESULTS_DIR, f'drift_scatter_before_after_{SESSION}.png')
if not os.path.exists(_scatter_save) and 'motion' in locals():
    print("[PLOT] Generating drift correction breakdown scatter...")
    T_VIZ_S = 12 * 60
    t_rec_start, t_rec_end = spike_t_gate.min(), spike_t_gate.max()
    windows = [(f'First 12 min', t_rec_start, t_rec_start + T_VIZ_S), (f'Last 12 min', t_rec_end - T_VIZ_S, t_rec_end)]
    fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharey=True, sharex='row')
    rng_sc = np.random.default_rng(7)
    for row_i, (win_label, t0, t1) in enumerate(windows):
        sel = np.where((spike_t_gate >= t0) & (spike_t_gate < t1))[0]
        if len(sel) > 200_000: sel = rng_sc.choice(sel, 200_000, replace=False)
        for col_i, (z_arr, panel_title) in enumerate([(z_raw[sel], 'Before'), (z_corr[sel], 'After DREDge')]):
            ax = axes[row_i, col_i]
            ax.scatter(spike_t_gate[sel], z_arr, c='gray', s=0.3, alpha=0.25, rasterized=True)
    plt.tight_layout()
    plt.savefig(_scatter_save, dpi=130, bbox_inches='tight')
    plt.close()

# 4. Waveform QC Spatial Mapping
_wf_save = os.path.join(RESULTS_DIR, f'waveform_qc_{SESSION}.png')
if not os.path.exists(_wf_save) and 'peaks' in locals():
    print("[PLOT] Generating multi-channel Waveform Spatial QC...")
    T_POOL_START, T_POOL_END = 100.0, 110.0
    pool_mask = (spike_t_gate >= T_POOL_START) & (spike_t_gate < T_POOL_END)
    pool_idx  = np.where(pool_mask)[0]
    if len(pool_idx) >= 5:
        rng = np.random.default_rng(2026)
        sel = rng.choice(pool_idx, 5, replace=False)
        fig, axes = plt.subplots(1, 5, figsize=(25, 12), sharey=True, sharex=True)
        for pi, _spi in enumerate(sel):
            ax = axes[pi]
            ax.scatter(ch_locs[:, 0], ch_locs[:, 1], s=8, c='lightgray', zorder=1)
            ax.scatter(spike_x[_spi], z_corr[_spi], s=150, c='red', marker='*', zorder=6)
        plt.tight_layout()
        plt.savefig(_wf_save, dpi=130, bbox_inches='tight')
        plt.close()

# %% [markdown]
# #### ALIGNMENT CHECK & FEATURE SCALING

# %%
ZSCORE_SPIKE_FEATURES = _args.zscore_spike_features
if ZSCORE_SPIKE_FEATURES:
    _sf_mean = spike_features.mean(axis=0)
    _sf_std  = spike_features.std(axis=0)
    _sf_std[_sf_std < 1e-9] = 1.0
    spike_features = ((spike_features - _sf_mean) / _sf_std).astype(np.float32)
    _zscore_params_sf = {'x_mean': float(_sf_mean[0]), 'x_std': float(_sf_std[0]), 'z_mean': float(_sf_mean[1]), 'z_std': float(_sf_std[1]), 'ptp_mean': float(_sf_mean[2]), 'ptp_std': float(_sf_std[2])}
else:
    _zscore_params_sf = None

_align_save = os.path.join(RESULTS_DIR, f'alignment_check_{SESSION}.png')
if not os.path.exists(_align_save):
    print("[PLOT] Generating behavioral tracking alignment check plot...")
    BIN_ALIGN = 1.0
    t_lo, t_hi = min(spike_t_gate.min(), beh_t.min()) - 2, max(spike_t_gate.max(), beh_t.max()) + 2
    bins_1s = np.arange(t_lo, t_hi + BIN_ALIGN, BIN_ALIGN)
    t_ctr = (bins_1s[:-1] + bins_1s[1:]) / 2
    mua_counts, _ = np.histogram(spike_t_gate, bins=bins_1s)
    mua_hz = uniform_filter1d(mua_counts.astype(float) / BIN_ALIGN, size=5)
    fig, axes = plt.subplots(3, 1, figsize=(22, 12), sharex=True)
    axes[0].fill_between(t_ctr, mua_hz, alpha=0.7, color='steelblue')
    axes[1].plot(beh_t, beh_var, color='darkorange', lw=0.8)
    axes[2].plot(t_ctr, zscore(mua_hz), color='steelblue', label='MUA')
    plt.tight_layout()
    plt.savefig(_align_save, dpi=130, bbox_inches='tight')
    plt.close()

_pos_save = os.path.join(RESULTS_DIR, 'spike_positions.png')
if not os.path.exists(_pos_save):
    print("[PLOT] Generating spike density profile overlay...")
    fig, ax = plt.subplots(figsize=(6, 14))
    ax.scatter(ch_locs[:, 0], ch_locs[:, 1], s=18, c='lightgray', edgecolors='gray', linewidths=0.5)
    mask = (spike_t_gate >= ap_t_offset) & (spike_t_gate < 5 + ap_t_offset)
    if mask.any():
        sc_h = ax.scatter(spike_x[mask], z_corr[mask], c=spike_ptp[mask], s=50, cmap='hot_r', edgecolors='k', linewidths=0.2)
    plt.tight_layout()
    plt.savefig(_pos_save, dpi=130, bbox_inches='tight')
    plt.close()

# %% [markdown]
# #### BEHAVIOURAL DATA WINDOWING

# %%
beh_raw = beh[BEHAVIOR_VAR].values.astype(np.float32)
beh_z = zscore(beh_raw, nan_policy='omit').astype(np.float32) if Z_SCORE_BEH else beh_raw.copy()

T_OV_START = max(beh_t.min(), ap_t_offset)
T_OV_END   = min(beh_t.max(), spike_t_gate.max())
trial_starts = np.arange(T_OV_START, T_OV_END - T_TRIAL, T_TRIAL)
trial_ends   = trial_starts + T_TRIAL
n_trials_beh = len(trial_starts)

CHUNK_STYLE     = _args.chunk_style
CHUNK_SIZE      = _args.chunk_size
TEST_PERCENTAGE = _args.test_percentage
TEST_FRAC       = TEST_PERCENTAGE / 100.0
TRAIN_FRAC      = 1.0 - TEST_FRAC

_VAR_KEY     = BEHAVIOR_VAR.lower().replace(' ', '_')
_params_json = os.path.join(RESULTS_DIR, f'decode_params_{_VAR_KEY}_{SESSION}.json')
if not os.path.exists(_params_json):
    _params_ts   = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    _params = {
        "session": SESSION, "started": _params_ts,
        "decode": {
            "BEHAVIOR_VAR": BEHAVIOR_VAR, "Z_SCORE_BEH": Z_SCORE_BEH, "T_TRIAL_s": float(T_TRIAL),
            "N_T_BINS": int(N_T_BINS), "LEARNING_RATE": float(LEARNING_RATE), "N_WORKERS": int(N_WORKERS),
            "chunk_style": CHUNK_STYLE, "test_percentage": float(TEST_PERCENTAGE),
            "zscore_spike_features": ZSCORE_SPIKE_FEATURES, "spike_feature_zscore_params": _zscore_params_sf,
        },
        "behavior": { "T_OV_START_s": float(T_OV_START), "T_OV_END_s": float(T_OV_END), "n_windows_total": int(n_trials_beh) }
    }
    with open(_params_json, 'w') as _f: json.dump(_params, _f, indent=2)

_beh_plot_save = os.path.join(RESULTS_DIR, f'behaviour_overview_{_VAR_KEY}_{SESSION}.png')
if not os.path.exists(_beh_plot_save):
    fig, ax = plt.subplots(figsize=(18, 2.5))
    ax.plot(beh_t, beh_z, lw=0.4, color='tab:blue', alpha=0.7)
    plt.tight_layout()
    plt.savefig(_beh_plot_save, dpi=130, bbox_inches='tight')
    plt.close()

# %% [markdown]
# #### PROCESS SPIKE FEATURES & BEHAVIOURS 

# %%
sys.path.insert(0, '/u/gongt/density_decoding')
from density_decoding.utils.data_utils import BaseDataLoader

data_loader = BaseDataLoader(t_before=0.0, t_after=T_TRIAL, n_t_bins=N_T_BINS)
SPIKE_CKPT = os.path.join(RESULTS_DIR, f'ckpt_spikebins_{SESSION}.pkl')

bin_beh_all = data_loader.process_behaviors(time_points=beh_t, raw_behaviors=beh_z, trial_start_times=trial_starts, trial_end_times=trial_ends)
valid_mask_beh = ~np.isnan(bin_beh_all).any(axis=1)
valid_trials_beh = np.where(valid_mask_beh)[0]

if os.path.exists(SPIKE_CKPT):
    with open(SPIKE_CKPT, 'rb') as _f: _ckpt = pickle.load(_f)
    bin_spike_features_all = _ckpt['bin_spike_features_all']
    bin_trial_idxs_all     = _ckpt['bin_trial_idxs_all']
    bin_time_idxs_all      = _ckpt['bin_time_idxs_all']
else:
    bin_spike_features_all, bin_trial_idxs_all, bin_time_idxs_all = data_loader.process_spike_features(
        spike_times=spike_t_gate, spike_channels=spike_channels, spike_features=spike_features,
        trial_start_times=trial_starts, trial_end_times=trial_ends, valid_trials=np.arange(n_trials_beh))
    with open(SPIKE_CKPT, 'wb') as _f: pickle.dump({'bin_spike_features_all': bin_spike_features_all, 'bin_trial_idxs_all': bin_trial_idxs_all, 'bin_time_idxs_all': bin_time_idxs_all}, _f, protocol=4)

bin_beh_v = bin_beh_all[valid_mask_beh]
bin_sf_v  = [bin_spike_features_all[k] for k in valid_trials_beh]
bin_ti_v, bin_tmi_v = [], []
for new_k, k in enumerate(valid_trials_beh):
    for t in range(N_T_BINS):
        bin_ti_v.append(np.full_like(bin_trial_idxs_all[k * N_T_BINS + t], new_k))
        bin_tmi_v.append(bin_time_idxs_all[k * N_T_BINS + t])

n_valid = len(valid_trials_beh)
if CHUNK_STYLE == 'interleaved' and n_valid >= CHUNK_SIZE:
    _n_test_per_chunk  = max(1, int(round(CHUNK_SIZE * TEST_FRAC)))
    _n_train_per_chunk = CHUNK_SIZE - _n_test_per_chunk
    train_idx = np.concatenate([np.arange(c * CHUNK_SIZE, c * CHUNK_SIZE + _n_train_per_chunk) for c in range(n_valid // CHUNK_SIZE) if c * CHUNK_SIZE + _n_train_per_chunk <= n_valid])
    test_idx = np.concatenate([np.arange(c * CHUNK_SIZE + _n_train_per_chunk, min((c + 1) * CHUNK_SIZE, n_valid)) for c in range(n_valid // CHUNK_SIZE)])
else:
    _n_train = int(n_valid * TRAIN_FRAC)
    train_idx = np.arange(_n_train)
    test_idx  = np.arange(_n_train, n_valid)

# 5. Trial Inspection Plot
_insp_path = os.path.join(RESULTS_DIR, f'trial_inspection_{_VAR_KEY}_{SESSION}.png')
if not os.path.exists(_insp_path):
    print("[PLOT] Generating behavioral window extraction quality inspection trace...")
    _bin_size_s = T_TRIAL / N_T_BINS
    fig_insp, axes_insp = plt.subplots(3, 1, figsize=(22, 10.5))
    for _row, _start in enumerate([0, n_valid//2, n_valid-21]):
        _vis = valid_trials_beh[_start : _start + 20]
        ax_l = axes_insp[_row]
        ax_r = ax_l.twinx()
        ax_r.plot(beh_t, beh_z, color='darkgray', lw=0.7, alpha=0.5)
    plt.tight_layout()
    fig_insp.savefig(_insp_path, dpi=130, bbox_inches='tight')
    plt.close(fig_insp)

# %% [markdown]
# #### ADVI LAYER CONFIG & DECODING

# %%
import torch
from density_decoding.decode_pipeline import decode_pipeline
from density_decoding.decoders.behavior_decoder import sliding_window_decoder, kfold_sliding_window_decoder
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

WEIGHT_CKPT = os.path.join(RESULTS_DIR, f'ckpt_weight_matrix_{_VAR_KEY}_{SESSION}.npy')
if os.path.exists(WEIGHT_CKPT):
    print(f"[CACHE] Loading ADVI learned neural mixture weights -> {WEIGHT_CKPT}")
    weight_matrix = np.load(WEIGHT_CKPT)
else:
    print("[COMPUTE] Fitting Variational GMM-GLM via ADVI...")
    weight_matrix = decode_pipeline(
        data_loader, bin_sf_v, bin_ti_v, bin_tmi_v, bin_behaviors=bin_beh_v, behavior_type="continuous",
        train=train_idx, test=test_idx, learning_rate=LEARNING_RATE, device=device, n_workers=N_WORKERS,
        checkpoint_dir=RESULTS_DIR, max_iter=8000, checkpoint_name=f"{_VAR_KEY}_{SESSION}",
        early_stopping=_args.early_stopping, es_patience=_args.es_patience
    )
    np.save(WEIGHT_CKPT, weight_matrix)

# 6. Putative Neurons Diagnostic Map (GLM weights & Spatial Means)
_pn_save = os.path.join(RESULTS_DIR, f'putative_neurons_{_VAR_KEY}_{SESSION}.png')
_post_params_path = os.path.join(RESULTS_DIR, f'ckpt_{_VAR_KEY}_{SESSION}_post_params.npz')
if os.path.exists(_post_params_path) and not os.path.exists(_pn_save):
    print("[DIAGNOSTIC] Generating ADVI/GLM structural tuning map...")
    _pp = np.load(_post_params_path)
    _pn_means = _pp['means']
    _pn_beta = _pp['U'] @ _pp['V']
    _pn_activity = weight_matrix.sum(axis=(0, 2))
    _pn_sizes = 10 + 190 * np.clip(_pn_activity / max(np.percentile(_pn_activity, 99), 1e-9), 0, 1)
    
    fig_pn, axes_pn = plt.subplots(1, 3, figsize=(18, 12), gridspec_kw={'width_ratios': [1, 1, 2]})
    axes_pn[0].scatter(_pn_means[:, 0], _pn_means[:, 1], s=_pn_sizes, c=_pn_beta.mean(axis=1), cmap='RdBu_r')
    axes_pn[1].scatter(_pn_means[:, 0], _pn_means[:, 1], s=_pn_sizes, c=np.clip(_pn_means[:, 2], 0, np.percentile(_pn_means[:, 2], 99)), cmap='hot_r')
    for _idx in np.argsort(_pn_activity)[-20:][::-1]:
        axes_pn[2].plot(_pn_beta[_idx], alpha=0.85)
    plt.tight_layout()
    plt.savefig(_pn_save, dpi=130, bbox_inches='tight')
    plt.close()

# 7. ADDED DIAGNOSTIC: ADVI ELBO Optimization Loss Track
_elbo_csv = os.path.join(RESULTS_DIR, f'ckpt_{_VAR_KEY}_{SESSION}_loss_history.csv')  # or appropriate filename generated by your module
_elbo_plot = os.path.join(RESULTS_DIR, f'advi_elbo_convergence_{SESSION}.png')
if os.path.exists(_elbo_csv) and not os.path.exists(_elbo_plot):
    try:
        print("[DIAGNOSTIC] Generating ADVI optimization convergence track...")
        loss_df = pd.read_csv(_elbo_csv)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(loss_df['iter'], loss_df['loss'], color='purple', label='Training ELBO')
        if 'val_loss' in loss_df.columns:
            ax.plot(loss_df['iter'], loss_df['val_loss'], color='teal', label='Validation Loss')
        ax.set_yscale('log')
        ax.set_xlabel('Iterations'); ax.set_ylabel('Negative ELBO (Log Scale)')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(_elbo_plot, dpi=110)
        plt.close()
    except Exception as e:
        print(f"Skipping ELBO convergence plot: {e}")

# ── RIDGE REGRESSION CODES & DECODING ──
_decode_npz = os.path.join(RESULTS_DIR, f'decode_results_{_VAR_KEY}_{SESSION}.npz')
_decode_txt = os.path.join(RESULTS_DIR, f'decode_results_{_VAR_KEY}_{SESSION}.txt')
_fig_path   = os.path.join(RESULTS_DIR, f'decode_traces_{_VAR_KEY}_{SESSION}.png')

if os.path.exists(_decode_npz) and os.path.exists(_decode_txt):
    print(f"\n[CACHE] Decoding results already exist -> {_decode_npz}")
    with open(_decode_txt, 'r') as _f:
        print(_f.read())
else:
    print("[COMPUTE] Executing Ridge Regression and Cross-Validation decoding loops...")
    _n_wm = weight_matrix.shape[0]
    _bin_beh_for_decode = bin_beh_v[:_n_wm]

    _, obs_beh, pred_beh, metrics_beh = sliding_window_decoder(weight_matrix, _bin_beh_for_decode, train_idx, test_idx, behavior_type="continuous")
    cv_results = kfold_sliding_window_decoder(weight_matrix, _bin_beh_for_decode, n_folds=5,
                                              embargo_trials=_args.embargo_trials)

    if not os.path.exists(_fig_path):
        _half_win = 7 // 2
        _n_win = N_T_BINS - _half_win + 1
        _n_pred_bins = _n_win - _half_win
        _bin_offsets = np.arange(_half_win, _n_win) * (T_TRIAL / N_T_BINS)
        _chunk_of_test = test_idx // CHUNK_SIZE if (CHUNK_STYLE == 'interleaved' and n_valid >= CHUNK_SIZE) else np.zeros(len(test_idx), dtype=int)
        _unique_chunks = np.unique(_chunk_of_test)
        _pred_by_trial = pred_beh.reshape(len(test_idx), _n_pred_bins)
        _obs_by_trial  = obs_beh.reshape(len(test_idx), _n_pred_bins)

        fig, axes = plt.subplots(len(_unique_chunks), 1, figsize=(18, max(2.2, 2.2 * len(_unique_chunks))), squeeze=False)
        
        _fold_r2_str = ", ".join([f"{x:+.3f}" for x in cv_results['fold_r2']])
        _dec_title = (
            f"Decoding Results — {BEHAVIOR_VAR}\n"
            f"Single Split  |  R²: {metrics_beh['r2']:+.4f}   Corr: {metrics_beh['corr']:.4f}   MSE: {metrics_beh['mse']:.4f}\n"
            f"5-Fold CV     |  mean R²: {cv_results['mean_r2']:+.4f} ± {cv_results['std_r2']:.4f}   "
            f"mean Corr: {cv_results['mean_corr']:.4f}   mean MSE: {cv_results['mean_mse']:.4f}\n"
            f"CV Folds R²   |  [{_fold_r2_str}]"
        )
        fig.suptitle(_dec_title, fontsize=11, fontweight='bold')
        
        for _pi, _ci in enumerate(_unique_chunks):
            _tidxs = np.where(_chunk_of_test == _ci)[0]
            _t_ax_chunk = np.concatenate([trial_starts[valid_trials_beh[test_idx[_j]]] + _bin_offsets for _j in _tidxs])
            axes[_pi, 0].plot(_t_ax_chunk, _obs_by_trial[_tidxs].ravel(), color='darkgray', lw=0.8)
            axes[_pi, 0].plot(_t_ax_chunk, _pred_by_trial[_tidxs].ravel(), color='tab:blue', lw=0.8, alpha=0.85)
            
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)  # Make room for the suptitle
        fig.savefig(_fig_path, dpi=130, bbox_inches='tight')
        plt.close()

    np.savez_compressed(
        _decode_npz, bin_beh=_bin_beh_for_decode, pred_test=pred_beh, train_idx=train_idx, test_idx=test_idx,
        weight_matrix=weight_matrix, trial_starts=trial_starts[valid_trials_beh], trial_ends=trial_ends[valid_trials_beh],
        T_TRIAL=np.float32(T_TRIAL), N_T_BINS=np.int32(N_T_BINS),
        cv_fold_r2=cv_results['fold_r2'], cv_fold_corr=cv_results['fold_corr'], cv_fold_mse=cv_results['fold_mse'],
    )
    
    summary_lines = [
        f"--- {BEHAVIOR_VAR} decoding (single train/test split) ---",
        f"  R2   : {metrics_beh['r2']:.4f}",
        f"  corr : {metrics_beh['corr']:.4f}",
        f"  MSE  : {metrics_beh['mse']:.4f}",
        f"--- {BEHAVIOR_VAR} decoding (5-fold CV) ---",
        f"  mean R2  : {cv_results['mean_r2']:+.4f} ± {cv_results['std_r2']:.4f}"
    ]
    with open(_decode_txt, 'w') as _f: _f.write('\n'.join(summary_lines) + '\n')
    print('\n'.join(summary_lines))