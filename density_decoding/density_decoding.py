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
_ap.add_argument('--es_patience',   type=int,   default=1000,
                 help='Early stopping patience in iterations (default 1000)')
_ap.add_argument('--zscore_spike_features', action=_argparse.BooleanOptionalAction, default=True,
                 help='Z-score x, z, and ptp independently before GMM/ADVI fitting. '
                      'Recommended when features have very different scales (e.g. z in µm vs ptp in µV). '
                      'Use --no-zscore-spike-features to disable. '
                      'If a spike-bins checkpoint already exists, delete it before toggling this flag.')
_ap.add_argument('--embargo_trials',  type=int,   default=1,       # NEW
                 help='Number of trials to drop from training on each side of every CV test '
                      'block boundary (default 1; set 0 to disable).')
_ap.add_argument('--context_before_bins', type=int, default=3,
                 help='Number of past time bins to include in the decoder context window '
                      '(default 3; set 0 for instantaneous decoding).')
_ap.add_argument('--context_after_bins', type=int, default=3,
                 help='Number of future time bins to include in the decoder context window '
                      '(default 3; set 0 for past-only decoding).')
_ap.add_argument('--animal_speed_gate', type=float, default=0.0,   ###CHANGED
                 help='If > 0, drop trial windows where the animal mean speed exceeds this '   ###CHANGED
                      'threshold (units match the "speed" column in texture_pca_df, '          ###CHANGED
                      'typically px/frame or normalised). '                                    ###CHANGED
                      'Keeps only stationary periods. 0 = off (default). '                    ###CHANGED
                      'Ported from rrr_texture_neural.py animal-speed gate.')                  ###CHANGED
_args = _ap.parse_args()

DF_DIR = r"/ptmp/gongt/behaviour"

metadata_df    = pd.read_pickle(os.path.join(DF_DIR, "metadata_df.pkl"))
texture_pca_df = pd.read_pickle(os.path.join(DF_DIR, "texture_pca_df_preproc.pkl"))

print(f"Texture PCA: {texture_pca_df.shape}")

animal  = _args.animal
exp     = _args.exp
SESSION = f"{animal}{exp}"                                    # e.g. GLC6962_g_g0

BASE_DIR    = rf"/nexus/posix0/MBR-neuralsystems/ephys/{animal}/{SESSION}"
PREPROC_DIR = os.path.join(BASE_DIR, "preprocessed")
PEAKS_FILE  = os.path.join(PREPROC_DIR, "thresholding_peaks", "peaks_combined2_sorted.npy")
LOC_FILE    = os.path.join(PREPROC_DIR, "thresholding_peaks", "localized_peaks_mt_raw.npy")
MOTION_DIR  = os.path.join(PREPROC_DIR, "thresholding_peaks", "motion_estimates", "motion_dredge")  # NEW: defined upfront

_beh_key_dir = _args.behavior_var.lower().replace(' ', '_')
_speedgate_tag = f"_gate{_args.animal_speed_gate:.3g}" if _args.animal_speed_gate > 0 else ""  ###CHANGED
RESULTS_DIR  = os.path.join(
    BASE_DIR,
    "density_decoding",
    f"{_beh_key_dir}_{_args.n_t_bins}bins_{_args.t_trial:.3g}s_{_args.chunk_style}_"
    f"{'zscoreboth' if _args.zscore_spike_features else 'nozscore'}_embargo{_args.embargo_trials}"
    f"{_speedgate_tag}"                                                                        ###CHANGED
)
os.makedirs(RESULTS_DIR, exist_ok=True)

# %%
sub = texture_pca_df[(texture_pca_df['animal'] == animal) & (texture_pca_df['background'] == 'eink')]

print(f"=== {animal} eink ===")
print(f"Number of ROWS (= video frames, not individual trials): {len(sub):,}")
print()

# Each row = one video frame at ~40 fps effective (every 3rd frame from 120fps camera)
tt = sub['trial_time'].dt.total_seconds()
abs_t = sub['time'].dt.total_seconds() ##these two are not the same, should go with abs_t! 
print(f"Session duration (trial_time): {tt.min():.4f} s  →  {tt.max():.4f} s")
print(f"Session duration (absolute)  : {abs_t.min():.4f} s  →  {abs_t.max():.4f} s")
print(f"  = {(tt.max() - tt.min()) / 60:.2f} min, {len(sub):,} frames at {len(sub)/(tt.max()-tt.min()):.2f} fps effective")
print()

# Show the first 10 rows
disp_cols = ['dataset', 'trial_idx', 'frame', 'trial_time', 'time', _args.behavior_var]
disp_cols = [c for c in disp_cols if c in sub.columns]
print(f"First 10 rows (each row = one video frame):")
print(sub[disp_cols].head(10).to_string())

# %% [markdown]
# #### NEURAL DATA

# %%
# ═════════════════════════════════════════════════════════════════════════════
# DECODING CONFIGURATION  — set via command-line args (see --help)
# ═════════════════════════════════════════════════════════════════════════════
# Available behavioral variables (texture_pca_df_preproc.pkl columns):
#   PCA components : 'PC1' … 'PC50'
#   texture motion : 'texture_speed'  (deg/s, continuous)
#   pupil          : 'area'           (px², continuous)
#   gaze           : 'x_coord', 'y_coord'
#   speed          : 'speed'
BEHAVIOR_VAR  = _args.behavior_var
Z_SCORE_BEH   = _args.zscore_spike_features   # NEW: was hardcoded True; now follows --zscore_spike_features flag
                                               # scale doesn't affect R²/corr; set True if ADVI shows instability
T_TRIAL       = _args.t_trial
N_T_BINS      = _args.n_t_bins
LEARNING_RATE = _args.learning_rate
N_WORKERS     = _args.n_workers
# ═════════════════════════════════════════════════════════════════════════════

# --- Load SI recording — single source for fs, t_start, ch_locs, and waveform access ---
from spikeinterface import NumpyRecording
rec_si      = NumpyRecording.load(PREPROC_DIR)
ch_locs     = rec_si.get_channel_locations()              # (n_ch, 2): [x µm, depth µm]
ap_fs       = rec_si.get_sampling_frequency()                 # 30000.2835 Hz
ap_t_offset = float(rec_si._recording_segments[0].t_start)   # gate-relative start
n_ch        = rec_si.get_num_channels()
n_samp      = rec_si.get_num_samples()
ap_dtype    = rec_si.get_dtype()

# --- Behavioral subset for this session ---
beh     = texture_pca_df[(texture_pca_df['animal'] == animal) & (texture_pca_df['background'] == 'eink')].copy()
beh_t   = beh['time'].dt.total_seconds().values   # gate-relative seconds
beh_var = beh[BEHAVIOR_VAR].values                # chosen decoding variable
beh_spd = beh['speed'].values if 'speed' in beh.columns else None

# ── SPIKE FEATURES CACHE ──────────────────────────────────────────────────────
# NEW: skip full raw-trace recompute if spike_features npz already exists.
# Delete OUT_PATH to force recomputation (e.g. after changing LOC_FILE or DREDge).
OUT_PATH = os.path.join(RESULTS_DIR, f'spike_features_{SESSION}.npz')

if os.path.exists(OUT_PATH):
    print(f"\n[CACHE] Loading cached spike features -> {OUT_PATH}")
    data             = np.load(OUT_PATH, allow_pickle=False)
    spike_times_samp = data['times_samples']
    spike_channels   = data['channels']
    spike_features   = data['point_source_localizations']   # (N, 3): [x, z_corr, ptp]

    spike_t_gate = spike_times_samp / ap_fs + ap_t_offset
    spike_x      = spike_features[:, 0]
    z_corr       = spike_features[:, 1]
    spike_ptp    = spike_features[:, 2]
    spike_amp    = spike_ptp   # alias used by waveform QC plots below

    # Re-load motion for drift plots (needs temporal_bins / spatial_bins)
    motion        = si.Motion.load(MOTION_DIR)
    temporal_bins = motion.temporal_bins_s[0]
    spatial_bins  = motion.spatial_bins_um
    z_raw         = None   # not available from cache; drift scatter will be skipped

    print(f"Loaded {len(spike_t_gate):,} spikes from cache.")
else:
    print("\n[COMPUTE] Computing spike features from raw data...")

    # --- Custom peak detection: peaks_combined2 + localized_peaks_mt ---
    peaks = np.load(PEAKS_FILE)   # structured: sample_index, channel_index, amplitude, segment_index
    # peaks_combined2_sorted.npy is already sorted by sample_index (same ordering used during localization)
    loc   = np.load(LOC_FILE)     # structured: x (lateral µm), y (depth µm), z (probe-normal µm), alpha

    # peaks['amplitude'] is negative (threshold crossings) — take abs for downstream use
    spike_times_samp = peaks['sample_index']    # int64: sample index in file
    spike_amp        = np.abs(peaks['amplitude']).astype(np.float32)  # µV magnitude
    spike_x          = loc['x'].astype(np.float32)   # lateral µm
    spike_z          = loc['y'].astype(np.float32)   # depth along probe µm  (field is 'y' in localized file)

    # Convert to gate-relative seconds — same clock as texture_pca_df['time']
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

    # ── RAW PTP AMPLITUDE: detected directly from waveform on detection channel ────
    # Uses the same ±0.5 ms window as SI's localize_peaks (ms_before=0.5, ms_after=0.5).
    # Recording is processed in 60-second chunks; spikes are pre-sorted by sample_index
    # so searchsorted correctly partitions them per chunk.

    N_BEFORE_PTP  = round(0.5 * ap_fs / 1000)   # 15 samples
    N_AFTER_PTP   = round(0.5 * ap_fs / 1000)   # 15 samples
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
        # include ALL spikes in [chunk_start, chunk_end) — no boundary exclusion
        lo = int(np.searchsorted(spike_times_samp, chunk_start))
        hi = int(np.searchsorted(spike_times_samp, chunk_end))
        if lo < hi:
            # read traces padded by ±N so every chunk-boundary spike gets its full window
            tr_start = max(0, chunk_start - N_BEFORE_PTP)
            tr_end   = min(n_samp, chunk_end + N_AFTER_PTP)
            traces   = rec_si.get_traces(start_frame=tr_start, end_frame=tr_end)  # (padded_samp, n_ch)
            idxs  = np.arange(lo, hi)
            s_rel = spike_times_samp[idxs] - tr_start   # spike positions relative to padded trace
            ch_loc = peaks['channel_index'][idxs].astype(int)
            # spikes that have their full ±window inside the loaded trace (the vast majority)
            full_mask = (s_rel >= N_BEFORE_PTP) & (s_rel + N_AFTER_PTP <= len(traces))
            if full_mask.any():
                t_off  = s_rel[full_mask][:, None] + np.arange(-N_BEFORE_PTP, N_AFTER_PTP)[None, :]
                ch_rep = np.broadcast_to(ch_loc[full_mask][:, None], t_off.shape)
                wfs    = traces[t_off, ch_rep].astype(np.float32)
                spike_ptp[idxs[full_mask]] = np.ptp(wfs, axis=1)
            # spikes within N samples of the absolute recording start/end — clip window
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
    print(f"  NaN remaining (spikes at absolute recording boundary with empty clip window): {n_nan_ptp} ({n_nan_ptp/n_spikes*100:.2f}%)")

    # ── DREDGE DRIFT CORRECTION ────────────────────────────────────────────────────
    # Corrected depth: z_corr = z_raw − displacement(spike_t_gate, z_raw)
    # Uses si.Motion.get_displacement_at_time_and_depth (bilinear interpolation).

    motion = si.Motion.load(MOTION_DIR)
    temporal_bins = motion.temporal_bins_s[0]
    spatial_bins  = motion.spatial_bins_um
    print(f"Motion loaded from  : {MOTION_DIR}")
    print(f"Temporal range      : {temporal_bins[0]:.2f} – {temporal_bins[-1]:.2f} s  (gate-relative)")
    print(f"Spatial range       : {spatial_bins[0]:.0f} – {spatial_bins[-1]:.0f} µm")

    z_raw = loc['y'].astype(np.float32)   # raw depth along probe (µm) — triangulated

    # spike_t_gate is already in absolute (gate-relative) seconds — matches Motion's time axis
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
        loc['x'].astype(np.float32),    # lateral position (µm) — triangulated
        z_corr,                          # depth along probe (µm) — triangulated + DREDge
        spike_ptp,                       # PTP amplitude (µV) — raw waveform ±0.5 ms window
    ])  # shape (N, 3)

    np.savez(
        OUT_PATH,
        # ── canonical density_decoding names ────────
        times_samples              = peaks['sample_index'].astype(np.int64),
        channels                   = spike_channels,
        point_source_localizations = spike_features,    # (N, 3): [x, z_corr, ptp]
        t_start_gate_s             = np.float64(ap_t_offset),
        sampling_rate              = np.float64(ap_fs),
    )
    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"Saved → {OUT_PATH}  ({size_mb:.0f} MB)")
    print(f"  spike_features shape: {spike_features.shape}  dtype: {spike_features.dtype}")

# %% [markdown]
# #### ALIGNMENT CHECK

# %%
# NEW: plot-guard — only generate if file does not already exist
_align_save = os.path.join(RESULTS_DIR, f'alignment_check_{SESSION}.png')
if not os.path.exists(_align_save):
    print("[PLOT] Generating alignment check ...")
    # ── 1. Bin MUA into 1-second windows (gate-relative time) ──────────────────
    BIN_ALIGN  = 1.0   # seconds
    t_lo = min(spike_t_gate.min(), beh_t.min()) - 2
    t_hi = max(spike_t_gate.max(), beh_t.max()) + 2
    bins_1s  = np.arange(t_lo, t_hi + BIN_ALIGN, BIN_ALIGN)
    t_ctr    = (bins_1s[:-1] + bins_1s[1:]) / 2

    mua_counts, _ = np.histogram(spike_t_gate, bins=bins_1s)
    mua_hz = uniform_filter1d(mua_counts.astype(float) / BIN_ALIGN, size=5)  # 5 s smoothing

    # ── 2. Bin behavioral signals into same windows ─────────────────────────────
    def bin_signal(t_arr, sig, bins):
        out = np.full(len(bins) - 1, np.nan)
        for i in range(len(bins) - 1):
            m = (t_arr >= bins[i]) & (t_arr < bins[i + 1])
            if m.any():
                out[i] = np.nanmean(sig[m])
        return out

    beh_var_1s = bin_signal(beh_t, beh_var, bins_1s)
    spd_1s = bin_signal(beh_t, beh_spd, bins_1s) if beh_spd is not None else None

    # ── 3. Build normalised overlay ─────────────────────────────────────────────
    def norm01(x):
        mn, mx = np.nanmin(x), np.nanmax(x)
        return (x - mn) / (mx - mn + 1e-9)

    # ── 4. Plot ──────────────────────────────────────────────────────────────────
    n_rows = 2 + (spd_1s is not None)
    fig, axes = plt.subplots(n_rows + 1, 1, figsize=(22, 4 * (n_rows + 1)), sharex=True)
    fig.suptitle(
        f"{SESSION}  ·  Alignment check  ·  Common clock: SpikeGLX gate-open",
        fontsize=11, y=1.002
    )

    # Row 0: MUA firing rate
    axes[0].fill_between(t_ctr, mua_hz, alpha=0.7, color='steelblue')
    axes[0].set_ylabel("MUA\n(sp / s)", fontsize=9)
    axes[0].set_title("All KS4 spikes — 1 s bins, 5 s smoothed", fontsize=8, loc='left', pad=2)

    # Row 1: chosen behavioral variable
    axes[1].plot(t_ctr, beh_var_1s, color='darkorange', lw=0.8)
    axes[1].set_ylabel(f"{BEHAVIOR_VAR}\n(1 s mean)", fontsize=9)

    row = 2
    if spd_1s is not None:
        axes[row].plot(t_ctr, spd_1s, color='forestgreen', lw=0.8)
        axes[row].set_ylabel("Speed\n(1 s mean)", fontsize=9)
        row += 1

    # Last row: normalised overlay to check temporal match
    ax_ov = axes[row]
    ax_ov.plot(t_ctr, norm01(mua_hz),     color='steelblue',   lw=1.0, alpha=0.85, label='MUA (norm)')
    ax_ov.plot(t_ctr, norm01(beh_var_1s), color='darkorange',  lw=0.8, alpha=0.85, label=f'{BEHAVIOR_VAR} (norm)')
    if spd_1s is not None:
        ax_ov.plot(t_ctr, norm01(spd_1s), color='forestgreen', lw=0.8, alpha=0.85, label='Speed (norm)')
    ax_ov.set_ylabel("Normalised\n[0 – 1]", fontsize=9)
    ax_ov.legend(fontsize=8, loc='upper right')
    ax_ov.set_title("Overlay — use this to judge temporal alignment", fontsize=8, loc='left', pad=2)

    # Shade windows on every axis
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

    # ── 5. Summary ───────────────────────────────────────────────────────────────
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
    # ── Channel positions already loaded in NEURAL DATA section ──────────────────────
    probe_x_min  = ch_locs[:, 0].min()
    probe_x_max  = ch_locs[:, 0].max()
    print(f"Channels: {n_ch}   "
          f"probe x: {probe_x_min:.1f} – {probe_x_max:.1f} µm   "
          f"depth: {ch_locs[:,1].min():.0f} – {ch_locs[:,1].max():.0f} µm")

    # ── Spike selection ────────────────────────────────────────────────────────────
    T_START = 0
    T_WIN = 5
    mask    = (spike_t_gate >= T_START + ap_t_offset) & (spike_t_gate < T_START + T_WIN + ap_t_offset)
    sx, sz, sa = spike_x[mask], z_corr[mask], spike_amp[mask]
    outside = (sx < probe_x_min) | (sx > probe_x_max)
    print(f"Spikes in {T_WIN*1e3:.0f} ms window : {mask.sum()}")
    print(f"  outside probe x range           : {outside.sum()}  ({outside.mean()*100:.0f}%)")

    # ── Plot — single panel ────────────────────────────────────────────────────────
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
                          vmin=0, vmax=amp_vmax,
                          edgecolors='k', linewidths=0.2, zorder=5,
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

# ── load back & prep for density decoding ─────────────────────────────────────
data = np.load(OUT_PATH, allow_pickle=False)

spike_times    = data['times_samples']              # (N,) int64 — sample indices
spike_channels = data['channels']                   # (N,) int32 — detection channel
spike_features = data['point_source_localizations'] # (N, 3) float32 — [x, z_corr, ptp]

print("Loaded spike features:")
print(f"  spike_times    shape {spike_times.shape},    dtype {spike_times.dtype}")
print(f"  spike_channels shape {spike_channels.shape}, dtype {spike_channels.dtype}")
print(f"  spike_features shape {spike_features.shape}, dtype {spike_features.dtype}")
print(f"  columns: x (µm, triangulated) | z (µm, triangulated + DREDge-corrected) | PTP amplitude (µV)")
print(f"  raw ranges: x=[{spike_features[:,0].min():.1f}, {spike_features[:,0].max():.1f}]  "
      f"z=[{spike_features[:,1].min():.1f}, {spike_features[:,1].max():.1f}]  "
      f"ptp=[{spike_features[:,2].min():.2f}, {spike_features[:,2].max():.2f}]")
print()

# ── scatter: raw localisations coloured by spike_ptp (plot-guarded) ───────────
p999 = float(np.nanpercentile(spike_ptp, 99.9))

_loc_save = os.path.join(RESULTS_DIR, f'spike_localisations_{SESSION}.png')
if not os.path.exists(_loc_save):
    print("[PLOT] Generating localisation scatter ...")
    rng_sf = np.random.default_rng(0)
    n_show = min(200_000, len(spike_features))
    isf    = rng_sf.choice(len(spike_features), n_show, replace=False)
    isf    = isf[np.argsort(spike_ptp[isf])]    # plot dim spikes first

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
# ── WAVEFORM GRID: 100 random spikes ordered by descending PTP ────────────────
# Reads waveforms for 100 random spikes; orders panels highest→lowest PTP.
# Blue arrows = max and min samples (the two points defining PTP).
# Orange dashed lines = ±ptp/2.
# Only runs when raw peaks were loaded (i.e. not from cache).

_wf_val_save = os.path.join(RESULTS_DIR, f'ptp_waveforms_{SESSION}.png')
if not os.path.exists(_wf_val_save) and 'peaks' in dir():
    print("[PLOT] Generating waveform grid ...")
    N_BEFORE_PTP_GRID = round(0.5 * ap_fs / 1000)
    N_AFTER_PTP_GRID  = round(0.5 * ap_fs / 1000)
    wf_len_val        = N_BEFORE_PTP_GRID + N_AFTER_PTP_GRID

    rng_val   = np.random.default_rng(42)
    val_idx   = rng_val.choice(len(peaks), 100, replace=False)
    wf_val    = np.full((100, wf_len_val), np.nan, dtype=np.float32)
    ptp_val   = spike_ptp[val_idx].copy()   # already computed for all spikes

    for k, idx in enumerate(val_idx):
        pk_ch = int(peaks['channel_index'][idx])
        samp  = int(spike_times_samp[idx])
        t0 = max(0, samp - N_BEFORE_PTP_GRID)
        t1 = min(n_samp,  samp + N_AFTER_PTP_GRID)
        tr = rec_si.get_traces(start_frame=t0, end_frame=t1)[:, pk_ch].astype(np.float32)
        if tr.shape[0] < wf_len_val:
            tr = np.pad(tr, (0, wf_len_val - tr.shape[0]))
        wf_val[k] = tr

    order_val = np.argsort(ptp_val)[::-1]   # descending PTP
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

        i_max = int(np.argmax(wf))
        i_min = int(np.argmin(wf))
        t_max, v_max = t_ax_val[i_max], float(wf[i_max])
        t_min, v_min = t_ax_val[i_min], float(wf[i_min])

        arrow_offset = y_half * 0.28
        ax2.annotate('', xy=(t_max, v_max),
                     xytext=(t_max, v_max + arrow_offset),
                     arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.0))
        ax2.annotate('', xy=(t_min, v_min),
                     xytext=(t_min, v_min - arrow_offset),
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


# ── SCATTER: SPIKE DEPTH BEFORE vs AFTER DRIFT CORRECTION (plot-guarded) ──────
# Only runs when raw peaks were loaded (z_raw not None means we came from compute path).
_scatter_save = os.path.join(RESULTS_DIR, f'drift_scatter_before_after_{SESSION}.png')
if not os.path.exists(_scatter_save) and 'z_raw' in dir() and z_raw is not None:
    print("[PLOT] Generating drift scatter before/after ...")
    # 4-panel grid: rows = first 12 min / last 12 min
    #               cols = raw depth (before) / DREDge-corrected depth (after)
    T_VIZ_S = 12 * 60    # 12 min in seconds
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
    fig.suptitle(
        f'{SESSION} — Spike depth vs time (grayscale)\n'
        'Left: raw localisation  |  Right: after DREDge correction\n'
        'Coloured lines on left panels = displacement trajectory per spatial bin '
        f'({len(spatial_bins)} bins, plasma = shallow→deep)',
        fontsize=10,
    )

    rng_sc   = np.random.default_rng(7)
    disp_cmap = plt.cm.plasma
    disp_norm = plt.Normalize(spatial_bins[0], spatial_bins[-1])

    for row_i, (win_label, t0, t1) in enumerate(windows):
        sel = np.where((spike_t_gate >= t0) & (spike_t_gate < t1))[0]
        if len(sel) > MAX_PTS:
            sel = rng_sc.choice(sel, MAX_PTS, replace=False)
            sel = sel[np.argsort(spike_t_gate[sel])]

        t_sel = spike_t_gate[sel]

        # Displacement trace mask for this time window
        t_disp = (temporal_bins >= t0) & (temporal_bins <= t1)
        t_line = temporal_bins[t_disp]

        for col_i, (z_arr, panel_title) in enumerate([
            (z_raw[sel],  'Before correction'),
            (z_corr[sel], 'After DREDge correction'),
        ]):
            ax = axes[row_i, col_i]

            ax.scatter(t_sel, z_arr, c='gray', s=0.3, alpha=0.25,
                       rasterized=True, linewidths=0)

            if col_i == 0:
                for k, z_bin in enumerate(spatial_bins):
                    apparent = z_bin + motion.displacement[0][t_disp, k]
                    ax.plot(t_line, apparent,
                            color=disp_cmap(disp_norm(z_bin)),
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


# ── WAVEFORM QC (post-DREDge) — probe-layout view (plot-guarded) ──────────────
_wf_save = os.path.join(RESULTS_DIR, f'waveform_qc_{SESSION}.png')
if not os.path.exists(_wf_save) and 'peaks' in dir():
    print("[PLOT] Generating waveform QC probe-layout ...")
    # Parameters
    T_POOL_START = 100.0    # pool spikes from this time range (s, gate-relative)
    T_POOL_END   = 110.0
    N_SHOW       = 5        # number of spikes to show
    N_BEFORE     = 20       # samples before trough
    N_AFTER      = 62       # samples after trough  (~2.7 ms @ 30 kHz)
    N_NBR        = 32       # nearest channels for waveform traces

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

        all_wfs = []
        all_nbr = []
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
            all_wfs.append(tr)
            all_nbr.append(nbr)

        all_ptp    = np.concatenate([np.ptp(w, axis=0) for w in all_wfs])
        global_ptp = float(np.percentile(all_ptp, 99))
        AMP_SCALE  = (0.7 * elec_spacing) / max(global_ptp, 1.0)

        print(f"Random spikes: {spike_amp[sel].min():.0f}–{spike_amp[sel].max():.0f} µV")
        print(f"Electrode spacing {elec_spacing:.0f} µm  |  column gap {col_gap:.0f} µm  |  "
              f"99th-pct PTP {global_ptp:.1f} µV  →  AMP_SCALE {AMP_SCALE:.4f} µm/µV")

        ncols = N_SHOW
        fig, axes = plt.subplots(1, ncols, figsize=(ncols * 5, 12), sharey=True, sharex=True)
        if ncols == 1:
            axes = [axes]

        fig.suptitle(
            f"Waveform QC (post-DREDge) — {SESSION}  ·  probe-layout view\n"
            "blue = waveform  |  ◆ orange = peak ch  |  ★ red + crosshairs = triangulated location\n"
            f"AMP_SCALE = {AMP_SCALE*1000:.2f} µm/mV  (99th-pct PTP ≈ {0.7*elec_spacing:.0f} µm)",
            fontsize=10,
        )

        for pi, (_spi, wf, nbr) in enumerate(zip(sel, all_wfs, all_nbr)):
            ax     = axes[pi]
            pk_ch  = int(peaks['channel_index'][_spi])
            px     = float(spike_x[_spi])
            pz     = float(z_corr[_spi])    # DREDge-corrected depth

            ax.scatter(ch_locs[:, 0], ch_locs[:, 1],
                       s=8, c='lightgray', edgecolors='none', zorder=1, rasterized=True)

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

            ax.set_title(
                f"spike {pi+1}  det_ch={pk_ch}\n"
                f"{spike_amp[_spi]:.0f} µV  x={px:.0f}  z={pz:.0f} µm (corr)",
                fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.18)

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

# ── BEHAVIOURAL DATA ──────────────────────────────────────────────────────────
# ── 0. Frame-interval diagnostic ─────────────────────────────────────────────
_dt = np.diff(beh_t)                          # inter-frame intervals (s)
_dt_median = float(np.median(_dt))
_dt_tol    = 0.01                # 5 % tolerance
_irregular = np.where(np.abs(_dt - _dt_median) > _dt_tol)[0]  # indices into _dt

print(f"Frame interval  : median={_dt_median*1000:.3f} ms  "
      f"min={_dt.min()*1000:.3f} ms  max={_dt.max()*1000:.3f} ms")
print(f"Irregular gaps  : {len(_irregular)} / {len(_dt)} intervals  "
      f"(>{_dt_tol*1000:.2f} ms from median)")
if len(_irregular):
    _gap_dt  = _dt[_irregular]
    _gap_t   = beh_t[_irregular]
    _gap_df  = pd.DataFrame({
        'frame_before': _irregular,
        't_before_s':   _gap_t,
        'interval_ms':  _gap_dt * 1000,
        'delta_from_median_ms': (_gap_dt - _dt_median) * 1000,
    }).sort_values('interval_ms', ascending=False)
    print(_gap_df.to_string(index=False))

# ── 1. Extract the chosen behavioral variable ─────────────────────────────────
beh_raw = beh[BEHAVIOR_VAR].values.astype(np.float32)   # (n_frames,)

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

# ── 2. Overlapping neural–behavioral window ───────────────────────────────────
T_OV_START = max(beh_t.min(), ap_t_offset)
T_OV_END   = min(beh_t.max(), spike_t_gate.max())
print(f"\nOverlap window: {T_OV_START:.2f} – {T_OV_END:.2f} s "
      f"({(T_OV_END - T_OV_START)/60:.2f} min)")

# ── 3. Chunk into windows ─────────────────────────────────────────────────────
trial_starts = np.arange(T_OV_START, T_OV_END - T_TRIAL, T_TRIAL)
trial_ends   = trial_starts + T_TRIAL
n_trials_beh = len(trial_starts)

CHUNK_STYLE     = _args.chunk_style
CHUNK_SIZE      = _args.chunk_size
TEST_PERCENTAGE = _args.test_percentage
TEST_FRAC       = TEST_PERCENTAGE / 100.0
TRAIN_FRAC      = 1.0 - TEST_FRAC
print(f"\nWindow length : {T_TRIAL} s  x  {N_T_BINS} bins = {T_TRIAL/N_T_BINS*1000:.0f} ms/bin")
print(f"Number of windows : {n_trials_beh}  (~{n_trials_beh*TRAIN_FRAC:.0f} train, ~{n_trials_beh*TEST_FRAC:.0f} test)")

# ── 4. Save params JSON (before any heavy compute) ────────────────────────────
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
        "N_T_BINS":      int(N_T_BINS),
        "ms_per_bin":    float(T_TRIAL / N_T_BINS * 1000),
        "LEARNING_RATE": float(LEARNING_RATE),
        "N_WORKERS":     int(N_WORKERS),
        "chunk_style":   CHUNK_STYLE,
        "test_percentage": float(TEST_PERCENTAGE),
        "train_frac":    float(TRAIN_FRAC),
        "test_frac":     float(TEST_FRAC),
        "embargo_trials": int(_args.embargo_trials),  # NEW
        "animal_speed_gate": float(_args.animal_speed_gate), ###CHANGED (use arg directly; ANIMAL_SPEED_GATE not yet defined here)
        "zscore_spike_features": ZSCORE_SPIKE_FEATURES,
        "spike_feature_zscore_params": _zscore_params_sf,
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
        "method":               "isosplit",   # all spikes used; no subsampling cap
        "K_init":               20,
        "min_cluster_size":     10,
        "whiten_cluster_pairs": 1,
        "refine_clusters":      1,
    },
}
with open(_params_json, 'w') as _f:
    json.dump(_params, _f, indent=2)
print(f"\n[PARAMS] Saved -> {_params_json}")

# ── 5. Quick overview plot (plot-guarded) ─────────────────────────────────────
_beh_plot_save = os.path.join(RESULTS_DIR, f'behaviour_overview_{_VAR_KEY}_{SESSION}.png')
if not os.path.exists(_beh_plot_save):
    print(f"[PLOT] Generating behaviour overview -> {_beh_plot_save}")
    fig, ax = plt.subplots(figsize=(18, 2.5))
    ax.plot(beh_t, beh_z, lw=0.4, color='tab:blue', alpha=0.7,
            label=f"{BEHAVIOR_VAR}{'  (z-scored)' if Z_SCORE_BEH else ''}")
    ax.axvspan(T_OV_START, T_OV_END, color='violet', alpha=0.08, label='Neural-behavioral overlap')
    ax.set_xlabel("Time from SpikeGLX gate open (s)", fontsize=9)
    ax.set_ylabel(BEHAVIOR_VAR, fontsize=9)
    ax.set_title(f"Behaviour: {BEHAVIOR_VAR} ({animal} eink)", fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(_beh_plot_save, dpi=130, bbox_inches='tight')
    print(f"Saved -> {_beh_plot_save}")
    plt.close()

# %%

# ── PROCESS SPIKE FEATURES & BEHAVIOURS ──────────────────────────────────────
sys.path.insert(0, '/u/gongt/density_decoding')

from density_decoding.utils.data_utils import BaseDataLoader

# Instantiate loader (always needed — cheap)
data_loader = BaseDataLoader(
    t_before=0.0,
    t_after=T_TRIAL,
    n_t_bins=N_T_BINS,
)

# ── CHECKPOINT: shared spike bins (reused across all BEHAVIOR_VAR runs) ───────
SPIKE_CKPT = os.path.join(RESULTS_DIR, f'ckpt_spikebins_{SESSION}.pkl')

# ── 1. Bin behavior for the chosen variable (cheap, always done) ──────────────
print(f"Processing behaviour ({BEHAVIOR_VAR}) ...")
bin_beh_all = data_loader.process_behaviors(
    time_points       = beh_t,
    raw_behaviors     = beh_z,
    trial_start_times = trial_starts,
    trial_end_times   = trial_ends,
)   # (n_trials_beh, N_T_BINS)

# Windows with any NaN bin -> no behavioral coverage for this variable -> drop
valid_mask_beh  = ~np.isnan(bin_beh_all).any(axis=1)
valid_trials_beh = np.where(valid_mask_beh)[0]
print(f"Valid windows (no NaN in {BEHAVIOR_VAR}): {valid_mask_beh.sum()} / {n_trials_beh}")

# ── Animal speed gate (ported from rrr_texture_neural.py) ────────────────────  ###CHANGED
# Drop trial windows where the animal is moving (mean speed over window > threshold).  ###CHANGED
# Uses the 'speed' column already present in texture_pca_df.                     ###CHANGED
ANIMAL_SPEED_GATE = _args.animal_speed_gate                                     ###CHANGED
_speed_gate_info  = dict(used=False)                                            ###CHANGED
if ANIMAL_SPEED_GATE > 0:                                                       ###CHANGED
    if beh_spd is None:                                                         ###CHANGED
        print("[SPEED GATE] WARNING: 'speed' column not found in df — gate DISABLED.") ###CHANGED
        ANIMAL_SPEED_GATE = 0.0                                                 ###CHANGED
    else:                                                                       ###CHANGED
        # Per-window mean animal speed (avg beh_spd over each trial window)     ###CHANGED
        _win_spd = np.array([                                                   ###CHANGED
            np.nanmean(beh_spd[                                                 ###CHANGED
                (beh_t >= trial_starts[k]) & (beh_t < trial_ends[k])           ###CHANGED
            ]) for k in range(n_trials_beh)                                     ###CHANGED
        ])                                                                      ###CHANGED
        _speed_gate_mask = _win_spd < ANIMAL_SPEED_GATE                        ###CHANGED
        # Combine with existing NaN-based validity mask                         ###CHANGED
        valid_mask_beh   = valid_mask_beh & _speed_gate_mask                   ###CHANGED
        valid_trials_beh = np.where(valid_mask_beh)[0]                         ###CHANGED
        _n_dropped_spd   = int((~_speed_gate_mask).sum())                      ###CHANGED
        _speed_gate_info = dict(                                                ###CHANGED
            used=True, threshold=float(ANIMAL_SPEED_GATE),                     ###CHANGED
            n_total=int(n_trials_beh), n_kept=int(valid_mask_beh.sum()),       ###CHANGED
            n_dropped_speed=_n_dropped_spd,                                    ###CHANGED
            pct_stationary=100.0 * valid_mask_beh.sum() / max(n_trials_beh, 1) ###CHANGED
        )                                                                       ###CHANGED
        print(f"[SPEED GATE] threshold={ANIMAL_SPEED_GATE:.4g} → "             ###CHANGED
              f"{valid_mask_beh.sum()} / {n_trials_beh} windows kept "         ###CHANGED
              f"({_speed_gate_info['pct_stationary']:.1f}% stationary); "      ###CHANGED
              f"{_n_dropped_spd} dropped for animal motion")                   ###CHANGED
print(f"Valid windows after all filters: {valid_mask_beh.sum()} / {n_trials_beh}")  ###CHANGED

# ── 2. Spike bins (cached, independent of variable choice) ────────────────────
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

# ── 3. Filter spike bins to behavior-valid windows ────────────────────────────
bin_beh_v = bin_beh_all[valid_mask_beh]              # (n_valid, N_T_BINS)
bin_sf_v  = [bin_spike_features_all[k] for k in valid_trials_beh]
bin_ti_v, bin_tmi_v = [], []
for new_k, k in enumerate(valid_trials_beh):
    for t in range(N_T_BINS):
        old = bin_trial_idxs_all[k * N_T_BINS + t]   # array of value k
        bin_ti_v.append(np.full_like(old, new_k))
        bin_tmi_v.append(bin_time_idxs_all[k * N_T_BINS + t])

n_valid  = len(valid_trials_beh)

if TEST_PERCENTAGE <= 0 or TEST_PERCENTAGE >= 100:
    raise ValueError(f"test_percentage must be in range (0, 100), got {TEST_PERCENTAGE}")
if CHUNK_STYLE == 'interleaved' and n_valid >= CHUNK_SIZE:
    # Divide valid trials into blocks of CHUNK_SIZE.
    # Within each block: first (100-test_percentage)% → train, last test_percentage% → test.
    # This distributes test windows evenly across the full recording.
    _n_test_per_chunk  = max(1, int(round(CHUNK_SIZE * TEST_FRAC)))
    _n_train_per_chunk = CHUNK_SIZE - _n_test_per_chunk
    train_idx = np.concatenate([
        np.arange(c * CHUNK_SIZE,
                  c * CHUNK_SIZE + _n_train_per_chunk)
        for c in range(n_valid // CHUNK_SIZE)
        if c * CHUNK_SIZE + _n_train_per_chunk <= n_valid
    ])
    test_idx = np.concatenate([
        np.arange(c * CHUNK_SIZE + _n_train_per_chunk,
                  min((c + 1) * CHUNK_SIZE, n_valid))
        for c in range(n_valid // CHUNK_SIZE)
    ])
    _n_chunks = n_valid // CHUNK_SIZE
    print(f"\nTrain/test split (style='{CHUNK_STYLE}', test={TEST_PERCENTAGE:.0f}%, {_n_chunks} chunks x {CHUNK_SIZE} trials, "
          f"{_n_train_per_chunk} train + {_n_test_per_chunk} test per chunk)")
    print(f"  train: {len(train_idx)} windows ({TRAIN_FRAC*100:.0f}%)  |  test: {len(test_idx)} windows ({TEST_PERCENTAGE:.0f}%)")
else:                                                           # contiguous split
    _n_train = int(n_valid * TRAIN_FRAC)
    train_idx = np.arange(_n_train)
    test_idx  = np.arange(_n_train, n_valid)
    print(f"\nTrain/test split (style='{CHUNK_STYLE}', test={TEST_PERCENTAGE:.0f}%)")
    print(f"  train: {len(train_idx)} windows ({TRAIN_FRAC*100:.0f}%)  |  test: {len(test_idx)} windows ({TEST_PERCENTAGE:.0f}%)")
    if n_valid > 0:
        print(f"  train covers {trial_starts[valid_trials_beh[0]]:.1f} – "
              f"{trial_ends[valid_trials_beh[_n_train-1]]:.1f} s")
        print(f"  test  covers {trial_starts[valid_trials_beh[_n_train]]:.1f} – "
              f"{trial_ends[valid_trials_beh[-1]]:.1f} s")
print(f"\nbin_beh_v shape : {bin_beh_v.shape}")

# ── Also save window timing for reference ─────────────────────────────────────
beh_save = os.path.join(RESULTS_DIR, f'beh_windows_{_VAR_KEY}_{SESSION}.npz')
np.savez_compressed(
    beh_save,
    bin_beh         = bin_beh_v,
    trial_starts    = trial_starts[valid_trials_beh],
    trial_ends      = trial_ends[valid_trials_beh],
    train_idx       = train_idx,
    test_idx        = test_idx,
    T_TRIAL         = np.float32(T_TRIAL),
    N_T_BINS        = np.int32(N_T_BINS),
)
print(f"Saved -> {beh_save}")


# %%

# ── TRIAL INSPECTION PLOT (plot-guarded) ──────────────────────────────────────
# 3 rows, each row = one randomly selected set of ~20 consecutive valid trials
# (20 × T_TRIAL = 30 s).  Within each row spike count (blue bars, left y-axis)
# is overlaid with the raw BEHAVIOR_VAR trace + binned BEHAVIOR_VAR mean (right y-axis).
#   thin silver verticals  = bin boundaries (50 ms)
#   thick dashed black     = trial boundaries (1.5 s)
#   alternating lavender / mint shading  = individual trials
#   top x-axis             = trial index labels
_insp_path = os.path.join(RESULTS_DIR, f'trial_inspection_{_VAR_KEY}_{SESSION}.png')
if not os.path.exists(_insp_path):
    print("[PLOT] Generating trial inspection ...")
    _bin_size_s       = T_TRIAL / N_T_BINS
    _n_trials_per_set = int(round(30.0 / T_TRIAL))   # 30 s / 1.5 s = 20
    _n_sets           = 3

    # Pick 3 non-overlapping start positions (in valid_trials_beh index space)
    _rng_inspect  = np.random.default_rng(seed=42)
    _n_valid      = len(valid_trials_beh)
    assert _n_valid >= _n_sets * _n_trials_per_set, \
        f"Not enough valid trials ({_n_valid}) for {_n_sets} sets of {_n_trials_per_set}"
    _pool = list(range(_n_valid - _n_trials_per_set + 1))
    _rng_inspect.shuffle(_pool)
    _set_starts = []
    for _s in _pool:
        if all(abs(_s - _s2) >= _n_trials_per_set for _s2 in _set_starts):
            _set_starts.append(_s)
        if len(_set_starts) == _n_sets:
            break
    _set_starts.sort()   # chronological order

    _shade_colors = ['#e8eaf6', '#e8f5e9']   # alternating lavender / mint per trial

    fig_insp, axes_insp = plt.subplots(
        _n_sets, 1,
        figsize=(22, 3.5 * _n_sets),
    )
    if _n_sets == 1:
        axes_insp = [axes_insp]

    for _row, _start in enumerate(_set_starts):
        _vis = valid_trials_beh[_start : _start + _n_trials_per_set]

        _t_row_start = trial_starts[_vis[0]]
        _t_row_end   = trial_ends[_vis[-1]]

        _all_bin_edges  = []
        _all_bin_ctrs   = []
        _all_scounts    = []
        _all_binned_beh = []

        for _ti, _k in enumerate(_vis):
            _t0 = trial_starts[_k]
            _t1 = trial_ends[_k]
            _be = _t0 + np.arange(N_T_BINS + 1) * _bin_size_s
            _all_bin_edges.append(_be)
            _all_bin_ctrs.append(0.5 * (_be[:-1] + _be[1:]))

            _smask = (spike_t_gate >= _t0) & (spike_t_gate < _t1)
            _sc, _ = np.histogram(spike_t_gate[_smask], bins=_be)
            _all_scounts.append(_sc)
            _all_binned_beh.append(bin_beh_v[_start + _ti])

        _bin_ctrs_all  = np.concatenate(_all_bin_ctrs)
        _scounts_all   = np.concatenate(_all_scounts)

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
        ax_l.tick_params(axis='x', labelsize=7)
        ax_l.set_xlim(_t_row_start - _bin_size_s * 0.1,
                      _t_row_end   + _bin_size_s * 0.1)

        ax_r.plot(beh_t[_bmask], beh_z[_bmask],
                  color='darkgray', lw=0.7, alpha=0.9, zorder=4, label=f'{BEHAVIOR_VAR} raw')

        for _ti, _be in enumerate(_all_bin_edges):
            ax_r.stairs(_all_binned_beh[_ti], _be,
                        color='tab:orange', lw=1.8, zorder=5, baseline=None,
                        label=f'{BEHAVIOR_VAR} binned' if _ti == 0 else None)
        ax_r.set_ylabel(f'{BEHAVIOR_VAR} (z)', fontsize=8, color='tab:orange')
        ax_r.tick_params(axis='y', labelcolor='tab:orange', labelsize=7)

        ax_top = ax_l.twiny()
        ax_top.set_xlim(ax_l.get_xlim())
        _trial_ctrs = [(trial_starts[_k] + trial_ends[_k]) / 2 for _k in _vis]
        ax_top.set_xticks(_trial_ctrs)
        ax_top.set_xticklabels(
            [str(valid_trials_beh[_start + _ti]) for _ti in range(_n_trials_per_set)],
            fontsize=5, rotation=90
        )
        ax_top.set_xlabel('trial index (in trial_starts)', fontsize=7)

        _h1, _l1 = ax_l.get_legend_handles_labels()
        _h2, _l2 = ax_r.get_legend_handles_labels()
        ax_l.legend(_h1 + _h2, _l1 + _l2, fontsize=7, loc='upper right', framealpha=0.7)

        ax_l.set_title(
            f"Set {_row + 1}  |  valid-trial indices {_start}–{_start + _n_trials_per_set - 1}  "
            f"|  t = {_t_row_start:.1f} – {_t_row_end:.1f} s  "
            f"|  {_n_trials_per_set} trials × {T_TRIAL*1e3:.0f} ms  "
            f"(bin = {_bin_size_s*1e3:.0f} ms)",
            fontsize=8, loc='left'
        )

        if _row == _n_sets - 1:
            ax_l.set_xlabel('Time from SpikeGLX gate open (s)', fontsize=8)

    fig_insp.suptitle(
        f"Trial inspection — {SESSION} — {BEHAVIOR_VAR} vs spike count  "
        f"(3 sets of {_n_trials_per_set} consecutive valid windows ≈ 30 s each)",
        fontsize=9
    )
    plt.tight_layout()
    fig_insp.savefig(_insp_path, dpi=130, bbox_inches='tight')
    print(f"Saved → {_insp_path}")
    plt.close(fig_insp)


# %%

# ── DENSITY DECODING ──────────────────────────────────────────────────────────
import torch
from density_decoding.decode_pipeline import decode_pipeline
from density_decoding.decoders.behavior_decoder import sliding_window_decoder

print("=" * 60)
print("GPU / DEVICE CHECK")
print("=" * 60)
print(f"  torch version      : {torch.__version__}")
print(f"  CUDA built with    : {torch.version.cuda}")
print(f"  CUDA available     : {torch.cuda.is_available()}")
print(f"  GPU count          : {torch.cuda.device_count()}")
for _i in range(torch.cuda.device_count()):
    print(f"  GPU {_i}              : {torch.cuda.get_device_name(_i)}")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"  -> using device     : {device}")
print("=" * 60)

print("=" * 60)
print(f"DECODE  {BEHAVIOR_VAR}{'  (z-scored)' if Z_SCORE_BEH else ''}  —  temporal train/test split")
print("=" * 60)

# ── CHECKPOINT: weight matrix (per variable) ──────────────────────────────────
WEIGHT_CKPT = os.path.join(RESULTS_DIR, f'ckpt_weight_matrix_{_VAR_KEY}_{SESSION}.npy')
if os.path.exists(WEIGHT_CKPT):
    print(f"\n[CHECKPOINT] Loading weight matrix -> {WEIGHT_CKPT}")
    weight_matrix = np.load(WEIGHT_CKPT)
    print(f"  shape: {weight_matrix.shape}")
else:
    # ── GMM diagnostic: feature stats for all channels + NaN audit ───────────
    _sf_all = np.concatenate([
        np.concatenate([_b for _b in _w if len(_b) > 0])
        for _w in bin_sf_v if any(len(_b) > 0 for _b in _w)
    ])  # (N_total, 4)
    _sf_chans = _sf_all[:, 0].astype(int)
    _sf_feats = _sf_all[:, 1:]   # [x, z, ptp]
    _unique_chans_diag = np.unique(_sf_chans)
    _nan_x   = int(np.isnan(_sf_feats[:, 0]).sum())
    _nan_z   = int(np.isnan(_sf_feats[:, 1]).sum())
    _nan_ptp = int(np.isnan(_sf_feats[:, 2]).sum())
    print("\n[GMM DIAGNOSTIC] Feature (x, z, ptp) mean ± std — all channels")
    print(f"  total spikes: {len(_sf_all):,}   unique channels: {len(_unique_chans_diag)}")
    print(f"  NaN counts:  x={_nan_x:,}  z={_nan_z:,}  ptp={_nan_ptp:,}")
    print(f"  {'chan':>5}  {'n':>8}  {'n_nan_ptp':>9}  "
          f"{'x_mean':>9} {'x_std':>9}  "
          f"{'z_mean':>9} {'z_std':>9}  "
          f"{'ptp_mean':>9} {'ptp_std':>9}")
    for _dc in _unique_chans_diag:
        _m = _sf_feats[_sf_chans == _dc]
        _n_nan = int(np.isnan(_m[:, 2]).sum())
        print(f"  {int(_dc):>5}  {len(_m):>8,}  {_n_nan:>9,}  "
              f"{_m[:,0].mean():>9.2f} {_m[:,0].std():>9.4f}  "
              f"{_m[:,1].mean():>9.2f} {_m[:,1].std():>9.4f}  "
              f"{np.nanmean(_m[:,2]):>9.2f} {np.nanstd(_m[:,2]):>9.4f}")
    del _sf_all, _sf_chans, _sf_feats
    print()

    # ── live loss-curve monitors (background threads) ────────────────────────
    import threading as _threading
    _advi_epoch_ckpt = os.path.join(RESULTS_DIR, f'ckpt_{_VAR_KEY}_{SESSION}_advi_epoch.pt')
    _advi_live_plot  = os.path.join(RESULTS_DIR, f'advi_loss_live_{_VAR_KEY}_{SESSION}.png')
    _glm_losses_ckpt = os.path.join(RESULTS_DIR, f'ckpt_{_VAR_KEY}_{SESSION}_glm_losses.npz')
    _glm_live_plot   = os.path.join(RESULTS_DIR, f'glm_loss_live_{_VAR_KEY}_{SESSION}.png')

    def _advi_loss_monitor(ckpt_path, plot_path, stop_evt, interval=60):
        import matplotlib
        matplotlib.use('Agg')
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
                _tr = _ec.get('elbos', [])
                _vl = _ec.get('val_elbos', [])
                _it = _ec.get('iter', len(_tr))
                if not _tr:
                    continue
                _fig, _ax = _plt.subplots(figsize=(12, 4))
                _ax.plot(np.arange(len(_tr)), _tr,
                         lw=0.8, color='steelblue', alpha=0.85, label='Train ELBO')
                if _vl:
                    _ax.plot([v[0] for v in _vl], [v[1] for v in _vl],
                             lw=1.2, color='darkorange', marker='o', markersize=3,
                             label='Val ELBO')
                _ax.set_xlabel('Iteration')
                _ax.set_ylabel('ELBO')
                _ax.set_title(f'ADVI training (live) — iter {_it}  '
                              f'{BEHAVIOR_VAR}  {SESSION}', fontsize=9)
                _ax.legend(fontsize=8)
                _ax.grid(True, alpha=0.25)
                _plt.tight_layout()
                _plt.savefig(plot_path, dpi=130, bbox_inches='tight')
                _plt.close(_fig)
            except Exception:
                pass

    def _glm_loss_monitor(ckpt_path, plot_path, stop_evt, interval=30):
        import matplotlib
        matplotlib.use('Agg')
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
                _tr     = _gl['train_losses']
                _te_ep  = _gl.get('test_epochs',  np.array([]))
                _te_val = _gl.get('test_losses', np.array([]))
                if len(_tr) == 0:
                    continue
                _fig, axes_glm = _plt.subplots(1, 2, figsize=(14, 4))
                # left: full loss curves
                ax_glm0 = axes_glm[0]
                ax_glm0.plot(np.arange(len(_tr)), _tr,
                             lw=0.8, color='steelblue', alpha=0.85, label='Train (Poisson NLL)')
                if len(_te_ep):
                    ax_glm0.plot(_te_ep, _te_val,
                                 lw=1.4, color='darkorange', marker='o', markersize=3,
                                 label='Test (Poisson NLL)')
                ax_glm0.set_xlabel('Epoch'); ax_glm0.set_ylabel('Poisson NLL')
                ax_glm0.set_title(f'GLM loss (live) — epoch {len(_tr)}/5000  {BEHAVIOR_VAR}  {SESSION}', fontsize=9)
                ax_glm0.legend(fontsize=8); ax_glm0.grid(True, alpha=0.25)
                # right: last 20% of training (zoom on convergence tail)
                ax_glm1 = axes_glm[1]
                _tail = max(1, len(_tr) * 4 // 5)
                ax_glm1.plot(np.arange(_tail, len(_tr)), _tr[_tail:],
                             lw=0.8, color='steelblue', alpha=0.85, label='Train (tail)')
                if len(_te_ep):
                    _te_tail_mask = _te_ep >= _tail
                    if _te_tail_mask.any():
                        ax_glm1.plot(_te_ep[_te_tail_mask], _te_val[_te_tail_mask],
                                     lw=1.4, color='darkorange', marker='o', markersize=3,
                                     label='Test (tail)')
                ax_glm1.set_xlabel('Epoch'); ax_glm1.set_ylabel('Poisson NLL')
                ax_glm1.set_title('Last 20% of epochs (convergence zoom)', fontsize=9)
                ax_glm1.legend(fontsize=8); ax_glm1.grid(True, alpha=0.25)
                _plt.tight_layout()
                _plt.savefig(plot_path, dpi=130, bbox_inches='tight')
                _plt.close(_fig)
            except Exception:
                pass

    _stop_monitor = _threading.Event()
    _advi_monitor_thread = _threading.Thread(
        target=_advi_loss_monitor,
        args=(_advi_epoch_ckpt, _advi_live_plot, _stop_monitor),
        daemon=True,
    )
    _glm_monitor_thread = _threading.Thread(
        target=_glm_loss_monitor,
        args=(_glm_losses_ckpt, _glm_live_plot, _stop_monitor),
        daemon=True,
    )
    _advi_monitor_thread.start()
    _glm_monitor_thread.start()
    print(f"  Live ADVI loss plot → {_advi_live_plot}  (updated every 60 s)")
    print(f"  Live GLM  loss plot → {_glm_live_plot}  (updated every 30 s)")

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
        max_iter         = 8000,
        checkpoint_name  = f"{_VAR_KEY}_{SESSION}",
        early_stopping   = _args.early_stopping,
        es_patience      = _args.es_patience,
    )
    _stop_monitor.set()
    _advi_monitor_thread.join(timeout=5)
    _glm_monitor_thread.join(timeout=5)
    np.save(WEIGHT_CKPT, weight_matrix)
    print(f"\n[CHECKPOINT] Saved weight matrix -> {WEIGHT_CKPT}")

# ── ADVI loss curve (plot-guarded) ────────────────────────────────────────────
_elbos_npz  = os.path.join(RESULTS_DIR, f'ckpt_{_VAR_KEY}_{SESSION}_elbos.npz')
_loss_plot  = os.path.join(RESULTS_DIR, f'advi_loss_{_VAR_KEY}_{SESSION}.png')
if os.path.exists(_elbos_npz) and not os.path.exists(_loss_plot):
    _ed         = np.load(_elbos_npz)
    _tr_elbos   = _ed['train_elbos']
    _val_iters  = _ed['val_iters']
    _val_elbos  = _ed['val_elbos']

    fig_l, ax_l = plt.subplots(figsize=(12, 4))
    ax_l.plot(np.arange(len(_tr_elbos)), _tr_elbos,
              lw=0.8, color='steelblue', alpha=0.85, label='Train ELBO')
    if len(_val_iters) > 0:
        ax_l.plot(_val_iters, _val_elbos,
                  lw=1.2, color='darkorange', marker='o', markersize=2, label='Val ELBO')
    ax_l.set_xlabel('Iteration')
    ax_l.set_ylabel('ELBO')
    ax_l.set_title(f'ADVI training — {BEHAVIOR_VAR}  {SESSION}', fontsize=9)
    ax_l.legend(fontsize=8)
    ax_l.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(_loss_plot, dpi=130, bbox_inches='tight')
    print(f"Saved → {_loss_plot}")
    plt.close()
else:
    if not os.path.exists(_elbos_npz):
        print("[INFO] No ADVI elbo file found — skipping loss plot (run decode_pipeline first).")

# %%

# ── PUTATIVE NEURON VISUALISATION (plot-guarded) ──────────────────────────────
# Loads post_params (ADVI checkpoint) and visualises each GMM component (putative
# neuron) as a point on the probe geometry.
#   Left   : (x, z) coloured by mean β (behavioural tuning); size ∝ activity
#   Middle : (x, z) coloured by mean PTP amplitude (µV);     size ∝ activity
#   Right  : β(t) tuning curves for the top-20 most active components

_post_params_path = os.path.join(RESULTS_DIR, f'ckpt_{_VAR_KEY}_{SESSION}_post_params.npz')
_pn_save          = os.path.join(RESULTS_DIR, f'putative_neurons_{_VAR_KEY}_{SESSION}.png')
if os.path.exists(_post_params_path) and not os.path.exists(_pn_save):
    print("\n[PUTATIVE NEURONS] Loading post_params ...")
    _pp          = np.load(_post_params_path)
    _pn_means    = _pp['means']                  # (n_c, 3): [x µm, z µm, ptp µV]
    _pn_beta     = _pp['U'] @ _pp['V']           # (n_c, n_t): behavioural tuning
    _pn_activity = weight_matrix.sum(axis=(0, 2))  # (n_c,): total posterior weight
    n_pn = len(_pn_means)

    print(f"  {n_pn} GMM components (putative neurons)")
    print(f"  x range    : {_pn_means[:,0].min():.1f} – {_pn_means[:,0].max():.1f} µm")
    print(f"  z range    : {_pn_means[:,1].min():.1f} – {_pn_means[:,1].max():.1f} µm")
    print(f"  PTP range  : {_pn_means[:,2].min():.1f} – {_pn_means[:,2].max():.1f} µV")
    print(f"  β range    : {_pn_beta.min():.4f} – {_pn_beta.max():.4f}")
    print(f"  activity   : {_pn_activity.min():.1f} – {_pn_activity.max():.1f}  (sum of posterior weights)")

    _act_99   = np.percentile(_pn_activity, 99)
    _act_norm = np.clip(_pn_activity / max(_act_99, 1e-9), 0, 1)
    _pn_sizes = 10 + 190 * _act_norm

    _beta_mean = _pn_beta.mean(axis=1)         # (n_c,): mean tuning
    _t_bins_ms = np.arange(N_T_BINS) * (T_TRIAL / N_T_BINS) * 1000

    fig_pn, axes_pn = plt.subplots(1, 3, figsize=(18, 12),
                                    gridspec_kw={'width_ratios': [1, 1, 2]})
    fig_pn.suptitle(
        f"Putative neurons (GMM components) — {SESSION} — {BEHAVIOR_VAR}\n"
        f"{n_pn} components  |  marker size ∝ total posterior weight",
        fontsize=10,
    )

    # ── Panel 0: behavioural tuning (mean β) ─────────────────────────────────
    ax_pn0 = axes_pn[0]
    _bm_abs = max(np.abs(_beta_mean).max(), 1e-9)
    sc_pn0 = ax_pn0.scatter(
        _pn_means[:, 0], _pn_means[:, 1],
        s=_pn_sizes, c=_beta_mean,
        cmap='RdBu_r', vmin=-_bm_abs, vmax=_bm_abs,
        edgecolors='k', linewidths=0.3, alpha=0.85,
    )
    plt.colorbar(sc_pn0, ax=ax_pn0, label=f'Mean β ({BEHAVIOR_VAR})', shrink=0.6)
    ax_pn0.set_xlabel('x (µm)'); ax_pn0.set_ylabel('Depth z (µm)')
    ax_pn0.set_title('Behavioural tuning (mean β)', fontsize=9)
    ax_pn0.grid(True, alpha=0.2)

    # ── Panel 1: mean PTP amplitude ───────────────────────────────────────────
    ax_pn1 = axes_pn[1]
    _ptp99 = np.percentile(_pn_means[:, 2], 99)
    sc_pn1 = ax_pn1.scatter(
        _pn_means[:, 0], _pn_means[:, 1],
        s=_pn_sizes, c=np.clip(_pn_means[:, 2], 0, _ptp99),
        cmap='hot_r', vmin=0, vmax=_ptp99,
        edgecolors='k', linewidths=0.3, alpha=0.85,
    )
    plt.colorbar(sc_pn1, ax=ax_pn1, label='Mean PTP amplitude (µV)', shrink=0.6)
    ax_pn1.set_xlabel('x (µm)'); ax_pn1.set_ylabel('Depth z (µm)')
    ax_pn1.set_title('Amplitude (mean PTP µV)', fontsize=9)
    ax_pn1.grid(True, alpha=0.2)

    # ── Panel 2: β(t) tuning curves, top-20 by total activity ────────────────
    ax_pn2 = axes_pn[2]
    _top20  = np.argsort(_pn_activity)[-20:][::-1]
    _cmap20 = plt.cm.tab20(np.linspace(0, 1, len(_top20)))
    for _ci, _idx in enumerate(_top20):
        ax_pn2.plot(_t_bins_ms, _pn_beta[_idx],
                    color=_cmap20[_ci], lw=1.2, alpha=0.85,
                    label=f'#{_idx}  act={_pn_activity[_idx]:.0f}')
    ax_pn2.axhline(0, color='k', lw=0.6, ls='--', alpha=0.4)
    ax_pn2.set_xlabel(f'Time within trial (ms, bin={T_TRIAL/N_T_BINS*1000:.0f} ms)')
    ax_pn2.set_ylabel(f'β ({BEHAVIOR_VAR} tuning)')
    ax_pn2.set_title('β(t) tuning curves — top-20 most active components', fontsize=9)
    ax_pn2.legend(fontsize=5, loc='upper right', ncol=2, framealpha=0.7)
    ax_pn2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(_pn_save, dpi=130, bbox_inches='tight')
    print(f"Saved → {_pn_save}")
    plt.close(fig_pn)
else:
    if not os.path.exists(_post_params_path):
        print(f"[INFO] post_params checkpoint not found — skipping putative neuron plot.\n"
              f"       (expected: {_post_params_path})")

# ── weight_matrix covers only the chunk-complete trials (n_chunks * CHUNK_SIZE).
# bin_beh_v has all n_valid rows; slice it to match so sliding_window_decoder
# can reshape (n_wm * n_windows_per_trial) correctly.
_n_wm = weight_matrix.shape[0]
_bin_beh_for_decode = bin_beh_v[:_n_wm]

# ── Decode on held-out test windows (same split used for ADVI — genuinely unseen) ──
# train_idx / test_idx were passed to decode_pipeline, so ADVI never saw y_tk
# for test_idx trials. This is the only metric reported.
print(f"\n[DECODE] Running decoder on held-out test windows ...")
_, obs_beh, pred_beh, metrics_beh = sliding_window_decoder(
    weight_matrix, _bin_beh_for_decode, train_idx, test_idx,
    behavior_type="continuous", verbose=False,
    context_before=_args.context_before_bins,
    context_after=_args.context_after_bins,
)
print(f"\n{'='*52}")
print(f"  {BEHAVIOR_VAR}  R² = {metrics_beh['r2']:+.4f}   r = {metrics_beh['corr']:.4f}")
print(f"  (n_test={len(test_idx)} windows, held out from ADVI + ridge)")
print(f"{'='*52}\n")


# %%

# ── VISUALISE DECODED vs OBSERVED — one panel per test chunk (plot-guarded) ──
_fig_path = os.path.join(RESULTS_DIR, f'decode_traces_{_VAR_KEY}_{SESSION}.png')
if not os.path.exists(_fig_path):
    print(f"[PLOT] Generating decoded vs observed traces -> {_fig_path}")
    # sliding_window: window_size=7 → half_window=3, n_windows=N_T_BINS-3+1.
    # Per trial only bins [half_win : n_win] are predicted (25 bins for 30-bin trials).
    _half_win    = 7 // 2
    _n_win       = N_T_BINS - _half_win + 1
    _n_pred_bins = _n_win - _half_win           # predicted bins per trial (= 25)
    _bin_dur     = T_TRIAL / N_T_BINS           # seconds per bin
    _bin_offsets = np.arange(_half_win, _n_win) * _bin_dur  # (n_pred_bins,) time offsets within trial

    _split_label = (f"interleaved {CHUNK_SIZE}-trial chunks"
                    if CHUNK_STYLE == 'interleaved' and n_valid >= CHUNK_SIZE
                    else f"contiguous last {TEST_PERCENTAGE:.0f}%")

    if CHUNK_STYLE == 'interleaved' and n_valid >= CHUNK_SIZE:
        _chunk_of_test = test_idx // CHUNK_SIZE
        _unique_chunks = np.unique(_chunk_of_test)
    else:
        _unique_chunks = np.array([0])
        _chunk_of_test = np.zeros(len(test_idx), dtype=int)

    _n_chunks_plot = len(_unique_chunks)

    _pred_by_trial = pred_beh.reshape(len(test_idx), _n_pred_bins)
    _obs_by_trial  = obs_beh.reshape(len(test_idx), _n_pred_bins)

    fig, axes = plt.subplots(
        _n_chunks_plot, 1,
        figsize=(18, max(2.2, 2.2 * _n_chunks_plot)),
        squeeze=False,
    )

    # Plot title: R² on genuinely held-out test windows (same split as ADVI)
    fig.suptitle(
        f"Density decoding — {SESSION} — {BEHAVIOR_VAR}\n"
        f"R² = {metrics_beh['r2']:+.4f}   r = {metrics_beh['corr']:.4f}"
        f"  ({len(test_idx)} test windows, held out from ADVI + ridge)",
        fontsize=10, y=1.002,
    )

    for _pi, _ci in enumerate(_unique_chunks):
        _mask = (_chunk_of_test == _ci)
        _tidxs = np.where(_mask)[0]

        _t_ax_chunk = np.concatenate([
            trial_starts[valid_trials_beh[test_idx[_j]]] + _bin_offsets
            for _j in _tidxs
        ])
        _obs_chunk  = _obs_by_trial[_tidxs].ravel()
        _pred_chunk = _pred_by_trial[_tidxs].ravel()

        _ax = axes[_pi, 0]
        _ax.plot(_t_ax_chunk, _obs_chunk,  color='darkgray', lw=0.8, label='observed')
        _ax.plot(_t_ax_chunk, _pred_chunk, color='tab:blue',  lw=0.8, alpha=0.85, label='decoded')

        for _j in _tidxs:
            _ts = trial_starts[valid_trials_beh[test_idx[_j]]]
            _te = trial_ends[valid_trials_beh[test_idx[_j]]]
            _ax.axvspan(_ts, _te, color='lightyellow', alpha=0.45, zorder=0)

        _ax.set_ylabel(BEHAVIOR_VAR, fontsize=8)
        _ax.set_xlabel(f"Time from gate open (s)  [chunk {_ci}]", fontsize=8)
        _ax.tick_params(labelsize=7)
        _ax.grid(True, alpha=0.2)
        if _pi == 0:
            _ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    fig.savefig(_fig_path, dpi=130, bbox_inches='tight')
    print(f"Figure   -> {_fig_path}")
    plt.close(fig)


# %%

# ── SAVE DECODING RESULTS ─────────────────────────────────────────────────────
_decode_npz = os.path.join(RESULTS_DIR, f'decode_results_{_VAR_KEY}_{SESSION}.npz')
_decode_txt = os.path.join(RESULTS_DIR, f'decode_results_{_VAR_KEY}_{SESSION}.txt')

# ── 1. Save arrays needed to reproduce/evaluate decoding ─────────────────────
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

# ── 2. Save human-readable summary ────────────────────────────────────────────
_ts_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
summary_lines = [
    f"Density decoding results — {SESSION}",
    f"Saved: {_ts_now}",
    f"",
    f"--- Design ---",
    f"  BEHAVIOR_VAR  : {BEHAVIOR_VAR}",
    f"  Z_SCORE_BEH   : {Z_SCORE_BEH}",
    f"  Window length : {T_TRIAL:.1f} s",
    f"  N_T_BINS      : {N_T_BINS}  ({T_TRIAL/N_T_BINS*1000:.0f} ms/bin)",
    f"  Valid windows : {n_valid}  (train={len(train_idx)}, test={len(test_idx)})",
    f"  Train period  : {trial_starts[valid_trials_beh[train_idx[0]]]:.1f} – "
        f"{trial_ends[valid_trials_beh[train_idx[-1]]]:.1f} s",
    f"  Test period   : {trial_starts[valid_trials_beh[test_idx[0]]]:.1f} – "
        f"{trial_ends[valid_trials_beh[test_idx[-1]]]:.1f} s",
    f"  Embargo trials: {_args.embargo_trials}",  # NEW
    f"  Speed gate    : {'threshold=%.4g (%.1f%% windows kept)' % (ANIMAL_SPEED_GATE, _speed_gate_info.get('pct_stationary', 100.0)) if ANIMAL_SPEED_GATE > 0 else 'OFF'}",  ###CHANGED
    f"  Drift corr    : DREDge ({MOTION_DIR})",
    f"",
    f"--- {BEHAVIOR_VAR} decoding ---",
    f"  R²      : {metrics_beh['r2']:+.4f}",
    f"  r       : {metrics_beh['corr']:.4f}",
    f"  n_test  : {len(test_idx)} windows (held out from ADVI + ridge)",
]
with open(_decode_txt, 'w') as _f:
    _f.write('\n'.join(summary_lines) + '\n')
print(f"Summary  -> {_decode_txt}")

# ── 3. Update params JSON with final results ──────────────────────────────────
_params['decode']['animal_speed_gate_info'] = _speed_gate_info   ###CHANGED
_params['results'] = {
    BEHAVIOR_VAR: {
        'r2':    float(metrics_beh['r2']),
        'corr':  float(metrics_beh['corr']),
        'mse':   float(metrics_beh['mse']),
        'n_test_windows': int(len(test_idx)),
        'note':  'metrics computed on test_idx only — same split used for ADVI training',
    }
}
_params['completed'] = _ts_now
with open(_params_json, 'w') as _f:
    json.dump(_params, _f, indent=2)
print(f"Params   -> {_params_json}")

print()
print('\n'.join(summary_lines))