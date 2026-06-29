# %% [markdown]
# #### **DENSITY DECODING — CIRCULAR SHIFT NULL DISTRIBUTION**
#
# Circularly shifts either spike times OR behaviour within the session
# overlap window to destroy the neural–behavioural relationship.
#
#   --shuffle_mode neural   : shift spike times forward by a random amount (≥60 s);
#                           behaviour stays real and unshifted.
#                           GMM/GLM/ADVI all see shifted spikes + real behaviour.
#                           Evaluation: shifted test spikes vs real behaviour.
#
#   --shuffle_mode behavior : shift the binned behaviour trace circularly across
#                           trials (same idea as scramble but at the trial level);
#                           spikes stay real and unshifted.
#                           GMM/GLM/ADVI all see real spikes + shifted behaviour.
#                           Evaluation: real test spikes vs shifted behaviour.
#
# No GMM or GLM checkpoint symlinks — everything is initialised from scratch
# on the shifted data so the null distribution is fully self-consistent.
#
# Directory layout:
#   SHUFFLE_DIR/  (<base_results_dir>_chunksize<N>_<mode>_shift)
#     shuffle_0/, shuffle_1/, …
#     shuffle_results_summary.npz

import sys, os, json, pickle, datetime, threading

_this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path[:] = [p for p in sys.path if os.path.realpath(p) != _this_dir]

import numpy as np
import pandas as pd
import spikeinterface as si
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import zscore
from pandas.core.arrays.categorical import Categorical
from pandas._libs.arrays import NDArrayBacked

_orig_categorical_setstate = Categorical.__setstate__
def _patched_categorical_setstate(self, state):
    if not isinstance(state, dict):
        dtype, codes = state; NDArrayBacked.__init__(self, codes, dtype)
    else: _orig_categorical_setstate(self, state)
Categorical.__setstate__ = _patched_categorical_setstate

import argparse as _argparse
_ap = _argparse.ArgumentParser(description='Density decoding — circular shift null')
_ap.add_argument('--animal',          default='GLC6962')
_ap.add_argument('--exp',             default='_g_g0')
_ap.add_argument('--behavior_var',    default='PC1')
_ap.add_argument('--bin_size_ms',     type=float, default=100.0)
_ap.add_argument('--t_trial',         type=float, default=5)
_ap.add_argument('--learning_rate',   type=float, default=1e-3)
_ap.add_argument('--n_workers',       type=int,   default=16)
_ap.add_argument('--chunk_style',     choices=['contiguous','interleaved'], default='interleaved')
_ap.add_argument('--test_percentage', type=float, default=20)
_ap.add_argument('--chunk_size',      type=int,   default=10)
_ap.add_argument('--early_stopping',  action=_argparse.BooleanOptionalAction, default=True)
_ap.add_argument('--es_patience',     type=int,   default=500)
_ap.add_argument('--batch_size',      type=int,   default=10)
_ap.add_argument('--zscore_spike_features', action=_argparse.BooleanOptionalAction, default=True)
_ap.add_argument('--embargo_trials',  type=int,   default=0)
_ap.add_argument('--animal_speed_gate', type=float, default=0.0)
_ap.add_argument('--selected_trials', default=None,
                 help='Path to similar_trials CSV (required).')
_ap.add_argument('--n_shuffles',      type=int,   default=100)
_ap.add_argument('--shuffle_index',   type=int,   default=None,
                 help='Run only this single shuffle index (for SLURM array jobs).')
_ap.add_argument('--min_shift_sec',   type=float, default=60.0,
                 help='Minimum circular shift in seconds (default 60).')
_ap.add_argument('--shuffle_mode',      choices=['neural', 'behavior'], default='neural',
                 help='What to shift: spike times (neural) or behaviour trace (behavior).')
_ap.add_argument('--seed',            type=int,   default=42)
_args = _ap.parse_args()

SHUFFLE_MODE = _args.shuffle_mode

# ── PATHS ─────────────────────────────────────────────────────────────────────
DF_DIR = r"/ptmp/gongt/behaviour"
metadata_df    = pd.read_pickle(os.path.join(DF_DIR, "metadata_df.pkl"))
texture_pca_df = pd.read_pickle(os.path.join(DF_DIR, "texture_pca_df_preproc.pkl"))

animal  = _args.animal
exp     = _args.exp
SESSION = f"{animal}{exp}"

BASE_DIR    = rf"/nexus/posix0/MBR-neuralsystems/ephys/{animal}/{SESSION}"
PREPROC_DIR = os.path.join(BASE_DIR, "preprocessed")
PEAKS_FILE  = os.path.join(PREPROC_DIR, "thresholding_peaks", "peaks_combined2_sorted.npy")
LOC_FILE    = os.path.join(PREPROC_DIR, "thresholding_peaks", "localized_peaks_mt_raw.npy")
MOTION_DIR  = os.path.join(PREPROC_DIR, "thresholding_peaks",
                            "motion_estimates", "motion_dredge")

BEHAVIOR_VAR  = _args.behavior_var
Z_SCORE_BEH   = _args.zscore_spike_features
BIN_SIZE_MS   = _args.bin_size_ms
T_TRIAL       = _args.t_trial
N_T_BINS      = max(1, round(T_TRIAL * 1000 / BIN_SIZE_MS))
LEARNING_RATE = _args.learning_rate
N_WORKERS     = _args.n_workers

_beh_key_dir   = BEHAVIOR_VAR.lower().replace(' ', '_')
_speedgate_tag = f"_gate{_args.animal_speed_gate:.3g}" if _args.animal_speed_gate > 0 else ""

if _args.selected_trials is None:
    raise ValueError("--selected_trials is required.")
_sel_stem     = os.path.splitext(os.path.basename(_args.selected_trials))[0]
_selected_tag = f"_selected-{_sel_stem}"

BASE_RESULTS_DIR = os.path.join(
    BASE_DIR, "density_decoding",
    f"{_beh_key_dir}_{BIN_SIZE_MS:.3g}ms_{T_TRIAL:.3g}s_{_args.chunk_style}_"
    f"{'zscoreboth' if _args.zscore_spike_features else 'nozscore'}"
    f"_embargo{_args.embargo_trials}{_speedgate_tag}{_selected_tag}"
    f"_chunksize{_args.chunk_size}"
)
# shuffle dir encodes chunksize and shuffle mode
SHUFFLE_DIR = BASE_RESULTS_DIR + f"_{SHUFFLE_MODE}_shuffle_all"
os.makedirs(SHUFFLE_DIR, exist_ok=True)
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

print(f"Base results dir  : {BASE_RESULTS_DIR}")
print(f"Shuffle parent dir: {SHUFFLE_DIR}")
print(f"Shuffle mode      : {SHUFFLE_MODE}")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
beh = texture_pca_df[(texture_pca_df['animal'] == animal) &
                     (texture_pca_df['background'] == 'eink')].copy()
beh_t   = beh['time'].dt.total_seconds().values
beh_spd = beh['speed'].values if 'speed' in beh.columns else None

from spikeinterface import NumpyRecording
rec_si      = NumpyRecording.load(PREPROC_DIR)
ch_locs     = rec_si.get_channel_locations()
ap_fs       = rec_si.get_sampling_frequency()
ap_t_offset = float(rec_si._recording_segments[0].t_start)
n_ch        = rec_si.get_num_channels()
n_samp      = rec_si.get_num_samples()
ap_dtype    = rec_si.get_dtype()

# ── spike features: compute from raw data (same as density_decoding_trial_selection.py)
# If this invocation is for a single shuffle index (typical SLURM array mode),
# cache inside that shuffle dir only; otherwise do not write a shared cache.
if _args.shuffle_index is not None:
    _sf_cache_dir = os.path.join(SHUFFLE_DIR, f"shuffle_{_args.shuffle_index}")
    os.makedirs(_sf_cache_dir, exist_ok=True)
    OUT_PATH = os.path.join(_sf_cache_dir, f'spike_features_{SESSION}.npz')
else:
    OUT_PATH = None

if OUT_PATH is not None and os.path.exists(OUT_PATH):
    print(f"\n[CACHE] Loading spike features → {OUT_PATH}")
    _sf_data                = np.load(OUT_PATH, allow_pickle=False)
    spike_times_samp        = _sf_data['times_samples'].copy()
    spike_channels          = _sf_data['channels']
    spike_features_unscaled = _sf_data['point_source_localizations'].copy()
    spike_features          = spike_features_unscaled.copy()
    print(f"Loaded {len(spike_times_samp):,} spikes from cache.")
else:
    print("\n[COMPUTE] Computing spike features from raw data...")
    peaks = np.load(PEAKS_FILE)
    loc   = np.load(LOC_FILE)

    spike_times_samp = peaks['sample_index']
    spike_amp        = np.abs(peaks['amplitude']).astype(np.float32)
    spike_x          = loc['x'].astype(np.float32)
    spike_z          = loc['y'].astype(np.float32)

    spike_t_gate_tmp = spike_times_samp / ap_fs + ap_t_offset

    # ── PTP computation ───────────────────────────────────────────────────────
    N_BEFORE_PTP   = round(0.5 * ap_fs / 1000)
    N_AFTER_PTP    = round(0.5 * ap_fs / 1000)
    CHUNK_S_PTP    = 60.0
    chunk_samp_ptp = int(CHUNK_S_PTP * ap_fs)
    n_spikes       = len(peaks)
    spike_ptp      = np.full(n_spikes, np.nan, dtype=np.float32)

    print(f"Computing raw PTP  window=±{N_BEFORE_PTP} samples  "
          f"chunks={int(np.ceil(n_samp / chunk_samp_ptp))} × {CHUNK_S_PTP:.0f} s ...")
    chunk_start = 0; ci = 0
    while chunk_start < n_samp:
        chunk_end = min(chunk_start + chunk_samp_ptp, n_samp)
        lo = int(np.searchsorted(spike_times_samp, chunk_start))
        hi = int(np.searchsorted(spike_times_samp, chunk_end))
        if lo < hi:
            tr_start = max(0, chunk_start - N_BEFORE_PTP)
            tr_end   = min(n_samp, chunk_end + N_AFTER_PTP)
            traces   = rec_si.get_traces(start_frame=tr_start, end_frame=tr_end)
            idxs   = np.arange(lo, hi)
            s_rel  = spike_times_samp[idxs] - tr_start
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
        chunk_start = chunk_end; ci += 1
        if ci % 10 == 0 or chunk_start >= n_samp:
            print(f"  chunk {ci}  ({chunk_start / ap_fs / 60:.1f} min processed)")

    # ── DREDge motion correction ──────────────────────────────────────────────
    motion    = si.Motion.load(MOTION_DIR)
    z_raw     = loc['y'].astype(np.float32)
    disp      = motion.get_displacement_at_time_and_depth(
                    spike_t_gate_tmp, z_raw.astype(np.float64)).astype(np.float32)
    z_corr    = z_raw - disp
    print(f"z_corr range: {z_corr.min():.1f} – {z_corr.max():.1f} µm  "
          f"(median correction {np.median(np.abs(disp)):.2f} µm)")

    spike_channels = peaks['channel_index'].astype(np.int32)
    spike_features_unscaled = np.column_stack([
        loc['x'].astype(np.float32),
        z_corr,
        spike_ptp,
    ])
    spike_features = spike_features_unscaled.copy()

    spike_times_samp = peaks['sample_index'].astype(np.int64)
    if OUT_PATH is not None:
        np.savez(
            OUT_PATH,
            times_samples              = peaks['sample_index'].astype(np.int64),
            channels                   = spike_channels,
            point_source_localizations = spike_features_unscaled,
            t_start_gate_s             = np.float64(ap_t_offset),
            sampling_rate              = np.float64(ap_fs),
        )
        print(f"Saved → {OUT_PATH}  ({os.path.getsize(OUT_PATH)/1e6:.0f} MB)")
    else:
        print("[CACHE] No on-disk spike-feature cache written (multi-shuffle invocation).")

spike_t_gate_orig = spike_times_samp / ap_fs + ap_t_offset
print(f"Spike time range: {spike_t_gate_orig.min():.2f} – {spike_t_gate_orig.max():.2f} s")

# ── z-score spike features ────────────────────────────────────────────────────
if _args.zscore_spike_features:
    _sf_mean = spike_features.mean(axis=0)
    _sf_std  = spike_features.std(axis=0)
    _sf_std[_sf_std < 1e-9] = 1.0
    spike_features = ((spike_features - _sf_mean) / _sf_std).astype(np.float32)
    print(f"[ZSCORE] Spike features z-scored.")

# ── behaviour ─────────────────────────────────────────────────────────────────
beh_raw = beh[BEHAVIOR_VAR].values.astype(np.float32)
if Z_SCORE_BEH:
    beh_z = zscore(beh_raw, nan_policy='omit').astype(np.float32)
    print(f"[ZSCORE] {BEHAVIOR_VAR}: mean={np.nanmean(beh_z):.4f}  std={np.nanstd(beh_z):.4f}")
else:
    beh_z = beh_raw.copy()

# ── overlap window ────────────────────────────────────────────────────────────
T_OV_START = max(beh_t.min(), ap_t_offset)
T_OV_END   = min(beh_t.max(), spike_t_gate_orig.max())
T_OV_DUR   = T_OV_END - T_OV_START
print(f"\nOverlap window: {T_OV_START:.2f} – {T_OV_END:.2f} s  ({T_OV_DUR/60:.2f} min)")

# ── selected trials ───────────────────────────────────────────────────────────
_sel_df = pd.read_csv(_args.selected_trials)
trial_starts = _sel_df['win_start_sec'].values
trial_ends   = _sel_df['win_end_sec'].values
_win_durations = trial_ends - trial_starts
T_TRIAL  = float(np.median(_win_durations))
N_T_BINS = max(1, round(T_TRIAL * 1000 / BIN_SIZE_MS))

_in_overlap = (trial_starts >= T_OV_START) & (trial_ends <= T_OV_END)
trial_starts = trial_starts[_in_overlap]
trial_ends   = trial_ends[_in_overlap]
_sel_df      = _sel_df[_in_overlap].reset_index(drop=True)
n_trials_beh = len(trial_starts)

_sel_meta = {
    'source_csv':    _args.selected_trials,
    'n_trials':      int(n_trials_beh),
    'align_mode':    str(_sel_df['align_mode'].iloc[0]) if 'align_mode' in _sel_df.columns else 'unknown',
    'rms_threshold': float(_sel_df['rms_threshold'].iloc[0]) if 'rms_threshold' in _sel_df.columns else None,
    'pc':            str(_sel_df['pc'].iloc[0]) if 'pc' in _sel_df.columns else 'unknown',
    'T_TRIAL_s':     float(T_TRIAL),
}
print(f"[SELECTED TRIALS] {n_trials_beh} trials  |  T_TRIAL={T_TRIAL:.4f} s  N_T_BINS={N_T_BINS}")

CHUNK_STYLE       = _args.chunk_style
CHUNK_SIZE        = _args.chunk_size
TEST_FRAC         = _args.test_percentage / 100.0
TRAIN_FRAC        = 1.0 - TEST_FRAC
ANIMAL_SPEED_GATE = _args.animal_speed_gate

# ── pipeline imports ──────────────────────────────────────────────────────────
sys.path.insert(0, '/u/gongt/density_decoding')
from density_decoding.utils.data_utils import BaseDataLoader
from density_decoding.decode_pipeline import decode_pipeline
from density_decoding.decoders.behavior_decoder import sliding_window_decoder

data_loader = BaseDataLoader(t_before=0.0, t_after=T_TRIAL, n_t_bins=N_T_BINS)

# ── real behaviour binned into trial windows (used in every shuffle) ──────────
print("\nBinning real behaviour (fixed across all shuffles) ...")
bin_beh_all_real = data_loader.process_behaviors(
    time_points       = beh_t,
    raw_behaviors     = beh_z,
    trial_start_times = trial_starts,
    trial_end_times   = trial_ends,
)
valid_mask_real  = ~np.isnan(bin_beh_all_real).any(axis=1)
if ANIMAL_SPEED_GATE > 0 and beh_spd is not None:
    _win_spd = np.array([
        np.nanmean(beh_spd[(beh_t >= trial_starts[k]) & (beh_t < trial_ends[k])])
        for k in range(n_trials_beh)
    ])
    valid_mask_real = valid_mask_real & (_win_spd < ANIMAL_SPEED_GATE)
valid_trials_real = np.where(valid_mask_real)[0]
bin_beh_v_real    = bin_beh_all_real[valid_mask_real]   # (n_valid, n_t), real

print(f"Valid behaviour windows: {valid_mask_real.sum()} / {n_trials_beh}")

# ── train/test split (fixed) ──────────────────────────────────────────────────
def make_split(n_valid):
    cs = CHUNK_STYLE
    if cs == 'interleaved' and n_valid < CHUNK_SIZE:
        cs = 'contiguous'
    if cs == 'interleaved':
        _n_test_pc  = max(1, int(round(CHUNK_SIZE * TEST_FRAC)))
        _n_train_pc = CHUNK_SIZE - _n_test_pc
        train_blocks = []
        test_blocks = []

        # Full chunks: first n_train_pc are train, remaining are test.
        n_full = n_valid // CHUNK_SIZE
        for c in range(n_full):
            base = c * CHUNK_SIZE
            train_blocks.append(np.arange(base, base + _n_train_pc, dtype=np.int64))
            test_blocks.append(np.arange(base + _n_train_pc, base + CHUNK_SIZE, dtype=np.int64))

        # Tail chunk policy requested by user:
        # - if tail <= n_train_pc: all tail trials are train
        # - if tail > n_train_pc : first n_train_pc train, remainder test
        tail_start = n_full * CHUNK_SIZE
        tail_len = n_valid - tail_start
        if tail_len > 0:
            if tail_len <= _n_train_pc:
                train_blocks.append(np.arange(tail_start, n_valid, dtype=np.int64))
            else:
                train_blocks.append(np.arange(tail_start, tail_start + _n_train_pc, dtype=np.int64))
                test_blocks.append(np.arange(tail_start + _n_train_pc, n_valid, dtype=np.int64))

        train_idx = np.concatenate(train_blocks) if len(train_blocks) else np.array([], dtype=np.int64)
        test_idx = np.concatenate(test_blocks) if len(test_blocks) else np.array([], dtype=np.int64)
    else:
        _n_train  = int(n_valid * TRAIN_FRAC)
        train_idx = np.arange(_n_train)
        test_idx  = np.arange(_n_train, n_valid)
    return train_idx, test_idx

train_idx_fixed, test_idx_fixed = make_split(len(valid_trials_real))
print(f"Train/test: {len(train_idx_fixed)} train | {len(test_idx_fixed)} test (fixed)")

# ── helper: bin spike features for a given set of spike times ─────────────────
def make_bin_sf(spike_t):
    bin_sf_all, bin_ti_all, bin_tmi_all = data_loader.process_spike_features(
        spike_times       = spike_t,
        spike_channels    = spike_channels,
        spike_features    = spike_features,
        trial_start_times = trial_starts,
        trial_end_times   = trial_ends,
        valid_trials      = valid_trials_real,
    )
    bin_sf_v  = [bin_sf_all[k] for k in range(len(valid_trials_real))]
    bin_ti_v, bin_tmi_v = [], []
    for new_k in range(len(valid_trials_real)):
        for t in range(N_T_BINS):
            old = bin_ti_all[new_k * N_T_BINS + t]
            bin_ti_v.append(np.full_like(old, new_k))
            bin_tmi_v.append(bin_tmi_all[new_k * N_T_BINS + t])
    return bin_sf_v, bin_ti_v, bin_tmi_v

def trial_spike_counts(bin_sf_v):
    """Return total spikes per valid trial from binned spike features."""
    if len(bin_sf_v) == 0:
        return np.array([], dtype=np.int64)
    return np.array([
        int(sum(len(bin_sf_v[k][t]) for t in range(N_T_BINS)))
        for k in range(len(bin_sf_v))
    ], dtype=np.int64)

def print_trial_spike_counts(orig_counts, new_counts, valid_trial_indices):
    """Print original vs shifted spike counts for every valid trial."""
    if len(orig_counts) != len(new_counts):
        print(f"  [WARN] count length mismatch: orig={len(orig_counts)} new={len(new_counts)}")
        return
    print("  Trial spike counts (original vs shifted):")
    print("    k_idx  valid_idx   original   shifted   delta")
    for k in range(len(orig_counts)):
        _vk = int(valid_trial_indices[k]) if k < len(valid_trial_indices) else -1
        _o = int(orig_counts[k]); _n = int(new_counts[k])
        print(f"    {k:5d}  {_vk:9d}  {_o:9d}  {_n:8d}  {_n - _o:+6d}")

# ── neural circular shift helper ──────────────────────────────────────────────
def circular_shift_spikes(spike_t_arr, shift_sec, t_start, t_end):
    """
    Circularly shift spike times by shift_sec within [t_start, t_end].
    Spikes outside the window are left unchanged.
    """
    dur  = t_end - t_start
    mask = (spike_t_arr >= t_start) & (spike_t_arr < t_end)
    shifted = spike_t_arr.copy()
    shifted[mask] = t_start + (spike_t_arr[mask] - t_start + shift_sec) % dur
    return shifted

# ── behaviour circular shift helper ──────────────────────────────────────────
def circular_shift_behavior(bin_beh, shift_bins):
    """
    Circularly shift the behaviour trace by shift_bins time bins across all
    trials simultaneously. bin_beh shape: (n_valid, n_t).
    """
    return np.roll(bin_beh, shift_bins, axis=1)

# ── draw shifts ───────────────────────────────────────────────────────────────
_min_shift = _args.min_shift_sec
if T_OV_DUR < 2 * _min_shift:
    raise ValueError(
        f"Overlap duration {T_OV_DUR:.1f} s < 2×min_shift ({2*_min_shift:.1f} s). "
        f"Reduce --min_shift_sec.")

rng_shift   = np.random.default_rng(_args.seed)
_shift_pool = rng_shift.uniform(_min_shift, T_OV_DUR - _min_shift, size=_args.n_shuffles)
print(f"\n[SHUFFLE] {_args.n_shuffles} {SHUFFLE_MODE}-shuffle shuffles planned")
print(f"  shift range: {_shift_pool.min():.1f} – {_shift_pool.max():.1f} s  "
      f"(median {np.median(_shift_pool):.1f} s)")

_run_indices = ([_args.shuffle_index] if _args.shuffle_index is not None
                else list(range(_args.n_shuffles)))

# ── GPU ───────────────────────────────────────────────────────────────────────
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")

# ── summary accumulator ───────────────────────────────────────────────────────
if _args.shuffle_index is not None:
    _summary_path = os.path.join(SHUFFLE_DIR, f"shuffle_{_args.shuffle_index}", "shuffle_results_summary.npz")
else:
    _summary_path = os.path.join(SHUFFLE_DIR, "shuffle_results_summary.npz")
if os.path.exists(_summary_path):
    _ex = np.load(_summary_path)
    _all_r2, _all_r = list(_ex['r2']), list(_ex['r'])
    _all_idx, _all_shift = list(_ex['shuffle_index']), list(_ex['shift_sec'])
    print(f"[SUMMARY] Loaded {len(_all_r2)} existing results.")
else:
    _all_r2, _all_r, _all_idx, _all_shift = [], [], [], []

_VAR_KEY = BEHAVIOR_VAR.lower().replace(' ', '_')

# ── z_corr for scatter plots ──────────────────────────────────────────────────
z_corr_plot        = spike_features_unscaled[:, 1]  # depth (um), never z-scored
trial_starts_valid = trial_starts[valid_trials_real]

# ── pre-bin original (unshifted) spikes once for diagnostics ─────────────────
# Only used in the diagnostic plot (col 0 scatter + col 1 orig spike bars).
# Computed once here so we don't repeat it per shuffle.
print("Pre-binning original spikes for diagnostics ...")
bin_sf_v_orig, _, _ = make_bin_sf(spike_t_gate_orig)
_trial_counts_orig = trial_spike_counts(bin_sf_v_orig)
print(f"  Computed original spike counts for {len(_trial_counts_orig)} valid trials.")
print("Done.")

# ── diagnostic plot function ──────────────────────────────────────────────────
N_DIAG_TRIALS = 5

def _plot_shift_diagnostic(shuf_dir, shuf_i, shift_sec,
                            bin_sf_orig, bin_sf_feed,
                            bin_beh_orig, bin_beh_feed,
                            train_idx, trial_starts_valid,
                            spike_t_gate_local, z_corr_local,
                            n_t, T_trial):
    """
    5-column diagnostic matching the scramble script layout:
      Col 0: original spike scatter (time vs depth)
      Col 1: original spike bar chart (count per bin)
      Col 2: original behaviour trace
      Col 3: decoder-feed spike bar chart
               (shifted/orange if neural mode, identical/blue if behavior mode)
      Col 4: decoder-feed behaviour trace
               (shifted/red if behavior mode, identical/blue if neural mode)
    """
    n_show  = min(N_DIAG_TRIALS, len(train_idx))
    t_edges = np.arange(n_t + 1) * (T_trial / n_t)
    t_ctrs  = 0.5 * (t_edges[:-1] + t_edges[1:])
    _bin_s  = T_trial / n_t

    fig, axes = plt.subplots(n_show, 5, figsize=(22, n_show * 2.8), squeeze=False)
    fig.suptitle(
        f"Shuffle diagnostic — {BEHAVIOR_VAR}  shuffle {shuf_i}  |  "
        f"mode={SHUFFLE_MODE}  shift={shift_sec:.2f} s\n"
        f"Col 0: orig scatter  |  Col 1: orig spike bars  |  Col 2: orig beh  |  "
        f"Col 3: decoder-feed spikes  |  Col 4: decoder-feed beh",
        fontsize=8,
    )

    _z_min = z_corr_local.min() if len(z_corr_local) else 0
    _z_max = z_corr_local.max() if len(z_corr_local) else 4000

    for row, tidx in enumerate(train_idx[:n_show]):
        t0 = float(trial_starts_valid[tidx])
        t1 = t0 + T_trial

        # ── Col 0: original spike scatter ─────────────────────────────────────
        ax0 = axes[row, 0]
        _m = (spike_t_gate_local >= t0) & (spike_t_gate_local < t1)
        ax0.scatter((spike_t_gate_local[_m] - t0), z_corr_local[_m],
                    s=1, color='black', alpha=0.15, rasterized=True)
        for _te in t_edges:
            ax0.axvline(_te, color='silver', lw=0.4, zorder=0)
        ax0.set_xlim(0, T_trial); ax0.set_ylim(_z_min, _z_max)
        ax0.set_ylabel(f'trial {tidx}\ndepth (µm)', fontsize=6)
        ax0.set_xlabel('Time (s)', fontsize=6)
        ax0.tick_params(labelsize=5)
        if row == 0: ax0.set_title('Orig spikes\n(scatter)', fontsize=7)

        # ── Col 1: original spike bar chart ───────────────────────────────────
        ax1 = axes[row, 1]
        _orig_counts = np.array([len(bin_sf_orig[tidx][t]) for t in range(n_t)])
        ax1.bar(t_ctrs, _orig_counts, width=_bin_s * 0.85,
                color='steelblue', alpha=0.7, edgecolor='none')
        ax1.set_xlim(0, T_trial); ax1.set_xlabel('Time (s)', fontsize=6)
        ax1.set_ylabel('Spikes / bin', fontsize=6)
        ax1.tick_params(labelsize=5); ax1.grid(True, alpha=0.15, axis='y')
        if row == 0: ax1.set_title('Orig spikes\n(bar/bin)', fontsize=7)

        # ── Col 2: original behaviour trace ───────────────────────────────────
        ax2 = axes[row, 2]
        ax2.stairs(bin_beh_orig[tidx], t_edges, color='steelblue', lw=1.5, baseline=None)
        ax2.set_xlim(0, T_trial); ax2.set_xlabel('Time (s)', fontsize=6)
        ax2.set_ylabel(BEHAVIOR_VAR, fontsize=6)
        ax2.tick_params(labelsize=5)
        if row == 0: ax2.set_title(f'Orig {BEHAVIOR_VAR}\n(→ decoder sees col 4)', fontsize=7)

        # ── Col 3: decoder-feed spike bar chart ───────────────────────────────
        ax3 = axes[row, 3]
        _feed_counts = np.array([len(bin_sf_feed[tidx][t]) for t in range(n_t)])
        if SHUFFLE_MODE == 'neural':
            ax3.bar(t_ctrs, _feed_counts, width=_bin_s * 0.85,
                    color='tab:orange', alpha=0.7, edgecolor='none')
            _total_orig = _orig_counts.sum(); _total_feed = _feed_counts.sum()
            ax3.set_xlabel(
                f'Time (s)  [orig={_total_orig} → shifted={_total_feed}]', fontsize=5)
            if row == 0: ax3.set_title('Decoder-feed spikes\n(SHIFTED)', fontsize=7)
        else:
            # behavior mode: spikes unchanged
            ax3.bar(t_ctrs, _feed_counts, width=_bin_s * 0.85,
                    color='steelblue', alpha=0.7, edgecolor='none')
            ax3.set_xlabel('Time (s)', fontsize=6)
            if row == 0: ax3.set_title('Decoder-feed spikes\n(unchanged)', fontsize=7)
        ax3.set_xlim(0, T_trial); ax3.set_ylabel('Spikes / bin', fontsize=6)
        ax3.tick_params(labelsize=5); ax3.grid(True, alpha=0.15, axis='y')

        # ── Col 4: decoder-feed behaviour trace ───────────────────────────────
        ax4 = axes[row, 4]
        if SHUFFLE_MODE == 'behavior':
            ax4.stairs(bin_beh_feed[tidx], t_edges, color='tab:red', lw=1.5,
                       baseline=None, label='decoder-feed (SHIFTED)')
            ax4.stairs(bin_beh_orig[tidx], t_edges, color='steelblue', lw=0.8,
                       baseline=None, alpha=0.4, linestyle='--', label='original')
            if row == 0: ax4.legend(fontsize=5, loc='upper right')
            if row == 0: ax4.set_title(f'Decoder-feed {BEHAVIOR_VAR}\n(SHIFTED)', fontsize=7)
        else:
            ax4.stairs(bin_beh_orig[tidx], t_edges, color='steelblue', lw=1.5, baseline=None)
            if row == 0: ax4.set_title(f'Decoder-feed {BEHAVIOR_VAR}\n(unchanged)', fontsize=7)
        ax4.set_xlim(0, T_trial); ax4.set_xlabel('Time (s)', fontsize=6)
        ax4.set_ylabel(BEHAVIOR_VAR, fontsize=6)
        ax4.tick_params(labelsize=5)

    plt.tight_layout()
    _diag_path = os.path.join(shuf_dir, f'shuffle_diagnostic_{_VAR_KEY}_{SESSION}.png')
    plt.savefig(_diag_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Diagnostic → {_diag_path}")


# ── shuffle loop ──────────────────────────────────────────────────────────────
for _shuf_i in _run_indices:
    _shift_sec = float(_shift_pool[_shuf_i])
    _shuf_dir  = os.path.join(SHUFFLE_DIR, f"shuffle_{_shuf_i}")
    _shuf_done = os.path.join(_shuf_dir, f"decode_results_{_VAR_KEY}_{SESSION}.npz")
    os.makedirs(_shuf_dir, exist_ok=True)

    # Guard against accidental checkpoint reuse under a different shift.
    _saved_shift_path = os.path.join(_shuf_dir, f'shift_sec_{_VAR_KEY}_{SESSION}.npy')
    if os.path.exists(_saved_shift_path):
        _saved_shift = float(np.load(_saved_shift_path))
        if not np.isclose(_saved_shift, _shift_sec, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"Existing shuffle dir has shift={_saved_shift:.9f}s but current run wants "
                f"shift={_shift_sec:.9f}s for shuffle {_shuf_i}. "
                "Delete this shuffle folder before rerunning with new seed/settings."
            )

    print(f"\n{'='*65}")
    print(f"  SHUFFLE {_shuf_i}/{_args.n_shuffles-1}  |  "
          f"mode={SHUFFLE_MODE}  shift={_shift_sec:.2f} s")
    print(f"{'='*65}")

    if os.path.exists(_shuf_done):
        print(f"  [SKIP] Already done.")
        _ex2 = np.load(_shuf_done)
        if _shuf_i not in _all_idx:
            _all_r2.append(float(_ex2.get('r2', np.nan)))
            _all_r.append(float(_ex2.get('r', np.nan)))
            _all_idx.append(_shuf_i); _all_shift.append(_shift_sec)
        continue

    _t_start = datetime.datetime.now()

    # ── step 1: apply circular shift ─────────────────────────────────────────
    if SHUFFLE_MODE == 'neural':
        # shift spike times; behaviour stays real
        spike_t_shifted = circular_shift_spikes(
            spike_t_gate_orig, _shift_sec, T_OV_START, T_OV_END)
        print(f"  Binning shifted spikes ...")
        bin_sf_v_s, bin_ti_v_s, bin_tmi_v_s = make_bin_sf(spike_t_shifted)
        _trial_counts_shift = trial_spike_counts(bin_sf_v_s)
        print_trial_spike_counts(_trial_counts_orig, _trial_counts_shift, valid_trials_real)
        bin_beh_v_s = bin_beh_v_real          # real, unshifted
        print(f"  Done. {len(bin_sf_v_s)} windows.")
    else:  # behavior
        # spikes stay real; circularly shift behaviour by shift_sec converted to bins
        spike_t_shifted = spike_t_gate_orig   # unchanged
        bin_sf_v_s   = bin_sf_v_orig          # original binned spikes (pre-computed)
        bin_ti_v_s   = None                   # will use real below — see decode_pipeline call
        bin_tmi_v_s  = None
        _shift_bins  = int(round(_shift_sec / (T_TRIAL / N_T_BINS))) % N_T_BINS
        bin_beh_v_s  = circular_shift_behavior(bin_beh_v_real, _shift_bins)
        print(f"  Behaviour shifted by {_shift_bins} bins ({_shift_sec:.2f} s).")

    # for behavior mode we need the real spike bins for decode_pipeline
    if SHUFFLE_MODE == 'behavior':
        # re-use pre-computed original bins
        bin_sf_v_s, bin_ti_v_s, bin_tmi_v_s = make_bin_sf(spike_t_gate_orig)
        _trial_counts_shift = trial_spike_counts(bin_sf_v_s)
        print_trial_spike_counts(_trial_counts_orig, _trial_counts_shift, valid_trials_real)

    # ── diagnostic plot ───────────────────────────────────────────────────────
    _diag_path = os.path.join(_shuf_dir, f'shuffle_diagnostic_{_VAR_KEY}_{SESSION}.png')
    if not os.path.exists(_diag_path):
        _plot_shift_diagnostic(
            shuf_dir           = _shuf_dir,
            shuf_i             = _shuf_i,
            shift_sec          = _shift_sec,
            bin_sf_orig        = bin_sf_v_orig,
            bin_sf_feed        = bin_sf_v_s,
            bin_beh_orig       = bin_beh_v_real,
            bin_beh_feed       = bin_beh_v_s,
            train_idx          = train_idx_fixed,
            trial_starts_valid = trial_starts_valid,
            spike_t_gate_local = spike_t_gate_orig,
            z_corr_local       = z_corr_plot,
            n_t                = N_T_BINS,
            T_trial            = T_TRIAL,
        )

    # ── save shift value ──────────────────────────────────────────────────────
    np.save(os.path.join(_shuf_dir, f'shift_sec_{_VAR_KEY}_{SESSION}.npy'),
            np.float64(_shift_sec))

    if len(train_idx_fixed) == 0 or len(test_idx_fixed) == 0:
        print(f"  [WARN] Empty train or test — skipping.")
        continue

    _batch_size_s = min(_args.batch_size, max(1, len(train_idx_fixed)))

    # no GMM/GLM checkpoint symlinks — both initialised from scratch on
    # shifted data (bin_sf_v_s for neural mode, bin_beh_v_s for behavior mode).

    # ── ADVI / GLM live monitors ──────────────────────────────────────────────
    _advi_ckpt_s = os.path.join(_shuf_dir, f"ckpt_{_VAR_KEY}_{SESSION}_advi_epoch.pt")
    _advi_live_s = os.path.join(_shuf_dir, f"advi_loss_live_{_VAR_KEY}_{SESSION}.png")
    _glm_ckpt_s  = os.path.join(_shuf_dir, f"ckpt_{_VAR_KEY}_{SESSION}_glm_losses.npz")
    _glm_live_s  = os.path.join(_shuf_dir, f"glm_loss_live_{_VAR_KEY}_{SESSION}.png")

    def _advi_mon(ckpt, plot, stop, interval=60):
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as _plt
        _lmt = 0
        while not stop.wait(interval):
            if not os.path.exists(ckpt): continue
            try:
                _mt = os.path.getmtime(ckpt)
                if _mt <= _lmt: continue
                _lmt = _mt
                _ec = torch.load(ckpt, map_location='cpu', weights_only=False)
                _tr = _ec.get('elbos', [])
                if not _tr: continue
                _f, _a = _plt.subplots(figsize=(10, 3))
                _a.plot(_tr, lw=0.8, color='steelblue', label='Train ELBO')
                _a.set_title(
                    f'ADVI live — shuffle {_shuf_i} ({SHUFFLE_MODE})  {BEHAVIOR_VAR}',
                    fontsize=9)
                _a.legend(fontsize=8); _a.grid(alpha=0.25)
                _plt.tight_layout(); _plt.savefig(plot, dpi=100); _plt.close(_f)
            except Exception: pass

    def _glm_mon(ckpt, plot, stop, interval=30):
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as _plt
        _lmt = 0
        while not stop.wait(interval):
            if not os.path.exists(ckpt): continue
            try:
                _mt = os.path.getmtime(ckpt)
                if _mt <= _lmt: continue
                _lmt = _mt
                _gl = np.load(ckpt); _tr = _gl['train_losses']
                if len(_tr) == 0: continue
                _skip = max(1, len(_tr) // 20)
                _yhi  = float(np.nanpercentile(_tr[_skip:], 99)) \
                        if len(_tr) > _skip else float(_tr.max())
                _ylo  = float(np.nanmin(_tr))
                _f, _a = _plt.subplots(figsize=(10, 3))
                _a.plot(_tr, lw=0.8, color='steelblue', label='Train NLL')
                _a.set_ylim(_ylo - abs(_ylo) * 0.05, _yhi + abs(_yhi) * 0.05)
                _a.set_title(
                    f'GLM live — shuffle {_shuf_i} ({SHUFFLE_MODE})  {BEHAVIOR_VAR}',
                    fontsize=9)
                _a.legend(fontsize=8); _a.grid(alpha=0.25)
                _plt.tight_layout(); _plt.savefig(plot, dpi=100); _plt.close(_f)
            except Exception: pass

    _stop_s = threading.Event()
    threading.Thread(target=_advi_mon,
                     args=(_advi_ckpt_s, _advi_live_s, _stop_s),
                     daemon=True).start()
    threading.Thread(target=_glm_mon,
                     args=(_glm_ckpt_s,  _glm_live_s,  _stop_s),
                     daemon=True).start()

    # ── decode_pipeline ───────────────────────────────────────────────────────
    # GMM init sees bin_sf_v_s  (shifted spikes if neural, real spikes if behavior)
    # GLM init sees bin_beh_v_s (real beh if neural, shifted beh if behavior)
    # ADVI trains on train_idx_fixed rows of the same arrays.
    # No checkpoint symlinks — everything computed from scratch on shifted data.
    weight_matrix_s = decode_pipeline(
        data_loader,
        bin_sf_v_s,
        bin_ti_v_s,
        bin_tmi_v_s,
        bin_behaviors    = bin_beh_v_s,
        behavior_type    = "continuous",
        train            = train_idx_fixed,
        test             = test_idx_fixed,
        learning_rate    = LEARNING_RATE,
        device           = device,
        n_workers        = N_WORKERS,
        checkpoint_dir   = _shuf_dir,
        max_iter         = 8000,
        checkpoint_name  = f"{_VAR_KEY}_{SESSION}",
        early_stopping   = _args.early_stopping,
        es_patience      = _args.es_patience,
        batch_size       = _batch_size_s,
    )
    _stop_s.set()

    # ── evaluate ──────────────────────────────────────────────────────────────
    # decode_pipeline returns rows only for trials referenced by train/test.
    # With interleaved chunking, leftover tail trials are dropped from split,
    # so bin_beh_v_s can be longer than weight_matrix_s. Align here.
    _n_wm = weight_matrix_s.shape[0]
    eval_train_idx = train_idx_fixed[train_idx_fixed < _n_wm]
    eval_test_idx  = test_idx_fixed[test_idx_fixed < _n_wm]
    if len(eval_train_idx) == 0 or len(eval_test_idx) == 0:
        raise ValueError(
            f"Empty eval split after alignment: n_wm={_n_wm}, "
            f"train={len(eval_train_idx)}, test={len(eval_test_idx)}")

    # Evaluate against the SHIFTED behaviour (bin_beh_v_s), consistent with
    # what the decoder was trained on. For neural mode this is real behaviour
    # (correct — we want to know if shifted spikes can predict real behaviour).
    # For behavior mode this is shifted behaviour (correct null — shifted beh
    # trained and tested on shifted beh).
    _, obs_s, pred_s, metrics_s = sliding_window_decoder(
        weight_matrix_s, bin_beh_v_s[:_n_wm],
        eval_train_idx, eval_test_idx,
        behavior_type="continuous", verbose=False,
    )

    _t_elapsed = (datetime.datetime.now() - _t_start).total_seconds()
    print(f"\n  Done in {_t_elapsed/60:.1f} min  |  "
          f"R²={metrics_s['r2']:+.4f}  r={metrics_s['corr']:.4f}")

    # ── save ──────────────────────────────────────────────────────────────────
    np.savez_compressed(
        _shuf_done,
        obs_test      = obs_s,
        pred_test     = pred_s,
        train_idx     = eval_train_idx,
        test_idx      = eval_test_idx,
        train_idx_global = train_idx_fixed,
        test_idx_global  = test_idx_fixed,
        weight_matrix = weight_matrix_s,
        shift_sec     = np.float64(_shift_sec),
        shuffle_index = np.int32(_shuf_i),
        shuffle_mode  = np.bytes_(SHUFFLE_MODE),
        r2            = np.float64(metrics_s['r2']),
        r             = np.float64(metrics_s['corr']),
        T_TRIAL       = np.float32(T_TRIAL),
        N_T_BINS      = np.int32(N_T_BINS),
        T_OV_START    = np.float64(T_OV_START),
        T_OV_END      = np.float64(T_OV_END),
    )
    _params_s = {
        "shuffle_index":        _shuf_i,
        "shift_sec":            float(_shift_sec),
        "shuffle_mode":         SHUFFLE_MODE,
        "session":              SESSION,
        "behavior_var":         BEHAVIOR_VAR,
        "shuffle_type":         "circular_shift",
        "T_TRIAL_s":            float(T_TRIAL),
        "N_T_BINS":             int(N_T_BINS),
        "bin_size_ms":          float(BIN_SIZE_MS),
        "chunk_style":          CHUNK_STYLE,
        "chunk_size":           int(CHUNK_SIZE),
        "test_percentage":      float(_args.test_percentage),
        "learning_rate":        float(LEARNING_RATE),
        "early_stopping":       bool(_args.early_stopping),
        "es_patience":          int(_args.es_patience),
        "batch_size":           int(_args.batch_size),
        "n_shuffles":           int(_args.n_shuffles),
        "min_shift_sec":        float(_args.min_shift_sec),
        "seed":                 int(_args.seed),
        "zscore_spike_features": bool(_args.zscore_spike_features),
        "embargo_trials":       int(_args.embargo_trials),
        "animal_speed_gate":    float(_args.animal_speed_gate),
        "selected_trials_meta": _sel_meta,
        "n_valid":              int(len(valid_trials_real)),
        "n_train":              int(len(train_idx_fixed)),
        "n_test":               int(len(test_idx_fixed)),
        "r2":                   float(metrics_s['r2']),
        "r":                    float(metrics_s['corr']),
        "elapsed_sec":          float(_t_elapsed),
        "timestamp":            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(_shuf_dir, f"params_{_VAR_KEY}_{SESSION}.json"), 'w') as _f:
        json.dump(_params_s, _f, indent=2)

    _all_r2.append(float(metrics_s['r2']))
    _all_r.append(float(metrics_s['corr']))
    _all_idx.append(_shuf_i); _all_shift.append(_shift_sec)

    np.savez_compressed(
        _summary_path,
        shuffle_index = np.array(_all_idx,  dtype=np.int32),
        shift_sec     = np.array(_all_shift, dtype=np.float64),
        r2            = np.array(_all_r2,   dtype=np.float64),
        r             = np.array(_all_r,    dtype=np.float64),
    )
    print(f"  Summary updated → {_summary_path}")

# ── final summary ──────────────────────────────────────────────────────────────
if _all_r2:
    print(f"\n{'='*65}")
    print(f"  CIRCULAR SHIFT SUMMARY  ({len(_all_r2)} shuffles)  mode={SHUFFLE_MODE}")
    print(f"  R²: mean={np.mean(_all_r2):+.4f}  std={np.std(_all_r2):.4f}  "
          f"range=[{np.min(_all_r2):+.4f}, {np.max(_all_r2):+.4f}]")
    print(f"  r : mean={np.mean(_all_r):+.4f}  std={np.std(_all_r):.4f}  "
          f"range=[{np.min(_all_r):+.4f}, {np.max(_all_r):+.4f}]")
    print(f"{'='*65}")