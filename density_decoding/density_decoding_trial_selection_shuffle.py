# %% [markdown]
# #### **DENSITY DECODING — CIRCULAR SHUFFLE NULL DISTRIBUTION**
#
# Identical setup to density_decoding_trial_selection.py up to and including
# spike binning (all checkpoints reused).  Then runs N_SHUFFLES iterations,
# each with a different circular shift of the behavioural time series.
#
# Directory layout:
#   SHUFFLE_DIR/                    ← parent, named <regular_results_dir>_shuffle
#     shuffle_0/                    ← one subdir per shuffle index
#     shuffle_1/
#     ...
#     shuffle_results_summary.npz   ← r² and r for all completed shuffles
#
# Each shuffle subdir contains the same outputs as a normal run
# (decode_results, params JSON, loss plots, etc.).
#
# Circular shift guarantees:
#   - behavioural autocorrelation structure preserved
#   - marginal distribution of PC values unchanged
#   - shift ≥ MIN_SHIFT_SEC (default 60 s) so fine-timescale correlation is destroyed

# %% ── IMPORTS ─────────────────────────────────────────────────────────────────
import sys
import os
import json
import pickle
import datetime
import threading

_this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path[:] = [p for p in sys.path if os.path.realpath(p) != _this_dir]

import numpy as np
import pandas as pd
import spikeinterface as si
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter1d
from scipy.stats import zscore
from pandas.core.arrays.categorical import Categorical
from pandas._libs.arrays import NDArrayBacked

# %% ── PANDAS PICKLE COMPAT ───────────────────────────────────────────────────
_orig_categorical_setstate = Categorical.__setstate__

def _patched_categorical_setstate(self, state):
    if not isinstance(state, dict):
        dtype, codes = state
        NDArrayBacked.__init__(self, codes, dtype)
    else:
        _orig_categorical_setstate(self, state)

Categorical.__setstate__ = _patched_categorical_setstate

# %% ── ARGS ───────────────────────────────────────────────────────────────────
import argparse as _argparse
_ap = _argparse.ArgumentParser(description='Density decoding — circular shuffle null distribution')
_ap.add_argument('--animal',        default='GLC6962')
_ap.add_argument('--exp',           default='_g_g0')
_ap.add_argument('--behavior_var',  default='PC1')
_ap.add_argument('--bin_size_ms',   type=float, default=100.0,
                 help='Time bin size in ms (default 100). N_T_BINS derived automatically.')
_ap.add_argument('--t_trial',       type=float, default=1.5)
_ap.add_argument('--learning_rate', type=float, default=1e-3)
_ap.add_argument('--n_workers',     type=int,   default=16)
_ap.add_argument('--chunk_style',   choices=['contiguous', 'interleaved'], default='interleaved')
_ap.add_argument('--test_percentage', type=float, default=20)
_ap.add_argument('--chunk_size',    type=int,   default=50)
_ap.add_argument('--early_stopping', action=_argparse.BooleanOptionalAction, default=True)
_ap.add_argument('--es_patience',   type=int,   default=500)
_ap.add_argument('--batch_size',    type=int,   default=32)
_ap.add_argument('--zscore_spike_features', action=_argparse.BooleanOptionalAction, default=True)
_ap.add_argument('--embargo_trials',  type=int, default=1)
_ap.add_argument('--animal_speed_gate', type=float, default=0.0)
_ap.add_argument('--selected_trials', default=None,
                 help='Path to similar_trials CSV from trial_structure.py (required for shuffle).')
# ── shuffle-specific ──────────────────────────────────────────────────────────
_ap.add_argument('--n_shuffles',    type=int,   default=100,
                 help='Number of circular-shift shuffles to run (default 100).')
_ap.add_argument('--shuffle_index', type=int,   default=None,
                 help='If set, run only this single shuffle index (0-based). '
                      'Useful for parallelising across SLURM array jobs. '
                      'If None, run shuffle_0 … shuffle_{n_shuffles-1} sequentially.')
_ap.add_argument('--min_shift_sec', type=float, default=60.0,
                 help='Minimum circular shift in seconds (default 60 s). '
                      'Ensures fine-timescale neural–behavioural correlation is destroyed.')
_ap.add_argument('--seed',          type=int,   default=42,
                 help='RNG seed for reproducible shift draws (default 42).')
_args = _ap.parse_args()

# %% ── PATHS & SETTINGS ───────────────────────────────────────────────────────
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
MOTION_DIR  = os.path.join(PREPROC_DIR, "thresholding_peaks", "motion_estimates", "motion_dredge")

BEHAVIOR_VAR  = _args.behavior_var
Z_SCORE_BEH   = _args.zscore_spike_features
T_TRIAL       = _args.t_trial
BIN_SIZE_MS   = _args.bin_size_ms
N_T_BINS      = max(1, round(T_TRIAL * 1000 / BIN_SIZE_MS))
LEARNING_RATE = _args.learning_rate
N_WORKERS     = _args.n_workers

_beh_key_dir   = BEHAVIOR_VAR.lower().replace(' ', '_')
_speedgate_tag = f"_gate{_args.animal_speed_gate:.3g}" if _args.animal_speed_gate > 0 else ""

if _args.selected_trials is None:
    raise ValueError("--selected_trials is required for the shuffle script.")
_sel_stem     = os.path.splitext(os.path.basename(_args.selected_trials))[0]
_selected_tag = f"_selected-{_sel_stem}"

# ── base results dir (same as regular run — spike/GMM checkpoints live here) ──
BASE_RESULTS_DIR = os.path.join(
    BASE_DIR, "density_decoding",
    f"{_beh_key_dir}_{BIN_SIZE_MS:.3g}ms_{T_TRIAL:.3g}s_{_args.chunk_style}_"
    f"{'zscoreboth' if _args.zscore_spike_features else 'nozscore'}_embargo{_args.embargo_trials}"
    f"{_speedgate_tag}{_selected_tag}"
)

# ── parent shuffle dir ────────────────────────────────────────────────────────
SHUFFLE_DIR = BASE_RESULTS_DIR + "_shuffle"
os.makedirs(SHUFFLE_DIR, exist_ok=True)
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)  # needed for shared checkpoints

print(f"Base results dir  : {BASE_RESULTS_DIR}")
print(f"Shuffle parent dir: {SHUFFLE_DIR}")

# %% ── LOAD BEHAVIOURAL & NEURAL DATA (identical to main script) ──────────────
sub = texture_pca_df[(texture_pca_df['animal'] == animal) &
                     (texture_pca_df['background'] == 'eink')]

beh = sub.copy()
beh_t   = beh['time'].dt.total_seconds().values
beh_var = beh[BEHAVIOR_VAR].values
beh_spd = beh['speed'].values if 'speed' in beh.columns else None

from spikeinterface import NumpyRecording
rec_si      = NumpyRecording.load(PREPROC_DIR)
ch_locs     = rec_si.get_channel_locations()
ap_fs       = rec_si.get_sampling_frequency()
ap_t_offset = float(rec_si._recording_segments[0].t_start)
n_ch        = rec_si.get_num_channels()
n_samp      = rec_si.get_num_samples()
ap_dtype    = rec_si.get_dtype()

# ── spike features (always from shared cache) ─────────────────────────────────
OUT_PATH = os.path.join(BASE_RESULTS_DIR, f'spike_features_{SESSION}.npz')
if not os.path.exists(OUT_PATH):
    raise FileNotFoundError(
        f"Spike features cache not found: {OUT_PATH}\n"
        f"Run density_decoding_trial_selection.py first to build the cache."
    )

print(f"\n[CACHE] Loading spike features → {OUT_PATH}")
_sf_data         = np.load(OUT_PATH, allow_pickle=False)
spike_times_samp = _sf_data['times_samples']
spike_channels   = _sf_data['channels']
spike_features   = _sf_data['point_source_localizations'].copy()  # will be z-scored below

spike_t_gate = spike_times_samp / ap_fs + ap_t_offset
spike_x      = spike_features[:, 0]
z_corr       = spike_features[:, 1]
spike_ptp    = spike_features[:, 2]
print(f"Loaded {len(spike_t_gate):,} spikes.")

# %% ── SPIKE FEATURE Z-SCORE ──────────────────────────────────────────────────
if _args.zscore_spike_features:
    _sf_mean = spike_features.mean(axis=0)
    _sf_std  = spike_features.std(axis=0)
    _sf_std[_sf_std < 1e-9] = 1.0
    print(f"[ZSCORE] Spike features BEFORE: means x={_sf_mean[0]:.3f} z={_sf_mean[1]:.3f} ptp={_sf_mean[2]:.3f}")
    print(f"                       BEFORE: stds  x={_sf_std[0]:.3f}  z={_sf_std[1]:.3f}  ptp={_sf_std[2]:.3f}")
    spike_features = ((spike_features - _sf_mean) / _sf_std).astype(np.float32)
    print(f"[ZSCORE] Spike features AFTER:  means x={spike_features[:,0].mean():.4f} "
          f"z={spike_features[:,1].mean():.4f} ptp={spike_features[:,2].mean():.4f}")
    print(f"                       AFTER:  stds  x={spike_features[:,0].std():.4f} "
          f"z={spike_features[:,1].std():.4f} ptp={spike_features[:,2].std():.4f}")
    _zscore_params_sf = {'x_mean': float(_sf_mean[0]), 'x_std': float(_sf_std[0]),
                         'z_mean': float(_sf_mean[1]), 'z_std': float(_sf_std[1]),
                         'ptp_mean': float(_sf_mean[2]), 'ptp_std': float(_sf_std[2])}
else:
    print("[ZSCORE] Spike features NOT z-scored.")
    _zscore_params_sf = None

# %% ── BEHAVIOURAL Z-SCORE (on full session trace) ────────────────────────────
beh_raw = beh[BEHAVIOR_VAR].values.astype(np.float32)

print(f"\n[ZSCORE] {BEHAVIOR_VAR} BEFORE: mean={np.nanmean(beh_raw):.4f}  "
      f"std={np.nanstd(beh_raw):.4f}  min={np.nanmin(beh_raw):.4f}  max={np.nanmax(beh_raw):.4f}")
if Z_SCORE_BEH:
    beh_z = zscore(beh_raw, nan_policy='omit').astype(np.float32)
    print(f"[ZSCORE] {BEHAVIOR_VAR} AFTER:  mean={np.nanmean(beh_z):.6f}  "
          f"std={np.nanstd(beh_z):.6f}  min={np.nanmin(beh_z):.4f}  max={np.nanmax(beh_z):.4f}")
else:
    beh_z = beh_raw.copy()
    print(f"[ZSCORE] {BEHAVIOR_VAR} NOT z-scored.")

# %% ── OVERLAP WINDOW & SELECTED TRIALS ──────────────────────────────────────
T_OV_START = max(beh_t.min(), ap_t_offset)
T_OV_END   = min(beh_t.max(), spike_t_gate.max())
print(f"\nOverlap window: {T_OV_START:.2f} – {T_OV_END:.2f} s  "
      f"({(T_OV_END - T_OV_START)/60:.2f} min)")

_sel_df = pd.read_csv(_args.selected_trials)
_required = {'win_start_sec', 'win_end_sec', 'win_duration_sec'}
_missing  = _required - set(_sel_df.columns)
if _missing:
    raise ValueError(f"CSV missing columns: {_missing}")

trial_starts = _sel_df['win_start_sec'].values
trial_ends   = _sel_df['win_end_sec'].values

_win_durations = trial_ends - trial_starts
T_TRIAL  = float(np.median(_win_durations))
N_T_BINS = max(1, round(T_TRIAL * 1000 / BIN_SIZE_MS))

_in_overlap = (trial_starts >= T_OV_START) & (trial_ends <= T_OV_END)
if (~_in_overlap).any():
    print(f"[WARN] Dropping {(~_in_overlap).sum()} trials outside overlap window.")
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
print(f"\n[SELECTED TRIALS] {n_trials_beh} trials  |  T_TRIAL={T_TRIAL:.4f} s  →  N_T_BINS={N_T_BINS}")

CHUNK_STYLE     = _args.chunk_style
CHUNK_SIZE      = _args.chunk_size
TEST_PERCENTAGE = _args.test_percentage
TEST_FRAC       = TEST_PERCENTAGE / 100.0
TRAIN_FRAC      = 1.0 - TEST_FRAC
ANIMAL_SPEED_GATE = _args.animal_speed_gate

# %% ── SHARED SPIKE BINS (built once, reused across all shuffles) ─────────────
sys.path.insert(0, '/u/gongt/density_decoding')
from density_decoding.utils.data_utils import BaseDataLoader
from density_decoding.decode_pipeline import decode_pipeline
from density_decoding.decoders.behavior_decoder import sliding_window_decoder

data_loader = BaseDataLoader(t_before=0.0, t_after=T_TRIAL, n_t_bins=N_T_BINS)

SPIKE_CKPT = os.path.join(BASE_RESULTS_DIR, f'ckpt_spikebins_{SESSION}.pkl')
if os.path.exists(SPIKE_CKPT):
    print(f"\n[CHECKPOINT] Loading spike bins → {SPIKE_CKPT}")
    with open(SPIKE_CKPT, 'rb') as _f:
        _ckpt = pickle.load(_f)
    bin_spike_features_all = _ckpt['bin_spike_features_all']
    bin_trial_idxs_all     = _ckpt['bin_trial_idxs_all']
    bin_time_idxs_all      = _ckpt['bin_time_idxs_all']
    if len(bin_spike_features_all) != n_trials_beh:
        raise RuntimeError(
            f"Cached {len(bin_spike_features_all)} windows != current {n_trials_beh}. "
            f"Delete {SPIKE_CKPT} and rerun density_decoding_trial_selection.py."
        )
    print(f"  {len(bin_spike_features_all)} cached windows.")
else:
    raise FileNotFoundError(
        f"Spike bins cache not found: {SPIKE_CKPT}\n"
        f"Run density_decoding_trial_selection.py first."
    )

# %% ── CIRCULAR SHIFT HELPER ──────────────────────────────────────────────────
def circular_shift_beh(beh_t_arr, beh_z_arr, shift_sec):
    """
    Circularly shift the behavioural time series by shift_sec seconds.

    The shift is applied by rolling the values array (which is on a regular
    grid at ~40 fps).  Returns a new array aligned to the same beh_t_arr.

    shift_sec > 0 means values are moved forward in time (later frames get
    earlier values), which is equivalent to looking at behavior from a
    different part of the session for each neural window.
    """
    dt_med  = float(np.median(np.diff(beh_t_arr)))
    n_shift = int(round(shift_sec / dt_med))
    return np.roll(beh_z_arr, n_shift)

def make_bin_beh(beh_z_shifted, valid_mask_in=None):
    """
    Re-bin the shifted behavioural trace into trial windows.
    Returns (bin_beh_v, valid_mask_beh, valid_trials_beh).
    """
    bin_beh_all = data_loader.process_behaviors(
        time_points       = beh_t,
        raw_behaviors     = beh_z_shifted,
        trial_start_times = trial_starts,
        trial_end_times   = trial_ends,
    )
    valid_mask_beh  = ~np.isnan(bin_beh_all).any(axis=1)

    # speed gate (if active) — same mask as real run
    if ANIMAL_SPEED_GATE > 0 and beh_spd is not None:
        _win_spd = np.array([
            np.nanmean(beh_spd[(beh_t >= trial_starts[k]) & (beh_t < trial_ends[k])])
            for k in range(n_trials_beh)
        ])
        valid_mask_beh = valid_mask_beh & (_win_spd < ANIMAL_SPEED_GATE)

    valid_trials_beh = np.where(valid_mask_beh)[0]
    bin_beh_v = bin_beh_all[valid_mask_beh]
    return bin_beh_v, valid_mask_beh, valid_trials_beh

def make_split(valid_trials_beh):
    """Reproduce the same train/test split logic as the main script."""
    n_valid = len(valid_trials_beh)
    cs = CHUNK_STYLE
    if cs == 'interleaved' and n_valid < CHUNK_SIZE:
        cs = 'contiguous'
    if cs == 'interleaved':
        _n_test_pc  = max(1, int(round(CHUNK_SIZE * TEST_FRAC)))
        _n_train_pc = CHUNK_SIZE - _n_test_pc
        train_idx = np.concatenate([
            np.arange(c * CHUNK_SIZE, c * CHUNK_SIZE + _n_train_pc)
            for c in range(n_valid // CHUNK_SIZE)
            if c * CHUNK_SIZE + _n_train_pc <= n_valid
        ])
        test_idx = np.concatenate([
            np.arange(c * CHUNK_SIZE + _n_train_pc, min((c + 1) * CHUNK_SIZE, n_valid))
            for c in range(n_valid // CHUNK_SIZE)
        ])
    else:
        _n_train  = int(n_valid * TRAIN_FRAC)
        train_idx = np.arange(_n_train)
        test_idx  = np.arange(_n_train, n_valid)
    return train_idx, test_idx, n_valid

def make_bin_sf_and_idx(valid_trials_beh):
    """Filter spike bins to the valid-trial subset."""
    bin_sf_v  = [bin_spike_features_all[k] for k in valid_trials_beh]
    bin_ti_v, bin_tmi_v = [], []
    for new_k, k in enumerate(valid_trials_beh):
        for t in range(N_T_BINS):
            old = bin_trial_idxs_all[k * N_T_BINS + t]
            bin_ti_v.append(np.full_like(old, new_k))
            bin_tmi_v.append(bin_time_idxs_all[k * N_T_BINS + t])
    return bin_sf_v, bin_ti_v, bin_tmi_v

# %% ── DRAW SHUFFLE SHIFTS ────────────────────────────────────────────────────
# Draw N_SHUFFLES shifts uniformly from [MIN_SHIFT, session_duration - MIN_SHIFT]
# ensuring each shift is ≥ MIN_SHIFT_SEC away from 0 (and its mirror).
_session_dur = beh_t.max() - beh_t.min()
_min_shift   = _args.min_shift_sec
if _session_dur < 2 * _min_shift:
    raise ValueError(
        f"Session duration {_session_dur:.1f} s is less than 2×min_shift "
        f"({2*_min_shift:.1f} s). Reduce --min_shift_sec."
    )

rng_shift = np.random.default_rng(_args.seed)
# Draw from [min_shift, session_dur - min_shift] to avoid near-zero shifts
_shift_pool = rng_shift.uniform(_min_shift, _session_dur - _min_shift,
                                size=_args.n_shuffles)
print(f"\n[SHUFFLE] {_args.n_shuffles} shuffles planned")
print(f"  min shift  : {_min_shift:.1f} s")
print(f"  session dur: {_session_dur:.1f} s")
print(f"  shift range: {_shift_pool.min():.1f} – {_shift_pool.max():.1f} s  "
      f"(median {np.median(_shift_pool):.1f} s)")

# which shuffle indices to actually run this job
if _args.shuffle_index is not None:
    _run_indices = [_args.shuffle_index]
else:
    _run_indices = list(range(_args.n_shuffles))

# %% ── GPU CHECK ──────────────────────────────────────────────────────────────
import torch
print("\n" + "="*60)
print(f"  torch {torch.__version__}  |  CUDA {torch.cuda.is_available()}  |  "
      f"GPUs: {torch.cuda.device_count()}")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"  using: {device}")
print("="*60)

# %% ── SUMMARY ACCUMULATOR ────────────────────────────────────────────────────
_summary_path = os.path.join(SHUFFLE_DIR, "shuffle_results_summary.npz")

# load existing summary if present (allows re-running to add more shuffles)
if os.path.exists(_summary_path):
    _existing = np.load(_summary_path)
    _all_r2   = list(_existing['r2'])
    _all_r    = list(_existing['r'])
    _all_idx  = list(_existing['shuffle_index'])
    _all_shift = list(_existing['shift_sec'])
    print(f"\n[SUMMARY] Loaded {len(_all_r2)} existing shuffle results.")
else:
    _all_r2, _all_r, _all_idx, _all_shift = [], [], [], []

# %% ── SHUFFLE LOOP ───────────────────────────────────────────────────────────
_VAR_KEY = BEHAVIOR_VAR.lower().replace(' ', '_')

for _shuf_i in _run_indices:
    _shift_sec   = float(_shift_pool[_shuf_i])
    _shuf_dir    = os.path.join(SHUFFLE_DIR, f"shuffle_{_shuf_i}")
    _shuf_done   = os.path.join(_shuf_dir, f"decode_results_{_VAR_KEY}_{SESSION}.npz")
    os.makedirs(_shuf_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  SHUFFLE {_shuf_i} / {_args.n_shuffles - 1}  |  "
          f"shift = {_shift_sec:.2f} s  ({_shift_sec/60:.2f} min into session)")
    print(f"  Output → {_shuf_dir}")
    print(f"{'='*65}")

    # skip if already done
    if os.path.exists(_shuf_done):
        print(f"  [SKIP] Already completed — loading existing results.")
        _ex = np.load(_shuf_done)
        if _shuf_i not in _all_idx:
            _all_r2.append(float(_ex.get('r2', np.nan)))
            _all_r.append(float(_ex.get('r', np.nan)))
            _all_idx.append(_shuf_i)
            _all_shift.append(_shift_sec)
        continue

    _t_shuf_start = datetime.datetime.now()

    # ── 1. circular-shift behaviour ───────────────────────────────────────────
    beh_z_shifted = circular_shift_beh(beh_t, beh_z, _shift_sec)

    # ── 2. bin shifted behaviour into trial windows ───────────────────────────
    bin_beh_v_s, valid_mask_s, valid_trials_s = make_bin_beh(beh_z_shifted)
    train_idx_s, test_idx_s, n_valid_s = make_split(valid_trials_s)
    bin_sf_v_s, bin_ti_v_s, bin_tmi_v_s = make_bin_sf_and_idx(valid_trials_s)

    print(f"  Valid windows: {n_valid_s}  (train={len(train_idx_s)}, test={len(test_idx_s)})")
    print(f"  bin_beh_v shape: {bin_beh_v_s.shape}")

    if len(train_idx_s) == 0 or len(test_idx_s) == 0:
        print(f"  [WARN] Empty train or test set — skipping this shuffle.")
        continue

    # ── 3. safe batch size ────────────────────────────────────────────────────
    _batch_size_s = min(_args.batch_size, max(1, len(train_idx_s)))
    if _batch_size_s < _args.batch_size:
        print(f"  [WARN] batch_size reduced to {_batch_size_s} ({len(train_idx_s)} train trials)")

    # ── 4. run decode_pipeline ────────────────────────────────────────────────
    # Each shuffle gets its own subdir for GLM/ADVI checkpoints so they don't
    # collide. The GMM checkpoint is shared from BASE_RESULTS_DIR (behavior-
    # independent), everything else is per-shuffle.
    _shuf_ckpt_name = f"{_VAR_KEY}_{SESSION}"

    # symlink or copy GMM checkpoint if present, so pipeline can reuse it
    _gmm_src  = os.path.join(BASE_RESULTS_DIR, f"ckpt_{_shuf_ckpt_name}_gmm.pkl")
    _gmm_dst  = os.path.join(_shuf_dir,        f"ckpt_{_shuf_ckpt_name}_gmm.pkl")
    if os.path.exists(_gmm_src) and not os.path.exists(_gmm_dst):
        os.symlink(_gmm_src, _gmm_dst)

    _iwm_src  = os.path.join(BASE_RESULTS_DIR, f"ckpt_{_shuf_ckpt_name}_init_weight_matrix.npy")
    _iwm_dst  = os.path.join(_shuf_dir,        f"ckpt_{_shuf_ckpt_name}_init_weight_matrix.npy")
    if os.path.exists(_iwm_src) and not os.path.exists(_iwm_dst):
        os.symlink(_iwm_src, _iwm_dst)

    weight_matrix_s = decode_pipeline(
        data_loader,
        bin_sf_v_s,
        bin_ti_v_s,
        bin_tmi_v_s,
        bin_behaviors    = bin_beh_v_s,
        behavior_type    = "continuous",
        train            = train_idx_s,
        test             = test_idx_s,
        learning_rate    = LEARNING_RATE,
        device           = device,
        n_workers        = N_WORKERS,
        checkpoint_dir   = _shuf_dir,
        max_iter         = 8000,
        checkpoint_name  = _shuf_ckpt_name,
        early_stopping   = _args.early_stopping,
        es_patience      = _args.es_patience,
        batch_size       = _batch_size_s,
    )

    # ── 5. decode ─────────────────────────────────────────────────────────────
    _n_wm_s = weight_matrix_s.shape[0]
    _, obs_s, pred_s, metrics_s = sliding_window_decoder(
        weight_matrix_s, bin_beh_v_s[:_n_wm_s], train_idx_s, test_idx_s,
        behavior_type="continuous", verbose=False,
    )

    _t_elapsed = (datetime.datetime.now() - _t_shuf_start).total_seconds()
    print(f"\n  Shuffle {_shuf_i} done in {_t_elapsed/60:.1f} min  |  "
          f"shift = {_shift_sec:.2f} s  |  "
          f"R² = {metrics_s['r2']:+.4f}   r = {metrics_s['corr']:.4f}")

    # ── 6. save per-shuffle results ───────────────────────────────────────────
    np.savez_compressed(
        _shuf_done,
        obs_test      = obs_s,
        pred_test     = pred_s,
        train_idx     = train_idx_s,
        test_idx      = test_idx_s,
        weight_matrix = weight_matrix_s,
        shift_sec     = np.float64(_shift_sec),
        shuffle_index = np.int32(_shuf_i),
        r2            = np.float64(metrics_s['r2']),
        r             = np.float64(metrics_s['corr']),
    )

    _params_s = {
        "shuffle_index":  _shuf_i,
        "shift_sec":      float(_shift_sec),
        "shift_min":      float(_shift_sec / 60),
        "session":        SESSION,
        "behavior_var":   BEHAVIOR_VAR,
        "T_TRIAL_s":      float(T_TRIAL),
        "N_T_BINS":       int(N_T_BINS),
        "bin_size_ms":    float(BIN_SIZE_MS),
        "selected_trials_meta": _sel_meta,
        "n_valid":        int(n_valid_s),
        "n_train":        int(len(train_idx_s)),
        "n_test":         int(len(test_idx_s)),
        "r2":             float(metrics_s['r2']),
        "r":              float(metrics_s['corr']),
        "elapsed_sec":    float(_t_elapsed),
        "timestamp":      datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(_shuf_dir, f"params_{_VAR_KEY}_{SESSION}.json"), 'w') as _f:
        json.dump(_params_s, _f, indent=2)

    _all_r2.append(float(metrics_s['r2']))
    _all_r.append(float(metrics_s['corr']))
    _all_idx.append(_shuf_i)
    _all_shift.append(_shift_sec)

    # ── 7. update running summary ──────────────────────────────────────────────
    np.savez_compressed(
        _summary_path,
        shuffle_index = np.array(_all_idx,  dtype=np.int32),
        shift_sec     = np.array(_all_shift, dtype=np.float64),
        r2            = np.array(_all_r2,   dtype=np.float64),
        r             = np.array(_all_r,    dtype=np.float64),
    )
    print(f"  Summary updated → {_summary_path}")

# %% ── FINAL SUMMARY ──────────────────────────────────────────────────────────
if len(_all_r2) > 0:
    print(f"\n{'='*65}")
    print(f"  SHUFFLE NULL DISTRIBUTION SUMMARY  ({len(_all_r2)} shuffles)")
    print(f"  {BEHAVIOR_VAR}  |  {SESSION}  |  selected trials: {_sel_meta['pc']} "
          f"{_sel_meta['align_mode']}-aligned  RMS≤{_sel_meta['rms_threshold']}")
    print(f"  Shuffle R²: mean={np.mean(_all_r2):+.4f}  "
          f"std={np.std(_all_r2):.4f}  "
          f"range=[{np.min(_all_r2):+.4f}, {np.max(_all_r2):+.4f}]")
    print(f"  Shuffle r : mean={np.mean(_all_r):+.4f}  "
          f"std={np.std(_all_r):.4f}  "
          f"range=[{np.min(_all_r):+.4f}, {np.max(_all_r):+.4f}]")
    print(f"  Summary   → {_summary_path}")
    print(f"{'='*65}")

    # ── plot null distribution ────────────────────────────────────────────────
    _null_plot = os.path.join(SHUFFLE_DIR, f"null_distribution_{_VAR_KEY}_{SESSION}.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"Shuffle null distribution — {SESSION} — {BEHAVIOR_VAR}\n"
        f"{len(_all_r2)} shuffles  |  circular shift {_min_shift:.0f}–"
        f"{_session_dur - _min_shift:.0f} s  |  "
        f"selected: {_sel_meta['pc']} {_sel_meta['align_mode']}-aligned "
        f"RMS≤{_sel_meta['rms_threshold']}",
        fontsize=10,
    )
    for ax, vals, label in [(axes[0], _all_r2, 'R²'), (axes[1], _all_r, 'Pearson r')]:
        ax.hist(vals, bins=20, color='steelblue', edgecolor='white', alpha=0.8,
                label='shuffle')
        ax.axvline(np.mean(vals), color='steelblue', lw=1.5, ls='--',
                   label=f'shuffle mean={np.mean(vals):+.3f}')
        ax.set_xlabel(label)
        ax.set_ylabel('Count')
        ax.set_title(f'Null distribution — {label}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(_null_plot, dpi=130, bbox_inches='tight')
    print(f"  Null plot → {_null_plot}")
    plt.close()