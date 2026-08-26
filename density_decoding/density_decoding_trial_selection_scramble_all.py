# %% [markdown]
# LEGACY — bin-scramble null distribution script. Superseded by
# density_decoding_unified.py --shuffle scramble (same logic, folded together
# with the real-run and shift-null scripts into one). Its own run_scramble_array.sh
# wrapper (referenced below) no longer exists — use run_shuffle_array.sh with
# unified.py's --shuffle scramble instead. Kept for reference only.
#
# #### **DENSITY DECODING — BIN SCRAMBLE NULL DISTRIBUTION**
#
# Instead of circularly shifting spike times (neural shuffle), this script
# randomly permutes the TIME-BIN ORDER within each trial. Two modes:
#
#   --scramble_mode neural   : permute the spike bins within each trial
#                              (behaviour stays in original order)
#   --scramble_mode behavior : permute the behaviour bins within each trial
#                              (neural data stays in original order)
#
# This breaks the within-trial temporal structure while preserving:
#   - Which spikes / behaviour values exist in each trial (counts / levels unchanged)
#   - The identity of which trial is train vs test
#   - Cross-trial variability
#
# Scrambled data is saved per-shuffle so analyse_shuffles.py can reconstruct
# exactly what was fed to the decoder.
#
# decode_pipeline is fed the FULL scrambled valid-trial set (train+test) plus
# the fixed train_idx_fixed/test_idx_fixed split (same fix as shuffle_all) —
# its internal Ridge baseline needs a non-empty test split to run. GMM/GLM/
# ADVI still only ever see scrambled data, computed from scratch per shuffle.
#
# SLURM array: sbatch run_scramble_array.sh
# (same pattern as neural shuffle — submit one index per job)

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
_ap = _argparse.ArgumentParser(description='Density decoding — bin scramble null')
_ap.add_argument('--animal',          default='GLC6962')
_ap.add_argument('--exp',             default='_g_g0')
_ap.add_argument('--behavior_var',    default='PC1')
_ap.add_argument('--bin_size_ms',     type=float, default=100.0)
_ap.add_argument('--t_trial',         type=float, default=5.0)
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
_ap.add_argument('--demean_beh', action=_argparse.BooleanOptionalAction, default=False,
                 help='Subtract per-trial mean from binned behaviour before decoding. '
                      'Must match the flag used in the real run.')
_ap.add_argument('--scramble_mode',   choices=['neural','behavior'], default='neural',
                 help='"neural": permute spike bins within each trial (behaviour fixed). '
                      '"behavior": permute behaviour bins within each trial (neural fixed).')
_ap.add_argument('--n_shuffles',      type=int,   default=100)
_ap.add_argument('--shuffle_index',   type=int,   default=None,
                 help='Run only this single shuffle index (for SLURM array jobs).')
_ap.add_argument('--seed',            type=int,   default=42)
_args = _ap.parse_args()

# ── PATHS ─────────────────────────────────────────────────────────────────────
DF_DIR = r"/ptmp/gongt/behaviour"
metadata_df    = pd.read_pickle(os.path.join(DF_DIR, "metadata_df.pkl"))
texture_pca_df = pd.read_pickle(os.path.join(DF_DIR, "texture_pca_df_preproc.pkl"))

animal  = _args.animal
exp     = _args.exp
SESSION = f"{animal}{exp}"

BASE_DIR    = rf"/nexus/posix0/MBR-neuralsystems/ephys/{animal}/{SESSION}"
PREPROC_DIR = os.path.join(BASE_DIR, "preprocessed")

BEHAVIOR_VAR  = _args.behavior_var
Z_SCORE_BEH   = _args.zscore_spike_features
BIN_SIZE_MS   = _args.bin_size_ms
T_TRIAL       = _args.t_trial
N_T_BINS      = max(1, round(T_TRIAL * 1000 / BIN_SIZE_MS))
LEARNING_RATE = _args.learning_rate
N_WORKERS     = _args.n_workers
SCRAMBLE_MODE = _args.scramble_mode

_beh_key_dir   = BEHAVIOR_VAR.lower().replace(' ', '_')
_speedgate_tag = f"_gate{_args.animal_speed_gate:.3g}" if _args.animal_speed_gate > 0 else ""

if _args.selected_trials is None:
    raise ValueError("--selected_trials is required.")
_sel_stem     = os.path.splitext(os.path.basename(_args.selected_trials))[0]
_selected_tag = f"_selected-{_sel_stem}"

_demean_tag = "_demeanbeh" if _args.demean_beh else ""

BASE_RESULTS_DIR = os.path.join(
    BASE_DIR, "density_decoding",
    f"{_beh_key_dir}_{BIN_SIZE_MS:.3g}ms_{T_TRIAL:.3g}s_{_args.chunk_style}_"
    f"{'zscoreboth' if _args.zscore_spike_features else 'nozscore'}"
    f"_embargo{_args.embargo_trials}{_speedgate_tag}{_selected_tag}{_demean_tag}_chunksize{_args.chunk_size}"
)
SHUFFLE_DIR = BASE_RESULTS_DIR + f"_{SCRAMBLE_MODE}_scramble_all"
os.makedirs(SHUFFLE_DIR, exist_ok=True)
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

print(f"Base results dir  : {BASE_RESULTS_DIR}")
print(f"Scramble dir      : {SHUFFLE_DIR}")
print(f"Scramble mode     : {SCRAMBLE_MODE}")

# ── LOAD AP RECORDING ─────────────────────────────────────────────────────────
from spikeinterface import NumpyRecording
rec_si      = NumpyRecording.load(PREPROC_DIR)
ap_fs       = rec_si.get_sampling_frequency()
ap_t_offset = float(rec_si._recording_segments[0].t_start)

# ── LOAD BEHAVIOUR ────────────────────────────────────────────────────────────
beh     = texture_pca_df[(texture_pca_df['animal'] == animal) &
                          (texture_pca_df['background'] == 'eink')].copy()
beh_t   = beh['time'].dt.total_seconds().values
beh_spd = beh['speed'].values if 'speed' in beh.columns else None

beh_raw = beh[BEHAVIOR_VAR].values.astype(np.float32)
if Z_SCORE_BEH:
    beh_z = zscore(beh_raw, nan_policy='omit').astype(np.float32)
else:
    beh_z = beh_raw.copy()

# ── LOAD SPIKE FEATURES FROM SHARED CACHE ────────────────────────────────────
OUT_PATH = os.path.join(BASE_RESULTS_DIR, f'spike_features_{SESSION}.npz')
if not os.path.exists(OUT_PATH):
    raise FileNotFoundError(
        f"Spike features cache not found: {OUT_PATH}\n"
        f"Run density_decoding_trial_selection.py first.")

print(f"\n[CACHE] Loading spike features → {OUT_PATH}")
_sf_data         = np.load(OUT_PATH, allow_pickle=False)
spike_times_samp = _sf_data['times_samples'].copy()
spike_channels   = _sf_data['channels']
spike_features   = _sf_data['point_source_localizations'].copy()
spike_t_gate     = spike_times_samp / ap_fs + ap_t_offset
print(f"Loaded {len(spike_t_gate):,} spikes.")

DEMEAN_BEH = _args.demean_beh

if _args.zscore_spike_features:
    _sf_mean = spike_features.mean(axis=0)
    _sf_std  = spike_features.std(axis=0)
    _sf_std[_sf_std < 1e-9] = 1.0
    spike_features = ((spike_features - _sf_mean) / _sf_std).astype(np.float32)

# ── SELECTED TRIALS ───────────────────────────────────────────────────────────
T_OV_START = max(beh_t.min(), ap_t_offset)
T_OV_END   = min(beh_t.max(), spike_t_gate.max())

_sel_df      = pd.read_csv(_args.selected_trials)
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
print(f"[SELECTED TRIALS] {n_trials_beh} trials  |  T_TRIAL={T_TRIAL:.4f}s  N_T_BINS={N_T_BINS}")

# ── DATA LOADER + BIN SPIKES + BIN BEHAVIOUR ──────────────────────────────────
sys.path.insert(0, '/u/gongt/density_decoding')
from density_decoding.utils.data_utils import BaseDataLoader
from density_decoding.decode_pipeline import decode_pipeline
from density_decoding.decoders.behavior_decoder import sliding_window_decoder

data_loader = BaseDataLoader(t_before=0.0, t_after=T_TRIAL, n_t_bins=N_T_BINS)

# load or compute spike bins (shared with real run)
SPIKE_CKPT = os.path.join(BASE_RESULTS_DIR, f'ckpt_spikebins_{SESSION}.pkl')
if os.path.exists(SPIKE_CKPT):
    print(f"\n[CHECKPOINT] Loading spike bins → {SPIKE_CKPT}")
    with open(SPIKE_CKPT, 'rb') as _f:
        _ckpt = pickle.load(_f)
    bin_spike_features_all = _ckpt['bin_spike_features_all']
    bin_trial_idxs_all     = _ckpt['bin_trial_idxs_all']
    bin_time_idxs_all      = _ckpt['bin_time_idxs_all']
else:
    print("Processing spike features ...")
    bin_spike_features_all, bin_trial_idxs_all, bin_time_idxs_all = \
        data_loader.process_spike_features(
            spike_times       = spike_t_gate,
            spike_channels    = spike_channels,
            spike_features    = spike_features,
            trial_start_times = trial_starts,
            trial_end_times   = trial_ends,
            valid_trials      = np.arange(n_trials_beh),
        )
    with open(SPIKE_CKPT, 'wb') as _f:
        pickle.dump({
            'bin_spike_features_all': bin_spike_features_all,
            'bin_trial_idxs_all':     bin_trial_idxs_all,
            'bin_time_idxs_all':      bin_time_idxs_all,
        }, _f, protocol=4)
    print(f"[CHECKPOINT] Saved → {SPIKE_CKPT}")

# bin behaviour
print("Binning behaviour ...")
bin_beh_all = data_loader.process_behaviors(
    time_points       = beh_t,
    raw_behaviors     = beh_z,
    trial_start_times = trial_starts,
    trial_end_times   = trial_ends,
)
valid_mask_beh   = ~np.isnan(bin_beh_all).any(axis=1)
valid_trials_beh = np.where(valid_mask_beh)[0]

ANIMAL_SPEED_GATE = _args.animal_speed_gate
if ANIMAL_SPEED_GATE > 0 and beh_spd is not None:
    _win_spd = np.array([
        np.nanmean(beh_spd[(beh_t >= trial_starts[k]) & (beh_t < trial_ends[k])])
        for k in range(n_trials_beh)
    ])
    valid_mask_beh   = valid_mask_beh & (_win_spd < ANIMAL_SPEED_GATE)
    valid_trials_beh = np.where(valid_mask_beh)[0]

# build valid-trial arrays (same as real run)
bin_beh_v = bin_beh_all[valid_mask_beh]          # (n_valid, n_t)

# ── per-trial demeaning — must match real run ─────────────────────────────────
if DEMEAN_BEH:
    bin_beh_v = bin_beh_v - bin_beh_v.mean(axis=1, keepdims=True)
    print(f"[DEMEAN_BEH] Per-trial mean subtracted from binned behaviour (matches real run).")
bin_sf_v  = [bin_spike_features_all[k] for k in valid_trials_beh]
bin_ti_v, bin_tmi_v = [], []
for new_k, k in enumerate(valid_trials_beh):
    for t in range(N_T_BINS):
        old = bin_trial_idxs_all[k * N_T_BINS + t]
        bin_ti_v.append(np.full_like(old, new_k))
        bin_tmi_v.append(bin_time_idxs_all[k * N_T_BINS + t])

n_valid = len(valid_trials_beh)
print(f"Valid trials: {n_valid}")

# ── TRAIN/TEST SPLIT (fixed across all shuffles) ──────────────────────────────
CHUNK_STYLE = _args.chunk_style
CHUNK_SIZE  = _args.chunk_size
TEST_FRAC   = _args.test_percentage / 100.0
TRAIN_FRAC  = 1.0 - TEST_FRAC

if CHUNK_STYLE == 'interleaved' and n_valid < CHUNK_SIZE:
    CHUNK_STYLE = 'contiguous'

if CHUNK_STYLE == 'interleaved' and n_valid >= CHUNK_SIZE:
    _n_test_pc  = max(1, int(round(CHUNK_SIZE * TEST_FRAC)))
    _n_train_pc = CHUNK_SIZE - _n_test_pc
    train_idx_fixed = np.concatenate([
        np.arange(c * CHUNK_SIZE, c * CHUNK_SIZE + _n_train_pc)
        for c in range(n_valid // CHUNK_SIZE)
        if c * CHUNK_SIZE + _n_train_pc <= n_valid
    ])
    test_idx_fixed = np.concatenate([
        np.arange(c * CHUNK_SIZE + _n_train_pc, min((c + 1) * CHUNK_SIZE, n_valid))
        for c in range(n_valid // CHUNK_SIZE)
    ])
else:
    _n_train = int(n_valid * TRAIN_FRAC)
    train_idx_fixed = np.arange(_n_train)
    test_idx_fixed  = np.arange(_n_train, n_valid)

print(f"Train/test: {len(train_idx_fixed)} train | {len(test_idx_fixed)} test (fixed)")

# ── GPU ───────────────────────────────────────────────────────────────────────
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")

# ── RNG — draw per-shuffle per-trial permutations ─────────────────────────────
# For reproducibility we pre-draw all permutation seeds.
rng_master = np.random.default_rng(_args.seed)
# Each shuffle gets its own seed so results are fully reproducible.
_shuffle_seeds = rng_master.integers(0, 2**31, size=_args.n_shuffles)

_VAR_KEY = BEHAVIOR_VAR.lower().replace(' ', '_')
_run_indices = ([_args.shuffle_index] if _args.shuffle_index is not None
                else list(range(_args.n_shuffles)))

# ── summary ───────────────────────────────────────────────────────────────────
_summary_path = os.path.join(SHUFFLE_DIR, "scramble_results_summary.npz")
if os.path.exists(_summary_path):
    _ex = np.load(_summary_path)
    _all_r2, _all_r = list(_ex['r2']), list(_ex['r'])
    _all_idx = list(_ex['shuffle_index'])
    print(f"[SUMMARY] Loaded {len(_all_r2)} existing results.")
else:
    _all_r2, _all_r, _all_idx = [], [], []

# ── helper: apply bin permutation to spike features ───────────────────────────
def _scramble_neural_bins(bin_sf, bin_ti, bin_tmi, perm, n_t):
    """
    Return new (bin_sf_s, bin_ti_s, bin_tmi_s) with time bins reordered
    according to perm (length n_t) for ALL trials.

    bin_sf  : list of length n_valid, each element is list of n_t arrays
              bin_sf[trial][time_bin] = spike_feature array (n_spk, 4)
    perm    : (n_t,) permutation of np.arange(n_t)

    The spike contents of each bin are swapped — bin 0 gets the spikes that
    were originally in bin perm[0], etc.  Trial index arrays are rewritten
    accordingly (they stay monotone per trial; only time index changes).
    """
    n_valid_loc = len(bin_sf)
    new_bin_sf  = []
    new_bin_ti  = []
    new_bin_tmi = []

    # build flat index: original position of bin (trial, t) in bin_ti/bin_tmi
    # bin_ti/bin_tmi are flat lists of length n_valid*n_t
    for new_k in range(n_valid_loc):
        trial_sf  = []
        for new_t in range(n_t):
            orig_t = perm[new_t]
            # original flat position for this trial and time bin
            orig_flat = new_k * n_t + orig_t
            trial_sf.append(bin_sf[new_k][orig_t])
            # new time index = new_t; trial index = new_k (unchanged)
            new_bin_ti.append(np.full_like(bin_ti[new_k * n_t + new_t], new_k))
            new_bin_tmi.append(np.full(len(bin_sf[new_k][orig_t]),
                                       new_t, dtype=bin_tmi[0].dtype))
        new_bin_sf.append(trial_sf)

    return new_bin_sf, new_bin_ti, new_bin_tmi


def _scramble_behavior_bins(bin_beh, perm):
    """
    Return new bin_beh (n_valid, n_t) with time bins reordered by perm.
    Each trial gets the same permutation applied independently.
    """
    return bin_beh[:, perm]   # (n_valid, n_t) with columns reordered


# ── DIAGNOSTIC PLOT for first 5 training trials ───────────────────────────────
N_DIAG_TRIALS = 5

def _plot_scramble_diagnostic(shuf_dir, shuf_i, perm,
                               bin_sf_orig, bin_beh_orig,
                               bin_sf_feed, bin_beh_feed,
                               train_idx, trial_starts_valid,
                               spike_t_gate_local, z_corr_local,
                               n_t, T_trial, ap_fs_loc, ap_t_offset_loc):
    """
    For training trials 0..N_DIAG_TRIALS-1, plot a 5-column row:
      Col 0: original spike scatter  (time vs depth within trial)
      Col 1: original spike bar chart (count per bin — same data as col 0, binned)
      Col 2: original beh trace
      Col 3: decoder-feed spike bar chart (scrambled if neural mode, else same as col 1)
      Col 4: decoder-feed beh trace (scrambled if behavior mode, else same as col 2)

    This layout lets you directly verify:
      - Col 1 vs Col 3: what changed in the neural input to the decoder
      - Col 2 vs Col 4: what changed in the behaviour input to the decoder
    """
    n_show  = min(N_DIAG_TRIALS, len(train_idx))
    t_edges = np.arange(n_t + 1) * (T_trial / n_t)
    t_ctrs  = 0.5 * (t_edges[:-1] + t_edges[1:])
    _bin_s  = T_trial / n_t

    fig, axes = plt.subplots(
        n_show, 5,
        figsize=(20, n_show * 2.8),
        squeeze=False,
    )
    fig.suptitle(
        f"Scramble diagnostic — {BEHAVIOR_VAR}  shuffle {shuf_i}\n"
        f"mode={SCRAMBLE_MODE}  |  perm={perm[:8]}{'…' if n_t>8 else ''}\n"
        f"Col 0: orig scatter  |  Col 1: orig spike bars (→ decoder sees col 3)  |  "
        f"Col 2: orig beh (→ decoder sees col 4)  |  "
        f"Col 3: decoder-feed spikes  |  Col 4: decoder-feed beh",
        fontsize=8,
    )

    _z_min = z_corr_local.min() if len(z_corr_local) else 0
    _z_max = z_corr_local.max() if len(z_corr_local) else 4000

    for row, tidx in enumerate(train_idx[:n_show]):
        t0   = float(trial_starts_valid[tidx])
        t1   = t0 + T_trial
        dur  = T_trial
        _m   = (spike_t_gate_local >= t0) & (spike_t_gate_local < t1)

        # ── Col 0: original spike scatter ────────────────────────────────────
        ax0 = axes[row, 0]
        ax0.scatter((spike_t_gate_local[_m] - t0), z_corr_local[_m],
                    s=1, color='black', alpha=0.15, rasterized=True)
        for _te in t_edges:
            ax0.axvline(_te, color='silver', lw=0.4, zorder=0)
        ax0.set_xlim(0, dur); ax0.set_ylim(_z_min, _z_max)
        ax0.set_ylabel(f'trial {tidx}\ndepth (µm)', fontsize=6)
        ax0.set_xlabel('Time (s)', fontsize=6)
        ax0.tick_params(labelsize=5)
        if row == 0: ax0.set_title('Orig spikes\n(scatter)', fontsize=7)

        # ── Col 1: original spike bar chart (count per bin) ───────────────────
        ax1 = axes[row, 1]
        # count spikes from the ORIGINAL (unscrambled) bin_sf_orig
        _orig_counts = np.array([len(bin_sf_orig[tidx][t]) for t in range(n_t)])
        ax1.bar(t_ctrs, _orig_counts, width=_bin_s * 0.85,
                color='steelblue', alpha=0.7, edgecolor='none')
        ax1.set_xlim(0, dur); ax1.set_xlabel('Time (s)', fontsize=6)
        ax1.set_ylabel('Spikes / bin', fontsize=6)
        ax1.tick_params(labelsize=5)
        ax1.grid(True, alpha=0.15, axis='y')
        if row == 0: ax1.set_title('Orig spikes\n(bar/bin — decoder sees col 3)', fontsize=7)

        # ── Col 2: original beh trace ─────────────────────────────────────────
        ax2 = axes[row, 2]
        ax2.stairs(bin_beh_orig[tidx], t_edges, color='steelblue', lw=1.5,
                   baseline=None)
        ax2.set_xlim(0, dur); ax2.set_xlabel('Time (s)', fontsize=6)
        ax2.set_ylabel(BEHAVIOR_VAR, fontsize=6)
        ax2.tick_params(labelsize=5)
        if row == 0: ax2.set_title(f'Orig {BEHAVIOR_VAR}\n(decoder sees col 4)', fontsize=7)

        # ── Col 3: decoder-feed spike bar chart ───────────────────────────────
        ax3 = axes[row, 3]
        if SCRAMBLE_MODE == 'neural':
            # scrambled bins — show count from bin_sf_feed (the permuted version)
            _feed_counts = np.array([len(bin_sf_feed[tidx][t]) for t in range(n_t)])
            ax3.bar(t_ctrs, _feed_counts, width=_bin_s * 0.85,
                    color='tab:green', alpha=0.7, edgecolor='none')
            # annotate total to confirm counts are preserved
            if row == 0:
                ax3.set_title(
                    f'Decoder-feed spikes (SCRAMBLED)\n'
                    f'same total counts, reordered bins', fontsize=7)
        else:
            # neural unchanged — same counts as original
            _feed_counts = _orig_counts
            ax3.bar(t_ctrs, _feed_counts, width=_bin_s * 0.85,
                    color='steelblue', alpha=0.7, edgecolor='none')
            if row == 0:
                ax3.set_title('Decoder-feed spikes\n(unchanged — same as col 1)', fontsize=7)
        ax3.set_xlim(0, dur); ax3.set_xlabel('Time (s)', fontsize=6)
        ax3.set_ylabel('Spikes / bin', fontsize=6)
        ax3.tick_params(labelsize=5)
        ax3.grid(True, alpha=0.15, axis='y')
        # sanity check: total spike counts must be preserved under neural scramble
        if SCRAMBLE_MODE == 'neural':
            _total_orig = _orig_counts.sum()
            _total_feed = _feed_counts.sum()
            _ok = '✓' if _total_orig == _total_feed else '✗ MISMATCH'
            ax3.set_xlabel(f'Time (s)  [total: orig={_total_orig} feed={_total_feed} {_ok}]',
                           fontsize=5)

        # ── Col 4: decoder-feed beh trace ─────────────────────────────────────
        ax4 = axes[row, 4]
        if SCRAMBLE_MODE == 'behavior':
            ax4.stairs(bin_beh_feed[tidx], t_edges, color='tab:red', lw=1.5,
                       baseline=None, label='decoder-feed (SCRAMBLED)')
            ax4.stairs(bin_beh_orig[tidx], t_edges, color='steelblue', lw=0.8,
                       baseline=None, alpha=0.4, label='original', linestyle='--')
            if row == 0: ax4.legend(fontsize=5, loc='upper right')
            if row == 0: ax4.set_title(f'Decoder-feed {BEHAVIOR_VAR}\n(SCRAMBLED)', fontsize=7)
        else:
            ax4.stairs(bin_beh_orig[tidx], t_edges, color='steelblue', lw=1.5,
                       baseline=None)
            if row == 0: ax4.set_title(f'Decoder-feed {BEHAVIOR_VAR}\n(unchanged)', fontsize=7)
        ax4.set_xlim(0, dur); ax4.set_xlabel('Time (s)', fontsize=6)
        ax4.set_ylabel(BEHAVIOR_VAR, fontsize=6)
        ax4.tick_params(labelsize=5)

    plt.tight_layout()
    _diag_path = os.path.join(shuf_dir,
                              f'scramble_diagnostic_{_VAR_KEY}_{SESSION}.png')
    plt.savefig(_diag_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Diagnostic → {_diag_path}")

# ── z_corr for scatter plots ──────────────────────────────────────────────────
z_corr_plot = _sf_data['point_source_localizations'][:, 1]   # unscored depth
trial_starts_valid = trial_starts[valid_trials_beh]

# ── SHUFFLE LOOP ──────────────────────────────────────────────────────────────
for _shuf_i in _run_indices:
    _shuf_dir  = os.path.join(SHUFFLE_DIR, f'shuffle_{_shuf_i}')
    _shuf_done = os.path.join(_shuf_dir,
                              f'decode_results_{_VAR_KEY}_{SESSION}.npz')
    os.makedirs(_shuf_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  SCRAMBLE {_shuf_i}/{_args.n_shuffles-1}  |  mode={SCRAMBLE_MODE}")
    print(f"{'='*65}")

    if os.path.exists(_shuf_done):
        print(f"  [SKIP] Already done.")
        _ex2 = np.load(_shuf_done)
        if _shuf_i not in _all_idx:
            _all_r2.append(float(_ex2.get('r2', np.nan)))
            _all_r.append(float(_ex2.get('r', np.nan)))
            _all_idx.append(_shuf_i)
        continue

    _t_start = datetime.datetime.now()

    # draw a per-trial-independent permutation for each time bin
    # each trial gets the SAME permutation within a shuffle so it's a
    # consistent temporal scrambling (like a "time-warp").
    # If you want per-trial-independent permutations use perm_per_trial below.
    rng_s = np.random.default_rng(int(_shuffle_seeds[_shuf_i]))
    perm  = rng_s.permutation(N_T_BINS)   # same perm applied to every trial

    print(f"  Permutation (first 10): {perm[:10]}")

    # ── step 1: scramble ALL valid trials (train + test) ─────────────────────
    # Both train and test receive the same permutation so that:
    #   - decode_pipeline (train only) sees scrambled training data
    #   - sliding_window_decoder evaluates against scrambled test data
    #     (not real behaviour) — this is the correct null distribution.
    if SCRAMBLE_MODE == 'neural':
        bin_sf_all_s, bin_ti_all_s, bin_tmi_all_s = _scramble_neural_bins(
            bin_sf_v, bin_ti_v, bin_tmi_v, perm, N_T_BINS)
        bin_beh_all_s = bin_beh_v.copy()           # behaviour unchanged
    else:  # behavior
        bin_sf_all_s  = bin_sf_v                   # neural unchanged
        bin_ti_all_s  = bin_ti_v
        bin_tmi_all_s = bin_tmi_v
        bin_beh_all_s = _scramble_behavior_bins(bin_beh_v, perm)

    # ── step 2: feed decode_pipeline the FULL scrambled valid-trial set ───────
    # (matches the shuffle_all fix) — decode_pipeline gets ALL n_valid trials
    # (scrambled) plus the real train_idx_fixed/test_idx_fixed indices into
    # that set. decode_pipeline's own internal generic_decoder step needs a
    # non-empty test split to run its Ridge baseline check; giving it the real
    # split (still scrambled data, still no leakage of the REAL behaviour/
    # spikes) satisfies that without us hand-rolling a pseudo split. GMM init,
    # GLM init, and ADVI all still see only scrambled data — no symlinks, no
    # reuse of checkpoints from the real run; everything is computed from
    # scratch. train_idx_fixed/test_idx_fixed are the SAME fixed split used
    # for every shuffle and for the real run.
    bin_sf_feed  = bin_sf_all_s
    bin_ti_feed  = bin_ti_all_s
    bin_tmi_feed = bin_tmi_all_s
    bin_beh_feed = bin_beh_all_s                    # (n_valid, n_t), scrambled

    if len(train_idx_fixed) == 0 or len(test_idx_fixed) == 0:
        print(f"  [WARN] Empty train or test — skipping.")
        continue

    _n_train_s = len(train_idx_fixed)

    # ── diagnostic plot (first 5 training trials) ─────────────────────────────
    _diag_path = os.path.join(_shuf_dir,
                              f'scramble_diagnostic_{_VAR_KEY}_{SESSION}.png')
    if not os.path.exists(_diag_path):
        _plot_scramble_diagnostic(
            shuf_dir           = _shuf_dir,
            shuf_i             = _shuf_i,
            perm               = perm,
            bin_sf_orig        = bin_sf_v,
            bin_beh_orig       = bin_beh_v,
            bin_sf_feed        = bin_sf_feed,
            bin_beh_feed       = bin_beh_feed,
            train_idx          = train_idx_fixed,
            trial_starts_valid = trial_starts_valid,
            spike_t_gate_local = spike_t_gate,
            z_corr_local       = z_corr_plot,
            n_t                = N_T_BINS,
            T_trial            = T_TRIAL,
            ap_fs_loc          = ap_fs,
            ap_t_offset_loc    = ap_t_offset,
        )

    # ── save permutation for later reconstruction ─────────────────────────────
    _perm_path = os.path.join(_shuf_dir,
                              f'scramble_perm_{_VAR_KEY}_{SESSION}.npy')
    np.save(_perm_path, perm)

    # no GMM/GLM checkpoint symlinks — GMM and GLM initialised from scratch
    # inside decode_pipeline using bin_sf_feed / bin_beh_feed (train, scrambled).

    # ── ADVI / GLM live loss monitors ─────────────────────────────────────────
    _advi_ckpt_s = os.path.join(_shuf_dir,
                                f"ckpt_{_VAR_KEY}_{SESSION}_advi_epoch.pt")
    _advi_live_s = os.path.join(_shuf_dir,
                                f"advi_loss_live_{_VAR_KEY}_{SESSION}.png")
    _glm_ckpt_s  = os.path.join(_shuf_dir,
                                f"ckpt_{_VAR_KEY}_{SESSION}_glm_losses.npz")
    _glm_live_s  = os.path.join(_shuf_dir,
                                f"glm_loss_live_{_VAR_KEY}_{SESSION}.png")

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
                    f'ADVI live — scramble {_shuf_i} ({SCRAMBLE_MODE})  {BEHAVIOR_VAR}',
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
                    f'GLM live — scramble {_shuf_i} ({SCRAMBLE_MODE})  {BEHAVIOR_VAR}',
                    fontsize=9)
                _a.legend(fontsize=8); _a.grid(alpha=0.25)
                _plt.tight_layout(); _plt.savefig(plot, dpi=100); _plt.close(_f)
            except Exception: pass

    _stop_s = threading.Event()
    threading.Thread(target=_advi_mon,
                     args=(_advi_ckpt_s, _advi_live_s, _stop_s),
                     daemon=True).start()
    threading.Thread(target=_glm_mon,
                     args=(_glm_ckpt_s, _glm_live_s, _stop_s),
                     daemon=True).start()

    # ── decode_pipeline ───────────────────────────────────────────────────────
    # bin_sf_feed / bin_ti_feed / bin_tmi_feed / bin_beh_feed cover ALL n_valid
    # trials (scrambled). train_idx_fixed/test_idx_fixed are the real, fixed
    # split indexing into that full set. GMM init, GLM init, and ADVI all see
    # only scrambled data — no reused checkpoints, everything from scratch.
    _batch_size_s = min(_args.batch_size, max(1, _n_train_s))

    weight_matrix_s = decode_pipeline(
        data_loader,
        bin_sf_feed,
        bin_ti_feed,
        bin_tmi_feed,
        bin_behaviors    = bin_beh_feed,
        behavior_type    = "continuous",
        train            = train_idx_fixed,
        test             = test_idx_fixed,
        learning_rate    = LEARNING_RATE,
        device           = device,
        n_workers        = N_WORKERS,
        checkpoint_dir   = _shuf_dir,
        max_iter         = 7000,
        checkpoint_name  = f"{_VAR_KEY}_{SESSION}",
        early_stopping   = _args.early_stopping,
        es_patience      = _args.es_patience,
        batch_size       = _batch_size_s,
    )
    _stop_s.set()

    # ── decode: evaluate test trials against SCRAMBLED behaviour ─────────────
    # bin_beh_all_s is the scrambled behaviour for all n_valid trials (built in
    # step 1 above). We evaluate the decoder on scrambled test data — that is
    # the correct null: can the decoder trained on scrambled train data predict
    # scrambled test behaviour?
    #
    # decode_pipeline returns rows only for trials referenced by train/test.
    # With interleaved chunking, leftover tail trials are dropped from split,
    # so bin_beh_all_s can be longer than weight_matrix_s. Align here (same
    # fix as shuffle_all).
    _n_wm_s = weight_matrix_s.shape[0]
    eval_train_idx = train_idx_fixed[train_idx_fixed < _n_wm_s]
    eval_test_idx  = test_idx_fixed[test_idx_fixed < _n_wm_s]
    if len(eval_train_idx) == 0 or len(eval_test_idx) == 0:
        raise ValueError(
            f"Empty eval split after alignment: n_wm={_n_wm_s}, "
            f"train={len(eval_train_idx)}, test={len(eval_test_idx)}")

    _, obs_s, pred_s, metrics_s = sliding_window_decoder(
        weight_matrix_s, bin_beh_all_s[:_n_wm_s],  # scrambled beh, aligned
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
        scramble_perm = perm,
        scramble_mode = np.bytes_(SCRAMBLE_MODE),
        shuffle_index = np.int32(_shuf_i),
        r2            = np.float64(metrics_s['r2']),
        r             = np.float64(metrics_s['corr']),
        trial_starts  = trial_starts_valid,
        trial_ends    = trial_starts_valid + T_TRIAL,
        T_TRIAL       = np.float32(T_TRIAL),
        N_T_BINS      = np.int32(N_T_BINS),
        T_OV_START    = np.float64(T_OV_START),
        T_OV_END      = np.float64(T_OV_END),
    )

    _params_s = {
        "shuffle_index":        _shuf_i,
        "scramble_mode":        SCRAMBLE_MODE,
        "scramble_perm":        perm.tolist(),
        "session":              SESSION,
        "behavior_var":         BEHAVIOR_VAR,
        "shuffle_type":         f"{SCRAMBLE_MODE}_bin_scramble",
        "T_TRIAL_s":            float(T_TRIAL),
        "N_T_BINS":             int(N_T_BINS),
        "bin_size_ms":          float(BIN_SIZE_MS),
        "selected_trials_meta": _sel_meta,
        "n_valid":              int(n_valid),
        "n_train":              int(len(train_idx_fixed)),
        "n_test":               int(len(test_idx_fixed)),
        "r2":                   float(metrics_s['r2']),
        "r":                    float(metrics_s['corr']),
        "elapsed_sec":          float(_t_elapsed),
        "note":                 ("r2/r: SCRAMBLED spikes vs REAL behaviour"
                                 if SCRAMBLE_MODE == 'neural' else
                                 "r2/r: REAL spikes vs SCRAMBLED behaviour"),
        "timestamp":            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(_shuf_dir,
                           f"params_{_VAR_KEY}_{SESSION}.json"), 'w') as _f:
        json.dump(_params_s, _f, indent=2)

    _all_r2.append(float(metrics_s['r2']))
    _all_r.append(float(metrics_s['corr']))
    _all_idx.append(_shuf_i)

    np.savez_compressed(
        _summary_path,
        shuffle_index = np.array(_all_idx,  dtype=np.int32),
        r2            = np.array(_all_r2,   dtype=np.float64),
        r             = np.array(_all_r,    dtype=np.float64),
    )
    print(f"  Summary → {_summary_path}")

# ── final summary ─────────────────────────────────────────────────────────────
if _all_r2:
    print(f"\n{'='*65}")
    print(f"  BIN SCRAMBLE ({SCRAMBLE_MODE}) SUMMARY  ({len(_all_r2)} shuffles)")
    print(f"  R²: mean={np.mean(_all_r2):+.4f}  std={np.std(_all_r2):.4f}  "
          f"[{np.min(_all_r2):+.4f}, {np.max(_all_r2):+.4f}]")
    print(f"  r : mean={np.mean(_all_r):+.4f}  std={np.std(_all_r):.4f}  "
          f"[{np.min(_all_r):+.4f}, {np.max(_all_r):+.4f}]")
    print(f"  Note: r²/r vs REAL behaviour — comparable to real run metrics")
    print(f"{'='*65}")