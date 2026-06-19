# %% [markdown]
# #### **DENSITY DECODING — NEURAL CIRCULAR SHUFFLE NULL DISTRIBUTION**
#
# Shifts spike times circularly within the session overlap window.
# Behaviour stays fixed and unshifted.
# The decoder is trained on shifted neural data + real behaviour,
# then predicts behaviour from shifted test-trial spikes.
# Predictions are compared against real behaviour for those test windows.
#
# Directory layout:
#   SHUFFLE_DIR/  (<base_results_dir>_neural_shuffle)
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
_ap = _argparse.ArgumentParser(description='Density decoding — neural circular shuffle')
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
                 help='Minimum circular shift of spike times in seconds (default 60).')
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
)
# ── shuffle dir tagged _neural_shuffle to distinguish from beh shuffle ────────
SHUFFLE_DIR = BASE_RESULTS_DIR + "_neural_shuffle"
os.makedirs(SHUFFLE_DIR, exist_ok=True)
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

print(f"Base results dir  : {BASE_RESULTS_DIR}")
print(f"Shuffle parent dir: {SHUFFLE_DIR}")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
beh = texture_pca_df[(texture_pca_df['animal'] == animal) &
                     (texture_pca_df['background'] == 'eink')].copy()
beh_t   = beh['time'].dt.total_seconds().values
beh_spd = beh['speed'].values if 'speed' in beh.columns else None

from spikeinterface import NumpyRecording
rec_si      = NumpyRecording.load(PREPROC_DIR)
ap_fs       = rec_si.get_sampling_frequency()
ap_t_offset = float(rec_si._recording_segments[0].t_start)

# ── load spike features from shared cache ─────────────────────────────────────
OUT_PATH = os.path.join(BASE_RESULTS_DIR, f'spike_features_{SESSION}.npz')
if not os.path.exists(OUT_PATH):
    raise FileNotFoundError(
        f"Spike features cache not found: {OUT_PATH}\n"
        f"Run density_decoding_trial_selection.py first.")

print(f"\n[CACHE] Loading spike features → {OUT_PATH}")
_sf_data         = np.load(OUT_PATH, allow_pickle=False)
spike_times_samp = _sf_data['times_samples'].copy()   # int64 — will shift per shuffle
spike_channels   = _sf_data['channels']
spike_features   = _sf_data['point_source_localizations'].copy()

# gate-relative spike times for un-shifted spikes (used to define overlap window)
spike_t_gate_orig = spike_times_samp / ap_fs + ap_t_offset
print(f"Loaded {len(spike_t_gate_orig):,} spikes.")
print(f"Spike time range: {spike_t_gate_orig.min():.2f} – {spike_t_gate_orig.max():.2f} s")

# ── z-score spike features (same as real run) ─────────────────────────────────
if _args.zscore_spike_features:
    _sf_mean = spike_features.mean(axis=0)
    _sf_std  = spike_features.std(axis=0)
    _sf_std[_sf_std < 1e-9] = 1.0
    spike_features = ((spike_features - _sf_mean) / _sf_std).astype(np.float32)
    print(f"[ZSCORE] Spike features z-scored.")

# ── behaviour (unshifted throughout) ─────────────────────────────────────────
beh_raw = beh[BEHAVIOR_VAR].values.astype(np.float32)
if Z_SCORE_BEH:
    beh_z = zscore(beh_raw, nan_policy='omit').astype(np.float32)
    print(f"[ZSCORE] {BEHAVIOR_VAR}: mean={np.nanmean(beh_z):.4f}  std={np.nanstd(beh_z):.4f}")
else:
    beh_z = beh_raw.copy()

# ── overlap window ─────────────────────────────────────────────────────────────
T_OV_START = max(beh_t.min(), ap_t_offset)
T_OV_END   = min(beh_t.max(), spike_t_gate_orig.max())
T_OV_DUR   = T_OV_END - T_OV_START
print(f"\nOverlap window: {T_OV_START:.2f} – {T_OV_END:.2f} s  ({T_OV_DUR/60:.2f} min)")

# ── selected trials ────────────────────────────────────────────────────────────
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

# ── real behaviour binned into trial windows (unshifted, used in every shuffle) 
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
bin_beh_v_real   = bin_beh_all_real[valid_mask_real]   # fixed for all shuffles

print(f"Valid behaviour windows: {valid_mask_real.sum()} / {n_trials_beh}")

# ── train/test split (fixed — same as real run since behaviour validity is fixed)
def make_split(n_valid):
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
    return train_idx, test_idx

train_idx_fixed, test_idx_fixed = make_split(len(valid_trials_real))
print(f"Train/test split: {len(train_idx_fixed)} train  |  {len(test_idx_fixed)} test  (fixed)")

# ── real behaviour for test windows (for final r²/r evaluation) ───────────────
_half_win_eval = 7 // 2
_n_win_eval    = N_T_BINS - _half_win_eval + 1
_n_pred_eval   = _n_win_eval - _half_win_eval
real_test_beh  = bin_beh_v_real[test_idx_fixed,
                                 _half_win_eval : _half_win_eval + _n_pred_eval].ravel()

# ── helper: bin spike features for shifted spike times ────────────────────────
def make_bin_sf_shifted(spike_t_shifted):
    """
    Bin shifted spike times into trial windows.
    Returns (bin_sf_v, bin_ti_v, bin_tmi_v) for the valid trial subset.
    """
    bin_sf_all, bin_ti_all, bin_tmi_all = data_loader.process_spike_features(
        spike_times       = spike_t_shifted,
        spike_channels    = spike_channels,
        spike_features    = spike_features,
        trial_start_times = trial_starts,
        trial_end_times   = trial_ends,
        valid_trials      = valid_trials_real,   # same valid mask as behaviour
    )
    bin_sf_v  = [bin_sf_all[k] for k in range(len(valid_trials_real))]
    bin_ti_v, bin_tmi_v = [], []
    for new_k in range(len(valid_trials_real)):
        for t in range(N_T_BINS):
            old = bin_ti_all[new_k * N_T_BINS + t]
            bin_ti_v.append(np.full_like(old, new_k))
            bin_tmi_v.append(bin_tmi_all[new_k * N_T_BINS + t])
    return bin_sf_v, bin_ti_v, bin_tmi_v

# ── neural circular shift helper ──────────────────────────────────────────────
def circular_shift_spikes(spike_t_arr, shift_sec, t_start, t_end):
    """
    Circularly shift spike times by shift_sec within [t_start, t_end].

    Spikes are shifted forward by shift_sec; those that fall beyond t_end
    are wrapped back to t_start + (spike_t - t_end - dt).
    Spikes outside [t_start, t_end] are left unchanged (they won't fall
    in any trial window anyway).
    """
    dur = t_end - t_start
    # only shift spikes inside the overlap window
    mask = (spike_t_arr >= t_start) & (spike_t_arr < t_end)
    shifted = spike_t_arr.copy()
    shifted[mask] = t_start + (spike_t_arr[mask] - t_start + shift_sec) % dur
    return shifted

# ── draw shifts ───────────────────────────────────────────────────────────────
_min_shift = _args.min_shift_sec
if T_OV_DUR < 2 * _min_shift:
    raise ValueError(
        f"Overlap duration {T_OV_DUR:.1f} s < 2×min_shift ({2*_min_shift:.1f} s). "
        f"Reduce --min_shift_sec.")

rng_shift   = np.random.default_rng(_args.seed)
_shift_pool = rng_shift.uniform(_min_shift, T_OV_DUR - _min_shift, size=_args.n_shuffles)
print(f"\n[SHUFFLE] {_args.n_shuffles} neural-shift shuffles planned")
print(f"  shift range: {_shift_pool.min():.1f} – {_shift_pool.max():.1f} s  "
      f"(median {np.median(_shift_pool):.1f} s)")

_run_indices = ([_args.shuffle_index] if _args.shuffle_index is not None
                else list(range(_args.n_shuffles)))

# ── GPU ───────────────────────────────────────────────────────────────────────
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")

# ── summary accumulator ───────────────────────────────────────────────────────
_summary_path = os.path.join(SHUFFLE_DIR, "shuffle_results_summary.npz")
if os.path.exists(_summary_path):
    _ex = np.load(_summary_path)
    _all_r2, _all_r = list(_ex['r2']), list(_ex['r'])
    _all_idx, _all_shift = list(_ex['shuffle_index']), list(_ex['shift_sec'])
    print(f"[SUMMARY] Loaded {len(_all_r2)} existing results.")
else:
    _all_r2, _all_r, _all_idx, _all_shift = [], [], [], []

_VAR_KEY = BEHAVIOR_VAR.lower().replace(' ', '_')

# ── shuffle loop ──────────────────────────────────────────────────────────────
for _shuf_i in _run_indices:
    _shift_sec = float(_shift_pool[_shuf_i])
    _shuf_dir  = os.path.join(SHUFFLE_DIR, f"shuffle_{_shuf_i}")
    _shuf_done = os.path.join(_shuf_dir, f"decode_results_{_VAR_KEY}_{SESSION}.npz")
    os.makedirs(_shuf_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  SHUFFLE {_shuf_i}/{_args.n_shuffles-1}  |  "
          f"neural shift = {_shift_sec:.2f} s")
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

    # 1. circularly shift spike times
    spike_t_shifted = circular_shift_spikes(
        spike_t_gate_orig, _shift_sec, T_OV_START, T_OV_END)

    # 2. re-bin shifted spikes into trial windows
    # Note: spike bins are re-computed from scratch each shuffle because
    # spike times changed. This is the main computational cost.
    print(f"  Binning shifted spikes ...")
    bin_sf_v_s, bin_ti_v_s, bin_tmi_v_s = make_bin_sf_shifted(spike_t_shifted)
    print(f"  Done. {len(bin_sf_v_s)} windows.")

    if len(train_idx_fixed) == 0 or len(test_idx_fixed) == 0:
        print(f"  [WARN] Empty train or test — skipping.")
        continue

    _batch_size_s = min(_args.batch_size, max(1, len(train_idx_fixed)))

    # symlink GMM / init_weight_matrix from real run
    for _src_name in [f"ckpt_{_VAR_KEY}_{SESSION}_gmm.pkl",
                      f"ckpt_{_VAR_KEY}_{SESSION}_init_weight_matrix.npy"]:
        _src = os.path.join(BASE_RESULTS_DIR, _src_name)
        _dst = os.path.join(_shuf_dir, _src_name)
        if os.path.exists(_src) and not os.path.exists(_dst):
            os.symlink(_src, _dst)

    # 3. decode_pipeline: shifted spikes + real (unshifted) behaviour
    # live loss monitors (same as real run)
    import threading as _threading
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
                _a.set_title(f'ADVI live — shuffle {_shuf_i}  {BEHAVIOR_VAR}', fontsize=9)
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
                # clip y-axis to avoid warm-up explosion
                _skip = max(1, len(_tr)//20)
                _yhi  = float(np.nanpercentile(_tr[_skip:], 99)) if len(_tr) > _skip else float(_tr.max())
                _ylo  = float(np.nanmin(_tr))
                _f, _a = _plt.subplots(figsize=(10, 3))
                _a.plot(_tr, lw=0.8, color='steelblue', label='Train NLL')
                _a.set_ylim(_ylo - abs(_ylo)*0.05, _yhi + abs(_yhi)*0.05)
                _a.set_title(f'GLM live — shuffle {_shuf_i}  {BEHAVIOR_VAR}', fontsize=9)
                _a.legend(fontsize=8); _a.grid(alpha=0.25)
                _plt.tight_layout(); _plt.savefig(plot, dpi=100); _plt.close(_f)
            except Exception: pass

    _stop_s = _threading.Event()
    _threading.Thread(target=_advi_mon, args=(_advi_ckpt_s,_advi_live_s,_stop_s), daemon=True).start()
    _threading.Thread(target=_glm_mon,  args=(_glm_ckpt_s, _glm_live_s, _stop_s), daemon=True).start()

    weight_matrix_s = decode_pipeline(
        data_loader,
        bin_sf_v_s,
        bin_ti_v_s,
        bin_tmi_v_s,
        bin_behaviors    = bin_beh_v_real,   # ← real, unshifted behaviour
        behavior_type    = "continuous",
        train            = train_idx_fixed,   # ← same split every shuffle
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

    # 4. sliding_window_decoder: predicts from shifted test spikes vs real behaviour
    _n_wm_s = weight_matrix_s.shape[0]
    _, obs_s, pred_s, metrics_s = sliding_window_decoder(
        weight_matrix_s, bin_beh_v_real[:_n_wm_s],
        train_idx_fixed, test_idx_fixed,
        behavior_type="continuous", verbose=False,
    )

    # obs_s comes from bin_beh_v_real (unshifted) — r²/r is already vs real behaviour
    _t_elapsed = (datetime.datetime.now() - _t_start).total_seconds()
    print(f"\n  Done in {_t_elapsed/60:.1f} min  |  "
          f"R²={metrics_s['r2']:+.4f}  r={metrics_s['corr']:.4f}")

    # 5. save
    np.savez_compressed(
        _shuf_done,
        obs_test      = obs_s,
        pred_test     = pred_s,
        train_idx     = train_idx_fixed,
        test_idx      = test_idx_fixed,
        weight_matrix = weight_matrix_s,
        shift_sec     = np.float64(_shift_sec),
        shuffle_index = np.int32(_shuf_i),
        r2            = np.float64(metrics_s['r2']),
        r             = np.float64(metrics_s['corr']),
        T_TRIAL       = np.float32(T_TRIAL),
        N_T_BINS      = np.int32(N_T_BINS),
    )
    _params_s = {
        "shuffle_index":        _shuf_i,
        "shift_sec":            float(_shift_sec),
        "session":              SESSION,
        "behavior_var":         BEHAVIOR_VAR,
        "shuffle_type":         "neural_circular_shift",
        "T_TRIAL_s":            float(T_TRIAL),
        "N_T_BINS":             int(N_T_BINS),
        "bin_size_ms":          float(BIN_SIZE_MS),
        "selected_trials_meta": _sel_meta,
        "n_valid":              int(len(valid_trials_real)),
        "n_train":              int(len(train_idx_fixed)),
        "n_test":               int(len(test_idx_fixed)),
        "r2":                   float(metrics_s['r2']),
        "r":                    float(metrics_s['corr']),
        "elapsed_sec":          float(_t_elapsed),
        "note":                 "r2/r vs REAL (unshifted) behaviour — directly comparable to real run",
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

# ── final summary ─────────────────────────────────────────────────────────────
if _all_r2:
    print(f"\n{'='*65}")
    print(f"  NEURAL SHUFFLE SUMMARY  ({len(_all_r2)} shuffles)")
    print(f"  R²: mean={np.mean(_all_r2):+.4f}  std={np.std(_all_r2):.4f}  "
          f"range=[{np.min(_all_r2):+.4f}, {np.max(_all_r2):+.4f}]")
    print(f"  r : mean={np.mean(_all_r):+.4f}  std={np.std(_all_r):.4f}  "
          f"range=[{np.min(_all_r):+.4f}, {np.max(_all_r):+.4f}]")
    print(f"  Note: r²/r are vs REAL behaviour — directly comparable to real run metrics")
    print(f"{'='*65}")