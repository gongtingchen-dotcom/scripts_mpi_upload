"""
analyse_shuffles.py  —  neural circular shuffle diagnostics
────────────────────────────────────────────────────────────
Usage:
    python analyse_shuffles.py \
        --shuffle_dir   <path_to_*_neural_shuffle/>  \
        --real_r2 0.095  --real_r 0.437  \
        [--behavior_var PC1]  [--ap_fs 30000]  [--ap_t_offset 0]

Outputs into shuffle_dir:
    shuffle_null_r2_r_{BEH}.png
    shuffle_decode_traces_{BEH}.png          decode traces: real run row + selected shuffle rows
    shuffle_neural_comparison_{BEH}.png      spike scatter + weight-matrix time-profiles
    shuffle_analysis_summary_{BEH}.txt
"""

import os, sys, json, pickle, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import pearsonr

plt.rcParams.update({
    'figure.dpi': 130,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 10,
})

# ── ARGS ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--shuffle_dir',
    default='/nexus/posix0/MBR-neuralsystems/ephys/GLC6962/GLC6962_g_g0/density_decoding/pc8_100ms_5s_interleaved_zscoreboth_embargo0_selected-similar_trials_PC8_peak_dur5s_chunksize10_neural_shuffle_all')
ap.add_argument('--real_results_dir', 
    default='/nexus/posix0/MBR-neuralsystems/ephys/GLC6962/GLC6962_g_g0/density_decoding/pc8_100ms_5s_interleaved_zscoreboth_embargo0_selected-similar_trials_PC8_peak_dur5s_chunksize10',
    help='Path to the real (non-shuffled) run results directory. '
         'If omitted, inferred by stripping the _neural_shuffle / '
         '_neural_scramble / _behavior_scramble suffix from shuffle_dir. '
         'r² and r are auto-loaded from the decode_params JSON in this dir.')
ap.add_argument('--real_r2',  type=float, default=None,
    help='R² of the real run. Auto-loaded from real_results_dir if omitted.')
ap.add_argument('--real_r',   type=float, default=None,
    help='Pearson r of the real run. Auto-loaded from real_results_dir if omitted.')
ap.add_argument('--behavior_var', default=None)
ap.add_argument('--ap_fs',       type=float, default=None,
    help='AP sampling rate Hz. Auto-loaded from preprocessed/ if omitted.')
ap.add_argument('--ap_t_offset', type=float, default=None,
    help='AP t_start offset (s). Auto-loaded from preprocessed/ if omitted.')
ap.add_argument('--n_trials_show', type=int, default=5,
    help='Number of randomly selected test trials to show in trace/scatter plots')
ap.add_argument('--n_train_show',  type=int, default=8,
    help='Number of training trials for weight-matrix template heatmap')
ap.add_argument('--seed', type=int, default=0,
    help='RNG seed for selecting which test trials to show')
args = ap.parse_args()

SHUFFLE_DIR = Path(args.shuffle_dir)
assert SHUFFLE_DIR.exists(), f"Not found: {SHUFFLE_DIR}"
_s = str(SHUFFLE_DIR)

# ── detect analysis mode from directory name ──────────────────────────────────
# 'scramble' → bin-scramble run;  'shuffle' → circular-shift run
if 'scramble' in Path(_s).name:
    ANALYSIS_MODE = 'scramble'
    # infer real dir by stripping the scramble suffix
    for _sfx in ('_neural_scramble', '_behavior_scramble'):
        if _s.endswith(_sfx):
            _inferred_real = Path(_s[:-len(_sfx)])
            break
    else:
        _inferred_real = None
else:
    ANALYSIS_MODE = 'shuffle'
    _inferred_real = Path(_s[:-len('_neural_shuffle')]) if _s.endswith('_neural_shuffle') else None

REAL_DIR = Path(args.real_results_dir) if args.real_results_dir else _inferred_real
if REAL_DIR and not REAL_DIR.exists():
    print(f"[WARN] real_results_dir not found: {REAL_DIR}"); REAL_DIR = None
print(f"[MODE] {ANALYSIS_MODE.upper()}  |  SHUFFLE_DIR={SHUFFLE_DIR.name}")
if REAL_DIR: print(f"[REAL] real_results_dir = {REAL_DIR}")

# ── auto-load real_r2 / real_r from decode_params JSON ───────────────────────
def _load_real_metrics(real_dir, beh_var):
    """
    Try to load r² and r from the real run's decode_params JSON.
    Falls back to the decode_results txt file.
    Returns (r2, r) or (None, None).
    """
    if real_dir is None: return None, None
    real_dir = Path(real_dir)
    _var_key_m = (beh_var or '').lower().replace(' ', '_')

    # 1. Try decode_params JSON (preferred — structured)
    _json_files = list(real_dir.glob(f'decode_params_{_var_key_m}_*.json'))
    if not _json_files: _json_files = list(real_dir.glob('decode_params_*.json'))
    if _json_files:
        try:
            with open(_json_files[0]) as _jf:
                _jp = json.load(_jf)
            # results section: {BEH_VAR: {r2: ..., corr: ...}}
            _results = _jp.get('results', {})
            # try exact key first, then any key in results
            _rk = beh_var if beh_var in _results else (list(_results.keys())[0] if _results else None)
            if _rk:
                _r2  = float(_results[_rk].get('r2',   _results[_rk].get('R2',  float('nan'))))
                _r   = float(_results[_rk].get('corr', _results[_rk].get('r',   float('nan'))))
                if not (np.isnan(_r2) and np.isnan(_r)):
                    print(f"[REAL METRICS] Loaded from {_json_files[0].name}: "
                          f"R²={_r2:+.4f}  r={_r:.4f}")
                    return _r2, _r
        except Exception as _e:
            print(f"  [WARN] decode_params JSON parse: {_e}")

    # 2. Fallback: parse decode_results txt
    _txt_files = list(real_dir.glob(f'decode_results_{_var_key_m}_*.txt'))
    if not _txt_files: _txt_files = list(real_dir.glob('decode_results_*.txt'))
    if _txt_files:
        try:
            _r2, _r = None, None
            with open(_txt_files[0]) as _tf:
                for _line in _tf:
                    if 'R²' in _line and ':' in _line:
                        _r2 = float(_line.split(':')[1].strip())
                    if _line.strip().startswith('r ') and ':' in _line:
                        _r = float(_line.split(':')[1].strip())
            if _r2 is not None and _r is not None:
                print(f"[REAL METRICS] Loaded from {_txt_files[0].name}: "
                      f"R²={_r2:+.4f}  r={_r:.4f}")
                return _r2, _r
        except Exception as _e:
            print(f"  [WARN] decode_results txt parse: {_e}")

    print("[WARN] Could not auto-load real run R²/r — pass --real_r2 and --real_r manually")
    return None, None

# load before behavior_var is finalised — may be None here, that's ok
# we re-run after BEH_VAR is set from the scan
_real_r2_auto, _real_r_auto = None, None

# ── auto-load ap_fs / ap_t_offset from SpikeInterface recording ───────────────
def _load_ap_params(real_dir):
    """Load ap_fs and ap_t_offset from preprocessed/ next to density_decoding/."""
    if real_dir is None: return None, None
    preproc_dir = Path(real_dir).parent.parent / 'preprocessed'
    if not preproc_dir.exists():
        print(f"[WARN] preprocessed/ not found at {preproc_dir}"); return None, None
    try:
        from spikeinterface import NumpyRecording
        rec = NumpyRecording.load(str(preproc_dir))
        _fs  = float(rec.get_sampling_frequency())
        _t0  = float(rec._recording_segments[0].t_start)
        print(f"[AP] ap_fs={_fs:.2f} Hz  ap_t_offset={_t0:.4f} s  "
              f"(from {preproc_dir})")
        return _fs, _t0
    except Exception as e:
        print(f"[WARN] Could not load SpikeInterface recording: {e}")
        return None, None

if args.ap_fs is None or args.ap_t_offset is None:
    _fs_auto, _t0_auto = _load_ap_params(REAL_DIR)
    if args.ap_fs       is None: args.ap_fs       = _fs_auto  or 30000.0
    if args.ap_t_offset is None: args.ap_t_offset = _t0_auto  or 0.0
print(f"[AP] Using ap_fs={args.ap_fs:.2f}  ap_t_offset={args.ap_t_offset:.4f}")

RNG = np.random.default_rng(args.seed)

# ── SCAN SHUFFLES ─────────────────────────────────────────────────────────────
print(f"Scanning {SHUFFLE_DIR} ...")
records = []
for item in sorted(SHUFFLE_DIR.iterdir()):
    if not item.is_dir() or not item.name.startswith('shuffle_'): continue
    pf = list(item.glob('params_*.json'))
    df_files = list(item.glob('decode_results_*.npz'))
    if not pf or not df_files: continue
    try:
        with open(pf[0]) as f: p = json.load(f)
        if args.behavior_var is None and 'behavior_var' in p:
            args.behavior_var = p['behavior_var']
        records.append({
            'shuffle_index':  int(p['shuffle_index']),
            # shuffle: has shift_sec; scramble: has scramble_perm/scramble_mode
            'shift_sec':      float(p['shift_sec']) if 'shift_sec' in p else 0.0,
            'scramble_mode':  str(p.get('scramble_mode', '')),
            'scramble_perm':  p.get('scramble_perm', []),
            'r2':             float(p['r2']),
            'r':              float(p['r']),
            'n_train':        int(p.get('n_train', 0)),
            'n_test':         int(p.get('n_test',  0)),
            'T_TRIAL':        float(p.get('T_TRIAL_s', p.get('T_TRIAL', 0))),
            'N_T_BINS':       int(p.get('N_T_BINS', 0)),
            'decode_npz':     str(df_files[0]),
            'folder':         str(item),
        })
    except Exception as e:
        print(f"  [WARN] {item.name}: {e}")

if not records:
    print("No completed shuffles found."); sys.exit(0)

df = pd.DataFrame(records).sort_values('shuffle_index').reset_index(drop=True)
BEH_VAR = args.behavior_var or 'PC'
_var_key = BEH_VAR.lower().replace(' ', '_')

# infer SCRAMBLE_MODE for scramble runs
SCRAMBLE_MODE = (df['scramble_mode'].iloc[0]
                 if ANALYSIS_MODE == 'scramble' and df['scramble_mode'].iloc[0]
                 else '')
_run_noun = 'scrambles' if ANALYSIS_MODE == 'scramble' else 'shuffles'
print(f"Found {len(df)} {_run_noun}"
      + (f"  mode={SCRAMBLE_MODE}" if SCRAMBLE_MODE else
         f"  shift {df.shift_sec.min():.0f}–{df.shift_sec.max():.0f}s"))
print(f"  R²: mean={df.r2.mean():+.4f}  std={df.r2.std():.4f}  "
      f"[{df.r2.min():+.4f}, {df.r2.max():+.4f}]")
print(f"  r : mean={df.r.mean():+.4f}  std={df.r.std():.4f}  "
      f"[{df.r.min():+.4f}, {df.r.max():+.4f}]")

idx_best_r   = df['r'].idxmax();  idx_worst_r  = df['r'].idxmin()
idx_best_r2  = df['r2'].idxmax(); idx_worst_r2 = df['r2'].idxmin()

for tag, i in [('Best r',idx_best_r),('Worst r',idx_worst_r),
               ('Best R²',idx_best_r2),('Worst R²',idx_worst_r2)]:
    ro = df.loc[i]
    _extra = (f"  shift={ro.shift_sec:.1f}s" if ANALYSIS_MODE == 'shuffle'
              else f"  mode={ro.scramble_mode}")
    print(f"  {tag:8s}: {_run_noun[:-1]} {int(ro.shuffle_index):3d}  "
          f"r={ro.r:+.4f}  R²={ro.r2:+.4f}{_extra}")

# selected shuffles: 2 best R² + 2 worst R² (deduplicated by shuffle index)
_best_order = df.sort_values('r2', ascending=False).index.tolist()
_worst_order = df.sort_values('r2', ascending=True).index.tolist()

_seen = set()
SELECTED_SHUFFLES = []   # list of (label, colour, df_idx)

def _add_selected(order, label_prefix, colors, max_n=2):
    added = 0
    for di in order:
        if added >= max_n:
            break
        si = int(df.loc[di, 'shuffle_index'])
        if si in _seen:
            continue
        _seen.add(si)
        clr = colors[min(added, len(colors) - 1)]
        SELECTED_SHUFFLES.append((f'{label_prefix} {added + 1}', clr, di))
        added += 1

_add_selected(_best_order, 'best R²', ['tab:green', 'mediumseagreen'], max_n=2)
_add_selected(_worst_order, 'worst R²', ['tab:red', 'indianred'], max_n=2)

print("[SELECTED] Runs for detailed plots:")
for lbl, _, di in SELECTED_SHUFFLES:
    ro = df.loc[di]
    _extra = (f"shift={ro.shift_sec:.1f}s" if ANALYSIS_MODE == 'shuffle'
              else f"mode={ro.scramble_mode}")
    print(f"  {lbl:10s}: {_run_noun[:-1]} {int(ro.shuffle_index):3d}  "
          f"R²={ro.r2:+.4f}  r={ro.r:+.4f}  {_extra}")

# ── auto-load real r2/r now that BEH_VAR is known ────────────────────────────
if args.real_r2 is None or args.real_r is None:
    _real_r2_auto, _real_r_auto = _load_real_metrics(REAL_DIR, BEH_VAR)
    if args.real_r2 is None: args.real_r2 = _real_r2_auto
    if args.real_r  is None: args.real_r  = _real_r_auto

if args.real_r2 is not None:
    print(f"[REAL METRICS] Using R²={args.real_r2:+.4f}  r={args.real_r:.4f}")
else:
    print("[REAL METRICS] No real run R²/r — null plot will show no real-run line")

# ══════════════════════════════════════════════════════════════════════════════
# ── 1. NULL DISTRIBUTION ──────────────────────────════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
_null_title_detail = (f"shift {df.shift_sec.min():.0f}–{df.shift_sec.max():.0f}s"
                      if ANALYSIS_MODE == 'shuffle'
                      else f"mode={SCRAMBLE_MODE}")
fig.suptitle(f"Null distribution — {BEH_VAR}  ({ANALYSIS_MODE})\n"
             f"{len(df)} {_run_noun} | {_null_title_detail}",
             fontsize=11)
for ax, col, label, real_val in [
    (axes[0], 'r2', 'R²',        args.real_r2),
    (axes[1], 'r',  'Pearson r', args.real_r),
]:
    vals = df[col].values
    ax.hist(vals, bins=min(20, len(df)//2+1),
            color='steelblue', edgecolor='white', alpha=0.8)
    for pct, ls in [(5,'--'),(50,'-'),(95,'--')]:
        v = np.percentile(vals, pct)
        ax.axvline(v, color='orange', lw=1.2, ls=ls, label=f'P{pct}={v:+.3f}')
    ax.axvline(0, color='black', lw=0.8, ls=':', alpha=0.5)
    if real_val is not None:
        ax.axvline(real_val, color='crimson', lw=2.5, label=f'Real={real_val:+.4f}')
        p_val = (vals >= real_val).mean()
        ax.text(real_val, ax.get_ylim()[1]*0.85,
                f' p={p_val:.3f}', color='crimson', fontsize=9, va='top')
    ax.axvline(vals.max(), color='darkgreen', lw=1.2, ls=':', label=f'Best={vals.max():+.3f}')
    ax.axvline(vals.min(), color='darkred',   lw=1.2, ls=':', label=f'Worst={vals.min():+.3f}')
    ax.set_xlabel(label); ax.set_ylabel('Count')
    ax.set_title(f'Null — {label}')
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.15)
plt.tight_layout()
_p = SHUFFLE_DIR / f'{ANALYSIS_MODE}_null_r2_r_{BEH_VAR}.png'
plt.savefig(_p, dpi=130, bbox_inches='tight'); plt.close()
print(f"\nSaved → {_p}")

# ══════════════════════════════════════════════════════════════════════════════
# ── SHARED: load real run decode_results ─────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
real_d = None
real_npz_path = None
if REAL_DIR:
    _rnpz = list(REAL_DIR.glob(f'decode_results_{_var_key}_*.npz'))
    if not _rnpz: _rnpz = list(REAL_DIR.glob('decode_results_*.npz'))
    if _rnpz:
        real_npz_path = str(_rnpz[0])
        try:
            real_d = np.load(real_npz_path)
            print(f"[REAL] Loaded {_rnpz[0].name}")
            print(f"  keys: {list(real_d.files)}")
        except Exception as e:
            print(f"[WARN] real decode_results: {e}"); real_d = None

def _get_real_timing():
    """Return (trial_starts, trial_ends, T_TRIAL, N_T_BINS) from real run."""
    if real_d is None: return None, None, None, None
    ts = real_d.get('trial_starts', None)
    te = real_d.get('trial_ends',   None)
    T  = float(real_d.get('T_TRIAL',  df['T_TRIAL'].iloc[0]))
    N  = int(real_d.get('N_T_BINS',   df['N_T_BINS'].iloc[0]))
    return ts, te, T, N

def _parse_obs_pred(d_npz, row_ser):
    """Load obs/pred from an npz (real or shuffle), return (obs, pred, t_ax, test_idx) or None."""
    try:
        d  = np.load(d_npz) if isinstance(d_npz, str) else d_npz
    except Exception as e:
        print(f"  [WARN] {d_npz}: {e}"); return None
    if 'obs_test' not in d.files: return None
    obs      = d['obs_test']
    pred     = d['pred_test']
    test_idx = d['test_idx']
    T_TRIAL  = float(d.get('T_TRIAL',  row_ser.get('T_TRIAL', 5.0)))
    N_T_BINS = int(d.get('N_T_BINS',   row_ser.get('N_T_BINS', 50)))
    if N_T_BINS < 7: return None
    _hw = 7 // 2
    _nw = N_T_BINS - _hw + 1
    _np = _nw - _hw
    t_ax = np.arange(_hw, _nw) * (T_TRIAL / N_T_BINS)
    n_test = len(test_idx)
    if len(obs) != n_test * _np:
        if len(obs) % n_test == 0:
            _np   = len(obs) // n_test
            t_ax  = np.arange(_np) * (T_TRIAL / N_T_BINS)
        else:
            return None
    obs  = obs.reshape(n_test, _np)
    pred = pred.reshape(n_test, _np)
    return obs, pred, t_ax, test_idx


# ── load spike features cache ─────────────────────────────────────────────────
sf_cache = None
if REAL_DIR:
    _sf_files = list(REAL_DIR.glob('spike_features_*.npz'))
    if _sf_files:
        try:
            sf_cache = np.load(_sf_files[0], allow_pickle=False)
            print(f"[SF] Loaded {_sf_files[0].name}  "
                  f"({len(sf_cache['times_samples']):,} spikes)")
        except Exception as e:
            print(f"[WARN] spike_features: {e}")

# ── load GMM ─────────────────────────────────────────────────────────────────
gmm_obj = None
if REAL_DIR:
    _gf = list(REAL_DIR.glob(f'ckpt_{_var_key}_*_gmm.pkl'))
    if not _gf: _gf = list(REAL_DIR.glob('ckpt_*_gmm.pkl'))
    if _gf:
        try:
            with open(_gf[0], 'rb') as f: gmm_obj = pickle.load(f)
            print(f"[GMM] Loaded {_gf[0].name}  {gmm_obj.means_.shape[0]} components")
        except Exception as e:
            print(f"[WARN] GMM: {e}")

# ── get trial timing ─────────────────────────────────────────────────────────
trial_starts_all, trial_ends_all, T_TRIAL_ref, N_T_BINS_ref = _get_real_timing()
if trial_starts_all is None and real_d is None:
    # fall back to shuffle npz
    _tmp_npz = np.load(df.iloc[0]['decode_npz'])
    trial_starts_all = _tmp_npz.get('trial_starts', None)
    trial_ends_all   = _tmp_npz.get('trial_ends',   None)
    T_TRIAL_ref      = float(_tmp_npz.get('T_TRIAL',  df['T_TRIAL'].iloc[0]))
    N_T_BINS_ref     = int(_tmp_npz.get('N_T_BINS',   df['N_T_BINS'].iloc[0]))

# get test_idx from real run (same for all shuffles)
_test_idx_ref = real_d['test_idx'] if real_d is not None else \
                np.load(df.iloc[0]['decode_npz'])['test_idx']

# determine which test trial positions to show — first N, same as trace plot
_n_show_t      = min(args.n_trials_show, len(_test_idx_ref))
_show_test_pos = np.arange(_n_show_t)

def _circ_shift_spikes(spike_t, shift_sec, t_ov_start, t_ov_end):
    """Reconstruct shifted spike times using same logic as shuffle script."""
    dur     = t_ov_end - t_ov_start
    mask    = (spike_t >= t_ov_start) & (spike_t < t_ov_end)
    shifted = spike_t.copy()
    shifted[mask] = t_ov_start + (spike_t[mask] - t_ov_start + shift_sec) % dur
    return shifted

def _reconstruct_scrambled_spike_times(spike_t_all, spike_z_all,
                                        trial_starts, trial_ends,
                                        test_idx, test_pos,
                                        perm, n_t, T_trial):
    """
    Reconstruct what the decoder saw for a scrambled trial.
    Bins were permuted (not spike times), so we take the real spikes
    in the trial window, assign each to its original bin, then
    remap to the new bin position according to perm.
    """
    _vp  = test_idx[test_pos]
    t0   = float(trial_starts[_vp]); t1 = float(trial_ends[_vp]); dur = t1 - t0
    mask = (spike_t_all >= t0) & (spike_t_all < t1)
    t_rel = spike_t_all[mask] - t0
    z_spk = spike_z_all[mask]
    if len(t_rel) == 0 or perm is None or len(perm) == 0:
        return t_rel, z_spk
    bin_s    = dur / n_t
    orig_bin = np.floor(t_rel / bin_s).astype(int).clip(0, n_t - 1)
    # perm[new_t] = orig_t  → build inverse: orig_t → new_t
    inv_perm = np.empty_like(perm)
    for new_t, orig_t in enumerate(perm):
        inv_perm[orig_t] = new_t
    new_bin     = inv_perm[orig_bin]
    t_scrambled = (new_bin + (t_rel % bin_s) / bin_s) * bin_s
    return t_scrambled, z_spk

def _spikes_in_window(spike_t_sec, spike_z, t0, t1):
    """Return (times_relative, z) for spikes in [t0, t1)."""
    mask = (spike_t_sec >= t0) & (spike_t_sec < t1)
    return spike_t_sec[mask] - t0, spike_z[mask]

# build T_OV_START / T_OV_END once from spike cache range
if sf_cache is not None:
    _spike_t_all = sf_cache['times_samples'] / args.ap_fs + args.ap_t_offset
    _z_all       = sf_cache['point_source_localizations'][:, 1]
    if trial_starts_all is not None:
        T_OV_START_ref = max(trial_starts_all.min() - 10, _spike_t_all.min())
    else:
        T_OV_START_ref = _spike_t_all.min()
    T_OV_END_ref = _spike_t_all.max()
else:
    _spike_t_all = None; _z_all = None
    T_OV_START_ref = 0.0; T_OV_END_ref = 1e9


# ══════════════════════════════════════════════════════════════════════════════
# ── 2. DECODE TRACES + SPIKE SCATTER ─────────────────────────────────────────
#
# Layout: for each run (real + selected shuffles/scrambles):
#   Row A: behaviour traces  — observed (grey), decoded (colour),
#                              + beh fed to decoder (red dashed, if different)
#   Row B: spike scatter     — spikes fed to decoder (real or shifted/scrambled)
#                              coloured by run, with original in grey for comparison
#
# Columns: N_TRIALS_SHOW randomly selected test trials
# ══════════════════════════════════════════════════════════════════════════════
_dummy_row = {'T_TRIAL': df['T_TRIAL'].iloc[0], 'N_T_BINS': df['N_T_BINS'].iloc[0]}

real_tr = _parse_obs_pred(real_d, _dummy_row) if real_d is not None else None

# choose which test trials to show
if real_tr is not None:
    _n_test_avail = real_tr[0].shape[0]
elif SELECTED_SHUFFLES:
    _tmp = _parse_obs_pred(df.loc[SELECTED_SHUFFLES[0][2], 'decode_npz'],
                           df.loc[SELECTED_SHUFFLES[0][2]])
    _n_test_avail = _tmp[0].shape[0] if _tmp else args.n_trials_show
else:
    _n_test_avail = args.n_trials_show

_n_show    = min(args.n_trials_show, _n_test_avail)
# use first N test trials — deterministic, same across runs, easy to compare
_show_cols = np.arange(_n_show)

# ── print test trial time windows (useful for cross-run comparison) ─────────────
print(f"\n[TEST TRIALS] first {_n_show} of {_n_test_avail} test trials:")
if real_d is not None and 'trial_starts' in real_d.files:
    _ts_print = real_d['trial_starts']; _te_print = real_d['trial_ends']
    _ti_print = real_d['test_idx']
    for _j in range(_n_show):
        _vp = _ti_print[_j]
        print(f"  test {_j:2d}  valid_idx={int(_vp):3d}  "
              f"t={_ts_print[_vp]:.2f}–{_te_print[_vp]:.2f}s")
else:
    print("  (no trial_starts in real decode_results — run with updated selection script)")

# ── pre-load what we need for spike scatter and beh feed reconstruction ───────────
_tr_timing = _get_real_timing()   # (trial_starts, trial_ends, T_TRIAL_ref2, N_T_BINS_ref2)
_ts_all2, _te_all2, _T2, _N2 = _tr_timing
if _ts_all2 is None and real_d is not None:
    _ts_all2 = real_d.get('trial_starts', None)
    _te_all2 = real_d.get('trial_ends',   None)
    _T2  = float(real_d.get('T_TRIAL', df['T_TRIAL'].iloc[0]))
    _N2  = int(real_d.get('N_T_BINS',  df['N_T_BINS'].iloc[0]))
_test_idx_for_tr = (real_d['test_idx'] if real_d is not None
                    else np.load(df.iloc[0]['decode_npz'])['test_idx'])

# load spike cache (already done above but guard in case)


def _spikes_for_trial(test_pos, test_idx, shift_s, perm):
    """
    Return (t_real, z_real, t_feed, z_feed) for a given test trial position.
    t_real/z_real = original unmodified spikes.
    t_feed/z_feed = what was fed to the decoder (shifted or bin-scrambled).
    All times are relative to trial start.
    """
    if _spike_t_all is None or _ts_all2 is None: return None, None, None, None
    _vp = test_idx[test_pos]
    t0  = float(_ts_all2[_vp]); t1 = float(_te_all2[_vp]); dur = t1 - t0

    # real spikes
    mask_r   = (_spike_t_all >= t0) & (_spike_t_all < t1)
    t_real   = _spike_t_all[mask_r] - t0
    z_real   = _z_all[mask_r]

    # feed spikes
    if ANALYSIS_MODE == 'shuffle' and shift_s > 0:
        _spt_shifted = _circ_shift_spikes(_spike_t_all, shift_s, T_OV_START_ref, T_OV_END_ref)
        mask_f = (_spt_shifted >= t0) & (_spt_shifted < t1)
        t_feed = _spt_shifted[mask_f] - t0
        z_feed = _z_all[mask_f]
    elif ANALYSIS_MODE == 'scramble' and perm is not None and SCRAMBLE_MODE == 'neural':
        t_feed, z_feed = _reconstruct_scrambled_spike_times(
            _spike_t_all, _z_all, _ts_all2, _te_all2,
            test_idx, test_pos, perm, _N2 or 50, _T2 or dur)
    else:
        # behaviour scramble or real run: neural unchanged
        t_feed, z_feed = t_real, z_real

    return t_real, z_real, t_feed, z_feed

# ── figure: 2 rows per run (beh trace + spike scatter), n_show columns ────────
_n_run_rows = 1 + len(SELECTED_SHUFFLES)   # real + selected
_n_fig_rows = _n_run_rows * 2              # 2 sub-rows per run
_row_heights = []
for _ in range(_n_run_rows):
    _row_heights += [2.2, 1.8]             # beh trace taller than spike scatter

fig_tr = plt.figure(figsize=(_n_show * 3.8, sum(_row_heights) + 0.6 * _n_run_rows))
gs_tr  = gridspec.GridSpec(
    _n_fig_rows, _n_show,
    figure=fig_tr,
    height_ratios=_row_heights,
    hspace=0.08, wspace=0.3,
    top=0.94, bottom=0.04, left=0.08, right=0.97,
)
fig_tr.suptitle(
    f"Decode traces + spike scatter — {BEH_VAR}  ({ANALYSIS_MODE})\n"
    f"Top row: observed vs decoded beh (test trials, unmodified)  |  "
    f"Bottom row: spikes fed to decoder (training trials)  |  "
    f"grey=real, colour=fed-to-decoder",
    fontsize=8,
)

# build list of (label, colour, obs_pred_result, shift_s, perm) per run
_tr_runs = []

# real run
_real_overall = ''
if args.real_r2 is not None and args.real_r is not None:
    _real_overall = f'\nR²={args.real_r2:+.3f}  r={args.real_r:+.3f}  shift=0s'
_tr_runs.append((f'Real run{_real_overall}', 'tab:blue', real_tr, 0.0, None))

# selected shuffles/scrambles
for lbl, clr, di in SELECTED_SHUFFLES:
    ro = df.loc[di]
    tr = _parse_obs_pred(ro['decode_npz'], ro)
    _perm_run = None
    if ANALYSIS_MODE == 'scramble':
        _perm_run = np.array(ro['scramble_perm'], dtype=int) if ro['scramble_perm'] else None
        if _perm_run is None:
            try:
                _d_tmp = np.load(ro['decode_npz'])
                if 'scramble_perm' in _d_tmp.files:
                    _perm_run = np.array(list(_d_tmp['scramble_perm']), dtype=int)
            except Exception: pass
    _ss = float(ro['shift_sec'])
    _tr_runs.append((
        f'{lbl}  ({_run_noun[:-1]} {int(ro.shuffle_index)})\n'
        f'R²={ro.r2:+.3f}  r={ro.r:+.3f}  '
        f'{("shift=" + format(_ss, ".0f") + "s") if ANALYSIS_MODE == "shuffle" else ("mode=" + str(ro.scramble_mode))}',
        clr, tr, _ss, _perm_run))

# For the spike scatter we want to show TRAINING trial spikes
# (not test trials) because that's what was actually fed to the decoder.
# We show the first N_TRAIN_SHOW train trials.
# For the real run: real spikes (black).
# For shuffle: circularly shifted spikes (coloured) with real in grey behind.
# For behaviour scramble: spikes unchanged (neural not touched), just show real.
# For neural scramble: bin-scrambled spike positions (coloured) with real in grey.

_n_train_scatter = min(args.n_train_show, len(_test_idx_for_tr))  # reuse n_train_show
# get train_idx from real decode_results (same for all runs)
_train_idx_for_scatter = (real_d['train_idx'] if real_d is not None
                          else np.load(df.iloc[0]['decode_npz'])['train_idx'])
_n_train_scatter = min(args.n_train_show, len(_train_idx_for_scatter))
_scatter_train_positions = np.arange(_n_train_scatter)  # first N training trials

for run_i, (run_lbl, run_clr, tr_result, shift_s, perm) in enumerate(_tr_runs):
    _base_row = run_i * 2   # beh trace row
    _scat_row = run_i * 2 + 1   # spike scatter row
    is_real   = (run_i == 0)

    # does this run change the neural data?
    _neural_differs = (
        (ANALYSIS_MODE == 'shuffle' and shift_s > 0) or
        (ANALYSIS_MODE == 'scramble' and perm is not None and SCRAMBLE_MODE == 'neural')
    )

    for j, col_j in enumerate(_show_cols):
        # ── top: beh traces (observed vs decoded) ────────────────────────────
        # Test trials are NEVER scrambled/shuffled — they are always the real
        # unmodified data. So just show observed (grey) vs decoded (colour).
        ax_beh = fig_tr.add_subplot(gs_tr[_base_row, j])

        if tr_result is not None:
            obs, pred, t_ax, test_idx_run = tr_result
            if col_j < len(obs):
                ax_beh.plot(t_ax, obs[col_j], color='darkgray', lw=1.3,
                            label='observed' if j == 0 else None)
                ax_beh.plot(t_ax, pred[col_j], color=run_clr, lw=1.3,
                            label='decoded'  if j == 0 else None)
                _ss_r = np.sum((obs[col_j]-pred[col_j])**2)
                _ss_t = np.sum((obs[col_j]-obs[col_j].mean())**2)
                _r2_j = 1 - _ss_r/_ss_t if _ss_t > 1e-12 else float('nan')
                ax_beh.set_title(f'test {col_j}  R²={_r2_j:+.3f}', fontsize=7)
        else:
            ax_beh.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax_beh.transAxes, fontsize=8, color='gray')
            ax_beh.set_title(f'test {col_j}', fontsize=7)

        ax_beh.tick_params(labelsize=5); ax_beh.grid(True, alpha=0.15)
        if j == 0:
            ax_beh.set_ylabel(f'{run_lbl}\n{BEH_VAR}', fontsize=6)
            if tr_result is not None:
                ax_beh.legend(fontsize=4.5, loc='upper right')
        plt.setp(ax_beh.get_xticklabels(), visible=False)
        ax_beh.set_xlabel('')

        # ── bottom: spike scatter for TRAINING trial col_j ───────────────────
        # Training trials DO have their neural data scrambled/shifted.
        # We show training trial col_j (same column position as the beh trace above
        # so the column is self-consistent in terms of position, though the
        # actual trial window differs between train and test).
        # Grey = real spikes always; colour overlay = what was fed to decoder.
        ax_spk = fig_tr.add_subplot(gs_tr[_scat_row, j])

        # pick a training trial to show for this column
        _train_pos = _scatter_train_positions[j] if j < len(_scatter_train_positions) else 0
        t_real_tr, z_real_tr, t_feed_tr, z_feed_tr = _spikes_for_trial(
            _train_pos, _train_idx_for_scatter, shift_s, perm)

        if t_real_tr is not None:
            # real spikes always in grey
            ax_spk.scatter(t_real_tr, z_real_tr, s=0.8,
                           color='black' if is_real else 'lightgray',
                           alpha=0.15 if is_real else 0.4, rasterized=True,
                           label='real spikes' if (j == 0 and not is_real) else None)
            if _neural_differs and t_feed_tr is not None:
                # overlay the data actually fed to the decoder
                ax_spk.scatter(t_feed_tr, z_feed_tr, s=0.8, color=run_clr,
                               alpha=0.25, rasterized=True,
                               label='fed to decoder' if j == 0 else None)

        if _z_all is not None:
            ax_spk.set_ylim(_z_all.min()*0.95, _z_all.max()*1.05)
        ax_spk.set_xlabel('Time (s)', fontsize=6)
        ax_spk.tick_params(labelsize=5); ax_spk.grid(True, alpha=0.12)
        if j == 0:
            ax_spk.set_ylabel('Depth (µm)', fontsize=6)
            if _neural_differs:
                ax_spk.legend(fontsize=4.5, loc='upper right')
        # label which training trial is shown
        _train_vp = _train_idx_for_scatter[_train_pos]
        if _ts_all2 is not None and _train_vp < len(_ts_all2):
            ax_spk.set_title(f'train {_train_pos}  '
                             f't={_ts_all2[_train_vp]:.0f}s', fontsize=6)

plt.savefig(SHUFFLE_DIR / f'{ANALYSIS_MODE}_decode_traces_{BEH_VAR}.png',
            dpi=130, bbox_inches='tight')
plt.close()
_p_tr = SHUFFLE_DIR / f'{ANALYSIS_MODE}_decode_traces_{BEH_VAR}.png'
print(f"Saved → {_p_tr}")

# ══════════════════════════════════════════════════════════════════════════════
# ── 3. NEURAL COMPARISON ──────────────────────────────────────────────────────
#
# For each of (real run, best-r shuffle, worst-r shuffle):
#
# Section A — SPIKE SCATTER  (one sub-panel per test trial)
#   x = time within trial window, y = z_corr (probe depth), dot per spike
#   real run spikes in grey; shifted spikes in colour
#   overlaid: GMM component centroids sized by weight_matrix value for that bin
#
# Section B — WEIGHT-MATRIX TIME PROFILES  (per test trial)
#   imshow: x = time bin, y = GMM component (sorted by depth)
#   show for each selected test trial (real & each shuffle side by side)
#
# Section C — TRAINING TEMPLATE
#   mean weight_matrix over all train trials → (n_comp, n_t)
#   "what the decoder learned to expect per time bin"
#
# ══════════════════════════════════════════════════════════════════════════════

# ── helpers for sorting components by depth ───────────────────────────────────
def _comp_sort_order(n_comp):
    if gmm_obj is not None and hasattr(gmm_obj, 'means_') and gmm_obj.means_.shape[1] >= 2:
        return np.argsort(gmm_obj.means_[:, 1])   # ascending z
    return np.arange(n_comp)

def _comp_depth_labels(sort_order):
    if gmm_obj is not None and hasattr(gmm_obj, 'means_') and gmm_obj.means_.shape[1] >= 2:
        return gmm_obj.means_[sort_order, 1]
    return sort_order.astype(float)

# ── load weight matrix from an npz ───────────────────────────────────────────
def _load_wm(npz_path_or_dict):
    d = np.load(npz_path_or_dict) if isinstance(npz_path_or_dict, str) else npz_path_or_dict
    if 'weight_matrix' not in d.files: return None, None, None
    wm       = d['weight_matrix']     # (n_valid, n_comp, n_t)
    train_i  = d['train_idx']
    test_i   = d['test_idx']
    return wm, train_i, test_i

# ── build FIGURE ──────────────────────────────────────────────────────────────
# Cases: real run + selected shuffles/scrambles
# Each case: (label, colour, decode_npz_or_dict, shift_sec_or_none, perm_or_none)
_cases = []
if real_d is not None:
    _real_detail = ''
    if args.real_r2 is not None and args.real_r is not None:
        _real_detail = f'\nshift=0s  R²={args.real_r2:+.3f}  r={args.real_r:+.3f}'
    _cases.append((f'Real run{_real_detail}', 'dimgray', real_d, 0.0, None))
for lbl, clr, di in SELECTED_SHUFFLES:
    ro = df.loc[di]
    if ANALYSIS_MODE == 'shuffle':
        _detail = f'shift={ro.shift_sec:.0f}s  R²={ro.r2:+.3f}  r={ro.r:+.3f}'
        _perm   = None
    else:
        _detail = f'mode={ro.scramble_mode}  R²={ro.r2:+.3f}  r={ro.r:+.3f}'
        _perm   = np.array(ro['scramble_perm'], dtype=int) if ro['scramble_perm'] else None
    _cases.append((
        f'{lbl}  ({_run_noun[:-1]} {int(ro.shuffle_index)})\n{_detail}',
        clr, ro['decode_npz'], float(ro['shift_sec']), _perm))

n_cases = len(_cases)

def _load_beta(folder_or_dir, var_key, session):
    """Load beta = U @ V from ckpt_*_post_params.npz in a directory."""
    p = Path(folder_or_dir)
    candidates = list(p.glob(f'ckpt_{var_key}_{session}_post_params.npz'))
    if not candidates:
        candidates = list(p.glob('ckpt_*_post_params.npz'))
    if not candidates:
        return None
    try:
        pp = np.load(candidates[0])
        beta = pp['U'] @ pp['V']   # (n_comp, n_t)
        return beta
    except Exception as e:
        print(f"  [WARN] beta load from {candidates[0]}: {e}")
        return None

def _infer_session(real_dir):
    """Infer SESSION string (e.g. GLC6962_g_g0) from real_dir path."""
    if real_dir is None: return ''
    for f in Path(real_dir).glob('ckpt_*_gmm.pkl'):
        # ckpt_{var_key}_{SESSION}_gmm.pkl
        stem = f.stem  # e.g. ckpt_pc1_GLC6962_g_g0_gmm
        if stem.endswith('_gmm'):
            inner = stem[len('ckpt_'):-len('_gmm')]  # pc1_GLC6962_g_g0
            parts = inner.split('_')
            for i, part in enumerate(parts):
                if part and part[0].isupper():
                    return '_'.join(parts[i:])
    return ''

_SESSION_str = _infer_session(REAL_DIR)

if n_cases == 0:
    print("[WARN] No cases to compare — skipping neural comparison plot")
else:
    # ── load all weight matrices and betas ─────────────────────────────────────
    all_case_data = []
    for lbl, clr, npz_or_dict, shift_sec, perm in _cases:
        wm, tr_i, te_i = _load_wm(npz_or_dict)
        # also try to load perm from npz (more reliable than from JSON list)
        if perm is None and ANALYSIS_MODE == 'scramble':
            try:
                _d_tmp = np.load(npz_or_dict) if isinstance(npz_or_dict, str) else npz_or_dict
                if 'scramble_perm' in _d_tmp.files:
                    perm = np.array(list(_d_tmp['scramble_perm']), dtype=int)
            except Exception:
                pass
        if shift_sec == 0.0 and REAL_DIR:
            beta = _load_beta(REAL_DIR, _var_key, _SESSION_str)
        else:
            _shuf_folder = Path(npz_or_dict).parent if isinstance(npz_or_dict, str) else None
            beta = _load_beta(_shuf_folder, _var_key, _SESSION_str) if _shuf_folder else None
        all_case_data.append({'lbl': lbl, 'clr': clr, 'wm': wm,
                               'train_i': tr_i, 'test_i': te_i,
                               'shift_sec': shift_sec, 'perm': perm, 'beta': beta})

    # ── figure layout ──────────────────────────────────────────────────────────
    # Columns: one per case + 1 decode-trace column
    # Rows:
    #   [A] _n_show_t spike-scatter rows
    #   [B] _n_show_t weight-heatmap rows
    #   [C] 1 training-template row
    #   [D] 1 beta row

    _n_cols       = n_cases + 1    # +1 for decode trace column
    _n_rows_A     = _n_show_t
    _n_rows_B     = _n_show_t
    _row_heights  = ([1.8]*_n_rows_A + [1.5]*_n_rows_B + [1.8] + [2.0])
    _n_rows_total = len(_row_heights)
    _fig_h = sum(_row_heights) + 0.5 * _n_rows_total
    _fig_w = _n_cols * 4.5

    fig_nc = plt.figure(figsize=(_fig_w, _fig_h))
    gs = gridspec.GridSpec(
        _n_rows_total, _n_cols,
        figure=fig_nc,
        height_ratios=_row_heights,
        hspace=0.55, wspace=0.35,
        top=0.96, bottom=0.03, left=0.06, right=0.97,
    )
    fig_nc.suptitle(
        f"Neural comparison — {BEH_VAR}  "
        f"({_n_show_t} test trials, R2-selected shuffles, depth-sorted GMM)"
        f"Section A: spike scatter  |  Section B: weight-matrix per test trial  "
        f"|  Section C: training template  |  Section D: beta (U@V)  "
        f"|  Col {n_cases+1}: decode traces",
        fontsize=8,
    )

    # ── also need obs/pred for decode trace column ─────────────────────────────
    _case_traces = []
    for cd in all_case_data:
        npz_src = cd['wm']   # wm already loaded; need raw npz for obs/pred
        # re-load obs/pred from the npz
        _npz_path = _cases[all_case_data.index(cd)][2]
        _tr = _parse_obs_pred(_npz_path, {'T_TRIAL': T_TRIAL_ref or 5.0,
                                           'N_T_BINS': N_T_BINS_ref or 50})
        _case_traces.append(_tr)

    for ci, cd in enumerate(all_case_data):
        wm      = cd['wm']
        train_i = cd['train_i']
        test_i  = cd['test_i']
        lbl     = cd['lbl']
        clr     = cd['clr']
        beta    = cd['beta']
        shift_s = cd['shift_sec']
        perm_s  = cd['perm']   # permutation array for scramble, None for shuffle

        if wm is None:
            continue

        n_comp = wm.shape[1]
        n_t    = wm.shape[2]
        sort_c = _comp_sort_order(n_comp)
        z_sorted = _comp_depth_labels(sort_c)
        _t_ax_sec = np.arange(n_t) * ((T_TRIAL_ref or 5.0) / (N_T_BINS_ref or n_t))

        # spike times to show: for shuffle = circularly shifted;
        # for scramble = original (bins were permuted, not spike times)
        if ANALYSIS_MODE == 'shuffle' and shift_s > 0 and _spike_t_all is not None:
            spike_t_case = _circ_shift_spikes(
                _spike_t_all, shift_s, T_OV_START_ref, T_OV_END_ref)
        else:
            spike_t_case = _spike_t_all

        # ── Section A: spike scatter ───────────────────────────────────────────
        for ti, test_pos in enumerate(_show_test_pos):
            ax = fig_nc.add_subplot(gs[ti, ci])

            if trial_starts_all is not None and test_i is not None:
                _vp  = test_i[test_pos]
                t0   = float(trial_starts_all[_vp])
                t1   = float(trial_ends_all[_vp])
                dur  = t1 - t0

                # compute R² for this test trial (obs vs pred)
                _tr_data = _case_traces[ci]
                _r2_str = ''
                if _tr_data is not None and test_pos < _tr_data[0].shape[0]:
                    _obs_j  = _tr_data[0][test_pos]
                    _pred_j = _tr_data[1][test_pos]
                    _ss_res = np.sum((_obs_j - _pred_j)**2)
                    _ss_tot = np.sum((_obs_j - _obs_j.mean())**2)
                    _r2_j   = 1 - _ss_res/_ss_tot if _ss_tot > 1e-12 else float('nan')
                    _r2_str = f'  R²={_r2_j:+.3f}'

                if spike_t_case is not None:
                    if ANALYSIS_MODE == 'scramble' and perm_s is not None and SCRAMBLE_MODE == 'neural':
                        # scramble: real spikes in grey underneath, scrambled on top
                        _t_real, _z_real = _spikes_in_window(spike_t_case, _z_all, t0, t1)
                        ax.scatter(_t_real, _z_real, s=1, color='lightgray',
                                   alpha=0.4, rasterized=True)
                        _t_scr, _z_scr = _reconstruct_scrambled_spike_times(
                            spike_t_case, _z_all, trial_starts_all, trial_ends_all,
                            test_i, test_pos, perm_s, n_t, T_TRIAL_ref or dur)
                        ax.scatter(_t_scr, _z_scr, s=1, color=clr,
                                   alpha=0.25, rasterized=True, label='scrambled')
                    else:
                        # shuffle or behaviour-scramble: just plot (shifted) spikes
                        _t_rel, _z_spk = _spikes_in_window(spike_t_case, _z_all, t0, t1)
                        ax.scatter(_t_rel, _z_spk,
                                   s=1, color=clr if shift_s > 0 else 'black',
                                   alpha=0.12, rasterized=True)

                    # GMM component centroids at right edge of trial window
                    if gmm_obj is not None and hasattr(gmm_obj, 'means_'):
                        _wm_t_mean = wm[_vp, :, :].mean(axis=1)  # (n_comp,)
                        _gmm_z = np.asarray(gmm_obj.means_[:, 1]).ravel()
                        # Guard against component-count mismatch between wm and GMM.
                        _n_plot = min(len(_wm_t_mean), len(_gmm_z))
                        if _n_plot > 0:
                            _wm_plot = _wm_t_mean[:_n_plot]
                            _z_plot = _gmm_z[:_n_plot]
                            _wmax = np.nanmax(_wm_plot) + 1e-9
                            _sizes_gm = (_wm_plot / _wmax * 80).clip(1, 80)
                            _x_centroids = np.full(_n_plot, dur * 1.06)
                            ax.scatter(_x_centroids, _z_plot,
                                       s=_sizes_gm, c=_wm_plot, cmap='hot_r',
                                       alpha=0.8, edgecolors='k', linewidths=0.3,
                                       zorder=5, clip_on=False)
                ax.set_xlim(-0.05 * dur, dur * 1.22)
                ax.set_ylim(_z_all.min() * 0.95 if _z_all is not None else 0,
                            _z_all.max() * 1.05 if _z_all is not None else 4000)
            else:
                ax.text(0.5, 0.5, 'no timing', ha='center', va='center',
                        transform=ax.transAxes, fontsize=7)
                _r2_str = ''

            ax.set_xlabel('Time in trial (s)', fontsize=6)
            ax.set_ylabel('Depth z (µm)', fontsize=6)
            ax.tick_params(labelsize=5); ax.grid(True, alpha=0.15)
            _sec_lbl = f'[A] test {test_pos}' if ti == 0 else f'test {test_pos}'
            _title   = f'{_sec_lbl}{_r2_str}'
            if ti == 0:
                _title = lbl.split('\n')[0] + '\n' + _title
            ax.set_title(_title, fontsize=6.5)

        # ── Section B: weight-matrix heatmap per test trial ────────────────────
        _wm_vmax = float(np.nanpercentile(wm[test_i], 97)) if test_i is not None else 3.0

        for ti, test_pos in enumerate(_show_test_pos):
            row = _n_rows_A + ti
            ax  = fig_nc.add_subplot(gs[row, ci])

            if test_i is not None:
                _vp = test_i[test_pos]
                _wm_sorted = wm[_vp][sort_c, :]   # (n_comp, n_t)
            else:
                _wm_sorted = np.zeros((n_comp, n_t))

            im = ax.imshow(_wm_sorted, aspect='auto', cmap='viridis',
                           vmin=0, vmax=_wm_vmax, origin='lower',
                           interpolation='nearest',
                           extent=[0, T_TRIAL_ref or 5.0, 0, n_comp])
            if ci == 0:
                ax.set_ylabel('Comp (depth ↑)', fontsize=6)
            ax.set_xlabel('Time (s)', fontsize=6)
            ax.tick_params(labelsize=5)
            _ttl2 = '[B] weight matrix' if ti == 0 else f'test {test_pos}'
            ax.set_title(_ttl2, fontsize=7)
            if ti == 0:
                plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8,
                             label='wt', fraction=0.04)
            _step = max(1, n_comp // 6)
            _ytk  = np.arange(0, n_comp, _step)
            ax.set_yticks(_ytk)
            ax.set_yticklabels([f'{z_sorted[i]:.0f}' for i in _ytk], fontsize=4)

        # ── Section C: training template ──────────────────────────────────────
        row_C = _n_rows_A + _n_rows_B
        ax_C  = fig_nc.add_subplot(gs[row_C, ci])
        _wm_tmpl = wm[train_i].mean(axis=0)[sort_c, :]   # (n_comp, n_t)
        _vmax_C  = float(np.nanpercentile(_wm_tmpl, 97))
        im_C = ax_C.imshow(_wm_tmpl, aspect='auto', cmap='viridis',
                           vmin=0, vmax=_vmax_C, origin='lower',
                           interpolation='nearest',
                           extent=[0, T_TRIAL_ref or 5.0, 0, n_comp])
        ax_C.set_title(f'[C] Training template ({len(train_i)} train trials)',
                       fontsize=7)
        ax_C.set_xlabel('Time (s)', fontsize=6)
        if ci == 0: ax_C.set_ylabel('Comp (depth ↑)', fontsize=6)
        ax_C.tick_params(labelsize=5)
        plt.colorbar(im_C, ax=ax_C, pad=0.02, shrink=0.8,
                     label='mean wt', fraction=0.04)
        _step = max(1, n_comp // 6)
        _ytk  = np.arange(0, n_comp, _step)
        ax_C.set_yticks(_ytk); ax_C.set_yticklabels(
            [f'{z_sorted[i]:.0f}' for i in _ytk], fontsize=4)

        # ── Section D: beta heatmap ────────────────────────────────────────────
        row_D = _n_rows_A + _n_rows_B + 1
        ax_D  = fig_nc.add_subplot(gs[row_D, ci])
        if beta is not None:
            _beta_sorted = beta[sort_c, :]   # (n_comp, n_t), depth-sorted
            _bmax = float(np.nanpercentile(np.abs(_beta_sorted), 99))
            im_D = ax_D.imshow(_beta_sorted, aspect='auto', cmap='RdBu_r',
                               vmin=-_bmax, vmax=_bmax, origin='lower',
                               interpolation='nearest',
                               extent=[0, T_TRIAL_ref or 5.0, 0, n_comp])
            ax_D.set_title('[D] Beta (U@V) — GLM tuning', fontsize=7)
            plt.colorbar(im_D, ax=ax_D, pad=0.02, shrink=0.8,
                         label='tuning coef', fraction=0.04)
        else:
            ax_D.text(0.5, 0.5, 'beta not found\nrun decode_pipeline first',
                      ha='center', va='center', transform=ax_D.transAxes,
                      fontsize=8, color='gray')
            ax_D.set_title('[D] Beta (U@V) — not found', fontsize=7)
        ax_D.set_xlabel('Time (s)', fontsize=6)
        if ci == 0: ax_D.set_ylabel('Comp (depth ↑)', fontsize=6)
        ax_D.tick_params(labelsize=5)
        _step = max(1, n_comp // 6)
        _ytk  = np.arange(0, n_comp, _step)
        ax_D.set_yticks(_ytk); ax_D.set_yticklabels(
            [f'{z_sorted[i]:.0f}' for i in _ytk], fontsize=4)

    # ── Column n_cases: decode traces (obs + all cases predicted) ─────────────
    col_tr = n_cases
    _tr_colours = ['tab:blue'] + [cd['clr'] for cd in all_case_data[1:]]
    _tr_labels  = ['Real run'] + [cd['lbl'].split('\n')[0] for cd in all_case_data[1:]]

    for ti, test_pos in enumerate(_show_test_pos):
        row_tr = ti   # decode traces go in rows 0..n_show-1 (section A rows)
        ax_tr  = fig_nc.add_subplot(gs[row_tr, col_tr])

        _plotted_obs = False
        for ci_tr, (tr_data, clr_tr, lbl_tr) in enumerate(
                zip(_case_traces, _tr_colours, _tr_labels)):
            if tr_data is None or test_pos >= tr_data[0].shape[0]:
                continue
            obs_j  = tr_data[0][test_pos]
            pred_j = tr_data[1][test_pos]
            t_ax_tr = tr_data[2]
            _ss_res = np.sum((obs_j - pred_j)**2)
            _ss_tot = np.sum((obs_j - obs_j.mean())**2)
            _r2_tr  = 1 - _ss_res/_ss_tot if _ss_tot > 1e-12 else float('nan')
            if not _plotted_obs:
                ax_tr.plot(t_ax_tr, obs_j, color='darkgray', lw=1.4,
                           label=f'observed')
                _plotted_obs = True
            ax_tr.plot(t_ax_tr, pred_j, color=clr_tr, lw=1.2, alpha=0.9,
                       label=f'{lbl_tr}  R²={_r2_tr:+.3f}')

        if row_tr == 0:
            ax_tr.set_title(f'Decode traces\ntest {test_pos}', fontsize=7)
            ax_tr.legend(fontsize=4.5, loc='upper right')
        else:
            ax_tr.set_title(f'test {test_pos}', fontsize=7)
        ax_tr.set_xlabel('Time (s)', fontsize=6)
        ax_tr.set_ylabel(BEH_VAR, fontsize=6)
        ax_tr.tick_params(labelsize=5); ax_tr.grid(True, alpha=0.15)

    # fill remaining rows in trace column with empty axes
    for row_empty in range(_n_show_t, _n_rows_total):
        ax_e = fig_nc.add_subplot(gs[row_empty, col_tr])
        ax_e.set_visible(False)

    _p_nc = SHUFFLE_DIR / f'{ANALYSIS_MODE}_neural_comparison_{BEH_VAR}.png'
    plt.savefig(_p_nc, dpi=130, bbox_inches='tight'); plt.close()
    print(f"Saved → {_p_nc}")

# ══════════════════════════════════════════════════════════════════════════════
# ── TEXT SUMMARY ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
summary_lines = [
    f"Shuffle null distribution analysis",
    f"Mode        : {ANALYSIS_MODE}" + (f"  ({SCRAMBLE_MODE})" if SCRAMBLE_MODE else ""),
    f"Dir         : {SHUFFLE_DIR}",
    f"Real dir    : {REAL_DIR}",
    f"N shuffles  : {len(df)}",
    f"BEH VAR     : {BEH_VAR}",
    f"",
    f"--- R² null ---",
    f"  mean={df.r2.mean():+.6f}  std={df.r2.std():.6f}",
    f"  P5={np.percentile(df.r2,5):+.6f}  P50={np.percentile(df.r2,50):+.6f}  "
    f"P95={np.percentile(df.r2,95):+.6f}",
    f"  min={df.r2.min():+.6f} (shuf {int(df.loc[idx_worst_r2,'shuffle_index'])})",
    f"  max={df.r2.max():+.6f} (shuf {int(df.loc[idx_best_r2, 'shuffle_index'])})",
]
if args.real_r2 is not None:
    summary_lines += [f"  real={args.real_r2:+.6f}  "
                      f"p={(df.r2.values>=args.real_r2).mean():.4f}"]
summary_lines += [
    f"",
    f"--- Pearson r null ---",
    f"  mean={df.r.mean():+.6f}  std={df.r.std():.6f}",
    f"  P5={np.percentile(df.r,5):+.6f}  P50={np.percentile(df.r,50):+.6f}  "
    f"P95={np.percentile(df.r,95):+.6f}",
    f"  min={df.r.min():+.6f} (shuf {int(df.loc[idx_worst_r,'shuffle_index'])})",
    f"  max={df.r.max():+.6f} (shuf {int(df.loc[idx_best_r, 'shuffle_index'])})",
]
if args.real_r is not None:
    summary_lines += [f"  real={args.real_r:+.6f}  "
                      f"p={(df.r.values>=args.real_r).mean():.4f}"]

_sp = SHUFFLE_DIR / f'{ANALYSIS_MODE}_analysis_summary_{BEH_VAR}.txt'
with open(_sp, 'w') as f: f.write('\n'.join(summary_lines) + '\n')
print(f"\nSaved → {_sp}")
print('\n'.join(summary_lines))