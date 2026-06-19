"""
analyse_shuffles.py
────────────────────
Analyse circular-shuffle null distributions from density_decoding_trial_shuffle.py.

Usage:
    python analyse_shuffles.py --shuffle_dir <path_to_shuffle_parent_dir>

The shuffle parent dir is the folder that CONTAINS shuffle_0/, shuffle_1/, etc.
It also contains shuffle_results_summary.npz if the shuffle script ran to completion,
but this script will re-scan from individual params JSON files so it works even
if the summary is partial or missing.

Outputs (all saved into shuffle_dir):
    shuffle_null_r2_r.png               — null distribution histograms
    shuffle_decode_traces_best_r.png    — decode traces for highest r shuffle
    shuffle_decode_traces_worst_r.png   — decode traces for lowest r shuffle
    shuffle_decode_traces_best_r2.png   — decode traces for highest R² (if differs)
    shuffle_decode_traces_worst_r2.png  — decode traces for lowest R² (if differs)
    shuffle_analysis_summary.txt        — text summary
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'figure.dpi': 130,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 10,
})

# ── ARGS ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(description='Analyse shuffle null distribution')
ap.add_argument('--shuffle_dir', default='/nexus/posix0/MBR-neuralsystems/ephys/GLC6962/GLC6962_g_g0/density_decoding/pc1_100ms_1.5s_interleaved_zscoreboth_embargo1_selected-similar_trials_PC1_peak_rms0.49_shuffle',
                help='Path to the parent folder containing shuffle_0/, shuffle_1/, …')
ap.add_argument('--real_r2',   type=float, default=-1.4,
                help='R² from the real (non-shuffled) run, for overlay on null plot')
ap.add_argument('--real_r',    type=float, default=0.48,
                help='Pearson r from the real run, for overlay on null plot')
ap.add_argument('--behavior_var', default='PC1',
                help='Behaviour variable label (inferred from JSON if not provided)')
args = ap.parse_args()


SHUFFLE_DIR = Path(args.shuffle_dir)
assert SHUFFLE_DIR.exists(), f"shuffle_dir not found: {SHUFFLE_DIR}"

# ── SCAN FOR COMPLETED SHUFFLE FOLDERS ───────────────────────────────────────
print(f"Scanning {SHUFFLE_DIR} for completed shuffles ...")

records = []
for item in sorted(SHUFFLE_DIR.iterdir()):
    if not item.is_dir() or not item.name.startswith('shuffle_'):
        continue

    # find params JSON
    params_files = list(item.glob('params_*.json'))
    if not params_files:
        continue
    params_path = params_files[0]

    # find decode results npz
    decode_files = list(item.glob('decode_results_*.npz'))
    if not decode_files:
        print(f"  [SKIP] {item.name} — no decode_results npz found")
        continue

    try:
        with open(params_path) as f:
            p = json.load(f)

        rec = {
            'shuffle_index': int(p['shuffle_index']),
            'shift_sec':     float(p['shift_sec']),
            'r2':            float(p['r2']),
            'r':             float(p['r']),
            'n_train':       int(p.get('n_train', 0)),
            'n_test':        int(p.get('n_test', 0)),
            'T_TRIAL':       float(p.get('T_TRIAL_s', p.get('T_TRIAL', 0))),
            'N_T_BINS':      int(p.get('N_T_BINS', 0)),
            'decode_npz':    str(decode_files[0]),
            'folder':        str(item),
        }
        records.append(rec)

        if args.behavior_var is None and 'behavior_var' in p:
            args.behavior_var = p['behavior_var']

    except Exception as e:
        print(f"  [WARN] {item.name} — failed to parse: {e}")
        continue

if not records:
    print("No completed shuffles found. Exiting.")
    sys.exit(0)

df = pd.DataFrame(records).sort_values('shuffle_index').reset_index(drop=True)
BEH_VAR = args.behavior_var or 'PC'

print(f"\nFound {len(df)} completed shuffles")
print(f"  R²: mean={df.r2.mean():+.4f}  std={df.r2.std():.4f}  "
      f"min={df.r2.min():+.4f}  max={df.r2.max():+.4f}")
print(f"  r : mean={df.r.mean():+.4f}  std={df.r.std():.4f}  "
      f"min={df.r.min():+.4f}  max={df.r.max():+.4f}")

# ── IDENTIFY EXTREME SHUFFLES ─────────────────────────────────────────────────
idx_best_r   = df['r'].idxmax()
idx_worst_r  = df['r'].idxmin()
idx_best_r2  = df['r2'].idxmax()
idx_worst_r2 = df['r2'].idxmin()

print(f"\n  Best  r  : shuffle {df.loc[idx_best_r,  'shuffle_index']}  "
      f"r={df.loc[idx_best_r,  'r']:+.4f}  R²={df.loc[idx_best_r,  'r2']:+.4f}")
print(f"  Worst r  : shuffle {df.loc[idx_worst_r, 'shuffle_index']}  "
      f"r={df.loc[idx_worst_r, 'r']:+.4f}  R²={df.loc[idx_worst_r, 'r2']:+.4f}")
print(f"  Best  R² : shuffle {df.loc[idx_best_r2,  'shuffle_index']}  "
      f"r={df.loc[idx_best_r2,  'r']:+.4f}  R²={df.loc[idx_best_r2,  'r2']:+.4f}")
print(f"  Worst R² : shuffle {df.loc[idx_worst_r2, 'shuffle_index']}  "
      f"r={df.loc[idx_worst_r2, 'r']:+.4f}  R²={df.loc[idx_worst_r2, 'r2']:+.4f}")

# check if best r == best R² etc.
same_best  = idx_best_r  == idx_best_r2
same_worst = idx_worst_r == idx_worst_r2
print(f"\n  Best  r and best  R² are {'the SAME' if same_best  else 'DIFFERENT'} shuffle")
print(f"  Worst r and worst R² are {'the SAME' if same_worst else 'DIFFERENT'} shuffle")

# ── NULL DISTRIBUTION PLOT ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"Shuffle null distribution — {BEH_VAR}\n"
    f"{len(df)} shuffles  |  "
    f"shift range: {df.shift_sec.min():.0f}–{df.shift_sec.max():.0f} s",
    fontsize=11,
)

for ax, col, label, real_val in [
    (axes[0], 'r2', 'R²',        args.real_r2),
    (axes[1], 'r',  'Pearson r', args.real_r),
]:
    vals = df[col].values
    ax.hist(vals, bins=min(20, len(df)//2 + 1),
            color='steelblue', edgecolor='white', alpha=0.8, label='shuffle')

    # percentile lines
    for pct, ls, clr in [(5, '--', 'orange'), (50, '-', 'orange'), (95, '--', 'orange')]:
        v = np.percentile(vals, pct)
        ax.axvline(v, color=clr, lw=1.2, ls=ls,
                   label=f'P{pct}={v:+.3f}' if pct == 50 else f'P{pct}={v:+.3f}')

    ax.axvline(0, color='black', lw=0.8, ls=':', alpha=0.5)

    # real run overlay
    if real_val is not None:
        ax.axvline(real_val, color='crimson', lw=2.5,
                   label=f'Real run = {real_val:+.4f}')
        # p-value: fraction of shuffles >= real
        p_val = (vals >= real_val).mean()
        ax.text(real_val, ax.get_ylim()[1] * 0.85,
                f' p={p_val:.3f}', color='crimson', fontsize=9, va='top')

    # mark best/worst
    ax.axvline(vals.max(), color='darkgreen', lw=1.2, ls=':',
               label=f'Best={vals.max():+.3f}')
    ax.axvline(vals.min(), color='darkred',   lw=1.2, ls=':',
               label=f'Worst={vals.min():+.3f}')

    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'Null distribution — {label}', fontsize=10)
    ax.legend(fontsize=7.5, loc='upper left' if col == 'r' else 'upper right')
    ax.grid(True, alpha=0.15)

plt.tight_layout()
_null_plot = SHUFFLE_DIR / f'shuffle_null_r2_r_{BEH_VAR}.png'
plt.savefig(_null_plot, dpi=130, bbox_inches='tight')
print(f"\nSaved → {_null_plot}")
plt.close()

# ── DECODE TRACE HELPER ───────────────────────────────────────────────────────
def plot_decode_traces(decode_npz_path, row, label, save_path):
    """Load decode results and plot one panel per test trial."""
    try:
        d = np.load(decode_npz_path)
    except Exception as e:
        print(f"  [WARN] Could not load {decode_npz_path}: {e}")
        return

    obs_test   = d['obs_test']     # flat array
    pred_test  = d['pred_test']
    train_idx  = d['train_idx']
    test_idx   = d['test_idx']
    T_TRIAL    = float(d.get('T_TRIAL', row['T_TRIAL']))
    N_T_BINS   = int(d.get('N_T_BINS', row['N_T_BINS']))

    if N_T_BINS < 7:
        print(f"  [WARN] N_T_BINS={N_T_BINS} too small for sliding window decoder — skipping")
        return

    _half_win    = 7 // 2
    _n_win       = N_T_BINS - _half_win + 1
    _n_pred_bins = _n_win - _half_win
    _bin_dur_s   = T_TRIAL / N_T_BINS
    _bin_offsets = np.arange(_half_win, _n_win) * _bin_dur_s

    n_test = len(test_idx)
    if len(obs_test) != n_test * _n_pred_bins:
        # try to infer n_pred_bins from data
        if len(obs_test) % n_test == 0:
            _n_pred_bins = len(obs_test) // n_test
            _bin_offsets = np.arange(_n_pred_bins) * _bin_dur_s
        else:
            print(f"  [WARN] obs_test shape mismatch — skipping trace plot")
            return

    pred_by_trial = pred_test.reshape(n_test, _n_pred_bins)
    obs_by_trial  = obs_test.reshape(n_test,  _n_pred_bins)

    _ncols = min(4, n_test)
    _nrows = int(np.ceil(n_test / _ncols))
    _panel_w = max(3.5, T_TRIAL * 0.8)

    fig, axes = plt.subplots(
        _nrows, _ncols,
        figsize=(_ncols * _panel_w, _nrows * 3.0),
        squeeze=False, sharey=True,
    )
    fig.suptitle(
        f"Decode traces — shuffle {int(row['shuffle_index'])}  [{label}]\n"
        f"{BEH_VAR}  |  r={row['r']:+.4f}  R²={row['r2']:+.4f}  "
        f"shift={row['shift_sec']:.1f} s  |  {n_test} test trials",
        fontsize=10,
    )

    for _j in range(n_test):
        _row, _col = divmod(_j, _ncols)
        _ax = axes[_row, _col]
        _ax.plot(_bin_offsets, obs_by_trial[_j],
                 color='darkgray', lw=1.2, label='observed' if _j == 0 else None)
        _ax.plot(_bin_offsets, pred_by_trial[_j],
                 color='tab:blue', lw=1.2, alpha=0.85,
                 label='decoded' if _j == 0 else None)
        _ax.set_title(f"test {_j}", fontsize=7, pad=2)
        _ax.set_xlabel('Time within trial (s)', fontsize=7)
        if _col == 0:
            _ax.set_ylabel(f'{BEH_VAR}', fontsize=7)
        _ax.tick_params(labelsize=6)
        _ax.grid(True, alpha=0.18)
        _ax.set_xlim(0, T_TRIAL)
        if _j == 0:
            _ax.legend(fontsize=6, loc='upper right')

    for _j in range(n_test, _nrows * _ncols):
        _r, _c = divmod(_j, _ncols)
        axes[_r, _c].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    print(f"Saved → {save_path}")
    plt.close()

# ── PLOT DECODE TRACES FOR EXTREME SHUFFLES ───────────────────────────────────
cases = [
    (idx_best_r,   f'shuffle_{int(df.loc[idx_best_r,   "shuffle_index"])}_best_r',
                   f'best r={df.loc[idx_best_r,   "r"]:+.4f}'),
    (idx_worst_r,  f'shuffle_{int(df.loc[idx_worst_r,  "shuffle_index"])}_worst_r',
                   f'worst r={df.loc[idx_worst_r,  "r"]:+.4f}'),
    (idx_best_r2,  f'shuffle_{int(df.loc[idx_best_r2,  "shuffle_index"])}_best_r2',
                   f'best R2={df.loc[idx_best_r2,  "r2"]:+.4f}'),
    (idx_worst_r2, f'shuffle_{int(df.loc[idx_worst_r2, "shuffle_index"])}_worst_r2',
                   f'worst R2={df.loc[idx_worst_r2, "r2"]:+.4f}'),
]
if same_best:
    print(f"  [NOTE] best r and best R2 are the same shuffle "
          f"({int(df.loc[idx_best_r, 'shuffle_index'])}) — plots will be identical")
if same_worst:
    print(f"  [NOTE] worst r and worst R2 are the same shuffle "
          f"({int(df.loc[idx_worst_r, 'shuffle_index'])}) — plots will be identical")

for idx, fname, label in cases:
    row = df.loc[idx]
    save_path = SHUFFLE_DIR / f'shuffle_decode_traces_{fname}_{BEH_VAR}.png'
    print(f"\nPlotting {label} (shuffle {int(row['shuffle_index'])}) ...")
    plot_decode_traces(row['decode_npz'], row, label, save_path)

# ── TEXT SUMMARY ──────────────────────────────────────────────────────────────
summary_lines = [
    f"Shuffle null distribution analysis",
    f"Shuffle dir : {SHUFFLE_DIR}",
    f"N shuffles  : {len(df)}",
    f"BEH VAR     : {BEH_VAR}",
    f"",
    f"--- R² null distribution ---",
    f"  mean  : {df.r2.mean():+.6f}",
    f"  std   : {df.r2.std():.6f}",
    f"  P5    : {np.percentile(df.r2, 5):+.6f}",
    f"  P50   : {np.percentile(df.r2, 50):+.6f}",
    f"  P95   : {np.percentile(df.r2, 95):+.6f}",
    f"  min   : {df.r2.min():+.6f}  (shuffle {int(df.loc[idx_worst_r2, 'shuffle_index'])})",
    f"  max   : {df.r2.max():+.6f}  (shuffle {int(df.loc[idx_best_r2,  'shuffle_index'])})",
]
if args.real_r2 is not None:
    p_r2 = (df.r2.values >= args.real_r2).mean()
    summary_lines += [
        f"  real  : {args.real_r2:+.6f}",
        f"  p-val : {p_r2:.4f}  (fraction of shuffles >= real R²)",
    ]
summary_lines += [
    f"",
    f"--- Pearson r null distribution ---",
    f"  mean  : {df.r.mean():+.6f}",
    f"  std   : {df.r.std():.6f}",
    f"  P5    : {np.percentile(df.r, 5):+.6f}",
    f"  P50   : {np.percentile(df.r, 50):+.6f}",
    f"  P95   : {np.percentile(df.r, 95):+.6f}",
    f"  min   : {df.r.min():+.6f}  (shuffle {int(df.loc[idx_worst_r, 'shuffle_index'])})",
    f"  max   : {df.r.max():+.6f}  (shuffle {int(df.loc[idx_best_r,  'shuffle_index'])})",
]
if args.real_r is not None:
    p_r = (df.r.values >= args.real_r).mean()
    summary_lines += [
        f"  real  : {args.real_r:+.6f}",
        f"  p-val : {p_r:.4f}  (fraction of shuffles >= real r)",
    ]
summary_lines += [
    f"",
    f"--- Best/worst shuffles ---",
    f"  Best  r  : shuffle {int(df.loc[idx_best_r,   'shuffle_index'])}  "
    f"r={df.loc[idx_best_r,  'r']:+.4f}  R²={df.loc[idx_best_r,  'r2']:+.4f}  "
    f"shift={df.loc[idx_best_r, 'shift_sec']:.1f} s",
    f"  Worst r  : shuffle {int(df.loc[idx_worst_r,  'shuffle_index'])}  "
    f"r={df.loc[idx_worst_r, 'r']:+.4f}  R²={df.loc[idx_worst_r, 'r2']:+.4f}  "
    f"shift={df.loc[idx_worst_r,'shift_sec']:.1f} s",
    f"  Best  R² : shuffle {int(df.loc[idx_best_r2,  'shuffle_index'])}  "
    f"r={df.loc[idx_best_r2, 'r']:+.4f}  R²={df.loc[idx_best_r2, 'r2']:+.4f}  "
    f"shift={df.loc[idx_best_r2,'shift_sec']:.1f} s",
    f"  Worst R² : shuffle {int(df.loc[idx_worst_r2, 'shuffle_index'])}  "
    f"r={df.loc[idx_worst_r2,'r']:+.4f}  R²={df.loc[idx_worst_r2,'r2']:+.4f}  "
    f"shift={df.loc[idx_worst_r2,'shift_sec']:.1f} s",
]

_summ_path = SHUFFLE_DIR / f'shuffle_analysis_summary_{BEH_VAR}.txt'
with open(_summ_path, 'w') as f:
    f.write('\n'.join(summary_lines) + '\n')
print(f"\nSaved → {_summ_path}")
print('\n'.join(summary_lines))