# %% [markdown]
# # PC Trial Check — windows defined by texture_speed
#
# **Pipeline:**
# 1. Run the full texture_speed peak-align + trim + RMS-filter pipeline
#    (identical logic to trial_structure.py) to obtain a set of trial windows.
# 2. For each of N_PCS principal components, extract those same windows from
#    the PC trace (no extra alignment or trimming — just the texture_speed-
#    defined windows).
# 3. Per PC, produce three panels:
#    a. Raw overlay traces (not demeaned)
#    b. Demeaned overlay traces
#    c. RMS-similarity plot  (traces coloured by RMS deviation from group
#       mean of demeaned traces, + RMS score histogram)  — same style as
#       trial_structure.py

# %% ── IMPORTS ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pandas.core.arrays import Categorical
from pandas._libs.arrays import NDArrayBacked

# ── patch pandas Categorical deserialisation for pickles saved with older pandas
_orig_categorical_setstate = Categorical.__setstate__

def _patched_categorical_setstate(self, state):
    if not isinstance(state, dict):
        dtype, codes = state
        NDArrayBacked.__init__(self, codes, dtype)
    else:
        _orig_categorical_setstate(self, state)

Categorical.__setstate__ = _patched_categorical_setstate

plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 11,
})

# %% ── SETTINGS ───────────────────────────────────────────────────────────────
behaviour_folder = "/ptmp/gongt/behaviour"
animal = 'GLC6962'
exp    = 'g_g0'

MIN_DUR    = 5
MAX_DUR    = 30
WINDOW_SEC = MIN_DUR          # 5 s window

# texture_speed filtering parameters (same as trial_structure.py)
TS_RMS_THRESHOLD = 1.2        # set to None on first run to see distribution
TS_ALIGN_MODE    = 'peak'     # peak-align texture_speed

# How many PCs to check (will use PC1 … PC{N_PCS})
N_PCS = 20

PLOT_DIR = Path(f"/ptmp/gongt/behaviour/{animal}/{animal}_{exp}/trial_filter/ts_pc_check")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# %% ── HELPERS ────────────────────────────────────────────────────────────────
def savefig(fig, name):
    p = PLOT_DIR / name
    fig.savefig(p, dpi=130, bbox_inches='tight')
    print(f"  Saved → {p}")
    plt.close(fig)

def rms_from_mean(traces):
    """Per-trial RMS deviation from group mean. traces: (n_trials, n_samples)."""
    mean_trace = np.nanmean(traces, axis=0)
    residuals  = traces - mean_trace[np.newaxis, :]
    return np.sqrt(np.nanmean(residuals ** 2, axis=1))

# %% ── LOAD DATA ──────────────────────────────────────────────────────────────
traj_path = (f"{behaviour_folder}/{animal}/{animal}_{exp}/"
             f"trajectory_times_sec.txt")
trajectory_times_sec = pd.read_csv(traj_path)

texture_pca_df = pd.read_pickle(f"{behaviour_folder}/texture_pca_df_preproc.pkl")
beh = texture_pca_df[
    (texture_pca_df['animal'] == animal) &
    (texture_pca_df['background'] == 'eink')
].copy()
beh_t = beh['time'].dt.total_seconds().values

# ── collect available PC columns ─────────────────────────────────────────────
pc_cols_available = [f'PC{i}' for i in range(1, N_PCS + 1) if f'PC{i}' in beh.columns]
if 'texture_speed' not in beh.columns:
    raise RuntimeError("'texture_speed' column not found in behaviour dataframe.")

all_check_cols = pc_cols_available      # PCs to check (texture_speed excluded)
print(f"PCs available for checking : {all_check_cols}")

# Build a big array: [texture_speed] + all PC cols for fast indexing
data_cols  = ['texture_speed'] + all_check_cols
beh_arr    = beh[data_cols].values.astype(float)
TS_IDX     = 0                          # column index of texture_speed
PC_INDICES = {col: i + 1 for i, col in enumerate(all_check_cols)}

dt        = float(np.median(np.diff(beh_t)))
FS        = 1.0 / dt
N_SAMPLES = int(round(WINDOW_SEC * FS))
t_axis    = np.arange(N_SAMPLES) / FS
half      = WINDOW_SEC / 2.0

print(f"Behaviour frames : {len(beh):,}")
print(f"Inferred FS      : {FS:.2f} Hz  →  {N_SAMPLES} samples / {WINDOW_SEC} s")
print(f"Trajectories     : {len(trajectory_times_sec)} trials")

onsets    = trajectory_times_sec['onset_sec'].values
offsets   = trajectory_times_sec['offset_sec'].values
peaks     = trajectory_times_sec['peaks_sec'].values
durations = offsets - onsets

# %% ── STEP 1 : duration filter ───────────────────────────────────────────────
duration_mask = (durations >= MIN_DUR) & (durations <= MAX_DUR)
sel_idx       = np.where(duration_mask)[0]
print(f"\nDuration filter [{MIN_DUR}–{MAX_DUR} s]: "
      f"{len(sel_idx)} / {len(durations)} trials")

# %% ── STEP 2 : extract texture_speed windows (peak-aligned) ─────────────────
print(f"\n── Extracting texture_speed windows (peak-aligned, {WINDOW_SEC} s) ──")
traces_peak_ts   = np.full((len(sel_idx), N_SAMPLES), np.nan)
peak_win_starts  = np.full(len(sel_idx), np.nan)
peak_win_ends    = np.full(len(sel_idx), np.nan)
valid_peak       = np.zeros(len(sel_idx), dtype=bool)

for ii, tri in enumerate(sel_idx):
    t0_pk = peaks[tri] - half
    t1_pk = peaks[tri] + half
    mask  = (beh_t >= t0_pk) & (beh_t < t1_pk)
    chunk = beh_arr[mask, TS_IDX]
    if len(chunk) >= N_SAMPLES:
        traces_peak_ts[ii]   = chunk[:N_SAMPLES]
        peak_win_starts[ii]  = t0_pk
        peak_win_ends[ii]    = t1_pk
        valid_peak[ii]       = True

traces_ts_v   = traces_peak_ts[valid_peak]
sel_idx_v     = sel_idx[valid_peak]
win_starts_v  = peak_win_starts[valid_peak]
win_ends_v    = peak_win_ends[valid_peak]
print(f"  Valid peak-aligned windows : {valid_peak.sum()} / {len(sel_idx)}")

# %% ── STEP 3 : texture_speed RMS filter ─────────────────────────────────────
traces_ts_demeaned = traces_ts_v - np.nanmean(traces_ts_v, axis=1, keepdims=True)
ts_rms_scores      = rms_from_mean(traces_ts_demeaned)

print(f"\ntexture_speed RMS scores — "
      f"min={ts_rms_scores.min():.4f}  "
      f"median={np.median(ts_rms_scores):.4f}  "
      f"max={ts_rms_scores.max():.4f}")
print(f"  P25={np.percentile(ts_rms_scores, 25):.4f}  "
      f"P50={np.percentile(ts_rms_scores, 50):.4f}  "
      f"P75={np.percentile(ts_rms_scores, 75):.4f}")

# ── texture_speed RMS distribution diagnostic plot ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle(
    f'{animal} {exp} — texture_speed,  peak-aligned,  '
    f'dur [{MIN_DUR}–{MAX_DUR} s],  win {WINDOW_SEC} s\n'
    f'Trials coloured by RMS deviation from group mean  (n={len(traces_ts_v)})',
    fontsize=11,
)
clr_ts   = 'mediumpurple'
cmap_rms = plt.cm.RdYlBu_r
ts_score_norm = ((ts_rms_scores - ts_rms_scores.min()) /
                 (ts_rms_scores.max() - ts_rms_scores.min() + 1e-12))
mean_ts_dm = np.nanmean(traces_ts_demeaned, axis=0)

ax = axes[0]
for ti in np.argsort(ts_rms_scores):
    ax.plot(t_axis, traces_ts_demeaned[ti],
            color=cmap_rms(ts_score_norm[ti]), alpha=0.6, lw=0.8)
ax.plot(t_axis, mean_ts_dm, color='black', lw=2.0,
        label='group mean (demeaned)')
if TS_RMS_THRESHOLD is not None:
    ax.fill_between(t_axis,
                    mean_ts_dm - TS_RMS_THRESHOLD,
                    mean_ts_dm + TS_RMS_THRESHOLD,
                    color='black', alpha=0.10,
                    label=f'±RMS thr ({TS_RMS_THRESHOLD})')
sm = plt.cm.ScalarMappable(cmap=cmap_rms,
                           norm=plt.Normalize(ts_rms_scores.min(),
                                              ts_rms_scores.max()))
sm.set_array([])
plt.colorbar(sm, ax=ax, label='RMS deviation (demeaned texture_speed units)')
ax.set_xlabel(f'Time relative to peak (s)  [peak at t={half:.1f}]')
ax.set_ylabel('texture_speed (demeaned)')
ax.set_title('Demeaned traces coloured by RMS score\n(red = most deviant)')
ax.legend(fontsize=8);  ax.grid(alpha=0.15)

ax2 = axes[1]
ax2.hist(ts_rms_scores, bins=20, color=clr_ts, edgecolor='white', alpha=0.8)
if TS_RMS_THRESHOLD is not None:
    ax2.axvline(TS_RMS_THRESHOLD, color='crimson', lw=2,
                label=f'threshold = {TS_RMS_THRESHOLD}')
    ax2.legend(fontsize=9)
for pct in [25, 50, 75]:
    v = np.percentile(ts_rms_scores, pct)
    ax2.axvline(v, color='orange', ls='--', lw=1, alpha=0.8)
    ax2.text(v, ax2.get_ylim()[1] * 0.93, f'P{pct}',
             ha='center', fontsize=8, color='darkorange')
ax2.set_xlabel('RMS deviation from group mean')
ax2.set_ylabel('Count')
ax2.set_title('texture_speed per-trial RMS score')
plt.tight_layout()
savefig(fig, f"{animal}_{exp}_texture_speed_peak_rms_score_distribution.png")

# ── apply threshold ───────────────────────────────────────────────────────────
if TS_RMS_THRESHOLD is None:
    print("\nTS_RMS_THRESHOLD is None — using ALL valid trials for PC check.")
    keep_mask = np.ones(len(traces_ts_v), dtype=bool)
else:
    keep_mask = ts_rms_scores <= TS_RMS_THRESHOLD
    print(f"\nRMS threshold = {TS_RMS_THRESHOLD}  →  "
          f"{keep_mask.sum()} / {len(traces_ts_v)} trials selected")

# Final trial set
ts_traces_final    = traces_ts_v[keep_mask]
sel_idx_final      = sel_idx_v[keep_mask]
win_starts_final   = win_starts_v[keep_mask]
win_ends_final     = win_ends_v[keep_mask]
n_final            = keep_mask.sum()

# %% ── STEP 4 : texture_speed before/after filter plot ───────────────────────
traces_ts_dm_all = traces_ts_demeaned
traces_ts_dm_sel = traces_ts_demeaned[keep_mask]
thr = TS_RMS_THRESHOLD if TS_RMS_THRESHOLD is not None else 0.0

fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=True, sharey=True)
fig.suptitle(
    f'{animal} {exp} — texture_speed,  peak-aligned,  '
    f'win {WINDOW_SEC} s,  dur [{MIN_DUR}–{MAX_DUR} s],  '
    f'RMS threshold ≤ {TS_RMS_THRESHOLD}  (demeaned)',
    fontsize=11,
)
panels = [
    (traces_ts_dm_all, f'Before  (n={len(traces_ts_v)})'),
    (traces_ts_dm_sel, f'After  (n={n_final})'),
]
for col_idx, (tr_set, title) in enumerate(panels):
    ax = axes[col_idx]
    m  = np.nanmean(tr_set, axis=0)
    for ti in range(len(tr_set)):
        ax.plot(t_axis, tr_set[ti], color=clr_ts, alpha=0.25, lw=0.7)
    s = np.nanstd(tr_set, axis=0)
    ax.fill_between(t_axis, m - s, m + s, color=clr_ts, alpha=0.20)
    ax.plot(t_axis, m, color='black', lw=2.0, label='mean ± 1 SD')
    if TS_RMS_THRESHOLD is not None:
        ax.fill_between(t_axis, m - thr, m + thr,
                        color='black', alpha=0.08, label=f'±RMS thr ({thr})')
        ax.plot(t_axis, m + thr, color='black', lw=0.8, ls='--')
        ax.plot(t_axis, m - thr, color='black', lw=0.8, ls='--')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f'Time relative to peak (s)  [peak at t={half:.1f}]')
    ax.set_ylabel('texture_speed (demeaned)' if col_idx == 0 else '')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.12)
plt.tight_layout()
savefig(fig, f"{animal}_{exp}_texture_speed_peak_before_vs_after_filter.png")

# %% ── STEP 5 : sanity-check — full trace with selected windows shaded ────────
# Show the first SANITY_PLOT_SEC seconds of the continuous recording for a
# handful of PCs (+ texture_speed), with every selected trial window shaded so
# you can visually confirm the same windows are being pulled for every channel.

SANITY_PLOT_SEC = 600          # how many seconds of the recording to show
SANITY_COLS     = ['texture_speed'] + all_check_cols[:4]   # ts + first 4 PCs

n_sanity = len(SANITY_COLS)
t_end    = beh_t[0] + SANITY_PLOT_SEC
t_mask   = (beh_t >= beh_t[0]) & (beh_t <= t_end)
t_seg    = beh_t[t_mask]

fig, axes = plt.subplots(n_sanity, 1, figsize=(22, 3 * n_sanity), sharex=True)
fig.suptitle(
    f'{animal} {exp} — full trace ({SANITY_PLOT_SEC} s) with selected windows shaded\n'
    f'(n={n_final} windows from texture_speed peak-align + RMS filter)',
    fontsize=11,
)

shade_color = 'gold'
for row, col in enumerate(SANITY_COLS):
    ax  = axes[row]
    ci  = TS_IDX if col == 'texture_speed' else PC_INDICES[col]
    seg = beh_arr[t_mask, ci]
    ax.plot(t_seg, seg, lw=0.6, color='steelblue' if col != 'texture_speed' else 'mediumpurple')
    # shade selected windows that fall within this plot range
    for t0, t1 in zip(win_starts_final, win_ends_final):
        if t1 < t_seg[0] or t0 > t_seg[-1]:
            continue
        ax.axvspan(max(t0, t_seg[0]), min(t1, t_seg[-1]),
                   color=shade_color, alpha=0.35, lw=0)
    ax.set_ylabel(col, fontsize=9)
    ax.grid(alpha=0.12)

axes[-1].set_xlabel('Time (s)')
plt.tight_layout()
savefig(fig, f"{animal}_{exp}_window_sanity_check_first{SANITY_PLOT_SEC}s.png")

# %% ── STEP 6 : per-PC check ──────────────────────────────────────────────────
x_label = f'Time relative to texture_speed peak (s)  [peak at t={half:.1f}]'

# colour palette for PCs (cycle through tab20)
cmap_pc = plt.colormaps['tab20'].resampled(len(all_check_cols))
PC_COLORS = {col: cmap_pc(i) for i, col in enumerate(all_check_cols)}

for pc_col in all_check_cols:  # STEP 6 loop
    pi  = PC_INDICES[pc_col]
    clr = PC_COLORS[pc_col]
    print(f"\n── {pc_col} ──")

    # ── extract windows using texture_speed-defined win_starts/win_ends ────────
    traces_pc = np.full((n_final, N_SAMPLES), np.nan)

    for ii, (t0, t1) in enumerate(zip(win_starts_final, win_ends_final)):
        mask  = (beh_t >= t0) & (beh_t < t1)
        chunk = beh_arr[mask, pi]
        if len(chunk) >= N_SAMPLES:
            traces_pc[ii] = chunk[:N_SAMPLES]
        # if too short, leave as NaN (edge-of-recording trials)

    # drop all-NaN rows (shouldn't happen since texture_speed windows are valid)
    valid_rows = ~np.all(np.isnan(traces_pc), axis=1)
    traces_pc  = traces_pc[valid_rows]
    n_pc       = len(traces_pc)
    print(f"  Valid traces : {n_pc} / {n_final}")

    if n_pc == 0:
        print(f"  [SKIP] No valid traces for {pc_col}")
        continue

    # ── demean ────────────────────────────────────────────────────────────────
    traces_pc_dm = traces_pc - np.nanmean(traces_pc, axis=1, keepdims=True)
    mean_raw     = np.nanmean(traces_pc,    axis=0)
    mean_dm      = np.nanmean(traces_pc_dm, axis=0)
    std_dm       = np.nanstd(traces_pc_dm,  axis=0)

    # ── RMS similarity scores (on demeaned) ───────────────────────────────────
    pc_rms_scores = rms_from_mean(traces_pc_dm)
    score_norm    = ((pc_rms_scores - pc_rms_scores.min()) /
                     (pc_rms_scores.max() - pc_rms_scores.min() + 1e-12))

    # ══ Figure : 3-panel per PC ════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(20, 4.5))
    fig.suptitle(
        f'{animal} {exp} — {pc_col}\n'
        f'Windows defined by texture_speed (peak-aligned, {WINDOW_SEC} s)  '
        f'[n={n_pc} trials]',
        fontsize=11,
    )

    # ── Panel A : raw (not demeaned) overlay ──────────────────────────────────
    ax = axes[0]
    for ti in range(n_pc):
        ax.plot(t_axis, traces_pc[ti], color=clr, alpha=0.25, lw=0.7)
    m_raw = np.nanmean(traces_pc, axis=0)
    s_raw = np.nanstd(traces_pc,  axis=0)
    ax.fill_between(t_axis, m_raw - s_raw, m_raw + s_raw,
                    color=clr, alpha=0.22)
    ax.plot(t_axis, m_raw, color='black', lw=2.0, label='mean ± 1 SD')
    ax.plot(t_axis, m_raw + s_raw, color='black', lw=0.8, ls='--')
    ax.plot(t_axis, m_raw - s_raw, color='black', lw=0.8, ls='--')
    ax.set_xlabel(x_label);  ax.set_ylabel(pc_col)
    ax.set_title('Raw traces (not demeaned)')
    ax.legend(loc='upper right', fontsize=8);  ax.grid(alpha=0.15)

    # ── Panel B : demeaned overlay ────────────────────────────────────────────
    ax = axes[1]
    for ti in range(n_pc):
        ax.plot(t_axis, traces_pc_dm[ti], color=clr, alpha=0.25, lw=0.7)
    ax.fill_between(t_axis, mean_dm - std_dm, mean_dm + std_dm,
                    color=clr, alpha=0.22)
    ax.plot(t_axis, mean_dm, color='black', lw=2.0, label='mean ± 1 SD')
    ax.plot(t_axis, mean_dm + std_dm, color='black', lw=0.8, ls='--')
    ax.plot(t_axis, mean_dm - std_dm, color='black', lw=0.8, ls='--')
    ax.set_xlabel(x_label);  ax.set_ylabel(f'{pc_col} (demeaned)')
    ax.set_title('Demeaned overlay')
    ax.legend(loc='upper right', fontsize=8);  ax.grid(alpha=0.15)

    # ── Panel C : RMS similarity (same style as trial_structure.py) ───────────
    ax = axes[2]
    for ti in np.argsort(pc_rms_scores):
        ax.plot(t_axis, traces_pc_dm[ti],
                color=cmap_rms(score_norm[ti]), alpha=0.6, lw=0.8)
    ax.plot(t_axis, mean_dm, color='black', lw=2.0,
            label='group mean (demeaned)')
    sm = plt.cm.ScalarMappable(cmap=cmap_rms,
                               norm=plt.Normalize(pc_rms_scores.min(),
                                                  pc_rms_scores.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=f'RMS deviation (demeaned {pc_col} units)')
    ax.set_xlabel(x_label);  ax.set_ylabel(f'{pc_col} (demeaned)')
    ax.set_title('RMS similarity\n(red = most deviant from group mean)')
    ax.legend(fontsize=8);  ax.grid(alpha=0.15)

    plt.tight_layout()
    fname = (f"{animal}_{exp}_{pc_col}_ts_windows_"
             f"dur{MIN_DUR}s_thr{TS_RMS_THRESHOLD}_check.png")
    savefig(fig, fname)

    print(f"  RMS scores — min={pc_rms_scores.min():.4f}  "
          f"median={np.median(pc_rms_scores):.4f}  "
          f"max={pc_rms_scores.max():.4f}")

print(f"\nAll done. Plots saved to {PLOT_DIR}")