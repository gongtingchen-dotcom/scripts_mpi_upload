# %% [markdown]
# # Trial Structure Analysis
# **Pipeline (per PC):**
# 1. Load data
# 2. Duration histogram → filter trials
# 3. For each PC separately:
#    a. Extract fixed window (onset-aligned AND peak-aligned) for that PC only
#    b. Quick-look: all duration-filtered trials, both alignments
#    c. Compute group mean trace → per-trial RMS deviation from mean
#    d. Threshold on RMS deviation → before vs after filter plots
#       (threshold shown as envelope around mean on the plot)
#    e. Trial-average ± SD plot
#    f. Export CSV with original onset/offset AND aligned window start/end

# %% ── IMPORTS ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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
WINDOW_SEC = MIN_DUR

PC_COLS   = ['PC1', 'PC2', 'PC3', 'texture_speed']
PC_COLORS = {'PC1': 'royalblue', 'PC2': 'darkorange', 'PC3': 'mediumseagreen',
             'texture_speed': 'mediumpurple'}

# ✏️  RMS threshold in PC units — per trial, RMS deviation from the group mean.
# On first run leave as None to see the score distribution, then set values.
RMS_THRESHOLDS = {
    'PC1':           0.5,
    'PC2':           1.5,
    'PC3':           0.8,
    'texture_speed': 1.2,
}

# ✏️  Set per-PC alignment after inspecting quick-look plots
ALIGN_MODES = {
    'PC1':           'peak',
    'PC2':           'peak',
    'PC3':           'peak',
    'texture_speed': 'peak',   # texture_speed has no clear peak — onset preferred
}

PLOT_DIR = Path("/ptmp/gongt/behaviour/GLC6962/GLC6962_g_g0/trial_filter")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# %% ── HELPERS ────────────────────────────────────────────────────────────────
def savefig(fig, name):
    p = PLOT_DIR / name
    fig.savefig(p, dpi=130, bbox_inches='tight')
    print(f"  Saved → {p}")
    plt.close(fig)

def make_tag(pc, align, thr):
    thr_str = f"thr{thr}" if thr is not None else "thrNone"
    return f"{animal}_{exp}_{pc}_{align}_dur{MIN_DUR}s_{thr_str}"

def rms_from_mean(traces):
    """
    Per-trial RMS deviation from the group mean trace.
    traces : (n_trials, n_samples)
    returns: (n_trials,) array of RMS values, in the same units as the PC.
    """
    mean_trace = np.nanmean(traces, axis=0)          # (n_samples,)
    residuals  = traces - mean_trace[np.newaxis, :]  # (n_trials, n_samples)
    return np.sqrt(np.nanmean(residuals ** 2, axis=1))

# %% ── LOAD DATA ──────────────────────────────────────────────────────────────
traj_path = f"{behaviour_folder}/{animal}/{animal}_{exp}/trajectory_times_sec.txt"
trajectory_times_sec = pd.read_csv(traj_path)

texture_pca_df = pd.read_pickle(f"{behaviour_folder}/texture_pca_df_preproc.pkl")
beh = texture_pca_df[
    (texture_pca_df['animal'] == animal) &
    (texture_pca_df['background'] == 'eink')
].copy()
beh_t   = beh['time'].dt.total_seconds().values

# guard: drop any col not present in the dataframe
_missing_cols = [c for c in PC_COLS if c not in beh.columns]
if _missing_cols:
    print(f"[WARN] Columns not found in beh dataframe: {_missing_cols}")
    print(f"  Available: {[c for c in beh.columns if any(k in c for k in ['PC','texture','speed'])]}")
    PC_COLS        = [c for c in PC_COLS if c in beh.columns]
    PC_COLORS      = {k: v for k, v in PC_COLORS.items()      if k in PC_COLS}
    RMS_THRESHOLDS = {k: v for k, v in RMS_THRESHOLDS.items() if k in PC_COLS}
    ALIGN_MODES    = {k: v for k, v in ALIGN_MODES.items()    if k in PC_COLS}
    print(f"  Continuing with: {PC_COLS}")

beh_arr = beh[PC_COLS].values.astype(float)

dt        = float(np.median(np.diff(beh_t)))
FS        = 1.0 / dt
N_SAMPLES = int(round(WINDOW_SEC * FS))
t_axis    = np.arange(N_SAMPLES) / FS

print(f"Behaviour frames : {len(beh):,}")
print(f"Inferred FS      : {FS:.2f} Hz  →  {N_SAMPLES} samples per {WINDOW_SEC} s window")
print(f"Trajectories     : {len(trajectory_times_sec)} trials")

onsets    = trajectory_times_sec['onset_sec'].values
offsets   = trajectory_times_sec['offset_sec'].values
peaks     = trajectory_times_sec['peaks_sec'].values
durations = offsets - onsets
peak_lags = peaks - onsets

# %% ── DURATION HISTOGRAM (once) ──────────────────────────────────────────────
duration_mask = (durations >= MIN_DUR) & (durations <= MAX_DUR)
sel_idx       = np.where(duration_mask)[0]
print(f"\nTrials passing duration filter [{MIN_DUR}–{MAX_DUR} s]: "
      f"{len(sel_idx)} / {len(durations)}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
ax.hist(durations, bins=30, color='steelblue', edgecolor='white', lw=0.5)
ax.axvline(np.percentile(durations, 25), color='orange', ls='--', lw=1.5, label='Q1/Q3')
ax.axvline(np.percentile(durations, 75), color='orange', ls='--', lw=1.5)
ax.axvline(np.median(durations),         color='crimson', ls='-',  lw=1.5, label='Median')
ax.axvspan(MIN_DUR, MAX_DUR, color='steelblue', alpha=0.12, label=f'[{MIN_DUR}–{MAX_DUR} s]')
ax.set_xlabel('Duration (s)');  ax.set_ylabel('Count')
ax.set_title('Trial duration distribution');  ax.legend(fontsize=9)
ax2 = axes[1]
ax2.hist(peak_lags, bins=30, color='mediumseagreen', edgecolor='white', lw=0.5)
ax2.set_xlabel('Peak lag from onset (s)');  ax2.set_ylabel('Count')
ax2.set_title('Peak lag distribution')
plt.tight_layout()
savefig(fig, f"{animal}_{exp}_dur{MIN_DUR}-{MAX_DUR}s_duration_histogram.png")

all_bins = np.linspace(durations.min(), durations.max(), 30)
fig, ax  = plt.subplots(figsize=(8, 3))
ax.hist(durations,                bins=all_bins, color='lightgray', edgecolor='white', label='all')
ax.hist(durations[duration_mask], bins=all_bins, color='steelblue', edgecolor='white',
        alpha=0.8, label='selected')
ax.axvline(MIN_DUR, color='black', ls=':', lw=1.5)
ax.axvline(MAX_DUR, color='black', ls=':', lw=1.5)
ax.set_xlabel('Duration (s)')
ax.set_title(f'Duration filter [{MIN_DUR}–{MAX_DUR} s]  '
             f'— {len(sel_idx)} / {len(durations)} trials selected')
ax.legend()
plt.tight_layout()
savefig(fig, f"{animal}_{exp}_dur{MIN_DUR}-{MAX_DUR}s_filter_overlay.png")

# %% ── PER-PC LOOP ────────────────────────────────────────────────────────────
for pc_col in PC_COLS:
    pi         = PC_COLS.index(pc_col)
    clr        = PC_COLORS[pc_col]
    thr        = RMS_THRESHOLDS[pc_col]
    align_mode = ALIGN_MODES[pc_col]
    tag        = make_tag(pc_col, align_mode, thr)
    half       = WINDOW_SEC / 2.0

    x_labels = {
        'onset': 'Time from onset (s)',
        'peak':  f'Time relative to peak (s)  [peak at t={half:.1f}]',
    }

    print(f"\n{'='*60}")
    print(f"  {pc_col}  |  align={align_mode}  |  RMS threshold={thr}")
    print(f"{'='*60}")

    # ── extract windows ───────────────────────────────────────────────────────
    traces_onset     = np.full((len(sel_idx), N_SAMPLES), np.nan)
    traces_peak      = np.full((len(sel_idx), N_SAMPLES), np.nan)
    onset_win_starts = np.full(len(sel_idx), np.nan)
    peak_win_starts  = np.full(len(sel_idx), np.nan)
    peak_win_ends    = np.full(len(sel_idx), np.nan)
    valid_onset      = np.zeros(len(sel_idx), dtype=bool)
    valid_peak       = np.zeros(len(sel_idx), dtype=bool)

    for ii, tri in enumerate(sel_idx):
        t0_on = onsets[tri]
        mask  = (beh_t >= t0_on) & (beh_t < t0_on + WINDOW_SEC)
        chunk = beh_arr[mask, pi]
        if len(chunk) >= N_SAMPLES:
            traces_onset[ii]     = chunk[:N_SAMPLES]
            onset_win_starts[ii] = t0_on
            valid_onset[ii]      = True

        t0_pk = peaks[tri] - half
        t1_pk = peaks[tri] + half
        mask  = (beh_t >= t0_pk) & (beh_t < t1_pk)
        chunk = beh_arr[mask, pi]
        if len(chunk) >= N_SAMPLES:
            traces_peak[ii]     = chunk[:N_SAMPLES]
            peak_win_starts[ii] = t0_pk
            peak_win_ends[ii]   = t1_pk
            valid_peak[ii]      = True

    traces_on_v = traces_onset[valid_onset];  sel_idx_on = sel_idx[valid_onset]
    traces_pk_v = traces_peak[valid_peak];    sel_idx_pk = sel_idx[valid_peak]

    print(f"  Valid onset-aligned : {valid_onset.sum()} / {len(sel_idx)}")
    print(f"  Valid peak-aligned  : {valid_peak.sum()} / {len(sel_idx)}")

    # ── quick-look: all duration-filtered trials, both alignments ─────────────
    for traces_loop, mode, n_valid in [
        (traces_on_v, 'onset', valid_onset.sum()),
        (traces_pk_v, 'peak',  valid_peak.sum()),
    ]:
        fig, ax = plt.subplots(figsize=(11, 4))
        fig.suptitle(
            f'{animal} {exp} — {pc_col},  {mode}-aligned,  '
            f'dur [{MIN_DUR}–{MAX_DUR} s],  win {WINDOW_SEC} s  (n={n_valid})',
            fontsize=11,
        )
        for ti in range(len(traces_loop)):
            ax.plot(t_axis, traces_loop[ti], color=clr, alpha=0.25, lw=0.7)
        mean_tr = np.nanmean(traces_loop, axis=0)
        ax.plot(t_axis, mean_tr, color='black', lw=2.0, label='mean')
        ax.set_xlabel(x_labels[mode]);  ax.set_ylabel(pc_col)
        ax.legend(loc='upper right', fontsize=9);  ax.grid(alpha=0.15)
        plt.tight_layout()
        # savefig(fig,
        #         f"{animal}_{exp}_{pc_col}_{mode}_dur{MIN_DUR}-{MAX_DUR}s"
        #         f"_win{WINDOW_SEC}s_quicklook.png")

    # ── choose alignment ──────────────────────────────────────────────────────
    if align_mode == 'onset':
        traces_v     = traces_on_v
        sel_idx_v    = sel_idx_on
        win_starts_v = onset_win_starts[valid_onset]
        win_ends_v   = win_starts_v + WINDOW_SEC
    else:
        traces_v     = traces_pk_v
        sel_idx_v    = sel_idx_pk
        win_starts_v = peak_win_starts[valid_peak]
        win_ends_v   = peak_win_ends[valid_peak]

    # ── per-trial demean (remove DC offset so we compare shapes not levels) ────
    traces_demeaned = traces_v - np.nanmean(traces_v, axis=1, keepdims=True)

    # ── RMS deviation from group mean of demeaned traces ─────────────────────
    trial_scores = rms_from_mean(traces_demeaned)
    mean_trace   = np.nanmean(traces_demeaned, axis=0)

    # ── score distribution plot ───────────────────────────────────────────────
    score_norm = (trial_scores - trial_scores.min()) / (trial_scores.max() - trial_scores.min() + 1e-12)
    cmap_rms   = plt.cm.RdYlBu_r

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(
        f'{animal} {exp} — {pc_col},  {align_mode}-aligned,  '
        f'dur [{MIN_DUR}–{MAX_DUR} s],  win {WINDOW_SEC} s\n'
        f'Trials coloured by RMS deviation from group mean  (n={len(traces_v)})',
        fontsize=11,
    )

    ax = axes[0]
    draw_order = np.argsort(trial_scores)
    for ti in draw_order:
        ax.plot(t_axis, traces_demeaned[ti], color=cmap_rms(score_norm[ti]),
                alpha=0.6, lw=0.8)
    ax.plot(t_axis, mean_trace, color='black', lw=2.0, label='group mean (demeaned)')
    if thr is not None:
        ax.fill_between(t_axis, mean_trace - thr, mean_trace + thr,
                        color='black', alpha=0.10, label=f'±RMS thr ({thr})')
    sm = plt.cm.ScalarMappable(cmap=cmap_rms,
                               norm=plt.Normalize(trial_scores.min(),
                                                  trial_scores.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=f'RMS deviation (demeaned {pc_col} units)')
    ax.set_xlabel(x_labels[align_mode]);  ax.set_ylabel(f'{pc_col} (demeaned)')
    ax.set_title('Demeaned traces coloured by RMS score\n(red = most deviant shape)')
    ax.legend(fontsize=8);  ax.grid(alpha=0.15)

    ax2 = axes[1]
    ax2.hist(trial_scores, bins=20, color=clr, edgecolor='white', alpha=0.8)
    if thr is not None:
        ax2.axvline(thr, color='crimson', lw=2, label=f'threshold = {thr}')
        ax2.legend(fontsize=9)
    for pct in [25, 50, 75]:
        v = np.percentile(trial_scores, pct)
        ax2.axvline(v, color='orange', ls='--', lw=1, alpha=0.8)
        ax2.text(v, ax2.get_ylim()[1] * 0.93, f'P{pct}',
                 ha='center', fontsize=8, color='darkorange')
    ax2.set_xlabel(f'RMS deviation from group mean  ({pc_col} units)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'{pc_col} per-trial RMS score\n'
                  f'(lower = closer to group mean)')
    plt.tight_layout()
    savefig(fig, f"{tag}_rms_score_distribution.png")

    print(f"  RMS score — min={trial_scores.min():.4f}  "
          f"median={np.median(trial_scores):.4f}  "
          f"max={trial_scores.max():.4f}  "
          f"(units: {pc_col})")
    print(f"  P25={np.percentile(trial_scores,25):.4f}  "
          f"P50={np.percentile(trial_scores,50):.4f}  "
          f"P75={np.percentile(trial_scores,75):.4f}")

    if thr is None:
        print(f"  → RMS_THRESHOLDS['{pc_col}'] not set — skipping filter plots.")
        continue

    # ── apply threshold ───────────────────────────────────────────────────────
    keep_mask      = trial_scores <= thr
    traces_sel     = traces_v[keep_mask]
    sel_idx_sel    = sel_idx_v[keep_mask]
    win_starts_sel = win_starts_v[keep_mask]
    win_ends_sel   = win_ends_v[keep_mask]
    scores_sel     = trial_scores[keep_mask]

    print(f"  Threshold = {thr}  →  "
          f"{keep_mask.sum()} selected / {len(traces_v)} total")

    # ── before vs after: show demeaned traces ─────────────────────────────────
    traces_demeaned_sel = traces_demeaned[keep_mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=True, sharey=True)
    fig.suptitle(
        f'{animal} {exp} — {pc_col},  {align_mode}-aligned,  '
        f'win {WINDOW_SEC} s,  dur [{MIN_DUR}–{MAX_DUR} s],  '
        f'RMS threshold ≤ {thr}  (demeaned)',
        fontsize=11,
    )
    panels = [
        (traces_demeaned,     f'Before  (n={len(traces_v)})'),
        (traces_demeaned_sel, f'After  (n={keep_mask.sum()})'),
    ]
    for col_idx, (tr_set, title) in enumerate(panels):
        ax = axes[col_idx]
        m  = np.nanmean(tr_set, axis=0)
        for ti in range(len(tr_set)):
            ax.plot(t_axis, tr_set[ti], color=clr, alpha=0.25, lw=0.7)
        s = np.nanstd(tr_set, axis=0)
        ax.fill_between(t_axis, m - s, m + s, color=clr, alpha=0.20)
        ax.plot(t_axis, m, color='black', lw=2.0, label='mean ± 1SD')
        ax.fill_between(t_axis, m - thr, m + thr,
                        color='black', alpha=0.08,
                        label=f'±RMS thr ({thr})')
        ax.plot(t_axis, m + thr, color='black', lw=0.8, ls='--')
        ax.plot(t_axis, m - thr, color='black', lw=0.8, ls='--')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(x_labels[align_mode])
        ax.set_ylabel(f'{pc_col} (demeaned)' if col_idx == 0 else '')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.12)
    plt.tight_layout()
    savefig(fig, f"{tag}_before_vs_after_filter.png")

    # ── trial average ± SD (selected, raw values) ─────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f'{animal} {exp} — {pc_col},  {align_mode}-aligned,  '
        f'win {WINDOW_SEC} s,  dur [{MIN_DUR}–{MAX_DUR} s],  '
        f'RMS threshold ≤ {thr}\n'
        f'Trial average ± 1 SD  (n={keep_mask.sum()} selected)',
        fontsize=11,
    )
    m = np.nanmean(traces_sel, axis=0)
    s = np.nanstd(traces_sel,  axis=0)
    for ti in range(len(traces_sel)):
        ax.plot(t_axis, traces_sel[ti], color=clr, alpha=0.15, lw=0.7)
    ax.fill_between(t_axis, m - s, m + s, color=clr, alpha=0.25, label='±1 SD')
    ax.plot(t_axis, m,     color=clr, lw=2.2, label='mean')
    ax.plot(t_axis, m + s, color=clr, lw=0.8, ls='--')
    ax.plot(t_axis, m - s, color=clr, lw=0.8, ls='--')
    ax.set_xlabel(x_labels[align_mode]);  ax.set_ylabel(pc_col)
    ax.legend(loc='upper right', fontsize=9);  ax.grid(alpha=0.15)
    plt.tight_layout()
    savefig(fig, f"{tag}_trial_average.png")

    # ── export CSV ────────────────────────────────────────────────────────────
    out_df = pd.DataFrame({
        'trial_original_idx':           sel_idx_sel,
        'original_onset_sec':           onsets[sel_idx_sel],
        'original_offset_sec':          offsets[sel_idx_sel],
        'original_duration_sec':        durations[sel_idx_sel],
        'peak_sec':                     peaks[sel_idx_sel],
        'win_start_sec':                win_starts_sel,
        'win_end_sec':                  win_ends_sel,
        'win_duration_sec':             win_ends_sel - win_starts_sel,
        'peak_offset_from_win_start_sec': peaks[sel_idx_sel] - win_starts_sel,
        'rms_score':                    scores_sel,
        'pc':                           pc_col,
        'align_mode':                   align_mode,
        'rms_threshold':                thr,
    })

    out_path = (f"{behaviour_folder}/{animal}/{animal}_{exp}/"
                f"similar_trials_{pc_col}_{align_mode}_dur{MIN_DUR}s.csv")
    out_df.to_csv(out_path, index=False)
    print(f"  Exported {len(out_df)} trials → {out_path}")
    print(out_df[['trial_original_idx', 'original_onset_sec', 'win_start_sec',
                  'win_end_sec', 'peak_sec', 'peak_offset_from_win_start_sec',
                  'rms_score']].head())

print(f"\nAll done. Plots saved to {PLOT_DIR}")