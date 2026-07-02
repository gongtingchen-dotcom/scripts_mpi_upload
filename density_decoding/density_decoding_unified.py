# %% [markdown]
# #### **DENSITY DECODING — UNIFIED PIPELINE**
#
# Single script for all three run types, selected via --shuffle:
#   --shuffle none      : the real decode (no perturbation)
#   --shuffle shift      : circular-shift null — shifts spike times (neural) or
#                          rolls the behaviour trace within-trial (behavior)
#                          across the session, looped over --n_shuffles
#   --shuffle scramble    : bin-scramble null — permutes the TIME-BIN ORDER
#                          within each trial for spikes (neural) or behaviour
#                          (behavior), looped over --n_shuffles
# --shuffle_mode {neural,behavior} selects the submode; ignored when
# --shuffle=none.
#
# DESIGN PRINCIPLES (this replaces 3 separately-drifting scripts):
#   - ONE make_split() used identically by all three modes (correct tail
#     handling + a working --embargo_trials — both were previously broken
#     or unused in one or more of the source scripts).
#   - Spike-feature extraction (PTP/DREDge), spike/behaviour binning, and
#     all QC plots are recomputed from scratch every run — no
#     os.path.exists()-guarded skip-and-load anywhere. Final outputs are
#     still written to disk (for later inspection), just never read back
#     to skip computation.
#   - checkpoint_dir behavior for decode_pipeline is UNCHANGED from the
#     original scripts: it is NOT wiped before the call, for --shuffle
#     none or shift/scramble. If a GMM/init-weight-matrix/ADVI-epoch
#     checkpoint already exists in that directory (RUN_DIR for real runs,
#     shuffle_N/ for null runs), decode_pipeline will pick it up as it
#     always did. Delete the relevant directory yourself first if you
#     need a guaranteed-fresh fit (e.g. after changing the train/test
#     split, chunk_size, or embargo_trials for a directory that already
#     has old checkpoints in it).
#   - One `plot_feed_diagnostic()` used by all three modes: 5-column
#     orig-vs-decoder-feed comparison (spike scatter, spike bar/bin, raw
#     behaviour, decoder-feed spikes, decoder-feed behaviour). For
#     --shuffle none, feed == orig, so columns 4-5 should visually match
#     1-2 — a sanity check on binning/z-scoring/demeaning, not a
#     perturbation check. For shift/scramble, columns 4-5 show the actual
#     perturbed data fed to decode_pipeline.
#   - All other plots follow the real-run script's set (more
#     comprehensive than the old shuffle/scramble scripts): alignment
#     check, spike positions, localisation scatter, waveform QC, drift
#     scatter, behaviour overview, trial inspection, putative neurons,
#     trial diagnostics, decode traces (the last three: --shuffle none
#     only — they depend on post_params / per-trial detail that doesn't
#     map onto the null loop).
#
# SLURM array usage (shift/scramble): submit one --shuffle_index per job.

import sys, os, json, pickle, datetime, threading, shutil, glob

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

# ── pandas pickle compatibility monkeypatch ────────────────────────────────────
_orig_categorical_setstate = Categorical.__setstate__
def _patched_categorical_setstate(self, state):
    if not isinstance(state, dict):
        dtype, codes = state
        NDArrayBacked.__init__(self, codes, dtype)
    else:
        _orig_categorical_setstate(self, state)
Categorical.__setstate__ = _patched_categorical_setstate

# %% [markdown]
# #### **ARGS**

# %%
import argparse as _argparse
_ap = _argparse.ArgumentParser(description='Density decoding — unified pipeline')
_ap.add_argument('--animal',          default='GLC6962')
_ap.add_argument('--exp',             default='_g_g0')
_ap.add_argument('--behavior_var',    default='PC1')
_ap.add_argument('--t_trial',         type=float, default=5.0,
                 help='Trial window length (s). Ignored when --selected_trials is given '
                      '(window length is read from the CSV).')
_ap.add_argument('--bin_size_ms',     type=float, default=100.0)
_ap.add_argument('--learning_rate',   type=float, default=1e-3)
_ap.add_argument('--n_workers',       type=int,   default=16)
_ap.add_argument('--chunk_style',     choices=['contiguous', 'interleaved'], default='interleaved')
_ap.add_argument('--test_percentage', type=float, default=20)
_ap.add_argument('--chunk_size',      type=int,   default=10)
_ap.add_argument('--early_stopping',  action=_argparse.BooleanOptionalAction, default=True)
_ap.add_argument('--es_patience',     type=int,   default=500)
_ap.add_argument('--batch_size',      type=int,   default=10)
_ap.add_argument('--zscore_spike_features', action=_argparse.BooleanOptionalAction, default=True)
_ap.add_argument('--embargo_trials',  type=int,   default=0,
                 help='Trials dropped on EACH side of EVERY train<->test boundary '
                      '(both directions: train->test and test->train). 0 = off.')
_ap.add_argument('--animal_speed_gate', type=float, default=0.0)
_ap.add_argument('--selected_trials', default=None,
                 help='Path to a similar_trials CSV from trial_structure.py. '
                      'win_start_sec/win_end_sec define trial windows; T_TRIAL is '
                      'overridden from the CSV.')
_ap.add_argument('--demean_beh', action=_argparse.BooleanOptionalAction, default=False,
                 help='Subtract each trial\'s own mean from its binned behaviour '
                      'before decoding (applied after global z-scoring).')
# ── unified shuffle switch ─────────────────────────────────────────────────────
_ap.add_argument('--shuffle',       choices=['none', 'shift', 'scramble'], default='none',
                 help='none: real decode. shift: circular-shift null. '
                      'scramble: within-trial bin-permutation null.')
_ap.add_argument('--shuffle_mode',  choices=['neural', 'behavior'], default='neural',
                 help='Which side gets perturbed. Ignored when --shuffle=none.')
_ap.add_argument('--n_shuffles',    type=int,   default=100,
                 help='Number of null draws (--shuffle shift/scramble only).')
_ap.add_argument('--shuffle_index', type=int,   default=None,
                 help='Run only this single null-draw index (SLURM array mode).')
_ap.add_argument('--min_shift_sec', type=float, default=60.0,
                 help='--shuffle shift only: minimum circular shift (s).')
_ap.add_argument('--seed',          type=int,   default=42)
_args = _ap.parse_args()

SHUFFLE      = _args.shuffle
SHUFFLE_MODE = _args.shuffle_mode

# %% [markdown]
# #### **PATHS**

# %%
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
_selected_tag  = "_selected"
_demean_tag = "_demeanbeh" if _args.demean_beh else ""

BASE_RESULTS_DIR = os.path.join(
    BASE_DIR, "density_decoding",
    f"{_beh_key_dir}_{BIN_SIZE_MS:.3g}ms_{T_TRIAL:.3g}s_{_args.chunk_style}_"
    f"{'zscoreboth' if _args.zscore_spike_features else 'nozscore'}"
    f"_embargo{_args.embargo_trials}{_speedgate_tag}{_selected_tag}{_demean_tag}"
    f"_chunksize{_args.chunk_size}"
)
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

if SHUFFLE == 'none':
    RUN_DIR = BASE_RESULTS_DIR
else:
    RUN_DIR = BASE_RESULTS_DIR + f"_{SHUFFLE_MODE}_{SHUFFLE}_all"
os.makedirs(RUN_DIR, exist_ok=True)

print(f"Base results dir : {BASE_RESULTS_DIR}")
print(f"Run dir          : {RUN_DIR}")
print(f"Mode             : shuffle={SHUFFLE}" +
      (f"  shuffle_mode={SHUFFLE_MODE}" if SHUFFLE != 'none' else ""))

# %% [markdown]
# #### **LOAD AP RECORDING + BEHAVIOUR**

# %%
from spikeinterface import NumpyRecording
rec_si      = NumpyRecording.load(PREPROC_DIR)
ch_locs     = rec_si.get_channel_locations()
ap_fs       = rec_si.get_sampling_frequency()
ap_t_offset = float(rec_si._recording_segments[0].t_start)
n_ch        = rec_si.get_num_channels()
n_samp      = rec_si.get_num_samples()
ap_dtype    = rec_si.get_dtype()

beh     = texture_pca_df[(texture_pca_df['animal'] == animal) &
                          (texture_pca_df['background'] == 'eink')].copy()
beh_t   = beh['time'].dt.total_seconds().values
beh_var = beh[BEHAVIOR_VAR].values
beh_spd = beh['speed'].values if 'speed' in beh.columns else None

print(f"Recording : {n_ch} ch  |  {ap_fs:.2f} Hz  |  {n_samp:,} samples "
      f"({n_samp / ap_fs / 60:.1f} min)  |  dtype {ap_dtype}")
print(f"t_start   : {ap_t_offset:.3f} s (gate-relative)")
print(f"Behavioral rows : {len(beh_t):,}   range {beh_t.min():.3f}–{beh_t.max():.3f} s")

# %% [markdown]
# #### **SPIKE FEATURES — ALWAYS COMPUTED FROM SCRATCH (no cache read)**

# %%
print("\n[COMPUTE] Computing spike features from raw data (fresh every run) ...")
peaks = np.load(PEAKS_FILE)
loc   = np.load(LOC_FILE)

spike_times_samp = peaks['sample_index'].astype(np.int64)
spike_amp        = np.abs(peaks['amplitude']).astype(np.float32)
spike_x          = loc['x'].astype(np.float32)
spike_z          = loc['y'].astype(np.float32)
spike_t_gate     = spike_times_samp / ap_fs + ap_t_offset

print(f"AP sampling rate    : {ap_fs:.4f} Hz")
print(f"Gate offset         : {ap_t_offset:.3f} s")
print(f"Total peaks         : {len(spike_t_gate):,}")
print(f"spike_t_gate range  : {spike_t_gate.min():.2f} – {spike_t_gate.max():.2f} s")

N_BEFORE_PTP   = round(0.5 * ap_fs / 1000)
N_AFTER_PTP    = round(0.5 * ap_fs / 1000)
CHUNK_S_PTP    = 60.0
chunk_samp_ptp = int(CHUNK_S_PTP * ap_fs)
n_spikes       = len(peaks)
spike_ptp      = np.full(n_spikes, np.nan, dtype=np.float32)

print(f"Computing raw PTP  window=±{N_BEFORE_PTP} samples (±0.5 ms)  "
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

n_nan_ptp = int(np.isnan(spike_ptp).sum())
p999 = float(np.nanpercentile(spike_ptp, 99.9))
print(f"spike_ptp: min={np.nanmin(spike_ptp):.1f}  median={np.nanmedian(spike_ptp):.1f}  "
      f"99.9th={p999:.1f}  max={np.nanmax(spike_ptp):.1f}  µV  (NaN: {n_nan_ptp})")

motion        = si.Motion.load(MOTION_DIR)
temporal_bins = motion.temporal_bins_s[0]
spatial_bins  = motion.spatial_bins_um
print(f"Motion loaded from  : {MOTION_DIR}")
print(f"Temporal range      : {temporal_bins[0]:.2f} – {temporal_bins[-1]:.2f} s")

z_raw = loc['y'].astype(np.float32)
disp_at_spike = motion.get_displacement_at_time_and_depth(
    spike_t_gate, z_raw.astype(np.float64)).astype(np.float32)
z_corr = z_raw - disp_at_spike
print(f"z_raw range  : {z_raw.min():.1f} – {z_raw.max():.1f} µm")
print(f"z_corr range : {z_corr.min():.1f} – {z_corr.max():.1f} µm  "
      f"(median |correction|={np.median(np.abs(disp_at_spike)):.2f} µm)")

spike_channels = peaks['channel_index'].astype(np.int32)
spike_features_unscaled = np.column_stack([
    loc['x'].astype(np.float32), z_corr, spike_ptp,
])
spike_features = spike_features_unscaled.copy()

# still SAVE for later manual inspection — just never read back to skip compute
OUT_PATH = os.path.join(BASE_RESULTS_DIR, f'spike_features_{SESSION}.npz')
np.savez(
    OUT_PATH,
    times_samples=spike_times_samp, channels=spike_channels,
    point_source_localizations=spike_features_unscaled,
    t_start_gate_s=np.float64(ap_t_offset), sampling_rate=np.float64(ap_fs),
)
print(f"Saved → {OUT_PATH}  ({os.path.getsize(OUT_PATH)/1e6:.0f} MB)")

# %% [markdown]
# #### **QC PLOTS — always regenerated**

# %%
_align_save = os.path.join(BASE_RESULTS_DIR, f'alignment_check_{SESSION}.png')
print("[PLOT] Alignment check ...")
BIN_ALIGN = 1.0
t_lo = min(spike_t_gate.min(), beh_t.min()) - 2
t_hi = max(spike_t_gate.max(), beh_t.max()) + 2
bins_1s = np.arange(t_lo, t_hi + BIN_ALIGN, BIN_ALIGN)
t_ctr   = (bins_1s[:-1] + bins_1s[1:]) / 2
mua_counts, _ = np.histogram(spike_t_gate, bins=bins_1s)
mua_hz = uniform_filter1d(mua_counts.astype(float) / BIN_ALIGN, size=5)

def _bin_signal(t_arr, sig, bins):
    out = np.full(len(bins) - 1, np.nan)
    for i in range(len(bins) - 1):
        m = (t_arr >= bins[i]) & (t_arr < bins[i + 1])
        if m.any():
            out[i] = np.nanmean(sig[m])
    return out

beh_var_1s = _bin_signal(beh_t, beh_var, bins_1s)
spd_1s = _bin_signal(beh_t, beh_spd, bins_1s) if beh_spd is not None else None

def _norm01(x):
    mn, mx = np.nanmin(x), np.nanmax(x)
    return (x - mn) / (mx - mn + 1e-9)

n_rows = 2 + (spd_1s is not None)
fig, axes = plt.subplots(n_rows + 1, 1, figsize=(22, 4 * (n_rows + 1)), sharex=True)
fig.suptitle(f"{SESSION}  ·  Alignment check  ·  Common clock: SpikeGLX gate-open",
             fontsize=11, y=1.002)
axes[0].fill_between(t_ctr, mua_hz, alpha=0.7, color='steelblue')
axes[0].set_ylabel("MUA\n(sp / s)", fontsize=9)
axes[0].set_title("All KS4 spikes — 1 s bins, 5 s smoothed", fontsize=8, loc='left', pad=2)
axes[1].plot(t_ctr, beh_var_1s, color='darkorange', lw=0.8)
axes[1].set_ylabel(f"{BEHAVIOR_VAR}\n(1 s mean)", fontsize=9)
row = 2
if spd_1s is not None:
    axes[row].plot(t_ctr, spd_1s, color='forestgreen', lw=0.8)
    axes[row].set_ylabel("Speed\n(1 s mean)", fontsize=9)
    row += 1
ax_ov = axes[row]
ax_ov.plot(t_ctr, _norm01(mua_hz), color='steelblue', lw=1.0, alpha=0.85, label='MUA (norm)')
ax_ov.plot(t_ctr, _norm01(beh_var_1s), color='darkorange', lw=0.8, alpha=0.85, label=f'{BEHAVIOR_VAR} (norm)')
if spd_1s is not None:
    ax_ov.plot(t_ctr, _norm01(spd_1s), color='forestgreen', lw=0.8, alpha=0.85, label='Speed (norm)')
ax_ov.set_ylabel("Normalised\n[0 – 1]", fontsize=9)
ax_ov.legend(fontsize=8, loc='upper right')
ax_ov.set_title("Overlay — use this to judge temporal alignment", fontsize=8, loc='left', pad=2)
beh_patch = mpatches.Patch(color='gold', alpha=0.35,
                            label=f"Behavioral  {beh_t.min():.0f}–{beh_t.max():.0f} s")
neu_patch = mpatches.Patch(color='violet', alpha=0.35,
                            label=f"Neural      {ap_t_offset:.0f}–{spike_t_gate.max():.0f} s")
for ax in axes:
    ax.axvspan(beh_t.min(), beh_t.max(), color='gold', alpha=0.10, zorder=0)
    ax.axvspan(ap_t_offset, spike_t_gate.max(), color='violet', alpha=0.08, zorder=0)
    ax.grid(True, alpha=0.25, lw=0.5)
axes[0].legend(handles=[beh_patch, neu_patch], fontsize=8, loc='upper right')
axes[-1].set_xlabel("Time from SpikeGLX gate open (s)", fontsize=9)
plt.tight_layout()
plt.savefig(_align_save, dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved → {_align_save}")

T_OV_START = max(beh_t.min(), ap_t_offset)
T_OV_END   = min(beh_t.max(), spike_t_gate.max())
T_OV_DUR   = T_OV_END - T_OV_START
print(f"Overlap window: {T_OV_START:.2f} – {T_OV_END:.2f} s  ({T_OV_DUR/60:.2f} min)")

# ── spike positions ────────────────────────────────────────────────────────────
_pos_save = os.path.join(BASE_RESULTS_DIR, 'spike_positions.png')
print("[PLOT] Spike positions ...")
probe_x_min = ch_locs[:, 0].min(); probe_x_max = ch_locs[:, 0].max()
T_START, T_WIN = 0, 5
mask = (spike_t_gate >= T_START + ap_t_offset) & (spike_t_gate < T_START + T_WIN + ap_t_offset)
sx, sz, sa = spike_x[mask], z_corr[mask], spike_amp[mask]
fig, ax = plt.subplots(figsize=(6, 14))
fig.suptitle(f"Spike positions  ·  t = {T_START}–{T_START+T_WIN:.3f} s", fontsize=10)
ax.scatter(ch_locs[:, 0], ch_locs[:, 1], s=18, c='lightgray',
           edgecolors='gray', linewidths=0.5, zorder=2, label='Electrodes')
ax.axvspan(probe_x_min, probe_x_max, color='dodgerblue', alpha=0.06, zorder=0)
ax.axvline(probe_x_min, color='dodgerblue', lw=1, alpha=0.6, ls='--', label='Probe x extent')
ax.axvline(probe_x_max, color='dodgerblue', lw=1, alpha=0.6, ls='--')
if mask.any():
    amp_vmax = np.percentile(spike_amp, 95)
    sc_h = ax.scatter(sx, sz, c=sa, s=50, cmap='hot_r', vmin=0, vmax=amp_vmax,
                       edgecolors='k', linewidths=0.2, zorder=5, label=f'Spikes (n={mask.sum()})')
    plt.colorbar(sc_h, ax=ax, label='Amplitude (µV)', shrink=0.55)
ax.set_xlabel('Lateral x (µm)'); ax.set_ylabel('Depth z (µm)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(_pos_save, dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved → {_pos_save}")

# ── localisation scatter ────────────────────────────────────────────────────────
_loc_save = os.path.join(BASE_RESULTS_DIR, f'spike_localisations_{SESSION}.png')
print("[PLOT] Localisation scatter ...")
rng_sf = np.random.default_rng(0)
n_show = min(200_000, len(spike_features_unscaled))
isf = rng_sf.choice(len(spike_features_unscaled), n_show, replace=False)
isf = isf[np.argsort(spike_ptp[isf])]
fig, ax = plt.subplots(figsize=(4, 10))
sc = ax.scatter(spike_features_unscaled[isf, 0], spike_features_unscaled[isf, 1],
                 c=np.clip(spike_ptp[isf], 0, p999), s=0.3, alpha=0.3, cmap='hot', rasterized=True)
plt.colorbar(sc, ax=ax, label='PTP (µV)')
ax.set_xlabel('x (µm)'); ax.set_ylabel('depth z (µm)')
ax.set_title(f'{SESSION} — localisations  ({n_show:,} random)')
plt.tight_layout()
plt.savefig(_loc_save, dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved → {_loc_save}")

# ── waveform QC grid + drift scatter + probe-layout waveform QC ───────────────
print("[PLOT] Waveform grid ...")
_wf_val_save = os.path.join(BASE_RESULTS_DIR, f'ptp_waveforms_{SESSION}.png')
N_BEFORE_G = round(0.5 * ap_fs / 1000); N_AFTER_G = round(0.5 * ap_fs / 1000)
wf_len_val = N_BEFORE_G + N_AFTER_G
rng_val = np.random.default_rng(42)
val_idx = rng_val.choice(len(peaks), 100, replace=False)
wf_val  = np.full((100, wf_len_val), np.nan, dtype=np.float32)
ptp_val = spike_ptp[val_idx].copy()
for k, idx in enumerate(val_idx):
    pk_ch = int(peaks['channel_index'][idx]); samp = int(spike_times_samp[idx])
    t0 = max(0, samp - N_BEFORE_G); t1 = min(n_samp, samp + N_AFTER_G)
    tr = rec_si.get_traces(start_frame=t0, end_frame=t1)[:, pk_ch].astype(np.float32)
    if tr.shape[0] < wf_len_val:
        tr = np.pad(tr, (0, wf_len_val - tr.shape[0]))
    wf_val[k] = tr
order_val = np.argsort(ptp_val)[::-1]
t_ax_val  = np.arange(wf_len_val) - N_BEFORE_G
y_half    = np.nanpercentile(ptp_val, 99) * 0.7
ylim_val  = (-y_half, y_half)
ncols_val, nrows_val = 10, 10
fig2, axes2 = plt.subplots(nrows_val, ncols_val, figsize=(ncols_val * 1.8, nrows_val * 1.8),
                            sharex=True, sharey=True)
fig2.suptitle('Waveforms on detection channel — 100 random spikes, ordered by descending PTP',
              fontsize=10)
for pi, k in enumerate(order_val):
    r, c = divmod(pi, ncols_val); ax2 = axes2[r, c]; wf = wf_val[k]
    ax2.plot(t_ax_val, wf, color='k', lw=0.8)
    i_max = int(np.argmax(wf)); i_min = int(np.argmin(wf))
    ax2.annotate('', xy=(t_ax_val[i_max], wf[i_max]),
                 xytext=(t_ax_val[i_max], wf[i_max] + y_half * 0.28),
                 arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.0))
    ax2.annotate('', xy=(t_ax_val[i_min], wf[i_min]),
                 xytext=(t_ax_val[i_min], wf[i_min] - y_half * 0.28),
                 arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.0))
    ax2.axhline(ptp_val[k] / 2, color='darkorange', lw=0.8, ls='--', alpha=0.85)
    ax2.axhline(-ptp_val[k] / 2, color='darkorange', lw=0.8, ls='--', alpha=0.85)
    ax2.axvline(0, color='gray', lw=0.5, ls=':')
    ax2.set_title(f'ptp={ptp_val[k]:.0f} µV', fontsize=5.5, pad=1)
    ax2.tick_params(labelsize=5)
for ax2 in axes2.flat:
    ax2.set_ylim(*ylim_val)
plt.tight_layout()
plt.savefig(_wf_val_save, dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved → {_wf_val_save}")

print("[PLOT] Drift scatter before/after ...")
_scatter_save = os.path.join(BASE_RESULTS_DIR, f'drift_scatter_before_after_{SESSION}.png')
T_VIZ_S = 12 * 60; MAX_PTS = 200_000
t_rec_start = spike_t_gate.min(); t_rec_end = spike_t_gate.max()
windows = [
    (f'First 12 min  ({t_rec_start:.0f}–{t_rec_start+T_VIZ_S:.0f} s)', t_rec_start, t_rec_start + T_VIZ_S),
    (f'Last 12 min   ({t_rec_end-T_VIZ_S:.0f}–{t_rec_end:.0f} s)', t_rec_end - T_VIZ_S, t_rec_end),
]
fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharey=True, sharex='row')
fig.suptitle(f'{SESSION} — Spike depth vs time\nLeft: raw  |  Right: DREDge corrected', fontsize=10)
rng_sc = np.random.default_rng(7)
disp_cmap = plt.cm.plasma; disp_norm = plt.Normalize(spatial_bins[0], spatial_bins[-1])
for row_i, (win_label, t0, t1) in enumerate(windows):
    sel = np.where((spike_t_gate >= t0) & (spike_t_gate < t1))[0]
    if len(sel) > MAX_PTS:
        sel = rng_sc.choice(sel, MAX_PTS, replace=False)
        sel = sel[np.argsort(spike_t_gate[sel])]
    t_sel = spike_t_gate[sel]
    t_disp = (temporal_bins >= t0) & (temporal_bins <= t1)
    t_line = temporal_bins[t_disp]
    for col_i, (z_arr, panel_title) in enumerate([(z_raw[sel], 'Before correction'),
                                                    (z_corr[sel], 'After DREDge correction')]):
        ax = axes[row_i, col_i]
        ax.scatter(t_sel, z_arr, c='gray', s=0.3, alpha=0.25, rasterized=True, linewidths=0)
        if col_i == 0:
            for k, z_bin in enumerate(spatial_bins):
                apparent = z_bin + motion.displacement[0][t_disp, k]
                ax.plot(t_line, apparent, color=disp_cmap(disp_norm(z_bin)), lw=1.5, alpha=0.85, zorder=5)
            sm = plt.cm.ScalarMappable(cmap=disp_cmap, norm=disp_norm); sm.set_array([])
            cb = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
            cb.set_label('Spatial bin depth (µm)', fontsize=7); cb.ax.tick_params(labelsize=6)
        ax.set_title(f'{win_label}  —  {panel_title}', fontsize=9)
        ax.set_xlabel('Time from gate open (s)', fontsize=8)
        if col_i == 0:
            ax.set_ylabel('Depth (µm)', fontsize=8)
        ax.grid(True, alpha=0.12, lw=0.4)
plt.tight_layout()
plt.savefig(_scatter_save, dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved → {_scatter_save}")

print("[PLOT] Waveform QC probe-layout ...")
_wf_save = os.path.join(BASE_RESULTS_DIR, f'waveform_qc_{SESSION}.png')
T_POOL_START, T_POOL_END, N_SHOW = 100.0, 110.0, 5
N_BEFORE, N_AFTER, N_NBR = 20, 62, 32
pool_mask = (spike_t_gate >= T_POOL_START) & (spike_t_gate < T_POOL_END)
pool_idx  = np.where(pool_mask)[0]
if len(pool_idx) < N_SHOW:
    print(f"[WARN] Not enough spikes in pooling window ({len(pool_idx)}) — skipping waveform QC")
else:
    rng = np.random.default_rng(2026)
    sel = rng.choice(pool_idx, N_SHOW, replace=False)
    wf_len = N_BEFORE + N_AFTER; n_total_samp = rec_si.get_num_samples()
    sort_depths = np.sort(np.unique(ch_locs[:, 1]))
    elec_spacing = float(np.median(np.diff(sort_depths)))
    col_xs = np.sort(np.unique(ch_locs[:, 0]))
    col_gap = float(np.min(np.diff(col_xs))) if len(col_xs) > 1 else elec_spacing
    t_um = np.linspace(-col_gap * 0.45, col_gap * 0.45, wf_len)
    all_wfs = []; all_nbr = []
    for _spi in sel:
        pk_ch = int(peaks['channel_index'][_spi])
        dists = np.linalg.norm(ch_locs - ch_locs[pk_ch], axis=1)
        nbr = np.argsort(dists)[:N_NBR]
        samp = int(spike_times_samp[_spi])
        t0 = max(0, samp - N_BEFORE); t1 = min(n_total_samp, samp + N_AFTER)
        tr = rec_si.get_traces(start_frame=t0, end_frame=t1)[:, nbr].astype(np.float32)
        if tr.shape[0] < wf_len:
            tr = np.pad(tr, ((0, wf_len - tr.shape[0]), (0, 0)))
        all_wfs.append(tr); all_nbr.append(nbr)
    all_ptp = np.concatenate([np.ptp(w, axis=0) for w in all_wfs])
    global_ptp = float(np.percentile(all_ptp, 99))
    AMP_SCALE = (0.7 * elec_spacing) / max(global_ptp, 1.0)
    fig, axes = plt.subplots(1, N_SHOW, figsize=(N_SHOW * 5, 12), sharey=True, sharex=True)
    fig.suptitle(f"Waveform QC (post-DREDge) — {SESSION}", fontsize=10)
    for pi, (_spi, wf, nbr) in enumerate(zip(sel, all_wfs, all_nbr)):
        ax = axes[pi]; pk_ch = int(peaks['channel_index'][_spi])
        px = float(spike_x[_spi]); pz = float(z_corr[_spi])
        ax.scatter(ch_locs[:, 0], ch_locs[:, 1], s=8, c='lightgray', edgecolors='none', zorder=1, rasterized=True)
        for j, ch_j in enumerate(nbr):
            xc, yc = ch_locs[ch_j, 0], ch_locs[ch_j, 1]
            ax.plot(xc + t_um, yc + wf[:, j] * AMP_SCALE, lw=1.2, color='royalblue', alpha=0.95, zorder=3)
        ax.scatter(ch_locs[pk_ch, 0], ch_locs[pk_ch, 1], s=60, c='darkorange', marker='D', zorder=5,
                   label='Peak ch' if pi == 0 else None)
        ax.axhline(pz, color='red', lw=1.1, ls='--', alpha=0.7, zorder=4)
        ax.axvline(px, color='red', lw=1.1, ls='--', alpha=0.7, zorder=4)
        ax.scatter(px, pz, s=150, c='red', marker='*', zorder=6, label='Triangulated' if pi == 0 else None)
        ax.set_title(f"spike {pi+1}  det_ch={pk_ch}\n{spike_amp[_spi]:.0f} µV  x={px:.0f}  z={pz:.0f} µm",
                     fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.18)
        if pi == 0:
            ax.legend(fontsize=8, loc='upper right')
    axes[0].set_ylabel('Depth (µm)', fontsize=9)
    for ax in axes:
        ax.set_xlabel('Lateral x (µm)', fontsize=9)
    plt.tight_layout()
    plt.savefig(_wf_save, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved → {_wf_save}")

# %% [markdown]
# #### **FEATURE Z-SCORING**

# %%
ZSCORE_SPIKE_FEATURES = _args.zscore_spike_features
if ZSCORE_SPIKE_FEATURES:
    _sf_mean = spike_features.mean(axis=0)
    _sf_std  = spike_features.std(axis=0)
    _sf_std[_sf_std < 1e-9] = 1.0
    spike_features = ((spike_features - _sf_mean) / _sf_std).astype(np.float32)
    print(f"[ZSCORE] Spike features z-scored.  means={_sf_mean}  stds={_sf_std}")
    _zscore_params_sf = {'x_mean': float(_sf_mean[0]), 'x_std': float(_sf_std[0]),
                          'z_mean': float(_sf_mean[1]), 'z_std': float(_sf_std[1]),
                          'ptp_mean': float(_sf_mean[2]), 'ptp_std': float(_sf_std[2])}
else:
    print("[ZSCORE] Spike features NOT z-scored.")
    _zscore_params_sf = None

# %% [markdown]
# #### **BEHAVIOURAL DATA**

# %%
beh_raw = beh[BEHAVIOR_VAR].values.astype(np.float32)
if Z_SCORE_BEH:
    beh_z = zscore(beh_raw, nan_policy='omit').astype(np.float32)
    print(f"[ZSCORE] {BEHAVIOR_VAR}: mean={np.nanmean(beh_z):.4f}  std={np.nanstd(beh_z):.4f}")
else:
    beh_z = beh_raw.copy()
    print(f"[ZSCORE] {BEHAVIOR_VAR} NOT z-scored.")

# %% [markdown]
# #### **TRIAL WINDOWS**

# %%
USING_SELECTED_TRIALS = _args.selected_trials is not None
if USING_SELECTED_TRIALS:
    _sel_df = pd.read_csv(_args.selected_trials)
    _required_cols = {'win_start_sec', 'win_end_sec', 'win_duration_sec'}
    _missing = _required_cols - set(_sel_df.columns)
    if _missing:
        raise ValueError(f"--selected_trials CSV is missing columns: {_missing}.")
    trial_starts = _sel_df['win_start_sec'].values
    trial_ends   = _sel_df['win_end_sec'].values
    _win_durations = trial_ends - trial_starts
    T_TRIAL  = float(np.median(_win_durations))
    N_T_BINS = max(1, round(T_TRIAL * 1000 / BIN_SIZE_MS))
    _in_overlap = (trial_starts >= T_OV_START) & (trial_ends <= T_OV_END)
    if (~_in_overlap).any():
        print(f"[WARN] {(~_in_overlap).sum()} selected trials fall outside the overlap window — dropped.")
        trial_starts = trial_starts[_in_overlap]
        trial_ends   = trial_ends[_in_overlap]
        _sel_df      = _sel_df[_in_overlap].reset_index(drop=True)
    n_trials_beh = len(trial_starts)
    _sel_meta = {
        'source_csv': _args.selected_trials, 'n_trials': int(n_trials_beh),
        'align_mode': str(_sel_df['align_mode'].iloc[0]) if 'align_mode' in _sel_df.columns else 'unknown',
        'rms_threshold': float(_sel_df['rms_threshold'].iloc[0]) if 'rms_threshold' in _sel_df.columns else None,
        'pc': str(_sel_df['pc'].iloc[0]) if 'pc' in _sel_df.columns else 'unknown',
        'T_TRIAL_s': float(T_TRIAL),
    }
    print(f"[SELECTED TRIALS] {n_trials_beh} trials  |  T_TRIAL={T_TRIAL:.4f}s  N_T_BINS={N_T_BINS}")
else:
    trial_starts = np.arange(T_OV_START, T_OV_END - T_TRIAL, T_TRIAL)
    trial_ends   = trial_starts + T_TRIAL
    n_trials_beh = len(trial_starts)
    _sel_meta    = None
    print(f"[REGULAR GRID] {n_trials_beh} sliding windows of {T_TRIAL} s")

CHUNK_STYLE     = _args.chunk_style
CHUNK_SIZE      = _args.chunk_size
TEST_PERCENTAGE = _args.test_percentage
DEMEAN_BEH      = _args.demean_beh
TEST_FRAC       = TEST_PERCENTAGE / 100.0
TRAIN_FRAC      = 1.0 - TEST_FRAC
EMBARGO_TRIALS  = _args.embargo_trials

if TEST_PERCENTAGE <= 0 or TEST_PERCENTAGE >= 100:
    raise ValueError(f"test_percentage must be in (0, 100), got {TEST_PERCENTAGE}")

_VAR_KEY = BEHAVIOR_VAR.lower().replace(' ', '_')

# ── params JSON (initial) ───────────────────────────────────────────────────────
_params_json = os.path.join(RUN_DIR, f'decode_params_{_VAR_KEY}_{SESSION}.json')
_params_ts   = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
_params = {
    "session": SESSION, "started": _params_ts, "shuffle": SHUFFLE,
    "shuffle_mode": SHUFFLE_MODE if SHUFFLE != 'none' else None,
    "decode": {
        "BEHAVIOR_VAR": BEHAVIOR_VAR, "Z_SCORE_BEH": Z_SCORE_BEH,
        "T_TRIAL_s": float(T_TRIAL), "bin_size_ms": float(BIN_SIZE_MS), "N_T_BINS": int(N_T_BINS),
        "LEARNING_RATE": float(LEARNING_RATE), "N_WORKERS": int(N_WORKERS),
        "chunk_style": CHUNK_STYLE, "test_percentage": float(TEST_PERCENTAGE),
        "embargo_trials": int(EMBARGO_TRIALS), "animal_speed_gate": float(_args.animal_speed_gate),
        "zscore_spike_features": ZSCORE_SPIKE_FEATURES, "spike_feature_zscore_params": _zscore_params_sf,
        "using_selected_trials": USING_SELECTED_TRIALS, "selected_trials_meta": _sel_meta,
        "demean_beh": DEMEAN_BEH,
    },
}
with open(_params_json, 'w') as _f:
    json.dump(_params, _f, indent=2)
print(f"[PARAMS] Saved → {_params_json}")

# ── behaviour overview plot ──────────────────────────────────────────────────────
_beh_plot_save = os.path.join(RUN_DIR, f'behaviour_overview_{_VAR_KEY}_{SESSION}.png')
print(f"[PLOT] Behaviour overview → {_beh_plot_save}")
fig, ax = plt.subplots(figsize=(18, 2.5))
ax.plot(beh_t, beh_z, lw=0.4, color='tab:blue', alpha=0.7,
        label=f"{BEHAVIOR_VAR}{'  (z-scored)' if Z_SCORE_BEH else ''} [session-wide]")
ax.axvspan(T_OV_START, T_OV_END, color='violet', alpha=0.08, label='Neural-behavioral overlap')
if USING_SELECTED_TRIALS:
    for _ts_mark, _te_mark in zip(trial_starts, trial_ends):
        ax.axvspan(_ts_mark, _te_mark, color='gold', alpha=0.25, zorder=0)
    ax.axvspan(0, 0, color='gold', alpha=0.5, label='Selected trial windows')
ax.set_xlabel("Time from SpikeGLX gate open (s)", fontsize=9)
ax.set_ylabel(BEHAVIOR_VAR, fontsize=9)
ax.set_title(f"Behaviour: {BEHAVIOR_VAR} ({animal} eink)  [shuffle={SHUFFLE}]", fontsize=9)
ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(_beh_plot_save, dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved → {_beh_plot_save}")

# %% [markdown]
# #### **BIN SPIKES + BEHAVIOUR — always computed fresh**

# %%
sys.path.insert(0, '/u/gongt/density_decoding')
from density_decoding.utils.data_utils import BaseDataLoader
from density_decoding.decode_pipeline import decode_pipeline
from density_decoding.decoders.behavior_decoder import sliding_window_decoder

data_loader = BaseDataLoader(t_before=0.0, t_after=T_TRIAL, n_t_bins=N_T_BINS)

def make_bin_sf(spike_t):
    """Bin spike features for the given spike-times array over ALL n_trials_beh windows."""
    return data_loader.process_spike_features(
        spike_times=spike_t, spike_channels=spike_channels, spike_features=spike_features,
        trial_start_times=trial_starts, trial_end_times=trial_ends,
        valid_trials=np.arange(n_trials_beh),
    )

print("Binning spike features (ALL windows) ...")
bin_spike_features_all, bin_trial_idxs_all, bin_time_idxs_all = make_bin_sf(spike_t_gate)
print(f"  Done — {len(bin_spike_features_all)} windows binned")

print(f"Binning behaviour ({BEHAVIOR_VAR}) ...")
bin_beh_all = data_loader.process_behaviors(
    time_points=beh_t, raw_behaviors=beh_z,
    trial_start_times=trial_starts, trial_end_times=trial_ends,
)
valid_mask_beh   = ~np.isnan(bin_beh_all).any(axis=1)
valid_trials_beh = np.where(valid_mask_beh)[0]
print(f"Valid windows (no NaN in {BEHAVIOR_VAR}): {valid_mask_beh.sum()} / {n_trials_beh}")

ANIMAL_SPEED_GATE = _args.animal_speed_gate
_speed_gate_info = dict(used=False)
if ANIMAL_SPEED_GATE > 0 and beh_spd is not None:
    _win_spd = np.array([
        np.nanmean(beh_spd[(beh_t >= trial_starts[k]) & (beh_t < trial_ends[k])])
        for k in range(n_trials_beh)
    ])
    _speed_gate_mask = _win_spd < ANIMAL_SPEED_GATE
    valid_mask_beh   = valid_mask_beh & _speed_gate_mask
    valid_trials_beh = np.where(valid_mask_beh)[0]
    _speed_gate_info = dict(used=True, threshold=float(ANIMAL_SPEED_GATE),
                            n_kept=int(valid_mask_beh.sum()), n_total=int(n_trials_beh))
    print(f"[SPEED GATE] threshold={ANIMAL_SPEED_GATE:.4g} → {valid_mask_beh.sum()} / {n_trials_beh} kept")

bin_beh_v = bin_beh_all[valid_mask_beh]
if DEMEAN_BEH:
    bin_beh_v = bin_beh_v - bin_beh_v.mean(axis=1, keepdims=True)
    print(f"[DEMEAN_BEH] Subtracted per-trial mean.")

bin_sf_v = [bin_spike_features_all[k] for k in valid_trials_beh]
bin_ti_v, bin_tmi_v = [], []
for new_k, k in enumerate(valid_trials_beh):
    for t in range(N_T_BINS):
        old = bin_trial_idxs_all[k * N_T_BINS + t]
        bin_ti_v.append(np.full_like(old, new_k))
        bin_tmi_v.append(bin_time_idxs_all[k * N_T_BINS + t])

n_valid = len(valid_trials_beh)
print(f"Valid trials: {n_valid}")

trial_starts_valid = trial_starts[valid_trials_beh]
z_corr_plot = spike_features_unscaled[:, 1]   # depth, never z-scored — for diagnostic plots

# %% [markdown]
# #### **TRAIN/TEST SPLIT — shared by all three modes**

# %%
def make_split(n_valid, embargo=0):
    """
    Fixed train/test split. Tail policy: leftover trials smaller than the
    per-chunk train count all go to train; if the tail exceeds the train
    count, the first train_pc trials go to train and the remainder go to
    test. Embargo: drop `embargo` trials on EACH side of EVERY train<->test
    boundary (both directions), excluded from both sets. embargo=0
    reproduces the un-embargoed split.
    """
    cs = CHUNK_STYLE
    if cs == 'interleaved' and n_valid < CHUNK_SIZE:
        cs = 'contiguous'
    label = np.full(n_valid, -1, dtype=np.int8)
    if cs == 'interleaved':
        _n_test_pc  = max(1, int(round(CHUNK_SIZE * TEST_FRAC)))
        _n_train_pc = CHUNK_SIZE - _n_test_pc
        n_full = n_valid // CHUNK_SIZE
        for c in range(n_full):
            base = c * CHUNK_SIZE
            label[base : base + _n_train_pc] = 0
            label[base + _n_train_pc : base + CHUNK_SIZE] = 1
        tail_start = n_full * CHUNK_SIZE
        tail_len = n_valid - tail_start
        if tail_len > 0:
            if tail_len <= _n_train_pc:
                label[tail_start:n_valid] = 0
            else:
                label[tail_start : tail_start + _n_train_pc] = 0
                label[tail_start + _n_train_pc : n_valid] = 1
    else:
        _n_train = int(n_valid * TRAIN_FRAC)
        label[:_n_train] = 0
        label[_n_train:] = 1

    drop = np.zeros(n_valid, dtype=bool)
    if embargo > 0:
        boundaries = np.where(label[:-1] != label[1:])[0]
        for i in boundaries:
            lo = max(0, i - embargo + 1)
            hi = min(n_valid, i + 1 + embargo)
            drop[lo:hi] = True

    train_idx = np.where((label == 0) & (~drop))[0].astype(np.int64)
    test_idx  = np.where((label == 1) & (~drop))[0].astype(np.int64)
    return train_idx, test_idx

train_idx_fixed, test_idx_fixed = make_split(n_valid, embargo=EMBARGO_TRIALS)
_n_embargoed = n_valid - len(train_idx_fixed) - len(test_idx_fixed)
print(f"Train/test: {len(train_idx_fixed)} train | {len(test_idx_fixed)} test | "
      f"{_n_embargoed} embargoed (fixed, embargo_trials={EMBARGO_TRIALS})")

if len(test_idx_fixed) == 0 or len(train_idx_fixed) == 0:
    _n_test_pc_est = max(1, int(round(CHUNK_SIZE * TEST_FRAC)))
    _n_train_pc_est = CHUNK_SIZE - _n_test_pc_est
    raise ValueError(
        f"embargo_trials={EMBARGO_TRIALS} eliminates the entire "
        f"{'test' if len(test_idx_fixed) == 0 else 'train'} set "
        f"(chunk_size={CHUNK_SIZE}, test_percentage={TEST_PERCENTAGE:.3g}% "
        f"→ ~{_n_test_pc_est} test / ~{_n_train_pc_est} train trials per chunk). "
        f"Increase --chunk_size, adjust --test_percentage, or reduce --embargo_trials."
    )

# %% [markdown]
# #### **GPU**

# %%
import torch
print("=" * 60)
print("GPU / DEVICE CHECK")
print(f"  torch version  : {torch.__version__}")
print(f"  CUDA available : {torch.cuda.is_available()}")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"  -> using device: {device}")
print("=" * 60)

# %% [markdown]
# #### **PERTURBATION HELPERS (--shuffle shift / scramble)**

# %%
def circular_shift_spikes(spike_t_arr, shift_sec, t_start, t_end):
    """Circularly shift spike times by shift_sec within [t_start, t_end]."""
    dur  = t_end - t_start
    mask = (spike_t_arr >= t_start) & (spike_t_arr < t_end)
    shifted = spike_t_arr.copy()
    shifted[mask] = t_start + (spike_t_arr[mask] - t_start + shift_sec) % dur
    return shifted

def circular_shift_behavior(bin_beh, shift_bins):
    """
    Roll the behaviour trace by shift_bins WITHIN each trial (axis=1).
    NOTE: intentionally within-trial, not across trial identity — this
    matches the originally-validated shuffle_all behavior-mode design,
    which was reviewed and confirmed as the intended methodology.
    """
    return np.roll(bin_beh, shift_bins, axis=1)

def scramble_neural_bins(bin_sf, bin_ti, bin_tmi, perm, n_t):
    """Reorder spike-bin contents within each trial according to perm (len n_t)."""
    n_valid_loc = len(bin_sf)
    new_bin_sf, new_bin_ti, new_bin_tmi = [], [], []
    for new_k in range(n_valid_loc):
        trial_sf = []
        for new_t in range(n_t):
            orig_t = perm[new_t]
            trial_sf.append(bin_sf[new_k][orig_t])
            new_bin_ti.append(np.full_like(bin_ti[new_k * n_t + new_t], new_k))
            new_bin_tmi.append(np.full(len(bin_sf[new_k][orig_t]), new_t, dtype=bin_tmi[0].dtype))
        new_bin_sf.append(trial_sf)
    return new_bin_sf, new_bin_ti, new_bin_tmi

def scramble_behavior_bins(bin_beh, perm):
    """Reorder behaviour time-bin columns by perm (same perm for every trial)."""
    return bin_beh[:, perm]

# %% [markdown]
# #### **FEED DIAGNOSTIC PLOT (orig vs what actually gets fed to decode_pipeline)**
# Runs for ALL modes, including --shuffle none: for none, feed == orig, so the
# right two columns should look identical to the left two — a quick visual
# check that binning/z-scoring/demeaning didn't do anything unexpected, even
# with no perturbation involved. For shift/scramble, the right columns show
# the actual perturbed data.

# %%
N_DIAG_TRIALS = 5

def plot_feed_diagnostic(run_dir, tag, bin_sf_orig, bin_sf_feed, bin_beh_orig, bin_beh_feed,
                          train_idx, trial_starts_valid_, extra_title=""):
    n_show = min(N_DIAG_TRIALS, len(train_idx))
    t_edges = np.arange(N_T_BINS + 1) * (T_TRIAL / N_T_BINS)
    t_ctrs  = 0.5 * (t_edges[:-1] + t_edges[1:])
    bin_s   = T_TRIAL / N_T_BINS
    _sf_changed  = (SHUFFLE != 'none') and (SHUFFLE_MODE == 'neural')
    _beh_changed = (SHUFFLE != 'none') and (SHUFFLE_MODE == 'behavior')
    _sf_tag  = {'shift': 'SHIFTED', 'scramble': 'SCRAMBLED'}.get(SHUFFLE, 'unchanged')
    _beh_tag = _sf_tag
    fig, axes = plt.subplots(n_show, 5, figsize=(22, n_show * 2.8), squeeze=False)
    fig.suptitle(f"Feed diagnostic — {BEHAVIOR_VAR}  {tag}  |  shuffle={SHUFFLE} mode={SHUFFLE_MODE}"
                 f"{extra_title}", fontsize=8)
    for row, tidx in enumerate(train_idx[:n_show]):
        t0 = float(trial_starts_valid_[tidx]); t1 = t0 + T_TRIAL
        ax0, ax1, ax2, ax3, ax4 = axes[row]
        m = (spike_t_gate >= t0) & (spike_t_gate < t1)
        ax0.scatter((spike_t_gate[m] - t0), z_corr_plot[m], s=1, color='black', alpha=0.15, rasterized=True)
        ax0.set_xlim(0, T_TRIAL)
        if row == 0: ax0.set_title('Orig spikes\n(scatter)', fontsize=7)
        orig_counts = np.array([len(bin_sf_orig[tidx][t]) for t in range(N_T_BINS)])
        ax1.bar(t_ctrs, orig_counts, width=bin_s * 0.85, color='steelblue', alpha=0.7)
        ax1.set_xlim(0, T_TRIAL)
        if row == 0: ax1.set_title('Orig spikes\n(bar/bin)', fontsize=7)
        ax2.stairs(bin_beh_orig[tidx], t_edges, color='steelblue', lw=1.5, baseline=None)
        ax2.set_xlim(0, T_TRIAL)
        if row == 0: ax2.set_title(f'Orig {BEHAVIOR_VAR}', fontsize=7)
        feed_counts = np.array([len(bin_sf_feed[tidx][t]) for t in range(N_T_BINS)])
        ax3.bar(t_ctrs, feed_counts, width=bin_s * 0.85,
                color=('tab:orange' if _sf_changed else 'steelblue'), alpha=0.7)
        ax3.set_xlim(0, T_TRIAL)
        if row == 0:
            ax3.set_title(f'Decoder-feed spikes\n({_sf_tag if _sf_changed else "unchanged"})', fontsize=7)
        if _beh_changed:
            ax4.stairs(bin_beh_feed[tidx], t_edges, color='tab:red', lw=1.5, baseline=None, label='feed')
            ax4.stairs(bin_beh_orig[tidx], t_edges, color='steelblue', lw=0.8, baseline=None,
                       alpha=0.4, linestyle='--', label='orig')
            if row == 0:
                ax4.legend(fontsize=5, loc='upper right')
        else:
            ax4.stairs(bin_beh_orig[tidx], t_edges, color='steelblue', lw=1.5, baseline=None)
        ax4.set_xlim(0, T_TRIAL)
        if row == 0:
            ax4.set_title(f'Decoder-feed {BEHAVIOR_VAR}\n({_beh_tag if _beh_changed else "unchanged"})', fontsize=7)
        for ax in (ax0, ax1, ax2, ax3, ax4):
            ax.tick_params(labelsize=5)
    plt.tight_layout()
    _p = os.path.join(run_dir, f'feed_diagnostic_{_VAR_KEY}_{SESSION}.png')
    plt.savefig(_p, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Diagnostic → {_p}")

def start_loss_monitors(run_dir, tag, stop_evt):
    advi_ckpt = os.path.join(run_dir, f"ckpt_{_VAR_KEY}_{SESSION}_advi_epoch.pt")
    advi_plot = os.path.join(run_dir, f"advi_loss_live_{_VAR_KEY}_{SESSION}.png")
    glm_ckpt  = os.path.join(run_dir, f"ckpt_{_VAR_KEY}_{SESSION}_glm_losses.npz")
    glm_plot  = os.path.join(run_dir, f"glm_loss_live_{_VAR_KEY}_{SESSION}.png")

    def _advi_mon(interval=60):
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as _plt
        _lmt = 0
        while not stop_evt.wait(interval):
            if not os.path.exists(advi_ckpt): continue
            try:
                _mt = os.path.getmtime(advi_ckpt)
                if _mt <= _lmt: continue
                _lmt = _mt
                _ec = torch.load(advi_ckpt, map_location='cpu', weights_only=False)
                _tr = _ec.get('elbos', [])
                if not _tr: continue
                _f, _a = _plt.subplots(figsize=(10, 3))
                _a.plot(_tr, lw=0.8, color='steelblue', label='Train ELBO')
                _a.set_title(f'ADVI live — {tag}  {BEHAVIOR_VAR}', fontsize=9)
                _a.legend(fontsize=8); _a.grid(alpha=0.25)
                _plt.tight_layout(); _plt.savefig(advi_plot, dpi=100); _plt.close(_f)
            except Exception:
                pass

    def _glm_mon(interval=30):
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as _plt
        _lmt = 0
        while not stop_evt.wait(interval):
            if not os.path.exists(glm_ckpt): continue
            try:
                _mt = os.path.getmtime(glm_ckpt)
                if _mt <= _lmt: continue
                _lmt = _mt
                _gl = np.load(glm_ckpt); _tr = _gl['train_losses']
                if len(_tr) == 0: continue
                _f, _a = _plt.subplots(figsize=(10, 3))
                _a.plot(_tr, lw=0.8, color='steelblue', label='Train NLL')
                _a.set_title(f'GLM live — {tag}  {BEHAVIOR_VAR}', fontsize=9)
                _a.legend(fontsize=8); _a.grid(alpha=0.25)
                _plt.tight_layout(); _plt.savefig(glm_plot, dpi=100); _plt.close(_f)
            except Exception:
                pass

    threading.Thread(target=_advi_mon, daemon=True).start()
    threading.Thread(target=_glm_mon, daemon=True).start()


# %% [markdown]
# #### **RUN**

# %%
if SHUFFLE == 'none':
    # ═══════════════════════════════════════════════════════════════════════
    # REAL RUN — single pass, full plot suite
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print(f"DECODE  {BEHAVIOR_VAR}{'  (z-scored)' if Z_SCORE_BEH else ''}  — real run")
    print("=" * 60)

    # ── trial inspection plot ────────────────────────────────────────────────
    _insp_path = os.path.join(RUN_DIR, f'trial_inspection_{_VAR_KEY}_{SESSION}.png')
    print("[PLOT] Trial inspection ...")
    _bin_size_s = T_TRIAL / N_T_BINS
    _n_trials_per_set = max(3, min(10, n_valid // 3))
    _n_sets = min(3, n_valid // _n_trials_per_set) if _n_trials_per_set > 0 else 0
    if _n_sets >= 1 and n_valid >= _n_trials_per_set:
        _rng_inspect = np.random.default_rng(seed=42)
        _pool = list(range(n_valid - _n_trials_per_set + 1))
        _rng_inspect.shuffle(_pool)
        _set_starts = []
        for _s in _pool:
            if all(abs(_s - _s2) >= _n_trials_per_set for _s2 in _set_starts):
                _set_starts.append(_s)
            if len(_set_starts) == _n_sets:
                break
        _set_starts.sort()
        _shade_colors = ['#e8eaf6', '#e8f5e9']
        fig_insp, axes_insp = plt.subplots(_n_sets, 1, figsize=(22, 3.5 * _n_sets))
        if _n_sets == 1:
            axes_insp = [axes_insp]
        for _row, _start in enumerate(_set_starts):
            _vis = valid_trials_beh[_start: _start + _n_trials_per_set]
            _t_row_start = trial_starts[_vis[0]]; _t_row_end = trial_ends[_vis[-1]]
            _all_bin_edges, _all_bin_ctrs, _all_scounts, _all_binned_beh = [], [], [], []
            for _ti, _k in enumerate(_vis):
                _t0 = trial_starts[_k]; _t1 = trial_ends[_k]
                _be = _t0 + np.arange(N_T_BINS + 1) * _bin_size_s
                _all_bin_edges.append(_be)
                _all_bin_ctrs.append(0.5 * (_be[:-1] + _be[1:]))
                _smask = (spike_t_gate >= _t0) & (spike_t_gate < _t1)
                _sc, _ = np.histogram(spike_t_gate[_smask], bins=_be)
                _all_scounts.append(_sc)
                _all_binned_beh.append(bin_beh_v[_start + _ti])
            _bin_ctrs_all = np.concatenate(_all_bin_ctrs)
            _scounts_all  = np.concatenate(_all_scounts)
            ax_l = axes_insp[_row]; ax_r = ax_l.twinx()
            for _ti, _k in enumerate(_vis):
                ax_l.axvspan(trial_starts[_k], trial_ends[_k], facecolor=_shade_colors[_ti % 2], alpha=0.55, zorder=0)
            for _be_val in np.unique(np.concatenate(_all_bin_edges)):
                ax_l.axvline(_be_val, color='silver', lw=0.35, zorder=1)
            for _k in _vis:
                ax_l.axvline(trial_starts[_k], color='#333333', lw=1.3, linestyle='--', zorder=2, alpha=0.75)
            ax_l.axvline(trial_ends[_vis[-1]], color='#333333', lw=1.3, linestyle='--', zorder=2, alpha=0.75)
            ax_l.bar(_bin_ctrs_all, _scounts_all, width=_bin_size_s * 0.88, color='steelblue',
                     alpha=0.55, edgecolor='none', zorder=3, label='spike count (MUA)')
            ax_l.set_ylabel('spikes / bin', fontsize=8, color='steelblue')
            ax_l.tick_params(axis='y', labelcolor='steelblue', labelsize=7)
            ax_l.set_xlim(_t_row_start - _bin_size_s * 0.1, _t_row_end + _bin_size_s * 0.1)
            _bmask = (beh_t >= _t_row_start) & (beh_t <= _t_row_end)
            ax_r.plot(beh_t[_bmask], beh_z[_bmask], color='darkgray', lw=0.7, alpha=0.9, zorder=4,
                      label=f'{BEHAVIOR_VAR} raw')
            for _ti, _be in enumerate(_all_bin_edges):
                ax_r.stairs(_all_binned_beh[_ti], _be, color='tab:orange', lw=1.8, zorder=5, baseline=None,
                            label=f'{BEHAVIOR_VAR} binned' if _ti == 0 else None)
            ax_r.set_ylabel(f'{BEHAVIOR_VAR}', fontsize=8, color='tab:orange')
            ax_r.tick_params(axis='y', labelcolor='tab:orange', labelsize=7)
            _h1, _l1 = ax_l.get_legend_handles_labels(); _h2, _l2 = ax_r.get_legend_handles_labels()
            ax_l.legend(_h1 + _h2, _l1 + _l2, fontsize=7, loc='upper right', framealpha=0.7)
            ax_l.set_title(f"Set {_row+1}  |  t = {_t_row_start:.1f}–{_t_row_end:.1f} s", fontsize=8, loc='left')
            if _row == _n_sets - 1:
                ax_l.set_xlabel('Time from SpikeGLX gate open (s)', fontsize=8)
        fig_insp.suptitle(f"Trial inspection — {SESSION} — {BEHAVIOR_VAR} vs spike count", fontsize=9)
        plt.tight_layout()
        fig_insp.savefig(_insp_path, dpi=130, bbox_inches='tight')
        plt.close(fig_insp)
        print(f"Saved → {_insp_path}")
    else:
        print(f"[WARN] Not enough valid trials ({n_valid}) for trial inspection plot — skipping.")

    # ── feed diagnostic: orig vs what's actually fed to decode_pipeline ──────
    # For --shuffle none these are the same arrays — this is a sanity check
    # that binning/z-scoring/demeaning produced what you'd expect, not a
    # perturbation check.
    plot_feed_diagnostic(RUN_DIR, "real", bin_sf_v, bin_sf_v, bin_beh_v, bin_beh_v,
                         train_idx_fixed, trial_starts_valid)

    # decode_pipeline uses checkpoint_dir=RUN_DIR as-is: if a previous run's
    # GMM/init-weight-matrix/ADVI checkpoints are still there, it will pick
    # them up and resume/reuse them. Delete RUN_DIR's ckpt_* files yourself
    # first if you want a guaranteed-fresh fit for a new train/test split.
    _n_train_trials = len(train_idx_fixed)
    _batch_size = min(_args.batch_size, max(1, _n_train_trials))
    print(f"[INFO] batch_size={_batch_size}  ({_n_train_trials} training trials)")

    _stop_monitor = threading.Event()
    start_loss_monitors(RUN_DIR, "real", _stop_monitor)

    weight_matrix = decode_pipeline(
        data_loader, bin_sf_v, bin_ti_v, bin_tmi_v,
        bin_behaviors=bin_beh_v, behavior_type="continuous",
        train=train_idx_fixed, test=test_idx_fixed,
        learning_rate=LEARNING_RATE, device=device, n_workers=N_WORKERS,
        checkpoint_dir=RUN_DIR, max_iter=7000, checkpoint_name=f"{_VAR_KEY}_{SESSION}",
        early_stopping=_args.early_stopping, es_patience=_args.es_patience, batch_size=_batch_size,
    )
    _stop_monitor.set()
    WEIGHT_CKPT = os.path.join(RUN_DIR, f'ckpt_weight_matrix_{_VAR_KEY}_{SESSION}.npy')
    np.save(WEIGHT_CKPT, weight_matrix)
    print(f"[SAVED] weight matrix → {WEIGHT_CKPT}")

    # ── ADVI loss curve ──────────────────────────────────────────────────────
    _elbos_npz = os.path.join(RUN_DIR, f'ckpt_{_VAR_KEY}_{SESSION}_elbos.npz')
    if os.path.exists(_elbos_npz):
        _ed = np.load(_elbos_npz)
        fig_l, ax_l = plt.subplots(figsize=(12, 4))
        ax_l.plot(_ed['train_elbos'], lw=0.8, color='steelblue', label='Train ELBO')
        if len(_ed['val_iters']) > 0:
            ax_l.plot(_ed['val_iters'], _ed['val_elbos'], lw=1.2, color='darkorange', marker='o',
                      markersize=2, label='Val ELBO')
        ax_l.set_xlabel('Iteration'); ax_l.set_ylabel('ELBO')
        ax_l.set_title(f'ADVI training — {BEHAVIOR_VAR}  {SESSION}', fontsize=9)
        ax_l.legend(fontsize=8); ax_l.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(RUN_DIR, f'advi_loss_{_VAR_KEY}_{SESSION}.png'), dpi=130, bbox_inches='tight')
        plt.close()

    # ── putative neurons + trial diagnostics ─────────────────────────────────
    _post_params_path = os.path.join(RUN_DIR, f'ckpt_{_VAR_KEY}_{SESSION}_post_params.npz')
    if os.path.exists(_post_params_path):
        print("[PLOT] Putative neurons ...")
        _pp = np.load(_post_params_path)
        _pn_means = _pp['means']; _pn_beta = _pp['U'] @ _pp['V']
        _pn_activity = weight_matrix.sum(axis=(0, 2)); n_pn = len(_pn_means)
        _act_99 = np.percentile(_pn_activity, 99)
        _pn_sizes = 10 + 190 * np.clip(_pn_activity / max(_act_99, 1e-9), 0, 1)
        _beta_mean = _pn_beta.mean(axis=1)
        _t_bins_ms = np.arange(N_T_BINS) * (T_TRIAL / N_T_BINS) * 1000
        fig_pn, axes_pn = plt.subplots(1, 3, figsize=(18, 12), gridspec_kw={'width_ratios': [1, 1, 2]})
        fig_pn.suptitle(f"Putative neurons — {SESSION} — {BEHAVIOR_VAR}\n{n_pn} components", fontsize=10)
        _bm_abs = max(np.abs(_beta_mean).max(), 1e-9)
        sc0 = axes_pn[0].scatter(_pn_means[:, 0], _pn_means[:, 1], s=_pn_sizes, c=_beta_mean,
                                  cmap='RdBu_r', vmin=-_bm_abs, vmax=_bm_abs, edgecolors='k', linewidths=0.3, alpha=0.85)
        plt.colorbar(sc0, ax=axes_pn[0], label=f'Mean β ({BEHAVIOR_VAR})', shrink=0.6)
        axes_pn[0].set_xlabel('x (µm)'); axes_pn[0].set_ylabel('Depth z (µm)')
        axes_pn[0].set_title('Behavioural tuning (mean β)', fontsize=9); axes_pn[0].grid(True, alpha=0.2)
        _ptp99 = np.percentile(_pn_means[:, 2], 99)
        sc1 = axes_pn[1].scatter(_pn_means[:, 0], _pn_means[:, 1], s=_pn_sizes,
                                  c=np.clip(_pn_means[:, 2], 0, _ptp99), cmap='hot_r', vmin=0, vmax=_ptp99,
                                  edgecolors='k', linewidths=0.3, alpha=0.85)
        plt.colorbar(sc1, ax=axes_pn[1], label='Mean PTP (µV)', shrink=0.6)
        axes_pn[1].set_xlabel('x (µm)'); axes_pn[1].set_ylabel('Depth z (µm)')
        axes_pn[1].set_title('Amplitude (mean PTP)', fontsize=9); axes_pn[1].grid(True, alpha=0.2)
        _top20 = np.argsort(_pn_activity)[-20:][::-1]
        _cmap20 = plt.cm.tab20(np.linspace(0, 1, len(_top20)))
        for _ci, _idx in enumerate(_top20):
            axes_pn[2].plot(_t_bins_ms, _pn_beta[_idx], color=_cmap20[_ci], lw=1.2, alpha=0.85,
                            label=f'#{_idx} act={_pn_activity[_idx]:.0f}')
        axes_pn[2].axhline(0, color='k', lw=0.6, ls='--', alpha=0.4)
        axes_pn[2].set_xlabel('Time within trial (ms)'); axes_pn[2].set_ylabel(f'β ({BEHAVIOR_VAR})')
        axes_pn[2].set_title('β(t) — top-20 most active', fontsize=9)
        axes_pn[2].legend(fontsize=5, loc='upper right', ncol=2, framealpha=0.7); axes_pn[2].grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(RUN_DIR, f'putative_neurons_{_VAR_KEY}_{SESSION}.png'), dpi=130, bbox_inches='tight')
        plt.close(fig_pn)

        print("[PLOT] Trial diagnostics ...")
        import matplotlib.gridspec as _gs
        _t_bins_ms_diag = np.arange(N_T_BINS) * (T_TRIAL / N_T_BINS * 1000)
        _beta_hmap = _pp['U'] @ _pp['V']
        _bmax = float(np.percentile(np.abs(_beta_hmap), 99))
        _spike_counts = np.zeros((len(train_idx_fixed), N_T_BINS))
        for _new_k, _k in enumerate(train_idx_fixed):
            for _tb in range(N_T_BINS):
                _spike_counts[_new_k, _tb] = len(bin_sf_v[_k][_tb])
        _sc_mean = _spike_counts.mean(axis=0); _sc_std = _spike_counts.std(axis=0)
        _beh_train = bin_beh_v[train_idx_fixed]
        _beh_mean = _beh_train.mean(axis=0); _beh_std = _beh_train.std(axis=0)
        fig_diag = plt.figure(figsize=(15, 11))
        _spec = _gs.GridSpec(3, 2, figure=fig_diag, hspace=0.08, wspace=0.03,
                              height_ratios=[2.5, 1.5, 1.5], width_ratios=[1, 0.03])
        ax0 = fig_diag.add_subplot(_spec[0, 0])
        ax1 = fig_diag.add_subplot(_spec[1, 0], sharex=ax0)
        ax2 = fig_diag.add_subplot(_spec[2, 0], sharex=ax0)
        ax_cb = fig_diag.add_subplot(_spec[0, 1])
        fig_diag.add_subplot(_spec[1, 1]).set_visible(False)
        fig_diag.add_subplot(_spec[2, 1]).set_visible(False)
        fig_diag.suptitle(f"{SESSION} - {BEHAVIOR_VAR} trial diagnostics\ntraining trials (n={len(train_idx_fixed)})",
                          fontsize=11)
        _im = ax0.imshow(_beta_hmap, aspect='auto', cmap='RdBu_r', vmin=-_bmax, vmax=_bmax,
                          extent=[_t_bins_ms_diag[0], _t_bins_ms_diag[-1], _beta_hmap.shape[0] - 0.5, -0.5],
                          interpolation='nearest')
        plt.colorbar(_im, cax=ax_cb, label='GLM tuning coefficient (U @ V)')
        ax0.set_ylabel('GMM component', fontsize=9); plt.setp(ax0.get_xticklabels(), visible=False)
        _bar_width = (T_TRIAL / N_T_BINS * 1000) * 0.75
        ax1.bar(_t_bins_ms_diag, _sc_mean, width=_bar_width, color='steelblue', alpha=0.75, label='mean', zorder=3)
        ax1.errorbar(_t_bins_ms_diag, _sc_mean, yerr=_sc_std, fmt='none', color='steelblue',
                     elinewidth=1.2, capsize=2.5, alpha=0.9, zorder=4, label='± 1SD')
        ax1.set_ylabel('Spikes / bin', fontsize=9); ax1.legend(fontsize=8, loc='upper right')
        ax1.grid(True, alpha=0.15, axis='y'); plt.setp(ax1.get_xticklabels(), visible=False)
        for _ti in range(len(train_idx_fixed)):
            ax2.plot(_t_bins_ms_diag, _beh_train[_ti], color='darkorange', alpha=0.15, lw=0.6)
        ax2.fill_between(_t_bins_ms_diag, _beh_mean - _beh_std, _beh_mean + _beh_std, color='darkorange', alpha=0.25)
        ax2.plot(_t_bins_ms_diag, _beh_mean, color='darkorange', lw=1.8, label='mean ± 1SD')
        ax2.set_xlabel('Time within trial (ms)', fontsize=9); ax2.set_ylabel(f'{BEHAVIOR_VAR}', fontsize=9)
        ax2.legend(fontsize=8, loc='upper right'); ax2.grid(True, alpha=0.15)
        plt.savefig(os.path.join(RUN_DIR, f'trial_diagnostics_{_VAR_KEY}_{SESSION}.png'), dpi=130, bbox_inches='tight')
        plt.close(fig_diag)

    # ── decode + evaluate ─────────────────────────────────────────────────────
    _n_wm = weight_matrix.shape[0]
    eval_train_idx = train_idx_fixed[train_idx_fixed < _n_wm]
    eval_test_idx  = test_idx_fixed[test_idx_fixed < _n_wm]
    if len(eval_train_idx) == 0 or len(eval_test_idx) == 0:
        raise ValueError(f"Empty eval split after alignment: n_wm={_n_wm}, "
                         f"train={len(eval_train_idx)}, test={len(eval_test_idx)}")
    _bin_beh_for_decode = bin_beh_v[:_n_wm]
    print(f"\n[DECODE] Running decoder on held-out test windows ...")
    _, obs_beh, pred_beh, metrics_beh = sliding_window_decoder(
        weight_matrix, _bin_beh_for_decode, eval_train_idx, eval_test_idx,
        behavior_type="continuous", verbose=False,
    )
    print(f"\n{'='*52}")
    print(f"  {BEHAVIOR_VAR}  R² = {metrics_beh['r2']:+.4f}   r = {metrics_beh['corr']:.4f}")
    print(f"  (n_test={len(eval_test_idx)} windows)")
    print(f"{'='*52}\n")

    # ── decode traces plot ────────────────────────────────────────────────────
    print("[PLOT] Decode traces ...")
    _half_win = 7 // 2
    _n_win = N_T_BINS - _half_win + 1
    _n_pred_bins = _n_win - _half_win
    _bin_dur_s = T_TRIAL / N_T_BINS
    _bin_offsets = np.arange(_half_win, _n_win) * _bin_dur_s
    _n_test = len(eval_test_idx)
    _pred_by_trial = pred_beh.reshape(_n_test, _n_pred_bins)
    _obs_by_trial  = obs_beh.reshape(_n_test, _n_pred_bins)
    _ncols = min(4, _n_test); _nrows = int(np.ceil(_n_test / _ncols)) if _ncols else 1
    _panel_w = max(3.5, T_TRIAL * 0.8)
    fig, axes = plt.subplots(_nrows, _ncols, figsize=(_ncols * _panel_w, _nrows * 3.0), squeeze=False, sharey=True)
    fig.suptitle(f"Density decoding — {SESSION} — {BEHAVIOR_VAR}\n"
                 f"R² = {metrics_beh['r2']:+.4f}   r = {metrics_beh['corr']:.4f}  ({_n_test} test trials)", fontsize=10)
    for _j in range(_n_test):
        _row, _col = divmod(_j, _ncols); _ax = axes[_row, _col]
        _ax.plot(_bin_offsets, _obs_by_trial[_j], color='darkgray', lw=1.2, label='observed')
        _ax.plot(_bin_offsets, _pred_by_trial[_j], color='tab:blue', lw=1.2, alpha=0.85, label='decoded')
        _orig_trial = valid_trials_beh[eval_test_idx[_j]]
        _ax.set_title(f"test trial {_j}  (t={trial_starts[_orig_trial]:.1f} s)", fontsize=7, pad=2)
        _ax.set_xlabel('Time within trial (s)', fontsize=7)
        if _col == 0:
            _ax.set_ylabel(f'{BEHAVIOR_VAR}', fontsize=7)
        _ax.tick_params(labelsize=6); _ax.grid(True, alpha=0.18); _ax.set_xlim(0, T_TRIAL)
        if _j == 0:
            _ax.legend(fontsize=6, loc='upper right')
    for _j in range(_n_test, _nrows * _ncols):
        _row, _col = divmod(_j, _ncols); axes[_row, _col].set_visible(False)
    plt.tight_layout()
    _fig_path = os.path.join(RUN_DIR, f'decode_traces_{_VAR_KEY}_{SESSION}.png')
    fig.savefig(_fig_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure → {_fig_path}")

    # ── save results ──────────────────────────────────────────────────────────
    _decode_npz = os.path.join(RUN_DIR, f'decode_results_{_VAR_KEY}_{SESSION}.npz')
    np.savez_compressed(
        _decode_npz, bin_beh=_bin_beh_for_decode, obs_test=obs_beh, pred_test=pred_beh,
        train_idx=eval_train_idx, test_idx=eval_test_idx,
        train_idx_global=train_idx_fixed, test_idx_global=test_idx_fixed,
        weight_matrix=weight_matrix, trial_starts=trial_starts_valid, trial_ends=trial_starts_valid + T_TRIAL,
        T_TRIAL=np.float32(T_TRIAL), N_T_BINS=np.int32(N_T_BINS),
    )
    print(f"Arrays → {_decode_npz}")

    _ts_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    summary_lines = [
        f"Density decoding results — {SESSION}  (shuffle=none)", f"Saved: {_ts_now}", "",
        f"  BEHAVIOR_VAR    : {BEHAVIOR_VAR}",
        f"  Valid windows   : {n_valid}  (train={len(train_idx_fixed)}, test={len(test_idx_fixed)}, "
        f"embargoed={_n_embargoed})",
        f"  R²              : {metrics_beh['r2']:+.4f}",
        f"  r               : {metrics_beh['corr']:.4f}",
        f"  n_test          : {len(eval_test_idx)} windows",
    ]
    with open(os.path.join(RUN_DIR, f'decode_results_{_VAR_KEY}_{SESSION}.txt'), 'w') as _f:
        _f.write('\n'.join(summary_lines) + '\n')
    _params['results'] = {BEHAVIOR_VAR: {'r2': float(metrics_beh['r2']), 'corr': float(metrics_beh['corr']),
                                          'n_test_windows': int(len(eval_test_idx))}}
    _params['completed'] = _ts_now
    with open(_params_json, 'w') as _f:
        json.dump(_params, _f, indent=2)
    print('\n'.join(summary_lines))

else:
    # ═══════════════════════════════════════════════════════════════════════
    # NULL DISTRIBUTION — shift or scramble, looped over shuffle indices
    # ═══════════════════════════════════════════════════════════════════════
    rng_master = np.random.default_rng(_args.seed)
    if SHUFFLE == 'shift':
        if T_OV_DUR < 2 * _args.min_shift_sec:
            raise ValueError(f"Overlap duration {T_OV_DUR:.1f}s < 2×min_shift "
                             f"({2*_args.min_shift_sec:.1f}s). Reduce --min_shift_sec.")
        _shift_pool = rng_master.uniform(_args.min_shift_sec, T_OV_DUR - _args.min_shift_sec, size=_args.n_shuffles)
    else:
        _shuffle_seeds = rng_master.integers(0, 2**31, size=_args.n_shuffles)

    _run_indices = ([_args.shuffle_index] if _args.shuffle_index is not None
                    else list(range(_args.n_shuffles)))

    _summary_path = os.path.join(RUN_DIR, f"{SHUFFLE}_results_summary.npz")
    if os.path.exists(_summary_path):
        _ex = np.load(_summary_path)
        _all_r2, _all_r, _all_idx = list(_ex['r2']), list(_ex['r']), list(_ex['shuffle_index'])
        print(f"[SUMMARY] Loaded {len(_all_r2)} existing results.")
    else:
        _all_r2, _all_r, _all_idx = [], [], []

    for _shuf_i in _run_indices:
        _shuf_dir  = os.path.join(RUN_DIR, f'shuffle_{_shuf_i}')
        _shuf_done = os.path.join(_shuf_dir, f'decode_results_{_VAR_KEY}_{SESSION}.npz')
        os.makedirs(_shuf_dir, exist_ok=True)

        print(f"\n{'='*65}")
        print(f"  {SHUFFLE.upper()} {_shuf_i}/{_args.n_shuffles-1}  |  mode={SHUFFLE_MODE}")
        print(f"{'='*65}")

        if os.path.exists(_shuf_done):
            print(f"  [SKIP] Already done — completed result, not a stale checkpoint.")
            _ex2 = np.load(_shuf_done)
            if _shuf_i not in _all_idx:
                _all_r2.append(float(_ex2.get('r2', np.nan)))
                _all_r.append(float(_ex2.get('r', np.nan)))
                _all_idx.append(_shuf_i)
            continue

        # decode_pipeline uses checkpoint_dir=_shuf_dir as-is: if this shuffle
        # folder already has checkpoints (e.g. a previous partial attempt), it
        # will pick them up and resume/reuse them. Delete _shuf_dir yourself
        # first if you want a guaranteed-fresh fit.

        _t_start = datetime.datetime.now()

        if SHUFFLE == 'shift':
            _shift_sec = float(_shift_pool[_shuf_i])
            if SHUFFLE_MODE == 'neural':
                spike_t_shifted = circular_shift_spikes(spike_t_gate, _shift_sec, T_OV_START, T_OV_END)
                _bsf_all, _bti_all, _btmi_all = make_bin_sf(spike_t_shifted)
                bin_sf_all_s = [_bsf_all[k] for k in valid_trials_beh]
                bin_ti_all_s, bin_tmi_all_s = [], []
                for new_k, k in enumerate(valid_trials_beh):
                    for t in range(N_T_BINS):
                        old = _bti_all[k * N_T_BINS + t]
                        bin_ti_all_s.append(np.full_like(old, new_k))
                        bin_tmi_all_s.append(_btmi_all[k * N_T_BINS + t])
                bin_beh_all_s = bin_beh_v.copy()
            else:  # behavior
                bin_sf_all_s, bin_ti_all_s, bin_tmi_all_s = bin_sf_v, bin_ti_v, bin_tmi_v
                _shift_bins = int(round(_shift_sec / (T_TRIAL / N_T_BINS))) % N_T_BINS
                bin_beh_all_s = circular_shift_behavior(bin_beh_v, _shift_bins)
            _perturb_meta = {'shift_sec': _shift_sec}
        else:  # scramble
            rng_s = np.random.default_rng(int(_shuffle_seeds[_shuf_i]))
            perm = rng_s.permutation(N_T_BINS)
            print(f"  Permutation (first 10): {perm[:10]}")
            if SHUFFLE_MODE == 'neural':
                bin_sf_all_s, bin_ti_all_s, bin_tmi_all_s = scramble_neural_bins(bin_sf_v, bin_ti_v, bin_tmi_v, perm, N_T_BINS)
                bin_beh_all_s = bin_beh_v.copy()
            else:
                bin_sf_all_s, bin_ti_all_s, bin_tmi_all_s = bin_sf_v, bin_ti_v, bin_tmi_v
                bin_beh_all_s = scramble_behavior_bins(bin_beh_v, perm)
            _perturb_meta = {'scramble_perm': perm.tolist()}

        if len(train_idx_fixed) == 0 or len(test_idx_fixed) == 0:
            print(f"  [WARN] Empty train or test — skipping.")
            continue

        _n_train_s = len(train_idx_fixed)
        _batch_size_s = min(_args.batch_size, max(1, _n_train_s))

        _extra_title = (f" | shift={_perturb_meta['shift_sec']:.2f}s" if SHUFFLE == 'shift'
                        else f" | perm={np.array(_perturb_meta['scramble_perm'])[:8]}")
        plot_feed_diagnostic(_shuf_dir, f"{SHUFFLE} {_shuf_i}", bin_sf_v, bin_sf_all_s,
                             bin_beh_v, bin_beh_all_s, train_idx_fixed, trial_starts_valid,
                             extra_title=_extra_title)

        _stop_s = threading.Event()
        start_loss_monitors(_shuf_dir, f"{SHUFFLE}_{_shuf_i}", _stop_s)

        weight_matrix_s = decode_pipeline(
            data_loader, bin_sf_all_s, bin_ti_all_s, bin_tmi_all_s,
            bin_behaviors=bin_beh_all_s, behavior_type="continuous",
            train=train_idx_fixed, test=test_idx_fixed,
            learning_rate=LEARNING_RATE, device=device, n_workers=N_WORKERS,
            checkpoint_dir=_shuf_dir, max_iter=7000, checkpoint_name=f"{_VAR_KEY}_{SESSION}",
            early_stopping=_args.early_stopping, es_patience=_args.es_patience, batch_size=_batch_size_s,
        )
        _stop_s.set()

        _n_wm_s = weight_matrix_s.shape[0]
        eval_train_idx = train_idx_fixed[train_idx_fixed < _n_wm_s]
        eval_test_idx  = test_idx_fixed[test_idx_fixed < _n_wm_s]
        if len(eval_train_idx) == 0 or len(eval_test_idx) == 0:
            raise ValueError(f"Empty eval split after alignment: n_wm={_n_wm_s}, "
                             f"train={len(eval_train_idx)}, test={len(eval_test_idx)}")

        _, obs_s, pred_s, metrics_s = sliding_window_decoder(
            weight_matrix_s, bin_beh_all_s[:_n_wm_s], eval_train_idx, eval_test_idx,
            behavior_type="continuous", verbose=False,
        )

        _t_elapsed = (datetime.datetime.now() - _t_start).total_seconds()
        print(f"\n  Done in {_t_elapsed/60:.1f} min  |  R²={metrics_s['r2']:+.4f}  r={metrics_s['corr']:.4f}")

        _save_kwargs = dict(
            obs_test=obs_s, pred_test=pred_s, train_idx=eval_train_idx, test_idx=eval_test_idx,
            train_idx_global=train_idx_fixed, test_idx_global=test_idx_fixed, weight_matrix=weight_matrix_s,
            shuffle_index=np.int32(_shuf_i), r2=np.float64(metrics_s['r2']), r=np.float64(metrics_s['corr']),
            trial_starts=trial_starts_valid, trial_ends=trial_starts_valid + T_TRIAL,
            T_TRIAL=np.float32(T_TRIAL), N_T_BINS=np.int32(N_T_BINS),
        )
        if SHUFFLE == 'shift':
            _save_kwargs['shift_sec'] = np.float64(_perturb_meta['shift_sec'])
        else:
            _save_kwargs['scramble_perm'] = np.array(_perturb_meta['scramble_perm'])
        np.savez_compressed(_shuf_done, **_save_kwargs)

        _params_s = {
            "shuffle_index": _shuf_i, "shuffle": SHUFFLE, "shuffle_mode": SHUFFLE_MODE,
            "session": SESSION, "behavior_var": BEHAVIOR_VAR,
            "n_valid": int(n_valid), "n_train": int(len(train_idx_fixed)), "n_test": int(len(test_idx_fixed)),
            "r2": float(metrics_s['r2']), "r": float(metrics_s['corr']), "elapsed_sec": float(_t_elapsed),
            "note": (f"r2/r: {'SCRAMBLED' if SHUFFLE=='scramble' else 'SHIFTED'} "
                    f"{'spikes vs REAL behaviour' if SHUFFLE_MODE=='neural' else 'REAL spikes vs perturbed behaviour'}"),
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **_perturb_meta,
        }
        with open(os.path.join(_shuf_dir, f"params_{_VAR_KEY}_{SESSION}.json"), 'w') as _f:
            json.dump(_params_s, _f, indent=2)

        _all_r2.append(float(metrics_s['r2'])); _all_r.append(float(metrics_s['corr'])); _all_idx.append(_shuf_i)
        np.savez_compressed(_summary_path, shuffle_index=np.array(_all_idx, dtype=np.int32),
                            r2=np.array(_all_r2, dtype=np.float64), r=np.array(_all_r, dtype=np.float64))
        print(f"  Summary → {_summary_path}")

    if _all_r2:
        print(f"\n{'='*65}")
        print(f"  {SHUFFLE.upper()} ({SHUFFLE_MODE}) SUMMARY  ({len(_all_r2)} shuffles)")
        print(f"  R²: mean={np.mean(_all_r2):+.4f}  std={np.std(_all_r2):.4f}  "
              f"[{np.min(_all_r2):+.4f}, {np.max(_all_r2):+.4f}]")
        print(f"  r : mean={np.mean(_all_r):+.4f}  std={np.std(_all_r):.4f}  "
              f"[{np.min(_all_r):+.4f}, {np.max(_all_r):+.4f}]")
        print(f"{'='*65}")