"""
Compare Boussard vs SpikeInterface monopolar-triangulation localisations.

Requires a GPU node (Boussard denoiser uses CUDA).
Run with:
    conda activate spikeinterface
    python scripts/compare_localisation.py

All plots saved to:
    /ptmp/gongt/GLC7189/GLC7189_g_g1/preprocessed/thresholding_peaks_3.3/
"""

import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import spikeinterface.full as si
import torch

sys.path.insert(0, '/u/gongt/spikes_localization_registration')
from localization_pipeline.localizer import LOCALIZER

# =============================================================================
# PATHS
# =============================================================================
peaks_folder = '/ptmp/gongt/GLC7189/GLC7189_g_g1/preprocessed/thresholding_peaks_3.3'
boussard_dir = peaks_folder + '/boussard_localization_merged'
REC_DIR      = '/ptmp/gongt/GLC7189/GLC7189_g_g1/preprocessed'
REPO         = '/u/gongt/spikes_localization_registration'

# =============================================================================
# SECTION 1 — Load data & plot depth / lateral / raw-trough-amplitude
# =============================================================================
print("Loading Boussard outputs...")
spike_index = np.load(peaks_folder + '/boussard_spike_index_30min.npy')   # (7.2M, 2)
valid       = np.load(boussard_dir + '/times_read.npy') == 1

b_z   = np.load(boussard_dir + '/results_z_merged.npy')[valid]
b_x   = np.load(boussard_dir + '/results_x_merged.npy')[valid]
b_ptp = np.load(boussard_dir + '/results_max_ptp_merged.npy')[valid]

n_30min  = len(spike_index)   # 7,225,769
print("Loading SI localizations...")
si_locs  = np.load(peaks_folder + '/localized_peaks_mt_raw.npy')    # (126M,)
si_peaks = np.load(peaks_folder + '/peaks_combined2_sorted.npy')    # (126M,)
si_z   = si_locs['y'][:n_30min][valid]
si_x   = si_locs['x'][:n_30min][valid]
si_amp = np.abs(si_peaks['amplitude'][:n_30min][valid])   # raw trough magnitude

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

lim_z = (min(si_z.min(), b_z.min()), max(si_z.max(), b_z.max()))
axes[0].scatter(si_z, b_z, s=0.5, alpha=0.1)
axes[0].plot(lim_z, lim_z, 'r--', lw=1)
axes[0].set_xlim(lim_z); axes[0].set_ylim(lim_z)
axes[0].set_aspect('equal')
axes[0].set_xlabel('SI depth (µm)'); axes[0].set_ylabel('Boussard depth (µm)')
axes[0].set_title('Depth: Boussard vs SI')

lim_x = (min(si_x.min(), b_x.min()), max(si_x.max(), b_x.max()))
axes[1].scatter(si_x, b_x, s=0.5, alpha=0.1)
axes[1].plot(lim_x, lim_x, 'r--', lw=1)
axes[1].set_xlim(lim_x); axes[1].set_ylim(lim_x)
axes[1].set_aspect('equal')
axes[1].set_xlabel('SI x (µm)'); axes[1].set_ylabel('Boussard x (µm)')
axes[1].set_title('Lateral: Boussard vs SI')

lim_amp = (min(si_amp.min(), b_ptp.min()), max(si_amp.max(), b_ptp.max()))
axes[2].scatter(si_amp, b_ptp, s=0.5, alpha=0.1)
axes[2].plot(lim_amp, lim_amp, 'r--', lw=1)
axes[2].set_xlim(lim_amp); axes[2].set_ylim(lim_amp)
axes[2].set_aspect('equal')
axes[2].set_xlabel('SI |raw trough| at detection ch (si_peaks["amplitude"])')
axes[2].set_ylabel('Boussard denoised max-PTP (over all channels)')
axes[2].set_title('Amplitude: raw trough vs denoised max-PTP')

plt.tight_layout()
out = peaks_folder + '/boussard_vs_si_comparison_ptp_peaks.png'
plt.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.close()

# =============================================================================
# SECTION 2 — Recover SI PTP (alpha / r) & amplitude distribution comparison
# =============================================================================
print("Computing SI recovered PTP...")
rec          = si.load(REC_DIR)
contact_locs = rec.get_channel_locations()   # (368, 2)

ch_idx = si_peaks['channel_index'][:n_30min][valid]
cx = contact_locs[ch_idx, 0];  cy = contact_locs[ch_idx, 1]
dx = si_locs['x'][:n_30min][valid] - cx
dy = si_locs['y'][:n_30min][valid] - cy
dz = si_locs['z'][:n_30min][valid]
r_detected = np.sqrt(dx**2 + dy**2 + dz**2)
si_ptp = si_locs['alpha'][:n_30min][valid] / r_detected   # µV, 549K spikes

ch_idx_all = si_peaks['channel_index'][:n_30min]
cx_all = contact_locs[ch_idx_all, 0];  cy_all = contact_locs[ch_idx_all, 1]
dx_all = si_locs['x'][:n_30min] - cx_all
dy_all = si_locs['y'][:n_30min] - cy_all
dz_all = si_locs['z'][:n_30min]
r_all      = np.sqrt(dx_all**2 + dy_all**2 + dz_all**2)
si_ptp_all = si_locs['alpha'][:n_30min] / r_all            # µV, 7.2M spikes

print(f"SI all 30-min spikes:       {len(si_ptp_all):,}")
print(f"Boussard valid (ptp > 6):   {len(b_ptp):,}  ({100*len(b_ptp)/len(si_ptp_all):.1f}%)")
print(f"si_ptp (valid) range: {si_ptp.min():.2f} -- {si_ptp.max():.2f} µV")
print(f"b_ptp  (valid) range: {b_ptp.min():.2f} -- {b_ptp.max():.2f} µV")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

lim_amp = (min(si_ptp.min(), b_ptp.min()), max(si_ptp.max(), b_ptp.max()))
axes[0].scatter(si_ptp, b_ptp, s=0.5, alpha=0.1)
axes[0].plot(lim_amp, lim_amp, 'r--', lw=1)
axes[0].set_xlim(lim_amp); axes[0].set_ylim(lim_amp)
axes[0].set_aspect('equal')
axes[0].set_xlabel('SI MT est. raw PTP at detect ch (µV)  [alpha / r_detected]')
axes[0].set_ylabel('Boussard denoised max-PTP over all channels (µV)')
axes[0].set_title(f'PTP scatter: SI raw-fit vs Boussard denoised-max (n={len(b_ptp):,})')

bins = np.linspace(0, np.percentile(si_ptp_all, 99.9), 200)
axes[1].hist(si_ptp_all, bins=bins, alpha=0.6, label=f'SI all 30-min ({len(si_ptp_all):,})')
axes[1].hist(b_ptp,      bins=bins, alpha=0.6, label=f'Boussard valid ({len(b_ptp):,})')
axes[1].axvline(6, color='r', lw=1, ls='--', label='Boussard threshold (6 µV)')
axes[1].set_xlabel('Amplitude (µV)'); axes[1].set_ylabel('Spike count')
axes[1].set_title('Amplitude distributions: full SI vs Boussard filtered')
axes[1].legend()

plt.tight_layout()
out = peaks_folder + '/amplitude_comparison_ptp_ptp.png'
plt.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.close()

# =============================================================================
# SECTION 3 — Find outlier spikes & plot raw vs denoised waveforms
# =============================================================================

# ── Channel identity and PTP ratio diagnostics ────────────────────────────────
print("\nDiagnostic: channel identity and PTP definitions")
b_max_ch_all = np.load(boussard_dir + '/results_max_channels.npy')[valid].astype(int)
si_ch_all    = si_peaks['channel_index'][:n_30min][valid]
n_same = np.sum(si_ch_all == b_max_ch_all)
print(f"  SI detect ch == Boussard results_max_channels: {n_same:,}/{len(si_ch_all):,} ({100*n_same/len(si_ch_all):.1f}%)")
print(f"  → 100% match because results_max_channels is the INPUT detection channel")
print(f"     (argmax-after-denoising code is commented out in localizer.py line 155)")
print(f"     b_ptp = np.ptp(denoised_wf, 0).max() = max PTP over ALL channels of denoised wf")
print(f"     so b_ptp is NOT necessarily at the detection channel")
ratio_ptp = b_ptp / si_ptp
print(f"  b_ptp / si_ptp: median={np.median(ratio_ptp):.2f}  mean={ratio_ptp.mean():.2f}")
print(f"  → SI MT fits raw PTP at neighborhood channels (feature='ptp' in monopolar.py)")
print(f"     so alpha/r = estimated raw PTP at detection channel")
print(f"     b_ptp = denoised max-channel PTP; ratio > 1 because:")
print(f"     (a) denoising amplifies neural signal, (b) max-PTP channel may differ from detection ch")

print("\nFinding outlier spikes (low si_ptp, high b_ptp)...")
outlier_local = np.where(
    (si_ptp < np.percentile(si_ptp, 10)) &
    (b_ptp  > np.percentile(b_ptp,  75))
)[0]
print(f"Outlier spikes: {len(outlier_local):,}")

rng = np.random.default_rng(42)
sel_local = rng.choice(outlier_local, size=min(6, len(outlier_local)), replace=False)

valid_idx   = np.where(valid)[0]
sel_30min   = valid_idx[sel_local]
sample_idxs = si_peaks['sample_index'][sel_30min]
ch_idxs     = si_peaks['channel_index'][sel_30min]

# Set up Boussard localizer (requires GPU)
print("Loading Boussard denoiser (requires CUDA)...")
bin_file              = REC_DIR + '/traces_cached_seg0.raw'
dtype_input           = 'float16'
n_channels            = 368
sampling_rate         = 30000
denoiser_weights      = REPO + '/pretrained_denoiser/denoise.pt'
n_filters_denoise     = [16, 8, 4]
filter_sizes_denoise  = [5, 11, 21]
offset_detector_denoiser = 18

with open(REC_DIR + '/probe.json') as f:
    _probe = json.load(f)
geom_array = np.array(_probe['probes'][0]['contact_positions'])  # (368, 2)

localizer_obj = LOCALIZER(
    bin_file, dtype_input,
    peaks_folder + '/boussard_spike_index_30min.npy',
    None,
    geom_array,
    denoiser_weights,
    offset_detector_denoiser,
    n_filters=n_filters_denoise,
    filter_sizes=filter_sizes_denoise,
    sampling_rate=sampling_rate,
)
localizer_obj.load_denoiser()
print("Denoiser loaded.")

# ── Helpers ───────────────────────────────────────────────────────────────────
N_HALF = 4
N_ROWS = 2 * N_HALF + 1   # 9 rows (±4 around reference channel)

# SI localize_peaks uses ms_before=0.5, ms_after=0.5 (defaults in main.py).
# At 30 kHz: nbefore = nafter = 15 samples.  Window = arange(-15, 15) → 30 samples.
# Trough is at Boussard position 42, so SI window maps to Boussard indices [27:57].
SI_NBEFORE   = int(0.5 * sampling_rate / 1000)   # 15
SI_NAFTER    = int(0.5 * sampling_rate / 1000)   # 15
SI_WIN_START = 42 - SI_NBEFORE                    # 27 (first SI sample in Boussard coords)
SI_WIN_END   = 42 + SI_NAFTER                     # 57 (exclusive slice end)

def get_chs(ref_ch, half=N_HALF):
    """±half channels around ref_ch, depth-sorted deepest-first."""
    lo  = max(0, ref_ch - half)
    hi  = min(n_channels - 1, ref_ch + half)
    chs = np.arange(lo, hi + 1)
    return chs[np.argsort(geom_array[chs, 1])[::-1]]

def plot_col(col_axes, raw_wf, denoised_wf, center_ch, si_ch=None, b_ch=None):
    """Plot waveforms centred on center_ch.
    si_ch: if set, mark blue ▽/△ (raw PTP within 30-sample SI window) + shading on that channel's row.
    b_ch:  if set, mark red ▽/△ (denoised PTP over full 121-sample window) on that channel's row.
    """
    chs = get_chs(center_ch)
    t   = np.arange(raw_wf.shape[0])
    for row in range(N_ROWS):
        ax = col_axes[row]
        ax.set_xticks([]); ax.set_yticks([])
        if row < len(chs):
            c = int(chs[row])
            ax.plot(t, raw_wf[:, c],      color='steelblue', lw=0.8, alpha=0.9)
            ax.plot(t, denoised_wf[:, c], color='crimson',   lw=0.8, alpha=0.9)
            ax.axvline(42, color='gray', lw=0.8, ls='--')
            if si_ch is not None and c == si_ch:
                ax.axvspan(SI_WIN_START, SI_WIN_END - 1, alpha=0.08, color='steelblue', zorder=0)
                raw_si = raw_wf[SI_WIN_START:SI_WIN_END, c]
                ti = SI_WIN_START + int(raw_si.argmin())
                ta = SI_WIN_START + int(raw_si.argmax())
                ax.scatter(ti, raw_wf[ti, c], color='steelblue', marker='v', s=40, zorder=5, clip_on=False)
                ax.scatter(ta, raw_wf[ta, c], color='steelblue', marker='^', s=40, zorder=5, clip_on=False)
            if b_ch is not None and c == b_ch:
                ti = int(denoised_wf[:, c].argmin())
                ta = int(denoised_wf[:, c].argmax())
                ax.scatter(ti, denoised_wf[ti, c], color='crimson', marker='v', s=40, zorder=5, clip_on=False)
                ax.scatter(ta, denoised_wf[ta, c], color='crimson', marker='^', s=40, zorder=5, clip_on=False)
            if si_ch is not None and c == si_ch:
                ch_marker = ' ◆'
            elif b_ch is not None and c == b_ch:
                ch_marker = ' ★'
            else:
                ch_marker = ''
            ax.set_ylabel(f'ch{c}{ch_marker}', fontsize=7, rotation=0, ha='right', va='center', labelpad=20)

legend_elements = [
    Line2D([0], [0], color='steelblue', lw=1.5, label='raw'),
    Line2D([0], [0], color='crimson',   lw=1.5, label='denoised'),
    Line2D([0], [0], color='none',      lw=0,   label='◆ = SI detection ch (left col per spike)'),
    Line2D([0], [0], color='none',      lw=0,   label='★ = Boussard denoised max-PTP ch (right col per spike)'),
    Line2D([0], [0], color='steelblue', lw=0, marker='v', markersize=6, label='raw min/max at ◆ within SI 30-sample window'),
    Line2D([0], [0], color='crimson',   lw=0, marker='v', markersize=6, label='denoised min/max at ★ (full 121-sample window)'),
]

# ── Plot: outlier waveforms ───────────────────────────────────────────────────
# Title shows amplitude values per spike:
#   raw_ptp_det  = raw PTP at detection ch (from wf, should ≈ si_ptp from monopole fit)
#   SI_mt        = alpha/r  (SI monopole estimate of raw PTP at detection ch)
#   den_ptp_det  = denoised PTP at detection ch
#   B_ptp_max    = results_max_ptp = max denoised PTP over ALL channels (possibly != detect ch)
n_plot = len(sel_local)
fig, axes = plt.subplots(N_ROWS, n_plot * 2,
                         figsize=(n_plot * 9.0, N_ROWS * 2.0),
                         squeeze=False)
fig.suptitle(
    "Outliers: low SI monopole PTP, high Boussard PTP\n"
    "Left col (per spike) = SI detection ch (◆)  |  Right col = Boussard denoised max-PTP ch (★)\n"
    "Raw (blue) / Denoised (red)  |  ▽/△ = PTP endpoints on respective trace  |  Shading = SI 30-sample window  |  Dashed = trough (pos 42)",
    fontsize=9
)

for col, (samp, mc) in enumerate(zip(sample_idxs, ch_idxs)):
    mc = int(mc)
    center = int(samp) + offset_detector_denoiser
    raw_wf, skip = localizer_obj.read_waveforms(np.array([center]))
    if len(skip) > 0:
        for row in range(N_ROWS):
            axes[row, 2*col].text(0.5, 0.5, 'skip', ha='center', va='center',
                                  transform=axes[row, 2*col].transAxes, fontsize=8)
            axes[row, 2*col+1].text(0.5, 0.5, 'skip', ha='center', va='center',
                                    transform=axes[row, 2*col+1].transAxes, fontsize=8)
        continue

    denoised_wf = localizer_obj.denoise_wf_nn_tmp(raw_wf)[0]
    raw_wf      = raw_wf[0]

    den_ptp_all    = np.ptp(denoised_wf, axis=0)
    den_max_ch     = int(den_ptp_all.argmax())
    den_ptp_max_val = float(den_ptp_all[den_max_ch])
    raw_ptp_si_win = float(raw_wf[SI_WIN_START:SI_WIN_END, mc].max() - raw_wf[SI_WIN_START:SI_WIN_END, mc].min())
    same_str = 'SAME' if den_max_ch == mc else f'det={mc}'
    print(f"  outlier spike {col}: detect_ch={mc}  den_max_ch={den_max_ch} ({'SAME' if den_max_ch==mc else 'DIFF'})"
          f"  raw_ptp_SI_win={raw_ptp_si_win:.1f}µV"
          f"  SI_mt={si_ptp[sel_local[col]]:.1f}µV"
          f"  B_ptp={b_ptp[sel_local[col]]:.1f}µV"
          f"  den_ptp_max={den_ptp_max_val:.1f}µV (ch{den_max_ch})")

    axes[0, 2*col].set_title(
        f'spike {col}  SI det_ch={mc}\n'
        f'raw_ptp(SI_win)={raw_ptp_si_win:.1f} µV\n'
        f'SI_mt={si_ptp[sel_local[col]]:.1f} µV',
        fontsize=7
    )
    axes[0, 2*col+1].set_title(
        f'spike {col}  B max_ch={den_max_ch} [{same_str}]\n'
        f'den_ptp_max={den_ptp_max_val:.1f} µV\n'
        f'B_ptp={b_ptp[sel_local[col]]:.1f} µV',
        fontsize=7
    )
    plot_col([axes[r, 2*col]   for r in range(N_ROWS)], raw_wf, denoised_wf, mc,         si_ch=mc)
    plot_col([axes[r, 2*col+1] for r in range(N_ROWS)], raw_wf, denoised_wf, den_max_ch, b_ch=den_max_ch)
    # unify y-axis within this spike's pair of columns
    spike_axes = [axes[r, 2*col] for r in range(N_ROWS)] + [axes[r, 2*col+1] for r in range(N_ROWS)]
    ylims = [ax.get_ylim() for ax in spike_axes if ax.lines]
    if ylims:
        ylo, yhi = min(v[0] for v in ylims), max(v[1] for v in ylims)
        for ax in spike_axes:
            ax.set_ylim(ylo, yhi)

axes[0, 0].legend(handles=legend_elements, fontsize=7, loc='upper left')
plt.tight_layout(rect=[0, 0, 1, 0.91])
out = peaks_folder + '/waveforms_boussard_si_outliers.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved → {out}")
plt.close()

# =============================================================================
# SECTION 4 — Normal spikes: b_ptp/si_ptp close to the population median (~1.57)
# =============================================================================
print("\nFinding normal spikes (b_ptp/si_ptp close to population median)...")
median_ratio  = np.median(b_ptp / si_ptp)
similar_local = np.where(
    (b_ptp / si_ptp > median_ratio * 0.85) &
    (b_ptp / si_ptp < median_ratio * 1.15) &
    (b_ptp > 10)
)[0]
print(f"Normal spikes (ratio {median_ratio*0.85:.2f}–{median_ratio*1.15:.2f}, b_ptp > 10 µV): {len(similar_local):,}")

rng2 = np.random.default_rng(123)
sel_similar   = rng2.choice(similar_local, size=min(6, len(similar_local)), replace=False)
valid_idx_s   = np.where(valid)[0]
sel_30min_s   = valid_idx_s[sel_similar]
sample_idxs_s = si_peaks['sample_index'][sel_30min_s]
ch_idxs_s     = si_peaks['channel_index'][sel_30min_s]

n_plot_s = len(sel_similar)
fig, axes = plt.subplots(N_ROWS, n_plot_s * 2,
                         figsize=(n_plot_s * 9.0, N_ROWS * 2.0),
                         squeeze=False)
fig.suptitle(
    f"Normal spikes: b_ptp/si_ptp near median ({median_ratio:.2f}), b_ptp > 10 µV\n"
    "Left col (per spike) = SI detection ch (◆)  |  Right col = Boussard denoised max-PTP ch (★)\n"
    "Raw (blue) / Denoised (red)  |  ▽/△ = PTP endpoints on respective trace  |  Shading = SI 30-sample window  |  Dashed = trough (pos 42)",
    fontsize=9
)

for col, (samp, mc) in enumerate(zip(sample_idxs_s, ch_idxs_s)):
    mc = int(mc)
    center = int(samp) + offset_detector_denoiser
    raw_wf, skip = localizer_obj.read_waveforms(np.array([center]))
    if len(skip) > 0:
        for row in range(N_ROWS):
            axes[row, 2*col].text(0.5, 0.5, 'skip', ha='center', va='center',
                                  transform=axes[row, 2*col].transAxes, fontsize=8)
            axes[row, 2*col+1].text(0.5, 0.5, 'skip', ha='center', va='center',
                                    transform=axes[row, 2*col+1].transAxes, fontsize=8)
        continue

    denoised_wf = localizer_obj.denoise_wf_nn_tmp(raw_wf)[0]
    raw_wf      = raw_wf[0]

    den_ptp_all    = np.ptp(denoised_wf, axis=0)
    den_max_ch     = int(den_ptp_all.argmax())
    den_ptp_max_val = float(den_ptp_all[den_max_ch])
    raw_ptp_si_win = float(raw_wf[SI_WIN_START:SI_WIN_END, mc].max() - raw_wf[SI_WIN_START:SI_WIN_END, mc].min())
    same_str = 'SAME' if den_max_ch == mc else f'det={mc}'
    print(f"  normal spike {col}: detect_ch={mc}  den_max_ch={den_max_ch} ({'SAME' if den_max_ch==mc else 'DIFF'})"
          f"  raw_ptp_SI_win={raw_ptp_si_win:.1f}µV"
          f"  SI_mt={si_ptp[sel_similar[col]]:.1f}µV"
          f"  B_ptp={b_ptp[sel_similar[col]]:.1f}µV"
          f"  den_ptp_max={den_ptp_max_val:.1f}µV (ch{den_max_ch})")

    axes[0, 2*col].set_title(
        f'spike {col}  SI det_ch={mc}\n'
        f'raw_ptp(SI_win)={raw_ptp_si_win:.1f} µV\n'
        f'SI_mt={si_ptp[sel_similar[col]]:.1f} µV',
        fontsize=7
    )
    axes[0, 2*col+1].set_title(
        f'spike {col}  B max_ch={den_max_ch} [{same_str}]\n'
        f'den_ptp_max={den_ptp_max_val:.1f} µV\n'
        f'B_ptp={b_ptp[sel_similar[col]]:.1f} µV',
        fontsize=7
    )
    plot_col([axes[r, 2*col]   for r in range(N_ROWS)], raw_wf, denoised_wf, mc,         si_ch=mc)
    plot_col([axes[r, 2*col+1] for r in range(N_ROWS)], raw_wf, denoised_wf, den_max_ch, b_ch=den_max_ch)
    # unify y-axis within this spike's pair of columns
    spike_axes = [axes[r, 2*col] for r in range(N_ROWS)] + [axes[r, 2*col+1] for r in range(N_ROWS)]
    ylims = [ax.get_ylim() for ax in spike_axes if ax.lines]
    if ylims:
        ylo, yhi = min(v[0] for v in ylims), max(v[1] for v in ylims)
        for ax in spike_axes:
            ax.set_ylim(ylo, yhi)

axes[0, 0].legend(handles=legend_elements, fontsize=7, loc='upper left')
plt.tight_layout(rect=[0, 0, 1, 0.91])
out = peaks_folder + '/waveforms_boussard_si_similar.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved → {out}")
plt.close()

print("Done.")
