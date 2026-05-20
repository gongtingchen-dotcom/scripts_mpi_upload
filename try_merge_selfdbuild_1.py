# %% [markdown]
# # **Auto-merging**
# 

# %% [markdown]
# #### **Imports & Settings**

# %%
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from kilosort import CCG
from kilosort.CCG import refract, check_CCG
from tqdm.auto import tqdm

import spikeinterface.full as si
import slay

# ─────────────────────────────────────────────────────────────────────────────

# %%
# =====================================================================
#  PATHS  — edit these before running
# =====================================================================
EXP            = r"/ptmp/gongt/GLC7189/GLC7189_g_g1"
SORTING_DIR    = r"/ptmp/gongt/GLC7189/GLC7189_g_g1/preprocessed/tomato_sorting2_all_try_merge2"
PHY_DIR        = r"/ptmp/gongt/GLC7189/GLC7189_g_g1/preprocessed/tomato_sorting2_all_try_merge2/export_to_phy"
RECORDING_PATH = r"/ptmp/gongt/GLC7189/GLC7189_g_g1/preprocessed"
# Peaks file from thresholding — used to derive peak channel per unit without
# reading the recording at all.  Set to None to fall back to estimate_sparsity.
PEAKS_FILE     = r"/ptmp/gongt/GLC7189/GLC7189_g_g1/preprocessed/thresholding_peaks_3.3/peaks_combined2_sorted.npy"

# Sparse analyzer: built once (or rebuilt when RUN_BUILD_ANALYZER=True).
# Contains mean templates (~32 channels, 90 samples) used for similarity.
ANALYZER_DIR   = PHY_DIR + "/_sparse_analyzer0"

# Output sub-folder for this run's results/plots.
# Change the suffix each time you run on freshly-merged data.
RUN_TAG        = "run1"

# =====================================================================
#  RUN FLAGS  — set True for the stages you want to execute
# =====================================================================
RUN_BUILD_ANALYZER   = False  # (re)build sparse SortingAnalyzer from scratch
RUN_LABEL_CLUSTERS   = False   # compute SNR / ContamPct → cluster_group.tsv
RUN_BUILD_METRICS    = True   # extract mean waveforms + pair similarity table
RUN_MERGE            = True   # find merge groups + save JSONs
RUN_PLOT_MERGES      = True   # render waveform / correlogram figures
RUN_ACCEPT_MERGES    = True  # apply accepted merges to spike arrays (destructive)

# =====================================================================
#  SPARSE ANALYZER SETTINGS  (only used when RUN_BUILD_ANALYZER=True)
# =====================================================================
TARGET_N_CHANNELS    = 32     # approx channels per unit neighbourhood
SAMPLES_BEFORE       = 30 # → pre_samples  = 30 @ 30 kHz
SAMPLES_AFTER        = 35 # → post_samples = 35 @ 30 kHz
MAX_SPIKES_PER_UNIT  = 1000     # for RANDOM spikes

from pathlib import Path
import os
import numpy as np
PHY_DIR        = Path(os.path.normpath(PHY_DIR))
ANALYZER_DIR   = Path(os.path.normpath(ANALYZER_DIR))
SORTING_DIR    = Path(os.path.normpath(SORTING_DIR))
OUT_DIR        = PHY_DIR / RUN_TAG
PLOTS_DIR      = OUT_DIR / "merges"
# OUT_DIR and PLOTS_DIR are created inside __main__ (after spike arrays are built)
# so that PHY_DIR is not pre-created here, which would skip the build-from-sorting step.
print(f"PHY_DIR     : {PHY_DIR}")
print(f"ANALYZER_DIR: {ANALYZER_DIR}")
print(f"OUT_DIR     : {OUT_DIR}")

# %% [markdown]
# #### **Self-build Parameters**

# %%
# ── PARAMETERS ─────────────────────────────────────────────────────────
SELF_BUILD_PARAMS = {
    # ── Clique similarity thresholds ─────────────────────────────────
    # Each member must have sim > strong_sim_thresh with ≥1 other member,
    # AND sim > weak_sim_thresh with ALL other members.
    "strong_sim_thresh": 0.95,   # seed / expand threshold
    "weak_sim_thresh":   0.9,    # all-pairs floor

    # ── Per-unit quality gate ─────────────────────────────────────────
    "max_contam_pct":    0.2,

    # ── SNR homogeneity gate ──────────────────────────────────────────
    # Two units can only merge if min(SNR_a, SNR_b) / max(SNR_a, SNR_b) >= this.
    # Prevents fusing a strong single-unit with a much weaker one.
    # 0.7 = weaker unit must have ≥70% of the stronger unit's SNR.
    "snr_ratio_min":     0.5,

    # ── Optional spatial gate ─────────────────────────────────────────
    "max_dist":          400,    # max peak-channel distance; large = disabled

    # ── XCorr gate (pairwise CCG co-firing check) ─────────────────────
    "window_size":       0.1,    # s — CCG window (±50 ms each side)
    "xcorr_bin_width":   0.001,  # s — CCG bin size (1 ms)
    "min_xcorr_rate":    10.0,   # Hz — min CCG rate for significance

    # ── Kept for plot_corr (slay) ─────────────────────────────────────
    "ref_pen_bin_width": 0.1,    # ms — bin size for ACG/CCG plots
    "overlap_tol":       0.0003, # s  — overlap tolerance

    # ── Waveform extraction ───────────────────────────────────────────
    # post_samples=35 (~1.17 ms) keeps only the spike + fast repolarisation
    # phase where waveform shape actually differs between units.
    # The SI analyzer still uses ms_after=2.0 for sparsity, but similarity
    # is computed on this tighter window.
    "pre_samples":       30,
    "post_samples":      35,
    "max_spikes":        1000,    # used when building mean waveforms (metrics cell)
    "max_spikes_for_plot": 500,  # max spikes displayed per cluster in plots (0 = no limit)
    "sample_rate":       30000.0,

    # ── Labelling thresholds (shared by label_clusters + _predict_merge_label) ──
    "contam_tbin_ms":    1.5,    # R12 CCG bin width; effective window = ±1..10 × tbin
    "isi_ref_ms":        1.5,    # ISI violation refractory threshold (ms)

    # ── Misc ──────────────────────────────────────────────────────────
    "backup_suffix":     "_orig",
    "results_folder":    RUN_TAG,
    "KS_folder":         str(PHY_DIR),
}

print("Parameters set.  OUT_DIR:", OUT_DIR)


# ── Module-level worker function: must be at top level for ProcessPoolExecutor pickling ──
def _contam_isi_one(args):
    """Compute ContamPct (R12) and ISI violation rate for a single cluster.

    args: (idx, st_sec_slice, st_slice, tbin, isi_ref_samples)
      idx           – position in all_cluster_ids (returned so caller can scatter results)
      st_sec_slice  – spike times in seconds for this cluster (1-D float64 array)
      st_slice      – spike times in samples for this cluster (1-D int/float array)
      tbin          – CCG bin width in seconds
      isi_ref       – ISI refractory period in samples
    """
    idx, st_sec_slice, st_slice, tbin, isi_ref = args
    from kilosort.CCG import check_CCG as _check_CCG
    import numpy as _np
    st_sec_sorted = _np.sort(st_sec_slice)
    contam_v = 0.0
    if len(st_sec_sorted) > 10 and (st_sec_sorted[-1] - st_sec_sorted[0]) > 0:
        _, _, contam_v = _check_CCG(st_sec_sorted, tbin=tbin)
    isi_v = 0.0
    if len(st_slice) > 1:
        isis = _np.diff(_np.sort(st_slice))
        isi_v = float((isis < isi_ref).sum()) / len(isis)
    return idx, contam_v, isi_v


# ── Module-level worker: XCorr + cosine metrics for a single unit pair ────────
def _pair_metrics_worker(args):
    c1, c2, st1, st2, mwf1_flat, mwf2_flat, n1_norm, n2_norm, sim_val, cp1, cp2, xp = args
    import numpy as _np
    import slay as _slay
    slay_sim = float(_np.dot(mwf1_flat, mwf2_flat) / (n1_norm * n2_norm)) \
               if (n1_norm > 0 and n2_norm > 0) else 0.0
    xgram = _slay.x_correlogram(
        st1, st2,
        window_size=xp["window_size"],
        bin_width=xp["bin_width"],
        overlap_tol=xp["overlap_tol"],
    )
    xcorr_sig = float(_slay.calc_xcorr_sig(
        xgram,
        xcorr_bin_width=xp["bin_width"],
        min_xcorr_rate=xp["min_xcorr_rate"],
    ))
    return {"Cluster 1": c1, "Cluster 2": c2,
            "Similarity": sim_val, "Slay Similarity": slay_sim,
            "XCorr": xcorr_sig, "ContamPct_1": cp1, "ContamPct_2": cp2}


# %%
# ── Load recording ────────────────────────────────────────────────────

if __name__ == "__main__":
    import time as _time
    _t_start = _time.time()
    def _ts(msg, **kw):
        """Print timestamped message, flushing immediately for sbatch log tailing."""
        elapsed = _time.time() - _t_start
        h, r = divmod(int(elapsed), 3600)
        m, s = divmod(r, 60)
        print(f"[{h:02d}:{m:02d}:{s:02d}] {msg}", flush=True, **kw)

    rec = si.load(str(RECORDING_PATH))
    _ts(f"Recording: {rec.get_num_channels()} ch  {rec.get_sampling_frequency():.0f} Hz  {rec.get_duration():.1f} s")

    # ── Derived waveform window (samples) ─────────────────────────────────
    _fs          = int(rec.get_sampling_frequency())
    pre_samples  = SAMPLES_BEFORE   # 30
    post_samples = SAMPLES_AFTER    # 35

    # ── Build minimal export_to_phy from SORTING_DIR if PHY_DIR is absent ──
    # Check for key files (not just directory existence) so that an empty or
    # partially-populated PHY_DIR also triggers the rebuild.
    if not (PHY_DIR / "spike_times.npy").exists() or not (PHY_DIR / "spike_clusters.npy").exists():
        _ts(f"PHY_DIR not found — building minimal export_to_phy from {SORTING_DIR} …")
        PHY_DIR.mkdir(parents=True, exist_ok=True)

        _sort = si.load(str(SORTING_DIR))
        _ts(f"  Sorting loaded: {len(_sort.unit_ids)} units")

        _st_list, _sc_list = [], []
        for _uid in _sort.unit_ids:
            _train = _sort.get_unit_spike_train(_uid, segment_index=0)
            _st_list.append(_train)
            _sc_list.append(np.full(len(_train), int(_uid), dtype="int32"))
        _spike_times   = np.concatenate(_st_list).astype("uint64")
        _spike_clusters = np.concatenate(_sc_list).astype("int32")
        _order = np.argsort(_spike_times, kind="stable")
        _spike_times    = _spike_times[_order]
        _spike_clusters = _spike_clusters[_order]
        np.save(PHY_DIR / "spike_times.npy",    _spike_times)
        np.save(PHY_DIR / "spike_clusters.npy", _spike_clusters)
        np.save(PHY_DIR / "spike_templates.npy", _spike_clusters.copy())

        # params.py — needed by si.read_phy
        _dtype_str = str(rec.get_dtype())
        (PHY_DIR / "params.py").write_text(
            f"n_channels_dat = {rec.get_num_channels()}\n"
            f"offset = 0\n"
            f"sample_rate = {rec.get_sampling_frequency()}\n"
            f"dtype = '{_dtype_str}'\n"
            f"hp_filtered = False\n"
        )

        _ts(f"  Saved spike_times ({len(_spike_times):,} spikes), "
              f"spike_clusters, spike_templates, params.py → {PHY_DIR}")

    # ── Now that PHY_DIR is guaranteed to exist, create output dirs ──────
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    PLOTS_DIR.mkdir(exist_ok=True, parents=True)

    # ── Load spike arrays ─────────────────────────────────────────────────
    _ts("Loading spike_times.npy …")
    spike_times    = np.load(PHY_DIR / "spike_times.npy").ravel().astype(np.int64)  # cast uint64→int64 to avoid float promotion in indexing
    _ts(f"  spike_times loaded: {len(spike_times):,} spikes")
    spike_clusters = np.load(PHY_DIR / "spike_clusters.npy").ravel()
    _ts(f"  spike_clusters loaded: {spike_clusters.max()+1} max id")
    _ts("Computing noise levels …")
    noise_levels   = si.get_noise_levels(rec, return_scaled=False)
    _ts(f"  noise_levels done: {len(noise_levels)} channels")

    group_path = PHY_DIR / "cluster_group.tsv"
    if not group_path.exists():
        _ts("cluster_group.tsv not found — creating empty one …")
        unique_ids = np.unique(spike_clusters)
        pd.DataFrame({"cluster_id": unique_ids, "label": "", "label_reason": ""}).to_csv(
            group_path, sep="\t", index=False)
        _ts(f"  Created with {len(unique_ids)} clusters")

    group_df        = pd.read_csv(group_path, sep="\t")
    all_cluster_ids = group_df["cluster_id"].values
    _ts(f"Total clusters: {len(all_cluster_ids)}")

    # ── Sync all metrics TSV files to include every ID in spike_clusters.npy ──
    # si.read_phy inner-joins all .tsv files. After a merge round, the new
    # merged cluster IDs are in spike_clusters.npy and cluster_group.tsv but
    # not in the metrics files (ContamPct, ISI, SNR) which were written by the
    # previous labelling step. The inner join would then silently drop those
    # new IDs, causing the analyzer to miss them entirely.
    # Fix: ensure every .tsv with a cluster_id column has all current IDs,
    # filling in NaN for any new ones.
    _sc_ids = set(int(x) for x in np.unique(spike_clusters))
    for _tsv in sorted(PHY_DIR.glob("*.tsv")):
        try:
            _df = pd.read_csv(_tsv, sep="\t")
        except Exception:
            continue
        if "cluster_id" not in _df.columns:
            continue
        _missing = _sc_ids - set(_df["cluster_id"].values)
        if _missing:
            _new_rows = pd.DataFrame({"cluster_id": sorted(_missing)})
            for col in _df.columns:
                if col != "cluster_id":
                    _new_rows[col] = np.nan
            _df = pd.concat([_df, _new_rows], ignore_index=True).sort_values("cluster_id")
            _df.to_csv(_tsv, sep="\t", index=False)
            print(f"  {_tsv.name}: added {len(_missing)} missing ID(s) with NaN metrics")

    # ── Build or load sparse SortingAnalyzer ──────────────────────────────
    # The analyzer stores mean templates for ~32 channels per unit (90 samples).
    # It is rebuilt whenever RUN_BUILD_ANALYZER=True or the folder is missing.
    #
    # IMPORTANT: we load the sorting from the phy export directory (si.read_phy),
    # NOT from SORTING_DIR. The NumpySorting in SORTING_DIR hard-codes unit_ids
    # from before any slay merges (0…N_orig). Slay's accept_all_merges only
    # updates the phy files (spike_clusters.npy etc.), not that folder.
    # Reading from phy ensures all current unit IDs (including merged ones) are
    # included in the sparsity mask.
    #
    # exclude_cluster_groups=[] keeps ALL units (including noise/mua) so that
    # sparsity_mask covers every cluster ID present in spike_clusters.npy.

    def _compute_radius_for_target(analyzer, target_n):
        locs  = analyzer.get_channel_locations()
        dists = np.linalg.norm(locs[:, np.newaxis] - locs[np.newaxis, :], axis=2)
        pitch = np.median(np.sort(dists, axis=1)[:, 1])
        best_r, best_diff = pitch, np.inf
        for r in np.arange(pitch, dists.max() + pitch, pitch / 2):
            med = np.median((dists <= r).sum(axis=1))
            if abs(med - target_n) < best_diff:
                best_diff, best_r = abs(med - target_n), r
            if med > target_n * 1.5:
                break
        _ts(f"  Probe pitch ≈ {pitch:.1f} µm  →  radius = {best_r:.1f} µm  "
              f"≈ {int(np.median((dists <= best_r).sum(axis=1)))} ch (median)")
        return best_r

    if RUN_BUILD_ANALYZER or not ANALYZER_DIR.exists():
        if ANALYZER_DIR.exists():
            _ts(f"Removing existing analyzer at {ANALYZER_DIR} …")
            shutil.rmtree(ANALYZER_DIR)

        # Build NumpySorting directly from already-loaded spike arrays.
        # This is ~100× faster than si.read_phy for 68k units / 129M spikes
        # because si.read_phy re-reads disk and groups in a Python loop.
        # We already have spike_times + spike_clusters in RAM — reuse them.
        _ts("Building NumpySorting from loaded spike arrays (fast path) …")
        import time as _time; _t0 = _time.time()
        _sort_uids = np.unique(spike_clusters)
        _fs_sort   = rec.get_sampling_frequency()
        # argsort once, then slice per-unit — O(n log n) total
        _order_s   = np.argsort(spike_clusters, kind='stable')
        _sc_s      = spike_clusters[_order_s]
        _st_s      = spike_times[_order_s]
        _su, _scnt = np.unique(_sc_s, return_counts=True)
        _sbounds   = np.concatenate([[0], np.cumsum(_scnt)])
        _trains    = {int(_su[_i]): _st_s[_sbounds[_i]:_sbounds[_i+1]]
                      for _i in range(len(_su))}
        del _order_s, _sc_s, _st_s, _su, _scnt, _sbounds
        sort = si.NumpySorting(
            sorting_dict={0: _trains},
            sampling_frequency=_fs_sort,
            unit_ids=_sort_uids,
        )
        del _trains
        _ts(f"  NumpySorting built: {len(sort.unit_ids)} units  "
              f"(ids {sort.unit_ids.min()}…{sort.unit_ids.max()})  "
              f"({_time.time()-_t0:.1f}s)")

        # Step 1 — build ChannelSparsity directly from the peaks file.
        # peaks_combined2_sorted.npy is already sorted by sample_index and
        # contains channel_index for every detected spike — no recording I/O
        # needed at all.  Strategy:
        #   1. searchsorted(peaks['sample_index'], spike_times) to match each
        #      spike to its detected peak channel (O(n log n), instant).
        #   2. Pre-sort spikes by cluster (argsort on spike_clusters) so the
        #      per-unit mode can be computed in one O(n_spikes) pass rather
        #      than 69 k × 129 M boolean scans.
        #   3. Build the radius mask from probe geometry + peak channel index.
        import time as _time; _t0 = _time.time()
        _r_um  = _compute_radius_for_target(rec, TARGET_N_CHANNELS)
        _locs  = rec.get_channel_locations()          # (n_ch, 2)
        _n_ch  = len(rec.channel_ids)

        _peaks_arr = np.load(str(PEAKS_FILE))         # sorted structured array
        _peak_si   = _peaks_arr['sample_index']       # int64, sorted
        _peak_ch   = _peaks_arr['channel_index'].astype(np.int32)

        # For every spike, find its channel via searchsorted on sorted peaks
        _idx     = np.searchsorted(_peak_si, spike_times)
        _idx     = np.clip(_idx, 0, len(_peaks_arr) - 1)
        _hit     = _peak_si[_idx] == spike_times       # True where times match
        _spike_ch = np.where(_hit, _peak_ch[_idx], np.int32(-1))  # -1 = no match
        _ts(f"  Peaks matched: {_hit.sum():,} / {len(spike_times):,} spikes  "
              f"({100*_hit.mean():.1f}%)")
        del _peaks_arr, _peak_si, _peak_ch, _idx, _hit

        # Pre-sort by cluster for O(n_spikes) total grouping
        _order      = np.argsort(spike_clusters, kind='stable')
        _sc_sorted  = spike_clusters[_order]
        _sch_sorted = _spike_ch[_order]
        _uids, _ucounts = np.unique(_sc_sorted, return_counts=True)
        _bounds = np.concatenate([[0], np.cumsum(_ucounts)])
        _uid_to_j = {int(uid): j for j, uid in enumerate(_uids)}
        del _order, _sc_sorted

        # For each unit, mode of matched peak channels → peak_chan index
        _peak_chan = np.zeros(len(sort.unit_ids), dtype=np.int32)
        for _i, _uid in enumerate(sort.unit_ids):
            _j = _uid_to_j.get(int(_uid), -1)
            if _j >= 0:
                _chs = _sch_sorted[_bounds[_j]:_bounds[_j + 1]]
                _chs = _chs[_chs >= 0]
                if len(_chs) > 0:
                    _peak_chan[_i] = np.bincount(_chs, minlength=_n_ch).argmax()
        del _sch_sorted, _bounds, _uid_to_j

        # Build radius mask: (n_units, n_ch)
        _dists = np.linalg.norm(
            _locs[:, np.newaxis] - _locs[np.newaxis, :], axis=2)  # (n_ch, n_ch)
        _mask  = _dists[_peak_chan, :] <= _r_um                    # (n_units, n_ch)
        sparsity = si.ChannelSparsity(
            mask=_mask, unit_ids=sort.unit_ids, channel_ids=rec.channel_ids)
        _ts(f"  Sparsity built from peaks  ({_time.time()-_t0:.1f}s)")
        _counts = _mask.sum(axis=1)
        _ts(f"  Channels/unit: min={_counts.min()}  median={int(np.median(_counts))}  max={_counts.max()}")

        # Step 2 — create sparse analyzer (sparsity metadata only, no waveforms)
        _ts(f"Building sparse analyzer at {ANALYZER_DIR} …")
        analyzer = si.create_sorting_analyzer(
            sorting=sort, recording=rec,
            format="binary_folder", folder=ANALYZER_DIR, overwrite=True,
            sparse=True, sparsity=sparsity)
        _ts("  Sparse analyzer built and saved.")
    else:
        _ts(f"Loading existing sparse analyzer from {ANALYZER_DIR} …")
        # load_extensions=False skips loading the waveforms .npy which can be
        # 100+ GB — only sparsity (core metadata) is needed by this script.
        analyzer = si.load_sorting_analyzer(str(ANALYZER_DIR), load_extensions=False)
        _ts("  Analyzer loaded.")

    # ── Load sparsity mask + mean templates from analyzer ─────────────────
    sparsity_mask = analyzer.sparsity.mask           # (n_units, n_ch) bool
    _ts(f"  sparsity_mask: {sparsity_mask.shape}")


    # %%
    JOB_KWARGS = dict(n_jobs=64, chunk_duration="1s", progress_bar=True, pool_engine="thread")

    # %%
    rec

    # %% [markdown]
    # #### **Quality Metrics & Labelling**

    # %%
    import os
    import numpy as np
    import pandas as pd
    from tqdm.auto import tqdm


    def _assign_label(snr, contam, n_sp):
        """Translate (SNR, ContamPct, spike count) into a good/mua/noise label.

        Single source of truth for label thresholds — called by both
        _compute_unit_metrics() and label_clusters() so that any change only
        needs to be made here.
        """
        if snr > 3.7:
            return "good" if contam < 0.2 else "mua"
        return "noise" if n_sp < 20 else "mua"


    def _compute_unit_metrics(st_samples, mean_wf, noise_levels, fs, contam_tbin_ms=1.5):
        """Single source of truth for SNR, ContamPct (R12 via check_CCG), and label.

        Called by both label_clusters() and _predict_merge_label() so that any
        change to labelling logic only needs to be made once.

        Parameters
        ----------
        st_samples    : 1-D int array  — spike times in samples
        mean_wf       : (n_ch, wf_len) float array — mean waveform (n_ch first)
        noise_levels  : (n_ch,) float array
        fs            : sampling frequency (Hz)
        contam_tbin_ms: CCG bin width in ms (must match label_clusters call site)

        Returns
        -------
        label  : str   — "good", "mua", or "noise"
        snr    : float
        contam : float — R12 ContamPct from check_CCG
        n_sp   : int
        """
        n_sp = len(st_samples)

        # SNR: peak ptp of mean waveform divided by channel noise
        ptp_per_chan = np.ptp(mean_wf, axis=1)       # (n_ch,)
        peak_ch      = int(np.argmax(ptp_per_chan))
        snr          = float(ptp_per_chan[peak_ch] / noise_levels[peak_ch]) \
                       if noise_levels[peak_ch] > 0 else 0.0

        # ContamPct: R12/CCG metric — same as label_clusters()
        contam = 0.0
        st_sec = st_samples.astype(np.float64) / fs
        if n_sp > 10 and (st_sec.max() - st_sec.min()) > 0:
            _, _, contam = check_CCG(st_sec, tbin=contam_tbin_ms / 1000.0)

        label = _assign_label(snr, contam, n_sp)
        return label, snr, contam, n_sp


    def _predict_merge_label(merge, spike_times, spike_clusters,
                              mean_wf_slay, noise_levels,
                              sample_rate=30000.0):
        """Predict the label (good/mua/noise) of a hypothetically merged cluster.

        Delegates to _compute_unit_metrics() — the single source of truth — so
        SNR, ContamPct method, and label thresholds are always consistent with
        label_clusters().
        """
        all_st = np.sort(np.concatenate(
            [spike_times[spike_clusters == uid] for uid in merge]))

        counts     = {uid: int((spike_clusters == uid).sum()) for uid in merge}
        total      = max(sum(counts.values()), 1)
        merged_mwf = sum(counts[uid] * mean_wf_slay[uid].astype(float)
                         for uid in merge) / total   # (n_ch, wf_len)

        label, snr_val, contam, n_sp = _compute_unit_metrics(
            all_st, merged_mwf, noise_levels, fs=sample_rate,
            contam_tbin_ms=SELF_BUILD_PARAMS["contam_tbin_ms"],
        )
        return label, snr_val, n_sp, contam


    def _extract_waveforms_batched(rec, spike_times_per_unit, pre_samples, post_samples,
                                    channel_mask=None, pbar=None, return_mean_only=True):
        """
        Extract waveforms for multiple units in a single pass over the recording.

        Parameters
        ----------
        rec                  : SpikeInterface recording
        spike_times_per_unit : dict  {uid: array of sample indices}
        pre_samples          : int
        post_samples         : int
        channel_mask         : dict {uid: array of channel indices} or None.
                               If provided, each unit's waveforms are sliced to
                               its active channels only.
        pbar                 : optional tqdm progress bar
        return_mean_only     : bool (default True)
            When True, accumulate a running sum+count per unit and return
            {uid: mean_wf (wf_len, n_active)} — uses O(n_units × wf_len × n_ch)
            RAM instead of O(n_units × n_spikes × wf_len × n_ch).
            When False, store every waveform and return
            {uid: (n_spikes, wf_len, n_active)}.

        Notes
        -----
        Spikes whose window straddles a 5-second chunk boundary are silently
        skipped (rare; < 0.1 % of spikes).
        """
        n_total    = rec.get_total_samples()
        n_channels = rec.get_num_channels()
        wf_len     = pre_samples + post_samples

        def _n_out(uid):
            if channel_mask is not None and uid in channel_mask:
                return len(channel_mask[uid])
            return n_channels

        all_spikes  = []
        valid_times = {}
        for uid, times in spike_times_per_unit.items():
            vtimes = []
            for t in times:
                if pre_samples <= t < n_total - post_samples:
                    all_spikes.append((t, uid, len(vtimes)))
                    vtimes.append(t)
            valid_times[uid] = vtimes
        all_spikes.sort(key=lambda x: x[0])

        if return_mean_only:
            # Running sum + count — O(n_units × wf_len × n_ch) RAM
            wf_sum = {uid: np.zeros((wf_len, _n_out(uid)), dtype=np.float64)
                      for uid in valid_times}
            wf_cnt = {uid: 0 for uid in valid_times}
        else:
            wfs = {uid: np.zeros((len(vtimes), wf_len, _n_out(uid)), dtype=np.float32)
                   for uid, vtimes in valid_times.items()}

        if len(all_spikes) == 0:
            if return_mean_only:
                return {uid: wf_sum[uid].astype(np.float32) for uid in wf_sum}
            return wfs

        fs             = rec.get_sampling_frequency()
        CHUNK          = int(5 * fs)
        OVERLAP        = pre_samples + post_samples
        chunk_start    = 0
        spike_ptr      = 0
        n_spikes_total = len(all_spikes)

        while spike_ptr < n_spikes_total:
            chunk_end       = min(chunk_start + CHUNK, n_total)
            spikes_in_chunk = []
            tmp_ptr         = spike_ptr
            while tmp_ptr < n_spikes_total and all_spikes[tmp_ptr][0] < chunk_end:
                t, uid, idx = all_spikes[tmp_ptr]
                if t - pre_samples >= chunk_start and t + post_samples <= chunk_end:
                    spikes_in_chunk.append((t, uid, idx))
                tmp_ptr += 1

            if spikes_in_chunk:
                load_start = max(spikes_in_chunk[0][0]  - pre_samples, 0)
                load_end   = min(spikes_in_chunk[-1][0] + post_samples, n_total)
                chunk_data = rec.get_traces(start_frame=load_start, end_frame=load_end)
                # Vectorised snippet extraction — build index array then slice once,
                # avoiding the per-spike Python loop (major bottleneck for 68k× 1000-spike units).
                _ts_arr  = np.array([t for t, _, _ in spikes_in_chunk], dtype=np.int64)
                _offsets = _ts_arr - pre_samples - np.int64(load_start)  # explicit int64 cast: spike_times are int64 but load_start could be uint64
                _idx_mat = _offsets[:, None] + np.arange(wf_len, dtype=np.int64)  # (n_sp_chunk, wf_len)
                _snips   = chunk_data[_idx_mat]  # (n_sp_chunk, wf_len, n_ch_all)
                for _k, (t, uid, idx) in enumerate(spikes_in_chunk):
                    snippet = _snips[_k]                        # (wf_len, n_ch_all)
                    if channel_mask is not None and uid in channel_mask:
                        snippet = snippet[:, channel_mask[uid]]  # (wf_len, n_active)
                    if return_mean_only:
                        wf_sum[uid] += snippet
                        wf_cnt[uid] += 1
                    else:
                        wfs[uid][idx] = snippet
                del _snips, _idx_mat, _offsets, _ts_arr
                spike_ptr = tmp_ptr
                if pbar is not None:
                    pbar.update(len(spikes_in_chunk))
            else:
                spike_ptr = tmp_ptr

            if chunk_end >= n_total:
                break
            chunk_start = chunk_end - OVERLAP

        if return_mean_only:
            return {
                uid: (wf_sum[uid] / wf_cnt[uid]).astype(np.float32)
                     if wf_cnt[uid] > 0
                     else np.zeros((wf_len, _n_out(uid)), dtype=np.float32)
                for uid in wf_sum
            }
        return wfs


    def label_clusters(
        spike_times, spike_clusters, all_cluster_ids, rec, noise_levels,
        sparsity_mask,
        pre_samples=30, post_samples=35, max_spikes_per_unit=1000,
        overwrite=False,
        seed=0,
        contam_tbin_ms=1.5,
        isi_ref_ms=1.5,
    ):
        """
        Compute SNR, ContamPct, and ISI violation rate for every cluster, assign
        good/mua/noise labels, and write:
            cluster_group.tsv / cluster_snr.tsv / cluster_ContamPct.tsv / cluster_isi_violation.tsv

        Waveforms are extracted only on the ~32 sparse channels for each unit
        (from sparsity_mask), which is faster and matches the SI template window.
        A fixed RNG seed is used for any subsampling so results are reproducible.

        Parameters
        ----------
        sparsity_mask : np.ndarray, shape (n_units, n_ch), bool
            sparsity_mask[uid] gives the active channels for unit uid.
            Must be indexed by cluster_id (sequential 0-N assumed).
        contam_tbin_ms : float
            CCG bin width in milliseconds for the R12 (ContamPct) metric.
            The refractory windows tested are ±1..10 × tbin.  Default 1.0 ms
            gives a maximum window of ±10 ms.  The lag-0 bin (double-spikes)
            is excluded by design (KiloSort convention).
        isi_ref_ms : float
            Refractory period threshold (ms) for the ISI violation rate saved
            as cluster_isi_violation.tsv.  Fraction of consecutive spike-pair
            ISIs shorter than this threshold.  Does NOT explicitly exclude lag-0
            spikes, but duplicate spikes are typically absent after KiloSort detection.
        """
        snr_path    = PHY_DIR / "cluster_snr.tsv"
        contam_path = PHY_DIR / "cluster_ContamPct.tsv"
        isi_path    = PHY_DIR / "cluster_isi_violation.tsv"
        group_path  = PHY_DIR / "cluster_group.tsv"

        metrics_exist = snr_path.exists() and contam_path.exists()
        if not overwrite and metrics_exist:
            print("Metrics files exist and overwrite=False; loading from disk.")
            snr_df     = pd.read_csv(snr_path, sep="\t")
            contam_df  = pd.read_csv(contam_path, sep="\t")
            snr        = snr_df.set_index("cluster_id").reindex(all_cluster_ids)["SNR"].values
            contam_all = contam_df.set_index("cluster_id").reindex(all_cluster_ids)["ContamPct"].values
            n_spikes   = np.array([np.sum(spike_clusters == clu) for clu in all_cluster_ids])
            if isi_path.exists():
                isi_df       = pd.read_csv(isi_path, sep="\t")
                isi_viol_all = isi_df.set_index("cluster_id").reindex(all_cluster_ids)["ISI_violation"].values
            else:
                isi_viol_all = np.full(len(all_cluster_ids), np.nan)

        else:
            # Map each cluster_id to its row index in sparsity_mask
            _sparsity_uid_to_row = {int(cid): i for i, cid in enumerate(all_cluster_ids)}

            rng      = np.random.default_rng(seed)
            peak_ptp = np.zeros(len(all_cluster_ids))
            peak_chan = np.zeros(len(all_cluster_ids), dtype=int)
            n_spikes = np.zeros(len(all_cluster_ids), dtype=int)

            # Pre-group spikes by cluster with a single argsort — O(n_spikes log n_spikes)
            # instead of O(n_clusters × n_spikes) np.where scans (was the main bottleneck
            # with 68 k clusters × 129 M spikes).
            print("  Pre-grouping spikes by cluster …")
            _grp_order = np.argsort(spike_clusters, kind='stable')
            _sc_grp    = spike_clusters[_grp_order]
            _st_grp    = spike_times[_grp_order]
            _uids_g, _ucnts_g = np.unique(_sc_grp, return_counts=True)
            _ubounds_g = np.concatenate([[0], np.cumsum(_ucnts_g)])
            _uid_to_g  = {int(u): i for i, u in enumerate(_uids_g)}
            print(f"  Grouping done.")

            # Build per-unit spike-time dict (same subsampling as before, same rng/seed)
            # and channel mask, then extract all waveforms in a single sorted pass.
            _st_per_unit = {}
            _ch_mask_lbl = {}
            for idx, clu in enumerate(all_cluster_ids):
                _gi = _uid_to_g.get(int(clu), -1)
                if _gi < 0:
                    continue
                s, e = int(_ubounds_g[_gi]), int(_ubounds_g[_gi + 1])
                n_spikes[idx] = e - s
                if n_spikes[idx] == 0:
                    continue
                if n_spikes[idx] > max_spikes_per_unit:
                    _sel = rng.choice(e - s, max_spikes_per_unit, replace=False)
                    _st_per_unit[int(clu)] = np.sort(_st_grp[s:e][_sel])
                else:
                    _st_per_unit[int(clu)] = np.sort(_st_grp[s:e])
                if clu in _sparsity_uid_to_row:
                    _ch_mask_lbl[int(clu)] = np.where(sparsity_mask[_sparsity_uid_to_row[clu]])[0]
                else:
                    _ch_mask_lbl[int(clu)] = np.arange(rec.get_num_channels())

            print(f"  Extracting waveforms in a single sorted pass over {len(_st_per_unit)} units …")
            # return_mean_only=True: accumulates running sum+count per unit —
            # returns {uid: mean_wf (wf_len, n_active)} using ~600 MB instead of ~550 GB.
            _mean_wfs = _extract_waveforms_batched(
                rec, _st_per_unit, pre_samples, post_samples, channel_mask=_ch_mask_lbl,
                return_mean_only=True)

            # Compute peak PTP from mean waveform per unit
            _clu_to_idx = {int(clu): idx for idx, clu in enumerate(all_cluster_ids)}
            for uid, mean_wf_sp in _mean_wfs.items():  # mean_wf_sp: (wf_len, n_active)
                idx = _clu_to_idx.get(uid, -1)
                if idx < 0 or not np.any(mean_wf_sp != 0):
                    continue
                ptp_per_ch = np.ptp(mean_wf_sp, axis=0)          # (n_active,)
                best_local = np.argmax(ptp_per_ch)
                active_ch  = _ch_mask_lbl[uid]
                peak_chan[idx] = active_ch[best_local]
                peak_ptp[idx]  = ptp_per_ch[best_local]
            del _mean_wfs

            snr = peak_ptp / noise_levels[peak_chan]

            # ContamPct = R12 (KiloSort metric): min ACG rate in ±1..10×tbin windows
            # divided by baseline rate.  Lag-0 bin (double-spikes) is excluded (K[nbins]=0).
            # ISI violation rate: fraction of consecutive ISIs < isi_ref_ms (simpler, no exclusion).
            print(f"Computing ContamPct (R12, tbin={contam_tbin_ms} ms) "
                  f"and ISI violation rate (ref={isi_ref_ms} ms) …")
            fs           = rec.get_sampling_frequency()
            _tbin        = contam_tbin_ms / 1000.0
            _isi_ref     = isi_ref_ms / 1000.0 * fs   # threshold in samples
            contam_all   = np.zeros(len(all_cluster_ids))
            isi_viol_all = np.zeros(len(all_cluster_ids))

            # Reuse the pre-sorted grouping built above — no per-cluster full scans.
            # Parallelise across clusters with ProcessPoolExecutor (check_CCG releases GIL
            # minimally, so threads don't help; processes avoid the GIL entirely).
            _st_sec_grp = _st_grp.astype(np.float64) / fs

            # Build job list — slice the per-cluster spike arrays here so that
            # _contam_isi_one (a module-level function) receives picklable plain
            # numpy arrays rather than indices into shared locals.
            import concurrent.futures as _cf
            _jobs_ci = []
            for idx, clu in enumerate(all_cluster_ids):
                _gi = _uid_to_g.get(int(clu), -1)
                if _gi < 0:
                    continue
                s, e = int(_ubounds_g[_gi]), int(_ubounds_g[_gi + 1])
                _jobs_ci.append((
                    idx,
                    _st_sec_grp[s:e].copy(),   # small per-unit float64 array
                    _st_grp[s:e].copy(),        # small per-unit int/float array
                    _tbin,
                    _isi_ref,
                ))

            _N_WORKERS_CI   = min(64, len(_jobs_ci))
            print(f"Computing ContamPct/ISI for {len(_jobs_ci)} clusters "
                  f"on {_N_WORKERS_CI} workers …")
            with _cf.ProcessPoolExecutor(max_workers=_N_WORKERS_CI) as _pool_ci:
                for _idx_ci, _c_ci, _i_ci in tqdm(
                        _pool_ci.map(_contam_isi_one, _jobs_ci, chunksize=256),
                        total=len(_jobs_ci), desc="ContamPct/ISI"):
                    contam_all[_idx_ci]   = _c_ci
                    isi_viol_all[_idx_ci] = _i_ci
            del _grp_order, _sc_grp, _st_grp, _uids_g, _ucnts_g, _ubounds_g, _uid_to_g, _st_sec_grp

            snr_df    = pd.DataFrame({"cluster_id": all_cluster_ids, "SNR": np.round(snr, 4)})
            contam_df = pd.DataFrame({"cluster_id": all_cluster_ids, "ContamPct": np.round(contam_all, 4)})
            isi_df    = pd.DataFrame({"cluster_id": all_cluster_ids, "ISI_violation": np.round(isi_viol_all, 4)})
            snr_df.to_csv(snr_path, sep="\t", index=False)
            contam_df.to_csv(contam_path, sep="\t", index=False)
            isi_df.to_csv(isi_path, sep="\t", index=False)

        # ── Assign labels ──────────────────────────────────────────────────
        labels = np.array([
            _assign_label(snr[i], contam_all[i], n_spikes[i])
            for i in range(len(all_cluster_ids))
        ])

        _ts(f"  good: {(labels=='good').sum()}, mua: {(labels=='mua').sum()}, noise: {(labels=='noise').sum()}")

        # ── Update cluster_group.tsv (preserve existing 'merged' labels) ──
        if group_path.exists():
            existing = pd.read_csv(group_path, sep="\t")
            if "cluster_id" in existing.columns:
                existing = existing.set_index("cluster_id")
        else:
            existing = pd.DataFrame(columns=["label", "label_reason"])
            existing.index.name = "cluster_id"

        new_rows = []
        preserved = assigned = 0
        for i, cid in enumerate(all_cluster_ids):
            is_merged = (
                cid in existing.index
                and "label" in existing.columns
                and existing.loc[cid, "label"] == "merged"
            )
            reason = (existing.loc[cid, "label_reason"]
                      if (cid in existing.index and "label_reason" in existing.columns) else "")
            if is_merged:
                new_rows.append({"cluster_id": cid, "label": "merged", "label_reason": reason})
                preserved += 1
            else:
                new_rows.append({"cluster_id": cid, "label": labels[i], "label_reason": reason})
                assigned += 1

        pd.DataFrame(new_rows, columns=["cluster_id", "label", "label_reason"]).to_csv(
            group_path, sep="\t", index=False)

        _ts(f"Preserved {preserved} 'merged' labels; assigned {assigned} SNR-based labels.")
        _ts(f"Written: cluster_group.tsv, cluster_snr.tsv, cluster_ContamPct.tsv, cluster_isi_violation.tsv")
        return snr, contam_all, isi_viol_all, labels, n_spikes


    # ── RUN_LABEL_CLUSTERS ────────────────────────────────────────────────
    # spike_times, spike_clusters, noise_levels, all_cluster_ids are already
    # loaded above in the spike-array loading section — no redundant reload.
    _ts(f"=== RUN_LABEL_CLUSTERS={RUN_LABEL_CLUSTERS} ===")
    if RUN_LABEL_CLUSTERS:
        snr, contam_all, isi_viol_all, labels, n_spikes_arr = label_clusters(
            spike_times, spike_clusters, all_cluster_ids, rec, noise_levels,
            sparsity_mask=sparsity_mask,
            pre_samples=pre_samples, post_samples=post_samples,
            max_spikes_per_unit=MAX_SPIKES_PER_UNIT,
            overwrite=True,
            seed=0,
            contam_tbin_ms=SELF_BUILD_PARAMS["contam_tbin_ms"],
            isi_ref_ms=SELF_BUILD_PARAMS["isi_ref_ms"],
        )
        _ts("label_clusters done.")
    else:
        # Load from disk so downstream cells that use snr/contam_all/labels work
        # even when the labelling step is skipped.
        _ts("RUN_LABEL_CLUSTERS=False — loading existing TSV files …")
        _snr_path    = PHY_DIR / "cluster_snr.tsv"
        _contam_path = PHY_DIR / "cluster_ContamPct.tsv"
        _isi_path    = PHY_DIR / "cluster_isi_violation.tsv"
        assert _snr_path.exists() and _contam_path.exists(), \
            "cluster_snr.tsv / cluster_ContamPct.tsv missing; run with RUN_LABEL_CLUSTERS=True first."
        _snr_df    = pd.read_csv(_snr_path,    sep="\t").set_index("cluster_id")
        _contam_df = pd.read_csv(_contam_path, sep="\t").set_index("cluster_id")
        snr        = _snr_df.reindex(all_cluster_ids)["SNR"].values
        contam_all = _contam_df.reindex(all_cluster_ids)["ContamPct"].values
        if _isi_path.exists():
            isi_viol_all = pd.read_csv(_isi_path, sep="\t").set_index("cluster_id").reindex(all_cluster_ids)["ISI_violation"].values
        else:
            isi_viol_all = np.full(len(all_cluster_ids), np.nan)
        n_spikes_arr = np.array([np.sum(spike_clusters == c) for c in all_cluster_ids])
        group_df_tmp = pd.read_csv(PHY_DIR / "cluster_group.tsv", sep="\t").set_index("cluster_id")
        labels = group_df_tmp.reindex(all_cluster_ids)["label"].fillna("mua").values
        _ts(f"  Loaded SNR/ContamPct/ISI for {len(all_cluster_ids)} clusters.")


    # %% [markdown]
    # #### **Waveform Extraction Helper** (function defined earlier, before label_clusters)

    # %%
    import matplotlib.pyplot as plt

    plt.ioff()


    # %% [markdown]
    # #### **Build Metrics Table**

    # %%
    if not RUN_BUILD_METRICS:
        # ── Load all precomputed outputs from disk ─────────────────────────
        _ts("=== RUN_BUILD_METRICS=False — loading from disk ===")

        _mwf_p = next((p for p in [OUT_DIR / "mean_waveforms.npy",
                                    PHY_DIR / "mean_waveforms.npy"] if p.exists()), None)
        assert _mwf_p is not None, \
            "mean_waveforms.npy not found; run with RUN_BUILD_METRICS=True first."
        mean_wf_slay = np.load(_mwf_p)

        _sim_p = PHY_DIR / "similarity_index.npy"
        assert _sim_p.exists(), \
            "similarity_index.npy not found; run with RUN_BUILD_METRICS=True first."
        cosine_sim_good = np.load(_sim_p)   # (n_good+n_mua, n_good+n_mua) proximity-filtered

        _met_p = OUT_DIR / "metrics.tsv"
        assert _met_p.exists(), \
            "metrics.tsv not found; run with RUN_BUILD_METRICS=True first."
        metrics_df = pd.read_csv(_met_p, sep="\t")

        contam_pct_df = pd.read_csv(PHY_DIR / "cluster_ContamPct.tsv", sep="\t").set_index("cluster_id")
        group_df_cur  = pd.read_csv(PHY_DIR / "cluster_group.tsv", sep="\t")
        good_ids      = group_df_cur[group_df_cur["label"] == "good"]["cluster_id"].values
        mua_ids       = group_df_cur[group_df_cur["label"] == "mua"]["cluster_id"].values

        _sim_ids_arr   = np.array(sorted(
            set(good_ids.astype(int)) | set(mua_ids.astype(int))), dtype=int)
        _sim_id_to_pos = {int(uid): i for i, uid in enumerate(_sim_ids_arr)}
        sim_matrix     = cosine_sim_good  # indexed by position in _sim_ids_arr
        def _sim_lookup(uid1, uid2):
            p1 = _sim_id_to_pos.get(int(uid1), -1)
            p2 = _sim_id_to_pos.get(int(uid2), -1)
            if p1 < 0 or p2 < 0:
                return 0.0
            return float(cosine_sim_good[p1, p2])

        _ts(f"  mean_wf_slay   : {mean_wf_slay.shape}")
        _ts(f"  sim_matrix     : {cosine_sim_good.shape} (good+mua, proximity-filtered)")
        N_wf = mean_wf_slay.shape[0]  # needed by RUN_MERGE (_peak_chans indexing)
        _ts(f"  metrics_df     : {len(metrics_df)} pairs")
        _ts(f"  good_ids       : {len(good_ids)}  |  mua_ids: {len(mua_ids)}")

    else:
        _ts("=== RUN_BUILD_METRICS: computing waveforms + cosine sim + XCorr ===")
        # ── Compute everything from scratch and save to disk ───────────────
        #   1. Extract per-cluster mean waveforms (sparse channels, max_spikes limit)
        #   2. Compute cosine similarity matrix  → saved to similarity_index.npy
        #   2b. Rebuild templates.npy / template_ind.npy (keeps phy consistent)
        #   3. Compute per-pair XCorr metrics → saved to metrics.tsv

        contam_pct_df = pd.read_csv(PHY_DIR / "cluster_ContamPct.tsv", sep="\t").set_index("cluster_id")
        group_df_cur  = pd.read_csv(PHY_DIR / "cluster_group.tsv", sep="\t")
        good_ids = group_df_cur[group_df_cur["label"] == "good"]["cluster_id"].values
        _ts(f"Good units: {len(good_ids)}  |  Total clusters: {len(group_df_cur)}")

        pre  = SELF_BUILD_PARAMS["pre_samples"]    # 30
        post = SELF_BUILD_PARAMS["post_samples"]   # 35
        n_ch = rec.get_num_channels()
        rng  = np.random.default_rng(0)

        N_wf = int(spike_clusters.max()) + 1

        # ── Per-unit channel mask from sparsity ────────────────────────────
        _sparsity_uid_to_row = {int(cid): i for i, cid in enumerate(all_cluster_ids)}
        def _channel_mask_for_unit(uid):
            if uid in _sparsity_uid_to_row:
                return np.where(sparsity_mask[_sparsity_uid_to_row[uid]])[0]
            return np.arange(n_ch)

        # ── 1. Extract mean waveforms ──────────────────────────────────────
        _ts(f"Extracting mean waveforms for {N_wf} clusters (sparse channels, "
              f"max {SELF_BUILD_PARAMS['max_spikes']} spikes each) …")

        active_ids = np.unique(spike_clusters)
        _spike_times_for_wf = {}
        for uid in active_ids:
            st = spike_times[spike_clusters == uid]
            if len(st) > SELF_BUILD_PARAMS["max_spikes"]:
                st = rng.choice(st, SELF_BUILD_PARAMS["max_spikes"], replace=False)
            _spike_times_for_wf[int(uid)] = np.sort(st)

        _ch_mask = {int(uid): _channel_mask_for_unit(int(uid)) for uid in active_ids}

        _all_mean_wfs = _extract_waveforms_batched(
            rec, _spike_times_for_wf, pre, post, channel_mask=_ch_mask,
            return_mean_only=True)  # returns {uid: mean_wf (wf_len, n_active)}

        mean_wf_slay = np.zeros((N_wf, n_ch, pre + post), dtype=np.float32)
        for uid in active_ids:
            uid        = int(uid)
            mwf_sparse = _all_mean_wfs.get(uid)   # (wf_len, n_active)
            if mwf_sparse is None or not np.any(mwf_sparse != 0):
                continue
            active_ch  = _ch_mask[uid]
            mean_wf_slay[uid][active_ch, :] = mwf_sparse.T       # (n_active, wf_len)
        del _all_mean_wfs

        _mean_wf_path = OUT_DIR / "mean_waveforms.npy"
        np.save(_mean_wf_path, mean_wf_slay)
        np.save(PHY_DIR / "mean_waveforms.npy", mean_wf_slay)
        _ts(f"  Mean waveforms saved → {_mean_wf_path}  shape={mean_wf_slay.shape}")

        # ── 2. Cosine similarity matrix (good+mua, proximity-filtered) ───
        # Include both good and mua units (for phy inspection of all non-noise
        # units), but zero out pairs whose sparsity channels don't overlap at all
        # (different brain regions → guaranteed non-matching).  This avoids the
        # full N_wf×N_wf (~19 GB) matrix while keeping every meaningful pair.
        mua_ids         = group_df_cur[group_df_cur["label"] == "mua"]["cluster_id"].values
        _sim_ids_arr    = np.array(sorted(
            set(good_ids.astype(int)) | set(mua_ids.astype(int))), dtype=int)
        _sim_id_to_pos  = {int(uid): i for i, uid in enumerate(_sim_ids_arr)}
        n_sim           = len(_sim_ids_arr)
        _ts(f"  Building similarity for {n_sim} units "
              f"(good={len(good_ids)}, mua={len(mua_ids)}) …")

        _W_all  = mean_wf_slay.reshape(N_wf, -1).astype(np.float32)
        _norms  = np.linalg.norm(_W_all, axis=1, keepdims=True)
        _norms[_norms == 0] = 1.0
        _W_all  /= _norms                                     # normalize in-place
        del _norms

        # Spatial proximity mask: True if two units share ≥1 sparsity channel
        _sp_rows = np.zeros((n_sim, sparsity_mask.shape[1]), dtype=bool)
        for _i, _uid in enumerate(_sim_ids_arr):
            if _uid in _sparsity_uid_to_row:
                _sp_rows[_i] = sparsity_mask[_sparsity_uid_to_row[_uid]]
        _prox_mask = (_sp_rows @ _sp_rows.T) > 0             # (n_sim, n_sim) bool
        del _sp_rows

        # Cosine similarity matrix — try GPU (cuBLAS via torch), fall back to CPU numpy.
        # With an A100 this matrix multiply is ~100× faster than CPU numpy for n_sim ~50k.
        _W_sim = _W_all[_sim_ids_arr]   # (n_sim, flat_wf)
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _ts(f"  Using GPU for cosine sim ({n_sim}×{n_sim}) …")
                _torch.backends.cuda.matmul.allow_tf32 = False  # keep full float32 precision
                _W_gpu = _torch.from_numpy(_W_sim).cuda()
                cosine_sim_good = (_W_gpu @ _W_gpu.T).cpu().numpy()
                del _W_gpu
                _torch.cuda.empty_cache()
            else:
                raise RuntimeError("CUDA not available")
        except Exception as _e:
            _ts(f"  GPU unavailable ({_e}), using CPU numpy …")
            cosine_sim_good = _W_sim @ _W_sim.T
        del _W_all, _W_sim
        np.fill_diagonal(cosine_sim_good, 1.0)
        cosine_sim_good[~_prox_mask] = 0.0                   # zero non-proximal pairs
        del _prox_mask

        sim_matrix = cosine_sim_good  # indexed by position in _sim_ids_arr
        def _sim_lookup(uid1, uid2):
            """Return cosine similarity between two unit IDs (0 if not both visible units)."""
            p1 = _sim_id_to_pos.get(int(uid1), -1)
            p2 = _sim_id_to_pos.get(int(uid2), -1)
            if p1 < 0 or p2 < 0:
                return 0.0
            return float(cosine_sim_good[p1, p2])
        np.save(PHY_DIR / "similarity_index.npy",  cosine_sim_good.astype(np.float32))
        np.save(PHY_DIR / "similar_templates.npy", cosine_sim_good.astype(np.float32))
        _n_prox_pairs = int(np.sum(np.triu(cosine_sim_good > 0, k=1)))
        _ts(f"  Cosine similarity (good+mua, proximity-filtered): {cosine_sim_good.shape}  "
              f"non-zero pairs: {_n_prox_pairs}  saved → similarity_index.npy")

        # ── 2b. Rebuild templates.npy / template_ind.npy ──────────────────
        _ts("Rebuilding templates.npy / template_ind.npy …")
        _existing_tmpl_path = PHY_DIR / "templates.npy"
        if _existing_tmpl_path.exists():
            _n_sparse_chans = int(np.load(_existing_tmpl_path).shape[2])
        else:
            _n_sparse_chans = TARGET_N_CHANNELS
        _ch_locs = np.array(rec.get_channel_locations())  # (n_ch, 2)

        def _sparse_chans_for_tmpl(uid):
            uid = int(uid)
            if uid in _sparsity_uid_to_row:
                return np.where(sparsity_mask[_sparsity_uid_to_row[uid]])[0][:_n_sparse_chans]
            amp_per_ch = np.max(np.abs(mean_wf_slay[uid]), axis=1)
            peak_ch    = int(np.argmax(amp_per_ch))
            dists      = np.linalg.norm(_ch_locs - _ch_locs[peak_ch], axis=1)
            return np.argsort(dists)[:_n_sparse_chans]

        _wf_len        = pre + post
        _templates_new = np.zeros((N_wf, _wf_len, _n_sparse_chans), dtype=np.float32)
        _tmpl_ind_new  = -np.ones((N_wf, _n_sparse_chans), dtype=np.int64)

        for _uid in active_ids:
            _ch = _sparse_chans_for_tmpl(_uid)
            _tmpl_ind_new[int(_uid), :len(_ch)]     = _ch
            _templates_new[int(_uid), :, :len(_ch)] = mean_wf_slay[int(_uid)][_ch, :].T

        np.save(PHY_DIR / "templates.npy",      _templates_new)
        np.save(PHY_DIR / "template_ind.npy",   _tmpl_ind_new)
        np.save(PHY_DIR / "pc_feature_ind.npy", _tmpl_ind_new)
        np.save(PHY_DIR / "spike_templates.npy", spike_clusters.astype(np.int32))
        _ts(f"  templates       : {_templates_new.shape}")
        _ts(f"  template_ind    : {_tmpl_ind_new.shape}")
        _ts(f"  spike_templates : overwritten to match spike_clusters  (max={int(spike_clusters.max())})")

        # ── 3. Spike times in seconds per good unit (for XCorr) ───────────
        fs = SELF_BUILD_PARAMS["sample_rate"]
        _times_sec = {
            int(uid): (spike_times[spike_clusters == uid] / fs).astype(np.float64)
            for uid in good_ids
        }

        # ── 4. Candidate pairs above weak similarity threshold (good×good) ─
        # cosine_sim_good covers good+mua but XCorr/merging only needs good×good
        _weak_thresh = SELF_BUILD_PARAMS["weak_sim_thresh"]
        _good_pos_in_sim = np.array([_sim_id_to_pos[int(uid)] for uid in good_ids
                                     if int(uid) in _sim_id_to_pos], dtype=int)
        _good_ids_for_pairs = np.array([int(uid) for uid in good_ids
                                         if int(uid) in _sim_id_to_pos], dtype=int)
        _sim_good_only  = cosine_sim_good[np.ix_(_good_pos_in_sim, _good_pos_in_sim)]
        _pair_ii, _pair_jj = np.where(np.triu(_sim_good_only > _weak_thresh, k=1))
        pairs = [(int(_good_ids_for_pairs[_pair_ii[k]]), int(_good_ids_for_pairs[_pair_jj[k]]))
                 for k in range(len(_pair_ii))]
        del _sim_good_only
        _ts(f"Candidate pairs (cosine sim > {_weak_thresh}): {len(pairs)}")

        # ── 5. Per-pair metrics (XCorr significance + quality columns) ────
        # Parallelised over pairs — each xcorr is independent CPU work.
        # Use processes (not threads) so slay's C extension releases the GIL properly.
        _xcorr_params = dict(
            window_size    = SELF_BUILD_PARAMS["window_size"],
            bin_width      = SELF_BUILD_PARAMS["xcorr_bin_width"],
            overlap_tol    = SELF_BUILD_PARAMS["overlap_tol"],
            min_xcorr_rate = SELF_BUILD_PARAMS["min_xcorr_rate"],
        )
        _contam_lookup = {int(cid): float(contam_pct_df.loc[cid, "ContamPct"])
                          if cid in contam_pct_df.index else float("nan")
                          for cid in np.unique(np.array(pairs).ravel()) if len(pairs) > 0}

        # _pair_metrics_worker is defined at module level (picklable for ProcessPoolExecutor)
        _xcorr_params_w = _xcorr_params  # kept for clarity; data is passed in args
        _mean_wf_flat   = {int(uid): mean_wf_slay[uid].flatten().astype(np.float32)
                           for uid in np.unique(np.array(pairs).ravel()) if len(pairs) > 0}
        _mean_wf_norms  = {uid: float(np.linalg.norm(v)) for uid, v in _mean_wf_flat.items()}

        _pair_jobs = []
        for c1, c2 in pairs:
            _f1 = _mean_wf_flat[c1]; _f2 = _mean_wf_flat[c2]
            _pair_jobs.append((
                c1, c2,
                _times_sec[c1], _times_sec[c2],
                _f1, _f2,
                _mean_wf_norms[c1], _mean_wf_norms[c2],
                _sim_lookup(c1, c2),
                _contam_lookup.get(c1, float("nan")),
                _contam_lookup.get(c2, float("nan")),
                _xcorr_params,
            ))
        del _mean_wf_flat, _mean_wf_norms

        import concurrent.futures as _cf
        _N_WORKERS_XC = min(64, max(1, len(_pair_jobs)))
        _ts(f"Computing pair XCorr metrics for {len(_pair_jobs)} pairs "
              f"on {_N_WORKERS_XC} workers …")
        rows = []
        with _cf.ProcessPoolExecutor(max_workers=_N_WORKERS_XC) as _pool_xc:
            for _row in tqdm(
                    _pool_xc.map(_pair_metrics_worker, _pair_jobs, chunksize=64),
                    total=len(_pair_jobs), desc="Pair metrics"):
                rows.append(_row)
        del _pair_jobs, _xcorr_params_w

        metrics_df   = pd.DataFrame(rows).sort_values("Similarity", ascending=False).reset_index(drop=True)
        metrics_path = OUT_DIR / "metrics.tsv"
        metrics_df.to_csv(metrics_path, sep="\t", index=False)
        _ts(f"\nMetrics table: {len(metrics_df)} pairs  →  {metrics_path}")
        metrics_df.head(10)


    # %% [markdown]
    # #### **Filter → Merge Groups → Save Outputs**

    # %%
    if not RUN_MERGE:
        _ts("=== RUN_MERGE=False — skipping. ===")
    else:
        _ts("=== RUN_MERGE: building merge groups ===")
        import json

        # ── CONNECTED-COMPONENTS + TRIM MERGE ALGORITHM ────────────────────
        # A merge group satisfies ALL of the following:
        #   1. Every member:  contam_pct < max_contam_pct
        #   2. Every pair:    sim > weak_sim_thresh  AND  xcorr_sig >= -5  (all-pairs floor)
        #   3. Every member:  sim > strong_sim_thresh  AND  xcorr_sig >= -5  with ≥1 other member
        #   4. Every pair:    min(SNR_a,SNR_b)/max(SNR_a,SNR_b) >= snr_ratio_min  (SNR homogeneity)
        #   5. (optional)     peak-channel distance < max_dist
        #
        # Algorithm:
        #   a. Build strong (conditions 1+3+4) and weak (conditions 1+2+4) adjacency matrices.
        #   b. Find connected components of the strong graph (order-independent, topology-complete).
        #   c. For each component: iteratively remove the node with the most weak-constraint
        #      violations until all pairs satisfy the weak floor (or component < 2 nodes).
        #   d. Trimmed nodes remain in the pending pool and are re-processed in subsequent
        #      iterations — allowing them to form smaller groups with other leftovers.
        #   e. Terminate when no new group is formed in an iteration.

        STRONG_THRESH = SELF_BUILD_PARAMS["strong_sim_thresh"]
        WEAK_THRESH   = SELF_BUILD_PARAMS["weak_sim_thresh"]
        SNR_RATIO_MIN = SELF_BUILD_PARAMS["snr_ratio_min"]
        MAX_DIST      = SELF_BUILD_PARAMS["max_dist"]

        # ── Valid units (pass contamination filters) ───────────────────────
        _valid_ids = np.array([
            int(uid) for uid in good_ids
            if (float(contam_pct_df.loc[int(uid), "ContamPct"])
                    if int(uid) in contam_pct_df.index else 1.0) <= SELF_BUILD_PARAMS["max_contam_pct"]
        ], dtype=int)
        _n = len(_valid_ids)
        _ts(f"Valid units after contam filter: {_n} / {len(good_ids)}")

        # ── Sub-matrix + optional distance mask ───────────────────────────
        # cosine_sim_good is indexed by position in _sim_ids_arr (good+mua); map _valid_ids
        _valid_pos  = np.array([_sim_id_to_pos[int(uid)] for uid in _valid_ids], dtype=int)
        _sim_sub    = cosine_sim_good[np.ix_(_valid_pos, _valid_pos)].copy()
        np.fill_diagonal(_sim_sub, 0.0)
        _peak_chans = np.argmax(np.max(mean_wf_slay, 2) - np.min(mean_wf_slay, 2), 1)

        if MAX_DIST < N_wf:  # skip if MAX_DIST >= total units (effectively disabled)
            _dist_mat = np.abs(_peak_chans[_valid_ids, None] - _peak_chans[None, _valid_ids])
            _sim_sub[_dist_mat >= MAX_DIST] = 0.0

        # ── XCorr gate matrix ─────────────────────────────────────────────
        _uid2idx  = {int(uid): idx for idx, uid in enumerate(_valid_ids)}
        _xcorr_ok = np.zeros((_n, _n), dtype=bool)
        for _, _mrow in metrics_df.iterrows():
            _c1, _c2 = int(_mrow["Cluster 1"]), int(_mrow["Cluster 2"])
            if _c1 in _uid2idx and _c2 in _uid2idx:
                _ok       = float(_mrow["XCorr"]) >= -5
                _i, _j    = _uid2idx[_c1], _uid2idx[_c2]
                _xcorr_ok[_i, _j] = _xcorr_ok[_j, _i] = _ok

        # ── SNR homogeneity mask ──────────────────────────────────────────
        # snr is positional (snr[i] = SNR of all_cluster_ids[i]), not indexed by cluster ID.
        _uid_to_snr = {int(all_cluster_ids[i]): snr[i] for i in range(len(all_cluster_ids))}
        _snr_arr = np.array([_uid_to_snr.get(uid, 0.0) for uid in _valid_ids])
        _snr_ratio = (np.minimum(_snr_arr[:, None], _snr_arr[None, :])
                      / (np.maximum(_snr_arr[:, None], _snr_arr[None, :]) + 1e-10))
        _snr_ok    = _snr_ratio >= SNR_RATIO_MIN

        _strong = (_sim_sub > STRONG_THRESH) & _xcorr_ok & _snr_ok
        _weak   = (_sim_sub > WEAK_THRESH)   & _xcorr_ok & _snr_ok
        _ts(f"Strong edges: {int(_strong.sum()) // 2}  |  Weak edges: {int(_weak.sum()) // 2}  |  SNR-blocked pairs: {int((~_snr_ok & (_sim_sub > WEAK_THRESH) & _xcorr_ok).sum()) // 2}")

        # ── Connected-components + trim ───────────────────────────────────
        pending         = set(range(_n))
        merge_groups_sb = []

        while True:
            _pend_arr = np.array(sorted(pending))
            if len(_pend_arr) < 2:
                break
            _adj = _strong[np.ix_(_pend_arr, _pend_arr)]
            _visited    = np.zeros(len(_pend_arr), dtype=bool)
            comps_local = []
            for _start in range(len(_pend_arr)):
                if _visited[_start]:
                    continue
                _comp = []; _q = [_start]; _visited[_start] = True
                while _q:
                    _node = _q.pop(); _comp.append(_node)
                    for _nbr in np.where(_adj[_node])[0]:
                        if not _visited[_nbr]:
                            _visited[_nbr] = True; _q.append(_nbr)
                if len(_comp) >= 2:
                    comps_local.append(_comp)
            if not comps_local:
                break
            _found_any_group = False
            for _comp_local in comps_local:
                _keep = [_pend_arr[i] for i in _comp_local]
                while len(_keep) >= 2:
                    _viol = {_m: sum(1 for _o in _keep if _o != _m and not _weak[_m, _o])
                             for _m in _keep}
                    if max(_viol.values()) == 0:
                        break
                    _keep.remove(max(_viol, key=_viol.get))
                if len(_keep) >= 2:
                    group = sorted(_valid_ids[list(_keep)].tolist())
                    merge_groups_sb.append(group)
                    _found_any_group = True
                    for _idx in _keep:
                        pending.discard(_idx)
            if not _found_any_group:
                break

        _ts(f"\n→ {len(merge_groups_sb)} raw merge groups (before progressive test):")
        for g in merge_groups_sb:
            print(f"  {g}")

        # ── Progressive merge test ─────────────────────────────────────────
        # For each group from connected-components:
        #   1. Order members by descending centrality (sum of pairwise similarities).
        #   2. Seed with the 2 most central members; test merged label.
        #   3. Add remaining members one at a time (next most central first).
        #      If adding a member makes the merged label flip to mua/noise, stop
        #      and keep the last good sub-group.  The dropped member goes into a
        #      leftover pool for a second pass.
        #   4. After all groups are processed, re-run connected-components on
        #      leftover IDs so they can form smaller groups with each other.
        _ts("Progressive merge test …")
        _sample_rate = SELF_BUILD_PARAMS["sample_rate"]

        pending_groups = list(merge_groups_sb)
        final_groups   = []
        leftover_ids   = []

        def _progressive_test(groups_to_test, out_final, out_leftover):
            for group in groups_to_test:
                if len(group) < 2:
                    continue
                centrality = [
                    sum(_sim_lookup(uid, other) for other in group if other != uid)
                    for uid in group
                ]
                ordered = [uid for _, uid in sorted(zip(centrality, group), reverse=True)]
                current = ordered[:2]
                lbl, snr_v, nsp_v, ctn_v = _predict_merge_label(
                    current, spike_times, spike_clusters,
                    mean_wf_slay, noise_levels, sample_rate=_sample_rate)
                print(f"  group {group}: seed {current} → {lbl} "
                      f"(SNR={snr_v:.2f} ContamPct={ctn_v:.3f})")
                if lbl != "good":
                    out_leftover.extend(group)
                    continue
                for uid in ordered[2:]:
                    candidate = current + [uid]
                    lbl2, snr_v2, _, ctn_v2 = _predict_merge_label(
                        candidate, spike_times, spike_clusters,
                        mean_wf_slay, noise_levels, sample_rate=_sample_rate)
                    print(f"    + {uid} → {lbl2} (SNR={snr_v2:.2f} ContamPct={ctn_v2:.3f})")
                    if lbl2 == "good":
                        current = candidate
                    else:
                        out_leftover.append(uid)
                if len(current) >= 2:
                    out_final.append(sorted(current))

        _progressive_test(pending_groups, final_groups, leftover_ids)

        # Second pass: re-run connected-components on leftovers, then progressive test
        if leftover_ids:
            print(f"\n  {len(leftover_ids)} leftover ID(s) for second pass: {leftover_ids}")
            _lo      = np.array(sorted(set(leftover_ids)), dtype=int)
            _vid_lut = {int(v): i for i, v in enumerate(_valid_ids)}
            _lo_idx  = np.array([_vid_lut[uid] for uid in _lo if uid in _vid_lut], dtype=int)
            if len(_lo_idx) >= 2:
                _adj_lo  = _strong[np.ix_(_lo_idx, _lo_idx)]
                _visited = np.zeros(len(_lo_idx), dtype=bool)
                _lo_comps = []
                for _s in range(len(_lo_idx)):
                    if _visited[_s]:
                        continue
                    _comp = []; _q = [_s]; _visited[_s] = True
                    while _q:
                        _nd = _q.pop(); _comp.append(_nd)
                        for _nb in np.where(_adj_lo[_nd])[0]:
                            if not _visited[_nb]:
                                _visited[_nb] = True; _q.append(_nb)
                    if len(_comp) >= 2:
                        _lo_comps.append([int(_lo[i]) for i in _comp])
                _lo_leftover = []
                _progressive_test(_lo_comps, final_groups, _lo_leftover)

        merge_groups_sb = final_groups
        _ts(f"→ {len(merge_groups_sb)} final merge groups (after progressive test):")
        for g in merge_groups_sb:
            print(f"  {g}")

        # ── Build old2new / new2old maps ──────────────────────────────────
        old2new_sb = {}
        new2old_sb = {}
        _new_id    = int(spike_clusters.max()) + 1
        for group in merge_groups_sb:
            new2old_sb[_new_id] = group
            for uid in group:
                old2new_sb[uid] = _new_id
            _new_id += 1

        # ── Spike times by cluster (needed for plot_corr) ─────────────────
        times_multi_sb = {int(uid): spike_times[spike_clusters == uid] for uid in good_ids}

        # ── Annotate metrics_df with BlockedBy reason ─────────────────────
        # For every pair in metrics_df:
        #   "merged"         – pair is proposed for merging
        #   otherwise        – semicolon-separated list of filters that prevented the merge,
        #                      reported in order: ContamPct → dist → xcorr → snr_ratio → sim
        #                      "trimmed" means all pairwise criteria passed but the node was
        #                      removed during the weak-constraint trim step (or had no remaining
        #                      strong-edge partners after earlier trim rounds).
        _merged_pairs = set()
        for _g in merge_groups_sb:
            for _ii in range(len(_g)):
                for _jj in range(_ii + 1, len(_g)):
                    _merged_pairs.add((min(_g[_ii], _g[_jj]), max(_g[_ii], _g[_jj])))

        def _block_reason(c1, c2, xcorr_val):
            if (min(c1, c2), max(c1, c2)) in _merged_pairs:
                return "merged"
            reasons = []
            # Contamination filter (units excluded from _valid_ids before any edge is built)
            _cp1 = float(contam_pct_df.loc[c1, "ContamPct"]) if c1 in contam_pct_df.index else 1.0
            _cp2 = float(contam_pct_df.loc[c2, "ContamPct"]) if c2 in contam_pct_df.index else 1.0
            if _cp1 > SELF_BUILD_PARAMS["max_contam_pct"]:
                reasons.append(f"ContamPct_1={_cp1:.2f}")
            if _cp2 > SELF_BUILD_PARAMS["max_contam_pct"]:
                reasons.append(f"ContamPct_2={_cp2:.2f}")
            # Distance filter (zeroes both strong and weak edges)
            if MAX_DIST < mean_wf_slay.shape[0]:
                _d = abs(int(_peak_chans[c1]) - int(_peak_chans[c2]))
                if _d >= MAX_DIST:
                    reasons.append(f"dist={_d}")
            # XCorr gate (threshold: xcorr >= -5, matching _xcorr_ok construction)
            if xcorr_val < -5:
                reasons.append(f"xcorr={xcorr_val:.3f}")
            # SNR homogeneity filter
            _s1 = _uid_to_snr.get(c1, 0.0);  _s2 = _uid_to_snr.get(c2, 0.0)
            _r  = min(_s1, _s2) / (max(_s1, _s2) + 1e-10)
            if _r < SNR_RATIO_MIN:
                reasons.append(f"snr_ratio={_r:.2f}")
            # Similarity filter — distinguish no-edge-at-all vs weak-only-no-strong
            _sim = _sim_lookup(c1, c2)
            if _sim <= WEAK_THRESH:
                reasons.append(f"sim<weak({_sim:.3f})")
            elif _sim <= STRONG_THRESH:
                reasons.append(f"sim<strong({_sim:.3f})")
            # If every pairwise criterion passed but the pair was never merged, one or both
            # nodes were trimmed out of their component during the weak-constraint trim step.
            if not reasons:
                reasons.append("trimmed")
            return "; ".join(reasons)

        metrics_df["BlockedBy"] = [
            _block_reason(int(_r["Cluster 1"]), int(_r["Cluster 2"]), float(_r["XCorr"]))
            for _r in metrics_df.to_dict("records")
        ]

        # ── Save outputs ──────────────────────────────────────────────────
        with open(OUT_DIR / "old2new.json", "w") as f:
            json.dump({str(k): int(v) for k, v in old2new_sb.items()}, f, separators=(",\n", ":"))
        with open(OUT_DIR / "new2old.json", "w") as f:
            json.dump({str(k): [int(x) for x in v] for k, v in new2old_sb.items()}, f, separators=(",\n", ":"))

        metrics_df.to_csv(OUT_DIR / "metrics.tsv", sep="\t", index=False)

        saveable_params = {k: v for k, v in SELF_BUILD_PARAMS.items()
                           if isinstance(v, (str, int, float, bool, list, type(None)))}
        with open(OUT_DIR / "run_params.json", "w") as f:
            json.dump(saveable_params, f, indent=2)


        _ts(f"Outputs saved to {OUT_DIR}/")
        _ts(f"  new2old.json  ({len(new2old_sb)} entries)")
        _ts(f"  old2new.json  ({len(old2new_sb)} entries)")

    # %% [markdown]
    # #### **Plot Proposed Merges (slay-style)**
    # 

    # %%
    if not RUN_PLOT_MERGES:
        _ts("=== RUN_PLOT_MERGES=False — skipping. ===")
    else:

        import json
        import math as _math
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
        from slay.xcorr import auto_correlogram as _auto_correlogram
        import slay.plot as slay_plot
        import gc

        plt.ioff()

        REJECTED_DIR = PLOTS_DIR / "rejected"
        REJECTED_DIR.mkdir(exist_ok=True, parents=True)

        _noise_levels_sb    = noise_levels
        pre                 = SELF_BUILD_PARAMS["pre_samples"]
        post                = SELF_BUILD_PARAMS["post_samples"]
        MAX_SPIKES_FOR_PLOT = SELF_BUILD_PARAMS["max_spikes_for_plot"]
        rng                 = np.random.default_rng(42)
        n_ch                = rec.get_num_channels()

        # ── Fallbacks for variables that depend on earlier cells ──────────
        if 'mean_wf_slay' not in globals():
            _mwf_p = next((p for p in [OUT_DIR / "mean_waveforms.npy",
                                        PHY_DIR / "mean_waveforms.npy"] if p.exists()), None)
            assert _mwf_p is not None, "mean_waveforms.npy not found; run RUN_BUILD_METRICS first."
            mean_wf_slay = np.load(_mwf_p)
            print(f"Loaded mean_wf_slay from {_mwf_p}  shape={mean_wf_slay.shape}")

        if 'metrics_df' not in globals():
            _mpath = OUT_DIR / "metrics.tsv"
            assert _mpath.exists(), f"metrics.tsv not found; run RUN_BUILD_METRICS first."
            metrics_df = pd.read_csv(_mpath, sep="\t")
            print(f"Loaded metrics_df ({len(metrics_df)} pairs) from {_mpath}")

        if 'new2old_sb' not in globals():
            with open(OUT_DIR / "new2old.json") as _f:
                new2old_sb = {int(k): [int(x) for x in v] for k, v in json.load(_f).items()}
            print(f"Loaded new2old_sb ({len(new2old_sb)} groups) from new2old.json")

        if 'old2new_sb' not in globals():
            with open(OUT_DIR / "old2new.json") as _f:
                old2new_sb = {int(k): int(v) for k, v in json.load(_f).items()}
            print(f"Loaded old2new_sb ({len(old2new_sb)} entries) from old2new.json")

        if 'times_multi_sb' not in globals():
            _all_needed = set(uid for g in new2old_sb.values() for uid in g)
            times_multi_sb = {int(uid): spike_times[spike_clusters == uid]
                              for uid in _all_needed}
            print(f"Rebuilt times_multi_sb for {len(times_multi_sb)} units")

        # ── Ensure _sparsity_uid_to_row is available (may not be if RUN_BUILD_METRICS=False) ──
        _sparsity_uid_to_row = {int(cid): i for i, cid in enumerate(all_cluster_ids)}

        # ── Build per-unit sparse channel mask ────────────────────────────
        _ch_mask_plot = {}
        for _uid in np.unique(spike_clusters):
            _uid = int(_uid)
            if _uid in _sparsity_uid_to_row:
                _ch_mask_plot[_uid] = np.where(sparsity_mask[_sparsity_uid_to_row[_uid]])[0]

        # _predict_merge_label is defined at module level (after _compute_unit_metrics)

        # ── Index metrics_df by pair for fast lookup ──────────────────────
        metrics_lookup = {
            (int(r["Cluster 1"]), int(r["Cluster 2"])): r
            for _, r in metrics_df.iterrows()
        }

        def _pair_metrics_title(merge, metrics_lookup):
            """Per-pair similarity/quality metrics — shown on the waveform and correlogram plots."""
            lines = []
            for ii in range(len(merge)):
                for jj in range(ii + 1, len(merge)):
                    c1, c2 = merge[ii], merge[jj]
                    row = metrics_lookup.get((c1, c2))
                    if row is None:
                        row = metrics_lookup.get((c2, c1))
                    if row is None:
                        lines.append(f"  [{c1} × {c2}]  (no metrics found)")
                        continue
                    lines.append(
                        f"  [{c1} × {c2}]  "
                        f"Sim={row['Similarity']:.3f}  "
                        f"SlaySim={row['Slay Similarity']:.3f}  "
                        f"XCorr={row['XCorr']:.2f}  |  "
                        f"ContamPct: {row['ContamPct_1']:.3f} / {row['ContamPct_2']:.3f}"
                    )
            return "\n".join(lines)

        def _merged_metrics_title(merge, pred_label, pred_snr, pred_nsp, pred_contam):
            """Merged cluster quality — shown only on the merged-cluster plot (page 3)."""
            verdict = "★ GOOD" if pred_label == "good" else f"✗ {pred_label.upper()}"
            return (
                f"Merged cluster  ({' + '.join(map(str, merge))})\n"
                f"  {verdict}  SNR={pred_snr:.2f}  n_spikes={pred_nsp}  ContamPct={pred_contam:.3f}"
            )

        def _add_suptitle(fig, title, fontsize=7):
            fig.suptitle(title, fontsize=fontsize, family="monospace")
            n_lines = title.count("\n") + 1
            if not fig.get_constrained_layout():
                top = max(0.68, 1.0 - n_lines * 0.045)
                try:
                    fig.tight_layout(rect=[0, 0, 1, top])
                except Exception:
                    fig.subplots_adjust(top=top)

        # ── Main plot loop ─────────────────────────────────────────────────
        good_merge_new_ids = set()

        for new_id, old_ids in tqdm(list(new2old_sb.items()), desc="Plotting merges"):
            merge     = sorted([int(x) for x in old_ids])
            merge_str = "_".join(map(str, merge))

            pred_label, pred_snr, pred_nsp, pred_contam = _predict_merge_label(
                merge, spike_times, spike_clusters,
                mean_wf_slay, _noise_levels_sb,
                sample_rate=SELF_BUILD_PARAMS["sample_rate"],
            )
            is_good  = (pred_label == "good")
            plot_dir = PLOTS_DIR if is_good else REJECTED_DIR
            wf_path     = plot_dir / f"merge_{merge_str}.png"
            corr_path   = plot_dir / f"merge_{merge_str}_corr.png"
            merged_path = plot_dir / f"merge_{merge_str}_merged.png"

            if is_good:
                good_merge_new_ids.add(new_id)

            if wf_path.exists() and corr_path.exists() and merged_path.exists():
                continue

            # ── Extract individual-cluster waveforms (subsampled) ─────────
            st_for_plot = {}
            for uid in merge:
                st = spike_times[spike_clusters == uid]
                if MAX_SPIKES_FOR_PLOT > 0 and len(st) > MAX_SPIKES_FOR_PLOT:
                    st = rng.choice(st, MAX_SPIKES_FOR_PLOT, replace=False)
                st_for_plot[uid] = np.sort(st)

            wfs_dict = _extract_waveforms_batched(
                rec, st_for_plot, pre, post, channel_mask=_ch_mask_plot,
                return_mean_only=False)

            # Reshape to slay format: (n_spikes, n_ch, wf_len)
            spikes_slay = {}
            for uid in merge:
                wfs_u = wfs_dict[uid]          # (n_sp, wf_len, n_active)
                full  = np.zeros((wfs_u.shape[0], n_ch, wfs_u.shape[1]), dtype=np.float32)
                if uid in _ch_mask_plot:
                    full[:, _ch_mask_plot[uid], :] = wfs_u.transpose(0, 2, 1)
                else:
                    full = wfs_u.transpose(0, 2, 1)
                spikes_slay[uid] = full

            pair_title   = _pair_metrics_title(merge, metrics_lookup)
            merged_title = _merged_metrics_title(merge, pred_label, pred_snr, pred_nsp, pred_contam)

            # ── Plot 1: Individual cluster waveforms ──────────────────────
            if not wf_path.exists():
                wf_fig = slay_plot.plot_wfs(merge, mean_wf_slay, spikes_slay,
                                             nchan=10, start=0, stop=pre + post)
                _add_suptitle(wf_fig, pair_title)
                wf_fig.savefig(wf_path, dpi=150, bbox_inches="tight")
                plt.close(wf_fig)

            # ── Plot 2: Individual cluster correlograms ────────────────────
            if not corr_path.exists():
                corr_fig = slay_plot.plot_corr(merge, times_multi_sb, SELF_BUILD_PARAMS)
                _add_suptitle(corr_fig, pair_title)
                corr_fig.savefig(corr_path, dpi=150, bbox_inches="tight")
                plt.close(corr_fig)

            # ── Plot 3: Merged cluster — waveforms + auto-correlogram ──────
            if not merged_path.exists():
                _all_st_merged = np.sort(np.concatenate(
                    [spike_times[spike_clusters == uid] for uid in merge]))

                _st_wf = _all_st_merged
                if MAX_SPIKES_FOR_PLOT > 0 and len(_st_wf) > MAX_SPIKES_FOR_PLOT:
                    _st_wf = rng.choice(_st_wf, MAX_SPIKES_FOR_PLOT, replace=False)
                _st_wf = np.sort(_st_wf)

                _merged_wfs = _extract_waveforms_batched(
                    rec, {0: _st_wf}, pre, post, channel_mask=None,
                    return_mean_only=False)[0]
                # shape: (n_sp, wf_len, n_ch)

                _counts_m = {uid: int((spike_clusters == uid).sum()) for uid in merge}
                _total_m  = max(sum(_counts_m.values()), 1)
                _mwf_m    = sum(_counts_m[uid] * mean_wf_slay[uid].astype(float)
                                for uid in merge) / _total_m

                _ptp_ch  = np.ptp(_mwf_m, axis=1)
                _peak_ch = int(np.argmax(_ptp_ch))
                _nchan_m = 10
                _ch0     = max(_peak_ch - _nchan_m // 2, 0)
                _ch1     = min(_ch0 + _nchan_m, n_ch)
                _ch0     = max(_ch1 - _nchan_m, 0)
                _disp_ch = list(range(_ch0, _ch1))
                _n_d     = len(_disp_ch)

                _sp_pk  = _merged_wfs[:, :, _peak_ch] if len(_merged_wfs) > 0 \
                          else np.zeros((1, pre + post))
                _sp_ptp = max(float(np.ptp(_sp_pk)), 1e-6)
                _ymin_m = float(_sp_pk.min()) - 0.1 * _sp_ptp
                _ymax_m = float(_sp_pk.max()) + 0.1 * _sp_ptp

                _acg = _auto_correlogram(
                    _all_st_merged.astype(np.float64) / SELF_BUILD_PARAMS["sample_rate"],
                    SELF_BUILD_PARAMS["window_size"],
                    SELF_BUILD_PARAMS["ref_pen_bin_width"] / 1000.0,
                    SELF_BUILD_PARAMS["overlap_tol"],
                )

                _n_wf_rows = _math.ceil(_n_d / 2)
                _fig_ht    = max(_n_wf_rows * 1.5, 5.0) + 1.5
                _fig_m     = plt.figure(figsize=(14, _fig_ht), constrained_layout=False)
                _fig_m.patch.set_facecolor('#e0e0e0')
                _gs_m    = GridSpec(1, 2, figure=_fig_m, wspace=0.3,
                                    left=0.07, right=0.97, top=0.87, bottom=0.07)
                _gs_wf_m = GridSpecFromSubplotSpec(_n_wf_rows, 2, subplot_spec=_gs_m[0],
                                                   wspace=0.08, hspace=0.08)
                _ax_acg  = _fig_m.add_subplot(_gs_m[1])
                _color_m = (0.35, 0.60, 0.99)

                for _ci, _ch in enumerate(_disp_ch):
                    _ax_w = _fig_m.add_subplot(_gs_wf_m[_ci // 2, _ci % 2])
                    _ax_w.set_facecolor("black")
                    _ax_w.set_ylim(_ymin_m, _ymax_m)
                    _ax_w.set_xticks([]); _ax_w.set_yticks([])
                    _ax_w.spines["top"].set_visible(False)
                    _ax_w.spines["right"].set_visible(False)
                    _ax_w.set_ylabel(str(_ch), size="small", color="black", weight="bold")
                    _xs = np.arange(pre + post)
                    for _j in range(len(_merged_wfs)):
                        _ax_w.plot(_xs, _merged_wfs[_j, :, _ch],
                                   color=_color_m, linewidth=0.2, alpha=0.35)
                    _ax_w.plot(_xs, _mwf_m[_ch], color='white', linewidth=1.3, alpha=0.95)

                _ax_acg.set_facecolor("black")
                _ax_acg.bar(range(len(_acg)), _acg, width=1, color=_color_m)
                _ax_acg.spines["top"].set_visible(False)
                _ax_acg.spines["right"].set_visible(False)
                _ax_acg.set_title("Auto-correlogram (merged)", color="black", size="small")
                _ax_acg.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                _ax_acg.set_yticks([0, max(int(_acg.max()), 1)])

                _fig_m.suptitle(merged_title, fontsize=8, family="monospace")
                _fig_m.savefig(merged_path, dpi=150, bbox_inches="tight")
                plt.close(_fig_m)

            gc.collect()

        # ── Rewrite JSONs: only keep good merges ──────────────────────────
        good_new2old_out = {k: v for k, v in new2old_sb.items() if k in good_merge_new_ids}
        good_old2new_out = {k: v for k, v in old2new_sb.items() if v in good_merge_new_ids}

        with open(OUT_DIR / "old2new.json", "w") as f:
            json.dump({str(k): int(v) for k, v in good_old2new_out.items()}, f, separators=(",\n", ":"))
        with open(OUT_DIR / "new2old.json", "w") as f:
            json.dump({str(k): [int(x) for x in v] for k, v in good_new2old_out.items()}, f, separators=(",\n", ":"))

        n_total_m = len(new2old_sb)
        n_good_m  = len(good_new2old_out)
        _ts(f"Merge quality check: {n_good_m} good / {n_total_m} total")
        _ts(f"  Good plots   → {PLOTS_DIR}/")
        _ts(f"  Rejected     → {REJECTED_DIR}/")
        _ts(f"  JSONs rewritten ({n_good_m} good merges kept)")

        plt.ion()

    # %% [markdown]
    # #### **Accept merges**

    # %%
    with open(OUT_DIR / "new2old.json") as f:
        new2old = json.load(f)

    num_new_clusters = len(new2old)
    num_old_clusters = sum(len(v) for v in new2old.values())
    _ts(f"Merge groups to accept : {num_new_clusters}")
    _ts(f"Old clusters merged in : {num_old_clusters}")

    # %%
    if not RUN_ACCEPT_MERGES:
        _ts("=== RUN_ACCEPT_MERGES=False — skipping. ===")
    else:
        from slay.stages import accept_all_merges

        _ks = str(PHY_DIR)

        # ── Load spike arrays from current state (consistent with amplitudes.npy) ──
        # Always load from the plain .npy files (not _orig). Slay's accept_all_merges
        # also loads amplitudes.npy / spike_templates.npy from disk, so everything
        # must come from the same consistent snapshot. Slay will then save the
        # current files as *_orig before overwriting, so _orig advances each round.
        spike_times_am    = np.load(PHY_DIR / "spike_times.npy").ravel()
        spike_clusters_am = np.load(PHY_DIR / "spike_clusters.npy").ravel()
        _ts(f"  spike_times:    spike_times.npy    (n={len(spike_times_am)})")
        _ts(f"  spike_clusters: spike_clusters.npy (n={len(spike_clusters_am)})")

        # ── mean_wf: prefer OUT_DIR copy (saved pre-merge, always safe) ───
        _mwf_options = [OUT_DIR / "mean_waveforms.npy", PHY_DIR / "mean_waveforms.npy"]
        _mwf_path    = next((p for p in _mwf_options if p.exists()), None)
        assert _mwf_path is not None, "mean_waveforms.npy not found"
        mean_wf_am = np.load(_mwf_path)
        _ts(f"  mean_wf: {_mwf_path.name}  shape={mean_wf_am.shape}")

        params_am = dict(SELF_BUILD_PARAMS)
        params_am["KS_folder"]      = _ks
        params_am["results_folder"] = RUN_TAG
        params_am["n_chan"]          = mean_wf_am.shape[1]

        _unique_am, _counts_am = np.unique(spike_clusters_am, return_counts=True)
        n_spikes_am = np.zeros(spike_clusters_am.max() + 1, dtype=int)
        n_spikes_am[_unique_am] = _counts_am

        times_multi_am = {int(c): spike_times_am[spike_clusters_am == c] for c in _unique_am}
        cl_labels_am   = pd.read_csv(PHY_DIR / "cluster_group.tsv", sep="\t", index_col=0)
        if "KSLabel" in cl_labels_am.columns:
            cl_labels_am = cl_labels_am.rename(columns={"KSLabel": "label"})

        # ── Sanity-check: spike_times must match amplitudes.npy ──────────
        _n_am = np.load(PHY_DIR / "amplitudes.npy").shape[0]
        assert len(spike_times_am) == _n_am, (
            f"Spike count mismatch: spike_times={len(spike_times_am)}  amplitudes={_n_am}\n"
            f"  amplitudes.npy is out of sync with spike_times.npy — check PHY_DIR state."
        )
        _ts(f"  Spike count OK: {len(spike_times_am)} spikes")

        # ── Run slay's merge acceptance ───────────────────────────────────
        vals = (None, cl_labels_am, mean_wf_am, n_spikes_am,
                spike_times_am, spike_clusters_am, times_multi_am)
        accept_all_merges(vals, params_am)

        _ts(f"Merges accepted and files saved.")



