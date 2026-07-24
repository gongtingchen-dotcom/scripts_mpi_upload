"""
Auto-merging pipeline for self-built KiloSort sortings (phy-compatible output).

Run this with the `slay` micromamba environment (has the old-API SLAy fork +
spikeinterface + kilosort installed as an editable package from
C:\\Users\\gongt\\SLAy). The rewritten PyPI `slay` package does NOT have the
functions this script needs (x_correlogram, calc_xcorr_sig, slay.plot,
slay.stages.accept_all_merges).

STAGES (each gated by a RUN_* flag below):
  1. build_analyzer   - sparse SortingAnalyzer (sparsity mask only, from peaks file)
  2. label_clusters   - SNR / ContamPct / ISI -> cluster_group.tsv (good/mua/noise)
  3. build_metrics    - mean waveforms, cosine similarity, phy template files,
                         per-pair XCorr metrics table
  4. find_merges      - connected-components + trim + progressive-test merge groups
                         -> old2new.json / new2old.json
  5. plot_merges      - waveform/correlogram QC figures, filters JSONs to
                         label-verified "good" merges only
  6. accept_merges    - apply merges to spike arrays via slay (destructive)

Each stage, when its RUN_* flag is False, tries to load its outputs from disk
instead of recomputing. If a flag is False but the cached outputs are missing,
the stage computes them anyway (a False flag never hard-fails a fresh run --
it just means "don't force a recompute"). Set the flag True to force a
recompute even when outputs already exist.

Output/behavior is intended to match the original try_merge_selfdbuild.py,
with two bug fixes carried over plus the phy-compatibility file writes
decoupled from the expensive metrics computation so that toggling
RUN_BUILD_METRICS off doesn't require redoing work just to get phy to open
the result.

Bug fixes carried over from try_merge_selfdbuild.py:
  - Centrality (progressive merge test) and the BlockedBy diagnostic now look
    up similarity through the good+mua position map (_sim_lookup) instead of
    indexing the compact similarity matrix directly by raw cluster id -- that
    raised IndexError once a group contained a merged-cluster id beyond the
    compact matrix's size.
  - similar_templates.npy is written at full (n_templates, n_templates) size
    (scattered from the compact good+mua matrix), matching what phy's loader
    asserts. Previously it was saved at the compact size, which made phy's
    `AssertionError: self.similar_templates.shape == (...)` fail.
"""

# %% [markdown]
# #### Imports

# %%
import gc
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from kilosort.CCG import check_CCG
import spikeinterface.full as si
import slay
import slay.plot as slay_plot
from slay.xcorr import auto_correlogram as _auto_correlogram
from slay.stages import accept_all_merges


# %% [markdown]
# #### Paths — edit these before running

# %%
EXP            = r"E:\GLC7189_c_g0"
SORTING_DIR    = r"E:\GLC7189_c_g0\tomato_sorting_selfbuild_stricter_run4"
PHY_DIR        = r"E:\GLC7189_c_g0\tomato_sorting_selfbuild_stricter_run4\export_to_phy"
RECORDING_PATH = r"E:\GLC7189_c_g0\preprocessed"
# Peaks file from thresholding — used to derive peak channel per unit without
# reading the recording at all. Set to None to fall back to estimate_sparsity.
PEAKS_FILE     = r"E:\GLC7189_c_g0\preprocessed\thresholding_peaks/peaks_combined2.npy"

# Sparse analyzer: built once (or rebuilt when RUN_BUILD_ANALYZER=True).
# Contains mean templates (~32 channels, 90 samples) used for similarity.
ANALYZER_DIR   = PHY_DIR + "/_sparse_analyzer5"

# Output sub-folder for this run's results/plots.
# Change the suffix each time you run on freshly-merged data.
RUN_TAG        = "run6"

# =====================================================================
#  RUN FLAGS
# =====================================================================
# True  = force this stage to (re)compute from scratch, overwriting cached outputs.
# False = reuse cached outputs from disk if present; if they're missing, the
#         stage computes them anyway (so a False flag never hard-fails a
#         fresh run — it's just "don't force a recompute").
RUN_BUILD_ANALYZER   = True    # (re)build sparse SortingAnalyzer
RUN_LABEL_CLUSTERS   = True    # compute SNR / ContamPct / ISI -> cluster_group.tsv
RUN_BUILD_METRICS    = True    # mean waveforms + similarity + phy template files + pair metrics
RUN_MERGE            = False   # find merge groups -> old2new.json / new2old.json
RUN_PLOT_MERGES      = False   # render waveform / correlogram QC figures
RUN_ACCEPT_MERGES    = False   # apply accepted merges to spike arrays (DESTRUCTIVE)

# =====================================================================
#  SPARSE ANALYZER SETTINGS  (only used when the analyzer is (re)built)
# =====================================================================
TARGET_N_CHANNELS    = 32    # approx channels per unit neighbourhood
SAMPLES_BEFORE       = 30    # -> pre_samples  = 30 @ 30 kHz
SAMPLES_AFTER        = 35    # -> post_samples = 35 @ 30 kHz
MAX_SPIKES_PER_UNIT  = 1000  # for labelling waveform means

PHY_DIR      = Path(PHY_DIR).resolve()
ANALYZER_DIR = Path(ANALYZER_DIR).resolve()
SORTING_DIR  = Path(SORTING_DIR).resolve()
OUT_DIR      = PHY_DIR / RUN_TAG
PLOTS_DIR    = OUT_DIR / "merges"
REJECTED_DIR = PLOTS_DIR / "rejected"
# OUT_DIR/PLOTS_DIR are created in main() (after spike arrays are confirmed to
# exist), so PHY_DIR isn't pre-created here — that would skip the
# build-from-sorting bootstrap step.

print(f"PHY_DIR     : {PHY_DIR}")
print(f"ANALYZER_DIR: {ANALYZER_DIR}")
print(f"OUT_DIR     : {OUT_DIR}")

# %% [markdown]
# #### Self-build parameters

# %%
SELF_BUILD_PARAMS = {
    # ── Clique similarity thresholds ─────────────────────────────────
    # Each merge-group member must have sim > strong_sim_thresh with >=1 other
    # member (this is what connects a component), AND sim > weak_sim_thresh
    # with every other member (the floor enforced by the trim step).
    "strong_sim_thresh": 0.9,    # seed / expand threshold
    "weak_sim_thresh":   0.85,   # all-pairs floor

    # ── Per-unit quality gate ─────────────────────────────────────────
    "max_contam_pct":    0.2,

    # ── SNR homogeneity gate ──────────────────────────────────────────
    # Two units can only merge if min(SNR_a, SNR_b) / max(SNR_a, SNR_b) >= this.
    # Prevents fusing a strong single-unit with a much weaker one.
    "snr_ratio_min":     0.5,    # e.g. 0.5 = weaker unit must have >=50% of the stronger unit's SNR

    # ── Optional spatial gate ─────────────────────────────────────────
    "max_dist":          400,    # max peak-channel distance; large = disabled

    # ── XCorr gate (pairwise CCG co-firing check) ─────────────────────
    "window_size":       0.1,    # s — CCG window (+-50 ms each side)
    "xcorr_bin_width":   0.001,  # s — CCG bin size (1 ms)
    "min_xcorr_rate":    10.0,   # Hz — min CCG rate for significance
    "xcorr_gate_min":    -5.0,   # min slay.calc_xcorr_sig value to pass the XCorr gate

    # ── Kept for plot_corr (slay) ─────────────────────────────────────
    "ref_pen_bin_width": 0.1,    # ms — bin size for ACG/CCG plots
    "overlap_tol":       0.0003, # s  — overlap tolerance

    # ── Waveform extraction ───────────────────────────────────────────
    # post_samples=35 (~1.17 ms) keeps only the spike + fast repolarisation
    # phase where waveform shape actually differs between units.
    # The SI analyzer still uses ms_after=2.0 for sparsity, but similarity
    # is computed on this tighter window.
    "pre_samples":         30,
    "post_samples":        35,
    "max_spikes":          1000,  # used when building mean waveforms (metrics stage)
    "max_spikes_for_plot": 500,   # max spikes displayed per cluster in plots (0 = no limit)
    "sample_rate":         30000.0,

    # ── Labelling thresholds ──────────────────────────────────────────
    # SNR/ContamPct/n_spikes cutoffs live in _assign_label(); contam_tbin_ms
    # and isi_ref_ms configure the CCG/ISI computations that feed it.
    "snr_good_min":      3.7,    # SNR must exceed this to be eligible for "good"
    "noise_min_spikes":  20,     # below snr_good_min: "noise" if fewer spikes than this, else "mua"
    "contam_tbin_ms":    1.5,    # R12 CCG bin width; effective window = +-1..10 x tbin
    "isi_ref_ms":        1.5,    # ISI violation refractory threshold (ms) — diagnostic only,
                                 # does not feed _assign_label (see docstring there)

    # ── Misc ──────────────────────────────────────────────────────────
    "backup_suffix":     "_orig",
    "results_folder":    RUN_TAG,
    "KS_folder":         str(PHY_DIR),
}

print("Parameters set. OUT_DIR:", OUT_DIR)


# %% [markdown]
# #### Shared metric / labelling helpers
#
# These are used by both the labelling stage (label_clusters) and the merge
# stages (_predict_merge_label, used by find_merges and plot_merges) so that
# "what makes a unit good/mua/noise" is defined in exactly one place.

# %%
def _assign_label(snr, contam, n_sp, params=SELF_BUILD_PARAMS):
    """Translate (SNR, ContamPct, spike count) into a good/mua/noise label.

    Single source of truth for label thresholds — called by both
    _compute_unit_metrics() and label_clusters() so that any change only
    needs to be made here (and in SELF_BUILD_PARAMS, which supplies the
    actual threshold values).
    """
    if snr > params["snr_good_min"]:
        return "good" if contam < params["max_contam_pct"] else "mua"
    return "noise" if n_sp < params["noise_min_spikes"] else "mua"


def _compute_unit_metrics(st_samples, mean_wf, noise_levels, fs,
                           contam_tbin_ms=SELF_BUILD_PARAMS["contam_tbin_ms"]):
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
                          sample_rate=SELF_BUILD_PARAMS["sample_rate"]):
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


def make_sim_lookup(cosine_sim, sim_ids_arr):
    """Build a (uid1, uid2) -> similarity lookup over a compact similarity matrix.

    `cosine_sim` is indexed by POSITION in `sim_ids_arr`, not by raw cluster
    id (the matrix only covers good+mua units, so raw cluster ids can exceed
    its bounds). Every place that needs a similarity value between two
    cluster ids must go through this — indexing cosine_sim directly by
    cluster id is the bug that caused `IndexError: index ... out of bounds`
    once a group contained a merged-cluster id beyond the compact matrix.
    """
    id_to_pos = {int(uid): i for i, uid in enumerate(sim_ids_arr)}

    def _lookup(uid1, uid2):
        p1 = id_to_pos.get(int(uid1), -1)
        p2 = id_to_pos.get(int(uid2), -1)
        if p1 < 0 or p2 < 0:
            return 0.0
        return float(cosine_sim[p1, p2])

    return _lookup, id_to_pos


def channel_mask_from_waveforms(mean_wf_slay, unit_ids):
    """Derive each unit's active-channel index array directly from
    mean_wf_slay, instead of needing a SortingAnalyzer/sparsity mask.

    mean_wf_slay is zero-padded outside each unit's own sparse channel
    neighborhood by construction (see compute_mean_waveforms), so "channels
    where this unit's mean waveform is ever nonzero" IS its sparsity mask.
    Used by try_merge_step_2.py, which works entirely from an existing
    mean_waveforms.npy and doesn't load the analyzer at all.
    """
    return {
        int(uid): np.where(np.any(mean_wf_slay[int(uid)] != 0, axis=1))[0]
        for uid in unit_ids
    }


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
            for t, uid, idx in spikes_in_chunk:
                s       = t - pre_samples - load_start
                snippet = chunk_data[s : s + wf_len]        # (wf_len, n_ch_all)
                if channel_mask is not None and uid in channel_mask:
                    snippet = snippet[:, channel_mask[uid]]  # (wf_len, n_active)
                if return_mean_only:
                    wf_sum[uid] += snippet
                    wf_cnt[uid] += 1
                else:
                    wfs[uid][idx] = snippet
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


def _connected_components(adjacency, node_ids, min_size=2):
    """Connected components of a boolean adjacency matrix, via BFS.

    `node_ids` labels each row/column of `adjacency` (any hashable/orderable
    type — used only to build the returned groups). Components smaller than
    `min_size` are dropped. Used by find_merges for both the main pass and
    the leftover second pass, so the BFS logic exists in exactly one place.
    """
    node_ids = list(node_ids)
    n = len(node_ids)
    visited = np.zeros(n, dtype=bool)
    components = []
    for start in range(n):
        if visited[start]:
            continue
        comp = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop()
            comp.append(node)
            for nbr in np.where(adjacency[node])[0]:
                if not visited[nbr]:
                    visited[nbr] = True
                    queue.append(nbr)
        if len(comp) >= min_size:
            components.append([node_ids[i] for i in comp])
    return components


# %% [markdown]
# #### Stage 2: label_clusters — SNR / ContamPct / ISI -> cluster_group.tsv

# %%
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
    overwrite : bool
        False (default): if cluster_snr.tsv and cluster_ContamPct.tsv already
        exist AND cover every id in all_cluster_ids, load them from disk
        instead of recomputing (the "skip if already there" path — call with
        overwrite=True to force a recompute regardless of what's cached).
        If the cache is missing ids (e.g. a newly-accepted merged cluster
        that didn't exist yet when the cache was written), it is always
        recomputed even when overwrite=False — a False flag skips redundant
        work, it never serves stale/incomplete data.
    contam_tbin_ms : float
        CCG bin width in milliseconds for the R12 (ContamPct) metric.
        The refractory windows tested are ±1..10 × tbin.  Default 1.5 ms
        gives a maximum window of ±15 ms.  The lag-0 bin (double-spikes)
        is excluded by design (KiloSort convention).
    isi_ref_ms : float
        Refractory period threshold (ms) for the ISI violation rate saved
        as cluster_isi_violation.tsv.  Fraction of consecutive spike-pair
        ISIs shorter than this threshold.  Diagnostic only — does not feed
        _assign_label.  Does NOT explicitly exclude lag-0 spikes, but
        duplicate spikes are typically absent after KiloSort detection.
    """
    snr_path    = PHY_DIR / "cluster_snr.tsv"
    contam_path = PHY_DIR / "cluster_ContamPct.tsv"
    isi_path    = PHY_DIR / "cluster_isi_violation.tsv"
    group_path  = PHY_DIR / "cluster_group.tsv"

    metrics_exist = snr_path.exists() and contam_path.exists()
    cache_covers_all = False
    snr_df = contam_df = None
    if metrics_exist:
        snr_df     = pd.read_csv(snr_path, sep="\t")
        contam_df  = pd.read_csv(contam_path, sep="\t")
        # Only count rows with an actual value — sync_metrics_tsvs_to_current_clusters()
        # (called earlier in main(), to keep si.read_phy's inner-join happy) inserts
        # NaN-valued placeholder rows for any new cluster id, so "cluster_id is a row"
        # is NOT the same as "this id has a real cached SNR/ContamPct".
        cached_ids = (set(int(x) for x in snr_df.loc[snr_df["SNR"].notna(), "cluster_id"])
                      & set(int(x) for x in contam_df.loc[contam_df["ContamPct"].notna(), "cluster_id"]))
        cache_covers_all = set(int(x) for x in all_cluster_ids) <= cached_ids

    if not overwrite and metrics_exist and not cache_covers_all:
        _missing_n = len(set(int(x) for x in all_cluster_ids) - cached_ids)
        print(f"  Cached SNR/ContamPct are missing {_missing_n} current cluster id(s) "
              f"(likely a newly-accepted merge) — recomputing instead of using a stale cache.")

    if not overwrite and cache_covers_all:
        print("  Metrics files exist and cover all current clusters — loading from disk.")
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
        sparsity_uid_to_row = {int(cid): i for i, cid in enumerate(all_cluster_ids)}

        rng       = np.random.default_rng(seed)
        peak_ptp  = np.zeros(len(all_cluster_ids))
        peak_chan = np.zeros(len(all_cluster_ids), dtype=int)
        n_spikes  = np.zeros(len(all_cluster_ids), dtype=int)

        # Pre-group spikes by cluster with a single argsort — O(n_spikes log n_spikes)
        # instead of O(n_clusters × n_spikes) np.where scans (was the main bottleneck
        # with 68 k clusters × 129 M spikes).
        print("  Pre-grouping spikes by cluster …")
        grp_order = np.argsort(spike_clusters, kind='stable')
        sc_grp    = spike_clusters[grp_order]
        st_grp    = spike_times[grp_order]
        uids_g, ucnts_g = np.unique(sc_grp, return_counts=True)
        ubounds_g = np.concatenate([[0], np.cumsum(ucnts_g)])
        uid_to_g  = {int(u): i for i, u in enumerate(uids_g)}
        print("  Grouping done.")

        # Build per-unit spike-time dict (same subsampling as before, same rng/seed)
        # and channel mask, then extract all waveforms in a single sorted pass.
        st_per_unit = {}
        ch_mask_lbl = {}
        for idx, clu in enumerate(all_cluster_ids):
            gi = uid_to_g.get(int(clu), -1)
            if gi < 0:
                continue
            s, e = int(ubounds_g[gi]), int(ubounds_g[gi + 1])
            n_spikes[idx] = e - s
            if n_spikes[idx] == 0:
                continue
            if n_spikes[idx] > max_spikes_per_unit:
                sel = rng.choice(e - s, max_spikes_per_unit, replace=False)
                st_per_unit[int(clu)] = np.sort(st_grp[s:e][sel])
            else:
                st_per_unit[int(clu)] = np.sort(st_grp[s:e])
            if clu in sparsity_uid_to_row:
                ch_mask_lbl[int(clu)] = np.where(sparsity_mask[sparsity_uid_to_row[clu]])[0]
            else:
                ch_mask_lbl[int(clu)] = np.arange(rec.get_num_channels())

        print(f"  Extracting waveforms in a single sorted pass over {len(st_per_unit)} units …")
        # return_mean_only=True: accumulates running sum+count per unit —
        # returns {uid: mean_wf (wf_len, n_active)} using ~600 MB instead of ~550 GB.
        mean_wfs = _extract_waveforms_batched(
            rec, st_per_unit, pre_samples, post_samples, channel_mask=ch_mask_lbl,
            return_mean_only=True)

        # Compute peak PTP from mean waveform per unit
        clu_to_idx = {int(clu): idx for idx, clu in enumerate(all_cluster_ids)}
        for uid, mean_wf_sp in mean_wfs.items():  # mean_wf_sp: (wf_len, n_active)
            idx = clu_to_idx.get(uid, -1)
            if idx < 0 or not np.any(mean_wf_sp != 0):
                continue
            ptp_per_ch = np.ptp(mean_wf_sp, axis=0)          # (n_active,)
            best_local = np.argmax(ptp_per_ch)
            active_ch  = ch_mask_lbl[uid]
            peak_chan[idx] = active_ch[best_local]
            peak_ptp[idx]  = ptp_per_ch[best_local]
        del mean_wfs

        snr = peak_ptp / noise_levels[peak_chan]

        # ContamPct = R12 (KiloSort metric): min ACG rate in ±1..10×tbin windows
        # divided by baseline rate.  Lag-0 bin (double-spikes) is excluded (K[nbins]=0).
        # ISI violation rate: fraction of consecutive ISIs < isi_ref_ms (simpler, no exclusion).
        print(f"  Computing ContamPct (R12, tbin={contam_tbin_ms} ms) "
              f"and ISI violation rate (ref={isi_ref_ms} ms) …")
        fs         = rec.get_sampling_frequency()
        tbin       = contam_tbin_ms / 1000.0
        isi_ref    = isi_ref_ms / 1000.0 * fs   # threshold in samples
        contam_all   = np.zeros(len(all_cluster_ids))
        isi_viol_all = np.zeros(len(all_cluster_ids))

        # Reuse the pre-sorted grouping built above — no per-cluster full scans.
        st_sec_grp = st_grp.astype(np.float64) / fs
        for idx, clu in enumerate(tqdm(all_cluster_ids, desc="ContamPct/ISI")):
            gi = uid_to_g.get(int(clu), -1)
            if gi < 0:
                continue
            s, e = int(ubounds_g[gi]), int(ubounds_g[gi + 1])
            st_clu = np.sort(st_sec_grp[s:e])
            if len(st_clu) > 10 and (st_clu.max() - st_clu.min()) > 0:
                _, _, contam_all[idx] = check_CCG(st_clu, tbin=tbin)
            if e - s > 1:
                isis = np.diff(np.sort(st_grp[s:e]))
                isi_viol_all[idx] = float((isis < isi_ref).sum()) / len(isis)
        del grp_order, sc_grp, st_grp, uids_g, ucnts_g, ubounds_g, uid_to_g, st_sec_grp

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

    print(f"  good: {(labels=='good').sum()}, mua: {(labels=='mua').sum()}, noise: {(labels=='noise').sum()}")

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

    print(f"  Preserved {preserved} 'merged' labels; assigned {assigned} SNR-based labels.")
    print("  Written: cluster_group.tsv, cluster_snr.tsv, cluster_ContamPct.tsv, cluster_isi_violation.tsv")
    return snr, contam_all, isi_viol_all, labels, n_spikes


# %% [markdown]
# #### Stage 3: build_metrics — mean waveforms, similarity, phy template files, pair metrics
#
# Split into four pieces so that the (cheap) "make phy able to open this"
# step doesn't require redoing the (expensive) waveform/similarity
# computation every time:
#   compute_mean_waveforms   — expensive: one pass over the raw recording
#   compute_similarity_matrix — cheap: just matrix algebra on mean_wf_slay
#   write_phy_template_files  — cheap: array reshuffling + np.save; ALWAYS
#                                run regardless of RUN_BUILD_METRICS, using
#                                whichever mean_wf_slay/similarity is in hand
#                                (freshly computed or loaded from disk), so a
#                                RUN_BUILD_METRICS=False run still leaves phy
#                                able to open the result.
#   compute_pair_metrics      — expensive: XCorr significance per candidate pair

# %%
def compute_mean_waveforms(rec, spike_times, spike_clusters, sparsity_mask,
                            sparsity_uid_to_row, params):
    """Per-cluster mean waveform on each unit's sparse channels (subsampled to
    params['max_spikes']). Returns mean_wf_slay, shape (N_wf, n_ch, wf_len),
    zero outside a unit's active channels — N_wf = spike_clusters.max()+1 so
    it can be indexed directly by raw cluster id.
    """
    pre  = params["pre_samples"]
    post = params["post_samples"]
    n_ch = rec.get_num_channels()
    rng  = np.random.default_rng(0)
    N_wf = int(spike_clusters.max()) + 1

    def channel_mask_for_unit(uid):
        if uid in sparsity_uid_to_row:
            return np.where(sparsity_mask[sparsity_uid_to_row[uid]])[0]
        return np.arange(n_ch)

    active_ids = np.unique(spike_clusters)
    print(f"  Extracting mean waveforms for {len(active_ids)} clusters (sparse channels, "
          f"max {params['max_spikes']} spikes each) …")

    spike_times_for_wf = {}
    for uid in active_ids:
        st = spike_times[spike_clusters == uid]
        if len(st) > params["max_spikes"]:
            st = rng.choice(st, params["max_spikes"], replace=False)
        spike_times_for_wf[int(uid)] = np.sort(st)

    ch_mask = {int(uid): channel_mask_for_unit(int(uid)) for uid in active_ids}

    all_mean_wfs = _extract_waveforms_batched(
        rec, spike_times_for_wf, pre, post, channel_mask=ch_mask,
        return_mean_only=True)  # {uid: mean_wf (wf_len, n_active)}

    mean_wf_slay = np.zeros((N_wf, n_ch, pre + post), dtype=np.float32)
    for uid in active_ids:
        uid        = int(uid)
        mwf_sparse = all_mean_wfs.get(uid)   # (wf_len, n_active)
        if mwf_sparse is None or not np.any(mwf_sparse != 0):
            continue
        active_ch  = ch_mask[uid]
        mean_wf_slay[uid][active_ch, :] = mwf_sparse.T       # (n_active, wf_len)

    mean_wf_path = OUT_DIR / "mean_waveforms.npy"
    np.save(mean_wf_path, mean_wf_slay)
    np.save(PHY_DIR / "mean_waveforms.npy", mean_wf_slay)
    print(f"  Mean waveforms saved -> {mean_wf_path}  shape={mean_wf_slay.shape}")
    return mean_wf_slay, active_ids


def compute_similarity_matrix(mean_wf_slay, sparsity_mask, sparsity_uid_to_row,
                               good_ids, mua_ids):
    """Cosine similarity between good+mua unit mean waveforms, zeroed for
    pairs whose sparse channel neighborhoods don't overlap at all (different
    probe regions -> guaranteed non-matching). This keeps the matrix at
    (n_good+n_mua)^2 instead of the full N_wf^2, which can be tens of GB for
    large sortings.

    Returns cosine_sim_good (n_sim, n_sim) and sim_ids_arr (the cluster ids
    in row/column order) — always look values up via make_sim_lookup(), not
    by raw cluster id.

    sim_ids_arr is also saved to similarity_ids.npy alongside
    similarity_index.npy, so a later run can verify the cached matrix still
    covers every currently-good/mua cluster id before trusting it (a new
    merged cluster id won't be in an old cache — see main()'s
    cached_metrics_ok check).
    """
    N_wf = mean_wf_slay.shape[0]
    sim_ids_arr = np.array(sorted(
        set(good_ids.astype(int)) | set(mua_ids.astype(int))), dtype=int)
    n_sim = len(sim_ids_arr)
    print(f"  Building similarity for {n_sim} units "
          f"(good={len(good_ids)}, mua={len(mua_ids)}) …")

    w_all  = mean_wf_slay.reshape(N_wf, -1).astype(np.float32)
    norms  = np.linalg.norm(w_all, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    w_all /= norms

    # Spatial proximity mask: True if two units share >=1 sparsity channel
    sp_rows = np.zeros((n_sim, sparsity_mask.shape[1]), dtype=bool)
    for i, uid in enumerate(sim_ids_arr):
        if uid in sparsity_uid_to_row:
            sp_rows[i] = sparsity_mask[sparsity_uid_to_row[uid]]
    prox_mask = (sp_rows @ sp_rows.T) > 0             # (n_sim, n_sim) bool

    cosine_sim_good = w_all[sim_ids_arr] @ w_all[sim_ids_arr].T
    np.fill_diagonal(cosine_sim_good, 1.0)
    cosine_sim_good[~prox_mask] = 0.0

    np.save(PHY_DIR / "similarity_index.npy", cosine_sim_good.astype(np.float32))
    np.save(PHY_DIR / "similarity_ids.npy", sim_ids_arr)
    n_prox_pairs = int(np.sum(np.triu(cosine_sim_good > 0, k=1)))
    print(f"  Cosine similarity (good+mua, proximity-filtered): {cosine_sim_good.shape}  "
          f"non-zero pairs: {n_prox_pairs}  saved -> similarity_index.npy")
    return cosine_sim_good, sim_ids_arr


def write_phy_template_files(mean_wf_slay, cosine_sim_good, sim_ids_arr,
                              sparsity_mask, sparsity_uid_to_row,
                              active_ids, spike_clusters, rec, params):
    """Write everything phy needs to open this PHY_DIR: templates.npy,
    template_ind.npy, pc_feature_ind.npy, spike_templates.npy, and
    similar_templates.npy.

    Deliberately separate from the expensive mean-waveform/similarity
    computation and always called by the build_metrics stage regardless of
    RUN_BUILD_METRICS, using whichever mean_wf_slay/cosine_sim_good is
    currently in hand (freshly computed this run, or loaded from disk) — so
    skipping the expensive recompute never leaves phy unable to open the
    result.

    similar_templates.npy MUST be (n_templates, n_templates) — phy's loader
    asserts this exact shape. cosine_sim_good is only (n_sim, n_sim) over the
    good+mua subset, so it's scattered into a full zero-initialized array at
    the right raw-cluster-id positions before saving.
    """
    N_wf = mean_wf_slay.shape[0]

    similar_templates_full = np.zeros((N_wf, N_wf), dtype=np.float32)
    similar_templates_full[np.ix_(sim_ids_arr, sim_ids_arr)] = cosine_sim_good
    np.save(PHY_DIR / "similar_templates.npy", similar_templates_full)
    print(f"  similar_templates.npy written full-size {(N_wf, N_wf)} for phy compatibility")

    print("  Rebuilding templates.npy / template_ind.npy …")
    existing_tmpl_path = PHY_DIR / "templates.npy"
    if existing_tmpl_path.exists():
        n_sparse_chans = int(np.load(existing_tmpl_path).shape[2])
    else:
        n_sparse_chans = TARGET_N_CHANNELS
    ch_locs = np.array(rec.get_channel_locations())  # (n_ch, 2)

    def sparse_chans_for_tmpl(uid):
        uid = int(uid)
        if uid in sparsity_uid_to_row:
            return np.where(sparsity_mask[sparsity_uid_to_row[uid]])[0][:n_sparse_chans]
        amp_per_ch = np.max(np.abs(mean_wf_slay[uid]), axis=1)
        peak_ch    = int(np.argmax(amp_per_ch))
        dists      = np.linalg.norm(ch_locs - ch_locs[peak_ch], axis=1)
        return np.argsort(dists)[:n_sparse_chans]

    wf_len       = params["pre_samples"] + params["post_samples"]
    templates_new = np.zeros((N_wf, wf_len, n_sparse_chans), dtype=np.float32)
    tmpl_ind_new  = -np.ones((N_wf, n_sparse_chans), dtype=np.int64)

    for uid in active_ids:
        ch = sparse_chans_for_tmpl(uid)
        tmpl_ind_new[int(uid), :len(ch)]     = ch
        templates_new[int(uid), :, :len(ch)] = mean_wf_slay[int(uid)][ch, :].T

    np.save(PHY_DIR / "templates.npy",      templates_new)
    np.save(PHY_DIR / "template_ind.npy",   tmpl_ind_new)
    np.save(PHY_DIR / "pc_feature_ind.npy", tmpl_ind_new)
    np.save(PHY_DIR / "spike_templates.npy", spike_clusters.astype(np.int32))
    print(f"  templates       : {templates_new.shape}")
    print(f"  template_ind    : {tmpl_ind_new.shape}")
    print(f"  spike_templates : overwritten to match spike_clusters  (max={int(spike_clusters.max())})")


def compute_pair_metrics(mean_wf_slay, cosine_sim_good, sim_ids_arr, sim_id_to_pos,
                          sim_lookup, good_ids, spike_times, spike_clusters,
                          contam_pct_df, params):
    """XCorr significance + quality columns for every good x good pair above
    the weak similarity threshold. Returns metrics_df, sorted by Similarity
    descending, and also saved to OUT_DIR/metrics.tsv.
    """
    weak_thresh = params["weak_sim_thresh"]
    good_pos_in_sim = np.array([sim_id_to_pos[int(uid)] for uid in good_ids
                                 if int(uid) in sim_id_to_pos], dtype=int)
    good_ids_for_pairs = np.array([int(uid) for uid in good_ids
                                     if int(uid) in sim_id_to_pos], dtype=int)
    sim_good_only = cosine_sim_good[np.ix_(good_pos_in_sim, good_pos_in_sim)]
    pair_ii, pair_jj = np.where(np.triu(sim_good_only > weak_thresh, k=1))
    pairs = [(int(good_ids_for_pairs[pair_ii[k]]), int(good_ids_for_pairs[pair_jj[k]]))
             for k in range(len(pair_ii))]
    print(f"  Candidate pairs (cosine sim > {weak_thresh}): {len(pairs)}")

    fs = params["sample_rate"]
    times_sec = {
        int(uid): (spike_times[spike_clusters == uid] / fs).astype(np.float64)
        for uid in good_ids
    }

    rows = []
    for c1, c2 in tqdm(pairs, desc="Pair metrics"):
        wf1 = mean_wf_slay[c1]    # (n_ch, wf_len)
        wf2 = mean_wf_slay[c2]

        v1 = wf1.flatten().astype(float)
        v2 = wf2.flatten().astype(float)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        slay_sim = float(np.dot(v1, v2) / (n1 * n2)) if (n1 > 0 and n2 > 0) else 0.0

        xgram = slay.x_correlogram(
            times_sec[c1], times_sec[c2],
            window_size=params["window_size"],
            bin_width=params["xcorr_bin_width"],
            overlap_tol=params["overlap_tol"],
        )
        xcorr_sig = slay.calc_xcorr_sig(
            xgram,
            xcorr_bin_width=params["xcorr_bin_width"],
            min_xcorr_rate=params["min_xcorr_rate"],
        )

        rows.append({
            "Cluster 1":       c1,
            "Cluster 2":       c2,
            "Similarity":      sim_lookup(c1, c2),  # good x good sub-matrix, via id->pos map
            "Slay Similarity": slay_sim,
            "XCorr":           float(xcorr_sig),
            "ContamPct_1":     float(contam_pct_df.loc[c1, "ContamPct"]) if c1 in contam_pct_df.index else np.nan,
            "ContamPct_2":     float(contam_pct_df.loc[c2, "ContamPct"]) if c2 in contam_pct_df.index else np.nan,
        })

    metrics_df   = pd.DataFrame(rows).sort_values("Similarity", ascending=False).reset_index(drop=True)
    metrics_path = OUT_DIR / "metrics.tsv"
    metrics_df.to_csv(metrics_path, sep="\t", index=False)
    print(f"  Metrics table: {len(metrics_df)} pairs  ->  {metrics_path}")
    return metrics_df


# %% [markdown]
# #### Stage 4: find_merges — connected-components + trim + progressive test
#
# A merge group satisfies ALL of the following:
#   1. Every member:  contam_pct < max_contam_pct
#   2. Every pair:    sim > weak_sim_thresh    AND xcorr_sig >= xcorr_gate_min  (all-pairs floor)
#   3. Every member:  sim > strong_sim_thresh  AND xcorr_sig >= xcorr_gate_min  with >=1 other member
#   4. Every pair:    min(SNR_a,SNR_b)/max(SNR_a,SNR_b) >= snr_ratio_min  (SNR homogeneity)
#   5. (optional)     peak-channel distance < max_dist
#
# Algorithm:
#   a. Build strong (conditions 1+3+4) and weak (conditions 1+2+4) adjacency matrices.
#   b. Find connected components of the strong graph (order-independent, topology-complete).
#   c. For each component: iteratively remove the node with the most weak-constraint
#      violations until all remaining pairs satisfy the weak floor (or component < 2 nodes).
#   d. Trimmed nodes remain in the pending pool and are re-processed in subsequent
#      iterations — allowing them to form smaller groups with other leftovers.
#   e. Terminate when no new group is formed in an iteration.
#   f. Progressive merge test: within each raw group, seed with the 2 most
#      central members (by summed pairwise similarity) and add the rest one
#      at a time, keeping only members that don't flip the predicted merged
#      label away from "good". Dropped members get a second connected-
#      components + trim + progressive-test pass among themselves.
#
# NOTE on the strong-connectivity guarantee: the trim step (c) only re-checks
# the WEAK floor when deciding whom to remove — it does not re-verify that
# the STRONG connectivity (condition 3) survives the removal. In a strong
# graph that's a path (A-B, B-C, C-D) with no direct A-D/A-C/B-D strong
# edges, trimming C could leave {A, B, D} passing the weak floor even though
# D no longer has a strong edge to anything in the remaining group. This is
# an accepted heuristic limitation (finding the true maximum quasi-clique is
# NP-hard) — flagging it here so it's not mistaken for an oversight.

# %%
def build_edge_matrices(valid_ids, cosine_sim_good, sim_id_to_pos, mean_wf_slay,
                        metrics_df, snr, all_cluster_ids, params):
    """Build the strong/weak boolean adjacency matrices over `valid_ids`
    (good units that passed the contamination filter). Also returns
    peak_chans (for the BlockedBy distance diagnostic) and uid_to_snr.
    """
    n = len(valid_ids)
    valid_pos = np.array([sim_id_to_pos[int(uid)] for uid in valid_ids], dtype=int)
    sim_sub   = cosine_sim_good[np.ix_(valid_pos, valid_pos)].copy()
    np.fill_diagonal(sim_sub, 0.0)

    peak_chans = np.argmax(np.max(mean_wf_slay, 2) - np.min(mean_wf_slay, 2), 1)

    max_dist = params["max_dist"]
    if max_dist < mean_wf_slay.shape[0]:  # skip if max_dist >= total units (effectively disabled)
        dist_mat = np.abs(peak_chans[valid_ids, None] - peak_chans[None, valid_ids])
        sim_sub[dist_mat >= max_dist] = 0.0

    # XCorr gate matrix
    uid2idx  = {int(uid): idx for idx, uid in enumerate(valid_ids)}
    xcorr_ok = np.zeros((n, n), dtype=bool)
    xcorr_gate_min = params["xcorr_gate_min"]
    for _, mrow in metrics_df.iterrows():
        c1, c2 = int(mrow["Cluster 1"]), int(mrow["Cluster 2"])
        if c1 in uid2idx and c2 in uid2idx:
            ok    = float(mrow["XCorr"]) >= xcorr_gate_min
            i, j  = uid2idx[c1], uid2idx[c2]
            xcorr_ok[i, j] = xcorr_ok[j, i] = ok

    # SNR homogeneity mask (snr is positional: snr[i] = SNR of all_cluster_ids[i])
    uid_to_snr = {int(all_cluster_ids[i]): snr[i] for i in range(len(all_cluster_ids))}
    snr_arr = np.array([uid_to_snr.get(uid, 0.0) for uid in valid_ids])
    snr_ratio = (np.minimum(snr_arr[:, None], snr_arr[None, :])
                 / (np.maximum(snr_arr[:, None], snr_arr[None, :]) + 1e-10))
    snr_ok = snr_ratio >= params["snr_ratio_min"]

    strong = (sim_sub > params["strong_sim_thresh"]) & xcorr_ok & snr_ok
    weak   = (sim_sub > params["weak_sim_thresh"])   & xcorr_ok & snr_ok
    n_blocked_by_snr = int((~snr_ok & (sim_sub > params["weak_sim_thresh"]) & xcorr_ok).sum()) // 2
    print(f"  Strong edges: {int(strong.sum()) // 2}  |  Weak edges: {int(weak.sum()) // 2}  "
          f"|  SNR-blocked pairs: {n_blocked_by_snr}")

    return strong, weak, peak_chans, uid_to_snr


def connected_components_trim(valid_ids, strong, weak):
    """Connected-components-then-trim pass: find strong-graph components,
    then within each iteratively drop the member with the most weak-floor
    violations until every remaining pair clears the weak floor. Repeats
    (using leftover/trimmed nodes) until no component >= 2 nodes remains.
    Returns a list of merge groups (lists of raw cluster ids).
    """
    n = len(valid_ids)
    pending = set(range(n))
    merge_groups = []

    while True:
        pend_arr = np.array(sorted(pending))
        if len(pend_arr) < 2:
            break
        adj = strong[np.ix_(pend_arr, pend_arr)]
        comps_local = _connected_components(adj, range(len(pend_arr)))
        if not comps_local:
            break
        found_any_group = False
        for comp_local in comps_local:
            keep = [pend_arr[i] for i in comp_local]
            while len(keep) >= 2:
                viol = {m: sum(1 for o in keep if o != m and not weak[m, o])
                        for m in keep}
                if max(viol.values()) == 0:
                    break
                keep.remove(max(viol, key=viol.get))
            if len(keep) >= 2:
                group = sorted(valid_ids[list(keep)].tolist())
                merge_groups.append(group)
                found_any_group = True
                for idx in keep:
                    pending.discard(idx)
        if not found_any_group:
            break

    return merge_groups


def progressive_merge_test(groups_to_test, sim_lookup, spike_times, spike_clusters,
                            mean_wf_slay, noise_levels, sample_rate):
    """Order each group by descending centrality (summed pairwise similarity
    to other group members), seed with the top 2, then add the rest one at a
    time — keeping a member only if it doesn't flip the predicted merged
    label away from "good". Returns (final_groups, leftover_ids) where
    leftover_ids collects every member that got dropped (for a second pass).
    """
    final_groups = []
    leftover_ids = []

    for group in groups_to_test:
        if len(group) < 2:
            continue
        centrality = [
            sum(sim_lookup(uid, other) for other in group if other != uid)
            for uid in group
        ]
        ordered = [uid for _, uid in sorted(zip(centrality, group), reverse=True)]
        current = ordered[:2]
        lbl, snr_v, nsp_v, ctn_v = _predict_merge_label(
            current, spike_times, spike_clusters,
            mean_wf_slay, noise_levels, sample_rate=sample_rate)
        print(f"  group {group}: seed {current} -> {lbl} "
              f"(SNR={snr_v:.2f} ContamPct={ctn_v:.3f})")
        if lbl != "good":
            leftover_ids.extend(group)
            continue
        for uid in ordered[2:]:
            candidate = current + [uid]
            lbl2, snr_v2, _, ctn_v2 = _predict_merge_label(
                candidate, spike_times, spike_clusters,
                mean_wf_slay, noise_levels, sample_rate=sample_rate)
            print(f"    + {uid} -> {lbl2} (SNR={snr_v2:.2f} ContamPct={ctn_v2:.3f})")
            if lbl2 == "good":
                current = candidate
            else:
                leftover_ids.append(uid)
        if len(current) >= 2:
            final_groups.append(sorted(current))

    return final_groups, leftover_ids


def annotate_blocked_by(metrics_df, merge_groups, contam_pct_df, peak_chans,
                         uid_to_snr, sim_lookup, mean_wf_slay, params):
    """Add a 'BlockedBy' diagnostic column to metrics_df: "merged" for pairs
    in a final merge group, else a semicolon-separated list of the filters
    that prevented the merge (ContamPct -> dist -> xcorr -> snr_ratio -> sim),
    or "trimmed" if every pairwise criterion passed but the pair never made
    it into a final group (i.e. one or both members were dropped during the
    weak-constraint trim step, or the progressive test).
    """
    merged_pairs = set()
    for g in merge_groups:
        for ii in range(len(g)):
            for jj in range(ii + 1, len(g)):
                merged_pairs.add((min(g[ii], g[jj]), max(g[ii], g[jj])))

    max_dist = params["max_dist"]
    weak_thresh, strong_thresh = params["weak_sim_thresh"], params["strong_sim_thresh"]
    xcorr_gate_min = params["xcorr_gate_min"]

    def block_reason(c1, c2, xcorr_val):
        if (min(c1, c2), max(c1, c2)) in merged_pairs:
            return "merged"
        reasons = []
        cp1 = float(contam_pct_df.loc[c1, "ContamPct"]) if c1 in contam_pct_df.index else 1.0
        cp2 = float(contam_pct_df.loc[c2, "ContamPct"]) if c2 in contam_pct_df.index else 1.0
        if cp1 > params["max_contam_pct"]:
            reasons.append(f"ContamPct_1={cp1:.2f}")
        if cp2 > params["max_contam_pct"]:
            reasons.append(f"ContamPct_2={cp2:.2f}")
        if max_dist < mean_wf_slay.shape[0]:
            d = abs(int(peak_chans[c1]) - int(peak_chans[c2]))
            if d >= max_dist:
                reasons.append(f"dist={d}")
        if xcorr_val < xcorr_gate_min:
            reasons.append(f"xcorr={xcorr_val:.3f}")
        s1 = uid_to_snr.get(c1, 0.0)
        s2 = uid_to_snr.get(c2, 0.0)
        r = min(s1, s2) / (max(s1, s2) + 1e-10)
        if r < params["snr_ratio_min"]:
            reasons.append(f"snr_ratio={r:.2f}")
        sim = sim_lookup(c1, c2)
        if sim <= weak_thresh:
            reasons.append(f"sim<weak({sim:.3f})")
        elif sim <= strong_thresh:
            reasons.append(f"sim<strong({sim:.3f})")
        if not reasons:
            reasons.append("trimmed")
        return "; ".join(reasons)

    metrics_df["BlockedBy"] = [
        block_reason(int(r["Cluster 1"]), int(r["Cluster 2"]), float(r["XCorr"]))
        for r in metrics_df.to_dict("records")
    ]
    return metrics_df


def find_merges(good_ids, contam_pct_df, cosine_sim_good, sim_ids_arr, sim_id_to_pos,
                 sim_lookup, mean_wf_slay, metrics_df, snr, all_cluster_ids,
                 spike_times, spike_clusters, noise_levels, params):
    """Run the full find-merges stage: filter to valid units, build edge
    matrices, connected-components + trim, progressive merge test (with a
    second pass over leftovers), annotate metrics_df, and save
    old2new.json / new2old.json / metrics.tsv / run_params.json to OUT_DIR.

    Returns (merge_groups, old2new, new2old, metrics_df).
    """
    valid_ids = np.array([
        int(uid) for uid in good_ids
        if (float(contam_pct_df.loc[int(uid), "ContamPct"])
                if int(uid) in contam_pct_df.index else 1.0) <= params["max_contam_pct"]
    ], dtype=int)
    print(f"Valid units after contam filter: {len(valid_ids)} / {len(good_ids)}")

    strong, weak, peak_chans, uid_to_snr = build_edge_matrices(
        valid_ids, cosine_sim_good, sim_id_to_pos, mean_wf_slay, metrics_df, snr,
        all_cluster_ids, params)

    merge_groups = connected_components_trim(valid_ids, strong, weak)
    print(f"\n-> {len(merge_groups)} raw merge groups (before progressive test):")
    for g in merge_groups:
        print(f"  {g}")

    print("\nProgressive merge test …", flush=True)
    sample_rate = params["sample_rate"]
    final_groups, leftover_ids = progressive_merge_test(
        merge_groups, sim_lookup, spike_times, spike_clusters,
        mean_wf_slay, noise_levels, sample_rate)

    # Second pass: re-run connected-components on leftovers, then progressive test.
    if leftover_ids:
        print(f"\n  {len(leftover_ids)} leftover ID(s) for second pass: {leftover_ids}")
        lo      = np.array(sorted(set(leftover_ids)), dtype=int)
        vid_lut = {int(v): i for i, v in enumerate(valid_ids)}
        lo_idx  = np.array([vid_lut[uid] for uid in lo if uid in vid_lut], dtype=int)
        if len(lo_idx) >= 2:
            adj_lo   = strong[np.ix_(lo_idx, lo_idx)]
            lo_comps = _connected_components(adj_lo, lo_idx.tolist())
            lo_comps = [[int(v) for v in comp] for comp in lo_comps]
            second_final, _second_leftover = progressive_merge_test(
                lo_comps, sim_lookup, spike_times, spike_clusters,
                mean_wf_slay, noise_levels, sample_rate)
            final_groups.extend(second_final)

    merge_groups = final_groups
    print(f"\n-> {len(merge_groups)} final merge groups (after progressive test):")
    for g in merge_groups:
        print(f"  {g}")

    old2new = {}
    new2old = {}
    new_id  = int(spike_clusters.max()) + 1
    for group in merge_groups:
        new2old[new_id] = group
        for uid in group:
            old2new[uid] = new_id
        new_id += 1

    metrics_df = annotate_blocked_by(
        metrics_df, merge_groups, contam_pct_df, peak_chans, uid_to_snr,
        sim_lookup, mean_wf_slay, params)

    with open(OUT_DIR / "old2new.json", "w") as f:
        json.dump({str(k): int(v) for k, v in old2new.items()}, f, separators=(",\n", ":"))
    with open(OUT_DIR / "new2old.json", "w") as f:
        json.dump({str(k): [int(x) for x in v] for k, v in new2old.items()}, f, separators=(",\n", ":"))
    metrics_df.to_csv(OUT_DIR / "metrics.tsv", sep="\t", index=False)

    saveable_params = {k: v for k, v in params.items()
                       if isinstance(v, (str, int, float, bool, list, type(None)))}
    with open(OUT_DIR / "run_params.json", "w") as f:
        json.dump(saveable_params, f, indent=2)

    print(f"\nOutputs saved to {OUT_DIR}/")
    print(f"  new2old.json  ({len(new2old)} entries)")
    print(f"  old2new.json  ({len(old2new)} entries)")

    return merge_groups, old2new, new2old, metrics_df


# %% [markdown]
# #### Stage 5: plot_merges — waveform/correlogram QC figures
#
# Re-predicts each group's merged label and only keeps the ones predicted
# "good" in the final old2new.json/new2old.json — rejected groups still get
# plotted (to REJECTED_DIR) but are dropped from the JSONs a curator would
# hand to accept_merges.

# %%
def _pair_metrics_title(merge, metrics_lookup):
    """Per-pair similarity/quality metrics — shown on the waveform and correlogram plots.

    Builds the line from whatever columns are actually present in
    metrics_lookup's rows, so callers that don't compute every column
    try_merge_selfdbuild_cleaned.py's own find_merges does (e.g. no XCorr or
    "Slay Similarity" — see try_merge_step_2.py) still get a sensible title
    instead of a KeyError.
    """
    lines = []
    for ii in range(len(merge)):
        for jj in range(ii + 1, len(merge)):
            c1, c2 = merge[ii], merge[jj]
            row = metrics_lookup.get((c1, c2))
            if row is None:
                row = metrics_lookup.get((c2, c1))
            if row is None:
                lines.append(f"  [{c1} x {c2}]  (no metrics found)")
                continue
            parts = [f"  [{c1} x {c2}] "]
            if "Similarity" in row and pd.notna(row["Similarity"]):
                parts.append(f"Sim={row['Similarity']:.3f}")
            if "Slay Similarity" in row and pd.notna(row["Slay Similarity"]):
                parts.append(f"SlaySim={row['Slay Similarity']:.3f}")
            if "XCorr" in row and pd.notna(row["XCorr"]):
                parts.append(f"XCorr={row['XCorr']:.2f}")
            if "ContamPct_1" in row and "ContamPct_2" in row:
                parts.append(f"ContamPct: {row['ContamPct_1']:.3f} / {row['ContamPct_2']:.3f}")
            if "PeakChanDist" in row and pd.notna(row["PeakChanDist"]):
                parts.append(f"PeakChanDist={row['PeakChanDist']:.0f}")
            if "SNRRatio" in row and pd.notna(row["SNRRatio"]):
                parts.append(f"SNRRatio={row['SNRRatio']:.2f}")
            lines.append("  ".join(parts))
    return "\n".join(lines)


def _merged_metrics_title(merge, pred_label, pred_snr, pred_nsp, pred_contam):
    """Merged cluster quality — shown only on the merged-cluster plot (page 3)."""
    verdict = "* GOOD" if pred_label == "good" else f"x {pred_label.upper()}"
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


def _plot_merged_cluster_figure(merge, merged_path, merged_title, spike_times,
                                 spike_clusters, mean_wf_slay, rec, params):
    """Page 3: merged-cluster waveforms (all constituent spikes overlaid on
    the weighted mean) + auto-correlogram of the merged spike train."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    pre, post = params["pre_samples"], params["post_samples"]
    n_ch = rec.get_num_channels()
    max_spikes_for_plot = params["max_spikes_for_plot"]
    rng = np.random.default_rng(42)

    all_st_merged = np.sort(np.concatenate(
        [spike_times[spike_clusters == uid] for uid in merge]))

    st_wf = all_st_merged
    if max_spikes_for_plot > 0 and len(st_wf) > max_spikes_for_plot:
        st_wf = rng.choice(st_wf, max_spikes_for_plot, replace=False)
    st_wf = np.sort(st_wf)

    merged_wfs = _extract_waveforms_batched(
        rec, {0: st_wf}, pre, post, channel_mask=None, return_mean_only=False)[0]
    # shape: (n_sp, wf_len, n_ch)

    counts_m = {uid: int((spike_clusters == uid).sum()) for uid in merge}
    total_m  = max(sum(counts_m.values()), 1)
    mwf_m    = sum(counts_m[uid] * mean_wf_slay[uid].astype(float)
                    for uid in merge) / total_m

    ptp_ch  = np.ptp(mwf_m, axis=1)
    peak_ch = int(np.argmax(ptp_ch))
    nchan_m = 10
    ch0     = max(peak_ch - nchan_m // 2, 0)
    ch1     = min(ch0 + nchan_m, n_ch)
    ch0     = max(ch1 - nchan_m, 0)
    disp_ch = list(range(ch0, ch1))
    n_d     = len(disp_ch)

    sp_pk  = merged_wfs[:, :, peak_ch] if len(merged_wfs) > 0 else np.zeros((1, pre + post))
    sp_ptp = max(float(np.ptp(sp_pk)), 1e-6)
    ymin_m = float(sp_pk.min()) - 0.1 * sp_ptp
    ymax_m = float(sp_pk.max()) + 0.1 * sp_ptp

    acg = _auto_correlogram(
        all_st_merged.astype(np.float64) / params["sample_rate"],
        params["window_size"],
        params["ref_pen_bin_width"] / 1000.0,
        params["overlap_tol"],
    )

    n_wf_rows = math.ceil(n_d / 2)
    fig_ht    = max(n_wf_rows * 1.5, 5.0) + 1.5
    fig_m     = plt.figure(figsize=(14, fig_ht), constrained_layout=False)
    fig_m.patch.set_facecolor('#e0e0e0')
    gs_m    = GridSpec(1, 2, figure=fig_m, wspace=0.3,
                        left=0.07, right=0.97, top=0.87, bottom=0.07)
    gs_wf_m = GridSpecFromSubplotSpec(n_wf_rows, 2, subplot_spec=gs_m[0],
                                       wspace=0.08, hspace=0.08)
    ax_acg  = fig_m.add_subplot(gs_m[1])
    color_m = (0.35, 0.60, 0.99)

    for ci, ch in enumerate(disp_ch):
        ax_w = fig_m.add_subplot(gs_wf_m[ci // 2, ci % 2])
        ax_w.set_facecolor("black")
        ax_w.set_ylim(ymin_m, ymax_m)
        ax_w.set_xticks([]); ax_w.set_yticks([])
        ax_w.spines["top"].set_visible(False)
        ax_w.spines["right"].set_visible(False)
        ax_w.set_ylabel(str(ch), size="small", color="black", weight="bold")
        xs = np.arange(pre + post)
        for j in range(len(merged_wfs)):
            ax_w.plot(xs, merged_wfs[j, :, ch], color=color_m, linewidth=0.2, alpha=0.35)
        ax_w.plot(xs, mwf_m[ch], color='white', linewidth=1.3, alpha=0.95)

    ax_acg.set_facecolor("black")
    ax_acg.bar(range(len(acg)), acg, width=1, color=color_m)
    ax_acg.spines["top"].set_visible(False)
    ax_acg.spines["right"].set_visible(False)
    ax_acg.set_title("Auto-correlogram (merged)", color="black", size="small")
    ax_acg.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_acg.set_yticks([0, max(int(acg.max()), 1)])

    fig_m.suptitle(merged_title, fontsize=8, family="monospace")
    fig_m.savefig(merged_path, dpi=150, bbox_inches="tight")
    plt.close(fig_m)


def plot_merges(new2old, old2new, metrics_df, spike_times, spike_clusters,
                 mean_wf_slay, noise_levels, ch_mask_by_uid,
                 rec, params, out_dir=None, plots_dir=None, rejected_dir=None):
    """Render per-group waveform/correlogram/merged-cluster QC figures, then
    rewrite old2new.json/new2old.json to keep only groups whose predicted
    merged label is "good". Returns (good_new2old, good_old2new).

    ch_mask_by_uid : dict {uid: active-channel-index array} or {} to use all
        channels for every unit. Callers build this however is convenient —
        try_merge_selfdbuild_cleaned.py's main() derives it from the
        SortingAnalyzer's sparsity mask; try_merge_step_2.py derives it
        directly from mean_wf_slay's nonzero channels (no analyzer needed).
    out_dir/plots_dir/rejected_dir : override where JSONs/figures are
        written — default to this module's OUT_DIR/PLOTS_DIR/REJECTED_DIR so
        try_merge_selfdbuild_cleaned.py's own call site is unaffected. Pass
        explicit dirs to reuse this function against a different output
        folder (as try_merge_step_2.py does).
    """
    import matplotlib.pyplot as plt
    matplotlib.use("Agg")

    out_dir      = OUT_DIR if out_dir is None else Path(out_dir)
    plots_dir    = PLOTS_DIR if plots_dir is None else Path(plots_dir)
    rejected_dir = REJECTED_DIR if rejected_dir is None else Path(rejected_dir)

    plots_dir.mkdir(exist_ok=True, parents=True)
    rejected_dir.mkdir(exist_ok=True, parents=True)
    plt.ioff()

    pre, post = params["pre_samples"], params["post_samples"]
    max_spikes_for_plot = params["max_spikes_for_plot"]
    rng  = np.random.default_rng(42)
    n_ch = rec.get_num_channels()

    ch_mask_plot = ch_mask_by_uid

    metrics_lookup = {
        (int(r["Cluster 1"]), int(r["Cluster 2"])): r
        for _, r in metrics_df.iterrows()
    }

    MAX_CORR_CLUST = 30
    good_merge_new_ids = set()

    for new_id, old_ids in tqdm(list(new2old.items()), desc="Plotting merges"):
        merge = sorted([int(x) for x in old_ids])
        full_merge_str = "_".join(map(str, merge))
        # Windows MAX_PATH: keep stem <= 200 chars to leave room for dir + suffix
        if len(full_merge_str) > 200:
            digest = hashlib.md5(full_merge_str.encode()).hexdigest()[:8]
            merge_str = full_merge_str[:191] + f"_{digest}"
        else:
            merge_str = full_merge_str

        pred_label, pred_snr, pred_nsp, pred_contam = _predict_merge_label(
            merge, spike_times, spike_clusters, mean_wf_slay, noise_levels,
            sample_rate=params["sample_rate"])
        is_good  = (pred_label == "good")
        plot_dir = plots_dir if is_good else rejected_dir
        wf_path     = plot_dir / f"merge_{merge_str}.png"
        corr_path   = plot_dir / f"merge_{merge_str}_corr.png"
        merged_path = plot_dir / f"merge_{merge_str}_merged.png"

        if is_good:
            good_merge_new_ids.add(new_id)

        if wf_path.exists() and corr_path.exists() and merged_path.exists():
            continue

        pair_title   = _pair_metrics_title(merge, metrics_lookup)
        merged_title = _merged_metrics_title(merge, pred_label, pred_snr, pred_nsp, pred_contam)

        # ── Plot 1: individual cluster waveforms ──────────────────────
        if not wf_path.exists():
            st_for_plot = {}
            for uid in merge:
                st = spike_times[spike_clusters == uid]
                if max_spikes_for_plot > 0 and len(st) > max_spikes_for_plot:
                    st = rng.choice(st, max_spikes_for_plot, replace=False)
                st_for_plot[uid] = np.sort(st)

            wfs_dict = _extract_waveforms_batched(
                rec, st_for_plot, pre, post, channel_mask=ch_mask_plot, return_mean_only=False)

            spikes_slay = {}
            for uid in merge:
                wfs_u = wfs_dict[uid]          # (n_sp, wf_len, n_active)
                full  = np.zeros((wfs_u.shape[0], n_ch, wfs_u.shape[1]), dtype=np.float32)
                if uid in ch_mask_plot:
                    full[:, ch_mask_plot[uid], :] = wfs_u.transpose(0, 2, 1)
                else:
                    full = wfs_u.transpose(0, 2, 1)
                spikes_slay[uid] = full
            del wfs_dict

            gc.collect()
            wf_fig = slay_plot.plot_wfs(merge, mean_wf_slay, spikes_slay,
                                         nchan=10, start=0, stop=pre + post)
            _add_suptitle(wf_fig, pair_title)
            wf_fig.savefig(wf_path, dpi=150, bbox_inches="tight")
            plt.close(wf_fig)
            del spikes_slay
            gc.collect()

        # ── Plot 2: individual cluster correlograms ────────────────────
        # Cap clusters shown to avoid n^2 cross-correlogram explosion
        if not corr_path.exists():
            times_multi = {int(uid): spike_times[spike_clusters == uid] for uid in merge}
            corr_merge = merge if len(merge) <= MAX_CORR_CLUST else merge[:MAX_CORR_CLUST]
            gc.collect()
            corr_fig = slay_plot.plot_corr(corr_merge, times_multi, params)
            _add_suptitle(corr_fig, pair_title)
            cw, ch = corr_fig.get_size_inches()
            # Divide by 2 to budget for bbox_inches='tight' expansion
            safe_dpi = min(150, int(3600 / max(cw, ch, 1)))
            try:
                corr_fig.savefig(corr_path, dpi=max(safe_dpi, 36), bbox_inches="tight")
            except MemoryError:
                corr_fig.savefig(corr_path, dpi=36)
            plt.close(corr_fig)
            gc.collect()

        # ── Plot 3: merged cluster — waveforms + auto-correlogram ──────
        if not merged_path.exists():
            gc.collect()
            _plot_merged_cluster_figure(
                merge, merged_path, merged_title, spike_times, spike_clusters,
                mean_wf_slay, rec, params)
            gc.collect()

    good_new2old = {k: v for k, v in new2old.items() if k in good_merge_new_ids}
    good_old2new = {k: v for k, v in old2new.items() if v in good_merge_new_ids}

    with open(out_dir / "old2new.json", "w") as f:
        json.dump({str(k): int(v) for k, v in good_old2new.items()}, f, separators=(",\n", ":"))
    with open(out_dir / "new2old.json", "w") as f:
        json.dump({str(k): [int(x) for x in v] for k, v in good_new2old.items()}, f, separators=(",\n", ":"))

    print(f"\nMerge quality check: {len(good_new2old)} good / {len(new2old)} total")
    print(f"  Good plots   -> {plots_dir}/")
    print(f"  Rejected     -> {rejected_dir}/")
    print(f"  JSONs rewritten ({len(good_new2old)} good merges kept)")

    plt.ion()
    return good_new2old, good_old2new


# %% [markdown]
# #### Stage 6: accept_merges — apply accepted merges to spike arrays (DESTRUCTIVE)

# %%
def accept_merges(params, out_dir=None, results_folder=None):
    """Apply the merges currently in <out_dir>/new2old.json to the spike
    arrays via slay's accept_all_merges. This overwrites spike_times.npy /
    spike_clusters.npy / spike_templates.npy in PHY_DIR (after saving *_orig
    backups) — irreversible without those backups.

    out_dir/results_folder : override which run's new2old.json gets applied
        — default to this module's OUT_DIR/RUN_TAG. slay's accept_all_merges
        resolves the JSON path as <KS_folder=PHY_DIR>/<results_folder>/new2old.json,
        so results_folder must be PHY_DIR-relative (a folder name, not a full
        path) — pass explicit values to accept merges found by a different
        pipeline stage sharing this PHY_DIR (as try_merge_step_2.py does).
    """
    out_dir        = OUT_DIR if out_dir is None else Path(out_dir)
    results_folder = RUN_TAG if results_folder is None else results_folder
    ks = str(PHY_DIR)

    # Always load from the plain .npy files (not _orig). Slay's accept_all_merges
    # also loads amplitudes.npy / spike_templates.npy from disk, so everything
    # must come from the same consistent snapshot. Slay will then save the
    # current files as *_orig before overwriting, so _orig advances each round.
    spike_times_am    = np.load(PHY_DIR / "spike_times.npy").ravel()
    spike_clusters_am = np.load(PHY_DIR / "spike_clusters.npy").ravel()
    print(f"  spike_times:    spike_times.npy    (n={len(spike_times_am)})")
    print(f"  spike_clusters: spike_clusters.npy (n={len(spike_clusters_am)})")

    # mean_wf: prefer out_dir copy (saved pre-merge, always safe)
    mwf_options = [out_dir / "mean_waveforms.npy", PHY_DIR / "mean_waveforms.npy"]
    mwf_path    = next((p for p in mwf_options if p.exists()), None)
    assert mwf_path is not None, "mean_waveforms.npy not found"
    mean_wf_am = np.load(mwf_path)
    print(f"  mean_wf: {mwf_path.name}  shape={mean_wf_am.shape}")

    params_am = dict(params)
    params_am["KS_folder"]      = ks
    params_am["results_folder"] = results_folder
    params_am["n_chan"]         = mean_wf_am.shape[1]

    unique_am, counts_am = np.unique(spike_clusters_am, return_counts=True)
    n_spikes_am = np.zeros(spike_clusters_am.max() + 1, dtype=int)
    n_spikes_am[unique_am] = counts_am

    times_multi_am = {int(c): spike_times_am[spike_clusters_am == c] for c in unique_am}
    cl_labels_am   = pd.read_csv(PHY_DIR / "cluster_group.tsv", sep="\t", index_col=0)
    if "KSLabel" in cl_labels_am.columns:
        cl_labels_am = cl_labels_am.rename(columns={"KSLabel": "label"})

    # Sanity-check: spike_times must match amplitudes.npy
    n_am = np.load(PHY_DIR / "amplitudes.npy").shape[0]
    assert len(spike_times_am) == n_am, (
        f"Spike count mismatch: spike_times={len(spike_times_am)}  amplitudes={n_am}\n"
        f"  amplitudes.npy is out of sync with spike_times.npy — check PHY_DIR state."
    )
    print(f"  Spike count OK: {len(spike_times_am)} spikes")

    vals = (None, cl_labels_am, mean_wf_am, n_spikes_am,
            spike_times_am, spike_clusters_am, times_multi_am)
    accept_all_merges(vals, params_am)
    print("Merges accepted and files saved.")


# %% [markdown]
# #### Bootstrap helpers — build PHY_DIR from a raw sorting, keep sidecar TSVs in sync

# %%
def bootstrap_phy_dir_from_sorting(rec):
    """If PHY_DIR doesn't exist yet, build a minimal phy export (spike_times,
    spike_clusters, spike_templates, params.py) from SORTING_DIR so the rest
    of the pipeline has something to read.
    """
    if PHY_DIR.exists():
        return
    print(f"PHY_DIR not found — building minimal export_to_phy from {SORTING_DIR} …")
    PHY_DIR.mkdir(parents=True, exist_ok=True)

    sort = si.load(str(SORTING_DIR))
    print(f"  Sorting loaded: {len(sort.unit_ids)} units")

    st_list, sc_list = [], []
    for uid in sort.unit_ids:
        train = sort.get_unit_spike_train(uid, segment_index=0)
        st_list.append(train)
        sc_list.append(np.full(len(train), int(uid), dtype="int32"))
    spike_times    = np.concatenate(st_list).astype("uint64")
    spike_clusters = np.concatenate(sc_list).astype("int32")
    order = np.argsort(spike_times, kind="stable")
    spike_times    = spike_times[order]
    spike_clusters = spike_clusters[order]
    np.save(PHY_DIR / "spike_times.npy",     spike_times)
    np.save(PHY_DIR / "spike_clusters.npy",  spike_clusters)
    np.save(PHY_DIR / "spike_templates.npy", spike_clusters.copy())

    # params.py — needed by si.read_phy
    dtype_str = str(rec.get_dtype())
    (PHY_DIR / "params.py").write_text(
        f"n_channels_dat = {rec.get_num_channels()}\n"
        f"offset = 0\n"
        f"sample_rate = {rec.get_sampling_frequency()}\n"
        f"dtype = '{dtype_str}'\n"
        f"hp_filtered = False\n"
    )
    print(f"  Saved spike_times ({len(spike_times):,} spikes), "
          f"spike_clusters, spike_templates, params.py -> {PHY_DIR}")


def ensure_cluster_group_tsv(spike_clusters):
    group_path = PHY_DIR / "cluster_group.tsv"
    if not group_path.exists():
        print("cluster_group.tsv not found — creating empty one …")
        unique_ids = np.unique(spike_clusters)
        pd.DataFrame({"cluster_id": unique_ids, "label": "", "label_reason": ""}).to_csv(
            group_path, sep="\t", index=False)
        print(f"  Created with {len(unique_ids)} clusters")
    return group_path


def clear_stale_phy_cache():
    """Remove phy's own cache directory (PHY_DIR/.phy) so it rebuilds
    new_cluster_id.json / spikes_per_cluster.pkl fresh next launch, instead
    of tripping `AssertionError: assert np.all(self._spike_clusters <
    self._new_cluster_id)`. That cache goes stale whenever something other
    than phy itself rewrites spike_clusters.npy — which is exactly what
    accept_merges() does — because phy never gets a chance to advance its
    own new_cluster_id counter.
    """
    phy_cache_dir = PHY_DIR / ".phy"
    if phy_cache_dir.exists():
        shutil.rmtree(phy_cache_dir)
        print(f"  Cleared stale phy cache: {phy_cache_dir}")


def sync_metrics_tsvs_to_current_clusters(spike_clusters):
    """si.read_phy inner-joins all .tsv files. After a merge round, the new
    merged cluster IDs are in spike_clusters.npy and cluster_group.tsv but
    not in the metrics files (ContamPct, ISI, SNR) written by the previous
    labelling step — the inner join would silently drop those new IDs,
    causing the analyzer to miss them entirely. Ensure every .tsv with a
    cluster_id column has all current IDs, filling in NaN for new ones.
    """
    sc_ids = set(int(x) for x in np.unique(spike_clusters))
    for tsv in sorted(PHY_DIR.glob("*.tsv")):
        try:
            df = pd.read_csv(tsv, sep="\t")
        except Exception:
            continue
        if "cluster_id" not in df.columns:
            continue
        missing = sc_ids - set(df["cluster_id"].values)
        if missing:
            new_rows = pd.DataFrame({"cluster_id": sorted(missing)})
            for col in df.columns:
                if col != "cluster_id":
                    new_rows[col] = np.nan
            df = pd.concat([df, new_rows], ignore_index=True).sort_values("cluster_id")
            df.to_csv(tsv, sep="\t", index=False)
            print(f"  {tsv.name}: added {len(missing)} missing ID(s) with NaN metrics")


def compute_radius_for_target(analyzer, target_n):
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
    print(f"  Probe pitch ~= {pitch:.1f} um  ->  radius = {best_r:.1f} um  "
          f"~= {int(np.median((dists <= best_r).sum(axis=1)))} ch (median)")
    return best_r


def build_sparsity_from_peaks(rec, sort, spike_times, spike_clusters, peaks_file, target_n_channels):
    """Build a ChannelSparsity mask directly from a peaks file — no recording
    I/O needed. Matches each spike to its detected peak channel via
    searchsorted (peaks file is sorted by sample_index), takes the per-unit
    channel mode, then builds a radius mask from probe geometry.
    """
    t0 = time.time()
    r_um  = compute_radius_for_target(rec, target_n_channels)
    locs  = rec.get_channel_locations()          # (n_ch, 2)
    n_ch  = len(rec.channel_ids)

    peaks_arr = np.load(str(peaks_file))         # sorted structured array
    peak_si   = peaks_arr['sample_index']        # int64, sorted
    peak_ch   = peaks_arr['channel_index'].astype(np.int32)

    idx = np.searchsorted(peak_si, spike_times)
    idx = np.clip(idx, 0, len(peaks_arr) - 1)
    hit = peak_si[idx] == spike_times            # True where times match
    spike_ch = np.where(hit, peak_ch[idx], np.int32(-1))  # -1 = no match
    print(f"  Peaks matched: {hit.sum():,} / {len(spike_times):,} spikes  "
          f"({100 * hit.mean():.1f}%)")
    del peaks_arr, peak_si, peak_ch, idx, hit

    # Pre-sort by cluster for O(n_spikes) total grouping
    order      = np.argsort(spike_clusters, kind='stable')
    sc_sorted  = spike_clusters[order]
    sch_sorted = spike_ch[order]
    uids, ucounts = np.unique(sc_sorted, return_counts=True)
    bounds = np.concatenate([[0], np.cumsum(ucounts)])
    uid_to_j = {int(uid): j for j, uid in enumerate(uids)}
    del order, sc_sorted

    # For each unit, mode of matched peak channels -> peak_chan index.
    # NOTE: units with zero matched peaks keep the default value 0 (channel
    # 0) rather than a sentinel — if the printed match rate above is well
    # under 100%, double-check any specific unit's sparsity looks sane.
    peak_chan = np.zeros(len(sort.unit_ids), dtype=np.int32)
    for i, uid in enumerate(sort.unit_ids):
        j = uid_to_j.get(int(uid), -1)
        if j >= 0:
            chs = sch_sorted[bounds[j]:bounds[j + 1]]
            chs = chs[chs >= 0]
            if len(chs) > 0:
                peak_chan[i] = np.bincount(chs, minlength=n_ch).argmax()
    del sch_sorted, bounds, uid_to_j

    dists = np.linalg.norm(locs[:, np.newaxis] - locs[np.newaxis, :], axis=2)  # (n_ch, n_ch)
    mask  = dists[peak_chan, :] <= r_um                                        # (n_units, n_ch)
    sparsity = si.ChannelSparsity(mask=mask, unit_ids=sort.unit_ids, channel_ids=rec.channel_ids)
    print(f"  Sparsity built from peaks  ({time.time() - t0:.1f}s)")
    counts = mask.sum(axis=1)
    print(f"  Channels/unit: min={counts.min()}  median={int(np.median(counts))}  max={counts.max()}")
    return sparsity


def build_or_load_analyzer(rec, force, required_ids):
    """(Re)build the sparse SortingAnalyzer if `force`, the folder is
    missing, or the existing analyzer doesn't cover every id in
    `required_ids` (e.g. a newly-accepted merged cluster); otherwise load the
    existing one. Sorting is loaded from the phy export directory
    (si.read_phy), NOT SORTING_DIR — SORTING_DIR's NumpySorting hard-codes
    unit_ids from before any merges, and slay's accept_all_merges only
    updates the phy files, not that folder. Reading from phy ensures merged
    unit IDs are included in the sparsity mask. exclude_cluster_groups=[]
    keeps ALL units (including noise/mua) so sparsity_mask covers every
    cluster ID present in spike_clusters.npy.
    """
    required_ids = set(int(u) for u in required_ids)

    if not force and ANALYZER_DIR.exists():
        print(f"Loading existing sparse analyzer from {ANALYZER_DIR} …")
        # load_extensions=False skips loading the waveforms .npy which can be
        # 100+ GB — only sparsity (core metadata) is needed by this script.
        analyzer = si.load_sorting_analyzer(str(ANALYZER_DIR), load_extensions=False)
        covered_ids = set(int(u) for u in analyzer.sparsity.unit_ids)
        if required_ids <= covered_ids:
            return analyzer
        missing = required_ids - covered_ids
        print(f"  Existing analyzer is missing {len(missing)} cluster id(s) "
              f"(likely new merged units) — rebuilding instead of using a stale analyzer.")

    if ANALYZER_DIR.exists():
        print(f"Removing existing analyzer at {ANALYZER_DIR} …")
        shutil.rmtree(ANALYZER_DIR)

    sort = si.read_phy(str(PHY_DIR), exclude_cluster_groups=[])
    print(f"  Sorting loaded from phy: {len(sort.unit_ids)} units  "
          f"(ids {sort.unit_ids.min()}...{sort.unit_ids.max()})")

    spike_times    = np.load(PHY_DIR / "spike_times.npy").ravel()
    spike_clusters = np.load(PHY_DIR / "spike_clusters.npy").ravel()
    sparsity = build_sparsity_from_peaks(
        rec, sort, spike_times, spike_clusters, PEAKS_FILE, TARGET_N_CHANNELS)

    print(f"Building sparse analyzer at {ANALYZER_DIR} …")
    analyzer = si.create_sorting_analyzer(
        sorting=sort, recording=rec,
        format="binary_folder", folder=ANALYZER_DIR, overwrite=True,
        sparse=True, sparsity=sparsity)
    print("  Sparse analyzer built and saved.")
    return analyzer


def try_load_cached_metrics(spike_clusters, good_ids, mua_ids, metrics_path):
    """Load mean_waveforms.npy / similarity_index.npy / similarity_ids.npy /
    metrics.tsv from disk — but only if they actually cover every currently
    active cluster id. Returns None if anything is missing or stale (e.g. a
    newly-accepted merged cluster isn't covered by these caches yet), so the
    caller recomputes rather than trusting a partial/misaligned cache. This
    is the "skip build_metrics only if the indexing still matches" check.

    Returns (mean_wf_slay, cosine_sim_good, sim_ids_arr, metrics_df, active_ids)
    or None.
    """
    mwf_p = next((p for p in [OUT_DIR / "mean_waveforms.npy", PHY_DIR / "mean_waveforms.npy"]
                  if p.exists()), None)
    sim_ids_path = PHY_DIR / "similarity_ids.npy"
    sim_path     = PHY_DIR / "similarity_index.npy"
    if mwf_p is None or not sim_ids_path.exists() or not sim_path.exists() or not metrics_path.exists():
        return None

    mean_wf_slay = np.load(mwf_p)
    if mean_wf_slay.shape[0] <= int(spike_clusters.max()):
        print(f"  Cached mean_waveforms.npy (N={mean_wf_slay.shape[0]}) doesn't cover the current "
              f"max cluster id ({int(spike_clusters.max())}) — cache is stale.")
        return None

    cosine_sim_good  = np.load(sim_path)
    cached_sim_ids   = np.load(sim_ids_path)
    required_sim_ids = np.array(sorted(
        set(good_ids.astype(int)) | set(mua_ids.astype(int))), dtype=int)
    if not np.array_equal(cached_sim_ids, required_sim_ids):
        missing = set(required_sim_ids.tolist()) - set(cached_sim_ids.tolist())
        print(f"  Cached similarity matrix doesn't match the current good+mua id set "
              f"({len(missing)} id(s) missing, e.g. a newly-accepted merge) — cache is stale.")
        return None

    metrics_df = pd.read_csv(metrics_path, sep="\t")
    active_ids = np.unique(spike_clusters)
    return mean_wf_slay, cosine_sim_good, cached_sim_ids, metrics_df, active_ids


def run_analyzer_label_metrics(rec, spike_times, spike_clusters, all_cluster_ids,
                                noise_levels, force_analyzer, force_labels, force_metrics):
    """Run stages 1-3 (analyzer, label_clusters, build_metrics) plus
    write_phy_template_files, and return everything the later stages need.

    Split out of main() so it can be called a SECOND time after
    accept_merges() creates a new merged cluster id — that id exists in
    spike_clusters.npy/cluster_group.tsv the moment accept_merges returns,
    but the analyzer/label/metrics caches from earlier in the same run
    predate it. Calling this again with force_*=False lets the coverage
    checks inside build_or_load_analyzer/label_clusters/try_load_cached_metrics
    detect exactly that gap and recompute only what's actually missing.
    """
    analyzer = build_or_load_analyzer(rec, force=force_analyzer, required_ids=all_cluster_ids)
    sparsity_mask = analyzer.sparsity.mask           # (n_units, n_ch) bool
    sparsity_uid_to_row = {int(cid): i for i, cid in enumerate(all_cluster_ids)}

    snr, contam_all, isi_viol_all, labels, n_spikes_arr = label_clusters(
        spike_times, spike_clusters, all_cluster_ids, rec, noise_levels,
        sparsity_mask=sparsity_mask,
        pre_samples=SAMPLES_BEFORE, post_samples=SAMPLES_AFTER,
        max_spikes_per_unit=MAX_SPIKES_PER_UNIT,
        overwrite=force_labels,
        seed=0,
        contam_tbin_ms=SELF_BUILD_PARAMS["contam_tbin_ms"],
        isi_ref_ms=SELF_BUILD_PARAMS["isi_ref_ms"],
    )

    contam_pct_df = pd.read_csv(PHY_DIR / "cluster_ContamPct.tsv", sep="\t").set_index("cluster_id")
    group_df_cur  = pd.read_csv(PHY_DIR / "cluster_group.tsv", sep="\t")
    good_ids = group_df_cur[group_df_cur["label"] == "good"]["cluster_id"].values
    mua_ids  = group_df_cur[group_df_cur["label"] == "mua"]["cluster_id"].values
    print(f"Good units: {len(good_ids)}  |  MUA: {len(mua_ids)}  |  Total clusters: {len(group_df_cur)}")

    # Even when force_metrics=False, only trust the cache if it actually
    # covers every current cluster id (see try_load_cached_metrics) — a
    # cluster id created by a just-accepted merge won't be in an old cache,
    # and using it anyway is exactly the IndexError/shape-mismatch bug that
    # used to crash phy. "Indexing matches -> skip; otherwise recompute."
    metrics_path = OUT_DIR / "metrics.tsv"
    cached = None if force_metrics else try_load_cached_metrics(
        spike_clusters, good_ids, mua_ids, metrics_path)

    if cached is not None:
        print("  Cached mean waveforms / similarity / metrics cover all current clusters — loading from disk …")
        mean_wf_slay, cosine_sim_good, sim_ids_arr, metrics_df, active_ids = cached
        print(f"  mean_wf_slay : {mean_wf_slay.shape}")
        print(f"  similarity   : {cosine_sim_good.shape} (good+mua, proximity-filtered)")
        print(f"  metrics_df   : {len(metrics_df)} pairs")
    else:
        if force_metrics:
            print("  Recomputing mean waveforms / similarity / metrics (forced) …")
        else:
            print("  Cached outputs are missing/stale — computing mean waveforms / similarity / metrics …")
        mean_wf_slay, active_ids = compute_mean_waveforms(
            rec, spike_times, spike_clusters, sparsity_mask, sparsity_uid_to_row,
            SELF_BUILD_PARAMS)
        cosine_sim_good, sim_ids_arr = compute_similarity_matrix(
            mean_wf_slay, sparsity_mask, sparsity_uid_to_row, good_ids, mua_ids)
        sim_lookup, sim_id_to_pos = make_sim_lookup(cosine_sim_good, sim_ids_arr)
        metrics_df = compute_pair_metrics(
            mean_wf_slay, cosine_sim_good, sim_ids_arr, sim_id_to_pos, sim_lookup,
            good_ids, spike_times, spike_clusters, contam_pct_df, SELF_BUILD_PARAMS)

    sim_lookup, sim_id_to_pos = make_sim_lookup(cosine_sim_good, sim_ids_arr)

    # Always keep phy able to open this PHY_DIR, whether mean_wf_slay/
    # cosine_sim_good above were just computed or loaded from cache — this is
    # the cheap step, decoupled from the expensive one above.
    write_phy_template_files(
        mean_wf_slay, cosine_sim_good, sim_ids_arr, sparsity_mask,
        sparsity_uid_to_row, active_ids, spike_clusters, rec, SELF_BUILD_PARAMS)

    return {
        "analyzer": analyzer, "sparsity_mask": sparsity_mask, "sparsity_uid_to_row": sparsity_uid_to_row,
        "snr": snr, "contam_all": contam_all, "contam_pct_df": contam_pct_df,
        "good_ids": good_ids, "mua_ids": mua_ids,
        "mean_wf_slay": mean_wf_slay, "cosine_sim_good": cosine_sim_good, "sim_ids_arr": sim_ids_arr,
        "sim_lookup": sim_lookup, "sim_id_to_pos": sim_id_to_pos, "metrics_df": metrics_df,
    }


# %% [markdown]
# #### main() — orchestrates all six stages according to the RUN_* flags
#
# After accept_merges (stage 6) actually runs, stages 1-3 are re-run
# automatically (with force_*=False, so only the newly-created merged
# cluster id triggers real recomputation — see run_analyzer_label_metrics's
# coverage checks) and phy's own cache is cleared, so a single invocation
# that includes RUN_ACCEPT_MERGES=True always ends with a directly
# phy-openable PHY_DIR, without needing a second manual run.

# %%
def main():
    rec = si.load(str(RECORDING_PATH))
    print(f"Recording: {rec.get_num_channels()} ch  {rec.get_sampling_frequency():.0f} Hz  "
          f"{rec.get_duration():.1f} s")

    bootstrap_phy_dir_from_sorting(rec)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    PLOTS_DIR.mkdir(exist_ok=True, parents=True)

    print("Loading spike arrays …")
    spike_times    = np.load(PHY_DIR / "spike_times.npy").ravel()
    spike_clusters = np.load(PHY_DIR / "spike_clusters.npy").ravel()
    noise_levels   = si.get_noise_levels(rec, return_scaled=False)

    ensure_cluster_group_tsv(spike_clusters)
    group_df        = pd.read_csv(PHY_DIR / "cluster_group.tsv", sep="\t")
    all_cluster_ids = group_df["cluster_id"].values
    print(f"Total clusters: {len(all_cluster_ids)}")

    sync_metrics_tsvs_to_current_clusters(spike_clusters)

    # ── Stages 1-3: analyzer, label_clusters, build_metrics ───────────────
    ctx = run_analyzer_label_metrics(
        rec, spike_times, spike_clusters, all_cluster_ids, noise_levels,
        force_analyzer=RUN_BUILD_ANALYZER, force_labels=RUN_LABEL_CLUSTERS,
        force_metrics=RUN_BUILD_METRICS)

    # ── Stage 4: find_merges ────────────────────────────────────────────────
    if not RUN_MERGE:
        print("RUN_MERGE=False — skipping.")
    else:
        find_merges(
            ctx["good_ids"], ctx["contam_pct_df"], ctx["cosine_sim_good"], ctx["sim_ids_arr"],
            ctx["sim_id_to_pos"], ctx["sim_lookup"], ctx["mean_wf_slay"], ctx["metrics_df"],
            ctx["snr"], all_cluster_ids, spike_times, spike_clusters, noise_levels, SELF_BUILD_PARAMS)

    # ── Stage 5: plot_merges ────────────────────────────────────────────────
    if not RUN_PLOT_MERGES:
        print("RUN_PLOT_MERGES=False — skipping.")
    else:
        new2old_path = OUT_DIR / "new2old.json"
        old2new_path = OUT_DIR / "old2new.json"
        assert new2old_path.exists() and old2new_path.exists(), \
            "new2old.json/old2new.json not found; run with RUN_MERGE=True first."
        with open(new2old_path) as f:
            new2old = {int(k): [int(x) for x in v] for k, v in json.load(f).items()}
        with open(old2new_path) as f:
            old2new = {int(k): int(v) for k, v in json.load(f).items()}
        ch_mask_by_uid = {
            int(uid): np.where(ctx["sparsity_mask"][ctx["sparsity_uid_to_row"][int(uid)]])[0]
            for uid in np.unique(spike_clusters)
            if int(uid) in ctx["sparsity_uid_to_row"]
        }
        plot_merges(
            new2old, old2new, ctx["metrics_df"], spike_times, spike_clusters,
            ctx["mean_wf_slay"], noise_levels, ch_mask_by_uid,
            rec, SELF_BUILD_PARAMS)

    # ── Stage 6: accept_merges ──────────────────────────────────────────────
    with open(OUT_DIR / "new2old.json") as f:
        new2old_final = json.load(f)
    print(f"Merge groups to accept : {len(new2old_final)}")
    print(f"Old clusters merged in : {sum(len(v) for v in new2old_final.values())}")

    if not RUN_ACCEPT_MERGES:
        print("RUN_ACCEPT_MERGES=False — skipping.")
    else:
        accept_merges(SELF_BUILD_PARAMS)

        # accept_merges() just created new merged cluster id(s) in
        # spike_clusters.npy/cluster_group.tsv. Everything computed above
        # (analyzer, labels, similarity, templates) predates that — re-run
        # stages 1-3 now (force_*=False; the coverage checks will correctly
        # detect the new id(s) and recompute only what's actually needed) so
        # this run ends with a directly phy-openable PHY_DIR, not one that
        # needs a second manual invocation first.
        print("\nRefreshing analyzer / labels / metrics for the newly merged cluster(s) …")
        spike_times    = np.load(PHY_DIR / "spike_times.npy").ravel()
        spike_clusters = np.load(PHY_DIR / "spike_clusters.npy").ravel()
        group_df        = pd.read_csv(PHY_DIR / "cluster_group.tsv", sep="\t")
        all_cluster_ids = group_df["cluster_id"].values
        sync_metrics_tsvs_to_current_clusters(spike_clusters)
        run_analyzer_label_metrics(
            rec, spike_times, spike_clusters, all_cluster_ids, noise_levels,
            force_analyzer=False, force_labels=False, force_metrics=False)

        clear_stale_phy_cache()
        print("PHY_DIR is now up to date and ready to open in phy.")


if __name__ == "__main__":
    main()
