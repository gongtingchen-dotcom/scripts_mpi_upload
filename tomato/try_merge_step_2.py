"""
try_merge_step_2.py — greedy core-cluster merge search.

Runs independently against ANY phy-openable output directory — set PHY_DIR /
RECORDING_PATH / PEAKS_FILE / ANALYZER_DIR below to point at whatever run you
want (not necessarily one produced by try_merge_selfdbuild_cleaned.py in this
same checkout / with its currently-hardcoded PATHS). The only real
requirement is that PHY_DIR already contains a valid phy export (needs, at
minimum: spike_times.npy, spike_clusters.npy, cluster_group.tsv,
cluster_ContamPct.tsv, cluster_snr.tsv, and mean_waveforms.npy — i.e. at
least try_merge_selfdbuild_cleaned.py's build_analyzer/label_clusters/
build_metrics stages have been run for this PHY_DIR at some point, by this
checkout or another). Same environment requirement as step 1 (the `slay`
micromamba env).

Reuses step 1's shared building blocks directly (imported as `s1`) instead
of re-deriving them, so "what counts as a good unit" and the QC plots stay
identical between the two scripts: s1._predict_merge_label /
s1._compute_unit_metrics / s1._assign_label for the merge-quality gate,
s1.plot_merges for the waveform/correlogram/merged-cluster figures, and
s1.accept_merges for applying accepted groups. Threshold PARAMETERS
(SELF_BUILD_PARAMS: snr_good_min, max_contam_pct, etc.) are deliberately
inherited from s1 wholesale, unmodified — see STEP2_PARAMS below. PATHS are
NOT inherited — s1's own PHY_DIR/RECORDING_PATH/etc. globals are overridden
immediately after import (see "Paths" section) so every s1.* helper called
below operates on THIS script's directories, whatever they're set to.

WHY THIS EXISTS
---------------
Step 1's find_merges only considers good x good candidate pairs, and
requires similarity + XCorr + SNR-ratio to all pass at once before two
units are even candidates. That leaves out cases where a small cluster
looks like mua/noise in isolation (too few spikes for a stable ContamPct,
or a noisy SNR) but would clearly belong to a large, well-isolated unit
once merged into it.

ALGORITHM
---------
1. Build the CORE QUEUE: every currently active cluster labelled 'good',
   sorted by spike count, descending. Only 'good' clusters may seed
   (become the anchor of) a merge group.
2. Walk the queue from most to fewest spikes. For each not-yet-consumed
   core cluster:
     a. Search over ALL active clusters (ANY label -- good/mua/noise) for
        candidates passing THREE pre-filters against the core (all computed
        from each cluster's own individual mean waveform / SNR -- i.e. the
        same values step 1 uses, not the core's evolving pooled group):
          - cosine similarity > core_sim_thresh
          - peak-channel index distance <= max_peak_chan_dist
          - SNR ratio min(SNR_core,SNR_cand)/max(SNR_core,SNR_cand) >= snr_ratio_min
            (same purpose as step 1's SNR-homogeneity gate: stop a strong
            core silently diluting its own contamination budget by
            absorbing many much-weaker candidates one at a time)
        (Small/contaminated-looking clusters are deliberately not excluded
        by LABEL here -- the whole point is that their true quality can
        only be judged once merged. But a candidate must still pass both
        gates above before it's even attempted.)

        NOTE: an XCorr (cross-correlogram significance) gate was tried here
        too, using the same xcorr_gate_min=-5.0 threshold step 1's
        find_merges uses. It was removed after checking a completed step 1
        run's metrics.tsv (813 candidate pairs): XCorr ranged 0.0-1.98,
        never negative at all, let alone near -5 -- the gate was filtering
        out nothing, in either script, on this data. If step 1's XCorr
        computation is ever recalibrated to a threshold that's actually
        selective, it would be worth reintroducing here the same way (per
        (core, candidate) pair, using each cluster's own spike train).
     b. Order the surviving candidates by descending similarity to the core.
     c. Add candidates to the core's group one at a time: after each
        addition, predict the merged unit's label (s1._predict_merge_label,
        the exact same SNR/ContamPct/label logic step 1 uses). Keep the
        addition only if the result is still 'good'. A rejected candidate
        is simply skipped -- it does NOT stop the walk over the core's
        remaining candidates, and it stays available for a different core.
     d. Every cluster ACCEPTED into a group is removed from further
        consideration -- both as a future core, and as a candidate for any
        other core. The core itself is marked consumed the moment it's
        picked off the queue, whether or not it ends up finding any
        partners.
3. Groups that end up with only the core (no accepted partner) aren't
   emitted -- there's nothing to do for them.

Every candidate that clears the (cheap) similarity + peak-channel-distance
screen gets a row in candidates.tsv, whether or not it's ultimately
accepted, with a BlockedBy column explaining exactly which gate stopped it
(snr_ratio=.., or trial_label=mua/noise with the SNR/ContamPct that caused
it) -- mirroring step 1's own BlockedBy diagnostic in find_merges.
Candidates that never clear the cheap screen aren't logged (same scoping
step 1 uses -- logging literally every unrelated pair on the probe would be
noise, not signal).

STAGES (RUN_* flags below, same "skip if already there but force if you
ask" spirit as step 1):
  1. find_core_merges -> STEP2_OUT_DIR/{old2new,new2old,candidates}.{json,tsv}
  2. plot_merges       -> waveform/correlogram/merged-cluster QC figures
                          (via s1.plot_merges, written under STEP2_OUT_DIR)
  3. accept_merges     -> apply accepted groups to spike arrays (DESTRUCTIVE),
                          then refresh step 1's analyzer/labels/metrics and
                          clear phy's cache so PHY_DIR stays directly
                          phy-openable (same as step 1's main() does).
"""

# %% [markdown]
# #### Imports

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import spikeinterface.full as si

import try_merge_selfdbuild_cleaned as s1

# %% [markdown]
# #### Paths — edit these before running (independent of try_merge_selfdbuild_cleaned.py's
# own hardcoded PATHS section — set these to whatever phy output you want to
# work on, whether or not it came from this checkout of step 1)

# %%
PHY_DIR        = r"E:\GLC7189_c_g0\tomato_sorting_selfbuild_stricter_run4\export_to_phy"
RECORDING_PATH = r"E:\GLC7189_c_g0\preprocessed"
# Peaks file from thresholding — used to derive peak channel per unit if the
# analyzer needs (re)building. Set to None to fall back to estimate_sparsity.
PEAKS_FILE     = r"E:\GLC7189_c_g0\preprocessed\thresholding_peaks/peaks_combined2.npy"
# Sparse analyzer for THIS PHY_DIR. Defaults to whatever step 1 would have
# built alongside it (PHY_DIR/_sparse_analyzer5); override if step 1 used a
# different suffix, or if this PHY_DIR wasn't built by step 1 at all.
ANALYZER_DIR   = None

# Run tag step 1 used when it last computed mean_waveforms.npy/metrics.tsv
# for THIS PHY_DIR (i.e. its OUT_DIR = PHY_DIR/STEP1_RUN_TAG) — used only so
# the post-accept refresh below can find and reuse that cache instead of
# recomputing it. Getting this wrong isn't destructive: a miss just means a
# cache-less (slower) recompute, since mean_waveforms.npy/similarity_index.npy
# also always live directly under PHY_DIR as a fallback.
STEP1_RUN_TAG  = "run4"

# Sparse analyzer settings — only used if ANALYZER_DIR needs (re)building.
TARGET_N_CHANNELS    = 32
SAMPLES_BEFORE       = 30
SAMPLES_AFTER        = 35
MAX_SPIKES_PER_UNIT  = 1000

PHY_DIR        = Path(PHY_DIR).resolve()
RECORDING_PATH = Path(RECORDING_PATH).resolve()
PEAKS_FILE     = Path(PEAKS_FILE) if PEAKS_FILE else None
ANALYZER_DIR   = Path(ANALYZER_DIR).resolve() if ANALYZER_DIR else (PHY_DIR / "_sparse_analyzer5")

# Point step 1's module-level globals at THIS script's own paths/settings, so
# every s1.* helper called below (run_analyzer_label_metrics, accept_merges,
# sync_metrics_tsvs_to_current_clusters, clear_stale_phy_cache, ...) operates
# on PHY_DIR/RECORDING_PATH/etc. above -- not whatever happens to be
# hardcoded at the top of try_merge_selfdbuild_cleaned.py right now. This
# works because those functions read PHY_DIR etc. as module globals at call
# time, not as parameters -- confirmed by inspection, not every s1.* function
# takes an explicit path argument. Threshold PARAMETERS (SELF_BUILD_PARAMS)
# are deliberately NOT touched here -- see STEP2_PARAMS below.
s1.PHY_DIR             = PHY_DIR
s1.RECORDING_PATH      = RECORDING_PATH
s1.PEAKS_FILE          = PEAKS_FILE
s1.ANALYZER_DIR        = ANALYZER_DIR
s1.OUT_DIR             = PHY_DIR / STEP1_RUN_TAG
s1.RUN_TAG             = STEP1_RUN_TAG
s1.TARGET_N_CHANNELS   = TARGET_N_CHANNELS
s1.SAMPLES_BEFORE      = SAMPLES_BEFORE
s1.SAMPLES_AFTER       = SAMPLES_AFTER
s1.MAX_SPIKES_PER_UNIT = MAX_SPIKES_PER_UNIT

print(f"PHY_DIR        : {PHY_DIR}")
print(f"RECORDING_PATH : {RECORDING_PATH}")
print(f"ANALYZER_DIR   : {ANALYZER_DIR}")

# %% [markdown]
# #### Run flags

# %%
RUN_FIND_CORE_MERGES = False    # walk cores, find + save merge groups
RUN_PLOT_MERGES      = False   # render waveform / correlogram / merged-cluster QC figures
RUN_ACCEPT_MERGES    = True   # apply accepted merges to spike arrays (DESTRUCTIVE)

# %% [markdown]
# #### Step-2 parameters
#
# Inherits step 1's SELF_BUILD_PARAMS wholesale (pre_samples/post_samples/
# sample_rate/etc -- must match what mean_waveforms.npy was built with, and
# this is also how the "good" label thresholds automatically stay identical
# to step 1's: s1._assign_label/_predict_merge_label read snr_good_min /
# max_contam_pct / noise_min_spikes / contam_tbin_ms straight off
# s1.SELF_BUILD_PARAMS, so step 2 never needs to restate them). It also
# inherits snr_ratio_min this way -- the SNR-homogeneity gate below uses
# step 1's exact same threshold value unless overridden here. (window_size /
# xcorr_bin_width / ref_pen_bin_width / overlap_tol / min_xcorr_rate /
# xcorr_gate_min are also inherited but unused by the merge-finding gate --
# see the XCorr note in the module docstring; window_size/ref_pen_bin_width/
# overlap_tol are still used by s1.plot_merges' correlogram figures.)
#
# NOTE: this inherits THRESHOLD VALUES only (a plain dict copy, taken before
# any path override above) -- not paths. SELF_BUILD_PARAMS also happens to
# carry "KS_folder"/"results_folder" entries baked in from s1's own hardcoded
# PATHS at s1's import time; harmless here since accept_merges() always
# overwrites both from its own out_dir/results_folder arguments (or from the
# now-patched s1.PHY_DIR/s1.RUN_TAG) before use, never from this dict.

# %%
STEP2_PARAMS = dict(s1.SELF_BUILD_PARAMS)
STEP2_PARAMS.update({
    "core_sim_thresh":    0.9,   # cosine similarity of a candidate to the CORE's mean waveform
    "max_peak_chan_dist": 40,    # peak-channel index distance from the core (same convention as
                                 # step 1's "max_dist": a channel-index difference, not microns)
})

STEP2_RUN_TAG      = STEP1_RUN_TAG + "_step2"
STEP2_OUT_DIR      = PHY_DIR / STEP2_RUN_TAG
STEP2_PLOTS_DIR    = STEP2_OUT_DIR / "merges"
STEP2_REJECTED_DIR = STEP2_PLOTS_DIR / "rejected"

print(f"STEP2_OUT_DIR  : {STEP2_OUT_DIR}")


# %% [markdown]
# #### Load step 1's outputs

# %%
def load_step1_outputs():
    """Load everything step 1 already computed: spike arrays, labels,
    ContamPct, and per-cluster mean waveforms. Asserts with a clear message
    if build_metrics/label_clusters haven't been run for this PHY_DIR yet --
    this script never recomputes them itself.
    """
    spike_times    = np.load(s1.PHY_DIR / "spike_times.npy").ravel()
    spike_clusters = np.load(s1.PHY_DIR / "spike_clusters.npy").ravel()

    group_path = s1.PHY_DIR / "cluster_group.tsv"
    assert group_path.exists(), \
        f"{group_path} not found — run try_merge_selfdbuild_cleaned.py (RUN_LABEL_CLUSTERS=True) first."
    group_df = pd.read_csv(group_path, sep="\t")
    label_of = {int(cid): lbl for cid, lbl in zip(group_df["cluster_id"], group_df["label"])}

    contam_path = s1.PHY_DIR / "cluster_ContamPct.tsv"
    assert contam_path.exists(), \
        f"{contam_path} not found — run try_merge_selfdbuild_cleaned.py (RUN_LABEL_CLUSTERS=True) first."
    contam_pct_df = pd.read_csv(contam_path, sep="\t").set_index("cluster_id")

    snr_path = s1.PHY_DIR / "cluster_snr.tsv"
    assert snr_path.exists(), \
        f"{snr_path} not found — run try_merge_selfdbuild_cleaned.py (RUN_LABEL_CLUSTERS=True) first."
    snr_df = pd.read_csv(snr_path, sep="\t")
    # dropna: sync_metrics_tsvs_to_current_clusters() (step 1) inserts NaN
    # placeholder rows for cluster ids new since the last labelling pass --
    # a real SNR isn't available for those yet, so treat them as unknown
    # (falls back to 0.0 via .get() below) rather than propagating NaN into
    # the ratio gate.
    snr_of = {int(cid): float(v) for cid, v in zip(snr_df["cluster_id"], snr_df["SNR"]) if pd.notna(v)}

    mwf_path = s1.PHY_DIR / "mean_waveforms.npy"
    assert mwf_path.exists(), \
        f"{mwf_path} not found — run try_merge_selfdbuild_cleaned.py (RUN_BUILD_METRICS=True) first."
    mean_wf_slay = np.load(mwf_path)

    print(f"  spike_clusters : {len(np.unique(spike_clusters))} active clusters")
    print(f"  mean_wf_slay   : {mean_wf_slay.shape}")
    return spike_times, spike_clusters, label_of, contam_pct_df, snr_of, mean_wf_slay


# %% [markdown]
# #### Stage 1: find_core_merges

# %%
def find_core_merges(spike_times, spike_clusters, label_of, snr_of, mean_wf_slay,
                      noise_levels, params):
    """Greedy core-driven merge search (see module docstring for the
    algorithm, including the SNR-ratio gate). Returns (merge_groups,
    candidate_df):
      merge_groups : list of sorted cluster-id lists, each len >= 2
      candidate_df : one row per candidate that cleared the cheap
                     similarity+distance screen for any core, with a
                     BlockedBy column explaining why it was (or wasn't)
                     ultimately accepted.
    """
    active_ids = np.unique(spike_clusters)

    uids_arr, counts_arr = np.unique(spike_clusters, return_counts=True)
    n_spikes = dict(zip(uids_arr.tolist(), counts_arr.tolist()))

    core_queue = sorted(
        (int(uid) for uid in active_ids if label_of.get(int(uid)) == "good"),
        key=lambda uid: n_spikes[uid], reverse=True,
    )
    print(f"Core queue: {len(core_queue)} 'good' clusters (of {len(active_ids)} total active)")

    # Normalized flattened waveforms + peak channel, over ALL active clusters
    # -- deliberately not restricted to good/mua like step 1's similarity
    # matrix, since candidates here may carry any label.
    id_to_pos = {int(uid): i for i, uid in enumerate(active_ids)}
    W = mean_wf_slay[active_ids].reshape(len(active_ids), -1).astype(np.float32)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    W /= norms

    ptp_per_ch = np.max(mean_wf_slay[active_ids], axis=2) - np.min(mean_wf_slay[active_ids], axis=2)
    peak_chan_of = {int(uid): int(np.argmax(ptp_per_ch[i])) for i, uid in enumerate(active_ids)}

    sim_thresh    = params["core_sim_thresh"]
    max_dist      = params["max_peak_chan_dist"]
    snr_ratio_min = params["snr_ratio_min"]
    sample_rate   = params["sample_rate"]

    consumed = set()
    merge_groups = []
    candidate_rows = []
    n_gate_blocked = 0

    for core in tqdm(core_queue, desc="Core walk"):
        if core in consumed:
            continue
        consumed.add(core)
        core_snr = snr_of.get(core, 0.0)

        remaining = np.array([uid for uid in active_ids if uid not in consumed], dtype=int)
        if len(remaining) == 0:
            continue

        remaining_pos = np.array([id_to_pos[u] for u in remaining])
        sims  = W[remaining_pos] @ W[id_to_pos[core]]
        dists = np.abs(np.array([peak_chan_of[u] for u in remaining]) - peak_chan_of[core])
        cheap_pass = (sims > sim_thresh) & (dists <= max_dist)

        cand_ids   = remaining[cheap_pass]
        cand_sims  = sims[cheap_pass]
        cand_dists = dists[cheap_pass]
        order = np.argsort(-cand_sims)

        current = [core]
        for idx in order:
            cid = int(cand_ids[idx])
            if cid in consumed:
                continue  # defensive; shouldn't happen since `order` is a fixed snapshot
            sim  = float(cand_sims[idx])
            dist = int(cand_dists[idx])

            # ── SNR-homogeneity gate ────────────────────────────────────
            # Compares each cluster's own individual SNR (not the growing
            # `current` group's SNR) -- same convention as step 1.
            cand_snr  = snr_of.get(cid, 0.0)
            snr_ratio = min(core_snr, cand_snr) / (max(core_snr, cand_snr) + 1e-10)
            snr_ok    = snr_ratio >= snr_ratio_min

            if not snr_ok:
                n_gate_blocked += 1
                candidate_rows.append({
                    "Core": core, "Candidate": cid, "Similarity": sim, "PeakChanDist": dist,
                    "SNRRatio": snr_ratio,
                    "TrialLabel": None, "TrialSNR": None, "TrialNSpikes": None, "TrialContamPct": None,
                    "Accepted": False, "BlockedBy": f"snr_ratio={snr_ratio:.2f}",
                })
                continue

            trial = current + [cid]
            label, snr_v, n_sp, contam = s1._predict_merge_label(
                trial, spike_times, spike_clusters, mean_wf_slay, noise_levels,
                sample_rate=sample_rate)
            accepted = (label == "good")
            candidate_rows.append({
                "Core": core, "Candidate": cid, "Similarity": sim, "PeakChanDist": dist,
                "SNRRatio": snr_ratio,
                "TrialLabel": label, "TrialSNR": snr_v, "TrialNSpikes": n_sp, "TrialContamPct": contam,
                "Accepted": accepted,
                "BlockedBy": "merged" if accepted
                             else f"trial_label={label} (SNR={snr_v:.2f} ContamPct={contam:.3f})",
            })
            if accepted:
                current.append(cid)
                consumed.add(cid)

        if len(current) >= 2:
            merge_groups.append(sorted(current))

    candidate_df = pd.DataFrame(candidate_rows)
    n_accepted = int(candidate_df["Accepted"].sum()) if len(candidate_df) else 0
    print(f"\n-> {len(merge_groups)} merge groups found from {len(core_queue)} core clusters")
    print(f"   {len(candidate_df)} candidates cleared sim+distance screen: "
          f"{n_accepted} accepted, {n_gate_blocked} blocked by snr_ratio, "
          f"{len(candidate_df) - n_accepted - n_gate_blocked} blocked by resulting label")
    for g in merge_groups:
        print(f"  {g}")

    return merge_groups, candidate_df


def save_core_merge_outputs(merge_groups, candidate_df, spike_clusters, params):
    """Save old2new.json / new2old.json / candidates.tsv / run_params.json
    to STEP2_OUT_DIR, mirroring step 1's find_merges output layout.
    """
    old2new = {}
    new2old = {}
    new_id = int(spike_clusters.max()) + 1
    for group in merge_groups:
        new2old[new_id] = group
        for uid in group:
            old2new[uid] = new_id
        new_id += 1

    STEP2_OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(STEP2_OUT_DIR / "old2new.json", "w") as f:
        json.dump({str(k): int(v) for k, v in old2new.items()}, f, separators=(",\n", ":"))
    with open(STEP2_OUT_DIR / "new2old.json", "w") as f:
        json.dump({str(k): [int(x) for x in v] for k, v in new2old.items()}, f, separators=(",\n", ":"))
    candidate_df.to_csv(STEP2_OUT_DIR / "candidates.tsv", sep="\t", index=False)

    saveable_params = {k: v for k, v in params.items()
                       if isinstance(v, (str, int, float, bool, list, type(None)))}
    with open(STEP2_OUT_DIR / "run_params.json", "w") as f:
        json.dump(saveable_params, f, indent=2)

    print(f"\nOutputs saved to {STEP2_OUT_DIR}/")
    print(f"  new2old.json   ({len(new2old)} entries)")
    print(f"  old2new.json   ({len(old2new)} entries)")
    print(f"  candidates.tsv ({len(candidate_df)} rows)")
    return old2new, new2old


def try_load_cached_core_merges(active_ids):
    """Load a previous find_core_merges run's outputs from STEP2_OUT_DIR, if
    they're still valid for the current active cluster set. Returns
    (merge_groups, candidate_df) or None if missing/stale (mirrors step 1's
    try_load_cached_metrics -- "skip only if the cache still matches reality").

    A cache is considered stale if any cluster id referenced in new2old.json
    is no longer active -- e.g. something else (a separate accept round)
    already consumed it, so re-running with the same groups would just be
    reapplying decisions made against data that no longer exists.
    """
    old2new_path    = STEP2_OUT_DIR / "old2new.json"
    new2old_path    = STEP2_OUT_DIR / "new2old.json"
    candidates_path = STEP2_OUT_DIR / "candidates.tsv"
    if not (old2new_path.exists() and new2old_path.exists() and candidates_path.exists()):
        return None

    with open(new2old_path) as f:
        new2old = {int(k): [int(x) for x in v] for k, v in json.load(f).items()}

    active_set = set(int(u) for u in active_ids)
    referenced_ids = {uid for group in new2old.values() for uid in group}
    if not referenced_ids <= active_set:
        missing = referenced_ids - active_set
        print(f"  Cached find_core_merges output references {len(missing)} cluster id(s) "
              f"no longer active (e.g. consumed elsewhere since) — cache is stale.")
        return None

    candidate_df = pd.read_csv(candidates_path, sep="\t")
    merge_groups = [sorted(int(x) for x in group) for group in new2old.values()]
    return merge_groups, candidate_df


# %% [markdown]
# #### Stage 2: plot_merges — reuses s1.plot_merges directly

# %%
def build_plot_metrics_df(candidate_df, contam_pct_df):
    """Adapt candidates.tsv into the (Cluster 1, Cluster 2, Similarity,
    ContamPct_1, ContamPct_2, ...) shape s1.plot_merges' pair-title builder
    expects. Only accepted candidates are included (those are the pairs
    that actually appear inside a merge group). Member-to-member pairs that
    don't include the core (e.g. two non-core members of the same group)
    were never directly compared, so they simply won't have a row here --
    s1._pair_metrics_title already falls back to "(no metrics found)" for
    any pair it can't look up.
    """
    accepted = candidate_df[candidate_df["Accepted"]].copy()
    accepted = accepted.rename(columns={"Core": "Cluster 1", "Candidate": "Cluster 2"})

    def _contam(cid):
        return float(contam_pct_df.loc[cid, "ContamPct"]) if cid in contam_pct_df.index else float("nan")

    accepted["ContamPct_1"] = accepted["Cluster 1"].map(_contam)
    accepted["ContamPct_2"] = accepted["Cluster 2"].map(_contam)
    return accepted[["Cluster 1", "Cluster 2", "Similarity", "PeakChanDist",
                     "SNRRatio", "ContamPct_1", "ContamPct_2"]]


# %% [markdown]
# #### main()

# %%
def main():
    spike_times, spike_clusters, label_of, contam_pct_df, snr_of, mean_wf_slay = load_step1_outputs()

    rec = s1.si.load(str(s1.RECORDING_PATH))
    noise_levels = s1.si.get_noise_levels(rec, return_scaled=False)

    # ── Stage 1: find_core_merges ──────────────────────────────────────────
    # RUN_FIND_CORE_MERGES=True forces a recompute; when False, reuse a
    # previous run's saved outputs if they're still valid for the current
    # active cluster set (see try_load_cached_core_merges) -- e.g. so a
    # crash during plotting doesn't force redoing the (slow) core walk.
    cached = None if RUN_FIND_CORE_MERGES else try_load_cached_core_merges(np.unique(spike_clusters))
    if cached is not None:
        merge_groups, candidate_df = cached
        print(f"RUN_FIND_CORE_MERGES=False — loaded {len(merge_groups)} cached merge group(s) "
              f"from {STEP2_OUT_DIR}")
    else:
        if RUN_FIND_CORE_MERGES:
            print("RUN_FIND_CORE_MERGES=True — recomputing the core walk …")
        else:
            print("RUN_FIND_CORE_MERGES=False but cached outputs are missing/stale — computing them …")
        merge_groups, candidate_df = find_core_merges(
            spike_times, spike_clusters, label_of, snr_of, mean_wf_slay, noise_levels, STEP2_PARAMS)
        save_core_merge_outputs(merge_groups, candidate_df, spike_clusters, STEP2_PARAMS)

    # ── Stage 2: plot_merges ────────────────────────────────────────────────
    if not RUN_PLOT_MERGES:
        print("RUN_PLOT_MERGES=False — skipping.")
    else:
        new2old_path = STEP2_OUT_DIR / "new2old.json"
        old2new_path = STEP2_OUT_DIR / "old2new.json"
        candidates_path = STEP2_OUT_DIR / "candidates.tsv"
        assert new2old_path.exists() and old2new_path.exists() and candidates_path.exists(), \
            "STEP2_OUT_DIR outputs not found; run with RUN_FIND_CORE_MERGES=True first."
        with open(new2old_path) as f:
            new2old = {int(k): [int(x) for x in v] for k, v in json.load(f).items()}
        with open(old2new_path) as f:
            old2new = {int(k): int(v) for k, v in json.load(f).items()}
        candidate_df = pd.read_csv(candidates_path, sep="\t")
        metrics_df = build_plot_metrics_df(candidate_df, contam_pct_df)

        ch_mask_by_uid = s1.channel_mask_from_waveforms(mean_wf_slay, np.unique(spike_clusters))

        s1.plot_merges(
            new2old, old2new, metrics_df, spike_times, spike_clusters,
            mean_wf_slay, noise_levels, ch_mask_by_uid, rec, STEP2_PARAMS,
            out_dir=STEP2_OUT_DIR, plots_dir=STEP2_PLOTS_DIR, rejected_dir=STEP2_REJECTED_DIR,
        )

    # ── Stage 3: accept_merges ──────────────────────────────────────────────
    new2old_final_path = STEP2_OUT_DIR / "new2old.json"
    if new2old_final_path.exists():
        with open(new2old_final_path) as f:
            new2old_final = json.load(f)
        print(f"Merge groups to accept : {len(new2old_final)}")
        print(f"Old clusters merged in : {sum(len(v) for v in new2old_final.values())}")

    if not RUN_ACCEPT_MERGES:
        print("RUN_ACCEPT_MERGES=False — skipping.")
    else:
        s1.accept_merges(STEP2_PARAMS, out_dir=STEP2_OUT_DIR, results_folder=STEP2_RUN_TAG)

        # Same reasoning as step 1's main(): accept_merges() just created new
        # merged cluster id(s) in spike_clusters.npy/cluster_group.tsv, so
        # the analyzer/label/metrics caches predate them. Refresh with
        # force_*=False -- s1's coverage checks detect exactly the new id(s)
        # and recompute only what's needed. This also writes a refreshed
        # mean_waveforms.npy/metrics.tsv copy into step 1's OUT_DIR (RUN_TAG)
        # as a side effect of reusing that function, since it isn't
        # STEP2_OUT_DIR-aware -- harmless, but PHY_DIR's own files (the ones
        # phy actually reads: templates.npy/similar_templates.npy/
        # cluster_snr.tsv/etc) are what gets correctly refreshed either way.
        print("\nRefreshing analyzer / labels / metrics for the newly merged cluster(s) …")
        spike_times    = np.load(s1.PHY_DIR / "spike_times.npy").ravel()
        spike_clusters = np.load(s1.PHY_DIR / "spike_clusters.npy").ravel()
        group_df        = pd.read_csv(s1.PHY_DIR / "cluster_group.tsv", sep="\t")
        all_cluster_ids = group_df["cluster_id"].values
        s1.sync_metrics_tsvs_to_current_clusters(spike_clusters)
        s1.run_analyzer_label_metrics(
            rec, spike_times, spike_clusters, all_cluster_ids, noise_levels,
            force_analyzer=False, force_labels=False, force_metrics=False)
        s1.clear_stale_phy_cache()
        print("PHY_DIR is now up to date and ready to open in phy.")


if __name__ == "__main__":
    main()
