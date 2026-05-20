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
SORTING_DIR    = r"/ptmp/gongt/GLC7189/GLC7189_g_g1/preprocessed/tomato_sorting2_all_try_merge"
PHY_DIR        = r"/ptmp/gongt/GLC7189/GLC7189_g_g1/preprocessed/tomato_sorting2_all_try_merge/export_to_phy"
RECORDING_PATH = r"/ptmp/gongt/GLC7189/GLC7189_g_g1/preprocessed"

# Sparse analyzer: built once (or rebuilt when RUN_BUILD_ANALYZER=True).
# Contains mean templates (~32 channels, 90 samples) used    for similarity.
ANALYZER_DIR   = PHY_DIR + "/_sparse_analyzer0"

# Output sub-folder for this run's results/plots.
# Change the suffix each time you run on freshly-merged data.
RUN_TAG        = "run1"

# =====================================================================
#  RUN FLAGS  — set True for the stages you want to execute
# =====================================================================
RUN_BUILD_ANALYZER   = True  # (re)build sparse SortingAnalyzer from scratch
                             # NOTE: set to False if waveforms are already computed to skip
                             #       the expensive waveform extraction step and go straight
                             #       to label_clusters / metrics.
RUN_LABEL_CLUSTERS   = True   # compute SNR / ContamPct → cluster_group.tsv
RUN_BUILD_METRICS    = True   # extract mean waveforms + pair similarity table
RUN_MERGE            = True   # find merge groups + save JSONs
RUN_PLOT_MERGES      = True   # render waveform / correlogram figures
RUN_ACCEPT_MERGES    = True  # apply accepted merges to spike arrays (destructive)
RUN_PLOT_ALL_PLOTS   = False  # render waveform + ACG figure for every cluster

# ──  cluster_plots settings  (only used when RUN_PLOT_ALL_PLOTS=True)  ──────────
CLUSTER_PLOTS_MIN_SPIKES = 2     # skip clusters with fewer spikes (0 = plot all)
CLUSTER_PLOTS_BATCH_SIZE = 5000  # clusters extracted per recording pass (only when MEAN_ONLY=False)
                                 #   memory ≈ BATCH × MAX_SPIKES_FOR_PLOT × 65samp × 32ch × 4B
                                 #   5000 → ~21 GB;  10000 → ~42 GB;  200 → ~0.8 GB (40–160 h IO!)
CLUSTER_PLOTS_SIM_THRESH = 0.9   # similarity threshold for listing similar clusters
CLUSTER_PLOTS_MEAN_ONLY  = True  # True  → plot mean waveform only (no individual traces);
                                 #          ALL data already in RAM — zero recording I/O,
                                 #          68k clusters in ~10–30 min on 32 cores.
                                 # False → also plot individual spike traces (requires
                                 #          scanning the 831 GB recording, adds 3–8 h).
N_PLOT_WORKERS           = 0     # parallel workers for figure rendering (0 = os.cpu_count())

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
PLOTS_DIR         = OUT_DIR / "merges"
CLUSTER_PLOTS_DIR = ANALYZER_DIR / "cluster_plots"
OUT_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)
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


# ─────────────────────────────────────────────────────────────────────────────
# PHY BOOTSTRAP HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _is_phy_valid(phy_dir):
    """Return True if phy_dir has the minimum required files to proceed."""
    required = ["spike_times.npy", "spike_clusters.npy", "amplitudes.npy"]
    return all((Path(phy_dir) / f).exists() for f in required)


def _bootstrap_phy_from_sorting(sorting_dir, phy_dir, rec):
    """Create a minimal phy-compatible directory from a NumpySorting folder.

    Reads sorting_dir/spikes.npy (structured array with fields sample_index,
    unit_index, segment_index) and writes the minimum files required for
    si.read_phy() and this pipeline into phy_dir:
        spike_times.npy / spike_clusters.npy / amplitudes.npy
        channel_map.npy / channel_positions.npy / params.py
        cluster_group.tsv
    """
    spikes_path = Path(sorting_dir) / "spikes.npy"
    assert spikes_path.exists(), f"spikes.npy not found in {sorting_dir}"
    print(f"\nPHY_DIR missing or incomplete — bootstrapping from {sorting_dir} …")
    Path(phy_dir).mkdir(parents=True, exist_ok=True)

    spikes             = np.load(spikes_path)
    order              = np.argsort(spikes["sample_index"], kind="stable")
    spike_times_arr    = spikes["sample_index"][order].astype(np.int64)
    spike_clusters_arr = spikes["unit_index"][order].astype(np.int32)

    np.save(Path(phy_dir) / "spike_times.npy",    spike_times_arr.reshape(-1, 1))
    np.save(Path(phy_dir) / "spike_clusters.npy", spike_clusters_arr)
    np.save(Path(phy_dir) / "amplitudes.npy",
            np.ones(len(spike_times_arr), dtype=np.float32).reshape(-1, 1))

    n_ch   = rec.get_num_channels()
    ch_pos = np.array(rec.get_channel_locations(), dtype=np.float32)
    np.save(Path(phy_dir) / "channel_map.npy",
            np.arange(n_ch, dtype=np.int32).reshape(-1, 1))
    np.save(Path(phy_dir) / "channel_positions.npy", ch_pos)

    fs = int(rec.get_sampling_frequency())
    (Path(phy_dir) / "params.py").write_text(
        f"sample_rate = {fs}\n"
        f"n_channels_dat = {n_ch}\n"
        f"dtype = 'int16'\n"
        f"offset = 0\n"
        f"hp_filtered = True\n"
    )

    unit_ids = np.unique(spike_clusters_arr)
    pd.DataFrame({
        "cluster_id":   unit_ids,
        "label":        "",
        "label_reason": "",
    }).to_csv(Path(phy_dir) / "cluster_group.tsv", sep="\t", index=False)

    print(f"  spike_times / spike_clusters : {len(spike_times_arr)} spikes")
    print(f"  units                        : {len(unit_ids)}  "
          f"(ids {unit_ids.min()} \u2026 {unit_ids.max()})")
    print(f"  channel_map / channel_positions: {n_ch} channels")
    print(f"  params.py : sample_rate={fs}  n_channels_dat={n_ch}")
    print(f"  \u2192 {phy_dir}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLUSTER PLOT WORKER  (module-level so it is picklable by ProcessPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────────

def _render_cluster_plot_worker(job):
    """Render and save one cluster plot in a subprocess.

    All data (waveforms, mean waveform, ACG, metadata) is pre-computed in the
    main process and passed via *job* dict so the worker needs no globals and
    no access to the recording object.

    job keys
    --------
    out_path        : str – destination .png path
    cid             : int
    n_sp            : int
    snr_b           : float
    contam_b        : float
    label_b         : str
    wfs_disp        : (n_sp, wf_len, n_disp) float32 – individual traces
    mwf_disp        : (n_disp, wf_len) float32 or None – mean waveform
    disp_ch_labels  : list[int] – channel indices (for y-axis labels only)
    ymin, ymax      : float
    acg             : 1-D float array
    sim_partners    : list[(cid, sim)] sorted by descending sim
    pre, post       : int
    sim_thresh      : float
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    import numpy as np
    import math

    out_path       = job["out_path"]
    cid            = job["cid"]
    n_sp           = job["n_sp"]
    snr_b          = job["snr_b"]
    contam_b       = job["contam_b"]
    label_b        = job["label_b"]
    wfs_disp       = job["wfs_disp"]       # (n_sp, wf_len, n_disp)
    mwf_disp       = job["mwf_disp"]       # (n_disp, wf_len) or None
    disp_ch_labels = job["disp_ch_labels"]  # list of int, for axis labels
    ymin           = job["ymin"]
    ymax           = job["ymax"]
    acg            = job["acg"]
    sim_partners   = job["sim_partners"]
    pre            = job["pre"]
    post           = job["post"]
    sim_thresh     = job["sim_thresh"]

    n_disp   = len(disp_ch_labels)
    n_cols_b = 4
    n_rows_b = math.ceil(n_disp / n_cols_b) if n_disp > 0 else 1
    color_cp = (0.35, 0.60, 0.99)
    xs       = np.arange(pre + post)

    fig = plt.figure(figsize=(20, max(n_rows_b * 1.5, 5.0)), constrained_layout=False)
    fig.patch.set_facecolor('#e0e0e0')
    gs = GridSpec(1, 3, figure=fig, width_ratios=[3, 1.3, 1.3],
                  wspace=0.3, left=0.05, right=0.97, top=0.85, bottom=0.07)

    # ── Waveform grid ────────────────────────────────────────────────────
    gs_wf = GridSpecFromSubplotSpec(n_rows_b, n_cols_b, subplot_spec=gs[0],
                                    wspace=0.06, hspace=0.06)
    for ci, ch_label in enumerate(disp_ch_labels):
        ax_w = fig.add_subplot(gs_wf[ci // n_cols_b, ci % n_cols_b])
        ax_w.set_facecolor("black")
        ax_w.set_ylim(ymin, ymax)
        ax_w.set_xticks([]); ax_w.set_yticks([])
        ax_w.spines["top"].set_visible(False)
        ax_w.spines["right"].set_visible(False)
        ax_w.set_ylabel(str(ch_label), size=6, color="white", labelpad=1)
        if len(wfs_disp) > 0 and ci < wfs_disp.shape[2]:
            for j in range(len(wfs_disp)):
                ax_w.plot(xs, wfs_disp[j, :, ci], color=color_cp,
                          linewidth=0.2, alpha=0.3)
        if mwf_disp is not None and ci < len(mwf_disp):
            ax_w.plot(xs, mwf_disp[ci], color='white', linewidth=1.3, alpha=0.95)
    for ci in range(n_disp, n_rows_b * n_cols_b):
        fig.add_subplot(gs_wf[ci // n_cols_b, ci % n_cols_b]).set_visible(False)

    # ── Auto-correlogram ─────────────────────────────────────────────────
    ax_acg = fig.add_subplot(gs[1])
    ax_acg.set_facecolor("black")
    ax_acg.bar(range(len(acg)), acg, width=1, color=color_cp)
    ax_acg.spines["top"].set_visible(False)
    ax_acg.spines["right"].set_visible(False)
    ax_acg.set_title("Auto-correlogram", color="black", size="small")
    ax_acg.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_acg.set_yticks([0, max(int(acg.max()), 1)])

    # ── Info text panel ──────────────────────────────────────────────────
    ax_txt = fig.add_subplot(gs[2])
    ax_txt.set_axis_off()
    sim_lines = [f"  {pid}: {psim:.3f}" for pid, psim in sim_partners[:15]]
    if len(sim_partners) > 15:
        sim_lines.append(f"  \u2026 ({len(sim_partners) - 15} more)")
    info_text = (
        f"cluster_id : {cid}\n"
        f"n_spikes   : {n_sp}\n"
        f"SNR        : {snr_b:.2f}\n"
        f"ContamPct  : {contam_b:.3f}\n"
        f"label      : {label_b}\n\n"
        f"Similar (>{sim_thresh}):\n"
        + ("\n".join(sim_lines) if sim_lines else "  (none)")
    )
    ax_txt.text(0.05, 0.95, info_text, transform=ax_txt.transAxes,
                fontsize=7, family="monospace", verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray",
                          boxstyle="round,pad=0.4"))

    fig.suptitle(
        f"Cluster {cid}  |  n={n_sp}  SNR={snr_b:.2f}  "
        f"Contam={contam_b:.3f}  [{label_b.upper()}]",
        fontsize=9, family="monospace",
    )
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


# %%
# ── Load recording ────────────────────────────────────────────────────

if __name__ == "__main__":
    rec = si.load(str(RECORDING_PATH))
    print(f"Recording: {rec.get_num_channels()} ch  {rec.get_sampling_frequency():.0f} Hz  {rec.get_duration():.1f} s")

    # ── Derived waveform window (samples) ─────────────────────────────────
    _fs          = int(rec.get_sampling_frequency())
    pre_samples  = SAMPLES_BEFORE   # 30
    post_samples = SAMPLES_AFTER    # 35

    # ── Bootstrap PHY_DIR if it is missing or incomplete ─────────────────
    if not _is_phy_valid(PHY_DIR):
        _bootstrap_phy_from_sorting(SORTING_DIR, PHY_DIR, rec)

    # ── Load spike arrays ─────────────────────────────────────────────────
    print("Loading spike arrays …")
    spike_times    = np.load(PHY_DIR / "spike_times.npy").ravel()
    spike_clusters = np.load(PHY_DIR / "spike_clusters.npy").ravel()
    noise_levels   = si.get_noise_levels(rec, return_scaled=False)

    group_path = PHY_DIR / "cluster_group.tsv"
    if not group_path.exists():
        print("cluster_group.tsv not found — creating empty one …")
        unique_ids = np.unique(spike_clusters)
        pd.DataFrame({"cluster_id": unique_ids, "label": "", "label_reason": ""}).to_csv(
            group_path, sep="\t", index=False)
        print(f"  Created with {len(unique_ids)} clusters")

    group_df        = pd.read_csv(group_path, sep="\t")
    all_cluster_ids = group_df["cluster_id"].values
    print(f"Total clusters: {len(all_cluster_ids)}")

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
        print(f"  Probe pitch ≈ {pitch:.1f} µm  →  radius = {best_r:.1f} µm  "
              f"≈ {int(np.median((dists <= best_r).sum(axis=1)))} ch (median)")
        return best_r

    import time as _time
    def _ts(msg):
        print(f"[{_time.strftime('%H:%M:%S')}] {msg}", flush=True)

    if RUN_BUILD_ANALYZER or not ANALYZER_DIR.exists():
        # ── Guard: if waveforms are already computed, don't delete them ──────
        # compute("templates") is intentionally skipped in the build:
        #   - It reads 108 GB waveforms.npy with random access on Lustre → hangs for hours.
        #   - No downstream code uses analyzer.get_extension("templates").
        #   - Mean waveforms are computed via _extract_waveforms_batched() in RUN_BUILD_METRICS.
        # compute("noise_levels") is also skipped: si.get_noise_levels(rec) is used instead.
        _waveforms_done = (ANALYZER_DIR / "extensions" / "waveforms" / "waveforms.npy").exists()
        _sparsity_done  = (ANALYZER_DIR / "sparsity_mask.npy").exists()
        if _waveforms_done and _sparsity_done:
            _ts(f"[INFO] Waveforms already exist — loading analyzer without rebuild to preserve waveforms.npy.")
            analyzer = si.load_sorting_analyzer(str(ANALYZER_DIR))
            _ts(f"  Loaded analyzer. Extensions present: {list(analyzer.extensions.keys())}")
        else:
            if ANALYZER_DIR.exists():
                _ts(f"Removing existing analyzer at {ANALYZER_DIR} …")
                shutil.rmtree(ANALYZER_DIR)

            # Load sorting from the phy export directory so merged unit IDs are included
            sort = si.read_phy(str(PHY_DIR), exclude_cluster_groups=[])
            _ts(f"  Sorting loaded from phy: {len(sort.unit_ids)} units  "
                f"(ids {sort.unit_ids.min()}…{sort.unit_ids.max()})")

            # Step 1 — dense temp analyzer (20 spikes) just to derive sparsity
            _temp_dir = PHY_DIR / "_temp_analyzer_for_sparsity"
            if _temp_dir.exists():
                shutil.rmtree(_temp_dir)
            _ts("Building temp dense analyzer to derive sparsity …")
            _a_temp = si.create_sorting_analyzer(
                sorting=sort, recording=rec,
                format="binary_folder", folder=_temp_dir, overwrite=True, sparse=False)
            _ts("  temp: compute random_spikes (20/unit) …")
            _a_temp.compute("random_spikes", max_spikes_per_unit=20)
            _ts("  temp: compute waveforms …")
            _a_temp.compute("waveforms", ms_before=SAMPLES_BEFORE/_fs*1000, ms_after=SAMPLES_AFTER/_fs*1000,
                            n_jobs=12, chunk_duration="1s", progress_bar=True)
            _ts("  temp: compute templates …")
            _a_temp.compute("templates")
            _ts("  temp: templates done.")

            _r_um    = _compute_radius_for_target(_a_temp, TARGET_N_CHANNELS)
            sparsity = si.compute_sparsity(_a_temp, method="radius", radius_um=_r_um)
            _counts  = np.array([len(v) for v in sparsity.unit_id_to_channel_indices.values()])
            print(f"  Channels/unit: min={_counts.min()}  median={int(np.median(_counts))}  max={_counts.max()}")
            del _a_temp
            shutil.rmtree(_temp_dir)

            # Step 2 — sparse analyzer (1000 spikes, fixed seed)
            _ts(f"Building sparse analyzer at {ANALYZER_DIR} …")
            analyzer = si.create_sorting_analyzer(
                sorting=sort, recording=rec,
                format="binary_folder", folder=ANALYZER_DIR, overwrite=True,
                sparse=True, sparsity=sparsity)
            _ts("  sparse: compute random_spikes (1000/unit) …")
            analyzer.compute("random_spikes", max_spikes_per_unit=MAX_SPIKES_PER_UNIT, seed=0)
            _ts("  sparse: compute waveforms (long step — large write) …")
            analyzer.compute("waveforms", ms_before=SAMPLES_BEFORE/_fs*1000, ms_after=SAMPLES_AFTER/_fs*1000,
                             n_jobs=12, chunk_duration="1s", progress_bar=True)
            _ts("  sparse: waveforms done. Skipping compute(templates) — not needed downstream.")
            _ts("  Sparse analyzer built and saved.")
            print("  Sparse analyzer built and saved.")
    else:
        print(f"Loading existing sparse analyzer from {ANALYZER_DIR} …")
        analyzer = si.load_sorting_analyzer(str(ANALYZER_DIR))
    # ── Load sparsity mask + mean templates from analyzer ─────────────────
    sparsity_mask = analyzer.sparsity.mask           # (n_units, n_ch) bool


    # %%
    JOB_KWARGS = dict(n_jobs=12, chunk_duration="1s", progress_bar=True)

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


    def label_clusters(
        spike_times, spike_clusters, all_cluster_ids, rec, noise_levels,
        sparsity_mask,
        pre_samples=30, post_samples=35, max_spikes_per_unit=1000,
        overwrite=False,
        seed=0,
        contam_tbin_ms=1.5,
        isi_ref_ms=1.5,
        wf_ext=None,
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

            rng        = np.random.default_rng(seed)
            peak_ptp   = np.zeros(len(all_cluster_ids))
            peak_chan  = np.zeros(len(all_cluster_ids), dtype=int)
            n_spikes   = np.zeros(len(all_cluster_ids), dtype=int)
            n_total    = rec.get_total_samples()

            for idx, clu in enumerate(tqdm(all_cluster_ids, desc="Waveforms/SNR")):
                all_indices = np.where(spike_clusters == clu)[0]
                n_spikes[idx] = len(all_indices)
                if len(all_indices) == 0:
                    continue

                # Active channels for this unit from sparsity mask
                if clu in _sparsity_uid_to_row:
                    active_ch = np.where(sparsity_mask[_sparsity_uid_to_row[clu]])[0]
                else:
                    active_ch = np.arange(rec.get_num_channels())

                if wf_ext is not None:
                    # Fast path: use precomputed waveforms from waveforms.npy.
                    # Each unit is a contiguous sequential read (~4 MB) vs 1000 random
                    # rec.get_traces() seeks on Lustre — 100-1000x faster.
                    try:
                        wfs_sparse = wf_ext.get_data(unit_id=int(clu))  # (n_spikes, n_samples, n_active_ch)
                    except Exception:
                        continue
                    if wfs_sparse is None or wfs_sparse.shape[0] == 0:
                        continue
                else:
                    # Slow path: extract waveforms from raw recording one spike at a time
                    if len(all_indices) > max_spikes_per_unit:
                        selected_indices = rng.choice(all_indices, max_spikes_per_unit, replace=False)
                    else:
                        selected_indices = all_indices
                    st = spike_times[selected_indices]
                    st = st[(st >= pre_samples) & (st < n_total - post_samples)]
                    if len(st) == 0:
                        continue
                    wfs_sparse = np.stack([
                        rec.get_traces(start_frame=int(t - pre_samples),
                                       end_frame=int(t + post_samples))[:, active_ch]
                        for t in st
                    ])  # (n_spikes, n_samples, n_active_ch)

                mean_wf_sparse = np.mean(wfs_sparse, axis=0).astype(np.float32)  # (n_samples, n_active_ch)
                ptp_per_ch     = np.ptp(mean_wf_sparse, axis=0)  # (n_active_ch,)
                best_local     = np.argmax(ptp_per_ch)
                peak_ch_abs    = active_ch[best_local]
                peak_chan[idx] = peak_ch_abs
                peak_ptp[idx]  = ptp_per_ch[best_local]

            snr = peak_ptp / noise_levels[peak_chan]

            # ContamPct = R12 (KiloSort metric): min ACG rate in ±1..10×tbin windows
            # divided by baseline rate.  Lag-0 bin (double-spikes) is excluded (K[nbins]=0).
            # ISI violation rate: fraction of consecutive ISIs < isi_ref_ms (simpler, no exclusion).
            print(f"Computing ContamPct (R12, tbin={contam_tbin_ms} ms) "
                  f"and ISI violation rate (ref={isi_ref_ms} ms) …")
            fs       = rec.get_sampling_frequency()
            st_sec   = spike_times.astype(np.float64) / fs
            _tbin    = contam_tbin_ms / 1000.0
            _isi_ref = isi_ref_ms / 1000.0 * fs   # threshold in samples
            contam_all   = np.zeros(len(all_cluster_ids))
            isi_viol_all = np.zeros(len(all_cluster_ids))

            for idx, clu in enumerate(tqdm(all_cluster_ids, desc="ContamPct/ISI")):
                st_clu = st_sec[spike_clusters == clu]
                if len(st_clu) > 10 and (st_clu.max() - st_clu.min()) > 0:
                    _, _, contam_all[idx] = check_CCG(st_clu, tbin=_tbin)
                st_samp = spike_times[spike_clusters == clu]
                if len(st_samp) > 1:
                    isis = np.diff(np.sort(st_samp))
                    isi_viol_all[idx] = float((isis < _isi_ref).sum()) / len(isis)

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

        print(f"Preserved {preserved} 'merged' labels; assigned {assigned} SNR-based labels.")
        print(f"Written: cluster_group.tsv, cluster_snr.tsv, cluster_ContamPct.tsv, cluster_isi_violation.tsv")
        return snr, contam_all, isi_viol_all, labels, n_spikes


    # %%
    print("Loading data files …")
    spike_times    = np.load(PHY_DIR / "spike_times.npy").ravel()
    spike_clusters = np.load(PHY_DIR / "spike_clusters.npy").ravel()

    group_path = PHY_DIR / "cluster_group.tsv"
    if not group_path.exists():
        print("cluster_group.tsv not found — creating empty one …")
        unique_ids = np.unique(spike_clusters)
        pd.DataFrame({"cluster_id": unique_ids, "label": "", "label_reason": ""}).to_csv(
            group_path, sep="\t", index=False)

    group_df        = pd.read_csv(group_path, sep="\t")
    all_cluster_ids = group_df["cluster_id"].values
    noise_levels    = si.get_noise_levels(rec, return_scaled=False)
    print(f"Total clusters: {len(all_cluster_ids)}")

    # %%
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
            wf_ext=analyzer.get_extension("waveforms"),
        )
    else:
        # Load from disk so downstream cells that use snr/contam_all/labels work
        # even when the labelling step is skipped.
        print("RUN_LABEL_CLUSTERS=False — loading existing TSV files …")
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
        print(f"  Loaded SNR/ContamPct/ISI for {len(all_cluster_ids)} clusters.")


    # %% [markdown]
    # #### **Waveform Extraction Helper**

    # %%
    import numpy as np
    from tqdm.auto import tqdm
    import matplotlib.pyplot as plt

    plt.ioff()


    def _extract_waveforms_batched(rec, spike_times_per_unit, pre_samples, post_samples,
                                    channel_mask=None, pbar=None):
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
                               its active channels only.  Shape of stored waveforms:
                                 - None   → (n_spikes, wf_len, n_ch_all)
                                 - given  → (n_spikes, wf_len, n_active_ch)
        pbar                 : optional tqdm progress bar

        Notes
        -----
        Spikes whose window straddles a 5-second chunk boundary are silently
        skipped (rare; < 0.1 % of spikes).  The corresponding waveform rows
        remain zero and the mean is taken only over extracted spikes.
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

        wfs = {uid: np.zeros((len(vtimes), wf_len, _n_out(uid)), dtype=np.float32)
               for uid, vtimes in valid_times.items()}

        if len(all_spikes) == 0:
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
                    wfs[uid][idx] = snippet
                spike_ptr = tmp_ptr
                if pbar is not None:
                    pbar.update(len(spikes_in_chunk))
            else:
                spike_ptr = tmp_ptr

            if chunk_end >= n_total:
                break
            chunk_start = chunk_end - OVERLAP

        return wfs


    # %% [markdown]
    # #### **Build Metrics Table**

    # %%
    if not RUN_BUILD_METRICS:
        # ── Load all precomputed outputs from disk ─────────────────────────
        print("RUN_BUILD_METRICS=False — loading precomputed outputs from disk …")

        _mwf_p = next((p for p in [OUT_DIR / "mean_waveforms.npy",
                                    PHY_DIR / "mean_waveforms.npy"] if p.exists()), None)
        assert _mwf_p is not None, \
            "mean_waveforms.npy not found; run with RUN_BUILD_METRICS=True first."
        mean_wf_slay = np.load(_mwf_p)

        _sim_p = PHY_DIR / "similarity_index.npy"
        assert _sim_p.exists(), \
            "similarity_index.npy not found; run with RUN_BUILD_METRICS=True first."
        sim_matrix = np.load(_sim_p)

        _met_p = OUT_DIR / "metrics.tsv"
        assert _met_p.exists(), \
            "metrics.tsv not found; run with RUN_BUILD_METRICS=True first."
        metrics_df = pd.read_csv(_met_p, sep="\t")

        contam_pct_df = pd.read_csv(PHY_DIR / "cluster_ContamPct.tsv", sep="\t").set_index("cluster_id")
        group_df_cur  = pd.read_csv(PHY_DIR / "cluster_group.tsv", sep="\t")
        good_ids      = group_df_cur[group_df_cur["label"] == "good"]["cluster_id"].values

        print(f"  mean_wf_slay   : {mean_wf_slay.shape}")
        print(f"  sim_matrix     : {sim_matrix.shape}")
        print(f"  metrics_df     : {len(metrics_df)} pairs")
        print(f"  good_ids       : {len(good_ids)}")

    else:
        # ── Compute everything from scratch and save to disk ───────────────
        #   1. Extract per-cluster mean waveforms (sparse channels, max_spikes limit)
        #   2. Compute cosine similarity matrix  → saved to similarity_index.npy
        #   2b. Rebuild templates.npy / template_ind.npy (keeps phy consistent)
        #   3. Compute per-pair XCorr metrics → saved to metrics.tsv

        contam_pct_df = pd.read_csv(PHY_DIR / "cluster_ContamPct.tsv", sep="\t").set_index("cluster_id")
        group_df_cur  = pd.read_csv(PHY_DIR / "cluster_group.tsv", sep="\t")
        good_ids = group_df_cur[group_df_cur["label"] == "good"]["cluster_id"].values
        print(f"Good units: {len(good_ids)}  |  Total clusters: {len(group_df_cur)}")

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
        print(f"Extracting mean waveforms for {N_wf} clusters (sparse channels, "
              f"max {SELF_BUILD_PARAMS['max_spikes']} spikes each) …")

        active_ids = np.unique(spike_clusters)
        _spike_times_for_wf = {}
        for uid in active_ids:
            st = spike_times[spike_clusters == uid]
            if len(st) > SELF_BUILD_PARAMS["max_spikes"]:
                st = rng.choice(st, SELF_BUILD_PARAMS["max_spikes"], replace=False)
            _spike_times_for_wf[int(uid)] = np.sort(st)

        _ch_mask = {int(uid): _channel_mask_for_unit(int(uid)) for uid in active_ids}

        _all_wfs = _extract_waveforms_batched(
            rec, _spike_times_for_wf, pre, post, channel_mask=_ch_mask)

        mean_wf_slay = np.zeros((N_wf, n_ch, pre + post), dtype=np.float32)
        for uid in active_ids:
            uid   = int(uid)
            wfs_u = _all_wfs[uid]
            if len(wfs_u) == 0:
                continue
            valid_mask = np.any(wfs_u != 0, axis=(1, 2))
            if not valid_mask.any():
                continue
            mwf_sparse = np.mean(wfs_u[valid_mask], axis=0)   # (wf_len, n_active)
            active_ch  = _ch_mask[uid]
            mean_wf_slay[uid][active_ch, :] = mwf_sparse.T       # (n_active, wf_len)
        del _all_wfs

        _mean_wf_path = OUT_DIR / "mean_waveforms.npy"
        np.save(_mean_wf_path, mean_wf_slay)
        np.save(PHY_DIR / "mean_waveforms.npy", mean_wf_slay)
        print(f"  Mean waveforms saved → {_mean_wf_path}  shape={mean_wf_slay.shape}")

        # ── 2. Cosine similarity matrix ────────────────────────────────────
        W     = mean_wf_slay.reshape(N_wf, -1).astype(np.float32)
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        W_n        = W / norms
        cosine_sim = W_n @ W_n.T    # (N_wf, N_wf)
        sim_matrix = cosine_sim
        np.save(PHY_DIR / "similarity_index.npy",  cosine_sim.astype(np.float32))
        np.save(PHY_DIR / "similar_templates.npy", cosine_sim.astype(np.float32))
        print(f"  Cosine similarity matrix: {cosine_sim.shape}  saved → similarity_index.npy")

        # ── 2b. Rebuild templates.npy / template_ind.npy ──────────────────
        print("Rebuilding templates.npy / template_ind.npy …")
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
        print(f"  templates       : {_templates_new.shape}")
        print(f"  template_ind    : {_tmpl_ind_new.shape}")
        print(f"  spike_templates : overwritten to match spike_clusters  (max={int(spike_clusters.max())})")

        # ── 3. Spike times in seconds per good unit (for XCorr) ───────────
        fs = SELF_BUILD_PARAMS["sample_rate"]
        _times_sec = {
            int(uid): (spike_times[spike_clusters == uid] / fs).astype(np.float64)
            for uid in good_ids
        }

        # ── 4. Candidate pairs above weak similarity threshold ────────────
        _weak_thresh = SELF_BUILD_PARAMS["weak_sim_thresh"]
        pairs = [
            (int(good_ids[ii]), int(good_ids[jj]))
            for ii in range(len(good_ids))
            for jj in range(ii + 1, len(good_ids))
            if sim_matrix[good_ids[ii], good_ids[jj]] > _weak_thresh
        ]
        print(f"Candidate pairs (cosine sim > {_weak_thresh}): {len(pairs)}")

        # ── 5. Per-pair metrics (XCorr significance + quality columns) ────
        rows = []
        for c1, c2 in tqdm(pairs, desc="Pair metrics"):
            wf1 = mean_wf_slay[c1]    # (n_ch, wf_len)
            wf2 = mean_wf_slay[c2]

            v1 = wf1.flatten().astype(float);  v2 = wf2.flatten().astype(float)
            n1 = np.linalg.norm(v1);           n2 = np.linalg.norm(v2)
            slay_sim = float(np.dot(v1, v2) / (n1 * n2)) if (n1 > 0 and n2 > 0) else 0.0

            xgram = slay.x_correlogram(
                _times_sec[c1], _times_sec[c2],
                window_size=SELF_BUILD_PARAMS["window_size"],
                bin_width=SELF_BUILD_PARAMS["xcorr_bin_width"],
                overlap_tol=SELF_BUILD_PARAMS["overlap_tol"],
            )
            xcorr_sig = slay.calc_xcorr_sig(
                xgram,
                xcorr_bin_width=SELF_BUILD_PARAMS["xcorr_bin_width"],
                min_xcorr_rate=SELF_BUILD_PARAMS["min_xcorr_rate"],
            )

            rows.append({
                "Cluster 1":       c1,
                "Cluster 2":       c2,
                "Similarity":      float(sim_matrix[c1, c2]),
                "Slay Similarity": slay_sim,
                "XCorr":           float(xcorr_sig),
                "ContamPct_1":     float(contam_pct_df.loc[c1, "ContamPct"]) if c1 in contam_pct_df.index else np.nan,
                "ContamPct_2":     float(contam_pct_df.loc[c2, "ContamPct"]) if c2 in contam_pct_df.index else np.nan,
            })

        metrics_df   = pd.DataFrame(rows).sort_values("Similarity", ascending=False).reset_index(drop=True)
        metrics_path = OUT_DIR / "metrics.tsv"
        metrics_df.to_csv(metrics_path, sep="\t", index=False)
        print(f"\nMetrics table: {len(metrics_df)} pairs  →  {metrics_path}")
        metrics_df.head(10)


    # %% [markdown]
    # #### **Plot All Clusters**

    # %%
    if not RUN_PLOT_ALL_PLOTS:
        print("RUN_PLOT_ALL_PLOTS=False — skipping.")
    else:
        import math as _math
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
        from slay.xcorr import auto_correlogram as _acg_fn
        import gc

        plt.ioff()
        CLUSTER_PLOTS_DIR.mkdir(exist_ok=True, parents=True)

        _pre_cp        = SELF_BUILD_PARAMS["pre_samples"]
        _post_cp       = SELF_BUILD_PARAMS["post_samples"]
        _max_sp_cp     = SELF_BUILD_PARAMS["max_spikes_for_plot"]
        _rng_cp        = np.random.default_rng(42)
        _n_ch_cp       = rec.get_num_channels()
        _fs_cp         = SELF_BUILD_PARAMS["sample_rate"]
        _sim_thresh_cp = CLUSTER_PLOTS_SIM_THRESH

        # ── Ensure sim_matrix and mean_wf_slay are available ───────────────
        if 'sim_matrix' not in globals():
            _sim_p = PHY_DIR / "similarity_index.npy"
            if _sim_p.exists():
                sim_matrix = np.load(_sim_p)
                print(f"Loaded sim_matrix from {_sim_p}  shape={sim_matrix.shape}")
            else:
                sim_matrix = None
                print("Warning: similarity_index.npy not found — similar-cluster "
                      "annotations will be omitted.  Run with RUN_BUILD_METRICS=True first.")

        if 'mean_wf_slay' not in globals():
            _mwf_p = next((p for p in [OUT_DIR / "mean_waveforms.npy",
                                        PHY_DIR / "mean_waveforms.npy"] if p.exists()), None)
            if _mwf_p is not None:
                mean_wf_slay = np.load(_mwf_p)
                print(f"Loaded mean_wf_slay from {_mwf_p}  shape={mean_wf_slay.shape}")
            else:
                mean_wf_slay = None
                print("Warning: mean_waveforms.npy not found — mean waveforms unavailable.")

        # ── Build sparsity channel mask ─────────────────────────────────────
        _sparsity_uid_to_row_cp = {int(cid): i for i, cid in enumerate(all_cluster_ids)}
        _ch_mask_cp = {
            int(uid): np.where(sparsity_mask[_sparsity_uid_to_row_cp[int(uid)]])[0]
            for uid in np.unique(spike_clusters)
            if int(uid) in _sparsity_uid_to_row_cp
        }

        # ── Index into quality arrays ───────────────────────────────────────
        _cid_to_idx_cp = {int(cid): i for i, cid in enumerate(all_cluster_ids)}

        # ── Pre-build similar-clusters lookup  (vectorised) ─────────────────
        # For each cluster find all others with sim > threshold using numpy
        # row-slice instead of a Python inner loop (O(N) numpy vs O(N²) Python).
        print("Building similar-cluster lookup …")
        _sim_partners_cp = {}
        if sim_matrix is not None:
            _valid_cp_arr = np.array(
                [int(c) for c in all_cluster_ids if int(c) < sim_matrix.shape[0]],
                dtype=np.int64)
            for _cid_int in tqdm(
                    [int(c) for c in all_cluster_ids], desc="Sim lookup", leave=False):
                if _cid_int >= sim_matrix.shape[0]:
                    _sim_partners_cp[_cid_int] = []
                    continue
                _sims_v = sim_matrix[_cid_int][_valid_cp_arr]  # one numpy row slice
                _mask_v = (_sims_v > _sim_thresh_cp) & (_valid_cp_arr != _cid_int)
                _pidx_v = np.where(_mask_v)[0]
                if len(_pidx_v) == 0:
                    _sim_partners_cp[_cid_int] = []
                else:
                    _sort_v = _pidx_v[np.argsort(-_sims_v[_pidx_v])]
                    _sim_partners_cp[_cid_int] = [
                        (int(_valid_cp_arr[k]), float(_sims_v[k])) for k in _sort_v
                    ]
        else:
            for _cid_key in all_cluster_ids:
                _sim_partners_cp[int(_cid_key)] = []

        # ── Filter clusters to plot ─────────────────────────────────────────
        _cids_to_plot = [
            int(cid) for i, cid in enumerate(all_cluster_ids)
            if n_spikes_arr[i] >= CLUSTER_PLOTS_MIN_SPIKES
        ]
        print(f"Clusters to plot: {len(_cids_to_plot)} / {len(all_cluster_ids)}  "
              f"(min_spikes >= {CLUSTER_PLOTS_MIN_SPIKES})")

        # ── Parallel figure rendering ──────────────────────────────────────
        # Two modes:
        #   CLUSTER_PLOTS_MEAN_ONLY=True  (default, fast):
        #     All data is already in RAM (mean_wf_slay, spike_times, sim_matrix).
        #     Zero recording I/O.  All 68k jobs are submitted at once and rendered
        #     in parallel.  Expected time: ~10–30 min on 32+ cores.
        #   CLUSTER_PLOTS_MEAN_ONLY=False (slow, shows individual spike traces):
        #     Waveform extraction (recording I/O) runs serially in the main process
        #     in batches, then render jobs are dispatched to workers.
        from concurrent.futures import ProcessPoolExecutor
        _n_workers_cp = N_PLOT_WORKERS if N_PLOT_WORKERS > 0 else None  # None = os.cpu_count()

        def _build_job(cid_b, wfs_disp_b=None):
            """Assemble the job dict for _render_cluster_plot_worker."""
            _idx_b       = _cid_to_idx_cp[cid_b]
            _n_sp_b      = int(n_spikes_arr[_idx_b])
            _contam_b    = float(contam_all[_idx_b])
            _label_b     = str(labels[_idx_b])
            _snr_b       = float(snr[_idx_b])
            _active_ch_b = _ch_mask_cp.get(cid_b, np.arange(_n_ch_cp))
            _n_disp_b    = min(32, len(_active_ch_b))
            _disp_ch_b   = _active_ch_b[:_n_disp_b]

            # Mean waveform sliced to displayed channels: (n_disp, wf_len)
            _mwf_disp_b = None
            if (mean_wf_slay is not None
                    and cid_b < mean_wf_slay.shape[0]
                    and len(_disp_ch_b) > 0):
                _mwf_disp_b = mean_wf_slay[cid_b][_disp_ch_b].copy()

            # Individual waveforms for displayed channels (may be empty)
            if wfs_disp_b is None:
                _wfs_disp_b = np.empty((0, _pre_cp + _post_cp, 0), dtype=np.float32)
            else:
                _wfs_u_b    = wfs_disp_b
                _wfs_disp_b = (_wfs_u_b[:, :, :_n_disp_b].copy()
                               if len(_wfs_u_b) > 0
                               else np.empty((0, _pre_cp + _post_cp, 0), dtype=np.float32))

            # Y-axis range
            if _mwf_disp_b is not None and len(_mwf_disp_b) > 0:
                _ptp_per_ch_b = np.ptp(_mwf_disp_b, axis=1)
                _sp_pk_b = _mwf_disp_b[int(np.argmax(_ptp_per_ch_b))]
            elif len(_wfs_disp_b) > 0 and _wfs_disp_b.shape[2] > 0:
                _sp_pk_b = np.mean(_wfs_disp_b[:, :, 0], axis=0)
            else:
                _sp_pk_b = np.zeros(_pre_cp + _post_cp)
            _ptp_b  = max(float(np.ptp(_sp_pk_b)), 1e-6)
            _ymin_b = float(_sp_pk_b.min()) - 0.15 * _ptp_b
            _ymax_b = float(_sp_pk_b.max()) + 0.15 * _ptp_b

            # ACG — computed from in-memory arrays, fast
            _all_st_sec_b = (spike_times[spike_clusters == cid_b]
                             .astype(np.float64) / _fs_cp)
            _acg_b = np.zeros(1)
            if len(_all_st_sec_b) > 1:
                try:
                    _acg_b = _acg_fn(
                        _all_st_sec_b,
                        SELF_BUILD_PARAMS["window_size"],
                        SELF_BUILD_PARAMS["ref_pen_bin_width"] / 1000.0,
                        SELF_BUILD_PARAMS["overlap_tol"],
                    )
                except Exception:
                    pass

            return {
                "out_path":       str(CLUSTER_PLOTS_DIR / f"cluster_{cid_b:06d}.png"),
                "cid":            cid_b,
                "n_sp":           _n_sp_b,
                "snr_b":          _snr_b,
                "contam_b":       _contam_b,
                "label_b":        _label_b,
                "wfs_disp":       _wfs_disp_b,
                "mwf_disp":       _mwf_disp_b,
                "disp_ch_labels": _disp_ch_b.tolist(),
                "ymin":           _ymin_b,
                "ymax":           _ymax_b,
                "acg":            _acg_b,
                "sim_partners":   _sim_partners_cp.get(cid_b, []),
                "pre":            _pre_cp,
                "post":           _post_cp,
                "sim_thresh":     _sim_thresh_cp,
            }

        if CLUSTER_PLOTS_MEAN_ONLY:
            # ── Fast path: all data already in RAM, submit all at once ─────
            print("CLUSTER_PLOTS_MEAN_ONLY=True: skipping recording I/O …")
            _todo_cp = [cid for cid in _cids_to_plot
                        if not (CLUSTER_PLOTS_DIR / f"cluster_{cid:06d}.png").exists()]
            print(f"  Building {len(_todo_cp)} job dicts (ACG + mean waveforms) …")
            _jobs_cp = [_build_job(cid) for cid in tqdm(_todo_cp, desc="Build jobs", leave=False)]
            print(f"  Submitting {len(_jobs_cp)} render jobs to workers …")
            with ProcessPoolExecutor(max_workers=_n_workers_cp) as _executor_cp:
                _futs_cp = [_executor_cp.submit(_render_cluster_plot_worker, j)
                            for j in _jobs_cp]
                for _f_cp in tqdm(_futs_cp, desc="Rendering"):
                    _f_cp.result()
        else:
            # ── Slow path: extract individual spike traces (recording I/O) ─
            print("CLUSTER_PLOTS_MEAN_ONLY=False: extracting waveforms from recording …")
            _n_batches_cp = _math.ceil(len(_cids_to_plot) / CLUSTER_PLOTS_BATCH_SIZE)

            with ProcessPoolExecutor(max_workers=_n_workers_cp) as _executor_cp:
                for _b_cp in tqdm(range(_n_batches_cp), desc="Plotting clusters (batches)"):
                    _batch_cids = _cids_to_plot[
                        _b_cp * CLUSTER_PLOTS_BATCH_SIZE : (_b_cp + 1) * CLUSTER_PLOTS_BATCH_SIZE]
                    _batch_cids = [cid for cid in _batch_cids
                                   if not (CLUSTER_PLOTS_DIR / f"cluster_{cid:06d}.png").exists()]
                    if not _batch_cids:
                        continue

                    _st_batch_cp = {}
                    for _uid_b in _batch_cids:
                        _st_b = spike_times[spike_clusters == _uid_b]
                        if _max_sp_cp > 0 and len(_st_b) > _max_sp_cp:
                            _st_b = _rng_cp.choice(_st_b, _max_sp_cp, replace=False)
                        _st_batch_cp[_uid_b] = np.sort(_st_b)

                    _wfs_batch_cp = _extract_waveforms_batched(
                        rec, _st_batch_cp, _pre_cp, _post_cp, channel_mask=_ch_mask_cp)

                    _batch_futs_cp = []
                    for _cid_b in _batch_cids:
                        _wfs_u_b = _wfs_batch_cp.get(
                            _cid_b, np.empty((0, _pre_cp + _post_cp, 1), dtype=np.float32))
                        _batch_futs_cp.append(
                            _executor_cp.submit(
                                _render_cluster_plot_worker,
                                _build_job(_cid_b, wfs_disp_b=_wfs_u_b)))

                    del _wfs_batch_cp
                    gc.collect()
                    for _f_cp in _batch_futs_cp:
                        _f_cp.result()

        print(f"\nAll-cluster plots saved to {CLUSTER_PLOTS_DIR}/")
        plt.ion()


    # %% [markdown]
    # #### **Filter → Merge Groups → Save Outputs**

    # %%
    if not RUN_MERGE:
        print("RUN_MERGE=False — skipping.")
    else:
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
        print(f"Valid units after contam filter: {_n} / {len(good_ids)}")

        # ── Sub-matrix + optional distance mask ───────────────────────────
        _sim_sub    = sim_matrix[np.ix_(_valid_ids, _valid_ids)].copy()
        np.fill_diagonal(_sim_sub, 0.0)
        _peak_chans = np.argmax(np.max(mean_wf_slay, 2) - np.min(mean_wf_slay, 2), 1)

        if MAX_DIST < sim_matrix.shape[0]:
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
        print(f"Strong edges: {int(_strong.sum()) // 2}  |  Weak edges: {int(_weak.sum()) // 2}  |  SNR-blocked pairs: {int((~_snr_ok & (_sim_sub > WEAK_THRESH) & _xcorr_ok).sum()) // 2}")

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

        print(f"\n→ {len(merge_groups_sb)} raw merge groups (before progressive test):")
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
        print("\nProgressive merge test …", flush=True)
        _sample_rate = SELF_BUILD_PARAMS["sample_rate"]

        pending_groups = list(merge_groups_sb)
        final_groups   = []
        leftover_ids   = []

        def _progressive_test(groups_to_test, out_final, out_leftover):
            for group in groups_to_test:
                if len(group) < 2:
                    continue
                centrality = [
                    sum(float(sim_matrix[uid, other]) for other in group if other != uid)
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
        print(f"\n→ {len(merge_groups_sb)} final merge groups (after progressive test):")
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
            _sim = float(sim_matrix[c1, c2])
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


        print(f"\nOutputs saved to {OUT_DIR}/")    
        print(f"  new2old.json  ({len(new2old_sb)} entries)")
        print(f"  old2new.json  ({len(old2new_sb)} entries)")

    # %% [markdown]
    # #### **Plot Proposed Merges (slay-style)**
    # 

    # %%
    if not RUN_PLOT_MERGES:
        print("RUN_PLOT_MERGES=False — skipping.")
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
                rec, st_for_plot, pre, post, channel_mask=_ch_mask_plot)

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
                    rec, {0: _st_wf}, pre, post, channel_mask=None)[0]
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
        print(f"\nMerge quality check: {n_good_m} good / {n_total_m} total")
        print(f"  Good plots   → {PLOTS_DIR}/")
        print(f"  Rejected     → {REJECTED_DIR}/")
        print(f"  JSONs rewritten ({n_good_m} good merges kept)")

        plt.ion()

    # %% [markdown]
    # #### **Accept merges**

    # %%
    with open(OUT_DIR / "new2old.json") as f:
        new2old = json.load(f)

    num_new_clusters = len(new2old)
    num_old_clusters = sum(len(v) for v in new2old.values())
    print(f"Merge groups to accept : {num_new_clusters}")
    print(f"Old clusters merged in : {num_old_clusters}")

    # %%
    if not RUN_ACCEPT_MERGES:
        print("RUN_ACCEPT_MERGES=False — skipping.")
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
        print(f"  spike_times:    spike_times.npy    (n={len(spike_times_am)})")
        print(f"  spike_clusters: spike_clusters.npy (n={len(spike_clusters_am)})")

        # ── mean_wf: prefer OUT_DIR copy (saved pre-merge, always safe) ───
        _mwf_options = [OUT_DIR / "mean_waveforms.npy", PHY_DIR / "mean_waveforms.npy"]
        _mwf_path    = next((p for p in _mwf_options if p.exists()), None)
        assert _mwf_path is not None, "mean_waveforms.npy not found"
        mean_wf_am = np.load(_mwf_path)
        print(f"  mean_wf: {_mwf_path.name}  shape={mean_wf_am.shape}")

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
        print(f"  Spike count OK: {len(spike_times_am)} spikes")

        # ── Run slay's merge acceptance ───────────────────────────────────
        vals = (None, cl_labels_am, mean_wf_am, n_spikes_am,
                spike_times_am, spike_clusters_am, times_multi_am)
        accept_all_merges(vals, params_am)

        print("Merges accepted and files saved.")



