#!/usr/bin/env python3
"""
cluster_from_spikes.py
======================
LEGACY — superseded by try_tomato.py (Tomato-based clustering). Kept for
reference: clusters peaks with SpikeInterface's own built-in clustering
methods instead of Tomato.

Full spike-sorting pipeline using SpikeInterface:
  1. Load recording + detected peaks
  2. Cluster peaks with the chosen method (HDBSCAN / isosplit / graph-clustering / ...)
  3. Create SortingAnalyzer + compute extensions
  4. Export to Phy

Run via run_cluster_from_spikes.sh (also legacy).

=======================================================================
CONFIGURE EVERYTHING IN THE "USER CONFIGURATION" SECTION BELOW.
=======================================================================
"""

import os
import sys
import json
import datetime
import numpy as np
import spikeinterface.full as si
from spikeinterface.sortingcomponents.clustering import find_clusters_from_peaks


# =====================================================================
#  USER CONFIGURATION — edit this section only
# =====================================================================

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BIN_PATH   = "/nexus/posix0/MBR-neuralsystems/ephys/GLC6962/GLC6962_g_g0/preprocessed"
PEAKS_PATH = os.path.join(BIN_PATH, "thresholding_peaks", "peaks_combined2.npy")

# ------------------------------------------------------------------
# Choose clustering method
#   "dummy"               – labels by detection channel (debug only)
#   "hdbscan-positions"   – HDBSCAN on 2-D spike locations
#   "random-projections"  – HDBSCAN on random waveform projections
#   "iterative-hdbscan"   – SVD + local HDBSCAN  (Spyking-Circus 2)
#   "iterative-isosplit"  – SVD + local ISOSPLIT  (Tridesclous 2)
#   "graph-clustering"    – global sparse graph + Louvain/Leiden/HDBSCAN
# ------------------------------------------------------------------
METHOD = "iterative-hdbscan"
ANALYZER_FOLDER = os.path.join(BIN_PATH, f"{METHOD}4_analyzer")

# ------------------------------------------------------------------
# Parallel-processing kwargs  (forwarded to compute-heavy steps)
# NOTE: Automatically optimized for current OS:
#       - Linux/macOS: uses multiprocessing with fork context for speed
#       - Windows: falls back to threading (safer with pickle limitations)
# ------------------------------------------------------------------
_IS_WINDOWS = sys.platform == "win32"
_IS_MACOS = sys.platform == "darwin"
_IS_LINUX = sys.platform.startswith("linux")

if _IS_LINUX:
    # ✓ OPTIMIZED FOR LINUX: Use multiprocessing with fork (5-10x faster)
    JOB_KWARGS = {
        "n_jobs":                -1,         # Use all cores (Raven: 36 cores)
        "chunk_duration":        "1s",
        "mp_context":            "fork",     # 5-10x faster than default on Linux
        "pool_engine":           "process",  # True parallelization
        "progress_bar":          True,
    }
elif _IS_MACOS:
    # macOS: spawn is safer than fork
    JOB_KWARGS = {
        "n_jobs":                -1,
        "chunk_duration":        "1s",
        "mp_context":            "spawn",
        "pool_engine":           "process",
        "progress_bar":          True,
    }
else:
    # Windows: threading is safer due to pickle limitations with multiprocessing
    JOB_KWARGS = {
        "n_jobs":                8,          # Conservative for Windows
        "chunk_duration":        "1s",
        "mp_context":            None,       # Default (required for Windows)
        "pool_engine":           "thread",   # Threading instead of multiprocessing
        "max_threads_per_worker": 1,
        "progress_bar":          True,
    }

print(f"[platform] Detected: {sys.platform.upper()}")
print(f"[platform] Using n_jobs={JOB_KWARGS['n_jobs']} with pool_engine='{JOB_KWARGS['pool_engine']}'")


# ------------------------------------------------------------------
# Method-specific parameters — only the block matching METHOD is used
# ------------------------------------------------------------------

# ── dummy ──────────────────────────────────────────────────────────
METHOD_KWARGS_dummy = {}
# No parameters; spikes are labelled by their detection channel.

# ── hdbscan-positions ─────────────────────────────────────────────
METHOD_KWARGS_hdbscan_positions = {
    # Set to a pre-computed array (from localize_peaks) to skip localisation;
    # leave as None to compute on-the-fly.
    "peak_locations": None,

    # Forwarded to localize_peaks() when peak_locations is None.
    "peak_localization_kwargs": {
        "method": "center_of_mass",   # or "monopolar_triangulation" (more accurate, slower)
    },

    # Forwarded to hdbscan.hdbscan().
    "hdbscan_kwargs": {
        "min_cluster_size":     20,    # ↑ = fewer, bigger clusters
        "allow_single_cluster": True,
        "core_dist_n_jobs":     -1,    # all CPUs
    },
}

# ── random-projections ────────────────────────────────────────────
METHOD_KWARGS_random_projections = {
    # HDBSCAN clusterer kwargs
    "clusterer": {
        "min_cluster_size":           15,
        "allow_single_cluster":       True,
        "core_dist_n_jobs":           -1,
        "cluster_selection_method":   "eom",  # "leaf" → more / smaller clusters
    },

    # Waveform snippet window
    "waveforms": {
        "ms_before": 0.5,
        "ms_after":  1.5,
    },

    # Sparse channel selection per spike
    "sparsity": {
        "method":         "snr",          # or "radius"
        "amplitude_mode": "peak_to_peak",
        "threshold":      0.25,           # lower → keep more channels
    },

    "radius_um":      50,     # spatial radius for waveform extraction (µm)
    "nb_projections": 10,     # number of random projection directions
    "feature":        "ptp",  # "ptp" (peak-to-peak) or "mean"
    "seed":           42,

    # Savitzky-Golay smoothing before projecting
    "smoothing": {
        "window_length_ms": 0.25,
    },

    # Template-similarity merge after clustering (None to skip)
    "merge_from_templates": {
        "similarity_thresh": 0.8,  # ↑ = stricter, less merging
        "num_shifts":        3,
    },

    "debug_folder": None,
    "verbose":      True,
}

# ── iterative-hdbscan ─────────────────────────────────────────────
# OPTIMIZED for reducing noise clusters while merging similar templates.
METHOD_KWARGS_iterative_hdbscan = {
    # SVD feature extraction
    "peaks_svd": {
        "n_components": 8,       # 8-D feature space (good balance)
        "ms_before":    1.0,
        "ms_after":     1.5,
        "radius_um":    120.0,   # spatial context for features
    },

    "seed": 42,

    # Local splitting step — TUNED TO REDUCE NOISE
    "split": {
        "split_radius_um": 100.0,   # neighbour channel radius for splitting (µm)
        "recursive":       True,
        "recursive_depth": 5,       # reduced: 10→5 (avoid over-splitting)
        "method_kwargs": {
            "clusterer": {
                "method":               "hdbscan",
                "min_cluster_size":     15,     # INCREASED: 2→15 (fewer tiny noise clusters)
                "allow_single_cluster": False,
                "min_samples":          20,     # stricter: 20→15 for balance
            },
            "n_pca_features":   8,      # reduced: 12→8 for consistency
            "projection_mode":  "tsvd",
        },
    },

    # Template-similarity merge — INCREASED TO MERGE MORE
    "merge_from_templates": {
        "similarity_thresh": 0.9,  # REDUCED: 0.9→0.75 (merge more similar templates)
        "num_shifts":        4,     # REDUCED: 10→5 (faster, still thorough)
        "use_lags":          True,
    },

    # Feature-distribution merge (None to disable)
    "merge_from_features": None,

    "debug_folder": None,
    "verbose":      True,
}

# ── iterative-isosplit ────────────────────────────────────────────
METHOD_KWARGS_iterative_isosplit = {
    # SVD feature extraction
    "peaks_svd": {
        "n_components": 6,
        "ms_before":    0.5,
        "ms_after":     1.5,
        "radius_um":    120.0,
        "motion":       None,   # pass Motion object if doing drift correction
    },

    # Pre-grouping strategy before local splitting
    "pre_label": {
        "mode": "channel",   # "channel" (default) or "vertical_bin"
    },

    "seed": None,   # set an int (e.g. 42) for reproducibility

    # Local splitting step
    "split": {
        "split_radius_um": 60.0,
        "recursive":       True,
        "recursive_depth": 3,
        "method_kwargs": {
            "clusterer": {
                "method":                  "isosplit",   # or "isosplit6" (C++ variant)
                "min_cluster_size":        10,
                "max_iterations_per_pass": 500,
                "isocut_threshold":        2.0,  # ↑ = less splitting (more conservative)
            },
            "min_size_split":  25,
            "n_pca_features":  10,    # default higher than hdbscan version
            "projection_mode": "tsvd",
        },
    },

    # Template-similarity merge (None to skip)
    "merge_from_templates": {
        "similarity_metric": "l1",
        "num_shifts":        3,
        "similarity_thresh": 0.8,
    },

    # Feature-distribution merge (None to disable)
    # enable with e.g. {"merge_radius_um": 50}
    "merge_from_features": None,

    # Remove tiny clusters after all merging
    "clean": {
        "minimum_cluster_size": 10,
    },

    "debug_folder": None,
    "verbose":      True,
}

# ── graph-clustering ──────────────────────────────────────────────
# NOTE: peaks_svd["radius_um"] must be >= graph_kwargs["neighbors_radius_um"] * 2
METHOD_KWARGS_graph_clustering = {
    "peaks_svd": {
        "n_components": 6,
        "ms_before":    0.5,
        "ms_after":     1.5,
        "radius_um":    100.0,   # must be >= neighbors_radius_um * 2
        "motion":       None,
    },

    "seed": None,

    # Graph construction
    "graph_kwargs": {
        "bin_mode":                 "channels",  # "channels" or "vertical_bins"
        "neighbors_radius_um":      50.0,
        # "bin_um": 30.0,             # used with bin_mode="vertical_bins"
        # "direction": "y",           # used with bin_mode="vertical_bins"
        "normed_distances":         True,
        "n_neighbors":              5,    # K for KNN graph
        "n_components":             10,   # local SVD dims for graph edges
        "sparse_mode":              "knn",
        "apply_local_svd":          True,
        "enforce_diagonal_to_zero": True,
    },

    # Community detection algorithm
    "clusterer": {
        "method": "sknetwork-louvain",   # pip install scikit-network
        # Alternatives:
        #   "method": "sknetwork-leiden"
        #   "method": "leidenalg"          # pip install leidenalg igraph
        #   "method": "hdbscan"            # + "min_cluster_size": 20, etc.
    },

    "debug_folder": None,
    "verbose":      True,
}

# ------------------------------------------------------------------
# SortingAnalyzer output folder
# (set to None to keep the analyzer in memory only; this is faster
#  but the analyzer is lost when the script exits)
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Phy export folder  (set to None to skip the Phy export)
# ------------------------------------------------------------------
PHY_FOLDER = os.path.join(ANALYZER_FOLDER, "export_to_phy")

# ------------------------------------------------------------------
# SortingAnalyzer computation settings
# ------------------------------------------------------------------
WAVEFORM_WINDOW = {"ms_before": 1.0, "ms_after": 2.0}
MAX_SPIKES_PER_UNIT = 1000
PC_N_COMPONENTS     = 8        # Increased from 6 to 8 (matches 32-channel target)
QUALITY_METRIC_NAMES = ["snr", "isi_violation", "presence_ratio", "amplitude_cutoff"]

# =====================================================================
#  END OF USER CONFIGURATION
# =====================================================================


# ── Map METHOD string → method_kwargs dict ────────────────────────
_METHOD_KWARGS_MAP = {
    "dummy":               METHOD_KWARGS_dummy,
    "hdbscan-positions":   METHOD_KWARGS_hdbscan_positions,
    "random-projections":  METHOD_KWARGS_random_projections,
    "iterative-hdbscan":   METHOD_KWARGS_iterative_hdbscan,
    "iterative-isosplit":  METHOD_KWARGS_iterative_isosplit,
    "graph-clustering":    METHOD_KWARGS_graph_clustering,
}


def _json_serialisable(obj):
    """
    Recursively convert a params dict to something json.dumps() can handle.
      - numpy arrays  → list  (peak_locations etc.)
      - numpy scalars → native Python scalar
      - float inf/nan → string so the value is not silently lost
      - everything else passes through unchanged (None → null, str, int, float, dict, list)
    """
    if isinstance(obj, dict):
        return {k: _json_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_serialisable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, float) and (obj != obj or abs(obj) == float("inf")):
        return str(obj)   # "nan", "inf", "-inf"
    return obj


def save_params_json(save_dir):
    """
    Write all user-configuration parameters to <save_dir>/params.json.
    Called before clustering so the file exists even if the run crashes.
    """
    os.makedirs(save_dir, exist_ok=True)

    params = {
        "timestamp":          datetime.datetime.now().isoformat(timespec="seconds"),
        "METHOD":             METHOD,
        "BIN_PATH":           BIN_PATH,
        "PEAKS_PATH":         PEAKS_PATH,
        "ANALYZER_FOLDER":    ANALYZER_FOLDER,
        "PHY_FOLDER":         PHY_FOLDER,
        "JOB_KWARGS":         JOB_KWARGS,
        "method_kwargs":      _METHOD_KWARGS_MAP[METHOD],
        "WAVEFORM_WINDOW":    WAVEFORM_WINDOW,
        "MAX_SPIKES_PER_UNIT":MAX_SPIKES_PER_UNIT,
        "PC_N_COMPONENTS":    PC_N_COMPONENTS,
        "QUALITY_METRIC_NAMES":QUALITY_METRIC_NAMES,
    }

    out_path = os.path.join(save_dir, "params.json")
    with open(out_path, "w") as f:
        json.dump(_json_serialisable(params), f, indent=2)
    print(f"[params] saved to {out_path!r}")


def load_data(file_path):
    """Load a recording from BIN_PATH (SpikeGLX or pre-processed binary)."""
    if "imec" in file_path:
        rec = si.read_spikeglx(file_path, stream_name="imec0.ap", load_sync_channel=False)
    else:
        rec = si.load(file_path, n_jobs=JOB_KWARGS["n_jobs"])
    return rec


def run_clustering(rec, peaks):
    """Cluster detected peaks with the chosen method."""
    method_kwargs = _METHOD_KWARGS_MAP[METHOD]
    print(f"\n[clustering] method = {METHOD!r}")
    print(f"[clustering] method_kwargs:\n{method_kwargs}\n")

    labels, peak_labels, extra_outputs = find_clusters_from_peaks(
        recording=rec,
        peaks=peaks,
        method=METHOD,
        method_kwargs=method_kwargs,
        extra_outputs=True,
        job_kwargs=JOB_KWARGS,
    )
    return labels, peak_labels, extra_outputs


def build_sorting(rec, peaks, peak_labels):
    """Remove noise spikes (label == -1) and build a NumpySorting object."""
    valid_mask   = peak_labels >= 0
    valid_peaks  = peaks[valid_mask]
    valid_labels = peak_labels[valid_mask]
    print(f"[sorting] kept {valid_peaks.shape[0]:,} / {peaks.shape[0]:,} spikes"
          f"  ({valid_mask.sum() / peaks.shape[0] * 100:.1f}%)")

    sorting = si.NumpySorting.from_samples_and_labels(
        samples_list=valid_peaks["sample_index"],
        labels_list=valid_labels,
        sampling_frequency=rec.get_sampling_frequency(),
    )
    print(f"[sorting] {len(sorting.unit_ids)} units")
    return sorting


def compute_analyzer(rec, sorting):
    """Create a SortingAnalyzer and compute all required extensions."""
    fmt    = "binary_folder" if ANALYZER_FOLDER is not None else "memory"
    kwargs = dict(sorting=sorting, recording=rec, sparse=True)
    if ANALYZER_FOLDER is not None:
        kwargs["folder"]    = ANALYZER_FOLDER
        kwargs["format"]    = "binary_folder"
        kwargs["n_jobs"]    = JOB_KWARGS["n_jobs"]
        kwargs["chunk_duration"] = JOB_KWARGS["chunk_duration"]
        kwargs["progress_bar"]   = True
    else:
        kwargs["format"] = "memory"

    print(f"\n[analyzer] creating SortingAnalyzer  (format={fmt!r})")
    analyzer = si.create_sorting_analyzer(**kwargs)

    # — required —
    print("[analyzer] computing random_spikes …")
    analyzer.compute("random_spikes", max_spikes_per_unit=MAX_SPIKES_PER_UNIT)

    print("[analyzer] computing waveforms …")
    analyzer.compute("waveforms", **WAVEFORM_WINDOW, **JOB_KWARGS)

    print("[analyzer] computing templates …")
    analyzer.compute("templates")

    # — recommended for Phy —
    print("[analyzer] computing noise_levels …")
    analyzer.compute("noise_levels")

    print("[analyzer] computing spike_amplitudes …")
    analyzer.compute("spike_amplitudes", **JOB_KWARGS)

    print("[analyzer] computing principal_components …")
    analyzer.compute("principal_components",
                     n_components=PC_N_COMPONENTS,
                     mode="by_channel_local",
                     **JOB_KWARGS)

    print("[analyzer] computing template_similarity …")
    analyzer.compute("template_similarity")

    # — optional quality metrics —
    print("[analyzer] computing quality_metrics …")
    analyzer.compute("quality_metrics", metric_names=QUALITY_METRIC_NAMES)

    print("[analyzer] computing template_metrics …")
    analyzer.compute("template_metrics")

    print("[analyzer] done!")
    return analyzer


def compute_radius_for_target_channels(analyzer, target_n_channels=32):
    """Find radius (µm) that encloses ~target_n_channels contacts around median unit."""
    locs = analyzer.get_channel_locations()
    dists = np.linalg.norm(locs[:, np.newaxis] - locs[np.newaxis, :], axis=2)

    min_dists = np.sort(dists, axis=1)[:, 1]
    pitch = np.median(min_dists)

    max_dist = dists.max()
    candidate_radii = np.arange(pitch, max_dist + pitch, pitch / 2)

    best_radius = pitch
    best_diff = np.inf
    for r in candidate_radii:
        counts = (dists <= r).sum(axis=1)
        median_count = np.median(counts)
        diff = abs(median_count - target_n_channels)
        if diff < best_diff:
            best_diff   = diff
            best_radius = r
        if median_count > target_n_channels * 1.5:
            break

    print(f"[phy-sparsity] Probe pitch ≈ {pitch:.1f} µm  →  "
          f"radius = {best_radius:.1f} µm  ≈ "
          f"{int(round(np.median((dists <= best_radius).sum(axis=1))))} "
          f"channels (median)")
    return best_radius


def build_radius_sparsity(analyzer, target_n_channels=32):
    """Build ChannelSparsity using radius derived from target channel count."""
    radius_um = compute_radius_for_target_channels(analyzer, target_n_channels)
    sparsity  = si.compute_sparsity(analyzer, method="radius", radius_um=radius_um)

    counts = np.array([len(v) for v in sparsity.unit_id_to_channel_indices.values()])
    print(f"[phy-sparsity] Channels per unit: min={counts.min()}  "
          f"median={int(np.median(counts))}  max={counts.max()}")
    return sparsity


def export_phy(analyzer):
    """Export the SortingAnalyzer to Phy format with proper sparsity handling.
    
    Follows the rebuild_export_to_phy.py pattern:
    - Computes sparsity based on target channel count
    - Ensures templates and PC features match sparsity
    - Properly handles spike cluster remapping for curated sortings
    - Does NOT include whitening (just identity matrix)
    """
    if PHY_FOLDER is None:
        print("\n[phy] PHY_FOLDER is None — skipping Phy export.")
        return

    from pathlib import Path
    print(f"\n[phy] exporting to {PHY_FOLDER!r} …")
    
    export_dir = Path(PHY_FOLDER)
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Build sparsity with 32 channels per unit
    print("\n[phy] Computing channel sparsity …")
    sparsity = build_radius_sparsity(analyzer, target_n_channels=32)
    
    # Get basic data
    unit_ids = analyzer.unit_ids
    sparse_dict = sparsity.unit_id_to_channel_indices
    n_channels = analyzer.get_channel_locations().shape[0]
    max_n_chans = 32  # Fixed to 32 channels for Phy waveform display
    
    print(f"\n[phy] Building Phy export with sparsity …")
    print(f"  Units: {len(unit_ids)}  |  Channels/unit: {max_n_chans} (fixed for Phy)")
    
    # Step 1: Templates and template_ind
    print(f"\n[phy-1] Extracting templates …")
    # Templates already computed in compute_analyzer()
    templates_full = analyzer.get_extension("templates").get_data()
    
    template_ind = -np.ones((len(unit_ids), max_n_chans), dtype="int64")
    n_units, n_samples, _ = templates_full.shape
    templates = np.zeros((n_units, n_samples, max_n_chans), dtype=templates_full.dtype)
    
    for unit_ind, unit_id in enumerate(unit_ids):
        sparse_channels = sparse_dict[unit_id]
        template_ind[unit_ind, :len(sparse_channels)] = sparse_channels
        templates[unit_ind, :, :len(sparse_channels)] = templates_full[unit_ind][:, sparse_channels]
    
    np.save(export_dir / "templates.npy", templates)
    np.save(export_dir / "template_ind.npy", template_ind)
    print(f"  ✓ templates: {templates.shape}")
    print(f"  ✓ template_ind: {template_ind.shape}")
    
    # Step 2: PC features
    print(f"\n[phy-2] Extracting PC features …")
    # Principal components already computed in compute_analyzer()
    pc_path = export_dir / "pc_features.npy"
    pca_ext = analyzer.get_extension("principal_components")
    pca_ext.run_for_all_spikes(pc_path, **JOB_KWARGS)
    
    pc_features = np.load(pc_path)
    # Pad to max_n_chans if needed
    if pc_features.shape[2] != max_n_chans:
        n_spikes, n_pcs_feat, _ = pc_features.shape
        pc_features_padded = np.zeros((n_spikes, n_pcs_feat, max_n_chans), dtype=pc_features.dtype)
        pc_features_padded[:, :, :pc_features.shape[2]] = pc_features
        pc_features = pc_features_padded
        np.save(pc_path, pc_features)
    print(f"  ✓ pc_features: {pc_features.shape}")
    
    # Step 3: Spike times and clusters
    print(f"\n[phy-3] Building spike times and clusters …")
    spike_times_list = []
    spike_clusters_list = []
    for unit_id in unit_ids:
        st = analyzer.sorting.get_unit_spike_train(unit_id, segment_index=0)
        spike_times_list.append(st)
        spike_clusters_list.append(np.full(len(st), unit_id, dtype='int32'))
    
    spike_times = np.concatenate(spike_times_list).astype('uint64')
    spike_clusters = np.concatenate(spike_clusters_list).astype('int32')
    sort_idx = np.argsort(spike_times, kind='stable')
    spike_times = spike_times[sort_idx]
    spike_clusters = spike_clusters[sort_idx]
    
    # Remap spike_clusters to sequential 0-N if needed
    sorted_ids = sorted([int(u) for u in unit_ids])
    if sorted_ids != list(range(len(sorted_ids))):
        print(f"  ⚠ Remapping cluster IDs: non-contiguous → sequential")
        unit_id_to_idx = {uid: idx for idx, uid in enumerate(sorted_ids)}
        spike_clusters = np.array([unit_id_to_idx.get(int(cid), 0) for cid in spike_clusters], dtype='int32')
        print(f"    Original: {sorted_ids[:10]} ... ")
        print(f"    Remapped range: 0-{spike_clusters.max()}")
    
    np.save(export_dir / "spike_times.npy", spike_times)
    np.save(export_dir / "spike_clusters.npy", spike_clusters)
    np.save(export_dir / "spike_templates.npy", spike_clusters)  # Same as spike_clusters
    print(f"  ✓ spike_times: {spike_times.shape}")
    print(f"  ✓ spike_clusters: {spike_clusters.shape}")
    
    # Step 4: Amplitudes
    print(f"\n[phy-4] Extracting amplitudes …")
    # Spike amplitudes already computed in compute_analyzer()
    amplitudes = analyzer.get_extension("spike_amplitudes").get_data().astype('float32')
    np.save(export_dir / "amplitudes.npy", amplitudes)
    print(f"  ✓ amplitudes: {amplitudes.shape}")
    
    # Step 5: Channel info
    print(f"\n[phy-5] Channel info …")
    channel_map = np.arange(n_channels, dtype='int32')
    channel_positions = analyzer.get_channel_locations()
    np.save(export_dir / "channel_map.npy", channel_map)
    np.save(export_dir / "channel_positions.npy", channel_positions)
    print(f"  ✓ channel_map: {channel_map.shape}")
    print(f"  ✓ channel_positions: {channel_positions.shape}")
    
    # Step 6: PC feature indices
    print(f"\n[phy-6] PC feature indices …")
    np.save(export_dir / "pc_feature_ind.npy", template_ind)
    print(f"  ✓ pc_feature_ind: {template_ind.shape}")
    
    # Step 7: Template similarity (no whitening)
    print(f"\n[phy-7] Template similarity …")
    # Template similarity already computed in compute_analyzer()
    similarity_ext = analyzer.get_extension("template_similarity")
    similarity_data = similarity_ext.get_data()
    if hasattr(similarity_data, 'values'):  # pandas DataFrame
        similarity_index = similarity_data.values.astype('float32')
    else:
        similarity_index = similarity_data.astype('float32')
    np.save(export_dir / "similar_templates.npy", similarity_index)
    print(f"  ✓ similar_templates: {similarity_index.shape}")
    
    # Step 8: Contamination index
    print(f"\n[phy-8] Contamination index …")
    # Quality metrics already computed in compute_analyzer()
    quality_ext = analyzer.get_extension("quality_metrics")
    metrics = quality_ext.get_data()
    if "contamination" in metrics.columns:
        contamination_index = metrics["contamination"].values.astype('float32')
    elif "isi_violation" in metrics.columns:
        contamination_index = metrics["isi_violation"].values.astype('float32')
    else:
        contamination_index = np.zeros(len(unit_ids), dtype='float32')
    np.save(export_dir / "contamination_index.npy", contamination_index)
    print(f"  ✓ contamination_index: {contamination_index.shape}")
    
    print(f"\n[phy] ✓ Export complete!  Open with:")
    print(f"    phy template-gui {export_dir / 'params.py'}")


# ── main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    if METHOD not in _METHOD_KWARGS_MAP:
        raise ValueError(
            f"Unknown METHOD={METHOD!r}. "
            f"Choose one of: {list(_METHOD_KWARGS_MAP)}"
        )

    # 1. Load data
    print(f"[data] loading recording from {BIN_PATH!r}")
    rec = load_data(BIN_PATH)
    print(rec)

    print(f"[data] loading peaks from {PEAKS_PATH!r}")
    detected_peaks = np.load(PEAKS_PATH)
    # Sort by sample index — required before passing to find_clusters_from_peaks
    detected_peaks = np.sort(detected_peaks, order="sample_index")
    print(f"[data] {detected_peaks.shape[0]:,} peaks loaded")

    # 2. Cluster
    labels, peak_labels, extra_outputs = run_clustering(rec, detected_peaks)
    print(f"[clustering] labels : {labels}")
    print(f"[clustering] peak_labels shape: {peak_labels.shape}")

    # 3. Build sorting
    sorting = build_sorting(rec, detected_peaks, peak_labels)

    # 4. SortingAnalyzer + extensions
    analyzer = compute_analyzer(rec, sorting)

    # 5. Phy export
    export_phy(analyzer)

    # 6. Save parameters
    _save_dir = ANALYZER_FOLDER if ANALYZER_FOLDER is not None else BIN_PATH
    save_params_json(_save_dir)