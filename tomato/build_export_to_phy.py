#!/usr/bin/env python3
"""
build_export_to_phy.py
=======================
Rebuild the entire export_to_phy folder with consistent sparsity.

This recomputes ALL files that need to match the sparsity setting:
- templates.npy (units × samples × channels_sparse)
- template_ind.npy (units × channels_sparse) - channel indices for each unit
- pc_features.npy (spikes × pcs × channels_sparse)
- pc_feature_ind.npy (units × channels_sparse) - same as template_ind
- amplitudes.npy (spikes,) - extracted at the best channels
- whitening_mat_inv.npy (channels_sparse × channels_sparse) - sparsified inverse whitening

This ensures phy sees consistent channel dimensions across all views.

Run this first after try_tomato.py (or merge_tomato_results.py, if the
recording was chunked) to turn its NumpySorting into a phy-openable folder.
try_merge_selfdbuild_cleaned.py / try_merge_step_2.py then operate on the
export_to_phy folder this script produces.

USAGE:
  python build_export_to_phy.py [--sorting_path P] [--rec_path P] [--export_dir P]
                                 [--target_n_channels N] [--n_pcs N]
                                 [--ms_before F] [--ms_after F]
  All args are optional and default to the paths/values hardcoded in the
  "USER CONFIGURATION" section below — edit those for routine use, or pass
  flags to override per-run without editing the file.

PARAMETERS: see USER CONFIGURATION section below for the full list
(TARGET_N_CHANNELS, N_PCS, MS_BEFORE/MS_AFTER, MAX_SPIKES_PER_UNIT,
MAX_SPIKES_FOR_TEMPLATE, JOB_KWARGS, SKIP_PC_FEATURES), each commented in place.

OUTPUTS, all in EXPORT_TO_PHY_DIR (rebuilt from scratch each run):
  templates.npy, template_ind.npy          -- per-unit mean waveforms + sparse channel indices
  pc_features.npy, pc_feature_ind.npy      -- per-spike PCA features (skipped if SKIP_PC_FEATURES)
  amplitudes.npy                            -- per-spike amplitude at best channel
  whitening_mat_inv.npy                     -- sparsified inverse whitening matrix
  spike_times.npy, spike_clusters.npy, spike_templates.npy, cluster_group.tsv
                                             -- copied/derived from SORTING_PATH

### things to modify:
- results are not saved in _sparse_analyzer, only exported to phy
- the 'good' labelling is very crude right now
- the contamination index is wrong (all zeros for now)

=======================================================================
CONFIGURE EVERYTHING IN THE "USER CONFIGURATION" SECTION BELOW
=======================================================================
"""

import os
import shutil
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import spikeinterface.full as si

# =====================================================================
#  USER CONFIGURATION
# =====================================================================

# Path to the recording that was sorted
RECORDING_PATH = (
    "/nexus/posix0/MBR-neuralsystems/ephys/GLC7189/GLC7189_c_g0/preprocessed"
)

# Path to SpikeInterface sorting (contains the spike times/clusters) --
# matches the tomato_sorting2_s{start}_e{end} folder try_tomato.py writes
# (see run_try_tomato.sh; with no SAMPLE_START/SAMPLE_END override, that's
# the full recording, samples 0..86545524).
SORTING_PATH = (
    RECORDING_PATH + "/tomato_sorting2_s0_e86545524"
)

# Path to export_to_phy folder (will be rebuilt)
EXPORT_TO_PHY_DIR = (
    SORTING_PATH + "/export_to_phy"
)

# Target number of channels per unit neighbourhood for PC/template computation.
# The radius is auto-computed from probe geometry to enclose approximately this many channels.
# Use 32 for good detail, 64 for more context (recommended for 355-channel probes)
TARGET_N_CHANNELS = 32

# Number of PCA components per channel
N_PCS = 3

# Waveform window
MS_BEFORE = 1.0
MS_AFTER = 2.0

# Max random spikes per unit for PCA training
MAX_SPIKES_PER_UNIT = 500

# Max spikes per unit for the minimal temp analyzer used ONLY to derive sparsity
# (small value avoids OOM; templates only need a rough estimate of peak channel)
MAX_SPIKES_FOR_TEMPLATE = 20

# Parallel job settings
JOB_KWARGS = dict(n_jobs=12, chunk_duration="1s", progress_bar=True)

# Skip PC feature computation entirely — phy feature view will be blank but everything
# else (templates, amplitudes, waveforms) works fine.  Set True to skip the slow step.
SKIP_PC_FEATURES = True

# =====================================================================
#  END OF USER CONFIGURATION
# =====================================================================


def load_recording(path):
    if "imec" in str(path):
        return si.read_spikeglx(path, stream_name="imec0.ap", load_sync_channel=False)
    return si.load(path)


def load_sorting_extractor(sorting_path):
    """Load sorting from SpikeInterface extractor."""
    sort = si.load(sorting_path)
    print(f"  Loaded {len(sort.unit_ids)} units")
    return sort


def remap_spike_clusters_to_sequential(spike_clusters, spike_templates, unit_ids):
    """Remap spike cluster IDs from non-contiguous to sequential 0-N if needed."""
    sorted_ids = sorted(unit_ids)
    if sorted_ids == list(range(len(sorted_ids))):
        return spike_clusters, spike_templates  # Already sequential
    
    unit_id_to_idx = {uid: idx for idx, uid in enumerate(sorted_ids)}
    print(f"  Remapping non-contiguous cluster IDs → sequential 0-{len(unit_ids)-1}")
    return (np.array([unit_id_to_idx.get(c, 0) for c in spike_clusters], dtype=spike_clusters.dtype),
            np.array([unit_id_to_idx.get(t, 0) for t in spike_templates], dtype=spike_templates.dtype))


def compute_radius_for_target_channels(analyzer, target_n_channels):
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

    print(f"  Probe pitch ≈ {pitch:.1f} µm  →  "
          f"radius = {best_radius:.1f} µm  ≈ "
          f"{int(round(np.median((dists <= best_radius).sum(axis=1))))} "
          f"channels (median)")
    return best_radius


def build_radius_sparsity(analyzer, target_n_channels):
    """Build ChannelSparsity using radius derived from target channel count."""
    radius_um = compute_radius_for_target_channels(analyzer, target_n_channels)
    sparsity  = si.compute_sparsity(analyzer, method="radius", radius_um=radius_um)

    counts = np.array([len(v) for v in sparsity.unit_id_to_channel_indices.values()])
    print(f"  Channels per unit: min={counts.min()}  "
          f"median={int(np.median(counts))}  max={counts.max()}")
    return sparsity


def rebuild_export_to_phy(export_dir, analyzer_sparse, sparsity, n_pcs, recording, sorting):
    """Rebuild export_to_phy folder, computing only MISSING files."""
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    unit_ids = analyzer_sparse.unit_ids
    sparse_dict = sparsity.unit_id_to_channel_indices
    n_channels = recording.get_num_channels()
    max_n_chans = max(len(v) for v in sparse_dict.values())
    
    print(f"\n{'='*70}")
    print(f"REBUILDING EXPORT_TO_PHY")
    print(f"{'='*70}\n")
    
    print(f"\n[1/6] Computing templates & template_ind (if missing) …")
    templates_path = export_dir / "templates.npy"
    template_ind_path = export_dir / "template_ind.npy"

    # Always ensure extension is in analyzer
    if not analyzer_sparse.has_extension("templates"):
        print(f"  Computing templates …")
        analyzer_sparse.compute("templates", save=True)

    if templates_path.exists() and template_ind_path.exists():
        templates = np.load(templates_path)
        template_ind = np.load(template_ind_path)
        print(f"  ✓ REUSING existing templates: {templates.shape}")
    else:
        templates_full = analyzer_sparse.get_extension("templates").get_data()
        template_ind = -np.ones((len(unit_ids), max_n_chans), dtype="int64")
        n_units, n_samples, _ = templates_full.shape
        templates = np.zeros((n_units, n_samples, max_n_chans), dtype=templates_full.dtype)
        for unit_ind, unit_id in enumerate(unit_ids):
            sparse_channels = sparse_dict[unit_id]
            template_ind[unit_ind, :len(sparse_channels)] = sparse_channels
            templates[unit_ind, :, :len(sparse_channels)] = templates_full[unit_ind][:, sparse_channels]
        np.save(templates_path, templates)
        np.save(template_ind_path, template_ind)
        print(f"  ✓ Saved templates: {templates.shape}, template_ind: {template_ind.shape}")

    print(f"\n[2/6] Computing PC features (if missing) …")
    pc_path = export_dir / "pc_features.npy"

    if SKIP_PC_FEATURES:
        if pc_path.exists():
            pc_path.unlink()
            print(f"  SKIP_PC_FEATURES=True: removed stale pc_features.npy")
        pc_features = None
        print(f"  Skipped (SKIP_PC_FEATURES=True — phy hides PC view, everything else works)")

    else:
        if pc_path.exists():
            # phy file already there — skip the slow PCA computation entirely
            # (the principal_components extension won't be in the analyzer, but phy doesn't need it)
            pc_features = np.load(pc_path)
            print(f"  ✓ REUSING existing PC features: {pc_features.shape}")
        else:
            if not analyzer_sparse.has_extension("principal_components"):
                print(f"  Computing principal_components …")
                analyzer_sparse.compute("principal_components",
                                        n_components=n_pcs,
                                        mode="by_channel_local",
                                        save=True,
                                        **JOB_KWARGS)
            print(f"  Exporting PC features for all spikes …")
            pca_ext = analyzer_sparse.get_extension("principal_components")
            pca_ext.run_for_all_spikes(pc_path, **JOB_KWARGS)
            pc_features = np.load(pc_path)
            if pc_features.shape[2] != max_n_chans:
                print(f"    Padding from {pc_features.shape[2]} → {max_n_chans} channels")
                n_spikes_t, n_pcs_t, _ = pc_features.shape
                pc_features_padded = np.zeros((n_spikes_t, n_pcs_t, max_n_chans), dtype=pc_features.dtype)
                pc_features_padded[:, :, :pc_features.shape[2]] = pc_features
                pc_features = pc_features_padded
                np.save(pc_path, pc_features)
            print(f"  ✓ Saved PC features: {pc_features.shape}")

    print(f"\n[3/6] Spike info + channel info + other files …")
    
    # Spike info (preserve from original if exists, else create)
    spike_times_path = export_dir / "spike_times.npy"
    spike_clusters_path = export_dir / "spike_clusters.npy"
    
    if spike_times_path.exists() and spike_clusters_path.exists():
        spike_times = np.load(spike_times_path).ravel()
        spike_clusters = np.load(spike_clusters_path).ravel()
        print(f"  ✓ REUSING spike_times and spike_clusters")
    else:
        print(f"  Deriving spike_times and spike_clusters from sorting …")
        unit_ids = sorting.unit_ids
        spike_times_list = []
        spike_clusters_list = []
        for unit_id in unit_ids:
            st = sorting.get_unit_spike_train(unit_id, segment_index=0)
            spike_times_list.append(st)
            spike_clusters_list.append(np.full(len(st), unit_id, dtype='int32'))
        spike_times   = np.concatenate(spike_times_list).astype('uint64')
        spike_clusters = np.concatenate(spike_clusters_list).astype('int32')
        sort_idx      = np.argsort(spike_times, kind='stable')
        spike_times   = spike_times[sort_idx]
        spike_clusters = spike_clusters[sort_idx]
        np.save(spike_times_path, spike_times)
        np.save(spike_clusters_path, spike_clusters)
        print(f"  ✓ Saved spike_times: {spike_times.shape},  spike_clusters: {spike_clusters.shape}")
    
    amplitudes_path = export_dir / "amplitudes.npy"
    # Always ensure extensions are in analyzer
    if not analyzer_sparse.has_extension("templates"):
        analyzer_sparse.compute("templates", save=True)
    if not analyzer_sparse.has_extension("noise_levels"):
        analyzer_sparse.compute("noise_levels", save=True)
    if not analyzer_sparse.has_extension("spike_amplitudes"):
        print(f"  Computing spike_amplitudes …")
        analyzer_sparse.compute("spike_amplitudes", save=True)
    if amplitudes_path.exists():
        amplitudes = np.load(amplitudes_path)
        print(f"  ✓ REUSING amplitudes: {amplitudes.shape}")
    else:
        amplitudes = analyzer_sparse.get_extension("spike_amplitudes").get_data().astype('float32')
        np.save(amplitudes_path, amplitudes)
        print(f"  ✓ Saved amplitudes: {amplitudes.shape}")
    
    # Channel info (always compute — these are static)
    channel_map = np.arange(n_channels, dtype='int32')
    np.save(export_dir / "channel_map.npy", channel_map)
    channel_positions = recording.get_channel_locations()
    np.save(export_dir / "channel_positions.npy", channel_positions)
    print(f"  ✓ Saved channel_map: {channel_map.shape}, channel_positions: {channel_positions.shape}")
    
    # params.py — write only if missing
    params_path = export_dir / "params.py"
    if params_path.exists():
        print(f"  ✓ REUSING params.py")
    else:
        try:
            # Works for BinaryRecordingExtractor and SpikeGLX
            file_paths = recording._kwargs.get("file_paths") or recording._kwargs.get("file_path")
            if file_paths is None:
                raise KeyError
            if not isinstance(file_paths, (list, tuple)):
                file_paths = [file_paths]
            dat_path_str = repr(str(file_paths[0]).replace("\\", "/"))
        except Exception:
            dat_path_str = repr(["<unknown — edit params.py manually>"])
        dtype_str = str(recording.get_dtype())
        params_content = (
            f"n_channels_dat = {recording.get_num_channels()}\n"
            f"offset = 0\n"
            f"sample_rate = {recording.get_sampling_frequency()}\n"
            f"dtype = '{dtype_str}'\n"
            f"hp_filtered = False\n"
            f"dat_path = {dat_path_str}\n"
        )
        params_path.write_text(params_content)
        print(f"  ✓ Saved params.py  (dtype={dtype_str}, n_chan={recording.get_num_channels()})")

    # Spike templates (same as spike_clusters for KS4)
    spike_templates_path = export_dir / "spike_templates.npy"
    if not spike_templates_path.exists():
        spike_templates = spike_clusters.copy()  # #CHANGED - Assign the variable, don't just save
        np.save(spike_templates_path, spike_templates)
        print(f"  ✓ Saved spike_templates: {spike_templates.shape}")
    else:
        spike_templates = np.load(spike_templates_path)
        print(f"  ✓ REUSING spike_templates: {spike_templates.shape}")
    
    # ──────────────────────────────────────────────────────────────────────
    # CRITICAL: Remap spike_clusters/spike_templates to sequential 0-N
    # ──────────────────────────────────────────────────────────────────────
    # If sorting was curated, unit_ids are non-contiguous (e.g., [0, 8, 23, 152, ...])
    # But templates are always 0-N. Remap to avoid phy index errors.
    spike_clusters, spike_templates = remap_spike_clusters_to_sequential(
        spike_clusters, spike_templates, unit_ids
    )
    
    # Save remapped versions (overwrite if they were non-contiguous)
    np.save(spike_clusters_path, spike_clusters)
    np.save(spike_templates_path, spike_templates)

    # Whitening matrix — delete if present (identity matrix causes phy template-view crashes)
    whitening_path = export_dir / "whitening_mat_inv.npy"
    if whitening_path.exists():
        whitening_path.unlink()
        print(f"  ✓ Deleted whitening_mat_inv.npy (not needed)")
    
    print(f"\n[4/6] Computing template similarity (if missing) …")
    similarity_path = export_dir / "similarity_index.npy"
    # Always ensure extension is in analyzer
    if not analyzer_sparse.has_extension("template_similarity"):
        print(f"  Computing template_similarity …")
        try:
            analyzer_sparse.compute("template_similarity", save=True)
        except Exception as e:
            print(f"  ⚠ template_similarity failed: {e}")
    if similarity_path.exists():
        similarity_index = np.load(similarity_path)
        print(f"  ✓ REUSING existing similarity_index: {similarity_index.shape}")
    else:
        try:
            similarity_ext = analyzer_sparse.get_extension("template_similarity")
            similarity_index = similarity_ext.get_data()
            if hasattr(similarity_index, 'values'):
                similarity_index = similarity_index.values.astype('float32')
            else:
                similarity_index = similarity_index.astype('float32')
            np.save(similarity_path, similarity_index)
            print(f"  ✓ Saved similarity_index: {similarity_index.shape}")
        except Exception as e:
            print(f"  ⚠ Failed: {e} → creating identity matrix")
            similarity_index = np.eye(len(unit_ids), dtype='float32')
            np.save(similarity_path, similarity_index)
    
    # Save as similar_templates too (phy uses this for visualization)
    np.save(export_dir / "similar_templates.npy", similarity_index)
    
    # Save pc_feature_ind (same as template_ind)
    pc_feature_ind_path = export_dir / "pc_feature_ind.npy"
    if not pc_feature_ind_path.exists():
        np.save(pc_feature_ind_path, template_ind)
        print(f"  ✓ Saved pc_feature_ind: {template_ind.shape}")
    else:
        print(f"  ✓ REUSING pc_feature_ind")

    print(f"\n[5/6] Computing quality metrics …")
    contamination_path = export_dir / "contamination_index.npy"
    contam_tsv_path = export_dir / "cluster_ContamPct.tsv"
    firing_rate_path = export_dir / "cluster_firing_rate.tsv"
    group_tsv_path = export_dir / "cluster_group.tsv"

    fs = recording.get_sampling_frequency()

    if not contamination_path.exists():
        print(f"  Computing Kilosort contamination metric (R12 / ContamPct) …")
        from kilosort.CCG import refract
        st_sec = spike_times.astype(np.float64) / fs
        _, R12_contamination = refract(
            spike_clusters, st_sec, acg_threshold=0.2, ccg_threshold=0.25
        )
        contamination_index = R12_contamination.astype('float32')
        np.save(contamination_path, contamination_index)
        pd.DataFrame({"cluster_id": np.arange(len(unit_ids)), "ContamPct": contamination_index}).to_csv(
            contam_tsv_path, sep="\t", index=False)
        print(f"  ✓ Saved contamination_index + cluster_ContamPct.tsv  "
              f"(median={float(np.median(contamination_index)):.4f})")
    else:
        contamination_index = np.load(contamination_path)
        print(f"  ✓ REUSING contamination_index (ContamPct)")

    if not firing_rate_path.exists():
        duration_sec = recording.get_duration()
        firing_rates = np.array([
            np.sum(spike_clusters == i) / duration_sec
            for i in range(len(unit_ids))
        ]).astype('float32')
        pd.DataFrame({"cluster_id": np.arange(len(unit_ids)), "firing_rate": firing_rates}).to_csv(
            firing_rate_path, sep="\t", index=False)
        print(f"  ✓ Saved cluster_firing_rate.tsv  "
              f"(mean={float(np.mean(firing_rates)):.2f} Hz)")
    else:
        print(f"  ✓ REUSING cluster_firing_rate.tsv")

    # cluster_group.tsv — start all unsorted for manual curation in phy
    pd.DataFrame({"cluster_id": np.arange(len(unit_ids)), "group": ""}).to_csv(
        group_tsv_path, sep="\t", index=False)
    print(f"  ✓ Saved cluster_group.tsv (all unsorted)")

    # Final step: all extensions are auto-saved via binary_folder format
    print(f"\n[6/6] Rebuild complete with {max_n_chans}-channel sparsity:")
    print(f"  - {len(unit_ids)} units with {len(spike_times):,} spikes")
    print(f"  - templates: {templates.shape}  (sparsified from 355 → {max_n_chans} channels)")
    print(f"  - template_ind: {template_ind.shape}")
    print(f"  - pc_features: {pc_features.shape if pc_features is not None else 'skipped'}")
    print(f"  - pc_feature_ind: {template_ind.shape}")
    print(f"  - amplitudes: {amplitudes.shape}")
    print(f"  - similarity_index: {similarity_index.shape}")
    print(f"  - contamination_index: {contamination_index.shape}")
    print(f"  - cluster_ContamPct.tsv, cluster_firing_rate.tsv, cluster_group.tsv")
    print(f"✓ All dimensions now match! Ready for phy2 template-gui")
    return {
        'templates': templates.shape,
        'template_ind': template_ind.shape,
        'pc_features': pc_features.shape if pc_features is not None else 'skipped',
        'pc_feature_ind': template_ind.shape,
        'amplitudes': amplitudes.shape,
        'similarity_index': similarity_index.shape,
        'contamination_index': contamination_index.shape,
        'cluster_ContamPct.tsv': contam_tsv_path.name,
        'cluster_firing_rate.tsv': firing_rate_path.name,
        'cluster_group.tsv': group_tsv_path.name,
        'spike_times': spike_times.shape,
        'spike_clusters': spike_clusters.shape,
    }


def main():
    parser = argparse.ArgumentParser(description="Rebuild export_to_phy with consistent sparsity")
    parser.add_argument("--sorting_path",     default=SORTING_PATH)
    parser.add_argument("--rec_path",         default=RECORDING_PATH)
    parser.add_argument("--export_dir",       default=EXPORT_TO_PHY_DIR)
    parser.add_argument("--target_n_channels", type=int, default=TARGET_N_CHANNELS)
    parser.add_argument("--n_pcs",            type=int, default=N_PCS)
    parser.add_argument("--ms_before",        type=float, default=MS_BEFORE)
    parser.add_argument("--ms_after",         type=float, default=MS_AFTER)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"REBUILDING EXPORT_TO_PHY WITH CONSISTENT {args.target_n_channels}-CHANNEL SPARSITY")
    print(f"{'='*70}")

    print(f"\n[1/4] Loading recording from {args.rec_path!r}")
    rec = load_recording(args.rec_path)
    print(f"  {rec.get_num_channels()} channels  "
          f"{rec.get_sampling_frequency()} Hz  "
          f"{rec.get_duration():.1f}s")

    print(f"\n[2/4] Loading sorting from {args.sorting_path!r}")
    sorting = load_sorting_extractor(args.sorting_path)

    sparse_dir = os.path.join(args.export_dir, '_sparse_analyzer')
    analyzer_sparse = None

    # Try to load existing sparse analyzer — skips expensive waveform extraction
    if os.path.exists(sparse_dir):
        try:
            analyzer_sparse = si.load_sorting_analyzer(sparse_dir)
            sparsity = analyzer_sparse.sparsity
            print(f"\n[2/4] Loaded existing sparse analyzer (skipping rebuild) …")
            if not analyzer_sparse.has_extension("waveforms"):
                print(f"  Waveforms missing — computing …")
                analyzer_sparse.compute("random_spikes", max_spikes_per_unit=MAX_SPIKES_PER_UNIT, save=True)
                analyzer_sparse.compute("waveforms",
                                        ms_before=args.ms_before,
                                        ms_after=args.ms_after,
                                        save=True,
                                        **JOB_KWARGS)
            print(f"  ✓ {len(analyzer_sparse.unit_ids)} units, waveforms present")
        except Exception as e:
            print(f"  ⚠ Failed to load ({e}), rebuilding from scratch …")
            shutil.rmtree(sparse_dir, ignore_errors=True)
            analyzer_sparse = None

    if analyzer_sparse is None:
        temp_dir = os.path.join(args.export_dir, '_temp_analyzer')

        print(f"\n[2/4] Building minimal temp analyzer for sparsity derivation …")
        print(f"  (dense, {MAX_SPIKES_FOR_TEMPLATE} spikes/unit, binary_folder → avoids shared-memory OOM)")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        analyzer_temp = si.create_sorting_analyzer(
            sorting=sorting,
            recording=rec,
            format="binary_folder",
            folder=temp_dir,
            overwrite=True,
            sparse=False,
        )
        analyzer_temp.compute("random_spikes", max_spikes_per_unit=MAX_SPIKES_FOR_TEMPLATE)
        analyzer_temp.compute("waveforms",
                              ms_before=args.ms_before,
                              ms_after=args.ms_after,
                              **JOB_KWARGS)
        analyzer_temp.compute("templates")

        print(f"\n[3/4] Deriving {args.target_n_channels}-channel sparsity …")
        sparsity = build_radius_sparsity(analyzer_temp, args.target_n_channels)
        del analyzer_temp
        shutil.rmtree(temp_dir, ignore_errors=True)

        print(f"\n[3/4] Building sparse SortingAnalyzer (binary_folder, {MAX_SPIKES_PER_UNIT} spikes/unit) …")
        analyzer_sparse = si.create_sorting_analyzer(
            sorting=sorting,
            recording=rec,
            format="binary_folder",
            folder=sparse_dir,
            overwrite=True,
            sparse=True,
            sparsity=sparsity,
        )
        analyzer_sparse.compute("random_spikes", max_spikes_per_unit=MAX_SPIKES_PER_UNIT)
        analyzer_sparse.compute("waveforms",
                                ms_before=args.ms_before,
                                ms_after=args.ms_after,
                                **JOB_KWARGS)

    print(f"\n[4/4] Rebuilding export_to_phy folder …")
    files_created = rebuild_export_to_phy(
        args.export_dir,
        analyzer_sparse,
        sparsity,
        args.n_pcs,
        rec,
        sorting
    )

    print(f"  Sparse analyzer saved at {sparse_dir!r}")

    print(f"\n{'='*70}")
    print(f"SUMMARY OF FILES CREATED/UPDATED IN {args.export_dir}:")
    print(f"{'='*70}")
    for fname, shape in files_created.items():
        print(f"  ✓ {fname:30s}  {str(shape):40s}")
    print(f"\nAll files now use consistent {args.target_n_channels}-channel sparsity!")
    print(f"phy should now load quickly without mismatch errors.")


if __name__ == "__main__":
    main()
