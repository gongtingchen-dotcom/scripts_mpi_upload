#!/usr/bin/env python3
"""
recompute_ks4_features.py
=========================
Recompute ONLY PC features (pc_features.npy and pc_feature_ind.npy) for an
existing Kilosort 4 output folder, using a physically-based channel neighbourhood.

The radius is auto-computed from the probe geometry so that it encloses
roughly TARGET_N_CHANNELS contacts around the best channel for the median unit.

Existing KS4 PC files are renamed to pc_features_old.npy and pc_feature_ind_old.npy,
then replaced with newly computed ones (in-place, no new folder created).

=======================================================================
CONFIGURE EVERYTHING IN THE "USER CONFIGURATION" SECTION BELOW
=======================================================================
"""

import os
import shutil
import argparse
from pathlib import Path

import numpy as np
import spikeinterface.full as si

# =====================================================================
#  USER CONFIGURATION
# =====================================================================

# Path to the KS4 sorter_output folder (contains spike_times.npy etc.)
KS4_OUTPUT_DIR = (
    "/nexus/posix0/MBR-neuralsystems/ephys/GLC6962/GLC6962_q_g0_r_g0_s_g0/KS4/nt0m20_car0_spc1_jobs-1_thrd4/sorter_output"
)

# Path to the recording that was sorted (SpikeGLX dir or SI folder).
RECORDING_PATH = (
    "/nexus/posix0/MBR-neuralsystems/ephys/GLC6962/GLC6962_q_g0_r_g0_s_g0/preprocessed"
)

# Target number of channels per unit neighbourhood.
# The radius is auto-computed from your probe geometry so that it
# encloses approximately this many channels around the best channel.
# Your spikes can span ~80 channels, so 64 is a reasonable target.
TARGET_N_CHANNELS = 64

# Number of PCA components per channel.
N_PCS = 12

# Waveform window used to compute PCs.
MS_BEFORE = 1.0
MS_AFTER  = 2.0

# Number of random spikes per unit used to fit PCA models.
MAX_SPIKES_PER_UNIT = 500

# Parallel job settings.
JOB_KWARGS = dict(n_jobs=8, chunk_duration="1s", progress_bar=True)

# =====================================================================
#  END OF USER CONFIGURATION
# =====================================================================


def load_recording(path):
    if "imec" in str(path):
        return si.read_spikeglx(path, stream_name="imec0.ap", load_sync_channel=False)
    return si.load(path)


def load_ks4_sorting(ks4_dir, recording):
    """
    Read spike_times.npy and spike_clusters.npy from a KS4 output folder
    and build a NumpySorting.
    """
    ks4_dir = Path(ks4_dir)
    spike_times    = np.load(ks4_dir / "spike_times.npy").ravel().astype(np.int64)
    spike_clusters = np.load(ks4_dir / "spike_clusters.npy").ravel().astype(np.int32)
    fs = recording.get_sampling_frequency()

    sorting = si.NumpySorting.from_samples_and_labels(
        samples_list=spike_times,
        labels_list=spike_clusters,
        sampling_frequency=fs,
    )
    print(f"  Loaded {len(spike_times):,} spikes  →  {len(sorting.unit_ids)} units")
    return sorting


def compute_radius_for_target_channels(analyzer, target_n_channels):
    """
    Find the radius (µm) that, when used with from_radius sparsity,
    encloses approximately target_n_channels contacts around the median
    unit's best channel.

    Strategy: compute pairwise distances between all channels, then for
    each channel count how many neighbours fall within candidate radii.
    Pick the radius where the median channel count across all channels
    is closest to target_n_channels.
    """
    locs = analyzer.get_channel_locations()   # (n_chans, 2)
    # Pairwise Euclidean distances (µm)
    dists = np.linalg.norm(locs[:, np.newaxis] - locs[np.newaxis, :], axis=2)

    # Probe pitch = median of the smallest non-zero inter-channel distance
    min_dists = np.sort(dists, axis=1)[:, 1]   # nearest neighbour per channel
    pitch = float(np.median(min_dists))

    # Search radii from 0 to cover the whole probe
    max_dist = dists.max()
    candidate_radii = np.arange(pitch, max_dist + pitch, pitch / 2)

    best_radius = pitch
    best_diff   = np.inf
    for r in candidate_radii:
        counts = (dists <= r).sum(axis=1)       # channels within radius, per channel
        median_count = float(np.median(counts))
        diff = abs(median_count - target_n_channels)
        if diff < best_diff:
            best_diff   = diff
            best_radius = r
        if median_count > target_n_channels * 1.5:
            break   # no point searching further

    print(f"  Probe pitch ≈ {pitch:.1f} µm  →  "
          f"radius = {best_radius:.1f} µm  ≈ "
          f"{int(round(float(np.median((dists <= best_radius).sum(axis=1)))))} "
          f"channels (median)")
    return float(best_radius)


def build_radius_sparsity(analyzer, target_n_channels):
    """
    Build a ChannelSparsity using a physically-derived radius chosen to
    enclose ~target_n_channels contacts around the best channel of each
    unit.  Uses SI's compute_sparsity with method='radius'.
    """
    radius_um = compute_radius_for_target_channels(analyzer, target_n_channels)
    sparsity  = si.compute_sparsity(analyzer, method="radius", radius_um=radius_um)

    counts = np.array([len(v) for v in sparsity.unit_id_to_channel_indices.values()])
    print(f"  Channels per unit: min={counts.min()}  "
          f"median={int(np.median(counts))}  max={counts.max()}")
    return sparsity


def recompute_and_save_pc_features(ks4_dir, analyzer_sparse, sparsity, n_pcs):
    """
    Recompute PC features and save them to the KS4 folder, replacing any
    existing files and backing up the old ones.
    """
    ks4_dir = Path(ks4_dir)
    unit_ids = analyzer_sparse.unit_ids

    # Backup existing PC files
    pc_features_old_path = ks4_dir / "pc_features_old.npy"
    pc_feature_ind_old_path = ks4_dir / "pc_feature_ind_old.npy"

    if (ks4_dir / "pc_features.npy").exists():
        print(f"  Backing up pc_features.npy → pc_features_old.npy")
        shutil.copy2(ks4_dir / "pc_features.npy", pc_features_old_path)

    if (ks4_dir / "pc_feature_ind.npy").exists():
        print(f"  Backing up pc_feature_ind.npy → pc_feature_ind_old.npy")
        shutil.copy2(ks4_dir / "pc_feature_ind.npy", pc_feature_ind_old_path)

    # ── PC features ───────────────────────────────────────────────────
    # run_for_all_spikes writes shape (n_spikes, n_pcs, n_chans_sparse).
    pc_path = ks4_dir / "pc_features.npy"
    print(f"  Computing PC features with {n_pcs} components (by_channel_local mode) …")
    pca_ext = analyzer_sparse.get_extension("principal_components")
    pca_ext.run_for_all_spikes(pc_path, **JOB_KWARGS)

    # pc_feature_ind: (n_units, n_chans_sparse) — same sparsity as PCA
    sparse_dict = sparsity.unit_id_to_channel_indices
    max_n_chans_pc = max(len(v) for v in sparse_dict.values())
    pc_feature_ind = -np.ones((len(unit_ids), max_n_chans_pc), dtype="int64")
    for unit_ind, unit_id in enumerate(unit_ids):
        chan_inds = sparse_dict[unit_id]
        pc_feature_ind[unit_ind, : len(chan_inds)] = chan_inds

    pc_feature_ind_path = ks4_dir / "pc_feature_ind.npy"
    np.save(pc_feature_ind_path, pc_feature_ind)

    pc_shape = np.load(pc_path).shape
    print(f"  ✓ Saved pc_features: {pc_shape}  pc_feature_ind: {pc_feature_ind.shape}")


def main():
    parser = argparse.ArgumentParser(description="Recompute KS4 PC features only")
    parser.add_argument("--ks4_dir",          default=KS4_OUTPUT_DIR)
    parser.add_argument("--rec_path",          default=RECORDING_PATH)
    parser.add_argument("--target_n_channels", type=int,   default=TARGET_N_CHANNELS,
                        help="Target number of channels per unit neighbourhood "
                             "(radius auto-derived from probe geometry).")
    parser.add_argument("--n_pcs",             type=int,   default=N_PCS)
    parser.add_argument("--ms_before",         type=float, default=MS_BEFORE)
    parser.add_argument("--ms_after",          type=float, default=MS_AFTER)
    args = parser.parse_args()

    ks4_dir = Path(args.ks4_dir)

    print(f"\n[1/5] Loading recording from {args.rec_path!r}")
    rec = load_recording(args.rec_path)
    print(f"  {rec.get_num_channels()} channels  "
          f"{rec.get_sampling_frequency()} Hz  "
          f"{rec.get_duration():.1f}s")

    print(f"\n[2/5] Loading KS4 sorting from {ks4_dir!r}")
    sorting = load_ks4_sorting(ks4_dir, rec)

    print(f"\n[3/5] Building initial SortingAnalyzer (dense, for sparsity derivation) …")
    analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=rec,
        format="memory",
        sparse=False,
    )

    print(f"\n[4/5] Computing templates (needed for sparsity + best channel) …")
    analyzer.compute("random_spikes", max_spikes_per_unit=MAX_SPIKES_PER_UNIT)
    analyzer.compute("waveforms",
                     ms_before=args.ms_before,
                     ms_after=args.ms_after,
                     **JOB_KWARGS)
    analyzer.compute("templates")

    print(f"\n[4b/5] Deriving radius sparsity targeting ~{args.target_n_channels} channels …")
    sparsity = build_radius_sparsity(analyzer, args.target_n_channels)

    print(f"\n[4c/5] Rebuilding sparse SortingAnalyzer with radius neighbourhood …")
    analyzer_sparse = si.create_sorting_analyzer(
        sorting=sorting,
        recording=rec,
        format="memory",
        sparse=True,
        sparsity=sparsity,
    )
    analyzer_sparse.compute("random_spikes", max_spikes_per_unit=MAX_SPIKES_PER_UNIT)
    analyzer_sparse.compute("waveforms",
                            ms_before=args.ms_before,
                            ms_after=args.ms_after,
                            **JOB_KWARGS)
    analyzer_sparse.compute("templates")

    print(f"\n[4d/5] Computing PCA (n_components={args.n_pcs}, mode=by_channel_local) …")
    analyzer_sparse.compute("principal_components",
                            n_components=args.n_pcs,
                            mode="by_channel_local",
                            **JOB_KWARGS)

    print(f"\n[5/5] Saving PC features to {ks4_dir!r} …")
    recompute_and_save_pc_features(ks4_dir, analyzer_sparse, sparsity, args.n_pcs)

    print(f"\n✓  Done.")
    print(f"  Old PC features backed up as pc_features_old.npy and pc_feature_ind_old.npy")
    print(f"  New PC features saved to {ks4_dir / 'pc_features.npy'} and {ks4_dir / 'pc_feature_ind.npy'}")


if __name__ == "__main__":
    main()