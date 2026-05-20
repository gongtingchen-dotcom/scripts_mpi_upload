#!/usr/bin/env python3
"""
merge_tomato_results.py
=======================
Merge partial Tomato clustering results produced by separate try_tomato.py
runs (each covering a different sample range of the recording).

Usage
-----
python merge_tomato_results.py \\
    /ptmp/gongt/GLC7189/GLC7189_g_g1/preprocessed/thresholding_peaks/tomato_windowed_s0_e540000000 \\
    /ptmp/gongt/GLC7189/GLC7189_g_g1/preprocessed/thresholding_peaks/tomato_windowed_s540000000_eEND \\
    [--output  /ptmp/gongt/.../tomato_sorting_merged]
    [--peaks   /ptmp/gongt/.../peaks_combined2_sorted.npy]
    [--recording /ptmp/gongt/.../preprocessed]
    [--n-clusters-override 'win0=3,win1=2']   # optional per-window cluster count

Key design choices
------------------
* The `valid_peak_idx` stored in each per-window NPZ is an index into the
  **run-local** filtered peaks array (peaks[sample_start..sample_end]).
  We recover the **global** sample times by: peaks_local = peaks_full[run_mask],
  then global_sample = peaks_local['sample_index'][valid_peak_idx].
* Global cluster IDs are allocated with a per-run offset so they never collide.
* The merged NumpySorting uses the original sample_index values, so it is
  fully compatible with try_merge_selfdbuild.py and build_export_to_phy.py.
* Duplicate spike coverage (if two runs overlap in sample range) is detected
  and a warning is raised.
"""

import argparse
import glob
import os
import re
import shutil
import sys

import numpy as np

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_peaks_full(peaks_path):
    print(f"Loading peaks from: {peaks_path}")
    peaks_full = np.load(peaks_path)
    print(f"  Total peaks in file: {len(peaks_full):,}")
    return peaks_full


def _parse_sample_range_from_folder(folder):
    """Extract sample_start, sample_end from a folder name like
    'tomato_windowed_s0_e540000000' or 'tomato_windowed_s540000000_eEND'.
    Returns (sample_start, sample_end_or_None).
    """
    base = os.path.basename(folder.rstrip("/"))
    m = re.search(r'_s(\d+)_e(\d+|END)', base)
    if m is None:
        raise ValueError(
            f"Cannot parse sample range from folder name: '{base}'\n"
            f"Expected pattern like '_s<start>_e<end>' or '_s<start>_eEND'."
        )
    s_start = int(m.group(1))
    s_end_str = m.group(2)
    s_end = None if s_end_str == "END" else int(s_end_str)
    return s_start, s_end


def _get_run_mask(peaks_full, s_start, s_end):
    """Return boolean mask over peaks_full for this run's sample range."""
    mask = peaks_full['sample_index'] >= s_start
    if s_end is not None:
        mask &= peaks_full['sample_index'] < s_end
    return mask


def _load_npz_files(folder):
    """Return sorted list of per-window NPZ paths in the given folder."""
    paths = sorted(glob.glob(os.path.join(folder, 'tomato_win*.npz')))
    if not paths:
        raise FileNotFoundError(f"No tomato_win*.npz files found in: {folder}")
    return paths


def _parse_window_tag(npz_path):
    """Return (win_idx, center_ch) from a filename like 'tomato_win02_ch048.npz'."""
    bname = os.path.basename(npz_path)
    w_idx = int(bname.split('win')[1].split('_')[0])
    center_ch = int(bname.split('_ch')[1].replace('.npz', ''))
    return w_idx, center_ch


def _relabel_contiguous(labels):
    """Re-number arbitrary integer labels to 0..k-1 (preserving -1 for unassigned)."""
    unique_ids = np.unique(labels[labels >= 0])
    remap = {old: new for new, old in enumerate(unique_ids)}
    out = np.full_like(labels, -1)
    for old, new in remap.items():
        out[labels == old] = new
    return out, len(unique_ids)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def merge_runs(run_folders, peaks_path, recording_path, output_folder,
               n_clusters_override=None, overwrite=False):
    """
    Parameters
    ----------
    run_folders : list[str]
        Each element is a path to a `tomato_windowed_s*_e*` directory.
    peaks_path : str
        Path to peaks_combined2_sorted.npy.
    recording_path : str
        Path to the preprocessed SpikeInterface recording folder.
    output_folder : str
        Destination for the merged NumpySorting.
    n_clusters_override : dict or None
        {(run_idx, win_idx): n_clusters} — only used for re-fitting labels.
        NOT needed here because we load pre-fitted final_labels_*.npy if
        available, or fall back to the raw labels in the NPZ.
    overwrite : bool
    """
    if n_clusters_override is None:
        n_clusters_override = {}

    peaks_full = _load_peaks_full(peaks_path)
    n_spikes_full = len(peaks_full)

    # ── Check for overlapping sample ranges ───────────────────────────────────
    print("\n=== Parsing run folders ===")
    run_info = []
    for folder in run_folders:
        s_start, s_end = _parse_sample_range_from_folder(folder)
        mask = _get_run_mask(peaks_full, s_start, s_end)
        covered = int(mask.sum())
        actual_end = peaks_full['sample_index'][mask].max() if covered > 0 else -1
        print(f"  {os.path.basename(folder)}: "
              f"samples [{s_start:,}, {s_end if s_end else 'END'})"
              f"  →  {covered:,} peaks")
        run_info.append({
            'folder':   folder,
            's_start':  s_start,
            's_end':    s_end,
            'mask':     mask,
            'n_peaks':  covered,
        })

    # Detect overlaps
    combined_mask = np.zeros(n_spikes_full, dtype=np.int32)
    for ri in run_info:
        combined_mask += ri['mask'].astype(np.int32)
    n_overlap = int((combined_mask > 1).sum())
    if n_overlap > 0:
        print(f"\n  ⚠ WARNING: {n_overlap:,} peaks appear in >1 run (overlapping sample ranges).")
        print("    These will be included in multiple clusters — consider fixing the ranges.")
    n_uncovered = int((combined_mask == 0).sum())
    if n_uncovered > 0:
        print(f"\n  ⚠ NOTE: {n_uncovered:,} peaks are not covered by any run "
              f"(gap in sample ranges).")

    # ── Merge across runs ─────────────────────────────────────────────────────
    # global_sample_times : list of sample indices (int64)
    # global_cluster_ids  : list of cluster IDs    (int64)
    global_spike_times   = []
    global_spike_clusters = []
    cluster_offset = 0  # running count across all runs

    print("\n=== Merging windows ===")
    for run_idx, ri in enumerate(run_info):
        folder  = ri['folder']
        mask    = ri['mask']               # bool (n_spikes_full,)
        # Run-local peaks subset  (same filtering as try_tomato.py does)
        peaks_local = peaks_full[mask]     # shape (n_local,)
        n_local = len(peaks_local)

        npz_files = _load_npz_files(folder)
        print(f"\nRun {run_idx}: {os.path.basename(folder)}")
        print(f"  {n_local:,} local peaks  |  {len(npz_files)} windows")

        run_spike_times   = []
        run_spike_clusters = []

        for npz_path in npz_files:
            w_idx, center_ch = _parse_window_tag(npz_path)
            d = np.load(npz_path, allow_pickle=False)

            # ── Recover valid_peak_idx (index into peaks_local) ───────────
            if 'valid_peak_idx' in d:
                valid_peak_idx_w = d['valid_peak_idx'].astype(np.int64)
            else:
                # Legacy fallback: try to re-derive from labels length
                n_labels = int(len(d['labels']))
                # We cannot reliably recover valid_peak_idx without re-running;
                # use all indices for this window as a best-effort.
                print(f"    [win {w_idx:02d}] WARNING: 'valid_peak_idx' not in NPZ — "
                      f"using raw label order (may be slightly off at boundaries)")
                valid_peak_idx_w = np.arange(n_labels, dtype=np.int64)

            # ── Load final cluster labels ──────────────────────────────────
            # Prefer the cached final_labels_*.npy (already re-fitted with n_clusters)
            tag = f'win{w_idx:02d}_ch{center_ch:03d}'
            cache_candidates = sorted(
                glob.glob(os.path.join(folder, f'final_labels_{tag}_k*.npy'))
            )
            if cache_candidates:
                # Sort NUMERICALLY by k (not alphabetically — 'k10' < 'k9' lexicographically)
                def _k_val(p):
                    m2 = re.search(r'_k(\d+)\.npy$', p)
                    return int(m2.group(1)) if m2 else -1
                cache_candidates.sort(key=_k_val)
                labels_w = np.load(cache_candidates[-1])  # highest numeric k
                n_unique = len(np.unique(labels_w[labels_w >= 0]))
                chosen_k = _k_val(cache_candidates[-1])
                print(f"    [win {w_idx:02d} ch {center_ch:03d}] "
                      f"loaded final_labels (k={chosen_k}, {n_unique} unique labels, {len(labels_w):,} spikes)")
            else:
                # Fall back to raw labels in NPZ
                labels_w = d['labels'].astype(np.int64)
                print(f"    [win {w_idx:02d} ch {center_ch:03d}] "
                      f"using raw NPZ labels ({len(np.unique(labels_w))} clusters, {len(labels_w):,} spikes)")

            # Sanity check: labels_w and valid_peak_idx_w MUST have the same length.
            # They were produced together from the same extraction call in try_tomato.py,
            # so a mismatch indicates a corrupted NPZ or wrong cache file.
            if len(labels_w) != len(valid_peak_idx_w):
                raise RuntimeError(
                    f"[win {w_idx:02d} ch {center_ch:03d}] FATAL: "
                    f"len(labels_w)={len(labels_w)} != len(valid_peak_idx_w)={len(valid_peak_idx_w)}.\n"
                    f"  This means the final_labels cache file does not match the NPZ file.\n"
                    f"  NPZ path  : {npz_path}\n"
                    f"  Cache used: {cache_candidates[-1] if cache_candidates else 'NPZ labels'}\n"
                    f"  Fix: delete the stale cache or re-run try_tomato.py for this window."
                )

            # bounds check
            out_of_bounds = valid_peak_idx_w >= n_local
            if out_of_bounds.any():
                print(f"    ⚠ WARNING [win {w_idx:02d}]: {out_of_bounds.sum()} "
                      f"out-of-bounds indices — dropping them")
                keep = ~out_of_bounds
                labels_w         = labels_w[keep]
                valid_peak_idx_w = valid_peak_idx_w[keep]

            # ── Map local peak indices → global sample times ───────────────
            # peaks_local['sample_index'][valid_peak_idx_w] are the original
            # recording sample indices, which are globally correct.
            sample_times_w = peaks_local['sample_index'][valid_peak_idx_w].astype(np.int64)

            # ── Re-number labels to contiguous 0..k-1 then offset ─────────
            assigned_mask = labels_w >= 0
            if not assigned_mask.any():
                continue
            local_labels  = labels_w[assigned_mask]
            times_assigned = sample_times_w[assigned_mask]

            local_labels_cont, n_clust_w = _relabel_contiguous(local_labels)
            global_cluster_ids_w = local_labels_cont + cluster_offset

            run_spike_times.append(times_assigned)
            run_spike_clusters.append(global_cluster_ids_w)
            cluster_offset += n_clust_w

        if run_spike_times:
            global_spike_times.append(np.concatenate(run_spike_times))
            global_spike_clusters.append(np.concatenate(run_spike_clusters))

    # ── Concatenate everything ────────────────────────────────────────────────
    if not global_spike_times:
        raise RuntimeError("No spikes collected from any run — check input folders.")

    spike_times_merged    = np.concatenate(global_spike_times).astype(np.int64)
    spike_clusters_merged = np.concatenate(global_spike_clusters).astype(np.int64)

    # Sort by time (required for NumpySorting and phy)
    sort_order = np.argsort(spike_times_merged)
    spike_times_merged    = spike_times_merged[sort_order]
    spike_clusters_merged = spike_clusters_merged[sort_order]

    n_total_clusters = int(spike_clusters_merged.max()) + 1

    print(f"\n=== Merge summary ===")
    print(f"  Total clusters  : {n_total_clusters}")
    print(f"  Total spikes    : {len(spike_times_merged):,}")
    print(f"  Peaks in full   : {n_spikes_full:,}")
    print(f"  Unassigned      : {n_spikes_full - len(spike_times_merged):,}")

    # ── Build spike_trains dict for NumpySorting ──────────────────────────────
    spike_trains_si = {}
    for uid in range(n_total_clusters):
        mask_u = spike_clusters_merged == uid
        spike_trains_si[uid] = np.sort(spike_times_merged[mask_u]).astype(np.int64)

    # ── Load recording to get sampling frequency ──────────────────────────────
    try:
        import spikeinterface.full as si
        rec = si.load(recording_path)
        fs = float(rec.get_sampling_frequency())
        print(f"  Recording       : {rec.get_num_channels()} ch  {fs:.0f} Hz  "
              f"{rec.get_duration():.1f} s")
    except Exception as e:
        print(f"  ⚠ Could not load recording (needed for NumpySorting): {e}")
        print("    Trying to infer sampling frequency from peaks structure …")
        # Fallback: try 30 kHz
        fs = 30000.0
        print(f"    Using default fs = {fs} Hz")

    # ── Create NumpySorting ───────────────────────────────────────────────────
    try:
        sorting = si.NumpySorting.from_dict(
            [spike_trains_si],
            sampling_frequency=fs
        )
    except Exception as _e1:
        try:
            sorting = si.NumpySorting.from_unit_dict(
                spike_trains_si,
                sampling_frequency=fs
            )
        except Exception as _e2:
            raise RuntimeError(f"Could not create NumpySorting:\n  {_e1}\n  {_e2}")

    print(f"\nNumpySorting created:")
    print(f"  n_units  : {len(sorting.unit_ids)}")
    print(f"  unit_ids : {sorting.unit_ids[:8]}...  "
          f"(type={type(sorting.unit_ids[0]).__name__})")

    # ── Save NumpySorting ─────────────────────────────────────────────────────
    if os.path.exists(output_folder):
        if not overwrite:
            raise FileExistsError(
                f"Output folder already exists: {output_folder}\n"
                f"Use --overwrite to replace it."
            )
        print(f"\n  Removing existing output folder …")
        shutil.rmtree(output_folder)

    sorting.save(folder=output_folder)
    print(f"\nSaved merged sorting to: {output_folder}")

    # ── Also save raw spike arrays alongside for easy inspection ─────────────
    np.save(os.path.join(output_folder, 'spike_times.npy'),    spike_times_merged)
    np.save(os.path.join(output_folder, 'spike_clusters.npy'), spike_clusters_merged)

    print(f"\n✓ DONE — {n_total_clusters} clusters, {len(spike_times_merged):,} spikes")
    print(f"  You can now run build_export_to_phy.py or try_merge_selfdbuild.py")
    print(f"  with SORTING_DIR = '{output_folder}'")

    return sorting


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_n_clusters_override(s):
    """Parse 'win0=3,win1=2' or '(0,0)=3,(0,1)=2' style strings."""
    out = {}
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        # Try 'winN=K' format (applies to all runs with that win_idx)
        m = re.match(r'win(\d+)=(\d+)$', part)
        if m:
            out[int(m.group(1))] = int(m.group(2))
            continue
        raise ValueError(f"Cannot parse n-clusters-override token: '{part}'. "
                         "Use format 'win0=3,win2=5'.")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Merge partial try_tomato.py results across sample ranges.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('run_folders', nargs='+',
                        help='Paths to tomato_windowed_s*_e* folders (one per run)')
    parser.add_argument('--output', default=None,
                        help='Destination folder for merged NumpySorting '
                             '(default: <first_folder_parent>/tomato_sorting_merged)')
    parser.add_argument('--peaks',
                        default='/ptmp/gongt/GLC7189/GLC7189_g_g1/preprocessed/'
                                'thresholding_peaks/peaks_combined2_sorted.npy',
                        help='Path to peaks_combined2_sorted.npy')
    parser.add_argument('--recording',
                        default='/ptmp/gongt/GLC7189/GLC7189_g_g1/preprocessed',
                        help='Path to the SpikeInterface preprocessed recording folder')
    parser.add_argument('--n-clusters-override', default='',
                        help="Per-window cluster overrides, e.g. 'win0=3,win2=5'")
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output folder')

    args = parser.parse_args()

    # Validate run_folders exist
    for f in args.run_folders:
        if not os.path.isdir(f):
            print(f"ERROR: folder not found: {f}", file=sys.stderr)
            sys.exit(1)

    # Default output folder
    if args.output is None:
        parent = os.path.dirname(args.run_folders[0].rstrip('/'))
        args.output = os.path.join(os.path.dirname(parent), 'tomato_sorting_merged')

    n_clusters_override = {}
    if args.n_clusters_override:
        try:
            n_clusters_override = _parse_n_clusters_override(args.n_clusters_override)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    print("="*60)
    print("Tomato results merger")
    print("="*60)
    print(f"Runs ({len(args.run_folders)}):")
    for f in args.run_folders:
        print(f"  {f}")
    print(f"Peaks file  : {args.peaks}")
    print(f"Recording   : {args.recording}")
    print(f"Output      : {args.output}")
    print()

    merge_runs(
        run_folders=args.run_folders,
        peaks_path=args.peaks,
        recording_path=args.recording,
        output_folder=args.output,
        n_clusters_override=n_clusters_override,
        overwrite=args.overwrite,
    )


if __name__ == '__main__':
    main()
