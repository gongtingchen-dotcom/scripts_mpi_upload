"""
mua_pipeline.py
---------------
Reusable MUA thresholding pipeline extracted from MUA_threhsold2.ipynb.

Call run_mua_pipeline() from any notebook to run the full pipeline on a
new recording.  Example usage in peak_localisation.ipynb:

    import sys
    sys.path.insert(0, r'C:\\Users\\gongt\\scripts')
    from mua_pipeline import run_mua_pipeline

    rec, peaks_combined2, peaks_combined1, peaks_matched, peaks0, peaks1, peaks2 = \\
        run_mua_pipeline(bin_path=r'D:\\path\\to\\recording')
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import json
import os

import numba
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import spikeinterface.full as si
from spikeinterface.core import get_noise_levels
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.tools import get_prototype_and_waveforms_from_peaks


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(file_path, n_jobs=6, ops=None):
    """
    Load a recording from multiple formats:
      - SpikeGLX .bin file (if 'imec' in path)
      - SpikeInterface-saved folder (with binary.json, probe.json)
      - temp_wh.dat (kilosort drift-corrected binary, if in a folder with binary.json)
    
    Parameters
    ----------
    file_path : str
        Path to recording directory or binary file
    n_jobs : int
        Number of parallel jobs for loading
    ops : dict, optional
        Kilosort ops dict containing channel coordinates (xc, yc).
        If provided, probe info will be extracted from ops instead of reading probe.json.
        
    Returns
    -------
    rec : BaseRecording
        SpikeInterface recording object
    """
    
    # ── Check if this is a temp_wh.dat file ────────────────────────────
    if os.path.isfile(file_path) and file_path.endswith('temp_wh.dat'):
        # file_path = /path/to/results_dir/temp_wh.dat
        results_dir = os.path.dirname(file_path)
        binary_json_path = os.path.join(results_dir, 'binary.json')
        
        if not os.path.exists(binary_json_path):
            raise FileNotFoundError(
                f"Could not find binary.json at {binary_json_path}.\n"
                f"binary.json must be in the same directory as temp_wh.dat.\n"
                f"(customised_KS4.py copies it automatically)"
            )
        
        # Load metadata from binary.json
        with open(binary_json_path, 'r') as f:
            binary_info = json.load(f)
        
        kwargs = binary_info.get('kwargs', {})
        # NOTE: temp_wh.dat is ALWAYS saved as int16 by io.save_preprocessing
        # Regardless of original dtype, so force it here
        rec = si.read_binary(
            file_path,
            sampling_frequency=kwargs.get('sampling_frequency'),
            num_channels=kwargs.get('num_channels'),
            dtype='int16'  # temp_wh.dat is always int16
        )
        
        # ── Attach probe info ────────────────────────────────────────────
        # Follow phy's convention: load from channel_positions.npy or create from ops
        
        # Try to find or create channel_positions.npy (phy standard)
        channel_positions_path = os.path.join(results_dir, 'channel_positions.npy')
        if not os.path.exists(channel_positions_path) and ops is not None and 'xc' in ops and 'yc' in ops:
            # Create channel_positions.npy from ops if it doesn't exist
            try:
                xc = ops['xc']
                yc = ops['yc']
                channel_positions = np.column_stack([xc, yc])
                np.save(channel_positions_path, channel_positions)
                print(f"  Created channel_positions.npy from ops ({len(xc)} channels)")
            except Exception as e:
                print(f"Warning: Could not create channel_positions.npy from ops: {e}")
        
        # Load and set channel locations using SpikeInterface
        if os.path.exists(channel_positions_path):
            try:
                channel_positions = np.load(channel_positions_path)
                rec_with_locs = rec.set_channel_locations(channel_positions)
                # set_channel_locations might return the recording or None (in_place behavior)
                if rec_with_locs is not None:
                    rec = rec_with_locs
                print(f"  Set channel locations from channel_positions.npy ({len(channel_positions)} channels)")
            except Exception as e:
                print(f"Warning: Could not set channel locations: {e}")
                # Continue anyway - channel locations are nice to have but not essential
        
        return rec
    
    # ── Check if this is a directory with binary.json (SI-saved format) ────
    if os.path.isdir(file_path):
        binary_json_path = os.path.join(file_path, 'binary.json')
        if os.path.exists(binary_json_path):
            # This is a SpikeInterface-saved folder
            rec = si.load(file_path, n_jobs=n_jobs)
            return rec
        
        # Check if it's a SpikeGLX directory
        if 'imec' in file_path or any('imec' in f for f in os.listdir(file_path)):
            rec = si.read_spikeglx(file_path, stream_name='imec0.ap',
                                    load_sync_channel=False)
            return rec
        
        # Try generic load as last resort
        try:
            rec = si.load(file_path, n_jobs=n_jobs)
            return rec
        except Exception as e:
            raise ValueError(
                f"Could not load recording from {file_path}.\n"
                f"Expected one of:\n"
                f"  - SpikeGLX directory (contains 'imec')\n"
                f"  - SpikeInterface-saved folder (contains binary.json)\n"
                f"  - SpikeInterface-compatible format\n"
                f"Error: {e}"
            )
    
    # ── Otherwise assume it's a SpikeGLX .bin file ─────────────────────
    rec = si.read_spikeglx(file_path, stream_name='imec0.ap',
                            load_sync_channel=False)
    return rec


# ── Rolling noise ─────────────────────────────────────────────────────────────

class RollingNoiseInfo:
    """Container for per-chunk MAD noise estimates."""
    def __init__(self, chunk_noise, chunk_starts, n_channels):
        self.chunk_noise  = chunk_noise   # (n_chunks, n_channels)
        self.chunk_starts = chunk_starts  # (n_chunks,) in samples
        self.n_channels   = n_channels

    def get_noise_for_peaks(self, sample_indices, channel_indices):
        """Return noise[chunk, channel] for each (sample, channel) pair."""
        chunk_idx = np.searchsorted(self.chunk_starts, sample_indices, side='right') - 1
        chunk_idx = np.clip(chunk_idx, 0, len(self.chunk_starts) - 1)
        return self.chunk_noise[chunk_idx, channel_indices]


def load_rolling_noise(rolling_noise_path, sampling_frequency):
    """Load rolling_noise_levels.npz and return a RollingNoiseInfo."""
    data           = np.load(rolling_noise_path)
    chunk_noise    = data['chunk_noise']     # (n_chunks, n_channels)
    chunk_starts_s = data['chunk_starts_s']  # (n_chunks,) in seconds
    chunk_starts   = (chunk_starts_s * sampling_frequency).astype(np.int64)
    return RollingNoiseInfo(chunk_noise, chunk_starts, chunk_noise.shape[1])


# ── Helper: convert boolean peak mask → structured peaks array ─────────────────

def convert_peak_mask_to_peaks(peak_mask, traces, chunk_start_frame=0):
    """
    peak_mask : bool array [n_samples_chunk, n_chans]
    chunk_start_frame : offset to add to sample indices so they are global
    """
    peak_sample_ind, peak_chan_ind = np.nonzero(peak_mask)

    peak_amplitudes = traces[peak_sample_ind, peak_chan_ind]
    peaks = np.zeros(len(peak_sample_ind), dtype=[
        ('sample_index', 'int64'), ('channel_index', 'int64'),
        ('amplitude', 'float32'), ('segment_index', 'int64'),
        ('z', 'int64')
    ])

    peaks['sample_index'] = peak_sample_ind + chunk_start_frame   # global index
    peaks['channel_index'] = peak_chan_ind
    peaks['amplitude'] = peak_amplitudes

    return peaks


# ── Stage 1: threshold detection ───────────────────────────────────────────────

def thresholding(traces, sampling_frequency, noise_levels,
                 detect_threshold=3.3, exclude_sweep_ms=0.1,
                 chunk_start_frame=0):
    abs_thresholds = noise_levels * detect_threshold

    peak_mask0 = traces < -abs_thresholds[None, :]

    peaks0 = convert_peak_mask_to_peaks(peak_mask0, traces,
                                        chunk_start_frame=chunk_start_frame)
    return peaks0


def _threshold_single_chunk(rec, chunk_start, chunk_end, fs, noise_levels,
                             detect_threshold, exclude_sweep_ms):
    # Keep as int16 to save memory - numpy will auto-cast during comparison
    traces_chunk = rec.get_traces(start_frame=chunk_start,
                                  end_frame=chunk_end)
    return thresholding(
        traces_chunk, fs, noise_levels,
        detect_threshold=detect_threshold,
        exclude_sweep_ms=exclude_sweep_ms,
        chunk_start_frame=chunk_start
    )


def thresholding_chunked(rec, sampling_frequency, noise_levels,
                         detect_threshold=3.3, exclude_sweep_ms=0.1,
                         chunk_size_s=60.0, n_jobs=4):
    """Runs thresholding in parallel chunks. Returns peaks with *global* sample indices."""
    fs              = sampling_frequency
    chunk_size_samp = int(chunk_size_s * fs)
    total_samples   = rec.get_num_samples()

    chunk_ranges = []
    chunk_start = 0
    while chunk_start < total_samples:
        chunk_end = min(chunk_start + chunk_size_samp, total_samples)
        chunk_ranges.append((chunk_start, chunk_end))
        chunk_start = chunk_end

    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(_threshold_single_chunk)(
            rec, start, end, fs, noise_levels, detect_threshold, exclude_sweep_ms
        )
        for start, end in tqdm(chunk_ranges, desc="Thresholding", unit="chunk")
    )

    all_peaks = [p for p in results if len(p) > 0]
    return np.concatenate(all_peaks) if all_peaks else np.array([], dtype=[
        ('sample_index', 'int64'), ('channel_index', 'int64'),
        ('amplitude', 'float32'), ('segment_index', 'int64'), ('z', 'int64')
    ])


# ── Stage 2: spatial & temporal filter ────────────────────────────────────────

@numba.jit(nopython=True)
def _group_peaks_numba(sorted_si, sorted_ci, snr_sorted, half_win, spatial_radius, n_chans):
    """Core grouping loop in Numba (nopython). Works on *global* sample indices."""
    n_peaks  = len(sorted_si)
    assigned = np.zeros(n_peaks, dtype=numba.boolean)
    in_group = np.zeros(n_peaks, dtype=numba.boolean)

    win_t_start = np.empty(n_peaks, dtype=numba.int64)
    win_t_end   = np.empty(n_peaks, dtype=numba.int64)
    win_c_low   = np.empty(n_peaks, dtype=numba.int64)
    win_c_high  = np.empty(n_peaks, dtype=numba.int64)
    win_winner  = np.empty(n_peaks, dtype=numba.int64)
    n_windows   = 0

    for i in range(n_peaks):
        if assigned[i]:
            continue

        t_seed  = sorted_si[i]
        t_start = t_seed - half_win
        t_end   = t_seed + half_win

        lo, hi = 0, n_peaks
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_si[mid] < t_start:
                lo = mid + 1
            else:
                hi = mid
        t_lo = lo

        lo, hi = t_lo, n_peaks
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_si[mid] <= t_end:
                lo = mid + 1
            else:
                hi = mid
        t_hi = lo

        for k in range(t_lo, t_hi):
            in_group[k] = False
        in_group[i] = True

        while True:
            c_min = n_chans
            c_max = -1
            for k in range(t_lo, t_hi):
                if in_group[k]:
                    c = sorted_ci[k]
                    if c < c_min:
                        c_min = c
                    if c > c_max:
                        c_max = c

            c_search_lo = c_min - spatial_radius
            c_search_hi = c_max + spatial_radius
            if c_search_lo < 0:
                c_search_lo = 0
            if c_search_hi >= n_chans:
                c_search_hi = n_chans - 1

            changed = False
            for k in range(t_lo, t_hi):
                if assigned[k] or in_group[k]:
                    continue
                c = sorted_ci[k]
                if c_search_lo <= c <= c_search_hi:
                    in_group[k] = True
                    changed     = True

            if not changed:
                break

        c_win_lo = n_chans
        c_win_hi = -1
        best_snr = -1.0
        best_k   = i

        for k in range(t_lo, t_hi):
            if not in_group[k]:
                continue
            assigned[k] = True
            c = sorted_ci[k]
            if c < c_win_lo:
                c_win_lo = c
            if c > c_win_hi:
                c_win_hi = c
            if snr_sorted[k] > best_snr:
                best_snr = snr_sorted[k]
                best_k   = k

        win_t_start[n_windows] = t_start
        win_t_end[n_windows]   = t_end
        win_c_low[n_windows]   = c_win_lo
        win_c_high[n_windows]  = c_win_hi
        win_winner[n_windows]  = best_k
        n_windows += 1

    return (win_t_start[:n_windows], win_t_end[:n_windows],
            win_c_low[:n_windows],   win_c_high[:n_windows],
            win_winner[:n_windows])


def spatial_and_temporal_filter(peaks0, noise_levels, sampling_frequency,
                                n_chans,
                                exclude_sweep_ms=0.1,
                                spatial_radius=5,
                                chunk_size_peaks=10_000_000):
    """
    Operates purely on peak metadata (sample_index, channel_index, amplitude).
    No traces array required — peaks already carry global sample indices.

    For large datasets (>chunk_size_peaks peaks), processes in chunks with
    half_win-sample overlap to keep memory within bounds.
    """
    half_win = int(exclude_sweep_ms * sampling_frequency / 1000.0)

    order      = np.argsort(peaks0['sample_index'], kind='stable')
    sorted_si  = peaks0['sample_index'][order].astype(np.int64)
    sorted_ci  = peaks0['channel_index'][order].astype(np.int64)
    snr_sorted = (np.abs(peaks0['amplitude'][order])
                  / noise_levels[sorted_ci]).astype(np.float64)

    n_peaks = len(sorted_si)

    if n_peaks <= chunk_size_peaks:
        # Original single-pass path
        t_starts, t_ends, c_lows, c_highs, winner_sorted_idx = _group_peaks_numba(
            sorted_si, sorted_ci, snr_sorted,
            np.int64(half_win), np.int64(spatial_radius), np.int64(n_chans)
        )
        all_t_starts        = t_starts
        all_t_ends          = t_ends
        all_c_lows          = c_lows
        all_c_highs         = c_highs
        all_winner_orig_idx = order[winner_sorted_idx]
    else:
        # Chunked path: process chunk_size_peaks peaks at a time.
        # Each chunk includes half_win-sample padding on each side so that
        # groups straddling a boundary are resolved correctly.
        # We only keep windows whose seed falls strictly within the chunk's
        # own time range, so each group is emitted by exactly one chunk.
        import logging as _logging
        _logging.getLogger(__name__).info(
            f'[spatial_and_temporal_filter] {n_peaks:,} peaks — '
            f'processing in chunks of {chunk_size_peaks:,}'
        )
        ts_list, te_list, cl_list, ch_list, wi_list = [], [], [], [], []

        chunk_boundaries = list(range(0, n_peaks, chunk_size_peaks)) + [n_peaks]
        for ci in range(len(chunk_boundaries) - 1):
            pk_start = chunk_boundaries[ci]
            pk_end   = chunk_boundaries[ci + 1]

            seed_t_start = int(sorted_si[pk_start])
            # Exclusive upper bound: use next chunk's first sample (or +1 for last chunk)
            seed_t_end_excl = int(sorted_si[pk_end]) if pk_end < n_peaks else int(sorted_si[-1]) + 1

            ext_lo = int(np.searchsorted(sorted_si, seed_t_start - half_win, side='left'))
            ext_hi = int(np.searchsorted(sorted_si, seed_t_end_excl - 1 + half_win, side='right'))

            ts, te, cl, ch, wc = _group_peaks_numba(
                sorted_si [ext_lo:ext_hi],
                sorted_ci [ext_lo:ext_hi],
                snr_sorted[ext_lo:ext_hi],
                np.int64(half_win), np.int64(spatial_radius), np.int64(n_chans)
            )

            # seed time = t_start + half_win; keep only seeds in this chunk's range
            seed_times = ts + half_win
            keep = (seed_times >= seed_t_start) & (seed_times < seed_t_end_excl)

            ts_list.append(ts[keep])
            te_list.append(te[keep])
            cl_list.append(cl[keep])
            ch_list.append(ch[keep])
            wi_list.append(order[ext_lo + wc[keep]])

        all_t_starts        = np.concatenate(ts_list)
        all_t_ends          = np.concatenate(te_list)
        all_c_lows          = np.concatenate(cl_list)
        all_c_highs         = np.concatenate(ch_list)
        all_winner_orig_idx = np.concatenate(wi_list)

    # Free large working arrays before the final output allocation.
    # sorted_si / sorted_ci / snr_sorted / order together occupy ~25 GB for an
    # 800 M-peak dataset; releasing them here before allocating the output
    # array avoids OOM on systems with 32-64 GB RAM.
    import gc as _gc
    del sorted_si, sorted_ci, snr_sorted, order
    del all_t_starts, all_t_ends, all_c_lows, all_c_highs
    _gc.collect()

    # Sort indices in-place so peaks0 is read sequentially (critical when
    # peaks0 is a memory-mapped file) and the output is time-ordered.
    all_winner_orig_idx.sort()
    max_peaks = peaks0[all_winner_orig_idx]

    return [], max_peaks


# ── Stage 3: spatial support filter ───────────────────────────────────────────

@numba.jit(nopython=True)
def _numba_check_neighbor_support_fixed_radius(
    peak_samples, peak_chans, traces, noise_levels,
    detect_threshold, support_fraction, n_required,
    half_sweep_samples,
    chunk_start_frame   # subtract to get local index into this chunk's traces
):
    num_peaks   = len(peak_samples)
    keep_mask   = np.zeros(num_peaks, dtype=numba.boolean)
    num_chans   = traces.shape[1]
    num_samples = traces.shape[0]

    for p in range(num_peaks):
        t_center_local = peak_samples[p] - chunk_start_frame
        c = peak_chans[p]
        support_count = 0

        t_start = max(0, t_center_local - half_sweep_samples)
        t_end   = min(num_samples, t_center_local + half_sweep_samples + 1)

        for offset in (-2, -1, 1, 2):
            neighbor = c + offset
            if neighbor < 0 or neighbor >= num_chans:
                continue

            support_thresh = -(noise_levels[neighbor] * detect_threshold * support_fraction)

            neighbor_supported = False
            for t_idx in range(t_start, t_end):
                if traces[t_idx, neighbor] < support_thresh:
                    neighbor_supported = True
                    break

            if neighbor_supported:
                support_count += 1

            if support_count >= n_required:
                break

        keep_mask[p] = (support_count >= n_required)

    return keep_mask


def filter_peaks_by_spatial_support(
    peaks, rec, noise_levels, sampling_frequency,
    detect_threshold=3.3, support_fraction=0.5, n_required=1,
    exclude_sweep_ms=0.1,
    chunk_size_s=60.0
):
    """
    Chunked spatial-support filter. Fetches only a small window of traces per
    chunk, runs the filter, then discards the traces.
    peaks must already carry *global* sample indices.
    """
    print(f"Filtering {len(peaks)} peaks for spatial support (window: {exclude_sweep_ms}ms, "
          f"chunk: {chunk_size_s}s)...")

    fs               = sampling_frequency
    half_sweep_s     = int(exclude_sweep_ms * fs / 2000.0)
    chunk_size_samp  = int(chunk_size_s * fs)
    total_samples    = rec.get_num_samples()

    keep_mask_global = np.zeros(len(peaks), dtype=bool)

    sort_order   = np.argsort(peaks['sample_index'], kind='stable')
    sorted_peaks = peaks[sort_order]

    n_chunks = int(np.ceil(total_samples / chunk_size_samp))
    chunk_start = 0
    with tqdm(total=n_chunks, desc="Spatial support filter", unit="chunk") as pbar:
        while chunk_start < total_samples:
            chunk_end = min(chunk_start + chunk_size_samp, total_samples)

            fetch_start = max(0, chunk_start - half_sweep_s)
            fetch_end   = min(total_samples, chunk_end + half_sweep_s)

            lo = np.searchsorted(sorted_peaks['sample_index'], chunk_start, side='left')
            hi = np.searchsorted(sorted_peaks['sample_index'], chunk_end,   side='left')

            if lo < hi:
                chunk_peaks      = sorted_peaks[lo:hi]
                chunk_peaks_orig = sort_order[lo:hi]

                traces_chunk = rec.get_traces(
                    start_frame=fetch_start,
                    end_frame=fetch_end
                )

                chunk_keep = _numba_check_neighbor_support_fixed_radius(
                    chunk_peaks['sample_index'].astype(np.int64),
                    chunk_peaks['channel_index'].astype(np.int64),
                    traces_chunk,
                    noise_levels.astype(np.float64),
                    np.float64(detect_threshold),
                    np.float64(support_fraction),
                    np.int64(n_required),
                    np.int64(half_sweep_s),
                    np.int64(fetch_start)
                )

                keep_mask_global[chunk_peaks_orig] = chunk_keep

            chunk_start = chunk_end
            pbar.update(1)
            pbar.set_postfix(kept=int(keep_mask_global.sum()))

    filtered = peaks[keep_mask_global]
    print(f"Discarded {len(peaks) - len(filtered)} single-channel artifacts.")
    print(f"Kept {len(filtered)} biologically plausible spikes.")
    return filtered


# ── Main pipeline entry point ──────────────────────────────────────────────────

def run_mua_pipeline(
    bin_path,
    ops=None,                   # Kilosort ops dict (for extracting probe geom)
    frame_slice=None,           # [t_start_s, t_end_s] or [] / None for full recording
    detect_threshold=3.3,
    exclude_sweep_ms=0.3,
    spatial_radius=10,
    chunk_size_s=60.0,
    n_jobs=6,
    proto_n_jobs=-1,
    load_computed_data=False,   # if True, load from disk if files exist; else compute
    folder_name="thresholding_peaks",  # subfolder to save peaks in
    rolling_noise_dir=None,     # path to rolling_snr/ dir; None → use si.get_noise_levels()
):
    """
    Run the full MUA thresholding pipeline on a recording.

    Parameters
    ----------
    bin_path : str
        Path to the recording. Can be one of:
          - SpikeGLX .bin file path (if 'imec' directory present)
          - SpikeInterface-saved folder (contains binary.json, probe.json)
          - Path to temp_wh.dat file (kilosort drift-corrected binary)
            Must be in a directory with binary.json and probe.json
          - SpikeInterface-compatible format
    ops : dict, optional
        Kilosort ops dictionary containing probe geometry (xc, yc channels).
        If provided, probe info will be extracted from ops instead of reading
        probe.json separately. This is the preferred method as it ensures
        consistency with drift correction parameters.
    frame_slice : list or None
        Optional [t_start_s, t_end_s] to slice the recording before processing.
    detect_threshold : float
        Detection threshold in units of noise (default 3.3σ).
    exclude_sweep_ms : float
        Temporal exclusion window in ms (default 0.3 ms).
    spatial_radius : int
        Spatial grouping radius in channels (default 10).
    chunk_size_s : float
        Chunk size for chunked operations in seconds (default 60 s).
    n_jobs : int
        Number of parallel workers (default 6).
    proto_n_jobs : int
        Workers for prototype/waveform extraction (-1 = all cores).
    load_computed_data : bool, default False
        If True, check whether all output .npy files are already present in
        ``<bin_path>/thresholding_peaks/`` and load them if so.  If any file
        is missing, fall through and compute (then save) them.
        If False, always compute and overwrite any existing files.

    Returns
    -------
    rec : BaseRecording
        The (optionally sliced, float32-converted) recording object.
    peaks_combined2 : np.ndarray
        Final spatially-filtered peaks (recommended for motion estimation).
    peaks_combined1 : np.ndarray
        Combined peaks after temporal/spatial grouping (before spatial support filter).
    peaks_matched : np.ndarray
        Peaks from matched-filter detection pass.
    peaks0 : np.ndarray
        Raw threshold detections.
    peaks1 : np.ndarray
        peaks0 after spatial/temporal grouping.
    peaks2 : np.ndarray
        peaks1 after spatial support filter.

    Also saves all arrays to ``<bin_path>/thresholding_peaks/`` (or
    ``<parent_dir>/thresholding_peaks/`` if bin_path is a file).
    """
    # ── 1. Load & optionally slice ──────────────────────────────────────────
    print("folder name:", folder_name)
    rec = load_data(bin_path, n_jobs=n_jobs, ops=ops)

    if frame_slice:
        fs_tmp = rec.get_sampling_frequency()
        rec = rec.frame_slice(
            start_frame=int(frame_slice[0] * fs_tmp),
            end_frame=int(frame_slice[1] * fs_tmp)
        )

    # Convert float16 to float32 for Numba compatibility
    rec = rec.astype('float32')

    # Determine save directory based on bin_path type
    if os.path.isfile(bin_path):
        # bin_path is a file (e.g., temp_wh.dat) → save to parent directory
        base_dir = os.path.dirname(bin_path)
    else:
        # bin_path is a directory
        base_dir = bin_path
    
    save_dir = os.path.join(base_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    # ── 1a. Save computation parameters to params.json ──────────────────────
    from datetime import datetime
    params_dict = {
        "bin_path": str(bin_path),
        "frame_slice": frame_slice,
        "detect_threshold": float(detect_threshold),
        "exclude_sweep_ms": float(exclude_sweep_ms),
        "spatial_radius": int(spatial_radius),
        "chunk_size_s": float(chunk_size_s),
        "n_jobs": int(n_jobs),
        "proto_n_jobs": int(proto_n_jobs),
        "load_computed_data": bool(load_computed_data),
        "folder_name": str(folder_name),
        "timestamp_created": datetime.now().isoformat(),
    }
    params_file = os.path.join(save_dir, "params.json")
    with open(params_file, 'w') as f:
        json.dump(params_dict, f, indent=2)
    print(f"Parameters saved to: {params_file}")

    # ── 1b. Early-return if loading from disk ────────────────────────────────
    _file_names = {
        "peaks_combined2": os.path.join(save_dir, "peaks_combined2.npy"),
        "peaks_combined1": os.path.join(save_dir, "peaks_combined1.npy"),
        "peaks_matched":   os.path.join(save_dir, "peaks_matched.npy"),
        "peaks0":          os.path.join(save_dir, "peaks0.npy"),
        "peaks1":          os.path.join(save_dir, "peaks1.npy"),
        "peaks2":          os.path.join(save_dir, "peaks2.npy"),
    }

    if load_computed_data:
        _missing = [k for k, p in _file_names.items() if not os.path.isfile(p)]
        if not _missing:
            print(f"Loading pre-computed peaks from {save_dir}")
            peaks_combined2 = np.load(_file_names["peaks_combined2"])
            peaks_combined1 = np.load(_file_names["peaks_combined1"])
            peaks_matched   = np.load(_file_names["peaks_matched"])
            peaks0          = np.load(_file_names["peaks0"])
            peaks1          = np.load(_file_names["peaks1"])
            peaks2          = np.load(_file_names["peaks2"])
            return rec, peaks_combined2, peaks_combined1, peaks_matched, peaks0, peaks1, peaks2
        else:
            print(f"Data not found in {save_dir} (missing: {', '.join(_missing)}), computing data...")
    else:
        print("Computing data...")

    # ── 2. Noise & config ───────────────────────────────────────────────────
    fs      = rec.sampling_frequency
    n_chans = rec.get_num_channels()

    rolling_noise_info = None
    if rolling_noise_dir is not None:
        noise_path = os.path.join(rolling_noise_dir, "rolling_noise_levels.npz")
        if os.path.isfile(noise_path):
            print(f"Loading rolling noise levels from: {noise_path}")
            rolling_noise_info = load_rolling_noise(noise_path, fs)
            print(f"  {rolling_noise_info.chunk_noise.shape[0]} chunks × "
                  f"{rolling_noise_info.chunk_noise.shape[1]} channels")
            noise_levels = None
        else:
            print(f"WARNING: rolling_noise_dir set but file not found: {noise_path}")
            print("  Falling back to si.get_noise_levels()")

    if rolling_noise_info is None:
        print("Computing static noise levels (si.get_noise_levels) ...")
        noise_levels = get_noise_levels(rec, return_in_uV=False)

    # ── 3. Threshold detection (chunked) ────────────────────────────────────
    print("thresholding (chunked)...")
    peaks0 = thresholding_chunked(
        rec, fs, noise_levels,
        detect_threshold=detect_threshold,
        exclude_sweep_ms=exclude_sweep_ms,
        chunk_size_s=chunk_size_s,
        n_jobs=n_jobs
    )
    print(f"peaks0: {len(peaks0)}")

    # Save peaks0 immediately and reload as a read-only memory-mapped file.
    # peaks0 can be 20-30 GB for long recordings; keeping it as a regular
    # array while spatial_and_temporal_filter allocates another ~25 GB of
    # working arrays (plus the ~7 GB output) causes OOM.
    # With mmap_mode='r' the OS pages the data in/out on demand.
    np.save(_file_names["peaks0"], peaks0)
    del peaks0
    import gc as _gc; _gc.collect()
    peaks0 = np.load(_file_names["peaks0"], mmap_mode='r')

    # ── 4. Spatial & temporal filter ────────────────────────────────────────
    bio_windows, peaks1 = spatial_and_temporal_filter(
        peaks0, noise_levels, fs,
        n_chans=n_chans,
        exclude_sweep_ms=exclude_sweep_ms,
        spatial_radius=spatial_radius
    )
    print(f"peaks1: {len(peaks1)}")
    np.save(_file_names["peaks1"], peaks1)
    print(f"  Saved: {_file_names['peaks1']}")

    # ── 5. Spatial support filter on peaks1 ─────────────────────────────────
    peaks2 = filter_peaks_by_spatial_support(
        peaks1, rec, noise_levels,
        sampling_frequency=fs,
        detect_threshold=detect_threshold,
        support_fraction=0.7,
        n_required=1,
        exclude_sweep_ms=exclude_sweep_ms,
        chunk_size_s=chunk_size_s
    )
    print(f"peaks2: {len(peaks2)}")
    np.save(_file_names["peaks2"], peaks2)
    print(f"  Saved: {_file_names['peaks2']}")

    # ── 6. Prototype & matched-filter detection ──────────────────────────────
    prototype, waveforms, few_peaks = get_prototype_and_waveforms_from_peaks(
        rec,
        peaks1,
        ms_before=1,
        ms_after=1,
        job_kwargs=dict(n_jobs=proto_n_jobs, chunk_duration='1s', progress_bar=True)
    )

    method_kwargs = dict(
        peak_sign='neg',
        detect_threshold=detect_threshold - 0.2,
        exclude_sweep_ms=0.1,
        prototype=prototype,
        ms_before=1
    )

    # Use smaller chunks and fewer workers for matched filtering to avoid memory issues
    job_kwargs_matched = dict(n_jobs=8, chunk_duration='1s', progress_bar=True)
    
    peaks_matched = detect_peaks(
        recording=rec,
        method='matched_filtering',
        method_kwargs=method_kwargs,
        verbose=True,
        job_kwargs=job_kwargs_matched
    )
    print(f"peaks_matched: {len(peaks_matched)}")
    np.save(_file_names["peaks_matched"], peaks_matched)
    print(f"  Saved: {_file_names['peaks_matched']}")

    # ── 7. Combine & filter ─────────────────────────────────────────────────
    # Use peaks1 (spatially-filtered peaks0) instead of raw peaks0.
    # peaks0 is now a memmap; concatenating it into RAM would reload ~28 GB.
    # peaks1 contains the same representative peaks at ~7 GB instead.
    peaks_combined = np.concatenate([peaks1, peaks_matched])

    _, peaks_combined1 = spatial_and_temporal_filter(
        peaks_combined, noise_levels, fs,
        n_chans=n_chans,
        exclude_sweep_ms=exclude_sweep_ms,
        spatial_radius=spatial_radius
    )
    print(f"peaks_combined1: {len(peaks_combined1)}")
    np.save(_file_names["peaks_combined1"], peaks_combined1)
    print(f"  Saved: {_file_names['peaks_combined1']}")

    peaks_combined2 = filter_peaks_by_spatial_support(
        peaks_combined1, rec, noise_levels,
        sampling_frequency=fs,
        detect_threshold=detect_threshold,
        support_fraction=0.7,
        n_required=1,
        exclude_sweep_ms=exclude_sweep_ms,
        chunk_size_s=chunk_size_s
    )
    print(f"peaks_combined2: {len(peaks_combined2)}")
    np.save(_file_names["peaks_combined2"], peaks_combined2)
    print(f"  Saved: {_file_names['peaks_combined2']}")

    # ── 8. Summary ────────────────────────────────────────────────────────────
    print(f"\n✓ All peaks saved to: {save_dir}")
    print(f"  params.json: computation parameters")
    print(f"  peaks0.npy: threshold detections")
    print(f"  peaks1.npy: after spatial/temporal grouping")
    print(f"  peaks2.npy: after spatial support filter")
    print(f"  peaks_matched.npy: matched-filter detections")
    print(f"  peaks_combined1.npy: combined peaks after grouping")
    print(f"  peaks_combined2.npy: final filtered peaks (recommended)")

    return rec, peaks_combined2, peaks_combined1, peaks_matched, peaks0, peaks1, peaks2
