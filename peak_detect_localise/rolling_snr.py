"""
rolling_snr.py
--------------
Compute a rolling (sliding-window) SNR across a recording.

For each window:
  1. Load raw traces and estimate per-channel noise via MAD.
  2. Find detected peaks whose sample_index falls in the window.
  3. SNR per spike  = |amplitude| / noise_level[channel_index]
  4. Window summary = mean & median SNR, spike count.

Output (.npz):
  window_centers_s   : (n_windows,)  centre time of each window (seconds)
  window_starts_s    : (n_windows,)  window start times (seconds)
  mean_snr           : (n_windows,)  mean   SNR of all spikes in window
  median_snr         : (n_windows,)  median SNR of all spikes in window
  spike_counts       : (n_windows,)  number of spikes in window
  mean_snr_per_chan   : (n_windows, n_channels)  per-channel mean SNR (NaN if no spike)
  params             : dict stored as a 0-d object array

Save/load layout:
  SHARED_NOISE_DIR/rolling_noise_levels.npz  — chunk-level noise, reused across runs
  BASE_PATH/FOLDER_NAME/rolling_snr.npz      — SNR results (threshold-specific)
  BASE_PATH/FOLDER_NAME/rolling_noise_per_window.npz  — per-window noise (threshold-specific)
  BASE_PATH/FOLDER_NAME/params.json          — run parameters
  BASE_PATH/FOLDER_NAME/rolling_snr_over_time.png     — plots
  BASE_PATH/FOLDER_NAME/rolling_snr_summary.txt       — text summary
"""

import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Add scripts folder to path so mua_pipeline loader is available
# ---------------------------------------------------------------------------
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from scripts.peak_detect_localise.mua_pipeline import load_data  # reuse existing recording loader

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_PATH   = r"/ptmp/gongt/GLC7189/GLC7189_g_g1/preprocessed"
FOLDER_NAME = "thresholding_peaks_8.5"   # subfolder with peaks files
PEAKS_NAME = "peaks2"  # name of peaks file within FOLDER_NAME
PEAKS_PATH  = os.path.join(BASE_PATH, FOLDER_NAME, f"{PEAKS_NAME}.npy")

# FIX: define OUTPUT_DIR so it can be used as default arg and in __main__
OUTPUT_DIR = os.path.join(BASE_PATH, FOLDER_NAME)

# Shared noise levels — computed once from raw recording, reused across threshold runs
SHARED_NOISE_DIR = os.path.join(BASE_PATH, "rolling_snr")

# Sliding window parameters
WINDOW_S  = 20.0   # window duration  (seconds)
STEP_S    =  5.0   # step size        (seconds)

# Minimum number of spikes in a window to compute a meaningful SNR.
MIN_SPIKES = 10

# How many seconds to load for the fast-path noise estimate per chunk.
NOISE_SAMPLE_S = 1.0

# If the fast-path window contains zeroed artifact samples, search this many
# seconds of the chunk for a clean 1 s window instead.
NOISE_SEARCH_S = STEP_S

# Whether to also save per-channel mean SNR
SAVE_PER_CHANNEL = True

# =============================================================================
# HELPER: MAD noise estimator (matches si.get_noise_levels behaviour)
# =============================================================================

def mad_noise(traces: np.ndarray, zero_thresh: float = 0.0) -> np.ndarray:
    """
    Per-channel noise via Median Absolute Deviation, exactly matching
    SpikeInterface's implementation:
        noise = median( |x - median(x)| ) / 0.6745

    Zeroed samples from artifact removal are replaced with NaN before both
    median steps so they do not bias the estimate toward zero.

    Parameters
    ----------
    traces     : (n_samples, n_channels) ndarray
    zero_thresh: samples with |value| <= this are treated as zeroed-out

    Returns
    -------
    noise : (n_channels,) float64  — NaN for fully-zeroed channels
    """
    t = traces.astype(np.float64)
    zero_mask = np.abs(t) <= zero_thresh
    if zero_mask.any():
        t[zero_mask] = np.nan
    med = np.nanmedian(t, axis=0, keepdims=True)
    return np.nanmedian(np.abs(t - med), axis=0) / 0.6744897501960817


# =============================================================================
# MAIN
# =============================================================================

def compute_rolling_snr(
    base_path:           str        = BASE_PATH,
    peaks_path:          str        = PEAKS_PATH,
    output_dir:          str        = OUTPUT_DIR,
    shared_noise_dir:    str        = SHARED_NOISE_DIR,
    window_s:            float      = WINDOW_S,
    step_s:              float      = STEP_S,
    min_spikes:          int        = MIN_SPIKES,
    noise_sample_s:      float      = NOISE_SAMPLE_S,
    noise_search_s:      float      = NOISE_SEARCH_S,
    save_per_channel:    bool       = SAVE_PER_CHANNEL,
    peaks_name:          str        = PEAKS_NAME,
    static_noise_levels: np.ndarray = None,
):
    """
    Three-phase rolling SNR.

    Noise save/load
    ---------------
    - If static_noise_levels is provided (1-D array shape (n_channels,)), it is
      broadcast into a flat chunk_noise array and Phases 0 & 1 are skipped.
    - Otherwise: shared_noise_dir/rolling_noise_levels.npz holds chunk_noise.
      If this file exists and matches the current recording shape it is loaded directly,
      skipping Phase 1 entirely.  Otherwise Phase 1 computes and saves it there.

    Output files are named with peaks_name as suffix so multiple peaks sets can
    coexist in the same output_dir without overwriting each other.
    All other outputs (rolling_snr_{peaks_name}.npz, etc.) go to output_dir.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1.  Load recording
    # ------------------------------------------------------------------
    print(f"Loading recording from: {base_path}")
    with tqdm(total=1, desc="Loading recording", bar_format="{l_bar}{bar}| {elapsed}") as _pbar:
        rec = load_data(base_path)
        _pbar.update(1)
    fs         = rec.sampling_frequency
    total_s    = rec.get_total_duration()
    n_channels = rec.get_num_channels()
    print(f"  Duration: {total_s:.1f} s | FS: {fs:.0f} Hz | Channels: {n_channels}")

    # ------------------------------------------------------------------
    # 2.  Load peaks
    # ------------------------------------------------------------------
    print(f"Loading peaks from: {peaks_path}")
    with tqdm(total=1, desc="Loading peaks", bar_format="{l_bar}{bar}| {elapsed}") as _pbar:
        peaks = np.load(peaks_path)
        _pbar.update(1)
    print(f"  {len(peaks):,} peaks loaded")
    print(f"  Peaks dtype fields: {peaks.dtype.names}")
    print(f"  sample_index range: {peaks['sample_index'].min()} – {peaks['sample_index'].max()}")
    print(f"  channel_index range: {peaks['channel_index'].min()} – {peaks['channel_index'].max()}")
    print(f"  amplitude range: {peaks['amplitude'].min():.4f} – {peaks['amplitude'].max():.4f}")
    print(f"  amplitude < 0: {(peaks['amplitude'] < 0).sum():,}  |  > 0: {(peaks['amplitude'] > 0).sum():,}")

    sample_indices = peaks["sample_index"]
    chan_indices   = peaks["channel_index"].astype(int)
    abs_amplitudes = np.abs(peaks["amplitude"].astype(np.float64))

    # Ensure time-sorted order — required for np.searchsorted(sample_indices, …)
    # in Phase 3.  Peaks from the combined pipeline may not be globally sorted
    # (peaks1 part is sorted, peaks_matched part is separately sorted, but the
    # concatenation is not globally sorted).
    sort_order = np.argsort(sample_indices, kind='stable')
    if not np.array_equal(sort_order, np.arange(len(sort_order))):
        print("  Peaks not time-sorted — sorting by sample_index …")
        sample_indices = sample_indices[sort_order]
        chan_indices   = chan_indices[sort_order]
        abs_amplitudes = abs_amplitudes[sort_order]

    # ------------------------------------------------------------------
    # 3.  Build grids
    # ------------------------------------------------------------------
    window_samples     = int(round(window_s       * fs))
    step_samples       = int(round(step_s         * fs))
    noise_load_samples = int(round(noise_sample_s * fs))
    noise_search_samps = int(round(noise_search_s * fs))
    total_samples      = rec.get_total_samples()

    window_starts = np.arange(0, total_samples - window_samples + 1, step_samples)
    n_windows     = len(window_starts)

    chunk_starts = np.arange(0, total_samples, step_samples)
    n_chunks     = len(chunk_starts)

    print(f"  {n_windows} rolling windows  (window={window_s:.0f} s, step={step_s:.0f} s)")
    print(f"  {n_chunks} noise chunks  "
          f"(fast-path: {noise_sample_s:.1f} s = {noise_load_samples:,} samples per chunk; "
          f"fallback search: {noise_search_s:.1f} s when artifact overlap detected)")

    # ------------------------------------------------------------------
    # PHASE 0: Use static noise if provided, else try shared cache
    # ------------------------------------------------------------------
    chunk_noise = None

    if static_noise_levels is not None:
        if len(static_noise_levels) != n_channels:
            raise ValueError(
                f"static_noise_levels has {len(static_noise_levels)} channels "
                f"but recording has {n_channels}"
            )
        chunk_noise = np.tile(
            static_noise_levels.astype(np.float64)[None, :], (n_chunks, 1)
        )
        print(f"  Using static noise levels broadcast to chunk_noise {chunk_noise.shape}")
        shared_noise_path = None  # nothing to save/load
    else:
        shared_noise_path = Path(shared_noise_dir) / "rolling_noise_levels.npz"

    if chunk_noise is None and shared_noise_path is not None and shared_noise_path.exists():
        print(f"\nLoading pre-computed noise levels from: {shared_noise_path}")
        noise_data = np.load(shared_noise_path)
        if "chunk_noise" in noise_data:
            stored_shape = noise_data["chunk_noise"].shape
            if stored_shape[0] == n_chunks and stored_shape[1] == n_channels:
                chunk_noise = noise_data["chunk_noise"]
                print(f"  ✓ Loaded chunk_noise: {chunk_noise.shape}")
            else:
                print(f"  ⚠ Shape mismatch: stored {stored_shape} vs expected "
                      f"({n_chunks}, {n_channels}) — recomputing")

    # ------------------------------------------------------------------
    # PHASE 1: Noise estimation (only when not cached)
    # ------------------------------------------------------------------
    if chunk_noise is None:
        print("\nComputing noise levels …")
        chunk_noise = np.full((n_chunks, n_channels), np.nan)
        _rng        = np.random.default_rng(0)
        n_fallback  = 0

        for c_idx, c_start in enumerate(tqdm(chunk_starts, desc="Phase 1: noise estimation", unit="chunk")):
            c_start  = int(c_start)
            fast_end = min(c_start + noise_load_samples, total_samples)
            traces   = rec.get_traces(start_frame=c_start, end_frame=fast_end)

            # FIX: this if-block was outside the for loop — now correctly indented
            if np.any(np.abs(traces) == 0):
                # Fallback: search the full step for a clean block
                n_fallback  += 1
                search_end   = min(c_start + noise_search_samps, total_samples)
                full_traces  = rec.get_traces(start_frame=c_start, end_frame=search_end)
                n_search     = full_traces.shape[0]

                if n_search >= noise_load_samples:
                    any_zero  = np.any(np.abs(full_traces) == 0, axis=1).astype(np.int32)
                    cum       = np.concatenate([[0], np.cumsum(any_zero)])
                    n_starts  = n_search - noise_load_samples + 1
                    zero_in_w = cum[noise_load_samples : noise_load_samples + n_starts] - cum[:n_starts]
                    clean_pos = np.where(zero_in_w == 0)[0]

                    if len(clean_pos) > 0:
                        chosen = int(_rng.choice(clean_pos))
                        traces = full_traces[chosen : chosen + noise_load_samples]
                    else:
                        traces = full_traces
                else:
                    traces = full_traces

            # FIX: this line was also outside the for loop
            chunk_noise[c_idx] = mad_noise(traces)

            if c_idx == 0:
                print(f"  [debug] noise sample shape: {traces.shape}, dtype: {traces.dtype}")
                print(f"  [debug] traces: min={traces.min()}, max={traces.max()}, mean={traces.mean():.3f}")
                print(f"  [debug] noise (chunk 0): min={chunk_noise[0].min():.4f}, "
                      f"max={chunk_noise[0].max():.4f}, mean={chunk_noise[0].mean():.4f}")

        print(f"  Fallback (artifact overlap): {n_fallback}/{n_chunks} chunks")

        # FIX: save chunk_noise only to shared_noise_dir (not output_dir)
        Path(shared_noise_dir).mkdir(parents=True, exist_ok=True)
        np.savez(
            shared_noise_path,
            chunk_noise    = chunk_noise,
            chunk_starts_s = chunk_starts / fs,
        )
        print(f"  Saved shared noise levels → {shared_noise_path}")

    # ------------------------------------------------------------------
    # PHASE 2: Per-spike SNR (vectorised, zero I/O)
    # ------------------------------------------------------------------
    print("Phase 2: assigning per-spike SNR (vectorised) …", flush=True)

    valid_ch_mask   = (chan_indices >= 0) & (chan_indices < n_channels)
    spike_chunk_idx = np.clip(
        np.searchsorted(chunk_starts, sample_indices, side="right") - 1,
        0, n_chunks - 1
    ).astype(int)

    spike_noise = np.where(
        valid_ch_mask,
        chunk_noise[spike_chunk_idx, np.clip(chan_indices, 0, n_channels - 1)],
        np.nan
    )
    spike_snr_all = abs_amplitudes / spike_noise
    spike_snr_all[~valid_ch_mask] = np.nan

    print(f"  [debug] per-spike SNR: min={np.nanmin(spike_snr_all):.2f}, "
          f"max={np.nanmax(spike_snr_all):.2f}, mean={np.nanmean(spike_snr_all):.2f}, "
          f"n_nan={np.isnan(spike_snr_all).sum():,}")

    window_lo    = np.searchsorted(sample_indices, window_starts,                  side="left")
    window_hi    = np.searchsorted(sample_indices, window_starts + window_samples, side="left")
    spike_counts = (window_hi - window_lo).astype(np.int64)

    # ------------------------------------------------------------------
    # PHASE 3: Rolling window aggregation (zero I/O)
    # ------------------------------------------------------------------
    mean_snr     = np.full(n_windows, np.nan)
    median_snr   = np.full(n_windows, np.nan)
    noise_levels = np.full((n_windows, n_channels), np.nan)

    if save_per_channel:
        mean_snr_per_chan = np.full((n_windows, n_channels), np.nan)
    else:
        mean_snr_per_chan = None

    w_width = len(str(n_windows))

    pbar = tqdm(total=n_windows, desc="Phase 3: rolling windows", unit="win",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for i in range(n_windows):
        lo      = int(window_lo[i])
        hi      = int(window_hi[i])
        t_start = window_starts[i] / fs
        t_end   = (window_starts[i] + window_samples) / fs
        n_sp    = int(spike_counts[i])

        c_lo = max(int(np.searchsorted(chunk_starts, window_starts[i],                  side="right") - 1), 0)
        c_hi = min(int(np.searchsorted(chunk_starts, window_starts[i] + window_samples, side="right")),     n_chunks)
        noise_levels[i] = np.nanmean(chunk_noise[c_lo:c_hi], axis=0)

        if n_sp < min_spikes:
            tqdm.write(
                f"  win {i+1:>{w_width}}/{n_windows}  "
                f"[{t_start:8.1f} – {t_end:8.1f} s]  "
                f"spikes={n_sp}  (< min={min_spikes}, SNR skipped)  "
                f"noise_mean={noise_levels[i].mean():.3f}"
            )
            pbar.update(1)
            continue

        snrs       = spike_snr_all[lo:hi]
        valid_snrs = snrs[~np.isnan(snrs)]
        if len(valid_snrs) == 0:
            pbar.update(1)
            continue

        mean_snr[i]   = np.mean(valid_snrs)
        median_snr[i] = np.median(valid_snrs)

        tqdm.write(
            f"  win {i+1:>{w_width}}/{n_windows}  "
            f"[{t_start:8.1f} – {t_end:8.1f} s]  "
            f"spikes={n_sp:6d}  "
            f"SNR mean={mean_snr[i]:6.2f}  median={median_snr[i]:6.2f}  "
            f"[min={valid_snrs.min():.2f}, max={valid_snrs.max():.2f}]  "
            f"noise_mean={noise_levels[i].mean():.3f}"
        )

        if save_per_channel:
            w_chans = chan_indices[lo:hi]
            w_valid = valid_ch_mask[lo:hi] & ~np.isnan(snrs)
            if w_valid.any():
                wc     = w_chans[w_valid]
                ws     = snrs[w_valid]
                ch_sum = np.bincount(wc, weights=ws, minlength=n_channels)
                ch_cnt = np.bincount(wc,             minlength=n_channels)
                hit    = ch_cnt > 0
                mean_snr_per_chan[i, hit] = ch_sum[hit] / ch_cnt[hit]

        pbar.update(1)

    pbar.close()

    # ------------------------------------------------------------------
    # 4.  Save results — everything goes to output_dir (BASE_PATH/FOLDER_NAME)
    # ------------------------------------------------------------------
    import json

    window_starts_s  = window_starts / fs
    window_centers_s = window_starts_s + window_s / 2.0

    params = dict(
        base_path       = base_path,
        peaks_path      = peaks_path,
        output_dir      = str(out_dir),
        shared_noise_dir= shared_noise_dir,
        window_s        = window_s,
        step_s          = step_s,
        min_spikes      = min_spikes,
        noise_sample_s  = noise_sample_s,
        noise_search_s  = noise_search_s,
        total_s         = float(total_s),
        fs              = float(fs),
        n_channels      = int(n_channels),
        n_windows       = int(n_windows),
        n_peaks_total   = int(len(peaks)),
        n_windows_with_spikes = int(np.sum(spike_counts >= min_spikes)),
        overall_mean_snr      = float(np.nanmean(mean_snr)),
        overall_median_snr    = float(np.nanmedian(mean_snr)),
        overall_min_snr       = float(np.nanmin(mean_snr)),
        overall_max_snr       = float(np.nanmax(mean_snr)),
    )

    # rolling_snr.npz — SNR time-series
    snr_dict = dict(
        window_centers_s = window_centers_s,
        window_starts_s  = window_starts_s,
        mean_snr         = mean_snr,
        median_snr       = median_snr,
        spike_counts     = spike_counts,
    )
    if save_per_channel:
        snr_dict["mean_snr_per_chan"] = mean_snr_per_chan

    snr_path = out_dir / f"rolling_snr_{peaks_name}.npz"
    np.savez(snr_path, **snr_dict)
    print(f"  Saved SNR              → {snr_path}")

    # rolling_noise_per_window_{peaks_name}.npz — per-window noise summary (threshold-specific)
    # FIX: chunk_noise is NOT re-saved here; it lives only in shared_noise_dir
    noise_path = out_dir / f"rolling_noise_per_window_{peaks_name}.npz"
    np.savez(
        noise_path,
        noise_levels     = noise_levels,      # (n_windows, n_channels)
        window_starts_s  = window_starts_s,
        window_centers_s = window_centers_s,
    )
    print(f"  Saved per-window noise → {noise_path}")

    # params_{peaks_name}.json
    json_path = out_dir / f"params_{peaks_name}.json"
    with open(json_path, "w") as _f:
        json.dump(params, _f, indent=2)
    print(f"  Saved params           → {json_path}")

    print(f"\nAll outputs saved to: {out_dir}")
    print(f"  Shared noise lives at : {shared_noise_path or 'N/A (static noise used)'}")
    print(f"  Windows computed      : {n_windows}")
    print(f"  Windows with >= {min_spikes} spikes: {params['n_windows_with_spikes']}")
    print(f"  Overall mean SNR      : {params['overall_mean_snr']:.2f}")
    print(f"  Overall median SNR    : {params['overall_median_snr']:.2f}")
    print(f"  Overall SNR range     : [{params['overall_min_snr']:.2f}, {params['overall_max_snr']:.2f}]")

    return dict(**snr_dict, noise_levels=noise_levels, chunk_noise=chunk_noise, params=params)


# =============================================================================
# Quick plotting helper
# =============================================================================

def plot_rolling_snr(output_dir: str = OUTPUT_DIR, peaks_name: str = None, ax=None):
    """Plot mean and median rolling SNR and spike count from a saved output folder."""
    import json
    import matplotlib.pyplot as plt

    if peaks_name is None:
        peaks_name = PEAKS_NAME

    out_dir    = Path(output_dir)
    data       = np.load(out_dir / f"rolling_snr_{peaks_name}.npz", allow_pickle=True)
    t          = data["window_centers_s"]
    mean_snr   = data["mean_snr"]
    median_snr = data["median_snr"]
    counts     = data["spike_counts"]
    with open(out_dir / f"params_{peaks_name}.json") as _f:
        params = json.load(_f)

    show = ax is None
    if show:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    else:
        axes = ax

    ax0, ax1 = axes

    ax0.plot(t / 60, mean_snr,   label="mean SNR",   lw=1)
    ax0.plot(t / 60, median_snr, label="median SNR", lw=1, linestyle="--")
    ax0.set_ylabel("SNR")
    ax0.legend(fontsize=8)
    ax0.set_title(
        f"Rolling SNR  (window={params['window_s']:.0f} s, step={params['step_s']:.0f} s)"
    )

    ax1.bar(t / 60, counts, width=params["step_s"] / 60, align="edge",
            color="steelblue", alpha=0.6)
    ax1.set_ylabel("Spikes / window")
    ax1.set_xlabel("Time (min)")

    if show:
        plt.tight_layout()
        plt.show()


# =============================================================================
# Analysis: create plots and summary after rolling SNR computation
# =============================================================================

def analyze_rolling_snr(output_dir: str = OUTPUT_DIR, peaks_name: str = None, detect_threshold: float = None):
    """
    Load rolling SNR results and create analysis plots/summaries.

    Saves to output_dir:
      - rolling_snr_over_time_{peaks_name}.png
      - rolling_snr_summary_{peaks_name}.txt
    """
    import json
    import matplotlib.pyplot as plt

    if peaks_name is None:
        peaks_name = PEAKS_NAME

    out_dir = Path(output_dir)

    print(f"\nAnalyzing rolling SNR from: {out_dir}")
    snr_data = np.load(out_dir / f"rolling_snr_{peaks_name}.npz", allow_pickle=True)

    with open(out_dir / f"params_{peaks_name}.json") as _f:
        params = json.load(_f)

    t          = snr_data["window_centers_s"]
    mean_snr   = snr_data["mean_snr"]
    median_snr = snr_data["median_snr"]
    counts     = snr_data["spike_counts"]

    min_spikes = params.get("min_spikes", 10)
    valid      = counts >= min_spikes

    overall_mean = np.nanmean(mean_snr[valid])
    snr_min      = np.nanmin(mean_snr[valid])
    snr_max      = np.nanmax(mean_snr[valid])

    print("  Creating rolling_snr_over_time.png ...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    t_min = t / 60.0

    ax = axes[0]
    ax.plot(t_min[valid],  mean_snr[valid],  lw=1.2, color="steelblue",
            label="mean SNR (valid windows)")
    ax.plot(t_min[~valid], mean_snr[~valid], lw=0.6, color="lightblue",
            alpha=0.5, label="mean SNR (sparse windows)")
    ax.axhline(overall_mean, color="tomato", lw=1.5, ls="--",
               label=f"overall mean = {overall_mean:.2f}")
    ax.fill_between(
        [t_min[0], t_min[-1]], snr_min, snr_max,
        color="tomato", alpha=0.10,
        label=f"full range  [{snr_min:.2f}, {snr_max:.2f}]",
    )
    ax.set_ylabel("Mean SNR")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Rolling SNR  –  mean per window")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.fill_between(t_min, counts, step="mid", alpha=0.6, color="steelblue",
                     label="spike count")
    ax2.axhline(min_spikes, color="red", lw=1.0, ls="--",
                label=f"min_spikes = {min_spikes}")
    ax2.set_ylabel("Spikes / window")
    ax2.set_xlabel("Time (min)")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / f"rolling_snr_over_time_{PEAKS_NAME}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {plot_path}")

    summary_text = f"""Rolling SNR Analysis Summary
==============================

Recording Parameters:
  Total duration    : {params['total_s']/60:.1f} minutes
  Sampling rate     : {params['fs']:.0f} Hz
  Channels          : {params['n_channels']}

Window Settings:
  Window size       : {params['window_s']:.1f} seconds
  Step size         : {params['step_s']:.1f} seconds
  Min spikes/window : {min_spikes}

Paths:
  Output dir        : {params['output_dir']}
  Shared noise dir  : {params.get('shared_noise_dir', 'N/A')}

Results:
  Total windows     : {len(t)}
  Valid windows     : {valid.sum()}  (with >= {min_spikes} spikes)
  Total peaks       : {params['n_peaks_total']:,}

SNR Statistics (over valid windows):
  Mean SNR          : {overall_mean:.3f}
  Median SNR        : {np.nanmedian(mean_snr[valid]):.3f}
  Min SNR           : {snr_min:.3f}
  Max SNR           : {snr_max:.3f}
  Std Dev           : {np.nanstd(mean_snr[valid]):.3f}

Spike Statistics:
  Mean spikes/win   : {counts[valid].mean():.1f}
  Median spikes/win : {np.median(counts[valid]):.1f}
  Max spikes/win    : {counts.max()}
  Min spikes/win    : {counts[valid].min()}
"""

    summary_path = out_dir / f"rolling_snr_summary_{peaks_name}.txt"
    with open(summary_path, "w") as _f:
        _f.write(summary_text)
    print(f"    Saved: {summary_path}")

    print(f"\n  Mean SNR : {overall_mean:.3f}  "
          f"(+{snr_max - overall_mean:.3f} / -{overall_mean - snr_min:.3f})  "
          f"range [{snr_min:.3f}, {snr_max:.3f}]")
    print("✓ Analysis complete")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    results = compute_rolling_snr()
    analyze_rolling_snr(OUTPUT_DIR, peaks_name=PEAKS_NAME)