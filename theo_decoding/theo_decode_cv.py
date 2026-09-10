#!/usr/bin/env python3
"""
LEGACY -- predecessor to theo_pipeline_new.py. Superseded: theo_pipeline_new.py
is a strict superset (per-model hyperparameter flags, the per-fold LagMat
leakage fix, --window-before/--window-after in SECONDS replacing
--bin-before/--bin-after, SLURM-array batch mode). Nothing imports this and its
only paired analysis notebook (theo_pipeline_analysis.ipynb) is also legacy.
Kept for reference. Use theo_pipeline_new.py via run_theo_pipeline_raven.sh.
"""

import sys

# In this environment, torch's optimizer construction (torch.optim.Adam.__init__,
# used by skorch/FNN) lazily imports torch._dynamo on its first call, which in
# turn imports triton purely to check availability (torch/utils/_triton.py's
# has_triton_package()) -- and triton's native extension segfaults/hangs on
# load here (torch==2.12.1+cu130 / triton==3.7.1 on this machine's GPU driver),
# regardless of whether training ever touches the GPU or calls torch.compile.
# Blocking the import (rather than uninstalling triton, which would touch the
# shared conda env) makes it fail as a clean, caught ImportError instead.
sys.modules.setdefault("triton", None)

# stdout is piped through `tee` (see run_theo_decode_cv.sh), not a tty, so
# CPython fully block-buffers it by default -- prints can sit invisible in
# the buffer for a long time instead of reaching the log file as they
# happen. Line-buffer explicitly so per-fold progress below (and everything
# else) shows up in the log in real time, not just at process exit.
sys.stdout.reconfigure(line_buffering=True)

import argparse
import datetime
import json
import platform
import subprocess
import time
import pickle
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spikeinterface.full as si
from sklearn.base import clone
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from Neural_Decoding.nonlinear import WienerCascade
from Neural_Decoding.preprocessing_funcs import LagMat, bin_spikes

DEFAULT_RIDGE_ALPHAS = np.logspace(-2, 4, 25)

WC_DEGREE = 3

SVR_C = 3.0
SVR_MAX_ITER = 50000


def _package_version(module_name):
    try:
        return __import__(module_name).__version__
    except Exception:
        return None


def _git_info():
    script_dir = str(Path(__file__).resolve().parent)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=script_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=script_dir, stderr=subprocess.DEVNULL
        ).decode().strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def build_run_params(
    args, pc_cols, selected_models, resolved_motion_path, drift_corrected, resolved_localized_peaks_path,
    channel_info,
):
    """Everything needed to reproduce this run: full CLI args, resolved
    settings decided at runtime (e.g. auto-derived motion path), the
    hyperparameters hardcoded per model in build_models()/make_xgb_estimator()
    (not exposed as CLI flags, so they'd otherwise be invisible in the output),
    and the software environment. Saved to the output dir up front, before the
    long CV loop, so it survives even if a later model/fold crashes.
    """
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "script": str(Path(__file__).resolve()),
        "hostname": platform.node(),
        "python_version": sys.version,
        "cli_args": vars(args),
        "resolved": {
            "pc_cols": pc_cols,
            "selected_models": selected_models,
            "motion_path": str(resolved_motion_path) if resolved_motion_path else None,
            "localized_peaks_path": str(resolved_localized_peaks_path) if resolved_localized_peaks_path else None,
            "drift_corrected": drift_corrected,
            "channels": channel_info,
        },
        "model_hyperparams": {
            "WF": {"regressor": "RidgeCV", "alphas": DEFAULT_RIDGE_ALPHAS.tolist()},
            "WC": {"regressor": "WienerCascade", "degree": WC_DEGREE},
            "SVR": {"regressor": "SVR", "C": SVR_C, "max_iter": SVR_MAX_ITER},
            "XGB": {"regressor": "XGBRegressor", **xgb_params(args)},
        },
        "package_versions": {
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "sklearn": _package_version("sklearn"),
            "xgboost": _package_version("xgboost"),
            "spikeinterface": _package_version("spikeinterface"),
            "joblib": _package_version("joblib"),
        },
        "git": _git_info(),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Run decoding CV on channel-wise MUA + PC outputs")
    p.add_argument("--recording-path", required=True)
    p.add_argument("--peaks-path", required=True)
    p.add_argument("--behavior-df-path", required=True)
    p.add_argument("--metadata-df-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--animal", default="GLC6962")
    p.add_argument("--background", default="eink")
    p.add_argument("--pc-cols", default="PC1,PC2,PC3,PC4,PC5")
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--bin-before", type=int, default=6)
    p.add_argument("--bin-current", type=int, default=1)
    p.add_argument("--bin-after", type=int, default=6)
    p.add_argument("--n-splits", type=int, default=3)
    p.add_argument("--models", default="WF,WC,SVR,XGB")
    p.add_argument("--normalize-pcs-separately", action="store_true")
    p.add_argument("--fnn-max-epochs", type=int, default=10)
    p.add_argument("--fnn-batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--motion-path",
        default=None,
        help="SpikeInterface Motion folder (si.Motion.load()). Defaults to "
        "<peaks-path's parent>/motion_estimates/motion_dredge, matching the "
        "peak set given in --peaks-path.",
    )
    p.add_argument(
        "--localized-peaks-path",
        default=None,
        help="Structured .npy with per-peak localized (x, y, z, alpha), index-aligned "
        "1:1 with --peaks-path (same length, same order -- e.g. produced alongside "
        "peaks_combined2_sorted.npy in the same thresholding_peaks* folder). Used as "
        "the depth input to drift correction instead of each peak's discrete home-"
        "channel depth. Defaults to <peaks-path's parent>/localized_peaks_mt_raw.npy.",
    )
    p.add_argument(
        "--skip-drift-correction",
        action="store_true",
        help="Skip motion/drift correction and use each peak's originally detected "
        "channel_index as-is (old behavior).",
    )
    p.add_argument(
        "--channel-smooth-window",
        type=int,
        default=0,
        help="Smooth active channels by sliding a window of this many (depth-sorted) "
        "channels before binning MUA, e.g. 20. This pipeline was built for ~10-20 "
        "sorted single units; with raw per-channel MUA as 'units' instead, neighboring "
        "channels (~20um apart) are highly spatially correlated rather than "
        "independent, and the feature count vastly exceeds this dataset's sample "
        "count. Each virtual channel's per-bin value is the MEAN spike count across "
        "its member channels. Default 0 disables smoothing (one feature per channel, "
        "old behavior). See --channel-smooth-overlap for the window's step size; the "
        "resulting number of windows is printed at runtime and saved, along with "
        "per-window membership and spike times, to <output-dir>/channel_grouping.npz.",
    )
    p.add_argument(
        "--channel-smooth-overlap",
        type=int,
        default=0,
        help="Number of channels shared between consecutive smoothing windows (must "
        "be < --channel-smooth-window). 0 = windows exactly tile the depth axis with "
        "no overlap; larger = more overlap between neighboring windows (heavier "
        "smoothing). Only used when --channel-smooth-window > 0.",
    )
    return p.parse_args()


def to_seconds(series: pd.Series) -> np.ndarray:
    if np.issubdtype(series.dtype, np.timedelta64):
        return series.dt.total_seconds().to_numpy(dtype=float)
    if np.issubdtype(series.dtype, np.datetime64):
        return (series - series.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    return series.to_numpy(dtype=float)


def bin_output_safe(outputs, output_times, dt, t_start, t_end):
    edges = np.arange(t_start, t_end, dt)
    num_bins = edges.shape[0] - 1
    out_dim = outputs.shape[1]
    out = np.full((num_bins, out_dim), np.nan, dtype=float)
    for i in range(num_bins):
        m = (output_times >= edges[i]) & (output_times < edges[i + 1])
        if np.any(m):
            out[i, :] = np.nanmean(outputs[m, :], axis=0)
    return out, edges


# --- Drift/motion correction -------------------------------------------------
# Each peak's position is corrected by the motion estimate at that position's
# depth and the peak's time, then snapped to whichever channel is now
# physically nearest. That snapped channel becomes the peak's corrected
# channel_index for MUA binning.

def get_corrected_positions(peak_times_sec, positions, motion_obj):
    """positions: (n_peaks, 2) [x, y] home-channel location of each peak.
    peak_times_sec must already be in the recording's ABSOLUTE time frame
    (matching motion_obj.temporal_bins_s), not relative to t_start.
    """
    if motion_obj is None:
        return positions.copy()
    depths = positions[:, 1]
    shifts = motion_obj.get_displacement_at_time_and_depth(peak_times_sec, depths)
    corrected = positions.copy()
    corrected[:, 1] = depths - shifts
    return corrected


def snap_to_nearest_channel(positions, channel_positions, chunk_size=10000):
    """Snap each (x, y) position to the nearest channel (no exclusions)."""
    if len(positions) == 0:
        return np.empty((0,), dtype=int)
    snapped_ch_idx = np.empty(len(positions), dtype=int)
    for start in range(0, len(positions), chunk_size):
        stop = min(start + chunk_size, len(positions))
        chunk = positions[start:stop]
        diff = chunk[:, None, :] - channel_positions[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        snapped_ch_idx[start:stop] = np.argmin(dist2, axis=1)
    return snapped_ch_idx


def group_channels_sliding(spike_times_active, active_channel_idx, channel_positions, window_channels, overlap_channels):
    """Slide a window of `window_channels` channels (sorted by depth), moving
    by `window_channels - overlap_channels` each step, to pool
    `spike_times_active` (already filtered to active_channel_idx) into
    overlapping virtual channels covering every active channel.

    Returns (spike_times_grouped, member_channel_idx, member_counts,
    depth_center_um, step_channels): first four have length n_groups (however
    many windows this geometry produces -- see module comment above);
    member_channel_idx[g] holds the ORIGINAL probe channel indices pooled
    into group g; member_counts[g] = len(member_channel_idx[g]).
    """
    if overlap_channels >= window_channels:
        raise ValueError(
            f"--channel-smooth-overlap ({overlap_channels}) must be < "
            f"--channel-smooth-window ({window_channels})"
        )

    active_pos = channel_positions[active_channel_idx]
    order = np.lexsort((active_pos[:, 0], active_pos[:, 1]))  # sort by y, tie-break x
    sorted_idx = active_channel_idx[order]
    sorted_y = active_pos[order, 1]
    sorted_spike_times = spike_times_active[order]
    n_active = sorted_idx.size

    window = min(window_channels, n_active)
    step = window - overlap_channels

    starts = list(range(0, n_active - window + 1, step))
    if not starts:
        starts = [0]
    tail_topped_up = False
    tail_overlap = overlap_channels
    if starts[-1] + window < n_active:
        # Regular striding leaves the last (n_active - (starts[-1] + window))
        # channels short of a full window -- rather than drop them, anchor
        # one final window to the END so every active channel is covered.
        # This can make its overlap with the previous window bigger than the
        # requested overlap_channels (never smaller/dropped).
        new_start = n_active - window
        tail_overlap = (starts[-1] + window) - new_start
        starts.append(new_start)
        tail_topped_up = True

    spike_times_grouped, member_channel_idx, member_counts, depth_center_um = [], [], [], []
    for lo in starts:
        hi = lo + window
        spike_times_grouped.append(np.concatenate(sorted_spike_times[lo:hi]))
        member_channel_idx.append(sorted_idx[lo:hi])
        member_counts.append(hi - lo)
        depth_center_um.append(float(sorted_y[lo:hi].mean()))

    return (
        np.asarray(spike_times_grouped, dtype=object),
        member_channel_idx,
        np.asarray(member_counts, dtype=int),
        np.asarray(depth_center_um, dtype=float),
        step,
        tail_topped_up,
        tail_overlap,
    )


def plot_drift_scatter(out_path, peak_times_abs, depths, title, fold_boundary_times):
    # All times here are ABSOLUTE (same convention as t_centers/fold_boundary_times
    # elsewhere in this script and as motion_obj.temporal_bins_s), not
    # relative to t_start -- keeps every diagnostic plot on the same x-axis.
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.scatter(peak_times_abs, depths, s=1, color="k", alpha=0.02, rasterized=True)
    for bt in fold_boundary_times:
        ax.axvline(bt, color="tab:red", lw=0.8, ls=":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Depth (um)")
    ax.set_title(title)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_motion_overlay(out_path, peak_times_abs, depths, motion_obj, fold_boundary_times):
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.scatter(peak_times_abs, depths, s=1, color="k", alpha=0.02, rasterized=True, label="Peaks (localized depth)")

    colors = ["red", "royalblue", "darkgreen", "darkorange", "deeppink", "saddlebrown", "teal"]
    t_bins_abs = motion_obj.temporal_bins_s[0]  # already absolute, same frame as peak_times_abs
    d_bins = motion_obj.spatial_bins_um
    displ = motion_obj.displacement[0]
    for di, start_d in enumerate(d_bins):
        drift_line = start_d + displ[:, di]
        ax.plot(
            t_bins_abs, drift_line,
            color=colors[di % len(colors)], lw=1.0, alpha=0.8,
            label="DREDge motion" if di == 0 else "",
        )
    for bt in fold_boundary_times:
        ax.axvline(bt, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Depth (um)")
    ax.set_title("Raw peak depths + motion (DREDge) overlay")
    ax.legend(loc="upper right", markerscale=8, fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def xgb_params(args):
    # Single source of truth for XGB hyperparams (XGB is not exposed via CLI
    # flags like the other models' knobs). Shared by make_xgb_estimator() --
    # both the production and diagnostic loss-curve fits -- and by
    # build_run_params()'s reproducibility dump, so the two can never drift.
    return dict(
        objective="reg:squarederror",
        max_depth=3,
        n_estimators=200,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=args.seed,
        n_jobs=1,
    )


def make_xgb_estimator(args, eval_metric=None):
    from xgboost import XGBRegressor

    kwargs = xgb_params(args)
    if eval_metric is not None:
        kwargs["eval_metric"] = eval_metric
    return XGBRegressor(**kwargs)


def build_models(args, y_dim, selected_models):
    # NOTE: LagMat is no longer a pipeline step here. It is applied once,
    # globally, in chronological order in main() before the CV split (see
    # X_lagged), instead of independently within each fold's train/test
    # slice. Building it per-fold-slice both (a) zero-pads real history at
    # fold edges instead of using the true neighboring bins, and (b) for any
    # fold whose training set is two non-adjacent chronological blocks
    # stitched together (e.g. train = block1 + block3 with the held-out
    # block2 removed from the middle), it silently manufactures lag features
    # that splice together two temporally distant bins as if they were
    # adjacent. Computing it once on the full, correctly time-ordered array
    # avoids both issues; only StandardScaler still needs to be fit per-fold
    # (on the training slice only) to avoid real leakage of test statistics.
    models = {}

    if "WF" in selected_models:
        models["WF"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", MultiOutputRegressor(RidgeCV(alphas=DEFAULT_RIDGE_ALPHAS))),
            ]
        )

    if "FNN" in selected_models:
        from skorch import NeuralNetRegressor
        from torch import optim
        from Neural_Decoding.nn import FNN

        models["FNN"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "fnn",
                    NeuralNetRegressor(
                        module=FNN,
                        lr=0.001,
                        iterator_train__shuffle=True,
                        optimizer=optim.Adam,
                        batch_size=args.fnn_batch_size,
                        module__n_targets=y_dim,
                        module__num_units=400,
                        module__frac_dropout=0.25,
                        module__n_layers=2,
                        max_epochs=args.fnn_max_epochs,
                        verbose=0,
                    ),
                ),
            ]
        )

    if "WC" in selected_models:
        models["WC"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("wc", MultiOutputRegressor(WienerCascade(degree=WC_DEGREE))),
            ]
        )

    if "SVR" in selected_models:
        models["SVR"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svr", MultiOutputRegressor(SVR(C=SVR_C, max_iter=SVR_MAX_ITER))),
            ]
        )

    if "XGB" in selected_models:
        try:
            models["XGB"] = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("xgb", MultiOutputRegressor(make_xgb_estimator(args))),
                ]
            )
        except Exception:
            pass

    return models


def plot_decoded_vs_observed(out_path, t_centers, y_true, y_pred, pc_cols, model_name):
    n_out = y_true.shape[1]
    fig, axes = plt.subplots(n_out, 1, figsize=(14, 2.3 * n_out), sharex=True)
    if n_out == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        ax.plot(t_centers, y_true[:, i], color="black", lw=0.8, label="Observed")
        ax.plot(t_centers, y_pred[:, i], color="tab:red", lw=0.8, alpha=0.9, label="Decoded")
        ax.set_ylabel(f"{pc_cols[i]}\n(R2={r2:.2f})")
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Decoded vs Observed ({model_name})", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_pred_vs_obs(out_path, y_true, y_pred, pc_cols, model_name):
    """Predicted-vs-observed scatter per output, with a y=x reference line.

    A tight diagonal cloud = good fit; a flat blob/scatter = the model isn't
    tracking the real signal. Easier to read at a glance than the time-series
    overlay, especially for spotting whether a model has just collapsed to
    predicting near-constant values.
    """
    n_out = y_true.shape[1]
    fig, axes = plt.subplots(1, n_out, figsize=(3.4 * n_out, 3.4))
    if n_out == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        yt, yp = y_true[:, i], y_pred[:, i]
        r2 = r2_score(yt, yp)
        lo = float(min(np.nanmin(yt), np.nanmin(yp)))
        hi = float(max(np.nanmax(yt), np.nanmax(yp)))
        ax.scatter(yt, yp, s=4, alpha=0.25, color="tab:blue", edgecolors="none")
        ax.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Decoded")
        ax.set_title(f"{pc_cols[i]} (R2={r2:.2f})")
        ax.grid(alpha=0.2)

    fig.suptitle(f"Decoded vs Observed scatter ({model_name}, OOF)", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_residuals_over_time(out_path, t_centers, y_true, y_pred, pc_cols, model_name, fold_boundary_times):
    """Decoded-minus-observed residual over the session.

    Flags *when* the model goes wrong (e.g. a specific stretch of the
    recording), which the aggregate R2 number hides. Dotted vertical lines
    mark internal CV fold boundaries so a bad stretch can be tied back to a
    specific fold (e.g. fold 0 being much worse than the others).
    """
    n_out = y_true.shape[1]
    resid = y_pred - y_true
    fig, axes = plt.subplots(n_out, 1, figsize=(14, 2.0 * n_out), sharex=True)
    if n_out == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t_centers, resid[:, i], color="tab:red", lw=0.6)
        ax.axhline(0, color="black", lw=0.8)
        for bt in fold_boundary_times:
            ax.axvline(bt, color="gray", lw=0.8, ls=":")
        ax.set_ylabel(f"{pc_cols[i]}\nresidual")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Decoded - Observed residuals over time ({model_name}, OOF)\n"
        "(dotted lines = CV fold boundaries)",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_firing_rate_heatmap(out_path, X, t_centers, fold_boundary_times):
    """QC plot, independent of any model: per-channel spike counts over time.

    Useful for spotting whether firing rates jump during specific periods
    (e.g. movement bouts / motion artifacts) or whether some channels are
    dead/saturated throughout.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(
        X.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[t_centers[0], t_centers[-1], 0, X.shape[1]],
    )
    for bt in fold_boundary_times:
        ax.axvline(bt, color="white", lw=1, ls=":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title(
        "MUA spike counts per bin, per channel\n"
        "(QC: watch for artifact bursts / dead channels; dotted lines = CV fold boundaries)"
    )
    fig.colorbar(im, ax=ax, label="Spikes per bin")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_density_before_after(
    out_path, active_channel_idx, channel_positions, active_channel_counts, duration_s,
    group_member_channel_idx=None, group_depth_center_um=None,
):
    """Spike density along probe depth, before vs after channel smoothing --
    same spirit as peak_detect_localise/spike_density_count.py's per-depth
    rate strip: y = depth (um), x = spike rate (spikes/s), one horizontal bar
    per depth resolved at each stage.

    "Before" sums counts across active channels sharing the same y (this
    probe has ~2 columns/depth row), at native ~row-pitch resolution.
    "After" sums counts across each smoothing window's member channels, at
    window resolution. This is the raw local rate for visualization -- a
    different (unnormalized) quantity than the per-member-channel MEAN used
    as the actual model feature (see group_channels_sliding), and NOT a
    strict repartition of "before" when windows overlap (a channel's spikes
    can then be counted in more than one window's bar).
    """
    active_y = channel_positions[active_channel_idx, 1]
    unique_y = np.sort(np.unique(active_y))
    row_pitch = float(np.min(np.diff(unique_y))) if unique_y.size > 1 else 20.0
    row_rate = np.array([active_channel_counts[active_y == y].sum() for y in unique_y]) / duration_s

    show_after = group_member_channel_idx is not None
    fig, axes = plt.subplots(1, 2 if show_after else 1, figsize=(11 if show_after else 5.5, 8), sharey=True)
    axes = np.atleast_1d(axes)

    axes[0].barh(unique_y, row_rate, height=row_pitch * 0.9, color="steelblue", edgecolor="none")
    axes[0].set_xlabel("Spike rate (spikes/s)")
    axes[0].set_ylabel("Depth (um)")
    axes[0].set_title(f"Before smoothing\n({unique_y.size} depth rows, ~{row_pitch:.0f}um pitch)")
    axes[0].grid(alpha=0.2, axis="x")

    if show_after:
        idx_to_count = dict(zip(active_channel_idx.tolist(), active_channel_counts.tolist()))
        group_rate = np.array(
            [sum(idx_to_count[int(ch)] for ch in members) for members in group_member_channel_idx]
        ) / duration_s
        group_span_um = np.array(
            [
                max(float(np.ptp(channel_positions[members, 1])), row_pitch) if len(members) > 1 else row_pitch
                for members in group_member_channel_idx
            ]
        )
        axes[1].barh(group_depth_center_um, group_rate, height=group_span_um * 0.9, color="tab:orange", edgecolor="none")
        axes[1].set_xlabel("Spike rate (spikes/s)")
        axes[1].set_title(f"After smoothing\n({len(group_member_channel_idx)} windows)")
        axes[1].grid(alpha=0.2, axis="x")

    fig.suptitle(
        "Spike density along probe depth: before vs after channel smoothing"
        if show_after else "Spike density along probe depth (channel smoothing off)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_xgb_loss_curves(out_path, curves, pc_cols, fold):
    """XGB's genuine per-boosting-round train vs held-out loss curve.

    This is the one model here that trains iteratively (one tree per round),
    so it's the only one with a real "convergence" curve to plot. If the
    held-out curve rises while the train curve keeps falling, that's a direct
    picture of overfitting -- the same failure mode diagnosed for WF/WC, but
    XGB's regularization (depth/shrinkage) keeps it much less severe.
    """
    n_out = len(pc_cols)
    fig, axes = plt.subplots(1, n_out, figsize=(3.6 * n_out, 3.2))
    if n_out == 1:
        axes = [axes]

    for i, col in enumerate(pc_cols):
        res = curves[col]
        train_curve = res["validation_0"]["rmse"]
        val_curve = res["validation_1"]["rmse"]
        rounds = np.arange(1, len(train_curve) + 1)
        ax = axes[i]
        ax.plot(rounds, train_curve, label="train", color="tab:blue")
        ax.plot(rounds, val_curve, label="held-out", color="tab:red")
        ax.set_title(col)
        ax.set_xlabel("Boosting round")
        if i == 0:
            ax.set_ylabel("RMSE")
            ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle(
        f"XGB train vs held-out loss curve (fold {fold})\n"
        "(held-out rising while train falls = overfitting)",
        y=1.06,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_xgb_channel_importance(out_path, channel_importance, x_positions, x_label, bar_width=1.0):
    """Per-channel (or per-depth-group) XGB importance, summed over lags and
    averaged over folds+outputs.

    Shows whether a handful of channels/groups actually drive predictions, or
    whether importance is smeared meaninglessly across all of them (which
    would be consistent with a mostly-noise feature set).

    x_positions/x_label are supplied by the caller so this reads correctly in
    both modes: original probe channel index (gaps = dropped zero-spike
    channels) when --channel-group-width-um is off, or depth-bin center in
    microns when channel grouping is on.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(x_positions, channel_importance, color="tab:green", width=bar_width)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean importance\n(summed over lags, avg over folds+outputs)")
    ax.set_title("XGB feature importance by channel")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_train_vs_test_r2(out_path, summary):
    """Train (in-sample) vs held-out (OOF) R2 per model, averaged over outputs.

    This is the single most direct overfitting check: if a model's train R2
    is near 1.0 while its test R2 is deeply negative, that gap *is* the
    overfitting diagnosis, visually, no need to infer it from side effects
    like RankWarning/ConvergenceWarning messages.
    """
    names = list(summary["models"].keys())
    if not names:
        return
    train_means = [np.mean(summary["models"][n]["mean_fold_train_r2_per_output"]) for n in names]
    test_means = [np.mean(summary["models"][n]["mean_fold_r2_per_output"]) for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2, 4))
    ax.bar(x - width / 2, train_means, width, label="Train R2 (in-sample)", color="tab:blue")
    ax.bar(x + width / 2, test_means, width, label="Test R2 (held-out, OOF)", color="tab:red")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("R2 (averaged across outputs)")
    ax.set_title("Train vs held-out R2 per model\n(large gap = overfitting)")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    models_dir = out_dir / "models"
    preds_dir = out_dir / "predictions"
    figs_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    rec = si.load(args.recording_path)
    peaks = np.load(args.peaks_path, allow_pickle=True)

    channel_ids = rec.get_channel_ids()
    channel_positions = rec.get_channel_locations()
    fs = float(rec.get_sampling_frequency())
    time_offset = float(rec.get_times()[0])

    peak_channel_index = peaks["channel_index"].astype(int)
    peak_times_sec = peaks["sample_index"].astype(float) / fs + time_offset

    n_channels = channel_ids.shape[0]
    if peak_channel_index.min() >= 0 and peak_channel_index.max() < n_channels:
        peak_row_index = peak_channel_index
        mapping_mode = "direct row index"
    else:
        ap_numbers = np.array([int(str(cid).split("AP")[-1]) for cid in channel_ids], dtype=int)
        ap_to_row = {ap: i for i, ap in enumerate(ap_numbers)}
        missing = np.setdiff1d(np.unique(peak_channel_index), ap_numbers)
        if missing.size > 0:
            raise ValueError(f"Cannot map channel_index values. First missing: {missing[:10]}")
        peak_row_index = np.array([ap_to_row[ch] for ch in peak_channel_index], dtype=int)
        mapping_mode = "mapped AP number -> row index"

    # --- Drift/motion correction (see get_corrected_positions/snap_to_nearest_channel) ---
    motion = None
    motion_path = None
    localized_peaks_path = None
    localized_y = None
    drift_corrected = False
    if not args.skip_drift_correction:
        motion_path = Path(args.motion_path) if args.motion_path else (
            Path(args.peaks_path).parent / "motion_estimates" / "motion_dredge"
        )
        localized_peaks_path = Path(args.localized_peaks_path) if args.localized_peaks_path else (
            Path(args.peaks_path).parent / "localized_peaks_mt_raw.npy"
        )
        if motion_path.exists() and localized_peaks_path.exists():
            motion = si.Motion.load(str(motion_path))
            print(f"Loaded motion estimate: {motion_path}")
            localized_peaks = np.load(str(localized_peaks_path), allow_pickle=True)
            if len(localized_peaks) != len(peaks):
                raise ValueError(
                    f"--localized-peaks-path has {len(localized_peaks):,} rows but "
                    f"--peaks-path has {len(peaks):,}; they must be index-aligned "
                    "(same peak set, same order)."
                )
            localized_y = localized_peaks["y"].astype(float)
            print(f"Loaded localized peaks: {localized_peaks_path}")
            drift_corrected = True
        elif not motion_path.exists():
            print(f"WARNING: motion path {motion_path} not found -- skipping drift correction.")
        else:
            print(f"WARNING: localized peaks path {localized_peaks_path} not found -- skipping drift correction.")

    texture_pca_df = pd.read_pickle(args.behavior_df_path)
    # Loaded but not consumed downstream: reads it here purely to fail fast on a
    # bad/unreadable --metadata-df-path before the long CV loop starts.
    pd.read_pickle(args.metadata_df_path)

    pc_cols = [c.strip() for c in args.pc_cols.split(",") if c.strip()]
    missing_cols = [c for c in pc_cols if c not in texture_pca_df.columns]
    if missing_cols:
        raise ValueError(f"Missing PC columns: {missing_cols}")

    if "background" in texture_pca_df.columns:
        beh = texture_pca_df[
            (texture_pca_df["animal"] == args.animal)
            & (texture_pca_df["background"].astype(str).str.lower() == args.background.lower())
        ].copy()
    else:
        beh = texture_pca_df[texture_pca_df["animal"] == args.animal].copy()

    if beh.empty:
        raise ValueError("No behavior rows after filtering animal/background")

    beh_t = to_seconds(beh["time"])
    beh_outputs = beh[pc_cols].to_numpy(dtype=float)

    neural_start_raw = float(np.min(peak_times_sec))
    neural_end_raw = float(np.max(peak_times_sec))
    beh_start_raw = float(np.min(beh_t))
    beh_end_raw = float(np.max(beh_t))

    t_start = max(neural_start_raw, beh_start_raw)
    t_end = min(neural_end_raw, beh_end_raw)
    if t_end <= t_start:
        raise ValueError("No overlap between neural and behavior")

    peak_keep = (peak_times_sec >= t_start) & (peak_times_sec < t_end)
    peak_times_trim = peak_times_sec[peak_keep]
    peak_rows_trim_raw = peak_row_index[peak_keep]
    raw_depths_trim = channel_positions[peak_rows_trim_raw, 1]

    if drift_corrected:
        localized_y_trim = localized_y[peak_keep]
        # x stays at the home channel's x (localized x is noisy across
        # shanks/columns); only the localized y (depth) is corrected.
        raw_positions_trim = np.column_stack(
            [channel_positions[peak_rows_trim_raw, 0], localized_y_trim]
        )
        corrected_positions_trim = get_corrected_positions(peak_times_trim, raw_positions_trim, motion)
        corrected_depths_trim = corrected_positions_trim[:, 1]
        peak_rows_trim = snap_to_nearest_channel(corrected_positions_trim, channel_positions)
        n_reassigned = int(np.sum(peak_rows_trim != peak_rows_trim_raw))
        print(
            f"Drift correction: {n_reassigned:,}/{len(peak_rows_trim):,} peaks "
            f"({100 * n_reassigned / len(peak_rows_trim):.1f}%) reassigned to a different channel."
        )
        # snap_to_nearest_channel is an argmin over distance to every channel,
        # so it always lands on the closest one by construction -- this just
        # prints the resulting distances as a sanity check on that claim.
        assigned_dist = np.linalg.norm(
            corrected_positions_trim - channel_positions[peak_rows_trim], axis=1
        )
        print(
            f"Post-snap distance to assigned channel: median={np.median(assigned_dist):.2f}um, "
            f"mean={np.mean(assigned_dist):.2f}um, max={np.max(assigned_dist):.2f}um."
        )
        # Raw (uncorrected) reference for the diagnostic plots below: the same
        # localized depth fed into correction, before the shift is applied --
        # keeps the raw-vs-corrected comparison apples-to-apples.
        raw_depths_trim = localized_y_trim
    else:
        corrected_depths_trim = raw_depths_trim
        peak_rows_trim = peak_rows_trim_raw

    spike_times_trim = np.asarray(
        [peak_times_trim[peak_rows_trim == ch] for ch in range(n_channels)], dtype=object
    )

    # Drop channels with zero spikes in the trimmed (neural/behavior overlap)
    # window -- they contribute nothing but dead columns to the feature
    # matrix, for free. Computed AFTER drift correction, since correction can
    # move a channel from zero spikes to some (or vice versa).
    channel_spike_counts = np.array([len(a) for a in spike_times_trim], dtype=int)
    active_channel_idx = np.flatnonzero(channel_spike_counts > 0)
    dropped_channel_idx = np.flatnonzero(channel_spike_counts == 0)
    print(
        f"Channels: using {active_channel_idx.size:,}/{n_channels:,} "
        f"({dropped_channel_idx.size:,} dropped, zero spikes in window)."
    )
    if dropped_channel_idx.size:
        print(f"Dropped channel indices: {dropped_channel_idx.tolist()}")
    print("Per-channel spike counts (ordered by channel_id AP number):")
    ap_order = sorted(range(n_channels), key=lambda i: int(str(channel_ids[i]).split("AP")[-1]))
    for ch in ap_order:
        print(f"  row {ch} (channel_id={channel_ids[ch]}): {channel_spike_counts[ch]:,} spikes")
    spike_times_trim = spike_times_trim[active_channel_idx]

    channel_grouping_path = None
    group_member_counts = None
    window_channels_used = None
    step_channels = None
    tail_topped_up = None
    tail_overlap = None
    if args.channel_smooth_window > 0:
        if args.channel_smooth_overlap >= args.channel_smooth_window:
            raise ValueError(
                f"--channel-smooth-overlap ({args.channel_smooth_overlap}) must be < "
                f"--channel-smooth-window ({args.channel_smooth_window})"
            )
        if args.channel_smooth_window > active_channel_idx.size:
            print(
                f"WARNING: --channel-smooth-window ({args.channel_smooth_window}) exceeds the "
                f"{active_channel_idx.size} active channels; using one window covering all of them."
            )
        (
            spike_times_used,
            group_member_channel_idx,
            group_member_counts,
            channel_x_positions,
            step_channels,
            tail_topped_up,
            tail_overlap,
        ) = group_channels_sliding(
            spike_times_trim, active_channel_idx, channel_positions,
            args.channel_smooth_window, args.channel_smooth_overlap,
        )
        n_channels_used = len(spike_times_used)
        window_channels_used = int(np.max(group_member_counts))
        channel_x_label = f"Depth bin center (um) -- {n_channels_used} windows"
        channel_bar_width = float(np.median(np.diff(channel_x_positions))) if n_channels_used > 1 else 1.0
        # For the human-readable "~Xum" estimate below only (not used in the
        # actual windowing math, which is purely channel-count/index based):
        # several channels can share a depth row (this probe has ~2 columns),
        # so divide by the number of DISTINCT depths, not raw channel count.
        unique_active_y = np.unique(channel_positions[active_channel_idx, 1])
        mean_pitch_um = (
            float(np.ptp(unique_active_y)) / max(unique_active_y.size - 1, 1)
        )
        print(
            f"Channel smoothing: {active_channel_idx.size:,} active channels -> "
            f"{n_channels_used:,} windows "
            f"(window={window_channels_used} channels [~{window_channels_used * mean_pitch_um:.0f}um], "
            f"step={step_channels} channels [~{step_channels * mean_pitch_um:.0f}um], "
            f"requested overlap={args.channel_smooth_overlap} channels)."
        )
        if tail_topped_up:
            print(
                f"NOTE: {active_channel_idx.size:,} active channels isn't an exact multiple of the "
                f"step -- a final window was anchored to the end so the last channels aren't dropped. "
                f"That window overlaps its neighbor by {tail_overlap} channels, not the requested "
                f"{args.channel_smooth_overlap} (window {n_channels_used - 1} of {n_channels_used}, 0-indexed)."
            )
        channel_grouping_path = out_dir / "channel_grouping.npz"
        np.savez_compressed(
            channel_grouping_path,
            channel_smooth_window=args.channel_smooth_window,
            channel_smooth_overlap=args.channel_smooth_overlap,
            step_channels=step_channels,
            tail_topped_up=tail_topped_up,
            tail_overlap=tail_overlap,
            n_active_channels_input=active_channel_idx.size,
            n_groups=n_channels_used,
            group_member_channel_idx=np.array(group_member_channel_idx, dtype=object),
            group_member_counts=group_member_counts,
            group_depth_center_um=channel_x_positions,
            group_spike_times_sec=spike_times_used,
        )
        print(f"Saved channel grouping -> {channel_grouping_path}")
    else:
        spike_times_used = spike_times_trim
        n_channels_used = active_channel_idx.size
        channel_x_positions = active_channel_idx
        channel_x_label = "Channel index (original probe index; gaps = dropped zero-spike channels)"
        channel_bar_width = 1.0

    # Single source of truth for every channel-stage fact -- both raw request
    # (--channel-smooth-window/-overlap, already in cli_args) and what was
    # actually RESOLVED at runtime (achieved window/step, whether the tail
    # needed topping up, and by how much). Written into both run_params.json
    # (saved early, survives a mid-run crash) and cv_summary.json (saved at
    # the end) so this doesn't only live in stdout/channel_grouping.npz.
    channel_info = {
        "total": int(n_channels),
        "active": int(active_channel_idx.size),
        "dropped_zero_spikes": dropped_channel_idx.tolist(),
        "channel_smooth_window_requested": args.channel_smooth_window,
        "channel_smooth_overlap_requested": (
            args.channel_smooth_overlap if args.channel_smooth_window > 0 else None
        ),
        "channel_smooth_window_used": window_channels_used,
        "channel_smooth_step_used": step_channels,
        "tail_topped_up": tail_topped_up,
        "tail_overlap": tail_overlap,
        "feature_channels": int(n_channels_used),
        "channel_grouping_path": str(channel_grouping_path) if channel_grouping_path else None,
    }

    beh_keep = (beh_t >= t_start) & (beh_t < t_end)
    beh_t_trim = beh_t[beh_keep]
    beh_outputs_trim = beh_outputs[beh_keep]

    neural_data = bin_spikes(spike_times_used, args.dt, t_start, t_end)
    if group_member_counts is not None:
        # Convert each group's summed per-bin count into a MEAN per-member-
        # channel count, so a group's value is comparable regardless of how
        # many raw channels happened to feed it (see group_channels_sliding).
        neural_data = neural_data / group_member_counts[None, :]
    y_binned, t_edges = bin_output_safe(beh_outputs_trim, beh_t_trim, args.dt, t_start, t_end)

    valid_mask = np.all(np.isfinite(y_binned), axis=1)
    X = neural_data[valid_mask].astype("float32")
    y = y_binned[valid_mask].astype("float32")
    t_centers = ((t_edges[:-1] + t_edges[1:]) / 2.0)[valid_mask]

    if args.normalize_pcs_separately:
        y_mu = np.nanmean(y, axis=0, keepdims=True)
        y_sd = np.nanstd(y, axis=0, keepdims=True) + 1e-9
        y = (y - y_mu) / y_sd

    formatted_path = out_dir / "formatted_data_like_hc.pickle"
    with open(formatted_path, "wb") as f:
        pickle.dump([X, y], f)

    # Build the lag-expanded feature matrix ONCE, in true chronological
    # order, before any CV split (see the note in build_models() for why).
    n_lags_total = args.bin_before + args.bin_current + args.bin_after
    lagger = LagMat(
        bin_before=args.bin_before,
        bin_current=args.bin_current,
        bin_after=args.bin_after,
        flat=True,
    )
    X_lagged = lagger.fit_transform(X)

    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    available_models = build_models(args, y.shape[1], selected)
    unknown = [m for m in selected if m not in available_models]
    if unknown:
        raise ValueError(f"Unknown/unavailable models requested: {unknown}. Available: {list(available_models)}")

    run_params = build_run_params(
        args, pc_cols, selected, motion_path, drift_corrected, localized_peaks_path, channel_info
    )
    with open(out_dir / "run_params.json", "w", encoding="utf-8") as f:
        json.dump(run_params, f, indent=2)
    print(f"Saved run params    -> {out_dir / 'run_params.json'}")

    kf = KFold(n_splits=args.n_splits, shuffle=False)
    fold_starts = [test_idx[0] for _, test_idx in kf.split(X_lagged)]
    fold_boundary_times = t_centers[fold_starts[1:]]  # internal boundaries only

    plot_firing_rate_heatmap(figs_dir / "firing_rate_heatmap.png", X, t_centers, fold_boundary_times)

    plot_density_before_after(
        figs_dir / "channel_density_before_after_smoothing.png",
        active_channel_idx, channel_positions, channel_spike_counts[active_channel_idx], (t_end - t_start),
        group_member_channel_idx=group_member_channel_idx if args.channel_smooth_window > 0 else None,
        group_depth_center_um=channel_x_positions if args.channel_smooth_window > 0 else None,
    )

    plot_drift_scatter(
        figs_dir / "drift_01_raw_scatter.png",
        peak_times_trim, raw_depths_trim,
        "Raw peak depths (localized, uncorrected)" if drift_corrected
        else "Raw peak depths (home channel, uncorrected)",
        fold_boundary_times,
    )
    if drift_corrected:
        plot_motion_overlay(
            figs_dir / "drift_02_motion_overlay.png",
            peak_times_trim, raw_depths_trim, motion, fold_boundary_times,
        )
        plot_drift_scatter(
            figs_dir / "drift_03_corrected_scatter.png",
            peak_times_trim, corrected_depths_trim,
            "Motion-corrected peak depths (localized depth, before channel-snap)",
            fold_boundary_times,
        )
        snapped_depths_trim = channel_positions[peak_rows_trim, 1]
        plot_drift_scatter(
            figs_dir / "drift_04_corrected_snapped_scatter.png",
            peak_times_trim, snapped_depths_trim,
            "Motion-corrected peak depths (snapped to nearest channel)",
            fold_boundary_times,
        )

    summary = {
        "config": {
            "recording_path": args.recording_path,
            "peaks_path": args.peaks_path,
            "behavior_df_path": args.behavior_df_path,
            "animal": args.animal,
            "background": args.background,
            "pc_cols": pc_cols,
            "dt": args.dt,
            "bin_before": args.bin_before,
            "bin_current": args.bin_current,
            "bin_after": args.bin_after,
            "n_splits": args.n_splits,
            "normalize_pcs_separately": args.normalize_pcs_separately,
            "mapping_mode": mapping_mode,
            "drift_corrected": drift_corrected,
            "motion_path": str(motion_path) if motion_path else None,
            "localized_peaks_path": str(localized_peaks_path) if localized_peaks_path else None,
            "time_offset": time_offset,
            "raw_ranges": {
                "neural": [neural_start_raw, neural_end_raw],
                "behavior": [beh_start_raw, beh_end_raw],
                "trimmed_overlap": [t_start, t_end],
            },
            "channels": channel_info,
            "shapes": {"X": list(X.shape), "y": list(y.shape), "X_lagged": list(X_lagged.shape)},
        },
        "models": {},
    }

    for name in selected:
        base_model = available_models[name]
        oof_pred = np.full_like(y, np.nan, dtype=float)
        fold_test_scores = []
        fold_train_scores = []
        xgb_channel_importances = []

        print(f"Model {name}: starting ({args.n_splits} folds)...")
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_lagged)):
            fold_start = time.time()
            print(f"Model {name}: starting fold {fold + 1}/{args.n_splits}...")
            model = clone(base_model)
            model.fit(X_lagged[train_idx], y[train_idx])
            pred = model.predict(X_lagged[test_idx])
            oof_pred[test_idx] = pred

            test_score = r2_score(y[test_idx], pred, multioutput="raw_values")
            fold_test_scores.append(test_score.tolist())

            # In-sample (train) R2: the direct overfitting check -- compare
            # against test_score above. A big train-vs-test gap IS overfitting.
            train_pred = model.predict(X_lagged[train_idx])
            train_score = r2_score(y[train_idx], train_pred, multioutput="raw_values")
            fold_train_scores.append(train_score.tolist())

            joblib.dump(model, models_dir / f"{name}_fold{fold}.joblib")

            if name == "XGB":
                # Diagnostic-only refit per output with eval_set, purely to get a
                # real per-boosting-round train/held-out loss curve (XGB is the
                # only model here trained iteratively, so it's the only one
                # that has a genuine "convergence" curve to show). Uses the
                # same fitted scaler as the production model above.
                scaler_fitted = model.named_steps["scaler"]
                Xtr_t = scaler_fitted.transform(X_lagged[train_idx])
                Xte_t = scaler_fitted.transform(X_lagged[test_idx])
                curves = {}
                for i, col in enumerate(pc_cols):
                    diag_reg = make_xgb_estimator(args, eval_metric="rmse")
                    diag_reg.fit(
                        Xtr_t,
                        y[train_idx][:, i],
                        eval_set=[(Xtr_t, y[train_idx][:, i]), (Xte_t, y[test_idx][:, i])],
                        verbose=False,
                    )
                    curves[col] = diag_reg.evals_result()
                plot_xgb_loss_curves(figs_dir / f"XGB_loss_curve_fold{fold}.png", curves, pc_cols, fold)

                fitted_xgb = model.named_steps["xgb"]
                fold_importance = np.array(
                    [est.feature_importances_ for est in fitted_xgb.estimators_]
                )  # (n_outputs, n_lags_total * n_channels_used)
                per_output_channel_importance = fold_importance.reshape(
                    len(pc_cols), n_lags_total, n_channels_used
                ).sum(axis=1)  # (n_outputs, n_channels_used)
                xgb_channel_importances.append(per_output_channel_importance)

            print(f"Model {name}: finished fold {fold + 1}/{args.n_splits} in {time.time() - fold_start:.1f}s")

        overall = r2_score(y, oof_pred, multioutput="raw_values")
        mean_train = np.mean(np.asarray(fold_train_scores), axis=0)
        mean_test = np.mean(np.asarray(fold_test_scores), axis=0)

        np.savez_compressed(
            preds_dir / f"{name}_oof_predictions.npz",
            t=t_centers,
            y_true=y,
            y_pred=oof_pred,
            feature_names=np.array(pc_cols, dtype=object),
        )

        plot_decoded_vs_observed(
            figs_dir / f"{name}_decoded_vs_observed.png",
            t_centers,
            y,
            oof_pred,
            pc_cols,
            name,
        )
        plot_scatter_pred_vs_obs(
            figs_dir / f"{name}_scatter_pred_vs_obs.png", y, oof_pred, pc_cols, name
        )
        plot_residuals_over_time(
            figs_dir / f"{name}_residuals_over_time.png",
            t_centers,
            y,
            oof_pred,
            pc_cols,
            name,
            fold_boundary_times,
        )

        if name == "XGB" and xgb_channel_importances:
            mean_channel_importance = np.mean(np.stack(xgb_channel_importances, axis=0), axis=(0, 1))
            plot_xgb_channel_importance(
                figs_dir / "XGB_channel_importance.png",
                mean_channel_importance,
                channel_x_positions,
                channel_x_label,
                bar_width=channel_bar_width,
            )

        summary["models"][name] = {
            "fold_r2_raw_values": fold_test_scores,
            "mean_fold_r2_per_output": mean_test.tolist(),
            "oof_r2_per_output": overall.tolist(),
            "fold_train_r2_raw_values": fold_train_scores,
            "mean_fold_train_r2_per_output": mean_train.tolist(),
        }

        print(f"{name} OOF R2 per output   (held-out):                     {np.round(overall, 4)}")
        print(f"{name} mean TRAIN R2 per output (in-sample, overfitting check): {np.round(mean_train, 4)}")

    plot_train_vs_test_r2(figs_dir / "train_vs_test_r2_summary.png", summary)

    with open(out_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved formatted data -> {formatted_path}")
    print(f"Saved CV summary    -> {out_dir / 'cv_summary.json'}")
    print(f"Saved models dir    -> {models_dir}")
    print(f"Saved preds dir     -> {preds_dir}")
    print(f"Saved figs dir      -> {figs_dir}")


if __name__ == "__main__":
    main()
