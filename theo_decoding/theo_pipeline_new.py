#!/usr/bin/env python3

import sys

# In this environment, torch's optimizer construction (torch.optim.Adam.__init__,
# used by skorch/FNN) lazily imports torch._dynamo on its first call, which in
# turn imports triton purely to check availability (torch/utils/_triton.py's
# has_triton_package()) -- and triton's native extension segfaults/hangs on
# load here (torch==2.12.1+cu130 / triton==3.7.1 on this machine's GPU driver),
# regardless of whether training ever touches the GPU or calls torch.compile.
# Blocking the import (rather than uninstalling triton, which would touch the
# shared conda env) makes it fail as a clean, caught ImportError instead. This
# also covers --device cuda runs -- the segfault trigger is optimizer
# construction itself, not GPU usage.
sys.modules.setdefault("triton", None)

# stdout is piped through `tee` (see run_theo_pipeline_new.sh), not a tty, so
# CPython fully block-buffers it by default -- prints can sit invisible in
# the buffer for a long time instead of reaching the log file as they
# happen. Line-buffer explicitly so per-fold progress below (and everything
# else) shows up in the log in real time, not just at process exit.
# Guarded: sys.stdout isn't a real file object with .reconfigure() when this
# module is imported (rather than run as __main__) under a Jupyter kernel --
# ipykernel's OutStream doesn't implement it -- so skip rather than crash.
if hasattr(sys.stdout, "reconfigure"):
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
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from Neural_Decoding.nonlinear import WienerCascade
from Neural_Decoding.preprocessing_funcs import LagMat, bin_spikes

try:
    from Neural_Decoding.nn import RNN as _RNNBase
    from torch import nn as _torch_nn

    class FixedRNN(_RNNBase):
        """Neural_Decoding.nn.RNN builds nn.RNN(...) without batch_first=True
        (unlike its GRU/LSTM siblings, which do). Fed the same (batch, seq,
        features) tensor as those, plain RNN would silently swap the batch
        and time axes rather than error -- no crash, just a meaningless
        scrambled-sequence result. This override adds the missing flag;
        everything else is unchanged. Defined at module level (not nested
        inside build_models()) so joblib/pickle can serialize fitted models
        by qualified name -- a local class can't be pickled."""

        def forward(self, X, **kwargs):
            if self.rnn is None:
                input_size = X.shape[-1]
                self.rnn = _torch_nn.RNN(
                    input_size=input_size,
                    hidden_size=self.num_units,
                    nonlinearity="relu",
                    dropout=self.frac_dropout,
                    batch_first=True,
                ).to(X.device)
            X, hidden_state = self.rnn(X)
            X = self.dropout(X)
            X = self.flatten(X)
            X = self.output(X)
            return X
except ImportError:
    FixedRNN = None

VALID_MODELS = {"WF", "WC", "SVR", "XGB", "FNN", "RNN", "GRU", "LSTM"}
SEQUENCE_MODELS = {"RNN", "GRU", "LSTM"}  # need the 3D (samples, timesteps, channels) lag tensor
EARLY_ABORT_EXIT_CODE = 75  # sysexits.h EX_TEMPFAIL -- distinct from a genuine crash (exit 1),
# so sacct/log review can tell "gave up early on purpose" apart from "actually broke" at a glance.


def pick_for_model(name, flat, seq):
    """flat (2D) or seq (3D, samples x timesteps x channels), whichever
    `name` needs -- see SEQUENCE_MODELS."""
    return seq if name in SEQUENCE_MODELS else flat


class SequenceScaler(BaseEstimator, TransformerMixin):
    """StandardScaler for 3D (samples, timesteps, channels) input.

    sklearn's StandardScaler only accepts 2D input. RNN/GRU/LSTM need their
    timestep axis kept intact, so this reshapes to (samples*timesteps,
    channels) to fit/apply per-channel scaling, then reshapes back.
    """

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        n, t, c = X.shape
        self.scaler_ = StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
        self.scaler_.fit(X.reshape(n * t, c))
        return self

    def transform(self, X):
        n, t, c = X.shape
        return self.scaler_.transform(X.reshape(n * t, c)).reshape(n, t, c).astype("float32")


def make_scaler_step(mode, sequence=False):
    if mode == "none":
        return "passthrough"
    with_std = mode == "zscore"
    if sequence:
        return SequenceScaler(with_mean=True, with_std=with_std)
    return StandardScaler(with_mean=True, with_std=with_std)


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


# --- Per-model hyperparameters: single source of truth ----------------------
# Each function returns a model's COMPLETE parameter dict (CLI-driven values
# plus every fixed constant), consumed both by build_models() to construct
# the actual estimator and by build_run_params() to log it -- so the log can
# never drift from what was actually run, and every model's full parameter
# set (searched or not) is always recorded, not just the ones this call ran.

def resolve_wf_alphas(args):
    if args.wf_alphas:
        return np.array([float(a) for a in args.wf_alphas.split(",") if a.strip()], dtype=float)
    return np.logspace(-2, 4, 25)


def wf_params(args):
    alphas = resolve_wf_alphas(args)
    return {"regressor": "RidgeCV", "alphas": alphas.tolist()}


def wc_params(args):
    return {"regressor": "WienerCascade", "degree": args.wc_degree}


def svr_params(args):
    return {
        "regressor": "SVR",
        "C": args.svr_c,
        "max_iter": args.svr_max_iter,
        "kernel": "rbf",
        "gamma": "scale",
        "epsilon": 0.1,
        "tol": 1e-3,
    }


def xgb_params(args):
    # Shared by make_xgb_estimator() (both the production and diagnostic
    # loss-curve fits) and build_run_params()'s reproducibility dump.
    return dict(
        objective="reg:squarederror",
        max_depth=args.xgb_max_depth,
        n_estimators=args.xgb_n_estimators,
        learning_rate=args.xgb_learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        reg_alpha=0,
        reg_lambda=1,
        tree_method="hist",
        random_state=args.seed,
        n_jobs=1,
    )


def _nn_params(name, args, batch_size, n_layers):
    """Shared shape for FNN/RNN/GRU/LSTM's parameter dicts -- each reads its
    num_units/frac_dropout/max_epochs from --<name>-* CLI flags. batch_size
    and n_layers differ per model (FNN's batch size is CLI-tunable via
    --fnn-batch-size; RNN's module has no n_layers param at all, unlike
    GRU/LSTM/FNN)."""
    prefix = name.lower()
    return {
        "regressor": name,
        "num_units": getattr(args, f"{prefix}_num_units"),
        "frac_dropout": getattr(args, f"{prefix}_frac_dropout"),
        "max_epochs": getattr(args, f"{prefix}_max_epochs"),
        "batch_size": batch_size,
        "n_layers": n_layers,
        "lr": 0.001,
        "device": args.device,
    }


def fnn_params(args):
    return _nn_params("FNN", args, batch_size=args.fnn_batch_size, n_layers=args.fnn_n_layers)


def rnn_params(args):
    return _nn_params("RNN", args, batch_size=32, n_layers=None)  # RNN's module has no such param -- not CLI-tunable


def gru_params(args):
    return _nn_params("GRU", args, batch_size=32, n_layers=args.gru_n_layers)


def lstm_params(args):
    return _nn_params("LSTM", args, batch_size=32, n_layers=args.lstm_n_layers)


MODEL_PARAM_FNS = {
    "WF": wf_params,
    "WC": wc_params,
    "SVR": svr_params,
    "XGB": xgb_params,
    "FNN": fnn_params,
    "RNN": rnn_params,
    "GRU": gru_params,
    "LSTM": lstm_params,
}


def build_output_tag(selected_models, args):
    """<model>_<key-params> tag for the output folder name. "multi" if more
    than one model was selected in this invocation (still fully
    disambiguated by run_params.json inside)."""
    if len(selected_models) != 1:
        return "multi"
    name = selected_models[0]
    params = MODEL_PARAM_FNS[name](args)
    if name == "WF":
        return f"WF_na{len(params['alphas'])}"
    if name == "WC":
        return f"WC_deg{params['degree']}"
    if name == "SVR":
        return f"SVR_c{params['C']}_mi{params['max_iter']}"
    if name == "XGB":
        return f"XGB_d{params['max_depth']}_n{params['n_estimators']}_lr{params['learning_rate']}"
    if name in ("FNN", "RNN", "GRU", "LSTM"):
        return f"{name}_u{params['num_units']}_do{params['frac_dropout']}_e{params['max_epochs']}"
    return name


def make_unique_output_dir(output_root, timestamp, tag):
    """Create <output_root>/results/<timestamp>_<tag>, appending _1, _2, ...
    on collision (e.g. two search jobs launched in the same second) rather
    than silently merging two runs' outputs together. Runs live under a
    results/ subfolder so they don't clutter --output-root alongside the
    manifests, logs/, analysis_plots/, etc."""
    base = Path(output_root) / "results" / f"{timestamp}_{tag}"
    candidate = base
    suffix = 1
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = Path(f"{base}_{suffix}")
            suffix += 1


def build_run_params(
    args, pc_cols, selected_models, resolved_motion_path, drift_corrected, resolved_localized_peaks_path,
    channel_info, n_holdout, holdout_start_time, out_dir,
):
    """Everything needed to reproduce this run: full CLI args, resolved
    settings decided at runtime, EVERY model's complete hyperparameter set
    (not just the ones this call ran -- see MODEL_PARAM_FNS), and the
    software environment. Saved to the output dir up front, before the long
    CV loop, so it survives even if a later model/fold crashes.
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
            "output_dir": str(out_dir),
            "motion_path": str(resolved_motion_path) if resolved_motion_path else None,
            "localized_peaks_path": str(resolved_localized_peaks_path) if resolved_localized_peaks_path else None,
            "drift_corrected": drift_corrected,
            "channels": channel_info,
            "input_preprocess": args.input_preprocess,
            "output_preprocess": args.output_preprocess,
            "device": args.device,
            "holdout_frac": args.holdout_frac,
            "evaluate_holdout": args.evaluate_holdout,
            "holdout_n_samples": n_holdout,
            "holdout_start_time": holdout_start_time,
        },
        "model_hyperparams": {name: fn(args) for name, fn in MODEL_PARAM_FNS.items()},
        "package_versions": {
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "sklearn": _package_version("sklearn"),
            "xgboost": _package_version("xgboost"),
            "spikeinterface": _package_version("spikeinterface"),
            "joblib": _package_version("joblib"),
            "torch": _package_version("torch"),
            "skorch": _package_version("skorch"),
        },
        "git": _git_info(),
    }


def build_parser():
    """Split out from parse_args() so callers that only want the argument
    definitions (e.g. analyze_param_search.ipynb, to read real defaults for
    backfilling old runs' cli_args) don't have to fake required args through
    an actual parse_args() call."""
    p = argparse.ArgumentParser(description="Run decoding CV on channel-wise MUA + PC outputs")
    p.add_argument("--recording-path", required=True)
    p.add_argument("--peaks-path", required=True)
    p.add_argument("--behavior-df-path", required=True)
    p.add_argument("--metadata-df-path", required=True)
    p.add_argument("--output-root", required=True, help="Parent directory; this run's own "
        "timestamped+tagged output folder is created underneath it (see build_output_tag).")
    p.add_argument("--animal", default="GLC6962")
    p.add_argument("--neural-prefix", default="g_g0", help="Identifies which recording "
        "session of this animal to use -- looked up against metadata_df's own "
        "(animal, neural_prefix) -> dataset mapping (see behavior-loading code below). "
        "Not the same as --background: an animal can have multiple sessions sharing one "
        "background (e.g. GLC7189 has three separate eink recordings), so background alone "
        "can't uniquely pick one.")
    p.add_argument("--pc-cols", default="PC1,PC2,PC3,PC4,PC5")
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--window-before", type=float, default=1.2, help="Lag window looking "
        "BACKWARD from each bin, in seconds (not bin count -- converted to "
        "round(window_before / dt) bins internally, so the same --window-before means the "
        "same physical lookback regardless of --dt). Default 1.2s matches this pipeline's "
        "historical bin-count default (bin_before=6) at its default --dt of 0.2.")
    p.add_argument("--bin-current", type=int, default=1, help="Bin count, NOT seconds -- "
        "unlike --window-before/--window-after, this is never swept as an independent "
        "physical duration in this project (always 1: just the current bin).")
    p.add_argument("--window-after", type=float, default=1.2, help="Lag window looking "
        "FORWARD from each bin, in seconds -- see --window-before.")
    p.add_argument("--n-splits", type=int, default=3)
    p.add_argument("--models", default="WF")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runtime-abort-budget-hours", type=float, default=23.0, help="After "
        "each model's 2nd fold finishes, project that model's total fold time (2-fold "
        "average x n_splits) plus whatever's already elapsed since the script started, and "
        "compare against this budget. If projected total exceeds it, abort immediately "
        "(nonzero exit) instead of continuing to burn the SLURM allocation on a run that "
        "can't finish in time -- see EARLY_ABORT_EXIT_CODE. Default leaves ~1h of margin "
        "under run_theo_pipeline_raven.sh's --time=23:59:59 for setup overhead already "
        "spent before the fold loop and any cleanup/plotting after it.")
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
        "--spike-thresh",
        type=int,
        default=0,
        help="Drop channels with fewer than this many spikes in the analysis window "
        "(counted after drift correction/channel-snapping, before --channel-smooth-window "
        "grouping) -- a blunt way to cut input feature count by dropping low-firing "
        "channels outright. 0 disables this (old behavior: only exactly-zero-spike "
        "channels are dropped).",
    )
    p.add_argument(
        "--channel-smooth-window",
        type=int,
        default=0,
        help="Smooth active channels by sliding a window of this many (depth-sorted) "
        "channels before binning MUA, e.g. 20. Default 0 disables smoothing (one "
        "feature per channel, old behavior). See --channel-smooth-overlap for the "
        "window's step size.",
    )
    p.add_argument(
        "--channel-smooth-overlap",
        type=int,
        default=0,
        help="Number of channels shared between consecutive smoothing windows (must "
        "be < --channel-smooth-window). Only used when --channel-smooth-window > 0.",
    )

    # --- Preprocessing (searchable) ---
    p.add_argument(
        "--input-preprocess", choices=["zscore", "demean", "none"], default="zscore",
        help="Preprocessing applied to the lagged feature matrix, fit per-fold on "
        "train only. zscore = today's default behavior.",
    )
    p.add_argument(
        "--output-preprocess", choices=["none", "demean", "zscore"], default="demean",
        help="Preprocessing applied to the target (y), fit per-fold on train only. "
        "Predictions are always inverse-transformed back to original PC units "
        "before scoring/plotting, so R2 stays comparable across choices.",
    )
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda",
        help="Device for FNN/RNN/GRU/LSTM (skorch NeuralNetRegressor). Ignored by "
        "WF/WC/SVR/XGB.")

    # --- Holdout (final, trustworthy number after a hyperparameter search) ---
    p.add_argument(
        "--holdout-frac", type=float, default=0.2,
        help="Fraction (0-1) of the session's LAST samples (chronological) reserved "
        "and excluded from the search's KFold entirely. 0 = off (identical to no "
        "holdout). See --evaluate-holdout.",
    )
    p.add_argument(
        "--evaluate-holdout", action="store_true",
        help="After the normal CV loop, refit each selected model on all non-holdout "
        "data and score once against the holdout block (requires --holdout-frac > 0). "
        "Recorded under a separate holdout_r2_per_output key -- never mixed into the "
        "per-fold search scores.",
    )

    # --- WF ---
    p.add_argument("--wf-alphas", default=None,
        help="Comma-separated floats overriding RidgeCV's alpha grid. Default: "
        "the 25-value np.logspace(-2, 4, 25) grid.")
    # --- WC ---
    p.add_argument("--wc-degree", type=int, default=3)
    # --- SVR ---
    p.add_argument("--svr-c", type=float, default=3.0)
    p.add_argument("--svr-max-iter", type=int, default=5000)
    # --- XGB ---
    p.add_argument("--xgb-max-depth", type=int, default=3)
    p.add_argument("--xgb-n-estimators", type=int, default=200)
    p.add_argument("--xgb-learning-rate", type=float, default=0.1)
    # --- FNN ---
    p.add_argument("--fnn-num-units", type=int, default=400)
    p.add_argument("--fnn-frac-dropout", type=float, default=0.25)
    p.add_argument("--fnn-max-epochs", type=int, default=10)
    p.add_argument("--fnn-batch-size", type=int, default=32)
    p.add_argument("--fnn-n-layers", type=int, default=2)
    # --- RNN --- (no --rnn-n-layers: Neural_Decoding.nn.RNN's constructor has
    # no n_layers param at all, unlike FNN/GRU/LSTM -- see rnn_params() below)
    p.add_argument("--rnn-num-units", type=int, default=400)
    p.add_argument("--rnn-frac-dropout", type=float, default=0.25)
    p.add_argument("--rnn-max-epochs", type=int, default=10)
    # --- GRU ---
    p.add_argument("--gru-num-units", type=int, default=400)
    p.add_argument("--gru-frac-dropout", type=float, default=0.25)
    p.add_argument("--gru-max-epochs", type=int, default=10)
    p.add_argument("--gru-n-layers", type=int, default=2)
    # --- LSTM ---
    p.add_argument("--lstm-num-units", type=int, default=400)
    p.add_argument("--lstm-frac-dropout", type=float, default=0.25)
    p.add_argument("--lstm-max-epochs", type=int, default=10)
    p.add_argument("--lstm-n-layers", type=int, default=2)

    return p


def parse_args():
    return build_parser().parse_args()


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


# --- Output (y) preprocessing: fit on train only, invert before scoring -----

def fit_output_preprocess(y_train, mode):
    """mu/sd fit on y_train only. sd is None unless mode == 'zscore'."""
    if mode == "none":
        return None, None
    mu = np.nanmean(y_train, axis=0, keepdims=True)
    sd = np.nanstd(y_train, axis=0, keepdims=True) + 1e-9 if mode == "zscore" else None
    return mu, sd


def apply_output_preprocess(y, mu, sd):
    if mu is None:
        return y
    y = y - mu
    if sd is not None:
        y = y / sd
    return y


def invert_output_preprocess(y, mu, sd):
    if mu is None:
        return y
    if sd is not None:
        y = y * sd
    return y + mu


def fit_with_output_preprocess(model, X_train, y_train, mode):
    """Fits `model` on X_train against y_train preprocessed per `mode` (fit
    on y_train only). Returns (mu, sd, y_train_preprocessed) -- the stats,
    for predict_original_scale() below, and the preprocessed training
    targets themselves, needed by XGB's diagnostic eval_set refit."""
    mu, sd = fit_output_preprocess(y_train, mode)
    y_train_p = apply_output_preprocess(y_train, mu, sd)
    model.fit(X_train, y_train_p)
    return mu, sd, y_train_p


def predict_original_scale(model, X, mu, sd):
    """model.predict(X), inverse-transformed back to original PC units."""
    return invert_output_preprocess(model.predict(X), mu, sd)


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
    many windows this geometry produces); member_channel_idx[g] holds the
    ORIGINAL probe channel indices pooled into group g; member_counts[g] =
    len(member_channel_idx[g]).
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
    t_bins_abs = motion_obj.temporal_bins_s[0]
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


def make_xgb_estimator(args, eval_metric=None):
    from xgboost import XGBRegressor

    kwargs = xgb_params(args)
    if eval_metric is not None:
        kwargs["eval_metric"] = eval_metric
    return XGBRegressor(**kwargs)


def make_nn_step(module, y_dim, params):
    """Shared skorch NeuralNetRegressor construction for FNN/RNN/GRU/LSTM --
    identical scaffolding (Adam, shuffled batches, verbose off) across all
    four, differing only in the module class and the hyperparameters in
    `params` (as returned by fnn_params/rnn_params/gru_params/lstm_params).
    n_layers is omitted entirely when None, matching RNN's module (unlike
    FNN/GRU/LSTM) having no such constructor param."""
    from skorch import NeuralNetRegressor
    from torch import optim

    kwargs = dict(
        module=module,
        lr=params["lr"],
        iterator_train__shuffle=True,
        optimizer=optim.Adam,
        batch_size=params["batch_size"],
        module__n_targets=y_dim,
        module__num_units=params["num_units"],
        module__frac_dropout=params["frac_dropout"],
        max_epochs=params["max_epochs"],
        device=params["device"],
        verbose=0,
    )
    if params["n_layers"] is not None:
        kwargs["module__n_layers"] = params["n_layers"]
    return NeuralNetRegressor(**kwargs)


def build_models(args, y_dim, selected_models):
    # NOTE: LagMat is NOT a pipeline step here -- it's applied in main()'s
    # lag() helper, separately to each fold's/split's own raw train or test
    # block, after that block has already been carved out (matches
    # Neural_Decoding_sklearn's own reference CV pattern: LagMat inside a
    # Pipeline that cross_validate() fits per-fold). The scaler step below
    # still needs to be fit per-fold (on the training slice only) to avoid
    # leakage the same way.
    models = {}
    scaler_flat = make_scaler_step(args.input_preprocess, sequence=False)

    if "WF" in selected_models:
        models["WF"] = Pipeline(
            [
                ("scaler", scaler_flat),
                ("ridge", MultiOutputRegressor(RidgeCV(alphas=wf_params(args)["alphas"]))),
            ]
        )

    if "WC" in selected_models:
        models["WC"] = Pipeline(
            [
                ("scaler", scaler_flat),
                ("wc", MultiOutputRegressor(WienerCascade(degree=wc_params(args)["degree"]))),
            ]
        )

    if "SVR" in selected_models:
        svr_p = svr_params(args)
        models["SVR"] = Pipeline(
            [
                ("scaler", scaler_flat),
                (
                    "svr",
                    MultiOutputRegressor(SVR(
                        C=svr_p["C"], max_iter=svr_p["max_iter"], kernel=svr_p["kernel"],
                        gamma=svr_p["gamma"], epsilon=svr_p["epsilon"], tol=svr_p["tol"],
                    )),
                ),
            ]
        )

    if "XGB" in selected_models:
        try:
            models["XGB"] = Pipeline(
                [
                    ("scaler", scaler_flat),
                    ("xgb", MultiOutputRegressor(make_xgb_estimator(args))),
                ]
            )
        except Exception:
            pass

    if "FNN" in selected_models:
        from Neural_Decoding.nn import FNN

        models["FNN"] = Pipeline(
            [
                ("scaler", scaler_flat),
                ("fnn", make_nn_step(FNN, y_dim, fnn_params(args))),
            ]
        )

    scaler_seq = make_scaler_step(args.input_preprocess, sequence=True)

    if "RNN" in selected_models:
        if FixedRNN is None:
            raise ImportError("RNN decoder requires torch to be importable (Neural_Decoding.nn.RNN).")

        models["RNN"] = Pipeline(
            [
                ("scaler", scaler_seq),
                ("rnn", make_nn_step(FixedRNN, y_dim, rnn_params(args))),
            ]
        )

    if "GRU" in selected_models:
        from Neural_Decoding.nn import GRU

        models["GRU"] = Pipeline(
            [
                ("scaler", scaler_seq),
                ("gru", make_nn_step(GRU, y_dim, gru_params(args))),
            ]
        )

    if "LSTM" in selected_models:
        from Neural_Decoding.nn import LSTM

        models["LSTM"] = Pipeline(
            [
                ("scaler", scaler_seq),
                ("lstm", make_nn_step(LSTM, y_dim, lstm_params(args))),
            ]
        )

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


def plot_spike_count_histogram(out_path, channel_spike_counts, n_channels, spike_thresh):
    """Linear + log histogram of per-channel spike counts, with a red line at
    --spike-thresh (if set) showing how many channels/spikes that cutoff
    keeps -- same view used to pick the threshold in channel_selection.ipynb."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(channel_spike_counts, bins=40, color="#3B6FA0", edgecolor="white")
    axes[0].set_xlabel("spike count (whole window)")
    axes[0].set_ylabel("# channels")
    axes[0].set_title("Linear scale")

    axes[1].hist(
        channel_spike_counts,
        bins=np.logspace(np.log10(max(channel_spike_counts.min(), 1)), np.log10(channel_spike_counts.max()), 40),
        color="#3B6FA0", edgecolor="white",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("spike count (whole window, log scale)")
    axes[1].set_title("Log scale -- easier to see the low-count tail")

    if spike_thresh > 0:
        n_kept = int((channel_spike_counts >= spike_thresh).sum())
        pct_lost = 100 * channel_spike_counts[channel_spike_counts < spike_thresh].sum() / channel_spike_counts.sum()
        label = f"spike_thresh={spike_thresh:,} -> keeps {n_kept}/{n_channels} channels, loses {pct_lost:.1f}% of spikes"
        for ax in axes:
            ax.axvline(spike_thresh, color="red", linestyle="--", linewidth=1.5, label=label)
            ax.legend(fontsize=7.5, loc="upper right")

    fig.suptitle(f"Per-channel spike counts, {n_channels} channels")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_xgb_loss_curves(out_path, curves, pc_cols, fold):
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
    job_start_time = time.time()  # for the per-model early-abort projection below, not wall-clock logging
    args = parse_args()
    np.random.seed(args.seed)

    selected = [m.strip() for m in args.models.split(",") if m.strip()]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = build_output_tag(selected, args)
    out_dir = make_unique_output_dir(args.output_root, timestamp, tag)
    models_dir = out_dir / "models"
    preds_dir = out_dir / "predictions"
    figs_dir = out_dir / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir -> {out_dir}")

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
    metadata_df = pd.read_pickle(args.metadata_df_path)

    pc_cols = [c.strip() for c in args.pc_cols.split(",") if c.strip()]
    missing_cols = [c for c in pc_cols if c not in texture_pca_df.columns]
    if missing_cols:
        raise ValueError(f"Missing PC columns: {missing_cols}")

    # (animal, background) isn't a unique key -- e.g. GLC7189 has three separate
    # eink recordings (neural_prefix c_g0/d_g0/g_g0 in metadata_df), so filtering
    # on background alone can silently lump unrelated sessions together.
    # (animal, neural_prefix) maps onto metadata_df's own "dataset" id, and
    # texture_pca_df carries that same "dataset" column -- match through it instead.
    #
    # Even (animal, neural_prefix) isn't always unique, though: GLC6962/g_g0 hits
    # both the real "freelymoving" session and a "reverse" bookkeeping row that
    # shares its neural_prefix but was never turned into actual texture_pca_df
    # frames (dataset "...191607_reverse" has zero rows there). Rather than
    # hardcoding "exclude type=reverse" (fragile, and future dataset variants
    # could add other labels), disambiguate operationally: keep only candidates
    # whose dataset id actually has rows in texture_pca_df.
    meta_match = metadata_df[
        (metadata_df["animal"] == args.animal) & (metadata_df["neural_prefix"] == args.neural_prefix)
    ]
    if len(meta_match) == 0:
        known = sorted(metadata_df.loc[metadata_df["animal"] == args.animal, "neural_prefix"].dropna().unique())
        raise ValueError(
            f"No metadata_df row for animal={args.animal!r}, neural_prefix={args.neural_prefix!r}. "
            f"Known --neural-prefix values for this animal: {known}"
        )
    if len(meta_match) > 1:
        has_data = meta_match["dataset"].isin(texture_pca_df["dataset"].unique())
        meta_match = meta_match[has_data]
        if len(meta_match) != 1:
            raise ValueError(
                f"animal={args.animal!r}, neural_prefix={args.neural_prefix!r} matches "
                f"{len(has_data)} metadata_df rows, and {len(meta_match)} of those have any "
                f"rows in texture_pca_df (need exactly 1): "
                f"{metadata_df.loc[metadata_df['animal'] == args.animal, ['dataset', 'type']].to_dict('records')}"
            )
    dataset_id = meta_match["dataset"].iloc[0]
    background = meta_match["background"].iloc[0]
    beh = texture_pca_df[texture_pca_df["dataset"] == dataset_id].copy()
    if beh.empty:
        raise ValueError(f"No behavior rows for dataset={dataset_id!r} in {args.behavior_df_path}")
    print(f"Behavior session: animal={args.animal}, neural_prefix={args.neural_prefix} -> "
          f"dataset={dataset_id!r} (background={background}), {len(beh):,} frames.")

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
        assigned_dist = np.linalg.norm(
            corrected_positions_trim - channel_positions[peak_rows_trim], axis=1
        )
        print(
            f"Post-snap distance to assigned channel: median={np.median(assigned_dist):.2f}um, "
            f"mean={np.mean(assigned_dist):.2f}um, max={np.max(assigned_dist):.2f}um."
        )
        raw_depths_trim = localized_y_trim
    else:
        corrected_depths_trim = raw_depths_trim
        peak_rows_trim = peak_rows_trim_raw

    spike_times_trim = np.asarray(
        [peak_times_trim[peak_rows_trim == ch] for ch in range(n_channels)], dtype=object
    )

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

    # --- Spike-count threshold (post drift-correction/snapping, pre channel-smoothing) ---
    # Blunter than --channel-smooth-window: outright drops low-firing channels
    # instead of grouping them, to cut input feature count for the lag matrix below.
    spike_thresh_dropped_idx = np.array([], dtype=int)
    if args.spike_thresh > 0:
        n_before = active_channel_idx.size
        keep_mask = channel_spike_counts[active_channel_idx] >= args.spike_thresh
        spike_thresh_dropped_idx = active_channel_idx[~keep_mask]
        active_channel_idx = active_channel_idx[keep_mask]
        spike_times_trim = spike_times_trim[keep_mask]
        print(
            f"Spike-count threshold (--spike-thresh {args.spike_thresh:,}): dropped "
            f"{spike_thresh_dropped_idx.size:,}/{n_before:,} more channels "
            f"({active_channel_idx.size:,} channels remain)."
        )
    plot_spike_count_histogram(
        figs_dir / "channel_spike_count_histogram.png", channel_spike_counts, n_channels, args.spike_thresh,
    )

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

    channel_info = {
        "total": int(n_channels),
        "active": int(active_channel_idx.size),
        "dropped_zero_spikes": dropped_channel_idx.tolist(),
        "spike_thresh": args.spike_thresh,
        "dropped_by_spike_thresh": spike_thresh_dropped_idx.tolist(),
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
        neural_data = neural_data / group_member_counts[None, :]
    y_binned, t_edges = bin_output_safe(beh_outputs_trim, beh_t_trim, args.dt, t_start, t_end)

    valid_mask = np.all(np.isfinite(y_binned), axis=1)
    X = neural_data[valid_mask].astype("float32")
    y = y_binned[valid_mask].astype("float32")
    t_centers = ((t_edges[:-1] + t_edges[1:]) / 2.0)[valid_mask]

    formatted_path = out_dir / "formatted_data_like_hc.pickle"
    with open(formatted_path, "wb") as f:
        pickle.dump([X, y], f)

    # --window-before/--window-after are in SECONDS (a physical duration,
    # independent of --dt); LagMat itself only understands bin COUNTS, so
    # convert here, once. round(), not int(), so e.g. window=1.2s at dt=0.2
    # gives exactly 6 bins, not 5 from float truncation.
    bin_before = int(round(args.window_before / args.dt))
    bin_after = int(round(args.window_after / args.dt))
    n_lags_total = bin_before + args.bin_current + bin_after
    need_seq = bool(set(selected) & SEQUENCE_MODELS)

    print(
        f"Lag window: {n_lags_total:,} bins/sample ({args.window_before:g}s before + current + "
        f"{args.window_after:g}s after, at dt={args.dt:g}s) x {n_channels_used:,} channels = "
        f"{n_channels_used * n_lags_total:,} input features per sample "
        f"(output dim = {len(pc_cols)}: {pc_cols})."
    )

    def lag(X_raw):
        """LagMat, applied to a single chronologically-contiguous raw block,
        AFTER any train/test/holdout split -- never before one, and never to
        a concatenation of two non-adjacent blocks. Rows within bin_before/
        bin_after of X_raw's own edges get zero-padded there (matches
        Neural_Decoding_sklearn's own reference CV pattern: LagMat inside
        the pipeline, so it only ever sees one fold's own train or test
        slice). See build_models() for the fuller rationale -- computing
        this globally before splitting instead would let real neighboring
        data leak across whatever boundary gets carved out afterward.
        Returns (flat, seq_or_None): flat (2D) is what WF/WC/SVR/XGB/FNN
        use; seq (3D, samples x timesteps x channels) is what RNN/GRU/LSTM
        use -- both come from the same LagMat call when seq is needed, so
        the identical windowing is never computed twice.

        Cost note: called twice per fold (train slice + test slice) instead
        of once globally, so total LagMat work scales with --n-splits, not
        just dataset size -- roughly an --n-splits x multiplier on top of
        LagMat's own O(samples x channels x lag_bins) cost, since each
        fold's train slice is nearly the whole dataset. This is the
        accepted trade for the leakage fix (see the fold loop below) and
        matches the reference pattern; it's worth watching at --dt values
        small enough to push lag_bins into the hundreds (window_before/
        window_after are in seconds now -- see build_search_grid.py's
        WINDOW_VALUES comment -- so a small --dt can quietly make lag_bins
        much larger than the old bin-count scheme ever did)."""
        seq = None
        if need_seq:
            seq = LagMat(bin_before=bin_before, bin_current=args.bin_current, bin_after=bin_after, flat=False).fit_transform(X_raw)
            flat = seq.reshape(seq.shape[0], -1)
        else:
            flat = LagMat(bin_before=bin_before, bin_current=args.bin_current, bin_after=bin_after, flat=True).fit_transform(X_raw)
        return flat, seq

    # --- Holdout: reserve the last holdout_frac of the session. Split on
    # the RAW (unlagged) data first, then lag the search and holdout blocks
    # SEPARATELY -- see lag()'s docstring for why the order matters.
    n_total = X.shape[0]
    n_holdout = int(round(n_total * args.holdout_frac)) if args.holdout_frac > 0 else 0
    holdout_start_idx = n_total - n_holdout

    X_search_raw = X[:holdout_start_idx]
    y_search = y[:holdout_start_idx]
    t_centers_search = t_centers[:holdout_start_idx]
    X_lagged_search, X_lagged_seq_search = lag(X_search_raw)

    if n_holdout > 0:
        y_holdout = y[holdout_start_idx:]
        t_centers_holdout = t_centers[holdout_start_idx:]
        X_lagged_holdout, X_lagged_seq_holdout = lag(X[holdout_start_idx:])
        holdout_start_time = float(t_centers[holdout_start_idx])
        print(
            f"Holdout: reserved last {n_holdout:,}/{n_total:,} samples "
            f"({args.holdout_frac:.1%}) starting at t={holdout_start_time:.1f}s, "
            "excluded from the search."
        )
    else:
        X_lagged_holdout = None
        X_lagged_seq_holdout = None
        y_holdout = None
        t_centers_holdout = None
        holdout_start_time = None

    available_models = build_models(args, y_search.shape[1], selected)
    unavailable = [m for m in selected if m not in available_models]
    if unavailable:
        raise ValueError(
            f"Model(s) {unavailable} failed to build (e.g. a required package like "
            f"xgboost/torch/skorch isn't importable). Available: {list(available_models)}"
        )

    run_params = build_run_params(
        args, pc_cols, selected, motion_path, drift_corrected, localized_peaks_path, channel_info,
        n_holdout, holdout_start_time, out_dir,
    )
    with open(out_dir / "run_params.json", "w", encoding="utf-8") as f:
        json.dump(run_params, f, indent=2)
    print(f"Saved run params    -> {out_dir / 'run_params.json'}")

    kf = KFold(n_splits=args.n_splits, shuffle=False)
    fold_starts = [test_idx[0] for _, test_idx in kf.split(X_search_raw)]
    fold_boundary_times = t_centers_search[fold_starts[1:]]  # internal boundaries only

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

    if args.spike_thresh > 0:
        # channel_positions[peak_rows_trim, 1] is the final assigned depth of
        # every trimmed peak regardless of whether drift correction ran above
        # (it's just each peak's home-channel depth when it didn't).
        final_depths_trim = channel_positions[peak_rows_trim, 1]
        kept_peak_mask = np.isin(peak_rows_trim, active_channel_idx)
        plot_drift_scatter(
            figs_dir / "drift_05_spike_thresh_dropped_scatter.png",
            peak_times_trim[kept_peak_mask], final_depths_trim[kept_peak_mask],
            f"Remaining spikes after --spike-thresh {args.spike_thresh:,} "
            f"({active_channel_idx.size}/{n_channels} channels kept, "
            f"{100 * kept_peak_mask.mean():.1f}% of peaks in window)",
            fold_boundary_times,
        )

    summary = {
        "config": {
            "recording_path": args.recording_path,
            "peaks_path": args.peaks_path,
            "behavior_df_path": args.behavior_df_path,
            "animal": args.animal,
            "neural_prefix": args.neural_prefix,
            "dataset": dataset_id,
            "background": background,
            "pc_cols": pc_cols,
            "dt": args.dt,
            "window_before": args.window_before,
            "bin_current": args.bin_current,
            "window_after": args.window_after,
            # Derived, not user-supplied -- window_before/window_after (seconds) above
            # are the actual source of truth; these are only recorded for
            # transparency/debugging (e.g. reading n_lags_total straight off this file).
            "bin_before": bin_before,
            "bin_after": bin_after,
            "n_splits": args.n_splits,
            "input_preprocess": args.input_preprocess,
            "output_preprocess": args.output_preprocess,
            "device": args.device,
            "holdout_frac": args.holdout_frac,
            "evaluate_holdout": args.evaluate_holdout,
            "holdout_n_samples": n_holdout,
            "holdout_start_time": holdout_start_time,
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
            "shapes": {
                "X": list(X.shape), "y": list(y.shape),
                "X_lagged_search": list(X_lagged_search.shape),
                "X_lagged_holdout": list(X_lagged_holdout.shape) if X_lagged_holdout is not None else None,
            },
        },
        "models": {},
    }

    for name in selected:
        base_model = available_models[name]
        oof_pred = np.full_like(y_search, np.nan, dtype=float)
        fold_test_scores = []
        fold_train_scores = []
        xgb_channel_importances = []

        print(f"Model {name}: starting ({args.n_splits} folds)...")
        model_start = time.time()
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_search_raw)):
            fold_start = time.time()
            print(f"Model {name}: starting fold {fold + 1}/{args.n_splits}...")
            model = clone(base_model)

            # Lag AFTER splitting, separately for train and test -- see
            # lag()'s docstring for why. train_idx is two non-adjacent raw
            # blocks glued together for every fold except the first/last
            # (KFold's test block sits in the middle); lag() zero-pads at
            # the resulting array's own start/end same as any other edge,
            # which does mean a handful of rows right at that internal seam
            # get a window blending two distant times as if adjacent -- a
            # minor, accepted imperfection (matches Neural_Decoding_sklearn's
            # own reference pattern), nowhere near as bad as the leakage it
            # replaces (real train-adjacent data reaching into test, or vice
            # versa, which is what applying LagMat globally before the split
            # used to do).
            X_train_flat, X_train_seq = lag(X_search_raw[train_idx])
            X_test_flat, X_test_seq = lag(X_search_raw[test_idx])
            X_train = pick_for_model(name, X_train_flat, X_train_seq)
            X_test = pick_for_model(name, X_test_flat, X_test_seq)

            y_train = y_search[train_idx]
            mu, sd, y_train_p = fit_with_output_preprocess(model, X_train, y_train, args.output_preprocess)

            pred = predict_original_scale(model, X_test, mu, sd)
            oof_pred[test_idx] = pred

            test_score = r2_score(y_search[test_idx], pred, multioutput="raw_values")
            fold_test_scores.append(test_score.tolist())

            # In-sample (train) R2: the direct overfitting check.
            train_pred = predict_original_scale(model, X_train, mu, sd)
            train_score = r2_score(y_train, train_pred, multioutput="raw_values")
            fold_train_scores.append(train_score.tolist())

            joblib.dump(model, models_dir / f"{name}_fold{fold}.joblib")

            if name == "XGB":
                # Diagnostic-only refit per output with eval_set, purely to get a
                # real per-boosting-round train/held-out loss curve, in the same
                # preprocessed-y space the production model above was fit in.
                scaler_fitted = model.named_steps["scaler"]
                Xtr_t = scaler_fitted.transform(X_train) if scaler_fitted != "passthrough" else X_train
                Xte_t = scaler_fitted.transform(X_test) if scaler_fitted != "passthrough" else X_test
                y_test_p = apply_output_preprocess(y_search[test_idx], mu, sd)
                curves = {}
                for i, col in enumerate(pc_cols):
                    diag_reg = make_xgb_estimator(args, eval_metric="rmse")
                    diag_reg.fit(
                        Xtr_t,
                        y_train_p[:, i],
                        eval_set=[(Xtr_t, y_train_p[:, i]), (Xte_t, y_test_p[:, i])],
                        verbose=False,
                    )
                    curves[col] = diag_reg.evals_result()
                plot_xgb_loss_curves(figs_dir / f"XGB_loss_curve_fold{fold}.png", curves, pc_cols, fold)

                fitted_xgb = model.named_steps["xgb"]
                fold_importance = np.array(
                    [est.feature_importances_ for est in fitted_xgb.estimators_]
                )
                per_output_channel_importance = fold_importance.reshape(
                    len(pc_cols), n_lags_total, n_channels_used
                ).sum(axis=1)
                xgb_channel_importances.append(per_output_channel_importance)

            print(f"Model {name}: finished fold {fold + 1}/{args.n_splits} in {time.time() - fold_start:.1f}s")

            # Runtime tripwire: 2 folds is enough to get a stable per-fold
            # time estimate (fold cost is ~constant across folds -- see the
            # dt=0.2/window=32 WC jobs from earlier this week, whose fold
            # times varied <2% fold-to-fold), so project the REST of this
            # model's folds at that pace, add whatever's already elapsed
            # since the script started (covers setup overhead already paid
            # and any earlier models in this same job), and compare against
            # budget. Catches the "doomed from the start" case (see
            # 29358087_5/6/7, cancelled by hand after ~8h -- this replaces
            # that manual monitoring) at ~2 folds' cost instead of the full
            # 24h SLURM allocation.
            if fold == 1:
                elapsed_this_model = time.time() - model_start
                avg_fold_time = elapsed_this_model / (fold + 1)
                remaining_folds = args.n_splits - (fold + 1)
                projected_job_total = (time.time() - job_start_time) + avg_fold_time * remaining_folds
                budget_sec = args.runtime_abort_budget_hours * 3600
                if projected_job_total > budget_sec:
                    print(
                        f"ABORT: Model {name} projected total job runtime "
                        f"{projected_job_total / 3600:.2f}h (avg {avg_fold_time:.1f}s/fold over "
                        f"the first 2 folds x {remaining_folds} folds remaining, plus "
                        f"{(time.time() - job_start_time) / 3600:.2f}h already elapsed) exceeds "
                        f"the {args.runtime_abort_budget_hours:g}h budget -- killing this run now "
                        f"instead of burning the full SLURM allocation with no result. "
                        f"(dt={args.dt:g}, window_before={args.window_before:g}, "
                        f"n_splits={args.n_splits}, models={args.models})"
                    )
                    sys.exit(EARLY_ABORT_EXIT_CODE)

        overall = r2_score(y_search, oof_pred, multioutput="raw_values")
        mean_train = np.mean(np.asarray(fold_train_scores), axis=0)
        mean_test = np.mean(np.asarray(fold_test_scores), axis=0)

        np.savez_compressed(
            preds_dir / f"{name}_oof_predictions.npz",
            t=t_centers_search,
            y_true=y_search,
            y_pred=oof_pred,
            feature_names=np.array(pc_cols, dtype=object),
        )

        plot_decoded_vs_observed(
            figs_dir / f"{name}_decoded_vs_observed.png", t_centers_search, y_search, oof_pred, pc_cols, name,
        )
        plot_scatter_pred_vs_obs(
            figs_dir / f"{name}_scatter_pred_vs_obs.png", y_search, oof_pred, pc_cols, name
        )
        plot_residuals_over_time(
            figs_dir / f"{name}_residuals_over_time.png", t_centers_search, y_search, oof_pred, pc_cols, name,
            fold_boundary_times,
        )

        if name == "XGB" and xgb_channel_importances:
            mean_channel_importance = np.mean(np.stack(xgb_channel_importances, axis=0), axis=(0, 1))
            plot_xgb_channel_importance(
                figs_dir / "XGB_channel_importance.png",
                mean_channel_importance, channel_x_positions, channel_x_label, bar_width=channel_bar_width,
            )

        model_summary = {
            "fold_r2_raw_values": fold_test_scores,
            "mean_fold_r2_per_output": mean_test.tolist(),
            "oof_r2_per_output": overall.tolist(),
            "fold_train_r2_raw_values": fold_train_scores,
            "mean_fold_train_r2_per_output": mean_train.tolist(),
        }

        print(f"{name} OOF R2 per output   (held-out):                     {np.round(overall, 4)}")
        print(f"{name} mean TRAIN R2 per output (in-sample, overfitting check): {np.round(mean_train, 4)}")

        # --- Holdout: refit on ALL non-holdout (search) data, score once ---
        # against data the search never touched. Kept entirely separate from
        # the per-fold search scores above (see spec doc for rationale).
        if args.evaluate_holdout and n_holdout > 0:
            print(f"Model {name}: refitting on full search set for holdout evaluation...")
            # X_lagged_search/X_lagged_holdout were already lagged separately
            # (see lag() above), so this refit's boundary with holdout is
            # already clean -- no further trimming needed here.
            X_for_model_refit = pick_for_model(name, X_lagged_search, X_lagged_seq_search)
            X_for_model_holdout = pick_for_model(name, X_lagged_holdout, X_lagged_seq_holdout)

            holdout_model = clone(base_model)
            mu_h, sd_h, _ = fit_with_output_preprocess(holdout_model, X_for_model_refit, y_search, args.output_preprocess)
            holdout_pred = predict_original_scale(holdout_model, X_for_model_holdout, mu_h, sd_h)
            holdout_score = r2_score(y_holdout, holdout_pred, multioutput="raw_values")

            joblib.dump(holdout_model, models_dir / f"{name}_holdout_refit.joblib")
            np.savez_compressed(
                preds_dir / f"{name}_holdout_predictions.npz",
                t=t_centers_holdout, y_true=y_holdout, y_pred=holdout_pred,
                feature_names=np.array(pc_cols, dtype=object),
            )
            plot_decoded_vs_observed(
                figs_dir / f"{name}_holdout_decoded_vs_observed.png",
                t_centers_holdout, y_holdout, holdout_pred, pc_cols, f"{name} (holdout)",
            )

            model_summary["holdout_r2_per_output"] = holdout_score.tolist()
            print(f"{name} HOLDOUT R2 per output (never touched by search): {np.round(holdout_score, 4)}")

        summary["models"][name] = model_summary

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
