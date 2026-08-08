#!/usr/bin/env python3
"""Build search_manifest.tsv for the theo_pipeline_new.py hyperparameter
search: one line per run, each line a complete, standalone set of CLI args
(FIXED_ARGS below plus whatever that row's grid/sweep point overrides).
run_theo_pipeline_raven.sh just forwards a line verbatim -- it knows nothing
about datasets or defaults.

Always scans <SEARCH_ROOT>/results/ and drops any row that's already been
run OR is currently running (any run dir with a run_params.json counts --
that file is written right at the start of a run, long before it finishes).
Always overwrites search_manifest.tsv/.json in place -- there's no
versioning, this is meant to be regenerated freely.

Usage:
    python build_search_grid.py
    nohup ./submit_search.sh search_manifest.tsv > submit_search.log 2>&1 &
    disown

Don't rerun this while a previous submit_search.sh batch is still
outstanding: a row that's queued but hasn't started yet has no results/
entry, so it would look "new" again and get submitted a second time.

Edit GRID / FIXED_ARGS / the sweep constants below to change what's
searched -- they're the single source of truth, and are copied into
search_manifest.json alongside the manifest for the record.
"""

import argparse
import itertools
import json
from pathlib import Path

SEARCH_ROOT = Path("/nexus/posix0/MBR-neuralsystems/ephys/GLC7189/GLC7189_c_g0/theo_pipeline_params_search")
RESULTS_ROOT = SEARCH_ROOT / "results"
MANIFEST_PATH = SEARCH_ROOT / "search_manifest.tsv"
GRID_RECORD_PATH = SEARCH_ROOT / "search_manifest.json"

# =============================================================================
# What to search
# =============================================================================

# Dataset paths and experiment-level defaults theo_pipeline_new.py needs.
# Prepended to every manifest row; anything a sweep/grid function overrides
# later in the same line wins (argparse takes the last value for a repeated
# flag). This is the one place to point the search at a different animal,
# session, or dataset path.
FIXED_ARGS = {
    "--recording-path": "/nexus/posix0/MBR-neuralsystems/ephys/GLC7189/GLC7189_c_g0/preprocessed",
    "--peaks-path": "/nexus/posix0/MBR-neuralsystems/ephys/GLC7189/GLC7189_c_g0/preprocessed/thresholding_peaks_3.5/peaks_combined2_sorted.npy",
    "--motion-path": "/nexus/posix0/MBR-neuralsystems/ephys/GLC7189/GLC7189_c_g0/preprocessed/thresholding_peaks_3.5/motion_estimates/motion_dredge",
    "--localized-peaks-path": "/nexus/posix0/MBR-neuralsystems/ephys/GLC7189/GLC7189_c_g0/preprocessed/thresholding_peaks_3.5/localized_peaks_mt_raw.npy",
    "--behavior-df-path": "/ptmp/gongt/behaviour/20260603_texture_pca_df.pkl",
    "--metadata-df-path": "/ptmp/gongt/behaviour/20260603_metadata_df.pkl",
    "--output-root": str(SEARCH_ROOT),
    "--animal": "GLC7189",
    "--neural-prefix": "c_g0",
    "--pc-cols": "PC1,PC2,PC3,PC4,PC5",
    "--dt": 0.2,
    "--bin-before": 6,
    "--bin-current": 1,
    "--bin-after": 6,
    "--n-splits": 10,
}
FIXED_ARGS_STR = " ".join(f"{k} {v}" for k, v in FIXED_ARGS.items())

# Per-model hyperparameter grid: every combination in each model's dict is
# tested. WF has no entry -- RidgeCV already internally cross-validates its
# alpha grid per fold, so it just runs once at its defaults (BASELINE_ONLY_MODELS).
GRID = {
    "WC": {
        "--wc-degree": [1, 2, 3, 4],
    },
    "SVR": {
        # max_iter=5000 (theo_pipeline_new.py's own default) isn't tested here:
        # it timed out (>8h) on most runs that hit it. 1000 added as a faster
        # option; see SVR_SAFE_MAX_ITER below for the anchor used elsewhere.
        "--svr-c": [0.3, 1.0, 2.0, 3.0, 10.0, 30.0],
        "--svr-max-iter": [1000, 2000, 3500],
    },
    "XGB": {
        "--xgb-max-depth": [1, 3, 5, 7, 9],
        "--xgb-n-estimators": [100, 200, 400, 600],
        "--xgb-learning-rate": [0.01, 0.05, 0.1, 0.2],
    },
    "FNN": {
        "--fnn-num-units": [100, 200, 400, 600],
        "--fnn-frac-dropout": [0.0, 0.25, 0.5],
        "--fnn-max-epochs": [10, 20, 30, 40],
        "--fnn-n-layers": [1, 2, 3, 4],
    },
    "RNN": {
        # No --rnn-n-layers: Neural_Decoding.nn.RNN's constructor has no
        # n_layers param at all, unlike FNN/GRU/LSTM (see theo_pipeline_new.py).
        "--rnn-num-units": [20, 50, 100, 200, 400, 600],
        "--rnn-frac-dropout": [0.0, 0.25, 0.5, 0.75, 1.0],
        "--rnn-max-epochs": [2, 5, 10, 20, 30, 40],
    },
    "GRU": {
        "--gru-num-units": [20, 50, 100, 200, 400, 600],
        "--gru-frac-dropout": [0.0, 0.25, 0.5, 0.75, 1.0],
        "--gru-max-epochs": [2, 5, 10, 20, 30, 40],
        "--gru-n-layers": [1, 2, 3, 4],
    },
    "LSTM": {
        "--lstm-num-units": [20, 50, 100, 200, 400, 600],
        "--lstm-frac-dropout": [0.0, 0.25, 0.5, 0.75, 1.0],
        "--lstm-max-epochs": [2, 5, 10, 20, 30, 40],
        "--lstm-n-layers": [1, 2, 3, 4],
    },
}
BASELINE_ONLY_MODELS = ["WF"]
ALL_MODELS = list(GRID) + BASELINE_ONLY_MODELS

# Additive pipeline sweep: dt / lag-window / n-splits apply the same way to
# every model, so instead of crossing them into GRID above (which would
# multiply its size by len(DT)*len(WINDOW)*len(N_SPLITS)), each combination
# is run once per model AT THAT MODEL'S DEFAULT hyperparameters. The lag
# window is swept symmetrically (--bin-before = --bin-after = window,
# --bin-current fixed at 1).
DT_VALUES = [0.1, 0.2, 0.5, 1, 2, 4]
WINDOW_VALUES = [1, 3, 5, 10, 15, 20]
N_SPLITS_VALUES = [5, 10, 15, 20]

# SVR-only additive sweep: --output-preprocess zscore instead of the default
# demean, crossed with each surviving --svr-max-iter at SVR's default C
# (PIPELINE_DEFAULTS["svr_c"] below).
SVR_OUTPUT_PREPROCESS_SWEEP = ["zscore"]

# --spike-thresh (theo_pipeline_new.py): drops channels with fewer than this
# many spikes, on top of the always-on zero-spike drop -- a cheap alternative
# to --channel-smooth-window for cutting input feature count. Additive sweep
# like the pipeline sweep above (each model at its own default hyperparams).
# Values chosen from channel_selection.ipynb's spike-count histogram: clearly
# bimodal, with a gap around 300-1000 and another past the upper mode by 20000.
SPIKE_THRESH_VALUES = [300, 5000, 20000]

# SVR's own default --svr-max-iter (5000) times out (see GRID["SVR"] above),
# so any additive sweep that leaves SVR's hyperparameters at their default
# needs this override instead, or it would silently re-hit the same timeout.
SVR_SAFE_MAX_ITER = 2000


# Per-model override needed when a sweep varies something else and leaves
# that model's own hyperparameters at their default -- SVR's own default
# --svr-max-iter (5000) times out (see GRID["SVR"] above), so any sweep that
# doesn't otherwise touch SVR's hyperparameters needs this instead. A dict
# (not an if-branch) so the next slow model is a data change, not new code.
MODEL_ANCHOR_OVERRIDES = {
    "SVR": {"--svr-max-iter": SVR_SAFE_MAX_ITER},
}


# =============================================================================
# Manifest row builders -- each returns a list of (model, overrides) pairs,
# where overrides is a {flag: value} dict of what that row sets beyond
# FIXED_ARGS. Kept structured (not formatted into a CLI string) until
# main() writes the manifest, so drop_already_done() can compare rows
# directly instead of re-parsing its own output.
# =============================================================================

def hyperparameter_grid_rows(models):
    """Every combination in GRID for each requested model, plus one bare row
    for each requested model in BASELINE_ONLY_MODELS."""
    rows = []
    for model in models:
        if model in BASELINE_ONLY_MODELS:
            rows.append((model, {}))
            continue
        flags = list(GRID[model])
        for combo in itertools.product(*GRID[model].values()):
            rows.append((model, dict(zip(flags, combo))))
    return rows


def pipeline_sweep_rows(models):
    """dt x lag-window x n-splits, each model at its own default hyperparams."""
    rows = []
    for model in models:
        for dt, window, n_splits in itertools.product(DT_VALUES, WINDOW_VALUES, N_SPLITS_VALUES):
            overrides = {
                "--dt": dt, "--bin-before": window, "--bin-current": 1, "--bin-after": window,
                "--n-splits": n_splits, **MODEL_ANCHOR_OVERRIDES.get(model, {}),
            }
            rows.append((model, overrides))
    return rows


def svr_output_preprocess_rows():
    """SVR at its default C, each surviving max_iter, each non-default
    --output-preprocess value."""
    rows = []
    for max_iter, output_preprocess in itertools.product(GRID["SVR"]["--svr-max-iter"], SVR_OUTPUT_PREPROCESS_SWEEP):
        overrides = {"--svr-c": PIPELINE_DEFAULTS["svr_c"], "--svr-max-iter": max_iter, "--output-preprocess": output_preprocess}
        rows.append(("SVR", overrides))
    return rows


def spike_thresh_sweep_rows(models):
    """Each model at its own default hyperparams, each --spike-thresh value."""
    rows = []
    for model in models:
        for spike_thresh in SPIKE_THRESH_VALUES:
            overrides = {"--spike-thresh": spike_thresh, **MODEL_ANCHOR_OVERRIDES.get(model, {})}
            rows.append((model, overrides))
    return rows


def format_row(model, overrides):
    """(model, overrides) -> the complete CLI-arg string for this row."""
    parts = [FIXED_ARGS_STR, f"--models {model}"]
    parts += [f"{flag} {val}" for flag, val in overrides.items()]
    return " ".join(parts)


# =============================================================================
# Skip anything already run or currently running (scans results/)
# =============================================================================

# What an unset flag resolves to, for matching a manifest row against a real
# run's cli_args -- must stay in sync with theo_pipeline_new.py's parse_args()
# defaults. n_splits is the one exception: every row here sets --n-splits
# explicitly via FIXED_ARGS, so this fallback only matters for results/
# entries from before FIXED_ARGS existed, which got --n-splits 10 from the
# old run_theo_pipeline_new_raven.sh wrapper, not argparse's own default of 3.
PIPELINE_DEFAULTS = {
    "dt": 0.2, "bin_before": 6, "bin_current": 1, "bin_after": 6, "n_splits": 10,
    "input_preprocess": "zscore", "output_preprocess": "demean", "spike_thresh": 0,
    "wc_degree": 3, "svr_c": 3.0, "svr_max_iter": 5000,
    "xgb_max_depth": 3, "xgb_n_estimators": 200, "xgb_learning_rate": 0.1,
    "fnn_num_units": 400, "fnn_frac_dropout": 0.25, "fnn_max_epochs": 10, "fnn_n_layers": 2,
    "rnn_num_units": 400, "rnn_frac_dropout": 0.25, "rnn_max_epochs": 10,  # no rnn_n_layers -- see GRID["RNN"]
    "gru_num_units": 400, "gru_frac_dropout": 0.25, "gru_max_epochs": 10, "gru_n_layers": 2,
    "lstm_num_units": 400, "lstm_frac_dropout": 0.25, "lstm_max_epochs": 10, "lstm_n_layers": 2,
}
NUMERIC_KEYS = {k for k, v in PIPELINE_DEFAULTS.items() if isinstance(v, (int, float))}


def resolve_config(values):
    """Fill in PIPELINE_DEFAULTS for anything missing from `values` (a dict
    keyed by PIPELINE_DEFAULTS' own names, e.g. {"dt": 0.5} -- either a run's
    cli_args, or a row's overrides after flag names are underscored), casting
    numeric fields to float for stable comparison."""
    cfg = {}
    for k, default in PIPELINE_DEFAULTS.items():
        v = values.get(k, default)
        cfg[k] = float(v) if k in NUMERIC_KEYS and v is not None else v
    return cfg


def row_config(model, overrides):
    """A manifest row's (model, config) pair, restricted to PIPELINE_DEFAULTS
    keys (dataset paths etc. never vary, so they're not part of matching)."""
    values = {flag[2:].replace("-", "_"): v for flag, v in overrides.items()}
    return model, resolve_config(values)


def existing_configs():
    """One (model, config) per results/ run dir that has at least a
    run_params.json -- written right at the start of a run, so this also
    catches runs still in progress, not just finished ones (deliberate: an
    in-flight run should be skipped the same as a finished one)."""
    configs = []
    for run_dir in sorted(RESULTS_ROOT.glob("[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_*")):
        try:
            params = json.loads((run_dir / "run_params.json").read_text())
            cli, models = params["cli_args"], params["resolved"]["selected_models"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            continue
        cfg = resolve_config(cli)
        configs.extend((model, cfg) for model in models)
    return configs


def configs_match(a, b, tol=1e-9):
    for k in PIPELINE_DEFAULTS:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, float) or isinstance(vb, float):
            if abs(float(va) - float(vb)) > tol:
                return False
        elif va != vb:
            return False
    return True


def drop_already_done(rows):
    """rows: list of (model, overrides). Returns (rows not yet in results/,
    number dropped because they are)."""
    by_model = {}
    for model, cfg in existing_configs():
        by_model.setdefault(model, []).append(cfg)

    kept = []
    for model, overrides in rows:
        _, cfg = row_config(model, overrides)
        if not any(configs_match(cfg, existing) for existing in by_model.get(model, [])):
            kept.append((model, overrides))
    return kept, len(rows) - len(kept)


# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default=None,
        help=f"Comma-separated subset of models to include (default: all of {ALL_MODELS}).")
    p.add_argument("--skip-pipeline-sweep", action="store_true", help="Omit the dt/window/n-splits sweep.")
    p.add_argument("--skip-svr-output-preprocess-sweep", action="store_true", help="Omit the SVR --output-preprocess sweep.")
    p.add_argument("--skip-spike-thresh-sweep", action="store_true", help="Omit the --spike-thresh sweep.")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",")] if args.models else ALL_MODELS
    unknown = [m for m in models if m not in ALL_MODELS]
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}. Known: {ALL_MODELS}")

    sections = {
        "hyperparameter grid": hyperparameter_grid_rows(models),
        "dt/window/n-splits sweep": [] if args.skip_pipeline_sweep else pipeline_sweep_rows(models),
        "SVR output-preprocess sweep": (
            [] if args.skip_svr_output_preprocess_sweep or "SVR" not in models else svr_output_preprocess_rows()
        ),
        "spike-thresh sweep": [] if args.skip_spike_thresh_sweep else spike_thresh_sweep_rows(models),
    }
    all_rows = [row for rows in sections.values() for row in rows]
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    new_rows, n_skipped = drop_already_done(all_rows)

    manifest_lines = [format_row(model, overrides) for model, overrides in new_rows]
    MANIFEST_PATH.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""))
    GRID_RECORD_PATH.write_text(json.dumps(
        {
            "fixed_args": FIXED_ARGS,
            "models": models,
            "grid": {m: GRID[m] for m in models if m in GRID},
            "dt_values": DT_VALUES,
            "window_values": WINDOW_VALUES,
            "n_splits_values": N_SPLITS_VALUES,
            "svr_output_preprocess_sweep": SVR_OUTPUT_PREPROCESS_SWEEP,
            "spike_thresh_values": SPIKE_THRESH_VALUES,
            "n_rows_considered": len(all_rows),
            "n_rows_already_done_or_running": n_skipped,
            "n_rows_new": len(new_rows),
        },
        indent=2,
    ))

    for name, rows in sections.items():
        print(f"{name:<28} -> {len(rows)} rows")
    print(f"{'considered':<28} -> {len(all_rows)} rows")
    print(f"{'already done/running':<28} -> {n_skipped} rows")
    print(f"{'new':<28} -> {len(new_rows)} rows")
    print(f"\nManifest        -> {MANIFEST_PATH}")
    print(f"Grid definition -> {GRID_RECORD_PATH}")
    if new_rows:
        print(f"\nSubmit with:\n  nohup ./submit_search.sh {MANIFEST_PATH} > submit_search.log 2>&1 &\n  disown")


if __name__ == "__main__":
    main()
