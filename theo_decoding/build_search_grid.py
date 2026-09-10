#!/usr/bin/env python3
"""Build search_manifest.tsv for the theo_pipeline_new.py hyperparameter
search: one line per run, each line a complete, standalone set of CLI args
(FIXED_ARGS below plus whatever that row's grid/sweep point overrides).
run_theo_pipeline_raven.sh just forwards a line verbatim -- it knows nothing
about datasets or defaults.

Always scans <SEARCH_ROOT>/results/ and drops any row that's already been
run OR is currently running (any run dir with a run_params.json counts --
that file is written right at the start of a run, long before it finishes).
The dedup match includes --pc-cols (see PIPELINE_DEFAULTS), so multiple PC
ranges can safely share one SEARCH_ROOT/results/ without a PC6-15 row ever
being mistaken for an already-done PC1-5 row of the same hyperparameters.
Always overwrites <manifest-name>.tsv/.json in place -- there's no
versioning, this is meant to be regenerated freely.

Usage:
    python build_search_grid.py
    nohup ./submit_search.sh search_manifest.tsv > submit_search.log 2>&1 &
    disown

Don't rerun this for the SAME --manifest-name while a previous
submit_search.sh batch of it is still outstanding: a row that's queued but
hasn't started yet has no results/ entry, so it would look "new" again and
get submitted a second time. If you need a second sweep target (a different
--pc-cols/--models combination) actively in flight against the same
SEARCH_ROOT at the same time, give it a distinct --manifest-name -- the
SLURM array tasks for each batch keep reading their own manifest file by
path for as long as any of its rows are still pending, so two concurrent
targets can never safely share one manifest file.

Edit GRID / FIXED_ARGS / the sweep constants below to change what's
searched -- they're the single source of truth, and are copied into
search_manifest.json alongside the manifest for the record.

Row order in the manifest follows ALL_MODELS x section order as written below
(WC, SVR, XGB, FNN, RNN, GRU, LSTM, WF). Standing practice on this project is
to submit cheap jobs first so expensive/doomed combos (small --dt, big
--window, many --n-splits -- especially SVR/XGB) surface late and can be
cancelled with little wasted compute; reorder ALL_MODELS / the sweep lists,
or split into separate --manifest-name batches, if you want that enforced.

OUTPUTS (in SEARCH_ROOT):
  <manifest-name>.tsv   -- one theo_pipeline_new.py arg-string per line; fed to
                           submit_search.sh, read line-by-line by run_theo_pipeline_raven.sh
  <manifest-name>.json  -- the GRID/FIXED_ARGS/sweep constants used, for the record
"""

import argparse
import itertools
import json
from pathlib import Path

# NOTE: this default is the ORIGINAL search folder, whose R2 numbers are
# inflated by the train/test lag-window leakage bug fixed 2026-08-16 (LagMat
# was applied once globally before the CV split; now per-fold). Post-fix work
# uses a SEPARATE folder so dedup can't confuse a leaky run with a correct one
# of the same hyperparameters -- pass:
#   python build_search_grid.py --search-root \
#     /nexus/posix0/MBR-neuralsystems/ephys/GLC7189/GLC7189_c_g0/theo_pipeline_params_search_lagfixed
# (build_search_grid_lagfixed_batch2.py already points at the lagfixed folder.)
SEARCH_ROOT = Path("/nexus/posix0/MBR-neuralsystems/ephys/GLC7189/GLC7189_c_g0/theo_pipeline_params_search")
RESULTS_ROOT = SEARCH_ROOT / "results"

# =============================================================================
# What to search
# =============================================================================

# Dataset paths and experiment-level defaults theo_pipeline_new.py needs.
# Prepended to every manifest row; anything a sweep/grid function overrides
# later in the same line wins (argparse takes the last value for a repeated
# flag). This is the one place to point the search at a different animal,
# session, or dataset path.
#
# All PC ranges for GLC7189_c_g0 share this one SEARCH_ROOT/results/ folder
# (pc_cols is part of PIPELINE_DEFAULTS below, so a PC6-15 row is never
# confused with an already-completed PC1-5 row of the same model +
# hyperparameters). Editing --pc-cols here changes which PC range the NEXT
# `python build_search_grid.py` invocation targets -- pair it with --models
# to restrict to the right subset:
#   - WF/WC/SVR/XGB are each wrapped in sklearn's MultiOutputRegressor,
#     fitting a completely independent single-output model per PC, so a
#     PC1-5 run and a PC6-15 run of these four are equivalent to one PC1-15
#     run -- generate/submit them as two separate --pc-cols passes with
#     --models WF,WC,SVR,XGB.
#   - FNN/RNN/GRU/LSTM are genuinely joint multi-output networks (one shared
#     hidden representation feeding every PC at once), so they can't be
#     split this way -- always generate/submit these with the full PC1-15
#     --pc-cols and --models FNN,RNN,GRU,LSTM.
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
    "--pc-cols": "PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10,PC11,PC12,PC13,PC14,PC15",
    "--dt": 0.2,
    "--window-before": 1.2,
    "--bin-current": 1,
    "--window-after": 1.2,
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
    # NOTE (SVR + --dt, PC1-15): --dt 0.5 is unworkable for SVR at this scale
    # -- roughly doubles the sample count vs --dt 1 (finer time resolution),
    # and SVR's fit cost blows up non-linearly with sample count. Confirmed
    # across the whole --bin-before range in the phase-2 custom sweep:
    # bin_before>=50 OOMs within ~35min (16GB mem limit), bin_before<50 runs
    # but needs 1-8+ days per run (per-fold cost up to ~3.5h at
    # bin_before=40), far past the 23:59:59 SLURM time cap. --dt 0.1 (part of
    # DT_VALUES below) would be even worse. If SVR is ever swept over --dt
    # again, keep it at --dt >= 1.
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
# window is swept symmetrically (--window-before = --window-after = window,
# --bin-current fixed at 1).
#
# WINDOW_VALUES is in SECONDS, not bin count -- passed straight through to
# --window-before/--window-after (theo_pipeline_new.py converts to bins
# internally via round(window / dt)). This is what makes the dt x window x
# n_splits heatmaps' window axis directly comparable across every dt facet:
# under the old bin-count scheme, the same literal "--bin-before 10" meant a
# 1s lookback at dt=0.1 but a 40s lookback at dt=4 -- confounding window
# duration with dt instead of letting them vary independently. Same 10
# literal values kept as before this migration (just reinterpreted as
# seconds), so this is a units fix, not a redesign of what's covered --
# note that DOES shift the actual feature-matrix size tested at small dt: at
# dt=0.1, window=40 now means 400 lag bins (was 40), not 40 bins (4s) --
# watch for the same OOM/runtime blowup pattern documented on GRID["SVR"]
# above if dt gets swept down near 0.1-0.2 again for a model sensitive to it.
DT_VALUES = [0.1, 0.2, 0.5, 1, 2, 4]
WINDOW_VALUES = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40]
N_SPLITS_VALUES = [5, 10, 15, 20, 25, 30, 35, 40]

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
    """dt x lag-window(seconds) x n-splits, each model at its own default
    hyperparams."""
    rows = []
    for model in models:
        for dt, window, n_splits in itertools.product(DT_VALUES, WINDOW_VALUES, N_SPLITS_VALUES):
            overrides = {
                "--dt": dt, "--window-before": window, "--bin-current": 1, "--window-after": window,
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
# defaults. dt/bin_*/n_splits/pc_cols instead reference FIXED_ARGS directly
# (not separately hardcoded) since those are never left at theo_pipeline_new.py's
# own argparse default here -- every row sets them explicitly via FIXED_ARGS,
# so the fallback that matters is THIS SEARCH_ROOT's current FIXED_ARGS, not
# the pipeline's own default. This also makes pc_cols part of a run's identity
# for dedup purposes -- required now that multiple PC ranges (see --pc-cols
# above) can share one SEARCH_ROOT/results/ folder: without it, a PC6-15 row
# would look identical to an already-completed PC1-5 row with the same
# hyperparameters and get wrongly skipped as "already done".
PIPELINE_DEFAULTS = {
    "dt": FIXED_ARGS["--dt"], "window_before": FIXED_ARGS["--window-before"],
    "bin_current": FIXED_ARGS["--bin-current"], "window_after": FIXED_ARGS["--window-after"],
    "n_splits": FIXED_ARGS["--n-splits"], "pc_cols": FIXED_ARGS["--pc-cols"],
    "input_preprocess": "zscore", "output_preprocess": "demean", "spike_thresh": 0,
    "wc_degree": 3, "svr_c": 3.0, "svr_max_iter": 5000,
    "xgb_max_depth": 3, "xgb_n_estimators": 200, "xgb_learning_rate": 0.1,
    "fnn_num_units": 400, "fnn_frac_dropout": 0.25, "fnn_max_epochs": 10, "fnn_n_layers": 2,
    "rnn_num_units": 400, "rnn_frac_dropout": 0.25, "rnn_max_epochs": 10,  # no rnn_n_layers -- see GRID["RNN"]
    "gru_num_units": 400, "gru_frac_dropout": 0.25, "gru_max_epochs": 10, "gru_n_layers": 2,
    "lstm_num_units": 400, "lstm_frac_dropout": 0.25, "lstm_max_epochs": 10, "lstm_n_layers": 2,
}
NUMERIC_KEYS = {k for k, v in PIPELINE_DEFAULTS.items() if isinstance(v, (int, float))}

# Runs from before --window-before/--window-after (seconds) existed stored
# bin_before/bin_after -- raw bin COUNTS -- in cli_args instead, under
# theo_pipeline_new.py's then-defaults of 6/6. Only used as the fallback
# bin count when deriving such a run's equivalent window in seconds below;
# never used directly as a PIPELINE_DEFAULTS value.
_LEGACY_BIN_DEFAULTS = {"window_before": 6, "window_after": 6}


def legacy_window_seconds(values, window_key):
    """window_before/window_after (seconds) for a run whose cli_args
    predates that flag -- a pre-migration run has no such key at all (it
    predates the flag), so naively falling back to today's default window
    would make every one of those runs look like it tested exactly that
    default regardless of what it actually ran, corrupting any downstream
    use of the value (dedup matching here, or historical-run plotting in
    analyze_param_search.ipynb, which imports and calls this same
    function -- the migration logic only lives here, once).

    Derives the true equivalent window in seconds from that run's own
    bin_before/bin_after (bin COUNT) * dt instead -- the same physical
    quantity, just read from its old representation."""
    legacy_key = "bin_before" if window_key == "window_before" else "bin_after"
    bin_count = values.get(legacy_key, _LEGACY_BIN_DEFAULTS[window_key])
    dt_val = values.get("dt", PIPELINE_DEFAULTS["dt"])
    return bin_count * dt_val


def resolve_config(values):
    """Fill in PIPELINE_DEFAULTS for anything missing from `values` (a dict
    keyed by PIPELINE_DEFAULTS' own names, e.g. {"dt": 0.5} -- either a run's
    cli_args, or a row's overrides after flag names are underscored), casting
    numeric fields to float for stable comparison. window_before/window_after
    get special-cased -- see legacy_window_seconds()."""
    cfg = {}
    for k, default in PIPELINE_DEFAULTS.items():
        if k in ("window_before", "window_after") and k not in values:
            v = legacy_window_seconds(values, k)
        else:
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
    global SEARCH_ROOT, RESULTS_ROOT, FIXED_ARGS_STR

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default=None,
        help=f"Comma-separated subset of models to include (default: all of {ALL_MODELS}).")
    p.add_argument("--skip-pipeline-sweep", action="store_true", help="Omit the dt/window/n-splits sweep.")
    p.add_argument("--skip-svr-output-preprocess-sweep", action="store_true", help="Omit the SVR --output-preprocess sweep.")
    p.add_argument("--skip-spike-thresh-sweep", action="store_true", help="Omit the --spike-thresh sweep.")
    p.add_argument("--manifest-name", default="search_manifest",
        help="Base filename (no extension) for the manifest/grid-record pair. Use a distinct "
        "name when generating for a different --pc-cols/--models target against a SEARCH_ROOT "
        "that still has another target's manifest actively being submitted -- submit_search.sh "
        "and the SLURM array tasks it launches keep reading this exact file by path for as long "
        "as any of their rows are still pending, so overwriting it out from under an in-flight "
        "submission would hand a pending task a completely different row than the one it was "
        "queued for.")
    p.add_argument("--search-root", default=None,
        help="Point this generation at a different results tree entirely (default: SEARCH_ROOT "
        "above). Use this for a search that shouldn't mix with an existing SEARCH_ROOT's "
        "results/ at all -- e.g. after a pipeline correctness fix, where old and new runs "
        "of the same hyperparameters are NOT the same experiment and dedup must never treat "
        "one as satisfying the other. Dataset/animal args (--recording-path etc.) are "
        "unaffected -- only where results get written and where dedup looks for existing ones.")
    args = p.parse_args()

    if args.search_root:
        SEARCH_ROOT = Path(args.search_root)
        RESULTS_ROOT = SEARCH_ROOT / "results"
        FIXED_ARGS["--output-root"] = str(SEARCH_ROOT)
        FIXED_ARGS_STR = " ".join(f"{k} {v}" for k, v in FIXED_ARGS.items())

    models = [m.strip() for m in args.models.split(",")] if args.models else ALL_MODELS
    unknown = [m for m in models if m not in ALL_MODELS]
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}. Known: {ALL_MODELS}")

    manifest_path = SEARCH_ROOT / f"{args.manifest_name}.tsv"
    grid_record_path = SEARCH_ROOT / f"{args.manifest_name}.json"

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
    manifest_path.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""))
    grid_record_path.write_text(json.dumps(
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
    print(f"\nManifest        -> {manifest_path}")
    print(f"Grid definition -> {grid_record_path}")
    if new_rows:
        print(f"\nSubmit with:\n  nohup ./submit_search.sh {manifest_path} > submit_search.log 2>&1 &\n  disown")


if __name__ == "__main__":
    main()
