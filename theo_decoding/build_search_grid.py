#!/usr/bin/env python3
"""Build a search manifest for the theo_pipeline_new.py hyperparameter search:
one line per run, each line a complete, standalone set of CLI args (FIXED_ARGS
below plus whatever that row's sweep point overrides). run_theo_pipeline_raven.sh
just forwards a line verbatim -- it knows nothing about datasets or defaults.

WHAT TO EDIT EACH TIME: SWEEPS (the named sweep blocks) and ACTIVE_BLOCKS (which
of them to generate), plus FIXED_ARGS if the dataset changes. That's the whole
search definition; the logic below never needs touching.

HOW ROWS ARE BUILT:
  1. Each SWEEPS block = {models, axes, fixed}. Within a block, every model is
     crossed with the FULL product of its axes (e.g. dt x window x n_splits x
     any hyperparameter flag). Blocks are then unioned. So "cross everything"
     and "one-factor-at-a-time" sweeps are both just block definitions.
     The special axis/fixed key "window" (no dashes) sets --window-before AND
     --window-after to the same value (symmetric lag window, in seconds).
  2. MODEL_ANCHOR_OVERRIDES are applied under every row (lowest priority), e.g.
     SVR's --svr-max-iter 2000 so a sweep that leaves SVR at defaults doesn't
     hit its 5000-iter timeout. A block that sets the flag itself wins.
  3. Validity filters drop rows that can't produce a real score (reported
     separately, and saved to <manifest-name>_filtered_combos.json):
       - zero-lag: a window > 0 was asked for but round(window/dt) = 0 bins;
       - test fold too small: LagMat zero-pads bin_before + bin_after rows at the
         edges of every CV test fold; if less than MIN_CLEAN_FRACTION of a
         fold's rows are un-padded, the fold is mostly padding. Needs the
         session duration: read from any completed run's cv_summary.json in
         SEARCH_ROOT/results/, or --duration-sec; if neither is available this
         filter is skipped (with a warning).
  4. Dedup: drops rows duplicated within this manifest, and rows already run OR
     currently running in SEARCH_ROOT/results/ (any run dir with a
     run_params.json counts -- written at the very start of a run). The match
     includes --pc-cols (see PIPELINE_DEFAULTS), so multiple PC ranges can share
     one results/ folder. Only flags in PIPELINE_DEFAULTS take part in the match.
  5. Ordering: fast models first (MODEL_SUBMIT_ORDER), then cheapest-first within
     a model (cost proxy n_splits * (window_before + window_after) / dt**2), so
     expensive/doomed combos surface last and can be cancelled cheaply.

Usage:
    python build_search_grid.py --dry-run                  # report only
    python build_search_grid.py                            # write the manifest
    python build_search_grid.py --blocks deep_WF,deep_WC --manifest-name wf_wc
    nohup ./submit_search.sh <SEARCH_ROOT>/search_manifest.tsv > submit_search.log 2>&1 &
    disown

Don't rerun this for the SAME --manifest-name while a previous submit_search.sh
batch of it is still outstanding: a row that's queued but hasn't started yet has
no results/ entry, so it would look "new" again and get submitted twice. For a
second target in flight at the same time, use a distinct --manifest-name.

OUTPUTS (in SEARCH_ROOT, overwritten each run):
  <manifest-name>.tsv                   -- one theo_pipeline_new.py arg-string per line
  <manifest-name>.json                  -- FIXED_ARGS + the blocks used + row counts
  <manifest-name>_filtered_combos.json  -- rows dropped by the validity filters, and why

(This replaces the former build_search_grid_lagfixed_batch2.py, whose per-model
full-cross sweeps, validity filters and ordering are now part of this script.)
"""

import argparse
import itertools
import json
import math
from pathlib import Path

# Results tree this search writes to and dedups against. A pipeline change that
# alters what a run computes needs a NEW folder, so dedup never treats an old run
# as satisfying a new row with the same hyperparameters:
#   theo_pipeline_params_search/          -- original; R2 inflated by the pre-2026-08-16
#                                            global-LagMat leakage bug
#   theo_pipeline_params_search_lagfixed/ -- per-fold LagMat, but neural<->behaviour
#                                            aligned on the time_offset clock
#   theo_pipeline_params_search_aligned/  -- per-fold LagMat + neural<->behaviour aligned
#                                            on the zeroed clock (current pipeline)
SEARCH_ROOT = Path("/nexus/posix0/MBR-neuralsystems/ephys/GLC7189/GLC7189_c_g0/theo_pipeline_params_search_aligned")
RESULTS_ROOT = SEARCH_ROOT / "results"

# =============================================================================
# Dataset / experiment defaults
# =============================================================================

# Prepended to every manifest row; anything a sweep sets later in the same line
# wins (argparse takes the last value for a repeated flag). The one place to point
# the search at a different animal, session, or dataset path.
#
# --pc-cols: WF/WC/SVR/XGB fit an independent model per PC (MultiOutputRegressor),
# so PC ranges can be split into separate passes for them; FNN/RNN/GRU/LSTM are
# joint multi-output networks, so always run those with the full PC range.
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

ALL_MODELS = ["WF", "WC", "SVR", "XGB", "FNN", "RNN", "GRU", "LSTM"]
MODEL_SUBMIT_ORDER = ["WF", "WC", "RNN", "GRU", "LSTM", "FNN", "SVR", "XGB"]  # fast -> slow

# Applied under every row (lowest priority). SVR's own default --svr-max-iter
# (5000) times out (>8h on most runs), so any row that doesn't set it gets 2000.
MODEL_ANCHOR_OVERRIDES = {
    "SVR": {"--svr-max-iter": 2000},
}

MIN_CLEAN_FRACTION = 0.5  # >= this fraction of each CV test fold must be outside the zero-padded zone

# =============================================================================
# THE SEARCH -- edit these each time
# =============================================================================
#
# name -> {"models": [...], "axes": {flag: [values]}, "fixed": {flag: value}}
# (full product of "axes" per model; "window" = symmetric --window-before/after, s)
#
# Notes carried over from past sweeps:
#   - SVR + small --dt: sample count grows as 1/dt and SVR's fit cost grows
#     non-linearly with it; dt=0.5 at large windows OOMs or needs 1-8+ days.
#     Keep SVR at --dt >= 1 unless you've got a reason.
#   - dt=0.2 at n_splits>=30 needs 1-2.5+ days per run under per-fold LagMat
#     (blows the 23:59:59 cap). WC at dt=0.2 + window=32 + n_splits>=20 also does.
#   - --spike-thresh values come from channel_selection.ipynb's spike-count
#     histogram (bimodal, gaps around 300-1000 and past 20000).
SWEEPS = {
    # --- Per-model hyperparameter grids, at FIXED_ARGS' dt/window/n_splits ---
    "hp_WF":   {"models": ["WF"],   "axes": {}},
    "hp_WC":   {"models": ["WC"],   "axes": {"--wc-degree": [1, 2, 3, 4]}},
    "hp_SVR":  {"models": ["SVR"],  "axes": {"--svr-c": [0.3, 1.0, 2.0, 3.0, 10.0, 30.0],
                                             "--svr-max-iter": [1000, 2000, 3500]}},
    "hp_XGB":  {"models": ["XGB"],  "axes": {"--xgb-max-depth": [1, 3, 5, 7, 9],
                                             "--xgb-n-estimators": [100, 200, 400, 600],
                                             "--xgb-learning-rate": [0.01, 0.05, 0.1, 0.2]}},
    "hp_FNN":  {"models": ["FNN"],  "axes": {"--fnn-num-units": [100, 200, 400, 600],
                                             "--fnn-frac-dropout": [0.0, 0.25, 0.5],
                                             "--fnn-max-epochs": [10, 20, 30, 40],
                                             "--fnn-n-layers": [1, 2, 3, 4]}},
    # (RNN has no --rnn-n-layers: Neural_Decoding.nn.RNN takes no n_layers.)
    "hp_RNN":  {"models": ["RNN"],  "axes": {"--rnn-num-units": [20, 50, 100, 200, 400, 600],
                                             "--rnn-frac-dropout": [0.0, 0.25, 0.5, 0.75, 1.0],
                                             "--rnn-max-epochs": [2, 5, 10, 20, 30, 40]}},
    "hp_GRU":  {"models": ["GRU"],  "axes": {"--gru-num-units": [20, 50, 100, 200, 400, 600],
                                             "--gru-frac-dropout": [0.0, 0.25, 0.5, 0.75, 1.0],
                                             "--gru-max-epochs": [2, 5, 10, 20, 30, 40],
                                             "--gru-n-layers": [1, 2, 3, 4]}},
    "hp_LSTM": {"models": ["LSTM"], "axes": {"--lstm-num-units": [20, 50, 100, 200, 400, 600],
                                             "--lstm-frac-dropout": [0.0, 0.25, 0.5, 0.75, 1.0],
                                             "--lstm-max-epochs": [2, 5, 10, 20, 30, 40],
                                             "--lstm-n-layers": [1, 2, 3, 4]}},

    # --- One shared dt x window x n_splits sweep, every model at its defaults ---
    "pipeline_shared": {"models": ALL_MODELS, "axes": {
        "--dt": [0.1, 0.2, 0.5, 1, 2, 4],
        "window": [1, 3, 5, 10, 15, 20, 25, 30, 35, 40],
        "--n-splits": [5, 10, 15, 20, 25, 30, 35, 40]}},

    # --- Small one-off sweeps ---
    "svr_output_zscore": {"models": ["SVR"], "axes": {"--svr-max-iter": [1000, 2000, 3500],
                                                      "--output-preprocess": ["zscore"]},
                          "fixed": {"--svr-c": 3.0}},
    "spike_thresh": {"models": ALL_MODELS, "axes": {"--spike-thresh": [300, 5000, 20000]}},

    # --- Per-model dt x window x n_splits deep sweeps (the former batch2, 2026-08-17) ---
    "deep_WF":   {"models": ["WF"],   "axes": {"--n-splits": [10, 20, 30, 40, 50, 60], "--dt": [0.5, 1, 2, 4, 8],
                                               "window": [2, 4, 8, 16, 32]}},
    "deep_WC":   {"models": ["WC"],   "axes": {"--n-splits": [10, 20, 30, 40, 50, 60], "--dt": [0.5, 1, 2, 4, 8],
                                               "window": [2, 4, 8, 16, 32], "--wc-degree": [2, 3, 4]}},
    "deep_RNN":  {"models": ["RNN"],  "axes": {"--n-splits": [10, 20, 30, 40, 50], "--dt": [0.2, 0.5, 1, 2, 4, 8],
                                               "window": [2, 4, 8]}},
    "deep_GRU":  {"models": ["GRU"],  "axes": {"--n-splits": [10, 20, 30, 40, 50, 60], "--dt": [0.1, 0.2, 0.5, 1, 2, 4, 8],
                                               "window": [2, 4, 8, 16, 32, 64, 128]}},
    "deep_LSTM": {"models": ["LSTM"], "axes": {"--n-splits": [10, 20, 30, 40, 50, 60], "--dt": [0.1, 0.2, 0.5, 1],
                                               "window": [2, 4, 8, 16, 32, 64, 128]}},
    "deep_FNN":  {"models": ["FNN"],  "axes": {"--n-splits": [10, 20, 30, 40], "--dt": [0.2, 0.5, 1, 2, 4, 8],
                                               "window": [2, 4, 8, 16, 32]},
                  "fixed": {"--fnn-frac-dropout": 0.5}},
    "deep_SVR":  {"models": ["SVR"],  "axes": {"--n-splits": [10, 20, 30, 40, 50, 60], "--dt": [0.5, 1, 2, 4, 8],
                                               "window": [2, 4, 8, 16]},
                  "fixed": {"--svr-max-iter": 2000, "--svr-c": 40.0}},
}

# Which blocks a plain `python build_search_grid.py` generates (override: --blocks).
ACTIVE_BLOCKS = ["deep_WF", "deep_WC", "deep_RNN", "deep_GRU", "deep_LSTM", "deep_FNN", "deep_SVR"]


# =============================================================================
# Row building
# =============================================================================

def expand_window(overrides):
    """'window' shorthand -> --window-before + --window-after (+ --bin-current 1)."""
    if "window" not in overrides:
        return overrides
    out = {k: v for k, v in overrides.items() if k != "window"}
    w = overrides["window"]
    out["--window-before"] = w
    out["--window-after"] = w
    out.setdefault("--bin-current", 1)
    return out


def block_rows(name, block, models_filter):
    """All (model, overrides) rows of one block: models x full product of axes."""
    axes = block.get("axes", {})
    fixed = block.get("fixed", {})
    flags = list(axes)
    rows = []
    for model in block["models"]:
        if models_filter and model not in models_filter:
            continue
        for combo in itertools.product(*axes.values()):
            overrides = {
                **MODEL_ANCHOR_OVERRIDES.get(model, {}),
                **expand_window(fixed),
                **expand_window(dict(zip(flags, combo))),
            }
            rows.append((model, overrides))
    return rows


def format_row(model, overrides):
    """(model, overrides) -> the complete CLI-arg string for this row."""
    parts = [FIXED_ARGS_STR, f"--models {model}"]
    parts += [f"{flag} {val}" for flag, val in overrides.items()]
    return " ".join(parts)


# =============================================================================
# Validity filters + ordering
# =============================================================================

def effective(overrides, flag, default):
    """A row's value for flag: its own override, else FIXED_ARGS, else default."""
    return float(overrides.get(flag, FIXED_ARGS.get(flag, default)))


def row_windows(overrides):
    dt = effective(overrides, "--dt", 0.2)
    wb = effective(overrides, "--window-before", 1.2)
    wa = effective(overrides, "--window-after", 1.2)
    return dt, wb, wa, int(round(wb / dt)), int(round(wa / dt))  # same rounding as theo_pipeline_new.py


def invalid_reason(overrides, duration_sec):
    """None if the row can produce a real score, else a short reason string."""
    dt, wb, wa, bb, ba = row_windows(overrides)
    if (wb > 0 and bb < 1) or (wa > 0 and ba < 1):
        return f"zero-lag: window {wb:g}/{wa:g}s at dt={dt:g} rounds to {bb}/{ba} bins"
    if duration_sec is None:
        return None
    n_splits = int(effective(overrides, "--n-splits", 3))
    holdout_frac = effective(overrides, "--holdout-frac", 0.2)
    n_total = int(duration_sec // dt)
    n_search = n_total - round(n_total * holdout_frac)
    if n_search < n_splits:
        return f"test-fold-too-small: {n_search} samples < {n_splits} folds"
    test_fold = n_search // n_splits  # KFold(shuffle=False)'s smallest fold
    clean = test_fold - (bb + ba)
    if clean < MIN_CLEAN_FRACTION * test_fold:
        return f"test-fold-too-small: fold={test_fold} rows, {clean} un-padded (<{MIN_CLEAN_FRACTION:.0%})"
    return None


def cost_proxy(overrides):
    """Relative cost within a model: small dt dominates (more samples AND more lag
    bins, and LagMat runs twice per fold), window and n_splits matter less."""
    dt, wb, wa, _, _ = row_windows(overrides)
    return effective(overrides, "--n-splits", 3) * (wb + wa) / dt ** 2


def infer_duration_sec():
    """Session duration from any completed run in RESULTS_ROOT (trimmed_overlap is
    the same physical overlap for every dt), or None if there is none yet."""
    for run_dir in RESULTS_ROOT.glob("[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_*"):
        try:
            t0, t1 = json.loads((run_dir / "cv_summary.json").read_text())["config"]["raw_ranges"]["trimmed_overlap"]
            return float(t1) - float(t0)
        except (FileNotFoundError, KeyError, json.JSONDecodeError, ValueError):
            continue
    return None


# =============================================================================
# Dedup: skip anything already run or currently running (scans results/)
# =============================================================================

# What an unset flag resolves to, for matching a manifest row against a real
# run's cli_args -- must stay in sync with theo_pipeline_new.py's parse_args()
# defaults. dt/window/n_splits/pc_cols instead reference FIXED_ARGS directly,
# since every row sets them via FIXED_ARGS. This also makes pc_cols part of a
# run's identity, so a PC6-15 row is never mistaken for a done PC1-5 row.
PIPELINE_DEFAULTS = {
    "dt": FIXED_ARGS["--dt"], "window_before": FIXED_ARGS["--window-before"],
    "bin_current": FIXED_ARGS["--bin-current"], "window_after": FIXED_ARGS["--window-after"],
    "n_splits": FIXED_ARGS["--n-splits"], "pc_cols": FIXED_ARGS["--pc-cols"],
    "input_preprocess": "zscore", "output_preprocess": "demean", "spike_thresh": 0,
    "wc_degree": 3, "svr_c": 3.0, "svr_max_iter": 5000,
    "xgb_max_depth": 3, "xgb_n_estimators": 200, "xgb_learning_rate": 0.1,
    "fnn_num_units": 400, "fnn_frac_dropout": 0.25, "fnn_max_epochs": 10, "fnn_n_layers": 2,
    "rnn_num_units": 400, "rnn_frac_dropout": 0.25, "rnn_max_epochs": 10,  # no rnn_n_layers
    "gru_num_units": 400, "gru_frac_dropout": 0.25, "gru_max_epochs": 10, "gru_n_layers": 2,
    "lstm_num_units": 400, "lstm_frac_dropout": 0.25, "lstm_max_epochs": 10, "lstm_n_layers": 2,
}
NUMERIC_KEYS = {k for k, v in PIPELINE_DEFAULTS.items() if isinstance(v, (int, float))}

# Runs from before --window-before/--window-after (seconds) existed stored
# bin_before/bin_after -- raw bin COUNTS -- in cli_args instead, under
# theo_pipeline_new.py's then-defaults of 6/6.
_LEGACY_BIN_DEFAULTS = {"window_before": 6, "window_after": 6}


def legacy_window_seconds(values, window_key):
    """window_before/window_after (seconds) for a run whose cli_args predates
    that flag: derived from its own bin_before/bin_after (bin COUNT) * dt, rather
    than today's default window (which would make every such run look like it
    tested the default). Also imported by analyze_param_search.ipynb."""
    legacy_key = "bin_before" if window_key == "window_before" else "bin_after"
    bin_count = values.get(legacy_key, _LEGACY_BIN_DEFAULTS[window_key])
    dt_val = values.get("dt", PIPELINE_DEFAULTS["dt"])
    return bin_count * dt_val


def resolve_config(values):
    """Fill in PIPELINE_DEFAULTS for anything missing from `values` (a dict keyed
    by PIPELINE_DEFAULTS' names -- a run's cli_args, or a row's overrides with
    flag names underscored), casting numerics to float for stable comparison."""
    cfg = {}
    for k, default in PIPELINE_DEFAULTS.items():
        if k in ("window_before", "window_after") and k not in values:
            v = legacy_window_seconds(values, k)
        else:
            v = values.get(k, default)
        cfg[k] = float(v) if k in NUMERIC_KEYS and v is not None else v
    return cfg


def row_config(model, overrides):
    """A manifest row's (model, config) pair, restricted to PIPELINE_DEFAULTS keys."""
    values = {flag[2:].replace("-", "_"): v for flag, v in overrides.items()}
    return model, resolve_config(values)


def config_key(model, cfg):
    return (model,) + tuple((k, round(v, 9) if isinstance(v, float) else v) for k, v in sorted(cfg.items()))


def existing_configs():
    """One (model, config) per results/ run dir that has at least a
    run_params.json -- written at the start of a run, so in-progress runs count."""
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


def drop_duplicates(rows):
    """rows: list of (model, overrides). Returns (kept, n_duplicate_in_manifest,
    n_already_in_results). First occurrence of a config wins."""
    done = {config_key(m, cfg) for m, cfg in existing_configs()}
    seen, kept, n_dup, n_done = set(), [], 0, 0
    for model, overrides in rows:
        key = config_key(*row_config(model, overrides))
        if key in seen:
            n_dup += 1
        elif key in done:
            n_done += 1
        else:
            seen.add(key)
            kept.append((model, overrides))
    return kept, n_dup, n_done


# =============================================================================

def main():
    global SEARCH_ROOT, RESULTS_ROOT, FIXED_ARGS_STR

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blocks", default=None,
        help=f"Comma-separated SWEEPS block names to generate (default: ACTIVE_BLOCKS). Known: {list(SWEEPS)}")
    p.add_argument("--models", default=None,
        help=f"Comma-separated subset of models to keep from those blocks (default: all of {ALL_MODELS}).")
    p.add_argument("--manifest-name", default="search_manifest",
        help="Base filename (no extension) for the manifest + records. Use a distinct name for "
        "a second target while another manifest is still being submitted.")
    p.add_argument("--search-root", default=None,
        help="Results tree to write to and dedup against (default: SEARCH_ROOT above).")
    p.add_argument("--duration-sec", type=float, default=None,
        help="Session duration for the test-fold-size filter, if no completed run exists yet.")
    p.add_argument("--dry-run", action="store_true", help="Print the report only; don't write files.")
    args = p.parse_args()

    if args.search_root:
        SEARCH_ROOT = Path(args.search_root)
        RESULTS_ROOT = SEARCH_ROOT / "results"
        FIXED_ARGS["--output-root"] = str(SEARCH_ROOT)
        FIXED_ARGS_STR = " ".join(f"{k} {v}" for k, v in FIXED_ARGS.items())

    block_names = [b.strip() for b in args.blocks.split(",")] if args.blocks else ACTIVE_BLOCKS
    unknown = [b for b in block_names if b not in SWEEPS]
    if unknown:
        raise ValueError(f"Unknown block(s) {unknown}. Known: {list(SWEEPS)}")
    models_filter = [m.strip() for m in args.models.split(",")] if args.models else None
    if models_filter and (bad := [m for m in models_filter if m not in ALL_MODELS]):
        raise ValueError(f"Unknown model(s) {bad}. Known: {ALL_MODELS}")

    duration_sec = args.duration_sec if args.duration_sec is not None else infer_duration_sec()
    if duration_sec is None:
        print("WARNING: no completed run in results/ and no --duration-sec -- skipping the "
              "test-fold-size filter (the zero-lag filter still applies).")
    else:
        print(f"Session duration for filtering: {duration_sec:.1f}s")

    # 1-2. Build every block's rows.
    all_rows, per_block = [], {}
    for name in block_names:
        rows = block_rows(name, SWEEPS[name], models_filter)
        per_block[name] = len(rows)
        all_rows.extend(rows)

    # 3. Validity filters.
    valid_rows, filtered = [], []
    for model, overrides in all_rows:
        reason = invalid_reason(overrides, duration_sec)
        if reason:
            filtered.append({"model": model, "overrides": overrides, "reason": reason})
        else:
            valid_rows.append((model, overrides))

    # 4. Dedup (within the manifest + against results/).
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    kept, n_dup, n_done = drop_duplicates(valid_rows)

    # 5. Fast models first, cheapest-first within a model.
    rank = {m: i for i, m in enumerate(MODEL_SUBMIT_ORDER)}
    kept.sort(key=lambda r: (rank.get(r[0], len(rank)), cost_proxy(r[1])))

    for name, n in per_block.items():
        print(f"  block {name:<22} -> {n} rows")
    print(f"{'considered':<30} {len(all_rows)}")
    print(f"{'filtered as not runnable':<30} {len(filtered)}")
    reasons = {}
    for f in filtered:
        reasons[f['reason'].split(':')[0]] = reasons.get(f['reason'].split(':')[0], 0) + 1
    for r, n in sorted(reasons.items()):
        print(f"  - {r:<28} {n}")
    print(f"{'duplicate within manifest':<30} {n_dup}")
    print(f"{'already done/running':<30} {n_done}")
    print(f"{'new rows':<30} {len(kept)}")
    per_model = {}
    for m, _ in kept:
        per_model[m] = per_model.get(m, 0) + 1
    print(f"{'new rows per model':<30} {per_model}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    manifest_path = SEARCH_ROOT / f"{args.manifest_name}.tsv"
    record_path = SEARCH_ROOT / f"{args.manifest_name}.json"
    filtered_path = SEARCH_ROOT / f"{args.manifest_name}_filtered_combos.json"
    manifest_lines = [format_row(m, o) for m, o in kept]
    manifest_path.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""))
    record_path.write_text(json.dumps({
        "fixed_args": FIXED_ARGS,
        "model_anchor_overrides": MODEL_ANCHOR_OVERRIDES,
        "blocks": {name: SWEEPS[name] for name in block_names},
        "models_filter": models_filter,
        "duration_sec": duration_sec,
        "n_rows_considered": len(all_rows),
        "n_rows_filtered": len(filtered),
        "n_rows_duplicate_in_manifest": n_dup,
        "n_rows_already_done_or_running": n_done,
        "n_rows_new": len(kept),
    }, indent=2))
    filtered_path.write_text(json.dumps(filtered, indent=2))

    print(f"\nManifest         -> {manifest_path}")
    print(f"Search record    -> {record_path}")
    print(f"Filtered combos  -> {filtered_path}")
    if kept:
        print(f"\nSubmit with:\n  nohup ./submit_search.sh {manifest_path} > submit_search.log 2>&1 &\n  disown")


if __name__ == "__main__":
    main()
