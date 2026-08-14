"""Phase 1 of the staged custom follow-up sweep:
- FNN/GRU/LSTM/SVR/XGB: hyperparameter-only search (the swept knobs from the
  user's spec), at DEFAULT dt/bin_before/n_splits. Cheap -- meant to be
  analyzed afterward to pick each model's best hyperparam combo, which then
  becomes the FIXED point for phase 2's dt x window x n_splits sweep (so
  that sweep is additive against one point, not crossed against the whole
  hyperparameter grid).
- RNN/WC/WF: no hyperparameter to pre-search (nothing was given as a list
  for these three -- everything is a fixed constant), so their dt x window x
  n_splits sweep can run directly, no phase-2 follow-up needed for them.

Row order in the manifest matches the user's explicit request: fast/cheap
models first, SVR and XGB last.
"""
import importlib.util
import itertools
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("bsg", "/raven/u/gongt/scripts/theo_decoding/build_search_grid.py")
bsg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bsg)

RESULTS_ROOT = bsg.RESULTS_ROOT
FIXED_ARGS_COMMON = {k: v for k, v in bsg.FIXED_ARGS.items()
                     if k not in ("--pc-cols", "--dt", "--bin-before", "--bin-current", "--bin-after", "--n-splits")}

PC1_5 = "PC1,PC2,PC3,PC4,PC5"
PC6_15 = "PC6,PC7,PC8,PC9,PC10,PC11,PC12,PC13,PC14,PC15"
PC1_15 = "PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10,PC11,PC12,PC13,PC14,PC15"

BASE_HP_DEFAULTS = {
    "wc_degree": 3, "svr_c": 3.0, "svr_max_iter": 5000,
    "xgb_max_depth": 3, "xgb_n_estimators": 200, "xgb_learning_rate": 0.1,
    "fnn_num_units": 400, "fnn_frac_dropout": 0.25, "fnn_max_epochs": 10, "fnn_n_layers": 2,
    "rnn_num_units": 400, "rnn_frac_dropout": 0.25, "rnn_max_epochs": 10,
    "gru_num_units": 400, "gru_frac_dropout": 0.25, "gru_max_epochs": 10, "gru_n_layers": 2,
    "lstm_num_units": 400, "lstm_frac_dropout": 0.25, "lstm_max_epochs": 10, "lstm_n_layers": 2,
}


def pipeline_defaults_for(pc_cols):
    d = dict(BASE_HP_DEFAULTS)
    d.update({
        "dt": 0.2, "bin_before": 6, "bin_current": 1, "bin_after": 6, "n_splits": 10,
        "pc_cols": pc_cols, "input_preprocess": "zscore", "output_preprocess": "demean",
        "spike_thresh": 0,
    })
    return d


NUMERIC_KEYS_ALL = {k for k, v in pipeline_defaults_for(PC1_5).items() if isinstance(v, (int, float))}


def resolve_config(values, pdefaults):
    cfg = {}
    for k, default in pdefaults.items():
        v = values.get(k, default)
        cfg[k] = float(v) if k in NUMERIC_KEYS_ALL and v is not None else v
    return cfg


def configs_match(a, b, pdefaults, tol=1e-9):
    for k in pdefaults:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, float) or isinstance(vb, float):
            if abs(float(va) - float(vb)) > tol:
                return False
        elif va != vb:
            return False
    return True


def existing_configs(pdefaults):
    configs = []
    for run_dir in RESULTS_ROOT.glob("[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_*"):
        try:
            params = json.loads((run_dir / "run_params.json").read_text())
            cli, models = params["cli_args"], params["resolved"]["selected_models"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            continue
        cfg = resolve_config(cli, pdefaults)
        configs.extend((model, cfg) for model in models)
    return configs


def format_row(model, overrides):
    fixed_str = " ".join(f"{k} {v}" for k, v in FIXED_ARGS_COMMON.items())
    parts = [fixed_str, f"--models {model}"]
    parts += [f"{flag} {val}" for flag, val in overrides.items()]
    return " ".join(parts)


# =============================================================================
# Ordered list (fast/cheap first, SVR/XGB last): each entry is
# (model, pc_targets, base_overrides, swept_dict_or_None).
# swept_dict=None means "no hyperparameter search -- go straight to the
# dt x window x n_splits sweep using the FULL ranges from the user's spec".
# =============================================================================

DT_WINDOW_NSPLITS = {
    "FNN": {"--dt": [0.5, 1, 2], "--bin-before": [10, 20, 30], "--n-splits": [40, 50, 60]},
    "GRU": {"--dt": [2, 4, 6, 8, 10], "--bin-before": [40, 50, 60], "--n-splits": [40, 50, 60]},
    "LSTM": {"--dt": [0.5, 1, 2, 4, 6, 8, 10], "--bin-before": [40, 50, 60], "--n-splits": [40, 50, 60]},
    "RNN": {"--dt": [0.5, 1, 2, 4, 6, 8, 10], "--bin-before": [40, 50, 60], "--n-splits": [40, 50, 60]},
    "SVR": {"--dt": [0.5, 1], "--bin-before": [10, 20, 30, 40, 50, 60], "--n-splits": [20, 30, 40, 50, 60]},
    "WC": {"--dt": [2, 4, 6, 8, 10], "--bin-before": [10, 20, 30, 40, 50, 60], "--n-splits": [20, 30, 40, 50, 60]},
    "WF": {"--dt": [2, 4, 6, 8, 10], "--bin-before": [10, 20, 30, 40, 50, 60], "--n-splits": [20, 30, 40, 50, 60]},
    "XGB": {"--dt": [2, 4, 6, 8, 10], "--bin-before": [10, 20, 30, 40, 50, 60], "--n-splits": [20, 30, 40, 50, 60]},
}

PLAN = [
    # -- ready now: no hyperparameter to pre-search, straight to dt/window/n_splits --
    ("RNN", [PC1_15], {"--rnn-num-units": 400, "--rnn-max-epochs": 10, "--rnn-frac-dropout": 0.25}, None),
    ("WC", [PC1_5, PC6_15], {}, None),
    ("WF", [PC1_5, PC6_15], {}, None),
    # -- phase 1: cheap hyperparameter-only search (default dt/window/n_splits) --
    ("FNN", [PC1_15], {"--fnn-num-units": 400, "--fnn-max-epochs": 10, "--fnn-n-layers": 2},
     {"--fnn-frac-dropout": [0, 0.5, 0.75]}),
    ("GRU", [PC1_15], {"--gru-num-units": 400, "--gru-max-epochs": 10, "--gru-n-layers": 4},
     {"--gru-frac-dropout": [0.25, 0.5, 0.75]}),
    ("LSTM", [PC1_15], {"--lstm-num-units": 400, "--lstm-max-epochs": 10, "--lstm-frac-dropout": 0.25},
     {"--lstm-n-layers": [2, 3]}),
    ("SVR", [PC1_5, PC6_15], {"--output-preprocess": "zscore"},
     {"--svr-max-iter": [2000, 3500, 5000], "--svr-c": [30, 40, 50]}),
    ("XGB", [PC1_5, PC6_15], {},
     {"--xgb-max-depth": [3, 5, 7], "--xgb-n-estimators": [600, 800, 1000], "--xgb-learning-rate": [0.1, 0.2]}),
]

ALL_ROWS = []  # (model, overrides, pc_cols, stage)
for model, pc_targets, base_overrides, hp_swept in PLAN:
    if hp_swept is None:
        # straight to full dt x window x n_splits sweep
        dwn = DT_WINDOW_NSPLITS[model]
        keys = list(dwn.keys())
        for pc_cols in pc_targets:
            for combo in itertools.product(*dwn.values()):
                overrides = {"--spike-thresh": 0, "--output-preprocess": "demean", **base_overrides}
                overrides.update(dict(zip(keys, combo)))
                overrides["--bin-after"] = overrides["--bin-before"]
                overrides["--bin-current"] = 1
                overrides["--pc-cols"] = pc_cols
                ALL_ROWS.append((model, overrides, pc_cols, "stage2-direct"))
    else:
        keys = list(hp_swept.keys())
        for pc_cols in pc_targets:
            for combo in itertools.product(*hp_swept.values()):
                overrides = {"--spike-thresh": 0, "--output-preprocess": "demean", **base_overrides}
                overrides.update(dict(zip(keys, combo)))
                overrides["--pc-cols"] = pc_cols
                # Explicit, not left to theo_pipeline_new.py's own argparse
                # defaults -- its --n-splits default is 3, not the 10 every
                # other row in this search (and PIPELINE_DEFAULTS here) uses,
                # so leaving it unset would silently test at n_splits=3 while
                # dedup assumed 10.
                overrides["--dt"] = 0.2
                overrides["--bin-before"] = 6
                overrides["--bin-current"] = 1
                overrides["--bin-after"] = 6
                overrides["--n-splits"] = 10
                ALL_ROWS.append((model, overrides, pc_cols, "stage1-hp-search"))

print(f"Considered (before dedup): {len(ALL_ROWS)} rows")
by_model_count = {}
for model, _, _, stage in ALL_ROWS:
    by_model_count.setdefault(model, {"stage1-hp-search": 0, "stage2-direct": 0})[stage] += 1
for m, d in by_model_count.items():
    print(f"  {m}: {d}")

cache = {}
kept = []
dropped = 0
for model, overrides, pc_cols, stage in ALL_ROWS:
    pdefaults = pipeline_defaults_for(pc_cols)
    if pc_cols not in cache:
        cache[pc_cols] = {}
        for m, cfg in existing_configs(pdefaults):
            cache[pc_cols].setdefault(m, []).append(cfg)
    values = {flag[2:].replace("-", "_"): v for flag, v in overrides.items()}
    cfg = resolve_config(values, pdefaults)
    existing_for_model = cache[pc_cols].get(model, [])
    if any(configs_match(cfg, e, pdefaults) for e in existing_for_model):
        dropped += 1
        continue
    kept.append((model, overrides))

print(f"\nAlready done/running: {dropped}")
print(f"New: {len(kept)}")

MANIFEST_PATH = bsg.SEARCH_ROOT / "search_manifest_custom_phase1.tsv"
lines = [format_row(model, overrides) for model, overrides in kept]
MANIFEST_PATH.write_text("\n".join(lines) + ("\n" if lines else ""))
print(f"\nManifest -> {MANIFEST_PATH}")

by_model_new = {}
for model, _ in kept:
    by_model_new[model] = by_model_new.get(model, 0) + 1
print("New rows per model (in manifest order):")
for m in ["RNN", "WC", "WF", "FNN", "GRU", "LSTM", "SVR", "XGB"]:
    print(f"  {m}: {by_model_new.get(m, 0)}")
