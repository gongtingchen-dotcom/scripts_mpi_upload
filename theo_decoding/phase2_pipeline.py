"""Run once phase 1 has fully drained:
1. Rank PC1..PC15 by the MEAN OF EACH PC'S OWN TOP-10 oof_r2 ever achieved
   (across EVERY completed run in the whole results/ folder), take the 4
   most decodable PCs. NOT a plain median/mean -- see compute_most_decodable_pcs
   docstring for why.
2. For each of FNN/GRU/LSTM/SVR/XGB, find the best hyperparameter combo
   from phase 1's cheap hyperparameter-only search (restricted to exactly
   the combos the user specified -- not the original full grid), scored by
   median oof_r2 over ONLY those 4 top PCs (not all 15 -- a hyperparameter
   choice shouldn't be diluted by PCs that are hard to decode regardless of
   model/config).
3. Build + submit phase 2: dt x window x n_splits crossed against that ONE
   winning combo (not the whole hyperparam grid), as a single PC1-15 job
   per model.
RNN/WC/WF already got their full dt/window/n_splits sweep in phase 1 (they
had nothing to pre-search), so nothing further happens for them here.

Self-contained: does not depend on the parent Claude session staying alive.
All decisions and actions are logged to LOG_PATH; the chosen top-4 PCs and
each model's winning hyperparameters are also written to
phase2_selected_params.json for a clean, non-log record.
"""
import importlib.util
import itertools
import json
import statistics
import subprocess
import sys
from pathlib import Path
from datetime import datetime

LOG_PATH = Path("/raven/u/gongt/scripts/theo_decoding/phase2_pipeline.log")


def log(msg):
    line = f"{datetime.now().isoformat()} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


spec = importlib.util.spec_from_file_location("bsg", "/raven/u/gongt/scripts/theo_decoding/build_search_grid.py")
bsg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bsg)

RESULTS_ROOT = bsg.RESULTS_ROOT
SEARCH_ROOT = bsg.SEARCH_ROOT
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
# Model specs: (model, pc_targets, base_overrides (fixed, non-swept),
# stage1_swept (what to search over for the "best combo"),
# phase2_dt_window_nsplits (what to sweep in phase 2))
# =============================================================================

# pc_targets: which PC range(s) phase 1's stage-1 hyperparameter-only rows
# were actually submitted at (needed to correctly read that data back for
# analysis) -- unchanged from phase 1, already submitted, not touched here.
# phase2_pc_targets: which PC range(s) phase 2's dt x window x n_splits
# sweep should run at. For WF/WC/SVR/XGB (each fits every PC completely
# independently via sklearn's MultiOutputRegressor -- see build_models() in
# theo_pipeline_new.py), a PC1-5 job + a PC6-15 job is mathematically
# identical to one PC1-15 job, so phase 2 now runs all of them as a single
# PC1-15 job like FNN/RNN/GRU/LSTM always have -- one pattern everywhere,
# per explicit request, since the split-for-parallelism benefit wasn't
# worth the confusion of two different conventions.
MODEL_SPECS = {
    "FNN": dict(pc_targets=[PC1_15], phase2_pc_targets=[PC1_15],
                base_overrides={"fnn_num_units": 400, "fnn_max_epochs": 10, "fnn_n_layers": 2},
                stage1_swept={"fnn_frac_dropout": [0, 0.5, 0.75]},
                phase2_dwn={"dt": [0.5, 1, 2], "bin_before": [10, 20, 30], "n_splits": [40, 50, 60]}),
    "GRU": dict(pc_targets=[PC1_15], phase2_pc_targets=[PC1_15],
                base_overrides={"gru_num_units": 400, "gru_max_epochs": 10, "gru_n_layers": 4},
                stage1_swept={"gru_frac_dropout": [0.25, 0.5, 0.75]},
                phase2_dwn={"dt": [2, 4, 6, 8, 10], "bin_before": [40, 50, 60], "n_splits": [40, 50, 60]}),
    "LSTM": dict(pc_targets=[PC1_15], phase2_pc_targets=[PC1_15],
                 base_overrides={"lstm_num_units": 400, "lstm_max_epochs": 10, "lstm_frac_dropout": 0.25},
                 stage1_swept={"lstm_n_layers": [2, 3]},
                 phase2_dwn={"dt": [0.5, 1, 2, 4, 6, 8, 10], "bin_before": [40, 50, 60], "n_splits": [40, 50, 60]}),
    "SVR": dict(pc_targets=[PC1_5, PC6_15], phase2_pc_targets=[PC1_15],
                base_overrides={"output_preprocess": "zscore"},
                stage1_swept={"svr_max_iter": [2000, 3500, 5000], "svr_c": [30, 40, 50]},
                # dt=0.5 dropped from the original [0.5, 1] spec: at PC1-15 it
                # roughly doubles the sample count vs dt=1 (finer time
                # resolution), and SVR's fit cost blows up non-linearly with
                # sample count. Confirmed unworkable across the whole
                # bin_before range -- bin_before>=50 OOMs within ~35min
                # (16GB mem limit), bin_before<50 runs but needs 1-8+ days to
                # finish (per-fold cost up to ~3.5h at bin_before=40), far
                # past the 23:59:59 SLURM time cap. The 19 in-flight
                # dt=0.5/bin_before<50 tasks were cancelled rather than left
                # to run out the clock for nothing.
                phase2_dwn={"dt": [1], "bin_before": [10, 20, 30, 40, 50, 60], "n_splits": [20, 30, 40, 50, 60]}),
    "XGB": dict(pc_targets=[PC1_5, PC6_15], phase2_pc_targets=[PC1_15],
                base_overrides={},
                stage1_swept={"xgb_max_depth": [3, 5, 7], "xgb_n_estimators": [600, 800, 1000], "xgb_learning_rate": [0.1, 0.2]},
                phase2_dwn={"dt": [2, 4, 6, 8, 10], "bin_before": [10, 20, 30, 40, 50, 60], "n_splits": [20, 30, 40, 50, 60]}),
}


def compute_most_decodable_pcs(n=4):
    """Rank PC1..PC15 by the mean of each PC's own top-10 oof_r2 ever
    achieved, across EVERY completed run in the whole results/ folder
    (every model, every hyperparameter combo, every PC range) --
    decodability treated as a property of the PC signal itself, not tied
    to any one model or sweep. Returns (top_n_pc_names,
    full_ranking_list_of_(pc,top10_mean,n)_sorted_desc)."""
    by_pc = {}
    for d in RESULTS_ROOT.iterdir():
        if not d.is_dir():
            continue
        pf, sf = d / "run_params.json", d / "cv_summary.json"
        if not pf.exists() or not sf.exists():
            continue
        try:
            params = json.loads(pf.read_text())
            summary = json.loads(sf.read_text())
        except json.JSONDecodeError:
            continue
        pc_cols = params["resolved"]["pc_cols"]
        for model, m in summary.get("models", {}).items():
            oof = m.get("oof_r2_per_output")
            if not oof:
                continue
            for pc, r2 in zip(pc_cols, oof):
                by_pc.setdefault(pc, []).append(r2)

    # mean of each PC's own top-10 R2 ever achieved -- NOT a plain median
    # or mean over everything. Two separate problems with those:
    #  1. A plain mean is wrecked by rare catastrophic numerical blowups
    #     elsewhere in the results (e.g. the known WC degree=3/bin_before=1
    #     divergence, oof_r2 ~ -1e24) that dominate an average over
    #     thousands of runs.
    #  2. A plain median is biased by how MUCH history exists per PC, not
    #     just how decodable it is: PC1-5 carry ~3100 extra older/less-
    #     refined runs from the original grid search that PC6-15 were never
    #     part of, which drags PC1/PC2's median down even though they've
    #     achieved the highest peak R2 of any PC (0.68, vs PC6's 0.42) --
    #     confirmed by checking top10_mean directly against max() per PC.
    # Top-10-mean sidesteps both: it reflects "what's achievable under a
    # good config" (matching what "most decodable" should mean) while still
    # averaging several runs, not just the single luckiest one.
    TOP_K = 10
    ranking = sorted(
        ((pc, statistics.mean(sorted(vals, reverse=True)[:TOP_K]), len(vals)) for pc, vals in by_pc.items()),
        key=lambda t: -t[1],
    )
    top_n = [pc for pc, _, _ in ranking[:n]]
    return top_n, ranking


def find_best_combo(model, spec, top_pcs):
    """Scan completed runs for this model's stage-1 hyperparam search (at
    default dt=0.2/bin_before=6/n_splits=10, base_overrides fixed, swept
    keys within EXACTLY the values the user specified), aggregate mean
    oof_r2 RESTRICTED TO top_pcs only (not all 15 -- the user specifically
    wants the hyperparameter choice driven by the most-decodable PCs, not
    diluted by PCs that are hard to decode for any model/config), return
    the best combo dict or None if nothing completed yet."""
    swept_keys = list(spec["stage1_swept"].keys())
    by_combo = {}  # combo tuple -> list of oof_r2, restricted to top_pcs, across pc_targets
    for d in RESULTS_ROOT.iterdir():
        if not d.is_dir():
            continue
        pf, sf = d / "run_params.json", d / "cv_summary.json"
        if not pf.exists() or not sf.exists():
            continue
        params = json.loads(pf.read_text())
        cli = params["cli_args"]
        if model not in params["resolved"]["selected_models"]:
            continue
        pc_cols = params["resolved"]["pc_cols"]
        if ",".join(pc_cols) not in spec["pc_targets"]:
            continue
        if cli.get("dt", 0.2) != 0.2 or cli.get("bin_before", 6) != 6 or cli.get("n_splits", 10) != 10:
            continue
        if any(cli.get(k) != v for k, v in spec["base_overrides"].items()):
            continue
        combo = tuple(cli.get(k) for k in swept_keys)
        if any(combo[i] not in spec["stage1_swept"][k] for i, k in enumerate(swept_keys)):
            continue
        summary = json.loads(sf.read_text())
        oof = summary.get("models", {}).get(model, {}).get("oof_r2_per_output")
        if not oof:
            continue
        wanted = [r2 for pc, r2 in zip(pc_cols, oof) if pc in top_pcs]
        if not wanted:
            continue  # this run's pc range doesn't overlap top_pcs at all
        by_combo.setdefault(combo, []).extend(wanted)

    if not by_combo:
        return None, by_combo
    best_combo, best_vals = max(by_combo.items(), key=lambda kv: statistics.median(kv[1]))
    return dict(zip(swept_keys, best_combo)), by_combo


def build_phase2_rows(model, spec, best_combo):
    keys = list(spec["phase2_dwn"].keys())
    rows = []
    for pc_cols in spec["phase2_pc_targets"]:
        for combo in itertools.product(*spec["phase2_dwn"].values()):
            overrides = {"spike_thresh": 0, "output_preprocess": spec["base_overrides"].get("output_preprocess", "demean")}
            overrides.update(spec["base_overrides"])
            overrides.update(best_combo)
            overrides.update(dict(zip(keys, combo)))
            overrides["bin_current"] = 1
            overrides["bin_after"] = overrides["bin_before"]
            overrides["pc_cols"] = pc_cols
            rows.append((model, overrides, pc_cols))
    return rows


def to_cli_overrides(model, overrides):
    flag_map = {
        "spike_thresh": "--spike-thresh", "output_preprocess": "--output-preprocess",
        "dt": "--dt", "bin_before": "--bin-before", "bin_current": "--bin-current", "bin_after": "--bin-after",
        "n_splits": "--n-splits", "pc_cols": "--pc-cols",
        "fnn_num_units": "--fnn-num-units", "fnn_max_epochs": "--fnn-max-epochs", "fnn_n_layers": "--fnn-n-layers",
        "fnn_frac_dropout": "--fnn-frac-dropout",
        "gru_num_units": "--gru-num-units", "gru_max_epochs": "--gru-max-epochs", "gru_n_layers": "--gru-n-layers",
        "gru_frac_dropout": "--gru-frac-dropout",
        "lstm_num_units": "--lstm-num-units", "lstm_max_epochs": "--lstm-max-epochs",
        "lstm_frac_dropout": "--lstm-frac-dropout", "lstm_n_layers": "--lstm-n-layers",
        "svr_max_iter": "--svr-max-iter", "svr_c": "--svr-c",
        "xgb_max_depth": "--xgb-max-depth", "xgb_n_estimators": "--xgb-n-estimators",
        "xgb_learning_rate": "--xgb-learning-rate",
    }
    return {flag_map[k]: v for k, v in overrides.items() if k in flag_map}


def phase1_still_outstanding():
    """True if submit_search.sh hasn't finished submitting everything, or if
    any theo_search SLURM task is still pending/running."""
    log_path = Path("/raven/u/gongt/scripts/theo_decoding/submit_search_custom_phase1.log")
    submitted_all = log_path.exists() and "All 694 rows submitted" in log_path.read_text()
    try:
        out = subprocess.run(["squeue", "-u", "gongt", "-n", "theo_search", "-r", "-h"],
                              capture_output=True, text=True, timeout=30).stdout
        n_outstanding = len([l for l in out.splitlines() if l.strip()])
    except Exception as e:
        log(f"WARNING: squeue check failed ({e}), assuming still outstanding")
        return True
    return not submitted_all or n_outstanding > 0


def main():
    log("=== phase2_pipeline started, waiting for phase 1 to drain ===")
    import time
    while phase1_still_outstanding():
        time.sleep(180)
    log("Phase 1 fully drained (all submitted, queue empty). Analyzing results...")

    top_pcs, full_ranking = compute_most_decodable_pcs(n=4)
    log("PC decodability ranking (mean of each PC's own top-10 oof_r2 ever achieved, across every completed run, every model, every combo):")
    for pc, mean_r2, n in full_ranking:
        marker = " <== TOP 4" if pc in top_pcs else ""
        log(f"  {pc}: top10_mean_oof_r2={mean_r2:.3f} (n={n}){marker}")
    log(f"Most decodable 4 PCs (used to score every model's stage-1 hyperparameter search below): {top_pcs}")

    selected_params = {"top_4_pcs": top_pcs, "pc_ranking": [{"pc": pc, "top10_mean_oof_r2": r2, "n": n} for pc, r2, n in full_ranking],
                        "models": {}}

    all_rows = []
    summary_lines = []
    for model, spec in MODEL_SPECS.items():
        best_combo, by_combo = find_best_combo(model, spec, top_pcs)
        n_combos_seen = len(by_combo)
        n_combos_expected = 1
        for v in spec["stage1_swept"].values():
            n_combos_expected *= len(v)
        if best_combo is None:
            log(f"{model}: NO completed stage-1 results found (among top-4 PCs) -- skipping phase 2 for this model.")
            summary_lines.append(f"{model}: SKIPPED (no stage-1 data)")
            selected_params["models"][model] = {"status": "skipped_no_data"}
            continue
        best_mean = statistics.median(by_combo[tuple(best_combo[k] for k in spec["stage1_swept"])])
        log(f"{model}: best combo {best_combo} (median oof_r2 over top-4 PCs={best_mean:.3f}, "
            f"seen {n_combos_seen}/{n_combos_expected} stage-1 combos completed)")
        summary_lines.append(f"{model}: best={best_combo} median_oof_r2_top4pcs={best_mean:.3f} "
                              f"({n_combos_seen}/{n_combos_expected} stage-1 combos in)")
        selected_params["models"][model] = {
            "status": "ok", "best_hyperparams": best_combo, "median_oof_r2_over_top4_pcs": best_mean,
            "stage1_combos_completed": f"{n_combos_seen}/{n_combos_expected}",
        }
        rows = build_phase2_rows(model, spec, best_combo)
        all_rows.extend(rows)

    params_path = Path("/raven/u/gongt/scripts/theo_decoding/phase2_selected_params.json")
    params_path.write_text(json.dumps(selected_params, indent=2))
    log(f"Selected params (top-4 PCs + each model's winning hyperparameters) written -> {params_path}")

    log(f"Phase 2 candidate rows (all models, before dedup): {len(all_rows)}")

    cache = {}
    kept = []
    for model, overrides, pc_cols in all_rows:
        pdefaults = pipeline_defaults_for(pc_cols)
        if pc_cols not in cache:
            cache[pc_cols] = {}
            for m, cfg in existing_configs(pdefaults):
                cache[pc_cols].setdefault(m, []).append(cfg)
        cfg = resolve_config(overrides, pdefaults)
        existing_for_model = cache[pc_cols].get(model, [])
        if any(configs_match(cfg, e, pdefaults) for e in existing_for_model):
            continue
        kept.append((model, overrides))

    log(f"Phase 2 new rows after dedup: {len(kept)}")

    if not kept:
        log("Nothing new to submit for phase 2 (either no stage-1 data completed for any model, "
            "or everything already exists). Stopping here.")
        return

    lines = [format_row(model, to_cli_overrides(model, overrides)) for model, overrides in kept]
    MANIFEST_PATH = SEARCH_ROOT / "search_manifest_custom_phase2.tsv"
    MANIFEST_PATH.write_text("\n".join(lines) + "\n")
    log(f"Phase 2 manifest written -> {MANIFEST_PATH}")
    log("Summary:\n  " + "\n  ".join(summary_lines))

    by_model_new = {}
    for model, _ in kept:
        by_model_new[model] = by_model_new.get(model, 0) + 1
    log("Phase 2 rows per model: " + ", ".join(f"{m}={c}" for m, c in sorted(by_model_new.items())))

    log("Submitting phase 2 via submit_search.sh ...")
    submit_log = Path("/raven/u/gongt/scripts/theo_decoding/submit_search_custom_phase2.log")
    with open(submit_log, "w") as f:
        subprocess.Popen(
            ["./submit_search.sh", str(MANIFEST_PATH)],
            cwd="/raven/u/gongt/scripts/theo_decoding",
            stdout=f, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    log(f"Phase 2 submission launched in background, log -> {submit_log}")
    log("=== phase2_pipeline finished its job; submit_search.sh continues independently ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("FATAL ERROR:\n" + traceback.format_exc())
        raise
