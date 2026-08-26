#!/usr/bin/env python3
"""Picks the fastest-to-retrain completed runs and refits them COMPLETELY
using her pipeline's own literal choices (see
/raven/u/gongt/Neural_Decoding_sklearn/Examples_sklearn_cross_validation.ipynb),
not just an sklearn-composed equivalent of this project's architecture --
that's what test_r2_method.py already checks (and confirms matches exactly,
since it's the same architecture either way). This script instead asks a
different question: how much do her specific choices actually change the
R2 numbers?

Two concrete differences from theo_pipeline_new.py's own pipeline, both
deliberately reproduced here exactly as her notebook has them:
  1. Pipeline order is scaler -> lagmat -> model (she scales the RAW,
     unlagged channels; this project's pipeline lags first, then scales
     the ALREADY-LAGGED feature matrix -- see the earlier pipeline
     comparison for why that's a real, not cosmetic, difference).
  2. No target/output preprocessing at all -- y goes into the pipeline
     raw, unlike theo_pipeline_new.py's fit-on-train-only demean/zscore
     (fit_with_output_preprocess/predict_original_scale).
Everything else (KFold(shuffle=False), LagMat's bin_before/current/after,
scoring via r2_score/"r2", per-fold not pooled) is identical -- see
test_r2_method.py's docstring and the architecture comparison this session
covered for the full list.

Only WF (RidgeCV) and WC (WienerCascade) are wired up -- both
deterministic and fast to refit; add entries to REGRESSOR_BUILDERS for
SVR/XGB if you need those checked too.

Usage:
    python test_pipeline_method.py                    # auto-picks 10 fastest WF/WC runs
    python test_pipeline_method.py --n-runs 5 --n-pcs 15
    python test_pipeline_method.py --run-dirs 20260818_070426_WF_na25,20260818_083450_WC_deg3
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Neural_Decoding.nonlinear import WienerCascade
from Neural_Decoding.preprocessing_funcs import LagMat

SEARCH_ROOT = Path("/nexus/posix0/MBR-neuralsystems/ephys/GLC7189/GLC7189_c_g0/theo_pipeline_params_search_lagfixed")
ALL_RUNS_CSV = SEARCH_ROOT / "analysis_plots" / "all_runs_long.csv"

# Deliberately NOT wrapped in MultiOutputRegressor -- see test_r2_method.py's
# same note: fitting one PC at a time (this script's loop below) IS what
# MultiOutputRegressor does internally for these naturally single-output
# estimators, so this is a faithful, not approximate, per-PC reproduction.
REGRESSOR_BUILDERS = {
    "WF": lambda cli: RidgeCV(alphas=(
        np.array([float(a) for a in cli["wf_alphas"].split(",") if a.strip()])
        if cli.get("wf_alphas") else np.logspace(-2, 4, 25)
    )),
    "WC": lambda cli: WienerCascade(degree=cli["wc_degree"]),
}


def pick_fastest_runs(n_runs):
    if not ALL_RUNS_CSV.exists():
        raise SystemExit(f"{ALL_RUNS_CSV} not found -- run analyze_param_search.ipynb first to generate it.")
    df = pd.read_csv(ALL_RUNS_CSV)
    sub = df[df.model.isin(REGRESSOR_BUILDERS)].drop_duplicates("run_dir")
    sub = sub.sort_values("runtime_min").head(n_runs)
    return sub[["run_dir", "model", "runtime_min"]].to_dict("records")


def check_one_run(run_dir_name, model, n_pcs, atol):
    run_path = SEARCH_ROOT / "results" / run_dir_name
    run_params = json.loads((run_path / "run_params.json").read_text())
    cv_summary = json.loads((run_path / "cv_summary.json").read_text())
    cli = run_params["cli_args"]

    dt = float(cli["dt"])
    window_before = float(cli.get("window_before", cli.get("bin_before", 6) * dt))
    n_splits = int(cli["n_splits"])
    holdout_frac = float(cli.get("holdout_frac", 0.2))
    pc_cols = run_params["resolved"]["pc_cols"]
    bin_before = round(window_before / dt)
    bin_after = bin_before

    with open(run_path / "formatted_data_like_hc.pickle", "rb") as f:
        X, y = pickle.load(f)
    n_total = X.shape[0]
    n_holdout = int(round(n_total * holdout_frac)) if holdout_frac > 0 else 0
    X_search_raw = X[: n_total - n_holdout]
    y_search = y[: n_total - n_holdout]

    kf = KFold(n_splits=n_splits, shuffle=False)
    saved_fold_r2 = cv_summary["models"][model]["fold_r2_raw_values"]

    n_pcs = min(n_pcs, len(pc_cols))
    per_pc_results = []
    for pc_idx in range(n_pcs):
        # HER literal pipeline: scaler -> lagmat -> model, no target
        # preprocessing at all (contrast with theo_pipeline_new.py's
        # lag -> scale order plus fit-on-train demean/zscore).
        her_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lagmat", LagMat(bin_before, 1, bin_after, flat=True)),
            ("regressor", REGRESSOR_BUILDERS[model](cli)),
        ])
        cv_results = cross_validate(
            her_pipeline, X_search_raw, y_search[:, pc_idx], cv=kf, scoring="r2", n_jobs=1,
        )
        her_r2 = cv_results["test_score"]
        saved_r2 = np.array([saved_fold_r2[f][pc_idx] for f in range(n_splits)])
        diff = her_r2 - saved_r2
        per_pc_results.append({
            "pc": pc_cols[pc_idx],
            "her_mean": her_r2.mean(), "saved_mean": saved_r2.mean(),
            "mean_diff": diff.mean(), "max_abs_diff": np.abs(diff).max(),
        })
    return per_pc_results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-runs", type=int, default=10, help="How many fastest WF/WC runs to auto-select (ignored if --run-dirs given).")
    p.add_argument("--run-dirs", default=None, help="Comma-separated explicit run_dir names instead of auto-selecting.")
    p.add_argument("--n-pcs", type=int, default=5, help="How many PCs to check per run, from PC1 (default: 5, for speed -- pass 15 for the full check).")
    p.add_argument("--atol", type=float, default=1e-3, help="Per-fold abs diff above which a PC is flagged as diverging.")
    args = p.parse_args()

    if args.run_dirs:
        # Need each run's own model -- look it up from run_params.json rather
        # than requiring the caller to pass it separately.
        runs = []
        for run_dir in args.run_dirs.split(","):
            rp = json.loads((SEARCH_ROOT / "results" / run_dir / "run_params.json").read_text())
            model = rp["resolved"]["selected_models"][0]
            if model not in REGRESSOR_BUILDERS:
                print(f"Skipping {run_dir}: model={model} not wired up (only {list(REGRESSOR_BUILDERS)}).")
                continue
            runs.append({"run_dir": run_dir, "model": model, "runtime_min": None})
    else:
        runs = pick_fastest_runs(args.n_runs)

    print(f"Checking {len(runs)} run(s), {args.n_pcs} PC(s) each, her literal pipeline vs saved results:\n")

    all_diffs = []
    for r in runs:
        rt = f"{r['runtime_min']:.2f}min" if r["runtime_min"] is not None else "?"
        print(f"=== {r['run_dir']} (model={r['model']}, original runtime={rt}) ===")
        results = check_one_run(r["run_dir"], r["model"], args.n_pcs, args.atol)
        for res in results:
            flag = "DIVERGES" if res["max_abs_diff"] > args.atol else "close"
            print(f"  {res['pc']:6s} [{flag:8s}] her_mean={res['her_mean']:+.4f}  "
                  f"saved_mean={res['saved_mean']:+.4f}  mean_diff={res['mean_diff']:+.4f}  "
                  f"max_abs_fold_diff={res['max_abs_diff']:.4f}")
            all_diffs.append(res["mean_diff"])
        print()

    all_diffs = np.array(all_diffs)
    print("=== Summary across everything checked ===")
    print(f"n comparisons: {len(all_diffs)}")
    print(f"mean(her - saved): {all_diffs.mean():+.4f}")
    print(f"std:               {all_diffs.std():.4f}")
    print(f"min / max:         {all_diffs.min():+.4f} / {all_diffs.max():+.4f}")


if __name__ == "__main__":
    main()
