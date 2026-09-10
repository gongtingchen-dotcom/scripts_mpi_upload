#!/usr/bin/env python3
"""Confirms theo_pipeline_new.py's saved per-fold R2 is arithmetically correct,
by loading each fold's ALREADY-FITTED model (models/<model>_fold<i>.joblib --
no retraining at all) and independently recomputing R2 on that same fold's
held-out predictions with sklearn's own r2_score -- the function scikit's "r2"
scoring string resolves to under the hood, same as
/raven/u/gongt/Neural_Decoding_sklearn/Examples_sklearn_cross_validation.ipynb.

RESULT: on the runs checked, the recomputed values match exactly (to float
tolerance). So the number the pipeline records IS a real sklearn r2_score on
held-out data -- not a bug or a nonstandard formula.

WHICH pipeline number this validates, precisely:
  cv_summary.json -> models -> <model> -> fold_r2_raw_values[fold][pc]
  i.e. the PER-FOLD, PER-PC held-out R2, each computed as
  r2_score(y_search[test_idx], pred, multioutput="raw_values").
Directly implied (it's just the fold-axis mean of the above):
  mean_fold_r2_per_output[pc].
NOT checked here -- different quantities:
  oof_r2_per_output[pc]  -- r2_score over ALL out-of-fold predictions pooled
                            into one array at once (not a mean of per-fold R2).
  holdout_r2_per_output[pc]  -- the reserved chronological holdout block.

This does NOT retrain anything: it's a pure re-scoring check, isolating the
R2 arithmetic from any question about whether refitting the model a second
time would reproduce the same fit (a separate, harder question this script
doesn't attempt -- see the earlier pipeline-comparison writeup for the
known architecture differences, e.g. her scaler-before-lag order vs this
project's lag-before-scale, that WOULD make a full retrain diverge).

Two things get cross-checked per fold, per PC:
  1. Predictions: model.predict() on that fold's own lagged test slice
     (inverting the demean/zscore target preprocessing the same way
     predict_original_scale() does) should exactly match the saved
     oof_pred slice in predictions/<model>_oof_predictions.npz.
  2. R2: r2_score(y_test, pred, multioutput="raw_values") recomputed here
     should exactly match fold_r2_raw_values[fold][pc] in cv_summary.json.

Usage:
    python test_r2_method.py                          # uses the default run below
    python test_r2_method.py <run_dir_name>
    python test_r2_method.py 20260816_021324_WF_na25 --n-pcs 5
"""

import argparse
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from Neural_Decoding.preprocessing_funcs import LagMat

SEARCH_ROOT = Path("/nexus/posix0/MBR-neuralsystems/ephys/GLC7189/GLC7189_c_g0/theo_pipeline_params_search_lagfixed")
DEFAULT_RUN_DIR = "20260816_021324_WF_na25"


def invert_output_preprocess(y, mu, sd):
    """Same as theo_pipeline_new.py's own invert_output_preprocess -- not
    imported directly since that requires the full theo_pipeline_new.py
    module (heavy imports: spikeinterface, torch, xgboost, ...) just for
    one four-line function."""
    if mu is None:
        return y
    if sd is not None:
        y = y * sd
    return y + mu


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", nargs="?", default=DEFAULT_RUN_DIR,
        help=f"Results dir name under {SEARCH_ROOT}/results/ (default: {DEFAULT_RUN_DIR}).")
    p.add_argument("--n-pcs", type=int, default=15, help="How many PCs to check, from PC1 (default: all 15).")
    p.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for the match check "
        "(combined with rtol=1e-5, per np.allclose -- tiny diffs are expected from float32 vs "
        "float64 arithmetic somewhere in the chain, not a real mismatch).")
    args = p.parse_args()

    run_path = SEARCH_ROOT / "results" / args.run_dir
    run_params = json.loads((run_path / "run_params.json").read_text())
    cv_summary = json.loads((run_path / "cv_summary.json").read_text())
    cli = run_params["cli_args"]
    selected = run_params["resolved"]["selected_models"]
    if len(selected) != 1:
        raise SystemExit(f"This run selected models={selected!r} -- expected exactly one.")
    model = selected[0]

    dt = float(cli["dt"])
    window_before = float(cli.get("window_before", cli.get("bin_before", 6) * dt))
    n_splits = int(cli["n_splits"])
    output_preprocess = cli.get("output_preprocess", "demean")
    holdout_frac = float(cli.get("holdout_frac", 0.2))
    pc_cols = run_params["resolved"]["pc_cols"]

    bin_before = round(window_before / dt)
    bin_after = bin_before  # symmetric sweep -- see build_search_grid.py's WINDOW_VALUES note

    print(f"Run: {args.run_dir}")
    print(f"  model={model}  dt={dt:g}  window_before={window_before:g}  n_splits={n_splits}  output_preprocess={output_preprocess}")

    # formatted_data_like_hc.pickle = [X, y] -- the FULL (pre-holdout,
    # pre-lag) arrays theo_pipeline_new.py itself saves before splitting.
    with open(run_path / "formatted_data_like_hc.pickle", "rb") as f:
        X, y = pickle.load(f)
    n_total = X.shape[0]
    n_holdout = int(round(n_total * holdout_frac)) if holdout_frac > 0 else 0
    X_search_raw = X[: n_total - n_holdout]
    y_search = y[: n_total - n_holdout]
    print(f"  X_search_raw: {X_search_raw.shape}, y_search: {y_search.shape}")

    d = np.load(run_path / "predictions" / f"{model}_oof_predictions.npz")
    saved_oof_pred = d["y_pred"]

    kf = KFold(n_splits=n_splits, shuffle=False)
    saved_fold_r2 = cv_summary["models"][model]["fold_r2_raw_values"]  # [fold][pc]
    n_pcs = min(args.n_pcs, len(pc_cols))

    print(f"\nLoading {n_splits} already-fitted models (no retraining) and re-scoring "
          f"{n_pcs}/{len(pc_cols)} PCs...\n")

    pred_all_match = True
    r2_all_match = True
    worst_pred_diff = 0.0
    worst_r2_diff = 0.0

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_search_raw)):
        fitted_model = joblib.load(run_path / "models" / f"{model}_fold{fold}.joblib")
        X_test = LagMat(bin_before, 1, bin_after, flat=True).fit_transform(X_search_raw[test_idx])

        # mu/sd are never saved to disk (computed transiently inside the
        # original fold loop and used immediately) -- recomputing them from
        # y_train is a cheap statistic, not a retrain.
        y_train = y_search[train_idx]
        mu = np.nanmean(y_train, axis=0, keepdims=True)
        sd = np.nanstd(y_train, axis=0, keepdims=True) + 1e-9 if output_preprocess == "zscore" else None

        pred = invert_output_preprocess(fitted_model.predict(X_test), mu, sd)
        pred_diff = np.abs(pred - saved_oof_pred[test_idx])
        worst_pred_diff = max(worst_pred_diff, pred_diff.max())
        if not np.allclose(pred, saved_oof_pred[test_idx], atol=args.atol, rtol=1e-5):
            pred_all_match = False
            print(f"fold {fold}: PREDICTION MISMATCH vs saved oof_pred, max|diff|={pred_diff.max():.3e}")

        recomputed_r2 = r2_score(y_search[test_idx, :n_pcs], pred[:, :n_pcs], multioutput="raw_values")
        saved_r2 = np.array(saved_fold_r2[fold][:n_pcs])
        r2_diff = np.abs(recomputed_r2 - saved_r2)
        worst_r2_diff = max(worst_r2_diff, r2_diff.max())
        if not np.allclose(recomputed_r2, saved_r2, atol=args.atol, rtol=1e-5):
            r2_all_match = False
            print(f"fold {fold}: R2 MISMATCH  saved={np.round(saved_r2, 4).tolist()}  "
                  f"recomputed={np.round(recomputed_r2, 4).tolist()}")

    print()
    print(f"Predictions vs saved oof_pred: {'ALL MATCH' if pred_all_match else 'MISMATCHES ABOVE'} "
          f"(worst abs diff: {worst_pred_diff:.3e})")
    print(f"R2 vs saved fold_r2_raw_values: {'ALL MATCH' if r2_all_match else 'MISMATCHES ABOVE'} "
          f"(worst abs diff: {worst_r2_diff:.3e})")


if __name__ == "__main__":
    main()
