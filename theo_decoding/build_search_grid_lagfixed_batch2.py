#!/usr/bin/env python3
"""Second targeted sweep against the lagfixed search
(theo_pipeline_params_search_lagfixed/results/), requested 2026-08-17: a
full n_splits x dt x window cross per model, with per-model value lists
(not one shared DT_VALUES/WINDOW_VALUES/N_SPLITS_VALUES like
build_search_grid.py's pipeline_sweep_rows -- this batch's ranges genuinely
differ per model, e.g. GRU/LSTM go down to dt=0.1 and up to window=128,
RNN only goes up to window=8).

Reuses build_search_grid.py's dedup/formatting machinery as a library
(FIXED_ARGS, format_row, drop_already_done, existing_configs) rather than
forking it, so a row here that's already in results/ -- from the original
lagfixed pipeline sweep OR a prior run of this exact script -- is skipped
the same way. Does NOT touch build_search_grid.py or its own
search_manifest.tsv (that manifest is still being actively submitted by a
long-running submit_search.sh as of this writing -- overwriting it out
from under that process would hand a queued task the wrong row; see
build_search_grid.py's own module docstring).

Two validity filters drop combinations that can't produce a meaningful
result, reported separately from "already done" so it's clear which rows
were never attempted at all:

  1. Zero-lag: bin_before = round(window/dt) must be >= 1 -- otherwise the
     lag window collapses to nothing (e.g. dt=8, window=2 -> round(0.25) = 0
     bins of history, not "a small amount of history", literally none).

  2. Test fold too small for the lag window: theo_pipeline_new.py's lag()
     zero-pads the first bin_before and last bin_after rows of EVERY fold's
     own test block (see analyze_param_search.ipynb's "LagMat zero-padded
     zone" shading -- added this same day after the WC boundary-artifact
     discussion). If the test fold itself is barely bigger than that
     zero-padded span, most or all of the fold is padding, not real signal
     -- e.g. n_splits=60 with a big window at a coarse dt can make the test
     fold *smaller* than the zero-padded span on its own. Requires at least
     half of each fold's test rows to be un-padded (test_fold_size >=
     2 * (2*bin_before)); flags anything under that bar as not viable
     rather than silently submitting a run that can't produce a real score.

Usage:
    python build_search_grid_lagfixed_batch2.py            # report + write manifest
    python build_search_grid_lagfixed_batch2.py --dry-run   # report only, no file written

    nohup ./submit_search.sh search_manifest_batch2.tsv > submit_search_batch2.log 2>&1 &
    disown
"""

import argparse
import importlib.util
import itertools
import json
from pathlib import Path

THEO_DECODING_DIR = Path("/raven/u/gongt/scripts/theo_decoding")
LAGFIXED_ROOT = Path("/nexus/posix0/MBR-neuralsystems/ephys/GLC7189/GLC7189_c_g0/theo_pipeline_params_search_lagfixed")
RESULTS_ROOT = LAGFIXED_ROOT / "results"
MANIFEST_NAME = "search_manifest_batch2"

# --- Import build_search_grid.py as a library, repointed at the lagfixed
# root (same effect as its own --search-root flag) -- so dedup/formatting
# can never drift out of sync with the one already used for the first
# lagfixed batch. ---
_spec = importlib.util.spec_from_file_location("build_search_grid", THEO_DECODING_DIR / "build_search_grid.py")
bsg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bsg)
bsg.SEARCH_ROOT = LAGFIXED_ROOT
bsg.RESULTS_ROOT = RESULTS_ROOT
bsg.FIXED_ARGS["--output-root"] = str(LAGFIXED_ROOT)
bsg.FIXED_ARGS_STR = " ".join(f"{k} {v}" for k, v in bsg.FIXED_ARGS.items())

# --- Session duration, inferred from any one completed lagfixed run rather
# than hardcoded -- every dt's cv_summary.json config.raw_ranges.trimmed_overlap
# is the same physical recording/behavior overlap regardless of dt, so any
# completed run gives the same answer. Needed for the test-fold-size filter
# (rule 2 above): n_total(dt) = floor(duration_sec / dt). ---
def infer_duration_sec():
    for run_dir in RESULTS_ROOT.glob("[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_*"):
        f = run_dir / "cv_summary.json"
        if not f.exists():
            continue
        try:
            cfg = json.loads(f.read_text())["config"]
            t0, t1 = cfg["raw_ranges"]["trimmed_overlap"]
            return float(t1) - float(t0)
        except (KeyError, json.JSONDecodeError, ValueError):
            continue
    raise RuntimeError(f"No completed run with a cv_summary.json found under {RESULTS_ROOT} to infer session duration from.")


DURATION_SEC = infer_duration_sec()
HOLDOUT_FRAC = 0.2  # must match FIXED_ARGS / theo_pipeline_new.py's --holdout-frac default
MIN_CLEAN_FRACTION = 0.5  # >=50% of each fold's test rows must be outside the zero-padded zone

# =============================================================================
# The requested grid: n_splits x dt x window, per model, full cross. Any
# hyperparameter not listed here is left unset in the row, so
# theo_pipeline_new.py's own argparse default applies -- confirmed to match
# what the first lagfixed batch actually used (see run_params.json audit),
# EXCEPT svr_c, which the first batch anchored at 40.0 (not the pipeline's
# own default of 3.0) -- carried forward explicitly below so "same as last
# submission batch" holds for SVR too.
# =============================================================================
MODEL_SWEEPS = {
    "FNN": {
        "n_splits": [10, 20, 30, 40],
        "dt": [0.2, 0.5, 1, 2, 4, 8],
        "window": [2, 4, 8, 16, 32],
        "fixed_overrides": {"--fnn-frac-dropout": 0.5},
    },
    "GRU": {
        "n_splits": [10, 20, 30, 40, 50, 60],
        "dt": [0.1, 0.2, 0.5, 1, 2, 4, 8],
        "window": [2, 4, 8, 16, 32, 64, 128],
        "fixed_overrides": {},
    },
    "LSTM": {
        "n_splits": [10, 20, 30, 40, 50, 60],
        "dt": [0.1, 0.2, 0.5, 1],
        "window": [2, 4, 8, 16, 32, 64, 128],
        "fixed_overrides": {},
    },
    "RNN": {
        "n_splits": [10, 20, 30, 40, 50],
        "dt": [0.2, 0.5, 1, 2, 4, 8],
        "window": [2, 4, 8],
        "fixed_overrides": {},
    },
    "SVR": {
        "n_splits": [10, 20, 30, 40, 50, 60],
        "dt": [0.5, 1, 2, 4, 8],
        "window": [2, 4, 8, 16],
        "fixed_overrides": {"--svr-max-iter": 2000, "--svr-c": 40.0},
    },
    "WC": {
        "n_splits": [10, 20, 30, 40, 50, 60],
        "dt": [0.5, 1, 2, 4, 8],
        "window": [2, 4, 8, 16, 32],
        "extra_grid": {"--wc-degree": [2, 3, 4]},
        "fixed_overrides": {},
    },
    "WF": {
        "n_splits": [10, 20, 30, 40, 50, 60],
        "dt": [0.5, 1, 2, 4, 8],
        "window": [2, 4, 8, 16, 32],
        "fixed_overrides": {},
    },
}

# Job-ordering: fast/cheap models first, slow ones last -- see the
# "job-ordering" project feedback (established twice on this project).
MODEL_SUBMIT_ORDER = ["WF", "WC", "RNN", "GRU", "LSTM", "FNN", "SVR"]


def bin_count(window, dt):
    return int(round(window / dt))


def zero_lag_invalid(window, dt):
    return bin_count(window, dt) < 1


def test_fold_too_small(dt, window, n_splits):
    """Returns (invalid: bool, test_fold_size: int, clean_rows: int)."""
    n_total = int(DURATION_SEC // dt)
    n_holdout = round(n_total * HOLDOUT_FRAC)
    n_search = n_total - n_holdout
    if n_search < n_splits:
        return True, 0, 0
    test_fold_size = n_search // n_splits  # KFold(shuffle=False)'s smallest fold
    lag_span = 2 * bin_count(window, dt)  # window_before == window_after, symmetric sweep
    clean_rows = test_fold_size - lag_span
    return clean_rows < MIN_CLEAN_FRACTION * test_fold_size, test_fold_size, clean_rows


def cost_proxy(dt, window, n_splits):
    """Rough relative-cost ordering within a model's own rows -- see the
    job-ordering feedback: small dt is the dominant real-world cost driver
    under the per-fold lag() redesign (more lag bins AND lag() gets called
    twice per fold), window/n_splits matter but less steeply."""
    return n_splits * window / dt ** 2


def build_rows_for_model(model, spec, invalid_log):
    rows = []
    extra_grid = spec.get("extra_grid", {})
    extra_flags = list(extra_grid)
    extra_combos = list(itertools.product(*extra_grid.values())) if extra_grid else [()]

    for n_splits, dt, window in itertools.product(spec["n_splits"], spec["dt"], spec["window"]):
        if zero_lag_invalid(window, dt):
            invalid_log.append((model, dt, window, n_splits, "zero-lag", f"round({window}/{dt})=0 bins"))
            continue
        too_small, test_fold_size, clean_rows = test_fold_too_small(dt, window, n_splits)
        if too_small:
            invalid_log.append((
                model, dt, window, n_splits, "test-fold-too-small",
                f"test_fold={test_fold_size}, clean_rows={clean_rows} (<{MIN_CLEAN_FRACTION:.0%} of fold)",
            ))
            continue
        for extra_combo in extra_combos:
            overrides = {
                "--dt": dt, "--window-before": window, "--bin-current": 1, "--window-after": window,
                "--n-splits": n_splits,
                **dict(zip(extra_flags, extra_combo)),
                **spec.get("fixed_overrides", {}),
            }
            rows.append((model, overrides, cost_proxy(dt, window, n_splits)))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="Print the report only; don't write the manifest.")
    args = p.parse_args()

    print(f"Inferred session duration: {DURATION_SEC:.1f}s (from an existing lagfixed run's cv_summary.json)")
    print()

    all_valid_rows = []  # (model, overrides, cost)
    invalid_log = []
    considered = 0

    for model in MODEL_SUBMIT_ORDER:
        spec = MODEL_SWEEPS[model]
        n_considered = len(spec["n_splits"]) * len(spec["dt"]) * len(spec["window"])
        considered += n_considered
        model_rows = build_rows_for_model(model, spec, invalid_log)
        all_valid_rows.extend(model_rows)
        n_invalid_this_model = sum(1 for e in invalid_log if e[0] == model)
        print(f"{model:6s}  considered={n_considered:4d} (n_splits x dt x window)  "
              f"invalid={n_invalid_this_model:4d}  valid_combos={n_considered - n_invalid_this_model:4d}  "
              f"-> {len(model_rows)} manifest rows (incl. any extra_grid multiplier)")

    print()
    print(f"{'Total combos considered':<32} {considered}")
    print(f"{'Filtered as not runnable':<32} {len(invalid_log)}")
    by_reason = {}
    for _, _, _, _, reason, _ in invalid_log:
        by_reason[reason] = by_reason.get(reason, 0) + 1
    for reason, n in sorted(by_reason.items()):
        print(f"  - {reason:<28} {n}")
    print(f"{'Rows before dedup':<32} {len(all_valid_rows)}")

    # Dedup against results/ -- reuses build_search_grid.py's own logic so
    # a row here is skipped by the EXACT SAME matching already used for the
    # first lagfixed batch (see build_search_grid.drop_already_done).
    rows_only = [(model, overrides) for model, overrides, _cost in all_valid_rows]
    kept, n_skipped = bsg.drop_already_done(rows_only)
    print(f"{'Already done/running (skipped)':<32} {n_skipped}")
    print(f"{'New rows to submit':<32} {len(kept)}")

    # Re-attach cost for ordering, then sort: model submit order first
    # (already the dict-insertion/model order since kept preserves input
    # order for a stable sort), then increasing cost within each model.
    kept_with_cost = []
    cost_by_key = {(m, tuple(sorted(o.items()))): c for m, o, c in all_valid_rows}
    for model, overrides in kept:
        cost = cost_by_key[(model, tuple(sorted(overrides.items())))]
        kept_with_cost.append((model, overrides, cost))
    model_rank = {m: i for i, m in enumerate(MODEL_SUBMIT_ORDER)}
    kept_with_cost.sort(key=lambda r: (model_rank[r[0]], r[2]))

    print()
    print("New rows per model (after dedup, in submit order):")
    from collections import Counter
    print(Counter(m for m, _, _ in kept_with_cost))

    if args.dry_run:
        print("\n--dry-run: not writing a manifest file.")
        return

    manifest_path = LAGFIXED_ROOT / f"{MANIFEST_NAME}.tsv"
    invalid_log_path = LAGFIXED_ROOT / f"{MANIFEST_NAME}_filtered_combos.json"

    manifest_lines = [bsg.format_row(model, overrides) for model, overrides, _cost in kept_with_cost]
    manifest_path.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""))
    invalid_log_path.write_text(json.dumps(
        [{"model": m, "dt": dt, "window": w, "n_splits": n, "reason": r, "detail": d}
         for m, dt, w, n, r, d in invalid_log],
        indent=2,
    ))

    print(f"\nManifest              -> {manifest_path} ({len(manifest_lines)} rows)")
    print(f"Filtered-combo record -> {invalid_log_path}")
    if manifest_lines:
        print(f"\nSubmit with:\n  nohup ./submit_search.sh {manifest_path} > submit_search_batch2.log 2>&1 &\n  disown")


if __name__ == "__main__":
    main()
