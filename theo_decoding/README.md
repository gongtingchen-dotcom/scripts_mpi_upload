# theo_decoding

Decode behaviour (texture PCs) from **channel-wise MUA** — drift-corrected peaks
binned per channel, lagged, fed to a regressor with K-fold CV. Grouped by
channel/depth rather than by sorted unit. See [`../README.md`](../README.md).

Named after Theo Woo, whose `Neural_Decoding` fork and reference CV notebook this
follows. The workflow is a **hyperparameter search**: build a grid → batch-submit
to SLURM → analyse.

## Environment

`conda env create -f ../environments/decoding.yml` (every `.sh` here activates the
`decoding` env). Needs `Neural_Decoding` (Theo Woo's fork of Josh Glaser's package,
not on PyPI) — already wired into `decoding.yml` as a git install. See
`../environments/README.md`.

## Main pipeline

1. **`build_search_grid.py`** — writes `<manifest-name>.tsv` (one
   `theo_pipeline_new.py` arg-string per line) + `.json` record. Scans
   `<SEARCH_ROOT>/results/` and drops rows already run or running.
   - **Edit** `GRID` / `FIXED_ARGS` / the sweep constants (single source of truth).
   - **`--search-root`** overrides the results tree. The hardcoded default is the
     **original, pre-leakage-fix** folder — for correct numbers pass
     `--search-root .../theo_pipeline_params_search_lagfixed`.
   - **Output:** `<SEARCH_ROOT>/<manifest-name>.tsv` + `.json`.

2. **`submit_search.sh <manifest.tsv>`** — submits every row as a sequence of
   SLURM array jobs sized to stay under the account's MaxSubmit cap, resubmitting
   the next batch as the queue drains. `nohup ./submit_search.sh … &  disown`.
   - Safe to re-run *after* regenerating the manifest with `build_search_grid.py`
     (never regenerate while a previous batch is still outstanding).

3. **`run_theo_pipeline_raven.sh`** — the array script `submit_search.sh` calls
   (one manifest line per task). Also usable directly for a **single manual run**:
   `bash run_theo_pipeline_raven.sh <full theo_pipeline_new.py args>`.

4. **`theo_pipeline_new.py`** — the decoder itself. One invocation = one
   hyperparameter point: load + drift-correct peaks → bin per-channel MUA + PCs at
   `--dt` → K-fold CV with **per-fold** `LagMat` (the leakage fix) → fit/score each
   model in `--models`. Full parameter list and outputs in its header.
   - **Models:** `WF WC SVR XGB FNN RNN GRU LSTM`.
   - **Output** (per run, under `<output-root>/<timestamp>_<tag>/`):
     `run_params.json`, `cv_summary.json`, `models/`, `predictions/`, `figures/`.

5. **`analyze_param_search.ipynb`** — scans every finished run under
   `<SEARCH_ROOT>/results/` into one table → coverage reports, per-model
   heatmaps/marginals, cross-model leaderboard, best/worst decoded-trace plots.
   All auto-detected from what was actually run.

## Supplementary

- **`monitor_runtime_budget.py`** — external watchdog: polls `squeue` + each
  task's SLURM log, `scancel`s any `theo_search` job whose projected total
  runtime exceeds the budget. Covers jobs that started before
  `theo_pipeline_new.py`'s own in-process `--runtime-abort-budget-hours` tripwire
  existed. `nohup python monitor_runtime_budget.py &  disown`.
- **`build_search_grid_lagfixed_batch2.py`** — a second, deeper sweep run on top
  of `build_search_grid.py`'s first pass. Difference from `build_search_grid.py`:
  it varies **only** dt × window × n_splits (not the per-model hyperparameter
  grids), with a **per-model** range for each of those three axes (GRU/LSTM down
  to dt=0.1 and up to window=128; RNN capped at window=8; etc.) instead of one
  shared list; and it drops combinations that can't produce a real score
  (window/dt < 1 lag bin; test fold mostly inside the LagMat zero-padded zone),
  which `build_search_grid.py` does not filter. Imports `build_search_grid.py` as
  a library so dedup/formatting stay identical. Both manifests coexist in the
  lagfixed folder and feed the same `results/`.
- **`channel_selection.ipynb`** — investigates dropping low-spike channels to
  fight XGB overfitting (feeds `--spike-thresh` / `--channel-smooth-window`).
- **`compare_neural_behaviour.py`** — same load/preprocess + PCA on the binned
  firing rates + a time-aligned neural-vs-behaviour comparison figure. **Note:**
  its neural↔behaviour alignment is NOT identical to the main pipeline's — it
  aligns on a neural-zeroed clock (peak `sample_index/fs`, no `time_offset`),
  whereas `theo_pipeline_new.py` aligns on the absolute `rec.get_times()` clock
  (`+ time_offset`, ~4.3s here). The drift-correction step itself is the same in
  both. See the script header.
- **`theo_data_prep_demo.py`** — standalone readable extract of just the
  data-prep half of `theo_pipeline_new.py` (raw → lagged X/y). Onboarding aid.
- **`test_r2_method.py`** — confirms the main pipeline's held-out R² arithmetic
  is correct: reloads each fold's fitted model (no retrain), recomputes
  `sklearn.r2_score(y_test, pred, multioutput="raw_values")`, checks it matches
  `cv_summary.json`'s `models.<model>.fold_r2_raw_values[fold][pc]`.

## Legacy / reference

- **`theo_decode_cv.py`** + **`run_theo_decode_cv.sh`** — predecessor to
  `theo_pipeline_new.py` (which is a strict superset: per-model hyperparam flags,
  the per-fold-lag leakage fix, window sizes in seconds, batch mode).
- **`run_theo_pipeline_new.sh`** — older "run locally" wrapper; `/gpfs/laur`
  paths (not on Raven) and passes `--background/--bin-before/--bin-after` that
  the current script no longer accepts. Use `run_theo_pipeline_raven.sh`.
- **`theo_pipeline_analysis.ipynb`** — analysis notebook for `theo_decode_cv.py`
  output. Superseded by `analyze_param_search.ipynb`.
- **`theo_pipeline.ipynb`** — the original exploratory notebook, predates both
  `.py` pipelines.

## Recommended practice

`build_search_grid.py --search-root <…_lagfixed>` → `submit_search.sh` →
`analyze_param_search.ipynb`. Always point new work at the `_lagfixed` results
folder — the original folder's R² numbers are inflated by the CV leakage bug
fixed 2026-08-16 and must not be compared against post-fix runs. Submit cheap
jobs (large `--dt`, small `--window`, few `--n-splits`) first so expensive or
doomed combos surface early and can be cancelled.
