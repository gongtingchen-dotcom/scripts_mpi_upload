# theo_decoding

Decode behaviour (texture PCs) from **channel peaks** — drift-corrected peaks
binned per channel, fed to a regressor with K-fold CV. Grouped by
channel rather than by sorted unit. See [`../README.md`](../README.md).

Follows Theodosia Woo's method, using her `Neural_Decoding` fork and reference CV notebook. The workflow now is mostly a **hyperparameter search**: build a parameter search grid → batch-submit to SLURM → analyse. For further using after the parameters are determined, simply submit to SLURM and analyse. 

## Environment

`conda env create -f ../environments/decoding.yml` (every `.sh` here activates the
`decoding` env). Needs `Neural_Decoding` (Theo Woo's fork of Josh Glaser's package,
not on PyPI) — already wired into `decoding.yml` as a git install. See
`../environments/README.md`.

## Main pipeline

1. **`build_search_grid.py`** — writes `<manifest-name>.tsv` (one
   `theo_pipeline_new.py` arg-string per line) + `.json` record. Scans
   `<SEARCH_ROOT>/results/`, check their .json parameter settings, and drops rows already run or running.
   - **Edit** `SWEEPS` (named blocks; each block = full cross of its axes per
     model, blocks are unioned) and `ACTIVE_BLOCKS` (which ones to generate), plus
     `FIXED_ARGS` for the dataset. Drops rows that can't give a real score
     (zero-lag, test fold mostly zero-padded), dedups, orders cheap jobs first.
     `--dry-run` to preview; `--blocks` to pick blocks from the CLI.
   - **`--search-root`** overrides the results tree. Default is
     `theo_pipeline_params_search_aligned` (current pipeline: per-fold LagMat +
     zeroed-clock neural↔behaviour alignment). Older folders `…_search` (leaky CV)
     and `…_search_lagfixed` (old alignment) are not comparable to it.
   - **Output:** `<SEARCH_ROOT>/<manifest-name>.tsv` + `.json` + `_filtered_combos.json`.

2. **`submit_search.sh <manifest.tsv>`** — submits every row as a sequence of
   SLURM array jobs sized to stay under the account's MaxSubmit cap, resubmitting
   the next batch as the queue drains. `nohup ./submit_search.sh … &  disown`.
   - Safe to re-run *after* regenerating the manifest with `build_search_grid.py`
     (never regenerate while a previous batch is still outstanding).

3. **`run_theo_pipeline_raven.sh`** — the array script `submit_search.sh` calls
   (one manifest line per task). Also usable directly for a **single manual run**:
   `bash run_theo_pipeline_raven.sh <parameter arguments you want to set>`.

4. **`theo_pipeline_new.py`** 
   load + drift-correct peaks → bin per-channel MUA + PCs at
   `--dt` → K-fold CV with per-fold `LagMat` (the leakage fix) → fit/score each model in `--models`. Full parameter list and outputs in its header.
   - Neural↔behaviour alignment uses the neural clock zeroed at the recording's
     first sample; only the motion lookup uses `rec.get_times()`'s clock. Writes
     `figures/alignment_check.png` + `neural_behaviour_comparison.png` to check it
     (`--no-alignment-plots` to skip).
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
- **`channel_selection.ipynb`** — investigates dropping low-spike channels to
  fight XGB overfitting (feeds `--spike-thresh` / `--channel-smooth-window`).
- **`compare_neural_behaviour.py`** — an attempt to check the neural↔behaviour and
  motion↔neural alignment. That alignment was found to be correct and is now used
  in `theo_pipeline_new.py` (neural↔behaviour on the neural clock zeroed at the
  recording's first sample; the motion lookup on `rec.get_times()`'s clock, +
  `time_offset`), along with its alignment plots. Only its sections 1-5 run (the
  rest is disabled in a `'''` block).
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

`build_search_grid.py` (edit `SWEEPS`/`ACTIVE_BLOCKS`, `--dry-run` first) →
`submit_search.sh` → `analyze_param_search.ipynb`, all against the
`_aligned` results folder. Don't compare its R² against the older `_search`
(leaky CV, fixed 2026-08-16) or `_search_lagfixed` (old neural↔behaviour
alignment) folders — those runs compute a different thing. Check
`figures/alignment_check.png` / `neural_behaviour_comparison.png` of a first run
when pointing the pipeline at a new dataset.
