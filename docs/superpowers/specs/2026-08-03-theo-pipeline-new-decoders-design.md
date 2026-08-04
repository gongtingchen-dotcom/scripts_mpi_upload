# theo_pipeline_new: additional decoders + hyperparameter-search-ready logging

## Purpose

`theo_decoding/theo_decode_cv.py` currently runs 5 decoders (WF, FNN, WC, SVR,
XGB) with their hyperparameters hardcoded as module-level constants, and writes
its output to a single fixed `--output-dir`. Before running a large-scale
hyperparameter search, we need: (1) three more decoders (RNN, GRU, LSTM), (2)
the hyperparameters we actually want to search exposed as CLI flags, (3) every
hyperparameter — searched or not — recorded in the run's JSON output, and (4)
each run landing in its own uniquely-named, timestamped output folder so many
search runs can coexist under one root.

This is built as a new, standalone script (`theo_pipeline_new.py` +
`run_theo_pipeline_new.sh`), leaving `theo_decode_cv.py`/`run_theo_decode_cv.sh`
untouched.

## Out of scope (explicitly deferred)

- **Kalman filter decoder**: dropped. Its `predict()` needs the true first
  test-set value to seed state (a well-known quirk of this decoder family) and
  it needs raw un-lagged input rather than the lag-expanded feature matrix
  every other model here uses — different enough to not be worth it now.
- **Sweep orchestration**: the script runs one model / one hyperparameter
  combo per invocation. Driving a search (bash loop, SLURM array, etc.) is the
  user's job, external to this script.
- **Results-table aggregation script**: not built now. `run_params.json` +
  `cv_summary.json` are designed to make this easy to build later, once real
  search output exists to test it against.

## New decoders: RNN, GRU, LSTM

- `Neural_Decoding.preprocessing_funcs.LagMat` already supports `flat=False`,
  producing a `(samples, timesteps, channels)` tensor. This is computed once,
  alongside the existing flat `X_lagged`, as a second input the RNN/GRU/LSTM
  models consume instead of the flat matrix.
- **Bug found in the vendored package** (`Neural_Decoding_sklearn`, which
  belongs to collaborator theowoo on GitHub — not our repo, so we don't edit
  it directly): `nn.RNN.forward()` constructs `nn.RNN(...)` without
  `batch_first=True`, unlike `nn.GRU`/`nn.LSTM` which correctly pass it. Fed
  the same `(batch, seq, features)` tensor as GRU/LSTM, plain `RNN` would
  silently swap the batch and time axes — no crash, just a meaningless
  scrambled-sequence result.
  - Fix: a local `FixedRNN(Neural_Decoding.nn.RNN)` subclass inside
    `theo_pipeline_new.py`, overriding just the `nn.RNN(...)` construction to
    add `batch_first=True`. The vendored package is left untouched.
- All three wired into `build_models()` the same way `FNN` is today (skorch
  `NeuralNetRegressor` inside a `Pipeline`), added to the set of valid
  `--models` tokens.

## Hyperparameter CLI flags

Only these are exposed as CLI flags; everything else is a fixed constant
(matching current hardcoded values) that still gets fully recorded (see
"JSON output" below).

| Model | CLI flags | Default |
|---|---|---|
| WF | `--wf-alphas` (comma-separated floats; overrides the whole RidgeCV alpha grid) | current 25-value `logspace(-2, 4, 25)` |
| WC | `--wc-degree` | 3 |
| SVR | `--svr-c`, `--svr-max-iter` | 3.0, 50000 |
| XGB | `--xgb-max-depth`, `--xgb-n-estimators`, `--xgb-learning-rate` | 3, 200, 0.1 |
| FNN | `--fnn-num-units`, `--fnn-frac-dropout`, `--fnn-max-epochs` (exists), `--fnn-batch-size` (exists) | 400, 0.25, 10, 32 |
| RNN | `--rnn-num-units`, `--rnn-frac-dropout`, `--rnn-max-epochs` | 400, 0.25, 10 |
| GRU | `--gru-num-units`, `--gru-frac-dropout`, `--gru-max-epochs` | 400, 0.25, 10 |
| LSTM | `--lstm-num-units`, `--lstm-frac-dropout`, `--lstm-max-epochs` | 400, 0.25, 10 |

Fixed-but-recorded values not exposed as flags: SVR's kernel/gamma/epsilon/tol;
XGB's subsample/colsample_bytree/min_child_weight/reg_alpha/reg_lambda/
tree_method/objective; every NN model's `lr` (0.001) and `n_layers` (2 for
GRU/LSTM/FNN; RNN's module has no such param); RNN/GRU/LSTM's batch_size
(fixed 32, no CLI flag).

**Single source of truth per model:** each model gets a `<model>_params(args)`
function (matching the existing `xgb_params()` pattern) returning its complete
parameter dict — CLI-driven values plus every fixed constant. Both the actual
estimator construction and `run_params.json` read from the same function, so
the two can never drift apart.

## `--device` flag

One global `--device` flag (`cpu` / `cuda`, default `cpu`), passed as
`device=` to `NeuralNetRegressor` for FNN/RNN/GRU/LSTM only. WF/WC/SVR/XGB
ignore it (plain sklearn/xgboost CPU estimators). Default stays `cpu`,
preserving current behavior.

## Output folder naming

`--output-dir` is replaced with `--output-root`. The script itself (not the
bash wrapper) builds and creates the actual run folder, since only Python
knows the final resolved model + hyperparameter values at the point the
folder needs to exist:

```
<output-root>/<timestamp>_<model-or-"multi">_<key-params>/
```

e.g. `20260803_143205_XGB_d5_lr0.05/`, `20260803_143210_RNN_u200_do0.1/`. If
`--models` selects more than one model in a single invocation, the tag is
`multi` (still fully disambiguated by the JSON files inside). Per-model tag
format (only CLI-tunable params, in this fixed order):

| Model | Tag format | Example |
|---|---|---|
| WF | `WF_na{n}` (n = number of alpha values in `--wf-alphas`) | `WF_na25` |
| WC | `WC_deg{degree}` | `WC_deg3` |
| SVR | `SVR_c{C}_mi{max_iter}` | `SVR_c3.0_mi50000` |
| XGB | `XGB_d{max_depth}_n{n_estimators}_lr{learning_rate}` | `XGB_d5_n200_lr0.05` |
| FNN | `FNN_u{num_units}_do{frac_dropout}_e{max_epochs}` | `FNN_u400_do0.25_e10` |
| RNN | `RNN_u{num_units}_do{frac_dropout}_e{max_epochs}` | `RNN_u200_do0.1_e10` |
| GRU | `GRU_u{num_units}_do{frac_dropout}_e{max_epochs}` | `GRU_u400_do0.25_e10` |
| LSTM | `LSTM_u{num_units}_do{frac_dropout}_e{max_epochs}` | `LSTM_u400_do0.25_e10` |

Numeric values are formatted with their given precision, no trailing zeros
added or removed. Inside, the
existing subfolder structure (`models/`, `predictions/`, `figures/`) and files
(`run_params.json`, `cv_summary.json`, `formatted_data_like_hc.pickle`, etc.)
are unchanged.

## Bash wrapper (`run_theo_pipeline_new.sh`)

Changes from "hardcoded single invocation" to "forward whatever flags it's
given" (`"$@"` passthrough), so a hyperparameter search is just repeated calls
to this script with different flags — a bash loop or SLURM array, each call
producing its own uniquely-named folder under a shared `--output-root`. The
existing environment-activation and timestamped-log-file behavior is
preserved as-is.

## `run_params.json` / `cv_summary.json`

`run_params.json`'s `model_hyperparams` section gains one full entry per
*available* model (all 7, not just the ones actually run this call) via the
`<model>_params(args)` functions above — so any single run's JSON is
self-sufficient to know every knob's value for every model, whether searched
or not. `cv_summary.json`'s per-model R2 score structure is unchanged (see
below for the new `holdout_r2_per_output` addition), and already joins with
`run_params.json` on the shared output folder for later analysis.

## Preprocessing as a search axis (input & output)

Prompted by the reference paper's preprocessing description (z-scored input
for all methods; output demeaned for most methods, z-scored for SVR) — made
configurable rather than hardcoded per-method, so it can itself be searched.

**Current state, for reference:** input (`X_lagged`) is always z-scored via a
hardcoded `StandardScaler` step in every model's `Pipeline`, fit per-fold on
train only (this part is already correct/leak-free). Output (`y`) is not
preprocessed by default; the optional `--normalize-pcs-separately` flag
z-scores it, but computes the mean/std over the *entire* dataset **before**
the CV split — leaking test-fold statistics into training. This flag is
removed in the new script.

**New flags:**
- `--input-preprocess {zscore,demean,none}` (default `zscore`, matching
  today). Implemented by picking the `Pipeline`'s scaler step:
  `StandardScaler()` (zscore), `StandardScaler(with_std=False)` (demean), or
  `"passthrough"` (none).
- `--output-preprocess {none,demean,zscore}` (default `none`, matching
  today's default-off behavior). Computed **inside the per-fold loop**, fit
  on `y[train_idx]` only (`mu`, and `sd` if zscore), applied to both train and
  eval portions using train's stats — fixing the leakage above.
- **Predictions are always inverse-transformed back to original PC units
  before computing R2 or plotting**, regardless of `--output-preprocess`.
  This is required for the R2 numbers to stay comparable across different
  `--output-preprocess` choices in a search — without it, switching this flag
  would silently change what "R2" measures, defeating the point of searching
  over it. (Note: for `demean`, R2 is mathematically shift-invariant so this
  step is a no-op for that case either way; for `zscore` it isn't, since the
  actual fitted model differs at that scale, so the inverse-transform
  matters.)

## Holdout evaluation (final, trustworthy number)

Purpose: doing a large hyperparameter search and picking a winner by looking
at test-fold R2 across many runs biases that winner's reported R2 upward
(you're implicitly selecting hyperparameters against the test signal). Rather
than rebuilding a full train/val/test split inside every fold of every search
run, one contiguous block is reserved up front, untouched by the entire
search, and evaluated only once for the config you ultimately pick.

- `--holdout-frac` (default `0.0` = off, identical to today). When set (e.g.
  `0.15`), the *last* `holdout_frac` fraction of the session (chronological
  order) is sliced off `X_lagged`/`y` before `KFold` runs. Every search run's
  CV folds only ever see the remaining portion — no code changes needed to
  the CV loop itself.
- `--evaluate-holdout` (flag, default off). When set: after the normal CV
  loop, each selected model is refit on **all** non-holdout data (every
  sample that fed the search's `KFold`) and scored once against the holdout
  block. Preprocessing stats (if `--output-preprocess`/`--input-preprocess`
  are non-`none`) are fit on that same non-holdout data.
- Result stored under a separate, clearly-labeled key,
  `holdout_r2_per_output`, in each model's `cv_summary.json` entry — never
  mixed into `fold_r2_raw_values`/`mean_fold_r2_per_output`, so a results
  table built later can't accidentally treat it as just another CV fold.
- `run_params.json` records `holdout_frac` and `evaluate_holdout` under
  `resolved`, and the exact holdout boundary (start index/time) once
  computed.
