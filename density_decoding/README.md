# density_decoding

Decode behaviour directly from peak density (no spike sorting) — drift-corrected
peaks + localization from `peak_detect_localise/`+`motion/` in, a GMM/GLM/ADVI decode
of a chosen behavioural PC out. See [`../README.md`](../README.md).

## Environment

`conda env create -f ../environments/density_decoding.yml` (every `.sh` here has
`conda activate density_decoding`). This env additionally needs the
`density_decoding` editable install — the actual GMM/GLM/ADVI pipeline all the
scripts here call into (`decode_pipeline`, `behavior_decoder`, `advi`, `glm`).
Already listed correctly in the `.yml` as an editable git install pinned to a
specific commit; repo: `git@gitlab.mpcdf.mpg.de:mpibr/laur/density_decoding.git`. See
`../environments/README.md`.

## Main pipeline

1. **`trial_structure.py`** — per behavioural PC: filters trials by duration, extracts
   a fixed peak- or onset-aligned window, computes per-trial RMS deviation from the
   group mean and filters on it, exports the surviving trials as a CSV.
   - **Usage:** edit `PC_COLS` (which PCs, rms_threshold/align/color per PC) at the
     top, then run top-to-bottom or cell-by-cell.
   - **Output:** `similar_trials_<pc>_<align>_dur<MIN_DUR>s.csv` — consumed by
     `--selected_trials` below. Filename does **not** encode `rms_threshold`; running
     the same PC/align combo again with a different threshold overwrites the earlier
     CSV — rename by hand if you need to keep more than one threshold sweep.

2. **`run_density_decoding_unified.sh` + `density_decoding_unified.py`** — the real
   decode run (`--shuffle none`). GMM/GLM/ADVI pipeline: extract spike features
   (PTP/DREDge), bin spikes+behaviour, train/test split, fit, decode.
   - **Parameters:** `--behavior_var`, `--selected_trials` (from step 1),
     `--bin_size_ms`, `--chunk_size`/`--chunk_style`, `--embargo_trials`,
     `--context_before_bins`/`--context_after_bins`, `--smooth_beh_window_sec`,
     `--demean_beh`, `--animal_speed_gate` — see the script's ARGS section for the
     full list and defaults.
   - **Output:** `<BASE_DIR>/density_decoding/<param-encoded name>/` —
     `decode_params_<VAR>_<SESSION>.json` (config + R²/r), ~15 QC/diagnostic PNGs.

3. **`run_shuffle_array.sh`** — SLURM array (100 tasks) of
   `density_decoding_unified.py --shuffle shift`, one `--shuffle_index` per task, to
   build a null distribution against step 2's real run. `--shuffle_mode`/`--shuffle`
   (neural/shift) are hardcoded in the script — edit directly for a behavior-mode or
   scramble-type null.
   - **Output:** same param-encoded dir as step 2 + `_<mode>_<shuffle>_all` suffix,
     one `decode_params_*.json`+plot set per shuffle index.

4. **`shuffle_analysis.py`** — compares the real run's R²/r (step 2) against the
   null distribution (step 3).
   - **Usage:** `python shuffle_analysis.py --shuffle_dir <step 3 output dir>
     [--real_results_dir <step 2 output dir>]` (real_r2/real_r/real_results_dir
     auto-loaded from step 2's `decode_params` JSON if omitted).
   - **Output:** `shuffle_null_r2_r_<BEH>.png`, `shuffle_decode_traces_<BEH>.png`,
     `shuffle_neural_comparison_<BEH>.png`, `shuffle_analysis_summary_<BEH>.txt` — all
     written into `--shuffle_dir`.

## Supplementary

- **`trial_defined_by_texture_speed.py`** — QC check: are texture_speed-defined trial
  windows also clean when applied to other PCs, or does each PC need its own
  peak-aligned window (as `trial_structure.py` does)? Plots only, nothing consumed
  downstream.
- **`density_decoding.ipynb`** — dev notebook behind the `density_decoding*.py`
  scripts (data loading/QC exploration); `density_decoding_unified.py` is the
  maintained pipeline.

## Legacy / reference

- **`density_decoding.py`** — earliest real-run script (no `--selected_trials`/
  `--demean_beh`, uses `--n_t_bins` instead of `--bin_size_ms`). Superseded by
  `density_decoding_trial_selection.py`.
- **`density_decoding_trial_selection.py`** — adds `--selected_trials`/`--demean_beh`/
  `--bin_size_ms` over `density_decoding.py`. Superseded by
  `density_decoding_unified.py --shuffle none`. `run_density_decoding.sh` still calls
  this — use `run_density_decoding_unified.sh` instead.
- **`density_decoding_trial_selection_shuffle_all.py`** — circular-shift null
  distribution, standalone. Superseded by `density_decoding_unified.py --shuffle shift`.
- **`density_decoding_trial_selection_scramble_all.py`** — bin-scramble null
  distribution, standalone. Superseded by `density_decoding_unified.py --shuffle scramble`.
  Its own `run_scramble_array.sh` wrapper no longer exists.
- **`run_density_decoding.sh`** — calls the now-legacy `density_decoding_trial_selection.py`.

## Recommended practice

Run the main-pipeline steps in order, manually. `density_decoding_unified.py`
replaced 3 separately-drifting scripts (real / shift-null / scramble-null) with one —
always use it (via `run_density_decoding_unified.sh`/`run_shuffle_array.sh`) over any
of the legacy standalone scripts above.
