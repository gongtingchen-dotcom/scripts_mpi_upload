# preprocess

Raw SpikeGLX recording → cleaned, artifact-removed, rescaled recording ready for
spike sorting or peak detection. See [`../README.md`](../README.md) for where this
fits in the overall pipeline.

## Environment

`conda env create -f ../environments/spikeinterface.yml` (every `.sh` here has
`conda activate spikeinterface`). No modified/local repo needed — see
`../environments/README.md`.

## Main pipeline

- **`run_preprocess.sh`** — SLURM submit script. Edit `--animal_tag`/`--exp_tag`, `sbatch run_preprocess.sh`.
- **`preprocess_artifact_removal_by_sqr_deviation.py`** — the pipeline. Steps: load recording (concatenate if multi-part) → raw preview plots → highpass filter + remove bad channels + common reference → extract TTLs → detect+zero eInk artifacts (squared-deviation-from-robust-baseline method — see the "FIND EINK ARTIFACTS" section inline in the script for why this method specifically) → confirm no artifacts remain → rescale + save as float16.
  - **Usage:** `sbatch run_preprocess.sh` (GPU node, ~23h limit).
  - **Parameters:** `--animal_tag`/`--exp_tag` (required; multiple `--exp_tag`s concatenate a multi-part session); `--start_sec`/`--end_sec` (restrict to a sub-window, mainly for quick testing).
  - **Outputs**, in the session folder: `preprocessed/` (the actual deliverable — float16 recording every later stage reads), `raw_neural_data/`/`processed_neural_data/` (QC preview plots), `raw_channel_std.png`/`processed_channel_std.png` (per-channel noise), `artifact_plots_sqd/` (detection + confirmation-sweep plots), `outliers_list_sqr_overview.txt` (detected artifact times), `ttl_overview_on2.txt` (TTL onset times).
  - **Recommended practice:** check the preview plots before trusting the output; watch the log for "There may be remaining outliers" (confirmation sweep found something it didn't remove); if passing all `--exp_tag`s of a multi-part session together, they will be automatically concatenated in the order of the exp_tags you enter. 

## Supplementary

- **`process_eInk.ipynb`** — optional, run after the main pipeline. Groups the raw per-sample `outliers_list_sqr_overview.txt` into discrete on/off artifact intervals (splits wherever the gap between timestamps exceeds `gap_threshold`, drops intervals shorter than `min_duration`). Output: `eInkOn.txt`/`eInkOff.txt`. Useful if you need explicit interval boundaries rather than the raw per-sample list.

## Legacy / reference

Each file has a "NOT USED" note at the top explaining why. Kept only as reference.

- **`preprocess_ks_CW_artifacts.py`**, **`preprocess_artifact_removal_by_mean_std.ipynb`** — the predecessor artifact-detection method (mean/std threshold on the raw window). Superseded because that threshold gets contaminated by the artifact it's trying to detect ("outlier masking") — the squared-deviation method fixes this directly.
- **`preprocess_artifact_removal_by_sqr_deviation.ipynb`** — interactive notebook version of the current pipeline, from before it was generalized with `argparse`.
- **`Margot_preprocess_ks_artifact_removal.py`/`.ipynb`** — not written by us; a colleague's downstream script that feeds this pipeline's output into Kilosort4 clustering.

## Recommended practice (repeated from top-level README)

Use `preprocess_artifact_removal_by_sqr_deviation.py` via `run_preprocess.sh` for every
new session. Nothing else here should be run as part of normal use.
