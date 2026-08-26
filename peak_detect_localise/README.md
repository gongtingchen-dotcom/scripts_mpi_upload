# peak_detect_localise

Preprocessed recording → detected + localized spikes, no full sorting. The other of two
independent ways to turn a preprocessed recording into analyzable output — see
[`../README.md`](../README.md). Feeds `tomato/`, `density_decoding/`, `theo_decoding/`,
and (after `motion/`) any analysis that needs drift-corrected spike positions.

## Environment

`conda env create -f ../environments/spikeinterface.yml` (every `.sh` here has
`conda activate spikeinterface`). The legacy `boussard_localization.py` additionally
needs the `subtraction_pipeline` editable install (int-brain-lab fork, has
uncommitted local edits not captured anywhere) — see `../environments/README.md`.
Nothing else here needs it.

## Main pipeline

Run in this order — as two separate, deliberately manual steps, not one script:

1. **`run_peak_localisation.sh` + `peak_localisation.py`** — detects peaks (via `mua_pipeline.py`: thresholding + matched filtering + spatial-support artifact rejection) and localizes them (`monopolar_triangulation`).
   - **Why not the full end-to-end version** (this script *can* also estimate/correct motion and re-detect on corrected data): each dataset needs different parameters at each step, a long recording can exceed the 24h SLURM limit if run as one job, and bundling hides which step went wrong. It is thus recommended to run them separately and use `motion/` for motion estimation.
   - **Usage:** `sbatch run_peak_localisation.sh` after editing the CLI args inside it.
   - **Parameters:** `--base_path` (session's `preprocessed/`), `--mua_detect_threshold`, `--support_radius`/`--support_fraction`/`--support_n_required` (spatial-support strictness), `--time_start_s`/`--time_end_s`, `--folder_name`.
   - **Outputs**, in `<base_path>/<folder_name>/`: `peaks0/1/2.npy`, `peaks_matched.npy`, `peaks_combined1/2.npy` (see `peak_localisation.py`'s own header for the full detection-branch derivation), `peaks_combined2_sorted.npy` (**the actual output — everything downstream uses this**), `localized_peaks_mt_raw.npy` (locations), `plots/` (QC).
2. **`slope_artifact_filter_analysis.ipynb`** — filters `peaks_combined2_sorted.npy` further using half-amplitude-width (found to separate real spikes from artifacts better than slope, via manual labeling).
   - **Why a notebook, not code in `mua_pipeline.py`:** the threshold choice could be dataset-specific and may need eyeballing. Recommended to only run this if the the waveforms of peaks_combined2_sorted suggest that there are narrow artifacts. As this has only been tested on 7248, the thresholds may need to be changed for other data. 
   - **Output:** a filtered `peaks_*.npy` next to the original, named with the threshold used. Point `motion/run_dredge_chunked.py`'s `PEAKS_FILENAME` at this file, not the raw one.

## Supplementary

- **`mua_pipeline.py`** — the detection engine `peak_localisation.py` imports. Not run directly; see its `run_mua_pipeline()` docstring for the authoritative parameter/return-value reference.
- **`boussard_localization.py` + `compare_localisation.py`** (+ `run_boussard_localisation.sh`, `boussard_localisation.ipynb`) — alternative localization method (the full IBL denoiser+localizer pipeline; same lineage as `monopolar_triangulation`, extra GPU-only denoising step). `compare_localisation.py` is the head-to-head that showed it wasn't meaningfully better for the extra cost — kept for comparison, not the default.
- **`spike_density_count.py`** — QC: spike-density / channel-noise-RMS / sorted-unit-rate maps laid out by real probe geometry. Run manually after the pipeline.
- **`specifc_spike_waveforms.ipynb`** — inspect one unit's waveform at a chosen timepoint (phy-style dark background).
- **`fix_peak_sample_indices.py`** — manual repair utility for old peak files with un-shifted sample indices. The current pipeline handles this automatically, so shouldn't be needed for anything it produces.

## Legacy / reference

- **`MUA_threhsold2.ipynb`** — where `mua_pipeline.py`'s functions were originally extracted from. Now a frozen, independently-diverged copy (doesn't import from `mua_pipeline.py`) missing matched-filtering, rolling-noise support, and the `run_mua_pipeline()` orchestrator. Not where active development happens.
- **`rolling_snr.py` + `rolling_snr_analysis.ipynb`** — abandoned attempt at an SNR-adaptive detection threshold instead of one fixed value. Not meaningfully better; still noisy unless the threshold was raised so high it missed too many peaks.

## Recommended practice

Run `peak_localisation.py` in detection+localization mode only, then
`slope_artifact_filter_analysis.ipynb`, then hand off to `motion/`. Don't use
`peak_localisation.py`'s own motion correction/re-detection sections.
