# scripts

Pipeline for processing Neuropixels/SpikeGLX recordings, from raw data to decoding.
Each folder below has its own `README.md` with full detail (scripts, inputs/outputs,
parameters, recommended practice). This file is the map: how the folders connect, and
which script in each one to actually reach for.

## Setup

Each pipeline stage runs in its own conda env (see table below); full details,
exact `conda env create` commands, and everything that needs installing on top of a
plain env create are in [`environments/README.md`](environments/README.md) — read
that first before running anything.

| stage | env | needs a repo beyond `conda env create`? |
|---|---|---|
| `preprocess/` | `spikeinterface` | no |
| `sorters/` | `phy2` | no extra repo, but see the hand-edit note below |
| `peak_detect_localise/` | `spikeinterface` | only for legacy `boussard_localization.py` (see below) |
| `motion/` | `spikeinterface` | no |
| `tomato/` | `spikeinterface` + `slay` + `phy2` (per-script, see `tomato/README.md`) | `slay` env needs the `slay` repo |
| `density_decoding/` | `density_decoding` | needs the `density_decoding` repo |
| `theo_decoding/` | `decoding` | no |

Three repos are hand-modified/local forks, not on PyPI — clone + `pip install -e .`
into the matching env (all already wired into the `.yml` files as pinned editable
installs, so a plain `conda env create` gets them automatically if you have repo
access):
- `slay` env ← `git@gitlab.mpcdf.mpg.de:mpibr/laur/slay.git`
- `density_decoding` env ← `git@gitlab.mpcdf.mpg.de:mpibr/laur/density_decoding.git`
- `spikeinterface` env, legacy Boussard localization path only ←
  `git@gitlab.mpcdf.mpg.de:mpibr/laur/spikes_localization_registration_gongt.git`
  (our fork of `int-brain-lab/spikes_localization_registration`, with local
  numpy-compatibility fixes — see `environments/README.md`)

One more thing a plain env create can't reproduce: the `phy2` env's installed
`kilosort` package has its own auto-merge step hand-disabled (a direct edit to the
installed package's source, not a repo). See `environments/README.md` and
`sorters/ks4_checkpoint.py`'s "ENVIRONMENT NOTE" for exactly what to redo.

## Pipeline

```
preprocess/
  │
  ├──▶ sorters/                              (traditional spike sorting: KS4 / IronClust / MountainSort5)
  │
  └──▶ peak_detect_localise/  ──▶  motion/    (detect + localize peaks, then motion-correct them — no full spike sorting)
                                      │                                      
                                      ├──▶ tomato/              (ToMATo: density-peak clustering on drift-corrected peaks)
                                      ├──▶ density_decoding/    (decode behaviour directly from peak density, no sorting)
                                      └──▶ theo_decoding/       (decode behaviour from peaks grouped by channel/depth)
```

`sorters/` and `peak_detect_localise/` are two independent ways of turning a preprocessed
recording into something you can analyze — one via full spike sorting, one by skipping
sorting and working with detected+localized threshold-crossings (i.e. peaks) directly. `tomato/`,
`density_decoding/`, and `theo_decoding/` all consume the second path's output
(`peaks_combined2_sorted.npy` + motion correction), not sorter output. 

## 1. Preprocess (`preprocess/`) — [full detail](preprocess/README.md)

Raw recording → cleaned, bad channel-removed, artifact-removed, rescaled recording.
- **`run_preprocess.sh` + `preprocess_artifact_removal_by_sqr_deviation.py`** — the pipeline, use this.
- `process_eInk.ipynb` — optional: turns the raw artifact-time list into discrete on/off intervals.
- Everything else: legacy methods this pipeline replaced. 

## 2. Spike sorting (`sorters/`) — [full detail](sorters/README.md)

Preprocessed recording → sorted single units (Phy-compatible).
- **`run_ks4_checkpoint.sh` + `ks4_checkpoint.py`** — Kilosort4, recommended choice (survives past the 24h SLURM limit).
- `run_ks4.sh` + `KS4.py` — same, no checkpointing, so will be automatically killed if runs for more than 24h on SLURM
- `run_IC.sh` + `ironclust.py`, `run_ms5.sh` + `ms5.py` — IronClust / MountainSort5, legacy scripts, run for comparison against Kilosort4.
- `matlab_test.sh` — unrelated GPU sanity check to test whether matlab is working for IC, not part of sorting.
- `customised_KS4.py` — legacy: fed our own detected spikes into KS4's clustering, results were poor.

## 3. Peak detection & localization (`peak_detect_localise/`) — [full detail](peak_detect_localise/README.md)

Preprocessed recording → detected + localized spikes (no sorting).
- **`run_peak_localisation.sh` + `peak_localisation.py`** — detects and localizes peaks. Contain the full pipeline to of detection --> localization --> motion, but recommended to detection + localization only (see folder README for why).
- **`slope_artifact_filter_analysis.ipynb`** — step 2: filters out artifacts too narrow to be real spikes. Recommended to check the waveforms after peak_localisation first, run this if there are visible narrow artifacts. 
- `boussard_localization.py` + `compare_localisation.py` — legacy, alternative localization method, kept for comparison (not meaningfully better, much slower).
- `spike_density_count.py`, `specifc_spike_waveforms.ipynb` — QC/inspection tools.
- `fix_peak_sample_indices.py` — one-off repair utility, rarely needed now.
- `MUA_threhsold2.ipynb`, `rolling_snr.py` — legacy/abandoned, see folder README.

## 4. Motion estimation & correction (`motion/`) — [full detail](motion/README.md)

Detected peaks → drift-corrected peak positions.
- **`run_dredge_chunks.sh` → `run_dredge_stitch.sh` → `run_correct_motion_chunks.sh`** — the pipeline: chunked estimation, then stitch (separately, after step 1 fully finishes), then QC plots.
- `combine_motions.py` — optional: merge per-depth-range estimates if one estimate is unreliable at the shallow/deep extremes.
- Everything else: comparison tools or legacy, see folder README.

## 5. ToMATo clustering (`tomato/`) — [full detail](tomato/README.md)

Drift-corrected peaks → sorted units, via ToMATo (persistence-based density-peak
clustering) instead of traditional spike sorting.
- **`run_try_tomato.sh` + `try_tomato.py`** — the clustering step, use this.
- **`run_try_merge_selfbuild.sh` + `try_merge_selfdbuild_cleaned.py`** — auto-merging
  pass over the phy export; `try_merge_step_2.py` is an optional follow-up pass.
- `merge_tomato_results.py` — only if run in separate sample-range chunks.
- `build_export_to_phy.py` — turns the sorting into a phy-openable folder.
- Everything else: supplementary tools or legacy, see folder README.

## 6. Density decoding (`density_decoding/`) — [full detail](density_decoding/README.md)

Decode behaviour directly from peak density (no spike sorting), via a GMM/GLM/ADVI
pipeline.
- **`trial_structure.py`** — first step: filters/aligns trials per behavioural PC,
  exports the CSV the decode step selects trials from.
- **`run_density_decoding_unified.sh` + `density_decoding_unified.py`** — the real
  decode run; `run_shuffle_array.sh` builds a matching null distribution
  (`--shuffle shift/scramble`); `shuffle_analysis.py` compares the two.
- Everything else: supplementary tools or legacy, see folder README.

## 7. Theo_decoding (not yet documented in this cleanup pass)

- `theo_decoding/` — decodes behaviour from peaks grouped by channel/depth; main entry `theo_pipeline_new.py`.
