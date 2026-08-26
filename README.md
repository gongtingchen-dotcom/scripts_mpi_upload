# scripts

Pipeline for processing Neuropixels/SpikeGLX recordings, from raw data to decoding.
Each stage below is a subfolder with its own SLURM submit script(s). Run stages in order.

## 1. Preprocess (`preprocess/`)

**Purpose:** clean up a raw recording before spike sorting — remove bad channels, remove
shared electrical noise, remove eInk-display artifacts, extract TTL event times, and
rescale/downcast for storage.

**Scripts:**
- `run_preprocess.sh` — SLURM submit script, edit `--animal_tag`/`--exp_tag` then `sbatch run_preprocess.sh`
- `preprocess_artifact_removal_by_sqr_deviation.py` — the actual pipeline (see its own header comment for the method details and why)

Everything else in `preprocess/` is a legacy/reference version or an unrelated one-off
utility — each has a note at the top of the file saying which and why. One useful
exception: `process_eInk.ipynb` — optional, run after the main pipeline if you need
discrete artifact start/end times rather than the raw per-sample list. It groups
`outliers_list_sqr_overview.txt` into on/off intervals (splitting wherever the gap
between consecutive artifact timestamps exceeds `gap_threshold`, dropping intervals
shorter than `min_duration`) and writes `eInkOn.txt` / `eInkOff.txt`.

**Input:** raw SpikeGLX session folder at
`/nexus/posix0/MBR-neuralsystems/ephys/<animal>/<animal>_<exp_tag>/` (an `imec0` folder,
or a `concat_rec` if multiple `--exp_tag`s were concatenated).

**Output**, written into that same session folder:
- `preprocessed/` — the cleaned, rescaled recording (float16, SpikeInterface binary format) — this is what spike sorting reads
- `raw_neural_data/`, `processed_neural_data/` — snapshot plots for visual QC, before/after
- `raw_channel_std.png`, `processed_channel_std.png` — per-channel noise, for spotting bad channels
- `artifact_plots_sqd/detection/`, `.../confirmation/` — plots from artifact detection and the post-removal sanity sweep
- `outliers_list_sqr_overview.txt` — detected artifact sample times
- `ttl_overview_on2.txt` — extracted TTL onset times

**Usage:** `sbatch run_preprocess.sh` (needs a GPU node; ~23h time limit set). Steps that
produce plots skip themselves if their output folder already has images in it, so
re-running after a partial failure won't redo finished work — except the final
`preprocessed/` save, which always overwrites.

**Key parameters to change per session** (edit inside `run_preprocess.sh`):
- `--animal_tag` / `--exp_tag` — which session to process. `--exp_tag` can take multiple space-separated tags to concatenate several recordings from one session before preprocessing.
- `--start_sec` / `--end_sec` — restrict to a sub-window of the recording (default: whole thing).

**Recommended practice:**
- Check `raw_neural_data/` and `processed_neural_data/` plots before trusting the output — this is the fastest way to catch a bad recording or a failed artifact removal.
- Watch the log for "There may be remaining outliers in the cleaned trace" — the confirmation sweep found artifacts it didn't remove; check `artifact_plots_sqd/confirmation/` manually.
- If a session was recorded in multiple parts, pass all `--exp_tag`s together (space-separated) so they get concatenated before preprocessing, rather than running each part separately.

## 2. Spike sorting (`sorters/`)

**Purpose:** cluster the preprocessed recording into putative single units.

**Scripts:**
- `run_ks4_checkpoint.sh` + `ks4_checkpoint.py` — **Kilosort4, best to use this by default.** Checkpoints itself and gets SLURM-requeued automatically, so it survives past the 24h walltime on long recordings (harmless no-op on short ones).
- `run_ks4.sh` + `KS4.py` — same Kilosort4 pipeline, same CLI, but no checkpointing — the job just dies if it doesn't finish in 24h. Only use if you have a specific reason not to use the checkpointed version.
- `run_IC.sh` + `ironclust.py` — IronClust, an alternative sorter, for comparison against Kilosort4. Paths are hardcoded at the top of `ironclust.py` — edit and rerun rather than passing CLI args. The export_to_phy function for ironclust results may have been depreciated in spikeinterface. 
- `run_ms5.sh` + `ms5.py` — MountainSort5, another alternative sorter, for comparison. `--root_folder`/`--data_folder` are CLI args.
- `matlab_test.sh` — unrelated one-off: checks MATLAB can see the GPU on this cluster. Not part of the sorting pipeline.

Everything else (`customised_KS4.py` / `run_customised_KS4.sh`) is legacy/reference — an
attempt to feed our own detected spikes (from `peak_detect_localise/`) into Kilosort4's
clustering instead of its own spike detector; sorting quality was still poor, may not be
worth pursuing further.

**Input:** the `preprocessed/` folder written by the preprocess stage.

**Output:** `<root_folder>/KS4/<run_name>/` for the Kilosort4 scripts, where `<run_name>`
encodes whichever parameters were changed from Kilosort's defaults (e.g. `nb1_thu9`);
IronClust/MountainSort5 write to the folder named in the script. Each contains a
`sorter_output/` with Phy-compatible spike times/clusters/templates, and
`kilosort_params.txt` recording exactly what was run.

**Usage:** `sbatch run_ks4_checkpoint.sh --root_folder <session dir> --data_folder <session dir>/preprocessed [kilosort params...]`.

**Key parameters to change per session:**
- `--root_folder` / `--data_folder` — session dir and its `preprocessed/` folder.
- `--nblocks` — number of drift-correction blocks (0 disables drift correction).
- `--Th_universal` / `--Th_learned` — detection thresholds (universal vs. learned templates).
- `--nearest_chans` / `--whitening_range` — spatial extent used for templates/whitening.
- Any other kilosort4 param exposed by `si.get_default_sorter_params('kilosort4')` can be passed the same way — the run folder name auto-encodes whichever ones you changed.

**Recommended practice:**
- Default to `ks4_checkpoint.py` unless you have a reason not to.
- Treat IronClust/MountainSort5 as comparison runs, not primary output.

## 3. Peak detection & localization (`peak_detect_localise/`)

**Purpose:** an alternative to sorter-based spike finding — detect spikes by thresholding
(not template matching), localize each one in 3D space on the probe, then estimate and
correct for probe drift/motion using those locations.

**Scripts:**
- `run_peak_localisation.sh` + `peak_localisation.py` — **the active pipeline, use this.** Detects peaks (via `mua_pipeline.py`), localizes them with SpikeInterface's `monopolar_triangulation`, estimates + corrects motion, then re-detects/re-localizes on the corrected data as a sanity check.
- `mua_pipeline.py` — the spike-detection engine used above (thresholding + matched filtering + spatial-support artifact filtering). Imported by the pipeline, not run directly.
- `MUA_threhsold2.ipynb` — where `mua_pipeline.py`'s functions were originally extracted from, but it's a frozen, independently-duplicated copy (doesn't import from `mua_pipeline.py`) that has since fallen behind — no matched-filtering detector, no rolling-noise support, no `run_mua_pipeline()` orchestrator. Not where active development happens; use `mua_pipeline.py` for what's actually in the pipeline.
- `run_boussard_localisation.sh` + `boussard_localization.py` — comparison localization method: the full IBL denoiser+localizer pipeline (GPU-only, slow). Same method family as monopolar triangulation (SpikeInterface credits it to the same authors) but with an extra neural-net denoising step. This was the method used for `density_decoding`. `compare_localisation.py` is the head-to-head comparison that showed it wasn't meaningfully better than monopolar triangulation for the extra cost, hence not the default. `boussard_localisation.ipynb` is its dev notebook.
- `fix_peak_sample_indices.py` — standalone manual repair utility for old peak files whose sample indices weren't shifted back to the original (unsliced) recording timeline. The current pipeline handles this automatically, so shouldn't be needed for anything it produces.
- `spike_density_count.py` — QC: spike density / channel-noise-RMS / sorted-unit-firing-rate maps laid out by real probe geometry (handles multiple shanks and deleted channels correctly). Run manually after the pipeline.
- `specifc_spike_waveforms.ipynb` — inspect one unit's waveform at a chosen timepoint, phy-style dark background.
- `slope_artifact_filter_analysis.ipynb` — **step 2 of the actual workflow, not just exploratory.** Manually-labeled analysis that found half-amplitude width separates real spikes from artifacts better than slope; applies that threshold to filter `peaks_combined2_sorted.npy` before motion estimation. Interactive by design (labeling/threshold choice is dataset-specific), not something worth hardening into `mua_pipeline.py`. Its filtered output — not the raw peaks file — is what `motion/run_dredge_chunked.py`'s `PEAKS_FILENAME` should point to.
- `rolling_snr.py` / `rolling_snr_analysis.ipynb` — abandoned: tried an SNR-adaptive detection threshold instead of one fixed value; not meaningfully better, still noisy unless the threshold was raised so high it missed too many peaks.

**Input:** the `preprocessed/` folder from the preprocess stage.

**Output**, in `<base_path>/<folder_name>/`: `peaks_combined2_sorted.npy` (and related `peaks*.npy`) for detected spikes, `localized_peaks_mt_raw.npy` for their locations, and verification plots.

**Usage:** `sbatch run_peak_localisation.sh` after editing the args below. In practice run in **detection + localization mode only** — leave the motion-related config flags in `peak_localisation.py` at their recommended settings (see the script's own "RECOMMENDED WORKFLOW / PREFERRED SETTINGS" comment block) and do motion estimation/correction separately with `motion/` (section 4) instead of this script's built-in (legacy, not well tested) motion correction. Reasons: each dataset needs different parameters at each step, a long recording can exceed the 24h SLURM limit if everything runs as one job, and running it all as one script hides which step actually went wrong; `motion/` also supports chunked estimation for long recordings, which this script doesn't.

**Key parameters to change per session:**
- `--base_path` — the `preprocessed/` folder for this session.
- `--mua_detect_threshold` — spike-detection threshold (units of noise).
- `--support_radius` / `--support_fraction` / `--support_n_required` — how strict the spatial-support artifact check is.
- `--time_start_s` / `--time_end_s` — restrict the whole pipeline to a sub-window.
- `--folder_name` — output subfolder name, so a rerun with different settings doesn't collide with an existing one.
- Between this stage and motion estimation, run `slope_artifact_filter_analysis.ipynb` on the detected peaks to filter out artifacts too narrow to be real spikes (see its note in this folder for the threshold it settled on).

## 4. Motion estimation & correction (`motion/`)

**Purpose:** estimate probe drift/motion from already-detected, already-localized (and
artifact-filtered) peaks, then correct for it. Kept as a separate stage from peak
detection/localization (see section 3) specifically so long recordings can be
chunked and each step checked individually.

**Scripts (in the order you run them):**
1. `run_dredge_chunks.sh` + `run_dredge_chunked.py` — **step 1, use this.** SLURM array job, one task per time chunk (edit chunk duration / `BASE_PATH` / `FOLDER_NAME` in `run_dredge_chunked.py` — it mirrors `peak_localisation.py`'s settings — and the `--array` range in the `.sh` to match).
2. `run_dredge_stitch.sh` — **step 2, run separately after step 1 fully finishes.** Combines the chunks (via `run_dredge_chunked.py`'s own `--stitch` mode) into one motion estimate, saved as `motion_dredge_stitched_<run_name>/` (a sibling of `chunks_<run_name>/`). Requires all chunks from step 1 to already exist — fails immediately and cleanly (`FileNotFoundError`) if not; check (e.g. via `sacct`) that every array task from step 1 succeeded before submitting this. **Never** call `--stitch` from inside step 1's array job itself (array tasks run in parallel with no ordering guarantee — the first to finish would stitch an incomplete set of chunks with no error) — these are two deliberately separate, manually-run steps, not one job.
3. `run_correct_motion_chunks.sh` + `correct_motion.py` — **step 3, QC only — read the caveat.** Produces plots showing what the correction (direct subtraction from peak depths / channel-shift, no recording interpolation) looks like — it does NOT save a corrected-peaks data file (checked: no `np.save`/`.save()` anywhere in the script). Whatever actually consumes the correction (e.g. `density_decoding.ipynb`) reads the motion estimate directly and reimplements the same nearest-bin-lookup math itself — see `correct_motion.py`'s header.
4. `combine_motions.py` — optional, only if a single estimate over the whole depth range is unreliable at the shallow/deep extremes: combines several per-depth-range estimates into one.

Everything else here is either a diagnostic/comparison tool or legacy — each has a note
at the top of the file saying which and why: `motion_comparison.py` (compares DREDge
against other SpikeInterface presets — its sibling `motion_comparison_tomerge.py` is
superseded, still points at an old `/gpfs/laur/...` path that no longer exists),
`run_dredge_chunked_boussard.py` (only needed for the Boussard-localization comparison
track, see section 3), `correct_motion_test.py` (dev scratch script), and the notebooks
(`correct_motion.ipynb`, `drift_diagnostics.ipynb` — keep, has a unique KS4-vs-DREDge
comparison nothing else does; `drift_plot.ipynb` — the oldest file in the folder,
predates the current approach, safe to consider legacy). `stitch_motion.py` does the
same job as `run_dredge_chunked.py`'s own built-in `--stitch` (confirmed: both align
overlapping chunks by matching mean displacement in the overlap, `stitch_motion.py`
just hardcoded to exactly 2 chunks with a different overlap-merge policy) — kept
temporarily, not yet marked fully legacy, pending an A/B comparison of the two merge
policies on real data.

**Input:** `peaks_combined2_sorted.npy` / `localized_peaks_mt_raw.npy` from the peak
detection/localization stage (ideally after `slope_artifact_filter_analysis.ipynb`).

**Output:** per-chunk motion estimates under `motion_estimates/chunks_<run_name>/motion_dredge_*min_*min/`, a stitched full-recording estimate at `motion_estimates/motion_dredge_stitched_<run_name>/` (a **sibling** of `chunks_<run_name>/`, not inside it), and QC plots from `correct_motion.py` — nothing downstream reads `correct_motion.py`'s output directly; they read the stitched motion estimate and apply the correction themselves.

**Usage:** `./submit_dredge_pipeline.sh` (submits steps 1+2 together, correctly ordered) → `./run_correct_motion_chunks.sh <run_name>`.

**Key parameters to change per session:**
- `BASE_PATH` / `FOLDER_NAME` / `PEAKS_FILENAME` in `run_dredge_chunked.py` — must match the peak-localization stage's output for this session.
- `CHUNK_DURATION_S` / `OVERLAP_S` in `run_dredge_chunked.py` — chunk size and overlap; the `--array` range in `run_dredge_chunks.sh` must cover the number of chunks this produces.
- `AMP_PERCENTILE` in `run_dredge_chunked.py` — restricting motion estimation to higher-amplitude peaks can substantially improve results on noisy datasets.
- `MOTION_BASE` in `run_correct_motion_chunks.sh` — session's peaks folder.
