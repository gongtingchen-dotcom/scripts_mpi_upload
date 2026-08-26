# motion

Detected+localized peaks → drift-corrected peak positions. Kept as a separate stage
from `peak_detect_localise/` specifically so long recordings can be chunked and each
step checked individually — see [`../README.md`](../README.md).

## Environment

`conda env create -f ../environments/spikeinterface.yml` (every `.sh` here has
`conda activate spikeinterface`). No modified/local repo needed — see
`../environments/README.md`.

## Main pipeline

Mainly run_dredge_chunks.sh + run_dredge_chunked.py, other scripts may also be needed depending on settings. 

1. **`run_dredge_chunks.sh` + `run_dredge_chunked.py`** — SLURM array job, one task per time chunk of the recording (DREDge motion estimation, chunked to avoid OOM on long recordings).
   - **Parameters:** `--run_name` (tags this batch), `--amp_pct` (restrict to high-amplitude peaks — improves noisy datasets), `--depth_min`/`--depth_max` (restrict to a depth sub-range — see `combine_motions.py` below); `CHUNK_DURATION_S`/`OVERLAP_S`/`DREDGE_PARAMS` (edit in the `.py` — see its header for what each `DREDGE_PARAMS` key does).
      - `CHUNK_DURATION_S`/`OVERLAP_S` allows this script to run by overlapping chunks, useful for long recordings; if running by chunks, need to run `run_dredge_stitch.sh` next. 
      - `--depth_min`/`--depth_max` allows this script to use only a certain depth of the probe, hence able to do several runs on different depths and estimate motion separately for them. if running by depth sections, need to run `combine_motions.py` next. 
   - **Output:** `motion_estimates/chunks_<run_name>/motion_dredge_<start>min_<end>min/` per chunk.

2.. **`run_correct_motion_chunks.sh` + `correct_motion.py`** — QC only. Produces plots of what the correction looks like; saves no data file (checked: no `np.save`/`.save()` in the script). Works on any single motion estimate — a chunk or the stitched result — not just "several chunks".
   - **Usage:** `python correct_motion.py --motion_path <one estimate dir>` or `--motion_root <dir containing several>` for batch QC.
   - **Outputs:** ~20 plot types (drift/motion-overlay/corrected scatter, at both the depth and channel level) — see the script's `PLOT SELECTION` block for the full one-line-each list. Plot time range always spans exactly the motion estimate you point it at (whole recording if given the stitched result, one chunk's window if given a chunk).

## Supplementary

- **`combine_motions.py`** — if a single estimate over the whole depth range is unreliable at the shallow/deep extremes (too few peaks there): combines several per-depth-range estimates (from step 1 above, using `--depth_min`/`--depth_max`) into one, by direct concatenation — every bin is a real fitted value from whichever source covers it; only the true gaps between disjoint ranges get SpikeInterface's own interpolation. Output: `chunks_combined/`.

- **`run_dredge_stitch.sh`** — after confirming (e.g. `sacct`) every chunk task succeeded: combines them via `run_dredge_chunked.py --stitch` (aligns overlapping chunks by matching mean displacement in their overlap).
   - **Output:** `motion_estimates/motion_dredge_stitched_<run_name>/` — a **sibling** of `chunks_<run_name>/`, not inside it. This is what downstream analyses (e.g. `density_decoding`) actually read.

- **`motion_comparison.py`** — sanity-checks DREDge against other SpikeInterface motion-estimation presets on the same data. Not part of the default pipeline (which calls DREDge directly).
- **`run_dredge_chunked_boussard.py`** — same chunked-DREDge idea, reading Boussard localization output instead of the standard peaks files. Only needed on the Boussard comparison track (see `peak_detect_localise/README.md`).
- `drift_diagnostics.ipynb` — standalone QC notebook; uniquely compares this pipeline's DREDge correction against Kilosort4's own internal drift correction (nothing else here does that) — has hardcoded Windows paths. 
- `correct_motion.ipynb` — interactive dev notebook `correct_motion.py` was built from.

## Legacy / reference

- **`stitch_motion.py`** — LEGACY, does a similar job as `run_dredge_chunked.py`'s own `--stitch`, just hardcoded to exactly 2 chunks with a different overlap-merge policy (uses chunk0's value in the overlap instead of averaging). Kept temporarily pending an A/B comparison of the two merge policies — not yet fully deprecated. the stitching method that `run_dredge_stitch.sh` is recommended instead. 
- **`motion_comparison_tomerge.py`** — LEGACY, superseded by `motion_comparison.py` (identical, just with stale pre-migration `/gpfs/laur/...` paths). 
- **`correct_motion_test.py`** — LEGACY, dev scratch copy of `correct_motion.py`, hardcoded to one dataset/chunk. 
- **`drift_plot.ipynb`** — LEGACY, early, abandoned exploration using SpikeInterface's own built-in `correct_motion` preprocessing directly. Predates the current approach.

## Recommended practice

Run the main-pipeline and any relevant following-up steps in order, manually, have to wait for the previous step to finish.
Use `correct_motion.py` for QC.
