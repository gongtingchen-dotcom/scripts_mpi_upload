# tomato

Drift-corrected peaks → sorted units, via ToMATo (Topological Mode Analysis TOol,
`gudhi.clustering.tomato` — persistence-based density-peak clustering), not spike
sorting in the traditional template-matching sense. See [`../README.md`](../README.md).

## Environment

Three different envs are used in this folder, one per stage — set up all three with
`conda env create -f ../environments/<name>.yml`:

| env | scripts |
|---|---|
| `spikeinterface` | `try_tomato.py`, `merge_tomato_results.py`, `build_export_to_phy.py`, `recompute_ks4_features.py`, and the legacy `cluster_from_spikes.py` |
| `slay` | `try_merge_selfdbuild_cleaned.py`, `try_merge_step_2.py`, and the legacy `try_merge_selfdbuild*.py` variants |
| `phy2` | the legacy `compute_in_si.py` only |

`slay` additionally needs the `slay` editable install (old-API fork — the PyPI
`slay` package will NOT work here) — see `../environments/README.md`. The other two
envs need no extra repo for anything in this folder.

## Main pipeline

1. **`run_try_tomato.sh` + `try_tomato.py`** — THE active clustering step. Re-applies
   the same drift correction as `correct_motion.py`, then windows the
   recording and runs Tomato per window on extracted features, currently auto-picking cluster
   count from the persistence diagram (manual override commented out in the script).
   Depending on the data, it is recommended to always use manual override, or even disabling 
   the merging completely. 
   - **Parameters:** `--data_path`, `--motion_path`, `--localized_peaks_path`,
     `--skip_drift_correction`, `--sample_start`/`--sample_end` (split a long recording
     into chunks, needed for longer recordings — see next step).
   - **Output:** `tomato_windowed_s<start>_e<end|END>/` — `persistence_win*.png`,
     `feats_*.npy`/`tomato_*.npz`, `final_labels_*.npy`, a NumpySorting folder,
     `run_summary.json`/`params.json`.

2. **`merge_tomato_results.py`** — needed if step 1 was run in separate
   sample-range chunks. Recovers each chunk's peaks in global sample-index terms,
   offsets cluster IDs per chunk, concatenates into one NumpySorting, warns on
   overlapping chunk ranges.
   - **Usage:** `python merge_tomato_results.py <chunk_folder> <chunk_folder> ... [--output DIR]`
   - **Output:** merged NumpySorting folder (`spike_times.npy`/`spike_clusters.npy` +
     `sorting.save()` folder).

3. **`build_export_to_phy.py`** — turns the NumpySorting (from step 1 or 2) into a
   phy-openable folder with consistent sparsity across all views.
   - **Usage:** `python build_export_to_phy.py [--sorting_path P] [--rec_path P] [--export_dir P] ...`
   - **Output:** `export_to_phy/` — `templates.npy`, `pc_features.npy`, `amplitudes.npy`,
     `whitening_mat_inv.npy`, `spike_times.npy`/`spike_clusters.npy`/`cluster_group.tsv`.

4. **`run_try_merge_selfbuild.sh` + `try_merge_selfdbuild_cleaned.py`** — auto-merging
   pass over the phy export (SNR/ContamPct labelling, similarity+XCorr-gated merge
   groups, QC figures, optional destructive merge-apply). Run in the `slay`
   micromamba environment (old-API SLAy fork, not the PyPI `slay` package).
   - **Parameters:** RUN_* flags (which stages to (re)compute) and `SELF_BUILD_PARAMS`
     (merge thresholds, waveform window, labelling cutoffs) — edit directly in the
     script, each commented in place.
   - **Output:** `cluster_group.tsv`, `cluster_ContamPct.tsv`, sparse analyzer,
     `<RUN_TAG>/old2new.json`/`new2old.json`, `<RUN_TAG>/merges/*.png` QC figures;
     with `RUN_ACCEPT_MERGES=True`, overwrites `spike_times.npy`/`spike_clusters.npy`
     in place (original backed up with `_orig` suffix).

5. **`try_merge_step_2.py`** — optional second merge pass, run after step 4. Finds
   good↔mua/noise merges that step 4's find_merges structurally can't (it only
   considers good×good pairs). Reuses step 4's labelling/plotting/accept logic
   directly (imported as `s1`) so results stay consistent between the two.
   - **Usage:** edit "Paths"/"Run flags" at the top, then run top-to-bottom or
     cell-by-cell. No dedicated `.sh` wrapper yet — run manually in the `slay` env.
   - **Output:** `STEP2_OUT_DIR/{old2new,new2old,candidates}.{json,tsv}`, QC figures,
     same destructive apply as step 4 if accepted.

## Supplementary

- **`recompute_ks4_features.py`** — not part of the ToMATo pipeline; redoes PC
  features on an *existing* Kilosort4 output (e.g. wanting more channels per unit in
  the PCA without re-sorting).
- **`try_tomato.ipynb`** — dev notebook for interactive exploration/prototyping;
  `try_tomato.py` is the maintained version.

## Legacy / reference

- **`cluster_from_spikes.py` + `.ipynb` + `run_cluster_from_spikes.sh`** — clusters
  peaks with SpikeInterface's own built-in methods (HDBSCAN/isosplit/graph-clustering),
  not Tomato. Superseded by `try_tomato.py`.
- **`compute_in_si.py` + `.ipynb`** — early precursor to
  `try_merge_selfdbuild_cleaned.py`; uses SpikeInterface's built-in SortingAnalyzer
  extensions directly instead of the custom labelling/merge-finding logic. Superseded
  by it. `run_compute.sh` runs this.
- **`combine_tomato_sortings.py`** — early, hardcoded-2-chunk version of
  `merge_tomato_results.py`, written during that script's development. Superseded by it.
- **`try_merge_selfdbuild.py`** — earliest version of the auto-merging pipeline.
  Superseded by `try_merge_selfdbuild_cleaned.py` (same logic + two bug fixes — see
  that file's docstring).
- **`try_merge_selfdbuild_1.py`** — HPC-scale variant with parallel/GPU performance
  options; those options were ported into `try_merge_selfdbuild_cleaned.py`'s
  PERFORMANCE SETTINGS (off by default). Superseded by it.
- **`try_merge_selfdbuild_local.py`** — earlier, smaller iteration (missing
  plot_all_clusters, the phy bootstrap-from-sorting step, and other stages). Superseded
  by `try_merge_selfdbuild_cleaned.py`.

## Recommended practice

Run the main-pipeline steps in order, manually, waiting for each to finish. Use
`try_merge_step_2.py` only if step 4's merges still look incomplete after reviewing
its QC figures.
