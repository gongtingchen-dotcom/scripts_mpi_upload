# sorters

Preprocessed recording → sorted single units. One of two independent ways to turn a
preprocessed recording into analyzable output — see [`../README.md`](../README.md).

## Environment

`conda env create -f ../environments/phy2.yml` (all scripts in this folder use the
`phy2` env — every `.sh` here has `conda activate phy2`). No extra repo to install —
`kilosort` is a stock pip install (`kilosort==4.1.1`, already pinned in the `.yml`).

**But**: this `phy2` env's installed kilosort package has its own auto-merge step
hand-disabled (`template_matching.merging_function` call commented out inside
`cluster_spikes()` in `kilosort/run_kilosort.py`) — a direct edit to the installed
package, not reproducible by `pip install` alone. See `ks4_checkpoint.py`'s
"ENVIRONMENT NOTE" and `../environments/README.md` for exactly what to redo after a
fresh install if you want this pipeline's original behavior (raw, un-merged KS4
clusters) preserved.

## Main pipeline

- **`run_ks4_checkpoint.sh` + `ks4_checkpoint.py`** — Kilosort4. Same pipeline and output as `KS4.py`, plus stage-level checkpointing: it gets SLURM-requeued automatically and resumes from its last completed stage, so a long sort that doesn't fit in one 24h allocation isn't lost. On short/medium recordings the checkpointing is just a harmless safety net. Also saves KS4's intermediate data at every stage, which is handy for troubleshooting. A reasonable default for everything; particularly worth it for long recordings. 
  - **Usage:** `sbatch run_ks4_checkpoint.sh --root_folder <session dir> --data_folder <session dir>/preprocessed [kilosort params...]`
  - **Parameters:** every `kilosort4` param from `si.get_default_sorter_params('kilosort4')` is a CLI flag. Commonly tuned: `--nblocks` (drift-correction blocks; 0 disables), `--Th_universal`/`--Th_learned` (detection thresholds), `--whitening_range` (preprocessing spatial extent), `--n_pcs`/`--nearest_chans` (pcs for clustering), etc. 
  - **Outputs**, under `<root_folder>/KS4/<run_name>/` (`run_name` auto-encodes changed params): `sorter_output/` (Phy-compatible spike times/clusters/templates/`ops.npy`), `kilosort_params.txt` (exact params used).

## Supplementary

- **`run_ks4.sh` + `KS4.py`** — plain Kilosort4, same CLI/parameters/output as `ks4_checkpoint.py`, just without the checkpoint/requeue machinery. Perfectly fine to use; the only catch is that a sort exceeding the 24h walltime is lost rather than resumed. 
- **`run_IC.sh` + `ironclust.py`** — IronClust, an alternative sorter, run for comparison against Kilosort4. No CLI — edit `root_folder`/`data_folder` in the file. Output: `<root_folder>/IC_test/` (`sorter_output/`, `si_analyzer/`, `export_to_phy/`). Note: the export_to_phy function for IronClust may have been changed / depreciated, see spikeinterface github for more info. 
- **`run_ms5.sh` + `ms5.py`** — MountainSort5, another comparison sorter. CLI: `--root_folder`/`--data_folder` + MountainSort5's own two-phase detection params (see script header). Output: `<root_folder>/ms5_<run_name>/`.
- **`matlab_test.sh`** — unrelated one-off: checks MATLAB can see the GPU on this cluster. Not part of sorting. Useful if need to use IronClust on Raven. 
- `ironclust.ipynb`, `ms5.ipynb` — interactive dev notebooks the two scripts above were built from; kept for reference.

## Legacy / reference

- **`customised_KS4.py` + `run_customised_KS4.sh`** — attempted to feed our own detected spikes (from `peak_detect_localise/`) into Kilosort4's clustering instead of its own detector. Sorting quality was still poor; but I did not do enough parameter searches (especially the ones that turn out to work well) on this. 
- **`kilosort_diagnostic_plots.ipynb`, `spikeinterface_sorter.py`/`.ipynb`, `spikeinterface_sorter2.py`** — deleted (confirmed superseded/redundant, see git history if needed again).

## Recommended practice

Either KS4 script is fine. `ks4_checkpoint.py` is a good default — the checkpointing
costs nothing on short runs and saves a long run from being lost at the walltime cap.
