# environments

Conda environment specs for every environment the pipeline uses, exported from the
working environments on this cluster (MPCDF Raven) via `conda env export`. Which
script uses which env is always stated at the top of that script's `.sh` wrapper
(`conda activate <name>`) — summary:

| env | used by |
|---|---|
| `spikeinterface` | `preprocess/`, `peak_detect_localise/`, `motion/`, most of `tomato/` |
| `phy2` | `sorters/` (KS4/IronClust/MountainSort5), `tomato/`'s legacy `compute_in_si.py` |
| `slay` | `tomato/`'s auto-merging scripts (`try_merge_selfdbuild_cleaned.py`, `try_merge_step_2.py`) |
| `density_decoding` | `density_decoding/` |
| `decoding` | `theo_decoding/` |

## Setup

```bash
conda env create -f environments/<name>.yml
```

These are full exports (`conda env export`), pinned to exact versions built for
Linux x86_64/CUDA on this cluster — the most reliable way to reproduce the same
behavior here, but not architecture-portable. On a different cluster/OS, treat the
`pip:` section's version pins as the source of truth and let conda resolve its own
build strings.

## Three environments need a hand-modified package installed on top

Three packages were developed/modified locally rather than used as-is from PyPI.
Each `.yml` above already lists the right one as an editable git install (pinned to
the exact commit the export was taken from) — `conda env create` will `git clone`
+ `pip install -e` it automatically, **provided you have SSH/HTTPS access to the
remote**:

- **`slay`** (used by the `slay` env) — old-API fork with the merge-finding
  functions `try_merge_selfdbuild_cleaned.py` needs (the rewritten PyPI `slay`
  package does NOT have them — do not let pip silently resolve to that one).
  Repo: `git@gitlab.mpcdf.mpg.de:mpibr/laur/slay.git`.
- **`density_decoding`** (used by the `density_decoding` env) — the actual
  GMM/GLM/ADVI decode pipeline `density_decoding_unified.py` calls.
  Repo: `git@gitlab.mpcdf.mpg.de:mpibr/laur/density_decoding.git`.
- **`subtraction_pipeline`** (used by the `spikeinterface` env, only for the
  legacy `peak_detect_localise/boussard_localization.py` comparison path) — our
  fork of int-brain-lab's `spikes_localization_registration`, with local
  numpy-compatibility fixes (`ndarray.ptp()` → `np.ptp()`) committed on top.
  Repo: `git@gitlab.mpcdf.mpg.de:mpibr/laur/spikes_localization_registration_gongt.git`.
  Legacy/not part of the main pipeline (see `peak_detect_localise/README.md`).

## One environment needs a one-line hand-edit after install

`kilosort` itself is a stock, version-pinned pip install everywhere it's used
(`kilosort==4.1.7` in `slay`, `==4.1.1` in `phy2` — both already correctly captured
in the `.yml` files above; `sorters/ks4_checkpoint.py`'s custom checkpointing never
edits the installed package — it's all done via runtime monkey-patching from within
that script, see its header). **Except**: in the `phy2` env specifically, kilosort's
own auto-merge step has been hand-disabled — a direct edit to the installed
package's source, made after `pip install`, and NOT captured by any `.yml` file or
git repo. To reproduce it in a fresh `phy2` env:

1. Find `cluster_spikes()` in `<env>/lib/python*/site-packages/kilosort/run_kilosort.py`.
2. Comment out the `Wall, clu, is_ref, st, tF = template_matching.merging_function(...)`
   call inside it (currently ~line 905 in kilosort 4.1.1), leaving `clu` as-is
   (kilosort4's raw, un-merged per-template clusters).

See `sorters/ks4_checkpoint.py`'s "ENVIRONMENT NOTE" for why this matters and what
it affects (every kilosort-based script in `sorters/`, via `cluster_spikes()` /
`run_kilosort()`). The `slay` env's kilosort is unaffected — this call is intact
there.
