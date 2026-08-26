#!/usr/bin/env python3
# %%
"""
combine_tomato_sortings.py
==========================
LEGACY — early, hardcoded-paths version of merge_tomato_results.py, written
while that script was still being developed. Superseded by
merge_tomato_results.py (general N-chunk merge with a CLI, global-sample-index
recovery, and overlap warnings). Kept for reference only.

Merge two NumpyFolderSorting results that cover consecutive time windows
into a single NumpyFolderSorting saved as tomato_sorting2_all.

Inputs
------
  tomato_sorting2_s0_e540000000       (units 0..35988, samples 0..~540M)
  tomato_sorting2_s540000000_e1082222502  (units 0..32981, samples ~540M..~1082M)

Both were sorted independently so unit IDs overlap at 0.
Strategy
--------
  - Keep sorting 1 unit_ids and unit_indices unchanged.
  - Shift sorting 2 unit_indices by N1 = len(sorting1_unit_ids).
  - New unit_ids = sorting1_unit_ids + [uid + N1 for uid in sorting2_unit_ids].
  - Concatenate all spikes and sort by sample_index.
"""

import json
import shutil
from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path("/ptmp/gongt/GLC7189/GLC7189_g_g1/preprocessed")
PATH_A = BASE / "tomato_sorting2_s0_e540000000"
PATH_B = BASE / "tomato_sorting2_s540000000_e1082222502"
OUT    = BASE / "tomato_sorting2_all"

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading sorting A …")
spikes_a = np.load(PATH_A / "spikes.npy")
info_a   = json.loads((PATH_A / "numpysorting_info.json").read_text())
unit_ids_a = np.array(info_a["unit_ids"], dtype=np.int64)
fs_a       = info_a["sampling_frequency"]

print("Loading sorting B …")
spikes_b = np.load(PATH_B / "spikes.npy")
info_b   = json.loads((PATH_B / "numpysorting_info.json").read_text())
unit_ids_b = np.array(info_b["unit_ids"], dtype=np.int64)
fs_b       = info_b["sampling_frequency"]

assert fs_a == fs_b, f"Sampling frequency mismatch: {fs_a} vs {fs_b}"
assert info_a["num_segments"] == 1 and info_b["num_segments"] == 1

print(f"  Sorting A: {len(spikes_a):,} spikes, {len(unit_ids_a)} units, "
      f"sample range {spikes_a['sample_index'].min()}–{spikes_a['sample_index'].max()}")
print(f"  Sorting B: {len(spikes_b):,} spikes, {len(unit_ids_b)} units, "
      f"sample range {spikes_b['sample_index'].min()}–{spikes_b['sample_index'].max()}")

# ── Sanity checks ──────────────────────────────────────────────────────────────
max_sample_a = spikes_a["sample_index"].max()
min_sample_b = spikes_b["sample_index"].min()
assert max_sample_a < min_sample_b, (
    f"Sorting A and B time windows overlap! "
    f"max_A={max_sample_a} >= min_B={min_sample_b}"
)
assert spikes_a["unit_index"].max() == len(unit_ids_a) - 1 or spikes_a["unit_index"].max() < len(unit_ids_a), \
    "Sorting A: unit_index out of range for unit_ids"
assert spikes_b["unit_index"].max() < len(unit_ids_b), \
    "Sorting B: unit_index out of range for unit_ids"

# ── Build merged unit_ids ──────────────────────────────────────────────────────
N_A = len(unit_ids_a)   # offset to add to sorting B's unit indices
# Sorting A unit_ids stay as-is; sorting B unit_ids are offset so they don't clash
unit_ids_b_shifted = unit_ids_b + N_A
merged_unit_ids = np.concatenate([unit_ids_a, unit_ids_b_shifted])

print(f"\nMerged unit_ids: {len(merged_unit_ids)} total "
      f"({N_A} from A + {len(unit_ids_b)} from B, B offset by {N_A})")
print(f"  unit_id range: {merged_unit_ids.min()}–{merged_unit_ids.max()}")

# ── Shift sorting B unit_indices ───────────────────────────────────────────────
spikes_b_shifted = spikes_b.copy()
spikes_b_shifted["unit_index"] = spikes_b["unit_index"] + N_A

# ── Concatenate and sort by sample_index ──────────────────────────────────────
print("\nConcatenating and sorting spikes …")
merged_spikes = np.concatenate([spikes_a, spikes_b_shifted])
sort_order    = np.argsort(merged_spikes["sample_index"], kind="stable")
merged_spikes = merged_spikes[sort_order]

print(f"  Total spikes: {len(merged_spikes):,}")
print(f"  Sample range: {merged_spikes['sample_index'].min()}–{merged_spikes['sample_index'].max()}")
print(f"  unit_index range: {merged_spikes['unit_index'].min()}–{merged_spikes['unit_index'].max()}")

# ── Verify no gaps / overlaps ──────────────────────────────────────────────────
assert merged_spikes["unit_index"].max() == len(merged_unit_ids) - 1 or \
       merged_spikes["unit_index"].max() < len(merged_unit_ids), \
    "unit_index out of range for merged unit_ids"
assert np.all(merged_spikes["unit_index"] >= 0), "Negative unit_index found"

# Check sample_index is non-decreasing
diffs = np.diff(merged_spikes["sample_index"])
assert np.all(diffs >= 0), "Merged spikes are not sorted by sample_index!"

print("\nAll sanity checks passed.")

# ── Write output ───────────────────────────────────────────────────────────────
print(f"\nWriting to: {OUT}")
OUT.mkdir(parents=True, exist_ok=True)

# spikes.npy
np.save(OUT / "spikes.npy", merged_spikes)
print(f"  Saved spikes.npy: {merged_spikes.shape}")

# numpysorting_info.json
merged_info = {
    "sampling_frequency": fs_a,
    "unit_ids": merged_unit_ids.tolist(),
    "num_segments": 1,
}
(OUT / "numpysorting_info.json").write_text(json.dumps(merged_info, indent=2))
print(f"  Saved numpysorting_info.json: {len(merged_unit_ids)} units")

# si_folder.json  (copy from sorting A, folder_path stays ".")
si_folder = json.loads((PATH_A / "si_folder.json").read_text())
si_folder["kwargs"]["folder_path"] = "."
(OUT / "si_folder.json").write_text(json.dumps(si_folder, indent=2))
print(f"  Saved si_folder.json")

print("\nDone!")
print(f"  Output: {OUT}")
print(f"  Units : {len(merged_unit_ids):,}  (A: {N_A}, B: {len(unit_ids_b)})")
print(f"  Spikes: {len(merged_spikes):,}  (A: {len(spikes_a):,}, B: {len(spikes_b):,})")

