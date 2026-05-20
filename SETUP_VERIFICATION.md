# Preprocessing + mua_pipeline Setup Verification

## 1. Preprocessing Configuration
✓ **Drift correction only**
- `nblocks=5`: Enables drift correction across 5 blocks
- `do_CAR=False`: Skips CAR (already applied to input data)
- `save_preprocessed_copy=False`: Uses temporary save, not permanent
- **Whitening matrix**: Computed internally, then replaced with identity matrix (lines 433-436)
  - `whitening_range=32` (default): Used to compute real whitening matrix
  - `ops["Wrot"] = identity`: Replaces with identity → no whitening applied

**No additional filtering** — only hp-filter (KS4 default) + CAR (already done) + drift correction.

---

## 2. Data Flow: customised_KS4.py → mua_pipeline.py

```
[RECORDING_PATH]
  (SI binary folder or SpikeGLX)
        ↓
   [load_data]
        ↓
  [Drift Correction]
        ↓
  [Save as SI binary] → /KS4_corrected/
        ↓
  [run_mua_pipeline]
        ↓
  [Spike Detection]
        ↓
  [KS4 Clustering]
        ↓
  [RESULTS_DIR]
```

---

## 3. Path Consistency

| Step | Location | Format | Handler |
|------|----------|--------|---------|
| Input | `RECORDING_PATH` | SI binary or SpikeGLX | `load_data()` |
| Temp KS4 | `RESULTS_DIR/temp_wh.dat` | KS4 binary | `io.save_preprocessing()` |
| Drift-corrected | `RESULTS_DIR/KS4_corrected/` | SI binary | `rec_si.save()` |
| mua_pipeline input | (same as above) | SI binary | `run_mua_pipeline()` → `load_data()` |
| Spike peaks | Returned directly | numpy array | No disk I/O needed |
| Final output | `RESULTS_DIR` | Phy format | `save_sorting()` |

---

## 4. Key Verifications

✓ **Whitening disabled**: Identity matrix replaces computed whitening
✓ **SI binary saved**: `KS4_corrected` folder created with proper format
✓ **Probe attached**: Channel locations set on SI recording before mua_pipeline
✓ **Path resolution**: All relative paths use `Path()` objects
✓ **mua_pipeline compatible**: load_data() handles SI folders via `si.load()`

---

## 5. To Run:

```bash
cd /u/gongt
python scripts/customised_KS4.py
```

Output structure:
```
RESULTS_DIR/
├── KS4_corrected/           ← drift-corrected SI binary (input to mua_pipeline)
├── thresholding_peaks/      ← peaks_combined2.npy, etc. (from mua_pipeline)
├── temp_wh.dat              ← temporary KS4 binary (can delete after run)
├── params.py                ← Phy output
├── spike_times.npy
├── spike_clusters.npy
└── [other KS4 outputs]
```
