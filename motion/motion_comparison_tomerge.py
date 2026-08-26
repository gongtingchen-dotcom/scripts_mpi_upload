#!/usr/bin/env python3
"""
SUPERSEDED by motion_comparison.py — this copy still points at
/gpfs/laur/... paths, an old storage location not reachable from this
cluster anymore (checked: doesn't exist). motion_comparison.py has the same
logic with corrected /ptmp/gongt/... paths. Kept only in case anything in
this exact copy is needed for reference; use motion_comparison.py instead.

Compare multiple SpikeInterface motion-estimation presets on the same data chunk.

For each preset in PRESETS, this script:
  1. Loads /ptmp/gongt/GLC6962_l_g1/preprocessed, restricted to a chosen time chunk
     (default: 0-70 min, matching the existing motion_estimates/chunks_1 layout).
  2. Runs si.compute_motion(rec_chunk, preset=preset) — this internally detects and
     localizes peaks (using THAT preset's own detect/localize parameters — different
     presets use different detection thresholds and localization methods, so each
     preset's peaks differ from the others) and estimates motion from them.
  3. Saves the resulting Motion object with motion.save() to
       .../thresholding_peaks/motion_estimates/chunks_1/motion_{preset}_{time_tag}/
     — the same flat format already used by motion_dredge_0000min_0070min, so it's
     directly loadable via si.Motion.load(...) and usable by combine_motions.py.
     Motion.save() alone is already enough to fully reconstruct the Motion object
     (displacement + temporal/spatial bins). On top of that this script also stashes
     the preset's own peaks.npy / peak_locations.npy / parameters.json / run_times.json
     in the same folder, so the exact preset run is fully reproducible/inspectable too.
  4. Calls correct_motion.py (unmodified) as a subprocess with
       --chunks_dir chunks_1 --motion_name motion_{preset}_{time_tag}
     which generates the full existing plot suite (drift scatter, motion overlay,
     corrected scatter, channel plots, etc.) into that motion's own folder — exactly
     the same plots already produced for the existing dredge motion.

Note: correct_motion.py plots the SAME shared background peak scatter every time
(thresholding_peaks/peaks_combined2_sorted.npy + localized_peaks_mt_raw.npy), with
each preset's drift curve overlaid on top. That's intentional — it keeps the
background identical across all 4 comparison plots so the drift curves themselves
are what differs, even though internally each preset estimated its motion from its
own, different set of detected peaks.

Also note: correct_motion.py's plot titles/legends hardcode the label "DREDge"
regardless of which preset actually produced the motion (that's baked into the
script and is not modified here) — use the folder name to know which preset a
given set of plots belongs to.
"""

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import spikeinterface.full as si

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
REC_DIR     = Path("/gpfs/laur/data/0_collaboration/ceph_phys/GLC6962/GLC6962_l_g1/preprocessed")
CHUNKS_DIR  = "chunks_1"                 # matches correct_motion.py's own default
MOTION_ROOT = REC_DIR / "preprocessed" / "motion_estimates" / CHUNKS_DIR

START_MIN = 0
END_MIN   = 70

PRESETS = [
    "nonrigid_accurate",
    "kilosort_like",
    "dredge_fast",
    "nonrigid_fast_and_accurate",
]

CORRECT_MOTION_SCRIPT = Path("/gpfs/laur/personal/gongt/scripts/motion/correct_motion.py")
RUN_PLOTS_AFTER       = True   # set False to only compute + save motions, skip plotting

N_JOBS = 8
# ══════════════════════════════════════════════════════════════

si.set_global_job_kwargs(n_jobs=N_JOBS)


def time_tag(start_min, end_min):
    return f"{int(start_min):04d}min_{int(end_min):04d}min"


def load_chunk(rec_dir, start_min, end_min):
    rec = si.load(rec_dir)
    fs = rec.get_sampling_frequency()
    start_frame = int(start_min * 60 * fs)
    end_frame = min(int(end_min * 60 * fs), rec.get_num_frames())
    rec = rec.frame_slice(start_frame=start_frame, end_frame=end_frame)

    # numba's peak-detection kernel (used by every preset here) can't JIT-compile
    # against float16 arrays ("NotImplementedError: float16"), and compute_motion()
    # does not auto-upcast dtype the way the higher-level correct_motion() wrapper
    # does. Preprocessed recordings in this pipeline are saved as float16, so cast
    # the (lazy, already-sliced) chunk up to float32 before motion estimation.
    if str(rec.get_dtype()) == "float16":
        print(f"Recording dtype is float16 — casting the {end_min - start_min}min chunk to float32 "
              f"(required for numba peak detection).")
        rec = rec.astype("float32")

    return rec


def run_preset(rec_chunk, preset, out_dir):
    """Estimate motion for one preset and save it (+ provenance) to out_dir."""
    if out_dir.exists():
        print(f"[{preset}] {out_dir.name} already exists — skipping motion estimation")
        return True

    print(f"[{preset}] estimating motion...")
    t0 = time.time()
    motion, motion_info = si.compute_motion(
        rec_chunk,
        preset=preset,
        output_motion_info=True,
        folder=None,          # we save manually below, to match the existing flat folder layout
        raise_error=False,    # don't let one bad preset kill the whole comparison
        progress_bar=True,
    )
    elapsed = time.time() - t0

    if motion is None:
        print(f"[{preset}] FAILED after {elapsed:.1f}s — motion estimation returned None, skipping save")
        return False

    print(f"[{preset}] done in {elapsed:.1f}s — saving to {out_dir}")
    motion.save(out_dir)   # creates out_dir; loadable via si.Motion.load(out_dir)

    # Extra provenance alongside Motion.save()'s own files (no filename collisions):
    # enough to fully reconstruct exactly how this preset's motion was computed.
    np.save(out_dir / "peaks.npy", motion_info["peaks"])
    np.save(out_dir / "peak_locations.npy", motion_info["peak_locations"])
    (out_dir / "parameters.json").write_text(
        json.dumps(motion_info["parameters"], indent=2, default=str)
    )
    (out_dir / "run_times.json").write_text(
        json.dumps({**motion_info["run_times"], "total_s": elapsed}, indent=2)
    )
    print(f"[{preset}] saved -> {out_dir}")
    return True


def run_plots(motion_name):
    """Interface with correct_motion.py (unmodified) to generate the standard plot suite."""
    cmd = [
        sys.executable, str(CORRECT_MOTION_SCRIPT),
        "--chunks_dir", CHUNKS_DIR,
        "--motion_name", motion_name,
    ]
    print(f"  running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    tag = time_tag(START_MIN, END_MIN)
    MOTION_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Loading recording chunk [{START_MIN}-{END_MIN}] min from {REC_DIR} ...")
    rec_chunk = load_chunk(REC_DIR, START_MIN, END_MIN)
    print(rec_chunk)

    for preset in PRESETS:
        name = f"motion_{preset}_{tag}"
        out_dir = MOTION_ROOT / name
        print(f"\n{'='*70}\n{preset}\n{'='*70}")

        try:
            ok = run_preset(rec_chunk, preset, out_dir)
        except Exception:
            print(f"[{preset}] ERROR during motion estimation:")
            traceback.print_exc()
            continue

        if not ok:
            continue

        if RUN_PLOTS_AFTER:
            print(f"[{preset}] generating comparison plots via correct_motion.py ...")
            try:
                run_plots(name)
            except subprocess.CalledProcessError as e:
                print(f"[{preset}] plotting failed (exit code {e.returncode}) — continuing to next preset")

    print(f"\nDone. Results under: {MOTION_ROOT}")


if __name__ == "__main__":
    main()