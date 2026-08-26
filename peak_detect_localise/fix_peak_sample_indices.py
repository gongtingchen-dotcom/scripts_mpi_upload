#!/usr/bin/env python3
"""Apply a sample-index offset to peak arrays.

Standalone manual repair utility — run by hand, not part of any .sh pipeline.
Rewrites every peak_*.npy file's 'sample_index' field in place, adding a
fixed offset (in frames, or seconds converted via --sampling-frequency).

Written for one specific problem: peak files that were produced from a
SLICED recording (e.g. --time_start_s set) but never got shifted back to be
relative to the ORIGINAL, unsliced recording — so their sample_index values
don't line up with anything computed from the full recording.

CONFIRMED (checked directly in peak_localisation.py): this shift now happens
automatically for any peaks produced by running peak_localisation.py itself
— shift_peaks_to_original_time() is called right after detection, and
peaks reloaded from a previous run are explicitly assumed to already be
shifted (see the comment on that code path). So this script shouldn't be
needed for peaks_*.npy files that came out of peak_localisation.py. It's
only useful for an old peaks file predating that fix, or peaks produced some
other way (e.g. calling mua_pipeline.run_mua_pipeline() directly, outside
peak_localisation.py) that hit the same kind of offset problem.

Examples:
    python fix_peak_sample_indices.py \
        --path /path/to/peaks_folder \
        --offset-seconds 5600 \
        --sampling-frequency 30000

    python fix_peak_sample_indices.py \
        --path /path/to/peaks2.npy \
        --offset-frames 168000000
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def iter_peak_files(path: Path):
    if path.is_file():
        yield path
        return
    if not path.exists():
        raise FileNotFoundError(path)
    for p in sorted(path.rglob("*.npy")):
        if p.name.startswith("localized_"):
            continue
        yield p


def apply_offset(path: Path, offset: int, dry_run: bool = False) -> bool:
    arr = np.load(path, allow_pickle=True)
    if not hasattr(arr, "dtype") or arr.dtype.names is None or "sample_index" not in arr.dtype.names:
        return False
    if dry_run:
        print(f"[dry-run] {path} -> sample_index + {offset}")
        return True
    arr2 = np.array(arr, copy=True)
    arr2["sample_index"] = arr2["sample_index"].astype(np.int64, copy=False) + offset
    np.save(path, arr2)
    print(f"Updated {path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Shift peak sample_index values by an offset")
    parser.add_argument("--path", required=True, help="Peak file or directory containing .npy peak files")
    parser.add_argument("--offset-frames", type=int, default=None, help="Frame offset to add to sample_index")
    parser.add_argument("--offset-seconds", type=float, default=None, help="Time offset in seconds")
    parser.add_argument("--sampling-frequency", type=float, default=None, help="Sampling frequency used with --offset-seconds")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without writing files")
    args = parser.parse_args()

    offset_frames = args.offset_frames
    if offset_frames is None:
        if args.offset_seconds is None or args.sampling_frequency is None:
            raise SystemExit("Provide either --offset-frames or both --offset-seconds and --sampling-frequency")
        offset_frames = int(round(args.offset_seconds * args.sampling_frequency))

    path = Path(args.path)
    updated = 0
    skipped = 0
    for p in iter_peak_files(path):
        ok = apply_offset(p, offset_frames, dry_run=args.dry_run)
        if ok:
            updated += 1
        else:
            skipped += 1
    print(f"Done. Updated {updated} files, skipped {skipped} non-peak files.")


if __name__ == "__main__":
    main()
