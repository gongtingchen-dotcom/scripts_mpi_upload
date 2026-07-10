#!/usr/bin/env python3
"""Apply a sample-index offset to peak arrays.

This is a manual repair utility for peak files that were produced from a
sliced recording but still need to be expressed relative to the original
recording timeline.

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
