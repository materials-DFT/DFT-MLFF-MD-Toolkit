#!/usr/bin/env python3
"""Split an extxyz dataset into all/high/normal force subsets.

The split criterion is the maximum per-atom force magnitude in each frame.
Frames with max |F| > threshold go to ``high_forces``; the rest go to
``normal_forces``. All frames are always written to ``all_forces``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write


def frame_max_force(frame) -> float:
    forces = frame.get_forces()
    return float(np.linalg.norm(forces, axis=1).max())


def resolve_input_path(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    preferred = sorted(
        [
            *path.glob("all_frames_*.xyz"),
            *path.glob("all_frames_*.extxyz"),
        ]
    )
    if len(preferred) == 1:
        return preferred[0]
    if len(preferred) > 1:
        names = ", ".join(p.name for p in preferred)
        raise ValueError(
            f"Multiple preferred dataset files found in {path}; pass one explicitly: {names}"
        )

    candidates = sorted(
        [
            *path.glob("*.xyz"),
            *path.glob("*.extxyz"),
        ]
    )
    if not candidates:
        raise FileNotFoundError(f"No .xyz or .extxyz dataset found in directory: {path}")
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise ValueError(
            f"Multiple dataset files found in {path}; pass one explicitly: {names}"
        )
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split an extxyz dataset by max per-atom force magnitude."
    )
    parser.add_argument(
        "input_xyz",
        type=Path,
        help="Path to an input .xyz/.extxyz file, or a directory containing exactly one such file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to create the split folders in. Defaults to the input file's parent.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Force threshold in eV/A for the high_forces split (default: 10.0).",
    )
    args = parser.parse_args()

    input_xyz = resolve_input_path(args.input_xyz)
    output_dir = args.output_dir or input_xyz.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = read(str(input_xyz), ":")

    all_dir = output_dir / "all_forces"
    high_dir = output_dir / "high_forces"
    normal_dir = output_dir / "normal_forces"
    for d in (all_dir, high_dir, normal_dir):
        d.mkdir(parents=True, exist_ok=True)

    stem = input_xyz.stem
    all_path = all_dir / f"{stem}.xyz"
    high_path = high_dir / "high_force_frames.xyz"
    normal_path = normal_dir / "normal_force_frames.xyz"

    high_frames = []
    normal_frames = []
    for frame in frames:
        if frame_max_force(frame) > args.threshold:
            high_frames.append(frame)
        else:
            normal_frames.append(frame)

    write(str(all_path), frames)
    write(str(high_path), high_frames)
    write(str(normal_path), normal_frames)

    print(f"input:   {input_xyz}")
    print(f"frames:  {len(frames)}")
    print(f"threshold: {args.threshold} eV/A")
    print(f"all:     {all_path} ({len(frames)} frames)")
    print(f"high:    {high_path} ({len(high_frames)} frames)")
    print(f"normal:  {normal_path} ({len(normal_frames)} frames)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
