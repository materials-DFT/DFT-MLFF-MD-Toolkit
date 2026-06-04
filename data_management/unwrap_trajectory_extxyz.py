#!/usr/bin/env python3
"""Unwrap wrapped LAMMPS trajectories stored as extxyz using ASE.

Consecutive-frame displacements are mapped through the minimum image
convention (ASE :func:`ase.geometry.find_mic`) and accumulated so atoms
do not jump across periodic boundaries.

Example::

    python unwrap_trajectory_extxyz.py /path/run1 /path/run2
    python unwrap_trajectory_extxyz.py --filename custom.extxyz ./sim
    python unwrap_trajectory_extxyz.py -r ./study_root
    python unwrap_trajectory_extxyz.py ./sim/trajectory.extxyz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.io import read, write


def unwrap_frames(frames: list[Atoms]) -> None:
    """Remove PBC jumps in-place; frames must share atom order and count."""
    if len(frames) < 2:
        return

    n0 = len(frames[0])
    for i, atoms in enumerate(frames):
        if len(atoms) != n0:
            raise ValueError(
                f"Frame {i} has {len(atoms)} atoms; expected {n0} throughout."
            )
        cell = atoms.get_cell()
        if cell is None or np.linalg.norm(cell) < 1e-12:
            raise ValueError(f"Frame {i} has no valid unit cell; unwrapping needs a cell.")
        if not np.any(atoms.get_pbc()):
            raise ValueError(
                f"Frame {i} has no periodic axes (pbc all False); nothing to unwrap."
            )

    prev_wrapped = frames[0].get_positions().copy()
    pos_unwrapped = prev_wrapped.copy()
    frames[0].set_positions(pos_unwrapped)

    for i in range(1, len(frames)):
        atoms = frames[i]
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        wrapped = atoms.get_positions().copy()
        dr = wrapped - prev_wrapped
        dr_mic, _ = find_mic(dr, cell, pbc=pbc)
        pos_unwrapped = pos_unwrapped + dr_mic
        atoms.set_positions(pos_unwrapped)
        prev_wrapped = wrapped


def collect_trajectory_paths(
    roots: list[Path],
    filename: str,
    recursive: bool,
    quiet: bool,
) -> tuple[list[Path], int]:
    """Expand CLI roots into concrete trajectory files. Returns (paths, n_errors)."""
    seen: set[Path] = set()
    paths: list[Path] = []
    errors = 0

    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            print(f"error: does not exist: {root}", file=sys.stderr)
            errors += 1
            continue

        if root.is_file():
            if root not in seen:
                seen.add(root)
                paths.append(root)
            continue

        if root.is_dir():
            if recursive:
                found = False
                for p in sorted(root.rglob(filename)):
                    if p.is_file() and p not in seen:
                        seen.add(p)
                        paths.append(p)
                        found = True
                if not found and not quiet:
                    print(
                        f"skip: no '{filename}' under {root} (recursive)",
                        file=sys.stderr,
                    )
            else:
                p = root / filename
                if p.is_file():
                    if p not in seen:
                        seen.add(p)
                        paths.append(p)
                elif not quiet:
                    print(f"skip: no file {p}", file=sys.stderr)
            continue

        print(f"error: not a file or directory: {root}", file=sys.stderr)
        errors += 1

    return paths, errors


def process_file(path: Path, output_name: str | None, quiet: bool) -> Path | None:
    """Read trajectory at path, unwrap in-place, write alongside input."""
    path = path.resolve()
    if not path.is_file():
        if not quiet:
            print(f"skip: not a file {path}", file=sys.stderr)
        return None

    if not quiet:
        print(f"reading {path} …", file=sys.stderr)
    frames = read(str(path), index=":")
    if not isinstance(frames, list):
        frames = [frames]

    if len(frames) == 0:
        if not quiet:
            print(f"skip: empty trajectory {path}", file=sys.stderr)
        return None

    unwrap_frames(frames)

    out = path.parent / (output_name or f"{path.stem}_unwrapped{path.suffix}")
    if not quiet:
        print(f"writing {out} ({len(frames)} frames) …", file=sys.stderr)
    write(str(out), frames)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Unwrap LAMMPS extxyz trajectories using ASE. "
            "Each argument may be a trajectory file or a directory "
            "(see --filename, --recursive)."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Trajectory .extxyz files and/or directories to search",
    )
    parser.add_argument(
        "-f",
        "--filename",
        default="trajectory.extxyz",
        help=(
            "When a path is a directory: basename of the trajectory inside it "
            "(default: trajectory.extxyz). With --recursive, match this name at any depth."
        ),
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="For directory arguments, find --filename under that tree (rglob).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output basename written next to each input; default: <stem>_unwrapped.extxyz",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )
    args = parser.parse_args()

    trajectories, n_err = collect_trajectory_paths(
        args.paths, args.filename, args.recursive, args.quiet
    )
    if n_err:
        return 1

    written: list[Path] = []
    for path in trajectories:
        out = process_file(path, args.output, args.quiet)
        if out is not None:
            written.append(out)

    if not args.quiet and written:
        print("done:", *written, sep="\n  ")
    if not written:
        if not args.quiet:
            print("warning: no trajectories were written", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
