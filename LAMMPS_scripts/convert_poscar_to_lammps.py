#!/usr/bin/env python3
"""
Convert POSCAR / CONTCAR files to LAMMPS data files using ASE.

By default we use ASE's triclinic LAMMPS box (force_skew=True). Orthogonal-only
exports omit xy/xz/yz even when the POSCAR cell is slightly non-orthogonal; that
representation has triggered segfaults in LAMMPS pair allegro/kk + LibTorch on
some GPU stacks. force_skew writes explicit tilt factors and avoids that path.

Velocities: by default ASE writes a ``Velocities`` block when ``velocities=True``.
Cartesian speeds from VASP (Å/fs in the POSCAR/CONTCAR convention ASE uses) are
converted to LAMMPS ``metal`` units (Å/ps). Use ``--no-velocities`` to omit the
block (then LAMMPS initializes v=0 unless you ``velocity create``).

Usage:
    python poscar_to_lammps.py POSCAR [output_filename]
    python poscar_to_lammps.py POSCAR data.lammps --no-force-skew   # legacy orthogonal
    python poscar_to_lammps.py CONTCAR data.lammps --no-velocities
    python poscar_to_lammps.py some_dir/                    # all POSCAR/CONTCAR under tree
    python poscar_to_lammps.py dir1/ dir2/ subdir/          # multiple roots, recursive

For each directory argument, every file named POSCAR or CONTCAR (case-insensitive)
under that directory and its subdirectories is converted. The LAMMPS file is
written as ``data.lammps`` in the same directory as each structure file.

When multiple path arguments are given and the second is an existing file that
looks like another structure (e.g. named POSCAR or CONTCAR), both are converted.
Use ``-o`` / ``--output`` to set the output path when converting exactly one file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

STRUCTURE_NAMES = frozenset({"POSCAR", "CONTCAR"})


def _is_structure_filename(name: str) -> bool:
    return name.upper() in STRUCTURE_NAMES


def iter_structure_files(*roots: Path) -> list[Path]:
    """Collect POSCAR/CONTCAR paths: files as-is; directories searched recursively."""
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        p = Path(root).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"No such path: {root}")
        if p.is_file():
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                out.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if not f.is_file() or not _is_structure_filename(f.name):
                    continue
                rp = f.resolve()
                if rp not in seen:
                    seen.add(rp)
                    out.append(f)
        else:
            raise OSError(f"Not a file or directory: {p}")
    return out


def convert_one(
    poscar: Path,
    output: Path,
    *,
    no_force_skew: bool,
    no_velocities: bool,
) -> None:
    atoms = read(str(poscar), format="vasp")
    write_vel = not no_velocities
    write(
        str(output),
        atoms,
        format="lammps-data",
        atom_style="atomic",
        masses=True,
        force_skew=not no_force_skew,
        velocities=write_vel,
        units="metal",
    )
    skew = "orthogonal (no force_skew)" if no_force_skew else "triclinic (force_skew=True)"
    vmax = None
    if atoms.get_velocities() is not None:
        vmax = float(np.max(np.abs(atoms.get_velocities())))
    vel_note = (
        "no Velocities section"
        if no_velocities
        else (f"velocities (|v|_max ≈ {vmax:.4g} Å/fs in ASE)" if vmax is not None else "velocities")
    )
    print(
        f"Wrote {output} from {poscar} ({len(atoms)} atoms, {skew}, {vel_note})."
    )


def _legacy_two_arg_mode(paths: list[str]) -> tuple[list[str], str | None] | None:
    """
    Old API: ``script POSCAR [output]`` with exactly two args where the second
    is the output filename (not an existing structure file).
    Returns (input_paths, single_output) or None if not legacy mode.
    """
    if len(paths) != 2:
        return None
    p0, p1 = Path(paths[0]).expanduser(), Path(paths[1]).expanduser()
    if not p0.is_file():
        return None
    if p1.is_dir():
        return None
    if p1.is_file() and _is_structure_filename(p1.name):
        return None
    return ([paths[0]], str(p1))


def main() -> None:
    p = argparse.ArgumentParser(description="Convert POSCAR to LAMMPS data (ASE).")
    p.add_argument(
        "paths",
        nargs="+",
        help="One or more POSCAR/CONTCAR files and/or directories (searched recursively)",
    )
    p.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Output LAMMPS path when exactly one input file is given (default: data.lammps in cwd)",
    )
    p.add_argument(
        "--no-force-skew",
        action="store_true",
        help="Let ASE omit triclinic tilt if nearly orthogonal (not recommended for Allegro/Kokkos GPU)",
    )
    p.add_argument(
        "--no-velocities",
        action="store_true",
        help="Do not write a Velocities section (default: write if ASE has velocities; VASP files without a block appear as zeros)",
    )
    args = p.parse_args()
    raw_paths = list(args.paths)

    legacy = _legacy_two_arg_mode(raw_paths)
    if legacy is not None:
        raw_paths, legacy_out = legacy
    else:
        legacy_out = None

    if args.output is not None and legacy_out is not None:
        print("Error: use either -o/--output or the legacy second positional, not both.", file=sys.stderr)
        sys.exit(2)

    effective_output = args.output if args.output is not None else legacy_out

    try:
        inputs = iter_structure_files(*[Path(x) for x in raw_paths])
    except OSError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not inputs:
        print("Error: no POSCAR/CONTCAR files found.", file=sys.stderr)
        sys.exit(1)

    if effective_output is not None:
        if len(inputs) != 1:
            print(
                "Error: -o/--output (or legacy output positional) only applies when exactly one structure file is selected.",
                file=sys.stderr,
            )
            sys.exit(2)
        out_paths = [Path(effective_output).expanduser().resolve()]
    else:
        if len(inputs) == 1:
            out_paths = [Path("data.lammps").resolve()]
        else:
            out_paths = [f.parent / "data.lammps" for f in inputs]

    failed = 0
    for in_path, out_path in zip(inputs, out_paths):
        try:
            convert_one(
                in_path,
                out_path,
                no_force_skew=args.no_force_skew,
                no_velocities=args.no_velocities,
            )
        except OSError as e:
            print(f"Error: {in_path}: {e}", file=sys.stderr)
            failed += 1
        except Exception as e:
            print(f"Error: {in_path}: {e}", file=sys.stderr)
            failed += 1

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
