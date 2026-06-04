#!/usr/bin/env python3
"""Remove LAMMPS or VASP output files from a run directory (prompts before deleting)."""

from __future__ import annotations

import fnmatch
import sys
from pathlib import Path

VASP_KEEP_NAMES = frozenset(
    {"INCAR", "POSCAR", "POTCAR", "KPOINTS", "submit.vasp6.sh"}
)

# POSCAR is kept for VASP runs but is also common in LAMMPS/Allegro workflows.
VASP_DIR_MARKERS = frozenset(
    {"INCAR", "OUTCAR", "vasprun.xml", "OSZICAR", "POTCAR", "KPOINTS", "CONTCAR"}
)
LAMMPS_DIR_MARKERS = frozenset({"data.lammps", "log.lammps"})


def is_regular_file(path: Path) -> bool:
    """True for ordinary files only (exclude symlinks, like find -type f)."""
    try:
        return path.is_file() and not path.is_symlink()
    except OSError:
        return False


def matches_lammps_pattern(name: str) -> bool:
    return (
        fnmatch.fnmatch(name, "restart*")
        or fnmatch.fnmatch(name, "job*")
        or fnmatch.fnmatch(name, "log.lammps")
        or fnmatch.fnmatch(name, "trajectory*")
    )


def is_lammps_input_name(name: str) -> bool:
    """Inputs and model artifacts that must never be removed by VASP cleanup."""
    if name in {"data.lammps", "best.ckpt", "best.nequip.pt2"}:
        return True
    if name.startswith("submit.") and "lammps" in name:
        return True
    return name.startswith("in.") and (
        name.endswith(".lammps")
        or "allegro" in name
        or fnmatch.fnmatch(name, "in.lammps*")
    )


def has_marker_file(target: Path, names: frozenset[str]) -> bool:
    return any((target / name).is_file() for name in names)


def is_lammps_directory(target: Path) -> bool:
    if has_marker_file(target, LAMMPS_DIR_MARKERS):
        return True
    try:
        for path in target.iterdir():
            if path.is_file() and is_lammps_input_name(path.name):
                return True
    except OSError:
        return False
    return False


def is_vasp_directory(target: Path) -> bool:
    # POSCAR alone is not a VASP run (Allegro/LAMMPS dirs often keep POSCAR).
    if is_lammps_directory(target):
        return False
    return has_marker_file(target, VASP_DIR_MARKERS)


def immediate_vasp_subdirectories(target: Path) -> list[Path]:
    """Direct child folders that look like standalone VASP runs."""
    roots: list[Path] = []
    try:
        for child in target.iterdir():
            if child.is_dir() and is_vasp_directory(child):
                roots.append(child)
    except OSError as e:
        print(f"Error: cannot read directory '{target}': {e}", file=sys.stderr)
        sys.exit(1)
    return sorted(roots)


def collect_vasp_files_to_delete(target: Path) -> list[Path]:
    files: list[Path] = []
    for path in target.rglob("*"):
        if not is_regular_file(path):
            continue
        if path.name in VASP_KEEP_NAMES or is_lammps_input_name(path.name):
            continue
        files.append(path)
    return sorted(files)


def collect_lammps_files_to_delete(target: Path) -> tuple[list[Path], str]:
    top: list[Path] = []
    try:
        for path in target.iterdir():
            if is_regular_file(path) and matches_lammps_pattern(path.name):
                top.append(path)
    except OSError as e:
        print(f"Error: cannot read directory '{target}': {e}", file=sys.stderr)
        sys.exit(1)

    if top:
        return sorted(top), f"top level of '{target}' only"

    sub: list[Path] = []
    for path in target.rglob("*"):
        if not is_regular_file(path):
            continue
        try:
            rel = path.relative_to(target)
        except ValueError:
            continue
        if len(rel.parts) < 2:
            continue
        if matches_lammps_pattern(path.name):
            sub.append(path)

    return sorted(sub), (
        f"subdirectories of '{target}' (recursive; top level had no matches)"
    )


def confirmed() -> bool:
    reply = input("Proceed with deletion? [y/N] ").strip().lower()
    return reply in ("y", "yes")


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <directory>", file=sys.stderr)
        sys.exit(1)

    raw = sys.argv[1].rstrip("/")
    target = Path(raw).expanduser()
    if not target.is_dir():
        print(
            f"Error: '{raw}' is not a directory or does not exist.",
            file=sys.stderr,
        )
        sys.exit(1)

    target = target.resolve()

    vasp_scope = ""
    if is_lammps_directory(target):
        mode = "LAMMPS"
        files, lammps_scope = collect_lammps_files_to_delete(target)
    elif is_vasp_directory(target):
        mode = "VASP"
        files = collect_vasp_files_to_delete(target)
        lammps_scope = ""
    elif vasp_children := immediate_vasp_subdirectories(target):
        mode = "VASP"
        files = sorted(
            p for root in vasp_children for p in collect_vasp_files_to_delete(root)
        )
        lammps_scope = ""
        vasp_scope = (
            "VASP run folder(s): "
            + ", ".join(str(c.relative_to(target)) for c in vasp_children)
        )
    else:
        mode = "LAMMPS"
        files, lammps_scope = collect_lammps_files_to_delete(target)

    if not files:
        if mode == "VASP":
            print(
                "No files to remove under "
                f"'{target}' (only kept inputs may be present, or directory is empty)."
            )
        else:
            print(f"No matching files in '{target}' or its subdirectories.")
        sys.exit(0)

    print(f"Mode: {mode}")
    print(f"Directory: {target}")
    if mode == "VASP" and vasp_scope:
        print(f"Scope: {vasp_scope}")
    if mode == "LAMMPS":
        print(f"Scope: {lammps_scope}")
    print(f"Files to delete ({len(files)}):")
    for p in files:
        print(f"  {p}")
    print()

    if not confirmed():
        print("Aborted.")
        sys.exit(1)

    deleted = 0
    for p in files:
        try:
            p.unlink()
            deleted += 1
        except OSError as e:
            print(f"Warning: could not remove '{p}': {e}", file=sys.stderr)

    print(f"Deleted {deleted} file(s).")


if __name__ == "__main__":
    main()
