#!/usr/bin/env python3
"""Create one VASP single-point directory per top-level .vasp file.

Each input file `foo.vasp` becomes exactly one directory `foo/` containing
`POSCAR` (converted from the .vasp). The original .vasp is removed.
"""

import argparse
import shutil
from collections import Counter
from pathlib import Path

from ase.io import read, write

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = SCRIPT_DIR / "delta_npt_frames" / "frame_0000_K24Mn32O64"

MAGMOM_MAP = {"Mn": 3.0}
DEFAULT_MAGMOM = 0.0


def generate_magmom(symbols, counts):
    parts = []
    for symbol, count in zip(symbols, counts):
        mag = MAGMOM_MAP.get(symbol, DEFAULT_MAGMOM)
        parts.append(f"{count}*{mag:g}")
    return " ".join(parts)


def get_element_counts(atoms):
    counter = Counter(atoms.get_chemical_symbols())
    symbols = sorted(counter.keys())
    return symbols, [counter[s] for s in symbols]


def write_incar(path, system_name, magmom):
    content = f"""System = {system_name} single-point

Starting parameters:
ISTART = 0
ICHARG = 2

Electronic Relaxation:
PREC = Accurate
ENCUT = 520
NELMIN = 6
NELM = 500
EDIFF = 1E-6
LREAL = .FALSE.
ISPIN = 2
MAGMOM = {magmom}
ALGO = All

METAGGA = SCAN
LMIXTAU = .TRUE.
LASPH = .TRUE.
LDIAG = .TRUE.

Single-point (no ionic relaxation):
# NSW = 0
IBRION = -1
# ISIF = 2
ISYM = 0

DOS related:
LORBIT = 10
ISMEAR = 0
SIGMA = 0.05

Parallelization:
NCORE = 8
NSIM = 4
LPLANE = .TRUE.
LSCALU = .FALSE.
KPAR = 1

IO Control:
LWAVE = .FALSE.
LCHARG = .FALSE.
"""
    path.write_text(content)


def write_kpoints(path):
    path.write_text(
        "Gamma-only\n"
        "0\n"
        "Gamma\n"
        "1 1 1\n"
        "0 0 0\n"
    )


def top_level_vasp_files(source: Path) -> list[Path]:
    """Return only direct-child .vasp files, never nested ones."""
    return sorted(
        p for p in source.iterdir()
        if p.is_file() and p.suffix == ".vasp"
    )


def nested_vasp_files(source: Path) -> list[Path]:
    return [
        p for p in source.rglob("*.vasp")
        if p.parent != source
    ]


def nested_calc_dirs(source: Path) -> list[Path]:
    """Directories below the first level (erroneous nested calc dirs)."""
    nested = []
    for path in source.iterdir():
        if not path.is_dir():
            continue
        for child in path.rglob("*"):
            if child.is_dir() and child != path:
                nested.append(child)
    return sorted(nested)


def prune_nested_dirs(source: Path, dry_run: bool = False) -> int:
    removed = 0
    for path in sorted(nested_calc_dirs(source), key=lambda p: len(p.parts), reverse=True):
        if dry_run:
            print(f"Would remove nested directory: {path.relative_to(source)}")
        else:
            shutil.rmtree(path)
            print(f"Removed nested directory: {path.relative_to(source)}")
        removed += 1
    return removed


def validate_source_layout(source: Path, prune_nested: bool = False, dry_run: bool = False):
    nested_vasps = nested_vasp_files(source)
    if nested_vasps:
        examples = ", ".join(str(p.relative_to(source)) for p in nested_vasps[:5])
        raise SystemExit(
            "Error: found .vasp files inside calculation subdirectories. "
            "Only top-level .vasp files are supported.\n"
            f"Examples: {examples}\n"
            "This usually means the setup script was re-run recursively and created "
            "nested directories. Remove nested .vasp files or use --prune-nested-dirs "
            "after backing up anything important."
        )

    nested_dirs = nested_calc_dirs(source)
    if nested_dirs:
        if not prune_nested:
            examples = ", ".join(str(p.relative_to(source)) for p in nested_dirs[:5])
            raise SystemExit(
                "Error: found nested calculation directories below the top level.\n"
                f"Examples: {examples}\n"
                "Re-run with --prune-nested-dirs to remove them, or clean up manually."
            )
        removed = prune_nested_dirs(source, dry_run=dry_run)
        print(f"{'Would remove' if dry_run else 'Removed'} {removed} nested director"
              f"{'y' if removed == 1 else 'ies'}.")


def setup_single_point_dft(
    source_dir,
    template_dir=None,
    dry_run=False,
    wrap=True,
    limit=None,
    prune_nested=False,
):
    source = Path(source_dir).resolve()
    if not source.is_dir():
        raise SystemExit(f"Error: '{source_dir}' is not a directory")

    if (source / "POSCAR").is_file() and (source / "INCAR").is_file():
        raise SystemExit(
            f"Error: '{source}' already looks like a VASP calculation directory. "
            "Pass the directory that contains .vasp files, not an individual calculation."
        )

    validate_source_layout(source, prune_nested=prune_nested, dry_run=dry_run)

    template = Path(template_dir).resolve() if template_dir else TEMPLATE_DIR
    potcar_template = template / "POTCAR"
    submit_template = template / "submit.vasp6.sh"
    for name, path in [("POTCAR", potcar_template), ("submit.vasp6.sh", submit_template)]:
        if not path.is_file():
            raise SystemExit(f"Error: template file not found: {path}")

    vasp_files = top_level_vasp_files(source)
    if not vasp_files:
        existing = sum(1 for p in source.iterdir() if p.is_dir() and (p / "POSCAR").is_file())
        if existing:
            print(f"No top-level .vasp files found. {existing} calculation director"
                  f"{'y' if existing == 1 else 'ies'} already present under {source}")
            return
        raise SystemExit(f"No top-level .vasp files found in '{source}'")

    if limit is not None:
        vasp_files = vasp_files[:limit]

    created = 0
    cleaned = 0
    skipped = 0

    for vasp_file in vasp_files:
        dest = source / vasp_file.stem
        if dest.parent != source:
            raise SystemExit(f"Internal error: destination is not a direct child: {dest}")

        poscar_path = dest / "POSCAR"

        if dest.is_dir() and poscar_path.is_file():
            skipped += 1
            if vasp_file.exists():
                if dry_run:
                    print(f"Would remove duplicate top-level file: {vasp_file.name}")
                else:
                    vasp_file.unlink()
                    cleaned += 1
            continue

        if dest.exists():
            raise SystemExit(
                f"Error: {dest.relative_to(source)} exists but has no POSCAR; "
                f"refusing to overwrite while processing {vasp_file.name}"
            )

        atoms = read(str(vasp_file), format="vasp")
        frames = atoms if isinstance(atoms, list) else [atoms]
        if len(frames) != 1:
            raise SystemExit(
                f"Error: {vasp_file.name} contains {len(frames)} structures. "
                "Expected exactly one structure per .vasp file."
            )
        atoms = frames[0]
        if wrap:
            atoms.wrap()

        symbols, counts = get_element_counts(atoms)
        system_name = "".join(f"{sym}{count}" for sym, count in zip(symbols, counts))
        magmom = generate_magmom(symbols, counts)

        if dry_run:
            print(
                f"Would create: {dest.name}/  "
                f"({system_name}, {len(atoms)} atoms) from {vasp_file.name}"
            )
            created += 1
            continue

        dest.mkdir(parents=False, exist_ok=False)
        write(str(poscar_path), atoms, format="vasp", vasp5=True, sort=True, direct=False)
        write_incar(dest / "INCAR", system_name, magmom)
        write_kpoints(dest / "KPOINTS")
        shutil.copy2(potcar_template, dest / "POTCAR")
        shutil.copy2(submit_template, dest / "submit.vasp6.sh")
        vasp_file.unlink()

        created += 1
        if created % 50 == 0:
            print(f"  Created {created}/{len(vasp_files)} directories...")

    print(f"\nDone! Created {created} director{'y' if created == 1 else 'ies'} under {source}")
    if skipped:
        print(f"Skipped {skipped} already-converted director{'y' if skipped == 1 else 'ies'}")
    if cleaned:
        print(f"Removed {cleaned} duplicate top-level .vasp file(s)")
    print(f"Top-level calculation directories: "
          f"{sum(1 for p in source.iterdir() if p.is_dir() and (p / 'POSCAR').is_file())}")


def main():
    parser = argparse.ArgumentParser(
        description="Create exactly one VASP directory per top-level .vasp file.",
        epilog=(
            "Each foo.vasp becomes foo/POSCAR. Nested .vasp processing is intentionally "
            "disabled to avoid multiplying directories on repeated runs."
        ),
    )
    parser.add_argument(
        "directory",
        help="Directory containing top-level .vasp files",
    )
    parser.add_argument(
        "--template-dir",
        help="Directory with template POTCAR and submit.vasp6.sh "
             f"(default: {TEMPLATE_DIR})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show planned directories only")
    parser.add_argument("--no-wrap", action="store_true", help="Do not wrap atoms into the cell")
    parser.add_argument("--limit", type=int, help="Process only the first N .vasp files")
    parser.add_argument(
        "--prune-nested-dirs",
        action="store_true",
        help="Remove erroneously nested subdirectories before processing",
    )
    args = parser.parse_args()

    setup_single_point_dft(
        args.directory,
        template_dir=args.template_dir,
        dry_run=args.dry_run,
        wrap=not args.no_wrap,
        limit=args.limit,
        prune_nested=args.prune_nested_dirs,
    )


if __name__ == "__main__":
    main()
