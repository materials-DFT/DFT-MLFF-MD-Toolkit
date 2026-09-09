#!/usr/bin/env python3
"""
Recursively prepare VASP calculation directories for ionic optimization.

Walks a directory tree looking for finished-calculation directories (INCAR +
POSCAR + POTCAR) and rewrites INCAR for a fresh SCAN ionic relaxation
(IBRION=2, ISIF=3), with MAGMOM built from the actual per-species atom
counts in POSCAR. POSCAR itself is left untouched.

Matches the settings used in optimization/unitcells/*/INCAR:
    PREC = Accurate, ENCUT = 520, METAGGA = SCAN, ISPIN = 2, ALGO = All,
    IBRION = 2, ISIF = 3, EDIFFG = -5E-02, POTIM = 0.5, ISYM = 0

Usage:
    python3 prepare_for_ionic_optimization.py <directory_path> [options]

Examples:
    # Dry run over a batch of supercell temperature directories
    python3 prepare_for_ionic_optimization.py /path/to/supercells --dry-run

    # Apply, and also delete heavy MD-stage output files (OUTCAR, CHG, ...)
    python3 prepare_for_ionic_optimization.py /path/to/supercells --clean

    # Different magnetic species/moment
    python3 prepare_for_ionic_optimization.py . --magmom "Mn:3.0,Fe:4.0"
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

DEFAULT_MAGMOM_MAP = {"Mn": 3.0}

DEFAULT_DELETE = [
    "CHG", "CHGCAR", "WAVECAR", "OUTCAR", "vasprun.xml", "PROCAR",
    "REPORT", "PCDAT", "ICONST", "HILLSPOT", "DOSCAR", "EIGENVAL",
    "IBZKPT", "XDATCAR", "OSZICAR", "job.out", "job.err", "MegaSAS.log",
]
DEFAULT_DELETE_GLOBS = ["*.btr"]

KPOINTS_CONTENT = """Automatic mesh
0
Monkhorst-Pack
1 1 1
0 0 0
"""


def parse_magmom_map(spec: str) -> Dict[str, float]:
    """Parse '--magmom Mn:3.0,Fe:4.0' into {'Mn': 3.0, 'Fe': 4.0}."""
    mapping = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        symbol, _, value = entry.partition(":")
        mapping[symbol.strip()] = float(value.strip())
    return mapping


def read_poscar_species(poscar_path: str) -> Tuple[List[str], List[int]]:
    """Read the element-symbol and element-count lines of a VASP5+ POSCAR."""
    with open(poscar_path, "r") as f:
        lines = f.readlines()

    symbols = lines[5].split()
    counts_str = lines[6].split()
    if not symbols or not counts_str or len(symbols) != len(counts_str) or not all(
        c.isdigit() for c in counts_str
    ):
        raise ValueError(f"Could not parse species/counts from {poscar_path}")

    counts = [int(c) for c in counts_str]
    return symbols, counts


def build_magmom(symbols: List[str], counts: List[int], magmom_map: Dict[str, float]) -> str:
    parts = []
    for symbol, count in zip(symbols, counts):
        value = magmom_map.get(symbol, 0.0)
        parts.append(f"{count}*{value:g}")
    return " ".join(parts)


def build_formula(symbols: List[str], counts: List[int]) -> str:
    return "".join(f"{symbol}{count}" for symbol, count in zip(symbols, counts))


def is_calc_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    try:
        names = set(os.listdir(path))
    except PermissionError:
        return False
    has_incar = "INCAR" in names
    has_structure = "POSCAR" in names
    has_potcar = "POTCAR" in names
    return has_incar and has_structure and has_potcar


def find_calc_dirs(root: str) -> List[str]:
    if is_calc_dir(root):
        return [root]
    found = []
    for dirpath, _dirnames, _filenames in os.walk(root):
        if is_calc_dir(dirpath):
            found.append(dirpath)
    return sorted(found)


def write_incar(incar_path: str, formula: str, magmom: str, args: argparse.Namespace):
    lines = []
    lines.append(f"System = {formula} ionic optimization\n\n")

    lines.append("Starting parameters:\n")
    lines.append("ISTART = 0\n")
    lines.append("ICHARG = 2\n\n")

    lines.append("Electronic Relaxation:\n")
    lines.append(f"PREC = {args.prec}\n")
    lines.append(f"ENCUT = {args.encut}\n")
    lines.append(f"NELMIN = {args.nelmin}\n")
    lines.append(f"NELM = {args.nelm}\n")
    lines.append(f"EDIFF = {args.ediff}\n")
    lines.append(f"LREAL = {args.lreal}\n")
    if not args.no_spin:
        lines.append("ISPIN = 2\n")
        lines.append(f"MAGMOM = {magmom}\n")
    lines.append(f"ALGO = {args.algo}\n\n")

    if not args.no_metagga:
        lines.append(f"METAGGA = {args.metagga}\n")
        lines.append("LMIXTAU = .TRUE.\n")
        lines.append("LASPH = .TRUE.\n")
        lines.append("LDIAG = .TRUE.\n\n")

    lines.append("Ionic relaxation:\n")
    lines.append(f"NSW = {args.nsw}\n")
    lines.append(f"IBRION = {args.ibrion}\n")
    lines.append(f"EDIFFG = {args.ediffg}\n")
    lines.append(f"ISIF = {args.isif}\n")
    lines.append(f"POTIM = {args.potim}\n")
    lines.append(f"ISYM = {args.isym}\n\n")

    lines.append("DOS related values:\n")
    lines.append(f"LORBIT = {args.lorbit}\n")
    lines.append(f"ISMEAR = {args.ismear}\n")
    lines.append(f"SIGMA = {args.sigma}\n\n")

    lines.append("Parallelization flags:\n")
    lines.append(f"NCORE = {args.ncore}\n")
    lines.append(f"NSIM = {args.nsim}\n")
    lines.append("LPLANE = .TRUE.\n")
    lines.append("LSCALU = .FALSE.\n")
    lines.append(f"KPAR = {args.kpar}\n")
    lines.append(f"NPAR = {args.npar}\n")

    with open(incar_path, "w") as f:
        f.writelines(lines)


def process_dir(path: str, args: argparse.Namespace) -> bool:
    poscar_path = os.path.join(path, "POSCAR")
    incar_path = os.path.join(path, "INCAR")
    kpoints_path = os.path.join(path, "KPOINTS")

    if not os.path.isfile(poscar_path):
        print(f"[skip] {path}: no POSCAR")
        return False

    try:
        symbols, counts = read_poscar_species(poscar_path)
    except ValueError as e:
        print(f"[skip] {path}: {e}")
        return False

    formula = build_formula(symbols, counts)
    magmom = build_magmom(symbols, counts, args.magmom_map)

    print(f"[{'dry-run' if args.dry_run else 'update'}] {path}")
    print(f"  Structure: {formula}")
    print(f"  MAGMOM: {magmom}")

    if not args.dry_run:
        write_incar(incar_path, formula, magmom, args)

    if not os.path.isfile(kpoints_path):
        print(f"  Writing default KPOINTS (1 1 1 Monkhorst-Pack)")
        if not args.dry_run:
            with open(kpoints_path, "w") as f:
                f.write(KPOINTS_CONTENT)

    if args.clean:
        import glob

        to_remove = list(DEFAULT_DELETE)
        for pattern in DEFAULT_DELETE_GLOBS:
            to_remove.extend(
                os.path.basename(p) for p in glob.glob(os.path.join(path, pattern))
            )
        for name in to_remove:
            fpath = os.path.join(path, name)
            if os.path.exists(fpath):
                if args.dry_run:
                    print(f"  Would remove {name}")
                else:
                    try:
                        os.remove(fpath)
                    except OSError as e:
                        print(f"  Could not remove {fpath}: {e}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare directories recursively for VASP ionic optimization (SCAN, IBRION=2, ISIF=3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 prepare_for_ionic_optimization.py /path/to/supercells --dry-run
  python3 prepare_for_ionic_optimization.py /path/to/supercells --clean
  python3 prepare_for_ionic_optimization.py . --magmom "Mn:3.0,Fe:4.0"
        """,
    )
    parser.add_argument("paths", nargs="*", default=["."], help="Directories to process (searched recursively).")

    parser.add_argument("--prec", default="Accurate")
    parser.add_argument("--encut", type=int, default=520)
    parser.add_argument("--nelmin", type=int, default=6)
    parser.add_argument("--nelm", type=int, default=1000)
    parser.add_argument("--ediff", default="1E-6")
    parser.add_argument("--lreal", default=".FALSE.")
    parser.add_argument("--algo", default="All")
    parser.add_argument("--no-spin", action="store_true", help="Omit ISPIN/MAGMOM (non-magnetic system).")
    parser.add_argument("--magmom", dest="magmom_spec", default="Mn:3.0",
                        help="Comma-separated symbol:moment pairs, e.g. 'Mn:3.0,Fe:4.0'. Unlisted species get 0.0.")
    parser.add_argument("--metagga", default="SCAN")
    parser.add_argument("--no-metagga", action="store_true", help="Skip METAGGA/LMIXTAU/LASPH/LDIAG (plain GGA).")

    parser.add_argument("--nsw", type=int, default=1000)
    parser.add_argument("--ibrion", type=int, default=2)
    parser.add_argument("--ediffg", default="-5E-02")
    parser.add_argument("--isif", type=int, default=3)
    parser.add_argument("--potim", default="0.5")
    parser.add_argument("--isym", type=int, default=0)

    parser.add_argument("--lorbit", type=int, default=10)
    parser.add_argument("--ismear", type=int, default=0)
    parser.add_argument("--sigma", default="0.05")

    parser.add_argument("--ncore", type=int, default=4)
    parser.add_argument("--npar", type=int, default=4)
    parser.add_argument("--kpar", type=int, default=1)
    parser.add_argument("--nsim", type=int, default=4)

    parser.add_argument("--clean", action="store_true",
                        help="Delete heavy leftover output files (OUTCAR, CHG*, vasprun.xml, WAVECAR, ...).")
    parser.add_argument("--dry-run", action="store_true", help="Show planned changes without modifying anything.")

    args = parser.parse_args()
    args.magmom_map = dict(DEFAULT_MAGMOM_MAP)
    args.magmom_map.update(parse_magmom_map(args.magmom_spec))

    all_dirs = []
    for path in args.paths:
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            print(f"[skip] '{path}': not a directory.", file=sys.stderr)
            continue
        all_dirs.extend(find_calc_dirs(path))

    all_dirs = sorted(set(all_dirs))
    if not all_dirs:
        print("No calculation directories found (need INCAR + POSCAR/CONTCAR + POTCAR).")
        sys.exit(1)

    processed = 0
    for d in all_dirs:
        if process_dir(d, args):
            processed += 1

    print(f"\n{'Would process' if args.dry_run else 'Processed'} {processed}/{len(all_dirs)} directories.")


if __name__ == "__main__":
    main()
