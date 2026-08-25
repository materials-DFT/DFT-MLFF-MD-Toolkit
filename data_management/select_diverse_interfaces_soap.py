#!/usr/bin/env python3
"""Pick a diverse subset of generated interface structures into one XYZ file.

Recursively scans a directory for generated interface structure files (as
written by 04_compare_interface_searches.py: ``outputs/interfaces/<pair>/*.vasp``,
but any directory tree containing structure files works), groups them by
"interface pair" (the two materials being interfaced, independent of Miller
index), and selects a diverse subset using dscribe's SOAP descriptor and
greedy farthest-point sampling (FPS) -- the same approach as
~/VASP-MACE-workflow/data_management/extract_diverse_frames_soap.py, applied
here to whole built interfaces instead of MD trajectory frames.

Selection policy
-----------------
1. Every interface pair must be represented in the output (hard constraint).
2. Selection targets interfaces with fewer than --max-atoms atoms. For a pair
   whose *every* built interface is at or above that threshold, the pair is
   still included -- via its smallest available interface -- rather than
   dropped.
3. Beyond the one guaranteed structure per pair, the remaining budget (up to
   --size total) is filled by running SOAP+FPS over the pool of
   sub-threshold candidates from every pair, so the bulk of the selection is
   still governed by structural diversity rather than size alone.

Output is a directory of directories: one subdirectory per selected
interface, named
``<materialA>__<materialB>__<miller>__c<candidate-id>__n<atoms>__<role>``
(role is "pair_anchor" or "diversity"; candidate-id is the numeric suffix
from the original build filename), each containing just the selected
structure as a ``POSCAR``. The name alone is enough to trace a pick back to
its source file under the scanned directory, so no separate metadata file is
written.

Usage:

    python select_diverse_interfaces_soap.py /path/to/campaign/outputs \\
        --size 50 --max-atoms 300 -o diverse_interfaces/

Requires: ase, dscribe, numpy
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

STRUCTURE_EXTENSIONS = {".vasp", ".poscar", ".cif", ".xyz", ".extxyz"}
STRUCTURE_BARE_NAMES = {"poscar", "contcar"}
PAIR_SEPARATOR = "__"
MATERIAL_NAME_PREFIX = "KMnO2_"


def strip_material_prefix(name: str) -> str:
    """Drop the common 'KMnO2_' prefix from a material name for display
    purposes (e.g. in directory names). The full name is still kept in
    pair_key/metadata for grouping and traceability."""
    return name[len(MATERIAL_NAME_PREFIX):] if name.startswith(MATERIAL_NAME_PREFIX) else name


@dataclass
class Candidate:
    path: Path
    atoms: Atoms
    n_atoms: int
    pair_key: str
    miller: str | None


def find_structure_files(root: Path) -> list[Path]:
    """Recursively find files that look like atomistic structure files."""
    found = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in STRUCTURE_EXTENSIONS or path.name.lower() in STRUCTURE_BARE_NAMES:
            found.append(path)
    return found


def looks_like_interface(path: Path) -> bool:
    """Heuristic: is this structure file a built *interface* (vs. a bulk
    material or a single-sided surface slab also commonly found alongside
    interface outputs)?

    True if any path component mentions "interface", or if the parent
    directory / filename encodes the ``materialA__materialB__miller`` naming
    convention written by 04_compare_interface_searches.py.
    """
    if any("interface" in part.lower() for part in path.parts):
        return True
    if parse_pair_miller(path.parent.name) is not None:
        return True
    if parse_pair_miller(path.stem) is not None:
        return True
    return False


def parse_pair_miller(token: str) -> tuple[str, str, str] | None:
    """Parse a ``materialA__materialB__miller[...]`` token.

    Matches the naming convention from _common.pair_search_name(): fields are
    joined with a literal double underscore, and the Miller field is exactly
    three '0'/'1' characters (extra suffixes, e.g. a trailing candidate id,
    are ignored).
    """
    parts = token.split(PAIR_SEPARATOR)
    if len(parts) < 3:
        return None
    material_a, material_b, miller_field = parts[0], parts[1], parts[2]
    miller = miller_field[:3]
    if len(miller) != 3 or any(c not in "01" for c in miller):
        return None
    if not material_a or not material_b:
        return None
    return material_a, material_b, miller


def pair_key_for(path: Path) -> tuple[str, str | None]:
    """Return (pair_key, miller) for a structure file.

    pair_key aggregates a material pair across Miller orientations (order
    independent). Falls back to the immediate parent directory name when the
    naming convention isn't recognized, so arbitrarily-organized interface
    trees still get a (coarser) grouping instead of crashing.
    """
    parsed = parse_pair_miller(path.parent.name) or parse_pair_miller(path.stem)
    if parsed is not None:
        material_a, material_b, miller = parsed
        pair_key = PAIR_SEPARATOR.join(sorted((material_a, material_b)))
        return pair_key, miller
    return path.parent.name, None


def load_candidates(paths: list[Path], *, verbose: bool) -> list[Candidate]:
    candidates = []
    for path in paths:
        try:
            atoms = read(path, index=0)
        except Exception as exc:  # noqa: BLE001 - best-effort structure loading
            warnings.warn(f"Skipping unreadable structure file {path}: {exc}")
            continue
        pair_key, miller = pair_key_for(path)
        candidates.append(
            Candidate(path=path, atoms=atoms, n_atoms=len(atoms), pair_key=pair_key, miller=miller)
        )
        if verbose:
            print(f"  loaded {path} ({len(atoms)} atoms, pair={pair_key}, miller={miller})")
    return candidates


def group_by_pair(candidates: list[Candidate]) -> dict[str, list[Candidate]]:
    groups: dict[str, list[Candidate]] = {}
    for c in candidates:
        groups.setdefault(c.pair_key, []).append(c)
    return groups


def select_coverage_anchors(
    groups: dict[str, list[Candidate]], *, max_atoms: int
) -> dict[str, Candidate]:
    """Pick exactly one guaranteed representative per pair.

    Prefers the smallest sub-threshold candidate; if a pair has none under
    max_atoms, falls back to its smallest candidate overall so the pair is
    never dropped.
    """
    anchors = {}
    for pair_key, members in groups.items():
        eligible = [m for m in members if m.n_atoms < max_atoms]
        pool = eligible if eligible else members
        anchors[pair_key] = min(pool, key=lambda m: (m.n_atoms, str(m.path)))
    return anchors


def warn_if_cutoff_too_large(candidates: list[Candidate], r_cut: float) -> None:
    """Mirror extract_diverse_frames_soap.py's cell-size sanity check: SOAP's
    cutoff sphere double-counts periodic images when 2*r_cut exceeds the
    smallest cell length."""
    short = []
    for c in candidates:
        try:
            min_len = float(min(c.atoms.cell.lengths()))
        except Exception:
            continue
        if min_len > 0 and 2 * r_cut > min_len:
            short.append((c.path, min_len))
    if short:
        print(
            f"Warning: 2*r_cut ({2 * r_cut:.2f} A) exceeds the smallest cell length "
            f"for {len(short)} interface(s); SOAP cutoff sphere may double-count "
            "periodic images there:"
        )
        for path, min_len in short[:10]:
            print(f"    {path}: smallest cell length {min_len:.2f} A")
        if len(short) > 10:
            print(f"    ... and {len(short) - 10} more")


def farthest_point_sample(X: np.ndarray, size: int) -> list[int]:
    """Greedy farthest-point sampling in Euclidean (SOAP) space.

    Starts from the point farthest from the dataset centroid (deterministic,
    no seed needed), then repeatedly adds whichever remaining point is
    farthest from the nearest already-selected point. Copied from
    ~/VASP-MACE-workflow/data_management/extract_diverse_frames_soap.py.
    """
    centroid = X.mean(axis=0)
    start = int(np.argmax(np.linalg.norm(X - centroid, axis=1)))
    selected = [start]
    min_dist = np.linalg.norm(X - X[start], axis=1)
    for _ in range(size - 1):
        min_dist[selected[-1]] = -np.inf
        nxt = int(np.argmax(min_dist))
        min_dist = np.minimum(min_dist, np.linalg.norm(X - X[nxt], axis=1))
        selected.append(nxt)
    return selected


def diverse_fill(
    pool: list[Candidate],
    budget: int,
    *,
    r_cut: float,
    n_max: int,
    l_max: int,
    sigma: float,
    n_jobs: int,
) -> list[Candidate]:
    if budget <= 0 or not pool:
        return []

    if budget >= len(pool):
        return list(pool)

    from dscribe.descriptors import SOAP

    species = sorted({sym for c in pool for sym in c.atoms.get_chemical_symbols()})
    soap = SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        periodic=True,
        average="outer",
    )
    X = np.atleast_2d(soap.create([c.atoms for c in pool], n_jobs=n_jobs))
    selected_idx = farthest_point_sample(X, budget)
    return [pool[i] for i in selected_idx]


def candidate_id(path: Path) -> str:
    """Numeric suffix from a build filename, e.g. '..._0001.vasp' -> '0001'."""
    m = re.search(r"_(\d+)$", path.stem)
    return m.group(1) if m else "0"


def write_selection(selected: list[tuple[Candidate, str]], *, output_dir: Path) -> list[Path]:
    """Write each selected interface into its own subdirectory of output_dir.

    Each subdirectory is named after the two phases (materials), the Miller
    index, the source file's candidate id, atom count, and selection role, so
    a pick is fully identifiable -- and traceable back to its source file --
    from the directory name alone. The original structure file is copied in
    verbatim (avoids any read/write round-trip through ASE) as POSCAR.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    written: list[Path] = []

    for candidate, role in selected:
        miller = candidate.miller if candidate.miller is not None else "na"
        pair_display = PAIR_SEPARATOR.join(
            strip_material_prefix(p) for p in candidate.pair_key.split(PAIR_SEPARATOR)
        )
        cand_id = candidate_id(candidate.path)
        base_name = f"{pair_display}__{miller}__c{cand_id}__n{candidate.n_atoms}__{role}"
        name = base_name
        suffix = 2
        while name in used_names:
            name = f"{base_name}__{suffix}"
            suffix += 1
        used_names.add(name)

        dest_dir = output_dir / name
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate.path, dest_dir / "POSCAR")
        written.append(dest_dir)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="Directory to search recursively for generated interfaces")
    parser.add_argument("-n", "--size", type=int, default=50, help="Number of interfaces to select (default: 50)")
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=300,
        help="Preferred atom-count ceiling; pairs with nothing under this are still included (default: 300)",
    )
    parser.add_argument("--r-cut", type=float, default=5.0, help="SOAP cutoff radius in Angstrom (default: 5.0)")
    parser.add_argument("--n-max", type=int, default=8, help="SOAP n_max (default: 8)")
    parser.add_argument("--l-max", type=int, default=6, help="SOAP l_max (default: 6)")
    parser.add_argument("--sigma", type=float, default=0.5, help="SOAP Gaussian width sigma (default: 0.5)")
    parser.add_argument("-j", "--n-jobs", type=int, default=1, help="Parallel workers for SOAP descriptor computation (default: 1)")
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Treat every structure file found as a candidate interface, skipping the interface-vs-surface heuristic filter",
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output directory (default: <directory>/diverse_interfaces)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print each loaded structure file")
    args = parser.parse_args()

    root = args.directory.resolve()
    if not root.is_dir():
        parser.error(f"{root} is not a directory")

    output_dir = args.output if args.output is not None else root / "diverse_interfaces"
    output_dir_resolved = output_dir.resolve()

    print(f"Scanning {root} for structure files...")
    all_files = [
        p for p in find_structure_files(root)
        if output_dir_resolved not in p.resolve().parents and p.resolve() != output_dir_resolved
    ]
    print(f"Found {len(all_files)} candidate structure files")

    if args.include_all:
        interface_files = all_files
    else:
        interface_files = [p for p in all_files if looks_like_interface(p)]
        skipped = len(all_files) - len(interface_files)
        if skipped:
            print(f"Filtered out {skipped} files that don't look like interfaces (use --include-all to keep them)")

    if not interface_files:
        sys.exit("No interface structure files found. Nothing to select.")

    candidates = load_candidates(interface_files, verbose=args.verbose)
    if not candidates:
        sys.exit("No structure files could be read. Nothing to select.")

    groups = group_by_pair(candidates)
    print(f"Loaded {len(candidates)} interfaces spanning {len(groups)} interface pairs")
    print(f"SOAP: r_cut={args.r_cut}, n_max={args.n_max}, l_max={args.l_max}, sigma={args.sigma}")
    warn_if_cutoff_too_large(candidates, args.r_cut)

    anchors = select_coverage_anchors(groups, max_atoms=args.max_atoms)
    n_oversized_anchors = sum(1 for a in anchors.values() if a.n_atoms >= args.max_atoms)
    if n_oversized_anchors:
        print(
            f"{n_oversized_anchors} pair(s) have no interface under {args.max_atoms} atoms; "
            "including their smallest available interface anyway."
        )

    if len(anchors) > args.size:
        warnings.warn(
            f"{len(anchors)} interface pairs must all be represented, which exceeds the "
            f"requested size of {args.size}. Output will contain {len(anchors)} structures "
            "(one per pair) instead."
        )

    anchor_paths = {a.path for a in anchors.values()}
    remaining_budget = max(0, args.size - len(anchors))
    stage2_pool = [
        c
        for members in groups.values()
        for c in members
        if c.n_atoms < args.max_atoms and c.path not in anchor_paths
    ]

    print(f"Filling {remaining_budget} additional diverse slot(s) via SOAP+FPS from {len(stage2_pool)} sub-{args.max_atoms}-atom candidates...")
    diverse_extra = diverse_fill(
        stage2_pool,
        remaining_budget,
        r_cut=args.r_cut,
        n_max=args.n_max,
        l_max=args.l_max,
        sigma=args.sigma,
        n_jobs=args.n_jobs,
    )

    selected: list[tuple[Candidate, str]] = [(a, "pair_anchor") for a in anchors.values()]
    selected += [(c, "diversity") for c in diverse_extra]
    selected.sort(key=lambda pair: (pair[0].pair_key, pair[0].n_atoms))

    if len(selected) < args.size and len(stage2_pool) + len(anchors) < args.size:
        warnings.warn(
            f"Only {len(selected)} interfaces were available in total; "
            f"requested size was {args.size}."
        )

    written_dirs = write_selection(selected, output_dir=output_dir)

    print(f"\nSelected {len(selected)} interfaces ({len(anchors)} pair-coverage anchors + {len(diverse_extra)} diversity picks)")
    print(f"Pairs represented: {len(groups)}/{len(groups)}")
    sizes = [c.n_atoms for c, _ in selected]
    print(f"Atom counts: min={min(sizes)}, max={max(sizes)}, mean={sum(sizes)/len(sizes):.1f}")
    print(f"Wrote {len(written_dirs)} interface directories under {output_dir}")


if __name__ == "__main__":
    main()
