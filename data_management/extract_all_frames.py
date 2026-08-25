#!/usr/bin/env python3
"""
Extract ALL frames from VASP OUTCAR(s) in a directory into a single extended XYZ file.

Uses the same extraction method as extract_frames_for_mlff.py and extract_optimized_frames.py:
ASE reads OUTCAR trajectory (energy, forces, stress, lattice) and writes extended XYZ.

Unlike extract_frames_for_mlff.py (stride/skip/cap) or extract_optimized_frames.py (lowest-energy
only), this script extracts every electronically converged frame from every OUTCAR found under
the given directory.

Usage:
  python extract_all_frames.py <directory> [-o output.xyz]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

try:
    from ase.io import read, write
    import ase.io.vasp_parsers.vasp_outcar_parsers as _vasp_outcar_parsers
except ImportError:
    print("Error: ASE is required. Install with: pip install ase", file=sys.stderr)
    sys.exit(1)

from vasp_step_convergence import filter_images_converged_only


# ============================================================================
# Work around an ASE limitation reading CONTCAR/POSCAR from interface
# structures built by concatenating multiple slabs without merging repeated
# species into contiguous blocks (e.g. "K Mn O K Mn O ..." instead of
# "K Mn O"). VASP wraps the atom-symbols/atom-counts header lines onto
# additional physical lines once there are too many atom-type groups, but
# ASE's POSCAR reader only consumes one physical line per header field, so
# it misparses the wrapped continuation as the next field (e.g. `int('O')`).
#
# When reading an OUTCAR, ASE separately peeks at CONTCAR/POSCAR in the same
# directory purely to recover selective-dynamics constraints
# (read_constraints_from_file); atom order/species for the actual frame
# always come from OUTCAR itself, never from this side file. So the fix
# below only needs to re-join wrapped header lines back into single logical
# lines -- it must NOT reorder atoms, since constraint indices are
# positional and must still line up with OUTCAR's atom order.
# ============================================================================

_orig_read_constraints_from_file = _vasp_outcar_parsers.read_constraints_from_file
_dewrap_rescue_count = 0


def _is_all_int(tokens: list[str]) -> bool:
    if not tokens:
        return False
    try:
        for t in tokens:
            int(t)
        return True
    except ValueError:
        return False


def _dewrap_poscar_lines(lines: list[str]) -> list[str] | None:
    """Re-join wrapped atom-symbols/atom-counts header lines in a
    POSCAR/CONTCAR into single logical lines, without reordering or
    deduplicating anything. Returns None if there's nothing to merge
    (file too short, or no wrapping detected)."""
    if len(lines) < 8:
        return None

    idx = 5  # 0=comment, 1=scale factor, 2-4=lattice vectors
    symbols_start = idx
    while idx < len(lines) and lines[idx].split() and not _is_all_int(lines[idx].split()):
        idx += 1
    symbols_end = idx

    counts_start = idx
    while idx < len(lines) and _is_all_int(lines[idx].split()):
        idx += 1
    counts_end = idx

    n_symbol_lines = symbols_end - symbols_start
    n_count_lines = counts_end - counts_start
    if n_count_lines == 0 or (n_symbol_lines <= 1 and n_count_lines <= 1):
        return None  # nothing wrapped (or not a shape we recognize)

    def _merge(block: list[str]) -> str:
        return " ".join(" ".join(l.split()) for l in block) + "\n"

    new_lines = list(lines[:symbols_start])
    if n_symbol_lines:
        new_lines.append(_merge(lines[symbols_start:symbols_end]))
    new_lines.append(_merge(lines[counts_start:counts_end]))
    new_lines.extend(lines[counts_end:])
    return new_lines


def _read_constraints_with_dewrap(directory):
    """Drop-in replacement for ASE's read_constraints_from_file that falls
    back to a de-wrapped CONTCAR/POSCAR copy when the original raises
    (interleaved-species header wrap). Preserves original behavior
    (including re-raising) for files the de-wrap can't fix, e.g. genuinely
    empty/truncated CONTCARs from crashed jobs."""
    global _dewrap_rescue_count
    try:
        return _orig_read_constraints_from_file(directory)
    except Exception as orig_exc:
        directory = Path(directory)
        for filename in ("CONTCAR", "POSCAR"):
            fpath = directory / filename
            if not fpath.is_file():
                continue
            try:
                with fpath.open("r") as f:
                    lines = f.readlines()
            except OSError:
                continue
            fixed = _dewrap_poscar_lines(lines)
            if fixed is None:
                continue
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".vasp", delete=False
                ) as tmp:
                    tmp.writelines(fixed)
                    tmp_path = tmp.name
                constraint = read(tmp_path, format="vasp").constraints
            except Exception:
                continue
            finally:
                if tmp_path is not None:
                    os.unlink(tmp_path)
            _dewrap_rescue_count += 1
            return constraint
        raise orig_exc


_vasp_outcar_parsers.read_constraints_from_file = _read_constraints_with_dewrap


# ============================================================================
# Logging (tee to stdout + log file) — same style as extract_frames_for_mlff.py
# ============================================================================

class Tee:
    """Write to both stdout and a log file."""
    def __init__(self, log_path, stream=None):
        self._stream = stream if stream is not None else sys.stdout
        self._log_path = log_path
        self._file = open(log_path, 'w', encoding='utf-8')

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()


def find_outcars(root: Path) -> list[Path]:
    """Find all OUTCAR files under root (recursive)."""
    return sorted(root.rglob("OUTCAR"))


def make_run_id(root: Path, outcar_path: Path) -> str:
    """Full directory path from ~ as run_id."""
    abs_dir = str(outcar_path.resolve().parent)
    home = str(Path.home())
    if abs_dir.startswith(home):
        return "~" + abs_dir[len(home):]
    return abs_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ALL frames from VASP OUTCAR(s) into one extended XYZ file."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory to search for OUTCAR files (searches recursively)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output extended XYZ file (default: all_frames_<count>.xyz)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Log file path (default: <output_stem>.log)",
    )
    args = parser.parse_args()

    root = args.directory.resolve()
    if not root.is_dir():
        print(f"Error: Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    outcars = find_outcars(root)
    if not outcars:
        print("Error: No OUTCAR files found.", file=sys.stderr)
        sys.exit(1)

    # Default output: all_frames_<count>.xyz (we'll set count after extraction)
    output_path = args.output
    if output_path is None:
        output_path = Path("all_frames.xyz")  # placeholder, renamed after

    # Log file: default to <output_stem>.log
    if args.log is None:
        args.log = os.path.splitext(str(output_path))[0] + ".log"

    tee = None
    try:
        tee = Tee(args.log)
        sys.stdout = tee

        if not args.quiet:
            print(f"Found {len(outcars)} OUTCAR(s). Extracting converged frames...")

        all_frames = []
        skipped = 0
        for p in outcars:
            try:
                images = read(str(p), index=":")
            except Exception as e:
                print(f"Warning: Could not read {p}: {e}", file=sys.stderr)
                continue
            run_id = make_run_id(root, p)
            filtered, err = filter_images_converged_only(p, images)
            if err is not None:
                print(
                    f"Warning: Skipping {p} ({err})",
                    file=sys.stderr,
                )
                skipped += 1
                continue
            assert filtered is not None
            to_write = filtered
            for atoms in to_write:
                atoms.info["run_id"] = run_id
            all_frames.extend(to_write)
            if not args.quiet:
                if len(to_write) != len(images):
                    print(
                        f"  {p.relative_to(root)}: {len(to_write)}/{len(images)} "
                        f"frame(s) (converged only)"
                    )
                else:
                    print(f"  {p.relative_to(root)}: {len(to_write)} frame(s)")
        if not args.quiet and skipped:
            print(
                f"  Skipped {skipped} OUTCAR(s) (convergence check failed or no data).",
                file=sys.stderr,
            )
        if not args.quiet and _dewrap_rescue_count:
            print(
                f"  Recovered {_dewrap_rescue_count} OUTCAR(s) with wrapped/interleaved "
                f"CONTCAR or POSCAR headers.",
                file=sys.stderr,
            )

        if not all_frames:
            print("Error: No frames could be read from any OUTCAR.", file=sys.stderr)
            sys.exit(1)

        if args.output is None:
            output_path = Path(f"all_frames_{len(all_frames)}_frames.xyz")

        write(str(output_path), all_frames, format="extxyz")

        if not args.quiet:
            print(f"Wrote {len(all_frames)} frame(s) to {output_path}")
    finally:
        if tee is not None:
            sys.stdout = tee._stream
            tee.close()
            print(f"Log written to {args.log}")


if __name__ == "__main__":
    main()
