#!/usr/bin/env python3
# The following SBATCH directives are optional. They set job parameters when
# submitting this script directly with sbatch. Route Slurm stdout/stderr to
# /dev/null so only the internal log (e.g. md_frames_soap.log) and xyz file
# are written in the working directory.
#
# cpus-per-task is set to 16 to match -j/--n-jobs 16 below; if you pass a
# different -j on the sbatch command line, update --cpus-per-task to match.
#SBATCH --job-name=extract_soap
#SBATCH --partition=cpucluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
"""
Sample maximally diverse MD frames across one or more directories of VASP
OUTCARs, using SOAP descriptors and greedy farthest-point sampling (FPS),
and write them to a single extended XYZ file.

Unlike extract_frames_from_md.py (which always takes exactly 2 frames per
OUTCAR, at 1/3 and 2/3 of the trajectory), this script pools every candidate
frame from every discovered OUTCAR, computes a whole-frame SOAP descriptor
for each (dscribe's atom-averaged SOAP), and runs FPS *globally* over that
pool with a single total frame budget (--n-frames). This means:

  - Low-temperature runs, whose frames barely move step to step, collapse
    into one or a few representative picks instead of wasting budget on
    near-duplicates.
  - High-temperature runs, which explore more configuration space, keep
    contributing frames for as long as they remain the most diverse
    remaining candidates.

There is no per-run quota: a trajectory that is fully redundant with frames
already selected from elsewhere can legitimately contribute zero frames.

By default, only electronically converged MD steps are considered as
candidates: per-step NELM (OSZICAR/INCAR) and OUTCAR chunk failure messages
(see --no-convergence-check to disable).

Reading each OUTCAR (ASE's OUTCAR parser) dominates runtime, not the SOAP
computation itself. Two things keep this efficient without changing results:
each OUTCAR is read only once (candidate Atoms are cached and reused for the
final write instead of being re-read), and -j/--n-jobs processes OUTCARs in
parallel worker processes, since each OUTCAR's read+SOAP pipeline is fully
independent. Candidates are sorted into a canonical order before farthest-
point sampling, so output is identical regardless of --n-jobs.

Requires: ase, dscribe, numpy
Run interactively:  python extract_diverse_frames_soap.py .
Run on cluster:     sbatch extract_diverse_frames_soap.py . -j 16
(Script is already executable; the #SBATCH header above requests the
cpucluster partition, 16 cores, 32G, and an 18-hour walltime. Pass a
different -j on the command line only if you also change --cpus-per-task.)

Usage:
  python extract_diverse_frames_soap.py [paths ...] [-n N_FRAMES]
      [--out frames.xyz] [--log ...] [--no-convergence-check]
      [--stride 1] [--r-cut 5.0] [--n-max 8] [--l-max 6] [--sigma 0.5]
      [-j N_JOBS]
  (-n/--n-frames defaults to 600 if omitted)
  (paths defaults to . if omitted; multiple directories are pooled together)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase.io import iread, write


def _load_per_step_convergence():
    """
    Import helper from local module, including sbatch spool fallback paths.

    Under `sbatch script.py`, Slurm executes a copied script from /var/spool,
    so sibling imports are not discoverable unless we probe explicit locations.
    """
    try:
        from vasp_step_convergence import per_step_electronically_converged as func
        return func
    except ModuleNotFoundError:
        candidates = [
            Path.cwd() / "vasp_step_convergence.py",
            Path(__file__).resolve().parent / "vasp_step_convergence.py",
            Path(os.environ.get("SLURM_SUBMIT_DIR", "")) / "vasp_step_convergence.py",
            Path.home() / "VASP-MACE-workflow" / "data_management" / "vasp_step_convergence.py",
        ]
        for module_path in candidates:
            if not module_path.is_file():
                continue
            spec = importlib.util.spec_from_file_location(
                "vasp_step_convergence", module_path
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "per_step_electronically_converged"):
                return module.per_step_electronically_converged
        raise


per_step_electronically_converged = _load_per_step_convergence()

# =============================================================================
# Logging (tee to stdout + log file)
# =============================================================================

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


# =============================================================================
# Discovery
# =============================================================================

def find_outcars(root: Path):
    """Yield (outcar_path, run_id) for each OUTCAR under root."""
    home = str(Path.home())
    for outcar in sorted(root.rglob("OUTCAR")):
        abs_dir = str(outcar.resolve().parent)
        if abs_dir.startswith(home):
            run_id = "~" + abs_dir[len(home):]
        else:
            run_id = abs_dir
        yield outcar, run_id


def discover_outcars(paths: list[Path]):
    """Pool (outcar, run_id) pairs across one or more root directories, deduped."""
    seen = set()
    results = []
    for root in paths:
        root = root.resolve()
        if not root.is_dir():
            print(f"Warning: not a directory, skipping: {root}")
            continue
        for outcar, run_id in find_outcars(root):
            resolved = outcar.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            results.append((outcar, run_id))
    return results


# =============================================================================
# Frame counting / streaming reads
# =============================================================================

def count_frames(outcar: Path) -> int:
    """Count frames in an OUTCAR trajectory without storing them all."""
    n = 0
    for _ in iread(outcar, index=":"):
        n += 1
    return n


def read_frames_at(outcar: Path, indices) -> dict:
    """Stream-read only the requested frame indices from an OUTCAR."""
    want = set(indices)
    got = {}
    if not want:
        return got
    for k, a in enumerate(iread(outcar, index=":")):
        if k in want:
            got[k] = a
            if len(got) == len(want):
                break
    return got


def candidate_indices(n_frames: int, stride: int, ok_list) -> list[int]:
    """Frame indices eligible as SOAP candidates: every `stride`-th frame,
    filtered to electronically converged steps if ok_list is given."""
    raw = range(0, n_frames, max(1, stride))
    if ok_list is None:
        return list(raw)
    return [i for i in raw if i < len(ok_list) and ok_list[i]]


# =============================================================================
# Farthest-point sampling
# =============================================================================

def farthest_point_sample(X: np.ndarray, size: int) -> list[int]:
    """Greedy farthest-point sampling in Euclidean space.

    Starts from the point farthest from the dataset centroid (deterministic,
    no seed needed), then repeatedly adds whichever remaining point is
    farthest from the nearest already-selected point.
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


# =============================================================================
# Per-OUTCAR worker (usable directly or via a process pool)
# =============================================================================

_WORKER_SOAP_CACHE = {}


def _get_worker_soap(species: tuple, r_cut: float, n_max: int, l_max: int, sigma: float):
    """Build (and cache, per-process) the SOAP calculator.

    Each process pool worker constructs its own instance on first use rather
    than receiving one via IPC, since SOAP objects aren't guaranteed picklable.
    """
    key = (species, r_cut, n_max, l_max, sigma)
    if key not in _WORKER_SOAP_CACHE:
        from dscribe.descriptors import SOAP
        _WORKER_SOAP_CACHE.clear()
        _WORKER_SOAP_CACHE[key] = SOAP(
            species=list(species),
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            sigma=sigma,
            periodic=True,
            average="outer",
        )
    return _WORKER_SOAP_CACHE[key]


def _process_one_outcar(
    outcar: Path,
    run_id: str,
    check_convergence: bool,
    stride: int,
    species: tuple,
    r_cut: float,
    n_max: int,
    l_max: int,
    sigma: float,
):
    """Read candidate frames from one OUTCAR and compute their SOAP descriptors.

    Runs standalone (no shared state needed beyond arguments), so it can be
    called directly for n_jobs=1 or submitted to a ProcessPoolExecutor for
    n_jobs>1 without any change in behavior. Returns the Atoms objects too
    (not just their descriptors) so the caller can write the final output
    without a second read pass over the OUTCAR.
    """
    if check_convergence:
        sim = outcar.parent
        ok_list = per_step_electronically_converged(outcar, sim / "INCAR", sim / "OSZICAR")
        if ok_list is None:
            return outcar, run_id, "could not parse per-step convergence (OUTCAR chunk read failed)", 0, []
        n_frames_in_run = len(ok_list)
    else:
        ok_list = None
        try:
            n_frames_in_run = count_frames(outcar)
        except Exception as e:
            return outcar, run_id, str(e), 0, []

    cand_idxs = candidate_indices(n_frames_in_run, stride, ok_list)
    if not cand_idxs:
        return outcar, run_id, "no candidate frames", n_frames_in_run, []

    frames_dict = read_frames_at(outcar, cand_idxs)
    if not frames_dict:
        return outcar, run_id, "could not read candidate frames", n_frames_in_run, []

    ordered_idxs = sorted(frames_dict.keys())
    atoms_list = [frames_dict[i] for i in ordered_idxs]
    soap = _get_worker_soap(species, r_cut, n_max, l_max, sigma)
    descriptors = np.atleast_2d(soap.create(atoms_list))

    results = [
        (frame_index, descriptors[i], atoms_list[i]) for i, frame_index in enumerate(ordered_idxs)
    ]
    return outcar, run_id, None, n_frames_in_run, results


# =============================================================================
# Main extraction pipeline
# =============================================================================

def run_extraction(
    paths: list[Path],
    out_path: Path,
    n_frames: int,
    check_convergence: bool = True,
    stride: int = 1,
    r_cut: float = 5.0,
    n_max: int = 8,
    l_max: int = 6,
    sigma: float = 0.5,
    n_jobs: int = 1,
    progress_interval: int = 30,
):
    print("=" * 80)
    print("SOAP-diverse MD frame sampling (global farthest-point sampling)")
    print("=" * 80)
    print(f"Search directories: {', '.join(str(p) for p in paths)}")
    print(f"Output: {out_path} (extended XYZ: energy, forces, stress)")
    print(f"Requested frame budget: {n_frames}")
    print(f"SOAP: r_cut={r_cut}, n_max={n_max}, l_max={l_max}, sigma={sigma}")
    print(f"Parallel workers: {n_jobs}")
    if check_convergence:
        print("Convergence filter: ON (only electronically converged steps are candidates)")
    else:
        print("Convergence filter: OFF")
    print()

    outcars_list = discover_outcars(paths)
    if not outcars_list:
        raise SystemExit("No OUTCAR files found under the given path(s).")
    print(f"Found {len(outcars_list)} OUTCAR(s)")
    print()

    # ---- Pass 1: global species scan + cutoff/cell sanity check -----------
    print("Scanning first frame of each OUTCAR for species and cell size...")
    global_species = set()
    valid_outcars = []
    short_cell_runs = []
    for outcar, run_id in outcars_list:
        try:
            atoms0 = next(iread(outcar, index="0"))
        except Exception as e:
            print(f"  Skip {outcar}: could not read first frame ({e})")
            continue
        global_species.update(atoms0.get_chemical_symbols())
        valid_outcars.append((outcar, run_id))
        try:
            min_len = float(min(atoms0.cell.lengths()))
            if min_len > 0 and 2 * r_cut > min_len:
                short_cell_runs.append((run_id, min_len))
        except Exception:
            pass

    if not valid_outcars:
        raise SystemExit("No OUTCAR could be read; nothing to sample from.")
    if short_cell_runs:
        print(
            f"  Warning: 2*r_cut ({2 * r_cut:.2f} A) exceeds the smallest cell "
            f"length for {len(short_cell_runs)} run(s); SOAP cutoff sphere may "
            "double-count periodic images there:"
        )
        for run_id, min_len in short_cell_runs[:10]:
            print(f"    {run_id}: smallest cell length {min_len:.2f} A")
        if len(short_cell_runs) > 10:
            print(f"    ... and {len(short_cell_runs) - 10} more")
    print(f"  Global species: {sorted(global_species)}")
    print()

    species_key = tuple(sorted(global_species))

    # ---- Pass 2: candidate descriptor computation --------------------------
    # Atoms objects are kept (not discarded) so the final write pass can reuse
    # them directly instead of re-reading each OUTCAR a second time. Memory
    # cost is proportional to (atoms/frame x total candidates); for typical
    # per-atom systems this is far smaller than the time saved by skipping a
    # second full OUTCAR scan.
    print("Computing SOAP descriptors for candidate frames...")
    candidates = []  # list of dicts: outcar, run_id, frame_index, vector
    atoms_cache = {}  # (outcar, frame_index) -> Atoms
    candidates_per_run = {}
    skipped = 0
    done = 0
    n_tasks = len(valid_outcars)
    t_start = time.time()

    # OUTCAR file size is a much better predictor of read time than a flat
    # per-file count (larger/longer trajectories take proportionally longer
    # to read), so weight progress and ETA by bytes rather than file count.
    outcar_size = {outcar: outcar.stat().st_size for outcar, _ in valid_outcars}
    total_bytes = sum(outcar_size.values())
    bytes_done = 0

    def _progress_fraction():
        if total_bytes > 0:
            return bytes_done / total_bytes
        return done / n_tasks if n_tasks else 0.0

    def _eta_seconds():
        elapsed = time.time() - t_start
        frac = _progress_fraction()
        return (elapsed / frac - elapsed) if frac > 0 else float("nan")

    def _report(outcar, run_id, err, n_frames_in_run, results):
        nonlocal skipped, done, bytes_done
        done += 1
        bytes_done += outcar_size.get(outcar, 0)
        progress = f"[{done}/{n_tasks}, ~{_progress_fraction() * 100:.0f}% by size, ETA {_eta_seconds() / 60:.1f} min]"
        if err is not None:
            print(f"{progress} Skip {outcar}: {err}")
            skipped += 1
            return
        for frame_index, vector, atoms in results:
            candidates.append(
                {"outcar": outcar, "run_id": run_id, "frame_index": frame_index, "vector": vector}
            )
            atoms_cache[(outcar, frame_index)] = atoms
        candidates_per_run[run_id] = len(results)
        print(f"{progress} {outcar}  ({n_frames_in_run} frames -> {len(results)} candidates)")

    stop_heartbeat = threading.Event()

    def _heartbeat():
        while not stop_heartbeat.wait(progress_interval):
            elapsed = time.time() - t_start
            print(
                f"  [heartbeat] {elapsed / 60:.1f} min elapsed, {done}/{n_tasks} OUTCARs done "
                f"(~{_progress_fraction() * 100:.0f}% by size), ETA {_eta_seconds() / 60:.1f} min",
                flush=True,
            )

    heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
        if n_jobs <= 1:
            for outcar, run_id in valid_outcars:
                result = _process_one_outcar(
                    outcar, run_id, check_convergence, stride, species_key, r_cut, n_max, l_max, sigma
                )
                _report(*result)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as ex:
                futures = [
                    ex.submit(
                        _process_one_outcar,
                        outcar, run_id, check_convergence, stride, species_key, r_cut, n_max, l_max, sigma,
                    )
                    for outcar, run_id in valid_outcars
                ]
                for fut in concurrent.futures.as_completed(futures):
                    _report(*fut.result())
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=1)

    if not candidates:
        raise SystemExit("No candidate frames collected; nothing to sample from.")
    print()
    print(f"Total candidate frames across all trajectories: {len(candidates)}")
    print()

    # Sort into a canonical order before FPS so the result is independent of
    # task-completion order (relevant when n_jobs > 1, where OUTCARs finish
    # in a nondeterministic order): guarantees identical output regardless
    # of --n-jobs.
    candidates.sort(key=lambda c: (str(c["outcar"]), c["frame_index"]))

    # ---- Pass 3: global farthest-point sampling ----------------------------
    X = np.array([c["vector"] for c in candidates])
    n_total = len(candidates)
    if n_frames >= n_total:
        print(
            f"Requested n_frames ({n_frames}) >= number of candidates ({n_total}); "
            "keeping all candidates."
        )
        selected_order = list(range(n_total))
    else:
        print(f"Running farthest-point sampling: {n_total} candidates -> {n_frames} frames...")
        selected_order = farthest_point_sample(X, n_frames)
    print()

    # ---- Pass 4: write output, reusing cached Atoms (no second OUTCAR read) --
    by_outcar = defaultdict(list)
    for rank, ci in enumerate(selected_order):
        c = candidates[ci]
        by_outcar[c["outcar"]].append((c["frame_index"], rank))

    try:
        out_path.unlink()
    except FileNotFoundError:
        pass

    first_write = True
    frames_written = 0
    selected_per_run = {}

    for outcar, run_id in valid_outcars:
        sel_list = by_outcar.get(outcar)
        if not sel_list:
            continue
        sel_list.sort(key=lambda t: t[0])
        n_written_here = 0
        for frame_index, rank in sel_list:
            a = atoms_cache.get((outcar, frame_index))
            if a is None:
                print(f"Warning: cached atoms missing for {outcar} frame {frame_index}; skipping")
                continue
            a.info["run_id"] = run_id
            a.info["frame_index"] = frame_index
            a.info["soap_pick_order"] = rank
            write(out_path, a, format="extxyz", append=(not first_write))
            first_write = False
            frames_written += 1
            n_written_here += 1
        selected_per_run[run_id] = n_written_here

    if frames_written == 0:
        raise SystemExit("No frames were written. Check candidate/selection logic above.")

    print("=" * 80)
    print("Writing output...")
    print(f"✓ Written {frames_written} frames to {out_path}")
    print()
    print("Per-run candidates -> selected (sorted by candidate count):")
    for run_id, n_cand in sorted(candidates_per_run.items(), key=lambda kv: -kv[1]):
        n_sel = selected_per_run.get(run_id, 0)
        print(f"  {run_id}: {n_cand} candidates -> {n_sel} selected")
    print()
    print("Extraction statistics:")
    print(f"  OUTCARs found:        {len(outcars_list)}")
    print(f"  OUTCARs skipped:      {skipped}")
    print(f"  Candidate frames:     {len(candidates)}")
    print(f"  Frames written:       {frames_written}")
    print()


def main():
    p = argparse.ArgumentParser(
        description="Sample maximally diverse MD frames via SOAP + farthest-point sampling"
    )
    p.add_argument(
        "paths",
        nargs="*",
        default=["."],
        type=Path,
        help="Directories to search recursively for OUTCARs (default: .); multiple directories are pooled together",
    )
    p.add_argument(
        "-n", "--n-frames",
        type=int,
        default=600,
        help="Total number of frames to select across all trajectories (default: 600)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("md_frames_soap.xyz"),
        help="Output XYZ file (default: md_frames_soap.xyz)",
    )
    p.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Log file path (default: <output_stem>.log)",
    )
    p.add_argument(
        "--no-convergence-check",
        action="store_true",
        help="Consider all frames as candidates even when OSZICAR/OUTCAR suggest non-converged SCF",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Only every Nth frame of each trajectory becomes a SOAP candidate (default: 1)",
    )
    p.add_argument("--r-cut", type=float, default=5.0, help="SOAP cutoff radius in Angstrom (default: 5.0)")
    p.add_argument("--n-max", type=int, default=8, help="SOAP n_max (default: 8)")
    p.add_argument("--l-max", type=int, default=6, help="SOAP l_max (default: 6)")
    p.add_argument("--sigma", type=float, default=0.5, help="SOAP Gaussian width sigma (default: 0.5)")
    p.add_argument(
        "-j", "--n-jobs",
        type=int,
        default=1,
        help="Number of OUTCARs to process in parallel worker processes (default: 1, "
             "sequential). Each OUTCAR's read+SOAP pipeline is independent, so this "
             "scales close to linearly with available CPUs; match it to --cpus-per-task "
             "if submitting via sbatch. Output is identical regardless of this value.",
    )
    p.add_argument(
        "--progress-interval",
        type=int,
        default=30,
        help="Seconds between heartbeat progress updates while descriptors are being "
             "computed (default: 30)",
    )
    args = p.parse_args()

    if args.n_frames < 1:
        raise SystemExit("--n-frames must be >= 1")

    log_path = args.log
    if log_path is None:
        log_path = args.out.with_suffix(".log")

    tee = None
    try:
        tee = Tee(log_path)
        sys.stdout = tee
        run_extraction(
            args.paths,
            args.out,
            args.n_frames,
            check_convergence=not args.no_convergence_check,
            stride=args.stride,
            r_cut=args.r_cut,
            n_max=args.n_max,
            l_max=args.l_max,
            sigma=args.sigma,
            n_jobs=args.n_jobs,
            progress_interval=args.progress_interval,
        )
    finally:
        if tee is not None:
            sys.stdout = tee._stream
            tee.close()
            print(f"Log written to {log_path}")


if __name__ == "__main__":
    main()
