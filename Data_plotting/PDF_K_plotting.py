#!/usr/bin/env python3
"""Plot the **pair distribution function** g(r) for K–X pairs from MD trajectories.

For isotropic fluids and the usual MD normalization, **g(r) is the same object**
whether you call it the pair distribution function (PDF) or the radial
distribution function (RDF). This script shows partial g(r) for every channel that
involves K, including **K–K** by default (use ``--no-kk`` to omit K–K). The analysis
code uses the name ``compute_rdf`` / result key ``rdf`` for
historical reasons only—the quantity plotted is g(r).

Simulation discovery matches ``RDF_MSD_evaluation.py``: directories are expanded with
``resolve_dirs``, and trajectories are read by the same
``analyze_one`` driver as in that workflow:

- **VASP:** ``XDATCAR`` (preferred) or ``OUTCAR``; timestep from ``INCAR`` / ``OUTCAR``.
- **LAMMPS:** ``*.extxyz`` / ``*.xyz``; timestep from ``in.*``.

**MLFF vs DFT:** pass two parent trees (e.g. ``vasp_runs/ mlff_runs/``) that each contain
matching temperature leaves (``300K/``, …). Curves at the same temperature are drawn in
one panel. When leaf folder names collide (two ``300K``), legend prefixes **DFT** /
**MLFF** come from ``finalize_plot_labels`` (VASP → DFT, LAMMPS → MLFF), same as
``RDF_MSD_evaluation.py``. Override with ``--labels`` or ``--series-prefixes``.

In a panel that contains **both** VASP and LAMMPS, **DFT is drawn with solid lines**
and **MLFF with dotted lines** (``:``). Each **pair type** (K–O, K–Mn, …) has its **own
subplot** within the temperature row; every subplot lives in the **same** matplotlib
window (one X11 figure).

Examples
--------
  %(prog)s 300K/ 700K/
  %(prog)s . --elements O Mn
  %(prog)s ~/vasp/oms6/md/ ~/lammps/oms6/       # DFT vs MLFF, auto labels if leaves match
  %(prog)s dft/ mlff/ --series-prefixes DFT MLFF
  %(prog)s dft/300K mlff/300K --labels DFT MLFF
  %(prog)s 300K/ --no-kk          # heteronuclear K–X only (drop K–K)

Requires a graphical session: ``DISPLAY`` must be set (X11). Figures are shown
interactively only; no image files are written.
"""

from __future__ import annotations

import argparse
import os
import sys
import re
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

_SUBFIG_HSPACE_TEMP_BLOCKS = 0.08
_TIGHT_LAYOUT_RDF_MSD = (0.055, 0.05, 0.985, 0.91)

ATOMIC_MASS = {
    "H": 1.008, "He": 4.002602, "Li": 6.94, "Be": 9.0121831, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998403163, "Ne": 20.1797,
    "Na": 22.98976928, "Mg": 24.305, "Al": 26.9815385, "Si": 28.085, "P": 30.973761998,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.0983, "Ca": 40.078,
    "Sc": 44.955908, "Ti": 47.867, "V": 50.9415, "Cr": 51.9961, "Mn": 54.938044,
    "Fe": 55.845, "Co": 58.933194, "Ni": 58.6934, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.921595, "Se": 78.971, "Br": 79.904,
    "Kr": 83.798,
}

_FLOAT_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


# ---------------------------------------------------------------------------
#  Trajectory I/O
# ---------------------------------------------------------------------------

def _parse_first_three_floats(line, source):
    clean = line.replace("\x00", " ")
    vals = _FLOAT_RE.findall(clean)
    if len(vals) < 3:
        raise ValueError(f"Could not parse 3 floats from {source}: {line!r}")
    return np.array(vals[:3], dtype=float)


def _parse_lattice(comment):
    m = re.search(r'Lattice="([^"]+)"', comment)
    if m is None:
        raise ValueError("No Lattice= field in XYZ comment line")
    return np.array(m.group(1).split(), dtype=float).reshape(3, 3)


def _vel_columns(comment):
    m = re.search(r"Properties=(\S+)", comment)
    if m is None:
        return None
    col, it = 0, 0
    parts = m.group(1).split(":")
    while it + 2 < len(parts):
        name = parts[it]
        ncols = int(parts[it + 2])
        if name == "vel":
            return (col, col + ncols)
        col += ncols
        it += 3
    return None


def read_outcar(path, skip=0, max_frames=None):
    positions, cells = [], []
    species_names, species_counts = [], []
    n_atoms = None
    cell = None

    with open(path) as fh:
        frame_idx, n_read = 0, 0
        while True:
            line = fh.readline()
            if not line:
                break

            if "TITEL" in line:
                species_names.append(line.split("=")[1].split()[1].split("_")[0])
                continue

            if "ions per type" in line:
                species_counts = [int(x) for x in line.split("=")[1].split()]
                n_atoms = sum(species_counts)
                continue

            if "direct lattice vectors" in line and "reciprocal" in line:
                cell = np.empty((3, 3))
                for i in range(3):
                    cell[i] = fh.readline().split()[:3]
                continue

            if n_atoms and line.startswith(" POSITION") and "TOTAL-FORCE" in line:
                fh.readline()
                if frame_idx < skip:
                    for _ in range(n_atoms):
                        fh.readline()
                    frame_idx += 1
                    continue
                pos = np.empty((n_atoms, 3))
                for a in range(n_atoms):
                    pos[a] = fh.readline().split()[:3]
                positions.append(pos)
                cells.append(cell.copy())
                n_read += 1
                frame_idx += 1
                if n_read % 5000 == 0:
                    print(f"    {n_read} frames ...", flush=True)
                if max_frames and n_read >= max_frames:
                    break

    species = []
    for name, count in zip(species_names, species_counts):
        species.extend([name] * count)
    print(f"    {n_read} frames, {n_atoms} atoms")
    return np.array(positions), np.array(cells), species, None


def read_xyz(path, skip=0, max_frames=None):
    positions, cells, velocities = [], [], []
    species = None
    vel_cols = None
    has_vel = None
    with open(path) as fh:
        idx, n_read = 0, 0
        while True:
            header = fh.readline()
            if not header:
                break
            nat = int(header)
            comment = fh.readline()
            if idx < skip:
                for _ in range(nat):
                    fh.readline()
                idx += 1
                continue
            cells.append(_parse_lattice(comment))
            if has_vel is None:
                vel_cols = _vel_columns(comment)
                has_vel = vel_cols is not None
            pos = np.empty((nat, 3))
            vel = np.empty((nat, 3)) if has_vel else None
            sp = [] if species is None else None
            for a in range(nat):
                tok = fh.readline().split()
                if sp is not None:
                    sp.append(tok[0])
                pos[a] = tok[1:4]
                if has_vel:
                    vel[a] = tok[vel_cols[0]:vel_cols[1]]
            positions.append(pos)
            if has_vel:
                velocities.append(vel)
            if sp is not None:
                species = sp
            n_read += 1
            idx += 1
            if n_read % 5000 == 0:
                print(f"    {n_read} frames ...", flush=True)
            if max_frames and n_read >= max_frames:
                break
    vel_tag = "with velocities" if has_vel else "positions only"
    print(f"    {n_read} frames, {nat} atoms ({vel_tag})")
    vel_arr = np.array(velocities) if has_vel else None
    return np.array(positions), np.array(cells), species, vel_arr


def read_xdatcar(path, skip=0, max_frames=None):
    positions, cells = [], []
    species = None
    nat = None
    n_read = 0
    seen = 0
    n_bad = 0
    with open(path) as fh:
        while True:
            comment = fh.readline()
            if not comment:
                break
            scale_line = fh.readline()
            if not scale_line:
                break
            scale = float(scale_line.split()[0])
            cell = np.empty((3, 3))
            for i in range(3):
                cell[i] = np.array(fh.readline().split()[:3], dtype=float)
            cell = cell * scale
            sp_line = fh.readline()
            cnt_line = fh.readline()
            if sp_line is None or cnt_line is None:
                break
            sp_tokens = sp_line.split()
            counts = [int(x) for x in cnt_line.split()]
            if species is None:
                if len(sp_tokens) != len(counts):
                    raise ValueError(
                        f"XDATCAR species/count mismatch in {path}: "
                        f"{sp_tokens!r} vs {counts!r}"
                    )
                species = []
                for s, n in zip(sp_tokens, counts):
                    species.extend([s] * n)
                nat = len(species)
            else:
                if sum(counts) != nat:
                    raise ValueError(
                        f"XDATCAR atom count changed in {path}: expected {nat}, got {sum(counts)}"
                    )
            cfg_line = fh.readline()
            if not cfg_line:
                break
            cfg_l = cfg_line.strip().lower()
            if "cartesian" in cfg_l:
                direct = False
            elif "direct" in cfg_l:
                direct = True
            else:
                direct = not cfg_l.startswith("c")
            seen += 1
            pos = np.empty((nat, 3))
            bad_frame = False
            for a in range(nat):
                ln = fh.readline()
                if not ln:
                    raise ValueError(f"XDATCAR truncated in {path}")
                try:
                    pos[a] = _parse_first_three_floats(ln, f"XDATCAR {path}")
                except ValueError as exc:
                    bad_frame = True
                    n_bad += 1
                    print(
                        f"    Warning: skipping corrupted XDATCAR frame {seen} in {path}: {exc}",
                        flush=True,
                    )
                    for _ in range(a + 1, nat):
                        if not fh.readline():
                            raise ValueError(f"XDATCAR truncated in {path}")
                    break
            if bad_frame:
                continue
            if direct:
                pos = pos @ cell
            if seen <= skip:
                continue
            positions.append(pos)
            cells.append(cell.copy())
            n_read += 1
            if n_read % 5000 == 0:
                print(f"    {n_read} frames ...", flush=True)
            if max_frames is not None and n_read >= max_frames:
                break
    bad_tag = f", skipped {n_bad} corrupted frame(s)" if n_bad else ""
    print(f"    {n_read} frames, {nat} atoms (XDATCAR, positions only{bad_tag})")
    return np.array(positions), np.array(cells), species, None


def find_trajectory(directory):
    d = Path(directory)
    if (d / "XDATCAR").exists():
        return str(d / "XDATCAR"), "xdatcar"
    outcar = d / "OUTCAR"
    if outcar.exists():
        return str(outcar), "outcar"
    for pattern in [
        "trajectory*.extxyz", "*.extxyz",
        "all_frames*.xyz", "*.xyz",
    ]:
        hits = sorted(d.glob(pattern))
        if hits:
            return str(hits[-1]), "xyz"
    return None, None


def read_timestep(directory):
    d = Path(directory)
    for candidate in [d / "INCAR", d / "OUTCAR"]:
        if candidate.exists():
            with open(candidate) as fh:
                for line in fh:
                    m = re.match(r"\s*POTIM\s*=\s*([0-9.eE+-]+)", line)
                    if m:
                        return float(m.group(1)), "VASP"
                    if candidate.name == "OUTCAR" and "POSITION" in line:
                        break
    for inp in sorted(d.glob("in.*")):
        units = "metal"
        with open(inp) as fh:
            for line in fh:
                line = line.split("#")[0].strip()
                mu = re.match(r"units\s+(\S+)", line)
                if mu:
                    units = mu.group(1)
                mt = re.match(r"timestep\s+([0-9.eE+-]+)", line)
                if mt:
                    dt_raw = float(mt.group(1))
                    if units == "metal":
                        return dt_raw * 1000.0, "LAMMPS"
                    elif units == "real":
                        return dt_raw, "LAMMPS"
                    else:
                        return dt_raw, "LAMMPS"
    return 1.0, "default"


def _infer_md_engine(directory, traj_fmt, dt_src):
    d = Path(directory)
    if traj_fmt in ("outcar", "xdatcar"):
        return "VASP"
    if dt_src == "LAMMPS":
        return "LAMMPS"
    if dt_src == "VASP":
        return "VASP"
    has_in = any(d.glob("in.*"))
    has_incar = (d / "INCAR").exists()
    if has_in and not has_incar:
        return "LAMMPS"
    if has_incar and not has_in:
        return "VASP"
    if has_in and has_incar:
        return "LAMMPS"
    return "unknown"


def _legend_prefix_from_engine(md_engine):
    if md_engine == "VASP":
        return "DFT"
    if md_engine == "LAMMPS":
        return "MLFF"
    return None


# ---------------------------------------------------------------------------
#  Physics: RDF
# ---------------------------------------------------------------------------

def compute_rdf(pos, cells, species, r_max=6.0, n_bins=300, stride=10):
    nat = pos.shape[1]
    dr = r_max / n_bins
    r = np.linspace(dr / 2, r_max - dr / 2, n_bins)
    shell = 4.0 / 3.0 * np.pi * ((r + dr / 2) ** 3 - (r - dr / 2) ** 3)
    sp = np.array(species)
    unique_sp = sorted(set(species))

    i_idx, j_idx = np.triu_indices(nat, k=1)

    hist_tot = np.zeros(n_bins)
    pair_list, pair_hists = [], {}
    for ia, a in enumerate(unique_sp):
        for b in unique_sp[ia:]:
            pair_list.append((a, b))
            pair_hists[f"{a}-{b}"] = np.zeros(n_bins)

    nf_used, vol_sum = 0, 0.0
    for fi in range(0, len(pos), stride):
        c = cells[fi]
        ci = np.linalg.inv(c)
        vol_sum += abs(np.linalg.det(c))
        dv = pos[fi][j_idx] - pos[fi][i_idx]
        df = dv @ ci
        df -= np.round(df)
        dist = np.linalg.norm(df @ c, axis=1)
        mask = dist < r_max
        bi = np.clip((dist[mask] / dr).astype(int), 0, n_bins - 1)
        np.add.at(hist_tot, bi, 1)
        vi, vj = i_idx[mask], j_idx[mask]
        for a, b in pair_list:
            if a == b:
                pm = (sp[vi] == a) & (sp[vj] == a)
            else:
                pm = (((sp[vi] == a) & (sp[vj] == b)) |
                      ((sp[vi] == b) & (sp[vj] == a)))
            np.add.at(pair_hists[f"{a}-{b}"], bi[pm], 1)
        nf_used += 1
        if nf_used % 500 == 0:
            print(f"    RDF: {nf_used} frames ...", flush=True)

    V = vol_sum / nf_used
    npairs_tot = nat * (nat - 1) / 2.0
    results = {"total": (r, hist_tot * V / (nf_used * npairs_tot * shell))}

    for a, b in pair_list:
        na_ = int(np.sum(sp == a))
        nb_ = int(np.sum(sp == b))
        npairs = na_ * (na_ - 1) / 2.0 if a == b else float(na_ * nb_)
        if npairs == 0:
            continue
        g = pair_hists[f"{a}-{b}"] * V / (nf_used * npairs * shell)
        results[f"{a}-{b}"] = (r, g)

    print(f"    RDF done ({nf_used} frames)")
    return results


# ---------------------------------------------------------------------------
#  Directory resolution
# ---------------------------------------------------------------------------

def _has_simulation(d):
    d = Path(d)
    if (d / "OUTCAR").exists() or (d / "INCAR").exists():
        return True
    if (d / "XDATCAR").exists() or (d / "POSCAR").exists():
        return True
    if any(d.glob("in.*")) or any(d.glob("*.extxyz")):
        return True
    return False


def _temp_sort_key(p):
    m = re.match(r"(\d+)", p.name)
    if m:
        return (0, int(m.group(1)), p.name)
    return (1, 0, p.name)


def _collect_simulation_leaves(root):
    root = Path(root)
    child_dirs = sorted([c for c in root.iterdir() if c.is_dir()], key=_temp_sort_key)
    leaf_hits = []
    for c in child_dirs:
        leaf_hits.extend(_collect_simulation_leaves(c))
    if leaf_hits:
        return leaf_hits
    if _has_simulation(root):
        return [root]
    return []


def resolve_dirs(raw_dirs):
    resolved = []
    origin_index = []
    for root_i, d in enumerate(raw_dirs):
        d = Path(d)
        leaves = _collect_simulation_leaves(d)
        if leaves:
            for c in leaves:
                resolved.append(c)
                origin_index.append(root_i)
        else:
            resolved.append(d)
            origin_index.append(root_i)
    return resolved, origin_index


# ---------------------------------------------------------------------------
#  Plotting helpers
# ---------------------------------------------------------------------------

def _configure_matplotlib_backend(save=None):
    import matplotlib
    display = os.environ.get("DISPLAY")
    if save and not display:
        matplotlib.use("Agg")
    elif display:
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")


def _maximize_figure_window(fig):
    try:
        mgr = fig.canvas.manager
    except Exception:
        return
    win = getattr(mgr, "window", None)
    if win is None:
        return
    try:
        win.wm_attributes("-zoomed", True)
        return
    except Exception:
        pass
    try:
        win.state("zoomed")
        return
    except Exception:
        pass
    try:
        win.showMaximized()
        return
    except Exception:
        pass
    try:
        win.update_idletasks()
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        win.geometry(f"{sw}x{sh}+0+0")
    except Exception:
        pass


def _finalize_md_figure(fig, save, maximize_window, tight_layout_rect=None):
    import matplotlib.pyplot as plt

    rect = tight_layout_rect if tight_layout_rect is not None else (0.055, 0.03, 0.985, 0.91)
    fig.tight_layout(rect=rect)
    display = os.environ.get("DISPLAY")
    if save:
        out_path = Path(save)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
        print(f"Saved figure → {out_path.resolve()}")
    if not save or display:
        if maximize_window and display:
            fig.canvas.draw()
            _maximize_figure_window(fig)
        plt.show()
    else:
        plt.close(fig)


def infer_temperature_from_path(sim_dir):
    name = Path(sim_dir).resolve().name
    m = re.match(r"^(\d+)\s*K?$", name, re.I)
    if m:
        n = int(m.group(1))
        return n, f"{n}K"
    m2 = re.search(r"(\d+)\s*K", name, re.I)
    if m2:
        n = int(m2.group(1))
        return n, f"{n}K"
    return None, None


def group_results_by_temperature(sim_dirs, results_list, labels):
    buckets = defaultdict(list)
    for d, R, lab in zip(sim_dirs, results_list, labels):
        sk, _ = infer_temperature_from_path(d)
        if sk is not None:
            key = ("T", sk)
        else:
            key = ("dir", str(Path(d).resolve()))
        buckets[key].append((R, lab))

    def sort_key(k):
        if k[0] == "T":
            return (0, k[1], "")
        return (1, 0, k[1])

    ordered = sorted(buckets.keys(), key=sort_key)
    out = []
    for k in ordered:
        pairs = buckets[k]
        if k[0] == "T":
            title = f"{k[1]}K"
        else:
            title = Path(k[1]).name
        out.append((title, pairs))
    return out


def finalize_plot_labels(dirs, user_labels, all_results,
                         origin_indices=None, n_raw_dirs=None,
                         series_prefixes=None):
    if user_labels and len(user_labels) >= len(dirs):
        return user_labels[:len(dirs)]

    if series_prefixes is not None and len(series_prefixes) > 0:
        if n_raw_dirs is None or len(series_prefixes) != n_raw_dirs:
            raise ValueError(
                "series_prefixes must have one entry per DIR argument "
                f"(expected {n_raw_dirs}, got {len(series_prefixes)})"
            )
        if origin_indices is None:
            raise ValueError("series_prefixes requires resolvable DIR roots")
        return [
            f"{series_prefixes[oi]}/{Path(d).resolve().name}"
            for d, oi in zip(dirs, origin_indices)
        ]

    names = [Path(d).resolve().name for d in dirs]
    if len(names) != len(set(names)):
        labels = []
        for d, R in zip(dirs, all_results):
            p = Path(d).resolve()
            pref = _legend_prefix_from_engine(R["meta"].get("md_engine", "unknown"))
            if pref:
                labels.append(f"{pref}/{p.name}")
            else:
                labels.append(f"{p.parent.name}/{p.name}")
        return labels
    return names


# ---------------------------------------------------------------------------
#  Per-directory driver (RDF only for this script)
# ---------------------------------------------------------------------------

def analyze_one(directory, skip, max_frames, rdf_stride, vacf_nframes,
                rdf_rmax, rdf_nbins, dt_override=None,
                use_precomputed_msd=False, compute_psd=False):
    d = Path(directory).resolve()
    print(f"\n{'═' * 60}")
    print(f"  {d}")
    print(f"{'═' * 60}")

    if dt_override is not None:
        dt, src = dt_override, "CLI"
    else:
        dt, src = read_timestep(d)
    print(f"  dt = {dt} fs  ({src})")

    traj_path, traj_fmt = find_trajectory(d)
    if traj_path is None:
        raise FileNotFoundError(
            f"No trajectory (OUTCAR / *.xyz / *.extxyz / XDATCAR) in {d}\n"
            "  Check that the directory contains simulation output.")
    print(f"  Trajectory: {Path(traj_path).name}  ({traj_fmt})")

    readers = {
        "outcar": read_outcar,
        "xyz": read_xyz,
        "xdatcar": read_xdatcar,
    }
    reader = readers.get(traj_fmt, read_xyz)
    try:
        pos, cells, species, vel = reader(traj_path, skip=skip, max_frames=max_frames)
    except Exception as exc:
        dpath = Path(directory)
        fallback_path, fallback_fmt = None, None
        if traj_fmt == "xyz":
            if (dpath / "XDATCAR").exists():
                fallback_path, fallback_fmt = str(dpath / "XDATCAR"), "xdatcar"
            elif (dpath / "OUTCAR").exists():
                fallback_path, fallback_fmt = str(dpath / "OUTCAR"), "outcar"
        elif traj_fmt == "xdatcar":
            if (dpath / "OUTCAR").exists():
                fallback_path, fallback_fmt = str(dpath / "OUTCAR"), "outcar"

        if fallback_path is not None:
            print(
                f"  {traj_fmt.upper()} read failed ({type(exc).__name__}: {exc}); "
                f"retrying with {Path(fallback_path).name} ({fallback_fmt})"
            )
            traj_path, traj_fmt = fallback_path, fallback_fmt
            reader = readers[fallback_fmt]
            pos, cells, species, vel = reader(traj_path, skip=skip, max_frames=max_frames)
        else:
            raise

    nf = len(pos)
    if nf == 0:
        raise ValueError(
            f"No readable frames found in trajectory {traj_path} "
            f"(skip={skip})."
        )
    u, c = np.unique(species, return_counts=True)
    print(f"  Species: {', '.join(f'{s}({n})' for s, n in zip(u, c))}")

    md_engine = _infer_md_engine(d, traj_fmt, src)
    print(f"  MD engine: {md_engine}")

    print(f"  RDF (stride={rdf_stride}, rmax={rdf_rmax} Å) ...")
    rdf_res = compute_rdf(pos, cells, species,
                          r_max=rdf_rmax, n_bins=rdf_nbins,
                          stride=rdf_stride)

    sk_dir, _ = infer_temperature_from_path(d)
    nominal_T = float(sk_dir) if sk_dir is not None else None

    meta = {
        "dt_fs": float(dt),
        "dt_source": src,
        "md_engine": md_engine,
        "trajectory": str(Path(traj_path).resolve()),
        "trajectory_format": traj_fmt,
        "n_frames": int(nf),
        "n_atoms": int(pos.shape[1]),
        "species": sorted(set(species)),
        "nominal_temperature_K": nominal_T,
    }
    return dict(
        rdf=rdf_res,
        meta=meta,
    )


# ---------------------------------------------------------------------------
#  K-pair selection and plotting
# ---------------------------------------------------------------------------

def _pair_species(pair_key: str) -> tuple[str, str] | None:
    parts = pair_key.split("-")
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def select_k_pairs(
    rdf_keys: set[str] | frozenset,
    *,
    elements: list[str] | None,
    include_kk: bool,
) -> list[str]:
    """Return sorted partial g(r) channel labels that involve potassium (``K``)."""
    out: list[str] = []
    for k in rdf_keys:
        if k == "total":
            continue
        sp = _pair_species(k)
        if sp is None:
            continue
        a, b = sp
        if a != "K" and b != "K":
            continue
        if not include_kk and a == "K" and b == "K":
            continue
        if elements is not None:
            elems_set = set(elements)
            other = b if a == "K" else a
            if other == "K":
                if not include_kk or "K" not in elems_set:
                    continue
            elif other not in elems_set:
                continue
        out.append(k)
    return sorted(out)


def _sort_series_pairs_for_comparison(
    series_pairs: list[tuple[dict, str]],
) -> list[tuple[dict, str]]:
    """VASP (DFT) before LAMMPS (MLFF) within each temperature panel."""
    _order = {"VASP": 0, "LAMMPS": 1}

    def _key(item: tuple[dict, str]) -> tuple[int, str]:
        R, lab = item
        eng = R.get("meta", {}).get("md_engine", "unknown")
        return (_order.get(eng, 99), lab)

    return sorted(series_pairs, key=_key)


def _panel_compare_dft_mlff(grp_results: list[dict]) -> bool:
    """True if this temperature panel has both VASP and LAMMPS (DFT vs MLFF overlay)."""
    engines = {R.get("meta", {}).get("md_engine") for R in grp_results}
    return "VASP" in engines and "LAMMPS" in engines


def _linestyle_for_overlay(R: dict, compare_dft_mlff: bool) -> str:
    """Solid for VASP (DFT), dotted for LAMMPS (MLFF); all solid if not comparing."""
    if not compare_dft_mlff:
        return "-"
    eng = R.get("meta", {}).get("md_engine", "")
    if eng == "LAMMPS":
        return ":"
    return "-"


def _plot_k_pair_pdf_figure(
    results_list: list[dict],
    labels: list[str],
    sim_dirs: list,
    pair_keys: list[str],
    *,
    maximize_window: bool,
) -> None:
    if not os.environ.get("DISPLAY"):
        raise SystemExit(
            "DISPLAY is not set — cannot open an interactive plot window. "
            "Use an X11 session or SSH with X11 forwarding (e.g. `ssh -X`). "
            "This script does not save PNG/PDF files."
        )
    _configure_matplotlib_backend(save=None)
    import matplotlib.pyplot as plt

    def _line_palette(n: int):
        if n <= 0:
            return []
        if n <= 10:
            return list(plt.cm.tab10.colors[:n])
        if n <= 20:
            return [plt.cm.tab20(i / 19.0) for i in np.linspace(0, 19, n)]
        return [plt.cm.turbo(i / max(n - 1, 1)) for i in range(n)]

    from matplotlib.ticker import AutoMinorLocator

    temp_groups = group_results_by_temperature(sim_dirs, results_list, labels)
    n_blocks = max(len(temp_groups), 1)
    n_pk = len(pair_keys)
    fig_w = float(np.clip(3.4 * max(n_pk, 1), 11.0, 30.0))
    fig_h = max(4.2, 2.85 * n_blocks + 0.9)
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(
        "K-involved pair distribution functions g(r) — MD",
        fontsize=13,
        weight="bold",
        y=0.995,
    )

    subfigs = fig.subfigures(n_blocks, 1, hspace=_SUBFIG_HSPACE_TEMP_BLOCKS)
    subfigs = np.atleast_1d(subfigs).ravel()

    for (group_title, series_pairs), sf in zip(temp_groups, subfigs):
        series_pairs = _sort_series_pairs_for_comparison(list(series_pairs))
        sf.suptitle(group_title, fontsize=10, weight="bold", y=1.02)
        grp_results = [R for R, _lab in series_pairs]
        grp_labels = [_lab for _R, _lab in series_pairs]
        multi_dir = len(grp_results) > 1
        compare_dm = _panel_compare_dft_mlff(grp_results)
        series_palette = _line_palette(len(grp_results))

        ax_row = sf.subplots(1, n_pk, sharey=True)
        axs = np.atleast_1d(ax_row).ravel()

        for ip, pk in enumerate(pair_keys):
            ax = axs[ip]
            for si, (R, lab) in enumerate(zip(grp_results, grp_labels)):
                if pk not in R["rdf"]:
                    continue
                r, g = R["rdf"][pk]
                ls = _linestyle_for_overlay(R, compare_dm)
                lbl = lab
                ax.plot(
                    r,
                    g,
                    color=series_palette[si % len(series_palette)],
                    ls=ls,
                    label=lbl,
                    lw=1.3,
                )
            ax.axhline(1, c="grey", lw=0.4, ls="--")
            ax.set_title(pk, fontsize=10, weight="bold")
            ax.set_xlabel("r (Å)")
            if ip == 0:
                ax.set_ylabel("g(r)")
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            if multi_dir:
                ax.legend(
                    fontsize=6.5,
                    loc="upper right",
                    framealpha=0.92,
                    fancybox=False,
                )

    _finalize_md_figure(
        fig,
        None,
        maximize_window,
        tight_layout_rect=_TIGHT_LAYOUT_RDF_MSD,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Plot the pair distribution function g(r) for K–element pairs "
            "(same directory resolution as RDF_MSD_evaluation.py)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "dirs",
        nargs="+",
        help=(
            "simulation or parent directories (VASP and/or LAMMPS). "
            "Pass two roots (e.g. vasp_tree mlff_tree) to overlay DFT vs MLFF per temperature."
        ),
    )
    ap.add_argument(
        "--labels",
        nargs="+",
        help="legend labels (default: auto from dir names)",
    )
    ap.add_argument(
        "--series-prefixes",
        nargs="+",
        default=None,
        metavar="PREFIX",
        help="one prefix per DIR argument; forces PREFIX/<T> (same as workflow script)",
    )
    ap.add_argument(
        "--elements",
        nargs="+",
        default=None,
        metavar="X",
        help="only K–X pairs for these X (e.g. O Mn). Default: all neighbors of K.",
    )
    ap.add_argument(
        "--no-kk",
        action="store_true",
        help="omit the K–K partial g(r) (default: K–K is plotted with other K–X channels)",
    )
    ap.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="max frames to read per directory",
    )
    ap.add_argument(
        "--dt",
        type=float,
        default=None,
        help="override timestep in fs (default: auto from INCAR/LAMMPS input)",
    )
    ap.add_argument(
        "--rdf-stride",
        type=int,
        default=10,
        metavar="N",
        help="frame stride when averaging g(r) (default: 10)",
    )
    ap.add_argument(
        "--rdf-rmax",
        type=float,
        default=6.0,
        metavar="R",
        help="g(r) cutoff radius in Å (default: 6.0)",
    )
    ap.add_argument(
        "--rdf-bins",
        type=int,
        default=300,
        metavar="N",
        help="number of histogram bins for g(r) (default: 300)",
    )
    ap.add_argument(
        "--no-maximize",
        action="store_true",
        help="do not maximize the plot window",
    )
    ap.add_argument(
        "--no-plot",
        action="store_true",
        help="skip matplotlib (exit after analysis prints)",
    )
    args = ap.parse_args()
    skip_frames = 0

    sim_dirs, origin_indices = resolve_dirs(args.dirs)
    print(f"Resolved {len(sim_dirs)} simulation(s):")
    for sd in sim_dirs:
        print(f"  {sd}")

    all_res: list[dict] = []
    for d in sim_dirs:
        res = analyze_one(
            str(d),
            skip_frames,
            args.max_frames,
            args.rdf_stride,
            5000,
            args.rdf_rmax,
            args.rdf_bins,
            dt_override=args.dt,
            use_precomputed_msd=False,
            compute_psd=False,
        )
        all_res.append(res)

    labels = finalize_plot_labels(
        sim_dirs,
        args.labels,
        all_res,
        origin_indices=origin_indices,
        n_raw_dirs=len(args.dirs),
        series_prefixes=args.series_prefixes,
    )
    print("Plot legend labels:")
    for lb, sd in zip(labels, sim_dirs):
        print(f"  {lb:24s} ← {sd}")

    union_keys: set[str] = set()
    for R in all_res:
        union_keys.update(R["rdf"].keys())
    pair_keys = select_k_pairs(
        union_keys,
        elements=args.elements,
        include_kk=not args.no_kk,
    )
    if not pair_keys:
        need = "K in the trajectory"
        if args.elements:
            need += f" and K–({', '.join(args.elements)}) pairs present in g(r)"
        raise SystemExit(
            f"No K-involved partial g(r) channels to plot ({need}). "
            f"Available pair keys: {sorted(k for k in union_keys if k != 'total')}"
        )
    print("K-involved partial g(r) channels:", ", ".join(pair_keys))

    if args.no_plot:
        return

    _plot_k_pair_pdf_figure(
        all_res,
        labels,
        sim_dirs,
        pair_keys,
        maximize_window=not args.no_maximize,
    )


if __name__ == "__main__":
    main()
