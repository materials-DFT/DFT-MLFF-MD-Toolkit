#!/usr/bin/env python3
"""Plot radial distribution function g(r) and mean square displacement.

Combines RDF computation with MSD analysis from trajectory files.
Supports VASP (XDATCAR, OUTCAR) and LAMMPS (extxyz) formats.

MSD computation uses NPT-aware unwrapping schemes:
- TOR (default): Displacement/TOR scheme (von Bulow et al., J. Chem. Phys. 153,
  021101 (2020)), recommended for diffusion coefficients in NPT simulations.
- Scaling: LAT scheme, accumulates fractional and scales by current cell.
- Hybrid: Kulke & Vermaas (JCTC 2022), reversible geometry-preserving scheme.

For NPT simulations (ISIF=3 in VASP), the XDATCAR reader detects per-step
lattice vectors automatically.
"""

import sys
import os
import re
import argparse
from datetime import datetime, timezone
from copy import deepcopy

import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

ATOMIC_MASS = {
    "H": 1.008, "He": 4.002602, "Li": 6.94, "Be": 9.0121831, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998403163, "Ne": 20.1797,
    "Na": 22.98976928, "Mg": 24.305, "Al": 26.9815385, "Si": 28.085, "P": 30.973761998,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.0983, "Ca": 40.078,
    "Sc": 44.955908, "Ti": 47.867, "V": 50.9415, "Cr": 51.9961, "Mn": 54.938044,
    "Fe": 55.845, "Co": 58.933194, "Ni": 58.6934, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.921595, "Se": 78.971, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.47, "Sr": 87.62, "Y": 88.91, "Zr": 91.22, "Nb": 92.91,
    "Mo": 95.95, "Tc": 98.00, "Ru": 101.1, "Rh": 102.9, "Pd": 106.4, "Ag": 107.9,
    "Cd": 112.4, "In": 114.8, "Sn": 118.7, "Sb": 121.8, "Te": 127.6, "I": 126.9,
    "Xe": 131.3, "Cs": 132.9, "Ba": 137.3, "La": 138.9, "Ce": 140.1, "Pr": 140.9,
    "Nd": 144.2, "Pm": 145.0, "Sm": 150.4, "Eu": 152.0, "Gd": 157.3, "Tb": 158.9,
    "Dy": 162.5, "Ho": 164.9, "Er": 167.3, "Tm": 168.9, "Yb": 173.0, "Lu": 175.0,
    "Hf": 178.5, "Ta": 180.9, "W": 183.8, "Re": 186.2, "Os": 190.2, "Ir": 192.2,
    "Pt": 195.1, "Au": 197.0, "Hg": 200.6, "Tl": 204.4, "Pb": 207.2, "Bi": 209.0,
    "Po": 209.0, "At": 210.0, "Rn": 222.0, "Fr": 223.0, "Ra": 226.0, "Ac": 227.0,
    "Th": 232.0, "Pa": 231.0, "U": 238.0, "Np": 237.0, "Pu": 244.0,
}

UNWRAP_SCHEMES = ("tor", "scaling", "hybrid")


def _get_masses(species):
    """Return atomic masses for species, using ASE if available, else fallback dict."""
    try:
        from ase.data import atomic_masses, atomic_numbers
        return np.array([atomic_masses[atomic_numbers[s]] for s in species], dtype=float)
    except ImportError:
        return np.array([ATOMIC_MASS.get(s, 1.0) for s in species], dtype=float)

_FLOAT_RE = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)


def _parse_first_three_floats(line, source):
    """Parse the first 3 float values from a coordinate line."""
    clean = line.replace("\x00", " ")
    vals = _FLOAT_RE.findall(clean)
    if len(vals) < 3:
        raise ValueError(f"Could not parse 3 floats from {source}: {line!r}")
    return np.array(vals[:3], dtype=float)


# ---------------------------------------------------------------------------
#  Trajectory I/O
# ---------------------------------------------------------------------------

def _parse_lattice(comment):
    """Extract 3x3 cell from an extended-XYZ comment line."""
    m = re.search(r'Lattice="([^"]+)"', comment)
    if m is None:
        raise ValueError("No Lattice= field in XYZ comment line")
    return np.array(m.group(1).split(), dtype=float).reshape(3, 3)


def read_outcar(path, skip=0, max_frames=None):
    """Parse VASP OUTCAR for positions and lattice vectors per ionic step.
    
    Handles concatenated OUTCARs (from VASP restarts) by detecting and skipping
    restart headers that may appear mid-file.
    """
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
                fh.readline()  # skip dashes line
                if frame_idx < skip:
                    for _ in range(n_atoms):
                        fh.readline()
                    frame_idx += 1
                    continue
                pos = np.empty((n_atoms, 3))
                bad_frame = False
                for a in range(n_atoms):
                    pos_line = fh.readline()
                    if not pos_line:
                        bad_frame = True
                        break
                    tokens = pos_line.split()[:3]
                    try:
                        pos[a] = [float(t) for t in tokens]
                    except ValueError:
                        # Hit a restart header or corrupted line; skip this frame
                        bad_frame = True
                        break
                if bad_frame:
                    continue
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
    return np.array(positions), np.array(cells), species


def read_xyz(path, skip=0, max_frames=None):
    """Fast reader for extended-XYZ / extxyz trajectories.
    
    Returns Cartesian positions and per-frame cells.
    """
    positions, cells = [], []
    species = None
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
            pos = np.empty((nat, 3))
            sp = [] if species is None else None
            for a in range(nat):
                tok = fh.readline().split()
                if sp is not None:
                    sp.append(tok[0])
                pos[a] = tok[1:4]
            positions.append(pos)
            if sp is not None:
                species = sp
            n_read += 1
            idx += 1
            if n_read % 5000 == 0:
                print(f"    {n_read} frames ...", flush=True)
            if max_frames and n_read >= max_frames:
                break
    print(f"    {n_read} frames, {nat} atoms")
    return np.array(positions), np.array(cells), species


def read_fractional_xyz(path, skip=0, max_frames=None):
    """Read an extended XYZ of fractional coordinates with per-frame lattices.
    
    The file is expected to have Lattice="..." tags with per-step cells.
    Coordinates are assumed to be fractional.
    
    Returns (species, scaled[T, N, 3], cells[T, 3, 3]).
    """
    file_path = Path(path)
    symbols = []
    scaled_frames = []
    cells = []

    with file_path.open("r", encoding="utf-8") as handle:
        idx, n_read = 0, 0
        while True:
            count_line = handle.readline()
            if not count_line:
                break
            count_line = count_line.strip()
            if not count_line:
                continue
            atom_count = int(count_line)

            comment = handle.readline()
            if not comment:
                raise ValueError(f"Unexpected EOF reading comment in {file_path}")
            
            if idx < skip:
                for _ in range(atom_count):
                    handle.readline()
                idx += 1
                continue

            cells.append(_parse_lattice(comment))

            frame_symbols = []
            coords = np.empty((atom_count, 3), dtype=float)
            for atom_idx in range(atom_count):
                fields = handle.readline().split()
                if len(fields) < 4:
                    raise ValueError(f"Malformed atom line in {file_path}")
                frame_symbols.append(fields[0])
                coords[atom_idx] = [float(v) for v in fields[1:4]]

            if not symbols:
                symbols = frame_symbols
            scaled_frames.append(coords)
            n_read += 1
            idx += 1

            if n_read % 5000 == 0:
                print(f"    {n_read} frames ...", flush=True)
            if max_frames and n_read >= max_frames:
                break

    if not scaled_frames:
        raise ValueError(f"No frames were read from {file_path}")

    print(f"    {n_read} frames, {atom_count} atoms (fractional XYZ)")
    return symbols, np.stack(scaled_frames, axis=0), np.stack(cells, axis=0)


def cartesian_to_fractional(cart_pos, cells):
    """Convert Cartesian positions to fractional coordinates.
    
    Parameters
    ----------
    cart_pos : ndarray
        Shape (T, N, 3) or (N, 3), Cartesian positions.
    cells : ndarray
        Shape (T, 3, 3) or (3, 3), lattice vectors as rows.
    
    Returns
    -------
    ndarray
        Fractional positions with same shape as input.
    """
    if cart_pos.ndim == 2:
        return cart_pos @ np.linalg.inv(cells)
    frac = np.empty_like(cart_pos)
    for i in range(cart_pos.shape[0]):
        cell_i = cells[i] if cells.ndim == 3 else cells
        frac[i] = cart_pos[i] @ np.linalg.inv(cell_i)
    return frac


def fractional_to_cartesian(frac_pos, cells):
    """Convert fractional positions to Cartesian coordinates.
    
    Parameters
    ----------
    frac_pos : ndarray
        Shape (T, N, 3) or (N, 3), fractional positions.
    cells : ndarray
        Shape (T, 3, 3) or (3, 3), lattice vectors as rows.
    
    Returns
    -------
    ndarray
        Cartesian positions with same shape as input.
    """
    if frac_pos.ndim == 2:
        return frac_pos @ cells
    cart = np.empty_like(frac_pos)
    for i in range(frac_pos.shape[0]):
        cell_i = cells[i] if cells.ndim == 3 else cells
        cart[i] = frac_pos[i] @ cell_i
    return cart


def read_xdatcar(path, skip=0, max_frames=None):
    """Parse VASP XDATCAR, honoring per-step lattices in NPT (ISIF=3) runs.
    
    Standard NVT XDATCAR files contain a single header (scale + 3 lattice
    vectors + species + counts) followed by "Direct configuration=" blocks.
    NPT runs repeat the full header before every configuration. This parser
    re-reads the header whenever a block is not introduced by a
    "Direct configuration=" line, so the cell tracks the box every step.
    
    Returns fractional positions and per-frame cells. Caller should convert
    to Cartesian as needed: cart = frac @ cell.
    """
    file_path = Path(path)
    positions_frac, cells = [], []
    species = None
    cell = np.eye(3)
    sp_list = []
    counts = []
    total = 0
    n_read = 0
    n_bad = 0

    with open(file_path) as fh:
        while True:
            line = fh.readline()
            if not line:
                break

            if "Direct configuration=" not in line and "Direct" not in line:
                scale_line = fh.readline()
                if not scale_line:
                    break
                scale = float(scale_line.split()[0])

                rows = []
                for _ in range(3):
                    rows.append([float(v) for v in fh.readline().split()])
                cell = np.asarray(rows, dtype=float) * scale

                sp_list = fh.readline().split()
                counts = [int(v) for v in fh.readline().split()]
                total = sum(counts)

                config_line = fh.readline()
                if not config_line:
                    break

            if total == 0:
                raise ValueError(f"Missing header before coordinates in {file_path}")

            coords = np.empty((total, 3), dtype=float)
            bad_frame = False
            for atom_idx in range(total):
                coord_line = fh.readline()
                if not coord_line:
                    raise ValueError(f"XDATCAR truncated in {file_path}")
                try:
                    coords[atom_idx] = _parse_first_three_floats(coord_line, f"XDATCAR {path}")
                except ValueError as exc:
                    bad_frame = True
                    n_bad += 1
                    print(f"    Warning: skipping corrupted frame: {exc}", flush=True)
                    for _ in range(atom_idx + 1, total):
                        if not fh.readline():
                            raise ValueError(f"XDATCAR truncated in {file_path}")
                    break

            if bad_frame:
                continue

            if species is None:
                species = []
                for elem, cnt in zip(sp_list, counts):
                    species.extend([elem] * cnt)

            if skip > 0:
                skip -= 1
                continue

            positions_frac.append(coords)
            cells.append(cell.copy())
            n_read += 1

            if n_read % 5000 == 0:
                print(f"    {n_read} frames ...", flush=True)
            if max_frames is not None and n_read >= max_frames:
                break

    if not positions_frac:
        raise ValueError(f"No frames were read from {file_path}")

    bad_tag = f", skipped {n_bad} corrupted frame(s)" if n_bad else ""
    print(f"    {n_read} frames, {total} atoms (XDATCAR NPT-aware{bad_tag})")
    return np.stack(positions_frac, axis=0), np.stack(cells, axis=0), species


def find_trajectory(directory):
    """Locate the best trajectory file in *directory*.
    
    Priority: XDATCAR (VASP MD), then OUTCAR, then extxyz/xyz files.
    For extxyz, prefers wrapped trajectories over unwrapped ones since
    the MSD calculation handles unwrapping internally.
    """
    d = Path(directory)
    if (d / "XDATCAR").exists():
        return str(d / "XDATCAR"), "xdatcar"
    outcar = d / "OUTCAR"
    if outcar.exists():
        return str(outcar), "outcar"
    
    if (d / "trajectory.extxyz").exists():
        return str(d / "trajectory.extxyz"), "xyz"
    
    for pattern in [
        "trajectory*.extxyz", "*.extxyz",
        "all_frames*.xyz", "*.xyz",
    ]:
        hits = sorted(d.glob(pattern))
        hits_wrapped = [h for h in hits if 'unwrapped' not in h.name.lower()]
        if hits_wrapped:
            return str(hits_wrapped[-1]), "xyz"
        if hits:
            return str(hits[-1]), "xyz"
    return None, None


def read_timestep(directory):
    """Auto-detect timestep in fs from INCAR / OUTCAR (VASP) or in.* (LAMMPS)."""
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
    """Classify a run as VASP (DFT-MD) vs LAMMPS (MLFF MD) for plot legends."""
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
    """Map detected engine to a short plot prefix."""
    if md_engine == "VASP":
        return "DFT"
    if md_engine == "LAMMPS":
        return "MLFF"
    return None


# ---------------------------------------------------------------------------
#  Physics: unwrap, RDF, MSD
# ---------------------------------------------------------------------------

def _unwrap_tor(scaled, cells):
    """Displacement/TOR unwrap: accumulate min-image displacements per step.
    
    This is the recommended scheme for diffusion coefficients (von Bulow et al.,
    J. Chem. Phys. 153, 021101 (2020)).
    """
    unwrapped = np.empty_like(scaled)
    unwrapped[0] = scaled[0] @ cells[0]
    for i in range(1, scaled.shape[0]):
        delta = scaled[i] - scaled[i - 1]
        delta -= np.round(delta)
        unwrapped[i] = unwrapped[i - 1] + delta @ cells[i]
    return unwrapped


def _unwrap_scaling(scaled, cells):
    """Scaling/LAT unwrap: accumulate fractional, scale by the current cell."""
    unwrapped_frac = np.empty_like(scaled)
    unwrapped_frac[0] = scaled[0]
    for i in range(1, scaled.shape[0]):
        delta = scaled[i] - scaled[i - 1]
        delta -= np.round(delta)
        unwrapped_frac[i] = unwrapped_frac[i - 1] + delta
    return np.einsum("tni,tij->tnj", unwrapped_frac, cells)


def _unwrap_hybrid(scaled, cells):
    """Hybrid unwrap (Kulke & Vermaas, JCTC 2022, Eq. 12).
    
    Reversible, geometry-preserving scheme that corrects the displacement
    scheme with a (L_{i+1} - L_i) term accounting for box rescaling.
    Defined for orthogonal boxes.
    """
    n_frames = scaled.shape[0]
    lengths = np.linalg.norm(cells, axis=2)
    wrapped_cart = scaled * lengths[:, None, :]
    unwrapped_cart = np.empty_like(scaled)
    unwrapped_cart[0] = wrapped_cart[0]

    def _round_half_up(values):
        return np.floor(values + 0.5)

    for i in range(1, n_frames):
        length = lengths[i]
        length_prev = lengths[i - 1]
        step = wrapped_cart[i] - wrapped_cart[i - 1]
        prev_offset = wrapped_cart[i - 1] - unwrapped_cart[i - 1]
        unwrapped_cart[i] = (
            unwrapped_cart[i - 1]
            + step
            - length * _round_half_up(step / length)
            - (length - length_prev) * _round_half_up(prev_offset / length_prev)
        )
    return unwrapped_cart


def unwrap_positions_npt(scaled, cells, scheme="tor"):
    """Unwrap wrapped fractional positions into continuous Cartesian space.
    
    Parameters
    ----------
    scaled : ndarray
        Wrapped fractional positions, shape (T, N, 3).
    cells : ndarray
        Per-frame row-vector cells, shape (T, 3, 3).
    scheme : str
        One of "tor" (default, recommended for diffusion), "scaling", or "hybrid".
    
    Returns
    -------
    ndarray
        Unwrapped Cartesian positions, shape (T, N, 3).
    """
    scaled = np.asarray(scaled, dtype=float)
    cells = np.asarray(cells, dtype=float)
    if scaled.shape[0] < 2:
        raise ValueError("Need at least two frames to unwrap a trajectory.")
    
    normalized = scheme.lower()
    if normalized == "tor":
        return _unwrap_tor(scaled, cells)
    if normalized == "scaling":
        return _unwrap_scaling(scaled, cells)
    if normalized == "hybrid":
        return _unwrap_hybrid(scaled, cells)
    raise ValueError(f"Unknown unwrap scheme {scheme!r}. Choose from {UNWRAP_SCHEMES}.")


def unwrap(pos, cells):
    """Remove PBC jumps from Cartesian trajectory (legacy compatibility).
    
    For new code, prefer unwrap_positions_npt() with fractional input.
    """
    nf = len(pos)
    uw = np.empty_like(pos)
    uw[0] = pos[0]
    for i in range(1, nf):
        dr = pos[i] - pos[i - 1]
        df = dr @ np.linalg.inv(cells[i])
        df -= np.round(df)
        uw[i] = uw[i - 1] + df @ cells[i]
    return uw


def remove_com_drift_cartesian(positions, masses):
    """Subtract mass-weighted COM position from each frame.
    
    Parameters
    ----------
    positions : ndarray
        Shape (T, N, 3), Cartesian positions.
    masses : ndarray
        Shape (N,), atomic masses.
    
    Returns
    -------
    ndarray
        Positions with COM drift removed.
    """
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        return positions
    com = np.sum(positions * masses[None, :, None], axis=1, keepdims=True) / total_mass
    return positions - com


def remove_com_drift(unwrapped, species):
    """Subtract mass-weighted COM position at every frame (legacy interface)."""
    masses = _get_masses(species)
    return remove_com_drift_cartesian(unwrapped, masses)


def compute_rdf(pos, cells, species,
                r_max=6.0, n_bins=300, stride=10):
    """Total + partial RDFs with minimum-image convention."""
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


def _ordered_elements(species):
    """Return unique element labels in first-appearance order."""
    ordered = []
    for s in species:
        if s not in ordered:
            ordered.append(s)
    return ordered


def compute_msd_npt(
    frac_positions, cells, species, dt_fs,
    scheme="tor", remove_com=True
):
    """Compute single-origin MSD from fractional positions using NPT-aware unwrapping.
    
    Uses the displacement/TOR scheme (von Bulow et al., J. Chem. Phys. 153, 021101 (2020))
    by default, which is recommended for diffusion coefficients in NPT simulations.
    
    Parameters
    ----------
    frac_positions : ndarray
        Fractional (scaled) positions, shape (T, N, 3).
    cells : ndarray
        Per-frame lattice vectors, shape (T, 3, 3), rows are lattice vectors.
    species : list
        Element symbols for each atom.
    dt_fs : float
        Timestep in femtoseconds.
    scheme : str
        Unwrapping scheme: "tor" (default), "scaling", or "hybrid".
    remove_com : bool
        Subtract mass-weighted center-of-mass drift (default True).
    
    Returns
    -------
    msd_t : ndarray
        Time array in picoseconds, shape (T-1,).
    msd : dict
        Dictionary mapping element symbol to MSD array (Angstrom^2), shape (T-1,).
    """
    n_frames = frac_positions.shape[0]
    if n_frames < 2:
        raise ValueError("Need at least two frames to compute MSD.")
    
    print(f"  Unwrapping trajectory ({scheme} scheme)...")
    unwrapped_cart = unwrap_positions_npt(frac_positions, cells, scheme=scheme)
    
    if remove_com:
        masses = _get_masses(species)
        unwrapped_cart = remove_com_drift_cartesian(unwrapped_cart, masses)
        print("  COM drift removed.")
    
    element_list = _ordered_elements(species)
    symbol_array = np.asarray(species)
    element_indices = {elem: np.where(symbol_array == elem)[0] for elem in element_list}
    
    origin = unwrapped_cart[0]
    displacements = unwrapped_cart[1:] - origin[None, :, :]
    squared_total = np.sum(displacements ** 2, axis=2)
    
    msd = {}
    for elem in element_list:
        indices = element_indices[elem]
        msd[elem] = np.mean(squared_total[:, indices], axis=1)
    
    msd_t = np.arange(1, n_frames) * dt_fs * 1e-3
    
    print(f"  MSD done ({n_frames - 1} time points, {len(element_list)} species)")
    return msd_t, msd


def compute_msd_from_cartesian(
    cart_positions, cells, species, dt_fs,
    scheme="tor", remove_com=True
):
    """Compute MSD from Cartesian positions by first converting to fractional.
    
    This is a convenience wrapper for trajectories that were read as Cartesian
    (e.g., from OUTCAR or extxyz).
    """
    print("  Converting Cartesian to fractional coordinates...")
    frac_positions = cartesian_to_fractional(cart_positions, cells)
    return compute_msd_npt(frac_positions, cells, species, dt_fs, scheme=scheme, remove_com=remove_com)


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def _configure_matplotlib_backend(save=None):
    import matplotlib
    display = os.environ.get("DISPLAY")
    # File export must be backend-independent; a stale DISPLAY variable is
    # common on clusters and does not guarantee that Tk is available.
    if save:
        matplotlib.use("Agg")
    elif display:
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")


def _apply_publication_style(plt):
    """Apply a compact, consistent style suitable for raster or vector export."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.fontsize": 7.5,
        "legend.framealpha": 0.95,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.axisbelow": True,
    })


def infer_temperature_from_path(sim_dir):
    """Infer (sort_key, display_label) from a simulation directory path."""
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


def group_indices_by_temperature(sim_dirs):
    """Group simulation indices by temperature-like folder naming."""
    from collections import defaultdict

    buckets = defaultdict(list)
    for idx, d in enumerate(sim_dirs):
        sk, _ = infer_temperature_from_path(d)
        if sk is not None:
            key = ("T", sk)
        else:
            key = ("dir", str(Path(d).resolve()))
        buckets[key].append(idx)

    def sort_key(k):
        if k[0] == "T":
            return (0, k[1], "")
        return (1, 0, k[1])

    ordered = sorted(buckets.keys(), key=sort_key)
    out = []
    for k in ordered:
        idxs = buckets[k]
        if k[0] == "T":
            title = f"{k[1]}K"
        else:
            title = Path(k[1]).name
        out.append((title, idxs))
    return out


def _path_from_home(directory):
    """Return absolute path with HOME replaced by '~' when possible."""
    p = Path(directory).resolve()
    home = Path.home().resolve()
    try:
        rel = p.relative_to(home)
        return str(Path("~") / rel)
    except ValueError:
        return str(p)


def _maximize_figure_window(fig):
    """Expand the matplotlib GUI window to fill the screen."""
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


# Compact vertical separation between stacked RDF/MSD blocks.
_SUBFIG_HSPACE_TEMP_BLOCKS = 0.001
_TIGHT_LAYOUT_RDF_MSD = (0.04, 0.06, 0.98, 0.92)


def _figure_uses_constrained_layout(fig):
    """True when Figure(layout='constrained') was used."""
    get = getattr(fig, "get_layout_engine", None)
    if get is None:
        return False
    le = get()
    return le is not None and type(le).__name__ == "ConstrainedLayoutEngine"


def _figure_uses_subfigures(fig):
    """Return True if the figure contains subfigures (incompatible with tight_layout)."""
    return hasattr(fig, 'subfigs') and fig.subfigs


def _finalize_md_figure(fig, save, maximize_window, tight_layout_rect=None, skip_tight_layout=False):
    """tight_layout, optional save, show or close."""
    import matplotlib.pyplot as plt

    if not skip_tight_layout and not _figure_uses_constrained_layout(fig) and not _figure_uses_subfigures(fig):
        rect = tight_layout_rect if tight_layout_rect is not None else (0.055, 0.03, 0.985, 0.91)
        fig.tight_layout(rect=rect)
    display = os.environ.get("DISPLAY")
    if save:
        out_path = Path(save)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
        print(f"Saved figure -> {out_path.resolve()}")
    if not save:
        if maximize_window and display:
            fig.canvas.draw()
            _maximize_figure_window(fig)
        plt.show()
    else:
        plt.close(fig)


def _plot_rdf_msd_axes(
    ax_rdf,
    ax_msd,
    results_list,
    labels,
    rdf_pairs,
    msd_tmax,
    colors,
    ls_cycle,
):
    from matplotlib.ticker import AutoMinorLocator

    pairs_to_plot = rdf_pairs or ["total"]
    multi_dir = len(results_list) > 1

    for i, (R, lab) in enumerate(zip(results_list, labels)):
        for ip, pk in enumerate(pairs_to_plot):
            if pk not in R["rdf"]:
                continue
            r, g = R["rdf"][pk]
            ls = ls_cycle[ip % len(ls_cycle)]
            lbl = f"{lab} ({pk})" if len(pairs_to_plot) > 1 else lab
            ax_rdf.plot(r, g, color=colors[i % 10], ls=ls, label=lbl, lw=1.3)
    ax_rdf.axhline(1, c="grey", lw=0.4, ls="--")
    ax_rdf.set_xlabel(r"$r$ (Å)")
    ax_rdf.set_ylabel("g(r)")
    ax_rdf.set_title("Radial distribution function")
    ax_rdf.legend(
        fontsize=7,
        ncol=2 if len(results_list) > 2 else 1,
        framealpha=0.95,
        loc="upper right",
        fancybox=False,
    )
    ax_rdf.xaxis.set_minor_locator(AutoMinorLocator())

    for i, (R, lab) in enumerate(zip(results_list, labels)):
        t_lag = R["msd_t"]
        msd_map = R["msd"] if isinstance(R["msd"], dict) else {"total": R["msd"]}
        for ispec, (spec, m) in enumerate(sorted(msd_map.items())):
            tl, ms = t_lag, m
            if msd_tmax is not None:
                cut = tl <= msd_tmax
                tl, ms = tl[cut], ms[cut]
            lbl = f"{lab} ({spec})" if multi_dir else spec
            ax_msd.plot(
                tl, ms,
                color=colors[i % 10], ls=ls_cycle[ispec % len(ls_cycle)],
                label=lbl, lw=1.3,
            )
    ax_msd.set_xlabel(r"$τ$ (ps)")
    ax_msd.set_ylabel(r"MSD (Å$^2$)")
    ax_msd.set_title("Mean square displacement")
    handles, leg_labs = ax_msd.get_legend_handles_labels()
    n_leg = len(handles)
    if n_leg <= 3:
        ncol = 1
    elif n_leg <= 8:
        ncol = 2
    else:
        ncol = 3
    ax_msd.legend(
        handles,
        leg_labs,
        fontsize=7,
        ncol=ncol,
        framealpha=0.92,
        loc="upper left",
        fancybox=False,
        columnspacing=0.9,
        handletextpad=0.5,
        handlelength=1.8,
    )
    ax_msd.xaxis.set_minor_locator(AutoMinorLocator())


def make_rdf_msd_figure(
    results_list,
    labels,
    sim_dirs=None,
    rdf_pairs=None,
    msd_tmax=None,
    save=None,
    maximize_window=True,
    show_dirs=False,
):
    """One window: RDF and MSD, optionally labeled with source directories."""
    _configure_matplotlib_backend(save=save)
    import matplotlib.pyplot as plt
    _apply_publication_style(plt)

    colors = plt.cm.tab10.colors
    ls_cycle = ["-", "--", "-.", ":"]

    if sim_dirs is None or len(sim_dirs) != len(results_list):
        sim_dirs = [f"series_{i + 1}" for i in range(len(results_list))]

    temp_groups = group_indices_by_temperature(sim_dirs)
    n_blocks = max(len(temp_groups), 1)
    # Keep each block compact while reserving a small, explicit header band
    # for the directory label above the axes.
    fig_h = max(4.8, 2.45 * n_blocks + 0.45)
    fig = plt.figure(figsize=(13.5, fig_h))
    fig.subplots_adjust(top=0.985, bottom=0.055, left=0.055, right=0.985)

    subfigs = fig.subfigures(n_blocks, 1, hspace=_SUBFIG_HSPACE_TEMP_BLOCKS)
    subfigs = np.atleast_1d(subfigs).ravel()

    for (_group_title, idxs), sf in zip(temp_groups, subfigs):
        header_dirs = [_path_from_home(sim_dirs[i]) for i in idxs]
        if len(header_dirs) == 1:
            header_text = header_dirs[0]
        else:
            header_text = "\n".join(header_dirs)
        if show_dirs:
            sf.suptitle(header_text, fontsize=8.5, weight="bold", y=0.985,
                        ha="center", wrap=True)
        ax_rdf, ax_msd = sf.subplots(
            1, 2,
            gridspec_kw={
                "wspace": 0.14,
                # Leave room for axes titles when no directory header is drawn.
                "top": 0.82 if show_dirs else 0.87,
                "bottom": 0.17,
            },
        )
        grp_results = [results_list[i] for i in idxs]
        grp_labels = [labels[i] for i in idxs]
        _plot_rdf_msd_axes(
            ax_rdf,
            ax_msd,
            grp_results,
            grp_labels,
            rdf_pairs,
            msd_tmax,
            colors,
            ls_cycle,
        )

    _finalize_md_figure(fig, save, maximize_window, tight_layout_rect=None, skip_tight_layout=True)


# ---------------------------------------------------------------------------
#  Per-directory driver
# ---------------------------------------------------------------------------

def analyze_one(directory, skip, max_frames,
                rdf_stride, rdf_rmax, rdf_nbins,
                dt_override=None, unwrap_scheme="tor"):
    """Analyze a single simulation directory for RDF and MSD.
    
    Args:
        directory: Path to simulation directory
        skip: Number of frames to skip at the beginning
        max_frames: Maximum frames to read (None for all)
        rdf_stride: Frame stride for RDF calculation
        rdf_rmax: RDF cutoff in Angstroms
        rdf_nbins: Number of RDF bins
        dt_override: Override timestep in fs (None for auto-detect)
        unwrap_scheme: Unwrapping scheme for MSD ("tor", "scaling", "hybrid")
    
    MSD is computed using the specified unwrapping scheme with per-frame lattice
    vectors. The TOR scheme (von Bulow et al.) is recommended for diffusion
    coefficients in NPT simulations.
    """
    d = Path(directory).resolve()
    print(f"\n{'=' * 60}")
    print(f"  {d}")
    print(f"{'=' * 60}")

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
    coords_are_fractional = (traj_fmt == "xdatcar")
    
    try:
        pos, cells, species = reader(traj_path, skip=skip, max_frames=max_frames)
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
            coords_are_fractional = (fallback_fmt == "xdatcar")
            pos, cells, species = reader(traj_path, skip=skip, max_frames=max_frames)
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

    if coords_are_fractional:
        frac_pos = pos
        cart_pos = fractional_to_cartesian(pos, cells)
    else:
        cart_pos = pos
        frac_pos = cartesian_to_fractional(pos, cells)

    print(f"  RDF (stride={rdf_stride}, rmax={rdf_rmax} A) ...")
    rdf_res = compute_rdf(cart_pos, cells, species,
                          r_max=rdf_rmax, n_bins=rdf_nbins,
                          stride=rdf_stride)

    print(f"  MSD (scheme={unwrap_scheme}) ...")
    msd_t, msd = compute_msd_npt(frac_pos, cells, species, dt,
                                  scheme=unwrap_scheme, remove_com=True)

    sk_dir, _ = infer_temperature_from_path(d)
    nominal_T = float(sk_dir) if sk_dir is not None else None

    meta = {
        "dt_fs": float(dt),
        "dt_source": src,
        "md_engine": md_engine,
        "trajectory": str(Path(traj_path).resolve()),
        "trajectory_format": traj_fmt,
        "n_frames": int(nf),
        "n_atoms": int(cart_pos.shape[1]),
        "species": sorted(set(species)),
        "nominal_temperature_K": nominal_T,
        "unwrap_scheme": unwrap_scheme,
    }
    return dict(
        rdf=rdf_res,
        msd_t=msd_t,
        msd=msd,
        meta=meta,
    )


# ---------------------------------------------------------------------------
#  Directory resolution
# ---------------------------------------------------------------------------

def _has_simulation(d):
    """True if *d* looks like it contains a simulation (trajectory or input)."""
    d = Path(d)
    if (d / "OUTCAR").exists() or (d / "INCAR").exists():
        return True
    if (d / "XDATCAR").exists() or (d / "POSCAR").exists():
        return True
    if any(d.glob("in.*")) or any(d.glob("*.extxyz")):
        return True
    return False


def _temp_sort_key(p):
    """Sort key that extracts leading number from dir name (e.g. '300K' -> 300)."""
    m = re.match(r"(\d+)", p.name)
    if m:
        return (0, int(m.group(1)), p.name)
    return (1, 0, p.name)


def _collect_simulation_leaves(root):
    """Recursively collect deepest simulation directories under *root*."""
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
    """Expand parent directories into their simulation sub-directories."""
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


def write_data_log(out_path, sim_dirs, labels, all_results):
    """Write RDF and MSD arrays used for plotting."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("# RDF_MSD_evaluation -- numerical data used for plots\n")
        fh.write(f"# generated_utc={utc}\n\n")
        for i, (d, lab, R) in enumerate(zip(sim_dirs, labels, all_results)):
            fh.write("=" * 80 + "\n")
            fh.write(f"# series_index={i}\n")
            fh.write(f"# label={lab}\n")
            fh.write(f"# directory={Path(d).resolve()}\n")
            if "meta" in R:
                m = R["meta"]
                for k in (
                    "dt_fs",
                    "dt_source",
                    "md_engine",
                    "trajectory",
                    "trajectory_format",
                    "n_frames",
                    "n_atoms",
                    "nominal_temperature_K",
                    "unwrap_scheme",
                ):
                    if k in m:
                        fh.write(f"# meta.{k}={m[k]}\n")
            fh.write("=" * 80 + "\n\n")

            if isinstance(R["msd"], dict):
                for spec in sorted(R["msd"].keys()):
                    fh.write(
                        f"# --- MSD species={spec!r}: columns tau_lag_ps, MSD_Angstrom^2 ---\n"
                    )
                    np.savetxt(
                        fh,
                        np.column_stack([R["msd_t"], R["msd"][spec]]),
                        fmt="%.10e",
                    )
                    fh.write("\n")
            else:
                fh.write(
                    "# --- MSD: columns tau_lag_ps, MSD_Angstrom^2 ---\n"
                )
                np.savetxt(
                    fh,
                    np.column_stack([R["msd_t"], R["msd"]]),
                    fmt="%.10e",
                )
                fh.write("\n")

            for pk in sorted(R["rdf"].keys()):
                r, g = R["rdf"][pk]
                fh.write(f"# --- RDF pair={pk!r}: columns r_A, g_r ---\n")
                np.savetxt(fh, np.column_stack([r, g]), fmt="%.10e")
                fh.write("\n")

    print(f"Data log written -> {out_path.resolve()}")


def finalize_plot_labels(dirs, user_labels, all_results,
                         origin_indices=None, n_raw_dirs=None,
                         series_prefixes=None):
    """Build legend labels after each directory has been analyzed."""
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
#  CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="RDF and MSD from MD trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 300K/
  %(prog)s dft/300K mlff/300K --labels DFT MLFF
  %(prog)s run/ --rdf-pairs total Mn-O --save rdf_msd.png
""",
    )
    ap.add_argument(
        "dirs",
        nargs="+",
        help="simulation or parent directories to analyze",
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
        help="one legend prefix per DIR argument; forces PREFIX/<T>",
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
        help="frame stride for RDF (default: 10)",
    )
    ap.add_argument(
        "--rdf-rmax",
        type=float,
        default=6.0,
        help="RDF cutoff in A (default: 6.0)",
    )
    ap.add_argument(
        "--rdf-bins",
        type=int,
        default=300,
        help="number of RDF bins (default: 300)",
    )
    ap.add_argument(
        "--rdf-pairs",
        nargs="+",
        default=None,
        help="partial RDFs to plot, e.g. Mn-O K-O (default: total)",
    )
    ap.add_argument(
        "--msd-tmax",
        type=float,
        default=None,
        help="max lag time t on MSD plot (ps; default: full range)",
    )
    ap.add_argument(
        "--unwrap-scheme",
        choices=list(UNWRAP_SCHEMES),
        default="tor",
        help="unwrapping scheme for MSD: tor (default, recommended for diffusion), "
             "scaling, or hybrid",
    )
    ap.add_argument(
        "--no-maximize",
        action="store_true",
        help="do not maximize the plot window to the screen",
    )
    ap.add_argument(
        "--show_dirs",
        action="store_true",
        help="show source directory labels above each RDF/MSD subplot row",
    )
    ap.add_argument(
        "--save",
        type=str,
        default=None,
        help="save figure to this path (png/pdf)",
    )
    ap.add_argument(
        "--data-log",
        type=str,
        default=None,
        metavar="FILE",
        help="write RDF and MSD data to FILE (tab-separated)",
    )
    ap.add_argument(
        "--no-plot",
        action="store_true",
        help="skip matplotlib; use with --data-log for analysis only",
    )
    args = ap.parse_args()
    skip_frames = 0

    sim_dirs, origin_indices = resolve_dirs(args.dirs)

    print(f"Resolved {len(sim_dirs)} simulation(s):")
    for sd in sim_dirs:
        print(f"  {sd}")

    all_res = []
    for d in sim_dirs:
        res = analyze_one(
            str(d),
            skip_frames,
            args.max_frames,
            args.rdf_stride,
            args.rdf_rmax,
            args.rdf_bins,
            dt_override=args.dt,
            unwrap_scheme=args.unwrap_scheme,
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
        print(f"  {lb:24s} <- {sd}")

    if args.data_log:
        write_data_log(args.data_log, sim_dirs, labels, all_res)

    if args.no_plot:
        if args.save:
            print("Note: --no-plot ignores --save (no figure written).", flush=True)
    else:
        make_rdf_msd_figure(
            all_res,
            labels,
            sim_dirs=sim_dirs,
            rdf_pairs=args.rdf_pairs,
            msd_tmax=args.msd_tmax,
            save=args.save,
            maximize_window=not args.no_maximize,
            show_dirs=args.show_dirs,
        )


if __name__ == "__main__":
    main()
