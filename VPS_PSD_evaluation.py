#!/usr/bin/env python3
"""
Self-contained velocity PSD tool: VASP / LAMMPS MD trajectories → S(ν̃) and ∫S (or T).

This file embeds the full trajectory analysis and plotting stack (formerly
shared with ``VACF_RDF_MSD_evaluation.py``) so it can run with no sibling imports.

CLI: ``python VPS_PSD_evaluation.py DIR … [options]`` opens **only** the PSD +
cumulative-integral figure (RDF/MSD are still computed internally when needed
for metadata but are not shown).

The full trajectory is used (no equilibration frame skipping).

Physics and formats match the original combined evaluator: Wiener–Khinchin
linear spectrum Re(FFT(R)), one-sided folding, wavenumber in cm⁻¹, optional
trim to where ∫S reaches a fraction of ⟨v²⟩, T from m⟨v²⟩/(3k_B) when SI
velocities apply.
"""

import sys
import os
import re
import argparse
from datetime import datetime, timezone

import numpy as np
from pathlib import Path

# Ensure progress prints appear immediately on network filesystems
sys.stdout.reconfigure(line_buffering=True)


# SI constants for kinetic temperature from ⟨v²⟩ (equipartition, 3 translational DOF / atom)
K_B_SI = 1.380649e-23   # J/K
AMU_KG = 1.66053906660e-27
# Wavenumber: ν̃ (cm⁻¹) = f (Hz) / c (cm/s)
SPEED_OF_LIGHT_CM_S = 2.99792458e10
# Default ν̃ cap: SI frequency (THz → cm⁻¹). Stricter than former 50 THz so the
# flat high-ν̃ tail is not shown by default; override with --psd-fmax.
DEFAULT_PSD_FMAX_THZ = 24.0
DEFAULT_PSD_FMAX_CM1 = DEFAULT_PSD_FMAX_THZ * 1e12 / SPEED_OF_LIGHT_CM_S
# Cumulative ∫S trim: lower fraction stops before the tail; small margin past that ν̃.
DEFAULT_PSD_TRIM_FRACTION = 0.92
DEFAULT_PSD_TRIM_MARGIN = 0.01
# Odd-length moving average (frequency bins) for PSD lines only; ∫S unchanged.
DEFAULT_PSD_SMOOTH = 15
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


_FLOAT_RE = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)


def _parse_first_three_floats(line, source):
    """Parse the first 3 float values from a coordinate line.

    Robust against null bytes and trailing non-numeric garbage sometimes seen in
    partially corrupted text files on shared/network filesystems.
    """
    clean = line.replace("\x00", " ")
    vals = _FLOAT_RE.findall(clean)
    if len(vals) < 3:
        raise ValueError(f"Could not parse 3 floats from {source}: {line!r}")
    return np.array(vals[:3], dtype=float)


def kinetic_temperature_from_mean_v_squared(v2, mass_amu, vel_to_m_s):
    """Kinetic temperature from mean square speed (equipartition).

    T = m ⟨v²⟩ / (3 k_B), with ⟨v²⟩ expressed in *trajectory* velocity units
    and *vel_to_m_s* = m/s per one unit of stored velocity (e.g. 100 for
    LAMMPS metal Å/ps, 1e5 for Å/fs).

    Parameters
    ----------
    v2 : float or array
        ⟨v²⟩ or partial ∫ S(f) df in the same squared units as velocities.
    mass_amu : float
        Atomic mass (amu).
    vel_to_m_s : float or None
        Conversion from trajectory velocity to m/s; if None, returns NaNs.
    """
    if vel_to_m_s is None or not np.isfinite(vel_to_m_s):
        return np.full(np.shape(v2), np.nan)
    m_kg = float(mass_amu) * AMU_KG
    v2 = np.asarray(v2, dtype=float)
    v2_si = v2 * (float(vel_to_m_s) ** 2)
    return (m_kg * v2_si) / (3.0 * K_B_SI)


def infer_velocity_si_scale(directory, traj_fmt, dt_source):
    """Return m/s per one unit of velocity as stored in the trajectory.

    LAMMPS ``units metal`` uses Å/ps → 100 m/s per unit.
    ``units real`` uses Å/fs → 1e5 m/s per unit.
    VASP finite-difference velocities use Å/fs → 1e5 m/s per unit.
    """
    d = Path(directory)
    if traj_fmt in ("outcar", "xdatcar"):
        return 1e5
    for inp in sorted(d.glob("in.*")):
        with open(inp) as fh:
            for line in fh:
                line = line.split("#")[0].strip()
                mu = re.match(r"units\s+(\S+)", line, re.I)
                if mu:
                    u = mu.group(1).lower()
                    if u == "metal":
                        return 100.0
                    if u == "real":
                        return 1e5
                    if u in ("si", "cgs"):
                        return 1.0
                    if u == "lj":
                        return None
    # For LAMMPS trajectories where explicit velocities are present in extxyz,
    # units are usually from the LAMMPS unit style (metal: Å/ps, real: Å/fs).
    # If velocities are reconstructed from positions elsewhere, the units are
    # always Å/fs and should use 1e5 m/s per unit instead.
    if dt_source == "LAMMPS":
        return 100.0
    return 1e5


# ---------------------------------------------------------------------------
#  Trajectory I/O
# ---------------------------------------------------------------------------

def _parse_lattice(comment):
    """Extract 3×3 cell from an extended-XYZ comment line."""
    m = re.search(r'Lattice="([^"]+)"', comment)
    if m is None:
        raise ValueError("No Lattice= field in XYZ comment line")
    return np.array(m.group(1).split(), dtype=float).reshape(3, 3)


def _vel_columns(comment):
    """Return (start, end) token indices for velocity columns, or None.

    Parses the ``Properties=...`` field to locate ``vel:R:3``.
    Token indices count from 0 where 0 is the species string.
    """
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
    """Parse VASP OUTCAR for positions and lattice vectors per ionic step.

    Returns the same 4-tuple as ``read_xyz`` so the downstream code is
    agnostic to the source format.
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
                fh.readline()                          # separator "----"
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
    """Fast reader for extended-XYZ / extxyz trajectories.

    Returns ``(positions, cells, species, velocities)`` where
      positions  : (n_frames, n_atoms, 3) Cartesian Å
      cells      : (n_frames, 3, 3)
      species    : list[str] of length n_atoms
      velocities : (n_frames, n_atoms, 3) or None if not in the file
    """
    positions, cells, velocities = [], [], []
    species = None
    vel_cols = None          # determined from first non-skipped frame
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
    """Parse VASP XDATCAR into Cartesian positions (Å) and lattice per frame.

    Each MD step repeats a POSCAR-like block: comment, scale, 3 lattice
    vectors, element symbols, per-type counts, then ``Direct`` or
    ``Cartesian`` (optionally ``… configuration= N``) and *nat* coordinate
    lines.  Direct coordinates are fractional; ``r_cart = r_frac @ cell``
    with lattice rows as in VASP.  Velocities are not in XDATCAR (None).

    Supports NPT trajectories where the cell changes every step.
    """
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
                    # Consume remaining atom lines for this frame to keep parser aligned.
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
    """Locate the best trajectory file in *directory*.

    Priority: **XDATCAR** (compact VASP MD), then OUTCAR, then
    LAMMPS-style extxyz / xyz. Preferring VASP-native trajectories avoids
    accidentally picking unrelated/aggregate xyz files in VASP directories.
    """
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
    """Auto-detect timestep in fs from INCAR / OUTCAR (VASP) or in.* (LAMMPS)."""
    d = Path(directory)

    # VASP: POTIM in fs — try INCAR first, then OUTCAR header
    for candidate in [d / "INCAR", d / "OUTCAR"]:
        if candidate.exists():
            with open(candidate) as fh:
                for line in fh:
                    m = re.match(r"\s*POTIM\s*=\s*([0-9.eE+-]+)", line)
                    if m:
                        return float(m.group(1)), "VASP"
                    if candidate.name == "OUTCAR" and "POSITION" in line:
                        break           # stop scanning past header

    # LAMMPS: "timestep <val>" — value is in time-units of the unit system.
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
                        return dt_raw * 1000.0, "LAMMPS"   # ps → fs
                    elif units == "real":
                        return dt_raw, "LAMMPS"             # already fs
                    else:
                        return dt_raw, "LAMMPS"

    return 1.0, "default"


def _infer_md_engine(directory, traj_fmt, dt_src):
    """Classify a run as VASP (DFT-MD) vs LAMMPS (MLFF MD) for plot legends.

    Uses trajectory priority (OUTCAR/XDATCAR → VASP), then timestep source,
    then presence of ``in.*`` vs ``INCAR``.
    """
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
        # Rare: prefer LAMMPS if we are not reading OUTCAR/XDATCAR
        return "LAMMPS"
    return "unknown"


def _legend_prefix_from_engine(md_engine):
    """Map detected engine to a short plot prefix."""
    if md_engine == "VASP":
        return "DFT"
    if md_engine == "LAMMPS":
        return "MLFF"
    return None


def read_precomputed_msd(directory):
    """Return (time_ps, msd_Å²) from msd_back_all.dat, or (None, None)."""
    p = Path(directory) / "msd_back_all.dat"
    if not p.exists():
        return None, None
    data = np.loadtxt(p, comments="#")
    return data[:, 0], data[:, 1]


# ---------------------------------------------------------------------------
#  Physics: unwrap, VACF → spectrum, RDF, MSD
# ---------------------------------------------------------------------------

def unwrap(pos, cells):
    """Remove PBC jumps from Cartesian trajectory."""
    nf = len(pos)
    uw = np.empty_like(pos)
    uw[0] = pos[0]
    for i in range(1, nf):
        dr = pos[i] - pos[i - 1]
        df = dr @ np.linalg.inv(cells[i])
        df -= np.round(df)
        uw[i] = uw[i - 1] + df @ cells[i]
    return uw


def _vacf_kernel(vel):
    """Unnormalized VACF from a velocity array (n_frames, n_atoms, 3).

    At τ=0 the value is the mean squared speed ⟨v²⟩ for the sampled atoms
    (in (Å/fs)² when velocities are in Å/fs), up to the usual FFT estimator.
    """
    nf, na, nd = vel.shape
    nfft = 1 << int(np.ceil(np.log2(2 * nf)))
    norm = np.arange(nf, 0, -1, dtype=float)
    c = np.zeros(nf)
    for d in range(nd):
        v = vel[:, :, d]
        vf = np.fft.rfft(v, n=nfft, axis=0)
        ac = np.fft.irfft(vf * np.conj(vf), n=nfft, axis=0)[:nf]
        c += np.sum(ac, axis=1) / norm
    c /= na
    return c


def _trapz_compat(y, x):
    """1D trapezoidal integral ∫ y dx (numpy 1.x / 2.x)."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def power_spectrum_from_vacf(vacf, dt_fs, normalize_area=False):
    """One-sided power spectrum from VACF (Wiener–Khinchin, discrete form).

    The symmetric ACF is embedded with τ = 0 at the **centre** of a length
    ``(2N−1)`` array.  Lag zero must be moved to index 0 via ``ifftshift``
    before ``rfft`` so that the transform is the Fourier transform of R(τ),
    **not** ``|FFT(R)|²`` (which would be a Parseval energy on R, unrelated
    to R(0)).

    The one-sided estimate uses ``2·Re(X[k])·Δt`` on interior bins (folding
    negative frequencies), ``Re(X[0])·Δt`` at DC, and ``Re(X[-1])·Δt`` at
    Nyquist when ``M`` is even.      Then ∫ S(f) df in Hz equals R(0) = ⟨v²⟩ (trapezoid).  For plotting vs
    wavenumber ν̃ in cm⁻¹, ``S_ν̃ = S_hz × c`` so ∫ S_ν̃ dν̃ = ⟨v²⟩.

    Parameters
    ----------
    vacf : (N,) array
        R(τ) with τ = 0, 1, … at spacing *dt_fs* (femtoseconds).
    dt_fs : float
        Time between velocity samples in femtoseconds.
    normalize_area : bool
        If True, divide S by ∫ S df (shape only, ∫ df = 1).

    Returns
    -------
    freq_cm1 : (K,) array
        Non-negative wavenumbers in cm⁻¹.
    power : (K,) array
        S(ν̃) in ⟨v²⟩/cm⁻¹.
    cumint : (K,) array
        Cumulative ∫_0^{ν̃} S(ν̃') dν̃' in ⟨v²⟩.
    """
    vacf = np.asarray(vacf, dtype=float)
    N = len(vacf)
    dt_s = float(dt_fs) * 1e-15
    if N < 2 or dt_s <= 0:
        z = np.array([0.0])
        return z, z.copy(), z.copy()

    M = 2 * N - 1
    center = N - 1
    acf2 = np.zeros(M)
    acf2[center] = vacf[0]
    for m in range(1, N):
        acf2[center - m] = vacf[m]
        acf2[center + m] = vacf[m]

    g = np.fft.ifftshift(acf2)
    spec = np.fft.rfft(g, n=M)
    freq_hz = np.fft.rfftfreq(M, d=dt_s)
    freq_cm1 = freq_hz / SPEED_OF_LIGHT_CM_S

    # One-sided PSD (Hz): interior frequencies doubled; DC and Nyquist once.
    p_hz = 2.0 * np.real(spec) * dt_s
    p_hz[0] = np.real(spec[0]) * dt_s
    if M % 2 == 0:
        p_hz[-1] = np.real(spec[-1]) * dt_s

    area_hz = _trapz_compat(p_hz, freq_hz)
    if normalize_area and area_hz > 0:
        p_hz = p_hz / area_hz

    # ⟨v²⟩/cm⁻¹: S_ν̃ dν̃ = S_hz df  ⇒  S_ν̃ = S_hz · c (cm/s)
    power_cm1 = p_hz * SPEED_OF_LIGHT_CM_S

    cumint = np.zeros_like(p_hz)
    df_hz = np.diff(freq_hz)
    if len(df_hz) > 0:
        cumint[1:] = np.cumsum(0.5 * (p_hz[1:] + p_hz[:-1]) * df_hz)
    return freq_cm1, power_cm1, cumint


def power_spectra_from_vacf_by_species(vacf_by_species, dt_fs, normalize_area=False):
    """Compute (freq_cm1, power dict, cumint dict) for each species VACF."""
    freq_cm1 = None
    power_out, cum_out = {}, {}
    for spec in sorted(vacf_by_species.keys()):
        f, p, c = power_spectrum_from_vacf(
            vacf_by_species[spec], dt_fs, normalize_area=normalize_area,
        )
        if freq_cm1 is None:
            freq_cm1 = f
        power_out[spec] = p
        cum_out[spec] = c
    return freq_cm1, power_out, cum_out


def _mass_array(species):
    """Return per-atom masses for mass-weighted COM removal."""
    masses = np.ones(len(species), dtype=float)
    for i, s in enumerate(species):
        masses[i] = ATOMIC_MASS.get(s, 1.0)
    return masses


def remove_com_drift(unwrapped, species):
    """Subtract mass-weighted COM position at every frame."""
    masses = _mass_array(species)
    msum = np.sum(masses)
    com = np.tensordot(unwrapped, masses, axes=([1], [0])) / msum  # (nf, 3)
    return unwrapped - com[:, None, :]


def remove_com_velocity(velocities, species):
    """Subtract mass-weighted COM velocity at every frame."""
    masses = _mass_array(species)
    msum = np.sum(masses)
    vcom = np.tensordot(velocities, masses, axes=([1], [0])) / msum  # (nf, 3)
    return velocities - vcom[:, None, :]


def compute_vacf(unwrapped, dt_fs, n_frames=5000, velocities=None):
    """VACF via FFT: unnormalized series and C(τ)/C(0).

    Uses *velocities* directly when available (LAMMPS extxyz),
    otherwise falls back to central finite-differences of *unwrapped*.
    """
    nv = min(n_frames, len(unwrapped) - 2)
    if velocities is not None:
        vel = velocities[:nv]
        tag = "from trajectory"
    else:
        vel = (unwrapped[2:nv + 2] - unwrapped[:nv]) / (2.0 * dt_fs)
        tag = "finite-diff"
    print(f"    VACF / spectrum: {len(vel)} frames ({tag})")
    vacf_raw = _vacf_kernel(vel)
    v0 = vacf_raw[0]
    vacf_norm = (vacf_raw / v0) if v0 != 0 else vacf_raw
    t = np.arange(len(vacf_raw)) * dt_fs * 1e-3   # ps
    return t, vacf_norm, vacf_raw


def compute_vacf_by_species(unwrapped, dt_fs, species, n_frames=5000, velocities=None):
    """Compute VACF per species: normalized (for export) and raw (for PSD / ⟨v²⟩)."""
    sp = np.array(species)
    out_norm, out_raw = {}, {}
    for s in sorted(set(species)):
        mask = sp == s
        vel_s = velocities[:, mask, :] if velocities is not None else None
        t, vacf_n, vacf_r = compute_vacf(
            unwrapped[:, mask, :],
            dt_fs,
            n_frames=n_frames,
            velocities=vel_s,
        )
        out_norm[s] = vacf_n
        out_raw[s] = vacf_r
    return t, out_norm, out_raw


def compute_rdf(pos, cells, species,
                r_max=6.0, n_bins=300, stride=10):
    """Total + partial RDFs with minimum-image convention.

    Returns dict  {label: (r, g(r))}  including key ``'total'``.
    """
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

    # total g(r): each unique pair counted once → factor 2
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


def _compute_msd_array(unwrapped):
    """Time-averaged MSD from unwrapped Cartesian coordinates (Å).

    For each lag *m*, returns the direct estimator
    ``mean_{t,a} |r_a(t+m)-r_a(t)|^2`` (Å²), averaged over time origins *t*
    and atoms *a*, computed via an O(N log N) FFT (same result as the nested
    loop to numerical precision).
    """
    nf, na, nd = unwrapped.shape
    nfft = 1 << int(np.ceil(np.log2(2 * nf)))
    norm = np.arange(nf, 0, -1, dtype=float)
    lag = np.arange(nf)
    msd = np.zeros(nf)
    for d in range(nd):
        x = unwrapped[:, :, d]                            # (nf, na)
        xf = np.fft.rfft(x, n=nfft, axis=0)
        acf = np.fft.irfft(xf * np.conj(xf), n=nfft, axis=0)[:nf]
        x2 = x ** 2
        cs = np.concatenate([np.zeros((1, na)), np.cumsum(x2, axis=0)])
        # D[m] = Σ_k [x²(k+m) + x²(k)]  for k=0..N-m-1
        D = cs[nf] - cs[lag] + cs[nf - lag]               # (nf, na)
        msd += np.sum((D - 2 * acf) / norm[:, None], axis=1)
    return msd / na


def compute_msd(unwrapped, dt_fs):
    """Total MSD for all atoms."""
    t = np.arange(unwrapped.shape[0]) * dt_fs * 1e-3
    return t, _compute_msd_array(unwrapped)


def compute_msd_by_species(unwrapped, dt_fs, species):
    """Compute MSD for each species separately."""
    sp = np.array(species)
    t = np.arange(unwrapped.shape[0]) * dt_fs * 1e-3
    out = {}
    for s in sorted(set(species)):
        out[s] = _compute_msd_array(unwrapped[:, sp == s, :])
    return t, out


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def _smooth_psd_for_plot(y, window):
    """Odd-length moving average for displayed PSD only (raw arrays unchanged)."""
    y = np.asarray(y, dtype=float)
    win = int(window)
    if win < 3 or len(y) < 3:
        return y
    if win % 2 == 0:
        win += 1
    if len(y) < win:
        return y
    k = np.ones(win, dtype=float) / win
    yp = np.pad(y, (win // 2, win // 2), mode="edge")
    out = np.convolve(yp, k, mode="valid")
    return out[: len(y)]


def _maximize_figure_window(fig):
    """Expand the matplotlib GUI window to fill the screen (Tk/Qt backends)."""
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


def _configure_matplotlib_backend(save=None):
    import matplotlib
    display = os.environ.get("DISPLAY")
    if save and not display:
        matplotlib.use("Agg")
    elif display:
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")


def infer_temperature_from_path(sim_dir):
    """Infer (sort_key, display_label) from a simulation directory path.

    Uses the leaf directory name first (e.g. ``300K`` → ``(300, \"300K\")``).
    If no match, returns ``(None, None)`` so callers can group by path.
    """
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
    """Group (result, label) tuples by temperature or by directory.

    Returns a list of ``(group_title, [(R, label), ...])`` sorted by
    temperature (numeric) then by path name.
    """
    from collections import defaultdict

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


def psd_nu_max_from_cumint(freq_cm1, cumint, fraction=DEFAULT_PSD_TRIM_FRACTION):
    """Smallest ν̃ (cm⁻¹) where cumulative ∫S dν̃ reaches ``fraction`` of the final value.

    Frequencies above this contribute only the tail (≈1−fraction of ⟨v²⟩_SI),
    where the PSD is usually flat and uninformative for peaks.
    """
    freq_cm1 = np.asarray(freq_cm1, dtype=float)
    cumint = np.asarray(cumint, dtype=float)
    if len(freq_cm1) < 2 or len(cumint) < 2:
        return float(freq_cm1[-1]) if len(freq_cm1) else 0.0
    total = float(cumint[-1])
    if not np.isfinite(total) or abs(total) < 1e-40:
        return 0.0
    target = fraction * total
    c = np.maximum.accumulate(np.nan_to_num(cumint, nan=0.0, posinf=0.0, neginf=0.0))
    idx = int(np.searchsorted(c, target, side="left"))
    idx = min(idx, len(freq_cm1) - 1)
    return float(freq_cm1[idx])


def psd_xmax_for_temperature_group(
    series_pairs,
    species_list,
    psd_fmax_cap,
    trim,
    trim_fraction,
    trim_margin,
):
    """Upper ν̃ (cm⁻¹) for PSD/cumint x-axis: min(cap, auto) when *trim* is True."""
    if not trim:
        return psd_fmax_cap
    nu_peak = 0.0
    for R, _ in series_pairs:
        f = R["psd_freq_cm1"]
        for spec in species_list:
            if spec not in R.get("psd_cumint", {}):
                continue
            cum = R["psd_cumint"][spec]
            nu = psd_nu_max_from_cumint(f, cum, fraction=trim_fraction)
            if np.isfinite(nu):
                nu_peak = max(nu_peak, nu)
    if nu_peak <= 0 or not np.isfinite(nu_peak):
        return psd_fmax_cap
    xmax = min(psd_fmax_cap, nu_peak * (1.0 + trim_margin))
    return max(xmax, 1e-6)


# Vertical gap between stacked temperature subfigures (fraction of mean subfig height).
_SUBFIG_HSPACE_TEMP_BLOCKS = 0.08

# tight_layout rect (left, bottom, right, top)
_TIGHT_LAYOUT_DEFAULT = (0.055, 0.03, 0.985, 0.91)
# RDF/MSD: legends on axes (upper right / upper left); no extra bottom margin for below-axes legend
_TIGHT_LAYOUT_RDF_MSD = (0.055, 0.05, 0.985, 0.91)

# ``make_psd_figure`` window: VPS column + cumulative ∫S/T; geometry vs default ``tight_layout`` rect.
_PSD_ONLY_TIGHT_LAYOUT = (0.055, 0.03, 0.985, 0.927)
_PSD_ONLY_SUPTITLE_Y = 0.968
# **VPS** = velocity power spectrum (left column: S(ν̃) vs cm⁻¹). The S(ν̃) label is for that column only,
# not the cumulative ∫S/T panel on the right. Gap = horizontal offset left of the VPS axes, in units of
# one VPS subplot width; **smaller → label closer to the VPS curves**.
_VPS_S_NU_LABEL_GAP_VS_AXIS_WIDTH = 0.038
_PSD_ONLY_COLUMN_TITLE_PAD = 5.0


def _apply_vps_psd_layout(fig):
    """After tight_layout: anchor S(ν̃) beside the VPS column; tweak suptitle and column titles."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for sf in fig.subfigs:
        lab = getattr(sf, "_supylabel", None)
        if lab is None:
            continue
        st = lab.get_text()
        if r"$S(\tilde{\nu})$" not in st and st.strip() != "S(ν̃)":
            continue
        axes_sf = [a for a in sf.get_axes() if a.get_subplotspec() is not None]
        if not axes_sf:
            bb = sf.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
            y_fig = 0.5 * (bb.y0 + bb.y1)
            lab.set_transform(fig.transFigure)
            lab.set_position((bb.x0 + 0.02 * bb.width, y_fig))
            continue
        x0s = [a.get_position().x0 for a in axes_sf]
        x_left = min(x0s)
        vps_axes = [
            a for a in axes_sf
            if abs(a.get_position().x0 - x_left) <= 0.02
        ]
        if not vps_axes:
            vps_axes = axes_sf
        pos = [a.get_position() for a in vps_axes]
        y0 = min(p.y0 for p in pos)
        y1 = max(p.y1 for p in pos)
        y_fig = 0.5 * (y0 + y1)
        axis_w = max(p.width for p in pos)
        x_fig = x_left - _VPS_S_NU_LABEL_GAP_VS_AXIS_WIDTH * axis_w
        lab.set_transform(fig.transFigure)
        lab.set_position((x_fig, y_fig))
    for txt in fig.texts:
        s = txt.get_text()
        if "Velocity power spectra — MD comparison" in s:
            txt.set_y(_PSD_ONLY_SUPTITLE_Y)
    for ax in fig.axes:
        ttl = ax.get_title()
        if ttl == "Velocity power spectrum" or ttl.startswith("Cumulative ∫"):
            # Use set_title(pad=…): ax.title may be Text-only (no set_pad) in some Matplotlib versions.
            ax.set_title(ttl, pad=_PSD_ONLY_COLUMN_TITLE_PAD)


def _finalize_psd_only_window(fig, save, maximize_window):
    """Like ``_finalize_md_figure`` but spectrum-figure rect + VPS label pass after ``tight_layout``."""
    import matplotlib.pyplot as plt

    fig.tight_layout(rect=_PSD_ONLY_TIGHT_LAYOUT)
    _apply_vps_psd_layout(fig)
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


def _finalize_md_figure(fig, save, maximize_window, tight_layout_rect=None):
    """tight_layout, optional save, show or close."""
    import matplotlib.pyplot as plt

    rect = tight_layout_rect if tight_layout_rect is not None else _TIGHT_LAYOUT_DEFAULT
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


def _draw_psd_temperature_subfig(
    sf,
    group_title,
    series_pairs,
    colors,
    ls_cycle,
    psd_fmax,
    psd_trim,
    psd_trim_fraction,
    psd_trim_margin,
    psd_smooth,
    *,
    show_velocity_spectra_suptitle=True,
    shared_psd_s_ylabel=False,
    show_psd_column_title=True,
    show_cum_column_title=True,
):
    """One temperature block: per-species S(ν̃) and cumulative ∫S (or T).

    If ``shared_psd_s_ylabel`` is True, species name is on each PSD row and
    the S(ν̃) axis label is drawn once per block (avoids repeated S(ν̃) on
    every row). If ``show_velocity_spectra_suptitle`` is False, the per-block
    "Velocity spectra — <T>" subfigure title is omitted.
    If ``show_psd_column_title`` / ``show_cum_column_title`` are False, the
    corresponding column titles are omitted (useful when several temperature
    blocks are stacked and titles should appear only once).
    """
    from matplotlib.ticker import AutoMinorLocator

    species_list = sorted(
        set().union(*(set(R["psd"].keys()) for R, _ in series_pairs))
    )
    nsp = len(species_list)
    if show_velocity_spectra_suptitle:
        sf.suptitle(
            f"Velocity spectra — {group_title}",
            fontsize=10,
            weight="bold",
            y=1.01,
        )
    gs = sf.add_gridspec(nsp, 2, width_ratios=[1.06, 1.0])
    ax_psd_ref = None
    axes_psd = []
    for row in range(nsp):
        ax_p = sf.add_subplot(gs[row, 0], sharex=ax_psd_ref)
        if ax_psd_ref is None:
            ax_psd_ref = ax_p
        axes_psd.append(ax_p)
    ax_cum = sf.add_subplot(gs[:, 1], sharex=ax_psd_ref)

    T_nom = None
    for R, _ in series_pairs:
        T_nom = R["meta"].get("nominal_temperature_K")
        if T_nom is not None:
            break

    has_T_axis = any(
        np.any(np.isfinite(R.get("psd_cumint_T", {}).get(s, [])))
        for R, _ in series_pairs
        for s in species_list
        if s in R.get("psd_cumint_T", {})
    )
    multi_series = len(series_pairs) > 1
    psd_si = any(
        R.get("meta", {}).get("psd_vacf_si") for R, _ in series_pairs
    )

    psd_xmax = psd_xmax_for_temperature_group(
        series_pairs,
        species_list,
        psd_fmax,
        psd_trim,
        psd_trim_fraction,
        psd_trim_margin,
    )
    if psd_trim and psd_xmax + 1e-6 < psd_fmax:
        print(
            f"  PSD ν̃ range: 0–{psd_xmax:.1f} cm⁻¹ "
            f"(∫S up to {psd_trim_fraction:.0%} of ⟨v²⟩ + {100*psd_trim_margin:.0f}% margin; "
            f"cap {psd_fmax:.0f} cm⁻¹)",
            flush=True,
        )

    for row, spec in enumerate(species_list):
        ax_psd = axes_psd[row]
        for j, (R, lab) in enumerate(series_pairs):
            if spec not in R["psd"]:
                continue
            f_cm1 = R["psd_freq_cm1"]
            cut = f_cm1 <= psd_xmax
            ff = f_cm1[cut]
            p = R["psd"][spec][cut]
            if psd_smooth and psd_smooth > 0:
                p = _smooth_psd_for_plot(p, psd_smooth)
            kw = dict(color=colors[j % 10], label=lab, lw=1.4)
            ax_psd.plot(ff, p, **kw)

        if shared_psd_s_ylabel:
            ax_psd.set_ylabel(spec)
        elif psd_si:
            ax_psd.set_ylabel(
                f"{spec}\n" + r"$S(\tilde{\nu})$ [(m/s)$^2$/cm$^{-1}$]"
            )
        else:
            ax_psd.set_ylabel(f"{spec}\nS(ν̃)")
        ax_psd.set_xlim(0, psd_xmax)
        ax_psd.xaxis.set_minor_locator(AutoMinorLocator())
        ax_psd.yaxis.set_minor_locator(AutoMinorLocator())
        if row == 0 and show_psd_column_title:
            ax_psd.set_title("Velocity power spectrum")
        if row == nsp - 1:
            ax_psd.set_xlabel("Wavenumber (cm⁻¹)")
        if row == 0:
            ax_psd.legend(
                fontsize=6.5,
                loc="upper right",
                framealpha=0.92,
                fancybox=False,
            )

    if shared_psd_s_ylabel:
        if psd_si:
            sf.supylabel(
                r"$S(\tilde{\nu})$ [(m/s)$^2$/cm$^{-1}$]",
                fontsize=10,
            )
        else:
            sf.supylabel("S(ν̃)", fontsize=10)

    for j, (R, lab) in enumerate(series_pairs):
        for si, spec in enumerate(species_list):
            if spec not in R["psd"]:
                continue
            f_cm1 = R["psd_freq_cm1"]
            cut = f_cm1 <= psd_xmax
            ff = f_cm1[cut]
            c_T = R.get("psd_cumint_T", {}).get(spec)
            if c_T is not None and np.any(np.isfinite(c_T)):
                c_plot = np.asarray(c_T[cut], dtype=float)
            else:
                c_plot = R["psd_cumint"][spec][cut]
            m = R["meta"].get("T_kin_from_vacf0_K", {}).get(spec)
            if multi_series:
                lab_c = f"{lab} ({spec}"
                if m is not None and np.isfinite(m):
                    lab_c += f", T≈{m:.0f} K"
                lab_c += ")"
            else:
                lab_c = (
                    f"{spec} (T≈{m:.0f} K)"
                    if m is not None and np.isfinite(m)
                    else spec
                )
            ax_cum.plot(
                ff, c_plot,
                color=colors[si % 10],
                ls=ls_cycle[j % len(ls_cycle)],
                label=lab_c,
                lw=1.35,
            )

    if T_nom is not None:
        ax_cum.axhline(
            T_nom, color="0.45", lw=1.0, ls="--", zorder=0,
        )

    ax_cum.set_ylabel(
        "T (K) from ∫ S df" if has_T_axis else "∫ S df",
    )
    ax_cum.set_xlim(0, psd_xmax)
    ax_cum.xaxis.set_minor_locator(AutoMinorLocator())
    ax_cum.yaxis.set_minor_locator(AutoMinorLocator())
    if show_cum_column_title:
        ax_cum.set_title(
            "Cumulative ∫ S df → T (K), all species"
            if has_T_axis
            else "Cumulative ∫ S df, all species",
        )
    ax_cum.set_xlabel("Wavenumber (cm⁻¹)")
    n_cum_leg = len(series_pairs) * len(species_list)
    ax_cum.legend(
        fontsize=5.8,
        loc="lower right",
        ncol=2 if n_cum_leg > 4 else 1,
        framealpha=0.92,
        fancybox=False,
        handlelength=2.2,
    )
    if T_nom is not None:
        ax_cum.text(
            0.02, 0.98,
            f"grey dash: nominal T = {T_nom:g} K",
            transform=ax_cum.transAxes,
            fontsize=6,
            color="0.35",
            va="top",
        )


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
    ax_rdf.set_xlabel("r (Å)")
    ax_rdf.set_ylabel("g(r)")
    ax_rdf.set_title("Radial distribution function")
    ax_rdf.legend(
        fontsize=6.5,
        ncol=2 if len(results_list) > 2 else 1,
        framealpha=0.92,
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
    ax_msd.set_xlabel("τ (ps)")
    ax_msd.set_ylabel("MSD (Å²)")
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
        fontsize=6.5,
        ncol=ncol,
        framealpha=0.92,
        loc="upper left",
        fancybox=False,
        columnspacing=0.9,
        handletextpad=0.5,
        handlelength=1.8,
    )
    ax_msd.xaxis.set_minor_locator(AutoMinorLocator())


def make_psd_figure(
    results_list,
    labels,
    sim_dirs,
    psd_fmax=DEFAULT_PSD_FMAX_CM1,
    save=None,
    psd_trim=True,
    psd_trim_fraction=DEFAULT_PSD_TRIM_FRACTION,
    psd_trim_margin=DEFAULT_PSD_TRIM_MARGIN,
    psd_smooth=DEFAULT_PSD_SMOOTH,
    maximize_window=True,
):
    """One window: velocity PSD and cumulative ∫S (or T) only, grouped by temperature."""
    _configure_matplotlib_backend(save=save)
    import matplotlib.pyplot as plt

    temp_groups = group_results_by_temperature(sim_dirs, results_list, labels)
    colors = plt.cm.tab10.colors
    ls_cycle = ["-", "--", "-.", ":"]

    heights = []
    for _title, series_pairs in temp_groups:
        species_set = set()
        for R, _ in series_pairs:
            species_set.update(R["psd"].keys())
        nsp = max(len(species_set), 1)
        heights.append(max(2.0, 2.15 * nsp))

    n_blocks = len(temp_groups)
    fig_h = max(6.0, 0.36 * sum(heights) + 0.85)
    fig = plt.figure(figsize=(13.5, fig_h))
    fig.suptitle(
        "Velocity power spectra — MD comparison",
        fontsize=13,
        weight="bold",
        y=_PSD_ONLY_SUPTITLE_Y,
    )
    subfigs = fig.subfigures(
        n_blocks, 1, height_ratios=heights, hspace=_SUBFIG_HSPACE_TEMP_BLOCKS
    )
    subfigs = np.atleast_1d(subfigs).ravel()

    for bi, ((_group_title, series_pairs), sf) in enumerate(
        zip(temp_groups, subfigs)
    ):
        _draw_psd_temperature_subfig(
            sf,
            _group_title,
            series_pairs,
            colors,
            ls_cycle,
            psd_fmax,
            psd_trim,
            psd_trim_fraction,
            psd_trim_margin,
            psd_smooth,
            show_velocity_spectra_suptitle=False,
            shared_psd_s_ylabel=True,
            show_psd_column_title=(bi == 0),
            show_cum_column_title=(bi == 0),
        )

    _finalize_psd_only_window(fig, save, maximize_window)


def make_rdf_msd_figure(
    results_list,
    labels,
    sim_dirs=None,
    rdf_pairs=None,
    msd_tmax=None,
    save=None,
    maximize_window=True,
):
    """One window: RDF and MSD only (stacked by temperature when available)."""
    _configure_matplotlib_backend(save=save)
    import matplotlib.pyplot as plt

    colors = plt.cm.tab10.colors
    ls_cycle = ["-", "--", "-.", ":"]

    if sim_dirs is None or len(sim_dirs) != len(results_list):
        sim_dirs = [f"series_{i + 1}" for i in range(len(results_list))]

    temp_groups = group_results_by_temperature(sim_dirs, results_list, labels)
    n_blocks = max(len(temp_groups), 1)
    fig_h = max(5.8, 2.8 * n_blocks + 0.8)
    fig = plt.figure(figsize=(13.5, fig_h))
    fig.suptitle("Structure & diffusion — MD comparison", fontsize=13, weight="bold", y=0.995)

    subfigs = fig.subfigures(n_blocks, 1, hspace=_SUBFIG_HSPACE_TEMP_BLOCKS)
    subfigs = np.atleast_1d(subfigs).ravel()

    for (group_title, series_pairs), sf in zip(temp_groups, subfigs):
        sf.suptitle(group_title, fontsize=10, weight="bold", y=1.01)
        ax_rdf, ax_msd = sf.subplots(1, 2)
        grp_results = [R for R, _lab in series_pairs]
        grp_labels = [_lab for _R, _lab in series_pairs]
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

    _finalize_md_figure(fig, save, maximize_window, tight_layout_rect=_TIGHT_LAYOUT_RDF_MSD)


def make_combined_figure(
    results_list,
    labels,
    sim_dirs,
    rdf_pairs=None,
    psd_fmax=DEFAULT_PSD_FMAX_CM1,
    msd_tmax=None,
    save=None,
    psd_trim=True,
    psd_trim_fraction=DEFAULT_PSD_TRIM_FRACTION,
    psd_trim_margin=DEFAULT_PSD_TRIM_MARGIN,
    psd_smooth=DEFAULT_PSD_SMOOTH,
    maximize_window=True,
):
    """Single X11 window: per temperature, S(f) one row per species; one shared ∫S df→T panel.

    PSD uses Re(FFT(R)) (one-sided); ∫ S df = ⟨v²⟩ then T = m⟨v²⟩/(3k_B).
    """
    _configure_matplotlib_backend(save=save)
    import matplotlib.pyplot as plt

    temp_groups = group_results_by_temperature(sim_dirs, results_list, labels)
    colors = plt.cm.tab10.colors
    ls_cycle = ["-", "--", "-.", ":"]

    heights = []
    for _title, series_pairs in temp_groups:
        species_set = set()
        for R, _ in series_pairs:
            species_set.update(R["psd"].keys())
        nsp = max(len(species_set), 1)
        heights.append(max(2.0, 2.15 * nsp))
    heights.append(2.75)

    n_blocks = len(temp_groups) + 1
    fig_h = max(7.0, 0.36 * sum(heights) + 1.0)
    fig = plt.figure(figsize=(13.5, fig_h))
    fig.suptitle("MD simulation comparison", fontsize=13, weight="bold", y=0.992)
    subfigs = fig.subfigures(
        n_blocks, 1, height_ratios=heights, hspace=_SUBFIG_HSPACE_TEMP_BLOCKS
    )
    subfigs = np.atleast_1d(subfigs).ravel()

    for (_group_title, series_pairs), sf in zip(temp_groups, subfigs[:-1]):
        _draw_psd_temperature_subfig(
            sf,
            _group_title,
            series_pairs,
            colors,
            ls_cycle,
            psd_fmax,
            psd_trim,
            psd_trim_fraction,
            psd_trim_margin,
            psd_smooth,
        )

    sf_last = subfigs[-1]
    sf_last.suptitle("Structure & diffusion", fontsize=10, weight="bold", y=1.01)
    ax_rdf, ax_msd = sf_last.subplots(1, 2)
    _plot_rdf_msd_axes(
        ax_rdf,
        ax_msd,
        results_list,
        labels,
        rdf_pairs,
        msd_tmax,
        colors,
        ls_cycle,
    )

    _finalize_md_figure(fig, save, maximize_window, tight_layout_rect=_TIGHT_LAYOUT_RDF_MSD)


def make_figure(results_list, labels,
                rdf_pairs=None, psd_fmax=DEFAULT_PSD_FMAX_CM1, msd_tmax=None,
                save=None, sim_dirs=None,
                psd_trim=True, psd_trim_fraction=DEFAULT_PSD_TRIM_FRACTION,
                psd_trim_margin=DEFAULT_PSD_TRIM_MARGIN,
                psd_smooth=DEFAULT_PSD_SMOOTH,
                maximize_window=True):
    """One matplotlib window: spectra by T & species, then RDF and MSD."""
    if sim_dirs is None or len(sim_dirs) != len(results_list):
        raise ValueError("make_figure requires sim_dirs with one entry per result")
    make_combined_figure(
        results_list,
        labels,
        sim_dirs,
        rdf_pairs=rdf_pairs,
        psd_fmax=psd_fmax,
        msd_tmax=msd_tmax,
        save=save,
        psd_trim=psd_trim,
        psd_trim_fraction=psd_trim_fraction,
        psd_trim_margin=psd_trim_margin,
        psd_smooth=psd_smooth,
        maximize_window=maximize_window,
    )


# ---------------------------------------------------------------------------
#  Per-directory driver
# ---------------------------------------------------------------------------

def analyze_one(directory, skip, max_frames,
                rdf_stride, vacf_nframes, rdf_rmax, rdf_nbins,
                dt_override=None, use_precomputed_msd=False,
                compute_psd=True):
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
        pos, cells, species, vel = reader(traj_path, skip=skip,
                                          max_frames=max_frames)
    except Exception as exc:
        dpath = Path(directory)
        fallback_path, fallback_fmt = None, None
        # Robust fallback path selection:
        # - xyz failure   -> try XDATCAR, then OUTCAR
        # - xdatcar fail  -> try OUTCAR
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
            pos, cells, species, vel = reader(traj_path, skip=skip,
                                              max_frames=max_frames)
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

    print("  Unwrapping ...")
    uw = unwrap(pos, cells)
    print("  Removing COM drift ...")
    uw = remove_com_drift(uw, species)
    vel_corr = remove_com_velocity(vel, species) if vel is not None else None

    vacf_t, vacf, vacf_raw = None, {}, {}
    psd_freq_cm1 = np.array([0.0])
    psd, psd_cumint = {}, {}
    psd_vacf_si = False
    md_engine = _infer_md_engine(d, traj_fmt, src)
    print(f"  MD engine: {md_engine}")
    vel_to_m_s = infer_velocity_si_scale(d, traj_fmt, src)
    # If trajectory has no explicit velocities, VACF is built from finite
    # differences of Cartesian positions / dt_fs, i.e. Å/fs regardless of
    # original engine. Use Å/fs -> m/s conversion in that case.
    if vel is None:
        vel_to_m_s = 1e5

    if compute_psd:
        print(f"  Power spectrum (from VACF) ...")
        vacf_t, vacf, vacf_raw = compute_vacf_by_species(
            uw, dt, species, n_frames=vacf_nframes, velocities=vel_corr
        )
        if vel_to_m_s is None:
            print("  Velocity units: LJ — PSD uses raw ⟨v²⟩; kinetic T in K not defined")
            vacf_for_psd = vacf_raw
            psd_vacf_si = False
        else:
            print(
                f"  Velocity → m/s scale: {vel_to_m_s:g} m/s per traj. unit "
                f"(PSD uses ⟨v²⟩ in (m/s)² for MLFF vs DFT overlays)"
            )
            vacf_for_psd = {
                s: vacf_raw[s] * (vel_to_m_s ** 2) for s in vacf_raw
            }
            psd_vacf_si = True

        psd_freq_cm1, psd, psd_cumint = power_spectra_from_vacf_by_species(
            vacf_for_psd, dt, normalize_area=False,
        )
    else:
        print("  Skipping VACF / power spectrum (structure-only analysis)")

    print(f"  RDF (stride={rdf_stride}, rmax={rdf_rmax} Å) ...")
    rdf_res = compute_rdf(pos, cells, species,
                          r_max=rdf_rmax, n_bins=rdf_nbins,
                          stride=rdf_stride)

    msd_tf, msd_f = None, None
    if use_precomputed_msd:
        msd_tf, msd_f = read_precomputed_msd(d)
    if msd_f is not None:
        print(f"  MSD: pre-computed ({len(msd_f)} points)")
        msd_t, msd = msd_tf, {"total": msd_f}
        msd_source = "precomputed(msd_back_all.dat,total_only)"
    else:
        print("  MSD (FFT) ...")
        msd_t, msd = compute_msd_by_species(uw, dt, species)
        msd_source = "fft_trajectory"

    sk_dir, _ = infer_temperature_from_path(d)
    nominal_T = float(sk_dir) if sk_dir is not None else None

    psd_cumint_T = {}
    T_kin_vacf0_K = {}
    if compute_psd and psd_cumint:
        for spec in psd_cumint:
            m_amu = ATOMIC_MASS.get(spec, 1.0)
            if psd_vacf_si:
                psd_cumint_T[spec] = kinetic_temperature_from_mean_v_squared(
                    psd_cumint[spec], m_amu, 1.0,
                )
            else:
                psd_cumint_T[spec] = kinetic_temperature_from_mean_v_squared(
                    psd_cumint[spec], m_amu, None,
                )
            T_kin_vacf0_K[spec] = float(
                kinetic_temperature_from_mean_v_squared(
                    vacf_raw[spec][0], m_amu, vel_to_m_s,
                )
            )
        if T_kin_vacf0_K:
            tstr = ", ".join(f"{s}≈{T_kin_vacf0_K[s]:.1f} K" for s in sorted(T_kin_vacf0_K))
            print(f"  Kinetic T from VACF(0): {tstr}")

    meta = {
        "dt_fs": float(dt),
        "dt_source": src,
        "md_engine": md_engine,
        "trajectory": str(Path(traj_path).resolve()),
        "trajectory_format": traj_fmt,
        "n_frames": int(nf),
        "n_atoms": int(pos.shape[1]),
        "msd_source": msd_source,
        "species": sorted(set(species)),
        "velocity_m_per_s_per_unit": vel_to_m_s,
        "nominal_temperature_K": nominal_T,
        "T_kin_from_vacf0_K": T_kin_vacf0_K,
        "psd_vacf_si": psd_vacf_si,
    }
    return dict(
        vacf_t=vacf_t,
        vacf=vacf,
        vacf_raw=vacf_raw,
        psd_freq_cm1=psd_freq_cm1,
        psd=psd,
        psd_cumint=psd_cumint,
        psd_cumint_T=psd_cumint_T,
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
    """Sort key that extracts leading number from dir name (e.g. '300K' → 300)."""
    m = re.match(r"(\d+)", p.name)
    if m:
        return (0, int(m.group(1)), p.name)
    return (1, 0, p.name)


def _collect_simulation_leaves(root):
    """Recursively collect deepest simulation directories under *root*.

    If a directory has simulation-bearing children, those children are preferred
    over the parent, even when the parent also contains aggregate trajectory files.
    """
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
    """Expand parent directories into their simulation sub-directories.

    Children with simulations are always preferred over a parent-level
    trajectory (which is often an aggregate file).  This lets the user
    pass ``./`` or a parent that holds ``300K/``, ``700K/``, etc.

    Returns
    -------
    resolved : list of Path
    origin_index : list of int
        For each resolved path, the index into *raw_dirs* of the CLI
        argument that produced it (for ``--series-prefixes`` only).
    """
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
            origin_index.append(root_i)  # let analyze_one raise a clear error
    return resolved, origin_index


def write_data_log(out_path, sim_dirs, labels, all_results):
    """Write power spectrum, VACF, MSD, and RDF arrays used for plotting."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("# VAC_RDF_MSD_evaluation — numerical data used for plots\n")
        fh.write(
            "# PSD columns: nu_tilde_cm-1, PSD (⟨v²⟩_SI/(cm-1) if meta.psd_vacf_si), "
            "cumint(⟨v²⟩_SI), [T_K]\n"
        )
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
                    "msd_source",
                    "velocity_m_per_s_per_unit",
                    "nominal_temperature_K",
                    "T_kin_from_vacf0_K",
                    "psd_vacf_si",
                ):
                    if k in m:
                        fh.write(f"# meta.{k}={m[k]}\n")
            fh.write("=" * 80 + "\n\n")

            for spec in sorted(R["psd"].keys()):
                cols = [
                    R["psd_freq_cm1"],
                    R["psd"][spec],
                    R["psd_cumint"][spec],
                ]
                hdr = "nu_tilde_cm-1, PSD, cumint(⟨v²⟩_SI if meta.psd_vacf_si)"
                if "psd_cumint_T" in R and spec in R["psd_cumint_T"]:
                    cols.append(R["psd_cumint_T"][spec])
                    hdr += ", T_K"
                fh.write(
                    f"# --- Power spectrum species={spec!r}: columns {hdr} ---\n"
                )
                np.savetxt(fh, np.column_stack(cols), fmt="%.10e")
                fh.write("\n")

            for spec in sorted(R["vacf"].keys()):
                fh.write(f"# --- VACF species={spec!r}: columns t_ps, C_t_over_C0 ---\n")
                np.savetxt(
                    fh,
                    np.column_stack([R["vacf_t"], R["vacf"][spec]]),
                    fmt="%.10e",
                )
                fh.write("\n")

            if isinstance(R["msd"], dict):
                for spec in sorted(R["msd"].keys()):
                    fh.write(
                        f"# --- MSD species={spec!r}: columns tau_lag_ps, MSD_Angstrom^2 "
                        "(full trajectory; no equilibration time offset) ---\n"
                    )
                    np.savetxt(
                        fh,
                        np.column_stack([R["msd_t"], R["msd"][spec]]),
                        fmt="%.10e",
                    )
                    fh.write("\n")
            else:
                fh.write(
                    "# --- MSD: columns tau_lag_ps, MSD_Angstrom^2 "
                    "(full trajectory; no equilibration time offset) ---\n"
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

    print(f"Data log written → {out_path.resolve()}")


def finalize_plot_labels(dirs, user_labels, all_results,
                         origin_indices=None, n_raw_dirs=None,
                         series_prefixes=None):
    """Build legend labels after each directory has been analyzed.

    When leaf folder names collide (e.g. two ``300K``), prefixes are
    ``DFT/<leaf>`` for VASP runs and ``MLFF/<leaf>`` for LAMMPS runs
    (from ``meta['md_engine']``). Unknown engine falls back to
    ``<parent>/<leaf>``.

    *series_prefixes* (one per CLI ``DIR`` argument) overrides engine-based
    prefixes and forces ``PREFIX/<leaf>`` for every series.
    """
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
#  CLI (PSD + cumulative ∫S figure only)
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Velocity PSD and cumulative ∫S (or T) — MD trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 300K/
  %(prog)s dft/300K mlff/300K --labels DFT MLFF
  %(prog)s run/ --psd-fmax 1300 --save psd.png
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
        help="one legend prefix per DIR argument; forces PREFIX/<T> and "
        "overrides VASP→DFT / LAMMPS→MLFF detection",
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
        "--vacf-frames",
        type=int,
        default=5000,
        help="frames for VACF / power spectrum (default: 5000)",
    )
    ap.add_argument(
        "--rdf-stride",
        type=int,
        default=10,
        help="frame stride for RDF computation (default: 10; not plotted here)",
    )
    ap.add_argument(
        "--rdf-rmax",
        type=float,
        default=6.0,
        help="RDF cutoff in Å (default: 6.0; not plotted here)",
    )
    ap.add_argument(
        "--rdf-bins",
        type=int,
        default=300,
        help="number of RDF bins (default: 300; not plotted here)",
    )
    ap.add_argument(
        "--psd-fmax",
        type=float,
        default=DEFAULT_PSD_FMAX_CM1,
        help="upper ν̃ cap in cm⁻¹; auto-trim never exceeds this "
        f"(default: ~{DEFAULT_PSD_FMAX_THZ:g} THz band)",
    )
    ap.add_argument(
        "--psd-no-trim",
        action="store_true",
        help="use full 0…psd-fmax on PSD axes (no shrink to peak-dominated window)",
    )
    ap.add_argument(
        "--psd-trim-fraction",
        type=float,
        default=DEFAULT_PSD_TRIM_FRACTION,
        metavar="F",
        help="trim PSD x-axis to ν̃ where cumulative ∫S reaches F×⟨v²⟩ "
        f"(default {DEFAULT_PSD_TRIM_FRACTION:g}; lower = stricter)",
    )
    ap.add_argument(
        "--psd-trim-margin",
        type=float,
        default=DEFAULT_PSD_TRIM_MARGIN,
        metavar="M",
        help="extend trimmed ν̃ by fraction M past that point "
        f"(default {DEFAULT_PSD_TRIM_MARGIN:g})",
    )
    ap.add_argument(
        "--psd-smooth",
        type=int,
        default=DEFAULT_PSD_SMOOTH,
        metavar="N",
        help="moving-average window (odd bins) for plotted PSD lines only; "
        f"0 disables (default {DEFAULT_PSD_SMOOTH})",
    )
    ap.add_argument(
        "--no-maximize",
        action="store_true",
        help="do not maximize the plot window to the screen (interactive only)",
    )
    ap.add_argument(
        "--precomputed-msd",
        action="store_true",
        help="use msd_back_all.dat when available (default: compute from trajectory)",
    )
    ap.add_argument(
        "--save",
        type=str,
        default=None,
        help="save figure to this path (png/pdf); use Agg when DISPLAY is unset",
    )
    ap.add_argument(
        "--data-log",
        type=str,
        default=None,
        metavar="FILE",
        help="write numerical data to FILE (tab-separated)",
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
            args.vacf_frames,
            args.rdf_rmax,
            args.rdf_bins,
            dt_override=args.dt,
            use_precomputed_msd=args.precomputed_msd,
            compute_psd=True,
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

    if args.data_log:
        write_data_log(args.data_log, sim_dirs, labels, all_res)

    if args.no_plot:
        if args.save:
            print("Note: --no-plot ignores --save (no figure written).", flush=True)
    else:
        make_psd_figure(
            all_res,
            labels,
            sim_dirs,
            psd_fmax=args.psd_fmax,
            save=args.save,
            psd_trim=not args.psd_no_trim,
            psd_trim_fraction=args.psd_trim_fraction,
            psd_trim_margin=args.psd_trim_margin,
            psd_smooth=args.psd_smooth,
            maximize_window=not args.no_maximize,
        )


if __name__ == "__main__":
    main()
