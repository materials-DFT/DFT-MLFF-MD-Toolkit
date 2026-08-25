#!/usr/bin/env python3
"""Convert an extended-XYZ MD trajectory into an extended-XYZ trajectory of the
repeated supercell, writing every frame (not just the first) as a repeated cell.

Usage:
    python make_repeated_xdatcar.py trajectory_unwrapped.extxyz [output.extxyz] --repeat 2 2 2
"""
import argparse

import numpy as np
from ase.io import iread

DEFAULT_OUTPUT = "unwrapped_repeated_trajectory.extxyz"


def repeat_frame(atoms, repeat):
    """Manually tile positions/velocities/forces/species in copy-major order
    (all atoms of translation 0, then translation 1, ...), matching
    Atoms.repeat()'s ordering but without the per-frame ASE object overhead
    and without losing the attached forces (Atoms.repeat() drops the calculator).
    """
    nx, ny, nz = repeat
    cell = np.array(atoms.get_cell())
    symbols = np.array(atoms.get_chemical_symbols())
    positions = atoms.get_positions()
    vel = atoms.arrays.get("vel")
    forces = atoms.calc.results.get("forces") if atoms.calc is not None else None

    frac_offsets = np.array([[i, j, k]
                              for i in range(nx)
                              for j in range(ny)
                              for k in range(nz)], dtype=float)
    cart_offsets = frac_offsets @ cell

    rep_positions = np.vstack([positions + off for off in cart_offsets])
    rep_symbols = np.tile(symbols, len(cart_offsets))
    rep_vel = np.tile(vel, (len(cart_offsets), 1)) if vel is not None else None
    rep_forces = np.tile(forces, (len(cart_offsets), 1)) if forces is not None else None

    return cell, rep_symbols, rep_positions, rep_vel, rep_forces


def write_frame(out, atoms, cell, symbols, positions, vel, forces):
    n = len(symbols)
    lattice_str = " ".join(f"{v:.8f}" for v in cell.flatten())

    props = "species:S:1:pos:R:3"
    if vel is not None:
        props += ":vel:R:3"
    if forces is not None:
        props += ":forces:R:3"

    info_bits = [f'Lattice="{lattice_str}"', f"Properties={props}"]
    for key in ("Timestep", "Time", "Potential_energy", "Temperature"):
        if key in atoms.info:
            info_bits.append(f"{key}={atoms.info[key]}")
    pbc = atoms.pbc
    pbc_str = " ".join("T" if p else "F" for p in pbc)
    info_bits.append(f'pbc="{pbc_str}"')

    out.write(f"{n}\n")
    out.write(" ".join(info_bits) + "\n")

    cols = [positions]
    if vel is not None:
        cols.append(vel)
    if forces is not None:
        cols.append(forces)
    stacked = np.hstack(cols)

    for sym, row in zip(symbols, stacked):
        out.write(sym + " " + " ".join(f"{v:.8f}" for v in row) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="input extended-XYZ trajectory")
    ap.add_argument("output", nargs="?", default=DEFAULT_OUTPUT,
                     help=f"output extended-XYZ path (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--repeat", type=int, nargs=3, default=(2, 2, 2),
                     metavar=("NX", "NY", "NZ"), help="supercell repeat factors (default: 2 2 2)")
    args = ap.parse_args()
    repeat = tuple(args.repeat)

    n_frames = 0
    n_atoms = None
    with open(args.output, "w") as out:
        for atoms in iread(args.input, index=":"):
            cell, symbols, positions, vel, forces = repeat_frame(atoms, repeat)
            write_frame(out, atoms, cell, symbols, positions, vel, forces)
            n_frames += 1
            n_atoms = len(symbols)

    print(f"Wrote {n_frames} frames ({repeat[0]}x{repeat[1]}x{repeat[2]} supercell, "
          f"{n_atoms} atoms/frame) to {args.output}")


if __name__ == "__main__":
    main()
