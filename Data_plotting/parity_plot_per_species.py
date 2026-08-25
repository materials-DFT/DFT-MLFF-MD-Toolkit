#!/usr/bin/env python3
"""Publication-style per-species force parity plots."""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib-cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.ticker import AutoMinorLocator, MaxNLocator  # noqa: E402


ML_TYPES = (
    ("MACE", "MACE_energy", "MACE_forces", "MACE_stress"),
    ("ALLEGRO", "ALLEGRO_energy", "ALLEGRO_forces", "ALLEGRO_stress"),
    ("UMA", "UMA_energy", "UMA_forces", "UMA_stress"),
)

PLOT_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
)


@dataclass
class RunningStats:
    count: int = 0
    abs_error_sum: float = 0.0
    sq_error_sum: float = 0.0

    def update(self, reference: np.ndarray, prediction: np.ndarray) -> None:
        diff = np.asarray(prediction, dtype=float) - np.asarray(reference, dtype=float)
        self.count += int(diff.size)
        self.abs_error_sum += float(np.sum(np.abs(diff)))
        self.sq_error_sum += float(np.sum(diff * diff))

    @property
    def mae(self) -> float:
        return self.abs_error_sum / self.count if self.count else math.nan

    @property
    def rmse(self) -> float:
        return math.sqrt(self.sq_error_sum / self.count) if self.count else math.nan


def detect_mlip(info):
    for label, e_key, f_key, s_key in ML_TYPES:
        if e_key in info:
            return label, e_key, f_key, s_key
    return None


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#4D4D4D",
            "axes.labelcolor": "#1F1F1F",
            "axes.labelsize": 10,
            "axes.linewidth": 0.9,
            "axes.titlecolor": "#1F1F1F",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "legend.fontsize": 8,
            "xtick.color": "#333333",
            "xtick.labelsize": 9,
            "ytick.color": "#333333",
            "ytick.labelsize": 9,
        }
    )


def finalize_axis(axis):
    axis.set_aspect("equal", adjustable="box")
    axis.grid(axis="both", color="#E2E2E2", linewidth=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_minor_locator(AutoMinorLocator())
    axis.tick_params(axis="both", which="major", direction="out", length=4, width=0.8)
    axis.tick_params(axis="both", which="minor", direction="out", length=2, width=0.6)


def padded_limits(*arrays):
    valid_arrays = [np.asarray(array, dtype=float).ravel() for array in arrays if np.asarray(array).size]
    if not valid_arrays:
        return -1.0, 1.0
    values = np.concatenate(valid_arrays)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    low = float(np.min(values))
    high = float(np.max(values))
    if math.isclose(low, high):
        pad = max(abs(low) * 0.05, 1.0)
    else:
        pad = 0.045 * (high - low)
    return low - pad, high + pad


def add_stats_box(axis, stats, unit, extra=""):
    lines = [
        f"RMSE = {stats.rmse:.4g} {unit}",
        f"MAE = {stats.mae:.4g} {unit}",
        f"N = {stats.count:,}",
    ]
    if extra:
        lines.append(extra)
    axis.text(
        0.04,
        0.96,
        "\n".join(lines),
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        color="#1F1F1F",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#BDBDBD",
            "linewidth": 0.8,
            "alpha": 0.92,
        },
    )


parser = argparse.ArgumentParser(
    description=(
        "Plot per-species force parity (MLIP vs REF) from an XYZ file. "
        "Auto-detects MACE, ALLEGRO, or UMA from frame.info. "
        "Shows combined force parity per species and directional (Fx, Fy, Fz) breakdown."
    )
)
parser.add_argument("xyz_file", type=str, help="Path to the input .xyz file")
args = parser.parse_args()


try:
    frames = read(args.xyz_file, index=":")
except Exception as exc:
    print(f"Error reading file {args.xyz_file}: {exc}")
    sys.exit(1)

if len(frames) == 0:
    print(f"No frames found in file {args.xyz_file}")
    sys.exit(1)

first_info = frames[0].info
detected = detect_mlip(first_info)
if detected is None:
    print(
        "No 'MACE_energy', 'ALLEGRO_energy', or 'UMA_energy' in frame.info. "
        "Make sure the XYZ file was written with one of these fields."
    )
    sys.exit(1)

ml_label, _, ml_forces_key, _ = detected

species_forces = defaultdict(
    lambda: {
        k: []
        for k in ("ref_fx", "ref_fy", "ref_fz", "ml_fx", "ml_fy", "ml_fz")
    }
)

for frame in frames:
    ref_f = frame.arrays["REF_forces"]
    ml_f = frame.arrays[ml_forces_key]
    symbols = frame.get_chemical_symbols()
    for j, sym in enumerate(symbols):
        species_forces[sym]["ref_fx"].append(ref_f[j, 0])
        species_forces[sym]["ref_fy"].append(ref_f[j, 1])
        species_forces[sym]["ref_fz"].append(ref_f[j, 2])
        species_forces[sym]["ml_fx"].append(ml_f[j, 0])
        species_forces[sym]["ml_fy"].append(ml_f[j, 1])
        species_forces[sym]["ml_fz"].append(ml_f[j, 2])

for sym in species_forces:
    for key in species_forces[sym]:
        species_forces[sym][key] = np.array(species_forces[sym][key], dtype=float)

species_list = sorted(species_forces.keys())
n_species = len(species_list)

configure_style()

nrows = 4
ncols = n_species
fig = plt.figure(figsize=(5.25 * ncols, 4.7 * nrows), constrained_layout=True)
gs = GridSpec(nrows, ncols, figure=fig)

dir_labels = ["x", "y", "z"]
dir_keys = [("ref_fx", "ml_fx"), ("ref_fy", "ml_fy"), ("ref_fz", "ml_fz")]

# Row 0: combined force parity per species
for col, sym in enumerate(species_list):
    sf = species_forces[sym]
    ref_all = np.concatenate([sf["ref_fx"], sf["ref_fy"], sf["ref_fz"]])
    ml_all = np.concatenate([sf["ml_fx"], sf["ml_fy"], sf["ml_fz"]])
    combined_stats = RunningStats()
    combined_stats.update(ref_all, ml_all)

    ax = fig.add_subplot(gs[0, col])
    ax.scatter(
        ref_all,
        ml_all,
        alpha=0.3,
        s=4,
        color=PLOT_COLORS[col % len(PLOT_COLORS)],
        rasterized=True,
    )
    lims = padded_limits(ref_all, ml_all)
    ax.plot(lims, lims, color="#222222", linestyle="--", linewidth=1.1, zorder=3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("REF force component (eV/Å)")
    ax.set_ylabel(f"{ml_label} force component (eV/Å)")
    ax.set_title(
        f"{sym} Forces",
        loc="left",
        fontsize=11,
        fontweight="semibold",
        pad=8,
    )
    finalize_axis(ax)
    add_stats_box(ax, combined_stats, "eV/Å", extra=f"Atoms = {len(sf['ref_fx']):,}")

# Rows 1-3: directional breakdown per species
for col, sym in enumerate(species_list):
    sf = species_forces[sym]
    for row_off, (d_label, (ref_key, ml_key)) in enumerate(zip(dir_labels, dir_keys)):
        ref_d = sf[ref_key]
        ml_d = sf[ml_key]
        stats = RunningStats()
        stats.update(ref_d, ml_d)

        ax = fig.add_subplot(gs[1 + row_off, col])
        ax.scatter(
            ref_d,
            ml_d,
            alpha=0.3,
            s=4,
            color=PLOT_COLORS[col % len(PLOT_COLORS)],
            rasterized=True,
        )
        lims = padded_limits(ref_d, ml_d)
        ax.plot(lims, lims, color="#222222", linestyle="--", linewidth=1.1, zorder=3)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"REF $F_{{{d_label}}}$ (eV/Å)")
        ax.set_ylabel(f"{ml_label} $F_{{{d_label}}}$ (eV/Å)")
        ax.set_title(
            f"{sym} - $F_{{{d_label}}}$",
            loc="left",
            fontsize=11,
            fontweight="semibold",
            pad=8,
        )
        finalize_axis(ax)
        add_stats_box(ax, stats, "eV/Å")

fig.suptitle(f"Per-Species Force Parity ({ml_label} vs REF)", fontsize=13, fontweight="semibold")

print(f"Detected MLIP: {ml_label}")
print(f"Read {len(frames)} frames, species: {', '.join(species_list)}\n")
print("Force RMSE/MAE/N by species:")
for sym in species_list:
    sf = species_forces[sym]
    ref_all = np.concatenate([sf["ref_fx"], sf["ref_fy"], sf["ref_fz"]])
    ml_all = np.concatenate([sf["ml_fx"], sf["ml_fy"], sf["ml_fz"]])
    combined_stats = RunningStats()
    combined_stats.update(ref_all, ml_all)
    print(
        f"  {sym:>4s}:  RMSE = {combined_stats.rmse:.6f}  "
        f"MAE = {combined_stats.mae:.6f} eV/Å  N = {combined_stats.count:,}"
    )
    for d_label, (ref_key, ml_key) in zip(dir_labels, dir_keys):
        ref_d = sf[ref_key]
        ml_d = sf[ml_key]
        stats = RunningStats()
        stats.update(ref_d, ml_d)
        print(
            f"         F_{d_label}:  RMSE = {stats.rmse:.6f}  "
            f"MAE = {stats.mae:.6f} eV/Å  N = {stats.count:,}"
        )

plt.show()
