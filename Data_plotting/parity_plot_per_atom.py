#!/usr/bin/env python3
"""Publication-style parity plots for eval XYZ outputs."""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib-cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ase.io import iread, read  # noqa: E402
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


def is_eval_xyz(path):
    try:
        first = read(path, index=0)
    except Exception:
        return False
    if first is None:
        return False
    detected = detect_mlip(first.info)
    if detected is None:
        return False
    _, _, ml_forces_key, _ = detected
    return "REF_energy" in first.info and "REF_forces" in first.arrays and ml_forces_key in first.arrays


def discover_eval_xyzs(paths):
    xyzs = []
    for raw in paths:
        candidate = Path(raw).expanduser().resolve()
        if not candidate.exists():
            print(f"Warning: path does not exist, skipping: {candidate}")
            continue
        if candidate.is_file():
            if is_eval_xyz(candidate):
                xyzs.append(candidate)
            else:
                print(f"Warning: not an evaluation-format XYZ, skipping: {candidate}")
            continue
        for xyz_file in sorted(candidate.rglob("*.xyz")):
            if is_eval_xyz(xyz_file):
                xyzs.append(xyz_file)
    return sorted(set(xyzs))


def label_from_path(xyz_path):
    """Short label: up to three parent directory names above the XYZ's folder (outer -> inner)."""
    resolved = xyz_path.resolve()
    names = []
    cur = resolved.parent
    for _ in range(3):
        parent = cur.parent
        if parent == cur:
            break
        if parent.name:
            names.append(parent.name)
        cur = parent
    if not names:
        return str(resolved.parent)
    return "/".join(reversed(names))


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


def aggregate_stats(datasets, key):
    stats = RunningStats()
    for dataset in datasets:
        item = dataset[key]
        stats.count += item.count
        stats.abs_error_sum += item.abs_error_sum
        stats.sq_error_sum += item.sq_error_sum
    return stats


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


def load_dataset(xyz_path, forced_ref_index=None):
    iterator = iread(str(xyz_path), index=":")
    try:
        first_frame = next(iterator)
    except StopIteration:
        raise RuntimeError(f"No frames in {xyz_path}")

    detected = detect_mlip(first_frame.info)
    if detected is None:
        raise RuntimeError(f"Could not detect MLIP type in {xyz_path}")
    ml_label, ml_energy_key, ml_forces_key, ml_stress_key = detected

    ml_energies, ref_energies = [], []
    n_atoms = []
    force_stats = RunningStats()
    stress_stats = RunningStats()
    ref_forces_plot_chunks = []
    ml_forces_plot_chunks = []
    ref_stresses_plot_chunks = []
    ml_stresses_plot_chunks = []
    force_plot_points = 0
    stress_plot_points = 0

    def process_frame(frame, i):
        if ml_energy_key not in frame.info or "REF_energy" not in frame.info:
            raise RuntimeError(
                f"Frame {i} in {xyz_path} missing '{ml_energy_key}' or 'REF_energy'."
            )
        if ml_forces_key not in frame.arrays or "REF_forces" not in frame.arrays:
            raise RuntimeError(
                f"Frame {i} in {xyz_path} missing '{ml_forces_key}' or 'REF_forces'."
            )
        ml_energies.append(frame.info[ml_energy_key])
        ref_energies.append(frame.info["REF_energy"])
        n_atoms.append(len(frame))
        ml_f = frame.arrays[ml_forces_key].ravel()
        ref_f = frame.arrays["REF_forces"].ravel()
        force_stats.update(ref_f, ml_f)
        return ml_f, ref_f

    def update_plot_buffers(
        ref_vals, ml_vals, ref_chunks, ml_chunks, points_so_far, max_points, rng
    ):
        if max_points is None:
            ref_chunks.append(ref_vals.copy())
            ml_chunks.append(ml_vals.copy())
            return points_so_far + ref_vals.size
        remaining = max_points - points_so_far
        if remaining <= 0:
            return points_so_far
        if ref_vals.size <= remaining:
            ref_chunks.append(ref_vals.copy())
            ml_chunks.append(ml_vals.copy())
            return points_so_far + ref_vals.size
        idx = rng.choice(ref_vals.size, size=remaining, replace=False)
        ref_chunks.append(ref_vals[idx].copy())
        ml_chunks.append(ml_vals[idx].copy())
        return points_so_far + remaining

    rng = np.random.default_rng(12345)
    max_force_points = 200000
    max_stress_points = 200000

    # Process first frame then remainder.
    ml_f0, ref_f0 = process_frame(first_frame, 0)
    force_plot_points = update_plot_buffers(
        ref_f0,
        ml_f0,
        ref_forces_plot_chunks,
        ml_forces_plot_chunks,
        force_plot_points,
        max_force_points,
        rng,
    )
    if ml_stress_key in first_frame.info and "REF_stress" in first_frame.info:
        ml_s0 = np.asarray(first_frame.info[ml_stress_key]).ravel()
        ref_s0 = np.asarray(first_frame.info["REF_stress"]).ravel()
        stress_stats.update(ref_s0, ml_s0)
        stress_plot_points = update_plot_buffers(
            ref_s0,
            ml_s0,
            ref_stresses_plot_chunks,
            ml_stresses_plot_chunks,
            stress_plot_points,
            max_stress_points,
            rng,
        )

    for i, frame in enumerate(iterator, start=1):
        ml_f, ref_f = process_frame(frame, i)
        force_plot_points = update_plot_buffers(
            ref_f,
            ml_f,
            ref_forces_plot_chunks,
            ml_forces_plot_chunks,
            force_plot_points,
            max_force_points,
            rng,
        )
        if ml_stress_key in frame.info and "REF_stress" in frame.info:
            ml_s = np.asarray(frame.info[ml_stress_key]).ravel()
            ref_s = np.asarray(frame.info["REF_stress"]).ravel()
            stress_stats.update(ref_s, ml_s)
            stress_plot_points = update_plot_buffers(
                ref_s,
                ml_s,
                ref_stresses_plot_chunks,
                ml_stresses_plot_chunks,
                stress_plot_points,
                max_stress_points,
                rng,
            )

    ml_energies = np.asarray(ml_energies)
    ref_energies = np.asarray(ref_energies)
    n_atoms = np.asarray(n_atoms)
    if ref_forces_plot_chunks:
        ref_forces_plot = np.concatenate(ref_forces_plot_chunks)
        ml_forces_plot = np.concatenate(ml_forces_plot_chunks)
    else:
        ref_forces_plot = np.empty(0, dtype=float)
        ml_forces_plot = np.empty(0, dtype=float)
    has_stress = stress_stats.count > 0
    if has_stress:
        if ref_stresses_plot_chunks:
            ref_stresses_plot = np.concatenate(ref_stresses_plot_chunks)
            ml_stresses_plot = np.concatenate(ml_stresses_plot_chunks)
        else:
            ref_stresses_plot = np.empty(0, dtype=float)
            ml_stresses_plot = np.empty(0, dtype=float)

    ref_per_atom = ref_energies / n_atoms
    ml_per_atom = ml_energies / n_atoms

    if forced_ref_index is not None:
        if forced_ref_index < 0 or forced_ref_index >= len(ref_per_atom):
            raise RuntimeError(
                f"Invalid --ref-index {forced_ref_index} for {xyz_path}, "
                f"must be in [0, {len(ref_per_atom)-1}]"
            )
        ref_idx = forced_ref_index
    else:
        ref_idx = int(np.argmin(ref_per_atom))

    delta_ref_per_atom = ref_per_atom - ref_per_atom[ref_idx]
    delta_ml_per_atom = ml_per_atom - ml_per_atom[ref_idx]

    per_atom_energy_stats = RunningStats()
    per_atom_energy_stats.update(delta_ref_per_atom, delta_ml_per_atom)

    data = {
        "xyz_path": xyz_path,
        "label": label_from_path(xyz_path),
        "ml_label": ml_label,
        "ref_idx": ref_idx,
        "n_frames": len(ref_energies),
        "delta_ref_per_atom": delta_ref_per_atom,
        "delta_ml_per_atom": delta_ml_per_atom,
        "ref_forces_plot": ref_forces_plot,
        "ml_forces_plot": ml_forces_plot,
        "has_stress": has_stress,
        "per_atom_energy_stats": per_atom_energy_stats,
        "force_stats": force_stats,
        "stress_stats": stress_stats,
    }
    if has_stress:
        data["ref_stresses_plot"] = ref_stresses_plot
        data["ml_stresses_plot"] = ml_stresses_plot
    return data


def _load_dataset_wrapper(payload):
    xyz, ref_index = payload
    try:
        return load_dataset(xyz, forced_ref_index=ref_index), None
    except Exception as exc:
        return None, f"Warning: skipping {xyz}: {exc}"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot parity across one or more eval datasets. Inputs can be files or "
            "directories. Directories are searched recursively for eval-format XYZs."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more XYZ files or directories containing eval-format XYZ files.",
    )
    parser.add_argument(
        "--ref-index",
        type=int,
        default=None,
        help=(
            "Index (0-based) used as relative-energy reference for each dataset. "
            "Default: per-dataset minimum REF energy/atom."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of processes used to load datasets in parallel. "
            "Use >1 to speed up loading when multiple files are provided."
        ),
    )
    args = parser.parse_args()

    xyz_files = discover_eval_xyzs(args.paths)
    if not xyz_files:
        print("No evaluation-format XYZ files found in the provided inputs.")
        sys.exit(1)

    datasets = []
    workers = max(1, int(args.workers))
    if workers == 1 or len(xyz_files) == 1:
        for xyz in xyz_files:
            ds, warn = _load_dataset_wrapper((xyz, args.ref_index))
            if warn:
                print(warn)
                continue
            datasets.append(ds)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            payloads = [(xyz, args.ref_index) for xyz in xyz_files]
            for ds, warn in executor.map(_load_dataset_wrapper, payloads):
                if warn:
                    print(warn)
                    continue
                datasets.append(ds)

    if not datasets:
        print("No valid datasets to plot after loading.")
        sys.exit(1)

    configure_style()

    ml_types = sorted({d["ml_label"] for d in datasets})
    has_stress_any = any(d["has_stress"] for d in datasets)
    has_stress_all = all(d["has_stress"] for d in datasets)
    include_stress = has_stress_any
    if has_stress_any and not has_stress_all:
        print("Note: some datasets have stress, some do not. Plotting stress where available.")

    ncols = 3 if include_stress else 2
    nrows = 1
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.7 * ncols, 4.6),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    colors = plt.cm.tab20(np.linspace(0, 1, max(2, len(datasets))))

    # Relative Per-Atom Energy
    ax = axes[0, 0]
    for i, ds in enumerate(datasets):
        ax.scatter(
            ds["delta_ref_per_atom"],
            ds["delta_ml_per_atom"],
            alpha=0.45,
            s=14,
            color=colors[i],
            label=ds["label"],
            rasterized=True,
        )
    lims = padded_limits(
        *(ds["delta_ref_per_atom"] for ds in datasets),
        *(ds["delta_ml_per_atom"] for ds in datasets),
    )
    ax.plot(lims, lims, color="#222222", linewidth=1.1, linestyle="--", zorder=3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("DFT ΔE/atom (eV/atom)")
    ax.set_ylabel("ML ΔE/atom (eV/atom)")
    ax.set_title("Relative Per-Atom Energy Parity", loc="left", fontsize=11, fontweight="semibold", pad=8)
    finalize_axis(ax)
    add_stats_box(ax, aggregate_stats(datasets, "per_atom_energy_stats"), "eV/atom")

    # Force
    ax = axes[0, 1]
    for i, ds in enumerate(datasets):
        ax.scatter(
            ds["ref_forces_plot"],
            ds["ml_forces_plot"],
            alpha=0.40,
            s=10,
            color=colors[i],
            label=ds["label"],
            rasterized=True,
        )
    lims = padded_limits(
        *(ds["ref_forces_plot"] for ds in datasets), *(ds["ml_forces_plot"] for ds in datasets)
    )
    ax.plot(lims, lims, color="#222222", linewidth=1.1, linestyle="--", zorder=3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("DFT force component (eV/Å)")
    ax.set_ylabel("ML force component (eV/Å)")
    ax.set_title("Force Parity", loc="left", fontsize=11, fontweight="semibold", pad=8)
    finalize_axis(ax)
    add_stats_box(ax, aggregate_stats(datasets, "force_stats"), "eV/Å")

    context_text = ", ".join(ds["label"] for ds in datasets)
    ml_text = "/".join(ml_types)
    wrapped_context = textwrap.fill(context_text, width=120)

    # Stress
    if include_stress:
        ax = axes[0, 2]
        stressable = [ds for ds in datasets if ds["has_stress"]]
        for i, ds in enumerate(stressable):
            ax.scatter(
                ds["ref_stresses_plot"],
                ds["ml_stresses_plot"],
                alpha=0.40,
                s=10,
                color=colors[i],
                label=ds["label"],
                rasterized=True,
            )
        lims = padded_limits(
            *(ds["ref_stresses_plot"] for ds in stressable),
            *(ds["ml_stresses_plot"] for ds in stressable),
        )
        ax.plot(lims, lims, color="#222222", linewidth=1.1, linestyle="--", zorder=3)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(r"DFT stress component (eV/Å$^3$)")
        ax.set_ylabel(r"ML stress component (eV/Å$^3$)")
        ax.set_title("Stress Parity (all components)", loc="left", fontsize=11, fontweight="semibold", pad=8)
        finalize_axis(ax)
        add_stats_box(ax, aggregate_stats(stressable, "stress_stats"), "eV/Å$^3$")

    fig.suptitle(
        f"Parity plots for {len(datasets)} dataset(s) | MLIP: {ml_text}\n{wrapped_context}",
        fontsize=12,
        fontweight="semibold",
    )

    if len(datasets) > 1:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        label_map = dict(zip(labels, handles))
        fig.legend(
            label_map.values(),
            label_map.keys(),
            loc="outside lower center",
            ncols=min(3, len(label_map)),
            frameon=False,
            title="Dataset",
        )

    for ds in datasets:
        print(f"\nDataset: {ds['label']}")
        print(f"  File: {ds['xyz_path']}")
        print(f"  MLIP: {ds['ml_label']}")
        print(f"  Reference frame index: {ds['ref_idx']}")
        print(
            f"  ΔE/atom:    RMSE = {ds['per_atom_energy_stats'].rmse:.6f} eV/atom, "
            f"MAE = {ds['per_atom_energy_stats'].mae:.6f} eV/atom, "
            f"N = {ds['per_atom_energy_stats'].count:,}"
        )
        print(
            f"  Forces:     RMSE = {ds['force_stats'].rmse:.6f} eV/Å, "
            f"MAE = {ds['force_stats'].mae:.6f} eV/Å, "
            f"N = {ds['force_stats'].count:,}"
        )
        if ds["has_stress"]:
            print(
                f"  Stress:     RMSE = {ds['stress_stats'].rmse:.6f} eV/Å³, "
                f"MAE = {ds['stress_stats'].mae:.6f} eV/Å³, "
                f"N = {ds['stress_stats'].count:,}"
            )
        else:
            print("  Stress:     N/A")

    plt.show()


if __name__ == "__main__":
    main()
