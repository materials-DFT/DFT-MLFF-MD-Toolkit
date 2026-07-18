#!/usr/bin/env python3
"""Publication-style parity plots for evaluation/validation XYZ outputs."""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib-cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ase.io import iread, read  # noqa: E402
from matplotlib.ticker import AutoMinorLocator, MaxNLocator  # noqa: E402


MLIP_KEYS = (
    ("ALLEGRO", "ALLEGRO_energy", "ALLEGRO_forces", "ALLEGRO_stress"),
    ("MACE", "MACE_energy", "MACE_forces", "MACE_stress"),
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


@dataclass
class PairSampler:
    max_points: int
    rng: np.random.Generator
    seen: int = 0
    ref_values: np.ndarray | None = None
    pred_values: np.ndarray | None = None

    def add(self, reference: np.ndarray, prediction: np.ndarray) -> None:
        reference = np.asarray(reference, dtype=float).ravel()
        prediction = np.asarray(prediction, dtype=float).ravel()
        if reference.size == 0:
            return
        if self.max_points <= 0:
            self._append_unlimited(reference, prediction)
            self.seen += int(reference.size)
            return

        if self.ref_values is None:
            self.ref_values = np.empty(self.max_points, dtype=float)
            self.pred_values = np.empty(self.max_points, dtype=float)

        assert self.pred_values is not None
        fill_start = min(self.seen, self.max_points)
        fill_count = min(reference.size, max(0, self.max_points - fill_start))
        if fill_count:
            self.ref_values[fill_start : fill_start + fill_count] = reference[:fill_count]
            self.pred_values[fill_start : fill_start + fill_count] = prediction[:fill_count]

        remaining_ref = reference[fill_count:]
        remaining_pred = prediction[fill_count:]
        if remaining_ref.size:
            positions = np.arange(self.seen + fill_count, self.seen + reference.size)
            keep_probability = self.max_points / (positions + 1.0)
            chosen = self.rng.random(remaining_ref.size) < keep_probability
            if np.any(chosen):
                replace_at = self.rng.integers(0, self.max_points, size=int(np.sum(chosen)))
                self.ref_values[replace_at] = remaining_ref[chosen]
                self.pred_values[replace_at] = remaining_pred[chosen]
        self.seen += int(reference.size)

    def _append_unlimited(self, reference: np.ndarray, prediction: np.ndarray) -> None:
        if self.ref_values is None:
            self.ref_values = reference.copy()
            self.pred_values = prediction.copy()
            return
        assert self.pred_values is not None
        self.ref_values = np.concatenate([self.ref_values, reference])
        self.pred_values = np.concatenate([self.pred_values, prediction])

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self.ref_values is None or self.pred_values is None:
            empty = np.empty(0, dtype=float)
            return empty, empty
        if self.max_points <= 0:
            return self.ref_values, self.pred_values
        count = min(self.seen, self.max_points)
        return self.ref_values[:count].copy(), self.pred_values[:count].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot parity from evaluation-format XYZ files containing REF_* and "
            "ML model prediction fields."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="XYZ files or directories. Directories are searched for output.xyz files.",
    )
    parser.add_argument(
        "--ref-index",
        type=int,
        default=None,
        help=(
            "Frame index used as the relative-energy reference for each dataset. "
            "Default: frame with minimum REF energy/atom."
        ),
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=120000,
        help="Maximum scatter points per panel across all datasets. Default: 120000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed used when subsampling dense force/stress points.",
    )
    parser.add_argument(
        "--include-total-energy",
        action="store_true",
        help="Also include relative total-energy parity. Default plots E/atom, forces, and stress.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the figure. If omitted, only an interactive plot is shown.",
    )
    parser.add_argument(
        "--prediction-xyz",
        type=Path,
        default=None,
        help=(
            "Prediction XYZ to use when a training run directory is given. "
            "Default: auto-detect evaluation/**/evaluation_on_training_dataset/output.xyz."
        ),
    )
    parser.add_argument(
        "--all-evaluation-outputs",
        action="store_true",
        help=(
            "For directories, plot all discovered evaluation output.xyz files instead of "
            "the reserved training validation subset."
        ),
    )
    return parser.parse_args()


def detect_mlip(info: dict) -> tuple[str, str, str, str] | None:
    for label, energy_key, forces_key, stress_key in MLIP_KEYS:
        if energy_key in info:
            return label, energy_key, forces_key, stress_key
    return None


def is_eval_xyz(path: Path) -> bool:
    try:
        atoms = read(path, index=0)
    except Exception:
        return False
    if atoms is None:
        return False
    detected = detect_mlip(atoms.info)
    if detected is None:
        return False
    _, _, forces_key, _ = detected
    return (
        "REF_energy" in atoms.info
        and "REF_forces" in atoms.arrays
        and forces_key in atoms.arrays
    )


def discover_eval_xyzs(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if not path.exists():
            print(f"warning: path does not exist, skipping: {path}", file=sys.stderr)
            continue
        if path.is_file():
            if is_eval_xyz(path):
                found.append(path)
            else:
                print(f"warning: not an evaluation-format XYZ, skipping: {path}", file=sys.stderr)
            continue

        output_files = sorted(path.rglob("output.xyz"))
        candidates = output_files if output_files else sorted(path.rglob("*.xyz"))
        for candidate in candidates:
            if is_eval_xyz(candidate):
                found.append(candidate)

    return sorted(set(found))


def load_yaml(path: Path) -> dict:
    import yaml

    with path.open() as handle:
        return yaml.safe_load(handle)


def natural_version_key(path: Path) -> tuple:
    parts = []
    for chunk in path.as_posix().split("/"):
        if chunk.startswith("version_"):
            suffix = chunk.removeprefix("version_")
            if suffix.isdigit():
                parts.append((chunk[:8], int(suffix)))
                continue
        parts.append((chunk, -1))
    return tuple(parts)


def training_config_path(run_dir: Path) -> Path | None:
    hparams = sorted((run_dir / "logs").glob("**/hparams.yaml"), key=natural_version_key)
    if hparams:
        return hparams[0]
    config = run_dir / "allegro.yaml"
    return config if config.is_file() else None


def split_config_from_training_dir(run_dir: Path) -> dict | None:
    config_path = training_config_path(run_dir)
    if config_path is None:
        return None

    config = load_yaml(config_path)
    if "info_dict" in config:
        data_config = config.get("info_dict", {}).get("data", {})
    else:
        data_config = config.get("data", {})

    split_dataset = data_config.get("split_dataset")
    if not isinstance(split_dataset, dict):
        return None

    dataset_config = split_dataset.get("dataset", {})
    dataset_file = dataset_config.get("file_path")
    if not dataset_file or "${" in str(dataset_file):
        dataset_file = "all_frames.xyz"

    return {
        "config_path": config_path,
        "dataset_file": (run_dir / dataset_file).resolve(),
        "seed": int(data_config.get("seed", 12345)),
        "split": {key: value for key, value in split_dataset.items() if key != "dataset"},
    }


def count_xyz_frames(path: Path) -> int:
    return sum(1 for _ in iread(str(path), index=":"))


def split_indices(num_frames: int, split: dict, subset_name: str, seed: int) -> list[int]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required to exactly reconstruct NequIP's random validation split"
        ) from exc

    subset_names = list(split.keys())
    if subset_name not in subset_names:
        raise RuntimeError(f"training config has no '{subset_name}' split")
    lengths = [split[name] for name in subset_names]
    generator = torch.Generator().manual_seed(seed)
    subsets = torch.utils.data.random_split(range(num_frames), lengths, generator=generator)
    return list(subsets[subset_names.index(subset_name)].indices)


def detect_training_prediction_xyz(run_dir: Path) -> Path | None:
    candidates = sorted(
        (run_dir / "evaluation").glob("**/evaluation_on_training_dataset/output.xyz")
    )
    for candidate in candidates:
        if is_eval_xyz(candidate):
            return candidate.resolve()
    return None


def label_from_path(path: Path) -> str:
    parts = path.resolve().parts
    if "evaluation" in parts:
        index = parts.index("evaluation")
        label_parts = list(parts[index + 1 : -1])
        if label_parts:
            return "/".join(label_parts[-3:])

    parent_names = [part for part in path.parent.parts[-3:] if part]
    return "/".join(parent_names) if parent_names else path.stem


def stress_arrays(frame, ml_stress_key: str) -> tuple[np.ndarray, np.ndarray] | None:
    if ml_stress_key not in frame.info:
        return None
    ref_stress = frame.info.get("REF_stress", frame.info.get("stress"))
    if ref_stress is None:
        return None
    return np.asarray(ref_stress, dtype=float).ravel(), np.asarray(
        frame.info[ml_stress_key], dtype=float
    ).ravel()


def load_dataset(
    path: Path,
    ref_index: int | None,
    max_points: int,
    seed: int,
    include_indices: set[int] | None = None,
    label_override: str | None = None,
) -> dict:
    ml_keys = None
    n_frames = 0
    n_seen = 0
    n_atoms = []
    ref_energies = []
    pred_energies = []
    force_stats = RunningStats()
    stress_stats = RunningStats()
    force_sampler = PairSampler(max_points, np.random.default_rng(seed))
    stress_sampler = PairSampler(max_points, np.random.default_rng(seed + 1))

    for frame_index, frame in enumerate(iread(str(path), index=":")):
        n_seen += 1
        if include_indices is not None and frame_index not in include_indices:
            continue
        if ml_keys is None:
            ml_keys = detect_mlip(frame.info)
            if ml_keys is None:
                raise RuntimeError(f"could not detect ML prediction keys in {path}")
        label, energy_key, forces_key, stress_key = ml_keys

        if energy_key not in frame.info or "REF_energy" not in frame.info:
            raise RuntimeError(f"frame {frame_index} missing energy keys in {path}")
        if forces_key not in frame.arrays or "REF_forces" not in frame.arrays:
            raise RuntimeError(f"frame {frame_index} missing force keys in {path}")

        n_frames += 1
        n_atoms.append(len(frame))
        ref_energies.append(float(frame.info["REF_energy"]))
        pred_energies.append(float(frame.info[energy_key]))

        ref_forces = np.asarray(frame.arrays["REF_forces"], dtype=float).ravel()
        pred_forces = np.asarray(frame.arrays[forces_key], dtype=float).ravel()
        force_stats.update(ref_forces, pred_forces)
        force_sampler.add(ref_forces, pred_forces)

        stress = stress_arrays(frame, stress_key)
        if stress is not None:
            ref_stress, pred_stress = stress
            stress_stats.update(ref_stress, pred_stress)
            stress_sampler.add(ref_stress, pred_stress)

    if n_frames == 0:
        raise RuntimeError(f"no frames found in {path}")
    assert ml_keys is not None
    ml_label = ml_keys[0]

    ref_energies = np.asarray(ref_energies, dtype=float)
    pred_energies = np.asarray(pred_energies, dtype=float)
    n_atoms = np.asarray(n_atoms, dtype=float)
    ref_energy_per_atom = ref_energies / n_atoms
    pred_energy_per_atom = pred_energies / n_atoms

    if ref_index is None:
        reference_index = int(np.argmin(ref_energy_per_atom))
    else:
        if ref_index < 0 or ref_index >= n_frames:
            raise RuntimeError(
                f"--ref-index {ref_index} is outside valid range 0-{n_frames - 1} for {path}"
            )
        reference_index = ref_index

    delta_ref_total = ref_energies - ref_energies[reference_index]
    delta_pred_total = pred_energies - pred_energies[reference_index]
    delta_ref_per_atom = ref_energy_per_atom - ref_energy_per_atom[reference_index]
    delta_pred_per_atom = pred_energy_per_atom - pred_energy_per_atom[reference_index]

    energy_total_stats = RunningStats()
    energy_total_stats.update(delta_ref_total, delta_pred_total)
    energy_per_atom_stats = RunningStats()
    energy_per_atom_stats.update(delta_ref_per_atom, delta_pred_per_atom)
    ref_forces, pred_forces = force_sampler.arrays()
    ref_stress, pred_stress = stress_sampler.arrays()

    return {
        "path": path,
        "label": label_override or label_from_path(path),
        "ml_label": ml_label,
        "n_frames": n_frames,
        "n_source_frames": n_seen,
        "ref_index": reference_index,
        "delta_ref_total": delta_ref_total,
        "delta_pred_total": delta_pred_total,
        "delta_ref_per_atom": delta_ref_per_atom,
        "delta_pred_per_atom": delta_pred_per_atom,
        "ref_forces": ref_forces,
        "pred_forces": pred_forces,
        "ref_stress": ref_stress,
        "pred_stress": pred_stress,
        "total_energy_stats": energy_total_stats,
        "per_atom_energy_stats": energy_per_atom_stats,
        "force_stats": force_stats,
        "stress_stats": stress_stats,
        "has_stress": stress_stats.count > 0,
    }


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


def subsample_arrays(
    reference: np.ndarray,
    prediction: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or reference.size <= max_points:
        return reference, prediction
    chosen = rng.choice(reference.size, size=max_points, replace=False)
    return reference[chosen], prediction[chosen]


def padded_limits(*arrays: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([np.asarray(array, dtype=float).ravel() for array in arrays if array.size])
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


def aggregate_stats(datasets: list[dict], key: str) -> RunningStats:
    stats = RunningStats()
    for dataset in datasets:
        item = dataset[key]
        stats.count += item.count
        stats.abs_error_sum += item.abs_error_sum
        stats.sq_error_sum += item.sq_error_sum
    return stats


def add_stats_box(axis, stats: RunningStats, unit: str, extra: str = "") -> None:
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


def scatter_panel(
    axis,
    datasets: list[dict],
    x_key: str,
    y_key: str,
    stats_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    unit: str,
    max_points: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    references = [dataset[x_key] for dataset in datasets if dataset[x_key].size]
    predictions = [dataset[y_key] for dataset in datasets if dataset[y_key].size]
    low, high = padded_limits(*references, *predictions)

    point_budget = max_points if max_points > 0 else sum(array.size for array in references)
    points_per_dataset = max(1, point_budget // max(1, len(datasets)))
    for index, dataset in enumerate(datasets):
        reference = dataset[x_key]
        prediction = dataset[y_key]
        if reference.size == 0 or prediction.size == 0:
            continue
        reference_plot, prediction_plot = subsample_arrays(
            reference, prediction, points_per_dataset, rng
        )
        axis.scatter(
            reference_plot,
            prediction_plot,
            s=12,
            alpha=0.42,
            linewidths=0,
            color=PLOT_COLORS[index % len(PLOT_COLORS)],
            rasterized=True,
            label=dataset["label"],
        )

    axis.plot([low, high], [low, high], color="#222222", linewidth=1.1, linestyle="--", zorder=3)
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title, loc="left", fontsize=11, fontweight="semibold", pad=8)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(axis="both", color="#E2E2E2", linewidth=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_minor_locator(AutoMinorLocator())
    axis.tick_params(axis="both", which="major", direction="out", length=4, width=0.8)
    axis.tick_params(axis="both", which="minor", direction="out", length=2, width=0.6)
    add_stats_box(axis, aggregate_stats(datasets, stats_key), unit)


def plot_parity(
    datasets: list[dict],
    include_total_energy: bool,
    max_points: int,
    seed: int,
    save_path: Path | None,
) -> None:
    configure_style()
    has_stress = any(dataset["has_stress"] for dataset in datasets)
    panels = [
        (
            "delta_ref_per_atom",
            "delta_pred_per_atom",
            "per_atom_energy_stats",
            "Relative Energy per Atom",
            "DFT delta E/atom (eV/atom)",
            "ML delta E/atom (eV/atom)",
            "eV/atom",
        ),
        (
            "ref_forces",
            "pred_forces",
            "force_stats",
            "Forces",
            "DFT force component (eV/A)",
            "ML force component (eV/A)",
            "eV/A",
        ),
    ]
    if include_total_energy:
        panels.insert(
            1,
            (
                "delta_ref_total",
                "delta_pred_total",
                "total_energy_stats",
                "Relative Total Energy",
                "DFT delta E (eV)",
                "ML delta E (eV)",
                "eV",
            ),
        )
    if has_stress:
        panels.append(
            (
                "ref_stress",
                "pred_stress",
                "stress_stats",
                "Stress",
                "DFT stress component (eV/A^3)",
                "ML stress component (eV/A^3)",
                "eV/A^3",
            )
        )

    ncols = 3 if len(panels) == 3 else (2 if len(panels) > 1 else 1)
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.25 * ncols, 5.2 * nrows),
        constrained_layout=True,
    )
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for panel_index, (axis, panel) in enumerate(zip(axes_list, panels)):
        scatter_panel(axis, datasets, *panel, max_points=max_points, seed=seed + panel_index)

    for axis in axes_list[len(panels) :]:
        axis.set_visible(False)

    mlip_text = "/".join(sorted({dataset["ml_label"] for dataset in datasets}))
    title = f"Validation Parity ({mlip_text})"
    fig.suptitle(title, fontsize=13, fontweight="semibold")
    if len(datasets) > 1:
        handles, labels = axes_list[0].get_legend_handles_labels()
        label_map = dict(zip(labels, handles))
        fig.legend(
            label_map.values(),
            label_map.keys(),
            loc="outside lower center",
            ncols=min(3, len(label_map)),
            frameon=False,
            title="Dataset",
        )

    if save_path is not None:
        save_path = save_path.expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Wrote {save_path}")
    plt.show()


def print_summary(datasets: list[dict]) -> None:
    for dataset in datasets:
        print(f"\nDataset: {dataset['label']}")
        print(f"  File: {dataset['path']}")
        print(f"  MLIP: {dataset['ml_label']}")
        if dataset.get("n_source_frames") and dataset["n_source_frames"] != dataset["n_frames"]:
            print(f"  Frames: {dataset['n_frames']:,} of {dataset['n_source_frames']:,}")
        else:
            print(f"  Frames: {dataset['n_frames']:,}")
        print(f"  Energy reference frame: {dataset['ref_index']}")
        print(f"  RMSE delta E/atom: {dataset['per_atom_energy_stats'].rmse:.6f} eV/atom")
        print(f"  RMSE forces:       {dataset['force_stats'].rmse:.6f} eV/A")
        if dataset["has_stress"]:
            print(f"  RMSE stress:       {dataset['stress_stats'].rmse:.6f} eV/A^3")
        else:
            print("  RMSE stress:       N/A")


def main() -> int:
    args = parse_args()
    datasets = []
    xyz_files: list[Path] = []

    if not args.all_evaluation_outputs:
        for raw_path in args.paths:
            path = raw_path.expanduser().resolve()
            if not path.is_dir():
                continue
            split_config = split_config_from_training_dir(path)
            if split_config is None:
                continue

            prediction_xyz = (
                args.prediction_xyz.expanduser().resolve()
                if args.prediction_xyz is not None
                else detect_training_prediction_xyz(path)
            )
            if prediction_xyz is None:
                print(
                    "error: found a training config, but could not find predictions for "
                    "the full training dataset. Expected "
                    "evaluation/**/evaluation_on_training_dataset/output.xyz, or pass "
                    "--prediction-xyz.",
                    file=sys.stderr,
                )
                return 1
            if not is_eval_xyz(prediction_xyz):
                print(f"error: not an evaluation-format XYZ: {prediction_xyz}", file=sys.stderr)
                return 1

            dataset_file = split_config["dataset_file"]
            if not dataset_file.is_file():
                print(f"error: training dataset not found: {dataset_file}", file=sys.stderr)
                return 1
            num_frames = count_xyz_frames(dataset_file)
            val_indices = split_indices(
                num_frames,
                split_config["split"],
                "val",
                split_config["seed"],
            )
            label = f"{path.name or path} validation split"
            print(
                f"Using reserved validation split from {split_config['config_path']} "
                f"({len(val_indices):,}/{num_frames:,} frames)."
            )
            print(f"Using predictions from {prediction_xyz}.")
            try:
                datasets.append(
                    load_dataset(
                        prediction_xyz,
                        args.ref_index,
                        args.max_points,
                        args.seed,
                        include_indices=set(val_indices),
                        label_override=label,
                    )
                )
            except Exception as exc:
                print(f"error: failed to load validation split: {exc}", file=sys.stderr)
                return 1

    if not datasets:
        xyz_files = discover_eval_xyzs(args.paths)
        if not xyz_files:
            print("error: no evaluation-format XYZ files found", file=sys.stderr)
            return 1

        for xyz_file in xyz_files:
            try:
                datasets.append(load_dataset(xyz_file, args.ref_index, args.max_points, args.seed))
            except Exception as exc:
                print(f"warning: skipping {xyz_file}: {exc}", file=sys.stderr)

    if not datasets:
        print("error: no valid datasets to plot", file=sys.stderr)
        return 1

    print_summary(datasets)
    plot_parity(datasets, args.include_total_energy, args.max_points, args.seed, args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
