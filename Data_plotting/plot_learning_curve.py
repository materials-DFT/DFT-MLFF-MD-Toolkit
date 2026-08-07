#!/usr/bin/env python3
"""Plot RMSE metrics over training epochs from discovered CSV log files."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib-cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import AutoMinorLocator, MaxNLocator  # noqa: E402


PLOT_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find training metrics CSV files under a directory and plot RMSE "
            "columns against epoch."
        )
    )
    parser.add_argument("directory", type=Path, help="Training run directory to scan")
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        help=(
            "Only plot metrics whose column name contains this text. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--derive-mse",
        action="store_true",
        help="Also plot epoch-level MSE columns as derived RMSE curves.",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Do not draw legends. Useful when plotting many runs.",
    )
    parser.add_argument(
        "--include-post-train-val",
        action="store_true",
        help=(
            "Include standalone validation rows written after training, such as "
            "the final validation of the best checkpoint."
        ),
    )
    return parser.parse_args()


def is_number(value: str | None) -> bool:
    if value is None or value == "":
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True


def natural_key(path: Path) -> list[int | str]:
    parts: list[int | str] = []
    for chunk in re.split(r"(\d+)", str(path)):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            parts.append(chunk)
    return parts


def discover_csv_files(root: Path) -> list[Path]:
    direct_metrics = root / "metrics.csv"
    if direct_metrics.is_file():
        return [direct_metrics]

    local_logs = root / "logs"
    if local_logs.is_dir():
        metrics_files = sorted(local_logs.rglob("metrics.csv"), key=natural_key)
        if metrics_files:
            return metrics_files

    metrics_files = sorted(root.rglob("metrics.csv"), key=natural_key)
    if metrics_files:
        return metrics_files

    csv_search_root = local_logs if local_logs.is_dir() else root
    candidates: list[Path] = []
    for path in sorted(csv_search_root.rglob("*.csv"), key=natural_key):
        try:
            with path.open(newline="") as handle:
                header = next(csv.reader(handle), [])
        except (OSError, UnicodeDecodeError):
            continue
        lowered = [column.lower() for column in header]
        if "epoch" in lowered and any("rmse" in column for column in lowered):
            candidates.append(path)
    return candidates


def wanted_column(column: str, filters: list[str]) -> bool:
    if not filters:
        return True
    lower = column.lower()
    return any(filter_text.lower() in lower for filter_text in filters)


def rmse_columns(fieldnames: list[str], filters: list[str]) -> list[tuple[str, str, bool]]:
    columns: list[tuple[str, str, bool]] = []
    for column in fieldnames:
        lower = column.lower()
        if "rmse" in lower and wanted_column(column, filters):
            columns.append((column, column, False))
    return columns


def derived_mse_columns(fieldnames: list[str], filters: list[str]) -> list[tuple[str, str, bool]]:
    columns: list[tuple[str, str, bool]] = []
    for column in fieldnames:
        lower = column.lower()
        if "mse" not in lower or "epoch" not in lower:
            continue
        label = column.replace("mse", "rmse").replace("MSE", "RMSE")
        if wanted_column(column, filters) or wanted_column(label, filters):
            columns.append((column, f"{label} (sqrt MSE)", True))
    return columns


def row_epoch(row: dict[str, str]) -> float | None:
    epoch_text = row.get("epoch")
    if not is_number(epoch_text):
        return None
    return float(epoch_text)


def read_series(
    path: Path,
    filters: list[str],
    derive_mse: bool,
    include_post_train_val: bool,
) -> dict[str, list[tuple[float, float]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return {}

        if "epoch" not in reader.fieldnames:
            return {}
        rows = list(reader)

        columns = rmse_columns(reader.fieldnames, filters)
        if derive_mse:
            columns.extend(derived_mse_columns(reader.fieldnames, filters))
        if not columns:
            return {}

        train_epoch_columns = [
            column for column in reader.fieldnames if column.startswith("train_loss_epoch/")
        ]
        trained_epochs = {
            epoch
            for row in rows
            if (epoch := row_epoch(row)) is not None
            and any(row.get(column) not in (None, "") for column in train_epoch_columns)
        }

        points_by_metric: dict[str, dict[float, float]] = defaultdict(dict)
        for row in rows:
            epoch = row_epoch(row)
            if epoch is None:
                continue
            for source_column, metric_label, should_sqrt in columns:
                if (
                    not include_post_train_val
                    and trained_epochs
                    and source_column.startswith("val")
                    and epoch not in trained_epochs
                ):
                    continue
                value_text = row.get(source_column)
                if not is_number(value_text):
                    continue
                value = float(value_text)
                if should_sqrt:
                    if value < 0:
                        continue
                    value = math.sqrt(value)
                points_by_metric[metric_label][epoch] = value

    series: dict[str, list[tuple[float, float]]] = {}
    for metric_label, points in points_by_metric.items():
        if points:
            series[metric_label] = sorted(points.items())
    return series


def collect_series(
    root: Path,
    filters: list[str],
    derive_mse: bool,
    include_post_train_val: bool,
) -> dict[str, list[tuple[str, list[float], list[float]]]]:
    points_by_metric: dict[str, dict[float, float]] = defaultdict(dict)
    for path in discover_csv_files(root):
        file_series = read_series(path, filters, derive_mse, include_post_train_val)
        for metric_label, points in file_series.items():
            for epoch, value in points:
                points_by_metric[metric_label][epoch] = value

    label = root.name or str(root)
    data: dict[str, list[tuple[str, list[float], list[float]]]] = defaultdict(list)
    for metric_label, points in points_by_metric.items():
        sorted_points = sorted(points.items())
        epochs = [epoch for epoch, _ in sorted_points]
        values = [value for _, value in sorted_points]
        data[metric_label].append((label, epochs, values))
    return data


def metric_title(metric: str) -> str:
    suffix = " (sqrt MSE)" if metric.endswith(" (sqrt MSE)") else ""
    base_metric = metric.removesuffix(" (sqrt MSE)")
    short_name = base_metric.split("/")[-1]
    titles = {
        "forces_rmse": "Forces RMSE",
        "per_atom_energy_rmse": "Per-Atom Energy RMSE",
        "stress_rmse": "Stress RMSE",
    }
    return titles.get(short_name, short_name.replace("_", " ").title()) + suffix


def is_total_energy_metric(metric: str) -> bool:
    short_name = metric.removesuffix(" (sqrt MSE)").split("/")[-1].lower()
    return "total_energy" in short_name or "total energy" in short_name


def metric_sort_key(metric: str) -> tuple[int, str]:
    short_name = metric.removesuffix(" (sqrt MSE)").split("/")[-1].lower()
    if "per_atom_energy" in short_name:
        return (0, short_name)
    if "forces" in short_name:
        return (1, short_name)
    if "stress" in short_name:
        return (2, short_name)
    return (3, short_name)


def configure_plot_style() -> None:
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


def plot_rmse(
    data: dict[str, list[tuple[str, list[float], list[float]]]],
    no_legend: bool,
) -> None:
    configure_plot_style()
    metrics = sorted(
        (metric for metric in data if not is_total_energy_metric(metric)),
        key=metric_sort_key,
    )
    ncols = max(1, len(metrics))
    nrows = 1
    fig_width = max(7.6, 4.2 * ncols)
    fig_height = 4.2
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        sharex=True,
        constrained_layout=True,
    )
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    show_legend = not no_legend and any(len(series) > 1 for series in data.values())

    for axis, metric in zip(axes_list, metrics):
        series_for_metric = sorted(data[metric], key=lambda item: item[0])
        for series_index, (source_label, epochs, values) in enumerate(series_for_metric):
            marker = "o" if len(epochs) <= 35 else None
            axis.plot(
                epochs,
                values,
                color=PLOT_COLORS[series_index % len(PLOT_COLORS)],
                linewidth=2.0,
                marker=marker,
                markersize=3.5,
                solid_capstyle="round",
                label=source_label,
            )
        axis.set_title(metric_title(metric), loc="left", fontsize=11, fontweight="semibold", pad=8)
        axis.set_ylabel("RMSE")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75)
        axis.grid(axis="x", color="#EAEAEA", linewidth=0.6, alpha=0.55)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.yaxis.set_minor_locator(AutoMinorLocator())
        axis.tick_params(axis="both", which="major", direction="out", length=4, width=0.8)
        axis.tick_params(axis="y", which="minor", direction="out", length=2, width=0.6)
        axis.margins(x=0.015, y=0.08)
        if show_legend:
            axis.legend(frameon=False, loc="upper right", borderaxespad=0.4)
        axis.set_xlabel("Epoch")

    fig.suptitle("RMSE over Training Epochs", fontsize=13, fontweight="semibold")
    fig.align_ylabels([axis for axis in axes_list[: len(metrics)]])
    plt.show()


def main() -> int:
    args = parse_args()
    root = args.directory.expanduser().resolve()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2

    data = collect_series(root, args.metric, args.derive_mse, args.include_post_train_val)
    if not data:
        print(
            f"error: no epoch/RMSE data found under {root}. "
            "Expected metrics.csv files with an epoch column and RMSE columns.",
            file=sys.stderr,
        )
        return 1

    plot_data = {metric: series for metric, series in data.items() if not is_total_energy_metric(metric)}
    if not plot_data:
        print(
            f"error: no plottable RMSE metrics found under {root} after excluding total energy.",
            file=sys.stderr,
        )
        return 1

    plot_rmse(plot_data, args.no_legend)
    n_series = sum(len(series) for series in plot_data.values())
    print(f"Plotted {len(plot_data)} RMSE metric(s) from {n_series} series.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
