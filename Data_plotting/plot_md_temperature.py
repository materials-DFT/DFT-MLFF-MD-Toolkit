#!/usr/bin/env python3
"""Plot MD temperature from LAMMPS (log.lammps) and/or VASP (OUTCAR) files."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THERMO_HEADER_RE = re.compile(r"^\s*Step\s+.*\bTemp\b", re.IGNORECASE)
EKIN_LAT_RE = re.compile(r"kin\.\s+lattice\s+EKIN_LAT\s*=", re.IGNORECASE)
TEMP_EQ_RE = re.compile(
    r"^\s*temperature\s*=\s*([0-9.eE+-]+)\s*K?\s*$", re.IGNORECASE
)
EKIN_PAREN_RE = re.compile(
    r"kinetic\s+energy\s+EKIN\s*=.*?\(temperature\s+([0-9.eE+-]+)",
    re.IGNORECASE,
)
TIMESTEP_RE = re.compile(r"^\s*timestep\s+([0-9.eE+-]+)", re.IGNORECASE)
POTIM_RE = re.compile(r"^\s*POTIM\s*=\s*([0-9.eE+-]+)", re.IGNORECASE)


def find_md_logs(roots: list[Path]) -> list[Path]:
    """Recursively collect log.lammps and OUTCAR files under each root."""
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            print(f"warning: not a directory, skipping: {root}", file=sys.stderr)
            continue
        for name in ("log.lammps", "OUTCAR"):
            for path in sorted(root.rglob(name)):
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    found.append(resolved)
    return found


def parse_lammps_log(path: Path) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Return (x_values, temperatures, timestep_ps). x_values are LAMMPS steps."""
    steps: list[int] = []
    temps: list[float] = []
    timestep_ps: float | None = None

    with path.open(encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    for line in lines:
        match = TIMESTEP_RE.match(line)
        if match:
            timestep_ps = float(match.group(1))

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if not THERMO_HEADER_RE.match(line):
            idx += 1
            continue

        cols = line.split()
        try:
            step_col = cols.index("Step")
            temp_col = next(i for i, name in enumerate(cols) if name.lower() == "temp")
        except (ValueError, StopIteration):
            idx += 1
            continue

        idx += 1
        while idx < len(lines):
            data = lines[idx].strip()
            if not data or data.startswith("Loop time"):
                break
            parts = data.split()
            if len(parts) <= max(step_col, temp_col):
                break
            try:
                step = int(float(parts[step_col]))
                temp = float(parts[temp_col])
            except ValueError:
                break
            steps.append(step)
            temps.append(temp)
            idx += 1

    return np.asarray(steps, dtype=float), np.asarray(temps, dtype=float), timestep_ps


def parse_outcar(path: Path) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Return (x_values, temperatures, potim_fs). x_values are ionic-step indices."""
    temps_lat: list[float] = []
    temps_eq: list[float] = []
    temps_paren: list[float] = []
    potim_fs: float | None = None

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            potim_match = POTIM_RE.match(line)
            if potim_match:
                potim_fs = float(potim_match.group(1))

            if EKIN_LAT_RE.search(line):
                parts = line.split()
                try:
                    temps_lat.append(float(parts[-2]))
                except (IndexError, ValueError):
                    pass
                continue

            eq_match = TEMP_EQ_RE.match(line)
            if eq_match:
                temps_eq.append(float(eq_match.group(1)))
                continue

            paren_match = EKIN_PAREN_RE.search(line)
            if paren_match:
                temps_paren.append(float(paren_match.group(1)))

    if temps_lat:
        temps = temps_lat
    elif temps_eq:
        temps = temps_eq
    else:
        temps = temps_paren

    steps = np.arange(len(temps), dtype=float)
    return steps, np.asarray(temps, dtype=float), potim_fs


def load_temperature(path: Path) -> tuple[np.ndarray, np.ndarray, str, float | None]:
    """Load one log file. Returns x, temperature, engine label, time step."""
    if path.name == "log.lammps":
        x, temp, dt = parse_lammps_log(path)
        return x, temp, "LAMMPS", dt
    if path.name == "OUTCAR":
        x, temp, dt = parse_outcar(path)
        return x, temp, "VASP", dt
    raise ValueError(f"unsupported file type: {path}")


def to_time_axis(
    x: np.ndarray, dt: float | None, engine: str, use_time: bool
) -> tuple[np.ndarray, str]:
    if not use_time:
        return x, "step"

    if dt is None:
        print(
            f"warning: no timestep found for {engine} run; using step index",
            file=sys.stderr,
        )
        return x, "step"

    if engine == "LAMMPS":
        return x * dt, "time (ps)"
    return x * dt, "time (fs)"


def label_for(path: Path, roots: list[Path]) -> str:
    path = path.resolve()
    for root in sorted(roots, key=lambda item: len(item.parts), reverse=True):
        root = root.resolve()
        try:
            rel = path.relative_to(root)
            parent = rel.parent.as_posix()
            if parent == ".":
                return f"{path.name}"
            return f"{parent}/{path.name}"
        except ValueError:
            continue
    return str(path)


def subplot_grid(n: int) -> tuple[int, int]:
    if n == 1:
        return 1, 1
    ncols = min(3, int(np.ceil(np.sqrt(n))))
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def plot_temperatures(
    roots: list[Path],
    output: Path | None,
    show: bool,
    use_time: bool,
    title: str | None,
) -> int:
    logs = find_md_logs(roots)
    if not logs:
        print("error: no log.lammps or OUTCAR files found", file=sys.stderr)
        return 1

    series: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    for path in logs:
        try:
            x, temp, engine, dt = load_temperature(path)
        except OSError as exc:
            print(f"warning: could not read {path}: {exc}", file=sys.stderr)
            continue

        if temp.size == 0:
            print(f"warning: no temperature data in {path}", file=sys.stderr)
            continue

        x_plot, x_label = to_time_axis(x, dt, engine, use_time)
        label = f"{label_for(path, roots)} ({engine})"
        series.append((x_plot, temp, x_label, label))

    if not series:
        print("error: no temperature data parsed from any file", file=sys.stderr)
        return 1

    nrows, ncols = subplot_grid(len(series))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 3.2 * nrows),
        squeeze=False,
    )

    for idx, (x_plot, temp, x_label, label) in enumerate(series):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        ax.plot(x_plot, temp, linewidth=1.0)
        ax.set_title(label, fontsize=9)
        ax.set_ylabel("T (K)")
        ax.grid(True, alpha=0.3)
        if row == nrows - 1:
            ax.set_xlabel(x_label)

    for idx in range(len(series), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(title or "MD temperature")
    fig.tight_layout()

    if output is not None:
        fig.savefig(output, dpi=200)
        print(f"wrote {output}")

    if show or output is None:
        plt.show()
    else:
        plt.close(fig)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find LAMMPS log.lammps and/or VASP OUTCAR files "
            "under the given directories and plot temperature in one figure "
            "with one subplot per run."
        )
    )
    parser.add_argument(
        "directories",
        nargs="+",
        type=Path,
        help="one or more root directories to search recursively",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="save plot to this image file (e.g. temperature.png)",
    )
    parser.add_argument(
        "--time",
        action="store_true",
        help="use physical time on x-axis (ps for LAMMPS, fs for VASP)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="do not open an interactive plot window",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="custom plot title",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return plot_temperatures(
        roots=args.directories,
        output=args.output,
        show=not args.no_show,
        use_time=args.time,
        title=args.title,
    )


if __name__ == "__main__":
    raise SystemExit(main())
