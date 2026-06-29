#!/usr/bin/env python3
"""Plot cell volume vs step from VASP OUTCAR and/or LAMMPS log.lammps.

Each argument must be a directory tree to search. Under each tree, any OUTCAR
is treated as VASP and any log.lammps as LAMMPS (both may appear in the same tree).

Series whose path labels contain a temperature tag (e.g. 300K, 700K) are grouped
into separate subplots, one panel per temperature.

The figure is shown with matplotlib's default GUI backend (on Linux, typically
Tk/Qt drawing to the X11/Wayland display referenced by DISPLAY—e.g. ssh -X).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, NamedTuple

import matplotlib.pyplot as plt

VASP_VOLUME_RE = re.compile(
    r"volume\s+of\s+cell\s*:\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)
TEMP_RE = re.compile(r"(\d+)\s*K", re.IGNORECASE)


class VolumeSeries(NamedTuple):
    label: str
    steps: list[int]
    volumes: list[float]


def _find_files(root: Path, name: str) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        return []
    hits: list[Path] = []
    if (root / name).is_file():
        hits.append(root / name)
    for p in sorted(root.rglob(name)):
        if p.is_file() and p not in hits:
            hits.append(p)
    return hits


def parse_vasp_outcar(path: Path) -> list[float]:
    """Extract volumes in file order from OUTCAR (streaming)."""
    volumes: list[float] = []
    with path.open("r", errors="replace") as f:
        for line in f:
            m = VASP_VOLUME_RE.search(line)
            if m:
                volumes.append(float(m.group(1)))
    return volumes


def parse_vasp_root(root: Path) -> list[VolumeSeries]:
    outcars = _find_files(root, "OUTCAR")
    series: list[VolumeSeries] = []
    root = root.resolve()
    for oc in outcars:
        vols = parse_vasp_outcar(oc)
        try:
            short = str(oc.parent.relative_to(root))
        except ValueError:
            short = oc.parent.name
        label = f"VASP ({short})" if short != "." else "VASP"
        series.append(
            VolumeSeries(label=label, steps=list(range(len(vols))), volumes=vols)
        )
    return series


def _parse_thermo_header(line: str) -> int | None:
    parts = line.split()
    if not parts or parts[0] != "Step":
        return None
    try:
        return parts.index("Volume")
    except ValueError:
        return None


def parse_lammps_log(path: Path) -> VolumeSeries:
    """Parse the longest thermo block; Volume column from header."""
    blocks: list[list[tuple[int, float]]] = []
    current: list[tuple[int, float]] = []
    vol_col: int | None = None

    with path.open("r", errors="replace") as f:
        for line in f:
            if vol_col is not None and not line.strip():
                continue
            if vol_col is None:
                vc = _parse_thermo_header(line)
                if vc is not None:
                    vol_col = vc
                    current = []
                continue

            parts = line.split()
            if (
                len(parts) > vol_col
                and parts[0].isdigit()
                and _looks_float(parts[vol_col])
            ):
                step = int(parts[0])
                vol = float(parts[vol_col])
                current.append((step, vol))
            else:
                if current:
                    blocks.append(current)
                current = []
                vol_col = _parse_thermo_header(line)
                if vol_col is not None:
                    current = []

    if current:
        blocks.append(current)

    if not blocks:
        return VolumeSeries(label=str(path), steps=[], volumes=[])

    best = max(blocks, key=len)
    steps = [s for s, _ in best]
    volumes = [v for _, v in best]
    return VolumeSeries(label=str(path), steps=steps, volumes=volumes)


def _looks_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_lammps_root(root: Path) -> list[VolumeSeries]:
    logs = _find_files(root, "log.lammps")
    root = root.resolve()
    series: list[VolumeSeries] = []
    for lg in logs:
        data = parse_lammps_log(lg)
        if not data.volumes:
            continue
        try:
            short = str(lg.parent.relative_to(root))
        except ValueError:
            short = lg.parent.name
        label = f"LAMMPS ({short})" if short != "." else "LAMMPS"
        series.append(VolumeSeries(label=label, steps=data.steps, volumes=data.volumes))
    return series


def extract_temperature(label: str) -> str | None:
    """Return e.g. '300K' when the series path label contains a temperature tag."""
    m = TEMP_RE.search(label)
    return f"{m.group(1)}K" if m else None


def _temperature_sort_key(temp: str) -> int:
    m = re.match(r"(\d+)K", temp, re.IGNORECASE)
    return int(m.group(1)) if m else 10**9


def _legend_label(label: str, temp: str | None) -> str:
    """Drop redundant temperature from legend when it is in the subplot title."""
    if temp:
        for variant in (f" ({temp})", f"({temp})"):
            if variant in label:
                return label.replace(variant, "").strip()
    if label.startswith("LAMMPS"):
        return "LAMMPS"
    if label.startswith("VASP"):
        return "VASP"
    return label


def _plot_on_axes(
    ax: plt.Axes,
    series: Iterable[VolumeSeries],
    *,
    title: str,
    temp: str | None = None,
) -> None:
    for s in series:
        if not s.volumes:
            continue
        ax.plot(
            s.steps,
            s.volumes,
            label=_legend_label(s.label, temp),
            linewidth=0.8,
            alpha=0.9,
        )

    ax.set_xlabel("Step (LAMMPS: MD step; VASP: sequential volume sample index)")
    ax.set_ylabel(r"Cell volume ($\mathrm{\AA}^3$)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_series(all_series: Iterable[VolumeSeries]) -> None:
    series_list = [s for s in all_series if s.volumes]
    if not series_list:
        return

    by_temp: dict[str | None, list[VolumeSeries]] = {}
    for s in series_list:
        by_temp.setdefault(extract_temperature(s.label), []).append(s)

    temps = sorted((t for t in by_temp if t is not None), key=_temperature_sort_key)
    other = by_temp.get(None, [])

    if not temps:
        fig, ax = plt.subplots(figsize=(10, 5))
        _plot_on_axes(ax, series_list, title="NPT / MD cell volume")
        fig.tight_layout()
        plt.show()
        return

    n_panels = len(temps) + (1 if other else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4.5 * n_panels), squeeze=False)
    panel_axes = list(axes[:, 0])

    for ax, temp in zip(panel_axes, temps):
        _plot_on_axes(
            ax,
            by_temp[temp],
            title=f"NPT / MD cell volume — {temp}",
            temp=temp,
        )

    if other:
        _plot_on_axes(
            panel_axes[-1],
            other,
            title="NPT / MD cell volume (no temperature tag)",
        )

    fig.tight_layout()
    plt.show()


def collect_series(root: Path) -> list[VolumeSeries]:
    """All VASP and LAMMPS volume series found under root (auto-detected)."""
    root = root.resolve()
    out: list[VolumeSeries] = []
    out.extend(parse_vasp_root(root))
    out.extend(parse_lammps_root(root))
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Plot volume: pass one or more run directories; "
        "OUTCAR → VASP, log.lammps → LAMMPS."
    )
    p.add_argument(
        "directories",
        nargs="+",
        type=Path,
        metavar="DIR",
        help="Directory to search (recursively) for OUTCAR and log.lammps",
    )
    args = p.parse_args()

    combined: list[VolumeSeries] = []
    for d in args.directories:
        if not d.is_dir():
            print(f"Not a directory: {d}", file=sys.stderr)
            return 1
        found = collect_series(d)
        if not found:
            print(
                f"No OUTCAR or parseable log.lammps under {d}",
                file=sys.stderr,
            )
            return 1
        combined.extend(found)

    if not combined:
        print("No volume data to plot.", file=sys.stderr)
        return 1

    plot_series(combined)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
