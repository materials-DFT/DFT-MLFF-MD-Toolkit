#!/usr/bin/env python3
"""Plot thermodynamic properties: volume, temperature, and total energy.

Compares MD simulations from VASP (DFT-MD) and LAMMPS (MLFF MD).
Supports VASP (OUTCAR, OSZICAR) and LAMMPS (log.lammps) formats.
"""

import sys
import os
import re
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)


def read_lammps_thermo(directory, skip=0, max_frames=None):
    """Parse LAMMPS log file for thermodynamic data.
    
    Expects thermo_style with: step temp press vol etotal (or similar).
    Returns arrays for step, temperature, volume, and total energy.
    """
    d = Path(directory)
    log_path = d / "log.lammps"
    if not log_path.exists():
        raise FileNotFoundError(f"No log.lammps in {d}")
    
    steps, temps, vols, energies = [], [], [], []
    dt_fs = None
    
    in_thermo = False
    header_cols = []
    
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
                        dt_fs = dt_raw * 1000.0
                    elif units == "real":
                        dt_fs = dt_raw
                    else:
                        dt_fs = dt_raw
    
    if dt_fs is None:
        dt_fs = 1.0
    
    with open(log_path) as fh:
        for line in fh:
            if line.strip().startswith("Step") and "Temp" in line:
                header_cols = line.split()
                in_thermo = True
                continue
            
            if in_thermo:
                if line.startswith("Loop time") or line.startswith("ERROR") or not line.strip():
                    in_thermo = False
                    continue
                
                parts = line.split()
                if len(parts) < len(header_cols):
                    in_thermo = False
                    continue
                
                try:
                    vals = [float(p) for p in parts[:len(header_cols)]]
                except ValueError:
                    in_thermo = False
                    continue
                
                col_map = {h.lower(): i for i, h in enumerate(header_cols)}
                
                step = int(vals[col_map.get("step", 0)])
                temp = vals[col_map.get("temp", 1)]
                vol = vals[col_map.get("vol", col_map.get("volume", 3))]
                etot = vals[col_map.get("toteng", col_map.get("etotal", 4))]
                
                steps.append(step)
                temps.append(temp)
                vols.append(vol)
                energies.append(etot)
    
    steps = np.array(steps)
    temps = np.array(temps)
    vols = np.array(vols)
    energies = np.array(energies)
    
    if skip > 0:
        steps = steps[skip:]
        temps = temps[skip:]
        vols = vols[skip:]
        energies = energies[skip:]
    
    if max_frames is not None and len(steps) > max_frames:
        steps = steps[:max_frames]
        temps = temps[:max_frames]
        vols = vols[:max_frames]
        energies = energies[:max_frames]
    
    time_ps = steps * dt_fs * 1e-3
    
    print(f"    {len(steps)} steps, dt={dt_fs} fs (LAMMPS)")
    return {
        "time_ps": time_ps,
        "step": steps,
        "temperature": temps,
        "volume": vols,
        "energy": energies,
        "dt_fs": dt_fs,
    }


def read_vasp_thermo(directory, skip=0, max_frames=None):
    """Parse VASP OSZICAR and OUTCAR for thermodynamic data.
    
    OSZICAR contains ionic step summary lines with T, E, F, E0, EK.
    OUTCAR contains volume of cell per ionic step.
    
    Returns arrays for step, temperature, volume, and total energy.
    """
    d = Path(directory)
    oszicar_path = d / "OSZICAR"
    outcar_path = d / "OUTCAR"
    
    if not oszicar_path.exists():
        raise FileNotFoundError(f"No OSZICAR in {d}")
    
    dt_fs = 1.0
    incar_path = d / "INCAR"
    if incar_path.exists():
        with open(incar_path) as fh:
            for line in fh:
                m = re.match(r"\s*POTIM\s*=\s*([0-9.eE+-]+)", line)
                if m:
                    dt_fs = float(m.group(1))
                    break
    
    if outcar_path.exists() and dt_fs == 1.0:
        with open(outcar_path) as fh:
            for line in fh:
                m = re.match(r"\s*POTIM\s*=\s*([0-9.eE+-]+)", line)
                if m:
                    dt_fs = float(m.group(1))
                    break
    
    steps, temps, energies = [], [], []
    ionic_step_pattern = re.compile(
        r"^\s*(\d+)\s+T=\s*([0-9.eE+-]+)\.\s+E=\s*([0-9.eE+-]+)"
    )
    
    with open(oszicar_path) as fh:
        for line in fh:
            m = ionic_step_pattern.match(line)
            if m:
                step = int(m.group(1))
                temp = float(m.group(2))
                energy = float(m.group(3))
                steps.append(step)
                temps.append(temp)
                energies.append(energy)
    
    volumes = []
    if outcar_path.exists():
        vol_pattern = re.compile(r"volume of cell\s*:\s*([0-9.eE+-]+)")
        with open(outcar_path) as fh:
            first_vol = True
            for line in fh:
                m = vol_pattern.search(line)
                if m:
                    if first_vol:
                        first_vol = False
                        continue
                    volumes.append(float(m.group(1)))
    
    steps = np.array(steps)
    temps = np.array(temps)
    energies = np.array(energies)
    
    if len(volumes) >= len(steps):
        volumes = np.array(volumes[:len(steps)])
    elif len(volumes) > 0:
        volumes = np.array(volumes)
        if len(volumes) < len(steps):
            steps = steps[:len(volumes)]
            temps = temps[:len(volumes)]
            energies = energies[:len(volumes)]
    else:
        volumes = np.full(len(steps), np.nan)
    
    if skip > 0:
        steps = steps[skip:]
        temps = temps[skip:]
        volumes = volumes[skip:]
        energies = energies[skip:]
    
    if max_frames is not None and len(steps) > max_frames:
        steps = steps[:max_frames]
        temps = temps[:max_frames]
        volumes = volumes[:max_frames]
        energies = energies[:max_frames]
    
    time_ps = steps * dt_fs * 1e-3
    
    print(f"    {len(steps)} steps, dt={dt_fs} fs (VASP)")
    return {
        "time_ps": time_ps,
        "step": steps,
        "temperature": temps,
        "volume": volumes,
        "energy": energies,
        "dt_fs": dt_fs,
    }


def _infer_md_engine(directory):
    """Classify a run as VASP (DFT-MD) vs LAMMPS (MLFF MD)."""
    d = Path(directory)
    has_lammps = (d / "log.lammps").exists() or any(d.glob("in.*"))
    has_vasp = (d / "OSZICAR").exists() or (d / "OUTCAR").exists() or (d / "INCAR").exists()
    
    if has_lammps and not has_vasp:
        return "LAMMPS"
    if has_vasp and not has_lammps:
        return "VASP"
    if has_lammps and has_vasp:
        return "LAMMPS"
    return "unknown"


def _legend_prefix_from_engine(md_engine):
    """Map detected engine to a short plot prefix."""
    if md_engine == "VASP":
        return "DFT"
    if md_engine == "LAMMPS":
        return "MLFF"
    return None


def read_thermo(directory, skip=0, max_frames=None):
    """Auto-detect format and read thermodynamic data."""
    d = Path(directory)
    md_engine = _infer_md_engine(d)
    
    if md_engine == "LAMMPS":
        data = read_lammps_thermo(d, skip=skip, max_frames=max_frames)
    elif md_engine == "VASP":
        data = read_vasp_thermo(d, skip=skip, max_frames=max_frames)
    else:
        if (d / "log.lammps").exists():
            data = read_lammps_thermo(d, skip=skip, max_frames=max_frames)
            md_engine = "LAMMPS"
        elif (d / "OSZICAR").exists():
            data = read_vasp_thermo(d, skip=skip, max_frames=max_frames)
            md_engine = "VASP"
        else:
            raise FileNotFoundError(
                f"No log.lammps or OSZICAR in {d}\n"
                "  Check that the directory contains simulation output."
            )
    
    data["md_engine"] = md_engine
    return data


def _has_simulation(d):
    """True if *d* looks like it contains a simulation."""
    d = Path(d)
    if (d / "OUTCAR").exists() or (d / "INCAR").exists():
        return True
    if (d / "OSZICAR").exists() or (d / "POSCAR").exists():
        return True
    if (d / "log.lammps").exists() or any(d.glob("in.*")):
        return True
    return False


def _temp_sort_key(p):
    """Sort key that extracts leading number from dir name."""
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


def _finalize_figure(fig, save, maximize_window, skip_tight_layout=False):
    """tight_layout, optional save, show or close."""
    import matplotlib.pyplot as plt

    if not skip_tight_layout:
        try:
            fig.tight_layout(rect=(0.055, 0.03, 0.985, 0.91))
        except Exception:
            pass
    
    display = os.environ.get("DISPLAY")
    if save:
        out_path = Path(save)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
        print(f"Saved figure -> {out_path.resolve()}")
    if not save or display:
        if maximize_window and display:
            fig.canvas.draw()
            _maximize_figure_window(fig)
        plt.show()
    else:
        plt.close(fig)


def compute_running_average(data, window):
    """Compute running average with given window size."""
    if window <= 1 or len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def _volume_decimal_places(vmin, vmax):
    """Choose y-tick precision so distinct volumes read as 737.9972 vs 738."""
    span = abs(vmax - vmin)
    if span < 1e-12:
        text = f"{vmax:.8f}".rstrip("0").rstrip(".")
        if "." not in text:
            return 0
        return min(len(text.split(".")[1]), 4)
    for decimals in range(5):
        if round(vmin, decimals) != round(vmax, decimals):
            return decimals
    return 4


def _configure_volume_axis(ax, vol_arrays):
    """Use a broad y-range for near-constant NVT volumes and label true values."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

    finite = np.concatenate(
        [arr[np.isfinite(arr)] for arr in vol_arrays if len(arr) > 0]
    )
    if finite.size == 0:
        return

    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    vmean = float(np.mean(finite))
    span = vmax - vmin

    # Matplotlib autoscale zooms in when span is tiny; keep a readable context window.
    if span < max(1.0, abs(vmean) * 1e-4):
        pad = max(1.0, abs(vmean) * 0.01)
        ax.set_ylim(vmin - pad, vmax + pad)
    else:
        pad = max(0.05 * span, 1e-6)
        ax.set_ylim(vmin - pad, vmax + pad)

    decimals = _volume_decimal_places(vmin, vmax)
    ylo, yhi = ax.get_ylim()

    if span < max(1.0, abs(vmean) * 1e-4):
        tick_vals = sorted({float(v) for v in np.unique(finite)})
        for edge in (np.floor(vmin) - 1, np.floor(vmin), np.ceil(vmax), np.ceil(vmax) + 1):
            if ylo <= edge <= yhi:
                tick_vals.append(float(edge))
        tick_vals = sorted(set(round(t, max(decimals, 0)) for t in tick_vals))
        ax.yaxis.set_major_locator(FixedLocator(tick_vals))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _pos, d=decimals: f"{y:.{d}f}")
    )


def plot_thermo_axes(ax_temp, ax_vol, ax_energy, results_list, labels, 
                     colors, ls_cycle, tmax=None, running_avg=None):
    """Plot temperature, volume, and energy on given axes."""
    from matplotlib.ticker import AutoMinorLocator
    
    plotted_volumes = []
    for i, (R, lab) in enumerate(zip(results_list, labels)):
        t = R["time_ps"]
        temp = R["temperature"]
        vol = R["volume"]
        energy = R["energy"]
        
        if tmax is not None:
            mask = t <= tmax
            t = t[mask]
            temp = temp[mask]
            vol = vol[mask]
            energy = energy[mask]
        
        plotted_volumes.append(vol)
        ls = ls_cycle[i % len(ls_cycle)]
        color = colors[i % len(colors)]
        
        if running_avg is not None and running_avg > 1:
            t_avg = compute_running_average(t, running_avg)
            temp_avg = compute_running_average(temp, running_avg)
            vol_avg = compute_running_average(vol, running_avg)
            energy_avg = compute_running_average(energy, running_avg)
            
            ax_temp.plot(t, temp, color=color, ls=ls, alpha=0.3, lw=0.5)
            ax_temp.plot(t_avg, temp_avg, color=color, ls=ls, label=lab, lw=1.5)
            
            ax_vol.plot(t, vol, color=color, ls=ls, alpha=0.3, lw=0.5)
            ax_vol.plot(t_avg, vol_avg, color=color, ls=ls, label=lab, lw=1.5)
            
            ax_energy.plot(t, energy, color=color, ls=ls, alpha=0.3, lw=0.5)
            ax_energy.plot(t_avg, energy_avg, color=color, ls=ls, label=lab, lw=1.5)
        else:
            ax_temp.plot(t, temp, color=color, ls=ls, label=lab, lw=1.0)
            ax_vol.plot(t, vol, color=color, ls=ls, label=lab, lw=1.0)
            ax_energy.plot(t, energy, color=color, ls=ls, label=lab, lw=1.0)
    
    ax_temp.set_xlabel("Time (ps)")
    ax_temp.set_ylabel("Temperature (K)")
    ax_temp.set_title("Temperature")
    ax_temp.legend(fontsize=7, framealpha=0.92, loc="best", fancybox=False)
    ax_temp.xaxis.set_minor_locator(AutoMinorLocator())
    
    ax_vol.set_xlabel("Time (ps)")
    ax_vol.set_ylabel("Volume (A^3)")
    ax_vol.set_title("Volume")
    _configure_volume_axis(ax_vol, plotted_volumes)
    ax_vol.legend(fontsize=7, framealpha=0.92, loc="best", fancybox=False)
    ax_vol.xaxis.set_minor_locator(AutoMinorLocator())
    
    ax_energy.set_xlabel("Time (ps)")
    ax_energy.set_ylabel("Total Energy (eV)")
    ax_energy.set_title("Total Energy")
    ax_energy.legend(fontsize=7, framealpha=0.92, loc="best", fancybox=False)
    ax_energy.xaxis.set_minor_locator(AutoMinorLocator())


def make_thermo_figure(results_list, labels, sim_dirs=None, tmax=None, 
                       running_avg=None, save=None, maximize_window=True):
    """Create figure with volume, temperature, and energy plots."""
    _configure_matplotlib_backend(save=save)
    import matplotlib.pyplot as plt

    colors = plt.cm.tab10.colors
    ls_cycle = ["-", "--", "-.", ":"]

    if sim_dirs is None or len(sim_dirs) != len(results_list):
        sim_dirs = [f"series_{i + 1}" for i in range(len(results_list))]

    temp_groups = group_indices_by_temperature(sim_dirs)
    n_blocks = max(len(temp_groups), 1)
    fig_h = max(8.0, 4.5 * n_blocks + 1.5)
    fig = plt.figure(figsize=(14, fig_h))
    fig.subplots_adjust(top=0.92, bottom=0.06, left=0.06, right=0.98)

    subfigs = fig.subfigures(n_blocks, 1, hspace=0.15)
    subfigs = np.atleast_1d(subfigs).ravel()

    for (_group_title, idxs), sf in zip(temp_groups, subfigs):
        header_dirs = [_path_from_home(sim_dirs[i]) for i in idxs]
        if len(header_dirs) == 1:
            header_text = header_dirs[0]
        else:
            header_text = "\n".join(header_dirs)
        sf.suptitle(header_text, fontsize=9, weight="bold", y=0.97)
        
        ax_vol, ax_temp, ax_energy = sf.subplots(
            1, 3, gridspec_kw={"wspace": 0.30, "top": 0.82, "bottom": 0.15}
        )
        
        grp_results = [results_list[i] for i in idxs]
        grp_labels = [labels[i] for i in idxs]
        
        plot_thermo_axes(
            ax_temp, ax_vol, ax_energy,
            grp_results, grp_labels,
            colors, ls_cycle,
            tmax=tmax, running_avg=running_avg,
        )

    _finalize_figure(fig, save, maximize_window, skip_tight_layout=True)


def analyze_one(directory, skip, max_frames):
    """Analyze a single simulation directory for thermodynamic properties."""
    d = Path(directory).resolve()
    print(f"\n{'=' * 60}")
    print(f"  {d}")
    print(f"{'=' * 60}")

    data = read_thermo(d, skip=skip, max_frames=max_frames)
    
    md_engine = data["md_engine"]
    print(f"  MD engine: {md_engine}")
    
    temp = data["temperature"]
    vol = data["volume"]
    energy = data["energy"]
    
    print(f"  Temperature: mean={np.mean(temp):.1f} K, std={np.std(temp):.1f} K")
    if not np.all(np.isnan(vol)):
        print(f"  Volume: mean={np.mean(vol):.2f} A^3, std={np.std(vol):.2f} A^3")
    print(f"  Energy: mean={np.mean(energy):.4f} eV, std={np.std(energy):.4f} eV")
    
    sk_dir, _ = infer_temperature_from_path(d)
    nominal_T = float(sk_dir) if sk_dir is not None else None

    meta = {
        "dt_fs": data["dt_fs"],
        "md_engine": md_engine,
        "n_steps": len(data["step"]),
        "nominal_temperature_K": nominal_T,
        "mean_temperature_K": float(np.mean(temp)),
        "std_temperature_K": float(np.std(temp)),
        "mean_volume_A3": float(np.nanmean(vol)),
        "std_volume_A3": float(np.nanstd(vol)),
        "mean_energy_eV": float(np.mean(energy)),
        "std_energy_eV": float(np.std(energy)),
    }
    
    return {
        "time_ps": data["time_ps"],
        "temperature": temp,
        "volume": vol,
        "energy": energy,
        "meta": meta,
    }


def finalize_plot_labels(dirs, user_labels, all_results, origin_indices=None, 
                         n_raw_dirs=None, series_prefixes=None):
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


def write_data_log(out_path, sim_dirs, labels, all_results):
    """Write thermodynamic data to a text file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("# thermo_evaluation -- numerical data used for plots\n")
        fh.write(f"# generated_utc={utc}\n\n")
        
        for i, (d, lab, R) in enumerate(zip(sim_dirs, labels, all_results)):
            fh.write("=" * 80 + "\n")
            fh.write(f"# series_index={i}\n")
            fh.write(f"# label={lab}\n")
            fh.write(f"# directory={Path(d).resolve()}\n")
            
            if "meta" in R:
                m = R["meta"]
                for k in (
                    "dt_fs", "md_engine", "n_steps", "nominal_temperature_K",
                    "mean_temperature_K", "std_temperature_K",
                    "mean_volume_A3", "std_volume_A3",
                    "mean_energy_eV", "std_energy_eV",
                ):
                    if k in m:
                        fh.write(f"# meta.{k}={m[k]}\n")
            fh.write("=" * 80 + "\n\n")

            fh.write("# --- Thermodynamic data: time_ps, temperature_K, volume_A3, energy_eV ---\n")
            np.savetxt(
                fh,
                np.column_stack([R["time_ps"], R["temperature"], R["volume"], R["energy"]]),
                fmt="%.10e",
            )
            fh.write("\n")

    print(f"Data log written -> {out_path.resolve()}")


def main():
    ap = argparse.ArgumentParser(
        description="Plot thermodynamic properties (V, T, E) from MD trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 300K/
  %(prog)s dft/300K mlff/300K --labels DFT MLFF
  %(prog)s run/ --save thermo.png --running-avg 100
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
        "--skip",
        type=int,
        default=0,
        help="number of initial frames to skip (default: 0)",
    )
    ap.add_argument(
        "--tmax",
        type=float,
        default=None,
        help="max time on x-axis (ps; default: full range)",
    )
    ap.add_argument(
        "--running-avg",
        type=int,
        default=None,
        metavar="N",
        help="plot running average with window size N (raw data shown faded)",
    )
    ap.add_argument(
        "--no-maximize",
        action="store_true",
        help="do not maximize the plot window to the screen",
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
        help="write thermodynamic data to FILE (tab-separated)",
    )
    ap.add_argument(
        "--no-plot",
        action="store_true",
        help="skip matplotlib; use with --data-log for analysis only",
    )
    args = ap.parse_args()

    sim_dirs, origin_indices = resolve_dirs(args.dirs)

    print(f"Resolved {len(sim_dirs)} simulation(s):")
    for sd in sim_dirs:
        print(f"  {sd}")

    all_res = []
    for d in sim_dirs:
        res = analyze_one(str(d), args.skip, args.max_frames)
        all_res.append(res)

    labels = finalize_plot_labels(
        sim_dirs,
        args.labels,
        all_res,
        origin_indices=origin_indices,
        n_raw_dirs=len(args.dirs),
        series_prefixes=args.series_prefixes,
    )
    print("\nPlot legend labels:")
    for lb, sd in zip(labels, sim_dirs):
        print(f"  {lb:24s} <- {sd}")

    if args.data_log:
        write_data_log(args.data_log, sim_dirs, labels, all_res)

    if args.no_plot:
        if args.save:
            print("Note: --no-plot ignores --save (no figure written).", flush=True)
    else:
        make_thermo_figure(
            all_res,
            labels,
            sim_dirs=sim_dirs,
            tmax=args.tmax,
            running_avg=args.running_avg,
            save=args.save,
            maximize_window=not args.no_maximize,
        )


if __name__ == "__main__":
    main()
