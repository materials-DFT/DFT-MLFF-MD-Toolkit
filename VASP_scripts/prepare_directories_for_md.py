#!/usr/bin/env python3
"""
VASP/LAMMPS MD Directory Preparation and Optimization Suite
============================================================

This comprehensive script prepares molecular dynamics simulations for both
VASP and LAMMPS codes, supporting NPT and NVT ensembles.

1. Directory Preparation:
   - Creates temperature-specific subdirectories (e.g., 300K, 500K, etc.)
   - Copies input files to each temperature directory
   - Sets up proper INCAR/KPOINTS (VASP) or in.* (LAMMPS) files
   - Cleans up unwanted output files

2. MD Parameter Optimization:
   - Mass- and temperature-aware Langevin gamma per species
   - For NPT: PMASS and LANGEVIN_GAMMA_L scaled by system size and temperature
   - For NVT: Only thermostat parameters (no barostat)
   - POTIM fixed at 1.0 fs; ISTART=1, ICHARG=1 (restart from WAVECAR/CHGCAR)
   - Tunable via --gamma-ref, --alpha, --beta, --gamma-min, --gamma-max

3. Parallelization Optimization (VASP only):
   - Analyzes system size and k-point density
   - Calculates optimal NCORE, NPAR, KPAR, NSIM parameters
   - Supports multiple cluster configurations
   - Ensures NCORE × NPAR = total available cores

4. LAMMPS Support:
   - Generates LAMMPS input files (in.npt_allegro or in.nvt_allegro)
   - Stable workflow: CG minimize, T ramp (Tinit=Tmd/6), fixed-cell burn-in,
     then production NPT/NVT with small timestep (0.00010 ps)
   - Translates VASP Langevin parameters to LAMMPS fix langevin scale factors
   - NPT barostat Pdamp/friction both derived from VASP LANGEVIN_GAMMA_L (PMASS has
     no LAMMPS press/langevin equivalent); geometry uses aniso flip no; restart-aware
     dump (burn-in frames + production append)
   - Supports Allegro/NequIP pair styles with Kokkos

Usage:
    python3 prepare_directories_for_md.py <directory_path> [options]

Examples:
    # VASP NPT (default)
    python3 prepare_directories_for_md.py /path/to/structures --temps 300,700

    # VASP NVT
    python3 prepare_directories_for_md.py /path/to/structures --nvt --temps 300,700

    # LAMMPS NPT (temperature from directory name when present, e.g. 700K/)
    python3 prepare_directories_for_md.py /path/to/NPT --lammps

    # LAMMPS NVT
    python3 prepare_directories_for_md.py /path/to/structures --lammps --nvt --temps 300,700

Author: AI Assistant
Version: 2.0 (Combined VASP/LAMMPS, NPT/NVT)
"""

import os
import sys
import argparse
import shutil
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings

# Default configurations
DEFAULT_TEMPS = [300, 500, 700, 900, 1100, 1300]
DEFAULT_DELETE_VASP = [
    "OUTCAR", "OSZICAR", "job.err", "job.out", "PROCAR",
    "REPORT", "vasprun.xml", "XDATCAR", "ICONST", "IBZKPT",
    "DOSCAR", "EIGENVAL"
]
DEFAULT_DELETE_LAMMPS = [
    "log.lammps", "job.err", "job.out", "trajectory.extxyz",
    "restart.lmp.a", "restart.lmp.b", "restart.lmp.last"
]

LAMMPS_STABLE_DEFAULTS = {
    "target_steps": 1_000_000,
    "timer_timeout": "1:57:00",
    "burnin_steps": 20_000,
    "burnin_timestep": 0.00005,
    "production_timestep": 0.00010,
    "burnin_lang_damp": 0.1,
    "dump_interval": 100,
    "neighbor_cutoff": 2.0,
    "tinit_ratio": 1.0 / 6.0,
}

DEFAULT_PARAMS = {
    "ISTART": "1",
    "ICHARG": "1",
    "IBRION": "0",
    "POTIM": "1",
    "ISIF": "3",
    "EDIFF": "1E-5",
    "PREC": "Normal",
    "ALGO": "Normal",
    "NELM": "150",
    "SIGMA": "0.05"
}

KPOINTS_CONTENT = """Automatic mesh
0
Monkhorst-Pack
1 1 1
0 0 0
"""

VASP_HINT_FILES = {"INCAR", "POSCAR", "CONTCAR", "KPOINTS", "POTCAR"}
LAMMPS_HINT_FILES = {"data.lammps", "POSCAR", "in.allegro", "in.npt_md_allegro", "in.nvt_allegro"}


@dataclass
class AtomicSpecies:
    """Data class for atomic species information."""
    symbol: str
    atomic_number: int
    atomic_mass: float
    count: int


@dataclass
class MDParameters:
    """Data class for MD simulation parameters (NPT or NVT)."""
    ensemble: str = "npt"  # "npt" or "nvt"
    
    # Core MD parameters
    ibrion: int = 0
    mdalgo: int = 3
    isif: int = 3  # 3 for NPT, 2 for NVT
    nsw: int = 10000
    potim: float = 1.0
    
    # Temperature control
    tebeg: Optional[float] = None
    teend: Optional[float] = None
    
    # Langevin thermostat parameters (per-species)
    langevin_gamma: List[float] = None
    
    # Barostat parameters (NPT only)
    langevin_gamma_l: float = 10.0
    pmass: float = 1000.0
    pstress: float = 0.0  # Target pressure in kbar
    
    # Additional parameters
    ediffg: float = -5e-2
    isym: int = 0


@dataclass
class SystemInfo:
    """Container for system analysis information."""
    path: str
    atoms: int
    kpoints: int
    kpoint_grid: Tuple[int, int, int]
    species_list: List[AtomicSpecies]
    lattice: np.ndarray
    current_ncore: Optional[int] = None
    current_npar: Optional[int] = None
    current_kpar: Optional[int] = None
    current_nsim: Optional[int] = None
    optimal_ncore: Optional[int] = None
    optimal_npar: Optional[int] = None
    optimal_kpar: Optional[int] = None
    optimal_nsim: Optional[int] = None
    optimal_md_params: Optional[MDParameters] = None


@dataclass
class ClusterConfig:
    """Cluster configuration parameters."""
    name: str
    total_cores: int
    cores_per_node: int
    nodes: int
    max_ncore: int


class MDDirectoryProcessor:
    """
    Comprehensive MD processor that handles directory preparation and
    parameter optimization for both VASP and LAMMPS, NPT and NVT.
    """
    
    ATOMIC_DATA: Dict[str, Dict[str, float]] = {
        'H':  {'Z': 1,  'mass': 1.008},
        'He': {'Z': 2,  'mass': 4.0026},
        'Li': {'Z': 3,  'mass': 6.94},
        'Be': {'Z': 4,  'mass': 9.0122},
        'B':  {'Z': 5,  'mass': 10.81},
        'C':  {'Z': 6,  'mass': 12.011},
        'N':  {'Z': 7,  'mass': 14.007},
        'O':  {'Z': 8,  'mass': 15.999},
        'F':  {'Z': 9,  'mass': 18.998},
        'Ne': {'Z': 10, 'mass': 20.180},
        'Na': {'Z': 11, 'mass': 22.990},
        'Mg': {'Z': 12, 'mass': 24.305},
        'Al': {'Z': 13, 'mass': 26.982},
        'Si': {'Z': 14, 'mass': 28.085},
        'P':  {'Z': 15, 'mass': 30.974},
        'S':  {'Z': 16, 'mass': 32.06},
        'Cl': {'Z': 17, 'mass': 35.45},
        'Ar': {'Z': 18, 'mass': 39.948},
        'K':  {'Z': 19, 'mass': 39.0983},
        'Ca': {'Z': 20, 'mass': 40.078},
        'Sc': {'Z': 21, 'mass': 44.9559},
        'Ti': {'Z': 22, 'mass': 47.867},
        'V':  {'Z': 23, 'mass': 50.9415},
        'Cr': {'Z': 24, 'mass': 51.9961},
        'Mn': {'Z': 25, 'mass': 54.938},
        'Fe': {'Z': 26, 'mass': 55.845},
        'Co': {'Z': 27, 'mass': 58.933},
        'Ni': {'Z': 28, 'mass': 58.693},
        'Cu': {'Z': 29, 'mass': 63.546},
        'Zn': {'Z': 30, 'mass': 65.38},
        'Ga': {'Z': 31, 'mass': 69.723},
        'Ge': {'Z': 32, 'mass': 72.63},
        'As': {'Z': 33, 'mass': 74.9216},
        'Se': {'Z': 34, 'mass': 78.971},
        'Br': {'Z': 35, 'mass': 79.904},
        'Kr': {'Z': 36, 'mass': 83.798},
        'Rb': {'Z': 37, 'mass': 85.468},
        'Sr': {'Z': 38, 'mass': 87.62},
        'Y':  {'Z': 39, 'mass': 88.906},
        'Zr': {'Z': 40, 'mass': 91.224},
        'Nb': {'Z': 41, 'mass': 92.906},
        'Mo': {'Z': 42, 'mass': 95.95},
        'Ru': {'Z': 44, 'mass': 101.07},
        'Rh': {'Z': 45, 'mass': 102.91},
        'Pd': {'Z': 46, 'mass': 106.42},
        'Ag': {'Z': 47, 'mass': 107.87},
        'Cd': {'Z': 48, 'mass': 112.41},
        'In': {'Z': 49, 'mass': 114.82},
        'Sn': {'Z': 50, 'mass': 118.71},
        'Sb': {'Z': 51, 'mass': 121.76},
        'Te': {'Z': 52, 'mass': 127.60},
        'I':  {'Z': 53, 'mass': 126.90},
        'Xe': {'Z': 54, 'mass': 131.29},
        'Cs': {'Z': 55, 'mass': 132.91},
        'Ba': {'Z': 56, 'mass': 137.33},
        'La': {'Z': 57, 'mass': 138.91},
        'Ce': {'Z': 58, 'mass': 140.12},
        'Pr': {'Z': 59, 'mass': 140.91},
        'Nd': {'Z': 60, 'mass': 144.24},
        'Sm': {'Z': 62, 'mass': 150.36},
        'Eu': {'Z': 63, 'mass': 151.96},
        'Gd': {'Z': 64, 'mass': 157.25},
        'Tb': {'Z': 65, 'mass': 158.93},
        'Dy': {'Z': 66, 'mass': 162.50},
        'Ho': {'Z': 67, 'mass': 164.93},
        'Er': {'Z': 68, 'mass': 167.26},
        'Tm': {'Z': 69, 'mass': 168.93},
        'Yb': {'Z': 70, 'mass': 173.05},
        'Lu': {'Z': 71, 'mass': 174.97},
        'Hf': {'Z': 72, 'mass': 178.49},
        'Ta': {'Z': 73, 'mass': 180.95},
        'W':  {'Z': 74, 'mass': 183.84},
        'Re': {'Z': 75, 'mass': 186.21},
        'Os': {'Z': 76, 'mass': 190.23},
        'Ir': {'Z': 77, 'mass': 192.22},
        'Pt': {'Z': 78, 'mass': 195.08},
        'Au': {'Z': 79, 'mass': 196.97},
        'Hg': {'Z': 80, 'mass': 200.59},
        'Tl': {'Z': 81, 'mass': 204.38},
        'Pb': {'Z': 82, 'mass': 207.2},
        'Bi': {'Z': 83, 'mass': 208.98},
        'Th': {'Z': 90, 'mass': 232.04},
        'U':  {'Z': 92, 'mass': 238.03},
    }
    
    def __init__(
        self,
        cluster_config: ClusterConfig,
        verbose: bool = True,
        nsw: int = 10000,
        gamma_ref: float = 2.0,
        alpha: float = 0.6,
        beta: float = 0.5,
        gamma_min: float = 0.5,
        gamma_max: float = 15.0,
        enforce_monotone: bool = True,
        ensemble: str = "npt",
        code: str = "vasp",
        lammps_timestep: float = LAMMPS_STABLE_DEFAULTS["production_timestep"],
        lammps_target_steps: int = LAMMPS_STABLE_DEFAULTS["target_steps"],
        lammps_timer_timeout: str = LAMMPS_STABLE_DEFAULTS["timer_timeout"],
        lammps_burnin_steps: int = LAMMPS_STABLE_DEFAULTS["burnin_steps"],
        lammps_burnin_timestep: float = LAMMPS_STABLE_DEFAULTS["burnin_timestep"],
        lammps_burnin_lang_damp: float = LAMMPS_STABLE_DEFAULTS["burnin_lang_damp"],
        lammps_dump_interval: int = LAMMPS_STABLE_DEFAULTS["dump_interval"],
        lammps_neighbor_cutoff: float = LAMMPS_STABLE_DEFAULTS["neighbor_cutoff"],
        lammps_tinit_ratio: float = LAMMPS_STABLE_DEFAULTS["tinit_ratio"],
    ):
        """Initialize the processor."""
        self.cluster_config = cluster_config
        self.verbose = verbose
        self.nsw = nsw
        self.gamma_ref = gamma_ref
        self.alpha = alpha
        self.beta = beta
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.enforce_monotone = enforce_monotone
        self.ensemble = ensemble.lower()
        self.code = code.lower()
        self.lammps_timestep = lammps_timestep
        self.lammps_target_steps = lammps_target_steps
        self.lammps_timer_timeout = lammps_timer_timeout
        self.lammps_burnin_steps = lammps_burnin_steps
        self.lammps_burnin_timestep = lammps_burnin_timestep
        self.lammps_burnin_lang_damp = lammps_burnin_lang_damp
        self.lammps_dump_interval = lammps_dump_interval
        self.lammps_neighbor_cutoff = lammps_neighbor_cutoff
        self.lammps_tinit_ratio = lammps_tinit_ratio
        self.processed_dirs = []
        self.skipped_dirs = []
        self.systems_analyzed = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log messages with appropriate formatting."""
        if self.verbose:
            print(f"{message}")
    
    def parse_temps(self, temp_string: str) -> List[int]:
        """Parse temperature string into list of integers."""
        temps = []
        for t in temp_string.split(","):
            t = t.strip()
            if t:
                temps.append(int(t))
        return temps

    @staticmethod
    def parse_temperature_from_dirname(path: str) -> Optional[int]:
        """Extract target temperature from a directory name like 300K or 700k."""
        match = re.fullmatch(r"(\d+)K", os.path.basename(path), re.IGNORECASE)
        return int(match.group(1)) if match else None
    
    def is_structure_dir(self, path: str) -> bool:
        """Check if directory contains structure files."""
        if not os.path.isdir(path):
            return False
        try:
            names = set(os.listdir(path))
        except PermissionError:
            return False
        
        # Check for LAMMPS or VASP files
        has_lammps = "data.lammps" in names or any(
            f.startswith("in.") and f.endswith("allegro") for f in names
        )
        has_vasp = len(VASP_HINT_FILES & names) > 0
        
        return has_lammps or has_vasp
    
    def detect_code(self, path: str) -> str:
        """Auto-detect whether directory is LAMMPS or VASP based on files present."""
        try:
            names = set(os.listdir(path))
        except PermissionError:
            return "vasp"
        
        # LAMMPS indicators: data.lammps or in.*allegro files
        has_lammps = "data.lammps" in names or any(
            f.startswith("in.") and "allegro" in f for f in names
        )
        
        # VASP indicators: INCAR, POTCAR, KPOINTS (not just POSCAR, since LAMMPS dirs may have POSCAR too)
        has_vasp_specific = bool({"INCAR", "POTCAR", "KPOINTS"} & names)
        
        if has_lammps and not has_vasp_specific:
            return "lammps"
        else:
            return "vasp"
    
    def read_poscar(self, poscar_path: str) -> Tuple[List[AtomicSpecies], np.ndarray]:
        """Read POSCAR file and extract atomic species information."""
        try:
            with open(poscar_path, 'r') as f:
                lines = f.readlines()
            
            scaling_factor = float(lines[1].strip().split()[0])
            
            lattice = np.array([
                [float(x) for x in lines[2].split()],
                [float(x) for x in lines[3].split()],
                [float(x) for x in lines[4].split()]
            ]) * scaling_factor
            
            species_line = lines[5].strip().split()
            counts_line = lines[6].strip().split()
            
            species_list = []
            for symbol, count_str in zip(species_line, counts_line):
                count = int(count_str)
                if symbol in self.ATOMIC_DATA:
                    atomic_info = self.ATOMIC_DATA[symbol]
                    species_list.append(AtomicSpecies(
                        symbol=symbol,
                        atomic_number=int(atomic_info['Z']),
                        atomic_mass=float(atomic_info['mass']),
                        count=count
                    ))
                else:
                    self.log(f"Unknown species '{symbol}' - using fallback mass=20, Z=0", "WARN")
                    species_list.append(AtomicSpecies(
                        symbol=symbol,
                        atomic_number=0,
                        atomic_mass=20.0,
                        count=count
                    ))
            
            return species_list, lattice
            
        except Exception as e:
            self.log(f"Error reading POSCAR {poscar_path}: {e}", "ERROR")
            return [], np.array([])

    def read_lammps_data(self, data_path: str) -> Tuple[List[AtomicSpecies], np.ndarray]:
        """Read LAMMPS data file and extract species information from Masses section."""
        try:
            with open(data_path, 'r') as f:
                lines = f.readlines()
            
            # Parse box dimensions
            lattice = np.zeros((3, 3))
            for line in lines:
                line_lower = line.lower()
                if 'xlo xhi' in line_lower:
                    parts = line.split()
                    lattice[0, 0] = float(parts[1]) - float(parts[0])
                elif 'ylo yhi' in line_lower:
                    parts = line.split()
                    lattice[1, 1] = float(parts[1]) - float(parts[0])
                elif 'zlo zhi' in line_lower:
                    parts = line.split()
                    lattice[2, 2] = float(parts[1]) - float(parts[0])
                elif 'xy xz yz' in line_lower:
                    parts = line.split()
                    lattice[0, 1] = float(parts[0])  # xy
                    lattice[0, 2] = float(parts[1])  # xz
                    lattice[1, 2] = float(parts[2])  # yz
            
            # Parse masses section
            in_masses = False
            masses = {}
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.lower() == 'masses':
                    in_masses = True
                    continue
                if in_masses:
                    if not stripped:
                        continue
                    # Check if we've hit another section
                    if stripped.lower().startswith('atoms') or stripped.lower().startswith('velocities'):
                        break
                    parts = stripped.split()
                    if len(parts) >= 2:
                        try:
                            type_id = int(parts[0])
                            mass = float(parts[1])
                            # Check for comment with element symbol: "1 39.098 # K"
                            symbol = f"Type{type_id}"
                            if '#' in stripped:
                                comment_part = stripped.split('#')[1].strip()
                                if comment_part:
                                    symbol = comment_part.split()[0]
                            masses[type_id] = (mass, symbol)
                        except (ValueError, IndexError):
                            continue
            
            # Parse header for counts
            n_atoms = 0
            n_types = 0
            for line in lines:
                stripped = line.strip().lower()
                # Match "96 atoms" but not "Atoms # atomic"
                if ' atoms' in stripped and not stripped.startswith('atoms'):
                    parts = line.split()
                    try:
                        n_atoms = int(parts[0])
                    except (ValueError, IndexError):
                        pass
                elif 'atom types' in stripped:
                    parts = line.split()
                    try:
                        n_types = int(parts[0])
                    except (ValueError, IndexError):
                        pass
            
            # Count atoms per type from Atoms section
            type_counts = {}
            in_atoms = False
            for line in lines:
                stripped = line.strip()
                stripped_lower = stripped.lower()
                # "Atoms # atomic" or just "Atoms"
                if stripped_lower.startswith('atoms'):
                    in_atoms = True
                    continue
                if in_atoms:
                    if not stripped:
                        continue
                    # Check if we've hit another section
                    if stripped_lower in ['velocities', 'bonds', 'angles', 'dihedrals', 'impropers']:
                        break
                    parts = stripped.split()
                    if len(parts) >= 2:
                        try:
                            # atom-id type x y z ... format
                            type_id = int(parts[1])
                            type_counts[type_id] = type_counts.get(type_id, 0) + 1
                        except (ValueError, IndexError):
                            continue
            
            # Build species list
            species_list = []
            for type_id in sorted(masses.keys()):
                mass, symbol = masses[type_id]
                count = type_counts.get(type_id, 0)
                
                # Try to identify element from mass if symbol is generic
                z = 0
                if symbol.startswith("Type"):
                    for elem, data in self.ATOMIC_DATA.items():
                        if abs(data['mass'] - mass) < 0.5:
                            symbol = elem
                            z = data['Z']
                            break
                else:
                    # Use the symbol from comment
                    if symbol in self.ATOMIC_DATA:
                        z = self.ATOMIC_DATA[symbol]['Z']
                
                species_list.append(AtomicSpecies(
                    symbol=symbol,
                    atomic_number=z,
                    atomic_mass=mass,
                    count=count
                ))
            
            return species_list, lattice
            
        except Exception as e:
            self.log(f"Error reading LAMMPS data {data_path}: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return [], np.array([])
    
    def analyze_kpoints(self, kpoints_path: str) -> Tuple[int, Tuple[int, int, int]]:
        """Analyze KPOINTS to determine k-point grid."""
        try:
            with open(kpoints_path, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                if line.strip().lower().startswith('monkhorst'):
                    if i + 1 < len(lines):
                        grid_line = lines[i + 1].strip()
                        grid = [int(x) for x in grid_line.split()]
                        if len(grid) >= 3:
                            kpoints = grid[0] * grid[1] * grid[2]
                            self.log(f"KPOINTS analysis: {kpoints} k-points ({grid[0]}×{grid[1]}×{grid[2]})")
                            return kpoints, (grid[0], grid[1], grid[2])
            
            for line in lines:
                if re.match(r'^\s*\d+\s+\d+\s+\d+', line.strip()):
                    grid = [int(x) for x in line.split()[:3]]
                    kpoints = grid[0] * grid[1] * grid[2]
                    return kpoints, (grid[0], grid[1], grid[2])
            
            raise ValueError("Could not find k-point grid in KPOINTS")
            
        except Exception as e:
            self.log(f"Error analyzing KPOINTS {kpoints_path}: {e}", "ERROR")
            return 1, (1, 1, 1)
    
    def read_incar(self, incar_path: str) -> Dict[str, Any]:
        """Read INCAR file and extract parameters."""
        params = {}
        try:
            with open(incar_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith('!'):
                        continue
                    
                    if '=' in line:
                        parts = line.split('=', 1)
                        key = parts[0].strip()
                        value_str = parts[1].strip()
                        for marker in ('#', '!'):
                            if marker in value_str:
                                value_str = value_str.split(marker, 1)[0].strip()

                        try:
                            if '.' not in value_str and 'E' not in value_str.upper():
                                params[key] = int(value_str)
                            else:
                                params[key] = float(value_str)
                        except ValueError:
                            first_token = value_str.split()[0] if value_str.split() else value_str
                            try:
                                if re.fullmatch(r'[+-]?\d+', first_token):
                                    params[key] = int(first_token)
                                elif re.fullmatch(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?', first_token):
                                    params[key] = float(first_token)
                                else:
                                    params[key] = value_str
                            except ValueError:
                                params[key] = value_str
                            
        except Exception as e:
            self.log(f"Error reading INCAR {incar_path}: {e}", "ERROR")
            
        return params
    
    # --------------- Physics heuristics --------------- #

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _compute_langevin_gammas(
        self,
        species_list: List[AtomicSpecies],
        temperature: float,
    ) -> List[float]:
        """Per-species gamma (ps^-1) for Langevin thermostat."""
        total_atoms = sum(s.count for s in species_list)
        avg_mass = sum(s.atomic_mass * s.count for s in species_list) / max(1, total_atoms)
        m_ref = avg_mass

        temp_factor = (max(1e-6, temperature) / 300.0) ** self.beta

        raw = []
        for s in species_list:
            g = self.gamma_ref * (m_ref / s.atomic_mass) ** self.alpha * temp_factor
            g = self._clamp(g, self.gamma_min, self.gamma_max)
            raw.append(g)

        if not self.enforce_monotone:
            return raw

        masses = [s.atomic_mass for s in species_list]
        gammas = raw[:]
        for _ in range(len(gammas)):
            changed = False
            for i in range(1, len(gammas)):
                if masses[i] > masses[i-1] and gammas[i] > gammas[i-1]:
                    new_val = 0.5 * (gammas[i] + gammas[i-1])
                    gammas[i] = min(gammas[i-1], new_val)
                    changed = True
            if not changed:
                break
        return [self._clamp(g, self.gamma_min, self.gamma_max) for g in gammas]

    def _compute_pmass(self, natoms: int, temperature: float) -> float:
        """PMASS scaled by system size and temperature."""
        size_factor = self._clamp(natoms / 50.0, 0.5, 3.0)
        t_factor = self._clamp(temperature / 300.0, 0.5, 2.0)
        pmass = 1000.0 * size_factor * t_factor
        return float(self._clamp(pmass, 800.0, 5000.0))

    def _compute_gamma_l(self, natoms: int) -> float:
        """Lattice friction (ps^-1) scaled by system size."""
        size_factor = self._clamp(natoms / 50.0, 0.5, 3.0)
        gamma_l = 3.0 * size_factor
        return float(self._clamp(gamma_l, 1.0, 10.0))

    def _compute_tinit(self, temperature: float) -> float:
        """Initial temperature for fresh-start ramp (default: Tmd/6)."""
        return round(temperature * self.lammps_tinit_ratio, 1)

    def calculate_optimal_md_parameters(
        self,
        species_list: List[AtomicSpecies],
        temperature: float,
        lattice: np.ndarray,
        existing_params: Dict[str, Any]
    ) -> MDParameters:
        """Calculate optimal MD parameters for NPT or NVT."""
        total_atoms = sum(s.count for s in species_list)

        langevin_gamma = self._compute_langevin_gammas(species_list, temperature)

        if self.ensemble == "npt":
            pmass = self._compute_pmass(total_atoms, temperature)
            gamma_l = self._compute_gamma_l(total_atoms)
            isif = 3
        else:
            pmass = 0.0
            gamma_l = 0.0
            isif = 2

        return MDParameters(
            ensemble=self.ensemble,
            ibrion=0,
            mdalgo=3,
            isif=isif,
            tebeg=temperature,
            teend=temperature,
            nsw=self.nsw,
            potim=1.0,
            langevin_gamma=langevin_gamma,
            langevin_gamma_l=gamma_l,
            pmass=pmass,
            ediffg=float(existing_params.get('EDIFFG', -5e-2)),
            isym=int(existing_params.get('ISYM', 0))
        )

    # --------------- Parallelization (VASP) --------------- #

    def analyze_incar_parallelization(self, incar_path: str) -> Dict[str, Optional[int]]:
        """Read current NCORE, NPAR, KPAR, and NSIM from an INCAR file."""
        params: Dict[str, Optional[int]] = {
            'ncore': None, 'npar': None, 'kpar': None, 'nsim': None
        }

        try:
            with open(incar_path, 'r') as f:
                for line in f:
                    line_upper = line.strip().upper()
                    if line_upper.startswith('#') or not line_upper:
                        continue
                    for param in params:
                        match = re.search(rf'^{param}\s*=\s*(\d+)', line_upper)
                        if match:
                            params[param] = int(match.group(1))
        except Exception as e:
            self.log(f"Error analyzing INCAR parallelization: {e}", "ERROR")

        return params

    def calculate_optimal_parallelization(self, atoms: int, kpoints: int) -> Dict[str, int]:
        """Calculate optimal parallelization parameters."""
        total_cores = self.cluster_config.total_cores

        if atoms <= 50:
            system_size = "small"
        elif atoms <= 100:
            system_size = "medium"
        else:
            system_size = "large"

        if system_size == "small":
            if kpoints >= 64:
                optimal_kpar = min(8, kpoints // 8)
                optimal_ncore = min(8, total_cores // 2)
            else:
                optimal_kpar = min(4, max(1, kpoints // 4))
                optimal_ncore = min(4, total_cores // 4)
        elif system_size == "medium":
            if kpoints >= 64:
                optimal_kpar = min(8, kpoints // 8)
            else:
                optimal_kpar = min(4, max(1, kpoints // 4))
            optimal_ncore = min(4, total_cores // 4)
        else:
            if kpoints >= 64:
                optimal_kpar = min(4, kpoints // 16)
            else:
                optimal_kpar = min(2, max(1, kpoints // 9))
            optimal_ncore = min(2, total_cores // 8)

        optimal_ncore = max(1, optimal_ncore)
        optimal_npar = max(1, total_cores // optimal_ncore)
        optimal_kpar = max(1, min(optimal_kpar, kpoints))
        optimal_ncore = max(1, min(optimal_ncore, self.cluster_config.max_ncore))

        return {
            'ncore': optimal_ncore,
            'npar': optimal_npar,
            'kpar': optimal_kpar,
            'nsim': 4,
        }

    def update_incar_parallelization(
        self, incar_path: str, parallelization_params: Dict[str, int]
    ) -> bool:
        """Update or append parallelization flags in INCAR."""
        try:
            with open(incar_path, 'r') as f:
                lines = f.readlines()

            updated_lines = []
            parallelization_section = False
            in_parallelization = False

            for line in lines:
                line_upper = line.strip().upper()

                if 'PARALLELIZATION' in line_upper and 'FLAGS' in line_upper:
                    parallelization_section = True
                    in_parallelization = True
                    updated_lines.append(line)
                    continue

                if in_parallelization and line.strip() and not line.startswith('#'):
                    if not any(p in line_upper for p in ['NCORE', 'NPAR', 'KPAR', 'NSIM', 'LPLANE', 'LSCALU']):
                        in_parallelization = False

                if in_parallelization:
                    if 'NCORE' in line_upper and '=' in line_upper:
                        line = f"NCORE = {parallelization_params['ncore']}\n"
                    elif 'NPAR' in line_upper and '=' in line_upper:
                        line = f"NPAR = {parallelization_params['npar']}\n"
                    elif 'KPAR' in line_upper and '=' in line_upper:
                        line = f"KPAR = {parallelization_params['kpar']}\n"
                    elif 'NSIM' in line_upper and '=' in line_upper:
                        line = f"NSIM = {parallelization_params['nsim']}\n"

                updated_lines.append(line)

            if not parallelization_section:
                updated_lines.append("\n# Optimized parallelization parameters\n")
                updated_lines.append(f"NCORE = {parallelization_params['ncore']}\n")
                updated_lines.append(f"NPAR = {parallelization_params['npar']}\n")
                updated_lines.append(f"KPAR = {parallelization_params['kpar']}\n")
                updated_lines.append(f"NSIM = {parallelization_params['nsim']}\n")
                updated_lines.append("LPLANE = .TRUE.\n")
                updated_lines.append("LSCALU = .FALSE.\n")

            with open(incar_path, 'w') as f:
                f.writelines(updated_lines)

            return True

        except Exception as e:
            self.log(f"Error updating INCAR parallelization: {e}", "ERROR")
            return False

    # --------------- File writers --------------- #

    def write_kpoints(self, kpoints_path: str):
        """Write standard KPOINTS file."""
        with open(kpoints_path, "w") as f:
            f.write(KPOINTS_CONTENT)
    
    def write_vasp_incar(
        self,
        incar_path: str,
        params: Dict[str, Any],
        temp: int,
        md_params: MDParameters
    ):
        """Write optimized VASP INCAR file."""
        all_params = params.copy()

        all_params['ISTART'] = 1
        all_params['ICHARG'] = 1
        all_params['TEBEG'] = temp
        all_params['TEEND'] = temp
        
        all_params.update({
            'IBRION': md_params.ibrion,
            'MDALGO': md_params.mdalgo,
            'ISIF': md_params.isif,
            'NSW': md_params.nsw,
            'POTIM': f'{md_params.potim:.3f}',
            'LANGEVIN_GAMMA': ' '.join(f'{g:.3f}' for g in md_params.langevin_gamma),
            'EDIFFG': md_params.ediffg,
            'ISYM': md_params.isym
        })
        
        if md_params.ensemble == "npt":
            all_params['LANGEVIN_GAMMA_L'] = f'{md_params.langevin_gamma_l:.3f}'
            all_params['PMASS'] = f'{md_params.pmass:.0f}'

        all_params['ALGO'] = 'Normal'
        all_params['PREC'] = 'Normal'
        all_params['EDIFF'] = 1E-5
        
        ensemble_label = "NPT" if md_params.ensemble == "npt" else "NVT"
        
        with open(incar_path, 'w') as f:
            f.write(f"System = Optimized {ensemble_label} Molecular Dynamics\n")
            f.write(f"# Parameters optimized for {temp}K (mass- and T-aware Langevin)\n\n")
            
            f.write("Starting parameters:\n")
            for key in ['ISTART', 'ICHARG']:
                if key in all_params:
                    f.write(f"{key} = {all_params[key]}\n")
            f.write("\n")
            
            f.write("Electronic Relaxation:\n")
            electronic_keys = ['PREC', 'ENCUT', 'NELMIN', 'NELM', 'EDIFF', 'LREAL', 
                             'ISPIN', 'MAGMOM', 'ALGO', 'METAGGA', 'LMIXTAU', 'LASPH', 'LDIAG',
                             'ISMEAR', 'SIGMA', 'LORBIT']
            for key in electronic_keys:
                if key in all_params:
                    f.write(f"{key} = {all_params[key]}\n")
            f.write("\n")
            
            f.write(f"Ionic Molecular Dynamics ({ensemble_label}, Langevin):\n")
            md_keys = ['NSW', 'IBRION', 'EDIFFG', 'ISIF', 'POTIM', 'ISYM', 
                      'MDALGO', 'LANGEVIN_GAMMA']
            if md_params.ensemble == "npt":
                md_keys.extend(['LANGEVIN_GAMMA_L', 'PMASS'])
            for key in md_keys:
                if key in all_params:
                    f.write(f"{key} = {all_params[key]}\n")
            f.write("\n")
            
            f.write("Temperature Control:\n")
            f.write(f"TEBEG = {temp}\n")
            f.write(f"TEEND = {temp}\n")
            f.write("\n")

            f.write("# Space-saving flags\n")
            f.write("LWAVE  = .FALSE.\n")
            f.write("LCHARG = .FALSE.\n")

    def write_lammps_input(
        self,
        input_path: str,
        species_list: List[AtomicSpecies],
        md_params: MDParameters,
        temperature: float,
    ):
        """Write LAMMPS input file for Allegro/NequIP with stable NPT or NVT workflow."""

        species_symbols = [s.symbol for s in species_list]
        elements_str = " ".join(species_symbols)
        tinit = self._compute_tinit(temperature)

        lang_damp = 1.0
        scale_lines = []
        for i, s in enumerate(species_list, start=1):
            gamma = md_params.langevin_gamma[i - 1]
            scale = 1.0 / (lang_damp * gamma)
            scale_lines.append(f"scale {i} {scale:.9f}")
        scale_str = " ".join(scale_lines)

        ensemble_label = "NPT" if md_params.ensemble == "npt" else "NVT"
        gamma_comment = " ".join(f"{g:.3f}" for g in md_params.langevin_gamma)

        content = f"""# LAMMPS input for {ensemble_label} MD with Allegro
# Auto-generated by prepare_directories_for_md.py
log           log.lammps append

units         metal
boundary      p p p
atom_style    atomic
atom_modify   map yes

newton        on
package       kokkos neigh half

processors    * * *

# --- continuous run settings ---
variable      target equal {self.lammps_target_steps}

timer         timeout {self.lammps_timer_timeout} every 100

restart       10000 restart.lmp.a restart.lmp.b

if "${{restartflag}} == 0" then "read_data ${{datafile}}" else "read_restart ${{restartfile}}"

if "${{restartflag}} == 0" then "jump SELF triclinic_maybe"
jump SELF after_triclinic
label triclinic_maybe
if "${{triclinic_fix}} == 1" then "change_box all xy final 0.0 remap units box"
label after_triclinic

pair_style    allegro/kk
pair_coeff    * * ${{model}} {elements_str}

neighbor      {self.lammps_neighbor_cutoff} bin
neigh_modify  delay 0 every 1 check yes

# Group atoms by type
"""
        for i, s in enumerate(species_list, start=1):
            content += f"group         {s.symbol:3} type {i}\n"

        content += """
thermo 100
thermo_style custom step temp press pxx pyy pzz pxy pxz pyz vol pe ke etotal
thermo_modify flush yes

if "${restartflag} == 0" then "run 0 post no"

# Fresh starts: relax local atomic forces at fixed cell before assigning velocities
# or enabling the NPT barostat.
if "${restartflag} == 0" then "min_style cg"
if "${restartflag} == 0" then "min_modify dmax 0.02"
if "${restartflag} == 0" then "minimize 1.0e-7 1.0e-5 1000 10000"
if "${restartflag} == 0" then "reset_timestep 0"
if "${restartflag} == 0" then "run 0 post no"

# Target T (K): matches VASP TEBEG/TEEND
"""
        content += f"variable      Tmd equal {temperature:.1f}\n"
        content += f"variable      Tinit equal {tinit:.1f}\n"

        if md_params.ensemble == "npt":
            content += f"""# Target pressure (bar)
variable      Ptarget equal {md_params.pstress * 10.0:.1f}
"""

        content += """
# Velocities
if "${restartflag} == 0" then "jump SELF maybe_vel_create"
jump SELF after_vel_create
label maybe_vel_create
if "${use_data_vel} == 1" then "jump SELF after_vel_create"
label do_vel_create
velocity all create ${Tinit} 12345 mom yes rot no dist gaussian
label after_vel_create

# Integration and fresh-start burn-in thermostat.
"""
        content += f"# VASP LANGEVIN_GAMMA = {gamma_comment} ({elements_str})\n"
        content += f"# LAMMPS scale = 1/(damp*gamma); damp = {lang_damp} ps\n"
        content += f"""fix nve all nve
fix lang_pre all langevin ${{Tinit}} ${{Tmd}} {self.lammps_burnin_lang_damp} 48279 &
    {scale_str}

# Fresh starts: fixed-cell burn-in at a very small timestep before enabling any barostat.
if "${{restartflag}} == 0" then "timestep {self.lammps_burnin_timestep:.5f}"
if "${{restartflag}} == 0" then "dump 1 all extxyz {self.lammps_dump_interval} trajectory.extxyz"
if "${{restartflag}} == 0" then "dump_modify 1 sort id"
if "${{restartflag}} == 0" then "dump_modify 1 element {elements_str}"
if "${{restartflag}} == 0" then "dump_modify 1 append yes"
if "${{restartflag}} == 0" then "run {self.lammps_burnin_steps}"
if "${{restartflag}} == 0" then "undump 1"
if "${{restartflag}} == 0" then "reset_timestep 0"
unfix lang_pre

# Production thermostat.
variable      lang_damp equal {lang_damp}
fix lang all langevin ${{Tmd}} ${{Tmd}} ${{lang_damp}} 48279 &
    {scale_str}
"""

        if md_params.ensemble == "npt":
            content += f"""# Barostat: VASP LANGEVIN_GAMMA_L = {md_params.langevin_gamma_l:.3f} ps^-1 (PMASS = {md_params.pmass:.0f} has no
# direct LAMMPS press/langevin equivalent: that fix takes a damping *time* (Pdamp), not a mass,
# and derives its own fictitious barostat mass internally). VASP's Langevin barostat only exposes
# one damping timescale (1/LANGEVIN_GAMMA_L), so we reuse it for both LAMMPS keywords.
# LAMMPS docs (fix press/langevin) recommend friction ~= Pdamp for well-behaved dynamics, and
# warn that Pdamp much larger than that makes the box take a very long time to equilibrate.
variable      gamma_L equal {md_params.langevin_gamma_l:.3f}
variable      friction_L equal 1.0/v_gamma_L
variable      baro_pd equal v_friction_L
fix baro all press/langevin aniso ${{Ptarget}} ${{Ptarget}} ${{baro_pd}} temp ${{Tmd}} ${{Tmd}} 48280 flip no friction ${{friction_L}}
"""

        content += f"""
log log.lammps append
thermo_modify flush yes

# Run simulation
# Trajectory output during production (and restart continuations)
dump 1 all extxyz {self.lammps_dump_interval} trajectory.extxyz
dump_modify 1 sort id
dump_modify 1 element {elements_str}
dump_modify 1 append yes

timestep {self.lammps_timestep:.5f}
run ${{target}} upto

# Checkpoints: restart 10000 restart.lmp.a restart.lmp.b (submit script picks latest)
"""

        with open(input_path, 'w') as f:
            f.write(content)

    # --------------- Directory processing --------------- #

    def process_structure_dir(
        self,
        struct_path: str,
        temperatures: List[int],
        files_to_delete: List[str],
        keep_top_level_files: bool,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Process a single structure directory."""
        struct_path = os.path.abspath(struct_path)
        in_place_temp = self.parse_temperature_from_dirname(struct_path)
        
        try:
            entries = os.listdir(struct_path)
        except PermissionError as e:
            self.log(f"Skipping '{struct_path}': {e}", "WARNING")
            return {'processed': False, 'error': str(e)}
        
        top_files = [f for f in entries if os.path.isfile(os.path.join(struct_path, f))]
        
        # Auto-detect code type for this directory
        detected_code = self.detect_code(struct_path)
        if self.code == "auto":
            effective_code = detected_code
        else:
            effective_code = self.code
        
        self.log(f"Detected: {detected_code}, using: {effective_code}")
        
        # Find structure file
        poscar_path = os.path.join(struct_path, 'POSCAR')
        data_lammps_path = os.path.join(struct_path, 'data.lammps')
        
        if effective_code == "lammps" and os.path.exists(data_lammps_path):
            species_list, lattice = self.read_lammps_data(data_lammps_path)
        elif os.path.exists(poscar_path):
            species_list, lattice = self.read_poscar(poscar_path)
        else:
            self.log(f"No structure file found in {struct_path}", "ERROR")
            return {'processed': False, 'error': 'No structure file found'}
        
        if not species_list:
            return {'processed': False, 'error': 'Failed to read structure'}
        
        atoms = sum(s.count for s in species_list)
        
        # For VASP, analyze k-points
        kpoints_path = os.path.join(struct_path, 'KPOINTS')
        if effective_code == "vasp" and os.path.exists(kpoints_path):
            kpoints, kpoint_grid = self.analyze_kpoints(kpoints_path)
        else:
            kpoints, kpoint_grid = 1, (1, 1, 1)
        
        # Read existing INCAR (for VASP) or use defaults
        incar_path = os.path.join(struct_path, 'INCAR')
        existing_params = self.read_incar(incar_path) if os.path.exists(incar_path) else DEFAULT_PARAMS.copy()

        parallelization_params = None
        current_parallel = {'ncore': None, 'npar': None, 'kpar': None, 'nsim': None}
        if effective_code == "vasp":
            if os.path.exists(incar_path):
                current_parallel = self.analyze_incar_parallelization(incar_path)
            parallelization_params = self.calculate_optimal_parallelization(atoms, kpoints)

        system_info = SystemInfo(
            path=struct_path,
            atoms=atoms,
            kpoints=kpoints,
            kpoint_grid=kpoint_grid,
            species_list=species_list,
            lattice=lattice,
            current_ncore=current_parallel.get('ncore'),
            current_npar=current_parallel.get('npar'),
            current_kpar=current_parallel.get('kpar'),
            current_nsim=current_parallel.get('nsim'),
            optimal_ncore=parallelization_params['ncore'] if parallelization_params else None,
            optimal_npar=parallelization_params['npar'] if parallelization_params else None,
            optimal_kpar=parallelization_params['kpar'] if parallelization_params else None,
            optimal_nsim=parallelization_params['nsim'] if parallelization_params else None,
        )
        
        results = {
            'processed': True,
            'atoms': atoms,
            'kpoints': kpoints,
            'species_count': len(species_list),
            'temperatures_processed': 0,
            'system_info': system_info
        }
        
        # LAMMPS: update in place (no temp subdirs). Prefer temperature from dir name (e.g. 700K).
        # VASP: create temperature subdirectories unless already in a temp-named dir.
        if effective_code == "lammps":
            if in_place_temp is not None:
                target_temperatures = [in_place_temp]
            else:
                target_temperatures = temperatures[:1] if temperatures else [300]
            in_place_mode = True
        else:
            target_temperatures = [in_place_temp] if in_place_temp is not None else temperatures
            in_place_mode = in_place_temp is not None
        
        for temp in target_temperatures:
            if in_place_mode or effective_code == "lammps":
                temp_dir = struct_path
            else:
                temp_dir = os.path.join(struct_path, f"{temp}K")
            
            if not dry_run:
                if not in_place_mode and effective_code != "lammps":
                    os.makedirs(temp_dir, exist_ok=True)
                
                # Copy top-level files (VASP only, when creating temp subdirs)
                if not in_place_mode and effective_code != "lammps":
                    for file in top_files:
                        src = os.path.join(struct_path, file)
                        dst = os.path.join(temp_dir, file)
                        try:
                            shutil.copy2(src, dst)
                        except Exception as e:
                            self.log(f"Couldn't copy {src} -> {dst}: {e}", "WARNING")
                
                # Calculate MD parameters
                md_params = self.calculate_optimal_md_parameters(
                    species_list, temp, lattice, existing_params
                )
                system_info.optimal_md_params = md_params
                
                if effective_code == "vasp":
                    temp_incar_path = os.path.join(temp_dir, "INCAR")
                    self.write_vasp_incar(temp_incar_path, existing_params, temp, md_params)
                    self.update_incar_parallelization(temp_incar_path, parallelization_params)
                    
                    temp_kpoints_path = os.path.join(temp_dir, "KPOINTS")
                    self.write_kpoints(temp_kpoints_path)
                    
                    # CONTCAR -> POSCAR
                    contcar = os.path.join(temp_dir, "CONTCAR")
                    poscar = os.path.join(temp_dir, "POSCAR")
                    if os.path.exists(contcar):
                        try:
                            shutil.copy2(contcar, poscar)
                        except Exception as e:
                            self.log(f"Couldn't copy CONTCAR->POSCAR: {e}", "WARNING")
                else:
                    # LAMMPS: overwrite every existing input, or create the standard one.
                    existing_inputs = sorted(
                        f for f in os.listdir(temp_dir)
                        if f.startswith("in.") or f.endswith(".in") or f.endswith(".txt")
                    )
                    if existing_inputs:
                        for existing_input in existing_inputs:
                            input_path = os.path.join(temp_dir, existing_input)
                            self.write_lammps_input(input_path, species_list, md_params, temp)
                            self.log(f"Overwrote {existing_input}")
                    else:
                        input_name = (
                            "in.npt_md_allegro"
                            if md_params.ensemble == "npt"
                            else "in.nvt_allegro"
                        )
                        input_path = os.path.join(temp_dir, input_name)
                        self.write_lammps_input(input_path, species_list, md_params, temp)
                        self.log(f"Created {input_name}")
                
                # Clean up unwanted files
                for unwanted in files_to_delete:
                    fpath = os.path.join(temp_dir, unwanted)
                    if os.path.exists(fpath):
                        try:
                            if os.path.isfile(fpath) or os.path.islink(fpath):
                                os.remove(fpath)
                            else:
                                shutil.rmtree(fpath)
                        except Exception as e:
                            self.log(f"Couldn't remove {fpath}: {e}", "WARNING")
            
            results['temperatures_processed'] += 1
            self.log(f"Processed {temp}K: {temp_dir}")
        
        # Remove top-level files only for VASP when creating temp subdirs
        if not dry_run and not in_place_mode and effective_code != "lammps" and not keep_top_level_files:
            for file in top_files:
                try:
                    os.remove(os.path.join(struct_path, file))
                except FileNotFoundError:
                    pass
                except Exception as e:
                    self.log(f"Couldn't remove {file}: {e}", "WARNING")
        
        self.systems_analyzed.append(system_info)
        return results
    
    def process_path(
        self,
        path: str,
        temperatures: List[int],
        files_to_delete: List[str],
        keep_top_level_files: bool,
        dry_run: bool = False
    ):
        """Process a path (structure or parent directory)."""
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            self.log(f"Skipping '{path}': not a directory.", "WARNING")
            return
        
        if self.is_structure_dir(path):
            self.log(f"Processing structure: {path}")
            result = self.process_structure_dir(
                path, temperatures, files_to_delete, 
                keep_top_level_files, dry_run
            )
            if result['processed']:
                self.processed_dirs.append(path)
            else:
                self.skipped_dirs.append((path, result.get('error', 'Unknown error')))
            return
        
        structure_dirs = []
        for root, _dirs, _files in os.walk(path):
            if self.is_structure_dir(root):
                structure_dirs.append(root)

        structure_dirs = sorted(set(structure_dirs), key=lambda p: (p.count(os.sep), p))

        if not structure_dirs:
            self.log(f"No structure folders found under '{path}' (recursive).", "WARNING")
            return
        
        self.log(f"Processing parent recursively: {path} (structures: {len(structure_dirs)})")
        for struct_path in structure_dirs:
            result = self.process_structure_dir(
                struct_path, temperatures, files_to_delete,
                keep_top_level_files, dry_run
            )
            if result['processed']:
                self.processed_dirs.append(struct_path)
            else:
                self.skipped_dirs.append((struct_path, result.get('error', 'Unknown error')))


def get_cluster_configs() -> Dict[str, ClusterConfig]:
    """Get predefined cluster configurations."""
    return {
        "default": ClusterConfig("Default", 16, 8, 1, 8),
        "small": ClusterConfig("Small", 8, 8, 1, 4),
        "medium": ClusterConfig("Medium", 32, 8, 4, 8),
        "large": ClusterConfig("Large", 64, 8, 8, 8),
        "hpc": ClusterConfig("HPC", 128, 16, 8, 16),
        "custom": ClusterConfig("Custom", 16, 8, 1, 8)
    }
    

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive VASP/LAMMPS MD directory preparation and optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect VASP/LAMMPS, NPT (default ensemble)
  python3 prepare_directories_for_md.py /path/to/structures --temps 300,700

  # Auto-detect, NVT ensemble
  python3 prepare_directories_for_md.py /path/to/structures --nvt --temps 300,700

  # Force LAMMPS mode
  python3 prepare_directories_for_md.py /path/to/structures --lammps --temps 700

  # Force VASP mode
  python3 prepare_directories_for_md.py /path/to/structures --vasp --temps 300,700
        """
    )
    
    parser.add_argument("paths", nargs="*", help="One or more paths to structure folders.")
    parser.add_argument("--temps", default=",".join(map(str, DEFAULT_TEMPS)),
                       help="Comma-separated temperatures (default: 300,500,700,900,1100,1300).")
    parser.add_argument("--delete", default=None,
                       help="Comma-separated filenames to delete (auto-selected by code).")
    parser.add_argument("--keep-top-level-files", action="store_true",
                       help="Do NOT remove top-level files after populating temp folders.")
    
    # Ensemble selection
    parser.add_argument("--nvt", action="store_true",
                       help="Use NVT ensemble (constant volume). Default is NPT.")
    
    # Code selection (auto-detect by default)
    parser.add_argument("--lammps", action="store_true",
                       help="Force LAMMPS mode (auto-detected if data.lammps present).")
    parser.add_argument("--vasp", action="store_true",
                       help="Force VASP mode (auto-detected if INCAR/POTCAR present).")
    
    # VASP parallelization
    parser.add_argument("--cluster", choices=["default", "small", "medium", "large", "hpc", "custom"],
                       default="default", help="Predefined cluster configuration (VASP)")
    parser.add_argument("--cores", type=int, help="Total cores (overrides cluster config)")
    parser.add_argument("--cores-per-node", type=int, help="Cores per node")
    parser.add_argument("--nodes", type=int, help="Number of nodes")
    parser.add_argument("--max-ncore", type=int, help="Maximum NCORE value")
    
    # MD steps
    parser.add_argument("--nsw", type=int, default=10000, help="Number of MD steps for VASP (default: 10000)")
    parser.add_argument("--lammps-steps", type=int, default=1_000_000,
                        help="Target steps for LAMMPS (default: 1000000)")
    parser.add_argument("--lammps-timestep", type=float, default=0.00010,
                        help="Production LAMMPS timestep in ps (burn-in uses 0.00005; default: 0.00010)")
    
    parser.add_argument("--dry-run", action="store_true", help="Analyze only, don't update files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    # Langevin parameters
    parser.add_argument("--gamma-ref", type=float, default=2.0,
                        help="Reference gamma (ps^-1) at 300 K. Default 2.0")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Mass exponent for gamma. Default 0.6")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Temperature exponent for gamma. Default 0.5")
    parser.add_argument("--gamma-min", type=float, default=0.5,
                        help="Lower clamp for species gamma. Default 0.5")
    parser.add_argument("--gamma-max", type=float, default=15.0,
                        help="Upper clamp for species gamma. Default 15.0")
    parser.add_argument("--no-monotone", action="store_true",
                        help="Do not enforce gamma decreasing with mass.")
    
    args = parser.parse_args()

    configs = get_cluster_configs()
    cluster_config = configs[args.cluster]

    if args.cores:
        cluster_config.total_cores = args.cores
    if args.cores_per_node:
        cluster_config.cores_per_node = args.cores_per_node
    if args.nodes:
        cluster_config.nodes = args.nodes
    if args.max_ncore:
        cluster_config.max_ncore = args.max_ncore
    
    ensemble = "nvt" if args.nvt else "npt"
    
    # Code selection: explicit flags override auto-detection
    if args.lammps and args.vasp:
        print("Error: Cannot specify both --lammps and --vasp")
        sys.exit(1)
    elif args.lammps:
        code = "lammps"
    elif args.vasp:
        code = "vasp"
    else:
        code = "auto"  # Auto-detect per directory
    
    processor = MDDirectoryProcessor(
        cluster_config,
        verbose=args.verbose and not args.quiet,
        nsw=args.nsw,
        gamma_ref=args.gamma_ref,
        alpha=args.alpha,
        beta=args.beta,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        enforce_monotone=(not args.no_monotone),
        ensemble=ensemble,
        code=code,
        lammps_timestep=args.lammps_timestep,
        lammps_target_steps=args.lammps_steps,
    )
    
    temperatures = processor.parse_temps(args.temps)
    
    if args.delete:
        files_to_delete = [x.strip() for x in args.delete.split(",") if x.strip()]
    else:
        files_to_delete = DEFAULT_DELETE_LAMMPS if args.lammps else DEFAULT_DELETE_VASP
    
    roots = args.paths if args.paths else ["."]
    seen = set()
    unique_roots = []
    for r in roots:
        if r not in seen:
            unique_roots.append(r)
            seen.add(r)
    
    for path in unique_roots:
        processor.process_path(
            path, temperatures, files_to_delete, 
            args.keep_top_level_files, args.dry_run
        )
    
    if processor.skipped_dirs and not args.dry_run:
        processor.log(f"Processing completed with {len(processor.skipped_dirs)} skipped directories", "WARNING")
        sys.exit(1)
    else:
        processor.log(f"Processing completed successfully! Processed {len(processor.processed_dirs)} directories", "INFO")
        sys.exit(0)


if __name__ == "__main__":
    main()
