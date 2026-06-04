#!/usr/bin/env python3
"""
VASP NPT Directory Preparation and Optimization Suite
=====================================================

This comprehensive script combines three major functionalities:

1. Directory Preparation (from prepare_directories_for_md.py):
   - Creates temperature-specific subdirectories (e.g., 300K, 500K, etc.)
   - Copies VASP input files to each temperature directory
   - Sets up proper INCAR and KPOINTS files
   - Cleans up unwanted output files

2. NPT Parameter Optimization (incorporates npt_incar_optimizer.py):
   - Mass- and temperature-aware Langevin gamma per species
   - PMASS and LANGEVIN_GAMMA_L scaled by system size and temperature
   - POTIM fixed at 1.0 fs; ISTART=1, ICHARG=1 (restart from WAVECAR/CHGCAR)
   - Tunable via --gamma-ref, --alpha, --beta, --gamma-min, --gamma-max

3. Parallelization Optimization (from optimize_vasp_incar.py):
   - Analyzes system size and k-point density
   - Calculates optimal NCORE, NPAR, NSIM parameters
   - Note: KPAR is not calculated and will be removed from INCAR if present
   - Supports multiple cluster configurations
   - Ensures proper core utilization

Usage:
    python3 prepare_directories_for_npt.py <directory_path> [options]

Examples:
    python3 prepare_directories_for_npt.py /path/to/structures
    python3 prepare_directories_for_npt.py . --temps 300,500,700 --cluster medium
    python3 prepare_directories_for_npt.py /data/vasp --dry-run --backup

Author: AI Assistant
Version: 1.0 (Combined)
"""

import os
import sys
import argparse
import shutil
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

# Default configurations
DEFAULT_TEMPS = [300, 500, 700, 900, 1100, 1300]
DEFAULT_DELETE = [
    "OUTCAR", "OSZICAR", "job.err", "job.out", "PROCAR",
    "REPORT", "vasprun.xml", "XDATCAR", "ICONST", "IBZKPT",
    "DOSCAR", "EIGENVAL"
]

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

@dataclass
class AtomicSpecies:
    """Data class for atomic species information."""
    symbol: str
    atomic_number: int
    atomic_mass: float
    count: int

@dataclass
class NPTParameters:
    """Data class for NPT simulation parameters."""
    # Core MD parameters
    ibrion: int = 0
    mdalgo: int = 3
    isif: int = 3
    nsw: int = 10000
    potim: float = 1.0
    
    # Temperature control
    tebeg: Optional[float] = None
    teend: Optional[float] = None
    
    # Langevin thermostat parameters
    langevin_gamma: List[float] = None
    langevin_gamma_l: float = 10.0
    
    # Barostat parameters
    pmass: float = 1000.0
    
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
    optimal_npt_params: Optional[NPTParameters] = None

@dataclass
class ClusterConfig:
    """Cluster configuration parameters."""
    name: str
    total_cores: int
    cores_per_node: int
    nodes: int
    max_ncore: int  # Maximum recommended NCORE value

class VASPNPTProcessor:
    """
    Comprehensive VASP NPT processor that combines directory preparation,
    NPT parameter optimization, and parallelization optimization.
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
    }
    
    def __init__(self, cluster_config: ClusterConfig, verbose: bool = True, nsw: int = 10000,
                 gamma_ref: float = 2.0, alpha: float = 0.6, beta: float = 0.5,
                 gamma_min: float = 0.5, gamma_max: float = 15.0,
                 enforce_monotone: bool = True):
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
    
    def is_structure_dir(self, path: str) -> bool:
        """Check if directory contains VASP structure files."""
        if not os.path.isdir(path):
            return False
        try:
            names = set(os.listdir(path))
        except PermissionError:
            return False
        return len(VASP_HINT_FILES & names) > 0
    
    def read_poscar(self, poscar_path: str) -> Tuple[List[AtomicSpecies], np.ndarray]:
        """Read POSCAR file and extract atomic species information."""
        try:
            with open(poscar_path, 'r') as f:
                lines = f.readlines()
            
            # Read scaling factor (line 1)
            scaling_factor = float(lines[1].strip().split()[0])
            
            # Read lattice vectors (lines 2-4) and apply scaling factor
            lattice = np.array([
                [float(x) for x in lines[2].split()],
                [float(x) for x in lines[3].split()],
                [float(x) for x in lines[4].split()]
            ]) * scaling_factor
            
            # Read species and counts (lines 5-6)
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
    
    def analyze_poscar(self, poscar_path: str) -> int:
        """Analyze POSCAR to determine number of atoms."""
        try:
            with open(poscar_path, 'r') as f:
                lines = f.readlines()
            
            # Clean lines and remove empty ones
            clean_lines = [line.strip() for line in lines if line.strip()]
            
            if len(clean_lines) < 7:
                raise ValueError("POSCAR file too short")
            
            # Find the atom count line by looking for a line with only integers
            atom_line_idx = None
            for i, line in enumerate(clean_lines):
                if re.match(r'^\s*\d+(\s+\d+)*\s*$', line):
                    if i > 4:  # Should be after lattice vectors
                        atom_line_idx = i
                        break
            
            if atom_line_idx is None:
                raise ValueError("Could not find atom count line in POSCAR")
            
            atom_counts = [int(x) for x in clean_lines[atom_line_idx].split()]
            total_atoms = sum(atom_counts)
            
            self.log(f"POSCAR analysis: {total_atoms} atoms ({atom_counts})")
            return total_atoms
            
        except Exception as e:
            self.log(f"Error analyzing POSCAR {poscar_path}: {e}", "ERROR")
            return 0
    
    def analyze_kpoints(self, kpoints_path: str) -> Tuple[int, Tuple[int, int, int]]:
        """Analyze KPOINTS to determine k-point grid."""
        try:
            with open(kpoints_path, 'r') as f:
                lines = f.readlines()
            
            # Find Monkhorst-Pack grid line
            for i, line in enumerate(lines):
                if line.strip().lower().startswith('monkhorst'):
                    if i + 1 < len(lines):
                        grid_line = lines[i + 1].strip()
                        grid = [int(x) for x in grid_line.split()]
                        if len(grid) >= 3:
                            kpoints = grid[0] * grid[1] * grid[2]
                            self.log(f"KPOINTS analysis: {kpoints} k-points ({grid[0]}×{grid[1]}×{grid[2]})")
                            return kpoints, (grid[0], grid[1], grid[2])
            
            # Fallback: look for any line with 3 integers
            for line in lines:
                if re.match(r'^\s*\d+\s+\d+\s+\d+', line.strip()):
                    grid = [int(x) for x in line.split()[:3]]
                    kpoints = grid[0] * grid[1] * grid[2]
                    self.log(f"KPOINTS analysis (fallback): {kpoints} k-points ({grid[0]}×{grid[1]}×{grid[2]})")
                    return kpoints, (grid[0], grid[1], grid[2])
            
            raise ValueError("Could not find k-point grid in KPOINTS")
            
        except Exception as e:
            self.log(f"Error analyzing KPOINTS {kpoints_path}: {e}", "ERROR")
            return 0, (1, 1, 1)
    
    def read_incar(self, incar_path: str) -> Dict[str, Any]:
        """Read INCAR file and extract parameters."""
        params = {}
        try:
            with open(incar_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if not line or line.startswith('#') or line.startswith('!'):
                        continue
                    
                    # Parse parameter = value
                    if '=' in line:
                        parts = line.split('=', 1)
                        key = parts[0].strip()
                        value_str = parts[1].strip()
                        # Strip inline comments commonly used in INCAR:
                        # EDIFFG = -5E-02  force criterion ! note
                        # EDIFFG = -5E-02  # note
                        for marker in ('#', '!'):
                            if marker in value_str:
                                value_str = value_str.split(marker, 1)[0].strip()

                        # Skip KPAR - remove it if found
                        if key.upper() == 'KPAR':
                            continue
                        
                        # Handle different value types
                        try:
                            # Try integer first
                            if '.' not in value_str and 'E' not in value_str.upper():
                                params[key] = int(value_str)
                            else:
                                # Try float
                                params[key] = float(value_str)
                        except ValueError:
                            # Some INCAR lines contain trailing prose after numeric values,
                            # e.g. "EDIFFG = -5E-02 force stopping-criterion".
                            # Fall back to parsing the first token when possible.
                            first_token = value_str.split()[0] if value_str.split() else value_str
                            try:
                                if re.fullmatch(r'[+-]?\d+', first_token):
                                    params[key] = int(first_token)
                                elif re.fullmatch(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?', first_token):
                                    params[key] = float(first_token)
                                else:
                                    params[key] = value_str
                            except ValueError:
                                # Keep as string
                                params[key] = value_str
                            
        except Exception as e:
            self.log(f"Error reading INCAR {incar_path}: {e}", "ERROR")
            
        return params
    
    # --------------- Physics heuristics from VASPNPTOptimizer --------------- #

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _compute_langevin_gammas(
        self,
        species_list: List[AtomicSpecies],
        temperature: float,
        gamma_ref: float = 2.0,
        alpha: float = 0.6,
        beta: float = 0.5,
        gamma_min: float = 0.5,
        gamma_max: float = 15.0,
        enforce_monotone: bool = True
    ) -> List[float]:
        """
        Per-species gamma that increases for lighter atoms and grows with T.
        Uses composition-weighted average mass as reference.
        """
        total_atoms = sum(s.count for s in species_list)
        avg_mass = sum(s.atomic_mass * s.count for s in species_list) / max(1, total_atoms)
        m_ref = avg_mass

        temp_factor = (max(1e-6, temperature) / 300.0) ** beta

        raw = []
        for s in species_list:
            g = gamma_ref * (m_ref / s.atomic_mass) ** alpha * temp_factor
            g = self._clamp(g, gamma_min, gamma_max)
            raw.append(g)

        if not enforce_monotone:
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
        gammas = [self._clamp(g, gamma_min, gamma_max) for g in gammas]
        return gammas

    def _compute_pmass(self, natoms: int, temperature: float) -> float:
        """PMASS scaled by system size and temperature, clamped to [800, 5000]."""
        size_factor = self._clamp(natoms / 50.0, 0.5, 3.0)
        t_factor = self._clamp(temperature / 300.0, 0.5, 2.0)
        pmass = 1000.0 * size_factor * t_factor
        return float(self._clamp(pmass, 800.0, 5000.0))

    def _compute_gamma_l(self, natoms: int) -> float:
        """Lattice friction scaled modestly by system size, clamped to [1, 10]."""
        size_factor = self._clamp(natoms / 50.0, 0.5, 3.0)
        gamma_l = 3.0 * size_factor
        return float(self._clamp(gamma_l, 1.0, 10.0))

    # -------------------------------------------------------------------- #

    def calculate_optimal_npt_parameters(self, species_list: List[AtomicSpecies],
                                         temperature: float, lattice: np.ndarray,
                                         existing_params: Dict[str, Any]) -> NPTParameters:
        """Calculate optimal NPT parameters using mass- and T-aware heuristics."""
        total_atoms = sum(s.count for s in species_list)

        langevin_gamma = self._compute_langevin_gammas(
            species_list=species_list,
            temperature=temperature,
            gamma_ref=self.gamma_ref,
            alpha=self.alpha,
            beta=self.beta,
            gamma_min=self.gamma_min,
            gamma_max=self.gamma_max,
            enforce_monotone=self.enforce_monotone
        )

        pmass = self._compute_pmass(total_atoms, temperature)
        gamma_l = self._compute_gamma_l(total_atoms)

        npt_params = NPTParameters(
            ibrion=0,
            mdalgo=3,
            isif=3,
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

        return npt_params

    def calculate_optimal_parallelization(self, atoms: int, kpoints: int) -> Dict[str, int]:
        """Calculate optimal parallelization parameters based on system characteristics."""
        total_cores = self.cluster_config.total_cores

        # Determine system size category
        if atoms <= 50:
            system_size = "small"
        elif atoms <= 100:
            system_size = "medium"
        else:
            system_size = "large"

        # Calculate optimal parameters based on system size and k-point density
        # Note: KPAR calculation is disabled - KPAR will be removed from INCAR if present
        if system_size == "small":
            if kpoints >= 64:
                # optimal_kpar = min(8, kpoints // 8)  # KPAR calculation disabled
                optimal_ncore = min(8, total_cores // 2)
                optimal_npar = total_cores // optimal_ncore
            else:
                # optimal_kpar = min(4, kpoints // 4)  # KPAR calculation disabled
                optimal_ncore = min(4, total_cores // 4)
                optimal_npar = total_cores // optimal_ncore

        elif system_size == "medium":
            if kpoints >= 64:
                # optimal_kpar = min(8, kpoints // 8)  # KPAR calculation disabled
                optimal_ncore = min(4, total_cores // 4)
                optimal_npar = total_cores // optimal_ncore
            else:
                # optimal_kpar = min(4, kpoints // 4)  # KPAR calculation disabled
                optimal_ncore = min(4, total_cores // 4)
                optimal_npar = total_cores // optimal_ncore

        else:  # large systems
            if kpoints >= 64:
                # optimal_kpar = min(4, kpoints // 16)  # KPAR calculation disabled
                optimal_ncore = min(2, total_cores // 8)
                optimal_npar = total_cores // optimal_ncore
            else:
                # optimal_kpar = min(2, kpoints // 9)  # KPAR calculation disabled
                optimal_ncore = min(2, total_cores // 8)
                optimal_npar = total_cores // optimal_ncore

        # Ensure NSIM is always set to 4 for optimal band parallelization
        optimal_nsim = 4

        # Validate that NCORE × NPAR = total_cores
        if optimal_ncore * optimal_npar != total_cores:
            optimal_npar = total_cores // optimal_ncore
            if optimal_ncore * optimal_npar != total_cores:
                self.log(f"Warning: Could not achieve exact core matching for {atoms} atoms", "WARNING")

        # KPAR calculation and validation disabled
        # optimal_kpar = min(optimal_kpar, kpoints)

        # Ensure all parameters are reasonable
        optimal_ncore = max(1, min(optimal_ncore, self.cluster_config.max_ncore))
        optimal_npar = max(1, optimal_npar)
        # optimal_kpar = max(1, optimal_kpar)  # KPAR validation disabled

        return {
            'ncore': optimal_ncore,
            'npar': optimal_npar,
            # 'kpar': optimal_kpar,  # KPAR not returned
            'nsim': optimal_nsim
        }
    
    def write_kpoints(self, kpoints_path: str):
        """Write standard KPOINTS file."""
        with open(kpoints_path, "w") as f:
            f.write(KPOINTS_CONTENT)
    
    def write_or_update_incar(self, incar_path: str, params: Dict[str, Any], 
                             temp: int, npt_params: NPTParameters,
                             parallelization_params: Dict[str, int]):
        """Write optimized INCAR file with all parameters."""
        try:
            # Prepare all parameters
            all_params = params.copy()

            # Remove KPAR if present (KPAR is not used)
            if 'KPAR' in all_params:
                del all_params['KPAR']
            
            # Force restart from WAVECAR and CHGCAR
            all_params['ISTART'] = 1
            all_params['ICHARG'] = 1

            # Add temperature parameters
            all_params['TEBEG'] = temp
            all_params['TEEND'] = temp
            
            # Add NPT parameters
            all_params.update({
                'IBRION': npt_params.ibrion,
                'MDALGO': npt_params.mdalgo,
                'ISIF': npt_params.isif,
                'NSW': npt_params.nsw,
                'POTIM': f'{npt_params.potim:.3f}',
                'LANGEVIN_GAMMA': ' '.join(f'{g:.3f}' for g in npt_params.langevin_gamma),
                'LANGEVIN_GAMMA_L': f'{npt_params.langevin_gamma_l:.3f}',
                'PMASS': f'{npt_params.pmass:.0f}',
                'EDIFFG': npt_params.ediffg,
                'ISYM': npt_params.isym
            })

            # Add parallelization parameters
            all_params.update({
                'NCORE': parallelization_params['ncore'],
                'NPAR': parallelization_params['npar'],
                # 'KPAR': parallelization_params['kpar'],  # KPAR not written
                'NSIM': parallelization_params['nsim'],
                'LPLANE': '.TRUE.',
                'LSCALU': '.FALSE.'
            })
            
            # Ensure critical parameters are set correctly (override any existing values)
            all_params['ALGO'] = 'Very Fast'
            all_params['PREC'] = 'Normal'
            all_params['EDIFF'] = 1E-5  # Suitable for MD (tighter not needed)
            
            # Write INCAR file with organized sections
            with open(incar_path, 'w') as f:
                # Header
                f.write("System = Optimized NPT Molecular Dynamics\n")
                f.write(f"# Parameters optimized for {temp}K (mass- and T-aware Langevin)\n\n")
                
                # Starting parameters
                f.write("Starting parameters:\n")
                for key in ['ISTART', 'ICHARG']:
                    if key in all_params:
                        f.write(f"{key} = {all_params[key]}\n")
                f.write("\n")
                
                # Electronic relaxation
                f.write("Electronic Relaxation:\n")
                electronic_keys = ['PREC', 'ENCUT', 'NELMIN', 'NELM', 'EDIFF', 'LREAL', 
                                 'ISPIN', 'MAGMOM', 'ALGO', 'METAGGA', 'LMIXTAU', 'LASPH', 'LDIAG',
                                 'ISMEAR', 'SIGMA', 'LORBIT']
                for key in electronic_keys:
                    if key in all_params:
                        f.write(f"{key} = {all_params[key]}\n")
                f.write("\n")
                
                # Ionic Molecular Dynamics
                f.write("Ionic Molecular Dynamics (NPT, Langevin):\n")
                md_keys = ['NSW', 'IBRION', 'EDIFFG', 'ISIF', 'POTIM', 'ISYM', 
                          'MDALGO', 'LANGEVIN_GAMMA', 'LANGEVIN_GAMMA_L', 'PMASS']
                for key in md_keys:
                    if key in all_params:
                        f.write(f"{key} = {all_params[key]}\n")
                f.write("\n")
                
                # Temperature control
                f.write("Temperature Control:\n")
                f.write(f"TEBEG = {temp}\n")
                f.write(f"TEEND = {temp}\n")
                f.write("\n")

                # Parallelization flags
                f.write("Parallelization flags:\n")
                parallelization_keys = ['NCORE', 'NPAR', 'NSIM', 'LPLANE', 'LSCALU']  # KPAR removed
                for key in parallelization_keys:
                    if key in all_params:
                        f.write(f"{key} = {all_params[key]}\n")
                f.write("\n")
                
                # Space-saving flags
                f.write("# Space-saving flags\n")
                f.write("LWAVE  = .FALSE.\n")
                f.write("LCHARG = .FALSE.\n")
                        
        except Exception as e:
            self.log(f"Error writing INCAR {incar_path}: {e}", "ERROR")
            raise
    
    def process_structure_dir(self, struct_path: str, temperatures: List[int], 
                            files_to_delete: List[str], keep_top_level_files: bool,
                            dry_run: bool = False) -> Dict[str, Any]:
        """Process a single structure directory with all optimizations."""
        struct_path = os.path.abspath(struct_path)
        temp_dir_match = re.fullmatch(r"(\d+)K", os.path.basename(struct_path))
        in_place_temp = int(temp_dir_match.group(1)) if temp_dir_match else None
        
        # Collect top-level files
        try:
            entries = os.listdir(struct_path)
        except PermissionError as e:
            self.log(f"Skipping '{struct_path}': {e}", "WARNING")
            return {'processed': False, 'error': str(e)}
        
        top_files = [f for f in entries if os.path.isfile(os.path.join(struct_path, f))]
        
        # Analyze the system for optimization
        poscar_path = os.path.join(struct_path, 'POSCAR')
        kpoints_path = os.path.join(struct_path, 'KPOINTS')
        incar_path = os.path.join(struct_path, 'INCAR')
        
        if not os.path.exists(poscar_path):
            self.log(f"POSCAR not found in {struct_path}", "ERROR")
            return {'processed': False, 'error': 'POSCAR not found'}
        
        # Read system information
        species_list, lattice = self.read_poscar(poscar_path)
        if not species_list:
            return {'processed': False, 'error': 'Failed to read POSCAR'}
        
        atoms = sum(s.count for s in species_list)
        
        # Analyze k-points
        kpoints, kpoint_grid = self.analyze_kpoints(kpoints_path) if os.path.exists(kpoints_path) else (1, (1, 1, 1))
        
        # Read existing INCAR parameters
        existing_params = self.read_incar(incar_path) if os.path.exists(incar_path) else DEFAULT_PARAMS.copy()

        # Calculate optimal parameters
        parallelization_params = self.calculate_optimal_parallelization(atoms, kpoints)

        # Store system info
        system_info = SystemInfo(
            path=struct_path,
            atoms=atoms,
            kpoints=kpoints,
            kpoint_grid=kpoint_grid,
            species_list=species_list,
            lattice=lattice,
            optimal_ncore=parallelization_params['ncore'],
            optimal_npar=parallelization_params['npar'],
            optimal_kpar=None,  # KPAR not calculated
            optimal_nsim=parallelization_params['nsim']
        )
        
        results = {
            'processed': True,
            'atoms': atoms,
            'kpoints': kpoints,
            'species_count': len(species_list),
            'temperatures_processed': 0,
            'system_info': system_info
        }
        
        # Process each temperature.
        # If the input directory is already a temperature directory (e.g. 300K),
        # update that directory in place and do not create nested temperature folders.
        target_temperatures = [in_place_temp] if in_place_temp is not None else temperatures
        for temp in target_temperatures:
            temp_dir = os.path.join(struct_path, f"{temp}K")
            if in_place_temp is not None:
                temp_dir = struct_path
            
            if not dry_run:
                if in_place_temp is None:
                    os.makedirs(temp_dir, exist_ok=True)
                
                # Copy top-level files into temp dir
                if in_place_temp is None:
                    for file in top_files:
                        src = os.path.join(struct_path, file)
                        dst = os.path.join(temp_dir, file)
                        try:
                            shutil.copy2(src, dst)
                        except Exception as e:
                            self.log(f"Couldn't copy {src} -> {dst}: {e}", "WARNING")
                
                # Calculate NPT parameters for this temperature
                npt_params = self.calculate_optimal_npt_parameters(
                    species_list, temp, lattice, existing_params
                )
                system_info.optimal_npt_params = npt_params
                
                # Write optimized INCAR
                temp_incar_path = os.path.join(temp_dir, "INCAR")
                self.write_or_update_incar(
                    temp_incar_path, existing_params, temp, 
                    npt_params, parallelization_params
                )
                
                # Write KPOINTS
                temp_kpoints_path = os.path.join(temp_dir, "KPOINTS")
                self.write_kpoints(temp_kpoints_path)
                
                # CONTCAR -> POSCAR (if available)
                contcar = os.path.join(temp_dir, "CONTCAR")
                poscar = os.path.join(temp_dir, "POSCAR")
                if os.path.exists(contcar):
                    try:
                        shutil.copy2(contcar, poscar)
                    except Exception as e:
                        self.log(f"Couldn't copy CONTCAR->POSCAR in {temp_dir}: {e}", "WARNING")
                
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
            self.log(f"Processed {temp}K directory: {temp_dir}")
        
        # Remove top-level files if requested (not applicable in-place temp mode)
        if not dry_run and in_place_temp is None and not keep_top_level_files:
            for file in top_files:
                try:
                    os.remove(os.path.join(struct_path, file))
                except FileNotFoundError:
                    pass
                except Exception as e:
                    self.log(f"Couldn't remove {file} in {struct_path}: {e}", "WARNING")
        
        self.systems_analyzed.append(system_info)
        return results
    
    def process_path(self, path: str, temperatures: List[int], files_to_delete: List[str],
                    keep_top_level_files: bool, dry_run: bool = False):
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
        
        # Otherwise, recursively discover structure directories.
        # A structure directory is any directory containing VASP hint files.
        structure_dirs = []
        for root, _dirs, _files in os.walk(path):
            if self.is_structure_dir(root):
                structure_dirs.append(root)

        # Process shallow paths first for stable and predictable behavior.
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
        "custom": ClusterConfig("Custom", 16, 8, 1, 8)  # Will be updated based on user input
    }
    
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive VASP NPT directory preparation and optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 prepare_directories_for_npt.py /path/to/structures
  python3 prepare_directories_for_npt.py . --temps 300,500,700 --cluster medium
  python3 prepare_directories_for_npt.py /data/vasp --dry-run
  python3 prepare_directories_for_npt.py /structures --cores 32 --verbose
        """
    )
    
    parser.add_argument("paths", nargs="*", help="One or more paths. Each path may be a structure folder or a parent folder.")
    parser.add_argument("--temps", default=",".join(map(str, DEFAULT_TEMPS)),
                       help="Comma-separated temperatures, e.g. 300,500,700 (default: 300,500,700,900,1100,1300).")
    parser.add_argument("--delete", default=",".join(DEFAULT_DELETE),
                       help="Comma-separated filenames to delete inside each temp folder.")
    parser.add_argument("--keep-top-level-files", action="store_true",
                       help="If set, do NOT remove top-level files in each structure folder after populating.")
    parser.add_argument("--cluster", choices=["default", "small", "medium", "large", "hpc", "custom"],
                       default="default", help="Predefined cluster configuration")
    parser.add_argument("--cores", type=int, help="Total number of cores (overrides cluster config)")
    parser.add_argument("--cores-per-node", type=int, help="Cores per node (overrides cluster config)")
    parser.add_argument("--nodes", type=int, help="Number of nodes (overrides cluster config)")
    parser.add_argument("--max-ncore", type=int, help="Maximum NCORE value")
    parser.add_argument("--nsw", type=int, default=10000, help="Number of MD steps (default: 10000)")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only, don't update files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    # NPT optimizer expert knobs
    parser.add_argument("--gamma-ref", type=float, default=2.0,
                        help="Reference gamma (ps^-1) at 300 K for avg mass. Default 2.0")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Mass exponent: gamma ~ (m_ref/m)^alpha. Default 0.6")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Temperature exponent: gamma ~ (T/300)^beta. Default 0.5")
    parser.add_argument("--gamma-min", type=float, default=0.5,
                        help="Lower clamp for species gamma (ps^-1). Default 0.5")
    parser.add_argument("--gamma-max", type=float, default=15.0,
                        help="Upper clamp for species gamma (ps^-1). Default 15.0")
    parser.add_argument("--no-monotone", action="store_true",
                        help="Do not enforce gamma decreasing with mass.")
    
    args = parser.parse_args()

    # Get cluster configuration
    configs = get_cluster_configs()
    cluster_config = configs[args.cluster]

    # Override with custom values if provided
    if args.cores:
        cluster_config.total_cores = args.cores
    if args.cores_per_node:
        cluster_config.cores_per_node = args.cores_per_node
    if args.nodes:
        cluster_config.nodes = args.nodes
    if args.max_ncore:
        cluster_config.max_ncore = args.max_ncore
    
    # Create processor
    processor = VASPNPTProcessor(
        cluster_config,
        verbose=args.verbose and not args.quiet,
        nsw=args.nsw,
        gamma_ref=args.gamma_ref,
        alpha=args.alpha,
        beta=args.beta,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        enforce_monotone=(not args.no_monotone)
    )
    
    # Parse temperatures and files to delete
    temperatures = processor.parse_temps(args.temps)
    files_to_delete = [x.strip() for x in args.delete.split(",") if x.strip()]
    
    # Default to current directory if none provided
    roots = args.paths if args.paths else ["."]
    # De-duplicate while preserving order
    seen = set()
    unique_roots = []
    for r in roots:
        if r not in seen:
            unique_roots.append(r)
            seen.add(r)
    
    # Process each path
    for path in unique_roots:
        processor.process_path(
            path, temperatures, files_to_delete, 
            args.keep_top_level_files, args.dry_run
        )
    
    # Exit with appropriate code
    if processor.skipped_dirs and not args.dry_run:
        processor.log(f"Processing completed with {len(processor.skipped_dirs)} skipped directories", "WARNING")
        sys.exit(1)
    else:
        processor.log(f"Processing completed successfully! Processed {len(processor.processed_dirs)} directories", "INFO")
        sys.exit(0)

if __name__ == "__main__":
    main()
