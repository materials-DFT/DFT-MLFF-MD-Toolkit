# The MIT License (MIT)
#
# Copyright (c) 2014 Muratahan Aykol
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE

import numpy as np
from copy import deepcopy
import re
import argparse
import os
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

_FLOAT_RE = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)


def _parse_first_three_floats(line, source=""):
    """Parse the first 3 float values from a coordinate line.

    Robust against null bytes and trailing non-numeric garbage sometimes seen in
    partially corrupted text files on shared/network filesystems.
    """
    clean = line.replace("\x00", " ")
    vals = _FLOAT_RE.findall(clean)
    if len(vals) < 3:
        raise ValueError(f"Could not parse 3 floats from {source}: {line!r}")
    return [float(v) for v in vals[:3]]


def read_xdatcar(filename):
    """
    Read VASP XDATCAR file and convert to fractional XYZ format.
    XDATCAR already contains fractional (Direct) coordinates.
    Returns list of frames, each frame is dict with 'lattice', 'elements', 'coords' (fractional).
    
    Robust against null bytes and corrupted lines on network filesystems.
    """
    frames = []
    n_bad = 0
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        comment = lines[i].strip()
        i += 1
        if i >= len(lines):
            break
            
        scale = float(lines[i].replace("\x00", " ").strip().split()[0])
        i += 1
        
        lattice = []
        for _ in range(3):
            vec = [float(x) * scale for x in lines[i].replace("\x00", " ").split()[:3]]
            lattice.append(vec)
            i += 1
        lattice = np.array(lattice)
        
        elements_line = lines[i].replace("\x00", " ").split()
        i += 1
        counts_line = [int(x) for x in lines[i].replace("\x00", " ").split()]
        i += 1
        
        elements = []
        for elem, count in zip(elements_line, counts_line):
            elements.extend([elem] * count)
        
        n_atoms = sum(counts_line)
        
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if line.startswith('Direct') or line.startswith('direct') or 'direct' in line.lower().replace("\x00", ""):
                i += 1
                coords = []
                bad_frame = False
                for _ in range(n_atoms):
                    if i >= len(lines):
                        break
                    try:
                        coord = _parse_first_three_floats(lines[i], f"XDATCAR {filename}")
                        coords.append(coord)
                    except ValueError:
                        bad_frame = True
                        n_bad += 1
                        for _ in range(n_atoms - len(coords)):
                            i += 1
                            if i >= len(lines):
                                break
                        break
                    i += 1
                
                if not bad_frame and len(coords) == n_atoms:
                    frames.append({
                        'lattice': lattice.copy(),
                        'elements': elements,
                        'coords': np.array(coords)
                    })
            elif any(c.isalpha() for c in line.replace("\x00", "")) and 'direct' not in line.lower().replace("\x00", ""):
                break
            else:
                i += 1
    
    if n_bad > 0:
        print(f"  Warning: skipped {n_bad} corrupted frame(s) in XDATCAR")
    
    return frames


def read_outcar(filename):
    """
    Read VASP OUTCAR file and extract positions and lattice vectors per ionic step.
    Returns list of frames in the same format as read_xdatcar.
    Positions in OUTCAR are Cartesian (Å), converted to fractional here.
    """
    frames = []
    species_names = []
    species_counts = []
    n_atoms = None
    cell = None
    
    with open(filename, 'r') as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            
            if "TITEL" in line:
                species_names.append(line.split("=")[1].split()[1].split("_")[0])
                continue
            
            if "ions per type" in line:
                species_counts = [int(x) for x in line.split("=")[1].split()]
                n_atoms = sum(species_counts)
                continue
            
            if "direct lattice vectors" in line and "reciprocal" in line:
                cell = np.empty((3, 3))
                for i in range(3):
                    cell[i] = [float(x) for x in fh.readline().split()[:3]]
                continue
            
            if n_atoms and line.startswith(" POSITION") and "TOTAL-FORCE" in line:
                fh.readline()  # separator "----"
                pos_cart = np.empty((n_atoms, 3))
                for a in range(n_atoms):
                    pos_cart[a] = [float(x) for x in fh.readline().split()[:3]]
                
                cell_inv = np.linalg.inv(cell)
                pos_frac = np.dot(pos_cart, cell_inv)
                pos_frac = pos_frac % 1.0
                
                elements = []
                for name, count in zip(species_names, species_counts):
                    elements.extend([name] * count)
                
                frames.append({
                    'lattice': cell.copy(),
                    'elements': elements,
                    'coords': pos_frac
                })
                
                if len(frames) % 5000 == 0:
                    print(f"    {len(frames)} frames ...", flush=True)
    
    print(f"  Read {len(frames)} frames from OUTCAR, {n_atoms} atoms")
    return frames


def read_extxyz(filename):
    """
    Read LAMMPS extended XYZ trajectory file.
    Converts Cartesian coordinates to fractional coordinates.
    Returns list of frames, each frame is dict with 'lattice', 'elements', 'coords' (fractional).
    """
    frames = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        try:
            n_atoms = int(lines[i].strip())
        except (ValueError, IndexError):
            i += 1
            continue
        i += 1
        
        if i >= len(lines):
            break
        comment = lines[i].strip()
        i += 1
        
        lattice_match = re.search(r'Lattice="([^"]+)"', comment)
        if lattice_match:
            lattice_values = [float(x) for x in lattice_match.group(1).split()]
            lattice = np.array(lattice_values).reshape(3, 3)
        else:
            raise ValueError(f"Could not find Lattice in comment line: {comment}")
        
        lattice_inv = np.linalg.inv(lattice)
        
        elements = []
        coords_cart = []
        for _ in range(n_atoms):
            if i >= len(lines):
                break
            parts = lines[i].split()
            elements.append(parts[0])
            cart_coord = [float(parts[1]), float(parts[2]), float(parts[3])]
            coords_cart.append(cart_coord)
            i += 1
        
        coords_cart = np.array(coords_cart)
        coords_frac = np.dot(coords_cart, lattice_inv)
        
        coords_frac = coords_frac % 1.0
        
        frames.append({
            'lattice': lattice,
            'elements': elements,
            'coords': coords_frac
        })
    
    return frames


def write_fract_xyz(frames, output_file):
    """
    Write frames to fractional XYZ format that MSD function expects.
    """
    with open(output_file, 'w') as f:
        for frame in frames:
            n_atoms = len(frame['elements'])
            f.write(f"{n_atoms}\n")
            f.write("comment\n")
            for elem, coord in zip(frame['elements'], frame['coords']):
                f.write(f"{elem} {coord[0]:.10f} {coord[1]:.10f} {coord[2]:.10f}\n")


def convert_trajectory(input_file, output_file=None, file_format=None, fallback_to_outcar=True):
    """
    Convert XDATCAR, OUTCAR, or extended XYZ to fractional XYZ format.
    Auto-detects format if not specified.
    Returns the frames and the average lattice vectors.
    
    If fallback_to_outcar is True and XDATCAR reading fails or returns no frames,
    attempts to read from OUTCAR in the same directory.
    """
    if file_format is None:
        with open(input_file, 'r') as f:
            first_lines = [f.readline() for _ in range(3)]
        
        if 'Lattice=' in first_lines[1] or 'Lattice=' in ''.join(first_lines):
            file_format = 'extxyz'
        elif 'outcar' in input_file.lower() or 'vasp' in first_lines[0].lower():
            file_format = 'outcar'
        else:
            file_format = 'xdatcar'
    
    print(f"Detected format: {file_format}")
    
    frames = None
    tried_fallback = False
    
    if file_format == 'xdatcar':
        try:
            frames = read_xdatcar(input_file)
        except Exception as e:
            print(f"  Error reading XDATCAR: {e}")
            frames = []
        
        if len(frames) == 0 and fallback_to_outcar:
            outcar_path = Path(input_file).parent / "OUTCAR"
            if outcar_path.exists():
                print(f"  XDATCAR failed or empty, falling back to OUTCAR...")
                tried_fallback = True
                frames = read_outcar(str(outcar_path))
    elif file_format == 'outcar':
        frames = read_outcar(input_file)
    elif file_format == 'extxyz':
        frames = read_extxyz(input_file)
    else:
        raise ValueError(f"Unknown format: {file_format}")
    
    if frames is None or len(frames) == 0:
        raise ValueError(f"No frames could be read from {input_file}")
    
    if not tried_fallback:
        print(f"Read {len(frames)} frames")
    
    if output_file is None:
        base = os.path.splitext(input_file)[0]
        output_file = base + "_fract.xyz"
    
    write_fract_xyz(frames, output_file)
    print(f"Wrote fractional coordinates to: {output_file}")
    
    avg_lattice = np.mean([f['lattice'] for f in frames], axis=0)
    print("Average lattice vectors:")
    for i, vec in enumerate(avg_lattice):
        print(f"  a{i+1} = [{vec[0]:.6f}, {vec[1]:.6f}, {vec[2]:.6f}]")
    
    return frames, avg_lattice


ATOMIC_MASSES = {
    'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.81, 'C': 12.01,
    'N': 14.01, 'O': 16.00, 'F': 19.00, 'Ne': 20.18, 'Na': 22.99, 'Mg': 24.31,
    'Al': 26.98, 'Si': 28.09, 'P': 30.97, 'S': 32.07, 'Cl': 35.45, 'Ar': 39.95,
    'K': 39.10, 'Ca': 40.08, 'Sc': 44.96, 'Ti': 47.87, 'V': 50.94, 'Cr': 52.00,
    'Mn': 54.94, 'Fe': 55.85, 'Co': 58.93, 'Ni': 58.69, 'Cu': 63.55, 'Zn': 65.38,
    'Ga': 69.72, 'Ge': 72.63, 'As': 74.92, 'Se': 78.97, 'Br': 79.90, 'Kr': 83.80,
    'Rb': 85.47, 'Sr': 87.62, 'Y': 88.91, 'Zr': 91.22, 'Nb': 92.91, 'Mo': 95.95,
    'Tc': 98.00, 'Ru': 101.1, 'Rh': 102.9, 'Pd': 106.4, 'Ag': 107.9, 'Cd': 112.4,
    'In': 114.8, 'Sn': 118.7, 'Sb': 121.8, 'Te': 127.6, 'I': 126.9, 'Xe': 131.3,
    'Cs': 132.9, 'Ba': 137.3, 'La': 138.9, 'Ce': 140.1, 'Pr': 140.9, 'Nd': 144.2,
    'Pm': 145.0, 'Sm': 150.4, 'Eu': 152.0, 'Gd': 157.3, 'Tb': 158.9, 'Dy': 162.5,
    'Ho': 164.9, 'Er': 167.3, 'Tm': 168.9, 'Yb': 173.0, 'Lu': 175.0, 'Hf': 178.5,
    'Ta': 180.9, 'W': 183.8, 'Re': 186.2, 'Os': 190.2, 'Ir': 192.2, 'Pt': 195.1,
    'Au': 197.0, 'Hg': 200.6, 'Tl': 204.4, 'Pb': 207.2, 'Bi': 209.0, 'Po': 209.0,
    'At': 210.0, 'Rn': 222.0, 'Fr': 223.0, 'Ra': 226.0, 'Ac': 227.0, 'Th': 232.0,
    'Pa': 231.0, 'U': 238.0, 'Np': 237.0, 'Pu': 244.0
}


def MSD(xyz_file, L, timestep=1.0, remove_com_drift=True):
    """
    Calculate MSD from fractional coordinate XYZ file.
    
    Args:
        xyz_file: Path to fractional coordinate XYZ file
        L: Lattice vectors (3x3 array)
        timestep: Time between frames in fs (default 1.0)
        remove_com_drift: If True, subtract center-of-mass drift (default True)
    
    Returns:
        Dictionary with 'steps', 'time', 'elements', and MSD data per element
    """
    a = []
    a.append(L[0]); a.append(L[1]); a.append(L[2])

    file = open(xyz_file, 'r')

    origin_list = []
    prev_list = []
    unwrapped_list = []

    msd = []
    msd_dict = {}
    msd_lattice = []
    msd_dict_lattice = {}

    element_list = []
    element_dict = {}
    masses = []

    content = file.readline()
    N = int(content)

    for i in range(N):
        msd.append(np.float64('0.0'))
        msd_lattice.append([0.0, 0.0, 0.0])

    file.readline()
    step = 0

    com_origin = None
    com_unwrapped = None
    com_prev = None

    results = {
        'steps': [],
        'time': [],
        'elements': None,
        'msd': {},
        'msd_x': {},
        'msd_y': {},
        'msd_z': {},
        'com_drift': []
    }

    while True:
        step += 1
        if step == 1:
            for i in range(N):
                t = file.readline().rstrip('\n').split()
                element = t[0]
                if element not in element_list:
                    element_list.append(element)
                if element not in element_dict:
                    element_dict[element] = 1.0
                else:
                    element_dict[element] += 1.0
                coords = np.array([float(s) for s in t[1:]])
                origin_list.append([element, coords])
                mass = ATOMIC_MASSES.get(element, 1.0)
                masses.append(mass)
            
            masses = np.array(masses)
            total_mass = np.sum(masses)
            
            unwrapped_list = deepcopy(origin_list)
            prev_list = deepcopy(origin_list)
            
            if remove_com_drift:
                origin_coords_cart = np.array([
                    origin_list[i][1][0]*a[0] + origin_list[i][1][1]*a[1] + origin_list[i][1][2]*a[2]
                    for i in range(N)
                ])
                com_origin = np.sum(masses[:, np.newaxis] * origin_coords_cart, axis=0) / total_mass
                com_unwrapped = com_origin.copy()
                com_prev_frac = np.array([origin_list[i][1] for i in range(N)])
                com_prev = np.sum(masses[:, np.newaxis] * com_prev_frac, axis=0) / total_mass
            
            results['elements'] = element_list
            for el in element_list:
                results['msd'][el] = []
                results['msd_x'][el] = []
                results['msd_y'][el] = []
                results['msd_z'][el] = []

        content = file.readline()
        if len(content) == 0:
            print("\n---End of file---\n")
            break
        N = int(content)
        file.readline()
        wrapped_list = []
        for i in range(N):
            t = file.readline().rstrip('\n').split()
            element = t[0]
            coords = np.array([float(s) for s in t[1:]])
            wrapped_list.append([element, coords])

        if remove_com_drift:
            wrapped_frac = np.array([wrapped_list[i][1] for i in range(N)])
            prev_frac = np.array([prev_list[i][1] for i in range(N)])
            
            com_wrapped_frac = np.sum(masses[:, np.newaxis] * wrapped_frac, axis=0) / total_mass
            com_prev_frac = np.sum(masses[:, np.newaxis] * prev_frac, axis=0) / total_mass
            
            com_diff = com_wrapped_frac - com_prev_frac
            for dim in range(3):
                if np.fabs(com_diff[dim]) > 0.5:
                    com_diff[dim] -= np.sign(com_diff[dim])
            
            com_unwrapped_frac_new = com_prev_frac + com_diff
            com_unwrapped += (com_diff[0]*a[0] + com_diff[1]*a[1] + com_diff[2]*a[2])
            
            com_drift = np.linalg.norm(com_unwrapped - com_origin)
            results['com_drift'].append(com_drift)

        for atom in range(N):
            msd[atom] = 0.0

            w1 = wrapped_list[atom][1][0]
            w2 = wrapped_list[atom][1][1]
            w3 = wrapped_list[atom][1][2]

            p1 = prev_list[atom][1][0]
            p2 = prev_list[atom][1][1]
            p3 = prev_list[atom][1][2]

            if np.fabs(w1 - p1) > 0.5:
                u1 = w1 - p1 - np.sign(w1 - p1)
            else:
                u1 = w1 - p1

            if np.fabs(w2 - p2) > 0.5:
                u2 = w2 - p2 - np.sign(w2 - p2)
            else:
                u2 = w2 - p2

            if np.fabs(w3 - p3) > 0.5:
                u3 = w3 - p3 - np.sign(w3 - p3)
            else:
                u3 = w3 - p3

            unwrapped_list[atom][1][0] += u1
            unwrapped_list[atom][1][1] += u2
            unwrapped_list[atom][1][2] += u3

            uw = unwrapped_list[atom][1][0]*a[0] + unwrapped_list[atom][1][1]*a[1] + unwrapped_list[atom][1][2]*a[2]
            ol = origin_list[atom][1][0]*a[0] + origin_list[atom][1][1]*a[1] + origin_list[atom][1][2]*a[2]

            if remove_com_drift:
                uw_corrected = uw - (com_unwrapped - com_origin)
                disp = uw_corrected - ol
            else:
                disp = uw - ol

            msd[atom] = np.linalg.norm(disp)**2
            msd_lattice[atom] = [disp[0]**2, disp[1]**2, disp[2]**2]

        prev_list = deepcopy(wrapped_list)

        for el in element_list:
            msd_dict[el] = 0.0
            msd_dict_lattice[el] = [0., 0., 0.]

        for atom in range(len(msd)):
            msd_dict[wrapped_list[atom][0]] += msd[atom] / element_dict[wrapped_list[atom][0]]
            for i in range(3):
                msd_dict_lattice[wrapped_list[atom][0]][i] += msd_lattice[atom][i] / element_dict[wrapped_list[atom][0]]

        results['steps'].append(step)
        results['time'].append(step * timestep)
        for el in element_list:
            results['msd'][el].append(msd_dict[el])
            results['msd_x'][el].append(msd_dict_lattice[el][0])
            results['msd_y'][el].append(msd_dict_lattice[el][1])
            results['msd_z'][el].append(msd_dict_lattice[el][2])

        if step % 1000 == 0:
            print(f"Processing step {step}...")

    file.close()
    
    for key in ['steps', 'time']:
        results[key] = np.array(results[key])
    results['com_drift'] = np.array(results['com_drift'])
    for el in element_list:
        results['msd'][el] = np.array(results['msd'][el])
        results['msd_x'][el] = np.array(results['msd_x'][el])
        results['msd_y'][el] = np.array(results['msd_y'][el])
        results['msd_z'][el] = np.array(results['msd_z'][el])
    
    if remove_com_drift:
        print(f"COM drift correction applied. Final COM drift: {results['com_drift'][-1]:.4f} Å")
    
    return results


def MSD_from_extxyz(extxyz_file, timestep=1.0, remove_com_drift=True):
    """
    Calculate MSD directly from extended XYZ trajectory file.
    
    This function properly handles NPT simulations by using per-frame lattice vectors
    and computing displacements in Cartesian space. This avoids artifacts from cell
    shape changes that occur when using fractional coordinates with a fixed reference lattice.
    
    Args:
        extxyz_file: Path to extended XYZ trajectory file (with Lattice= in comment)
        timestep: Time between frames in fs (default 1.0)
        remove_com_drift: If True, subtract center-of-mass drift (default True)
    
    Returns:
        Dictionary with 'steps', 'time', 'elements', and MSD data per element
    """
    frames = read_extxyz(extxyz_file)
    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError(f"No frames read from {extxyz_file}")
    
    N = len(frames[0]['elements'])
    element_list = []
    element_dict = {}
    masses = []
    
    for elem in frames[0]['elements']:
        if elem not in element_list:
            element_list.append(elem)
        if elem not in element_dict:
            element_dict[elem] = 1.0
        else:
            element_dict[elem] += 1.0
        masses.append(ATOMIC_MASSES.get(elem, 1.0))
    
    masses = np.array(masses)
    total_mass = np.sum(masses)
    
    results = {
        'steps': [],
        'time': [],
        'elements': element_list,
        'msd': {el: [] for el in element_list},
        'msd_x': {el: [] for el in element_list},
        'msd_y': {el: [] for el in element_list},
        'msd_z': {el: [] for el in element_list},
        'com_drift': []
    }
    
    lattice0 = frames[0]['lattice']
    frac0 = frames[0]['coords']
    cart0 = frac0 @ lattice0
    
    if remove_com_drift:
        com_origin = np.sum(masses[:, np.newaxis] * cart0, axis=0) / total_mass
    
    unwrapped_frac = frac0.copy()
    prev_frac = frac0.copy()
    
    print(f"Processing {n_frames} frames...")
    
    for step in range(1, n_frames):
        lattice = frames[step]['lattice']
        wrapped_frac = frames[step]['coords']
        
        delta_frac = wrapped_frac - prev_frac
        delta_frac = delta_frac - np.round(delta_frac)
        
        unwrapped_frac = unwrapped_frac + delta_frac
        
        unwrapped_cart = unwrapped_frac @ lattice
        origin_cart = frac0 @ lattice
        
        if remove_com_drift:
            com_current = np.sum(masses[:, np.newaxis] * unwrapped_cart, axis=0) / total_mass
            com_origin_current = np.sum(masses[:, np.newaxis] * origin_cart, axis=0) / total_mass
            com_drift_vec = com_current - com_origin_current
            com_drift = np.linalg.norm(com_drift_vec)
            results['com_drift'].append(com_drift)
            
            disp = (unwrapped_cart - com_drift_vec) - origin_cart
        else:
            disp = unwrapped_cart - origin_cart
        
        msd_per_atom = np.sum(disp**2, axis=1)
        msd_components = disp**2
        
        msd_dict = {el: 0.0 for el in element_list}
        msd_dict_xyz = {el: np.zeros(3) for el in element_list}
        
        for atom in range(N):
            elem = frames[step]['elements'][atom]
            msd_dict[elem] += msd_per_atom[atom] / element_dict[elem]
            msd_dict_xyz[elem] += msd_components[atom] / element_dict[elem]
        
        results['steps'].append(step)
        results['time'].append(step * timestep)
        for el in element_list:
            results['msd'][el].append(msd_dict[el])
            results['msd_x'][el].append(msd_dict_xyz[el][0])
            results['msd_y'][el].append(msd_dict_xyz[el][1])
            results['msd_z'][el].append(msd_dict_xyz[el][2])
        
        prev_frac = wrapped_frac.copy()
        
        if step % 1000 == 0:
            print(f"Processing step {step}...")
    
    print("\n---End of file---\n")
    
    for key in ['steps', 'time']:
        results[key] = np.array(results[key])
    results['com_drift'] = np.array(results['com_drift']) if results['com_drift'] else np.array([])
    for el in element_list:
        results['msd'][el] = np.array(results['msd'][el])
        results['msd_x'][el] = np.array(results['msd_x'][el])
        results['msd_y'][el] = np.array(results['msd_y'][el])
        results['msd_z'][el] = np.array(results['msd_z'][el])
    
    if remove_com_drift and len(results['com_drift']) > 0:
        print(f"COM drift correction applied. Final COM drift: {results['com_drift'][-1]:.4f} Å")
    
    return results


def plot_msd(results, title='Mean Square Displacement', save_path=None, show_components=False):
    """
    Plot MSD vs time for each element.
    
    Args:
        results: Dictionary returned by MSD function
        title: Plot title
        save_path: If provided, save figure to this path
        show_components: If True, also plot x, y, z components
    """
    elements = results['elements']
    time_ps = results['time'] / 1000.0  # Convert fs to ps
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(elements)))
    
    if show_components:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax_total = axes[0, 0]
        ax_x = axes[0, 1]
        ax_y = axes[1, 0]
        ax_z = axes[1, 1]
        
        for el, color in zip(elements, colors):
            ax_total.plot(time_ps, results['msd'][el], label=el, color=color, linewidth=1.5)
            ax_x.plot(time_ps, results['msd_x'][el], label=el, color=color, linewidth=1.5)
            ax_y.plot(time_ps, results['msd_y'][el], label=el, color=color, linewidth=1.5)
            ax_z.plot(time_ps, results['msd_z'][el], label=el, color=color, linewidth=1.5)
        
        ax_total.set_xlabel('Time (ps)')
        ax_total.set_ylabel('MSD (Å²)')
        ax_total.set_title('Total MSD')
        ax_total.legend()
        ax_total.grid(True, alpha=0.3)
        
        ax_x.set_xlabel('Time (ps)')
        ax_x.set_ylabel('MSD (Å²)')
        ax_x.set_title('MSD - X component')
        ax_x.legend()
        ax_x.grid(True, alpha=0.3)
        
        ax_y.set_xlabel('Time (ps)')
        ax_y.set_ylabel('MSD (Å²)')
        ax_y.set_title('MSD - Y component')
        ax_y.legend()
        ax_y.grid(True, alpha=0.3)
        
        ax_z.set_xlabel('Time (ps)')
        ax_z.set_ylabel('MSD (Å²)')
        ax_z.set_title('MSD - Z component')
        ax_z.legend()
        ax_z.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for el, color in zip(elements, colors):
            ax.plot(time_ps, results['msd'][el], label=el, color=color, linewidth=1.5)
        
        ax.set_xlabel('Time (ps)', fontsize=12)
        ax.set_ylabel('MSD (Å²)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    return fig


def save_msd_data(results, output_file="msd.out"):
    """Save MSD results to text file."""
    with open(output_file, 'w') as f:
        header = "step time(fs) "
        for el in results['elements']:
            header += f"{el}_msd {el}_msd_x {el}_msd_y {el}_msd_z "
        if len(results['com_drift']) > 0:
            header += "com_drift"
        f.write(header.strip() + "\n")
        
        for i, (step, time) in enumerate(zip(results['steps'], results['time'])):
            line = f"{step} {time:.1f} "
            for el in results['elements']:
                line += f"{results['msd'][el][i]:.6f} "
                line += f"{results['msd_x'][el][i]:.6f} "
                line += f"{results['msd_y'][el][i]:.6f} "
                line += f"{results['msd_z'][el][i]:.6f} "
            if len(results['com_drift']) > 0:
                line += f"{results['com_drift'][i]:.6f}"
            f.write(line.strip() + "\n")
    print(f"MSD data saved to: {output_file}")


def infer_temperature_from_path(sim_dir):
    """Infer (sort_key, display_label) from a simulation directory path.

    Uses the leaf directory name first (e.g. ``300K`` → ``(300, "300K")``).
    If no match, returns ``(None, None)`` so callers can group by path.
    """
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
    """Configure matplotlib backend based on display availability."""
    display = os.environ.get("DISPLAY")
    if save and not display:
        matplotlib.use("Agg")
    elif display:
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")


def _plot_msd_axes(ax_msd, results_list, labels, msd_tmax, colors, ls_cycle, show_components=False):
    """Plot MSD on a single axis."""
    from matplotlib.ticker import AutoMinorLocator
    
    multi_dir = len(results_list) > 1
    elements_used = []
    
    for i, (res_data, lab) in enumerate(zip(results_list, labels)):
        results = res_data['results']
        time_ps = results['time'] / 1000.0
        
        for ispec, el in enumerate(sorted(results['elements'])):
            if el not in elements_used:
                elements_used.append(el)
            tl, ms = time_ps, results['msd'][el]
            if msd_tmax is not None:
                cut = tl <= msd_tmax
                tl, ms = tl[cut], ms[cut]
            lbl = f"{lab} ({el})" if multi_dir else el
            ax_msd.plot(
                tl, ms,
                color=colors[i % 10], ls=ls_cycle[ispec % len(ls_cycle)],
                label=lbl, lw=1.3,
            )
    
    ax_msd.set_xlabel("Time (ps)")
    ax_msd.set_ylabel("MSD (Å²)")
    handles, leg_labs = ax_msd.get_legend_handles_labels()
    n_leg = len(handles)
    if n_leg <= 3:
        ncol = 1
    elif n_leg <= 8:
        ncol = 2
    else:
        ncol = 3
    ax_msd.legend(
        handles, leg_labs,
        fontsize=6.5, ncol=ncol, framealpha=0.92,
        loc="upper left", fancybox=False,
        columnspacing=0.9, handletextpad=0.5, handlelength=1.8,
    )
    ax_msd.xaxis.set_minor_locator(AutoMinorLocator())
    ax_msd.grid(True, alpha=0.3)


def plot_msd_comparison(all_results, labels, sim_dirs=None, title="MSD Comparison", 
                        save_path=None, show_components=False, msd_tmax=None):
    """
    Plot MSD comparison from multiple directories, grouped by temperature.
    Uses subfigures similar to RDF_MSD_evaluation.py.
    
    Args:
        all_results: List of dicts with 'results', 'lattice', 'traj_path', 'dir'
        labels: List of legend labels
        sim_dirs: List of simulation directories (for grouping by temperature)
        title: Plot title
        save_path: If provided, save figure to this path
        show_components: If True, also plot x, y, z components
        msd_tmax: Max lag time on MSD plot (ps)
    """
    import textwrap
    _configure_matplotlib_backend(save=save_path)
    
    colors = plt.cm.tab10.colors
    ls_cycle = ['-', '--', '-.', ':']
    
    if sim_dirs is None:
        sim_dirs = [str(res['dir']) for res in all_results]
    
    dir_lines = [f"{lab}: {str(sim_dirs[i])}" for i, lab in enumerate(labels)]
    dir_text = "\n".join(dir_lines)
    full_title = f"{title}\n{dir_text}"
    
    temp_groups = group_indices_by_temperature(sim_dirs)
    n_blocks = max(len(temp_groups), 1)
    
    if show_components:
        fig_h = max(8.0, 4.5 * n_blocks + 0.8)
        fig = plt.figure(figsize=(14, fig_h), layout="constrained")
        fig.suptitle(full_title, fontsize=10)
        
        subfigs = fig.subfigures(n_blocks, 1, hspace=0.08)
        subfigs = np.atleast_1d(subfigs).ravel()
        
        for (group_title, idxs), sf in zip(temp_groups, subfigs):
            axes = sf.subplots(2, 2, gridspec_kw={"wspace": 0.25, "hspace": 0.35})
            ax_total, ax_x = axes[0, 0], axes[0, 1]
            ax_y, ax_z = axes[1, 0], axes[1, 1]
            
            grp_results = [all_results[i] for i in idxs]
            grp_labels = [labels[i] for i in idxs]
            
            for ax, msd_key, subtitle in [
                (ax_total, 'msd', 'Total MSD'),
                (ax_x, 'msd_x', 'MSD - X component'),
                (ax_y, 'msd_y', 'MSD - Y component'),
                (ax_z, 'msd_z', 'MSD - Z component'),
            ]:
                for i, (res_data, lab) in enumerate(zip(grp_results, grp_labels)):
                    results = res_data['results']
                    time_ps = results['time'] / 1000.0
                    
                    for ispec, el in enumerate(sorted(results['elements'])):
                        tl, ms = time_ps, results[msd_key][el]
                        if msd_tmax is not None:
                            cut = tl <= msd_tmax
                            tl, ms = tl[cut], ms[cut]
                        lbl = f"{lab} ({el})" if len(grp_results) > 1 else el
                        ax.plot(tl, ms, color=colors[i % 10], 
                               ls=ls_cycle[ispec % len(ls_cycle)],
                               label=lbl, lw=1.3)
                
                ax.set_xlabel("Time (ps)")
                ax.set_ylabel("MSD (Å²)")
                ax.set_title(subtitle)
                ax.legend(fontsize=6.5, loc="upper left", framealpha=0.92)
                ax.grid(True, alpha=0.3)
    else:
        fig_h = max(5.8, 2.8 * n_blocks + 0.8)
        fig = plt.figure(figsize=(13.5, fig_h), layout="constrained")
        fig.suptitle(full_title, fontsize=10)
        
        subfigs = fig.subfigures(n_blocks, 1, hspace=0.08)
        subfigs = np.atleast_1d(subfigs).ravel()
        
        for (group_title, idxs), sf in zip(temp_groups, subfigs):
            ax_msd = sf.subplots(1, 1)
            grp_results = [all_results[i] for i in idxs]
            grp_labels = [labels[i] for i in idxs]
            _plot_msd_axes(ax_msd, grp_results, grp_labels, msd_tmax, colors, ls_cycle)
    
    display = os.environ.get("DISPLAY")
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
        print(f"Figure saved to: {Path(save_path).resolve()}")
    
    if not save_path or display:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def read_lat_vec():
    lat_file = open('lattice.vectors', 'r')
    line = []
    for i in range(3):
        line.append([float(x) for x in lat_file.readline().rstrip('\n').split()])
        print(line[i])
    lattice = np.array([line[0], line[1], line[2]])
    return lattice


def _has_simulation(d):
    """True if *d* looks like it contains a simulation (trajectory or input)."""
    d = Path(d)
    if (d / "OUTCAR").exists() or (d / "INCAR").exists():
        return True
    if (d / "XDATCAR").exists() or (d / "POSCAR").exists():
        return True
    if any(d.glob("in.*")) or any(d.glob("*.extxyz")):
        return True
    return False


def _temp_sort_key(p):
    """Sort key that extracts leading number from dir name (e.g. '300K' → 300)."""
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
    """Expand parent directories into their simulation sub-directories.

    Returns resolved list of Paths and origin indices.
    """
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


def find_trajectory(directory):
    """Locate the best trajectory file in *directory*.

    Priority: XDATCAR (VASP MD), then OUTCAR, then extxyz/xyz files.
    For extxyz, prefers 'trajectory.extxyz' over 'trajectory_unwrapped.extxyz'
    since the MSD_from_extxyz function handles unwrapping internally.
    Returns (path, format) or (None, None).
    """
    d = Path(directory)
    if (d / "XDATCAR").exists():
        return str(d / "XDATCAR"), "xdatcar"
    if (d / "OUTCAR").exists():
        return str(d / "OUTCAR"), "outcar"
    
    if (d / "trajectory.extxyz").exists():
        return str(d / "trajectory.extxyz"), "extxyz"
    
    for pattern in [
        "trajectory*.extxyz", "*.extxyz",
        "all_frames*.xyz", "*.xyz",
    ]:
        hits = sorted(d.glob(pattern))
        hits = [h for h in hits if 'unwrapped' not in h.name.lower()]
        if hits:
            return str(hits[-1]), "extxyz"
        hits_all = sorted(d.glob(pattern))
        if hits_all:
            return str(hits_all[0]), "extxyz"
    return None, None


def read_timestep(directory):
    """Auto-detect timestep in fs from INCAR (VASP) or in.* (LAMMPS)."""
    d = Path(directory)

    for candidate in [d / "INCAR", d / "OUTCAR"]:
        if candidate.exists():
            with open(candidate) as fh:
                for line in fh:
                    m = re.match(r"\s*POTIM\s*=\s*([0-9.eE+-]+)", line)
                    if m:
                        return float(m.group(1)), "VASP"
                    if candidate.name == "OUTCAR" and "POSITION" in line:
                        break

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
                        return dt_raw * 1000.0, "LAMMPS"
                    elif units == "real":
                        return dt_raw, "LAMMPS"
                    else:
                        return dt_raw, "LAMMPS"

    return 1.0, "default"


def analyze_directory(directory, timestep=None, remove_com_drift=True):
    """Analyze a single simulation directory for MSD.

    Auto-detects trajectory file and timestep if not provided.
    For extxyz files (typically from NPT LAMMPS), uses MSD_from_extxyz which
    properly handles per-frame lattice vectors to avoid artifacts from cell shape changes.
    Returns (results, lattice, input_file).
    """
    d = Path(directory).resolve()
    print(f"\n{'=' * 60}")
    print(f"  {d}")
    print(f"{'=' * 60}")

    if timestep is not None:
        dt, src = timestep, "CLI"
    else:
        dt, src = read_timestep(d)
    print(f"  dt = {dt} fs  ({src})")

    traj_path, traj_fmt = find_trajectory(d)
    if traj_path is None:
        raise FileNotFoundError(
            f"No trajectory (XDATCAR / *.xyz / *.extxyz) in {d}\n"
            "  Check that the directory contains simulation output.")
    print(f"  Trajectory: {Path(traj_path).name}  ({traj_fmt})")

    if traj_fmt == 'extxyz':
        print(f"\n  Running MSD calculation directly from extxyz (timestep = {dt} fs, COM correction = {remove_com_drift})...")
        print(f"  (Using per-frame lattice vectors for NPT compatibility)")
        results = MSD_from_extxyz(traj_path, timestep=dt, remove_com_drift=remove_com_drift)
        frames = read_extxyz(traj_path)
        lattice = np.mean([f['lattice'] for f in frames], axis=0)
    else:
        frames, lattice = convert_trajectory(traj_path, file_format=traj_fmt)

        base = os.path.splitext(traj_path)[0]
        xyz_file = base + "_fract.xyz"
        write_fract_xyz(frames, xyz_file)
        print(f"  Wrote fractional coordinates to: {xyz_file}")

        print(f"\n  Running MSD calculation (timestep = {dt} fs, COM correction = {remove_com_drift})...")
        results = MSD(xyz_file, lattice, timestep=dt, remove_com_drift=remove_com_drift)
    
    print(f"  MSD calculation complete. Processed {len(results['steps'])} frames.")

    return results, lattice, traj_path


def main():
    parser = argparse.ArgumentParser(
        description='Calculate MSD from XDATCAR or LAMMPS extended XYZ trajectory files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a simulation directory (auto-detect trajectory and timestep)
  python msd.py 300K/
  
  # Analyze multiple directories
  python msd.py dft/300K mlff/300K --labels DFT MLFF
  
  # Analyze parent directory (expands to sub-directories with simulations)
  python msd.py ./
  
  # Directly specify a trajectory file (legacy mode)
  python msd.py XDATCAR
  python msd.py trajectory.extxyz
  
  # Plot with directional components (x, y, z)
  python msd.py 300K/ --components
  
  # Save plot to file instead of displaying
  python msd.py 300K/ --save-plot msd.png
  
  # Save MSD data to text file
  python msd.py 300K/ --save-data msd.out
  
  # Override timestep (default: auto-detect from INCAR/LAMMPS input)
  python msd.py 300K/ --timestep 2.0
  
  # Just convert without running MSD
  python msd.py trajectory.extxyz --convert-only
  
  # Use existing fractional XYZ with manual lattice
  python msd.py coords.xyz --format xyz --lattice 16.08,0,0,0,16.08,0,0,0,16.08
        """
    )
    parser.add_argument('inputs', nargs='+',
                        help='Input directories or trajectory files (XDATCAR or extended XYZ)')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Legend labels for multiple directories (default: auto from dir names)')
    parser.add_argument('--format', '-f', choices=['xdatcar', 'extxyz', 'xyz'],
                        help='Input file format (auto-detected if not specified)')
    parser.add_argument('--output', '-o', help='Output fractional XYZ file name')
    parser.add_argument('--convert-only', action='store_true',
                        help='Only convert to fractional XYZ, do not run MSD')
    parser.add_argument('--lattice', '-l', 
                        help='Manual lattice vectors as comma-separated values: a1x,a1y,a1z,a2x,a2y,a2z,a3x,a3y,a3z')
    parser.add_argument('--timestep', '-t', type=float, default=None,
                        help='Timestep between frames in fs (default: auto-detect)')
    parser.add_argument('--components', '-c', action='store_true',
                        help='Show x, y, z components of MSD in addition to total')
    parser.add_argument('--save-plot', '-p', metavar='FILE',
                        help='Save plot to file instead of displaying (e.g., msd.png)')
    parser.add_argument('--save-data', '-d', metavar='FILE',
                        help='Save MSD data to text file (e.g., msd.out)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Do not display plot (use with --save-data)')
    parser.add_argument('--title', default=None,
                        help='Custom title for the plot')
    parser.add_argument('--msd-tmax', type=float, default=None,
                        help='Max lag time τ on MSD plot (ps; default: full range)')
    parser.add_argument('--no-com-correction', action='store_true',
                        help='Disable center-of-mass drift correction')
    
    args = parser.parse_args()
    
    first_input = Path(args.inputs[0])
    is_directory_mode = first_input.is_dir() or (
        len(args.inputs) > 1 and any(Path(p).is_dir() for p in args.inputs)
    )
    
    if is_directory_mode:
        sim_dirs, origin_indices = resolve_dirs(args.inputs)
        print(f"Resolved {len(sim_dirs)} simulation(s):")
        for sd in sim_dirs:
            print(f"  {sd}")
        
        all_results = []
        for d in sim_dirs:
            remove_com = not args.no_com_correction
            results, lattice, traj_path = analyze_directory(
                str(d), timestep=args.timestep, remove_com_drift=remove_com
            )
            all_results.append({
                'results': results,
                'lattice': lattice,
                'traj_path': traj_path,
                'dir': d
            })
        
        if args.labels and len(args.labels) >= len(sim_dirs):
            labels = args.labels[:len(sim_dirs)]
        else:
            labels = [Path(d).name for d in sim_dirs]
            if len(labels) != len(set(labels)):
                labels = [f"{Path(d).parent.name}/{Path(d).name}" for d in sim_dirs]
        
        if args.save_data:
            for i, (res_data, label) in enumerate(zip(all_results, labels)):
                suffix = f"_{label.replace('/', '_')}" if len(all_results) > 1 else ""
                out_file = args.save_data.replace('.out', f'{suffix}.out') if '.out' in args.save_data else f"{args.save_data}{suffix}"
                save_msd_data(res_data['results'], out_file)
        
        if not args.no_plot:
            title = args.title if args.title else "MSD Comparison"
            if len(all_results) == 1:
                title = args.title if args.title else f"MSD - {labels[0]}"
                plot_msd(all_results[0]['results'], title=title, 
                        save_path=args.save_plot, show_components=args.components)
            else:
                plot_msd_comparison(all_results, labels, sim_dirs=[str(d) for d in sim_dirs],
                                   title=title, save_path=args.save_plot, 
                                   show_components=args.components, msd_tmax=args.msd_tmax)
    
    else:
        input_file = args.inputs[0]
        
        if args.format == 'xyz':
            if args.lattice is None:
                parser.error("For pre-converted XYZ files, you must specify --lattice")
            lattice_vals = [float(x) for x in args.lattice.split(',')]
            if len(lattice_vals) != 9:
                parser.error("Lattice must have 9 values: a1x,a1y,a1z,a2x,a2y,a2z,a3x,a3y,a3z")
            lattice = np.array(lattice_vals).reshape(3, 3)
            print("Using manual lattice:")
            for i, vec in enumerate(lattice):
                print(f"  a{i+1} = [{vec[0]:.6f}, {vec[1]:.6f}, {vec[2]:.6f}]")
            xyz_file = input_file
        else:
            frames, lattice = convert_trajectory(input_file, args.output, args.format)
            xyz_file = args.output if args.output else os.path.splitext(input_file)[0] + "_fract.xyz"
        
        if not args.convert_only:
            remove_com = not args.no_com_correction
            timestep = args.timestep if args.timestep else 1.0
            print(f"\nRunning MSD calculation (timestep = {timestep} fs, COM correction = {remove_com})...")
            results = MSD(xyz_file, lattice, timestep=timestep, remove_com_drift=remove_com)
            print(f"MSD calculation complete. Processed {len(results['steps'])} frames.")
            
            if args.save_data:
                save_msd_data(results, args.save_data)
            
            if not args.no_plot:
                title = args.title if args.title else f"MSD - {os.path.basename(input_file)}"
                plot_msd(results, title=title, save_path=args.save_plot, show_components=args.components)


if __name__ == '__main__':
    main()
