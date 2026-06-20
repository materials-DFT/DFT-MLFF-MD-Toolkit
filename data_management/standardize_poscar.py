#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import tempfile
import re
from ase.build.tools import sort
from ase.io import read, write


def split_concatenated_poscars(file_path):
    """
    Split a file containing multiple concatenated POSCAR structures into individual frames.
    Returns a list of Atoms objects.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    frames = []
    frame_lines = []
    in_coordinates = False
    coord_count = 0
    expected_atoms = 0
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Detect start of a new POSCAR frame
        # A new frame starts with a comment line, followed by scaling factor (typically "1.0")
        if len(frame_lines) == 0 or (in_coordinates and coord_count >= expected_atoms):
            # Check if this could be the start of a new frame
            # Look ahead: next line should be scaling factor (a single number)
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                try:
                    float(next_line)
                    # This looks like a new frame starting
                    if frame_lines and coord_count >= expected_atoms:
                        # Save previous frame
                        frame_text = ''.join(frame_lines)
                        atoms = parse_poscar_text(frame_text)
                        if atoms is not None:
                            frames.append(atoms)
                        frame_lines = []
                        in_coordinates = False
                        coord_count = 0
                        expected_atoms = 0
                except ValueError:
                    pass
        
        frame_lines.append(lines[i])
        
        # Track where we are in the POSCAR structure
        frame_line_count = len(frame_lines)
        
        if frame_line_count == 7:
            # Line 7 contains atom counts - parse to know when coordinates end
            counts_line = lines[i].strip()
            try:
                expected_atoms = sum(int(x) for x in counts_line.split())
            except ValueError:
                expected_atoms = 0
        
        if frame_line_count == 8:
            coord_type = line.lower()
            if coord_type.startswith('s'):  # Selective dynamics
                pass  # Next line will be coord type
            elif coord_type.startswith('c') or coord_type.startswith('d') or coord_type.startswith('k'):
                in_coordinates = True
                coord_count = 0
        elif frame_line_count == 9 and not in_coordinates:
            # Could be coordinate type after selective dynamics
            if line.lower().startswith('c') or line.lower().startswith('d') or line.lower().startswith('k'):
                in_coordinates = True
                coord_count = 0
        elif in_coordinates and frame_line_count > 8:
            coord_count += 1
        
        i += 1
    
    # Don't forget the last frame
    if frame_lines:
        frame_text = ''.join(frame_lines)
        atoms = parse_poscar_text(frame_text)
        if atoms is not None:
            frames.append(atoms)
    
    return frames


def parse_poscar_text(text):
    """Parse POSCAR text and return an Atoms object."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.vasp', delete=False) as tmp:
        tmp.write(text)
        tmp_path = tmp.name
    try:
        atoms = read(tmp_path, format='vasp')
        return atoms
    except Exception as e:
        print(f"  Warning: Failed to parse a frame: {e}")
        return None
    finally:
        os.unlink(tmp_path)


def write_concatenated_poscars(file_path, atoms_list):
    """Write multiple Atoms objects as concatenated POSCAR structures."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.vasp', delete=False) as tmp:
        tmp_path = tmp.name
    
    poscar_texts = []
    for atoms in atoms_list:
        write(tmp_path, atoms, format='vasp')
        with open(tmp_path, 'r') as f:
            poscar_texts.append(f.read())
    
    os.unlink(tmp_path)
    
    with open(file_path, 'w') as f:
        f.write(''.join(poscar_texts))

def main():
    """
    Main function to process POSCAR files from command-line input.
    """
    if len(sys.argv) < 2:
        print("Usage: python reorder_poscar.py <file_or_directory_path>")
        sys.exit(1)

    path = sys.argv[1]
    # Note: new_order is not used anymore since ASE's sort automatically sorts by element
    # But keeping it for potential future use or compatibility
    new_order = ['K', 'Mn', 'O']

    if os.path.isfile(path):
        process_file(path, new_order)
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            if 'POSCAR' in filenames:
                file_path = os.path.join(dirpath, 'POSCAR')
                process_file(file_path, new_order)
    else:
        print(f"Error: The path '{path}' does not exist or is not a valid file/directory.", file=sys.stderr)
        sys.exit(1)

def process_file(file_path, new_order):
    """
    Reads, sorts atoms by element type using ASE, and overwrites the POSCAR file.
    Handles both single-frame and multi-frame (concatenated POSCAR) files.
    """
    print(f"Processing {file_path}...")
    try:
        # First try standard ASE read
        atoms_list = read(file_path, index=':')
        
        # Handle case where read returns a single Atoms object instead of list
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        
        # If ASE only found one frame, check if file might have concatenated POSCARs
        if len(atoms_list) == 1:
            # Check file for multiple POSCAR blocks by looking for multiple "Cartesian" or "Direct" lines
            with open(file_path, 'r') as f:
                content = f.read()
            coord_markers = len(re.findall(r'^(Cartesian|Direct|Kartesian)', content, re.MULTILINE | re.IGNORECASE))
            
            if coord_markers > 1:
                print(f"  Detected concatenated POSCAR file with ~{coord_markers} frames, parsing...")
                atoms_list = split_concatenated_poscars(file_path)
        
        num_frames = len(atoms_list)
        print(f"  Found {num_frames} frame(s)")
        
        # Sort atoms by element type for each frame
        atoms_sorted_list = [sort(atoms) for atoms in atoms_list]
        
        # Write back to file
        if num_frames == 1:
            write(file_path, atoms_sorted_list[0], format='vasp')
        else:
            # Write all frames as concatenated POSCARs
            write_concatenated_poscars(file_path, atoms_sorted_list)
        
        print(f"Successfully reordered and overwrote {file_path} ({num_frames} frame(s))")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

