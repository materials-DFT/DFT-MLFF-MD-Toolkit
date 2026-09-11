#!/usr/bin/env python3
import os, shutil, numpy as np, sys, argparse
from pathlib import Path

# Default sweep settings
# percent strain about current volume (e.g., -6% ... +6%)
DEFAULT_VOL_PCT_MIN, DEFAULT_VOL_PCT_MAX, DEFAULT_NPTS = -100.0, 100.0, 41  # includes 0%
COPY_FILES = ["INCAR", "KPOINTS", "POTCAR", "submit.vasp6.sh"]     # symlink if possible, else copy
TEMPLATE_FILES_TO_REMOVE = ["INCAR", "KPOINTS", "POSCAR", "POTCAR", "submit.vasp6.sh"]  # Files to remove from template dir after setup

def read_poscar(path):
    with open(path, "r") as f:
        lines = [l.rstrip() for l in f]
    scale = float(lines[1].split()[0])
    A = np.array([[float(x) for x in lines[i].split()] for i in range(2,5)], float) * scale

    # Find counts line (first all-int line)
    counts_idx = None
    for i in range(5, 12):
        parts = lines[i].split()
        if parts and all(p.isdigit() for p in parts):
            counts_idx = i
            break
    if counts_idx is None:
        raise RuntimeError("Could not find atom counts line in POSCAR")

    nat = sum(int(x) for x in lines[counts_idx].split())
    coord_start = counts_idx + 1
    sel_dyn = lines[coord_start].strip().lower().startswith("s")
    if sel_dyn: coord_start += 1
    direct = lines[coord_start].strip().lower().startswith("d")
    coord_start += 1
    
    # Read coordinates
    coords = []
    for i in range(nat):
        coords.append([float(x) for x in lines[coord_start+i].split()[:3]])
    coords = np.array(coords, float)
    
    # Convert Cartesian to Direct (fractional) if needed
    converted = False
    if not direct:
        # Cartesian coordinates: convert to fractional using inverse lattice matrix
        # r_frac = A^-1 * r_cart
        A_inv = np.linalg.inv(A)
        frac = coords @ A_inv.T
        converted = True
    else:
        frac = coords
    
    header = lines[0]
    species_line = None
    if not all(p.isdigit() for p in lines[counts_idx-1].split()):
        species_line = lines[counts_idx-1]
    return header, A, frac, nat, species_line, lines, counts_idx, sel_dyn, converted

def write_poscar(path, header, A, frac, species_line, counts_line, sel_dyn):
    with open(path, "w") as f:
        f.write(f"{header}\n")
        f.write("1.0\n")
        for i in range(3):
            f.write(f"{A[i,0]:18.10f} {A[i,1]:18.10f} {A[i,2]:18.10f}\n")
        if species_line is not None:
            f.write(species_line+"\n")
        f.write(counts_line+"\n")
        if sel_dyn:
            f.write("Selective dynamics\n")
        f.write("Direct\n")
        for r in frac:
            f.write(f"{r[0]:.10f} {r[1]:.10f} {r[2]:.10f}\n")

def modify_incar(incar_path, nelm=500, nsw=1, lwave=False, lcharg=False):
    """Modify INCAR to set NELM, NSW and disable writing WAVECAR/CHGCAR/CHG.

    Args:
        incar_path: Path to INCAR file
        nelm: Value for NELM parameter (default: 500)
        nsw: Value for NSW parameter (default: 1)
        lwave: Sets LWAVE (False disables WAVECAR writing)
        lcharg: Sets LCHARG (False disables CHGCAR/CHG writing)
    """
    if not os.path.isfile(incar_path):
        return

    with open(incar_path, "r") as f:
        lines = f.readlines()

    nelm_found = False
    nsw_found = False
    lwave_found = False
    lcharg_found = False
    modified_lines = []

    # Helper to format logicals as VASP expects
    def vbool(x):
        return ".TRUE." if bool(x) else ".FALSE."

    for line in lines:
        original_line = line
        stripped = line.strip()

        # Preserve empty or full-comment lines
        if not stripped or stripped.startswith("#") or stripped.startswith("!"):
            modified_lines.append(original_line)
            continue

        # Extract parameter name token (exact key before '=' or first word)
        if "=" in stripped:
            param_name = stripped.split("=")[0].strip().upper()
        else:
            words = stripped.split()
            param_name = words[0].upper() if words else None

        def preserve_comment(s):
            # keep trailing inline comments starting with # or ! (not at col 0)
            for c in ("#", "!"):
                idx = s.find(c)
                if idx > 0:
                    return s[idx:].rstrip("\n")
            return ""

        # Update NELM
        if param_name == "NELM" and not nelm_found:
            comment = preserve_comment(line)
            newline = "\n" if line.endswith("\n") else ""
            modified_lines.append(f"NELM = {nelm}{comment}{newline}")
            nelm_found = True
            continue

        # Update NSW
        if param_name == "NSW" and not nsw_found:
            comment = preserve_comment(line)
            newline = "\n" if line.endswith("\n") else ""
            modified_lines.append(f"NSW = {nsw}{comment}{newline}")
            nsw_found = True
            continue

        # Update LWAVE
        if param_name == "LWAVE" and not lwave_found:
            comment = preserve_comment(line)
            newline = "\n" if line.endswith("\n") else ""
            modified_lines.append(f"LWAVE = {vbool(lwave)}{comment}{newline}")
            lwave_found = True
            continue

        # Update LCHARG
        if param_name == "LCHARG" and not lcharg_found:
            comment = preserve_comment(line)
            newline = "\n" if line.endswith("\n") else ""
            modified_lines.append(f"LCHARG = {vbool(lcharg)}{comment}{newline}")
            lcharg_found = True
            continue

        # Keep original if not modified
        modified_lines.append(original_line)

    # Append any missing params at the end (ensures they're at the bottom)
    if not nelm_found:
        modified_lines.append(f"NELM = {nelm}\n")
    if not nsw_found:
        modified_lines.append(f"NSW = {nsw}\n")
    if not lwave_found:
        modified_lines.append(f"LWAVE = {vbool(lwave)}\n")
    if not lcharg_found:
        modified_lines.append(f"LCHARG = {vbool(lcharg)}\n")

    with open(incar_path, "w") as f:
        f.writelines(modified_lines)

def setup_calculation(template_dir, vol_pct_min, vol_pct_max, npts, cleanup=True):
    """Set up bulk modulus calculations in a template directory.
    
    Args:
        template_dir: Directory containing CONTCAR or POSCAR and other input files
        vol_pct_min: Minimum volume strain percentage (e.g., -10.0 for -10%)
        vol_pct_max: Maximum volume strain percentage (e.g., 10.0 for +10%)
        npts: Number of strain points to generate (includes 0% if range spans it)
        cleanup: If True (default), remove template files after creating V_* directories
    """
    contcar_t = os.path.join(template_dir, "CONTCAR")
    poscar_t = os.path.join(template_dir, "POSCAR")
    
    if os.path.isfile(contcar_t) and os.path.getsize(contcar_t) > 0:
        structure_file = contcar_t
        print(f"\n  Using CONTCAR (optimized structure) from: {template_dir}")
    elif os.path.isfile(poscar_t):
        structure_file = poscar_t
        print(f"\n  Using POSCAR (no CONTCAR found) from: {template_dir}")
    else:
        print(f"Warning: No CONTCAR or POSCAR found in {template_dir}, skipping...")
        return False
    
    print(f"Setting up calculations in: {template_dir}")
    
    header, A0, frac, nat, species_line, lines, counts_idx, sel_dyn, converted = read_poscar(structure_file)
    if converted:
        print(f"  Note: Converted Cartesian coordinates to Direct (fractional) coordinates")
    counts_line = lines[counts_idx]
    V0 = abs(np.linalg.det(A0))

    grid = np.linspace(vol_pct_min, vol_pct_max, npts)
    print(f"  Reference volume V0 = {V0:.6f} Å^3")
    print(f"  Volume strain range: {vol_pct_min}% to {vol_pct_max}% ({npts} points)")
    
    for pct in grid:
        scaleV = 1.0 + pct/100.0
        s = scaleV ** (1.0/3.0)   # isotropic length scale so det(s*A0) = scaleV*V0
        A = A0 * s
        tag = f"V_{pct:+05.1f}%".replace('+', '+').replace('-', '–')  # nice minus sign
        tag = tag.replace('–', '-')  # if you prefer ASCII minus
        tag_dir = os.path.join(template_dir, tag)
        os.makedirs(tag_dir, exist_ok=True)

        # copy control files (must copy, not symlink, because template files are removed during cleanup)
        for fn in COPY_FILES:
            src = os.path.join(template_dir, fn)
            dst = os.path.join(tag_dir, fn)
            if os.path.isfile(src):
                if os.path.exists(dst):
                    os.remove(dst)
                # Always copy files (not symlink) so each directory has its own copy
                # This is necessary because template files are removed during cleanup
                shutil.copy2(src, dst)
                
                # Modify INCAR file to set NELM=500 and NSW=1
                if fn == "INCAR":
                    modify_incar(dst, nelm=500, nsw=1)

        # write POSCAR
        pos_path = os.path.join(tag_dir, "POSCAR")
        write_poscar(pos_path, header, A, frac, species_line, counts_line, sel_dyn)

        # write a small README with volume info
        with open(os.path.join(tag_dir, "README.eos"), "w") as g:
            g.write(f"Target volume scale: {scaleV:.6f} (pct {pct:+.1f}%)\n")
            g.write(f"Cell volume: {abs(np.linalg.det(A)):.6f} Å^3 (V0={V0:.6f})\n")

    # Clean up all files (but keep directories) if requested
    if cleanup:
        removed_count = 0
        for item in os.listdir(template_dir):
            item_path = os.path.join(template_dir, item)
            # Only remove files, not directories (keep V_* directories)
            if os.path.isfile(item_path):
                os.remove(item_path)
                removed_count += 1
        if removed_count > 0:
            print(f"  Removed {removed_count} file(s) from {template_dir} (kept directories)")

    print(f"  Made {npts} EOS points from {vol_pct_min}% to {vol_pct_max}%.")
    return True

def find_structure_directories(root_dir):
    """Recursively find all directories containing CONTCAR or POSCAR files."""
    struct_dirs = []
    root_path = Path(root_dir).resolve()
    
    for name in ("CONTCAR", "POSCAR"):
        for struct_file in root_path.rglob(name):
            parent = struct_file.parent
            if "V_" in parent.name:
                continue
            if name == "CONTCAR" and os.path.getsize(struct_file) == 0:
                continue
            struct_dirs.append(str(parent))
    
    return sorted(set(struct_dirs))

def main():
    parser = argparse.ArgumentParser(
        description="Set up bulk modulus calculations recursively in a directory structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 13 structures from -10% to +10% strain
  python bulkmodulus_setup.py /path/to/directory

  # Custom strain range: 11 structures from -5% to +5%
  python bulkmodulus_setup.py /path/to/directory --min -5.0 --max 5.0 --npts 11

  # Single directory mode with custom parameters
  python bulkmodulus_setup.py /path/to/directory --single --min -6.0 --max 6.0 --npts 15
        """
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Root directory to search for POSCAR files recursively"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Treat the specified directory as a single template (don't recurse)"
    )
    parser.add_argument(
        "--min",
        type=float,
        default=DEFAULT_VOL_PCT_MIN,
        help=f"Minimum volume strain percentage (default: {DEFAULT_VOL_PCT_MIN})"
    )
    parser.add_argument(
        "--max",
        type=float,
        default=DEFAULT_VOL_PCT_MAX,
        help=f"Maximum volume strain percentage (default: {DEFAULT_VOL_PCT_MAX})"
    )
    parser.add_argument(
        "--npts",
        type=int,
        default=DEFAULT_NPTS,
        help=f"Number of strain points to generate (default: {DEFAULT_NPTS})"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep template files (INCAR, KPOINTS, POSCAR, POTCAR, submit.vasp6.sh) in parent directories (cleanup is default)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.min >= args.max:
        print(f"Error: --min ({args.min}) must be less than --max ({args.max})")
        sys.exit(1)
    if args.npts < 2:
        print(f"Error: --npts must be at least 2 (got {args.npts})")
        sys.exit(1)
    
    root_dir = os.path.abspath(args.directory)
    if not os.path.isdir(root_dir):
        print(f"Error: Directory '{root_dir}' does not exist.")
        sys.exit(1)
    
    # Print configuration
    cleanup_enabled = not args.no_cleanup
    print(f"Configuration:")
    print(f"  Volume strain range: {args.min}% to {args.max}%")
    print(f"  Number of points: {args.npts}")
    print(f"  Root directory: {root_dir}")
    print(f"  Cleanup template files: {cleanup_enabled} (template files will be removed after setup)")
    print()
    
    if args.single:
        # Single directory mode
        if not setup_calculation(root_dir, args.min, args.max, args.npts, cleanup_enabled):
            sys.exit(1)
    else:
        # Recursive mode
        print(f"Searching for CONTCAR/POSCAR files in: {root_dir}")
        struct_dirs = find_structure_directories(root_dir)
        
        if not struct_dirs:
            print(f"No CONTCAR or POSCAR files found in {root_dir}")
            sys.exit(1)
        
        print(f"Found {len(struct_dirs)} directory(ies) with structure files:")
        for d in struct_dirs:
            print(f"  - {d}")
        
        print("\nSetting up calculations...")
        success_count = 0
        for template_dir in struct_dirs:
            if setup_calculation(template_dir, args.min, args.max, args.npts, cleanup_enabled):
                success_count += 1
        
        print(f"\n{'='*60}")
        print(f"Completed: {success_count}/{len(struct_dirs)} directories processed successfully.")

if __name__ == "__main__":
    main()
