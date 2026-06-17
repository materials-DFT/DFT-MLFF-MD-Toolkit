#!/usr/bin/env python3
import os
import shutil
import argparse
import sys

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
    "POTIM": "0.5",
    "ISIF": "3",
    "EDIFF": "1E-5",
    "PREC": "Normal",
    "ALGO": "Fast",
    "NELM": "1000",
    "SIGMA": "0.05"
}

KPOINTS_CONTENT = """Automatic mesh
0
Monkhorst-Pack
1 1 1
0 0 0
"""

VASP_HINT_FILES = {"INCAR", "POSCAR", "CONTCAR", "KPOINTS", "POTCAR"}

def parse_args():
    p = argparse.ArgumentParser(
        description="Populate per-temperature subfolders for each structure folder (auto-detect structure vs parent)."
    )
    p.add_argument("paths", nargs="*", help="One or more paths. Each path may be a structure folder or a parent folder.")
    p.add_argument("--temps", default=",".join(map(str, DEFAULT_TEMPS)),
                   help="Comma-separated temperatures, e.g. 300,500,700 (default: 300,500,700,900,1100,1300).")
    p.add_argument("--delete", default=",".join(DEFAULT_DELETE),
                   help="Comma-separated filenames to delete inside each temp folder.")
    p.add_argument("--keep-top-level-files", action="store_true",
                   help="If set, do NOT remove top-level files in each structure folder after populating.")
    return p.parse_args()

def parse_temps(s):
    vals = []
    for t in s.split(","):
        t = t.strip()
        if t:
            vals.append(int(t))
    return vals

def is_structure_dir(path):
    """Heuristic: a directory is a VASP structure folder if it contains any of the common VASP input/output files."""
    if not os.path.isdir(path):
        return False
    try:
        names = set(os.listdir(path))
    except PermissionError:
        return False
    return len(VASP_HINT_FILES & names) > 0

def write_or_update_incar(incar_path, params, temp):
    # If missing, create a minimal INCAR
    if not os.path.exists(incar_path):
        with open(incar_path, "w") as f:
            for k, v in params.items():
                f.write(f"{k} = {v}\n")
            f.write(f"TEBEG = {temp}\n")
            f.write(f"TEEND = {temp}\n")
            f.write("\n# Space-saving flags (appended by script)\n")
            f.write("LWAVE  = .FALSE.\n")
            f.write("LCHARG = .FALSE.\n")
        return

    with open(incar_path, 'r') as f:
        lines = f.readlines()

    updated_lines = []
    keys_written = set()
    # Remove any existing LWAVE/LCHARG occurrences; we will append canonical ones at bottom
    for line in lines:
        if '=' in line:
            left, right = line.split('=', 1)
            key_raw = left.strip()
            key = key_raw.lstrip("#").strip().upper()
            if key in {"LWAVE", "LCHARG"}:
                continue
            if key in params:
                updated_lines.append(f"{key} = {params[key]}\n")
                keys_written.add(key)
            elif key in {"TEBEG", "TEEND"}:
                updated_lines.append(f"{key} = {temp}\n")
                keys_written.add(key)
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)

    for key in params:
        if key not in keys_written:
            updated_lines.append(f"{key} = {params[key]}\n")
    if "TEBEG" not in keys_written:
        updated_lines.append(f"TEBEG = {temp}\n")
    if "TEEND" not in keys_written:
        updated_lines.append(f"TEEND = {temp}\n")

    # Append canonical LWAVE/LCHARG at the bottom
    updated_lines.append("\n# Space-saving flags (appended by script)\n")
    updated_lines.append("LWAVE  = .FALSE.\n")
    updated_lines.append("LCHARG = .FALSE.\n")

    with open(incar_path, 'w') as f:
        f.writelines(updated_lines)

def write_kpoints(kpoints_path):
    with open(kpoints_path, "w") as f:
        f.write(KPOINTS_CONTENT)

def process_structure_dir(struct_path, temperatures, files_to_delete, keep_top_level_files):
    """Process a single structure directory: create {temp}K subfolders, copy files, edit INCAR/KPOINTS, clean outputs."""
    struct_path = os.path.abspath(struct_path)
    # collect only top-level FILES (ignore subdirectories)
    try:
        entries = os.listdir(struct_path)
    except PermissionError as e:
        print(f"[WARN] Skipping '{struct_path}': {e}", file=sys.stderr)
        return

    top_files = [f for f in entries if os.path.isfile(os.path.join(struct_path, f))]

    for temp in temperatures:
        temp_dir = os.path.join(struct_path, f"{temp}K")
        os.makedirs(temp_dir, exist_ok=True)

        # copy top-level files into each temp dir
        for file in top_files:
            src = os.path.join(struct_path, file)
            dst = os.path.join(temp_dir, file)
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"[WARN] Couldn't copy {src} -> {dst}: {e}", file=sys.stderr)

        # INCAR update/create
        incar_path = os.path.join(temp_dir, "INCAR")
        write_or_update_incar(incar_path, DEFAULT_PARAMS, temp)

        # Overwrite KPOINTS with requested content
        kpoints_path = os.path.join(temp_dir, "KPOINTS")
        write_kpoints(kpoints_path)

        # CONTCAR -> POSCAR (if available in this temp dir after copy)
        contcar = os.path.join(temp_dir, "CONTCAR")
        poscar  = os.path.join(temp_dir, "POSCAR")
        if os.path.exists(contcar):
            try:
                shutil.copy2(contcar, poscar)
            except Exception as e:
                print(f"[WARN] Couldn't copy CONTCAR->POSCAR in {temp_dir}: {e}", file=sys.stderr)

        # cleanup inside each temp dir
        for unwanted in files_to_delete:
            fpath = os.path.join(temp_dir, unwanted)
            if os.path.exists(fpath):
                try:
                    if os.path.isfile(fpath) or os.path.islink(fpath):
                        os.remove(fpath)
                    else:
                        shutil.rmtree(fpath)
                except Exception as e:
                    print(f"[WARN] Couldn't remove {fpath}: {e}", file=sys.stderr)

    # remove top-level files so only temp dirs (and any pre-existing subdirs) remain
    if not keep_top_level_files:
        for file in top_files:
            try:
                os.remove(os.path.join(struct_path, file))
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"[WARN] Couldn't remove {file} in {struct_path}: {e}", file=sys.stderr)

def process_path(path, temperatures, files_to_delete, keep_top_level_files):
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        print(f"[WARN] Skipping '{path}': not a directory.", file=sys.stderr)
        return

    if is_structure_dir(path):
        print(f"[INFO] Processing structure: {path}")
        process_structure_dir(path, temperatures, files_to_delete, keep_top_level_files)
        return

    # Otherwise, treat immediate subdirectories as structures (original behavior)
    subdirs = [os.path.join(path, d) for d in os.listdir(path)
               if os.path.isdir(os.path.join(path, d))]

    if not subdirs:
        print(f"[INFO] No structure subfolders found under '{path}'.")
        return

    print(f"[INFO] Processing parent: {path} (structures: {len(subdirs)})")
    for struct_path in subdirs:
        process_structure_dir(struct_path, temperatures, files_to_delete, keep_top_level_files)

def main():
    args = parse_args()
    temperatures = parse_temps(args.temps)
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

    for path in unique_roots:
        process_path(path, temperatures, files_to_delete, args.keep_top_level_files)

if __name__ == "__main__":
    main()
