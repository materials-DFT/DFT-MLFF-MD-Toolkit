#!/usr/bin/env python3
"""Update INCAR files for post-relaxation electronic-structure calculations."""

import argparse
import os
import re
from pathlib import Path


def modify_incar(incar_path):
    with open(incar_path, "r") as f:
        lines = f.readlines()

    updated_lines = []
    flags_found = {
        "ISTART": False,
        "ICHARG": False,
        "IBRION": False,
        "LREAL": False,
        "NEDOS": False,
        "EMIN": False,
        "EMAX": False,
    }

    for line in lines:
        original = line.strip()
        key = re.sub(r"^#+", "", original).split("=")[0].strip() if "=" in original else None

        if key == "ISTART":
            updated_lines.append("ISTART = 1\n")
            flags_found["ISTART"] = True
        elif key == "ICHARG":
            updated_lines.append("ICHARG = 1\n")
            flags_found["ICHARG"] = True
        elif key == "IBRION":
            updated_lines.append("IBRION = -1\n")
            flags_found["IBRION"] = True
        elif key == "LREAL":
            updated_lines.append("LREAL = .FALSE.\n")
            flags_found["LREAL"] = True
        elif key == "NEDOS":
            updated_lines.append("NEDOS = 1501\n")
            flags_found["NEDOS"] = True
        elif key == "EMIN":
            updated_lines.append("EMIN = -20.0\n")
            flags_found["EMIN"] = True
        elif key == "EMAX":
            updated_lines.append("EMAX = 10.0\n")
            flags_found["EMAX"] = True
        elif key in {"ISIF", "POTIM", "NSW"}:
            if original.startswith("#"):
                updated_lines.append(line)
            else:
                updated_lines.append(f"# {original}\n")
        else:
            updated_lines.append(line)

    if not flags_found["ISTART"]:
        updated_lines.append("ISTART = 1\n")
    if not flags_found["ICHARG"]:
        updated_lines.append("ICHARG = 1\n")
    if not flags_found["IBRION"]:
        updated_lines.append("IBRION = -1\n")
    if not flags_found["LREAL"]:
        updated_lines.append("LREAL = .FALSE.\n")
    if not flags_found["NEDOS"]:
        updated_lines.append("NEDOS = 1501\n")
    if not flags_found["EMIN"]:
        updated_lines.append("EMIN = -20.0\n")
    if not flags_found["EMAX"]:
        updated_lines.append("EMAX = 10.0\n")

    with open(incar_path, "w") as f:
        f.writelines(updated_lines)

    print(f"Updated: {incar_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare INCAR files for electronic-structure calculations."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to search recursively for INCAR files (default: current directory)",
    )
    args = parser.parse_args()

    target = Path(args.directory).resolve()
    if not target.is_dir():
        raise SystemExit(f"Error: '{args.directory}' is not a directory")

    count = 0
    for root, _, files in os.walk(target):
        for file in files:
            if file == "INCAR":
                modify_incar(os.path.join(root, file))
                count += 1

    if count == 0:
        print(f"No INCAR files found under {target}")
    else:
        print(f"\nDone! Updated {count} INCAR file(s).")


if __name__ == "__main__":
    main()
