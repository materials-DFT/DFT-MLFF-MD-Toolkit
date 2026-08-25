from __future__ import annotations

import os
import sys
from importlib import import_module
from getpass import getpass
from pathlib import Path

def _register_module_alias(alias: str, target: str) -> None:
    if alias in sys.modules:
        return
    sys.modules[alias] = import_module(target)


def _ensure_pymatgen_compat() -> None:
    aliases = {
        "pymatgen.core.graphs": "pymatgen.analysis.graphs",
        "pymatgen.core.local_env": "pymatgen.analysis.local_env",
        "pymatgen.core.bond_valence": "pymatgen.analysis.bond_valence",
        "pymatgen.core.structure_matcher": "pymatgen.analysis.structure_matcher",
        "pymatgen.core.molecule_matcher": "pymatgen.analysis.molecule_matcher",
        "pymatgen.core.structure_analyzer": "pymatgen.analysis.structure_analyzer",
        "pymatgen.core.entries": "pymatgen.entries.computed_entries",
    }

    for alias, target in aliases.items():
        try:
            _register_module_alias(alias, target)
        except Exception:
            # Leave the alias unset if the target module is unavailable.
            pass


_ensure_pymatgen_compat()

from mp_api.client import MPRester
from pymatgen.io.vasp import Poscar


def main() -> None:
    api_key = getpass("Materials Project API key: ").strip()
    if not api_key:
        raise SystemExit("No API key provided.")

    raw_ids = input(
        "Enter Materials Project IDs to download, separated by commas: "
    ).strip()
    if not raw_ids:
        raise SystemExit("No material IDs provided.")

    mpids = [mid.strip() for mid in raw_ids.split(",") if mid.strip()]
    if not mpids:
        raise SystemExit("No valid material IDs provided.")

    out_dir = Path.cwd()

    with MPRester(api_key=api_key) as mpr:
        for material_id in mpids:
            structure = mpr.get_structure_by_material_id(material_id)
            if structure is None:
                print(f"{material_id}: no structure returned")
                continue

            output_path = out_dir / f"{material_id}.POSCAR"
            Poscar(structure).write_file(output_path)
            print(f"{material_id}: wrote {output_path}")


if __name__ == "__main__":
    main()
