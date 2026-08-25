# DFT-MLFF-MD-Toolkit

Utilities for VASP/MLFF workflows on HPC systems: structure preparation, INCAR/KPOINTS automation, SLURM job management, trajectory conversion/extraction, MLFF evaluation, and plotting.

## What Is In This Repo

- `VASP_scripts`: VASP input generation and recursive INCAR/KPOINTS tuning.
- `data_management`: dataset extraction/conversion/filtering and structure curation.
- `Data_plotting`: parity plots, MD diagnostics, EOS and dataset distribution visualization.
- `MLFF`: MACE/Allegro/UMA training-evaluation helpers and model conversion utilities.
- `cluster_management`: bulk submit/check/cancel helpers for SLURM (plus PBS variant).
- `software`: cluster-specific build/run guides and scripts (Polaris, Palmetto).
- `supplemental`: example Allegro training artifacts/config snapshots.

## Quick Start

```bash
# 1) create env (example)
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib ase pandas

# 2) common workflow examples
python VASP_scripts/kpoints_calculator.py /path/to/structures --k 25
python VASP_scripts/prepare_directories_for_npt.py /path/to/structures --temps 300,500,700
python data_management/extract_optimized_frames.py /path/to/runs -o optimized_frames.xyz
python Data_plotting/parity_plot_per_atom.py output.xyz
```

## Repository Map

### `VASP_scripts`

- `prepare_directories_for_npt.py`: combined workflow for temperature folder creation, NPT parameter updates, and parallelization tuning.
- `bulkmodulus_setup.py`: creates strained `V_...%` directories and prepares single-point EOS runs.
- `extract_poscars_from_xyz.py`: splits an extxyz file into per-structure DFT folders (`POSCAR/INCAR/KPOINTS`).
- `kpoints_calculator.py`: recommends/writes KPOINTS grids from POSCAR lattice lengths.
- `set_magmom_nelect.py`: recursively updates `MAGMOM` and `NELECT` using `POSCAR`+`POTCAR`.
- `smass_set_recursive.py`: computes and writes `SMASS` from `TEBEG`, POSCAR geometry, and DOF.
- `optimize_vasp_incar.py`: analyzes system size + k-point density to tune `NCORE/NPAR/KPAR/NSIM`.
- `setup_dft_directories.py`: organizes `POSCAR.*` files into run directories and regenerates INCAR/KPOINTS.
- `update_incar_files.py`: bulk rewrite of selected INCAR keys for electronic-structure style runs.
- `prepare_for_electronic_structure.py`: switches INCAR defaults for post-relaxation electronic calculations.

### `data_management`

- Trajectory extraction:
  - `extract_all_frames.py`: all OUTCAR frames -> one extxyz.
  - `extract_optimized_frames.py`: lowest-energy frame per OUTCAR.
  - `extract_frames_from_md.py`: midpoint frame extraction for MD runs.
  - `extract_diverse_frames_soap.py`: SOAP descriptor + global farthest-point sampling to pick maximally diverse frames across one or more directories of MD OUTCARs.
  - `extract_bulk_modulus_lowest_10_frames.py`: lowest-energy subset from EOS directories.
- Conversion/prep:
  - `convert_trajectory_to_extxyz.py`, `poscar_to_lammps.py`, `gaussian_to_poscar.py`, `unwrap_xdatcar.py`.
- Dataset statistics and curation:
  - `calculate_element_concentration_in_xyz.py`, `convergence_separation.py`, `extract_average_md_volume.py`.
  - `cull_combined_global.py` and `cull_amorphous_structures.py` (QUEST-based compression/stratified culling).
  - `filter_structures.py` (distance-based filtering), `combine_pngs.py`.
- Volume/EOS helpers:
  - `predict_optimal_volume.py`, `write_optimal_volume_poscars.py`, `match_interfaces_to_dataframe.py`.
- Shell helpers:
  - `cleanup_lammps_outputs.sh`, `copy_a_file_recursively.sh`, `plot_md_temperature.sh`.

### `Data_plotting`

- MLFF diagnostics:
  - `parity_plot_per_atom.py`, `parity_plot_per_species.py`, `plot_xyz_energies_forces.py`.
- MD analysis:
  - `plot_md_pvt.py` (OUTCAR quick T/V/P plotting, GUI or ASCII fallback).
  - `VACF_RDF_MSD_evaluation.py`, `RDF_MSD_evaluation.py`, `DFT_RDF_MSD_plotting.py`, `VPS_PSD_evaluation.py`, `PDF_K_plotting.py`.
- EOS/volume plots:
  - `plot_bulk_modulus.py`, `plot_bulk_modulus_individual.py`, `plot_bulk_modulus_per_atom.py`, `plot_volume_lammps_vs_vasp.py`.
- Dataset composition:
  - `plot_frame_distribution.py`.

### `MLFF`

- Evaluation:
  - `eval_configs.sh`: autodetects MACE (`.model`) vs Allegro (`.ckpt`) and writes `output.xyz`.
  - `eval_configs_uma.sh`: UMA checkpoint evaluation with optional stress metrics.
  - `identify_mlff_outliers.py`: ranks worst structures by relative energy/force/stress errors.
- MACE training submit scripts:
  - `MACE_scripts/mace_1gpu.sh`, `mace_2gpus.sh`, `mace_4gpus.sh`.
- Allegro conversion and launch scripts:
  - `ALLEGRO_scripts/convert_ckpt_to_pth.py`, `convert_ckpt_to_pt2.py`.
  - `ALLEGRO_scripts/allegro_training_submit_scripts/*.sh`.
  - `ALLEGRO_scripts/lammps_allegro_submit_scripts/*.sh`.
  - `ALLEGRO_scripts/setup_nequip_env_for_pt2_lammps.sh`.

### `cluster_management`

- `submit_all_jobs.sh`: recursive `sbatch` of `submit.vasp6.sh`.
- `check_jobs.sh`: prints state/reason and attempts run progress from `OSZICAR`/`INCAR`.
- `cancel_jobs_in_dirs.py`: cancel SLURM jobs under chosen root paths.
- `cancel_jobs_in_dirs_pbs.py`: PBS variant.
- `clean_dev_shm_themis.sh`: cleanup utility for shared-memory usage.

### `software`

- `Polaris_software`:
  - GPU VASP compile/run guides and scripts (`VASP_GPU_COMPILATION_GUIDE.md`, `compile_vasp_gpu_final.sh`, `submit.vasp6.sh`).
  - LAMMPS/Kokkos/MACE install guide and script.
- `Palmetto_software`:
  - `build_lammps_mace_palmetto.sh`.

## Dependencies

Core Python packages used across scripts:

- `numpy`
- `scipy`
- `matplotlib`
- `ase`
- `pandas`

Optional or script-specific packages:

- `pymatgen` (some POSCAR/composition tasks)
- `torch`, `mace-torch`, `nequip`, `nequip-allegro`, `fairchem-core` (MLFF workflows)
- `quests` (dataset compression scripts)
- `dscribe` (SOAP descriptors for `extract_diverse_frames_soap.py`)

External tooling:

- VASP, LAMMPS, SLURM (and PBS for one script), plus cluster module stack where applicable.

## Typical Commands

```bash
# Bulk modulus setup
python VASP_scripts/bulkmodulus_setup.py /path/to/bulk_modulus_calculations --min -20 --max 20 --npts 40

# Unified MLFF evaluation (MACE or Allegro) from a submit directory
sbatch MLFF/eval_configs.sh

# UMA evaluation
sbatch MLFF/eval_configs_uma.sh --task omat

# Plot OUTCAR thermodynamics quickly
python Data_plotting/plot_md_pvt.py OUTCAR --ascii

# Recursively submit VASP jobs
bash cluster_management/submit_all_jobs.sh /path/to/tree
```

## Notes

- Most scripts are standalone and are intended to run from CLI without package installation.
- Many defaults are cluster-specific (partitions, conda env names, module names); adjust before production use.
- Some scripts intentionally recurse through large directory trees; test with a small subset first.
