#!/bin/bash
#SBATCH --job-name=MLFF_Eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=eval_output.log
#SBATCH --partition=gpuquick
#SBATCH --gres=gpu:1

# Unified evaluation script for MACE and Allegro (NequIP) models.
# Run from the evaluation directory, or pass it as the first argument:
#   sbatch eval_configs.sh
#   sbatch eval_configs.sh .
#   sbatch eval_configs.sh path/to/high_forces
# Autodetects backend by model file: .model → MACE, .ckpt → Allegro.
# Expects one .xyz (configs) in the work directory; model may be local or in a parent models/ dir.

set -euo pipefail

# ===== Initialize Conda (env set later based on backend) =====
export PATH="$HOME/miniconda3/bin:$PATH"
set +u
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
set -u

usage() {
  cat <<'USAGE'
Usage:
  sbatch eval_configs.sh [WORK_DIR]

WORK_DIR defaults to the directory where you run sbatch (SLURM_SUBMIT_DIR).
  - Put exactly one .xyz file (structures to evaluate) in WORK_DIR.
  - Put one model file in WORK_DIR, or Allegro last.ckpt/last.ckpt in a parent models/ dir.
  - Backend is auto-detected from the model extension (.model → MACE, .ckpt → Allegro).
  - Optional: -m/--model FILE, -c/--configs FILE, -h/--help.
USAGE
}

MODEL_FILE=""
CONFIGS_FILE=""
WORK_DIR=""

# ===== Optional args (overrides; not required) =====
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)
      [[ $# -ge 2 ]] || { echo "Error: --model requires a value"; exit 2; }
      MODEL_FILE="$2"; shift 2;;
    -c|--configs|--config)
      [[ $# -ge 2 ]] || { echo "Error: --configs requires a value"; exit 2; }
      CONFIGS_FILE="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    -*)
      echo "Error: Unknown option: $1"; usage; exit 2;;
    *)
      if [[ -n "$WORK_DIR" ]]; then
        echo "Error: Unexpected extra argument: $1"; usage; exit 2
      fi
      WORK_DIR="$1"; shift;;
  esac
done

# ===== Move to work directory =====
if [[ -z "$WORK_DIR" ]]; then
  WORK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
fi
if [[ ! -d "$WORK_DIR" ]]; then
  echo "❌ Error: Work directory not found: $WORK_DIR"
  exit 1
fi
cd "$WORK_DIR"
echo "✅ Work directory: $(pwd)"

find_allegro_model_in_parents() {
  local dir="$1"
  local depth=0
  while [[ $depth -lt 12 && "$dir" != "/" ]]; do
    if [[ -f "$dir/models/last.ckpt" ]]; then
      echo "$dir/models/last.ckpt"
      return 0
    fi
    dir="$(dirname "$dir")"
    depth=$((depth + 1))
  done
  dir="$1"
  depth=0
  while [[ $depth -lt 12 && "$dir" != "/" ]]; do
    if [[ -f "$dir/models/last.ckpt" ]]; then
      echo "$dir/models/last.ckpt"
      return 0
    fi
    dir="$(dirname "$dir")"
    depth=$((depth + 1))
  done
  return 1
}

# ===== Autodetect CONFIGS_FILE: single .xyz in submit directory =====
# Ignore output.xyz — it is written by this script; a leftover file must not count as a second input.
if [[ -z "${CONFIGS_FILE}" ]]; then
  shopt -s nullglob
  ALL_XYZ=( ./*.xyz )
  shopt -u nullglob
  XYZ_FILES=()
  for f in "${ALL_XYZ[@]}"; do
    [[ "$(basename "$f")" == "output.xyz" ]] && continue
    XYZ_FILES+=( "$f" )
  done
  if [[ ${#XYZ_FILES[@]} -eq 0 ]]; then
    echo "❌ Error: No .xyz file in current directory. Put exactly one .xyz (configs) here."
    echo "   (output.xyz is ignored when picking the input file; use --configs FILE if your input is only output.xyz.)"
    exit 1
  elif [[ ${#XYZ_FILES[@]} -gt 1 ]]; then
    echo "❌ Error: Multiple .xyz files found. Keep only one or use --configs FILE:"
    printf '  %s\n' "${XYZ_FILES[@]}"
    exit 1
  else
    CONFIGS_FILE="${XYZ_FILES[0]}"
  fi
fi

# ===== Autodetect MODEL_FILE and BACKEND from submit directory =====
# Rule: .model → MACE, .ckpt → Allegro. One model file in cwd; if none, Allegro can use ../models/last.ckpt or last.ckpt.
if [[ -z "${MODEL_FILE}" ]]; then
  shopt -s nullglob
  MODEL_FILES=( ./*.model )
  CKPT_FILES=( ./*.ckpt )
  shopt -u nullglob
  N_MODEL=${#MODEL_FILES[@]}
  N_CKPT=${#CKPT_FILES[@]}

  if [[ $N_MODEL -eq 1 && $N_CKPT -eq 0 ]]; then
    MODEL_FILE="${MODEL_FILES[0]}"
    BACKEND="mace"
  elif [[ $N_MODEL -eq 0 && $N_CKPT -eq 1 ]]; then
    MODEL_FILE="${CKPT_FILES[0]}"
    BACKEND="allegro"
  elif [[ $N_MODEL -eq 1 && $N_CKPT -eq 1 ]]; then
    echo "❌ Error: Both a .model and a .ckpt found in current directory. Keep only one (MACE .model or Allegro .ckpt)."
    exit 1
  elif [[ $N_MODEL -gt 1 ]]; then
    echo "❌ Error: Multiple .model files found. Keep only one or use --model FILE:"
    printf '  %s\n' "${MODEL_FILES[@]}"
    exit 1
  elif [[ $N_CKPT -gt 1 ]]; then
    echo "❌ Error: Multiple .ckpt files found. Keep only one or use --model FILE:"
    printf '  %s\n' "${CKPT_FILES[@]}"
    exit 1
  else
    # No model in cwd: walk up to find Allegro-style models/last.ckpt or models/last.ckpt
    if MODEL_FILE="$(find_allegro_model_in_parents "$(pwd)")"; then
      BACKEND="allegro"
    else
      echo "❌ Error: No model in current directory. Put one .model (MACE) or one .ckpt (Allegro) here, or put last.ckpt/last.ckpt in a parent models/ directory for Allegro."
      exit 1
    fi
  fi
else
  # User passed --model: infer backend from extension
  if [[ "$MODEL_FILE" == *.model ]]; then
    BACKEND="mace"
  elif [[ "$MODEL_FILE" == *.ckpt ]]; then
    BACKEND="allegro"
  else
    echo "❌ Error: Model file must end with .model (MACE) or .ckpt (Allegro): $MODEL_FILE"
    exit 1
  fi
fi

# ===== Validate files =====
if [[ ! -f "$CONFIGS_FILE" ]]; then
  echo "❌ Error: Configs file not found: $CONFIGS_FILE"
  exit 1
fi
if [[ ! -f "$MODEL_FILE" ]]; then
  echo "❌ Error: Model file not found: $MODEL_FILE"
  exit 1
fi

echo "✅ Backend: $BACKEND"
echo "✅ Using configs file: $CONFIGS_FILE"
echo "✅ Using model file: $MODEL_FILE"

# ===== Activate backend-specific conda env =====
set +u
if [[ "$BACKEND" == "mace" ]]; then
  conda activate mace
else
  conda activate nequip
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
set -u

# ===== Debug information =====
echo "🔍 Debug info:"
echo "  - SLURM_JOB_ID: ${SLURM_JOB_ID:-'Not set'}"
echo "  - SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-'Not set'}"
echo "  - CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-'Not set'}"
echo "  - Python: $(which python) ($(python --version 2>&1))"
[[ "$BACKEND" == "allegro" ]] && python -c "import torch; print('  - CUDA available:', torch.cuda.is_available()); print('  - Device count:', torch.cuda.device_count())"

# ===== Stress: only if configs contain stress data =====
if grep -q 'stress=' "$CONFIGS_FILE"; then
  echo "✅ Detected stress data in $CONFIGS_FILE — enabling stress computation"
  COMPUTE_STRESS=1
else
  echo "ℹ No stress data in $CONFIGS_FILE — skipping stress computation"
  COMPUTE_STRESS=0
fi

# ===== Run evaluation by backend =====
# Remove stale output.xyz so evaluation can overwrite (unless the configs file *is* output.xyz).
if [[ "$(basename "$CONFIGS_FILE")" != "output.xyz" ]] && [[ -f output.xyz ]]; then
  echo "ℹ Removing existing output.xyz so the run can write a fresh file."
  rm -f output.xyz
fi

if [[ "$BACKEND" == "mace" ]]; then
  COMPUTE_STRESS_ARGS=()
  [[ "$COMPUTE_STRESS" -eq 1 ]] && COMPUTE_STRESS_ARGS=(--compute_stress)
  python -m mace.cli.eval_configs \
    --configs="$CONFIGS_FILE" \
    --model="$MODEL_FILE" \
    --output="output.xyz" \
    --default_dtype="float64" \
    --device="cuda" \
    --batch_size=8 \
    "${COMPUTE_STRESS_ARGS[@]}"
  ADD_REF_STRESS="${HOME}/scripts/add_ref_stress_to_output.py"
  if [[ -f "$ADD_REF_STRESS" ]]; then
    python "$ADD_REF_STRESS" --configs="$CONFIGS_FILE" --output="output.xyz"
  fi
  echo "✅ MACE evaluation completed"
else
  # Allegro: run embedded Python evaluator
  export FINAL_FRAMES="$CONFIGS_FILE"
  export MODEL="$MODEL_FILE"
  export COMPUTE_STRESS
  python << 'PYTHON_SCRIPT'
import os
import numpy as np
from pathlib import Path

import ase.io
from nequip.ase import NequIPCalculator

configs_path = Path(os.environ["FINAL_FRAMES"])
model_path = Path(os.environ["MODEL"])
output_path = Path("output.xyz")
device = "cuda"
compute_stress = bool(int(os.environ.get("COMPUTE_STRESS", "0")))

if not configs_path.exists():
    raise FileNotFoundError(f"Configs file not found: {configs_path}")
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")

print(f"Using configs file: {configs_path}")
print(f"Using model file: {model_path}")
print(f"Compute stress: {compute_stress}")

def _normalize_stress(stress_like):
    if stress_like is None:
        return None
    s = np.asarray(stress_like, dtype=float)
    if s.shape == (3, 3):
        return s
    if s.shape == (6,):
        from ase.stress import voigt_6_to_full_3x3_stress
        return voigt_6_to_full_3x3_stress(s)
    return None

calc = NequIPCalculator._from_saved_model(
    str(model_path),
    device=device,
    chemical_species_to_atom_type_map=True,
)

frames = ase.io.read(str(configs_path), index=":")
n_frames = len(frames)
print(f"Loaded {n_frames} frames")

ref_energies = []
ref_forces_list = []
pred_energies = []
pred_forces_list = []
ref_stresses_list = []
pred_stresses_list = []

for i, at in enumerate(frames):
    if at.calc is not None and hasattr(at.calc, "results"):
        if "energy" in at.calc.results:
            ref_e = float(at.calc.results["energy"])
            ref_energies.append(ref_e)
            at.info["REF_energy"] = ref_e
        if "forces" in at.calc.results:
            ref_f = np.asarray(at.calc.results["forces"]).copy()
            ref_forces_list.append(ref_f)
            at.arrays["REF_forces"] = ref_f
        if compute_stress:
            ref_s = _normalize_stress(at.calc.results.get("stress"))
            if ref_s is None:
                ref_s = _normalize_stress(at.info.get("stress"))
            if ref_s is not None:
                ref_stresses_list.append(ref_s.copy())
                at.info["REF_stress"] = ref_s

    at.calc = calc
    pred_e = at.get_potential_energy()
    pred_f = at.get_forces()
    pred_energies.append(pred_e)
    pred_forces_list.append(pred_f.copy())
    at.info["ALLEGRO_energy"] = pred_e
    at.arrays["ALLEGRO_forces"] = pred_f

    if compute_stress:
        try:
            pred_s = _normalize_stress(at.get_stress())
        except Exception:
            pred_s = None
        if pred_s is not None:
            pred_stresses_list.append(pred_s.copy())
            at.info["ALLEGRO_stress"] = pred_s

    if (i + 1) % 100 == 0 or i == 0:
        print(f"  Frame {i + 1}/{n_frames}")

print("Writing output file...")
with open(str(output_path), "w") as f:
    for i, at in enumerate(frames):
        ase.io.write(f, [at], format="extxyz", write_results=False)
        if (i + 1) % 100 == 0:
            print(f"  Wrote frame {i + 1}/{n_frames}")
print(f"Wrote {output_path}")

if ref_energies and ref_forces_list:
    ref_e = np.array(ref_energies)
    pred_e = np.array(pred_energies)
    mae_e = np.mean(np.abs(pred_e - ref_e))
    rmse_e = np.sqrt(np.mean((pred_e - ref_e) ** 2))
    n_atoms = np.mean([len(f) for f in frames])
    mae_e_per_atom = mae_e / n_atoms
    rmse_e_per_atom = rmse_e / n_atoms
    ref_f = np.concatenate(ref_forces_list, axis=0)
    pred_f = np.concatenate(pred_forces_list, axis=0)
    mae_f = np.mean(np.abs(pred_f - ref_f))
    rmse_f = np.sqrt(np.mean((pred_f - ref_f) ** 2))
    print("ALLEGRO evaluation completed")
    print(f"  Energy  MAE = {mae_e:.6f} eV  (per atom: {mae_e_per_atom:.6f})")
    print(f"  Energy  RMSE = {rmse_e:.6f} eV  (per atom: {rmse_e_per_atom:.6f})")
    print(f"  Forces  MAE = {mae_f:.6f} eV/Å")
    print(f"  Forces  RMSE = {rmse_f:.6f} eV/Å")
    if compute_stress and ref_stresses_list and pred_stresses_list:
        n_stress = min(len(ref_stresses_list), len(pred_stresses_list))
        ref_s = np.array(ref_stresses_list[:n_stress]).reshape(n_stress, 9)
        pred_s = np.array(pred_stresses_list[:n_stress]).reshape(n_stress, 9)
        mae_s = np.mean(np.abs(pred_s - ref_s))
        rmse_s = np.sqrt(np.mean((pred_s - ref_s) ** 2))
        print(f"  Stress  MAE = {mae_s:.6f} eV/Å^3")
        print(f"  Stress  RMSE = {rmse_s:.6f} eV/Å^3")
    elif compute_stress:
        print("  Stress metrics unavailable (missing REF_stress or ALLEGRO_stress in some/all frames)")
else:
    print("ALLEGRO evaluation completed (no REF_energy/REF_forces in configs for metrics)")
PYTHON_SCRIPT
  echo "✅ Allegro evaluation completed"
fi

echo "Done. Check output.xyz and eval_output.log"
