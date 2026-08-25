#!/bin/bash
# Submit N independent one-GPU LAMMPS simulations as one Slurm job.
#
# Usage (submit from a login node; do not prefix with sbatch):
#   ./submit.lammps_multirun_1gpu.sh 1 2
#   ./submit.lammps_multirun_1gpu.sh /path/to/run_a /path/to/run_b /path/to/run_c
#
# gpuquick presently has a single node with four GPUs, so this launcher accepts
# two through four directories. Each directory must contain the normal LAMMPS
# input, data, and model files; no per-directory submit script is required.

set -euo pipefail

readonly PARTITION="gpuquick"
readonly TIME_LIMIT="02:00:00"
readonly MAX_RUNS=4
readonly LAMMPS_PREFIX="/Apps/chem/lammps_allegro"
readonly LMP_BINARY="${LAMMPS_PREFIX}/bin/lmp"
readonly LIBTORCH_DIR="/opt/pytorch/libtorch-gpu"

usage() {
    echo "Usage: $(basename "$0") DIR1 DIR2 [DIR3 [DIR4]]" >&2
    echo "Submit 2-${MAX_RUNS} independent one-GPU LAMMPS runs in one gpuquick job." >&2
}

resolve_dirs() {
    local raw dir known
    RUN_DIRS=()

    for raw in "$@"; do
        if [[ ! -d "$raw" ]]; then
            echo "ERROR: not a directory: $raw" >&2
            return 2
        fi
        dir="$(cd -- "$raw" && pwd -P)"
        for known in "${RUN_DIRS[@]}"; do
            if [[ "$dir" == "$known" ]]; then
                echo "ERROR: directory was supplied more than once: $dir" >&2
                return 2
            fi
        done
        if [[ ! -f "$dir/in.npt_md_allegro" ]]; then
            echo "ERROR: missing in.npt_md_allegro in $dir" >&2
            return 2
        fi
        RUN_DIRS+=("$dir")
    done
}

# This is the complete per-directory one-GPU runner. It deliberately does not
# submit another Slurm job and does not delete or modify restart files itself.
run_one_gpu() {
    local dir="$1" input_file data_file model_file basename
    local restart_file=none restart_flag=0 triclinic_fix use_data_vel
    local gcc_libstdcpp_dir pt2_env_lib
    local -a restart_candidates=() models=()

    cd -- "$dir"
    input_file="./in.npt_md_allegro"
    [[ -f "$input_file" ]] || { echo "ERROR: input file not found: $input_file" >&2; return 1; }

    if [[ -n "${LAMMPS_DATA_FILE:-}" ]]; then
        data_file="${LAMMPS_DATA_FILE}"
    elif [[ -f "data.lammps" ]]; then
        data_file="./data.lammps"
    elif [[ -f "data.data" ]]; then
        data_file="./data.data"
    else
        data_file="$(find . -maxdepth 1 \( -name 'data.*' -o -name '*.data' -o -name '*.lammps' \) | head -1)"
    fi
    [[ -n "$data_file" && -f "$data_file" ]] || { echo "ERROR: no LAMMPS data file in $dir" >&2; return 1; }

    if [[ -n "${LAMMPS_MODEL_FILE:-}" ]]; then
        model_file="${LAMMPS_MODEL_FILE}"
    else
        shopt -s nullglob
        models=( *.nequip.pt2 )
        (( ${#models[@]} == 0 )) && models=( *.nequip.pth )
        (( ${#models[@]} == 0 )) && models=( *.pt )
        shopt -u nullglob
        (( ${#models[@]} > 0 )) || { echo "ERROR: no Allegro model in $dir" >&2; return 1; }
        model_file="${models[0]}"
    fi
    [[ -f "$model_file" ]] || { echo "ERROR: model file not found: $model_file" >&2; return 1; }

    module purge
    module load gcc-toolset/12
    module load mpi/openmpi-x86_64
    module load cuda-toolkit/12.6
    [[ -x "$LMP_BINARY" ]] || { echo "ERROR: LAMMPS executable not found: $LMP_BINARY" >&2; return 1; }

    gcc_libstdcpp_dir="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"
    pt2_env_lib="${PT2_ENV_LIB:-/Users/924322630/miniconda3/envs/nequip/lib}"
    export LD_LIBRARY_PATH="${pt2_env_lib}:${gcc_libstdcpp_dir}:${LIBTORCH_DIR}/lib:${LAMMPS_PREFIX}/lib64"
    export MKL_THREADING_LAYER=INTEL
    export OMP_NUM_THREADS=1 OMP_PROC_BIND=spread OMP_PLACES=threads
    export TORCH_NUM_INTRAOP_THREADS=1 TORCH_NUM_INTEROP_THREADS=1
    export MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE
    export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader

    if [[ "${FORCE_FRESH_RUN:-0}" != "1" ]]; then
        if [[ -f "restart.lmp.last" ]]; then
            restart_file=restart.lmp.last
            restart_flag=1
        else
            [[ -f "restart.lmp.a" ]] && restart_candidates+=("restart.lmp.a")
            [[ -f "restart.lmp.b" ]] && restart_candidates+=("restart.lmp.b")
            if (( ${#restart_candidates[@]} > 0 )); then
                restart_file="$(ls -t "${restart_candidates[@]}" | head -1)"
                restart_flag=1
            fi
        fi
    fi

    triclinic_fix="${TRICLINIC_FIX:-0}"
    use_data_vel="${USE_DATA_VEL:-0}"
    basename="$(basename "$input_file")"
    mpirun -np 1 "$LMP_BINARY" -log none -k on g 1 -sf kk -pk kokkos newton on neigh half \
        -var model "$model_file" -var datafile "$data_file" \
        -var restartfile "$restart_file" -var restartflag "$restart_flag" \
        -var triclinic_fix "$triclinic_fix" -var use_data_vel "$use_data_vel" \
        -var use_extra_dump 1 -in "$basename"
}

target_step() {
    awk '$1 == "variable" && $2 == "target" && $3 == "equal" { print $4; exit }' \
        "$1/in.npt_md_allegro"
}

simulation_complete() {
    local dir="$1" target last_step
    target="$(target_step "$dir" 2>/dev/null || true)"
    if [[ ! "$target" =~ ^[0-9]+$ ]]; then
        echo "ERROR: cannot read a numeric target step from $dir/in.npt_md_allegro" >&2
        return 2
    fi
    last_step=0
    if [[ -f "$dir/log.lammps" ]]; then
        last_step="$(awk '($1 ~ /^[0-9]+$/) { step=$1 } END { print step+0 }' "$dir/log.lammps")"
    fi
    [[ "$last_step" =~ ^[0-9]+$ ]] || last_step=0
    echo "[$dir] last step: $last_step / target: $target"
    (( last_step >= target ))
}

submit() {
    local launcher run_count
    launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"
    run_count="$#"

    echo "Submitting $run_count independent one-GPU simulations as one ${PARTITION} job."
    exec sbatch \
        --partition="$PARTITION" \
        --nodes=1 \
        --ntasks="$run_count" \
        --cpus-per-task=1 \
        --gres="gpu:${run_count}" \
        --time="$TIME_LIMIT" \
        --requeue \
        --job-name="lmp-${run_count}x1gpu" \
        --output=/dev/null \
        --error=/dev/null \
        "$launcher" --inside "$@"
}

run_inside_allocation() {
    local dir pid status complete_status launcher
    local -a pids=()
    status=0
    launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"

    echo "SLURM_JOB_ID=${SLURM_JOB_ID:-notset}"
    echo "Launching ${#RUN_DIRS[@]} independent one-GPU steps."

    for dir in "${RUN_DIRS[@]}"; do
        echo "Launching: $dir"
        # --exclusive prevents steps sharing a CPU; --gpus-per-task and --gpu-bind
        # request and expose one distinct GPU to each LAMMPS process.
        srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=1 \
            --gpus-per-task=1 --gpu-bind=verbose,single:1 \
            --chdir="$dir" --output=/dev/null --error=/dev/null \
            bash "$launcher" --run-one "$dir" &
        pids+=("$!")
    done

    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "ERROR: one simulation step failed (PID $pid)." >&2
            status=1
        fi
    done
    if (( status != 0 )); then
        echo "At least one simulation failed; not requeueing." >&2
        return "$status"
    fi

    complete_status=0
    for dir in "${RUN_DIRS[@]}"; do
        if simulation_complete "$dir"; then
            echo "[$dir] complete."
        else
            case $? in
                1) complete_status=1 ;;
                *) return 1 ;;
            esac
        fi
    done

    if (( complete_status == 0 )); then
        echo "All simulations complete."
        return 0
    fi

    echo "At least one simulation is incomplete; requeueing job ${SLURM_JOB_ID}."
    scontrol requeue "$SLURM_JOB_ID" || {
        echo "ERROR: requeue failed; resubmit with the same directory arguments." >&2
        return 1
    }
}

if [[ "${1:-}" == "--run-one" ]]; then
    [[ $# == 2 ]] || { echo "ERROR: --run-one expects one directory." >&2; exit 2; }
    run_one_gpu "$2"
elif [[ "${1:-}" == "--inside" ]]; then
    shift
    if (( $# < 2 || $# > MAX_RUNS )); then
        usage
        exit 2
    fi
    resolve_dirs "$@"
    run_inside_allocation
else
    if (( $# < 2 || $# > MAX_RUNS )); then
        usage
        exit 2
    fi
    resolve_dirs "$@"
    submit "${RUN_DIRS[@]}"
fi
