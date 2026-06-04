#!/bin/bash
#SBATCH --job-name=vg2_time
#SBATCH --partition=gpuquick
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=job.out
#SBATCH --error=job.err

# Continuous MD slice for gpuquick walltime.
# Requeues until cumulative ionic steps reach NSW in INCAR, appending
# logs with continuous numbering. Saves WAVECAR for fast restart.

set -euo pipefail

source /etc/profile.d/modules.sh
module purge
module load gcc-toolset/12

sanitize_ld_library_path() {
  local p cleaned=""
  local IFS=:
  for p in ${LD_LIBRARY_PATH:-}; do
    case "$p" in
      *miniconda*|*anaconda*|*mambaforge*) ;;
      *) cleaned="${cleaned:+$cleaned:}${p}" ;;
    esac
  done
  LD_LIBRARY_PATH="$cleaned"
}

NVROOT="${NVROOT:-${HOME}/nvidia_hpc_sdk/install/Linux_x86_64/25.9}"
CUDA_TAG="12.9"
HPCX_INIT="${HOME}/nvidia_hpc_sdk/nvhpc_2025_259_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/25.9/comm_libs/${CUDA_TAG}/hpcx/hpcx-2.24/hpcx-init-ompi.sh"
BIN="${BIN:-${HOME}/vasp6/vasp.6.4.3/bin/vasp_gam}"
NUMA_COMPAT_DIR="${HOME}/vasp6/.compat/lib"

mkdir -p "${NUMA_COMPAT_DIR}"
if [[ ! -e "${NUMA_COMPAT_DIR}/libnuma.so" && -e /lib64/libnuma.so.1 ]]; then
  ln -s /lib64/libnuma.so.1 "${NUMA_COMPAT_DIR}/libnuma.so"
fi

export PATH="${NVROOT}/compilers/bin:${NVROOT}/cuda/${CUDA_TAG}/bin:${PATH}"
sanitize_ld_library_path
# shellcheck source=/dev/null
set +u
source "${HPCX_INIT}"
hpcx_load
set -u

export CUDA_HOME="${NVROOT}/cuda/${CUDA_TAG}"
sanitize_ld_library_path
NV_COMP="${NVROOT}/compilers/lib:${NVROOT}/compilers/extras/qd/lib:${CUDA_HOME}/lib64"
GCC_STDCXX="${GCC_STDCXX:-/opt/rh/gcc-toolset-12/root/usr/lib64}"
export LD_LIBRARY_PATH="${NUMA_COMPAT_DIR}:${NV_COMP}:${LD_LIBRARY_PATH}:${GCC_STDCXX}:/usr/lib64"
export ACC_DEVICE_TYPE=nvidia

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

cd "${SLURM_SUBMIT_DIR}"

WRITE_CHGCAR="${WRITE_CHGCAR:-no}"

echo "=== continuous vasp_gam 2xGPU cpus-per-task=${SLURM_CPUS_PER_TASK} ==="
echo "Start: $(date -Is)"
echo "SLURM_NTASKS=${SLURM_NTASKS} SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK} OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L
fi

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: $BIN" >&2
  exit 1
fi

TARGET_STEPS="${TARGET_STEPS:-$(awk -F= '/^[[:space:]]*NSW[[:space:]]*=/{gsub(/[[:space:]]/,"",$2); print $2; exit}' INCAR)}"
if [[ -z "${TARGET_STEPS}" ]]; then
  echo "ERROR: Could not determine TARGET_STEPS. Set TARGET_STEPS or NSW in INCAR." >&2
  exit 1
fi

# Largest ionic step index in OSZICAR (first field on MD lines "N T=...").
# Use max, not "last line in file": after a bad merge the tail can restart at 1
# while earlier segments still hold the true progress (e.g. max=7316, last=112).
# BEGIN/END + printf: never emit a blank line (empty would make DONE_BEFORE break [[ -gt 0 ]] and raw cat append).
max_ionic_step() {
  local osz_file="$1"
  [[ -f "$osz_file" ]] || { echo 0; return; }
  awk 'BEGIN{m=0} /^[[:space:]]*[0-9]+[[:space:]]+T=/{v=$1+0; if(v>m)m=v} END{printf "%d\n", m}' "$osz_file"
}

# Largest frame index on "Direct configuration=" lines in XDATCAR.
max_xdat_config() {
  local f="$1"
  [[ -f "$f" ]] || { echo 0; return; }
  awk 'BEGIN{m=0} /^Direct configuration=/{rest=$0; sub(/^Direct configuration=[[:space:]]*/, "", rest); match(rest,/^[0-9]+/); if(RSTART){n=substr(rest,RSTART,RLENGTH)+0; if(n>m)m=n}} END{printf "%d\n", m}' "$f"
}

# Largest index on "MD step No." lines in REPORT.
max_report_step() {
  local f="$1"
  [[ -f "$f" ]] || { echo 0; return; }
  awk 'BEGIN{m=0} /MD step No\./{rest=$0; if(match(rest,/MD step No\.[[:space:]]+/)){rest=substr(rest,RSTART+RLENGTH); match(rest,/^[0-9]+/); if(RSTART){n=substr(rest,RSTART,RLENGTH)+0; if(n>m)m=n}}} END{printf "%d\n", m}' "$f"
}

sanitize_uint() {
  local v="${1:-0}"
  if [[ "${v}" =~ ^[0-9]+$ ]]; then
    echo "${v}"
  else
    echo 0
  fi
}

# Append OSZICAR with shifted ionic step numbers (always awk: off=0 is cold start, never raw cat).
append_oszicar_shifted() {
  local src="$1" dst="$2" off
  off="$(sanitize_uint "${3:-0}")"
  [[ -f "$src" ]] || return 0
  awk -v off="$off" '
    /^[[:space:]]*[0-9]+[[:space:]]+T=/ {
      match($0, /^[[:space:]]*/)
      spaces = substr($0, 1, RLENGTH)
      rest = substr($0, RLENGTH+1)
      match(rest, /^[0-9]+/)
      n = substr(rest, 1, RLENGTH)
      tail = substr(rest, RLENGTH+1)
      printf "%s%d%s\n", spaces, n+off, tail
      next
    }
    {print}
  ' "$src" >> "$dst"
}

# Append XDATCAR with shifted configuration numbers (per-file max offset; never unshifted append to existing dst).
append_xdatcar_shifted() {
  local src="$1" dst="$2" off
  off="$(sanitize_uint "${3:-0}")"
  [[ -f "$src" ]] || return 0
  if [[ ! -f "$dst" ]]; then
    cp "$src" "$dst"
    return 0
  fi
  awk -v off="$off" '
    BEGIN {started=0}
    /^Direct configuration=/ {
      started=1
      rest = $0
      sub(/^Direct configuration=[[:space:]]*/, "", rest)
      match(rest, /^[0-9]+/)
      n = substr(rest, RSTART, RLENGTH) + 0
      printf "Direct configuration=%6d\n", n + off
      next
    }
    started {print}
  ' "$src" >> "$dst"
}

# Append REPORT with shifted MD step numbers (always awk).
append_report_shifted() {
  local src="$1" dst="$2" off
  off="$(sanitize_uint "${3:-0}")"
  [[ -f "$src" ]] || return 0
  awk -v off="$off" '
    /MD step No\./ {
      match($0, /[0-9]+/)
      n = substr($0, RSTART, RLENGTH)
      pre = substr($0, 1, RSTART-1)
      post = substr($0, RSTART+RLENGTH)
      printf "%s%d%s\n", pre, n+off, post
      next
    }
    {print}
  ' "$src" >> "$dst"
}

append_text_file() {
  local src="$1" dst="$2"
  [[ -f "$src" ]] || return 0
  if [[ -f "$dst" ]]; then
    cat "$src" >> "$dst"
  else
    cp "$src" "$dst"
  fi
}

RUN_DIR="${SLURM_SUBMIT_DIR}/.vasp_chunk"
mkdir -p "${RUN_DIR}"

MERGE_MARKER="${SLURM_SUBMIT_DIR}/.merged_chunk_${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}"
if [[ -f "${MERGE_MARKER}" ]]; then
  echo "Found existing merge marker for this requeue attempt: ${MERGE_MARKER}"
  echo "Skipping run to avoid duplicate append."
  exit 0
fi

DONE_BEFORE="$(max_ionic_step "${SLURM_SUBMIT_DIR}/OSZICAR")"
DONE_BEFORE="$(sanitize_uint "${DONE_BEFORE}")"
XDAT_BEFORE="$(max_xdat_config "${SLURM_SUBMIT_DIR}/XDATCAR")"
XDAT_BEFORE="$(sanitize_uint "${XDAT_BEFORE}")"
REPORT_BEFORE="$(max_report_step "${SLURM_SUBMIT_DIR}/REPORT")"
REPORT_BEFORE="$(sanitize_uint "${REPORT_BEFORE}")"

if [[ "${DONE_BEFORE}" -ge "${TARGET_STEPS}" ]]; then
  echo "Target ionic steps already reached (${DONE_BEFORE} >= ${TARGET_STEPS}); not running VASP."
  echo "End: $(date -Is)"
  exit 0
fi

NSW_REM=$(( TARGET_STEPS - DONE_BEFORE ))
if [[ "${NSW_REM}" -lt 1 ]]; then
  NSW_REM=1
fi

on_walltime_signal() {
  echo "LSTOP = .TRUE." > "${RUN_DIR}/STOPCAR"
  echo "USR1 caught -> STOPCAR written."
}
trap on_walltime_signal USR1

rm -f "${RUN_DIR}/STOPCAR"
rm -f "${RUN_DIR}/WAVECAR" "${RUN_DIR}/CHGCAR" "${RUN_DIR}/CHG"
cp INCAR "${RUN_DIR}/INCAR"
cp KPOINTS POTCAR "${RUN_DIR}/"

# Set NSW to remaining steps
if awk 'BEGIN{IGNORECASE=1} /^[[:space:]]*NSW[[:space:]]*=/{found=1} END{exit(found?0:1)}' "${RUN_DIR}/INCAR"; then
  sed -i "s/^[[:space:]]*NSW[[:space:]]*=.*$/NSW = ${NSW_REM}/" "${RUN_DIR}/INCAR"
else
  echo "NSW = ${NSW_REM}" >> "${RUN_DIR}/INCAR"
fi

# Use CONTCAR if available
if [[ -s CONTCAR ]]; then
  cp CONTCAR "${RUN_DIR}/POSCAR"
else
  cp POSCAR "${RUN_DIR}/POSCAR"
fi

# Always write WAVECAR for restart
if awk 'BEGIN{IGNORECASE=1} /^[[:space:]]*LWAVE[[:space:]]*=/{found=1} END{exit(found?0:1)}' "${RUN_DIR}/INCAR"; then
  sed -i 's/^[[:space:]]*LWAVE[[:space:]]*=.*/LWAVE = .TRUE./' "${RUN_DIR}/INCAR"
else
  echo 'LWAVE = .TRUE.' >> "${RUN_DIR}/INCAR"
fi

# CHGCAR controlled by env
want_chg=0
case "${WRITE_CHGCAR,,}" in
  1|yes|true|on) want_chg=1 ;;
esac
if [[ "${want_chg}" -eq 1 ]]; then
  if awk 'BEGIN{IGNORECASE=1} /^[[:space:]]*LCHARG[[:space:]]*=/{found=1} END{exit(found?0:1)}' "${RUN_DIR}/INCAR"; then
    sed -i 's/^[[:space:]]*LCHARG[[:space:]]*=.*/LCHARG = .TRUE./' "${RUN_DIR}/INCAR"
  else
    echo 'LCHARG = .TRUE.' >> "${RUN_DIR}/INCAR"
  fi
else
  if awk 'BEGIN{IGNORECASE=1} /^[[:space:]]*LCHARG[[:space:]]*=/{found=1} END{exit(found?0:1)}' "${RUN_DIR}/INCAR"; then
    sed -i 's/^[[:space:]]*LCHARG[[:space:]]*=.*/LCHARG = .FALSE./' "${RUN_DIR}/INCAR"
  else
    echo 'LCHARG = .FALSE.' >> "${RUN_DIR}/INCAR"
  fi
fi

# Copy WAVECAR if exists for restart
HAVE_WAVE=0
if [[ -s "${SLURM_SUBMIT_DIR}/WAVECAR" ]]; then
  HAVE_WAVE=1
  cp "${SLURM_SUBMIT_DIR}/WAVECAR" "${RUN_DIR}/WAVECAR"
fi
if [[ "${want_chg}" -eq 1 ]] && [[ -s "${SLURM_SUBMIT_DIR}/CHGCAR" ]]; then
  cp "${SLURM_SUBMIT_DIR}/CHGCAR" "${RUN_DIR}/CHGCAR"
fi

# Set ISTART/ICHARG based on restart state
if [[ "${HAVE_WAVE}" -eq 1 ]]; then
  echo "Continuing from WAVECAR (max ionic step index before chunk: ${DONE_BEFORE})."
  if awk 'BEGIN{IGNORECASE=1} /^[[:space:]]*ISTART[[:space:]]*=/{found=1} END{exit(found?0:1)}' "${RUN_DIR}/INCAR"; then
    sed -i 's/^[[:space:]]*ISTART[[:space:]]*=.*/ISTART = 1/' "${RUN_DIR}/INCAR"
  else
    echo 'ISTART = 1' >> "${RUN_DIR}/INCAR"
  fi
  if awk 'BEGIN{IGNORECASE=1} /^[[:space:]]*ICHARG[[:space:]]*=/{found=1} END{exit(found?0:1)}' "${RUN_DIR}/INCAR"; then
    sed -i 's/^[[:space:]]*ICHARG[[:space:]]*=.*/ICHARG = 0/' "${RUN_DIR}/INCAR"
  else
    echo 'ICHARG = 0' >> "${RUN_DIR}/INCAR"
  fi
elif [[ "${DONE_BEFORE}" -eq 0 ]]; then
  echo "Cold start (no prior steps). Using INCAR defaults for ISTART/ICHARG."
else
  echo "WARN: Continuing without WAVECAR (done=${DONE_BEFORE}); using ISTART=0/ICHARG=2."
  if awk 'BEGIN{IGNORECASE=1} /^[[:space:]]*ISTART[[:space:]]*=/{found=1} END{exit(found?0:1)}' "${RUN_DIR}/INCAR"; then
    sed -i 's/^[[:space:]]*ISTART[[:space:]]*=.*/ISTART = 0/' "${RUN_DIR}/INCAR"
  else
    echo 'ISTART = 0' >> "${RUN_DIR}/INCAR"
  fi
  if awk 'BEGIN{IGNORECASE=1} /^[[:space:]]*ICHARG[[:space:]]*=/{found=1} END{exit(found?0:1)}' "${RUN_DIR}/INCAR"; then
    sed -i 's/^[[:space:]]*ICHARG[[:space:]]*=.*/ICHARG = 2/' "${RUN_DIR}/INCAR"
  else
    echo 'ICHARG = 2' >> "${RUN_DIR}/INCAR"
  fi
fi

echo "TARGET_STEPS=${TARGET_STEPS} NSW_THIS_CHUNK=${NSW_REM}"
echo "Merge offsets before chunk: OSZICAR max=${DONE_BEFORE} XDATCAR max=${XDAT_BEFORE} REPORT max=${REPORT_BEFORE}"

SECONDS=0
(
  cd "${RUN_DIR}"
  srun "$BIN"
) &
SRUN_PID=$!
set +e
wait "${SRUN_PID}"
SRUN_RC=$?
set -e
echo "wall_seconds=${SECONDS}"

if [[ "${SRUN_RC}" -ne 0 ]]; then
  if [[ -f "${RUN_DIR}/STOPCAR" ]]; then
    echo "srun exited with code ${SRUN_RC} after STOPCAR; continuing with merge."
  else
    echo "ERROR: srun failed with code ${SRUN_RC}." >&2
    exit "${SRUN_RC}"
  fi
fi

# Save WAVECAR/CHGCAR back
if [[ -s "${RUN_DIR}/WAVECAR" ]]; then
  cp "${RUN_DIR}/WAVECAR" "${SLURM_SUBMIT_DIR}/WAVECAR"
fi
if [[ "${want_chg}" -eq 1 ]] && [[ -s "${RUN_DIR}/CHGCAR" ]]; then
  cp "${RUN_DIR}/CHGCAR" "${SLURM_SUBMIT_DIR}/CHGCAR"
fi

# Re-read merge offsets from disk immediately before append (authoritative; avoids stale 0).
OSZ_MERGE_OFF="$(sanitize_uint "$(max_ionic_step "${SLURM_SUBMIT_DIR}/OSZICAR")")"
XDAT_MERGE_OFF="$(sanitize_uint "$(max_xdat_config "${SLURM_SUBMIT_DIR}/XDATCAR")")"
REPORT_MERGE_OFF="$(sanitize_uint "$(max_report_step "${SLURM_SUBMIT_DIR}/REPORT")")"

# Refuse silent corruption: cumulative logs already have MD frames but parser returned 0.
if [[ -f "${SLURM_SUBMIT_DIR}/OSZICAR" ]] && [[ -s "${SLURM_SUBMIT_DIR}/OSZICAR" ]]; then
  OSZ_MD_COUNT="$(awk 'BEGIN{n=0} /^[[:space:]]*[0-9]+[[:space:]]+T=/{n++} END{printf "%d", n}' "${SLURM_SUBMIT_DIR}/OSZICAR")"
  if [[ "${OSZ_MD_COUNT}" -gt 0 && "${OSZ_MERGE_OFF}" -eq 0 ]]; then
    echo "ERROR: OSZICAR has ${OSZ_MD_COUNT} MD lines but max ionic index is 0 (parse/path failure). Refusing merge that would restart at step 1." >&2
    exit 1
  fi
fi
if [[ -f "${SLURM_SUBMIT_DIR}/XDATCAR" ]] && [[ -s "${SLURM_SUBMIT_DIR}/XDATCAR" ]]; then
  XD_COUNT="$(awk 'BEGIN{n=0} /^Direct configuration=/{n++} END{printf "%d", n}' "${SLURM_SUBMIT_DIR}/XDATCAR")"
  if [[ "${XD_COUNT}" -gt 0 && "${XDAT_MERGE_OFF}" -eq 0 ]]; then
    echo "ERROR: XDATCAR has ${XD_COUNT} frames but max configuration index is 0. Refusing merge." >&2
    exit 1
  fi
fi
echo "Merge append offsets (from disk): OSZICAR=${OSZ_MERGE_OFF} XDATCAR=${XDAT_MERGE_OFF} REPORT=${REPORT_MERGE_OFF}"

# Append logs with shifted numbering (offset = max index in cumulative file before this append)
if [[ -f "${SLURM_SUBMIT_DIR}/OUTCAR" ]] && [[ "${OSZ_MERGE_OFF}" -gt 0 ]]; then
  printf '\n===== CONTINUATION (after step %d) =====\n\n' "${OSZ_MERGE_OFF}" >> "${SLURM_SUBMIT_DIR}/OUTCAR"
fi
append_text_file "${RUN_DIR}/OUTCAR" "${SLURM_SUBMIT_DIR}/OUTCAR"

append_oszicar_shifted "${RUN_DIR}/OSZICAR" "${SLURM_SUBMIT_DIR}/OSZICAR" "${OSZ_MERGE_OFF}"
append_report_shifted "${RUN_DIR}/REPORT" "${SLURM_SUBMIT_DIR}/REPORT" "${REPORT_MERGE_OFF}"
append_xdatcar_shifted "${RUN_DIR}/XDATCAR" "${SLURM_SUBMIT_DIR}/XDATCAR" "${XDAT_MERGE_OFF}"

if [[ -f "${SLURM_SUBMIT_DIR}/vasprun.xml" ]] && [[ "${OSZ_MERGE_OFF}" -gt 0 ]]; then
  printf '\n<!-- continuation after step %d -->\n' "${OSZ_MERGE_OFF}" >> "${SLURM_SUBMIT_DIR}/vasprun.xml"
fi
append_text_file "${RUN_DIR}/vasprun.xml" "${SLURM_SUBMIT_DIR}/vasprun.xml"

if [[ -s "${RUN_DIR}/CONTCAR" ]]; then
  cp "${RUN_DIR}/CONTCAR" "${SLURM_SUBMIT_DIR}/CONTCAR"
  cp "${RUN_DIR}/CONTCAR" "${SLURM_SUBMIT_DIR}/POSCAR"
fi

touch "${MERGE_MARKER}"

DONE_STEPS="$(max_ionic_step "${SLURM_SUBMIT_DIR}/OSZICAR")"
DONE_STEPS="$(sanitize_uint "${DONE_STEPS}")"
FINAL_MAX="${DONE_STEPS}"
echo "Cumulative ionic steps after chunk: ${DONE_STEPS} (max step index ${FINAL_MAX})"
echo "Target ionic steps: ${TARGET_STEPS}"

if [[ "${DONE_STEPS}" -ge "${TARGET_STEPS}" ]]; then
  echo "Target reached. Not requeueing."
  echo "End: $(date -Is)"
  exit 0
fi

echo "Target not reached. Requeueing job ${SLURM_JOB_ID}..."
scontrol requeue "${SLURM_JOB_ID}" || {
  echo "WARNING: requeue failed; resubmit manually."
  echo "End: $(date -Is)"
  exit 1
}

echo "End: $(date -Is)"
