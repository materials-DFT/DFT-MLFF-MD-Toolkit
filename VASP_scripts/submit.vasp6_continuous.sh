#!/bin/bash
#PBS -N oms6_md_700K_cont
#PBS -q default
#PBS -l nodes=2:ppn=12
#PBS -l walltime=9999:00:00
#PBS -r y
#PBS -e job.err
#PBS -o job.out
#PBS -m e -M nicoleal@sfsu.edu
#PBS -V

set -euo pipefail

export LD_LIBRARY_PATH="/opt/intel/compilers_and_libraries_2017.1.132/linux/mkl/lib/intel64:/opt/intel/compilers_and_libraries_2017.1.132/linux/compiler/lib/intel64_lin/"
export PATH="/opt/intel/compilers_and_libraries_2017.1.132/linux/mpi/intel64/bin/:$PATH"

BIN="${BIN:-/opt/bin/vasp6_gam}"
WORKDIR="${PBS_O_WORKDIR:-$(pwd)}"
PBS_JOBID_FULL="${PBS_JOBID:-local}"
PBS_JOBNUM="${PBS_JOBID_FULL%%.*}"
cd "${WORKDIR}"

RUN_DIR="${WORKDIR}/.vasp_chunk"
mkdir -p "${RUN_DIR}"
cleanup_stopcar() {
  rm -f "${WORKDIR}/STOPCAR" "${RUN_DIR}/STOPCAR"
}

# Remove stale STOPCAR from a prior stop (recreate STOPCAR to stop this run).
cleanup_stopcar
# Prevent duplicate appends from stale chunk outputs across submissions.
rm -f "${RUN_DIR}/"{CHG,CHGCAR,CONTCAR,DOSCAR,EIGENVAL,IBZKPT,OSZICAR,OUTCAR,PCDAT,REPORT,vasprun.xml,XDATCAR,WAVECAR,.vasp_xml_done,.xdat_state.tmp} 2>/dev/null || true

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: $BIN" >&2
  exit 1
fi

max_ionic_step() {
  local f="$1"
  [[ -f "$f" ]] || { echo 0; return; }
  awk 'BEGIN{m=0} /^[[:space:]]*[0-9]+[[:space:]]+T=/{v=$1+0; if(v>m)m=v} END{printf "%d\n",m}' "$f"
}

max_xdat_config() {
  local f="$1"
  [[ -f "$f" ]] || { echo 0; return; }
  awk 'BEGIN{m=0} /^Direct configuration=/{rest=$0; sub(/^Direct configuration=[[:space:]]*/,"",rest); match(rest,/^[0-9]+/); if(RSTART){n=substr(rest,RSTART,RLENGTH)+0; if(n>m)m=n}} END{printf "%d\n",m}' "$f"
}

sanitize_uint() {
  local v="${1:-0}"
  [[ "$v" =~ ^[0-9]+$ ]] && echo "$v" || echo 0
}

append_oszicar_delta() {
  local src="$1" dst="$2" off done_lines
  off=$(sanitize_uint "${3:-0}")
  done_lines=$(sanitize_uint "${4:-0}")
  [[ -f "$src" ]] || { echo "$done_lines"; return; }
  local total
  total=$(wc -l < "$src")
  if (( total <= done_lines )); then
    echo "$done_lines"
    return
  fi
  awk -v off="$off" -v start=$((done_lines+1)) '
    NR < start { next }
    /^[[:space:]]*[0-9]+[[:space:]]+T=/ {
      match($0,/^[[:space:]]*/)
      spaces=substr($0,1,RLENGTH)
      rest=substr($0,RLENGTH+1)
      match(rest,/^[0-9]+/)
      n=substr(rest,1,RLENGTH)
      tail=substr(rest,RLENGTH+1)
      printf "%s%d%s\n", spaces, n+off, tail
      next
    }
    { print }
  ' "$src" >> "$dst"
  echo "$total"
}

append_outcar_delta() {
  local src="$1" dst="$2" done_bytes
  done_bytes=$(sanitize_uint "${3:-0}")
  [[ -f "$src" ]] || { echo "$done_bytes"; return; }
  local total
  total=$(wc -c < "$src")
  if (( total <= done_bytes )); then
    echo "$done_bytes"
    return
  fi
  dd if="$src" bs=1 skip="$done_bytes" status=none >> "$dst"
  echo "$total"
}

append_xdatcar_delta() {
  local src="$1" dst="$2" off="$3" done_lines="$4" started="$5"
  [[ -f "$src" ]] || { echo "$done_lines $started"; return; }
  local total
  total=$(wc -l < "$src")
  if (( total <= done_lines )); then
    echo "$done_lines $started"
    return
  fi

  awk -v off="$off" -v start=$((done_lines+1)) -v started="$started" '
    NR < start { next }
    {
      if (started==0) {
        if ($0 ~ /^Direct configuration=/) started=1
        else next
      }
      if ($0 ~ /^Direct configuration=/) {
        rest=$0
        sub(/^Direct configuration=[[:space:]]*/,"",rest)
        match(rest,/^[0-9]+/)
        n=substr(rest,RSTART,RLENGTH)+0
        printf "Direct configuration=%6d\n", n+off
      } else {
        print
      }
    }
    END { printf "__STATE__ %d\n", started > "/dev/stderr" }
  ' "$src" >> "$dst" 2>"${RUN_DIR}/.xdat_state.tmp"

  if grep -q "__STATE__ 1" "${RUN_DIR}/.xdat_state.tmp" 2>/dev/null; then
    started=1
  fi
  rm -f "${RUN_DIR}/.xdat_state.tmp"
  echo "$total $started"
}

clean_shm_on_alloc_nodes() {
  if [[ -z "${PBS_NODEFILE:-}" || ! -f "${PBS_NODEFILE}" ]]; then
    return 0
  fi
  echo "Cleaning /dev/shm psm artifacts on allocated nodes..."
  awk '{print $1}' "${PBS_NODEFILE}" | sort -u | while read -r n; do
    [[ -n "$n" ]] || continue
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$n" "rm -f /dev/shm/psm_shm.* /dev/shm/shm-col-space* 2>/dev/null || true" >/dev/null 2>&1 || true
  done
}

TARGET_STEPS="${TARGET_STEPS:-$(awk -F= '/^[[:space:]]*NSW[[:space:]]*=/{gsub(/[[:space:]]/,"",$2); print $2; exit}' INCAR)}"
TARGET_STEPS="$(sanitize_uint "$TARGET_STEPS")"
if (( TARGET_STEPS < 1 )); then
  echo "ERROR: invalid TARGET_STEPS/NSW" >&2
  exit 1
fi

DONE_BEFORE="$(sanitize_uint "$(max_ionic_step "${WORKDIR}/OSZICAR")")"
XDAT_BEFORE="$(sanitize_uint "$(max_xdat_config "${WORKDIR}/XDATCAR")")"

if (( DONE_BEFORE >= TARGET_STEPS )); then
  echo "Target already reached (${DONE_BEFORE} >= ${TARGET_STEPS}); exiting."
  exit 0
fi

NSW_REM=$((TARGET_STEPS - DONE_BEFORE))

cp INCAR "${RUN_DIR}/INCAR"
cp KPOINTS POTCAR "${RUN_DIR}/"
if [[ -s CONTCAR ]]; then cp CONTCAR "${RUN_DIR}/POSCAR"; else cp POSCAR "${RUN_DIR}/POSCAR"; fi

if awk 'BEGIN{IGNORECASE=1} /^[[:space:]]*NSW[[:space:]]*=/{f=1} END{exit(f?0:1)}' "${RUN_DIR}/INCAR"; then
  sed -i "s/^[[:space:]]*NSW[[:space:]]*=.*$/NSW = ${NSW_REM}/" "${RUN_DIR}/INCAR"
else
  echo "NSW = ${NSW_REM}" >> "${RUN_DIR}/INCAR"
fi

if awk 'BEGIN{IGNORECASE=1} /^[[:space:]]*LWAVE[[:space:]]*=/{f=1} END{exit(f?0:1)}' "${RUN_DIR}/INCAR"; then
  sed -i 's/^[[:space:]]*LWAVE[[:space:]]*=.*/LWAVE = .TRUE./' "${RUN_DIR}/INCAR"
else
  echo 'LWAVE = .TRUE.' >> "${RUN_DIR}/INCAR"
fi

WANT_WAVE_RESTART=0
if awk 'BEGIN{IGNORECASE=1} /^[[:space:]]*ISTART[[:space:]]*=[[:space:]]*1/{f=1} END{exit(f?0:1)}' "${RUN_DIR}/INCAR"; then
  WANT_WAVE_RESTART=1
fi
if [[ "${USE_WAVECAR:-}" =~ ^(1|yes|true|on)$ ]]; then
  WANT_WAVE_RESTART=1
fi
if [[ "${USE_WAVECAR:-}" =~ ^(0|no|false|off)$ ]]; then
  WANT_WAVE_RESTART=0
fi
if [[ "${WANT_WAVE_RESTART}" -eq 1 && -s "${WORKDIR}/WAVECAR" ]]; then
  cp "${WORKDIR}/WAVECAR" "${RUN_DIR}/WAVECAR"
  if awk 'BEGIN{IGNORECASE=1} /^[[:space:]]*ISTART[[:space:]]*=/{f=1} END{exit(f?0:1)}' "${RUN_DIR}/INCAR"; then
    sed -i 's/^[[:space:]]*ISTART[[:space:]]*=.*/ISTART = 1/' "${RUN_DIR}/INCAR"
  else
    echo 'ISTART = 1' >> "${RUN_DIR}/INCAR"
  fi
  if awk 'BEGIN{IGNORECASE=1} /^[[:space:]]*ICHARG[[:space:]]*=/{f=1} END{exit(f?0:1)}' "${RUN_DIR}/INCAR"; then
    sed -i 's/^[[:space:]]*ICHARG[[:space:]]*=.*/ICHARG = 0/' "${RUN_DIR}/INCAR"
  else
    echo 'ICHARG = 0' >> "${RUN_DIR}/INCAR"
  fi
  echo "Using WAVECAR for electronic restart (ISTART=1)."
else
  echo "Continuing from CONTCAR; not reading WAVECAR (set ISTART=1 in INCAR or USE_WAVECAR=yes to enable)."
fi

echo "=== live-merge vasp6_gam PBS ==="
echo "Start: $(date -Is)"
echo "PBS_JOBID=${PBS_JOBID:-local}"
echo "TARGET_STEPS=${TARGET_STEPS} NSW_THIS_RUN=${NSW_REM}"
echo "Merge offsets: OSZICAR=${DONE_BEFORE} XDATCAR=${XDAT_BEFORE}"
echo "To stop: touch STOPCAR here (forwarded to ${RUN_DIR}/) or cp STOPCAR ${RUN_DIR}/STOPCAR"
printf '%s %s\n' "${PBS_JOBNUM}" "${DONE_BEFORE}" > "${WORKDIR}/.cj_baseline"
clean_shm_on_alloc_nodes

osz_done=0
out_done=0
xdat_done=0
xdat_started=0
[[ -s "${WORKDIR}/XDATCAR" ]] && xdat_started=1

if [[ -f "${WORKDIR}/OUTCAR" ]] && [[ "$DONE_BEFORE" -gt 0 ]]; then
  printf '\n===== CONTINUATION (after step %d) =====\n\n' "$DONE_BEFORE" >> "${WORKDIR}/OUTCAR"
fi
if [[ -f "${WORKDIR}/vasprun.xml" ]] && [[ "$DONE_BEFORE" -gt 0 ]]; then
  printf '\n<!-- continuation after step %d -->\n' "$DONE_BEFORE" >> "${WORKDIR}/vasprun.xml"
fi

forward_stopcar() {
  # VASP runs in RUN_DIR; parent STOPCAR must be copied there. stat() avoids stale NFS -f.
  if stat "${WORKDIR}/STOPCAR" >/dev/null 2>&1; then
    cp -f "${WORKDIR}/STOPCAR" "${RUN_DIR}/STOPCAR"
  fi
}

(
  cd "${RUN_DIR}"
  mpirun "$BIN"
) &
VASP_PID=$!

# Independent watcher: keeps forwarding even if the merge loop hits set -e errors.
(
  while kill -0 "$VASP_PID" 2>/dev/null; do
    forward_stopcar || true
    sleep 2
  done
) &
STOPCAR_PID=$!

while kill -0 "$VASP_PID" 2>/dev/null; do
  forward_stopcar || true
  set +e
  osz_done="$(append_oszicar_delta "${RUN_DIR}/OSZICAR" "${WORKDIR}/OSZICAR" "$DONE_BEFORE" "$osz_done")"
  out_done="$(append_outcar_delta "${RUN_DIR}/OUTCAR" "${WORKDIR}/OUTCAR" "$out_done")"
  read -r xdat_done xdat_started < <(append_xdatcar_delta "${RUN_DIR}/XDATCAR" "${WORKDIR}/XDATCAR" "$XDAT_BEFORE" "$xdat_done" "$xdat_started")
  if [[ -f "${RUN_DIR}/vasprun.xml" ]]; then
    if [[ ! -f "${RUN_DIR}/.vasp_xml_done" ]]; then echo 0 > "${RUN_DIR}/.vasp_xml_done"; fi
    xml_done=0
    if [[ -f "${RUN_DIR}/.vasp_xml_done" ]]; then
      xml_done=$(<"${RUN_DIR}/.vasp_xml_done")
    fi
    xml_done="$(append_outcar_delta "${RUN_DIR}/vasprun.xml" "${WORKDIR}/vasprun.xml" "$xml_done")"
    echo "$xml_done" > "${RUN_DIR}/.vasp_xml_done"
  fi
  set -e
  sleep 20
done

wait "$STOPCAR_PID" 2>/dev/null || true

set +e
wait "$VASP_PID"
VASP_RC=$?
set -e

# Final flush
set +e
osz_done="$(append_oszicar_delta "${RUN_DIR}/OSZICAR" "${WORKDIR}/OSZICAR" "$DONE_BEFORE" "$osz_done")"
out_done="$(append_outcar_delta "${RUN_DIR}/OUTCAR" "${WORKDIR}/OUTCAR" "$out_done")"
read -r xdat_done xdat_started < <(append_xdatcar_delta "${RUN_DIR}/XDATCAR" "${WORKDIR}/XDATCAR" "$XDAT_BEFORE" "$xdat_done" "$xdat_started")
if [[ -f "${RUN_DIR}/vasprun.xml" ]]; then
  xml_done=0
  if [[ -f "${RUN_DIR}/.vasp_xml_done" ]]; then
    xml_done=$(<"${RUN_DIR}/.vasp_xml_done")
  fi
  append_outcar_delta "${RUN_DIR}/vasprun.xml" "${WORKDIR}/vasprun.xml" "$xml_done" >/dev/null
fi

if [[ -s "${RUN_DIR}/CONTCAR" ]]; then
  cp "${RUN_DIR}/CONTCAR" "${WORKDIR}/CONTCAR"
  cp "${RUN_DIR}/CONTCAR" "${WORKDIR}/POSCAR"
fi
if [[ -s "${RUN_DIR}/WAVECAR" ]]; then
  cp "${RUN_DIR}/WAVECAR" "${WORKDIR}/WAVECAR"
fi

DONE_AFTER="$(sanitize_uint "$(max_ionic_step "${WORKDIR}/OSZICAR")")"
echo "Cumulative ionic steps now: ${DONE_AFTER}"
set -e
echo "End: $(date -Is)"

STOPPED_BY_STOPCAR=0
if [[ -f "${RUN_DIR}/STOPCAR" || -f "${WORKDIR}/STOPCAR" ]]; then
  STOPPED_BY_STOPCAR=1
elif [[ -f "${RUN_DIR}/OUTCAR" ]] && grep -q "soft stop encountered" "${RUN_DIR}/OUTCAR" 2>/dev/null; then
  STOPPED_BY_STOPCAR=1
fi

if [[ "$STOPPED_BY_STOPCAR" -eq 1 ]]; then
  cleanup_stopcar
  echo "Stopped via STOPCAR; removed STOPCAR files for clean resubmit."
  VASP_RC=0
elif [[ "$VASP_RC" -ne 0 ]]; then
  echo "ERROR: mpirun exited with code ${VASP_RC}" >&2
  exit "$VASP_RC"
fi

exit 0
