#!/bin/bash
set -uo pipefail  # drop -e to avoid killing the loop on benign nonzero statuses

USER=$(whoami)

# qselect is the most reliable way to list job IDs by state in PBS/Torque/PBS Pro
# -s R => running jobs only (same as your Slurm -t R)
while read -r job; do
  [[ -z "$job" ]] && continue
  echo "Job ID: $job"

  # Fetch job info (don't let an empty result kill the loop)
  JOB_INFO=$(qstat -f "$job" 2>/dev/null || true)
  if [[ -z "$JOB_INFO" ]]; then
      echo "Could not retrieve job info with qstat. Skipping..."
      echo ""
      continue
  fi

  # Extract fields (each grep may return empty; that's fine)
  # PBS/PBS Pro key names:
  #   job_state (R, Q, H, S, E, T, W, C)
  #   Job_Name
  #   Output_Path  (often host:/path/to/stdout)
  #   init_work_dir (PBS Pro), or Variable_List includes PBS_O_WORKDIR=...
  #   comment (reason/notes; typically for non-running jobs)
  JOB_STATE=$(echo "$JOB_INFO" | grep -oP 'job_state = \K\S+' || true)
  JOB_NAME=$(echo "$JOB_INFO" | grep -oP 'Job_Name = \K.+' || true)
  RAW_OUTPUT_PATH=$(echo "$JOB_INFO" | grep -oP 'Output_Path = \K.+' || true)
  INIT_WORK_DIR=$(echo "$JOB_INFO" | grep -oP 'init_work_dir = \K.+' || true)
  REASON=$(echo "$JOB_INFO" | grep -oP 'comment = \K.+' || true)

  # Normalize Output_Path: strip "host:" prefix and handle "(null)"
  OUTPUT_PATH=""
  if [[ -n "${RAW_OUTPUT_PATH:-}" && "${RAW_OUTPUT_PATH}" != "(null)" ]]; then
      # If the path looks like "node123:/path/file", keep only the path part
      if [[ "$RAW_OUTPUT_PATH" == *:* ]]; then
          OUTPUT_PATH="${RAW_OUTPUT_PATH#*:}"
      else
          OUTPUT_PATH="$RAW_OUTPUT_PATH"
      fi
  fi

  # Elapsed runtime: use resources_used.walltime if present
  RUNTIME=$(echo "$JOB_INFO" | grep -oP 'resources_used.walltime = \K\S+' || true)
  [[ -z "$RUNTIME" ]] && RUNTIME="00:00:00"

  echo "State: ${JOB_STATE:-unknown}"
  echo "Name:  ${JOB_NAME:-unknown}"
  echo "Runtime: $RUNTIME"

  # Resolve job working directory:
  # Prefer init_work_dir (PBS Pro). Otherwise try PBS_O_WORKDIR from Variable_List.
  WORK_DIR="$INIT_WORK_DIR"
  if [[ -z "$WORK_DIR" ]]; then
      # Variable_List can wrap across lines; PBS_O_WORKDIR=... usually sits intact on a single line
      WORK_DIR=$(echo "$JOB_INFO" | grep -oP 'PBS_O_WORKDIR=\K[^,]+' || true)
  fi

  # Choose a job dir to inspect for VASP status:
  JOB_DIR=""
  if [[ -n "${OUTPUT_PATH:-}" ]]; then
      echo "Output path: $OUTPUT_PATH"
      JOB_DIR=$(dirname "$OUTPUT_PATH")
  elif [[ -n "${WORK_DIR:-}" ]]; then
      echo "WorkDir: $WORK_DIR"
      JOB_DIR="$WORK_DIR"
  else
      echo "No Output_Path or WorkDir available."
  fi

  # If somehow we still see a queued/held job (shouldn't, since we filtered -s R)
  if [[ "${JOB_STATE:-}" == "Q" || "${JOB_STATE:-}" == "H" || "${JOB_STATE:-}" == "W" || "${JOB_STATE:-}" == "S" ]]; then
      echo "Reason: ${REASON:-unknown}"
      echo ""
      continue
  fi

  # Progress from OSZICAR/INCAR
  if [[ -n "$JOB_DIR" && -f "$JOB_DIR/OSZICAR" && -f "$JOB_DIR/INCAR" ]]; then
      CURRENT_STEP=$(grep -E "^[[:space:]]*[0-9]+[[:space:]]+T=|^[[:space:]]*[0-9]+[[:space:]]+F=" "$JOB_DIR/OSZICAR" \
                     | tail -1 | awk '{print $1}')
      TOTAL_STEPS=$(grep -i "^[[:space:]]*NSW[[:space:]]*=" "$JOB_DIR/INCAR" \
                    | awk -F'=' '{print $2}' | awk '{print $1}')
      if [[ -n "$CURRENT_STEP" && -n "$TOTAL_STEPS" && "$TOTAL_STEPS" -gt 0 ]]; then
          PERCENT=$(awk -v cur="$CURRENT_STEP" -v total="$TOTAL_STEPS" 'BEGIN { printf "%.2f", (cur / total) * 100 }')
          echo "Completion: $CURRENT_STEP / $TOTAL_STEPS steps (${PERCENT}%)"
      else
          echo "Could not parse step count or NSW from $JOB_DIR"
      fi
  else
      echo "Missing OSZICAR or INCAR in $JOB_DIR"
  fi

  echo ""
done < <(qselect -u "$USER" -s R 2>/dev/null || true)
