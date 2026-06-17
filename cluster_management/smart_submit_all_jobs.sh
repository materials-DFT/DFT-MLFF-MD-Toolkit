#!/bin/bash
#
# smart_submit_all_jobs.sh - Submit all jobs with automatic retry when queue limits are hit
#
# This script bypasses cluster job submission caps by:
# 1. Finding all submit scripts in the target directory
# 2. Attempting to submit each one
# 3. Tracking which submissions fail due to queue limits
# 4. Automatically retrying failed submissions when slots become available
#
# By default, runs in the background (daemonized) so you can close your terminal.
#
# Usage: smart_submit_all_jobs.sh <target_directory> [options]
#
# Options:
#   --check-interval <seconds>   How often to check for queue slots (default: 60)
#   --max-retries <n>            Maximum retry attempts per job (default: 1000)
#   --dry-run                    Show what would be submitted without actually submitting
#   --foreground                 Run in foreground instead of daemonizing
#   --verbose                    Show detailed output (only useful with --foreground)
#

set -euo pipefail

# Default settings
CHECK_INTERVAL=60
MAX_RETRIES=1000
DRY_RUN=false
VERBOSE=false
FOREGROUND=false
TARGET_DIR=""
DAEMONIZED=false
RETRY_FILE=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --check-interval)
            CHECK_INTERVAL="$2"
            shift 2
            ;;
        --max-retries)
            MAX_RETRIES="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            FOREGROUND=true  # dry-run implies foreground
            shift
            ;;
        --foreground)
            FOREGROUND=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --daemonized)
            # Internal flag: we are the daemonized child process
            DAEMONIZED=true
            shift
            ;;
        --retry-file)
            # Internal: file containing jobs to retry
            RETRY_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 <target_directory> [options]"
            echo ""
            echo "Automatically submits all jobs and retries any that hit queue limits."
            echo "Runs in background by default (safe to close terminal)."
            echo ""
            echo "Options:"
            echo "  --check-interval <seconds>   How often to check for queue slots (default: 60)"
            echo "  --max-retries <n>            Maximum retry attempts per job (default: 1000)"
            echo "  --dry-run                    Show what would be submitted without actually submitting"
            echo "  --foreground                 Run in foreground instead of background"
            echo "  --verbose                    Show detailed output (only useful with --foreground)"
            exit 0
            ;;
        *)
            if [[ -z "$TARGET_DIR" ]]; then
                TARGET_DIR="$1"
            else
                echo "Error: Unknown argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate target directory
if [[ -z "$TARGET_DIR" ]]; then
    echo "Usage: $0 <target_directory> [options]"
    echo "Use --help for more information."
    exit 1
fi

if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Error: Directory $TARGET_DIR does not exist."
    exit 1
fi

cd "$TARGET_DIR" || exit 1
TARGET_DIR="$(pwd)"  # Get absolute path

# Daemonize if not already daemonized and not in foreground mode
if [[ "$FOREGROUND" == "false" ]] && [[ "$DAEMONIZED" == "false" ]]; then
    # Find all submit scripts
    mapfile -t ALL_SCRIPTS < <(find . -type f -name '*submit*' | sort)
    
    if [[ ${#ALL_SCRIPTS[@]} -eq 0 ]]; then
        echo "No submit scripts found in $TARGET_DIR"
        exit 0
    fi
    
    # Try to submit each job, track what succeeds vs what needs retry
    declare -a SUBMITTED_NOW=()
    declare -a WILL_RETRY=()
    
    for script in "${ALL_SCRIPTS[@]}"; do
        script_dir=$(dirname "$script")
        script_name=$(basename "$script")
        
        output=$(cd "$script_dir" && sbatch "$script_name" 2>&1) || true
        
        if echo "$output" | grep -q "Submitted batch job"; then
            job_id=$(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+')
            SUBMITTED_NOW+=("$script (Job $job_id)")
        else
            WILL_RETRY+=("$script_dir|$script_name")
        fi
    done
    
    # Show results
    if [[ ${#SUBMITTED_NOW[@]} -gt 0 ]]; then
        echo "Submitted:"
        for s in "${SUBMITTED_NOW[@]}"; do
            echo "  $s"
        done
    fi
    
    if [[ ${#WILL_RETRY[@]} -gt 0 ]]; then
        echo "Will submit when queue allows:"
        for entry in "${WILL_RETRY[@]}"; do
            IFS='|' read -r script_dir script_name <<< "$entry"
            echo "  $script_dir/$script_name"
        done
        
        # Write retry list to temp file for daemon to read
        RETRY_FILE=$(mktemp)
        printf '%s\n' "${WILL_RETRY[@]}" > "$RETRY_FILE"
        
        # Start daemon to handle retries
        nohup "$0" "$TARGET_DIR" --daemonized --check-interval "$CHECK_INTERVAL" --max-retries "$MAX_RETRIES" \
            --retry-file "$RETRY_FILE" </dev/null >/dev/null 2>&1 &
        disown
    fi
    
    exit 0
fi

# When daemonized, all output goes to /dev/null anyway, but we can skip formatting
log() {
    [[ "$DAEMONIZED" == "true" ]] && return
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    [[ "$DAEMONIZED" == "true" ]] && return
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✓${NC} $1"
}

log_warning() {
    [[ "$DAEMONIZED" == "true" ]] && return
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠${NC} $1"
}

log_error() {
    [[ "$DAEMONIZED" == "true" ]] && return
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ✗${NC} $1"
}

verbose_log() {
    [[ "$DAEMONIZED" == "true" ]] && return
    if [[ "$VERBOSE" == "true" ]]; then
        log "$1"
    fi
}

# Function to check if submission failed due to queue limit
is_queue_limit_error() {
    local output="$1"
    # Common SLURM queue limit error messages
    # Note: QOSMaxSubmitJobPerUserLimit (singular Job) is different from QOSMaxSubmitJobsPerUser (plural Jobs)
    if echo "$output" | grep -qiE "MaxSubmitJob|job.?limit|submission.?limit|too many|AssocMaxSubmit|QOSMax.*Limit|violates.*policy.*limit"; then
        return 0
    fi
    return 1
}

# Function to get current number of jobs in queue for user
get_queue_count() {
    squeue -u "$(whoami)" -h 2>/dev/null | wc -l
}

# Function to attempt job submission
# Returns: 0 = success, 1 = queue limit hit, 2 = other error
attempt_submit() {
    local script_dir="$1"
    local submit_script="$2"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would submit: $script_dir/$submit_script"
        return 0
    fi
    
    # Capture both stdout and stderr
    local output
    local exit_code
    
    output=$(cd "$script_dir" && sbatch "$submit_script" 2>&1) || exit_code=$?
    exit_code=${exit_code:-0}
    
    if [[ $exit_code -eq 0 ]] && echo "$output" | grep -q "Submitted batch job"; then
        local job_id
        job_id=$(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+')
        log_success "Submitted $script_dir/$submit_script (Job ID: $job_id)"
        return 0
    elif is_queue_limit_error "$output"; then
        verbose_log "Queue limit hit for $script_dir/$submit_script"
        return 1
    else
        log_error "Failed to submit $script_dir/$submit_script: $output"
        return 2
    fi
}

# Find all submit scripts or load from retry file
declare -a PENDING_JOBS=()
declare -a FAILED_JOBS=()

if [[ -n "$RETRY_FILE" ]] && [[ -f "$RETRY_FILE" ]]; then
    # Daemon mode: load jobs from retry file
    while IFS= read -r entry; do
        [[ -n "$entry" ]] && PENDING_JOBS+=("$entry")
    done < "$RETRY_FILE"
    rm -f "$RETRY_FILE"  # Clean up temp file
else
    log "Scanning for submit scripts in $TARGET_DIR..."
    while IFS= read -r script; do
        script_dir=$(dirname "$script")
        script_name=$(basename "$script")
        PENDING_JOBS+=("$script_dir|$script_name")
    done < <(find . -type f -name '*submit*' | sort)
fi

TOTAL_JOBS=${#PENDING_JOBS[@]}

if [[ $TOTAL_JOBS -eq 0 ]]; then
    log_warning "No submit scripts found in $TARGET_DIR"
    exit 0
fi

# Track statistics
SUBMITTED=0
QUEUE_BLOCKED=0
OTHER_FAILED=0
RETRY_COUNT=0

declare -a RETRY_QUEUE=()

# If we're a daemon with a retry file, skip initial pass - jobs are already for retry
if [[ "$DAEMONIZED" == "true" ]] && [[ ${#PENDING_JOBS[@]} -gt 0 ]]; then
    RETRY_QUEUE=("${PENDING_JOBS[@]}")
else
    log "Found $TOTAL_JOBS submit scripts to process"
    echo ""

    # First pass: try to submit everything
    log "Starting initial submission pass..."

    for job_entry in "${PENDING_JOBS[@]}"; do
        IFS='|' read -r script_dir script_name <<< "$job_entry"
        
        # Capture return code without triggering set -e
        result=0
        attempt_submit "$script_dir" "$script_name" || result=$?
        
        case $result in
            0)
                ((SUBMITTED++)) || true
                ;;
            1)
                RETRY_QUEUE+=("$job_entry")
                ((QUEUE_BLOCKED++)) || true
                ;;
            2)
                FAILED_JOBS+=("$job_entry")
                ((OTHER_FAILED++)) || true
                ;;
        esac
    done

    echo ""
    log "Initial pass complete: $SUBMITTED submitted, ${#RETRY_QUEUE[@]} blocked by queue limit, $OTHER_FAILED failed"
fi

# Retry loop for queue-blocked jobs
if [[ ${#RETRY_QUEUE[@]} -gt 0 ]] && [[ "$DRY_RUN" == "false" ]]; then
    echo ""
    log "Starting automatic retry loop for ${#RETRY_QUEUE[@]} queue-blocked jobs..."
    log "Checking every $CHECK_INTERVAL seconds for available slots (Ctrl+C to stop)"
    echo ""
    
    while [[ ${#RETRY_QUEUE[@]} -gt 0 ]] && [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; do
        ((RETRY_COUNT++)) || true
        
        current_queue=$(get_queue_count)
        verbose_log "Retry attempt $RETRY_COUNT: ${#RETRY_QUEUE[@]} jobs pending, $current_queue jobs in queue"
        
        declare -a STILL_BLOCKED=()
        
        for job_entry in "${RETRY_QUEUE[@]}"; do
            IFS='|' read -r script_dir script_name <<< "$job_entry"
            
            # Capture return code without triggering set -e
            result=0
            attempt_submit "$script_dir" "$script_name" || result=$?
            
            case $result in
                0)
                    ((SUBMITTED++)) || true
                    ;;
                1)
                    STILL_BLOCKED+=("$job_entry")
                    ;;
                2)
                    FAILED_JOBS+=("$job_entry")
                    ((OTHER_FAILED++)) || true
                    ;;
            esac
        done
        
        RETRY_QUEUE=("${STILL_BLOCKED[@]+"${STILL_BLOCKED[@]}"}")
        
        if [[ ${#RETRY_QUEUE[@]} -gt 0 ]]; then
            log "Waiting $CHECK_INTERVAL seconds before retry... (${#RETRY_QUEUE[@]} jobs remaining)"
            sleep "$CHECK_INTERVAL"
        fi
    done
fi

# Final summary
echo ""
echo "========================================"
log "SUBMISSION COMPLETE"
echo "========================================"
echo -e "  ${GREEN}Submitted:${NC}      $SUBMITTED / $TOTAL_JOBS"
if [[ ${#RETRY_QUEUE[@]} -gt 0 ]]; then
    echo -e "  ${YELLOW}Still blocked:${NC}  ${#RETRY_QUEUE[@]}"
fi
if [[ $OTHER_FAILED -gt 0 ]]; then
    echo -e "  ${RED}Failed:${NC}         $OTHER_FAILED"
fi
if [[ $RETRY_COUNT -gt 0 ]]; then
    echo -e "  ${BLUE}Retry attempts:${NC} $RETRY_COUNT"
fi
echo "========================================"

# List failed jobs if any
if [[ ${#FAILED_JOBS[@]} -gt 0 ]]; then
    echo ""
    log_error "Failed jobs (non-queue errors):"
    for job_entry in "${FAILED_JOBS[@]}"; do
        IFS='|' read -r script_dir script_name <<< "$job_entry"
        echo "  - $script_dir/$script_name"
    done
fi

# List still-blocked jobs if any
if [[ ${#RETRY_QUEUE[@]} -gt 0 ]]; then
    echo ""
    log_warning "Jobs still blocked after $RETRY_COUNT retries:"
    for job_entry in "${RETRY_QUEUE[@]}"; do
        IFS='|' read -r script_dir script_name <<< "$job_entry"
        echo "  - $script_dir/$script_name"
    done
    echo ""
    log "You can re-run this script later to retry these jobs."
fi

# Exit with appropriate code
if [[ ${#RETRY_QUEUE[@]} -gt 0 ]] || [[ ${#FAILED_JOBS[@]} -gt 0 ]]; then
    exit 1
fi
exit 0
