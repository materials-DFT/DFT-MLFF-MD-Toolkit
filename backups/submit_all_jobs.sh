#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <target_directory> [<target_directory> ...]"
    exit 1
fi

ORIG_DIR="$(pwd)"

for TARGET_DIR in "$@"; do
    if [ ! -d "$TARGET_DIR" ]; then
        echo "Error: Directory $TARGET_DIR does not exist."
        exit 1
    fi

    echo "Processing $TARGET_DIR"

    cd "$TARGET_DIR" || exit

    find . -type f -name *submit* | while read -r script; do
        echo "Submitting $script"
        script_dir=$(dirname "$script")
        (cd "$script_dir" && sbatch *submit*)
    done

    cd "$ORIG_DIR" || exit
done
