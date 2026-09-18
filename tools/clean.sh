#!/usr/bin/env bash
# Remove only generated experimental artifacts from this repository.
# Usage: bash tools/clean.sh [--dry-run]

set -euo pipefail

if [[ $# -gt 1 || (${1:-} != "" && ${1:-} != "--dry-run") ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

dry_run=false
if [[ ${1:-} == "--dry-run" ]]; then
    dry_run=true
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/.." && pwd -P)"
output_dir="$project_root/outputs"

# Guard against an unexpected script location before any deletion is attempted.
case "$output_dir" in
    "$project_root/outputs") ;;
    *)
        echo "Refusing to clean an unexpected path: $output_dir" >&2
        exit 1
        ;;
esac

if [[ ! -d "$output_dir" ]]; then
    echo "No outputs directory found: $output_dir"
    exit 0
fi

shopt -s nullglob
artifacts=(
    "$output_dir"/history_*.json
    "$output_dir"/model_*.pt
    "$output_dir"/figure
)

existing=()
for artifact in "${artifacts[@]}"; do
    [[ -e "$artifact" || -L "$artifact" ]] && existing+=("$artifact")
done

if (( ${#existing[@]} == 0 )); then
    echo "No generated artifacts to remove from $output_dir"
    exit 0
fi

printf '%s generated artifact(s) under %s:
' "${#existing[@]}" "$output_dir"
printf '  %s
' "${existing[@]##$project_root/}"

if "$dry_run"; then
    echo "Dry run: nothing removed."
    exit 0
fi

for artifact in "${existing[@]}"; do
    rm -rf -- "$artifact"
done

echo "Generated artifacts removed."
