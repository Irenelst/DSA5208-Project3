
#!/bin/bash
# DSA5208 Project 3 - Serial scaling experiments
#
# Usage:
# bash scripts/serial/run_serial_scaling.sh \
#   <dimension> <iterations> <repeats> <output_csv> <N1> [N2 ...]

set -euo pipefail

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <dimension> <iterations> <repeats> <output_csv> <N1> [N2 ...]"
    exit 1
fi

DIMENSION="$1"
ITERATIONS="$2"
REPEATS="$3"
OUTPUT_CSV="$4"

shift 4

if [ "$DIMENSION" != "2D" ] &&
   [ "$DIMENSION" != "3D" ]; then
    echo "ERROR: dimension must be 2D or 3D."
    exit 1
fi

for VALUE in "$ITERATIONS" "$REPEATS" "$@"; do
    if ! [[ "$VALUE" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: all numeric arguments must be positive integers."
        exit 1
    fi
done

RUNNER="scripts/serial/run_serial_repeats.sh"

if [ ! -f "$RUNNER" ]; then
    echo "ERROR: runner not found: $RUNNER"
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_CSV")"

echo "========================================"
echo "Serial scaling experiment"
echo "Dimension  : $DIMENSION"
echo "Iterations : $ITERATIONS"
echo "Repeats    : $REPEATS"
echo "Grid sizes : $*"
echo "Output CSV : $OUTPUT_CSV"
echo "========================================"

for N in "$@"; do
    echo
    echo "Running $DIMENSION, N=$N"

    bash "$RUNNER" \
        scaling \
        "$DIMENSION" \
        "$N" \
        "$ITERATIONS" \
        "$REPEATS" \
        "$OUTPUT_CSV"
done

echo "Serial scaling experiments completed."
