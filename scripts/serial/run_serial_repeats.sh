
#!/bin/bash
# DSA5208 Project 3 - Serial repeated benchmark
#
# Usage:
# bash scripts/serial/run_serial_repeats.sh \
#   <experiment> <dimension> <N> <iterations> <repeats> <output_csv>

set -euo pipefail

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <experiment> <dimension> <N> <iterations> <repeats> <output_csv>"
    exit 1
fi

EXPERIMENT="$1"
DIMENSION="$2"
N="$3"
ITERATIONS="$4"
REPEATS="$5"
OUTPUT_CSV="$6"

if [ "$DIMENSION" != "2D" ] &&
   [ "$DIMENSION" != "3D" ]; then
    echo "ERROR: dimension must be 2D or 3D."
    exit 1
fi

for VALUE in "$N" "$ITERATIONS" "$REPEATS"; do
    if ! [[ "$VALUE" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: N, iterations and repeats must be positive integers."
        exit 1
    fi
done

RUNNER="scripts/serial/run_serial_benchmark.sh"

if [ ! -f "$RUNNER" ]; then
    echo "ERROR: runner not found: $RUNNER"
    exit 1
fi

echo "========================================"
echo "Serial repeated benchmark"
echo "Experiment : $EXPERIMENT"
echo "Dimension  : $DIMENSION"
echo "N          : $N"
echo "Iterations : $ITERATIONS"
echo "Repeats    : $REPEATS"
echo "Output CSV : $OUTPUT_CSV"
echo "========================================"

for ((REPEAT=1; REPEAT<=REPEATS; REPEAT++)); do
    echo "Starting repeat $REPEAT / $REPEATS"

    bash "$RUNNER" \
        "$EXPERIMENT" \
        "$DIMENSION" \
        "$N" \
        "$ITERATIONS" \
        "$REPEAT" \
        "$OUTPUT_CSV"
done

echo "All serial repeats completed."
echo "Results: $OUTPUT_CSV"
