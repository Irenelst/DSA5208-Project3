
#!/bin/bash
# DSA5208 Project 3 - Serial single-run benchmark
#
# Usage:
# bash scripts/serial/run_serial_benchmark.sh \
#   <experiment> <dimension> <N> <iterations> <repeat> <output_csv>

set -euo pipefail

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <experiment> <dimension> <N> <iterations> <repeat> <output_csv>"
    exit 1
fi

EXPERIMENT="$1"
DIMENSION="$2"
N="$3"
ITERATIONS="$4"
REPEAT="$5"
OUTPUT_CSV="$6"

case "$DIMENSION" in
    2D)
        EXECUTABLE="./bin/poisson_2d_serial_vanda"
        ;;
    3D)
        EXECUTABLE="./bin/poisson_3d_serial_vanda"
        ;;
    *)
        echo "ERROR: dimension must be 2D or 3D."
        exit 1
        ;;
esac

for VALUE in "$N" "$ITERATIONS" "$REPEAT"; do
    if ! [[ "$VALUE" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: N, iterations and repeat must be positive integers."
        exit 1
    fi
done

if [ ! -x "$EXECUTABLE" ]; then
    echo "ERROR: executable not found: $EXECUTABLE"
    echo "Compile the serial source code into bin/ first."
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_CSV")"

echo "========================================"
echo "Serial benchmark"
echo "Experiment : $EXPERIMENT"
echo "Dimension  : $DIMENSION"
echo "N          : $N"
echo "Iterations : $ITERATIONS"
echo "Repeat     : $REPEAT"
echo "Output CSV : $OUTPUT_CSV"
echo "========================================"

PROGRAM_OUTPUT="$(
    "$EXECUTABLE" benchmark "$N" "$ITERATIONS"
)"

echo "$PROGRAM_OUTPUT"

CSV_COUNT="$(
    echo "$PROGRAM_OUTPUT" | grep -c '^CSV,' || true
)"

if [ "$CSV_COUNT" -ne 1 ]; then
    echo "ERROR: expected exactly one CSV line."
    exit 1
fi

CSV_LINE="$(
    echo "$PROGRAM_OUTPUT" | grep '^CSV,'
)"

RESULT_FIELDS="$(
    echo "$CSV_LINE" | cut -d',' -f5-
)"

if [ ! -f "$OUTPUT_CSV" ]; then
    echo "experiment,dimension,N,iterations,repeat,total_runtime,time_per_iteration,final_residual" \
        > "$OUTPUT_CSV"
fi

echo "$EXPERIMENT,$DIMENSION,$N,$ITERATIONS,$REPEAT,$RESULT_FIELDS" \
    >> "$OUTPUT_CSV"

echo "Result saved: $OUTPUT_CSV"
