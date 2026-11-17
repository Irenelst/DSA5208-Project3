
#!/bin/bash

# ============================================================
# DSA5208 Project 3
# Serial Benchmark Runner
#
# Run ONE serial benchmark configuration for ONE repeat
# and append the result to a CSV file.
#
# Usage:
#   bash scripts/run_serial_benchmark.sh \
#       <experiment> <dimension> <N> <iterations> \
#       <repeat> <output_csv>
#
# Example:
#   bash scripts/run_serial_benchmark.sh \
#       pilot 2D 512 1000 1 results/test_serial_2d.csv
# ============================================================

set -euo pipefail

# ------------------------------------------------------------
# 1. Check arguments
# ------------------------------------------------------------

if [ "$#" -ne 6 ]; then
    echo "Usage:"
    echo "  $0 <experiment> <dimension> <N> <iterations> <repeat> <output_csv>"
    exit 1
fi

# ------------------------------------------------------------
# 2. Read arguments
# ------------------------------------------------------------

EXPERIMENT="$1"
DIMENSION="$2"
N="$3"
ITERATIONS="$4"
REPEAT="$5"
OUTPUT_CSV="$6"

# ------------------------------------------------------------
# 3. Validate dimension and select executable
#
# Source files:
#   src/serial/poisson_2d_serial_vanda.cpp
#   src/serial/poisson_3d_serial_vanda.cpp
#
# Compiled executables:
#   bin/poisson_2d_serial_vanda
#   bin/poisson_3d_serial_vanda
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# 4. Validate integer arguments
# ------------------------------------------------------------

if ! [[ "$N" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: N must be a positive integer."
    exit 1
fi

if ! [[ "$ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: iterations must be a positive integer."
    exit 1
fi

if ! [[ "$REPEAT" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: repeat must be a positive integer."
    exit 1
fi

# ------------------------------------------------------------
# 5. Check executable
# ------------------------------------------------------------

if [ ! -x "$EXECUTABLE" ]; then
    echo "ERROR: executable not found or not executable:"
    echo "  $EXECUTABLE"
    echo
    echo "Compile the serial source code into bin/ first."
    exit 1
fi

# ------------------------------------------------------------
# 6. Create output directory if necessary
# ------------------------------------------------------------

OUTPUT_DIR="$(dirname "$OUTPUT_CSV")"
mkdir -p "$OUTPUT_DIR"

# ------------------------------------------------------------
# 7. Create CSV header if output file does not exist
# ------------------------------------------------------------

if [ ! -f "$OUTPUT_CSV" ]; then

    echo \
"experiment,dimension,N,iterations,repeat,total_runtime,time_per_iteration,final_residual" \
    > "$OUTPUT_CSV"

fi

# ------------------------------------------------------------
# 8. Display benchmark configuration
# ------------------------------------------------------------

echo "============================================================"
echo "Serial benchmark"
echo "------------------------------------------------------------"
echo "Experiment : $EXPERIMENT"
echo "Dimension  : $DIMENSION"
echo "N          : $N"
echo "Iterations : $ITERATIONS"
echo "Repeat     : $REPEAT"
echo "Output CSV : $OUTPUT_CSV"
echo "============================================================"

# ------------------------------------------------------------
# 9. Run serial benchmark
# ------------------------------------------------------------

PROGRAM_OUTPUT="$(
    "$EXECUTABLE" benchmark "$N" "$ITERATIONS"
)"

# Display full solver output
echo "$PROGRAM_OUTPUT"

# ------------------------------------------------------------
# 10. Extract machine-readable CSV line
#
# Expected solver output:
# CSV,dimension,N,K,total_runtime,time_per_iteration,final_residual
# ------------------------------------------------------------

CSV_LINE="$(
    echo "$PROGRAM_OUTPUT" |
    grep '^CSV,'
)"

# ------------------------------------------------------------
# 11. Verify exactly one CSV line was produced
# ------------------------------------------------------------

CSV_COUNT="$(
    echo "$PROGRAM_OUTPUT" |
    grep -c '^CSV,' || true
)"

if [ "$CSV_COUNT" -ne 1 ]; then
    echo "ERROR: expected exactly one CSV line from solver."
    echo "Found: $CSV_COUNT"
    exit 1
fi

# ------------------------------------------------------------
# 12. Extract result fields
#
# Remove:
# CSV,dimension,N,K
#
# Keep:
# total_runtime,time_per_iteration,final_residual
# ------------------------------------------------------------

RESULT_FIELDS="$(
    echo "$CSV_LINE" |
    cut -d',' -f5-
)"

# ------------------------------------------------------------
# 13. Append result to CSV
# ------------------------------------------------------------

echo \
"$EXPERIMENT,$DIMENSION,$N,$ITERATIONS,$REPEAT,$RESULT_FIELDS" \
>> "$OUTPUT_CSV"

echo
echo "Result appended successfully:"
echo "  $OUTPUT_CSV"
