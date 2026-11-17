#!/bin/bash

# ============================================================
# run_serial_repeats.sh
#
# Run the same serial benchmark configuration multiple times.
#
# Usage:
#
#   ./scripts/run_serial_repeats.sh \
#       <experiment> \
#       <dimension> \
#       <N> \
#       <iterations> \
#       <repeats> \
#       <output_csv>
#
# Example:
#
#   ./scripts/run_serial_repeats.sh \
#       pilot 3D 128 1000 5 results/test_serial_3d_repeats.csv
#
# ============================================================

set -euo pipefail


# ------------------------------------------------------------
# Check arguments
# ------------------------------------------------------------

if [ "$#" -ne 6 ]; then
    echo "Usage:"
    echo "  $0 <experiment> <dimension> <N> <iterations> <repeats> <output_csv>"
    exit 1
fi


EXPERIMENT="$1"
DIMENSION="$2"
N="$3"
ITERATIONS="$4"
REPEATS="$5"
OUTPUT_CSV="$6"


# ------------------------------------------------------------
# Validate dimension
# ------------------------------------------------------------

if [ "$DIMENSION" != "2D" ] &&
   [ "$DIMENSION" != "3D" ]; then

    echo "ERROR: dimension must be 2D or 3D."
    exit 1
fi


# ------------------------------------------------------------
# Validate integer arguments
# ------------------------------------------------------------

if ! [[ "$N" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: N must be a positive integer."
    exit 1
fi

if ! [[ "$ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: iterations must be a positive integer."
    exit 1
fi

if ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: repeats must be a positive integer."
    exit 1
fi


# ------------------------------------------------------------
# Check Level-1 runner
# ------------------------------------------------------------

RUNNER="./scripts/run_serial_benchmark.sh"

if [ ! -x "$RUNNER" ]; then
    echo "ERROR: benchmark runner not found or not executable:"
    echo "  $RUNNER"
    exit 1
fi


# ------------------------------------------------------------
# Display configuration
# ------------------------------------------------------------

echo "============================================================"
echo "Serial repeated benchmark"
echo "------------------------------------------------------------"
echo "Experiment : $EXPERIMENT"
echo "Dimension  : $DIMENSION"
echo "N          : $N"
echo "Iterations : $ITERATIONS"
echo "Repeats    : $REPEATS"
echo "Output CSV : $OUTPUT_CSV"
echo "============================================================"


# ------------------------------------------------------------
# Run repeats
# ------------------------------------------------------------

for (( REPEAT=1; REPEAT<=REPEATS; REPEAT++ ))
do
    echo
    echo "############################################################"
    echo "Starting repeat $REPEAT / $REPEATS"
    echo "############################################################"
    echo

    "$RUNNER" \
        "$EXPERIMENT" \
        "$DIMENSION" \
        "$N" \
        "$ITERATIONS" \
        "$REPEAT" \
        "$OUTPUT_CSV"
done


# ------------------------------------------------------------
# Finished
# ------------------------------------------------------------

echo
echo "============================================================"
echo "All serial repeats completed successfully."
echo "Results:"
echo "  $OUTPUT_CSV"
echo "============================================================"