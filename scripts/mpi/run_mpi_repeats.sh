
#!/bin/bash

# ============================================================
# DSA5208 Project 3
# MPI Repeated Benchmark Runner
#
# Runs the same MPI benchmark multiple times.
#
# Usage:
#
#   bash scripts/mpi/run_mpi_repeats.sh \
#       <experiment> \
#       <dimension> \
#       <N> \
#       <iterations> \
#       <mpi_ranks> \
#       <repeats> \
#       <output_csv>
#
# Example:
#
#   bash scripts/mpi/run_mpi_repeats.sh \
#       pilot 2D 512 1000 4 5 \
#       results/pilot_2d.csv
#
# ============================================================

set -e


# ============================================================
# Check arguments
# ============================================================

if [ "$#" -ne 7 ]; then
    echo "Usage:"
    echo "  $0 <experiment> <dimension> <N> <iterations> <mpi_ranks> <repeats> <output_csv>"
    echo
    echo "Example:"
    echo "  $0 pilot 2D 512 1000 4 5 results/pilot_2d.csv"
    exit 1
fi


# ============================================================
# Read arguments
# ============================================================

EXPERIMENT="$1"
DIMENSION="$2"
N="$3"
ITERATIONS="$4"
MPI_RANKS="$5"
REPEATS="$6"
OUTPUT_CSV="$7"


# ============================================================
# Check repeats
# ============================================================

if ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: repeats must be a positive integer."
    exit 1
fi


# ============================================================
# Locate single-run benchmark script
# ============================================================

RUNNER="./scripts/mpi/run_mpi_benchmark.sh"

if [ ! -f "$RUNNER" ]; then
    echo "Error: benchmark runner not found or not executable:"
    echo "  $RUNNER"
    exit 1
fi


# ============================================================
# Display experiment information
# ============================================================

echo
echo "========================================"
echo "MPI Repeated Benchmark"
echo "========================================"
echo "Experiment : $EXPERIMENT"
echo "Dimension  : $DIMENSION"
echo "N          : $N"
echo "Iterations : $ITERATIONS"
echo "MPI ranks  : $MPI_RANKS"
echo "Repeats    : $REPEATS"
echo "Output CSV : $OUTPUT_CSV"
echo "========================================"


# ============================================================
# Run repeated benchmarks
# ============================================================

for ((REPEAT=1; REPEAT<=REPEATS; REPEAT++))
do

    echo
    echo "########################################"
    echo "Starting repeat $REPEAT / $REPEATS"
    echo "########################################"
    echo

    bash "$RUNNER" \
        "$EXPERIMENT" \
        "$DIMENSION" \
        "$N" \
        "$ITERATIONS" \
        "$MPI_RANKS" \
        "$REPEAT" \
        "$OUTPUT_CSV"

done


# ============================================================
# Finished
# ============================================================

echo
echo "========================================"
echo "All repeats completed successfully"
echo "========================================"
echo "Results saved to:"
echo "$OUTPUT_CSV"

