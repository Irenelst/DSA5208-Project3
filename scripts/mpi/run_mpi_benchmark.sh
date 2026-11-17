
#!/bin/bash

# ============================================================
# DSA5208 Project 3
# MPI Benchmark Runner
#
# Usage:
#   bash scripts/run_mpi_benchmark.sh \
#       <experiment> <dimension> <N> <iterations> \
#       <mpi_ranks> <repeat> <output_csv>
#
# Example:
#   bash scripts/run_mpi_benchmark.sh \
#       pilot 2D 512 1000 4 1 results/test_2d.csv
# ============================================================

set -e

# ------------------------------------------------------------
# 1. Check arguments
# ------------------------------------------------------------

if [ "$#" -ne 7 ]; then
    echo "Usage:"
    echo "  $0 <experiment> <dimension> <N> <iterations> <mpi_ranks> <repeat> <output_csv>"
    exit 1
fi

# ------------------------------------------------------------
# 2. Read arguments
# ------------------------------------------------------------

EXPERIMENT="$1"
DIMENSION="$2"
N="$3"
ITERATIONS="$4"
MPI_RANKS="$5"
REPEAT="$6"
OUTPUT_CSV="$7"

# ------------------------------------------------------------
# 3. Select executable
#
# Source files:
#   src/mpi/poisson_2d_mpi_vanda.cpp
#   src/mpi/poisson_3d_mpi_vanda.cpp
#
# Compiled executables:
#   bin/poisson_2d_mpi_vanda
#   bin/poisson_3d_mpi_vanda
# ------------------------------------------------------------

if [ "$DIMENSION" = "2D" ]; then

    EXECUTABLE="./bin/poisson_2d_mpi_vanda"

elif [ "$DIMENSION" = "3D" ]; then

    EXECUTABLE="./bin/poisson_3d_mpi_vanda"

else

    echo "Error: dimension must be 2D or 3D."
    exit 1

fi

# ------------------------------------------------------------
# 4. Check executable
# ------------------------------------------------------------

if [ ! -x "$EXECUTABLE" ]; then
    echo "Error: executable not found or not executable:"
    echo "  $EXECUTABLE"
    echo
    echo "Compile the MPI source code into bin/ first."
    exit 1
fi

# ------------------------------------------------------------
# 5. Create output directory
# ------------------------------------------------------------

OUTPUT_DIR=$(dirname "$OUTPUT_CSV")
mkdir -p "$OUTPUT_DIR"

# ------------------------------------------------------------
# 6. Write CSV header if output file does not exist
# ------------------------------------------------------------

if [ ! -f "$OUTPUT_CSV" ]; then

    echo "experiment,dimension,N,iterations,mpi_ranks,repeat,critical_rank,parallel_runtime,critical_comm_time,critical_compute_time,communication_fraction,min_rank_runtime,mean_rank_runtime,max_rank_runtime,rank_imbalance,final_residual,final_max_error" \
        > "$OUTPUT_CSV"

fi

# ------------------------------------------------------------
# 7. Run MPI benchmark
# ------------------------------------------------------------

echo "========================================"
echo "MPI Benchmark Runner"
echo "========================================"
echo "Experiment : $EXPERIMENT"
echo "Dimension  : $DIMENSION"
echo "N          : $N"
echo "Iterations : $ITERATIONS"
echo "MPI ranks  : $MPI_RANKS"
echo "Repeat     : $REPEAT"
echo "Output CSV : $OUTPUT_CSV"
echo "========================================"

PROGRAM_OUTPUT=$(
    mpirun -np "$MPI_RANKS" \
        "$EXECUTABLE" \
        benchmark \
        "$N" \
        "$ITERATIONS"
)

# Show solver output
echo "$PROGRAM_OUTPUT"

# ------------------------------------------------------------
# 8. Extract machine-readable CSV line
# ------------------------------------------------------------

SOLVER_CSV=$(
    echo "$PROGRAM_OUTPUT" |
    grep '^CSV,'
)

if [ -z "$SOLVER_CSV" ]; then
    echo "Error: no CSV output was produced by the solver."
    exit 1
fi

# Remove:
# CSV,dimension,N,iterations,mpi_ranks,
#
# Retain all fields from critical_rank onward.

SOLVER_DATA=$(
    echo "$SOLVER_CSV" |
    cut -d',' -f6-
)

# ------------------------------------------------------------
# 9. Append result to CSV
# ------------------------------------------------------------

echo \
"$EXPERIMENT,$DIMENSION,$N,$ITERATIONS,$MPI_RANKS,$REPEAT,$SOLVER_DATA" \
>> "$OUTPUT_CSV"

echo
echo "========================================"
echo "Result saved successfully"
echo "========================================"
echo "$OUTPUT_CSV"
