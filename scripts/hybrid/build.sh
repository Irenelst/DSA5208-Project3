
#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root regardless of the calling directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$ROOT_DIR"
mkdir -p bin

# Intel oneAPI toolchain on NUS Vanda.
# GCC/OpenMPI alternative:
# MPICXX=mpicxx OMP_FLAG=-fopenmp bash scripts/hybrid/build.sh
MPICXX="${MPICXX:-mpiicpx}"
OMP_FLAG="${OMP_FLAG:--qopenmp}"

echo "========================================"
echo "Building 2D and 3D Hybrid Poisson solvers"
echo "========================================"

for d in 2 3; do
    echo "Compiling ${d}D MPI+OpenMP solver..."

    "$MPICXX" -O3 -std=c++17 "$OMP_FLAG" \
        "src/hybrid/poisson_${d}d_hybrid.cpp" \
        -o "bin/poisson_${d}d_hybrid"

    echo "Compiling ${d}D MPI reference solver..."

    "$MPICXX" -O3 -std=c++17 "$OMP_FLAG" \
        "src/mpi/poisson_${d}d_mpi_vanda.cpp" \
        -o "bin/poisson_${d}d_mpi_reference"
done

echo "========================================"
echo "Build completed successfully."
echo "Executables are available in bin/"
echo "========================================"
