#!/bin/bash

# ============================================================
# DSA5208 Project 3
# Submit 3D MPI Weak Scaling Formal Experiments
#
# Base problem:
#   N1 = 128 at P = 1
#   iterations = 1000
#   repeats = 5
#
# Weak scaling:
#   N(P) = round(N1 * P^(1/3))
#
# P = 1, 2, 4, 8, 16, 32, 36, 72, 144, 288
# ============================================================

set -e

PROJECT="personal-e0869267"

N1=128
ITERATIONS=1000
REPEATS=5

P_VALUES=(1 2 4 8 16 32 36 72 144 288)

mkdir -p scripts/generated_pbs
mkdir -p results


for P in "${P_VALUES[@]}"
do

    # --------------------------------------------------------
    # Calculate N for 3D weak scaling:
    #
    # N(P) = round(N1 * P^(1/3))
    #
    # This keeps approximately:
    #
    # N(P)^3 / P = constant
    # --------------------------------------------------------

    N=$(python3 -c "print(round(${N1} * (${P} ** (1.0 / 3.0))))")

    PBS_FILE="scripts/generated_pbs/weak_3d_p${P}.pbs"
    OUTPUT_CSV="results/weak_3d_p${P}.csv"


    # --------------------------------------------------------
    # Determine PBS resource allocation
    # --------------------------------------------------------

    if [ "$P" -le 72 ]; then

        SELECT_LINE="select=1:ncpus=${P}:mpiprocs=${P}:ompthreads=1:mem=6gb"

    elif [ "$P" -eq 144 ]; then

        SELECT_LINE="select=2:ncpus=72:mpiprocs=72:ompthreads=1:mem=480gb"

    elif [ "$P" -eq 288 ]; then

        SELECT_LINE="select=4:ncpus=72:mpiprocs=72:ompthreads=1:mem=480gb"

    else

        echo "ERROR: unsupported MPI rank count P=${P}"
        exit 1

    fi


    # --------------------------------------------------------
    # Generate PBS file
    # --------------------------------------------------------

    cat > "$PBS_FILE" <<EOF
#!/bin/bash
#PBS -N weak3d_p${P}
#PBS -P ${PROJECT}
#PBS -j oe
#PBS -k oed
#PBS -l walltime=00:10:00
#PBS -l ${SELECT_LINE}

cd \$PBS_O_WORKDIR

module purge
module load intel/2024a

echo "===== 3D MPI Weak Scaling Formal Experiment ====="
echo "P = ${P}"
echo "N = ${N}"
echo "N^3/P = \$(( ${N} * ${N} * ${N} / ${P} ))"
echo "Hostname: \$(hostname)"
echo "Date: \$(date)"
echo

mkdir -p results

bash scripts/run_mpi_repeats.sh \\
    weak \\
    3D \\
    ${N} \\
    ${ITERATIONS} \\
    ${P} \\
    ${REPEATS} \\
    ${OUTPUT_CSV}
EOF


    # --------------------------------------------------------
    # Submit PBS job
    # --------------------------------------------------------

    echo
    echo "Submitting P=${P}, N=${N}"
    echo "PBS file   : ${PBS_FILE}"
    echo "Output CSV : ${OUTPUT_CSV}"

    qsub "$PBS_FILE"

done


echo
echo "========================================"
echo "All 3D weak-scaling jobs submitted."
echo "========================================"
