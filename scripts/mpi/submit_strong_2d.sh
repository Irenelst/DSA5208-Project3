
#!/bin/bash

# ============================================================
# DSA5208 Project 3
# Submit 2D MPI Strong Scaling Formal Experiments
#
# Fixed problem:
#   N = 8192
#   iterations = 1000
#   repeats = 5
#
# P = 2, 4, 8, 16, 32, 36, 72, 144, 288
#
# P=1 has already been completed separately.
# ============================================================

set -e

PROJECT="personal-e0869267"
N=8192
ITERATIONS=1000
REPEATS=5

P_VALUES=(2 4 8 16 32 36 72 144 288)

mkdir -p scripts/mpi/generated_pbs
mkdir -p results


for P in "${P_VALUES[@]}"
do

    PBS_FILE="scripts/mpi/generated_pbs/strong_2d_p${P}.pbs"
    OUTPUT_CSV="results/strong_2d_p${P}.csv"

    # --------------------------------------------------------
    # Determine PBS resource allocation
    #
    # Vanda compute nodes provide up to 72 CPU cores per node.
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
#PBS -N strong2d_p${P}
#PBS -P ${PROJECT}
#PBS -j oe
#PBS -k oed
#PBS -l walltime=00:10:00
#PBS -l ${SELECT_LINE}

cd \$PBS_O_WORKDIR

module purge
module load intel/2024a

echo "===== 2D MPI Strong Scaling Formal Experiment: P=${P} ====="
echo "Hostname: \$(hostname)"
echo "Date: \$(date)"
echo

mkdir -p results

bash scripts/mpi/run_mpi_repeats.sh \\
    strong \\
    2D \\
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
    echo "Submitting P=${P}"
    echo "PBS file   : ${PBS_FILE}"
    echo "Output CSV : ${OUTPUT_CSV}"

    qsub "$PBS_FILE"

done


echo
echo "========================================"
echo "All 2D strong-scaling jobs submitted."
echo "========================================"
