# DSA5208 Project 3 — Parallelizing a Poisson Equation Solver with MPI+X

## 1. Project Overview

This project investigates the parallelization and performance of Jacobi iterative solvers for the two-dimensional (2D) and three-dimensional (3D) Poisson equations.

Four implementations are developed and evaluated:

1. **Serial C++:** Sequential baseline for numerical validation and runtime scaling.
2. **MPI:** Distributed-memory parallelization using domain decomposition and halo exchange.
3. **MPI + OpenMP:** Hybrid distributed-memory and shared-memory parallelization.
4. **CUDA:** Single-GPU implementations, including additional block-configuration experiments.

The project evaluates numerical correctness, runtime scaling, parallel speedup, efficiency, communication overhead, and CPU/GPU performance trade-offs.

Experiments were conducted on the **NUS Vanda High-Performance Computing (HPC) cluster**. Both 2D and 3D problems are investigated throughout the project.

## 2. Repository Structure

```text
DSA5208-Project3/
├── src/
│   ├── serial/
│   │   ├── poisson_2d_serial_vanda.cpp
│   │   └── poisson_3d_serial_vanda.cpp
│   ├── mpi/
│   │   ├── poisson_2d_mpi_vanda.cpp
│   │   └── poisson_3d_mpi_vanda.cpp
│   ├── hybrid/
│   │   ├── poisson_2d_hybrid.cpp
│   │   └── poisson_3d_hybrid.cpp
│   └── cuda/
│       ├── poisson_2d_cuda.cu
│       ├── poisson_3d_cuda.cu
│       ├── poisson_2d_cuda_block.cu
│       └── poisson_3d_cuda_block.cu
│
├── scripts/
│   ├── serial/
│   │   ├── formal_serial_2d.pbs
│   │   ├── formal_serial_3d.pbs
│   │   ├── run_serial_benchmark.sh
│   │   ├── run_serial_repeats.sh
│   │   └── run_serial_scaling.sh
│   ├── mpi/
│   │   ├── formal_strong_2d_p1.pbs
│   │   ├── run_mpi_benchmark.sh
│   │   ├── run_mpi_repeats.sh
│   │   ├── submit_strong_2d.sh
│   │   ├── submit_strong_3d.sh
│   │   ├── submit_weak_2d.sh
│   │   └── submit_weak_3d.sh
│   ├── hybrid/
│   │   ├── build.sh
│   │   ├── run_config.py
│   │   ├── run_suite.py
│   │   ├── run_72cores.pbs
│   │   ├── run_144cores.pbs
│   │   ├── validate.py
│   │   ├── validation.pbs
│   │   └── summarize_results.py
│   └── cuda/
│       ├── block_benchmark.pbs
│       ├── gpu_benchmark.pbs
│       ├── run_block_benchmarks.py
│       ├── run_gpu_benchmarks.py
│       ├── validate_2d.pbs
│       ├── validate_3d.pbs
│       └── validate_block.pbs
│
└── README.md
```

The `src/` directory contains the numerical implementations. The `scripts/` directory contains the corresponding compilation, validation, benchmarking, and PBS job-submission scripts.

Generated executables, intermediate PBS files, and experimental CSV outputs are not required to be committed to the source repository.

## 3. Software Requirements

### 3.1 HPC Environment

The primary experimental environment is the NUS Vanda HPC cluster, using the PBS job scheduler.

**CPU environment**

- C++17
- Intel oneAPI C++ compiler (`icpx`)
- Intel MPI (`mpiicpx`, `mpirun`)
- OpenMP
- Intel environment module: `intel/2024a`

**GPU environment**

- NVIDIA CUDA C++ compiler (`nvcc`)
- CUDA Toolkit module: `cuda12.4/toolkit/12.4.1`
- NVIDIA GPU compute node
- NVIDIA A40 GPU used in the reported experiments

**Additional requirements**

- Python 3
- Standard Python libraries used by the benchmark scripts
- Bash
- PBS commands, including `qsub`

The reported CPU experiments used Intel oneAPI 2024.2.0 and Intel MPI 2021.13.

The CUDA experiments used NVIDIA CUDA Toolkit 12.4.1 with the `-O3 -std=c++17` compilation options.

### 3.2 Environment Setup

For CPU experiments:

```bash
module purge
module load intel/2024a
```

For GPU experiments, use the CUDA environment specified in the GPU PBS scripts:

```bash
module load cuda12.4/toolkit/12.4.1
```

Modules should be loaded in the appropriate CPU or GPU job environment. The CPU and CUDA commands above do not necessarily need to be executed in the same shell session.

PBS scripts contain project-specific resource requests. Replace the PBS project ID with an account or project authorized to submit jobs on the target HPC system.

All paths in this README are relative to the repository root unless otherwise stated.

## 4. Compilation

Create an output directory for compiled executables:

```bash
mkdir -p bin
```

### 4.1 Serial C++

Load the Intel compiler environment and compile the two serial implementations:

```bash
module load intel/2024a

icpx -O3 -std=c++17 \
  src/serial/poisson_2d_serial_vanda.cpp \
  -o bin/poisson_2d_serial_vanda

icpx -O3 -std=c++17 \
  src/serial/poisson_3d_serial_vanda.cpp \
  -o bin/poisson_3d_serial_vanda
```

### 4.2 MPI

Compile the distributed-memory implementations:

```bash
module load intel/2024a

mpiicpx -O3 -std=c++17 \
  src/mpi/poisson_2d_mpi_vanda.cpp \
  -o bin/poisson_2d_mpi_vanda

mpiicpx -O3 -std=c++17 \
  src/mpi/poisson_3d_mpi_vanda.cpp \
  -o bin/poisson_3d_mpi_vanda
```

### 4.3 MPI + OpenMP

The hybrid implementation uses the Intel MPI C++ compiler with OpenMP support.

Compile the hybrid executables and their MPI reference dependencies using:

```bash
module load intel/2024a
bash scripts/hybrid/build.sh
```

The hybrid build process prepares the executables required by the benchmark and validation scripts, including the 2D and 3D hybrid solvers.

The validation workflow also requires MPI reference executables compiled from the corresponding MPI source implementations.

### 4.4 CUDA

Load the CUDA environment:

```bash
module load cuda12.4/toolkit/12.4.1
mkdir -p bin
```

Compile the four CUDA implementations:

```bash
nvcc -O3 -std=c++17 \
  src/cuda/poisson_2d_cuda.cu \
  -o bin/poisson_2d_cuda

nvcc -O3 -std=c++17 \
  src/cuda/poisson_3d_cuda.cu \
  -o bin/poisson_3d_cuda

nvcc -O3 -std=c++17 \
  src/cuda/poisson_2d_cuda_block.cu \
  -o bin/poisson_2d_cuda_block

nvcc -O3 -std=c++17 \
  src/cuda/poisson_3d_cuda_block.cu \
  -o bin/poisson_3d_cuda_block
```

The CUDA PBS and Python scripts may use implementation-specific executable names or working directories. Ensure these agree with the compiled executable paths before submitting GPU jobs.

## 5. Numerical Validation

Correctness is assessed before performance benchmarking.

The validation process checks numerical convergence, relative residuals, maximum analytical errors, and consistency between implementations.

### 5.1 Serial and MPI Validation

Small-grid tests are used to verify the numerical solver before larger-scale experiments.

The primary validation grid is:

- 2D: N = 16
- 3D: N = 16

MPI validation also compares results across different process counts, including 1, 2, and 4 MPI ranks.

The validation tests check:

- Convergence under the specified stopping criterion
- Final relative residual
- Maximum error against the analytical solution
- Agreement between serial and MPI implementations

The serial and MPI source files support validation and fixed-iteration benchmark modes.

Example validation commands:

```bash
./bin/poisson_2d_serial_vanda validation 16
./bin/poisson_3d_serial_vanda validation 16

mpirun -np 4 ./bin/poisson_2d_mpi_vanda validation 16
mpirun -np 4 ./bin/poisson_3d_mpi_vanda validation 16
```

### 5.2 Hybrid Validation

The hybrid validation procedure tests combinations of MPI ranks and OpenMP threads, including uneven domain decomposition.

Submit the validation job:

```bash
qsub scripts/hybrid/validation.pbs
```

The validation script is:

```text
scripts/hybrid/validate.py
```

It performs fixed-iteration comparisons and convergence checks against the MPI reference implementation and analytical expectations.

The configured validation matrix contains 54 test cases.

Expected output location:

```text
results/hybrid/All_54_Validation_Tests.csv
```

### 5.3 CUDA Validation

CUDA validation is performed using the supplied PBS scripts:

```bash
qsub scripts/cuda/validate_2d.pbs
qsub scripts/cuda/validate_3d.pbs
qsub scripts/cuda/validate_block.pbs
```

These tests cover the 2D and 3D CUDA solvers and the additional block-configuration implementations.

GPU validation requires access to a suitable NVIDIA GPU compute node.

## 6. Benchmark Methodology

All reported formal performance benchmarks use a fixed number of **1,000 Jacobi iterations**.

For each experimental configuration:

- Five independent runs are performed.
- The median runtime is used as the representative performance measurement.
- Fixed-iteration timing is used to compare implementations under consistent computational workloads.
- Numerical validation is conducted separately from the formal performance measurements.

Fixed-iteration runtime should not be interpreted as time to convergence.

The experiments cover serial runtime scaling, MPI strong scaling, MPI weak scaling, MPI+OpenMP performance, and CUDA performance.

## 7. Serial Runtime Scaling

Serial scaling evaluates the increase in computational runtime as the problem size grows while using a single CPU core.

### 7.1 Experiment Configuration

| Dimension | Grid sizes (N) | CPU cores | Iterations | Repeats |
|---|---|---|---|---|
| 2D | 128, 256, 512, 1024, 2048, 4096 | 1 | 1000 | 5 |
| 3D | 32, 64, 128, 256 | 1 | 1000 | 5 |

### 7.2 Reproduction

The relevant scripts are:

```text
scripts/serial/run_serial_benchmark.sh
scripts/serial/run_serial_repeats.sh
scripts/serial/run_serial_scaling.sh
scripts/serial/formal_serial_2d.pbs
scripts/serial/formal_serial_3d.pbs
```

Make the shell scripts executable if required:

```bash
chmod +x scripts/serial/*.sh
```

Submit the formal PBS jobs:

```bash
qsub scripts/serial/formal_serial_2d.pbs
qsub scripts/serial/formal_serial_3d.pbs
```

The benchmark scripts execute the compiled serial solvers and collect runtime measurements.

The results are used to examine computational complexity and runtime growth with grid size.

## 8. MPI Strong and Weak Scaling

MPI experiments use distributed-memory parallelization and halo exchange.

Both 2D and 3D implementations are evaluated.

### 8.1 MPI Process Counts

The tested MPI process counts are:

```text
1, 2, 4, 8, 16, 32, 36, 72, 144, 288
```

One MPI process is assigned per allocated CPU core for the MPI-only experiments.

Each configuration performs 1,000 iterations and is repeated five times.

### 8.2 Strong Scaling

Strong scaling keeps the global problem size fixed while increasing the number of MPI processes.

| Dimension | Global grid size (N) | MPI ranks |
|---|---|---|
| 2D | 8192 | 1–288 |
| 3D | 384 | 1–288 |

The submission scripts are:

```text
scripts/mpi/submit_strong_2d.sh
scripts/mpi/submit_strong_3d.sh
```

Run from the repository root:

```bash
chmod +x scripts/mpi/*.sh

bash scripts/mpi/submit_strong_2d.sh
bash scripts/mpi/submit_strong_3d.sh
```

The 2D single-process strong-scaling case also has a dedicated PBS script. Submit the P=1 baseline experiment separately:

```bash
qsub scripts/mpi/formal_strong_2d_p1.pbs
```

### 8.3 Weak Scaling

Weak scaling increases the global problem size with the number of MPI processes to maintain approximately constant computational work per process.

The grid sizes follow:

**2D:**

```text
N(P) = round(1024 × sqrt(P))
```

**3D:**

```text
N(P) = round(128 × P^(1/3))
```

where P denotes the number of MPI processes.

The submission scripts are:

```text
scripts/mpi/submit_weak_2d.sh
scripts/mpi/submit_weak_3d.sh
```

Run from the repository root:

```bash
bash scripts/mpi/submit_weak_2d.sh
bash scripts/mpi/submit_weak_3d.sh
```

The submission scripts generate PBS job files and submit the requested MPI configurations.

The generated PBS files are intermediate artifacts and do not need to be stored in the repository.

### 8.4 MPI Benchmark Scripts

The MPI workflow uses:

```text
scripts/mpi/run_mpi_benchmark.sh
scripts/mpi/run_mpi_repeats.sh
```

These scripts execute the compiled MPI solvers, repeat the benchmark runs, and collect timing information.

MPI performance analysis includes runtime, speedup, parallel efficiency, computation time, and communication overhead.

## 9. MPI + OpenMP Hybrid Experiments

The hybrid implementation combines MPI domain decomposition with OpenMP threading within each MPI process.

The experiments compare different MPI-rank and OpenMP-thread configurations while keeping the total allocated CPU core count fixed.

### 9.1 Experiment Configuration

The hybrid experiments use the same global problem sizes as MPI strong scaling:

- 2D: N = 8192
- 3D: N = 384

Two resource configurations are tested.

**72 total CPU cores, one node**

| MPI ranks | OpenMP threads per rank | Total CPU cores |
|---|---|---|
| 72 | 1 | 72 |
| 36 | 2 | 72 |
| 18 | 4 | 72 |

**144 total CPU cores, two nodes**

| MPI ranks | OpenMP threads per rank | Total CPU cores |
|---|---|---|
| 144 | 1 | 144 |
| 72 | 2 | 144 |
| 36 | 4 | 144 |

Both 2D and 3D problems are tested.

Each configuration uses 1,000 iterations and five independent runs.

The experiment therefore contains 12 configurations and 60 measured benchmark runs.

### 9.2 Reproduction

First compile the hybrid implementation:

```bash
module load intel/2024a
bash scripts/hybrid/build.sh
```

Submit the benchmark jobs:

```bash
qsub scripts/hybrid/run_72cores.pbs
qsub scripts/hybrid/run_144cores.pbs
```

The PBS scripts invoke:

```text
scripts/hybrid/run_suite.py
```

The shared benchmark configuration is provided by:

```text
scripts/hybrid/run_config.py
```

The experiments use OpenMP binding settings, including:

```bash
export I_MPI_PIN_DOMAIN=omp
export OMP_DYNAMIC=FALSE
export OMP_PLACES=cores
export OMP_PROC_BIND=close
```

These settings help control process and thread placement during hybrid execution.

### 9.3 Hybrid Results Processing

Hybrid results are summarized using:

```text
scripts/hybrid/summarize_results.py
```

The script:

1. Requires five distinct repeats per configuration.
2. Selects the complete run corresponding to the median parallel runtime.
3. Calculates the runtime standard deviation.
4. Computes speedup relative to the matching one-thread-per-rank configuration at the same total CPU core count.

Example usage, after benchmark CSV files have been generated:

```bash
python3 scripts/hybrid/summarize_results.py \
  results/hybrid/New_Runs/*/*.csv
```

Default summary output:

```text
results/hybrid/Median_Runtime_Results.csv
```

The hybrid speedup calculated by this script uses the corresponding hybrid configuration with one OpenMP thread per MPI rank as its reference. It is not automatically equivalent to a comparison against the separately implemented MPI-only solver.

## 10. CUDA GPU Experiments

The CUDA implementation evaluates GPU acceleration for the 2D and 3D Poisson solvers.

Additional experiments investigate the effects of CUDA block configurations.

### 10.1 GPU Environment

The reported GPU experiments used:

- NUS Vanda GPU compute nodes
- One NVIDIA A40 GPU
- CUDA Toolkit 12.4.1
- `nvcc -O3 -std=c++17`

The PBS GPU jobs request the GPU and associated CPU and memory resources required by the benchmark scripts.

### 10.2 CUDA Validation

The CUDA validation scripts are:

```text
scripts/cuda/validate_2d.pbs
scripts/cuda/validate_3d.pbs
scripts/cuda/validate_block.pbs
```

Submit the validation jobs:

```bash
qsub scripts/cuda/validate_2d.pbs
qsub scripts/cuda/validate_3d.pbs
qsub scripts/cuda/validate_block.pbs
```

### 10.3 GPU Benchmarking

The standard CUDA benchmark workflow uses:

```text
scripts/cuda/gpu_benchmark.pbs
scripts/cuda/run_gpu_benchmarks.py
```

The block-configuration experiments use:

```text
scripts/cuda/block_benchmark.pbs
scripts/cuda/run_block_benchmarks.py
```

Submit the benchmark jobs:

```bash
qsub scripts/cuda/gpu_benchmark.pbs
qsub scripts/cuda/block_benchmark.pbs
```

The GPU benchmark suite contains 60 formal runs across 12 configurations, while the block-configuration study contains 50 runs across 10 configurations.

These experiments are used to evaluate GPU runtime, acceleration relative to CPU baselines, and sensitivity to block configuration.

The original GPU benchmark package includes raw-run and median-summary CSV outputs. Such results may be retained separately from the source repository.

**Path note:** The CUDA scripts were reorganized into `scripts/cuda/`, while the CUDA source files are located in `src/cuda/`. Any working-directory or executable-path assumptions inherited from the original standalone GPU package must be checked before rerunning the PBS jobs.

## 11. Performance Metrics and Results Processing

### 11.1 Representative Runtime

Each formal configuration is executed five times.

The reported representative runtime is the median of the five measured runtimes.

For the hybrid experiments, the representative observation is selected as a complete benchmark record rather than calculating an independent median for every timing field.

### 11.2 Strong-Scaling Speedup

Strong-scaling speedup is calculated as:

```text
S(P) = T(1) / T(P)
```

where:

- `T(1)` is the runtime using one MPI process.
- `T(P)` is the runtime using P MPI processes.

### 11.3 Parallel Efficiency

Parallel efficiency is calculated as:

```text
E(P) = S(P) / P
```

Efficiency measures the fraction of ideal linear speedup achieved by the parallel implementation.

### 11.4 Weak Scaling

Weak-scaling analysis compares execution times as the number of processes and global problem size increase together.

The problem size is selected to keep the computational workload per process approximately constant.

### 11.5 Communication Analysis

MPI timing measurements are used to examine the relative costs of computation and communication.

The analysis considers the effects of halo exchange, synchronization, process count, and problem dimensionality.

### 11.6 CPU/GPU Comparison

GPU performance is compared against the relevant CPU implementations using compatible problem sizes and benchmark definitions.

These comparisons account for differences in hardware allocation and parallel execution models.

### 11.7 Result Reproduction

The benchmark scripts generate runtime measurements and, where supported, machine-readable CSV output.

The hybrid implementation includes a dedicated summary script. Serial, MPI, and CUDA results can be summarized from their corresponding benchmark outputs using the statistical definitions above.

The repository focuses on reproducing the main numerical experiments. A separate script for regenerating every report figure is not included.

For complete numerical findings, plots, and interpretation, refer to the accompanying project report.

## 12. Reproducibility Notes

To reproduce the experiments:

1. Obtain access to NUS Vanda or an HPC environment with compatible compilers, MPI, CUDA, and PBS scheduling.
2. Clone the repository.
3. Load the appropriate CPU or GPU software modules.
4. Compile the required implementations.
5. Run the validation tests before formal benchmarking.
6. Submit the relevant PBS scripts.
7. Collect benchmark outputs and calculate the reported performance metrics.

Clone the repository using:

```bash
git clone https://github.com/IreneIst/DSA5208-Project3.git
cd DSA5208-Project3
```

If necessary, enable execution permissions for shell scripts:

```bash
chmod +x scripts/serial/*.sh
chmod +x scripts/mpi/*.sh
chmod +x scripts/hybrid/*.sh
```

The scripts are designed around the original NUS Vanda environment. Reproduction on another HPC cluster may require changes to:

- PBS project identifiers and resource requests
- Environment module names
- MPI launch configuration
- CPU and GPU hardware allocation
- Executable paths and working directories

Performance measurements may differ across systems due to hardware characteristics, memory bandwidth, MPI communication behavior, resource placement, and runtime variability.

The reported results use fixed-iteration timing and should not be interpreted as a universal comparison of convergence speed.

## 13. Submission Contents

This repository provides:

- Serial C++ implementations for 2D and 3D Poisson equations
- MPI implementations for 2D and 3D Poisson equations
- MPI + OpenMP hybrid implementations
- CUDA GPU implementations and block-configuration variants
- Compilation and validation scripts
- Serial and MPI scaling experiment scripts
- Hybrid benchmark and result-summary scripts
- CUDA validation and benchmark scripts
- Instructions for reproducing the main experiments

Compiled binaries, temporary PBS job files, debugging logs, and terminal output are excluded from the source submission.

The accompanying project report contains the detailed numerical results, performance plots, discussion, limitations, and conclusions.

---

**Course:** DSA5208 — Scalable Distributed Computing for Data Science

**Project:** Project 3 — Parallelizing an Algorithm with MPI+X

**Platform:** NUS Vanda HPC Cluster

**Algorithm:** Jacobi iterative solver for 2D and 3D Poisson equations
