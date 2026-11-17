# DSA5208 Project 3 — Parallelizing a Poisson Equation Solver with MPI+X

## 1. Project Overview

This project investigates the parallelization of Jacobi iterative solvers for the 2D and 3D Poisson equations using four implementations:

- **Serial C++:** Sequential baseline
- **MPI:** Distributed-memory parallelization with domain decomposition and halo exchange
- **MPI + OpenMP:** Hybrid distributed- and shared-memory parallelization
- **CUDA:** GPU acceleration and block-configuration experiments

Experiments were conducted on the **NUS Vanda HPC cluster** to evaluate numerical correctness, runtime, strong and weak scaling, speedup, efficiency, and communication overhead.

All formal benchmarks use **1,000 fixed Jacobi iterations**, with **five independent runs per configuration**. The median runtime is used for performance comparisons.

## 2. Repository Structure

```text
DSA5208-Project3/
├── src/
│   ├── serial/       # Serial C++ solvers (2D and 3D)
│   ├── mpi/          # MPI solvers (2D and 3D)
│   ├── hybrid/       # MPI + OpenMP solvers (2D and 3D)
│   └── cuda/         # CUDA solvers and block variants
├── scripts/
│   ├── serial/       # Serial benchmark and PBS scripts
│   ├── mpi/          # MPI strong/weak scaling scripts
│   ├── hybrid/       # Hybrid build, validation, and benchmarks
│   └── cuda/         # GPU validation and benchmark scripts
└── README.md
```

Source files are stored in `src/`, while compilation, validation, benchmarking, and PBS submission scripts are organized under `scripts/`.

Generated executables and experimental results are stored locally in `bin/` and `results/`, respectively.

## 3. Software Requirements

Experiments were performed on NUS Vanda using the PBS job scheduler.

| Component | Software / Environment |
|---|---|
| Language | C++17, Python 3, Bash |
| CPU compiler | Intel oneAPI `icpx` |
| MPI | Intel MPI (`mpiicpx`, `mpirun`) |
| Shared-memory parallelism | OpenMP |
| CPU module | `intel/2024a` |
| GPU compiler | NVIDIA `nvcc` |
| CUDA module | `cuda12.4/toolkit/12.4.1` |
| GPU hardware | NVIDIA A40 |
| Job scheduler | PBS (`qsub`) |

CPU experiments used Intel oneAPI 2024.2.0 and Intel MPI 2021.13.

Before submitting PBS jobs, replace project-specific PBS account identifiers with an authorized account if necessary.

**All commands below should be executed from the repository root**, unless otherwise stated.

## 4. Compilation

Clone the repository and create a directory for executables:

```bash
git clone https://github.com/Irenelst/DSA5208-Project3.git
cd DSA5208-Project3
mkdir -p bin
```

### 4.1 Serial

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

The hybrid build script compiles both hybrid solvers and their MPI reference executables.

```bash
module load intel/2024a
bash scripts/hybrid/build.sh
```

### 4.4 CUDA

```bash
module load cuda12.4/toolkit/12.4.1

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

CPU and GPU compilation should be performed in their respective software environments.

## 5. Numerical Validation

Numerical correctness is assessed before performance benchmarking using convergence, relative residuals, analytical errors, and agreement between implementations.

### Serial and MPI

The primary validation grid is N = 16 for both 2D and 3D. MPI validation includes process counts of 1, 2, and 4.

Example commands:

```bash
./bin/poisson_2d_serial_vanda validation 16
./bin/poisson_3d_serial_vanda validation 16

mpirun -np 4 ./bin/poisson_2d_mpi_vanda validation 16
mpirun -np 4 ./bin/poisson_3d_mpi_vanda validation 16
```

### MPI + OpenMP

Submit the hybrid validation job:

```bash
qsub scripts/hybrid/validation.pbs
```

The validation workflow uses `scripts/hybrid/validate.py` and covers 54 test cases.

### CUDA

Submit the GPU validation jobs:

```bash
qsub scripts/cuda/validate_2d.pbs
qsub scripts/cuda/validate_3d.pbs
qsub scripts/cuda/validate_block.pbs
```

These jobs require a compatible NVIDIA GPU compute node.

## 6. Reproducing the Main Experiments

All formal experiments use 1,000 fixed iterations and five repeated runs per configuration.

### 6.1 Serial Runtime Scaling

| Dimension | Grid sizes (N) |
|---|---|
| 2D | 128, 256, 512, 1024, 2048, 4096 |
| 3D | 32, 64, 128, 256 |

Submit the formal experiments:

```bash
qsub scripts/serial/formal_serial_2d.pbs
qsub scripts/serial/formal_serial_3d.pbs
```

The PBS jobs invoke the serial scaling, repeat, and benchmark scripts under `scripts/serial/`.

### 6.2 MPI Strong Scaling

Strong scaling uses fixed global grid sizes:

| Dimension | Grid size (N) | MPI ranks |
|---|---|---|
| 2D | 8192 | 1, 2, 4, 8, 16, 32, 36, 72, 144, 288 |
| 3D | 384 | 1, 2, 4, 8, 16, 32, 36, 72, 144, 288 |

Submit the experiments:

```bash
bash scripts/mpi/submit_strong_2d.sh
bash scripts/mpi/submit_strong_3d.sh
```

The 2D P=1 baseline uses a separate PBS script:

```bash
qsub scripts/mpi/formal_strong_2d_p1.pbs
```

### 6.3 MPI Weak Scaling

Weak scaling adjusts the global grid size according to the number of MPI processes P:

- **2D:** N(P) = round(1024 × sqrt(P))
- **3D:** N(P) = round(128 × P^(1/3))

Submit the experiments:

```bash
bash scripts/mpi/submit_weak_2d.sh
bash scripts/mpi/submit_weak_3d.sh
```

The MPI submission scripts generate PBS files and invoke `run_mpi_repeats.sh` and `run_mpi_benchmark.sh` to collect timing measurements.

### 6.4 MPI + OpenMP Hybrid

Hybrid experiments use the same fixed global grid sizes as MPI strong scaling: N = 8192 (2D) and N = 384 (3D).

| Total CPU cores | MPI ranks × OpenMP threads |
|---|---|
| 72 | 72×1, 36×2, 18×4 |
| 144 | 144×1, 72×2, 36×4 |

Submit the benchmark jobs:

```bash
qsub scripts/hybrid/run_72cores.pbs
qsub scripts/hybrid/run_144cores.pbs
```

The workflow uses `scripts/hybrid/run_suite.py` and `run_config.py`, with controlled MPI/OpenMP process and thread binding.

### 6.5 CUDA GPU Benchmarks

CUDA experiments evaluate both standard GPU implementations and alternative block configurations.

Submit the formal benchmark jobs:

```bash
qsub scripts/cuda/gpu_benchmark.pbs
qsub scripts/cuda/block_benchmark.pbs
```

The corresponding Python drivers are:

- `scripts/cuda/run_gpu_benchmarks.py`
- `scripts/cuda/run_block_benchmarks.py`

The formal GPU study includes 60 runs across 12 configurations; the block-configuration study includes 50 runs across 10 configurations.

## 7. Results and Performance Metrics

Benchmark scripts generate runtime measurements and CSV results in the configured output directories.

Performance is evaluated using:

- **Median runtime:** Median of five independent runs
- **Strong-scaling speedup:** S(P) = T(1) / T(P)
- **Parallel efficiency:** E(P) = S(P) / P
- **Weak scaling:** Runtime as process count and global workload increase together
- **Communication overhead:** MPI computation and communication timing
- **GPU acceleration:** Comparison against relevant CPU baselines

For hybrid results, the summary script is:

```bash
python3 scripts/hybrid/summarize_results.py \
  results/hybrid/New_Runs/*/*.csv
```

It validates repeated measurements and generates summary statistics, including median runtime and speedup relative to the matching one-thread-per-rank hybrid configuration.

Detailed numerical results, plots, and performance analysis are provided in the accompanying project report.

## 8. Reproducibility Notes

- The scripts are configured for the original NUS Vanda environment.
- PBS project IDs, queue settings, module names, and resource requests may require adjustment on another HPC system.
- MPI and OpenMP process placement can affect performance.
- GPU benchmarks require compatible CUDA hardware and executable paths.
- Generated PBS files, executables, and benchmark CSV outputs are not included in the source repository.
- Fixed-iteration runtime measurements are distinct from time-to-convergence measurements.

The repository provides the source code, experiment scripts, and instructions needed to reproduce the main numerical experiments. Exact performance measurements may vary across hardware and runtime environments.
