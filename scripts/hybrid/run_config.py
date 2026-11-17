
"""Run one allocated CPU configuration. Never run formal jobs on a login node."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess

FIELDS = ('dimension,N,iterations,mpi_ranks,critical_rank,parallel_runtime,'
          'critical_comm_time,critical_compute_time,communication_fraction,'
          'min_rank_runtime,mean_rank_runtime,max_rank_runtime,rank_imbalance,'
          'final_residual,final_max_error').split(',')


def launch(exe, ranks, threads, mode, n, iterations=1000):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC='FALSE',
               OMP_PLACES='cores', OMP_PROC_BIND='close')

    # PBS provides allocated hosts; Intel MPI uses omp-sized pinning domains.
    env.setdefault('I_MPI_PIN_DOMAIN', 'omp')

    cmd = shlex.split(env.get('MPIEXEC', 'mpirun')) + [
        '-np', str(ranks), str(exe), mode, str(n)
    ]

    if mode == 'benchmark':
        cmd.append(str(iterations))

    result = subprocess.run(
        cmd, env=env, text=True, capture_output=True, check=True
    )

    if f'THREADS,{threads}' not in result.stdout.splitlines():
        raise RuntimeError(
            'Actual thread count missing or different from requested count'
        )

    return result.stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dimension', type=int, choices=[2, 3], required=True)
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--ranks', type=int, required=True)
    p.add_argument('--threads', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=5)

    a = p.parse_args()

    if min(a.n, a.ranks, a.threads) < 1 or a.ranks > a.n or a.repeats != 5:
        p.error(
            'Require 1 <= ranks <= N, positive threads and exactly 5 repeats'
        )

    # Repository root: scripts/hybrid/ -> repository root
    root = Path(__file__).resolve().parents[2]

    exe = root / 'bin' / f'poisson_{a.dimension}d_hybrid'

    a.output.parent.mkdir(parents=True, exist_ok=True)

    # Exclusive file creation prevents accidental duplicate or overwritten runs.
    metadata = dict(
        host=platform.node(),
        job=os.getenv('PBS_JOBID'),
        nodefile=os.getenv('PBS_NODEFILE'),
        ranks=a.ranks,
        loaded_modules=os.getenv('LOADEDMODULES'),
        mpiexec=os.getenv('MPIEXEC', 'mpirun'),
        threads=a.threads,
        total_cpu_cores=a.ranks * a.threads,
        binding=(
            'OMP_PLACES=cores; OMP_PROC_BIND=close; I_MPI_PIN_DOMAIN='
            + os.getenv('I_MPI_PIN_DOMAIN', 'omp')
        ),
        environment={
            k: v for k, v in os.environ.items()
            if k.startswith(('OMP_', 'I_MPI_'))
        }
    )

    if os.getenv('PBS_NODEFILE'):
        metadata['allocated_hosts'] = Path(
            os.environ['PBS_NODEFILE']
        ).read_text().splitlines()

    metadata['cpu'] = subprocess.run(
        ['lscpu'], text=True, capture_output=True
    ).stdout

    fields = [
        'implementation', 'omp_threads', 'total_cpu_cores', 'repeat'
    ] + FIELDS

    with a.output.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for repeat in range(1, 6):
            out = launch(exe, a.ranks, a.threads, 'benchmark', a.n)

            lines = [
                s for s in out.splitlines() if s.startswith('CSV,')
            ]

            if len(lines) != 1:
                raise RuntimeError('Expected exactly one CSV row')

            values = next(csv.reader([lines[0]]))[1:]

            if len(values) != len(FIELDS):
                raise RuntimeError('Unexpected solver CSV schema')

            row = dict(zip(FIELDS, values))

            if (
                row['dimension'],
                int(row['N']),
                int(row['iterations']),
                int(row['mpi_ranks'])
            ) != (
                f'{a.dimension}D',
                a.n,
                1000,
                a.ranks
            ):
                raise RuntimeError('Solver configuration mismatch')

            row.update(
                implementation='hybrid',
                omp_threads=a.threads,
                total_cpu_cores=a.ranks * a.threads,
                repeat=repeat
            )

            writer.writerow(row)
            f.flush()

    a.output.with_suffix('.environment.json').write_text(
        json.dumps(metadata, indent=2)
    )


if __name__ == '__main__':
    main()
