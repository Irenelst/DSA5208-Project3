
"""Run inside a PBS allocation with >= 16 CPU cores. Results are not benchmarks."""
import csv
import math
import os
from pathlib import Path
import re
import shlex
import subprocess

from run_config import FIELDS, launch


# Repository root: scripts/hybrid/ -> repository root
root = Path(__file__).resolve().parents[2]

records = []

for dim in (2, 3):
    exe = root / 'bin' / f'poisson_{dim}d_hybrid'

    for n in (16, 17):  # 17 exercises uneven row/slab ownership.
        # On this sine forcing, the discrete solution is a single eigenmode.
        h = 1.0 / (n + 1)
        rho = math.cos(math.pi * h)
        factor = math.pi**2 * h * h / (4 * math.sin(math.pi * h / 2)**2)
        peak = max(math.sin(math.pi * i * h) for i in range(1, n + 1))**dim

        reference_exe = root / 'bin' / f'poisson_{dim}d_mpi_reference'

        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', I_MPI_PIN_DOMAIN='omp')

        ref_out = subprocess.run(
            shlex.split(env.get('MPIEXEC', 'mpirun')) +
            ['-np', '1', str(reference_exe), 'benchmark', str(n), '1000'],
            env=env,
            text=True,
            capture_output=True,
            check=True
        ).stdout

        ref_line = next(
            s for s in ref_out.splitlines() if s.startswith('CSV,')
        )
        reference = dict(
            zip(FIELDS, next(csv.reader([ref_line]))[1:])
        )

        for ranks in (1, 2, 4):
            for threads in (1, 2, 4):
                out = launch(exe, ranks, threads, 'benchmark', n)

                line = next(
                    s for s in out.splitlines() if s.startswith('CSV,')
                )
                row = dict(
                    zip(FIELDS, next(csv.reader([line]))[1:])
                )

                for key in ('final_residual', 'final_max_error'):
                    if not math.isclose(
                        float(row[key]),
                        float(reference[key]),
                        rel_tol=2e-7,
                        abs_tol=2e-12
                    ):
                        raise AssertionError(
                            ('Part1/Hybrid mismatch', key, dim, n, ranks, threads)
                        )

                want_residual = rho**1000
                want_error = abs(factor * (1 - rho**1000) - 1) * peak

                if not math.isclose(
                    float(row['final_residual']),
                    want_residual,
                    rel_tol=2e-5,
                    abs_tol=2e-12
                ):
                    raise AssertionError(
                        ('residual', dim, n, ranks, threads, row)
                    )

                if not math.isclose(
                    float(row['final_max_error']),
                    want_error,
                    rel_tol=2e-7,
                    abs_tol=2e-10
                ):
                    raise AssertionError(
                        ('analytical error', dim, n, ranks, threads, row)
                    )

                row.update(omp_threads=threads, test='fixed1000')
                records.append(row)

    for ranks in (1, 2, 4):
        for threads in (1, 2, 4):
            out = launch(exe, ranks, threads, 'validation', 16)

            if 'Status: CONVERGED' not in out:
                raise AssertionError(
                    ('not converged', dim, ranks, threads)
                )

            residual = float(
                re.search(r'Final relative residual:\s*(\S+)', out)[1]
            )
            error = float(
                re.search(r'Maximum analytical error:\s*(\S+)', out)[1]
            )

            # Same check schedule as Part1:
            # first multiple of 100 below tolerance.
            steps = next(
                k for k in range(100, 100001, 100)
                if math.cos(math.pi / 17)**k < 1e-6
            )

            if (
                int(re.search(r'Final iteration:\s*(\d+)', out)[1]) != steps
                or residual >= 1e-6
            ):
                raise AssertionError(
                    ('stopping rule', dim, ranks, threads)
                )

            h = 1 / 17
            factor = math.pi**2 * h * h / (
                4 * math.sin(math.pi * h / 2)**2
            )
            peak = max(
                math.sin(math.pi * i * h) for i in range(1, 17)
            )**dim

            want_error = abs(
                factor * (1 - math.cos(math.pi * h)**steps) - 1
            ) * peak

            if not math.isclose(
                error,
                want_error,
                rel_tol=2e-7,
                abs_tol=2e-10
            ):
                raise AssertionError(
                    ('converged solution error', dim, ranks, threads)
                )

            records.append(
                dict(
                    test='convergence',
                    dimension=f'{dim}D',
                    N=16,
                    mpi_ranks=ranks,
                    omp_threads=threads,
                    iterations=steps,
                    final_residual=residual,
                    final_max_error=error
                )
            )

# Save validation results in the repository's results directory.
path = root / 'results' / 'hybrid' / 'All_54_Validation_Tests.csv'
path.parent.mkdir(parents=True, exist_ok=True)

with path.open('w', newline='') as f:
    writer = csv.DictWriter(
        f,
        fieldnames=['test', 'omp_threads'] + FIELDS
    )
    writer.writeheader()
    writer.writerows(records)

print(
    f'PASS: {len(records)} cases, including uneven ownership '
    f'and convergence. Saved {path}'
)
