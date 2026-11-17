
"""Select the median-runtime WHOLE run, and compare matching thread=1 jobs."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics


def representative(rows):
    if len(rows) != 5 or {int(r['repeat']) for r in rows} != {1, 2, 3, 4, 5}:
        raise ValueError(
            'Each configuration requires exactly five distinct repeats; do not mix reruns'
        )

    chosen = dict(sorted(rows, key=lambda r: float(r['parallel_runtime']))[2])
    chosen['runtime_std'] = statistics.stdev(
        float(r['parallel_runtime']) for r in rows
    )
    return chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument('files', nargs='+', type=Path)
    p.add_argument(
        '--output',
        type=Path,
        default=Path('results/hybrid/Median_Runtime_Results.csv')
    )
    a = p.parse_args()

    groups = defaultdict(list)

    for path in a.files:
        with path.open() as f:
            for r in csv.DictReader(f):
                key = tuple(
                    r[k] for k in (
                        'dimension',
                        'N',
                        'iterations',
                        'mpi_ranks',
                        'omp_threads',
                        'total_cpu_cores'
                    )
                )
                groups[key].append(r)

    selected = [representative(g) for g in groups.values()]

    base = {
        (
            r['dimension'],
            r['N'],
            r['iterations'],
            r['total_cpu_cores']
        ): float(r['parallel_runtime'])
        for r in selected
        if int(r['omp_threads']) == 1
    }

    for r in selected:
        key = tuple(
            r[k] for k in (
                'dimension',
                'N',
                'iterations',
                'total_cpu_cores'
            )
        )
        r['hybrid_vs_mpi_speedup'] = (
            base[key] / float(r['parallel_runtime'])
            if key in base else ''
        )

    a.output.parent.mkdir(parents=True, exist_ok=True)

    with a.output.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(selected[0]))
        w.writeheader()
        w.writerows(selected)

    print(a.output)


if __name__ == '__main__':
    main()
