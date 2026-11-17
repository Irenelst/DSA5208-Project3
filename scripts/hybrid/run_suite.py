
"""Run selected standard configurations sequentially in one PBS allocation."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def get_hosts(nodefile, nodes):
    hosts = list(dict.fromkeys(Path(nodefile).read_text().split()))
    if len(hosts) != nodes:
        raise ValueError(
            'Expected {} allocated hosts, got {}'.format(nodes, hosts)
        )
    return hosts


def launcher(hosts, threads):
    if threads not in (1, 2, 4):
        raise ValueError('Threads must be 1, 2 or 4')

    return 'mpirun -ppn {} -hosts {}'.format(
        72 // threads, ','.join(hosts)
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--nodes', type=int, choices=(1, 2, 4), required=True
    )
    p.add_argument(
        '--threads',
        type=int,
        nargs='+',
        choices=(1, 2, 4),
        default=[1, 2, 4]
    )

    a = p.parse_args()

    if len(set(a.threads)) != len(a.threads):
        p.error('Thread configurations must not repeat')

    if not os.getenv('PBS_JOBID') or not os.getenv('PBS_NODEFILE'):
        p.error(
            'Run this script in a PBS CPU allocation, not on the login node'
        )

    hosts = get_hosts(os.environ['PBS_NODEFILE'], a.nodes)

    # Repository root: scripts/hybrid/ -> repository root
    root = Path(__file__).resolve().parents[2]

    # Store generated experiment results outside source directories
    output = root / 'results' / 'hybrid' / 'New_Runs' / os.environ['PBS_JOBID']

    output.mkdir(parents=True, exist_ok=False)

    print(
        'Hosts: {}. Outputs: {}'.format(hosts, output),
        flush=True
    )

    for dim, n in ((2, 8192), (3, 384)):
        for threads in a.threads:

            ranks = a.nodes * (72 // threads)
            cores = a.nodes * 72

            name = '{}D_{}cores_{}MPI_{}threads.csv'.format(
                dim, cores, ranks, threads
            )

            env = os.environ.copy()

            # Intel MPI otherwise ignores -ppn under PBS scheduler placement.
            env['I_MPI_JOB_RESPECT_PROCESS_PLACEMENT'] = 'off'
            env['MPIEXEC'] = launcher(hosts, threads)

            print(
                'Running {} (5 repeats)'.format(name),
                flush=True
            )

            subprocess.run(
                [
                    sys.executable,
                    str(root / 'scripts' / 'hybrid' / 'run_config.py'),
                    '--dimension', str(dim),
                    '--n', str(n),
                    '--ranks', str(ranks),
                    '--threads', str(threads),
                    '--output', str(output / name)
                ],
                env=env,
                check=True
            )

    print(
        'COMPLETE: {} configurations, five repeats each.'.format(
            2 * len(a.threads)
        ),
        flush=True
    )


if __name__ == '__main__':
    main()
