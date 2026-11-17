
#!/usr/bin/env python3
"""
DSA5208 Project 3
CUDA Block-Size Benchmark Runner

Evaluate CUDA thread-block configurations for the
2D and 3D Poisson Jacobi solvers.

Repository structure:
    src/cuda/       CUDA source files
    bin/            Compiled CUDA executables
    scripts/cuda/   Benchmark scripts
    results/cuda/   Generated experiment results

Usage (from repository root):
    python3 scripts/cuda/run_block_benchmarks.py

Smoke test:
    python3 scripts/cuda/run_block_benchmarks.py \
        --smoke --outdir results/cuda/block_smoke
"""

import argparse
import csv
import datetime
import math
import os
from pathlib import Path
import re
import subprocess
import sys

# ------------------------------------------------------------
# Experiment configurations
# ------------------------------------------------------------

CONFIGS = {
    "2D": [
        (8, 8),
        (16, 16),
        (32, 8),
        (8, 32),
        (32, 16),
    ],
    "3D": [
        (8, 8, 4),
        (8, 8, 8),
        (16, 4, 4),
        (16, 8, 4),
        (32, 4, 4),
    ],
}

PAT = re.compile(
    r"^CSV,benchmark,(2D|3D),(\d+),(\d+),"
    r"([^,]+),([^,]+),([^,\s]+)\s*$",
    re.M,
)

GPUPAT = re.compile(
    r"^GPU\s*=\s*(.+)$",
    re.M,
)

FIELDS = [
    "timestamp_utc",
    "dimension",
    "N",
    "K",
    "block",
    "threads_per_block",
    "repeat",
    "gpu",
    "compute_s",
    "end_to_end_s",
    "total_runtime_s",
    "residual",
    "selection_metric",
    "job_id",
    "hostname",
    "status",
    "error",
]

SUMMARY = [
    "dimension",
    "N",
    "K",
    "block",
    "threads_per_block",
    "repeats_successful",
    "representative_repeat",
    "gpu",
    "compute_s",
    "end_to_end_s",
    "total_runtime_s",
    "residual",
    "selection_metric",
    "job_id",
    "hostname",
]


def write_rows(path, fields, rows):
    """Write a CSV file with a header."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    parser.add_argument(
        "--outdir",
        default="results/cuda/block_size",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Repository paths
    # --------------------------------------------------------

    # scripts/cuda/run_block_benchmarks.py
    # -> repository root
    root = Path(__file__).resolve().parents[2]

    bin_dir = root / "bin"

    out = (root / args.outdir).resolve()

    # Prevent accidental overwriting of experiment results
    if out.exists() and any(out.iterdir()):
        print(
            "Output directory not empty; "
            "use a new --outdir to avoid overwriting:",
            out,
            file=sys.stderr,
        )
        return 2

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = []
    summaries = []
    failed = False

    # --------------------------------------------------------
    # Run 2D and 3D block-size experiments
    # --------------------------------------------------------

    for dim in ("2D", "3D"):

        executable = bin_dir / (
            "poisson_2d_cuda_block"
            if dim == "2D"
            else "poisson_3d_cuda_block"
        )

        if not executable.is_file():
            print(
                "Missing executable:",
                executable,
                file=sys.stderr,
            )
            return 2

        # Formal benchmark problem sizes
        N = (
            32
            if args.smoke
            else (
                4096
                if dim == "2D"
                else 256
            )
        )

        configs = (
            CONFIGS[dim][:1]
            if args.smoke
            else CONFIGS[dim]
        )

        for cfg in configs:

            block = "x".join(
                map(str, cfg)
            )

            group = []

            repeats = (
                1
                if args.smoke
                else 5
            )

            for rep in range(1, repeats + 1):

                command = [
                    str(executable),
                    "benchmark",
                    str(N),
                    "1000",
                    *map(str, cfg),
                ]

                proc = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    cwd=root,
                )

                log = out / (
                    f"{dim}_N{N}_B{block}_R{rep}.log"
                )

                log.write_text(
                    f"command={command}\n"
                    f"exit={proc.returncode}\n"
                    f"STDOUT:\n{proc.stdout}\n"
                    f"STDERR:\n{proc.stderr}\n",
                    encoding="utf-8",
                )

                match = PAT.search(proc.stdout)

                row = dict(
                    timestamp_utc=(
                        datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat()
                    ),
                    dimension=dim,
                    N=N,
                    K=1000,
                    block=block,
                    threads_per_block=math.prod(cfg),
                    repeat=rep,
                    gpu="",
                    compute_s="",
                    end_to_end_s="",
                    total_runtime_s="",
                    residual="",
                    selection_metric="compute_s",
                    job_id=os.getenv("PBS_JOBID", ""),
                    hostname=os.uname().nodename,
                    status="failed",
                    error="",
                )

                valid = (
                    proc.returncode == 0
                    and match is not None
                    and match.group(1) == dim
                    and int(match.group(2)) == N
                    and int(match.group(3)) == 1000
                )

                if valid:

                    try:
                        compute, e2e, residual = map(
                            float,
                            match.group(4, 5, 6),
                        )

                        if not (
                            all(
                                map(
                                    math.isfinite,
                                    (compute, e2e, residual),
                                )
                            )
                            and 0 < compute <= e2e
                            and residual >= 0
                        ):
                            raise ValueError(
                                "Invalid timing/residual"
                            )

                        gpu_match = GPUPAT.search(
                            proc.stdout
                        )

                        row.update(
                            compute_s=compute,
                            end_to_end_s=e2e,
                            total_runtime_s=compute,
                            residual=residual,
                            gpu=(
                                gpu_match.group(1)
                                if gpu_match
                                else "unknown"
                            ),
                            status="ok",
                        )

                    except ValueError as exc:
                        row["error"] = str(exc)

                else:
                    row["error"] = (
                        f"exit={proc.returncode}; "
                        f"see {log.name}"
                    )

                group.append(row)
                all_rows.append(row)

                write_rows(
                    out / "raw_runs.csv",
                    FIELDS,
                    all_rows,
                )

                print(
                    dim,
                    N,
                    block,
                    "repeat",
                    rep,
                    row["status"],
                    row["compute_s"],
                    flush=True,
                )

            # ------------------------------------------------
            # Select median-runtime observation
            # ------------------------------------------------

            successful = [
                row
                for row in group
                if row["status"] == "ok"
            ]

            expected = (
                1
                if args.smoke
                else 5
            )

            if len(successful) != expected:

                failed = True

                print(
                    "INCOMPLETE:",
                    dim,
                    block,
                    len(successful),
                    "/",
                    expected,
                    file=sys.stderr,
                )

                continue

            representative = sorted(
                successful,
                key=lambda row: (
                    row["total_runtime_s"],
                    row["repeat"],
                ),
            )[len(successful) // 2]

            summary = {
                key: (
                    len(successful)
                    if key == "repeats_successful"
                    else representative["repeat"]
                    if key == "representative_repeat"
                    else representative[key]
                )
                for key in SUMMARY
            }

            summaries.append(summary)

            write_rows(
                out / "median_summary.csv",
                SUMMARY,
                summaries,
            )

    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
