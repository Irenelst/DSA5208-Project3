
#!/usr/bin/env python3
"""
DSA5208 Project 3
CUDA GPU Benchmark Runner

Runs 2D and 3D CUDA Poisson solvers.
Preserves five independent runs and selects the
complete median-runtime observation.

Repository structure:
    src/cuda/       CUDA source files
    bin/            Compiled CUDA executables
    scripts/cuda/   Benchmark scripts
    results/cuda/   Generated experiment results

Usage (from repository root):
    python3 scripts/cuda/run_gpu_benchmarks.py

Smoke test:
    python3 scripts/cuda/run_gpu_benchmarks.py --smoke
"""

import argparse
import csv
import datetime as dt
import os
from pathlib import Path
import re
import subprocess
import sys

SIZES = {
    "2D": [128, 256, 512, 1024, 2048, 4096, 8192],
    "3D": [32, 64, 128, 256, 384],
}

FIELDS = [
    "timestamp_utc",
    "dimension",
    "N",
    "K",
    "repeat",
    "gpu",
    "gpu_compute_s",
    "gpu_end_to_end_s",
    "final_residual",
    "total_runtime_s",
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
    "repeats_successful",
    "representative_repeat",
    "gpu",
    "total_runtime_s",
    "gpu_compute_s",
    "gpu_end_to_end_s",
    "final_residual",
    "selection_metric",
    "job_id",
    "hostname",
]

PATTERN = re.compile(
    r"^CSV,benchmark,(2D|3D),(\d+),(\d+),"
    r"([^,]+),([^,]+),([^,\s]+)\s*$",
    re.M,
)

GPU = re.compile(r"^GPU\s*=\s*(.+)$", re.M)


def append_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    exists = path.exists() and path.stat().st_size > 0

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not exists:
            writer.writeheader()

        writer.writerows(rows)
        f.flush()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dimension",
        choices=["2D", "3D", "both"],
        default="both",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--outdir",
        default="results/cuda/gpu",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Only N=32 for 2D and 3D; one run each",
    )

    args = parser.parse_args()

    if args.repeats != 5 and not args.smoke:
        parser.error(
            "Formal experiments require exactly 5 repeats"
        )

    # --------------------------------------------------------
    # Repository paths
    # --------------------------------------------------------

    # scripts/cuda/run_gpu_benchmarks.py
    # -> repository root
    root = Path(__file__).resolve().parents[2]

    bin_dir = root / "bin"

    out = (root / args.outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    dimensions = (
        ["2D", "3D"]
        if args.dimension == "both"
        else [args.dimension]
    )

    any_failed = False

    # --------------------------------------------------------
    # Run benchmarks
    # --------------------------------------------------------

    for dimension in dimensions:

        executable = bin_dir / (
            "poisson_2d_cuda"
            if dimension == "2D"
            else "poisson_3d_cuda"
        )

        if not executable.is_file():
            print(
                "Missing executable:",
                executable,
                file=sys.stderr,
            )
            return 2

        sizes = (
            [32]
            if args.smoke
            else SIZES[dimension]
        )

        for n in sizes:

            rows = []

            repeats = (
                1
                if args.smoke
                else args.repeats
            )

            for repeat in range(1, repeats + 1):

                now = dt.datetime.now(
                    dt.timezone.utc
                ).isoformat()

                command = [
                    str(executable),
                    "benchmark",
                    str(n),
                    "1000",
                ]

                proc = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    cwd=root,
                )

                # Save detailed execution log
                log = out / (
                    f"{dimension}_N{n}_repeat{repeat}.log"
                )

                log.write_text(
                    "COMMAND: " + " ".join(command)
                    + "\nRETURN CODE: "
                    + str(proc.returncode)
                    + "\nSTDOUT:\n"
                    + proc.stdout
                    + "\nSTDERR:\n"
                    + proc.stderr,
                    encoding="utf-8",
                )

                match = PATTERN.search(proc.stdout)

                valid = (
                    proc.returncode == 0
                    and match is not None
                    and match.group(1) == dimension
                    and int(match.group(2)) == n
                    and int(match.group(3)) == 1000
                )

                row = dict(
                    timestamp_utc=now,
                    dimension=dimension,
                    N=n,
                    K=1000,
                    repeat=repeat,
                    gpu="",
                    gpu_compute_s="",
                    gpu_end_to_end_s="",
                    final_residual="",
                    total_runtime_s="",
                    selection_metric="gpu_compute_s",
                    job_id=os.environ.get(
                        "PBS_JOBID", ""
                    ),
                    hostname=os.uname().nodename,
                    status="ok" if valid else "failed",
                    error="",
                )

                if valid:

                    try:
                        compute = float(match.group(4))
                        e2e = float(match.group(5))
                        residual = float(match.group(6))

                        if not (
                            0 < compute <= e2e
                            and residual >= 0
                        ):
                            raise ValueError(
                                "Invalid timing/residual values"
                            )

                        gpu_match = GPU.search(proc.stdout)

                        row.update(
                            gpu_compute_s=compute,
                            gpu_end_to_end_s=e2e,
                            final_residual=residual,
                            total_runtime_s=compute,
                            gpu=(
                                gpu_match.group(1)
                                if gpu_match
                                else "unknown"
                            ),
                        )

                    except (ValueError, OverflowError) as exc:
                        row.update(
                            status="failed",
                            error=str(exc),
                        )

                else:
                    row["error"] = (
                        f"exit={proc.returncode}; "
                        f"see {log.name}"
                    )

                rows.append(row)

                print(
                    dimension,
                    n,
                    "repeat",
                    repeat,
                    row["status"],
                    "compute_s",
                    row["gpu_compute_s"],
                    flush=True,
                )

            # Save all independent runs
            append_csv(
                out / "raw_runs.csv",
                FIELDS,
                rows,
            )

            successful = [
                row
                for row in rows
                if row["status"] == "ok"
            ]

            expected = 1 if args.smoke else 5

            if len(successful) != expected:

                any_failed = True

                print(
                    f"INCOMPLETE {dimension} N={n}: "
                    f"{len(successful)}/{expected} "
                    "successful; no representative selected",
                    file=sys.stderr,
                    flush=True,
                )

                continue

            # ------------------------------------------------
            # Median runtime selection
            #
            # Select the complete observation corresponding
            # to the median GPU compute runtime.
            # ------------------------------------------------

            representative = sorted(
                successful,
                key=lambda row: (
                    row["total_runtime_s"],
                    row["repeat"],
                ),
            )[len(successful) // 2]

            summary_row = dict(
                dimension=dimension,
                N=n,
                K=1000,
                repeats_successful=len(successful),
                representative_repeat=representative["repeat"],
                gpu=representative["gpu"],
                total_runtime_s=representative["total_runtime_s"],
                gpu_compute_s=representative["gpu_compute_s"],
                gpu_end_to_end_s=representative["gpu_end_to_end_s"],
                final_residual=representative["final_residual"],
                selection_metric="gpu_compute_s",
                job_id=representative["job_id"],
                hostname=representative["hostname"],
            )

            summary_file = (
                "smoke_summary.csv"
                if args.smoke
                else "median_summary.csv"
            )

            append_csv(
                out / summary_file,
                SUMMARY,
                [summary_row],
            )

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
