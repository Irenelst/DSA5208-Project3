#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

constexpr double PI = 3.14159265358979323846;


// ============================================================
// Analytical solution and source term
//
// Poisson equation:
//     ∇²u = f
//
// Exact solution:
//     u(x,y) = sin(pi*x) sin(pi*y)
//
// Therefore:
//     f(x,y) = -2*pi^2*sin(pi*x)sin(pi*y)
//
// Boundary condition:
//     u = 0 on all boundaries
// ============================================================

double exact_solution(double x, double y)
{
    return std::sin(PI * x) * std::sin(PI * y);
}

double source_term(double x, double y)
{
    return -2.0 * PI * PI
         * std::sin(PI * x)
         * std::sin(PI * y);
}


// ============================================================
// Local indexing
// ============================================================

std::size_t idx(int i, int j, int cols)
{
    return static_cast<std::size_t>(i) * cols + j;
}


// ============================================================
// Row-wise domain decomposition
// ============================================================

void compute_decomposition(
    int N,
    int rank,
    int size,
    int& local_n,
    int& global_start_row)
{
    const int base = N / size;
    const int remainder = N % size;

    local_n =
        base + (rank < remainder ? 1 : 0);

    global_start_row =
        rank * base + std::min(rank, remainder) + 1;
}


// ============================================================
// Initialize source term
// ============================================================

void initialize_source(
    std::vector<double>& f,
    int N,
    int local_n,
    int global_start_row,
    double h)
{
    const int cols = N + 2;

    for (int i = 1; i <= local_n; ++i)
    {
        const int global_i =
            global_start_row + i - 1;

        const double x = global_i * h;

        for (int j = 1; j <= N; ++j)
        {
            const double y = j * h;

            f[idx(i, j, cols)] =
                source_term(x, y);
        }
    }
}


// ============================================================
// Halo exchange
//
// Blocking MPI_Sendrecv is used as the MPI baseline.
// ============================================================

void exchange_halos(
    std::vector<double>& u,
    int N,
    int local_n,
    int rank,
    int size)
{
    const int cols = N + 2;

    const int upper =
        (rank == 0)
        ? MPI_PROC_NULL
        : rank - 1;

    const int lower =
        (rank == size - 1)
        ? MPI_PROC_NULL
        : rank + 1;

    // Send first owned row upward.
    // Receive lower neighbour into bottom ghost row.
    MPI_Sendrecv(
        &u[idx(1, 0, cols)],
        cols,
        MPI_DOUBLE,
        upper,
        0,

        &u[idx(local_n + 1, 0, cols)],
        cols,
        MPI_DOUBLE,
        lower,
        0,

        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE);

    // Send last owned row downward.
    // Receive upper neighbour into top ghost row.
    MPI_Sendrecv(
        &u[idx(local_n, 0, cols)],
        cols,
        MPI_DOUBLE,
        lower,
        1,

        &u[idx(0, 0, cols)],
        cols,
        MPI_DOUBLE,
        upper,
        1,

        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE);
}


// ============================================================
// One Jacobi step
// ============================================================

void jacobi_step(
    const std::vector<double>& u,
    std::vector<double>& u_new,
    const std::vector<double>& f,
    int N,
    int local_n,
    double h)
{
    const int cols = N + 2;
    const double h2 = h * h;

    for (int i = 1; i <= local_n; ++i)
    {
        for (int j = 1; j <= N; ++j)
        {
            u_new[idx(i, j, cols)] =
                0.25 *
                (
                    u[idx(i - 1, j, cols)]
                    + u[idx(i + 1, j, cols)]
                    + u[idx(i, j - 1, cols)]
                    + u[idx(i, j + 1, cols)]
                    - h2 * f[idx(i, j, cols)]
                );
        }
    }
}


// ============================================================
// Relative residual
//
//     R = ||f - A_h u||_2 / ||f||_2
// ============================================================

double compute_relative_residual(
    const std::vector<double>& u,
    const std::vector<double>& f,
    int N,
    int local_n,
    double h)
{
    const int cols = N + 2;
    const double h2 = h * h;

    double local_residual_sq = 0.0;
    double local_f_sq = 0.0;

    for (int i = 1; i <= local_n; ++i)
    {
        for (int j = 1; j <= N; ++j)
        {
            const double Au =
                (
                    u[idx(i - 1, j, cols)]
                    + u[idx(i + 1, j, cols)]
                    + u[idx(i, j - 1, cols)]
                    + u[idx(i, j + 1, cols)]
                    - 4.0 * u[idx(i, j, cols)]
                ) / h2;

            const double r =
                f[idx(i, j, cols)] - Au;

            local_residual_sq += r * r;

            const double fij =
                f[idx(i, j, cols)];

            local_f_sq += fij * fij;
        }
    }

    double global_residual_sq = 0.0;
    double global_f_sq = 0.0;

    MPI_Allreduce(
        &local_residual_sq,
        &global_residual_sq,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD);

    MPI_Allreduce(
        &local_f_sq,
        &global_f_sq,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD);

    return std::sqrt(
        global_residual_sq / global_f_sq);
}


// ============================================================
// Global maximum analytical error
// ============================================================

double compute_global_max_error(
    const std::vector<double>& u,
    int N,
    int local_n,
    int global_start_row,
    double h)
{
    const int cols = N + 2;

    double local_max_error = 0.0;

    for (int i = 1; i <= local_n; ++i)
    {
        const int global_i =
            global_start_row + i - 1;

        const double x = global_i * h;

        for (int j = 1; j <= N; ++j)
        {
            const double y = j * h;

            const double error =
                std::abs(
                    u[idx(i, j, cols)]
                    - exact_solution(x, y));

            local_max_error =
                std::max(
                    local_max_error,
                    error);
        }
    }

    double global_max_error = 0.0;

    MPI_Allreduce(
        &local_max_error,
        &global_max_error,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        MPI_COMM_WORLD);

    return global_max_error;
}


// ============================================================
// Check that each rank owns at least one row
// ============================================================

void check_decomposition(
    int N,
    int local_n,
    int rank,
    int size)
{
    if (local_n == 0)
    {
        if (rank == 0)
        {
            std::cerr
                << "Error: number of MPI processes ("
                << size
                << ") must not exceed N ("
                << N
                << ") for this row-wise decomposition.\n";
        }

        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}


// ============================================================
// VALIDATION MODE
// ============================================================

void run_validation(
    int N,
    int rank,
    int size)
{
    const double tolerance = 1.0e-6;
    const int check_interval = 100;
    const int max_iterations = 100000;

    const double h =
        1.0 / static_cast<double>(N + 1);

    int local_n = 0;
    int global_start_row = 0;

    compute_decomposition(
        N,
        rank,
        size,
        local_n,
        global_start_row);

    check_decomposition(
        N,
        local_n,
        rank,
        size);

    const int cols = N + 2;

    std::vector<double> u(
        static_cast<std::size_t>(local_n + 2) * cols,
        0.0);

    std::vector<double> u_new(
        static_cast<std::size_t>(local_n + 2) * cols,
        0.0);

    std::vector<double> f(
        static_cast<std::size_t>(local_n + 2) * cols,
        0.0);

    initialize_source(
        f,
        N,
        local_n,
        global_start_row,
        h);

    if (rank == 0)
    {
        std::cout
            << "========================================\n"
            << "2D Poisson MPI Validation\n"
            << "========================================\n"
            << "Global interior grid: "
            << N << " x " << N << '\n'
            << "MPI processes: "
            << size << '\n'
            << "h = "
            << std::setprecision(10)
            << h << '\n'
            << "Tolerance = "
            << tolerance << '\n'
            << "Check interval = "
            << check_interval << '\n'
            << "========================================\n";
    }

    exchange_halos(
        u,
        N,
        local_n,
        rank,
        size);

    double residual =
        compute_relative_residual(
            u,
            f,
            N,
            local_n,
            h);

    if (rank == 0)
    {
        std::cout
            << "Iteration 0, relative residual = "
            << std::setprecision(10)
            << residual << '\n';
    }

    int iteration = 0;
    bool converged = false;

    while (iteration < max_iterations)
    {
        exchange_halos(
            u,
            N,
            local_n,
            rank,
            size);

        jacobi_step(
            u,
            u_new,
            f,
            N,
            local_n,
            h);

        u.swap(u_new);

        ++iteration;

        if (iteration % check_interval == 0)
        {
            exchange_halos(
                u,
                N,
                local_n,
                rank,
                size);

            residual =
                compute_relative_residual(
                    u,
                    f,
                    N,
                    local_n,
                    h);

            if (rank == 0)
            {
                std::cout
                    << "Iteration "
                    << iteration
                    << ", relative residual = "
                    << std::setprecision(10)
                    << residual
                    << '\n';
            }

            if (residual < tolerance)
            {
                converged = true;
                break;
            }
        }
    }

    exchange_halos(
        u,
        N,
        local_n,
        rank,
        size);

    residual =
        compute_relative_residual(
            u,
            f,
            N,
            local_n,
            h);

    const double max_error =
        compute_global_max_error(
            u,
            N,
            local_n,
            global_start_row,
            h);

    if (rank == 0)
    {
        std::cout
            << "========================================\n"
            << "Final iteration: "
            << iteration << '\n'
            << "Final relative residual: "
            << std::setprecision(10)
            << residual << '\n'
            << "Maximum analytical error: "
            << std::setprecision(10)
            << max_error << '\n'
            << "Status: "
            << (converged
                ? "CONVERGED"
                : "NOT CONVERGED")
            << '\n'
            << "========================================\n";
    }
}


// ============================================================
// BENCHMARK MODE - Vanda Version
//
// Fixed number of Jacobi iterations.
//
// Timed region contains:
//     1. halo exchange
//     2. Jacobi computation
//     3. local grid swap
//
// Initialization and final validation are excluded.
//
// IMPORTANT:
// Instead of independently taking max(total), max(comm),
// and max(compute), all rank timings are gathered.
//
// We identify:
//
//     critical_rank = argmax_r(total_time_r)
//
// Communication and computation breakdown are then taken from
// the SAME critical rank.
//
// This provides a consistent critical-path timing breakdown.
// ============================================================

void run_benchmark(
    int N,
    int iterations,
    int rank,
    int size)
{
    const double h =
        1.0 / static_cast<double>(N + 1);

    int local_n = 0;
    int global_start_row = 0;

    compute_decomposition(
        N,
        rank,
        size,
        local_n,
        global_start_row);

    check_decomposition(
        N,
        local_n,
        rank,
        size);

    const int cols = N + 2;

    std::vector<double> u(
        static_cast<std::size_t>(local_n + 2) * cols,
        0.0);

    std::vector<double> u_new(
        static_cast<std::size_t>(local_n + 2) * cols,
        0.0);

    std::vector<double> f(
        static_cast<std::size_t>(local_n + 2) * cols,
        0.0);

    initialize_source(
        f,
        N,
        local_n,
        global_start_row,
        h);


    // ========================================================
    // Synchronize all ranks before timing.
    //
    // Initialization is excluded from benchmark runtime.
    // ========================================================

    MPI_Barrier(MPI_COMM_WORLD);

    const double total_start =
        MPI_Wtime();

    double local_communication_time = 0.0;
    double local_compute_time = 0.0;


    // ========================================================
    // Fixed-iteration benchmark
    // ========================================================

    for (int iteration = 0;
         iteration < iterations;
         ++iteration)
    {
        // ----------------------------------------------------
        // Halo communication timing
        // ----------------------------------------------------

        const double communication_start =
            MPI_Wtime();

        exchange_halos(
            u,
            N,
            local_n,
            rank,
            size);

        const double communication_end =
            MPI_Wtime();

        local_communication_time +=
            communication_end
            - communication_start;


        // ----------------------------------------------------
        // Jacobi computation timing
        // ----------------------------------------------------

        const double compute_start =
            MPI_Wtime();

        jacobi_step(
            u,
            u_new,
            f,
            N,
            local_n,
            h);

        const double compute_end =
            MPI_Wtime();

        local_compute_time +=
            compute_end
            - compute_start;

        u.swap(u_new);
    }


    const double total_end =
        MPI_Wtime();

    const double local_total_time =
        total_end - total_start;


    // ========================================================
    // Gather timing information from every MPI rank.
    //
    // Only rank 0 allocates the receive vectors.
    // ========================================================

    std::vector<double> all_total_times;
    std::vector<double> all_communication_times;
    std::vector<double> all_compute_times;

    if (rank == 0)
    {
        all_total_times.resize(size);
        all_communication_times.resize(size);
        all_compute_times.resize(size);
    }

    MPI_Gather(
        &local_total_time,
        1,
        MPI_DOUBLE,
        rank == 0
            ? all_total_times.data()
            : nullptr,
        1,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD);

    MPI_Gather(
        &local_communication_time,
        1,
        MPI_DOUBLE,
        rank == 0
            ? all_communication_times.data()
            : nullptr,
        1,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD);

    MPI_Gather(
        &local_compute_time,
        1,
        MPI_DOUBLE,
        rank == 0
            ? all_compute_times.data()
            : nullptr,
        1,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD);


    // ========================================================
    // Final correctness sanity check.
    //
    // OUTSIDE timed region.
    // ========================================================

    exchange_halos(
        u,
        N,
        local_n,
        rank,
        size);

    const double final_residual =
        compute_relative_residual(
            u,
            f,
            N,
            local_n,
            h);

    const double final_max_error =
        compute_global_max_error(
            u,
            N,
            local_n,
            global_start_row,
            h);


    // ========================================================
    // Rank 0 performs timing analysis.
    // ========================================================

    if (rank == 0)
    {
        // ----------------------------------------------------
        // Find critical rank:
        //
        //     r* = argmax_r T_r
        //
        // This rank determines parallel runtime.
        // ----------------------------------------------------

        int critical_rank = 0;

        for (int r = 1; r < size; ++r)
        {
            if (all_total_times[r] >
                all_total_times[critical_rank])
            {
                critical_rank = r;
            }
        }


        // ----------------------------------------------------
        // Critical-path timing
        //
        // All three values come from the SAME rank.
        // ----------------------------------------------------

        const double parallel_runtime =
            all_total_times[critical_rank];

        const double critical_communication_time =
            all_communication_times[critical_rank];

        const double critical_compute_time =
            all_compute_times[critical_rank];


        // ----------------------------------------------------
        // Across-rank runtime statistics
        // ----------------------------------------------------

        double min_rank_runtime =
            all_total_times[0];

        double max_rank_runtime =
            all_total_times[0];

        double sum_rank_runtime = 0.0;

        for (int r = 0; r < size; ++r)
        {
            min_rank_runtime =
                std::min(
                    min_rank_runtime,
                    all_total_times[r]);

            max_rank_runtime =
                std::max(
                    max_rank_runtime,
                    all_total_times[r]);

            sum_rank_runtime +=
                all_total_times[r];
        }

        const double mean_rank_runtime =
            sum_rank_runtime /
            static_cast<double>(size);


        // ----------------------------------------------------
        // Derived metrics
        // ----------------------------------------------------

        const double time_per_iteration =
            parallel_runtime /
            static_cast<double>(iterations);

        const double communication_fraction =
            100.0 *
            critical_communication_time /
            parallel_runtime;

        const double compute_fraction =
            100.0 *
            critical_compute_time /
            parallel_runtime;

        const double other_fraction =
            100.0
            - communication_fraction
            - compute_fraction;

        const double rank_imbalance =
            100.0 *
            (max_rank_runtime - min_rank_runtime) /
            max_rank_runtime;


        // ====================================================
        // Human-readable output
        // ====================================================

        std::cout
            << "========================================\n"
            << "2D Poisson MPI Benchmark - Vanda Version\n"
            << "========================================\n"

            << "Global interior grid: "
            << N << " x " << N << '\n'

            << "MPI processes: "
            << size << '\n'

            << "Fixed iterations: "
            << iterations << '\n'

            << "h = "
            << std::setprecision(10)
            << h << '\n'

            << "----------------------------------------\n"
            << "Critical-path timing\n"
            << "----------------------------------------\n"

            << "Critical rank: "
            << critical_rank << '\n'

            << "Parallel runtime: "
            << parallel_runtime
            << " s\n"

            << "Time per iteration: "
            << time_per_iteration
            << " s\n"

            << "Critical-rank halo communication time: "
            << critical_communication_time
            << " s\n"

            << "Critical-rank Jacobi compute time: "
            << critical_compute_time
            << " s\n"

            << "Communication fraction: "
            << communication_fraction
            << " %\n"

            << "Compute fraction: "
            << compute_fraction
            << " %\n"

            << "Other/timing overhead fraction: "
            << other_fraction
            << " %\n"

            << "----------------------------------------\n"
            << "Across-rank runtime statistics\n"
            << "----------------------------------------\n"

            << "Minimum rank runtime: "
            << min_rank_runtime
            << " s\n"

            << "Mean rank runtime: "
            << mean_rank_runtime
            << " s\n"

            << "Maximum rank runtime: "
            << max_rank_runtime
            << " s\n"

            << "Rank runtime imbalance: "
            << rank_imbalance
            << " %\n"

            << "----------------------------------------\n"
            << "Correctness sanity check\n"
            << "----------------------------------------\n"

            << "Final relative residual: "
            << final_residual << '\n'

            << "Final max analytical error: "
            << final_max_error << '\n'

            << "========================================\n";


        // ====================================================
        // Machine-readable CSV row
        //
        // Format:
        //
        // CSV,
        // dimension,
        // N,
        // iterations,
        // mpi_ranks,
        // critical_rank,
        // parallel_runtime,
        // critical_comm_time,
        // critical_compute_time,
        // communication_fraction,
        // min_rank_runtime,
        // mean_rank_runtime,
        // max_rank_runtime,
        // rank_imbalance,
        // final_residual,
        // final_max_error
        //
        // A Vanda runner can later extract lines beginning
        // with "CSV," automatically.
        // ====================================================

        std::cout
            << "CSV,"
            << "2D,"
            << N << ","
            << iterations << ","
            << size << ","
            << critical_rank << ","
            << parallel_runtime << ","
            << critical_communication_time << ","
            << critical_compute_time << ","
            << communication_fraction << ","
            << min_rank_runtime << ","
            << mean_rank_runtime << ","
            << max_rank_runtime << ","
            << rank_imbalance << ","
            << final_residual << ","
            << final_max_error
            << '\n';
    }
}


// ============================================================
// Usage information
// ============================================================

void print_usage(const char* program_name)
{
    std::cout
        << "Usage:\n\n"

        << "Validation mode:\n"
        << "  mpirun -np <P> "
        << program_name
        << " validation <N>\n\n"

        << "Benchmark mode:\n"
        << "  mpirun -np <P> "
        << program_name
        << " benchmark <N> <iterations>\n\n"

        << "Examples:\n"
        << "  mpirun -np 4 "
        << program_name
        << " validation 16\n"

        << "  mpirun -np 4 "
        << program_name
        << " benchmark 512 1000\n";
}


// ============================================================
// Main
// ============================================================

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;

    MPI_Comm_rank(
        MPI_COMM_WORLD,
        &rank);

    MPI_Comm_size(
        MPI_COMM_WORLD,
        &size);


    // ========================================================
    // Parse command-line arguments
    // ========================================================

    if (argc < 3)
    {
        if (rank == 0)
        {
            print_usage(argv[0]);
        }

        MPI_Finalize();
        return 1;
    }

    const std::string mode = argv[1];

    int N = 0;

    try
    {
        N = std::stoi(argv[2]);
    }
    catch (...)
    {
        if (rank == 0)
        {
            std::cerr
                << "Error: N must be an integer.\n";
        }

        MPI_Finalize();
        return 1;
    }

    if (N <= 0)
    {
        if (rank == 0)
        {
            std::cerr
                << "Error: N must be positive.\n";
        }

        MPI_Finalize();
        return 1;
    }


    // ========================================================
    // Validation mode
    // ========================================================

    if (mode == "validation")
    {
        run_validation(
            N,
            rank,
            size);
    }


    // ========================================================
    // Benchmark mode
    // ========================================================

    else if (mode == "benchmark")
    {
        if (argc < 4)
        {
            if (rank == 0)
            {
                print_usage(argv[0]);
            }

            MPI_Finalize();
            return 1;
        }

        int iterations = 0;

        try
        {
            iterations =
                std::stoi(argv[3]);
        }
        catch (...)
        {
            if (rank == 0)
            {
                std::cerr
                    << "Error: iterations must be an integer.\n";
            }

            MPI_Finalize();
            return 1;
        }

        if (iterations <= 0)
        {
            if (rank == 0)
            {
                std::cerr
                    << "Error: iterations must be positive.\n";
            }

            MPI_Finalize();
            return 1;
        }

        run_benchmark(
            N,
            iterations,
            rank,
            size);
    }


    // ========================================================
    // Invalid mode
    // ========================================================

    else
    {
        if (rank == 0)
        {
            std::cerr
                << "Error: unknown mode \""
                << mode
                << "\".\n\n";

            print_usage(argv[0]);
        }

        MPI_Finalize();
        return 1;
    }


    MPI_Finalize();

    return 0;
}