#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <string>

constexpr double PI = 3.14159265358979323846;


// =========================================================
// Exact solution
// =========================================================

double exact_solution(double x, double y)
{
    return std::sin(PI * x) * std::sin(PI * y);
}


// =========================================================
// Source term for Laplacian(u) = f
// =========================================================

double source_term(double x, double y)
{
    return -2.0 * PI * PI
           * std::sin(PI * x)
           * std::sin(PI * y);
}


// =========================================================
// Convert 2D index to 1D index
// =========================================================

inline int idx(int i, int j, int size)
{
    return i * size + j;
}


// =========================================================
// One Jacobi iteration
// =========================================================

void jacobi_step(
    const std::vector<double>& u_old,
    std::vector<double>& u_new,
    const std::vector<double>& f,
    int N,
    int size,
    double h)
{
    const double h2 = h * h;

    for (int i = 1; i <= N; ++i)
    {
        for (int j = 1; j <= N; ++j)
        {
            u_new[idx(i, j, size)] =
                (
                    u_old[idx(i - 1, j, size)]
                    + u_old[idx(i + 1, j, size)]
                    + u_old[idx(i, j - 1, size)]
                    + u_old[idx(i, j + 1, size)]
                    - h2 * f[idx(i, j, size)]
                ) / 4.0;
        }
    }
}


// =========================================================
// Relative residual
// =========================================================

double compute_relative_residual(
    const std::vector<double>& u,
    const std::vector<double>& f,
    int N,
    int size,
    double h)
{
    const double h2 = h * h;

    double residual_sum = 0.0;
    double f_sum = 0.0;

    for (int i = 1; i <= N; ++i)
    {
        for (int j = 1; j <= N; ++j)
        {
            const double Au =
                (
                    u[idx(i - 1, j, size)]
                    + u[idx(i + 1, j, size)]
                    + u[idx(i, j - 1, size)]
                    + u[idx(i, j + 1, size)]
                    - 4.0 * u[idx(i, j, size)]
                ) / h2;

            const double r =
                f[idx(i, j, size)] - Au;

            residual_sum += r * r;

            const double f_value =
                f[idx(i, j, size)];

            f_sum += f_value * f_value;
        }
    }

    return std::sqrt(residual_sum)
           / std::sqrt(f_sum);
}


// =========================================================
// Maximum analytical error
// =========================================================

double compute_max_error(
    const std::vector<double>& u,
    int N,
    int size,
    double h)
{
    double max_error = 0.0;

    for (int i = 1; i <= N; ++i)
    {
        const double x = i * h;

        for (int j = 1; j <= N; ++j)
        {
            const double y = j * h;

            const double exact =
                exact_solution(x, y);

            const double error =
                std::abs(
                    u[idx(i, j, size)] - exact
                );

            max_error =
                std::max(max_error, error);
        }
    }

    return max_error;
}


// =========================================================
// Initialize source term
// =========================================================

void initialize_source(
    std::vector<double>& f,
    int N,
    int size,
    double h)
{
    for (int i = 1; i <= N; ++i)
    {
        const double x = i * h;

        for (int j = 1; j <= N; ++j)
        {
            const double y = j * h;

            f[idx(i, j, size)] =
                source_term(x, y);
        }
    }
}


// =========================================================
// Validation mode
// =========================================================

void run_validation(int N)
{
    const int size = N + 2;
    const double h = 1.0 / (N + 1);
    const int total_points = size * size;

    const double tolerance = 1e-6;
    const int max_iterations = 100000;
    const int check_frequency = 100;

    std::vector<double> u_old(total_points, 0.0);
    std::vector<double> u_new(total_points, 0.0);
    std::vector<double> f(total_points, 0.0);

    initialize_source(f, N, size, h);

    double residual =
        compute_relative_residual(
            u_old, f, N, size, h
        );

    int iteration = 0;
    bool converged = false;

    while (iteration < max_iterations)
    {
        jacobi_step(
            u_old, u_new, f, N, size, h
        );

        std::swap(u_old, u_new);

        ++iteration;

        if (iteration % check_frequency == 0)
        {
            residual =
                compute_relative_residual(
                    u_old, f, N, size, h
                );

            if (residual < tolerance)
            {
                converged = true;
                break;
            }
        }
    }

    residual =
        compute_relative_residual(
            u_old, f, N, size, h
        );

    const double max_error =
        compute_max_error(
            u_old, N, size, h
        );

    std::cout << "\nValidation Results\n";
    std::cout << "------------------------------\n";

    std::cout << "N                 = "
              << N << "\n";

    std::cout << "Iterations        = "
              << iteration << "\n";

    std::cout << "Relative residual = "
              << residual << "\n";

    std::cout << "Maximum error     = "
              << max_error << "\n";

    std::cout << "Status            = "
              << (converged
                  ? "CONVERGED"
                  : "NOT CONVERGED")
              << "\n";
}


// =========================================================
// Benchmark mode
// =========================================================

void run_benchmark(int N, int K)
{
    const int size = N + 2;
    const double h = 1.0 / (N + 1);
    const int total_points = size * size;

    std::vector<double> u_old(total_points, 0.0);
    std::vector<double> u_new(total_points, 0.0);
    std::vector<double> f(total_points, 0.0);

    initialize_source(f, N, size, h);


    // -----------------------------------------------------
    // Start timer
    //
    // Timing includes ONLY the K Jacobi iterations.
    // Initialization and final validation are excluded.
    // -----------------------------------------------------

    const auto start =
        std::chrono::steady_clock::now();


    for (int iteration = 0;
         iteration < K;
         ++iteration)
    {
        jacobi_step(
            u_old,
            u_new,
            f,
            N,
            size,
            h
        );

        std::swap(u_old, u_new);
    }


    const auto end =
        std::chrono::steady_clock::now();


    // -----------------------------------------------------
    // Timing result
    // -----------------------------------------------------

    const std::chrono::duration<double> elapsed =
        end - start;

    const double total_time =
        elapsed.count();

    const double time_per_iteration =
        total_time / K;


    // -----------------------------------------------------
    // Final validation AFTER timer stops
    // -----------------------------------------------------

    const double final_residual =
        compute_relative_residual(
            u_old,
            f,
            N,
            size,
            h
        );


    // -----------------------------------------------------
    // Human-readable output
    // -----------------------------------------------------

    std::cout << "\nBenchmark Results\n";
    std::cout << "------------------------------\n";

    std::cout << "Dimension          = 2D\n";

    std::cout << "N                  = "
              << N << "\n";

    std::cout << "Grid size          = "
              << size << " x " << size << "\n";

    std::cout << "K                  = "
              << K << "\n";

    std::cout << "Total runtime (s)  = "
              << total_time << "\n";

    std::cout << "Time/iteration (s) = "
              << time_per_iteration << "\n";

    std::cout << "Final residual     = "
              << final_residual << "\n";


    // -----------------------------------------------------
    // Machine-readable output
    //
    // Format:
    // CSV,dimension,N,K,total_runtime,
    //     time_per_iteration,final_residual
    // -----------------------------------------------------

    std::cout << "CSV,"
              << "2D,"
              << N << ","
              << K << ","
              << total_time << ","
              << time_per_iteration << ","
              << final_residual
              << "\n";
}


// =========================================================
// Main
// =========================================================

int main(int argc, char* argv[])
{
    std::cout << std::setprecision(10);


    // -----------------------------------------------------
    // Command-line interface
    //
    // Validation:
    //   ./poisson_2d_serial_vanda validation N
    //
    // Benchmark:
    //   ./poisson_2d_serial_vanda benchmark N K
    // -----------------------------------------------------

    if (argc < 3)
    {
        std::cerr
            << "Usage:\n"
            << "  " << argv[0]
            << " validation <N>\n"
            << "  " << argv[0]
            << " benchmark <N> <K>\n";

        return 1;
    }


    const std::string mode = argv[1];

    int N = 0;
    int K = 0;

    try
    {
        N = std::stoi(argv[2]);

        if (mode == "benchmark")
        {
            if (argc != 4)
            {
                std::cerr
                    << "ERROR: benchmark mode requires N and K.\n";

                return 1;
            }

            K = std::stoi(argv[3]);
        }
        else if (mode == "validation")
        {
            if (argc != 3)
            {
                std::cerr
                    << "ERROR: validation mode requires only N.\n";

                return 1;
            }
        }
        else
        {
            std::cerr
                << "ERROR: Unknown mode: "
                << mode << "\n";

            return 1;
        }
    }
    catch (...)
    {
        std::cerr
            << "ERROR: N and K must be valid integers.\n";

        return 1;
    }


    if (N <= 0)
    {
        std::cerr
            << "ERROR: N must be positive.\n";

        return 1;
    }

    if (mode == "benchmark" && K <= 0)
    {
        std::cerr
            << "ERROR: K must be positive.\n";

        return 1;
    }


    std::cout << "=====================================\n";
    std::cout << "2D Poisson Serial Solver\n";
    std::cout << "=====================================\n";

    std::cout << "Mode = "
              << mode << "\n";


    if (mode == "validation")
    {
        run_validation(N);
    }
    else
    {
        run_benchmark(N, K);
    }


    return 0;
}
