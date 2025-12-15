#!/usr/bin/env julia
#
# Benchmark LinearAlgebraMPI.jl vs SafePETSc.jl for sparse operations.
#
# This script compares performance on:
# 1. Matrix-matrix multiplication (A * A)
# 2. Sparse direct solves (LDLT factorization)
#
# Usage: mpiexec -n 4 julia --project=tools --threads=2 tools/benchmark_vs_petsc.jl
#
# First time setup:
#   cd tools && julia --project=. -e 'using Pkg; Pkg.instantiate()'

using MPI
MPI.Init()

using SparseArrays
using LinearAlgebra
using Printf
using Statistics
using Dates
using BenchmarkTools
using LinearAlgebraMPI
using SafePETSc

SafePETSc.Init()

# Configure PETSc to use MUMPS direct solver for fair comparison with LinearAlgebraMPI's direct solver
# MUMPS is bundled with petsc_jll
SafePETSc.petsc_options_insert_string("-MPIAIJ_ksp_type preonly -MPIAIJ_pc_type lu -MPIAIJ_pc_factor_mat_solver_type mumps")

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nranks = MPI.Comm_size(comm)

"""
    laplacian_2d_sparse(n)

Create a 2D Laplacian matrix for an approximately sqrt(n) × sqrt(n) grid.
Returns a SparseMatrixCSC.
"""
function laplacian_2d_sparse(n)
    grid_size = max(1, round(Int, sqrt(n)))
    e = ones(grid_size)
    L1D = spdiagm(-1 => -e[1:end-1], 0 => 2*e, 1 => -e[1:end-1])
    I_g = sparse(I, grid_size, grid_size)
    L2D = kron(I_g, L1D) + kron(L1D, I_g)
    return L2D
end

"""
    benchmark_matmat_linearalgebrampi(A_local, n_samples)

Benchmark LinearAlgebraMPI.jl matrix-matrix multiplication (A * A).
Returns median time in seconds.
"""
function benchmark_matmat_linearalgebrampi(A_local, n_samples)
    A_mpi = SparseMatrixMPI{Float64}(A_local)

    # Warmup
    C = A_mpi * A_mpi

    # Benchmark
    MPI.Barrier(comm)
    times = Float64[]
    for _ in 1:n_samples
        MPI.Barrier(comm)
        t_start = MPI.Wtime()
        C = A_mpi * A_mpi
        MPI.Barrier(comm)
        t_end = MPI.Wtime()
        push!(times, t_end - t_start)
    end

    return median(times)
end

"""
    benchmark_matmat_safepetsc(A_local, n_samples)

Benchmark SafePETSc.jl matrix-matrix multiplication (A * A).
Returns median time in seconds.
"""
function benchmark_matmat_safepetsc(A_local, n_samples)
    A_petsc = SafePETSc.Mat_uniform(A_local)

    # Warmup
    C = A_petsc * A_petsc

    # Benchmark
    MPI.Barrier(comm)
    times = Float64[]
    for _ in 1:n_samples
        MPI.Barrier(comm)
        t_start = MPI.Wtime()
        C = A_petsc * A_petsc
        MPI.Barrier(comm)
        t_end = MPI.Wtime()
        push!(times, t_end - t_start)
    end

    return median(times)
end

"""
    benchmark_linearalgebrampi(A_local, b_local, n_samples)

Benchmark LinearAlgebraMPI.jl solve time.
Returns median time in seconds for the solve phase only.
"""
function benchmark_linearalgebrampi(A_local, b_local, n_samples)
    # Create distributed matrix and vector
    A_mpi = SparseMatrixMPI{Float64}(A_local)
    b_mpi = VectorMPI(b_local)

    # Warmup: first factorization generates the plan
    F = ldlt(A_mpi)
    x = F \ b_mpi

    # Verify solution
    residual = norm(Vector(A_mpi * x - b_mpi)) / norm(b_local)
    if rank == 0 && residual > 1e-10
        println("  Warning: LinearAlgebraMPI residual = $residual")
    end
    finalize!(F)  # Clean up warmup factorization

    # Benchmark solve phase only (factorization already done in warmup conceptually,
    # but LDLT doesn't cache, so we measure factorization + solve)
    # For a fair comparison with PETSc (which also does factorization in KSP setup),
    # we measure: factorization time separately, then solve time

    # Benchmark factorization
    MPI.Barrier(comm)
    fact_times = Float64[]
    F = nothing
    for _ in 1:n_samples
        if F !== nothing
            finalize!(F)  # Clean up previous factorization
        end
        MPI.Barrier(comm)
        t_start = MPI.Wtime()
        F = ldlt(A_mpi)
        MPI.Barrier(comm)
        t_end = MPI.Wtime()
        push!(fact_times, t_end - t_start)
    end

    # Benchmark solve (using last factorization from above)
    MPI.Barrier(comm)
    solve_times = Float64[]
    for _ in 1:n_samples
        MPI.Barrier(comm)
        t_start = MPI.Wtime()
        x = F \ b_mpi
        MPI.Barrier(comm)
        t_end = MPI.Wtime()
        push!(solve_times, t_end - t_start)
    end

    finalize!(F)  # Clean up final factorization
    return median(fact_times), median(solve_times)
end

"""
    benchmark_safepetsc(A_local, b_local, n_samples)

Benchmark SafePETSc.jl solve time.
Returns median time in seconds for KSP setup and solve phases.
"""
function benchmark_safepetsc(A_local, b_local, n_samples)
    # Create PETSc matrix and vector
    A_petsc = SafePETSc.Mat_uniform(A_local)
    b_petsc = SafePETSc.Vec_uniform(b_local)

    # Warmup
    x = A_petsc \ b_petsc

    # Verify solution
    x_julia = SafePETSc.J(x)
    residual = norm(A_local * x_julia - b_local) / norm(b_local)
    if rank == 0 && residual > 1e-10
        println("  Warning: SafePETSc residual = $residual")
    end

    # Benchmark KSP setup (includes factorization for direct solvers)
    GC.gc()
    MPI.Barrier(comm)
    setup_times = Float64[]
    for _ in 1:n_samples
        GC.gc()
        MPI.Barrier(comm)
        t_start = MPI.Wtime()
        ksp = SafePETSc.KSP(A_petsc)
        MPI.Barrier(comm)
        t_end = MPI.Wtime()
        push!(setup_times, t_end - t_start)
    end

    # Benchmark solve (using pre-computed KSP)
    ksp = SafePETSc.KSP(A_petsc)
    x_petsc = SafePETSc.zeros_like(b_petsc)

    GC.gc()
    MPI.Barrier(comm)
    solve_times = Float64[]
    for _ in 1:n_samples
        GC.gc()
        MPI.Barrier(comm)
        t_start = MPI.Wtime()
        LinearAlgebra.ldiv!(ksp, x_petsc, b_petsc)
        MPI.Barrier(comm)
        t_end = MPI.Wtime()
        push!(solve_times, t_end - t_start)
    end

    return median(setup_times), median(solve_times)
end

function format_time(t)
    if t >= 1.0
        return @sprintf("%.3f s", t)
    elseif t >= 0.001
        return @sprintf("%.3f ms", t * 1000)
    else
        return @sprintf("%.3f μs", t * 1_000_000)
    end
end

function run_benchmarks()
    n = 10_000  # Degrees of freedom
    n_samples = 5  # Number of benchmark samples

    if rank == 0
        println("=" ^ 70)
        println("LinearAlgebraMPI.jl vs SafePETSc.jl Benchmark")
        println("=" ^ 70)
        println("Date: $(Dates.now())")
        println("MPI ranks: $nranks")
        println("Julia threads: $(Threads.nthreads())")
        println("BLAS threads: $(BLAS.get_num_threads())")
        println()
    end

    # Create 2D Laplacian
    A_local = laplacian_2d_sparse(n)
    actual_n = size(A_local, 1)
    b_local = ones(actual_n)

    if rank == 0
        println("Problem size: $actual_n × $actual_n")
        println("Non-zeros: $(nnz(A_local))")
        println("Samples per benchmark: $n_samples")
        println()
    end

    # ========================================================================
    # Matrix-Matrix Multiplication Benchmark (A * A)
    # ========================================================================
    if rank == 0
        println("=" ^ 70)
        println("Matrix-Matrix Multiplication (A * A)")
        println("=" ^ 70)
        println()
    end

    # Benchmark LinearAlgebraMPI.jl matmat
    if rank == 0
        println("Benchmarking LinearAlgebraMPI.jl (A * A)...")
    end
    lampi_matmat_time = benchmark_matmat_linearalgebrampi(A_local, n_samples)
    if rank == 0
        println("  Time: $(format_time(lampi_matmat_time))")
        println()
    end

    # Benchmark SafePETSc.jl matmat
    if rank == 0
        println("Benchmarking SafePETSc.jl (A * A)...")
    end
    petsc_matmat_time = benchmark_matmat_safepetsc(A_local, n_samples)
    if rank == 0
        println("  Time: $(format_time(petsc_matmat_time))")
        println()
    end

    # Matmat summary
    if rank == 0
        matmat_speedup = petsc_matmat_time / lampi_matmat_time
        println(@sprintf("Matrix-Matrix Speedup (LinearAlgebraMPI vs SafePETSc): %.2fx", matmat_speedup))
        println()
    end

    # ========================================================================
    # Sparse Direct Solve Benchmark (LDLT)
    # ========================================================================
    if rank == 0
        println("=" ^ 70)
        println("Sparse Direct Solve (LDLT)")
        println("=" ^ 70)
        println()
    end

    # Benchmark LinearAlgebraMPI.jl
    if rank == 0
        println("Benchmarking LinearAlgebraMPI.jl (LDLT)...")
    end
    lampi_fact_time, lampi_solve_time = benchmark_linearalgebrampi(A_local, b_local, n_samples)

    if rank == 0
        println("  Factorization: $(format_time(lampi_fact_time))")
        println("  Solve:         $(format_time(lampi_solve_time))")
        println("  Total:         $(format_time(lampi_fact_time + lampi_solve_time))")
        println()
    end

    # Benchmark SafePETSc.jl
    if rank == 0
        println("Benchmarking SafePETSc.jl (KSP direct solver)...")
    end
    petsc_setup_time, petsc_solve_time = benchmark_safepetsc(A_local, b_local, n_samples)

    if rank == 0
        println("  KSP Setup:     $(format_time(petsc_setup_time))")
        println("  Solve:         $(format_time(petsc_solve_time))")
        println("  Total:         $(format_time(petsc_setup_time + petsc_solve_time))")
        println()
    end

    # Summary comparison
    if rank == 0
        println("=" ^ 70)
        println("Summary (Solve Time Only)")
        println("=" ^ 70)
        println()
        println(@sprintf("%-25s %15s %15s", "Library", "Solve Time", "Speedup"))
        println("-" ^ 55)

        # Use PETSc as baseline
        baseline = petsc_solve_time
        println(@sprintf("%-25s %15s %15s", "SafePETSc.jl", format_time(petsc_solve_time), "1.00x"))
        speedup = petsc_solve_time / lampi_solve_time
        println(@sprintf("%-25s %15s %15.2fx", "LinearAlgebraMPI.jl", format_time(lampi_solve_time), speedup))

        println()
        println("=" ^ 70)
        println("Summary (Factorization + Solve)")
        println("=" ^ 70)
        println()
        println(@sprintf("%-25s %15s %15s %15s", "Library", "Factorization", "Solve", "Total"))
        println("-" ^ 70)

        petsc_total = petsc_setup_time + petsc_solve_time
        lampi_total = lampi_fact_time + lampi_solve_time

        println(@sprintf("%-25s %15s %15s %15s", "SafePETSc.jl",
                        format_time(petsc_setup_time), format_time(petsc_solve_time), format_time(petsc_total)))
        println(@sprintf("%-25s %15s %15s %15s", "LinearAlgebraMPI.jl",
                        format_time(lampi_fact_time), format_time(lampi_solve_time), format_time(lampi_total)))

        total_speedup = petsc_total / lampi_total
        println()
        println(@sprintf("Total speedup (LinearAlgebraMPI vs SafePETSc): %.2fx", total_speedup))
        println()
    end

    # Save results to file
    if rank == 0
        results_file = joinpath(@__DIR__, "benchmark_vs_petsc_results.txt")
        open(results_file, "w") do f
            println(f, "# LinearAlgebraMPI.jl vs SafePETSc.jl Benchmark Results")
            println(f, "# Date: $(Dates.now())")
            println(f, "# MPI ranks: $nranks")
            println(f, "# Julia threads: $(Threads.nthreads())")
            println(f, "# Problem size: $actual_n x $actual_n, nnz=$(nnz(A_local))")
            println(f, "#")
            println(f, "# Matrix-Matrix Multiplication (A * A)")
            println(f, "# library,matmat_time")
            println(f, "SafePETSc,$petsc_matmat_time")
            println(f, "LinearAlgebraMPI,$lampi_matmat_time")
            println(f, "#")
            println(f, "# Sparse Direct Solve (LDLT)")
            println(f, "# library,factorization_time,solve_time,total_time")
            println(f, "SafePETSc,$petsc_setup_time,$petsc_solve_time,$(petsc_setup_time + petsc_solve_time)")
            println(f, "LinearAlgebraMPI,$lampi_fact_time,$lampi_solve_time,$(lampi_fact_time + lampi_solve_time)")
        end
        println("Results saved to: $results_file")
    end
end

run_benchmarks()

# MPI.Finalize() is called automatically by MPI.jl's atexit hook
