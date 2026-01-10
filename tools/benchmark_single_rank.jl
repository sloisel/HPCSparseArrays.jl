#!/usr/bin/env julia
#
# Benchmark HPCLinearAlgebra types vs native Julia types (single rank)
#
# This script identifies performance bottlenecks in the distributed code path
# by comparing against native Julia operations with 1 MPI rank.
#
# Usage: mpiexec -n 1 julia --project=tools tools/benchmark_single_rank.jl
#

using MPI
MPI.Init()

using SparseArrays
using LinearAlgebra
using Printf
using Dates
using Random
using BenchmarkTools
using HPCLinearAlgebra

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nranks = MPI.Comm_size(comm)

# Test sizes - limited to avoid long runtimes
const SIZES = [100, 1000, 10000]
const NNZ_PER_ROW = 10  # Realistic sparsity for sparse matrices

# BenchmarkTools config - faster benchmarks
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1

# ============================================================================
# Test Matrix/Vector Generators
# ============================================================================

"""Generate a random dense vector."""
function generate_vector(n, ::Type{T}) where T
    if T <: Complex
        return randn(n) + im * randn(n)
    else
        return randn(n)
    end
end

"""Generate a sparse matrix with approximately nnz_per_row nonzeros per row."""
function generate_sparse(n, ::Type{T}; nnz_per_row=NNZ_PER_ROW) where T
    I = Int[]
    J = Int[]
    V = T[]

    for i in 1:n
        # Random column indices for this row
        ncols = min(nnz_per_row, n)
        cols = randperm(n)[1:ncols]
        for j in cols
            push!(I, i)
            push!(J, j)
            if T <: Complex
                push!(V, randn() + im * randn())
            else
                push!(V, randn())
            end
        end
    end

    A = sparse(I, J, V, n, n)
    # Make symmetric for testing (not strictly necessary but realistic)
    return A + A'
end

"""Generate a dense matrix."""
function generate_dense(n, ::Type{T}) where T
    if T <: Complex
        return randn(n, n) + im * randn(n, n)
    else
        return randn(n, n)
    end
end

# ============================================================================
# Benchmark Functions
# ============================================================================

function format_time(t)
    if t >= 1.0
        return @sprintf("%.3f s ", t)
    elseif t >= 0.001
        return @sprintf("%.3f ms", t * 1000)
    elseif t >= 0.000001
        return @sprintf("%.3f μs", t * 1_000_000)
    else
        return @sprintf("%.3f ns", t * 1_000_000_000)
    end
end

function format_ratio(r)
    if r >= 2.0
        return @sprintf("%.2fx  ", r)  # RED - bad
    elseif r >= 1.2
        return @sprintf("%.2fx  ", r)  # YELLOW - needs work
    else
        return @sprintf("%.2fx  ", r)  # GREEN - acceptable
    end
end

function ratio_color(r)
    if r >= 2.0
        return "31"  # Red
    elseif r >= 1.2
        return "33"  # Yellow
    else
        return "32"  # Green
    end
end

function colored_ratio(r)
    color = ratio_color(r)
    return "\e[$(color)m$(format_ratio(r))\e[0m"
end

"""Benchmark vector operations."""
function benchmark_vector_ops(n, ::Type{T}) where T
    # Generate native vectors
    v_native = generate_vector(n, T)
    w_native = generate_vector(n, T)

    # Create MPI versions
    v_mpi = HPCVector(v_native)
    w_mpi = HPCVector(w_native)

    results = Dict{String, Tuple{Float64, Float64}}()

    # dot product
    t_native = @belapsed dot($v_native, $w_native)
    t_mpi = @belapsed dot($v_mpi, $w_mpi)
    results["dot(v,w)"] = (t_mpi, t_native)

    # norm
    t_native = @belapsed norm($v_native)
    t_mpi = @belapsed norm($v_mpi)
    results["norm(v)"] = (t_mpi, t_native)

    # norm(v, 1)
    t_native = @belapsed norm($v_native, 1)
    t_mpi = @belapsed norm($v_mpi, 1)
    results["norm(v,1)"] = (t_mpi, t_native)

    # addition
    t_native = @belapsed $v_native + $w_native
    t_mpi = @belapsed $v_mpi + $w_mpi
    results["v + w"] = (t_mpi, t_native)

    # subtraction
    t_native = @belapsed $v_native - $w_native
    t_mpi = @belapsed $v_mpi - $w_mpi
    results["v - w"] = (t_mpi, t_native)

    # scalar multiplication
    alpha = T <: Complex ? (2.0 + 1.0im) : 2.0
    t_native = @belapsed $alpha * $v_native
    t_mpi = @belapsed $alpha * $v_mpi
    results["α * v"] = (t_mpi, t_native)

    # sum
    t_native = @belapsed sum($v_native)
    t_mpi = @belapsed sum($v_mpi)
    results["sum(v)"] = (t_mpi, t_native)

    return results
end

"""Benchmark sparse matrix-vector multiplication."""
function benchmark_sparse_matvec(n, ::Type{T}) where T
    # Generate native sparse matrix and vector
    A_native = generate_sparse(n, T)
    x_native = generate_vector(n, T)

    # Create MPI versions
    A_mpi = HPCSparseMatrix{T}(A_native)
    x_mpi = HPCVector(x_native)

    results = Dict{String, Tuple{Float64, Float64}}()

    # Warmup to create and cache the plan
    _ = A_mpi * x_mpi

    # Sparse A * x
    t_native = @belapsed $A_native * $x_native
    t_mpi = @belapsed $A_mpi * $x_mpi
    results["sparse A*x"] = (t_mpi, t_native)

    return results
end

"""Benchmark dense matrix-vector multiplication."""
function benchmark_dense_matvec(n, ::Type{T}) where T
    # For dense matrices, limit size due to memory
    if n > 10000
        return Dict{String, Tuple{Float64, Float64}}()
    end

    # Generate native dense matrix and vector
    A_native = generate_dense(n, T)
    x_native = generate_vector(n, T)

    # Create MPI versions
    A_mpi = HPCMatrix(A_native)
    x_mpi = HPCVector(x_native)

    results = Dict{String, Tuple{Float64, Float64}}()

    # Warmup to create and cache the plan
    _ = A_mpi * x_mpi

    # Dense A * x
    t_native = @belapsed $A_native * $x_native
    t_mpi = @belapsed $A_mpi * $x_mpi
    results["dense A*x"] = (t_mpi, t_native)

    return results
end

"""Benchmark sparse matrix-matrix multiplication."""
function benchmark_sparse_matmat(n, ::Type{T}) where T
    # For matrix-matrix, limit size due to fill-in and long runtime
    if n > 1000
        return Dict{String, Tuple{Float64, Float64}}()
    end

    # Generate native sparse matrices
    A_native = generate_sparse(n, T)
    B_native = generate_sparse(n, T)

    # Create MPI versions
    A_mpi = HPCSparseMatrix{T}(A_native)
    B_mpi = HPCSparseMatrix{T}(B_native)

    results = Dict{String, Tuple{Float64, Float64}}()

    # Warmup to create and cache the plan
    _ = A_mpi * B_mpi

    # Sparse A * B
    t_native = @belapsed $A_native * $B_native
    t_mpi = @belapsed $A_mpi * $B_mpi
    results["sparse A*B"] = (t_mpi, t_native)

    return results
end

# ============================================================================
# Main Benchmark Runner
# ============================================================================

function run_benchmarks()
    println("=" ^ 75)
    println("HPCLinearAlgebra Single-Rank Performance Benchmark")
    println("=" ^ 75)
    println("Date: $(Dates.now())")
    println("MPI ranks: $nranks")
    println("Julia threads: $(Threads.nthreads())")
    println("Julia version: $(VERSION)")
    println()

    if nranks != 1
        println("WARNING: This benchmark is designed for 1 MPI rank.")
        println("         Results with multiple ranks measure distributed overhead.")
        println()
    end

    # Collect all results
    all_results = Dict{Tuple{Int, DataType, String}, Tuple{Float64, Float64}}()

    for T in [Float64, ComplexF64]
        type_name = T == Float64 ? "Float64" : "ComplexF64"

        println("=" ^ 75)
        println("Element Type: $type_name")
        println("=" ^ 75)
        println()

        for n in SIZES
            println("-" ^ 75)
            println("Size n = $n")
            println("-" ^ 75)
            println(@sprintf("%-20s %15s %15s %10s", "Operation", "MPI Time", "Native Time", "Ratio"))
            println("-" ^ 75)

            # Vector operations
            vec_results = benchmark_vector_ops(n, T)
            for (op, (t_mpi, t_native)) in sort(collect(vec_results), by=x->x[1])
                ratio = t_mpi / t_native
                println(@sprintf("%-20s %15s %15s %s", op, format_time(t_mpi), format_time(t_native), colored_ratio(ratio)))
                all_results[(n, T, op)] = (t_mpi, t_native)
            end

            # Sparse matrix-vector
            sparse_mv_results = benchmark_sparse_matvec(n, T)
            for (op, (t_mpi, t_native)) in sort(collect(sparse_mv_results), by=x->x[1])
                ratio = t_mpi / t_native
                println(@sprintf("%-20s %15s %15s %s", op, format_time(t_mpi), format_time(t_native), colored_ratio(ratio)))
                all_results[(n, T, op)] = (t_mpi, t_native)
            end

            # Dense matrix-vector (limited sizes)
            dense_mv_results = benchmark_dense_matvec(n, T)
            for (op, (t_mpi, t_native)) in sort(collect(dense_mv_results), by=x->x[1])
                ratio = t_mpi / t_native
                println(@sprintf("%-20s %15s %15s %s", op, format_time(t_mpi), format_time(t_native), colored_ratio(ratio)))
                all_results[(n, T, op)] = (t_mpi, t_native)
            end

            # Sparse matrix-matrix (limited sizes)
            sparse_mm_results = benchmark_sparse_matmat(n, T)
            for (op, (t_mpi, t_native)) in sort(collect(sparse_mm_results), by=x->x[1])
                ratio = t_mpi / t_native
                println(@sprintf("%-20s %15s %15s %s", op, format_time(t_mpi), format_time(t_native), colored_ratio(ratio)))
                all_results[(n, T, op)] = (t_mpi, t_native)
            end

            println()
        end
    end

    # Summary of worst offenders
    println("=" ^ 75)
    println("Summary: Operations with >2x overhead (sorted by ratio)")
    println("=" ^ 75)
    println()

    # Collect and sort by ratio
    sorted_results = sort(collect(all_results), by=x -> x[2][1] / x[2][2], rev=true)

    println(@sprintf("%-15s %-12s %-20s %10s", "Size", "Type", "Operation", "Ratio"))
    println("-" ^ 60)

    count = 0
    for ((n, T, op), (t_mpi, t_native)) in sorted_results
        ratio = t_mpi / t_native
        if ratio >= 2.0 || count < 10  # Show at least top 10
            type_name = T == Float64 ? "Float64" : "ComplexF64"
            println(@sprintf("%-15d %-12s %-20s %s", n, type_name, op, colored_ratio(ratio)))
            count += 1
        end
        if count >= 20
            break
        end
    end

    println()
    println("Legend: \e[32mgreen\e[0m < 1.2x, \e[33myellow\e[0m 1.2-2x, \e[31mred\e[0m > 2x")
    println()
    println("=" ^ 75)
end

# Run benchmarks
run_benchmarks()
