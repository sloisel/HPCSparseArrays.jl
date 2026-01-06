# MPI test for matrix multiplication
# This file is executed under mpiexec by runtests.jl
# Parameterized over scalar types and backends (CPU/GPU)

# Check Metal availability BEFORE loading MPI
const METAL_AVAILABLE = try
    using Metal
    Metal.functional()
catch e
    false
end

using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra: norm
using Test

include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

include(joinpath(@__DIR__, "test_utils.jl"))
using .TestUtils

comm = MPI.COMM_WORLD

ts = @testset QuietTestSet "Matrix Multiplication" begin

for (T, to_backend, backend_name) in TestUtils.ALL_CONFIGS
    TOL = TestUtils.tolerance(T)
    VT, ST, MT = TestUtils.expected_types(T, to_backend)

    println(io0(), "[test] Matrix multiplication ($T, $backend_name)")

    n = 8
    A = TestUtils.tridiagonal_matrix(T, n)

    # Second tridiagonal matrix with different values
    I_B = [1:n; 1:n-1; 2:n]
    J_B = [1:n; 2:n; 1:n-1]
    V_B = if T <: Complex
        T.([1.5*ones(n); 0.25*ones(n-1); 0.25*ones(n-1)]) .+
        im .* T.([-0.1*ones(n); 0.1*ones(n-1); 0.1*ones(n-1)])
    else
        T.([1.5*ones(n); 0.25*ones(n-1); 0.25*ones(n-1)])
    end
    B = sparse(I_B, J_B, V_B, n, n)

    Adist = to_backend(SparseMatrixMPI{T}(A))
    Bdist = to_backend(SparseMatrixMPI{T}(B))
    Cdist = assert_type(Adist * Bdist, ST)
    C_ref = A * B
    C_ref_dist = to_backend(SparseMatrixMPI{T}(C_ref))

    # Convert to CPU for norm comparison
    Cdist_cpu = TestUtils.to_cpu(Cdist)
    C_ref_dist_cpu = TestUtils.to_cpu(C_ref_dist)
    err = assert_uniform(norm(Cdist_cpu - C_ref_dist_cpu, Inf), name="matmul_err")
    @test err < TOL


    println(io0(), "[test] Non-square matrix multiplication ($T, $backend_name)")

    m, k, n2 = 6, 8, 10
    I_A2 = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
    J_A2 = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2]
    V_A2 = T <: Complex ? T.(1:length(I_A2)) .+ im .* T.(length(I_A2):-1:1) : T.(1:length(I_A2))
    A2 = sparse(I_A2, J_A2, V_A2, m, k)

    I_B2 = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3]
    J_B2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    V_B2 = T <: Complex ? T.(1:length(I_B2)) .+ im .* T.(length(I_B2):-1:1) : T.(1:length(I_B2))
    B2 = sparse(I_B2, J_B2, V_B2, k, n2)

    Adist2 = to_backend(SparseMatrixMPI{T}(A2))
    Bdist2 = to_backend(SparseMatrixMPI{T}(B2))
    Cdist2 = assert_type(Adist2 * Bdist2, ST)
    C_ref2 = A2 * B2
    C_ref_dist2 = to_backend(SparseMatrixMPI{T}(C_ref2))

    Cdist2_cpu = TestUtils.to_cpu(Cdist2)
    C_ref_dist2_cpu = TestUtils.to_cpu(C_ref_dist2)
    err2 = assert_uniform(norm(Cdist2_cpu - C_ref_dist2_cpu, Inf), name="nonsquare_matmul_err")
    @test err2 < TOL

end  # for (T, to_backend, backend_name)

end  # QuietTestSet

# Aggregate counts across ranks
local_counts = [
    get(ts.counts, :pass, 0),
    get(ts.counts, :fail, 0),
    get(ts.counts, :error, 0),
    get(ts.counts, :broken, 0),
    get(ts.counts, :skip, 0),
]
global_counts = similar(local_counts)
MPI.Allreduce!(local_counts, global_counts, +, comm)

println(io0(), "Test Summary: Matrix Multiplication | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")

MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
