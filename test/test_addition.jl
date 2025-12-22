# MPI test for addition and subtraction
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

ts = @testset QuietTestSet "Addition" begin

for (T, to_backend, backend_name) in TestUtils.ALL_CONFIGS
    TOL = TestUtils.tolerance(T)

    println(io0(), "[test] Matrix addition ($T, $backend_name)")

    n = 8
    A = TestUtils.tridiagonal_matrix(T, n)

    # Second matrix with different values
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
    Cdist = Adist + Bdist
    C_ref = A + B
    C_ref_dist = to_backend(SparseMatrixMPI{T}(C_ref))

    Cdist_cpu = TestUtils.to_cpu(Cdist)
    C_ref_dist_cpu = TestUtils.to_cpu(C_ref_dist)
    err = norm(Cdist_cpu - C_ref_dist_cpu, Inf)
    @test err < TOL


    println(io0(), "[test] Matrix subtraction ($T, $backend_name)")

    I_A2 = [1:n; 1:n-1; 2:n]
    J_A2 = [1:n; 2:n; 1:n-1]
    V_A2 = if T <: Complex
        T.([3.0*ones(n); -0.7*ones(n-1); -0.7*ones(n-1)]) .+
        im .* T.([0.2*ones(n); 0.1*ones(n-1); -0.1*ones(n-1)])
    else
        T.([3.0*ones(n); -0.7*ones(n-1); -0.7*ones(n-1)])
    end
    A2 = sparse(I_A2, J_A2, V_A2, n, n)

    V_B2 = if T <: Complex
        T.([1.0*ones(n); 0.3*ones(n-1); 0.3*ones(n-1)]) .+
        im .* T.([0.1*ones(n); -0.1*ones(n-1); 0.1*ones(n-1)])
    else
        T.([1.0*ones(n); 0.3*ones(n-1); 0.3*ones(n-1)])
    end
    B2 = sparse(I_B, J_B, V_B2, n, n)

    Adist2 = to_backend(SparseMatrixMPI{T}(A2))
    Bdist2 = to_backend(SparseMatrixMPI{T}(B2))
    Cdist2 = Adist2 - Bdist2
    C_ref2 = A2 - B2
    C_ref_dist2 = to_backend(SparseMatrixMPI{T}(C_ref2))

    Cdist2_cpu = TestUtils.to_cpu(Cdist2)
    C_ref_dist2_cpu = TestUtils.to_cpu(C_ref_dist2)
    err2 = norm(Cdist2_cpu - C_ref_dist2_cpu, Inf)
    @test err2 < TOL


    println(io0(), "[test] Different sparsity patterns ($T, $backend_name)")

    I_A3 = [1, 1, 2, 3, 4, 5, 6, 7, 8]
    J_A3 = [1, 2, 2, 3, 4, 5, 6, 7, 8]
    V_A3 = T <: Complex ? T.(1:9) .+ im .* T.(9:-1:1) : T.(1:9)
    A3 = sparse(I_A3, J_A3, V_A3, n, n)

    I_B3 = [1, 2, 2, 3, 4, 5, 6, 7, 8]
    J_B3 = [1, 1, 2, 3, 4, 5, 6, 7, 8]
    V_B3 = T <: Complex ? T.(9:-1:1) .+ im .* T.(1:9) : T.(9:-1:1)
    B3 = sparse(I_B3, J_B3, V_B3, n, n)

    Adist3 = to_backend(SparseMatrixMPI{T}(A3))
    Bdist3 = to_backend(SparseMatrixMPI{T}(B3))
    Cdist3 = Adist3 + Bdist3
    C_ref3 = A3 + B3
    C_ref_dist3 = to_backend(SparseMatrixMPI{T}(C_ref3))

    Cdist3_cpu = TestUtils.to_cpu(Cdist3)
    C_ref_dist3_cpu = TestUtils.to_cpu(C_ref_dist3)
    err3 = norm(Cdist3_cpu - C_ref_dist3_cpu, Inf)
    @test err3 < TOL


    println(io0(), "[test] Cached addition path ($T, $backend_name)")

    # Test that repeating the same addition uses the cached plan
    Cdist3_repeat = Adist3 + Bdist3
    Cdist3_repeat_cpu = TestUtils.to_cpu(Cdist3_repeat)
    err3_repeat = norm(Cdist3_repeat_cpu - C_ref_dist3_cpu, Inf)
    @test err3_repeat < TOL


    println(io0(), "[test] Cached subtraction path ($T, $backend_name)")

    # Test that repeating the same subtraction uses the cached plan
    Ddist = Adist3 - Bdist3
    D_ref = A3 - B3
    D_ref_dist = to_backend(SparseMatrixMPI{T}(D_ref))
    D_ref_dist_cpu = TestUtils.to_cpu(D_ref_dist)
    Ddist_cpu = TestUtils.to_cpu(Ddist)
    err_sub1 = norm(Ddist_cpu - D_ref_dist_cpu, Inf)
    @test err_sub1 < TOL

    Ddist_repeat = Adist3 - Bdist3
    Ddist_repeat_cpu = TestUtils.to_cpu(Ddist_repeat)
    err_sub2 = norm(Ddist_repeat_cpu - D_ref_dist_cpu, Inf)
    @test err_sub2 < TOL

end  # for (T, to_backend, backend_name)

end  # QuietTestSet

# Aggregate counts across ranks
local_counts = [ts.counts[:pass], ts.counts[:fail], ts.counts[:error], ts.counts[:broken], ts.counts[:skip]]
global_counts = similar(local_counts)
MPI.Allreduce!(local_counts, global_counts, +, comm)

println(io0(), "Test Summary: Addition | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")

MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
