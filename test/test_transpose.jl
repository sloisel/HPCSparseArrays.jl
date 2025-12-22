# MPI test for transpose
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

ts = @testset QuietTestSet "Transpose" begin

for (T, to_backend, backend_name) in TestUtils.ALL_CONFIGS
    TOL = TestUtils.tolerance(T)

    println(io0(), "[test] Transpose ($T, $backend_name)")

    m, n = 10, 8
    I_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 3, 5, 7, 9]
    J_idx = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2,   3, 5, 7, 1, 4]
    V = T <: Complex ? T.(1:length(I_idx)) .+ im .* T.(length(I_idx):-1:1) : T.(1:length(I_idx))
    A = sparse(I_idx, J_idx, V, m, n)

    Adist = to_backend(SparseMatrixMPI{T}(A))
    plan = LinearAlgebraMPI.TransposePlan(Adist)
    ATdist = LinearAlgebraMPI.execute_plan!(plan, Adist)
    AT_ref = sparse(transpose(A))
    AT_ref_dist = to_backend(SparseMatrixMPI{T}(AT_ref))

    ATdist_cpu = TestUtils.to_cpu(ATdist)
    AT_ref_dist_cpu = TestUtils.to_cpu(AT_ref_dist)
    err = norm(ATdist_cpu - AT_ref_dist_cpu, Inf)
    @test err < TOL


    println(io0(), "[test] Square matrix transpose ($T, $backend_name)")

    n2 = 8
    I_idx2 = [1:n2; 1:n2-1; 2:n2]
    J_idx2 = [1:n2; 2:n2; 1:n2-1]
    V2 = if T <: Complex
        T.([2.0*ones(n2); 0.3*ones(n2-1); 0.7*ones(n2-1)]) .+
        im .* T.([0.1*ones(n2); -0.1*ones(n2-1); 0.2*ones(n2-1)])
    else
        T.([2.0*ones(n2); 0.3*ones(n2-1); 0.7*ones(n2-1)])
    end
    A2 = sparse(I_idx2, J_idx2, V2, n2, n2)

    Adist2 = to_backend(SparseMatrixMPI{T}(A2))
    plan2 = LinearAlgebraMPI.TransposePlan(Adist2)
    ATdist2 = LinearAlgebraMPI.execute_plan!(plan2, Adist2)
    AT_ref2 = sparse(transpose(A2))
    AT_ref_dist2 = to_backend(SparseMatrixMPI{T}(AT_ref2))

    ATdist2_cpu = TestUtils.to_cpu(ATdist2)
    AT_ref_dist2_cpu = TestUtils.to_cpu(AT_ref_dist2)
    err2 = norm(ATdist2_cpu - AT_ref_dist2_cpu, Inf)
    @test err2 < TOL

end  # for (T, to_backend, backend_name)

end  # QuietTestSet

# Aggregate counts across ranks
local_counts = [ts.counts[:pass], ts.counts[:fail], ts.counts[:error], ts.counts[:broken], ts.counts[:skip]]
global_counts = similar(local_counts)
MPI.Allreduce!(local_counts, global_counts, +, comm)

println(io0(), "Test Summary: Transpose | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")

MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
