# MPI test for transpose
# This file is executed under mpiexec by runtests.jl

using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra: norm
using Test

include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

const TOL = 1e-12

ts = @testset QuietTestSet "Transpose" begin

if rank == 0
    println("[test] Transpose")
    flush(stdout)
end

m, n = 10, 8
I_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 3, 5, 7, 9]
J_idx = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2,   3, 5, 7, 1, 4]
V = Float64.(1:length(I_idx))
A = sparse(I_idx, J_idx, V, m, n)

Adist = SparseMatrixMPI{Float64}(A)
plan = LinearAlgebraMPI.TransposePlan(Adist)
ATdist = LinearAlgebraMPI.execute_plan!(plan, Adist)
AT_ref = sparse(A')
AT_ref_dist = SparseMatrixMPI{Float64}(AT_ref)
err = norm(ATdist - AT_ref_dist, Inf)
@test err < TOL

MPI.Barrier(comm)

if rank == 0
    println("[test] Transpose with ComplexF64")
    flush(stdout)
end

V_c = ComplexF64.(1:length(I_idx)) .+ im .* ComplexF64.(length(I_idx):-1:1)
A_c = sparse(I_idx, J_idx, V_c, m, n)

Adist_c = SparseMatrixMPI{ComplexF64}(A_c)
plan_c = LinearAlgebraMPI.TransposePlan(Adist_c)
ATdist_c = LinearAlgebraMPI.execute_plan!(plan_c, Adist_c)
AT_ref_c = sparse(transpose(A_c))
AT_ref_dist_c = SparseMatrixMPI{ComplexF64}(AT_ref_c)
err_c = norm(ATdist_c - AT_ref_dist_c, Inf)
@test err_c < TOL

MPI.Barrier(comm)

if rank == 0
    println("[test] Square matrix transpose")
    flush(stdout)
end

n2 = 8
I_idx2 = [1:n2; 1:n2-1; 2:n2]
J_idx2 = [1:n2; 2:n2; 1:n2-1]
V2 = [2.0*ones(Float64, n2); 0.3*ones(n2-1); 0.7*ones(n2-1)]
A2 = sparse(I_idx2, J_idx2, V2, n2, n2)

Adist2 = SparseMatrixMPI{Float64}(A2)
plan2 = LinearAlgebraMPI.TransposePlan(Adist2)
ATdist2 = LinearAlgebraMPI.execute_plan!(plan2, Adist2)
AT_ref2 = sparse(A2')
AT_ref_dist2 = SparseMatrixMPI{Float64}(AT_ref2)
err2 = norm(ATdist2 - AT_ref_dist2, Inf)
@test err2 < TOL

MPI.Barrier(comm)

end  # QuietTestSet

# Aggregate counts across ranks
local_counts = [ts.counts[:pass], ts.counts[:fail], ts.counts[:error], ts.counts[:broken], ts.counts[:skip]]
global_counts = similar(local_counts)
MPI.Allreduce!(local_counts, global_counts, +, comm)

if rank == 0
    println("Test Summary: Transpose | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")
    flush(stdout)
end

MPI.Barrier(comm)
MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
