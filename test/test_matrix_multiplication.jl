# MPI test for matrix multiplication
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

const TOL = 1e-12

ts = @testset QuietTestSet "Matrix Multiplication" begin

println(io0(), "[test] Matrix multiplication")

n = 8
I_A = [1:n; 1:n-1; 2:n]
J_A = [1:n; 2:n; 1:n-1]
V_A = [2.0*ones(Float64, n); -0.5*ones(n-1); -0.5*ones(n-1)]
A = sparse(I_A, J_A, V_A, n, n)

I_B = [1:n; 1:n-1; 2:n]
J_B = [1:n; 2:n; 1:n-1]
V_B = [1.5*ones(Float64, n); 0.25*ones(n-1); 0.25*ones(n-1)]
B = sparse(I_B, J_B, V_B, n, n)

Adist = SparseMatrixMPI{Float64}(A)
Bdist = SparseMatrixMPI{Float64}(B)
Cdist = Adist * Bdist
C_ref = A * B
C_ref_dist = SparseMatrixMPI{Float64}(C_ref)
err = norm(Cdist - C_ref_dist, Inf)
@test err < TOL

println(io0(), "[test] Matrix multiplication with ComplexF64")

V_A_c = ComplexF64.([2.0*ones(n); -0.5*ones(n-1); -0.5*ones(n-1)]) .+
        im .* ComplexF64.([0.1*ones(n); 0.2*ones(n-1); -0.2*ones(n-1)])
A_c = sparse(I_A, J_A, V_A_c, n, n)

V_B_c = ComplexF64.([1.5*ones(n); 0.25*ones(n-1); 0.25*ones(n-1)]) .+
        im .* ComplexF64.([-0.1*ones(n); 0.1*ones(n-1); 0.1*ones(n-1)])
B_c = sparse(I_B, J_B, V_B_c, n, n)

Adist_c = SparseMatrixMPI{ComplexF64}(A_c)
Bdist_c = SparseMatrixMPI{ComplexF64}(B_c)
Cdist_c = Adist_c * Bdist_c
C_ref_c = A_c * B_c
C_ref_dist_c = SparseMatrixMPI{ComplexF64}(C_ref_c)
err_c = norm(Cdist_c - C_ref_dist_c, Inf)
@test err_c < TOL

println(io0(), "[test] Non-square matrix multiplication")

m, k, n2 = 6, 8, 10
I_A2 = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
J_A2 = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2]
V_A2 = Float64.(1:length(I_A2))
A2 = sparse(I_A2, J_A2, V_A2, m, k)

I_B2 = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3]
J_B2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
V_B2 = Float64.(1:length(I_B2))
B2 = sparse(I_B2, J_B2, V_B2, k, n2)

Adist2 = SparseMatrixMPI{Float64}(A2)
Bdist2 = SparseMatrixMPI{Float64}(B2)
Cdist2 = Adist2 * Bdist2
C_ref2 = A2 * B2
C_ref_dist2 = SparseMatrixMPI{Float64}(C_ref2)
err2 = norm(Cdist2 - C_ref_dist2, Inf)
@test err2 < TOL

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
