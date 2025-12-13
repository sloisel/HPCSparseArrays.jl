"""
Tests for distributed LU and LDLT factorization.
"""

using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra
using Test

include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

const TOL = 1e-10

# Create deterministic test matrices
function create_spd_tridiagonal(n::Int)
    # Symmetric positive definite tridiagonal matrix
    I_A = [1:n; 1:n-1; 2:n]
    J_A = [1:n; 2:n; 1:n-1]
    V_A = [4.0*ones(n); -1.0*ones(n-1); -1.0*ones(n-1)]
    return sparse(I_A, J_A, V_A, n, n)
end

function create_general_tridiagonal(n::Int)
    # General (unsymmetric) tridiagonal matrix
    I_A = [1:n; 1:n-1; 2:n]
    J_A = [1:n; 2:n; 1:n-1]
    V_A = [2.0*ones(n); -0.5*ones(n-1); -0.8*ones(n-1)]
    return sparse(I_A, J_A, V_A, n, n)
end

function create_symmetric_indefinite(n::Int)
    # Symmetric indefinite matrix
    I_A = [1:n; 1:n-1; 2:n]
    J_A = [1:n; 2:n; 1:n-1]
    # Alternating signs on diagonal
    diag_vals = [(-1.0)^i * 2.0 for i in 1:n]
    V_A = [diag_vals; -1.0*ones(n-1); -1.0*ones(n-1)]
    return sparse(I_A, J_A, V_A, n, n)
end

ts = @testset QuietTestSet "Distributed Factorization Tests" begin

# Test 1: LU factorization of small matrix
if rank == 0
    println("[test] LU factorization - small matrix")
    flush(stdout)
end

n = 8
A_full = create_general_tridiagonal(n)
A = SparseMatrixMPI{Float64}(A_full)

F = lu(A)
@test F isa LinearAlgebraMPI.LUFactorizationMPI{Float64}
@test size(F) == (n, n)

b_full = ones(n)
b = VectorMPI(b_full)
x = solve(F, b)

x_full = Vector(x)
residual = A_full * x_full - b_full
err = norm(residual, Inf)

if rank == 0
    println("  LU solve residual: $err")
end
@test err < TOL

MPI.Barrier(comm)

# Test 2: LDLT factorization of SPD matrix
if rank == 0
    println("[test] LDLT factorization - SPD matrix")
    flush(stdout)
end

n = 10
A_full = create_spd_tridiagonal(n)
A = SparseMatrixMPI{Float64}(A_full)

F = ldlt(A)
@test F isa LinearAlgebraMPI.LDLTFactorizationMPI{Float64}
@test size(F) == (n, n)

b_full = ones(n)
b = VectorMPI(b_full)
x = solve(F, b)

x_full = Vector(x)
residual = A_full * x_full - b_full
err = norm(residual, Inf)

if rank == 0
    println("  LDLT solve residual (SPD): $err")
end
@test err < TOL

MPI.Barrier(comm)

# Test 3: LDLT with symmetric indefinite matrix
if rank == 0
    println("[test] LDLT factorization - indefinite matrix")
    flush(stdout)
end

n = 8
A_full = create_symmetric_indefinite(n)
A = SparseMatrixMPI{Float64}(A_full)

F = ldlt(A)

b_full = collect(1.0:n)
b = VectorMPI(b_full)
x = solve(F, b)

x_full = Vector(x)
residual = A_full * x_full - b_full
err = norm(residual, Inf)

if rank == 0
    println("  LDLT solve residual (indefinite): $err")
end
@test err < TOL

MPI.Barrier(comm)

# Test 4: Plan reuse (same structure, different values)
if rank == 0
    println("[test] Plan reuse")
    flush(stdout)
end

n = 8
A1_full = create_spd_tridiagonal(n)
A1 = SparseMatrixMPI{Float64}(A1_full)
F1 = ldlt(A1; reuse_symbolic=true)

A2_full = create_spd_tridiagonal(n)
A2_full.nzval .*= 2.0
A2 = SparseMatrixMPI{Float64}(A2_full)
F2 = ldlt(A2; reuse_symbolic=true)

b_full = ones(n)
b = VectorMPI(b_full)

x1 = solve(F1, b)
x2 = solve(F2, b)

x1_full = Vector(x1)
x2_full = Vector(x2)

err1 = norm(A1_full * x1_full - b_full, Inf)
err2 = norm(A2_full * x2_full - b_full, Inf)

if rank == 0
    println("  Residual 1: $err1")
    println("  Residual 2: $err2")
end

@test err1 < TOL
@test err2 < TOL

MPI.Barrier(comm)

# Test 5: Complex-valued matrix (LU)
if rank == 0
    println("[test] LU factorization - complex")
    flush(stdout)
end

n = 6
A_full_real = create_general_tridiagonal(n)
A_full = Complex{Float64}.(A_full_real) + im * spdiagm(0 => 0.1*ones(n))
A = SparseMatrixMPI{ComplexF64}(A_full)

F = lu(A)

b_full = ones(ComplexF64, n)
b = VectorMPI(b_full)
x = solve(F, b)

x_full = Vector(x)
residual = A_full * x_full - b_full
err = norm(residual, Inf)

if rank == 0
    println("  LU solve residual (complex): $err")
end
@test err < TOL

# Test 6: Direct A \ b solve
if rank == 0
    println("[test] Direct A \\ b solve")
    flush(stdout)
end

n = 8
A_full = create_general_tridiagonal(n)
A = SparseMatrixMPI{Float64}(A_full)
b_full = ones(n)
b = VectorMPI(b_full)

# Direct solve without explicit factorization
x = A \ b

x_full = Vector(x)
residual = A_full * x_full - b_full
err = norm(residual, Inf)

if rank == 0
    println("  Direct solve residual: $err")
end
@test err < TOL

MPI.Barrier(comm)

# Test 7: Transpose solve - transpose(A) \ b
if rank == 0
    println("[test] Transpose solve")
    flush(stdout)
end

x_t = transpose(A) \ b

x_t_full = Vector(x_t)
residual_t = transpose(A_full) * x_t_full - b_full
err_t = norm(residual_t, Inf)

if rank == 0
    println("  Transpose solve residual: $err_t")
end
@test err_t < TOL

MPI.Barrier(comm)

# Test 8: Adjoint solve - A' \ b
if rank == 0
    println("[test] Adjoint solve")
    flush(stdout)
end

x_a = A' \ b

x_a_full = Vector(x_a)
residual_a = A_full' * x_a_full - b_full
err_a = norm(residual_a, Inf)

if rank == 0
    println("  Adjoint solve residual: $err_a")
end
@test err_a < TOL

MPI.Barrier(comm)

# Test 9: Factorization transpose/adjoint with F
if rank == 0
    println("[test] Factorization transpose/adjoint")
    flush(stdout)
end

F = lu(A)

# transpose(F) \ b should solve transpose(A) * x = b
x_Ft = transpose(F) \ b
x_Ft_full = Vector(x_Ft)
err_Ft = norm(transpose(A_full) * x_Ft_full - b_full, Inf)

# F' \ b should solve A' * x = b
x_Fa = F' \ b
x_Fa_full = Vector(x_Fa)
err_Fa = norm(A_full' * x_Fa_full - b_full, Inf)

if rank == 0
    println("  F transpose solve residual: $err_Ft")
    println("  F adjoint solve residual: $err_Fa")
end
@test err_Ft < TOL
@test err_Fa < TOL

end  # QuietTestSet

# Aggregate results across ranks
local_counts = [ts.counts[:pass], ts.counts[:fail], ts.counts[:error], ts.counts[:broken], ts.counts[:skip]]
global_counts = similar(local_counts)
MPI.Allreduce!(local_counts, global_counts, +, comm)

if rank == 0
    total = sum(global_counts)
    println("\nTest Summary: distributed factorization | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])  Total: $total")
    flush(stdout)
end

MPI.Barrier(comm)
MPI.Finalize()

exit_code = global_counts[2] + global_counts[3] > 0 ? 1 : 0
exit(exit_code)
