# Tests for indexing operations (range getindex/setindex!, HPCVector indexing)
# NOTE: Scalar indexing (v[i], A[i,j]) was removed to prevent MPI desync
using MPI
MPI.Init()

# Check CUDA availability BEFORE loading HPCSparseArrays
const CUDA_AVAILABLE = try
    using CUDA
    CUDA.device!(MPI.Comm_rank(MPI.COMM_WORLD) % length(CUDA.devices()))
    using NCCL_jll
    using CUDSS_jll
    CUDA.functional()
catch e
    false
end

using HPCSparseArrays
using SparseArrays
using LinearAlgebra
using Test

include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Create default CPU backend
backend = BACKEND_CPU_MPI
backend_int = backend_cpu_mpi(Int)  # Int backend for index vectors

# Create deterministic test data (same on all ranks)
n = 12

# HPCVector test data
v_global = collect(1.0:Float64(n))
v = HPCVector(v_global, backend)

# HPCSparseMatrix test data
I_vals = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3, 5, 7]
J_vals = [1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 8]
V_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.1, 0.3, 0.5, 0.7]
A_global = sparse(I_vals, J_vals, V_vals, n, n)
A = HPCSparseMatrix(A_global, backend)

# HPCMatrix test data
M_global = Float64[i + j * 0.1 for i in 1:n, j in 1:n]
M = HPCMatrix(M_global, backend)

ts = @testset QuietTestSet "Indexing" begin

# ============================================================================
# Range Indexing Tests
# ============================================================================

println(io0(), "[test] HPCVector range getindex")

w = v[3:7]
@test length(w) == 5
w_full = v[1:n]
@test length(w_full) == n


println(io0(), "[test] HPCVector range setindex! (scalar)")

v_modify = HPCVector(copy(v_global), backend)
v_modify[3:6] = 99.0
@test length(v_modify) == n


println(io0(), "[test] HPCVector range setindex! (vector)")

v_modify2 = HPCVector(copy(v_global), backend)
v_modify2[3:6] = [100.0, 200.0, 300.0, 400.0]
@test length(v_modify2) == n


println(io0(), "[test] HPCVector range setindex! (HPCVector source)")

v_modify3 = HPCVector(copy(v_global), backend)
src_vec = HPCVector([100.0, 200.0, 300.0, 400.0], backend)
v_modify3[3:6] = src_vec
@test length(v_modify3) == n


println(io0(), "[test] HPCMatrix range getindex")

M_sub = M[2:6, 3:8]
@test size(M_sub) == (5, 6)


println(io0(), "[test] HPCMatrix range getindex with Colon")

M_rows = M[2:5, :]
@test size(M_rows) == (4, n)
M_cols = M[:, 3:7]
@test size(M_cols) == (n, 5)
M_full = M[:, :]
@test size(M_full) == (n, n)


println(io0(), "[test] HPCMatrix range setindex! (scalar)")

M_modify = HPCMatrix(copy(M_global), backend)
M_modify[2:4, 3:5] = 0.0
@test size(M_modify) == (n, n)


println(io0(), "[test] HPCMatrix range setindex! (matrix)")

M_modify2 = HPCMatrix(copy(M_global), backend)
new_vals = Float64[10*i + j for i in 1:3, j in 1:4]
M_modify2[2:4, 3:6] = new_vals
@test size(M_modify2) == (n, n)


println(io0(), "[test] HPCMatrix range setindex! (HPCMatrix source)")

M_modify3 = HPCMatrix(zeros(n, n), backend)
src_matrix = HPCMatrix(Float64[10*i + j for i in 1:4, j in 1:5], backend)
M_modify3[2:5, 3:7] = src_matrix
@test size(M_modify3) == (n, n)


println(io0(), "[test] HPCSparseMatrix range getindex")

A_sub = A[2:6, 2:7]
@test size(A_sub) == (5, 6)


println(io0(), "[test] HPCSparseMatrix range getindex with Colon")

A_rows = A[2:5, :]
@test size(A_rows) == (4, n)
A_cols = A[:, 3:8]
@test size(A_cols) == (n, 6)


println(io0(), "[test] HPCSparseMatrix range setindex! (scalar)")

A_modify = HPCSparseMatrix(copy(A_global), backend)
A_modify[2:4, 2:5] = 0.0
@test size(A_modify) == (n, n)


println(io0(), "[test] HPCSparseMatrix range setindex! (HPCSparseMatrix source)")

A_modify2 = HPCSparseMatrix(copy(A_global), backend)
new_block = Float64[100*i + j for i in 1:3, j in 1:4]
A_modify2[2:4, 3:6] = HPCSparseMatrix(sparse(new_block), backend)
@test size(A_modify2) == (n, n)


println(io0(), "[test] HPCVector range partition consistency")

w1 = v[3:8]
w2 = v[3:8]
@test w1.partition == w2.partition
@test w1.structural_hash == w2.structural_hash


# ============================================================================
# HPCVector Indexing Tests
# ============================================================================

println(io0(), "[test] HPCVector getindex with HPCVector indices")

idx = HPCVector([3, 1, 5, 2, 6, 4], backend_int)
result = v[idx]
@test result.partition == idx.partition
@test length(result) == length(idx)


println(io0(), "[test] HPCVector setindex! with HPCVector indices")

v_modify = HPCVector(copy(v_global), backend)
idx_set = HPCVector([2, 4, 6], backend_int)
src_values = HPCVector([20.0, 40.0, 60.0], backend)
v_modify[idx_set] = src_values
@test length(v_modify) == n


println(io0(), "[test] HPCMatrix getindex with HPCVector indices")

row_idx = HPCVector([2, 5, 1, 4], backend_int)
col_idx = HPCVector([3, 1], backend_int)
result_dense = M[row_idx, col_idx]
@test size(result_dense) == (4, 2)


println(io0(), "[test] HPCMatrix setindex! with HPCVector indices")

M_modify = HPCMatrix(zeros(6, 4), backend)
row_idx_set = HPCVector([1, 3, 5], backend_int)
col_idx_set = HPCVector([2, 4], backend_int)
src_dense = HPCMatrix(ones(3, 2) * 7.0, backend)
M_modify[row_idx_set, col_idx_set] = src_dense
@test size(M_modify) == (6, 4)


println(io0(), "[test] HPCSparseMatrix getindex with HPCVector indices")

row_idx_sparse = HPCVector([2, 4, 1], backend_int)
col_idx_sparse = HPCVector([1, 3, 5], backend_int)
result_sparse = A[row_idx_sparse, col_idx_sparse]
@test result_sparse isa HPCMatrix
@test size(result_sparse) == (3, 3)


println(io0(), "[test] HPCSparseMatrix setindex! with HPCVector indices")

A_modify = HPCSparseMatrix(spzeros(6, 6), backend)
row_idx_set = HPCVector([1, 3, 5], backend_int)
col_idx_set = HPCVector([2, 4], backend_int)
src = HPCMatrix(ones(3, 2) * 9.0, backend)
A_modify[row_idx_set, col_idx_set] = src
@test size(A_modify) == (6, 6)


# ============================================================================
# Mixed Indexing Tests (HPCVector + range/Colon)
# ============================================================================

println(io0(), "[test] HPCMatrix getindex with HPCVector rows and range columns")

row_idx = HPCVector([2, 5, 8], backend_int)
M_mix = M[row_idx, 3:7]
@test size(M_mix) == (3, 5)


println(io0(), "[test] HPCMatrix getindex with range rows and HPCVector columns")

col_idx = HPCVector([1, 4, 7, 10], backend_int)
M_mix2 = M[2:5, col_idx]
@test size(M_mix2) == (4, 4)


println(io0(), "[test] HPCMatrix getindex with HPCVector rows and Colon")

row_idx = HPCVector([1, 6, n], backend_int)
M_colon = M[row_idx, :]
@test size(M_colon) == (3, n)


println(io0(), "[test] HPCMatrix getindex with Colon and HPCVector columns")

col_idx = HPCVector([2, 5, 8, 11], backend_int)
M_colon2 = M[:, col_idx]
@test size(M_colon2) == (n, 4)


println(io0(), "[test] HPCMatrix getindex with HPCVector rows and Int column")

row_idx = HPCVector([1, 4, 7, 10], backend_int)
M_int_col = M[row_idx, 5]
@test M_int_col isa HPCVector
@test length(M_int_col) == 4


println(io0(), "[test] HPCMatrix getindex with Int row and HPCVector columns")

col_idx = HPCVector([2, 4, 6, 8], backend_int)
M_int_row = M[3, col_idx]
@test M_int_row isa HPCVector
@test length(M_int_row) == 4


println(io0(), "[test] HPCSparseMatrix getindex with HPCVector rows and range columns")

row_idx = HPCVector([1, 3, 5, 7], backend_int)
A_mix = A[row_idx, 2:6]
@test size(A_mix) == (4, 5)


println(io0(), "[test] HPCSparseMatrix getindex with range rows and HPCVector columns")

col_idx = HPCVector([1, 3, 5, 7], backend_int)
A_mix2 = A[2:5, col_idx]
@test size(A_mix2) == (4, 4)


println(io0(), "[test] HPCSparseMatrix getindex with HPCVector rows and Int column")

row_idx = HPCVector([1, 3, 5, 7], backend_int)
A_int_col = A[row_idx, 3]
@test A_int_col isa HPCVector
@test length(A_int_col) == 4


println(io0(), "[test] HPCSparseMatrix getindex with Int row and HPCVector columns")

col_idx = HPCVector([1, 2, 3, 4], backend_int)
A_int_row = A[2, col_idx]
@test A_int_row isa HPCVector
@test length(A_int_row) == 4


# ============================================================================
# Column Extraction Tests
# ============================================================================

println(io0(), "[test] HPCMatrix column extraction (A[:, k])")

M_col = M[:, 5]
@test M_col isa HPCVector
@test length(M_col) == n


println(io0(), "[test] HPCSparseMatrix column extraction (A[:, k])")

A_col = A[:, 2]
@test A_col isa HPCVector
@test length(A_col) == n


# ============================================================================
# Empty Range Tests
# ============================================================================

println(io0(), "[test] Empty ranges")

@test size(M[1:0, 1:5]) == (0, 5)
@test size(M[1:5, 1:0]) == (5, 0)
@test size(M[1:0, 1:0]) == (0, 0)
@test size(A[1:0, 1:5]) == (0, 5)
@test size(A[1:5, 1:0]) == (5, 0)
@test nnz(SparseMatrixCSC(A[1:0, 1:0])) == 0


# ============================================================================
# Setindex! with mixed HPCVector/range
# ============================================================================

println(io0(), "[test] HPCMatrix setindex! with HPCVector rows and range columns")

M_setmix = HPCMatrix(zeros(n, n), backend)
row_idx = HPCVector([1, 4, 7], backend_int)
M_setmix[row_idx, 2:5] = HPCMatrix(ones(3, 4) * 55.0, backend)
@test size(M_setmix) == (n, n)


println(io0(), "[test] HPCMatrix setindex! with range rows and HPCVector columns")

M_setmix2 = HPCMatrix(zeros(n, n), backend)
col_idx = HPCVector([1, 5, 9], backend_int)
M_setmix2[2:4, col_idx] = HPCMatrix(ones(3, 3) * 66.0, backend)
@test size(M_setmix2) == (n, n)


println(io0(), "[test] HPCSparseMatrix setindex! with HPCVector rows and range columns")

A_setmix = HPCSparseMatrix(spzeros(n, n), backend)
row_idx = HPCVector([1, 4, 7], backend_int)
A_setmix[row_idx, 2:4] = HPCMatrix(ones(3, 3) * 77.0, backend)
@test size(A_setmix) == (n, n)


println(io0(), "[test] HPCSparseMatrix setindex! with range rows and HPCVector columns")

A_setmix2 = HPCSparseMatrix(spzeros(n, n), backend)
col_idx = HPCVector([1, 5, 9], backend_int)
A_setmix2[2:4, col_idx] = HPCMatrix(ones(3, 3) * 88.0, backend)
@test size(A_setmix2) == (n, n)


println(io0(), "[test] HPCMatrix setindex! with HPCVector rows and Int column")

M_setint = HPCMatrix(zeros(n, n), backend)
row_idx = HPCVector([1, 3, 5, 7], backend_int)
src_col = HPCVector([10.0, 20.0, 30.0, 40.0], backend)
M_setint[row_idx, 5] = src_col
@test size(M_setint) == (n, n)


println(io0(), "[test] HPCMatrix setindex! with Int row and HPCVector columns")

M_setint2 = HPCMatrix(zeros(n, n), backend)
col_idx = HPCVector([2, 4, 6, 8], backend_int)
src_row = HPCVector([100.0, 200.0, 300.0, 400.0], backend)
M_setint2[3, col_idx] = src_row
@test size(M_setint2) == (n, n)


println(io0(), "[test] HPCSparseMatrix setindex! with HPCVector rows and Int column")

A_setint = HPCSparseMatrix(spzeros(n, n), backend)
row_idx = HPCVector([1, 3, 5, 7], backend_int)
src_col = HPCVector([10.0, 20.0, 30.0, 40.0], backend)
A_setint[row_idx, 5] = src_col
@test size(A_setint) == (n, n)


println(io0(), "[test] HPCSparseMatrix setindex! with Int row and HPCVector columns")

A_setint2 = HPCSparseMatrix(spzeros(n, n), backend)
col_idx = HPCVector([2, 4, 6, 8], backend_int)
src_row = HPCVector([100.0, 200.0, 300.0, 400.0], backend)
A_setint2[3, col_idx] = src_row
@test size(A_setint2) == (n, n)


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

println(io0(), "Test Summary: Indexing | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")

MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
