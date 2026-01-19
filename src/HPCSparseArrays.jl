module HPCSparseArrays

using MPI
using Blake3Hash
using SparseArrays
using MUMPS
using StaticArrays
import SparseArrays: nnz, issparse, dropzeros, spdiagm, blockdiag
import LinearAlgebra
import LinearAlgebra: tr, diag, triu, tril, Transpose, Adjoint, norm, opnorm, mul!, ldlt, BLAS, issymmetric, UniformScaling, dot, Symmetric

export HPCSparseMatrix, HPCMatrix, HPCVector, clear_plan_cache!, uniform_partition, repartition
export HPCVector_CPU, HPCMatrix_CPU, HPCSparseMatrix_CPU  # Type aliases for CPU-backed types
export SparseMatrixCSR  # Type alias for Transpose{SparseMatrixCSC} (CSR storage format)
export map_rows, map_rows_gpu, vertex_indices  # Row-wise map over distributed vectors/matrices
export HPCVector_local, HPCMatrix_local, HPCSparseMatrix_local  # Local constructors
export mean  # Our mean function for HPCSparseMatrix and HPCVector
export io0   # Utility for rank-selective output
export get_backend  # Get the KernelAbstractions backend for a distributed array

# Backend conversion function
export to_backend

# Factorization exports (generic interface, implementation details hidden)
export solve, solve!, finalize!, clear_mumps_analysis_cache!

# HPCBackend abstraction exports
export HPCBackend, AbstractDevice, AbstractComm, AbstractSolver
export DeviceCPU, DeviceMetal, DeviceCUDA
export CommSerial, CommMPI
export SolverMUMPS, AbstractSolverCuDSS
export HPCBackendCPU, HPCBackendMetal, HPCBackendCUDA
export backend_cpu_serial, backend_cpu_mpi, backend_metal_mpi, backend_cuda_serial, backend_cuda_mpi
export BACKEND_CPU_SERIAL, BACKEND_CPU_MPI  # Pre-constructed CPU backend constants
# CUDA backends: use backend_cuda_serial() and backend_cuda_mpi() after loading CUDA
export comm_rank, comm_size
export array_type, matrix_type
export backends_compatible, assert_backends_compatible

# Type alias for 256-bit Blake3 hash (always mandatory, never Nothing)
const Blake3Hash = NTuple{32,UInt8}

# ============================================================================
# SparseMatrixCSR Type Alias and Constructors
# ============================================================================

"""
    SparseMatrixCSR{Tv,Ti} = Transpose{Tv, SparseMatrixCSC{Tv,Ti}}

Type alias for CSR (Compressed Sparse Row) storage format.

## The Dual Life of Transpose{SparseMatrixCSC}

In Julia, the type `Transpose{Tv, SparseMatrixCSC{Tv,Ti}}` has two interpretations:

1. **Semantic interpretation**: A lazy transpose wrapper around a CSC matrix.
   When you call `transpose(A)` on a SparseMatrixCSC, you get this wrapper that
   represents A^T without copying data.

2. **Storage interpretation**: CSR (row-major) access to sparse data.
   The underlying CSC stores columns contiguously, but through the transpose wrapper,
   we can iterate efficiently over rows instead of columns.

This alias clarifies intent: use `SparseMatrixCSR` when you want row-major storage
semantics, and `transpose(A)` when you want the mathematical transpose.

## CSR vs CSC Storage

- **CSC (Compressed Sparse Column)**: Julia's native sparse format. Efficient for
  column-wise operations, matrix-vector products with column access.
- **CSR (Compressed Sparse Row)**: Efficient for row-wise operations, matrix-vector
  products with row access, and row-partitioned distributed matrices.

For `SparseMatrixCSR`, the underlying `parent::SparseMatrixCSC` stores the *transposed*
matrix. If `B = SparseMatrixCSR(A)` represents matrix M, then `B.parent` is a CSC
storing M^T. This means:
- `B.parent.colptr` acts as row pointers for M
- `B.parent.rowval` contains column indices for M
- `B.parent.nzval` contains values in row-major order

## Usage Note

Julia will still display this type as `Transpose{Float64, SparseMatrixCSC{...}}`,
not as `SparseMatrixCSR`. The alias improves code clarity but doesn't affect
type printing.
"""
const SparseMatrixCSR{Tv,Ti} = Transpose{Tv, SparseMatrixCSC{Tv,Ti}}

"""
    SparseMatrixCSR(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}

Convert a CSC matrix to CSR format representing the **same** matrix.

If A represents matrix M in CSC format, the result represents M in CSR format.
Element access is unchanged: `B[i,j] == A[i,j]`.

Internally, this:
1. Materializes A^T as CSC (physical transpose)
2. Wraps in lazy transpose to get M back, but with row-major storage

# Example
```julia
A_csc = sparse([1,2,2], [1,1,2], [1.0, 2.0, 3.0], 2, 2)
A_csr = SparseMatrixCSR(A_csc)  # Same matrix, CSR storage
A_csr[1,1] == A_csc[1,1]        # true - same elements
```
"""
function SparseMatrixCSR(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    return transpose(SparseMatrixCSC(transpose(A)))
end

"""
    SparseMatrixCSC(A::SparseMatrixCSR{Tv,Ti}) where {Tv,Ti}

Convert a CSR matrix to CSC format representing the **same** matrix.

This physically transposes the underlying storage to produce a CSC matrix.
Element access is unchanged: the result represents the same matrix as the input.
"""
function SparseArrays.SparseMatrixCSC(A::SparseMatrixCSR{Tv,Ti}) where {Tv,Ti}
    # Use sparse() to avoid dispatching back to our method
    return sparse(transpose(A.parent))
end

# Cache for memoized MatrixPlans
# Key: (A_hash, B_hash, T, Ti, AV) - use full 256-bit hashes, includes array type for GPU support
const _plan_cache = Dict{Tuple{Blake3Hash,Blake3Hash,DataType,DataType,Type},Any}()

# Cache for memoized VectorPlans (for A * x)
# Key: (A_hash, x_hash, T, Ti, AV) - includes index type Ti and array type for GPU support
# Uses Type instead of DataType to support GPU UnionAll types like MtlVector{Float32}
const _vector_plan_cache = Dict{Tuple{Blake3Hash,Blake3Hash,Type,Type,Type},Any}()

# Cache for memoized DenseMatrixVectorPlans (for HPCMatrix * HPCVector)
# Key: (A_hash, x_hash, T, Backend) - uses backend type for GPU support
const _dense_vector_plan_cache = Dict{Tuple{Blake3Hash,Blake3Hash,Type,Type},Any}()

# Cache for memoized DenseTransposePlans (for transpose(HPCMatrix))
# Key: (A_hash, T, Backend) - uses backend type for GPU support
const _dense_transpose_plan_cache = Dict{Tuple{Blake3Hash,Type,Type},Any}()

# Cache for memoized RepartitionPlans (for repartition)
# Key includes (hash_A, target_hash, T, Ti) for sparse and (hash_A, target_hash, T) for others
const _repartition_plan_cache = Dict{Any,Any}()

# Cache for diagonal matrix structure
# Key: (vector's structural hash, Ti type) - Ti is needed for index type
# Value: DiagStructureCache with colptr, rowval, col_indices, and structural hash
struct DiagStructureCache{Ti<:Integer}
    colptr::Vector{Ti}
    rowval::Vector{Ti}
    col_indices::Vector{Int}  # Global indices stay Int (row/col space indexing)
    structural_hash::Blake3Hash
end
const _diag_structure_cache = Dict{Tuple{Blake3Hash,DataType},Any}()

# Cache for memoized AdditionPlans (for A + B and A - B)
# Key: (A_hash, B_hash, T, Ti, AV) - use full 256-bit hashes, includes array type for GPU support
const _addition_plan_cache = Dict{Tuple{Blake3Hash,Blake3Hash,DataType,DataType,Type},Any}()

# Cache for memoized IdentityAdditionPlans (for A + λI)
# Key: (A_hash, T, Ti) - use full 256-bit hash
const _identity_addition_plan_cache = Dict{Tuple{Blake3Hash,DataType,DataType},Any}()

# ============================================================================
# Backend Conversion Functions (stubs - implementations after includes)
# ============================================================================

# Forward declarations - implementations are after include("backends.jl")
function _convert_array end
function to_backend end
function _array_to_device end  # GPU extensions provide CPU→GPU conversion

"""
    clear_plan_cache!()

Clear all memoized plan caches, including the MUMPS analysis cache.
This is a collective operation that must be called on all MPI ranks together.
"""
function clear_plan_cache!()
    empty!(_plan_cache)
    empty!(_vector_plan_cache)
    empty!(_dense_vector_plan_cache)
    empty!(_dense_transpose_plan_cache)
    empty!(_repartition_plan_cache)
    empty!(_diag_structure_cache)
    empty!(_addition_plan_cache)
    empty!(_identity_addition_plan_cache)
    if isdefined(@__MODULE__, :_dense_transpose_vector_plan_cache)
        empty!(_dense_transpose_vector_plan_cache)
    end
    # Clear partition hash cache
    if isdefined(@__MODULE__, :_partition_hash_cache)
        empty!(_partition_hash_cache)
    end
    # Also clear MUMPS analysis cache (defined in mumps_factorization.jl)
    if isdefined(@__MODULE__, :clear_mumps_analysis_cache!)
        clear_mumps_analysis_cache!()
    end
end

"""
    cache_sizes() -> NamedTuple

Return the sizes of all internal caches. Useful for debugging memory leaks.
"""
function cache_sizes()
    mumps_size = isdefined(@__MODULE__, :_mumps_analysis_cache) ? length(_mumps_analysis_cache) : 0
    dense_transpose_vector_size = isdefined(@__MODULE__, :_dense_transpose_vector_plan_cache) ? length(_dense_transpose_vector_plan_cache) : 0
    return (
        plan = length(_plan_cache),
        vector_plan = length(_vector_plan_cache),
        dense_vector_plan = length(_dense_vector_plan_cache),
        dense_transpose_plan = length(_dense_transpose_plan_cache),
        dense_transpose_vector_plan = dense_transpose_vector_size,
        repartition_plan = length(_repartition_plan_cache),
        diag_structure = length(_diag_structure_cache),
        addition_plan = length(_addition_plan_cache),
        identity_addition_plan = length(_identity_addition_plan_cache),
        partition_hash = length(_partition_hash_cache),
        mumps_analysis = mumps_size,
    )
end

"""
    check_cache_sizes!(; max_entries::Int=20, io::IO=stderr)

Check that no cache exceeds `max_entries`. If any cache is too large,
print a warning and return false. Otherwise return true.
"""
function check_cache_sizes!(; max_entries::Int=20, io::IO=stderr)
    sizes = cache_sizes()
    ok = true
    for (name, sz) in pairs(sizes)
        if sz > max_entries
            println(io, "WARNING: Cache $name has $sz entries (max=$max_entries)")
            ok = false
        end
    end
    return ok
end

export cache_sizes, check_cache_sizes!

# Cache for partition hashes: objectid(partition) -> Blake3Hash
# This avoids recomputing hashes for the same partition vector
const _partition_hash_cache = Dict{UInt, Blake3Hash}()

"""
    compute_partition_hash(partition::Vector{Int}) -> Blake3Hash

Compute a Blake3 hash of a partition vector's contents.
"""
function compute_partition_hash(partition::Vector{Int})::Blake3Hash
    ctx = Blake3Ctx()
    update!(ctx, reinterpret(UInt8, partition))
    return Blake3Hash(digest(ctx))
end

"""
    uniform_partition(n::Int, nranks::Int) -> Vector{Int}

Compute a balanced partition of `n` elements across `nranks` ranks.
Returns a vector of length `nranks + 1` with 1-indexed partition boundaries.

The first `mod(n, nranks)` ranks get `div(n, nranks) + 1` elements,
the remaining ranks get `div(n, nranks)` elements.

# Example
```julia
partition = uniform_partition(10, 4)  # [1, 4, 7, 9, 11]
# Rank 0: 1:3 (3 elements)
# Rank 1: 4:6 (3 elements)
# Rank 2: 7:8 (2 elements)
# Rank 3: 9:10 (2 elements)
```
"""
function uniform_partition(n::Int, nranks::Int)
    per_rank = div(n, nranks)
    remainder = mod(n, nranks)
    partition = Vector{Int}(undef, nranks + 1)
    partition[1] = 1
    for r in 1:nranks
        extra = r <= remainder ? 1 : 0
        partition[r+1] = partition[r] + per_rank + extra
    end
    return partition
end

# Include the component files (order matters: backends first, then vectors, then dense/sparse, then blocks, then indexing)
include("backends.jl")
include("vectors.jl")
include("dense.jl")
include("sparse.jl")
include("blocks.jl")
include("indexing.jl")

# Include MUMPS factorization module
include("mumps_factorization.jl")

# ============================================================================
# _convert_array Implementations (after backends.jl is loaded)
# ============================================================================

"""
    _convert_array(v::AbstractVector, device::AbstractDevice) -> AbstractVector
    _convert_array(A::AbstractMatrix, device::AbstractDevice) -> AbstractMatrix

Convert an array to the appropriate type for the given device.
Base implementations handle CPU cases; extensions add GPU conversions.

This is the low-level helper used by `to_backend()`.
"""
# CPU → CPU: identity (no copy)
_convert_array(v::Vector, ::DeviceCPU) = v
_convert_array(A::Matrix, ::DeviceCPU) = A
# GPU → CPU: copy to Array
_convert_array(v::AbstractVector, ::DeviceCPU) = Array(v)
_convert_array(A::AbstractMatrix, ::DeviceCPU) = Array(A)

# Extensions add methods for DeviceMetal and DeviceCUDA:
# _convert_array(v::Vector, ::DeviceMetal) = MtlVector(v)
# _convert_array(v::MtlVector, ::DeviceMetal) = v
# _convert_array(v::Vector, ::DeviceCUDA) = CuVector(v)
# _convert_array(v::CuVector, ::DeviceCUDA) = v

# ============================================================================
# to_backend() Implementations (after types are defined)
# ============================================================================

"""
    to_backend(v::HPCVector, backend::HPCBackend) -> HPCVector

Convert a HPCVector to use a different backend.
"""
function to_backend(v::HPCVector{T}, backend::B) where {T, B<:HPCBackend}
    new_v = _convert_array(v.v, backend.device)
    HPCVector{T,B}(v.structural_hash, v.partition, new_v, backend)
end

"""
    to_backend(A::HPCMatrix, backend::HPCBackend) -> HPCMatrix

Convert a HPCMatrix to use a different backend.
"""
function to_backend(A::HPCMatrix{T}, backend::B) where {T, B<:HPCBackend}
    new_A = _convert_array(A.A, backend.device)
    HPCMatrix{T,B}(A.structural_hash, A.row_partition, A.col_partition, new_A, backend)
end

"""
    to_backend(A::HPCSparseMatrix, backend::HPCBackend) -> HPCSparseMatrix

Convert a HPCSparseMatrix to use a different backend.
The nzval and target structure arrays are converted; CPU structure arrays remain on CPU.
"""
function to_backend(A::HPCSparseMatrix{T,Ti}, backend::B) where {T, Ti, B<:HPCBackend}
    new_nzval = _convert_array(A.nzval, backend.device)
    new_rowptr_target = _to_target_device(A.rowptr, backend.device)
    new_colval_target = _to_target_device(A.colval, backend.device)
    HPCSparseMatrix{T,Ti,B}(
        A.structural_hash,
        A.row_partition,
        A.col_partition,
        A.col_indices,
        A.rowptr,
        A.colval,
        new_nzval,
        A.nrows_local,
        A.ncols_compressed,
        nothing,  # Invalidate cached_transpose
        A.cached_symmetric,
        new_rowptr_target,
        new_colval_target,
        backend
    )
end

# ============================================================================
# Symmetry Check
# ============================================================================

"""
    _compare_rows_distributed(A::HPCSparseMatrix{T}, B::HPCSparseMatrix{T}) where T

Compare two sparse matrices with potentially different row partitions.
Redistributes B's rows to match A's row partition, then compares locally.
Returns true if all corresponding entries are equal.
"""
function _compare_rows_distributed(A::HPCSparseMatrix{T,Ti,Bk}, B::HPCSparseMatrix{T,Ti,Bk}) where {T,Ti,Bk<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    # A's local rows
    my_row_start = A.row_partition[rank + 1]
    my_row_end = A.row_partition[rank + 2] - 1
    my_nrows = my_row_end - my_row_start + 1

    # For each of A's local rows, determine which rank owns that row in B
    # B has row_partition = A.col_partition (since B = transpose(A))
    rows_needed_from = [Int[] for _ in 1:nranks]  # rows_needed_from[r+1] = rows we need from rank r
    for row in my_row_start:my_row_end
        owner = searchsortedlast(B.row_partition, row) - 1
        push!(rows_needed_from[owner + 1], row)
    end

    # Exchange: tell each rank which rows we need from them
    send_counts = Int32[length(rows_needed_from[r + 1]) for r in 0:nranks-1]
    recv_counts = comm_alltoall(comm, MPI.UBuffer(send_counts, 1))

    # Send row requests
    send_reqs = MPI.Request[]
    for r in 0:nranks-1
        if send_counts[r + 1] > 0 && r != rank
            req = comm_isend(comm, rows_needed_from[r + 1], r, 80)
            push!(send_reqs, req)
        end
    end

    # Receive row requests
    rows_to_send = Dict{Int, Vector{Int}}()
    recv_reqs = MPI.Request[]
    for r in 0:nranks-1
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            req = comm_irecv!(comm, buf, r, 80)
            push!(recv_reqs, req)
            rows_to_send[r] = buf
        end
    end

    comm_waitall(comm, vcat(send_reqs, recv_reqs))

    # Now send the actual row data from B
    # For each row, we send: (num_entries, col_indices..., values...)
    BT = _get_csc(B)  # underlying CSC (columns = local rows of B)
    B_row_start = B.row_partition[rank + 1]

    # Prepare send buffers: pack row data
    send_data = Vector{Vector{UInt8}}(undef, nranks)
    for r in 0:nranks-1
        rows = get(rows_to_send, r, Int[])
        if isempty(rows) || r == rank
            send_data[r + 1] = UInt8[]
            continue
        end

        # Pack all requested rows into a buffer
        io = IOBuffer()
        for global_row in rows
            local_row = global_row - B_row_start + 1
            ptr_start = BT.colptr[local_row]
            ptr_end = BT.colptr[local_row + 1] - 1
            nnz_row = ptr_end - ptr_start + 1

            write(io, Int32(nnz_row))
            for ptr in ptr_start:ptr_end
                global_col = B.col_indices[BT.rowval[ptr]]
                write(io, Int32(global_col))
            end
            for ptr in ptr_start:ptr_end
                write(io, BT.nzval[ptr])
            end
        end
        send_data[r + 1] = take!(io)
    end

    # Exchange message sizes so we know how much to receive
    send_sizes = Int32[length(send_data[r + 1]) for r in 0:nranks-1]
    recv_sizes = comm_alltoall(comm, MPI.UBuffer(send_sizes, 1))

    # Now send and receive row data with known sizes
    send_data_reqs = MPI.Request[]
    for r in 0:nranks-1
        if r != rank && send_sizes[r + 1] > 0
            req = comm_isend(comm, send_data[r + 1], r, 81)
            push!(send_data_reqs, req)
        end
    end

    recv_data = Vector{Vector{UInt8}}(undef, nranks)
    recv_data_reqs = MPI.Request[]
    for r in 0:nranks-1
        if r != rank && recv_sizes[r + 1] > 0
            recv_data[r + 1] = Vector{UInt8}(undef, recv_sizes[r + 1])
            req = comm_irecv!(comm, recv_data[r + 1], r, 81)
            push!(recv_data_reqs, req)
        else
            recv_data[r + 1] = UInt8[]
        end
    end

    comm_waitall(comm, vcat(send_data_reqs, recv_data_reqs))

    # Receive row data and compare with A's local rows
    local_match = true
    AT = _get_csc(A)
    A_row_start = A.row_partition[rank + 1]

    # First handle rows we own in both A and B (rank == rank case)
    for global_row in rows_needed_from[rank + 1]
        local_row_A = global_row - A_row_start + 1
        local_row_B = global_row - B_row_start + 1

        # Get A's row entries
        ptr_start_A = AT.colptr[local_row_A]
        ptr_end_A = AT.colptr[local_row_A + 1] - 1
        nnz_A = ptr_end_A - ptr_start_A + 1

        # Get B's row entries
        ptr_start_B = BT.colptr[local_row_B]
        ptr_end_B = BT.colptr[local_row_B + 1] - 1
        nnz_B = ptr_end_B - ptr_start_B + 1

        if nnz_A != nnz_B
            local_match = false
            break
        end

        # Compare entries (need to handle potentially different orderings)
        A_entries = Dict{Int, T}()
        for ptr in ptr_start_A:ptr_end_A
            global_col = A.col_indices[AT.rowval[ptr]]
            A_entries[global_col] = AT.nzval[ptr]
        end

        for ptr in ptr_start_B:ptr_end_B
            global_col = B.col_indices[BT.rowval[ptr]]
            if !haskey(A_entries, global_col) || A_entries[global_col] != BT.nzval[ptr]
                local_match = false
                break
            end
        end

        if !local_match
            break
        end
    end

    # Now compare rows received from other ranks
    for r in 0:nranks-1
        if r == rank || isempty(recv_data[r + 1])
            continue
        end

        if !local_match
            # Already know it doesn't match, data already received
            continue
        end

        io = IOBuffer(recv_data[r + 1])
        for global_row in rows_needed_from[r + 1]
            local_row_A = global_row - A_row_start + 1

            # Read B's row data
            nnz_B = read(io, Int32)
            B_cols = [read(io, Int32) for _ in 1:nnz_B]
            B_vals = [read(io, T) for _ in 1:nnz_B]

            # Get A's row entries
            ptr_start_A = AT.colptr[local_row_A]
            ptr_end_A = AT.colptr[local_row_A + 1] - 1
            nnz_A = ptr_end_A - ptr_start_A + 1

            if nnz_A != nnz_B
                local_match = false
                break
            end

            A_entries = Dict{Int, T}()
            for ptr in ptr_start_A:ptr_end_A
                global_col = A.col_indices[AT.rowval[ptr]]
                A_entries[global_col] = AT.nzval[ptr]
            end

            for (col, val) in zip(B_cols, B_vals)
                if !haskey(A_entries, col) || A_entries[col] != val
                    local_match = false
                    break
                end
            end

            if !local_match
                break
            end
        end
    end

    # Allreduce to check if all ranks matched
    global_match = comm_allreduce(comm, local_match ? 1 : 0, MPI.BAND)
    return global_match == 1
end

"""
    LinearAlgebra.issymmetric(A::HPCSparseMatrix{T}) where T

Check if A is symmetric by materializing the transpose and comparing rows.
Returns true if A == transpose(A). Result is cached for subsequent calls.
"""
function LinearAlgebra.issymmetric(A::HPCSparseMatrix{T}) where T
    # Return cached result if available
    if A.cached_symmetric !== nothing
        return A.cached_symmetric
    end

    m, n = size(A)
    if m != n
        A.cached_symmetric = false
        return false
    end

    At = HPCSparseMatrix(transpose(A))
    result = _compare_rows_distributed(A, At)
    A.cached_symmetric = result
    return result
end

# ============================================================================
# Direct Solve Interface (A \ b) with Backslash Caching
# ============================================================================
#
# The backslash operator caches factorizations by structural hash.
# On cache hit, we update values and refactorize (skip symbolic analysis).
# On cache miss, we create a full factorization and cache it.

"""
    Base.:\\(A::HPCSparseMatrix{T,Ti,HPCBackend{T,Ti,D,C,SolverMUMPS}}, b::HPCVector{T,HPCBackend{T,Ti,D,C,SolverMUMPS}}) where {T,Ti,D,C}

Solve A*x = b using LU factorization with MUMPS.

Uses backslash caching: first call for a given sparsity pattern performs full
analysis + factorization. Subsequent calls with the same pattern skip analysis
and only refactorize (much faster).

For symmetric matrices, use `Symmetric(A) \\ b` to use the faster LDLT factorization.
For repeated solves with the same values, compute factorization once with `lu(A)`.

Note: This method is specific to MUMPS backends. GPU backends (cuDSS) have their own
specialized backslash methods defined in the CUDA extension.
"""
function Base.:\(A::HPCSparseMatrix{T,Ti,HPCBackend{T,Ti,D,C,SolverMUMPS}},
                 b::HPCVector{T,HPCBackend{T,Ti,D,C,SolverMUMPS}}) where {T,Ti,D,C}
    structural_hash = A.structural_hash
    cache_key = (structural_hash, false, T)  # false = not symmetric (LU)

    if haskey(_mumps_backslash_cache, cache_key)
        # Cache hit: refactorize and solve (skip analysis!)
        F = _mumps_backslash_cache[cache_key]::MUMPSFactorization{T,HPCBackend{T,Ti,D,C,SolverMUMPS},_mumps_internal_type(T)}
        return _refactorize_and_solve!(F, A, b)
    else
        # Cache miss: create full factorization and cache it
        F = _create_fresh_mumps_factorization(A, false)
        _mumps_backslash_cache[cache_key] = F
        return solve(F, b)
    end
end

"""
    Base.:\\(A::Symmetric{T,<:HPCSparseMatrix{T,Ti,HPCBackend{T,Ti,D,C,SolverMUMPS}}}, b::HPCVector{T,HPCBackend{T,Ti,D,C,SolverMUMPS}}) where {T,Ti,D,C}

Solve A*x = b for a symmetric matrix using LDLT with MUMPS.

Uses backslash caching: first call for a given sparsity pattern performs full
analysis + factorization. Subsequent calls with the same pattern skip analysis
and only refactorize (much faster).

Use `Symmetric(A)` to wrap a known-symmetric matrix and skip the expensive symmetry check.

Note: This method is specific to MUMPS backends. GPU backends (cuDSS) have their own
specialized backslash methods defined in the CUDA extension.
"""
function Base.:\(A::Symmetric{T,<:HPCSparseMatrix{T,Ti,HPCBackend{T,Ti,D,C,SolverMUMPS}}},
                 b::HPCVector{T,HPCBackend{T,Ti,D,C,SolverMUMPS}}) where {T,Ti,D,C}
    A_inner = parent(A)
    structural_hash = A_inner.structural_hash
    cache_key = (structural_hash, true, T)  # true = symmetric (LDLT)

    if haskey(_mumps_backslash_cache, cache_key)
        # Cache hit: refactorize and solve (skip analysis!)
        F = _mumps_backslash_cache[cache_key]::MUMPSFactorization{T,HPCBackend{T,Ti,D,C,SolverMUMPS},_mumps_internal_type(T)}
        return _refactorize_and_solve!(F, A_inner, b)
    else
        # Cache miss: create full factorization and cache it
        F = _create_fresh_mumps_factorization(A_inner, true)
        _mumps_backslash_cache[cache_key] = F
        return solve(F, b)
    end
end

"""
    Base.:\\(At::Transpose{T,<:HPCSparseMatrix{T,Ti,HPCBackend{T,Ti,D,C,SolverMUMPS}}}, b::HPCVector{T,HPCBackend{T,Ti,D,C,SolverMUMPS}}) where {T,Ti,D,C}

Solve transpose(A)*x = b using LU factorization with MUMPS.

Uses backslash caching on the materialized transpose.

Note: This method is specific to MUMPS backends. GPU backends (cuDSS) have their own
specialized backslash methods defined in the CUDA extension.
"""
function Base.:\(At::Transpose{T,<:HPCSparseMatrix{T,Ti,HPCBackend{T,Ti,D,C,SolverMUMPS}}},
                 b::HPCVector{T,HPCBackend{T,Ti,D,C,SolverMUMPS}}) where {T,Ti,D,C}
    # Materialize the transpose and use backslash caching on it
    A_t = HPCSparseMatrix(At)
    return A_t \ b
end

# ============================================================================
# Right Division Interface (b / A)
# ============================================================================

# Right division: x * A = b, so x = b * A^(-1) = transpose(transpose(A) \ transpose(b))
# For row vectors: transpose(v) / A solves x * A = transpose(v)

"""
    Base.:/(vt::Transpose{T,HPCVector{T,HPCBackend{T,Ti,D,C,SolverMUMPS}}}, A::HPCSparseMatrix{T,Ti,HPCBackend{T,Ti,D,C,SolverMUMPS}}) where {T,Ti,D,C}

Solve x * A = transpose(v), returning x as a transposed HPCVector.
Equivalent to transpose(transpose(A) \\ v).

Note: This method is specific to MUMPS backends.
"""
function Base.:/(vt::Transpose{T,HPCVector{T,HPCBackend{T,Ti,D,C,SolverMUMPS}}},
                 A::HPCSparseMatrix{T,Ti,HPCBackend{T,Ti,D,C,SolverMUMPS}}) where {T,Ti,D,C}
    v = vt.parent
    x = transpose(A) \ v
    return transpose(x)
end

"""
    Base.:/(vt::Transpose{T,HPCVector{T,HPCBackend{T,Ti,D,C,SolverMUMPS}}}, At::Transpose{T,<:HPCSparseMatrix{T,Ti,HPCBackend{T,Ti,D,C,SolverMUMPS}}}) where {T,Ti,D,C}

Solve x * transpose(A) = transpose(v), returning x as a transposed HPCVector.

Note: This method is specific to MUMPS backends.
"""
function Base.:/(vt::Transpose{T,HPCVector{T,HPCBackend{T,Ti,D,C,SolverMUMPS}}},
                 At::Transpose{T,<:HPCSparseMatrix{T,Ti,HPCBackend{T,Ti,D,C,SolverMUMPS}}}) where {T,Ti,D,C}
    v = vt.parent
    A = At.parent
    x = A \ v
    return transpose(x)
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    io0(io=stdout; r::Set{Int}=Set{Int}([0]), dn=devnull)

Return `io` if the current MPI rank is in set `r`, otherwise return `dn` (default: `devnull`).

This is useful for printing only from specific ranks:
```julia
println(io0(), "Hello from rank 0!")
println(io0(r=Set([0,1])), "Hello from ranks 0 and 1!")
```

With string interpolation:
```julia
println(io0(), "Matrix A = \$A")
```
"""
function io0(io::IO=stdout; r::Set{Int}=Set{Int}([0]), dn::IO=devnull)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    return rank ∈ r ? io : dn
end

# ============================================================================
# Conversion Functions: MPI types -> Native Julia types
# ============================================================================

"""
    Vector(v::HPCVector{T}) where T

Gather a distributed HPCVector to a full Vector on all ranks.
Requires MPI communication (Allgatherv).
"""
function Base.Vector(v::HPCVector{T,B}) where {T, B<:HPCBackend}
    comm = v.backend.comm
    nranks = comm_size(comm)

    # Compute counts per rank
    counts = Int32[v.partition[r+2] - v.partition[r+1] for r in 0:nranks-1]

    # Ensure local data is on CPU for MPI (GPU arrays not supported by MPI.jl)
    v_cpu = _ensure_cpu(v.v)

    # Use Allgatherv to gather the full vector
    full_v = Vector{T}(undef, length(v))
    comm_allgatherv!(comm, v_cpu, MPI.VBuffer(full_v, counts))

    return full_v
end

"""
    Matrix(A::HPCMatrix{T}) where T

Gather a distributed HPCMatrix to a full Matrix on all ranks.
Requires MPI communication (Allgatherv).
"""
function Base.Matrix(A::HPCMatrix{T,B}) where {T, B<:HPCBackend}
    comm = A.backend.comm
    nranks = comm_size(comm)

    m, n = size(A)

    # Compute row counts per rank (each rank's local rows * ncols = elements to gather)
    row_counts = Int32[A.row_partition[r+2] - A.row_partition[r+1] for r in 0:nranks-1]
    element_counts = Int32.(row_counts .* n)

    # Allocate full matrix
    full_M = Matrix{T}(undef, m, n)

    # Flatten local matrix (column-major order)
    # Ensure data is on CPU for MPI (GPU arrays not supported by MPI.jl)
    local_flat = _ensure_cpu(vec(A.A))

    # Gather all flattened matrices
    full_flat = Vector{T}(undef, m * n)
    comm_allgatherv!(comm, local_flat, MPI.VBuffer(full_flat, element_counts))

    # Reconstruct full matrix from gathered data
    # Each rank's data is stored row-by-row in column-major chunks
    offset = 0
    for r in 0:nranks-1
        row_start = A.row_partition[r+1]
        row_end = A.row_partition[r+2] - 1
        local_nrows = row_end - row_start + 1
        if local_nrows > 0
            # Reshape rank r's data into (local_nrows, n) and copy to full matrix
            rank_data = reshape(@view(full_flat[offset+1:offset+local_nrows*n]), local_nrows, n)
            full_M[row_start:row_end, :] = rank_data
            offset += local_nrows * n
        end
    end

    return full_M
end

"""
    SparseArrays.SparseMatrixCSC(A::HPCSparseMatrix{T}) where T

Gather a distributed HPCSparseMatrix to a full SparseMatrixCSC on all ranks.
Requires MPI communication (Allgatherv).
"""
function SparseArrays.SparseMatrixCSC(A::HPCSparseMatrix{T,Ti,B}) where {T, Ti, B<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    m, n = size(A)
    my_row_start = A.row_partition[rank+1]

    # Extract local triplets (I, J, V)
    AT = _get_csc(A)  # underlying CSC storage
    local_nnz = nnz(AT)

    local_I = Vector{Ti}(undef, local_nnz)
    local_J = Vector{Ti}(undef, local_nnz)
    local_V = Vector{T}(undef, local_nnz)

    idx = 1
    for col in 1:AT.n  # AT.n = number of local rows
        global_row = my_row_start + col - 1
        for ptr in AT.colptr[col]:(AT.colptr[col+1]-1)
            # AT.rowval contains LOCAL column indices, convert to global
            global_col = A.col_indices[AT.rowval[ptr]]
            local_I[idx] = global_row
            local_J[idx] = global_col
            local_V[idx] = AT.nzval[ptr]
            idx += 1
        end
    end

    # Gather counts
    local_count = Int32(local_nnz)
    all_counts = comm_allgather(comm, local_count)
    total_nnz = sum(all_counts)

    # Gather all triplets
    global_I = Vector{Ti}(undef, total_nnz)
    global_J = Vector{Ti}(undef, total_nnz)
    global_V = Vector{T}(undef, total_nnz)

    comm_allgatherv!(comm, local_I, MPI.VBuffer(global_I, all_counts))
    comm_allgatherv!(comm, local_J, MPI.VBuffer(global_J, all_counts))
    comm_allgatherv!(comm, local_V, MPI.VBuffer(global_V, all_counts))

    # Build global sparse matrix
    return sparse(global_I, global_J, global_V, m, n)
end

# ============================================================================
# Show Methods
# ============================================================================

"""
    Base.show(io::IO, v::HPCVector)

Display a HPCVector by gathering it to a full vector and showing that.
"""
function Base.show(io::IO, v::HPCVector{T}) where T
    full_v = Vector(v)
    print(io, "HPCVector{$T}(")
    show(io, full_v)
    print(io, ")")
end

"""
    Base.show(io::IO, ::MIME"text/plain", v::HPCVector)

Pretty-print a HPCVector.
"""
function Base.show(io::IO, ::MIME"text/plain", v::HPCVector{T}) where T
    full_v = Vector(v)
    println(io, length(v), "-element HPCVector{$T}:")
    show(io, MIME("text/plain"), full_v)
end

"""
    Base.show(io::IO, A::HPCMatrix)

Display a HPCMatrix by gathering it to a full matrix and showing that.
"""
function Base.show(io::IO, A::HPCMatrix{T}) where T
    full_A = Matrix(A)
    print(io, "HPCMatrix{$T}(")
    show(io, full_A)
    print(io, ")")
end

"""
    Base.show(io::IO, ::MIME"text/plain", A::HPCMatrix)

Pretty-print a HPCMatrix.
"""
function Base.show(io::IO, ::MIME"text/plain", A::HPCMatrix{T}) where T
    full_A = Matrix(A)
    m, n = size(A)
    println(io, "$m×$n HPCMatrix{$T}:")
    show(io, MIME("text/plain"), full_A)
end

"""
    Base.show(io::IO, A::HPCSparseMatrix)

Display a HPCSparseMatrix by gathering it to a full SparseMatrixCSC and showing that.
"""
function Base.show(io::IO, A::HPCSparseMatrix{T}) where T
    full_A = SparseMatrixCSC(A)
    print(io, "HPCSparseMatrix{$T}(")
    show(io, full_A)
    print(io, ")")
end

"""
    Base.show(io::IO, ::MIME"text/plain", A::HPCSparseMatrix)

Pretty-print a HPCSparseMatrix.
"""
function Base.show(io::IO, ::MIME"text/plain", A::HPCSparseMatrix{T}) where T
    full_A = SparseMatrixCSC(A)
    m, n = size(A)
    println(io, "$m×$n HPCSparseMatrix{$T} with $(nnz(full_A)) stored entries:")
    show(io, MIME("text/plain"), full_A)
end

# ============================================================================
# map_rows: Row-wise map over distributed vectors/matrices
# ============================================================================

"""
    _get_row_partition(A::HPCVector) -> Vector{Int}
    _get_row_partition(A::HPCMatrix) -> Vector{Int}

Get the row partition from a distributed type.
"""
_get_row_partition(A::HPCVector) = A.partition
_get_row_partition(A::HPCMatrix) = A.row_partition

"""
    _align_to_partition(A::HPCVector{T}, p::Vector{Int}) where T
    _align_to_partition(A::HPCMatrix{T}, p::Vector{Int}) where T

Repartition a distributed type to match partition p.
"""
_align_to_partition(A::HPCVector, p::Vector{Int}) = repartition(A, p)
_align_to_partition(A::HPCMatrix, p::Vector{Int}) = repartition(A, p)

# Helper: Convert local matrix to Vector{SVector} by transposing then reinterpreting
# For column-major storage, transpose gives us rows as contiguous columns
function _matrix_to_svectors(::Val{K}, M::AbstractMatrix{T}) where {K, T}
    # M is (nrows, K) in column-major. transpose(M) is (K, nrows) column-major.
    # Each column of transpose(M) is a row of M, which can be reinterpreted as SVector{K,T}
    MT = copy(transpose(M))  # Materialize to contiguous memory (stays on GPU if M is GPU)
    vec(reinterpret(reshape, SVector{K, T}, MT))
end
_matrix_to_svectors(M::AbstractMatrix{T}) where {T} = _matrix_to_svectors(Val(size(M, 2)), M)

# Helper: Convert Vector{SVector} back to Matrix
function _svectors_to_matrix(v::AbstractVector{SVector{K,T}}) where {K,T}
    # reinterpret as (K, nrows), then transpose to (nrows, K)
    # Use copy(transpose(...)) to preserve GPU array type (not Matrix(...) which forces CPU)
    copy(transpose(reinterpret(reshape, T, v)))
end

# Helper: Convert to SVector representation for map_rows
# Vectors pass through as-is (each element is a "row")
_to_svectors(v::AbstractVector{T}) where {T<:Number} = v
# Matrices get converted to Vector{SVector}
_to_svectors(M::AbstractMatrix{T}) where {T} = _matrix_to_svectors(M)

# Helper: Convert result back based on type
_from_result(v::AbstractVector{SVector{K,T}}) where {K,T} = _svectors_to_matrix(v)
_from_result(v::AbstractVector{T}) where {T<:Number} = v

# ============================================================================
# GPU kernel infrastructure for map_rows_gpu
# ============================================================================

"""
    _map_rows_gpu_kernel(f, matrices...)

Apply function `f` row-wise over matrices. Each row is converted to an SVector,
`f` is applied, and results are collected into an output matrix.

This is the CPU fallback implementation using broadcasting. The Metal extension
overrides this for MtlMatrix inputs to use efficient GPU kernels.

For HPCVector inputs, the vector should be reshaped to a 1-column matrix before calling.
"""
function _map_rows_gpu_kernel(f, arg1::AbstractMatrix{T}, rest::AbstractMatrix...) where T
    # Convert to SVector representation for broadcasting
    local_arrays = (_matrix_to_svectors(arg1), map(_matrix_to_svectors, rest)...)

    # Broadcast f over all local arrays
    results = f.(local_arrays...)

    # Convert results back to matrix
    _from_result(results)
end

# CPU version (same as GPU kernel for CPU arrays)
_map_rows_cpu_kernel(f, args...) = _map_rows_gpu_kernel(f, args...)

"""
    map_rows_gpu(f, A...)

Apply function `f` to corresponding rows of distributed vectors/matrices (GPU-native).

Each argument in `A...` must be either a `HPCVector` or `HPCMatrix`. All inputs
are repartitioned to match the partition of the first argument before applying `f`.

This implementation uses GPU-friendly broadcasting: matrices are converted to
Vector{SVector} via transpose+reinterpret, then f is broadcast over all arguments.
This avoids GPU->CPU->GPU round-trips when the underlying arrays are on GPU.

**Important**: The function `f` must be isbits-compatible (no captured non-isbits data)
for GPU execution. Use [`map_rows`](@ref) for functions with arbitrary closures.

For each row index i, `f` is called with:
- For `HPCVector`: the scalar element at index i
- For `HPCMatrix`: an SVector containing the i-th row

## Result Type

The result type depends on what `f` returns:

| `f` returns | Result |
|-------------|--------|
| scalar (`Number`) | `HPCVector` (one element per input row) |
| `SVector{K,T}` | `HPCMatrix` (K columns, one row per input row) |

## Examples

```julia
# Element-wise product of two vectors
u = HPCVector([1.0, 2.0, 3.0])
v = HPCVector([4.0, 5.0, 6.0])
w = map_rows_gpu((a, b) -> a * b, u, v)  # HPCVector([4.0, 10.0, 18.0])

# Row norms of a matrix
A = HPCMatrix(randn(5, 3))
norms = map_rows_gpu(r -> norm(r), A)  # HPCVector of row norms

# Return SVector to build a matrix
A = HPCMatrix(randn(3, 2))
result = map_rows_gpu(r -> SVector(sum(r), prod(r)), A)  # 3×2 HPCMatrix

# Mixed inputs: matrix rows combined with vector elements
A = HPCMatrix(randn(4, 3))
w = HPCVector([1.0, 2.0, 3.0, 4.0])
result = map_rows_gpu((row, wi) -> sum(row) * wi, A, w)  # HPCVector
```

See also: [`map_rows`](@ref) for CPU fallback version (handles arbitrary closures)
"""
function map_rows_gpu(f, A...)
    isempty(A) && error("map_rows_gpu requires at least one argument")

    # Get target partition from first argument
    target_partition = _get_row_partition(A[1])

    # Align all arguments to target partition
    aligned = map(a -> _align_to_partition(a, target_partition), A)

    # Check if all inputs are HPCMatrix (can use GPU kernel path)
    all_matrices = all(a -> a isa HPCMatrix, aligned)

    if all_matrices
        # Extract raw matrices and use GPU kernel (specialized by Metal extension for MtlMatrix)
        raw_matrices = map(a -> a.A, aligned)
        local_result = _map_rows_gpu_kernel(f, raw_matrices...)
    else
        # Mixed HPCVector/HPCMatrix: use broadcasting approach
        # HPCVector.v passes through, HPCMatrix.A gets transposed and reinterpreted
        local_arrays = map(aligned) do a
            if a isa HPCVector
                a.v  # Vector{T} - each element is a "row"
            else
                _to_svectors(a.A)  # Vector{SVector{K,T}} - each SVector is a row
            end
        end

        # Broadcast f over all local arrays
        results = f.(local_arrays...)

        # Convert results back to appropriate type
        local_result = _from_result(results)
    end

    # Wrap in MPI type using first argument's partition info
    first_arg = aligned[1]
    row_partition = first_arg isa HPCVector ? first_arg.partition : first_arg.row_partition
    # For HPCVector, structural_hash == partition hash, so reuse directly
    # For HPCMatrix, structural_hash includes more info, so just compute partition hash (cached)
    if first_arg isa HPCVector
        hash = first_arg.structural_hash
    else
        # HPCMatrix: compute partition hash (uses cache)
        hash = compute_partition_hash(row_partition)
    end

    # Get backend from first argument, but adjust T to match result's element type
    # (e.g., norm of ComplexF64 returns Float64)
    orig_backend = first_arg.backend
    result_T = eltype(local_result)
    orig_Ti = indextype_backend(typeof(orig_backend))
    result_backend = HPCBackend{result_T,orig_Ti,typeof(orig_backend.device),typeof(orig_backend.comm),typeof(orig_backend.solver)}(
        orig_backend.device, orig_backend.comm, orig_backend.solver)

    if local_result isa AbstractMatrix
        return HPCMatrix(
            hash,
            row_partition,
            [1, size(local_result, 2) + 1],  # Full columns on each rank
            local_result,
            result_backend
        )
    else
        return HPCVector(
            hash,
            row_partition,
            local_result,
            result_backend
        )
    end
end

"""
    map_rows(f, A...)

Apply function `f` to corresponding rows of distributed arrays, with CPU fallback.

This is the safe version that handles functions with arbitrary closures by
converting GPU arrays to CPU, applying the function, and converting back.
Use `map_rows_gpu` for performance-critical inner loops where `f` is isbits-compatible.

# Arguments
- `f`: Function to apply to each row (can capture non-isbits data)
- `A...`: One or more distributed arrays (HPCVector or HPCMatrix)

# Returns
- HPCVector or HPCMatrix depending on the return type of `f`

See also: [`map_rows_gpu`](@ref) for GPU-native version (requires isbits closures)
"""
function map_rows(f, A...)
    isempty(A) && error("map_rows requires at least one argument")

    # Get a CPU backend with same T, Ti, comm, solver as first arg
    first_backend = A[1] isa HPCVector ? A[1].backend : A[1].backend
    T = eltype_backend(typeof(first_backend))
    Ti = indextype_backend(typeof(first_backend))
    cpu_backend = HPCBackend{T,Ti,DeviceCPU,typeof(first_backend.comm),typeof(first_backend.solver)}(
        DeviceCPU(), first_backend.comm, first_backend.solver)

    # Convert all args to CPU
    cpu_args = map(a -> to_backend(a, cpu_backend), A)

    # Apply function on CPU (handles arbitrary closures)
    result_cpu = map_rows_gpu(f, cpu_args...)

    # Convert back to original device, but with result's element type
    # (e.g., norm of ComplexF64 returns Float64)
    result_T = eltype(result_cpu isa HPCVector ? result_cpu.v : result_cpu.A)
    target_backend = HPCBackend{result_T,Ti,typeof(first_backend.device),typeof(first_backend.comm),typeof(first_backend.solver)}(
        first_backend.device, first_backend.comm, first_backend.solver)
    return to_backend(result_cpu, target_backend)
end

# ============================================================================
# Backend Conversion Helpers for Distributed Types (legacy, simplified)
# ============================================================================

"""
    _to_same_backend(result, template)

Convert a distributed array to the same backend as the template.
This is a simplified helper that uses to_backend() internally.
"""
_to_same_backend(result::HPCVector, template::HPCVector) = to_backend(result, template.backend)
_to_same_backend(result::HPCMatrix, template::HPCMatrix) = to_backend(result, template.backend)
_to_same_backend(result::HPCVector, template::HPCMatrix) = to_backend(result, template.backend)

"""
    vertex_indices(A::AbstractHPCVector)

Create a HPCVector of vertex indices (1:n) with the same partitioning and backend as `A`.

This is useful for barrier functions that need to index into captured arrays.
The indices are created on the same device (CPU/GPU) as the input array.

# Arguments
- `A`: A distributed array whose partitioning and backend to match

# Returns
- HPCVector{Int,AV} where AV matches the integer array type for A's backend

# Example
```julia
Dz = ...  # HPCVector on GPU
indices = vertex_indices(Dz)  # GPU HPCVector of 1:n
map_rows_gpu(f, indices, Dz)  # f receives (j, Dz[j,:])
```
"""
function vertex_indices(A::HPCVector{T,B}) where {T, B<:HPCBackend}
    # Get this rank's local row range (global indices)
    rank = comm_rank(A.backend.comm)
    start_idx = A.partition[rank + 1]
    end_idx = A.partition[rank + 2] - 1

    # Create local indices as global row numbers for this rank
    local_indices = collect(start_idx:end_idx)

    # Create CPU backend for initial construction (preserve T, Ti from source)
    Ti = indextype_backend(B)
    cpu_backend = HPCBackend{T,Ti,DeviceCPU,typeof(A.backend.comm),typeof(A.backend.solver)}(
        DeviceCPU(), A.backend.comm, A.backend.solver)
    indices_cpu = HPCVector_local(local_indices, cpu_backend)

    # Convert to same device as A
    return to_backend(indices_cpu, A.backend)
end

function vertex_indices(A::HPCMatrix{T,B}) where {T, B<:HPCBackend}
    # Get this rank's local row range (global indices)
    rank = comm_rank(A.backend.comm)
    start_idx = A.row_partition[rank + 1]
    end_idx = A.row_partition[rank + 2] - 1

    # Create local indices as global row numbers for this rank
    local_indices = collect(start_idx:end_idx)

    # Create CPU backend for initial construction (preserve T, Ti from source)
    Ti = indextype_backend(B)
    cpu_backend = HPCBackend{T,Ti,DeviceCPU,typeof(A.backend.comm),typeof(A.backend.solver)}(
        DeviceCPU(), A.backend.comm, A.backend.solver)
    indices_cpu = HPCVector_local(local_indices, cpu_backend)

    # Convert to same device as A
    return to_backend(indices_cpu, A.backend)
end


# ============================================================================
# Base.zeros for Distributed Types
# ============================================================================

# Helper to create a zero array with the correct device type
_zeros_device(::DeviceCPU, ::Type{T}, dims...) where T = zeros(T, dims...)
# GPU extensions will add methods for DeviceMetal and DeviceCUDA

"""
    Base.zeros(::Type{T}, ::Type{HPCVector}, backend::HPCBackend, n::Integer) where T

Create a distributed zero vector of length `n` with element type `T`.

The vector is uniformly partitioned across ranks according to the backend's comm.

# Examples
```julia
# CPU zero vector with serial backend
backend = backend_cpu_serial()
v = zeros(Float64, HPCVector, backend, 100)

# CPU zero vector with MPI backend
backend = backend_cpu_mpi(MPI.COMM_WORLD)
v = zeros(Float64, HPCVector, backend, 100)
```
"""
function Base.zeros(::Type{T}, ::Type{HPCVector}, backend::HPCBackend, n::Integer) where T
    comm = backend.comm
    nranks = comm_size(comm)
    rank = comm_rank(comm)

    partition = uniform_partition(n, nranks)
    local_size = partition[rank + 2] - partition[rank + 1]

    local_v = _zeros_device(backend.device, T, local_size)
    hash = compute_partition_hash(partition)

    return HPCVector{T, typeof(backend)}(hash, partition, local_v, backend)
end

"""
    Base.zeros(::Type{HPCVector}, backend::HPCBackend, n::Integer)

Create a distributed zero vector of length `n` with default element type Float64.
"""
Base.zeros(::Type{HPCVector}, backend::HPCBackend, n::Integer) = zeros(Float64, HPCVector, backend, n)

"""
    Base.zeros(::Type{T}, ::Type{HPCMatrix}, backend::HPCBackend, m::Integer, n::Integer) where T

Create a distributed zero matrix of size `m × n` with element type `T`.

The matrix is row-partitioned across ranks according to the backend's comm.

# Examples
```julia
# CPU zero matrix
backend = backend_cpu_serial()
A = zeros(Float64, HPCMatrix, backend, 100, 50)
```
"""
function Base.zeros(::Type{T}, ::Type{HPCMatrix}, backend::HPCBackend, m::Integer, n::Integer) where T
    comm = backend.comm
    nranks = comm_size(comm)
    rank = comm_rank(comm)

    row_partition = uniform_partition(m, nranks)
    col_partition = uniform_partition(n, nranks)  # Used for transpose operations
    local_nrows = row_partition[rank + 2] - row_partition[rank + 1]

    local_A = _zeros_device(backend.device, T, local_nrows, n)

    # Compute structural hash eagerly
    structural_hash = compute_dense_structural_hash(row_partition, col_partition, size(local_A), backend.comm)

    return HPCMatrix{T, typeof(backend)}(structural_hash, row_partition, col_partition, local_A, backend)
end

"""
    Base.zeros(::Type{HPCMatrix}, backend::HPCBackend, m::Integer, n::Integer)

Create a distributed zero matrix of size `m × n` with default element type Float64.
"""
Base.zeros(::Type{HPCMatrix}, backend::HPCBackend, m::Integer, n::Integer) = zeros(Float64, HPCMatrix, backend, m, n)

"""
    Base.zeros(::Type{T}, ::Type{Ti}, ::Type{HPCSparseMatrix}, backend::HPCBackend, m::Integer, n::Integer) where {T,Ti<:Integer}

Create a distributed zero sparse matrix of size `m × n`.

A zero sparse matrix has no nonzero entries, so the resulting matrix has:
- Empty `rowptr` (all ones)
- Empty `colval` and `nzval`

# Examples
```julia
# CPU zero sparse matrix
backend = backend_cpu_serial()
A = zeros(Float64, Int, HPCSparseMatrix, backend, 100, 100)
```
"""
function Base.zeros(::Type{T}, ::Type{Ti}, ::Type{HPCSparseMatrix}, backend::HPCBackend, m::Integer, n::Integer) where {T, Ti<:Integer}
    comm = backend.comm
    nranks = comm_size(comm)
    rank = comm_rank(comm)

    row_partition = uniform_partition(m, nranks)
    col_partition = uniform_partition(n, nranks)
    local_nrows = row_partition[rank + 2] - row_partition[rank + 1]

    # Empty sparse structure
    rowptr = ones(Ti, local_nrows + 1)  # All rows have 0 entries
    colval = Ti[]
    nzval = _zeros_device(backend.device, T, 0)  # Empty but correct type
    col_indices = Int[]  # No columns referenced

    # For CPU, rowptr_target/colval_target are the same as rowptr/colval
    # For GPU, they would be GPU copies (but empty arrays don't matter)
    rowptr_target = _to_target_device(rowptr, backend.device)
    colval_target = _to_target_device(colval, backend.device)

    # Compute structural hash eagerly
    structural_hash = compute_structural_hash(row_partition, col_indices, rowptr, colval, backend.comm)

    return HPCSparseMatrix{T, Ti, typeof(backend)}(
        structural_hash,
        row_partition,
        col_partition,
        col_indices,
        rowptr,
        colval,
        nzval,
        local_nrows,
        0,  # ncols_compressed = 0 (no columns referenced)
        nothing,  # cached_transpose
        true,  # cached_symmetric (zero matrix is symmetric)
        rowptr_target,
        colval_target,
        backend
    )
end

"""
    Base.zeros(::Type{HPCSparseMatrix}, backend::HPCBackend, m::Integer, n::Integer)

Create a distributed zero sparse matrix of size `m × n` with default types Float64 and Int.
"""
Base.zeros(::Type{HPCSparseMatrix}, backend::HPCBackend, m::Integer, n::Integer) = zeros(Float64, Int, HPCSparseMatrix, backend, m, n)

# ============================================================================
# Precompilation Workload
# ============================================================================

using PrecompileTools

# TODO: Re-enable after HPCBackend refactor is complete
# Temporarily disabled during the HPCVector/HPCMatrix/HPCSparseMatrix refactor
#=
@setup_workload begin
    # Small test data for precompilation (runs with single MPI rank)
    n = 8

    # Sparse matrix (tridiagonal) - Float64
    I_sp = Int[]; J_sp = Int[]; V_sp = Float64[]
    for i in 1:n
        push!(I_sp, i); push!(J_sp, i); push!(V_sp, 4.0)
        if i > 1
            push!(I_sp, i); push!(J_sp, i-1); push!(V_sp, -1.0)
            push!(I_sp, i-1); push!(J_sp, i); push!(V_sp, -1.0)
        end
    end
    A_sparse_f64 = sparse(I_sp, J_sp, V_sp, n, n)

    # Sparse matrix - ComplexF64
    A_sparse_c64 = sparse(I_sp, J_sp, ComplexF64.(V_sp), n, n)

    # Dense matrix - Float64
    A_dense_f64 = Float64[i == j ? 4.0 : (abs(i-j) == 1 ? -1.0 : 0.0) for i in 1:n, j in 1:n]

    # Dense matrix - ComplexF64
    A_dense_c64 = ComplexF64.(A_dense_f64)

    # Vectors
    v_f64 = ones(Float64, n)
    v_c64 = ones(ComplexF64, n)

    # Identity for SPD construction
    I_sparse = sparse(1.0 * LinearAlgebra.I, n, n)

    @compile_workload begin
        # === MPI Jail Escape ===
        # When precompiling under mpiexec, the subprocess inherits MPI environment
        # variables but isn't part of the MPI job. Clean them to allow MPI.Init()
        # to succeed as a fresh single-rank process.
        for k in collect(keys(ENV))
            if startswith(k, "PMI") || startswith(k, "PMIX") || startswith(k, "OMPI_") || startswith(k, "MPI_")
                delete!(ENV, k)
            end
        end

        MPI.Init()
        backend = backend_cpu_mpi(MPI.COMM_WORLD)

        # === HPCVector operations (Float64) ===
        v = HPCVector(v_f64, backend)
        w = HPCVector(2.0 .* v_f64, backend)
        _ = v + w
        _ = v - w
        _ = 2.0 * v
        _ = v * 2.0
        _ = norm(v)
        _ = dot(v, w)
        _ = conj(v)
        _ = length(v)
        _ = size(v)

        # HPCVector (ComplexF64)
        vc = HPCVector(v_c64, backend)
        _ = conj(vc)
        _ = norm(vc)

        # === HPCSparseMatrix operations (Float64) ===
        A = HPCSparseMatrix{Float64}(A_sparse_f64, backend)
        B = HPCSparseMatrix{Float64}(A_sparse_f64, backend)
        _ = A + B
        _ = A - B
        _ = 2.0 * A
        _ = A * v
        _ = A * B
        _ = transpose(A)
        At = HPCSparseMatrix(transpose(A))
        _ = size(A)
        _ = nnz(A)
        _ = norm(A)

        # HPCSparseMatrix (ComplexF64)
        Ac = HPCSparseMatrix{ComplexF64}(A_sparse_c64, backend)
        _ = Ac * vc

        # === HPCMatrix operations (Float64) ===
        D = HPCMatrix(A_dense_f64)
        _ = 2.0 * D
        _ = D * v
        _ = transpose(D)
        Dt = copy(transpose(D))  # Materialize dense transpose
        _ = size(D)
        _ = norm(D)

        # HPCMatrix (ComplexF64)
        Dc = HPCMatrix(A_dense_c64)
        _ = Dc * vc

        # === Mixed operations ===
        _ = A * D  # Sparse * Dense

        # === Indexing (scalar indexing removed to prevent MPI desync) ===
        # Slice indexing still works:
        _ = v[1:2]
        _ = D[:, 1]

        # === Factorization (MUMPS) ===
        # Make symmetric positive definite: A + A^T + 10I
        At_mat = HPCSparseMatrix(transpose(A))
        I_dist = HPCSparseMatrix{Float64}(I_sparse)
        A_spd = A + At_mat + I_dist * 10.0
        F = LinearAlgebra.ldlt(A_spd)
        x = F \ v
        finalize!(F)

        # LU factorization
        F_lu = LinearAlgebra.lu(A)
        x = F_lu \ v
        finalize!(F_lu)

        # === Block operations ===
        _ = cat(v, w; dims=1)
        _ = blockdiag(A, B)

        # === Conversions ===
        _ = Vector(v)
        _ = Matrix(D)
        _ = SparseMatrixCSC(A)

        # Clear caches
        clear_plan_cache!()
    end
end
=#

end # module HPCSparseArrays
