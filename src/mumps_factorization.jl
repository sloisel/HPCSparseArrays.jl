"""
MUMPS-based distributed sparse factorization.

Uses MUMPS with distributed matrix input (ICNTL(18)=3) for efficient
parallel direct solve of sparse linear systems.

Caching strategy:
- lu(A) / ldlt(A): Creates fresh MUMPS factorization each time. User owns
  the factorization and is responsible for calling finalize!().
- A \\ b: Uses backslash cache keyed by structural hash. On cache hit,
  updates values and refactorizes (skips expensive symbolic analysis).
  On cache miss, creates full factorization and caches it.
"""

using MPI
using SparseArrays
using MUMPS
using MUMPS: Mumps, set_icntl!, MUMPS_INT, MUMPS_INT8, suppress_printing!
import MUMPS: invoke_mumps_unsafe!

# ============================================================================
# MUMPS Factorization Type
# ============================================================================

# Helper to determine the MUMPS-compatible internal type
_mumps_internal_type(::Type{T}) where T<:Real = Float64
_mumps_internal_type(::Type{T}) where T<:Complex = ComplexF64

"""
    MUMPSFactorization{Tin, Bin, Tinternal}

Distributed MUMPS factorization result. Can be reused for multiple solves.

Type parameters:
- `Tin`: Original element type of input matrix (e.g., Float32)
- `Bin`: Original HPCBackend type of input matrix
- `Tinternal`: MUMPS-compatible type used internally (Float64 or ComplexF64)

This allows factorizing matrices with any element type or backend (e.g., GPU Float32),
with automatic conversion to/from MUMPS-compatible types during factorization and solve.
"""
mutable struct MUMPSFactorization{Tin, Bin<:HPCBackend, Tinternal}
    mumps::Any  # Mumps{Tinternal,R}
    irn_loc::Vector{MUMPS_INT}  # Row indices (kept alive for MUMPS pointers)
    jcn_loc::Vector{MUMPS_INT}  # Column indices (kept alive for MUMPS pointers)
    a_loc::Vector{Tinternal}    # Values (kept alive for MUMPS pointers, updated for refactorize)
    nzval_perm::Vector{Int}     # Permutation: a_loc[k] = nzval[nzval_perm[k]]
    n::Int
    symmetric::Bool
    row_partition::Vector{Int}
    col_partition::Vector{Int}
    rhs_buffer::Vector{Tinternal}
    backend::Bin  # Original backend for comm access during solve
end

Base.size(F::MUMPSFactorization) = (F.n, F.n)
Base.eltype(::MUMPSFactorization{Tin}) where Tin = Tin

# ============================================================================
# Backslash Cache
# ============================================================================
#
# The backslash operator (A \ b) caches complete factorizations by structural hash.
# On cache hit, we update values and refactorize (skip symbolic analysis).
# The cache owns the MUMPS objects - they are never finalized until cache is cleared.

# Cache mapping (structural_hash, symmetric, element_type) -> MUMPSFactorization
const _mumps_backslash_cache = Dict{Tuple{NTuple{32,UInt8}, Bool, DataType}, Any}()

"""
    clear_mumps_backslash_cache!()

Clear the MUMPS backslash cache. This is a collective operation that must
be called on all MPI ranks together.
"""
function clear_mumps_backslash_cache!()
    for (key, F) in _mumps_backslash_cache
        F.mumps._finalized = false
        MUMPS.finalize!(F.mumps)
    end
    empty!(_mumps_backslash_cache)
end

# Legacy alias for backwards compatibility
const clear_mumps_analysis_cache! = clear_mumps_backslash_cache!

# For diagnostics: expose cache reference
const _mumps_analysis_cache = _mumps_backslash_cache

# ============================================================================
# Extract COO from HPCSparseMatrix
# ============================================================================

"""
    extract_local_coo(A::HPCSparseMatrix{T,Ti,B}; symmetric::Bool=false) where {T,Ti,B}

Extract local COO entries from a distributed sparse matrix for MUMPS input.
Returns (irn_loc, jcn_loc, a_loc, nzval_perm) with 1-based global indices.

The nzval_perm array maps COO indices to AT.nzval indices, enabling fast
value updates: a_loc .= AT.nzval[nzval_perm]

For symmetric matrices, only lower triangular entries (row >= col) are extracted.
"""
function extract_local_coo(A::HPCSparseMatrix{T,Ti,B}; symmetric::Bool=false) where {T,Ti,B}
    comm = A.backend.comm
    rank = comm_rank(comm)

    row_start = A.row_partition[rank + 1]

    irn_loc = MUMPS_INT[]
    jcn_loc = MUMPS_INT[]
    a_loc = T[]
    nzval_perm = Int[]

    # _get_csc(A) reconstructs the underlying CSC storage from explicit arrays
    # Columns in the CSC correspond to local rows (CSR stored as transpose of CSC)
    # A.col_indices maps local column indices to global
    AT = _get_csc(A)

    for local_row in 1:AT.n  # AT.n = number of local rows
        global_row = row_start + local_row - 1

        for ptr in AT.colptr[local_row]:(AT.colptr[local_row + 1] - 1)
            local_col_idx = AT.rowval[ptr]
            global_col = A.col_indices[local_col_idx]
            val = AT.nzval[ptr]

            # For symmetric matrices, only include lower triangular (row >= col)
            if !symmetric || global_row >= global_col
                push!(irn_loc, MUMPS_INT(global_row))
                push!(jcn_loc, MUMPS_INT(global_col))
                push!(a_loc, val)
                push!(nzval_perm, ptr)
            end
        end
    end

    return irn_loc, jcn_loc, a_loc, nzval_perm
end

# ============================================================================
# Create Fresh MUMPS Factorization (for lu/ldlt)
# ============================================================================

"""
    _create_fresh_mumps_factorization(A::HPCSparseMatrix{T, Ti, B}, symmetric::Bool) where {T, Ti, B}

Create a fresh MUMPS factorization (no caching). Used by lu() and ldlt().
The caller owns the factorization and must call finalize!() when done.
"""
function _create_fresh_mumps_factorization(A::HPCSparseMatrix{T, Ti, B}, symmetric::Bool) where {T, Ti, B}
    comm = A.backend.comm
    rank = comm_rank(comm)

    m, n = size(A)
    @assert m == n "Matrix must be square for factorization"

    # Convert to MUMPS-compatible type
    Tinternal = _mumps_internal_type(T)

    # Extract local COO entries
    irn_loc, jcn_loc, a_loc_orig, nzval_perm = extract_local_coo(A; symmetric=symmetric)
    a_loc = Tinternal.(a_loc_orig)
    nz_loc = length(a_loc)

    # Create MUMPS instance
    mumps_sym = symmetric ? MUMPS.mumps_definite : MUMPS.mumps_unsymmetric
    mumps = Mumps{Tinternal}(mumps_sym, MUMPS.default_icntl, MUMPS.default_cntl64)
    mumps._finalized = true  # Disable MUMPS GC finalizer to avoid post-MPI crash

    # Suppress all MUMPS output
    suppress_printing!(mumps)

    # Configure MUMPS for distributed input
    set_icntl!(mumps, 5, 0; displaylevel=0)    # Assembled matrix format
    set_icntl!(mumps, 14, 50; displaylevel=0)  # Memory relaxation (50%)
    set_icntl!(mumps, 18, 3; displaylevel=0)   # Distributed matrix input
    set_icntl!(mumps, 20, 0; displaylevel=0)   # Dense RHS
    set_icntl!(mumps, 21, 0; displaylevel=0)   # Centralized solution on host
    set_icntl!(mumps, 7, 5; displaylevel=0)    # METIS ordering (better fill-in)

    # Enable OpenMP threading in MUMPS
    omp_threads = parse(Int, get(ENV, "OMP_NUM_THREADS", "1"))
    set_icntl!(mumps, 16, omp_threads; displaylevel=0)

    # Set matrix dimension and data pointers
    mumps.n = MUMPS_INT(n)
    mumps.nz_loc = MUMPS_INT(nz_loc)
    mumps.nnz_loc = MUMPS_INT8(nz_loc)
    mumps.irn_loc = pointer(irn_loc)
    mumps.jcn_loc = pointer(jcn_loc)
    mumps.a_loc = pointer(a_loc)

    # Analysis phase (job = 1)
    mumps.job = MUMPS_INT(1)
    invoke_mumps_unsafe!(mumps)
    _check_mumps_error(mumps, "analysis")

    # Factorization phase (job = 2)
    mumps.job = MUMPS_INT(2)
    invoke_mumps_unsafe!(mumps)
    _check_mumps_error(mumps, "factorization")

    # Pre-allocate RHS buffer on rank 0
    rhs_buffer = rank == 0 ? zeros(Tinternal, n) : Tinternal[]

    return MUMPSFactorization{T, B, Tinternal}(
        mumps, irn_loc, jcn_loc, a_loc, nzval_perm,
        n, symmetric, copy(A.row_partition), copy(A.col_partition),
        rhs_buffer, A.backend
    )
end

"""
    _check_mumps_error(mumps::Mumps, phase::String)

Check for MUMPS errors and throw descriptive exception if found.
"""
function _check_mumps_error(mumps::Mumps, phase::String)
    if mumps.infog[1] < 0
        error("MUMPS $phase error: INFOG(1) = $(mumps.infog[1]), INFOG(2) = $(mumps.infog[2])")
    end
end

# ============================================================================
# LinearAlgebra Interface (lu / ldlt)
# ============================================================================

"""
    LinearAlgebra.lu(A::HPCSparseMatrix{T,Ti,HPCBackend{T,Ti,D,C,SolverMUMPS}}) where {T,Ti,D,C}

Compute LU factorization of a distributed sparse matrix using MUMPS.
Returns a `MUMPSFactorization` for use with `\\` or `solve`.

Creates a fresh factorization each time (no caching). The caller is responsible
for calling `finalize!(F)` when the factorization is no longer needed.

Note: This method is specific to MUMPS backends. GPU backends (cuDSS) define their own
lu method in the CUDA extension.
"""
function LinearAlgebra.lu(A::HPCSparseMatrix{T,Ti,HPCBackend{T,Ti,D,C,SolverMUMPS}}) where {T,Ti,D,C}
    return _create_fresh_mumps_factorization(A, false)
end

"""
    LinearAlgebra.ldlt(A::HPCSparseMatrix{T,Ti,HPCBackend{T,Ti,D,C,SolverMUMPS}}) where {T,Ti,D,C}

Compute LDLT factorization of a distributed symmetric sparse matrix using MUMPS.
The matrix must be symmetric; only the lower triangular part is used.
Returns a `MUMPSFactorization` for use with `\\` or `solve`.

Creates a fresh factorization each time (no caching). The caller is responsible
for calling `finalize!(F)` when the factorization is no longer needed.

Note: This method is specific to MUMPS backends. GPU backends (cuDSS) define their own
ldlt method in the CUDA extension.
"""
function LinearAlgebra.ldlt(A::HPCSparseMatrix{T,Ti,HPCBackend{T,Ti,D,C,SolverMUMPS}}) where {T,Ti,D,C}
    return _create_fresh_mumps_factorization(A, true)
end

# ============================================================================
# Solve Interface
# ============================================================================

# Helper to create output vector with original type/backend
function _create_output_vector(::Type{Tin}, backend::B, n::Int, partition) where {Tin, B<:HPCBackend}
    v_cpu = zeros(Tin, n)
    v_dist = HPCVector(v_cpu, backend; partition=partition)
    # Convert to original device if needed (extensions handle GPU conversion)
    return _convert_vector_to_device(v_dist, backend.device)
end

# Convert HPCVector to match a device type.
# WARNING: This function exists ONLY for MUMPS, which is a CPU-only solver.
# MUMPS requires GPU->CPU->GPU cycling. Do NOT use this for general operations.
# The base module only defines the identity case (CPU device).
# Extensions define CPU->GPU conversions (e.g., Vector -> MtlVector).
_convert_vector_to_device(v::HPCVector, ::DeviceCPU) = v

"""
    solve(F::MUMPSFactorization{Tin, Bin, Tinternal}, b::HPCVector) where {Tin, Bin, Tinternal}

Solve the linear system A*x = b using the precomputed MUMPS factorization.

The input vector b can have any compatible element type and backend.
The result is returned with the same element type and backend as the
original matrix used for factorization.
"""
function solve(F::MUMPSFactorization{Tin, Bin, Tinternal}, b::HPCVector) where {Tin, Bin, Tinternal}
    # Create output vector with original type/backend
    x = _create_output_vector(Tin, F.backend, F.n, b.partition)
    solve!(x, F, b)
    return x
end

"""
    solve!(x::HPCVector, F::MUMPSFactorization{Tin, Bin, Tinternal}, b::HPCVector) where {Tin, Bin, Tinternal}

Solve A*x = b in-place using MUMPS factorization.

Automatically converts inputs to CPU Float64/ComplexF64 for MUMPS,
then converts results back to the element type and backend of x.
"""
function solve!(x::HPCVector, F::MUMPSFactorization{Tin, Bin, Tinternal}, b::HPCVector) where {Tin, Bin, Tinternal}
    comm = F.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    # Convert b to CPU with internal type
    b_cpu = _ensure_cpu(b.v)
    b_internal = Tinternal.(b_cpu)

    # Gather RHS to rank 0
    counts = Int32[b.partition[r+2] - b.partition[r+1] for r in 0:nranks-1]

    # Use direct MPI calls for Gatherv/Scatterv since we don't have thin wrappers for these
    # (they have complex root parameter semantics)
    mpi_comm = _get_mpi_comm(comm)
    if rank == 0
        MPI.Gatherv!(b_internal, MPI.VBuffer(F.rhs_buffer, counts), mpi_comm; root=0)

        # Set RHS in MUMPS
        F.mumps.nrhs = MUMPS_INT(1)
        F.mumps.lrhs = MUMPS_INT(F.n)
        F.mumps.rhs = pointer(F.rhs_buffer)
    else
        MPI.Gatherv!(b_internal, nothing, mpi_comm; root=0)
    end

    # Solve phase (job = 3)
    F.mumps.job = MUMPS_INT(3)
    invoke_mumps_unsafe!(F.mumps)
    _check_mumps_error(F.mumps, "solve")

    # Scatter solution from rank 0
    # Result is in F.rhs_buffer on rank 0, need local buffer for scatter
    local_size = x.partition[rank+2] - x.partition[rank+1]
    x_internal = Vector{Tinternal}(undef, local_size)

    if rank == 0
        MPI.Scatterv!(MPI.VBuffer(F.rhs_buffer, counts), x_internal, mpi_comm; root=0)
    else
        MPI.Scatterv!(nothing, x_internal, mpi_comm; root=0)
    end

    # Convert result back to original type and backend
    Tx = eltype(x)
    x_converted = Tx.(x_internal)
    _copy_to_vector!(x, x_converted)

    return x
end

# Helper to extract MPI.Comm from AbstractComm (for operations not wrapped in thin wrappers)
_get_mpi_comm(c::CommMPI) = c.comm
_get_mpi_comm(::CommSerial) = error("Gatherv/Scatterv not supported for CommSerial in MUMPS solve")

# Helper to copy values into a HPCVector (handles GPU arrays)
# Unified: _convert_array handles CPU->GPU conversion, copyto! handles the copy
function _copy_to_vector!(x::HPCVector{T,B}, values::Vector) where {T,B}
    copyto!(x.v, _convert_array(values, x.backend.device))
    return x
end

"""
    Base.:\\(F::MUMPSFactorization, b::HPCVector)

Solve A*x = b using the backslash operator.
"""
function Base.:\(F::MUMPSFactorization, b::HPCVector)
    return solve(F, b)
end

# ============================================================================
# Refactorize and Solve (for backslash cache)
# ============================================================================

"""
    _update_values_and_refactorize!(F::MUMPSFactorization{Tin,Bin,Tinternal}, A::HPCSparseMatrix) where {Tin,Bin,Tinternal}

Update values in a cached factorization from matrix A and refactorize.
Skips symbolic analysis (job=1), only does numeric factorization (job=2).
"""
function _update_values_and_refactorize!(F::MUMPSFactorization{Tin,Bin,Tinternal}, A::HPCSparseMatrix) where {Tin,Bin,Tinternal}
    # Update values using cached permutation (zero-allocation loop)
    nzval_cpu = _ensure_cpu(A.nzval)
    @inbounds for k in eachindex(F.nzval_perm)
        F.a_loc[k] = Tinternal(nzval_cpu[F.nzval_perm[k]])
    end

    # Refactorize (skip analysis - symbolic factorization already done)
    F.mumps.job = MUMPS_INT(2)
    invoke_mumps_unsafe!(F.mumps)
    _check_mumps_error(F.mumps, "refactorization")
end

"""
    _refactorize_and_solve!(F::MUMPSFactorization, A::HPCSparseMatrix, b::HPCVector)

Update values from A, refactorize, and solve. Used by backslash cache on cache hit.
"""
function _refactorize_and_solve!(F::MUMPSFactorization{Tin,Bin,Tinternal}, A::HPCSparseMatrix, b::HPCVector) where {Tin,Bin,Tinternal}
    _update_values_and_refactorize!(F, A)
    return solve(F, b)
end

# ============================================================================
# Cleanup Interface
# ============================================================================

"""
    finalize!(F::MUMPSFactorization)

Release MUMPS resources. Must be called on all ranks together.

This properly finalizes the MUMPS object and frees all associated memory.
After calling finalize!, the factorization should not be used.
"""
function finalize!(F::MUMPSFactorization)
    F.mumps._finalized = false  # Re-enable MUMPS finalization
    MUMPS.finalize!(F.mumps)
    return F
end
