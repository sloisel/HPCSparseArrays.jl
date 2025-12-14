"""
MUMPS-based distributed sparse factorization.

Uses MUMPS with distributed matrix input (ICNTL(18)=3) for efficient
parallel direct solve of sparse linear systems.
"""

using MPI
using SparseArrays
using MUMPS
using MUMPS: Mumps, set_icntl!, MUMPS_INT, MUMPS_INT8, suppress_printing!
import MUMPS: invoke_mumps_unsafe!

# ============================================================================
# MUMPS Factorization Type
# ============================================================================

"""
    MUMPSFactorizationMPI{T}

Distributed MUMPS factorization result.

Wraps a MUMPS object that has been analyzed and factorized. The factorization
can be reused for multiple solves with different right-hand sides.

# Important: Manual Cleanup Required

Unlike other types in LinearAlgebraMPI, factorization objects require explicit cleanup
via `finalize!(F)` when you are done using them. This is necessary because MUMPS
cleanup routines call MPI functions, and Julia's garbage collector may run finalizers
after MPI has shut down, causing crashes.

```julia
F = lu(A)
x = F \\ b
finalize!(F)  # Required! Call when done with factorization
```

If you forget to call `finalize!`, the program will still work but MUMPS resources
will not be released until program exit (minor memory leak, but no crash).

# Fields
- `mumps`: MUMPS solver object
- `irn_loc`: Row indices (GC-protected)
- `jcn_loc`: Column indices (GC-protected)
- `a_loc`: Matrix values (GC-protected)
- `n`: Matrix dimension
- `symmetric`: Whether this is a symmetric factorization (LDLT vs LU)
- `row_partition`: Row partition for VectorMPI compatibility
- `rhs_buffer`: Pre-allocated buffer for RHS (on rank 0)

Note: For complex matrices, MUMPS uses Mumps{ComplexF64, Float64} (complex values,
real control parameters), so we use a more flexible type for the mumps field.
"""
mutable struct MUMPSFactorizationMPI{T}
    mumps::Any  # Mumps{T,R} where R is the real type (Float64 for both real and complex)
    irn_loc::Vector{MUMPS_INT}
    jcn_loc::Vector{MUMPS_INT}
    a_loc::Vector{T}
    n::Int
    symmetric::Bool
    row_partition::Vector{Int}
    rhs_buffer::Vector{T}
end

Base.size(F::MUMPSFactorizationMPI) = (F.n, F.n)
Base.eltype(::MUMPSFactorizationMPI{T}) where T = T

# ============================================================================
# Extract COO from SparseMatrixMPI
# ============================================================================

"""
    extract_local_coo(A::SparseMatrixMPI{T}; symmetric::Bool=false)

Extract local COO entries from a distributed sparse matrix for MUMPS input.
Returns (irn_loc, jcn_loc, a_loc) with 1-based global indices.

For symmetric matrices, only lower triangular entries (row >= col) are extracted.
"""
function extract_local_coo(A::SparseMatrixMPI{T}; symmetric::Bool=false) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    row_start = A.row_partition[rank + 1]

    irn_loc = MUMPS_INT[]
    jcn_loc = MUMPS_INT[]
    a_loc = T[]

    # A.A.parent is the underlying CSC storage
    # Columns in A.A.parent correspond to local rows
    # A.col_indices maps local column indices to global
    AT = A.A.parent

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
            end
        end
    end

    return irn_loc, jcn_loc, a_loc
end

# ============================================================================
# Create MUMPS Factorization
# ============================================================================

"""
    _create_mumps_factorization(A::SparseMatrixMPI{T}, symmetric::Bool) where T

Create and compute a MUMPS factorization of the distributed matrix A.
"""
function _create_mumps_factorization(A::SparseMatrixMPI{T}, symmetric::Bool) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    m, n = size(A)
    @assert m == n "Matrix must be square for factorization"

    # Extract local COO entries
    irn_loc, jcn_loc, a_loc = extract_local_coo(A; symmetric=symmetric)
    nz_loc = length(a_loc)

    # Create MUMPS instance
    # sym=0: unsymmetric, sym=1: SPD, sym=2: general symmetric
    mumps_sym = symmetric ? MUMPS.mumps_definite : MUMPS.mumps_unsymmetric
    mumps = Mumps{T}(mumps_sym, MUMPS.default_icntl, MUMPS.default_cntl64)
    mumps._finalized = true  # Disable GC finalizer to avoid post-MPI crash

    # Suppress all MUMPS output
    suppress_printing!(mumps)

    # Configure MUMPS for distributed input (displaylevel=0 suppresses verbose messages)
    set_icntl!(mumps, 5, 0; displaylevel=0)    # Assembled matrix format
    set_icntl!(mumps, 14, 50; displaylevel=0)  # Memory relaxation (50%)
    set_icntl!(mumps, 18, 3; displaylevel=0)   # Distributed matrix input
    set_icntl!(mumps, 20, 0; displaylevel=0)   # Dense RHS
    set_icntl!(mumps, 21, 0; displaylevel=0)   # Centralized solution on host

    # Set matrix dimension
    mumps.n = MUMPS_INT(n)

    # Set distributed matrix data
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
    rhs_buffer = rank == 0 ? zeros(T, n) : T[]

    return MUMPSFactorizationMPI{T}(
        mumps, irn_loc, jcn_loc, a_loc,
        n, symmetric, copy(A.row_partition), rhs_buffer
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
# LinearAlgebra Interface
# ============================================================================

"""
    LinearAlgebra.lu(A::SparseMatrixMPI{T}) where T

Compute LU factorization of a distributed sparse matrix using MUMPS.

Returns a `MUMPSFactorizationMPI` that can be used with the backslash operator
or `solve` function.

**Important:** Call `finalize!(F)` when done to release MUMPS resources.
See `MUMPSFactorizationMPI` for details.

# Example
```julia
F = lu(A)
x = F \\ b
finalize!(F)
```
"""
function LinearAlgebra.lu(A::SparseMatrixMPI{T}) where T
    return _create_mumps_factorization(A, false)
end

"""
    LinearAlgebra.ldlt(A::SparseMatrixMPI{T}) where T

Compute LDLT factorization of a distributed symmetric sparse matrix using MUMPS.

The matrix must be symmetric. MUMPS uses only the lower triangular part.
Returns a `MUMPSFactorizationMPI` that can be used with the backslash operator
or `solve` function.

**Important:** Call `finalize!(F)` when done to release MUMPS resources.
See `MUMPSFactorizationMPI` for details.

# Example
```julia
F = ldlt(A)
x = F \\ b
finalize!(F)
```
"""
function LinearAlgebra.ldlt(A::SparseMatrixMPI{T}) where T
    return _create_mumps_factorization(A, true)
end

# ============================================================================
# Solve Interface
# ============================================================================

"""
    solve(F::MUMPSFactorizationMPI{T}, b::VectorMPI{T}) where T

Solve the linear system A*x = b using the precomputed MUMPS factorization.
"""
function solve(F::MUMPSFactorizationMPI{T}, b::VectorMPI{T}) where T
    x = VectorMPI(zeros(T, F.n); partition=b.partition)
    solve!(x, F, b)
    return x
end

"""
    solve!(x::VectorMPI{T}, F::MUMPSFactorizationMPI{T}, b::VectorMPI{T}) where T

Solve A*x = b in-place using MUMPS factorization.
"""
function solve!(x::VectorMPI{T}, F::MUMPSFactorizationMPI{T}, b::VectorMPI{T}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # Gather RHS to rank 0
    counts = Int32[b.partition[r+2] - b.partition[r+1] for r in 0:nranks-1]

    if rank == 0
        MPI.Gatherv!(b.v, MPI.VBuffer(F.rhs_buffer, counts), comm; root=0)

        # Set RHS in MUMPS
        F.mumps.nrhs = MUMPS_INT(1)
        F.mumps.lrhs = MUMPS_INT(F.n)
        F.mumps.rhs = pointer(F.rhs_buffer)
    else
        MPI.Gatherv!(b.v, nothing, comm; root=0)
    end

    # Solve phase (job = 3)
    F.mumps.job = MUMPS_INT(3)
    invoke_mumps_unsafe!(F.mumps)
    _check_mumps_error(F.mumps, "solve")

    # Scatter solution from rank 0
    # Result is in F.rhs_buffer on rank 0
    if rank == 0
        MPI.Scatterv!(MPI.VBuffer(F.rhs_buffer, counts), x.v, comm; root=0)
    else
        MPI.Scatterv!(nothing, x.v, comm; root=0)
    end

    return x
end

"""
    Base.:\\(F::MUMPSFactorizationMPI{T}, b::VectorMPI{T}) where T

Solve A*x = b using the backslash operator.
"""
function Base.:\(F::MUMPSFactorizationMPI{T}, b::VectorMPI{T}) where T
    return solve(F, b)
end

# ============================================================================
# Cleanup Interface
# ============================================================================

"""
    finalize!(F)

Release resources associated with a MUMPS factorization. **This must be called
explicitly** when you are done using a factorization returned by `lu()` or `ldlt()`.

Unlike most Julia objects, MUMPS factorizations cannot rely on garbage collection
for cleanup because MUMPS internally uses MPI, and Julia's GC may run after MPI
has shut down. To avoid crashes at program exit, automatic GC finalization is
disabled for these objects.

If you forget to call `finalize!`, the program will still run correctly, but
MUMPS memory will not be released until the program exits (a minor memory leak).

# Example
```julia
F = lu(A)
x = F \\ b
finalize!(F)  # Required! Free resources when done
```
"""
function finalize!(F::MUMPSFactorizationMPI)
    F.mumps._finalized = false  # Re-enable finalization
    MUMPS.finalize!(F.mumps)
    return F
end

