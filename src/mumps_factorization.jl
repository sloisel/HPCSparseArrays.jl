"""
MUMPS-based distributed sparse factorization.

Uses MUMPS with distributed matrix input (ICNTL(18)=3) for efficient
parallel direct solve of sparse linear systems.

Analysis caching: The symbolic analysis phase (ordering, symbolic factorization)
depends only on sparsity structure, not numerical values. We cache the analyzed
MUMPS object by structural hash, so subsequent factorizations with the same
structure skip the expensive analysis phase.
"""

using MPI
using SparseArrays
using MUMPS
using MUMPS: Mumps, set_icntl!, MUMPS_INT, MUMPS_INT8, suppress_printing!
import MUMPS: invoke_mumps_unsafe!

# ============================================================================
# MUMPS Automatic Finalization Management
# ============================================================================
#
# MUMPS cleanup requires synchronized MPI calls across all ranks, but Julia's
# GC runs asynchronously on each rank. This system handles automatic cleanup:
#
# 1. Each MUMPS factorization gets a unique integer ID (_mumps_count)
# 2. Objects are registered in _mumps_registry by ID
# 3. Julia's GC finalizer queues the ID to _destroy_list (no MPI calls)
# 4. When creating a new factorization, _process_finalizers() is called:
#    - All ranks broadcast their _destroy_list
#    - Lists are merged, sorted, uniqued
#    - Objects are finalized in deterministic order across all ranks
#
# This ensures synchronized cleanup without blocking in finalizers.

# Global counter for unique MUMPS object IDs
const _mumps_count = Ref{Int}(0)

# Registry mapping ID -> MUMPSFactorization (prevents GC until removed from registry)
const _mumps_registry = Dict{Int, Any}()

# List of MUMPS IDs queued for destruction by this rank's GC
const _destroy_list = Int[]

# Lock for thread-safe access to _destroy_list (finalizers may run from GC thread)
const _destroy_list_lock = ReentrantLock()

# ============================================================================
# MUMPS Analysis Cache
# ============================================================================
#
# The symbolic analysis phase (job=1) depends only on sparsity structure, not
# numerical values. We cache the analyzed MUMPS object by structural hash,
# allowing subsequent factorizations to skip analysis and only do numeric
# factorization (job=2).
#
# The cache stores MUMPSAnalysisPlan objects, which contain:
# - A pre-analyzed MUMPS object (ready for job=2)
# - The COO index arrays (structure is fixed)
# - Metadata for validation

"""
    MUMPSAnalysisPlan{T}

Cached MUMPS symbolic analysis for a given sparsity structure.
Stores a pre-analyzed MUMPS object that can be reused for numeric
factorization with different values but the same structure.
"""
mutable struct MUMPSAnalysisPlan{T}
    mumps::Any  # Mumps{T,R} after analysis (job=1)
    irn_loc::Vector{MUMPS_INT}  # Row indices (structure, immutable)
    jcn_loc::Vector{MUMPS_INT}  # Column indices (structure, immutable)
    a_loc::Vector{T}  # Value array (updated for each factorization)
    nzval_perm::Vector{Int}  # Permutation: a_loc[k] = AT.nzval[nzval_perm[k]]
    n::Int
    symmetric::Bool
    row_partition::Vector{Int}
    structural_hash::NTuple{32,UInt8}
end

# Cache mapping (structural_hash, symmetric, element_type) -> MUMPSAnalysisPlan
const _mumps_analysis_cache = Dict{Tuple{NTuple{32,UInt8}, Bool, DataType}, Any}()

"""
    clear_mumps_analysis_cache!()

Clear the MUMPS analysis cache. This is a collective operation that must
be called on all MPI ranks together.
"""
function clear_mumps_analysis_cache!()
    for (key, plan) in _mumps_analysis_cache
        # Finalize the cached MUMPS objects
        plan.mumps._finalized = false
        MUMPS.finalize!(plan.mumps)
    end
    empty!(_mumps_analysis_cache)
end

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

Note: The MUMPS object is shared with the analysis cache. The factorization
does not own the MUMPS object and should not finalize it directly.
"""
mutable struct MUMPSFactorization{Tin, Bin<:HPCBackend, Tinternal}
    id::Int  # Unique ID for finalization tracking
    mumps::Any  # Mumps{Tinternal,R} - shared with cache, do not finalize
    irn_loc::Vector{MUMPS_INT}
    jcn_loc::Vector{MUMPS_INT}
    a_loc::Vector{Tinternal}
    n::Int
    symmetric::Bool
    row_partition::Vector{Int}
    col_partition::Vector{Int}
    rhs_buffer::Vector{Tinternal}
    owns_mumps::Bool  # Whether this factorization owns the MUMPS object
    backend::Bin  # Original backend for comm access during solve
end

Base.size(F::MUMPSFactorization) = (F.n, F.n)
Base.eltype(::MUMPSFactorization{Tin}) where Tin = Tin

# ============================================================================
# Automatic Finalization Functions
# ============================================================================

"""
    _queue_for_destruction(F::MUMPSFactorization)

Julia finalizer callback. Queues the factorization ID for later synchronized
destruction. Does NOT call MPI (unsafe from GC thread).
"""
function _queue_for_destruction(F::MUMPSFactorization)
    lock(_destroy_list_lock) do
        push!(_destroy_list, F.id)
    end
    return nothing
end

"""
    _process_finalizers(comm::AbstractComm)

Process pending MUMPS finalizations in a synchronized manner across all ranks.
This is a **collective operation** - all ranks must call it together.

Called automatically when creating new factorizations. Gathers pending
destruction requests from all ranks, merges them, and finalizes in
deterministic order.
"""
function _process_finalizers(comm::AbstractComm)
    nranks = comm_size(comm)

    # Thread-safe: detach current destroy list, replace with empty
    local_list = lock(_destroy_list_lock) do
        list = copy(_destroy_list)
        empty!(_destroy_list)
        list
    end

    # Allgather counts of how many IDs each rank has
    local_count = Int32(length(local_list))
    all_counts = comm_allgather(comm, local_count)

    # Allgatherv to collect all IDs from all ranks
    total_count = sum(all_counts)
    if total_count == 0
        return  # Nothing to finalize
    end

    all_ids = Vector{Int}(undef, total_count)
    comm_allgatherv!(comm, local_list, MPI.VBuffer(all_ids, all_counts))

    # Sort and unique to get deterministic order across all ranks
    dead_list = sort!(unique(all_ids))

    # Finalize each in order (check registry to avoid double-finalize)
    for id in dead_list
        if haskey(_mumps_registry, id)
            F = _mumps_registry[id]
            delete!(_mumps_registry, id)
            # Only finalize if we own the MUMPS object (not shared with cache)
            if F.owns_mumps
                F.mumps._finalized = false
                MUMPS.finalize!(F.mumps)
            end
        end
    end
end

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
# Create MUMPS Factorization
# ============================================================================

"""
    _get_or_create_analysis_plan(A::HPCSparseMatrix{T,Ti,B}, symmetric::Bool) where {T,Ti,B}

Get a cached analysis plan or create a new one. Returns the plan with
values updated from matrix A.
"""
function _get_or_create_analysis_plan(A::HPCSparseMatrix{T,Ti,B}, symmetric::Bool) where {T,Ti,B}
    # Note: MUMPS uses its own MPI communicator, but we use A's backend for consistency

    # Ensure structural hash is computed
    structural_hash = _ensure_hash(A)
    cache_key = (structural_hash, symmetric, T)

    if haskey(_mumps_analysis_cache, cache_key)
        # Cache hit: reuse existing analysis
        plan = _mumps_analysis_cache[cache_key]::MUMPSAnalysisPlan{T}

        # Update values from matrix A (structure is already correct)
        _update_values!(plan, A, symmetric)

        return plan, true  # true = cache hit
    end

    # Cache miss: create new analysis
    m, n = size(A)
    @assert m == n "Matrix must be square for factorization"

    # Extract local COO entries
    irn_loc, jcn_loc, a_loc, nzval_perm = extract_local_coo(A; symmetric=symmetric)
    nz_loc = length(a_loc)

    # Create MUMPS instance
    # sym=0: unsymmetric, sym=1: SPD, sym=2: general symmetric
    mumps_sym = symmetric ? MUMPS.mumps_definite : MUMPS.mumps_unsymmetric
    mumps = Mumps{T}(mumps_sym, MUMPS.default_icntl, MUMPS.default_cntl64)
    mumps._finalized = true  # Disable MUMPS GC finalizer to avoid post-MPI crash

    # Suppress all MUMPS output
    suppress_printing!(mumps)

    # Configure MUMPS for distributed input (displaylevel=0 suppresses verbose messages)
    set_icntl!(mumps, 5, 0; displaylevel=0)    # Assembled matrix format
    set_icntl!(mumps, 14, 50; displaylevel=0)  # Memory relaxation (50%)
    set_icntl!(mumps, 18, 3; displaylevel=0)   # Distributed matrix input
    set_icntl!(mumps, 20, 0; displaylevel=0)   # Dense RHS
    set_icntl!(mumps, 21, 0; displaylevel=0)   # Centralized solution on host
    set_icntl!(mumps, 7, 5; displaylevel=0)    # METIS ordering (better fill-in)

    # Enable OpenMP threading in MUMPS
    # ICNTL(16) = number of OpenMP threads (0 = use OMP_NUM_THREADS)
    omp_threads = parse(Int, get(ENV, "OMP_NUM_THREADS", "1"))
    set_icntl!(mumps, 16, omp_threads; displaylevel=0)

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

    # Create and cache the analysis plan
    plan = MUMPSAnalysisPlan{T}(
        mumps, irn_loc, jcn_loc, a_loc, nzval_perm,
        n, symmetric, copy(A.row_partition), structural_hash
    )
    _mumps_analysis_cache[cache_key] = plan

    return plan, false  # false = cache miss
end

"""
    _update_values!(plan::MUMPSAnalysisPlan{T}, A::HPCSparseMatrix{T}, symmetric::Bool) where T

Update the values in a cached analysis plan from a new matrix with the same structure.
Uses cached nzval_perm for fast vectorized copy.
"""
function _update_values!(plan::MUMPSAnalysisPlan{T}, A::HPCSparseMatrix{T}, symmetric::Bool) where T
    # Fast vectorized copy using cached permutation
    nzval_cpu = _ensure_cpu(A.nzval)
    plan.a_loc .= nzval_cpu[plan.nzval_perm]
end

"""
    _convert_to_mumps_compatible(A::HPCSparseMatrix{T,Ti,B}) where {T,Ti,B}

Convert a HPCSparseMatrix to CPU with MUMPS-compatible element type.
Returns the converted matrix with element type Float64 or ComplexF64.
The backend's comm is preserved (creates a CPU backend with same comm).
"""
function _convert_to_mumps_compatible(A::HPCSparseMatrix{T,Ti,B}) where {T,Ti,B}
    Tinternal = _mumps_internal_type(T)

    # If already compatible (CPU + correct type), just return as-is
    if T === Tinternal && A.backend.device isa DeviceCPU
        return A  # Already CPU and correct type
    end

    # Convert nzval to CPU and/or convert element type
    nzval_cpu = _ensure_cpu(A.nzval)
    nzval_converted = Tinternal.(nzval_cpu)

    # Create CPU backend with same comm (MUMPS is CPU-only)
    cpu_backend = HPCBackend(DeviceCPU(), A.backend.comm, A.backend.solver)

    # Create new HPCSparseMatrix with converted type (CPU Vector)
    # Structural arrays (rowptr, colval, col_indices) stay the same
    new_rowptr = copy(A.rowptr)
    new_colval = copy(A.colval)
    # For CPU, rowptr_target and colval_target are the same as rowptr and colval
    return HPCSparseMatrix{Tinternal,Ti,typeof(cpu_backend)}(
        A.structural_hash,
        copy(A.row_partition),
        copy(A.col_partition),
        copy(A.col_indices),
        new_rowptr,
        new_colval,
        nzval_converted,
        A.nrows_local,
        A.ncols_compressed,
        nothing,  # Don't copy cached transpose
        A.cached_symmetric,
        new_rowptr,  # rowptr_target (same as rowptr for CPU)
        new_colval,  # colval_target (same as colval for CPU)
        cpu_backend
    )
end

"""
    _create_mumps_factorization(A::HPCSparseMatrix{T, Ti, B}, symmetric::Bool) where {T, Ti, B}

Create and compute a MUMPS factorization of the distributed matrix A.
Uses cached symbolic analysis when available for the same sparsity structure.

Automatically converts the matrix to CPU Float64/ComplexF64 internally if needed,
storing the original backend for later reconstruction during solve.
"""
function _create_mumps_factorization(A::HPCSparseMatrix{T, Ti, B}, symmetric::Bool) where {T, Ti, B}
    comm = A.backend.comm
    rank = comm_rank(comm)

    # Process any pending finalizations first (collective operation)
    _process_finalizers(comm)

    # Assign unique ID for this factorization
    id = _mumps_count[]
    _mumps_count[] += 1

    # Convert to MUMPS-compatible type (CPU Float64 or ComplexF64)
    Tinternal = _mumps_internal_type(T)
    A_internal = _convert_to_mumps_compatible(A)

    # Get or create analysis plan (may be cached)
    plan, cache_hit = _get_or_create_analysis_plan(A_internal, symmetric)

    # Update value pointer (values may have been updated)
    plan.mumps.a_loc = pointer(plan.a_loc)

    # Factorization phase (job = 2)
    plan.mumps.job = MUMPS_INT(2)
    invoke_mumps_unsafe!(plan.mumps)
    _check_mumps_error(plan.mumps, "factorization")

    # Pre-allocate RHS buffer on rank 0
    rhs_buffer = rank == 0 ? zeros(Tinternal, plan.n) : Tinternal[]

    # Create factorization object with ID
    # Store original backend for comm access during solve
    # Note: We copy the value array since the plan's array is reused.
    # The MUMPS object is shared with the cache (owns_mumps=false).
    F = MUMPSFactorization{T, B, Tinternal}(
        id, plan.mumps, plan.irn_loc, plan.jcn_loc, copy(plan.a_loc),
        plan.n, symmetric, copy(plan.row_partition), copy(A.col_partition),
        rhs_buffer, false, A.backend
    )

    # Register in global registry (prevents GC until removed)
    _mumps_registry[id] = F

    # Attach Julia finalizer to queue for synchronized destruction
    finalizer(_queue_for_destruction, F)

    return F
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
    LinearAlgebra.lu(A::HPCSparseMatrix{T,Ti,B}) where {T,Ti,B}

Compute LU factorization of a distributed sparse matrix using MUMPS.
Returns a `MUMPSFactorization` for use with `\\` or `solve`.
"""
function LinearAlgebra.lu(A::HPCSparseMatrix{T,Ti,B}) where {T,Ti,B}
    return _create_mumps_factorization(A, false)
end

"""
    LinearAlgebra.ldlt(A::HPCSparseMatrix{T,Ti,B}) where {T,Ti,B}

Compute LDLT factorization of a distributed symmetric sparse matrix using MUMPS.
The matrix must be symmetric; only the lower triangular part is used.
Returns a `MUMPSFactorization` for use with `\\` or `solve`.
"""
function LinearAlgebra.ldlt(A::HPCSparseMatrix{T,Ti,B}) where {T,Ti,B}
    return _create_mumps_factorization(A, true)
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
# MUMPS requires GPU→CPU→GPU cycling. Do NOT use this for general operations.
# The base module only defines the identity case (CPU device).
# Extensions define CPU→GPU conversions (e.g., Vector → MtlVector).
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
function _copy_to_vector!(x::HPCVector{T,B}, values::Vector) where {T,B}
    if x.backend.device isa DeviceCPU
        x.v .= values
    else
        # GPU array - need to copy through appropriate method
        copyto!(x.v, _convert_to_device_array(values, x.backend.device))
    end
    return x
end

# Convert a CPU vector to a target device
function _convert_to_device_array(v::Vector{T}, device::AbstractDevice) where T
    if device isa DeviceCPU
        return v
    else
        # For GPU devices, use extension-defined function
        return _array_to_device(v, device)
    end
end

# Fallback for CPU device
_array_to_device(v::Vector{T}, ::DeviceCPU) where T = v

"""
    Base.:\\(F::MUMPSFactorization, b::HPCVector)

Solve A*x = b using the backslash operator.
"""
function Base.:\(F::MUMPSFactorization, b::HPCVector)
    return solve(F, b)
end

# ============================================================================
# Cleanup Interface
# ============================================================================

"""
    finalize!(F::MUMPSFactorization)

Release MUMPS resources. Must be called on all ranks together.

Note: If the MUMPS object is shared with the analysis cache (owns_mumps=false),
this only removes the factorization from the registry. The MUMPS object itself
is finalized when `clear_mumps_analysis_cache!()` is called.
"""
function finalize!(F::MUMPSFactorization)
    # Check if already finalized (removed from registry)
    if !haskey(_mumps_registry, F.id)
        return F  # Already finalized, no-op
    end

    # Remove from registry
    delete!(_mumps_registry, F.id)

    # Only finalize the MUMPS object if we own it (not shared with cache)
    if F.owns_mumps
        F.mumps._finalized = false  # Re-enable MUMPS finalization
        MUMPS.finalize!(F.mumps)
    end

    return F
end

