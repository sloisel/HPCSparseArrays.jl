# HPCMatrix type and dense matrix operations

"""
    compute_dense_structural_hash(row_partition, col_partition, local_size, comm::AbstractComm) -> Blake3Hash

Compute a structural hash for a dense matrix that is identical across all ranks.

1. Hash local data: row_partition, col_partition, local matrix size
2. Allgather all local hashes
3. Hash the gathered hashes to produce a global hash
"""
function compute_dense_structural_hash(row_partition::Vector{Int}, col_partition::Vector{Int},
    local_size::Tuple{Int,Int}, comm::AbstractComm)::Blake3Hash
    # Step 1: Compute rank-local hash
    # IMPORTANT: Prefix each vector with its length to disambiguate boundaries.
    # Without length prefixes, different structures could hash to the same value
    # if the concatenated bytes happen to match.
    ctx = Blake3Ctx()
    update!(ctx, reinterpret(UInt8, Int[length(row_partition)]))
    update!(ctx, reinterpret(UInt8, row_partition))
    update!(ctx, reinterpret(UInt8, Int[length(col_partition)]))
    update!(ctx, reinterpret(UInt8, col_partition))
    update!(ctx, reinterpret(UInt8, Int[2]))  # local_size is always 2 elements
    update!(ctx, reinterpret(UInt8, collect(local_size)))
    local_hash = digest(ctx)

    # Step 2: Allgather all local hashes
    all_hashes = comm_allgather(comm, local_hash)

    # Step 3: Hash them together to produce global hash
    ctx2 = Blake3Ctx()
    update!(ctx2, all_hashes)
    return Blake3Hash(digest(ctx2))
end

"""
    HPCMatrix{T, B<:HPCBackend{T}}

A distributed dense matrix partitioned by rows across MPI ranks.

# Type Parameters
- `T`: Element type (e.g., `Float64`, `ComplexF64`)
- `B<:HPCBackend{T}`: Backend configuration with matching element type

# Fields
- `structural_hash::Blake3Hash`: 256-bit Blake3 hash of the structural pattern
- `row_partition::Vector{Int}`: Row partition boundaries, length = nranks + 1 (always on CPU)
- `col_partition::Vector{Int}`: Column partition boundaries, length = nranks + 1 (always on CPU)
- `A::AbstractMatrix{T}`: Local rows (NOT transposed), size = (local_nrows, ncols)
- `backend::B`: The HPC backend configuration

# Invariants
- `row_partition` and `col_partition` are sorted
- `row_partition[nranks+1]` = total number of rows + 1
- `col_partition[nranks+1]` = total number of columns + 1
- `size(A, 1) == row_partition[rank+2] - row_partition[rank+1]`
- `size(A, 2) == col_partition[end] - 1`
"""
mutable struct HPCMatrix{T, B<:HPCBackend{T}} <: AbstractMatrix{T}
    structural_hash::Blake3Hash
    row_partition::Vector{Int}
    col_partition::Vector{Int}
    A::AbstractMatrix{T}
    backend::B
    # Inner constructor that takes all arguments
    function HPCMatrix{T,B}(hash::Blake3Hash, row_partition::Vector{Int}, col_partition::Vector{Int}, A::AbstractMatrix{T}, backend::B) where {T, B<:HPCBackend{T}}
        new{T,B}(hash, row_partition, col_partition, A, backend)
    end
end

# Type aliases for common backend configurations (using Int as default index type)
const HPCMatrix_CPU{T} = HPCMatrix{T, HPCBackend_CPU_MPI{T,Int}}
const HPCMatrix_CPU_Serial{T} = HPCMatrix{T, HPCBackend_CPU_Serial{T,Int}}

# Convenience constructor that infers B from the backend type
function HPCMatrix{T}(hash::Blake3Hash, row_partition::Vector{Int}, col_partition::Vector{Int}, A::AbstractMatrix{T}, backend::B) where {T, B<:HPCBackend{T}}
    HPCMatrix{T,B}(hash, row_partition, col_partition, A, backend)
end

# Constructor that infers T from the matrix
function HPCMatrix(hash::Blake3Hash, row_partition::Vector{Int}, col_partition::Vector{Int}, A::AbstractMatrix{T}, backend::B) where {T, B<:HPCBackend{T}}
    HPCMatrix{T,B}(hash, row_partition, col_partition, A, backend)
end

# Get the HPCBackend for a HPCMatrix
get_hpc_backend(M::HPCMatrix) = M.backend

# Adapt.jl support for converting HPCMatrix between KernelAbstractions backends (GPU/CPU)
# Note: This adapts the array storage but preserves the HPCBackend
function Adapt.adapt_structure(to, M::HPCMatrix{T,B}) where {T,B}
    new_A = adapt(to, M.A)
    return HPCMatrix{T,B}(M.structural_hash, M.row_partition, M.col_partition, new_A, M.backend)
end

"""
    HPCMatrix_local(A_local::AbstractMatrix{T}, backend::HPCBackend; col_partition=...) where T

Create a HPCMatrix from a local matrix on each rank.

Unlike `HPCMatrix(M_global, backend)` which takes a global matrix and partitions it,
this constructor takes only the local rows of the matrix that each rank owns.
The row partition is computed by gathering the local row counts from all ranks.

The type of `A_local` determines the storage backend (CPU Matrix, GPU MtlMatrix, etc.).

All ranks must have local matrices with the same number of columns.
A collective error is raised if the column counts don't match.

# Arguments
- `A_local::AbstractMatrix{T}`: Local rows owned by this rank
- `backend::HPCBackend`: The HPC backend configuration

# Keyword Arguments
- `col_partition::Vector{Int}`: Column partition boundaries (default: `uniform_partition(size(A_local,2), nranks)`)

# Example
```julia
backend = backend_cpu_mpi(MPI.COMM_WORLD)
# Rank 0 has 2×3 matrix, Rank 1 has 3×3 matrix
A = HPCMatrix_local(randn(2, 3), backend)  # on rank 0
A = HPCMatrix_local(randn(3, 3), backend)  # on rank 1
# Result: 5×3 distributed matrix with row_partition [1, 3, 6]
```
"""
function HPCMatrix_local(A_local::AbstractMatrix{T}, backend::B;
                         col_partition::Vector{Int}=uniform_partition(size(A_local, 2), comm_size(backend.comm))) where {T, B<:HPCBackend}
    nranks = comm_size(backend.comm)

    local_nrows, local_ncols = size(A_local)

    # Gather local row counts and column counts from all ranks
    local_info = Int32[local_nrows, local_ncols]
    all_info = comm_allgather(backend.comm, local_info)

    # Extract row counts and verify column counts match
    all_row_counts = [all_info[2*(r-1)+1] for r in 1:nranks]
    all_col_counts = [all_info[2*(r-1)+2] for r in 1:nranks]

    # Check that all column counts are the same
    if !all(c == all_col_counts[1] for c in all_col_counts)
        error("HPCMatrix_local: All ranks must have the same number of columns. " *
              "Got column counts: $all_col_counts")
    end

    # Build row_partition from row counts (inferred via Allgather)
    row_partition = Vector{Int}(undef, nranks + 1)
    row_partition[1] = 1
    for r in 1:nranks
        row_partition[r+1] = row_partition[r] + all_row_counts[r]
    end

    # Convert to target device (no-op for CPU, copies to GPU for GPU backends)
    A_target = _convert_array(A_local isa Matrix ? A_local : Matrix(A_local), backend.device)

    # Compute structural hash eagerly
    structural_hash = compute_dense_structural_hash(row_partition, col_partition, size(A_target), backend.comm)
    return HPCMatrix{T,B}(structural_hash, row_partition, col_partition, A_target, backend)
end

"""
    HPCMatrix(M::Matrix{T}, backend::HPCBackend; row_partition=..., col_partition=...) where T

Create a HPCMatrix from a global matrix M, partitioning it by rows across MPI ranks.

Each rank extracts only its local rows from `M`, so:

- **Simple usage**: Pass identical `M` to all ranks
- **Efficient usage**: Pass a matrix with correct `size(M)` on all ranks,
  but only populate the rows that each rank owns (other rows are ignored)

# Arguments
- `M::Matrix{T}`: The global matrix
- `backend::HPCBackend`: The HPC backend configuration

# Keyword Arguments
- `row_partition::Vector{Int}`: Row partition boundaries (default: `uniform_partition(size(M,1), nranks)`)
- `col_partition::Vector{Int}`: Column partition boundaries (default: `uniform_partition(size(M,2), nranks)`)

Use `uniform_partition(n, nranks)` to compute custom partitions.

# Example
```julia
backend = backend_cpu_mpi(MPI.COMM_WORLD)
A = HPCMatrix(randn(100, 50), backend)
```
"""
function HPCMatrix(M::Matrix{T}, backend::B;
                   row_partition::Vector{Int}=uniform_partition(size(M, 1), comm_size(backend.comm)),
                   col_partition::Vector{Int}=uniform_partition(size(M, 2), comm_size(backend.comm))) where {T, B<:HPCBackend}
    rank = comm_rank(backend.comm)

    # Local row range (1-indexed, Julia style)
    row_start = row_partition[rank+1]
    row_end = row_partition[rank+2] - 1

    # Extract local rows from M (NOT transposed, unlike HPCSparseMatrix)
    A_cpu = M[row_start:row_end, :]
    # Convert to target device (no-op for CPU, copies to GPU for GPU backends)
    A = _convert_array(A_cpu, backend.device)

    # Compute structural hash eagerly
    structural_hash = compute_dense_structural_hash(row_partition, col_partition, size(A), backend.comm)
    return HPCMatrix{T,B}(structural_hash, row_partition, col_partition, A, backend)
end

# ============================================================================
# Helpers for Distributed Operations (cat, etc.)
# ============================================================================

"""
    _gather_dense_rows(A::HPCMatrix{T,B}, global_rows::AbstractVector{Int}) where {T,B}

Gather specific rows from a HPCMatrix by global row index.

Returns a Matrix{T} with the requested rows, where result[i,:] corresponds to
global row global_rows[i]. Uses point-to-point communication (Isend/Irecv)
to minimize data transfer - only exchanges rows that are actually needed.

# Arguments
- `A::HPCMatrix{T,B}`: The distributed dense matrix
- `global_rows::AbstractVector{Int}`: Global row indices to gather

# Returns
Matrix{T} of size (length(global_rows), ncols) with the requested rows.

Communication tags used: 34 (row indices), 35 (row values)
"""
function _gather_dense_rows(A::HPCMatrix{T,B}, global_rows::AbstractVector{Int}) where {T,B}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    n_requested = length(global_rows)
    ncols = size(A, 2)

    # NOTE: Do NOT early return here even if n_requested == 0!
    # All ranks must participate in collectives below.

    # Ensure matrix data is on CPU for row extraction and MPI (GPU arrays not supported)
    A_cpu = _ensure_cpu(A.A)

    my_row_start = A.row_partition[rank+1]

    # Build position mapping for received rows
    position_map = Dict{Int,Int}()  # global_row => position in result
    for (pos, global_row) in enumerate(global_rows)
        position_map[global_row] = pos
    end

    result = Matrix{T}(undef, n_requested, ncols)

    # Step 1: Partition rows by owner
    rows_by_owner = _partition_rows_by_owner(global_rows, A.row_partition)

    # Step 2: Exchange row request counts via Alltoall
    send_counts = Int32[haskey(rows_by_owner, r) ? length(rows_by_owner[r]) : 0 for r in 0:(nranks-1)]
    recv_counts = comm_alltoall(comm, MPI.UBuffer(send_counts, 1))

    # Step 3: Send row requests to owner ranks
    row_send_reqs = []
    row_recv_bufs = Dict{Int,Vector{Int32}}()
    row_recv_reqs = []

    for r in 0:(nranks-1)
        if send_counts[r+1] > 0 && r != rank
            rows_to_request = Int32.(rows_by_owner[r])
            req = comm_isend(comm, rows_to_request, r, 34)
            push!(row_send_reqs, req)
        end
    end

    # Step 4: Receive row requests from other ranks
    for r in 0:(nranks-1)
        if recv_counts[r+1] > 0 && r != rank
            buf = Vector{Int32}(undef, recv_counts[r+1])
            req = comm_irecv!(comm, buf, r, 34)
            push!(row_recv_reqs, req)
            row_recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, row_recv_reqs)
    comm_waitall(comm, row_send_reqs)

    # Step 5: Send row data for received requests
    row_data_send_bufs = Dict{Int,Matrix{T}}()
    row_data_send_reqs = []

    for r in 0:(nranks-1)
        if recv_counts[r+1] > 0 && r != rank
            requested_rows = row_recv_bufs[r]
            n_rows_to_send = length(requested_rows)
            data_to_send = Matrix{T}(undef, n_rows_to_send, ncols)

            for (i, global_row) in enumerate(requested_rows)
                local_row = global_row - my_row_start + 1
                data_to_send[i, :] = A_cpu[local_row, :]
            end

            row_data_send_bufs[r] = data_to_send
            req = comm_isend(comm, vec(data_to_send), r, 35)
            push!(row_data_send_reqs, req)
        end
    end

    # Step 6: Receive row data
    row_data_recv_bufs = Dict{Int,Vector{T}}()
    row_data_recv_reqs = []
    recv_positions = Dict{Int,Vector{Int}}()  # Track where each rank's data goes

    for r in 0:(nranks-1)
        if send_counts[r+1] > 0 && r != rank
            n_rows = send_counts[r+1]
            buf = Vector{T}(undef, n_rows * ncols)
            req = comm_irecv!(comm, buf, r, 35)
            push!(row_data_recv_reqs, req)
            row_data_recv_bufs[r] = buf
            # Track positions for this rank's rows
            recv_positions[r] = [position_map[row] for row in rows_by_owner[r]]
        end
    end

    # Step 7: Copy local rows directly
    if haskey(rows_by_owner, rank)
        for global_row in rows_by_owner[rank]
            local_row = global_row - my_row_start + 1
            result_pos = position_map[global_row]
            result[result_pos, :] = A_cpu[local_row, :]
        end
    end

    comm_waitall(comm, row_data_recv_reqs)

    # Step 8: Unpack received row data
    for r in 0:(nranks-1)
        if send_counts[r+1] > 0 && r != rank
            buf = row_data_recv_bufs[r]
            positions = recv_positions[r]
            n_rows = length(positions)
            data_matrix = reshape(buf, n_rows, ncols)
            for (i, pos) in enumerate(positions)
                result[pos, :] = data_matrix[i, :]
            end
        end
    end

    comm_waitall(comm, row_data_send_reqs)

    return result
end

"""
    _partition_rows_by_owner(global_rows::AbstractVector{Int}, partition::Vector{Int}) -> Dict{Int, Vector{Int}}

Partition a list of global row indices by their owner rank.

Returns a Dict mapping owner rank (0-indexed) to the list of global row indices
owned by that rank, according to the given partition.
"""
function _partition_rows_by_owner(global_rows::AbstractVector{Int}, partition::Vector{Int})
    nranks = length(partition) - 1
    result = Dict{Int, Vector{Int}}()

    for global_row in global_rows
        owner = searchsortedlast(partition, global_row) - 1
        owner = clamp(owner, 0, nranks - 1)
        if !haskey(result, owner)
            result[owner] = Int[]
        end
        push!(result[owner], global_row)
    end

    return result
end

# Dense matrix-vector plan and multiplication

"""
    DenseMatrixVectorPlan{T}

A communication plan for gathering vector elements needed for HPCMatrix * x.

For a dense matrix A with ncols columns, we need all elements x[1:ncols].
The plan gathers these elements from the distributed vector x based on x's partition.

# Fields
- `send_rank_ids::Vector{Int}`: Ranks we send elements to (0-indexed)
- `send_indices::Vector{Vector{Int}}`: For each rank, local indices to send
- `send_bufs::Vector{Vector{T}}`: Pre-allocated send buffers
- `send_reqs::Vector{Any}`: Pre-allocated send request handles
- `recv_rank_ids::Vector{Int}`: Ranks we receive elements from (0-indexed)
- `recv_bufs::Vector{Vector{T}}`: Pre-allocated receive buffers
- `recv_reqs::Vector{Any}`: Pre-allocated receive request handles
- `recv_perm::Vector{Vector{Int}}`: For each recv rank, indices into gathered
- `local_src_indices::Vector{Int}`: Source indices for local copy (into x.v)
- `local_dst_indices::Vector{Int}`: Destination indices for local copy (into gathered)
- `gathered::AV`: Pre-allocated buffer for gathered elements (same type as input)
- `gathered_cpu::Vector{T}`: CPU staging buffer (used when AV is GPU array)
"""
mutable struct DenseMatrixVectorPlan{T,AV<:AbstractVector{T}}
    send_rank_ids::Vector{Int}
    send_indices::Vector{Vector{Int}}
    send_bufs::Vector{Vector{T}}      # Always CPU for MPI
    send_reqs::Vector{Any}
    recv_rank_ids::Vector{Int}
    recv_bufs::Vector{Vector{T}}      # Always CPU for MPI
    recv_reqs::Vector{Any}
    recv_perm::Vector{Vector{Int}}
    local_src_indices::Vector{Int}
    local_dst_indices::Vector{Int}
    gathered::AV                       # Same type as input vector
    gathered_cpu::Vector{T}            # CPU staging buffer
    # Partition for result vector (computed at plan creation)
    result_partition_hash::Blake3Hash
    result_partition::Vector{Int}
end

# Type alias for CPU-backed DenseMatrixVectorPlan (backwards compatible)
const DenseMatrixVectorPlan_CPU{T} = DenseMatrixVectorPlan{T,Vector{T}}

"""
    DenseMatrixVectorPlan(A::HPCMatrix{T,B}, x::HPCVector{T,B}) where {T,B<:HPCBackend}

Create a communication plan to gather all x elements (x[1:ncols]) for dense matrix-vector multiplication.
The gathered buffer will have the same storage type as the input vector x.
"""
function DenseMatrixVectorPlan(A::HPCMatrix{T,B}, x::HPCVector{T,B}) where {T,B<:HPCBackend}
    assert_backends_compatible(A.backend, x.backend)
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    # For dense matrix, we need all elements 1:ncols
    ncols = A.col_partition[end] - 1
    col_indices = collect(1:ncols)
    n_gathered = ncols

    my_x_start = x.partition[rank+1]

    # Step 1: Group col_indices by owner rank in x's partition
    needed_from = [Tuple{Int,Int}[] for _ in 1:nranks]
    for (dst_idx, global_idx) in enumerate(col_indices)
        owner = searchsortedlast(x.partition, global_idx) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(needed_from[owner+1], (global_idx, dst_idx))
    end

    # Step 2: Exchange counts via Alltoall
    send_counts = [length(needed_from[r+1]) for r in 0:(nranks-1)]
    recv_counts = comm_alltoall(comm, MPI.UBuffer(send_counts, 1))

    # Step 3: Send requested indices to each owner rank
    struct_send_bufs = Dict{Int,Vector{Int}}()
    struct_send_reqs = []
    recv_rank_ids = Int[]
    recv_perm_map = Dict{Int,Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r+1] > 0 && r != rank
            push!(recv_rank_ids, r)
            indices = [t[1] for t in needed_from[r+1]]
            dst_indices = [t[2] for t in needed_from[r+1]]
            recv_perm_map[r] = dst_indices
            struct_send_bufs[r] = indices
            req = comm_isend(comm, indices, r, 30)
            push!(struct_send_reqs, req)
        end
    end

    # Step 4: Receive requests from other ranks
    send_rank_ids = Int[]
    struct_recv_bufs = Dict{Int,Vector{Int}}()
    struct_recv_reqs = []

    for r in 0:(nranks-1)
        if recv_counts[r+1] > 0 && r != rank
            push!(send_rank_ids, r)
            buf = Vector{Int}(undef, recv_counts[r+1])
            req = comm_irecv!(comm, buf, r, 30)
            push!(struct_recv_reqs, req)
            struct_recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, struct_recv_reqs)
    comm_waitall(comm, struct_send_reqs)

    # Step 5: Convert received global indices to local indices for sending
    send_indices_map = Dict{Int,Vector{Int}}()
    for r in send_rank_ids
        global_indices = struct_recv_bufs[r]
        local_indices = [idx - my_x_start + 1 for idx in global_indices]
        send_indices_map[r] = local_indices
    end

    # Step 6: Handle local elements
    local_src_indices = Int[]
    local_dst_indices = Int[]
    for (global_idx, dst_idx) in needed_from[rank+1]
        local_idx = global_idx - my_x_start + 1
        push!(local_src_indices, local_idx)
        push!(local_dst_indices, dst_idx)
    end

    # Step 7: Build final arrays and buffers
    sort!(send_rank_ids)
    sort!(recv_rank_ids)

    send_indices_final = [send_indices_map[r] for r in send_rank_ids]
    recv_perm_final = [recv_perm_map[r] for r in recv_rank_ids]

    # MPI buffers are always on CPU
    send_bufs = [Vector{T}(undef, length(inds)) for inds in send_indices_final]
    recv_bufs = [Vector{T}(undef, send_counts[r+1]) for r in recv_rank_ids]
    send_reqs = Vector{Any}(undef, length(send_rank_ids))
    recv_reqs = Vector{Any}(undef, length(recv_rank_ids))

    # CPU staging buffer (always needed for MPI)
    gathered_cpu = Vector{T}(undef, n_gathered)

    # Gathered buffer matches source vector's storage type
    gathered = similar(x.v, n_gathered)

    # Compute result partition hash eagerly
    result_partition = copy(A.row_partition)
    result_partition_hash = compute_partition_hash(result_partition)

    return DenseMatrixVectorPlan{T,typeof(gathered)}(
        send_rank_ids, send_indices_final, send_bufs, send_reqs,
        recv_rank_ids, recv_bufs, recv_reqs, recv_perm_final,
        local_src_indices, local_dst_indices, gathered, gathered_cpu,
        result_partition_hash, result_partition
    )
end

"""
    execute_plan!(plan::DenseMatrixVectorPlan{T,AV}, x::HPCVector{T,B}) where {T,AV,B}

Execute a dense vector communication plan to gather elements from x.
Returns plan.gathered containing x[1:ncols] for the associated matrix.

Works uniformly for CPU and GPU vectors - MPI communication always uses CPU
buffers, with automatic staging for GPU arrays.
"""
function execute_plan!(plan::DenseMatrixVectorPlan{T,AV}, x::HPCVector{T,B}) where {T,AV,B}
    comm = x.backend.comm

    # Get CPU data for MPI operations (no-op for CPU, copy for GPU)
    x_cpu = _ensure_cpu(x.v)

    # Step 1: Copy local values to CPU staging buffer
    @inbounds for i in eachindex(plan.local_src_indices, plan.local_dst_indices)
        plan.gathered_cpu[plan.local_dst_indices[i]] = x_cpu[plan.local_src_indices[i]]
    end

    # Step 2: Fill send buffers and send
    send_reqs = []
    @inbounds for i in eachindex(plan.send_rank_ids)
        r = plan.send_rank_ids[i]
        send_idx = plan.send_indices[i]
        buf = plan.send_bufs[i]
        for k in eachindex(send_idx)
            buf[k] = x_cpu[send_idx[k]]
        end
        req = comm_isend(comm, buf, r, 31)
        push!(send_reqs, req)
    end

    # Step 3: Receive values to CPU buffers
    recv_reqs = []
    @inbounds for i in eachindex(plan.recv_rank_ids)
        req = comm_irecv!(comm, plan.recv_bufs[i], plan.recv_rank_ids[i], 31)
        push!(recv_reqs, req)
    end

    comm_waitall(comm, recv_reqs)

    # Step 4: Scatter received values into CPU staging buffer
    @inbounds for i in eachindex(plan.recv_rank_ids)
        perm = plan.recv_perm[i]
        buf = plan.recv_bufs[i]
        for k in eachindex(perm)
            plan.gathered_cpu[perm[k]] = buf[k]
        end
    end

    comm_waitall(comm, send_reqs)

    # Step 5: Copy to output buffer (no-op if CPU, copy if GPU)
    _copy_to_output!(plan.gathered, plan.gathered_cpu)

    return plan.gathered
end

"""
    get_dense_vector_plan(A::HPCMatrix{T,B}, x::HPCVector{T,B}) where {T,B<:HPCBackend}

Get a memoized DenseMatrixVectorPlan for A * x.
The plan is cached based on the structural hashes of A and x, plus the backend type.
"""
function get_dense_vector_plan(A::HPCMatrix{T,B}, x::HPCVector{T,B}) where {T,B<:HPCBackend}
    key = (A.structural_hash, x.structural_hash, T, B)
    if haskey(_dense_vector_plan_cache, key)
        return _dense_vector_plan_cache[key]::DenseMatrixVectorPlan{T}
    end
    plan = DenseMatrixVectorPlan(A, x)
    _dense_vector_plan_cache[key] = plan
    return plan
end


"""
    LinearAlgebra.mul!(y::HPCVector{T,B}, A::HPCMatrix{T,B}, x::HPCVector{T,B}) where {T,B<:HPCBackend}

In-place dense matrix-vector multiplication: y = A * x.

Requires matching backends.
"""
function LinearAlgebra.mul!(y::HPCVector{T,B}, A::HPCMatrix{T,B}, x::HPCVector{T,B}) where {T,B<:HPCBackend}
    assert_backends_compatible(A.backend, x.backend)
    assert_backends_compatible(A.backend, y.backend)

    plan = get_dense_vector_plan(A, x)
    execute_plan!(plan, x)

    # Unified CPU/GPU path: plan.gathered has correct type after execute_plan!
    LinearAlgebra.mul!(y.v, A.A, plan.gathered)
    return y
end

"""
    Base.:*(A::HPCMatrix{T,B}, x::HPCVector{T,B}) where {T,B<:HPCBackend}

Dense matrix-vector multiplication returning a new HPCVector.
The result has the same row partition as A and inherits the backend.
"""
function Base.:*(A::HPCMatrix{T,B}, x::HPCVector{T,B}) where {T,B<:HPCBackend}
    assert_backends_compatible(A.backend, x.backend)

    rank = comm_rank(A.backend.comm)
    local_rows = A.row_partition[rank + 2] - A.row_partition[rank + 1]

    # Get the plan
    plan = get_dense_vector_plan(A, x)

    # Execute the plan to gather vector elements
    execute_plan!(plan, x)

    # Unified CPU/GPU path: similar() preserves array type, plan.gathered has correct type
    y_v = similar(A.A, local_rows)
    LinearAlgebra.mul!(y_v, A.A, plan.gathered)

    return HPCVector{T,B}(
        plan.result_partition_hash,
        plan.result_partition,  # Partition is immutable, no need to copy
        y_v,
        A.backend
    )
end

# Note: _create_output_like is defined in vectors.jl (loaded first)

# Dense transpose plan

"""
    DenseTransposePlan{T}

A communication plan for computing the transpose of a HPCMatrix.

The transpose of A (with row_partition R and col_partition C) will have:
- row_partition = C (columns of A become rows of A^T)
- col_partition = R (rows of A become columns of A^T)

# Fields
- `rank_ids::Vector{Int}`: Ranks we send data to (0-indexed)
- `send_row_ranges::Vector{UnitRange{Int}}`: For each rank, local row range to send
- `send_col_ranges::Vector{UnitRange{Int}}`: For each rank, column range to send
- `send_bufs::Vector{Vector{T}}`: Pre-allocated send buffers
- `send_reqs::Vector{Any}`: Pre-allocated send request handles
- `recv_rank_ids::Vector{Int}`: Ranks we receive data from (0-indexed)
- `recv_row_ranges::Vector{UnitRange{Int}}`: For each recv rank, row range in result
- `recv_col_ranges::Vector{UnitRange{Int}}`: For each recv rank, column range in result
- `recv_bufs::Vector{Vector{T}}`: Pre-allocated receive buffers
- `recv_reqs::Vector{Any}`: Pre-allocated receive request handles
- `local_row_range::UnitRange{Int}`: Local rows that stay on this rank
- `local_col_range::UnitRange{Int}`: Local columns that stay on this rank
- `AT::Matrix{T}`: Pre-allocated result matrix
- `row_partition::Vector{Int}`: Row partition for the transposed matrix
- `col_partition::Vector{Int}`: Col partition for the transposed matrix
"""
mutable struct DenseTransposePlan{T}
    rank_ids::Vector{Int}
    send_row_ranges::Vector{UnitRange{Int}}
    send_col_ranges::Vector{UnitRange{Int}}
    send_bufs::Vector{Vector{T}}
    send_reqs::Vector{Any}
    recv_rank_ids::Vector{Int}
    recv_row_ranges::Vector{UnitRange{Int}}
    recv_col_ranges::Vector{UnitRange{Int}}
    recv_bufs::Vector{Vector{T}}
    recv_reqs::Vector{Any}
    local_row_range::UnitRange{Int}
    local_col_range::UnitRange{Int}
    AT::Matrix{T}
    row_partition::Vector{Int}
    col_partition::Vector{Int}
    # Structural hash for transpose result (computed at plan creation)
    structural_hash::Blake3Hash
end

"""
    DenseTransposePlan(A::HPCMatrix{T,B}) where {T,B}

Create a communication plan for computing A^T.

The algorithm:
1. A^T has row_partition = A.col_partition, col_partition = A.row_partition
2. For each destination rank r (owning rows in A^T = columns in A):
   - Determine which columns of our local A need to go to rank r
   - These become rows r's local A^T
3. Exchange data via point-to-point communication
"""
function DenseTransposePlan(A::HPCMatrix{T,B}) where {T,B}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    # The transpose has swapped partitions
    result_row_partition = A.col_partition
    result_col_partition = A.row_partition

    my_row_start = A.row_partition[rank+1]
    my_row_end = A.row_partition[rank+2] - 1
    my_local_nrows = my_row_end - my_row_start + 1
    ncols = A.col_partition[end] - 1

    # Result dimensions
    result_row_start = result_row_partition[rank+1]
    result_row_end = result_row_partition[rank+2] - 1
    result_local_nrows = result_row_end - result_row_start + 1
    result_ncols = result_col_partition[end] - 1

    # Pre-allocate result matrix
    AT = Matrix{T}(undef, result_local_nrows, result_ncols)

    # Determine send/receive patterns
    # We need to send A[my_rows, cols] where cols go to different destination ranks
    # In A^T, these become A^T[cols, my_rows] on the destination rank

    rank_ids = Int[]
    send_row_ranges = UnitRange{Int}[]
    send_col_ranges = UnitRange{Int}[]
    recv_rank_ids = Int[]
    recv_row_ranges = UnitRange{Int}[]
    recv_col_ranges = UnitRange{Int}[]

    # For each rank, determine which columns of A (= rows of A^T) they own
    for r in 0:(nranks-1)
        dest_row_start = result_row_partition[r+1]
        dest_row_end = result_row_partition[r+2] - 1

        if dest_row_end >= dest_row_start
            # These are columns dest_row_start:dest_row_end in A
            col_start = dest_row_start
            col_end = dest_row_end

            if r != rank
                # Send A[1:local_nrows, col_start:col_end] to rank r
                push!(rank_ids, r)
                push!(send_row_ranges, 1:my_local_nrows)
                # Convert global column indices to local
                local_col_start = col_start
                local_col_end = col_end
                push!(send_col_ranges, local_col_start:local_col_end)
            end
        end

        # Determine what we receive from rank r
        # Rank r owns rows src_row_start:src_row_end in A
        # These become columns src_row_start:src_row_end in A^T
        src_row_start = A.row_partition[r+1]
        src_row_end = A.row_partition[r+2] - 1

        if src_row_end >= src_row_start
            # We receive data for our local rows in A^T (= columns in A that we own)
            if r != rank
                push!(recv_rank_ids, r)
                # Our local rows in A^T
                push!(recv_row_ranges, 1:result_local_nrows)
                # The columns in A^T corresponding to rank r's rows in A
                push!(recv_col_ranges, src_row_start:src_row_end)
            end
        end
    end

    # Determine local copy range (data that stays on this rank)
    # This is the block where both row in A^T (= col in A) and col in A^T (= row in A)
    # are owned by this rank
    local_row_range = 1:0  # empty by default
    local_col_range = 1:0  # empty by default

    # My columns in A that I own in A^T (rows result_row_start:result_row_end)
    my_AT_row_start = result_row_start
    my_AT_row_end = result_row_end

    # My columns in A^T are my rows in A (my_row_start:my_row_end)
    # So local copy: A^T[1:result_local_nrows, my_row_start:my_row_end]
    #              = transpose of A[1:my_local_nrows, my_AT_row_start:my_AT_row_end]
    if my_AT_row_end >= my_AT_row_start && my_row_end >= my_row_start
        local_row_range = 1:result_local_nrows
        local_col_range = my_row_start:my_row_end
    end

    # Allocate send/recv buffers
    send_bufs = Vector{Vector{T}}(undef, length(rank_ids))
    for (i, r) in enumerate(rank_ids)
        nrows_to_send = length(send_row_ranges[i])
        ncols_to_send = length(send_col_ranges[i])
        send_bufs[i] = Vector{T}(undef, nrows_to_send * ncols_to_send)
    end

    recv_bufs = Vector{Vector{T}}(undef, length(recv_rank_ids))
    for (i, r) in enumerate(recv_rank_ids)
        nrows_to_recv = length(recv_row_ranges[i])
        ncols_to_recv = length(recv_col_ranges[i])
        recv_bufs[i] = Vector{T}(undef, nrows_to_recv * ncols_to_recv)
    end

    send_reqs = Vector{Any}(undef, length(rank_ids))
    recv_reqs = Vector{Any}(undef, length(recv_rank_ids))

    # Compute structural hash eagerly
    structural_hash = compute_dense_structural_hash(result_row_partition, result_col_partition, size(AT), comm)

    return DenseTransposePlan{T}(
        rank_ids, send_row_ranges, send_col_ranges, send_bufs, send_reqs,
        recv_rank_ids, recv_row_ranges, recv_col_ranges, recv_bufs, recv_reqs,
        local_row_range, local_col_range,
        AT, result_row_partition, result_col_partition,
        structural_hash
    )
end

"""
    execute_plan!(plan::DenseTransposePlan{T}, A::HPCMatrix{T,B}) where {T,B}

Execute a dense transpose plan to compute A^T.
Returns a HPCMatrix representing the transpose.
"""
function execute_plan!(plan::DenseTransposePlan{T}, A::HPCMatrix{T,B}) where {T,B}
    comm = A.backend.comm
    rank = comm_rank(comm)

    my_row_start = A.row_partition[rank+1]
    result_row_start = plan.row_partition[rank+1]

    # Stage GPU data to CPU for scalar indexing (MPI requires CPU buffers anyway)
    A_local = Array(A.A)

    # Step 1: Copy local values (transpose the block)
    # A^T[local_row_range, local_col_range] = transpose(A[1:nrows, result_row_start:result_row_end])
    if !isempty(plan.local_row_range) && !isempty(plan.local_col_range)
        local_A_cols = result_row_start:(result_row_start + length(plan.local_row_range) - 1)
        for (dst_col_idx, src_row) in enumerate(plan.local_col_range)
            local_src_row = src_row - my_row_start + 1
            for (dst_row_idx, src_col) in enumerate(local_A_cols)
                plan.AT[dst_row_idx, dst_col_idx + my_row_start - 1] = A_local[local_src_row, src_col]
            end
        end
    end

    # Step 2: Fill send buffers (pack transposed data) and send
    send_reqs = []
    for (i, r) in enumerate(plan.rank_ids)
        row_range = plan.send_row_ranges[i]
        col_range = plan.send_col_ranges[i]
        buf = plan.send_bufs[i]
        buf_idx = 1
        # Pack: for each local row (= column in A^T on destination), pack all columns
        # Order must match unpack: outer loop over source rows, inner loop over columns
        for local_row in row_range
            for c in col_range
                buf[buf_idx] = A_local[local_row, c]
                buf_idx += 1
            end
        end
        req = comm_isend(comm, buf, r, 40)
        push!(send_reqs, req)
    end

    # Step 3: Receive values
    recv_reqs = []
    for (i, r) in enumerate(plan.recv_rank_ids)
        req = comm_irecv!(comm, plan.recv_bufs[i], r, 40)
        push!(recv_reqs, req)
    end

    comm_waitall(comm, recv_reqs)

    # Step 4: Unpack received values into AT (transposing in the process)
    for (i, r) in enumerate(plan.recv_rank_ids)
        row_range = plan.recv_row_ranges[i]
        col_range = plan.recv_col_ranges[i]
        buf = plan.recv_bufs[i]
        buf_idx = 1
        # Data was packed in column-major order of A (row-major of A^T)
        # So unpack as: for each source row in A (= dest col in A^T), assign to dest rows
        for src_row in col_range
            dst_col = src_row
            for dst_row in row_range
                plan.AT[dst_row, dst_col] = buf[buf_idx]
                buf_idx += 1
            end
        end
    end

    comm_waitall(comm, send_reqs)

    # Create result with a copy of AT
    result_AT = copy(plan.AT)

    # Unified CPU/GPU path: _convert_array is no-op for CPU, copies for GPU
    result_A = _convert_array(result_AT, A.backend.device)
    return HPCMatrix{T,B}(plan.structural_hash, plan.row_partition, plan.col_partition, result_A, A.backend)
end

"""
    get_dense_transpose_plan(A::HPCMatrix{T,B}) where {T,B}

Get a memoized DenseTransposePlan for A^T.
The plan is cached based on the structural hash of A and the element type.
"""
function get_dense_transpose_plan(A::HPCMatrix{T,B}) where {T,B}
    key = (A.structural_hash, T, B)
    if haskey(_dense_transpose_plan_cache, key)
        return _dense_transpose_plan_cache[key]::DenseTransposePlan{T}
    end
    plan = DenseTransposePlan(A)
    _dense_transpose_plan_cache[key] = plan
    return plan
end

# Lazy transpose support for HPCMatrix

"""
    Base.transpose(A::HPCMatrix{T,B}) where {T,B}

Return a lazy transpose wrapper around A.
"""
Base.transpose(A::HPCMatrix{T,B}) where {T,B} = Transpose(A)

"""
    Base.conj(A::HPCMatrix{T,B}) where {T,B}

Return a new HPCMatrix with conjugated values.
"""
function Base.conj(A::HPCMatrix{T,B}) where {T,B}
    return HPCMatrix{T,B}(A.structural_hash, A.row_partition, A.col_partition, conj.(A.A), A.backend)
end

"""
    Base.adjoint(A::HPCMatrix{T,B}) where {T,B}

Return transpose(conj(A)), i.e., the conjugate transpose.
"""
Base.adjoint(A::HPCMatrix{T,B}) where {T,B} = transpose(conj(A))

# Type alias for transpose of HPCMatrix (matches any backend)
const TransposedHPCMatrix{T,B} = Transpose{T,HPCMatrix{T,B}}

"""
    LinearAlgebra.copy(At::Transpose{T,<:HPCMatrix{T}}) where T

Materialize a transposed HPCMatrix.
"""
function LinearAlgebra.copy(At::Transpose{T,<:HPCMatrix{T}}) where T
    A = At.parent
    plan = get_dense_transpose_plan(A)
    return execute_plan!(plan, A)
end

# transpose(A) * x for HPCMatrix

"""
    DenseTransposeVectorPlan{T,AV}

A communication plan for computing transpose(A) * x where A is HPCMatrix.

For transpose(A) * x:
- We need to gather x[1:nrows] according to A's row_partition
- Then compute transpose(A.A) * gathered_x locally
- Result has A.col_partition

# Type Parameters
- `T`: Element type
- `AV<:AbstractVector{T}`: Storage type for gathered buffer (matches input HPCVector)
"""
mutable struct DenseTransposeVectorPlan{T,AV<:AbstractVector{T}}
    send_rank_ids::Vector{Int}
    send_indices::Vector{Vector{Int}}
    send_bufs::Vector{Vector{T}}      # Always CPU for MPI
    send_reqs::Vector{Any}
    recv_rank_ids::Vector{Int}
    recv_bufs::Vector{Vector{T}}      # Always CPU for MPI
    recv_reqs::Vector{Any}
    recv_perm::Vector{Vector{Int}}
    local_src_indices::Vector{Int}
    local_dst_indices::Vector{Int}
    gathered::AV                       # Same type as input vector
    gathered_cpu::Vector{T}            # CPU staging buffer
    # Partition for result vector (computed at plan creation)
    result_partition_hash::Blake3Hash
    result_partition::Vector{Int}
end

# Cache for DenseTransposeVectorPlans (includes backend type in key)
const _dense_transpose_vector_plan_cache = Dict{Tuple{Blake3Hash,Blake3Hash,DataType,DataType},Any}()

"""
    DenseTransposeVectorPlan(A::HPCMatrix{T,B}, x::HPCVector{T,B}) where {T,B<:HPCBackend}

Create a plan to gather x elements according to A's row_partition for transpose(A) * x.
The gathered buffer has the same storage type as x (CPU or GPU).
"""
function DenseTransposeVectorPlan(A::HPCMatrix{T,B}, x::HPCVector{T,B}) where {T,B<:HPCBackend}
    assert_backends_compatible(A.backend, x.backend)
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    # For transpose(A) * x, we need x[1:nrows] where nrows = A.row_partition[end] - 1
    nrows = A.row_partition[end] - 1
    col_indices = collect(1:nrows)
    n_gathered = nrows

    my_x_start = x.partition[rank+1]

    # Group col_indices by owner rank in x's partition
    needed_from = [Tuple{Int,Int}[] for _ in 1:nranks]
    for (dst_idx, global_idx) in enumerate(col_indices)
        owner = searchsortedlast(x.partition, global_idx) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(needed_from[owner+1], (global_idx, dst_idx))
    end

    # Exchange counts via Alltoall
    send_counts = [length(needed_from[r+1]) for r in 0:(nranks-1)]
    recv_counts = comm_alltoall(comm, MPI.UBuffer(send_counts, 1))

    # Send requested indices to each owner rank
    struct_send_bufs = Dict{Int,Vector{Int}}()
    struct_send_reqs = []
    recv_rank_ids = Int[]
    recv_perm_map = Dict{Int,Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r+1] > 0 && r != rank
            push!(recv_rank_ids, r)
            indices = [t[1] for t in needed_from[r+1]]
            dst_indices = [t[2] for t in needed_from[r+1]]
            recv_perm_map[r] = dst_indices
            struct_send_bufs[r] = indices
            req = comm_isend(comm, indices, r, 50)
            push!(struct_send_reqs, req)
        end
    end

    # Receive requests from other ranks
    send_rank_ids = Int[]
    struct_recv_bufs = Dict{Int,Vector{Int}}()
    struct_recv_reqs = []

    for r in 0:(nranks-1)
        if recv_counts[r+1] > 0 && r != rank
            push!(send_rank_ids, r)
            buf = Vector{Int}(undef, recv_counts[r+1])
            req = comm_irecv!(comm, buf, r, 50)
            push!(struct_recv_reqs, req)
            struct_recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, struct_recv_reqs)
    comm_waitall(comm, struct_send_reqs)

    # Convert received global indices to local indices for sending
    send_indices_map = Dict{Int,Vector{Int}}()
    for r in send_rank_ids
        global_indices = struct_recv_bufs[r]
        local_indices = [idx - my_x_start + 1 for idx in global_indices]
        send_indices_map[r] = local_indices
    end

    # Handle local elements
    local_src_indices = Int[]
    local_dst_indices = Int[]
    for (global_idx, dst_idx) in needed_from[rank+1]
        local_idx = global_idx - my_x_start + 1
        push!(local_src_indices, local_idx)
        push!(local_dst_indices, dst_idx)
    end

    # Build final arrays and buffers
    sort!(send_rank_ids)
    sort!(recv_rank_ids)

    send_indices_final = [send_indices_map[r] for r in send_rank_ids]
    recv_perm_final = [recv_perm_map[r] for r in recv_rank_ids]

    send_bufs = [Vector{T}(undef, length(inds)) for inds in send_indices_final]
    recv_bufs = [Vector{T}(undef, send_counts[r+1]) for r in recv_rank_ids]
    send_reqs = Vector{Any}(undef, length(send_rank_ids))
    recv_reqs = Vector{Any}(undef, length(recv_rank_ids))

    # Create gathered buffer with same type as input and CPU staging buffer
    gathered_cpu = Vector{T}(undef, n_gathered)
    gathered = similar(x.v, n_gathered)

    # Compute result partition hash eagerly
    result_partition = copy(A.col_partition)
    result_partition_hash = compute_partition_hash(result_partition)

    return DenseTransposeVectorPlan{T,typeof(gathered)}(
        send_rank_ids, send_indices_final, send_bufs, send_reqs,
        recv_rank_ids, recv_bufs, recv_reqs, recv_perm_final,
        local_src_indices, local_dst_indices, gathered, gathered_cpu,
        result_partition_hash, result_partition
    )
end

"""
    execute_plan!(plan::DenseTransposeVectorPlan{T,AV}, x::HPCVector{T,B}) where {T,AV,B}

Execute the transpose vector plan to gather x elements.

Works uniformly for CPU and GPU vectors - MPI communication always uses CPU
buffers, with automatic staging for GPU arrays.
"""
function execute_plan!(plan::DenseTransposeVectorPlan{T,AV}, x::HPCVector{T,B}) where {T,AV,B}
    comm = x.backend.comm

    # Get CPU data for MPI operations (no-op for CPU, copy for GPU)
    x_cpu = _ensure_cpu(x.v)

    # Step 1: Copy local values to CPU staging buffer
    @inbounds for i in eachindex(plan.local_src_indices, plan.local_dst_indices)
        plan.gathered_cpu[plan.local_dst_indices[i]] = x_cpu[plan.local_src_indices[i]]
    end

    # Step 2: Fill send buffers and send
    send_reqs = []
    @inbounds for i in eachindex(plan.send_rank_ids)
        r = plan.send_rank_ids[i]
        send_idx = plan.send_indices[i]
        buf = plan.send_bufs[i]
        for k in eachindex(send_idx)
            buf[k] = x_cpu[send_idx[k]]
        end
        req = comm_isend(comm, buf, r, 51)
        push!(send_reqs, req)
    end

    # Step 3: Receive values to CPU buffers
    recv_reqs = []
    @inbounds for i in eachindex(plan.recv_rank_ids)
        req = comm_irecv!(comm, plan.recv_bufs[i], plan.recv_rank_ids[i], 51)
        push!(recv_reqs, req)
    end

    comm_waitall(comm, recv_reqs)

    # Step 4: Scatter received values into CPU staging buffer
    @inbounds for i in eachindex(plan.recv_rank_ids)
        perm = plan.recv_perm[i]
        buf = plan.recv_bufs[i]
        for k in eachindex(perm)
            plan.gathered_cpu[perm[k]] = buf[k]
        end
    end

    comm_waitall(comm, send_reqs)

    # Step 5: Copy to output buffer (no-op if CPU, copy if GPU)
    _copy_to_output!(plan.gathered, plan.gathered_cpu)

    return plan.gathered
end

"""
    get_dense_transpose_vector_plan(A::HPCMatrix{T,B}, x::HPCVector{T,B}) where {T,B}

Get a memoized DenseTransposeVectorPlan for transpose(A) * x.
Cache key includes backend type for GPU/CPU separation.
"""
function get_dense_transpose_vector_plan(A::HPCMatrix{T,B}, x::HPCVector{T,B}) where {T,B}
    key = (A.structural_hash, x.structural_hash, T, B)
    if haskey(_dense_transpose_vector_plan_cache, key)
        return _dense_transpose_vector_plan_cache[key]::DenseTransposeVectorPlan{T}
    end
    plan = DenseTransposeVectorPlan(A, x)
    _dense_transpose_vector_plan_cache[key] = plan
    return plan
end

"""
    Base.:*(At::Transpose{T,HPCMatrix{T,B}}, x::HPCVector{T,B}) where {T,B<:HPCBackend}

Compute transpose(A) * x without materializing the transpose.
"""
function Base.:*(At::Transpose{T,HPCMatrix{T,B}}, x::HPCVector{T,B}) where {T,B<:HPCBackend}
    A = At.parent
    assert_backends_compatible(A.backend, x.backend)

    comm = A.backend.comm
    rank = comm_rank(comm)

    plan = get_dense_transpose_vector_plan(A, x)

    execute_plan!(plan, x)

    # Local computation: transpose(A.A) * local_gathered
    # A.A has shape (local_nrows, ncols), gathered has shape (nrows,)
    # We need gathered[my_row_start:my_row_end] for the local rows
    my_row_start = A.row_partition[rank+1]
    my_row_end = A.row_partition[rank+2] - 1

    # Unified CPU/GPU path:
    # 1. Get slice and copy to contiguous array (fixes GPU view issues)
    # 2. Compute on backend
    # 3. Ensure CPU for Allreduce (no-op for CPU, copy for GPU)
    local_gathered_slice = plan.gathered[my_row_start:my_row_end]
    local_gathered_contiguous = copy(local_gathered_slice)
    partial_result_backend = transpose(A.A) * local_gathered_contiguous
    partial_result = _ensure_cpu(partial_result_backend)

    # Allreduce to sum contributions from all ranks
    full_result = comm_allreduce(comm, partial_result, +)

    # Extract our portion according to col_partition
    my_col_start = A.col_partition[rank+1]
    my_col_end = A.col_partition[rank+2] - 1
    local_result_cpu = full_result[my_col_start:my_col_end]

    # Unified: _convert_array is no-op for CPU, copies for GPU
    local_result = _convert_array(local_result_cpu, A.backend.device)

    # Create result vector (partition is immutable, no need to copy)
    y = HPCVector{T,B}(
        plan.result_partition_hash,
        plan.result_partition,
        local_result,
        A.backend
    )
    return y
end

# Vector * HPCMatrix operations

"""
    Base.:*(vt::Transpose{<:Any, HPCVector{T}}, A::HPCMatrix{T}) where T

Compute transpose(v) * A = transpose(transpose(A) * v).
"""
function Base.:*(vt::Transpose{<:Any, HPCVector{T}}, A::HPCMatrix{T}) where T
    v = vt.parent
    result = transpose(A) * v
    return transpose(result)
end

# ============================================================================
# Dense Matrix-Matrix Multiplication with Transpose
# ============================================================================

"""
    Base.:*(At::TransposedHPCMatrix{T,B}, Bmat::HPCMatrix{T,B}) where {T,B}

Compute transpose(A) * B for dense matrices.
Uses column-by-column multiplication via transpose(A) * b_col.
"""
function Base.:*(At::TransposedHPCMatrix{T,B}, Bmat::HPCMatrix{T,B}) where {T,B}
    A = At.parent
    assert_backends_compatible(A.backend, Bmat.backend)
    n = size(Bmat, 2)

    rank = comm_rank(A.backend.comm)

    # Multiply column by column using existing transpose(A) * vector
    columns = Vector{HPCVector{T,B}}(undef, n)
    for k in 1:n
        b_col = Bmat[:, k]
        columns[k] = transpose(A) * b_col
    end

    # Get partition from first column result
    result_partition = columns[1].partition
    local_m = result_partition[rank+2] - result_partition[rank+1]

    # Build local matrix from column results (preserves GPU array type)
    local_result = reduce(hcat, [columns[k].v for k in 1:n])

    return HPCMatrix_local(local_result, A.backend)
end

# Scalar multiplication for HPCMatrix

"""
    Base.:*(a::Number, A::HPCMatrix{T,B}) where {T,B}

Scalar times dense matrix.
"""
function Base.:*(a::Number, A::HPCMatrix{T,B}) where {T,B}
    RT = promote_type(typeof(a), T)
    return HPCMatrix{RT,B}(A.structural_hash, A.row_partition, A.col_partition, RT.(a .* A.A), A.backend)
end

"""
    Base.:*(A::HPCMatrix{T,B}, a::Number) where {T,B}

Dense matrix times scalar.
"""
Base.:*(A::HPCMatrix{T,B}, a::Number) where {T,B} = a * A

"""
    Base.:*(a::Number, At::TransposedHPCMatrix{T,B}) where {T,B}

Scalar times transposed dense matrix.
"""
Base.:*(a::Number, At::TransposedHPCMatrix{T,B}) where {T,B} = transpose(a * At.parent)

"""
    Base.:*(At::TransposedHPCMatrix{T,B}, a::Number) where {T,B}

Transposed dense matrix times scalar.
"""
Base.:*(At::TransposedHPCMatrix{T,B}, a::Number) where {T,B} = transpose(At.parent * a)

# Utility methods for HPCMatrix

"""
    Base.size(A::HPCMatrix)

Return the size of the distributed dense matrix as (nrows, ncols).
"""
function Base.size(A::HPCMatrix)
    nrows = A.row_partition[end] - 1
    ncols = A.col_partition[end] - 1
    return (nrows, ncols)
end

Base.size(A::HPCMatrix, d::Integer) = size(A)[d]

Base.eltype(::HPCMatrix{T}) where T = T
Base.eltype(::Type{HPCMatrix{T}}) where T = T

"""
    Base.sum(A::HPCMatrix{T,B}; dims=nothing) where {T,B}

Compute the sum of all elements in the distributed matrix.
If dims is specified, sum along that dimension.
"""
function Base.sum(A::HPCMatrix{T,B}; dims=nothing) where {T,B}
    comm = A.backend.comm

    if dims === nothing
        # Sum all elements
        local_sum = sum(A.A; init=zero(T))
        return comm_allreduce(comm, local_sum, +)
    elseif dims == 1
        # Sum along rows (each rank has some rows, result is a row vector)
        # Each rank computes column sums of its local rows
        local_colsums = sum(A.A, dims=1)
        # Reduce across all ranks
        global_colsums = comm_allreduce(comm, local_colsums, +)
        return global_colsums
    elseif dims == 2
        # Sum along columns (result is distributed column vector)
        local_rowsums = sum(A.A, dims=2)
        # Result inherits row partition from A
        return HPCMatrix{T,B}(A.structural_hash, A.row_partition, A.col_partition, local_rowsums, A.backend)
    else
        error("dims must be nothing, 1, or 2")
    end
end

"""
    LinearAlgebra.norm(A::HPCMatrix{T,B}, p::Real=2) where {T,B}

Compute the p-norm of A treated as a vector of elements.
- `p=2` (default): Frobenius norm
- `p=1`: Sum of absolute values
- `p=Inf`: Maximum absolute value
"""
function LinearAlgebra.norm(A::HPCMatrix{T,B}, p::Real=2) where {T,B}
    comm = A.backend.comm
    local_vals = A.A

    if p == 2
        local_sum = sum(abs2, local_vals; init=zero(real(T)))
        global_sum = comm_allreduce(comm, local_sum, +)
        return sqrt(global_sum)
    elseif p == 1
        local_sum = sum(abs, local_vals; init=zero(real(T)))
        return comm_allreduce(comm, local_sum, +)
    elseif p == Inf
        local_max = isempty(local_vals) ? zero(real(T)) : maximum(abs, local_vals)
        return comm_allreduce(comm, local_max, max)
    else
        local_sum = sum(x -> abs(x)^p, local_vals; init=zero(real(T)))
        global_sum = comm_allreduce(comm, local_sum, +)
        return global_sum^(1 / p)
    end
end

"""
    LinearAlgebra.opnorm(A::HPCMatrix{T,B}, p::Real=1) where {T,B}

Compute the induced operator norm of dense matrix A.
- `p=1`: Maximum absolute column sum
- `p=Inf`: Maximum absolute row sum
"""
function LinearAlgebra.opnorm(A::HPCMatrix{T,B}, p::Real=1) where {T,B}
    comm = A.backend.comm

    if p == Inf
        # Maximum absolute row sum
        # Each rank owns some rows
        local_nrows = size(A.A, 1)
        local_max = zero(real(T))
        for i in 1:local_nrows
            row_sum = sum(abs, @view A.A[i, :]; init=zero(real(T)))
            local_max = max(local_max, row_sum)
        end
        return comm_allreduce(comm, local_max, max)

    elseif p == 1
        # Maximum absolute column sum
        ncols = size(A.A, 2)
        local_col_sums = zeros(real(T), ncols)
        for j in 1:ncols
            local_col_sums[j] = sum(abs, @view A.A[:, j]; init=zero(real(T)))
        end
        global_col_sums = comm_allreduce(comm, local_col_sums, +)
        return maximum(global_col_sums; init=zero(real(T)))

    else
        error("opnorm(A, $p) is not implemented. Use p=1 or p=Inf.")
    end
end

# mapslices for HPCMatrix

"""
    Base.mapslices(f, A::HPCMatrix{T,B}; dims) where {T,B}

Apply function f to slices of distributed matrix A along dimension dims.

- `dims=2`: Apply f to each row (local operation, no MPI communication)
- `dims=1`: Apply f to each column (requires MPI communication to gather columns)

Returns a HPCMatrix with the results.

# Example
```julia
backend = backend_cpu_mpi(MPI.COMM_WORLD)
A = HPCMatrix(randn(5,3), backend)
# Apply function to each row, returning 2-element result
B = mapslices(x -> [norm(x), maximum(x)], A; dims=2)  # Returns 5×2 HPCMatrix
```
"""
function Base.mapslices(f, A::HPCMatrix{T,B}; dims) where {T,B}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    if dims == 2
        # Apply f to each row - LOCAL operation (rows are distributed)
        local_result = mapslices(f, A.A; dims=2)

        # Result has same row count, possibly different column count
        new_ncols = size(local_result, 2)

        # Create new col_partition for result
        new_col_partition = uniform_partition(new_ncols, nranks)

        # Compute structural hash
        new_hash = compute_dense_structural_hash(
            A.row_partition, new_col_partition, size(local_result), comm)

        RT = eltype(local_result)
        return HPCMatrix{RT,B}(new_hash, A.row_partition, new_col_partition, local_result, A.backend)

    elseif dims == 1
        # Apply f to each column - REQUIRES MPI (columns span ranks)
        m_global = A.row_partition[end] - 1
        n = size(A.A, 2)  # columns are not partitioned

        results = Vector{Any}(undef, n)
        for j in 1:n
            # Gather full column j from all ranks
            # Unified: _ensure_cpu is no-op for CPU, Array() for GPU
            local_col = A.A[:, j]
            local_col_cpu = _ensure_cpu(local_col)
            counts = Int32[A.row_partition[r+1] - A.row_partition[r] for r in 1:nranks]
            full_col = Vector{T}(undef, m_global)
            comm_allgatherv!(comm, local_col_cpu, MPI.VBuffer(full_col, counts))

            # Apply f to full column
            results[j] = f(full_col)
        end

        # Stack results - each is a column of the output
        # Result is (output_len × n)
        local_full_result = hcat(results...)

        # Redistribute rows according to standard partition
        output_m = size(local_full_result, 1)
        new_row_partition = uniform_partition(output_m, nranks)
        row_start = new_row_partition[rank+1]
        row_end = new_row_partition[rank+2] - 1
        local_result = local_full_result[row_start:row_end, :]

        new_col_partition = uniform_partition(n, nranks)
        new_hash = compute_dense_structural_hash(
            new_row_partition, new_col_partition, size(local_result), comm)

        RT = eltype(local_result)
        return HPCMatrix{RT,B}(new_hash, new_row_partition, new_col_partition, local_result, A.backend)
    else
        error("dims must be 1 or 2")
    end
end

# ============================================================================
# DenseRepartitionPlan: Repartition a HPCMatrix to a new row partition
# ============================================================================

# Helper to get CPU copy of matrix data (no-op for CPU, copy for GPU)
# Used for MPI communication which always requires CPU buffers
_ensure_cpu(M::Matrix) = M
_ensure_cpu(M::AbstractMatrix) = Array(M)

"""
    DenseRepartitionPlan{T}

Communication plan for repartitioning a HPCMatrix to a new row partition.

# Fields
- `send_rank_ids::Vector{Int}`: Ranks we send rows to (0-indexed)
- `send_row_ranges::Vector{UnitRange{Int}}`: For each rank, range of local rows to send
- `send_bufs::Vector{Matrix{T}}`: Pre-allocated send buffers
- `send_reqs::Vector{Any}`: Pre-allocated send request handles
- `recv_rank_ids::Vector{Int}`: Ranks we receive rows from (0-indexed)
- `recv_row_counts::Vector{Int}`: Number of rows to receive from each rank
- `recv_bufs::Vector{Matrix{T}}`: Pre-allocated receive buffers
- `recv_reqs::Vector{Any}`: Pre-allocated receive request handles
- `recv_dst_ranges::Vector{UnitRange{Int}}`: Destination row ranges in result for each recv
- `local_src_range::UnitRange{Int}`: Source row range for local copy
- `local_dst_range::UnitRange{Int}`: Destination row range for local copy
- `result_row_partition::Vector{Int}`: Target row partition (copy of p)
- `result_col_partition::Vector{Int}`: Column partition (unchanged from source)
- `result_structural_hash::Blake3Hash`: Hash of result matrix
- `result_local_nrows::Int`: Number of rows this rank owns after repartition
- `ncols::Int`: Number of columns in the matrix
"""
mutable struct DenseRepartitionPlan{T}
    send_rank_ids::Vector{Int}
    send_row_ranges::Vector{UnitRange{Int}}
    send_bufs::Vector{Matrix{T}}
    send_reqs::Vector{Any}
    recv_rank_ids::Vector{Int}
    recv_row_counts::Vector{Int}
    recv_bufs::Vector{Matrix{T}}
    recv_reqs::Vector{Any}
    recv_dst_ranges::Vector{UnitRange{Int}}
    local_src_range::UnitRange{Int}
    local_dst_range::UnitRange{Int}
    result_row_partition::Vector{Int}
    result_col_partition::Vector{Int}
    result_structural_hash::Blake3Hash
    result_local_nrows::Int
    ncols::Int
end

"""
    DenseRepartitionPlan(A::HPCMatrix{T,B}, p::Vector{Int}) where {T,B}

Create a communication plan to repartition `A` to have row partition `p`.
The col_partition remains unchanged.

The plan computes:
1. Which rows to send to each rank based on partition overlap
2. Which rows to receive from each rank
3. Pre-allocates all buffers for allocation-free execution
4. Computes the result structural hash eagerly
"""
function DenseRepartitionPlan(A::HPCMatrix{T,B}, p::Vector{Int}) where {T,B}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    # Source partition info
    src_start = A.row_partition[rank+1]
    src_end = A.row_partition[rank+2] - 1
    local_nrows = max(0, src_end - src_start + 1)

    # Target partition info
    dst_start = p[rank+1]
    dst_end = p[rank+2] - 1
    result_local_nrows = max(0, dst_end - dst_start + 1)

    ncols = A.col_partition[end] - 1

    # Step 1: Determine which rows we send to each rank
    send_row_ranges_map = Dict{Int, UnitRange{Int}}()
    for r in 0:(nranks-1)
        r_start = p[r+1]
        r_end = p[r+2] - 1
        if r_end < r_start
            continue  # rank r has no rows in target partition
        end
        # Intersection of our rows with rank r's target
        overlap_start = max(src_start, r_start)
        overlap_end = min(src_end, r_end)
        if overlap_start <= overlap_end
            # Convert to local row indices in A.A
            local_start = overlap_start - src_start + 1
            local_end = overlap_end - src_start + 1
            send_row_ranges_map[r] = local_start:local_end
        end
    end

    # Step 2: Exchange counts via Alltoall
    send_counts = Int32[haskey(send_row_ranges_map, r) ? length(send_row_ranges_map[r]) : 0 for r in 0:(nranks-1)]
    recv_counts_raw = comm_alltoall(comm, MPI.UBuffer(send_counts, 1))

    # Step 3: Build send/recv structures
    send_rank_ids = Int[]
    send_row_ranges = UnitRange{Int}[]
    recv_rank_ids = Int[]
    recv_row_counts = Int[]
    recv_dst_ranges = UnitRange{Int}[]

    local_src_range = 1:0  # empty range
    local_dst_range = 1:0  # empty range

    # Handle local copy separately
    if haskey(send_row_ranges_map, rank)
        local_src_range = send_row_ranges_map[rank]
        # Compute destination range: where do these rows go in the result?
        global_start = src_start + local_src_range.start - 1
        local_dst_start = global_start - dst_start + 1
        local_dst_end = local_dst_start + length(local_src_range) - 1
        local_dst_range = local_dst_start:local_dst_end
    end

    # Build send arrays (excluding local)
    for r in 0:(nranks-1)
        if haskey(send_row_ranges_map, r) && r != rank
            push!(send_rank_ids, r)
            push!(send_row_ranges, send_row_ranges_map[r])
        end
    end

    # Build recv arrays (excluding local)
    for r in 0:(nranks-1)
        if recv_counts_raw[r+1] > 0 && r != rank
            push!(recv_rank_ids, r)
            push!(recv_row_counts, recv_counts_raw[r+1])

            # Rows from rank r: their source range is [A.row_partition[r+1], A.row_partition[r+2]-1]
            # intersected with our target range [dst_start, dst_end]
            r_src_start = A.row_partition[r+1]
            r_src_end = A.row_partition[r+2] - 1
            overlap_start = max(r_src_start, dst_start)
            overlap_end = min(r_src_end, dst_end)
            # Destination range in our result
            dst_range_start = overlap_start - dst_start + 1
            dst_range_end = overlap_end - dst_start + 1
            push!(recv_dst_ranges, dst_range_start:dst_range_end)
        end
    end

    # Pre-allocate buffers
    send_bufs = [Matrix{T}(undef, length(r), ncols) for r in send_row_ranges]
    recv_bufs = [Matrix{T}(undef, c, ncols) for c in recv_row_counts]
    send_reqs = Vector{Any}(undef, length(send_rank_ids))
    recv_reqs = Vector{Any}(undef, length(recv_rank_ids))

    # Compute result structural hash eagerly
    result_local_size = (result_local_nrows, ncols)
    result_structural_hash = compute_dense_structural_hash(p, A.col_partition, result_local_size, comm)

    return DenseRepartitionPlan{T}(
        send_rank_ids, send_row_ranges, send_bufs, send_reqs,
        recv_rank_ids, recv_row_counts, recv_bufs, recv_reqs, recv_dst_ranges,
        local_src_range, local_dst_range,
        copy(p), copy(A.col_partition), result_structural_hash,
        result_local_nrows, ncols
    )
end

"""
    execute_plan!(plan::DenseRepartitionPlan{T}, A::HPCMatrix{T,B}) where {T,B}

Execute a dense repartition plan to redistribute rows from A to a new partition.
Returns a new HPCMatrix with the target row partition, preserving the backend.
"""
function execute_plan!(plan::DenseRepartitionPlan{T}, A::HPCMatrix{T,B}) where {T,B}
    comm = A.backend.comm

    # Get CPU version of input for MPI operations
    A_cpu = _ensure_cpu(A.A)

    # Allocate CPU result buffer (MPI requires CPU)
    result_cpu = Matrix{T}(undef, plan.result_local_nrows, plan.ncols)

    # Step 1: Local copy (from CPU staging)
    if !isempty(plan.local_src_range) && !isempty(plan.local_dst_range)
        result_cpu[plan.local_dst_range, :] = A_cpu[plan.local_src_range, :]
    end

    # Step 2: Fill send buffers and send (from CPU staging)
    send_reqs = []
    @inbounds for i in eachindex(plan.send_rank_ids)
        r = plan.send_rank_ids[i]
        row_range = plan.send_row_ranges[i]
        buf = plan.send_bufs[i]
        buf .= @view A_cpu[row_range, :]
        req = comm_isend(comm, vec(buf), r, 93)
        push!(send_reqs, req)
    end

    # Step 3: Post receives
    recv_reqs = []
    @inbounds for i in eachindex(plan.recv_rank_ids)
        req = comm_irecv!(comm, vec(plan.recv_bufs[i]), plan.recv_rank_ids[i], 93)
        push!(recv_reqs, req)
    end

    comm_waitall(comm, recv_reqs)

    # Step 4: Copy received rows into result
    @inbounds for i in eachindex(plan.recv_rank_ids)
        dst_range = plan.recv_dst_ranges[i]
        buf = plan.recv_bufs[i]
        result_cpu[dst_range, :] = buf
    end

    comm_waitall(comm, send_reqs)

    # Copy result back to target array type using dispatch (no type checks)
    result_A = _matrix_to_backend(result_cpu, A.A)

    return HPCMatrix{T,B}(plan.result_structural_hash, plan.result_row_partition, plan.result_col_partition, result_A, A.backend)
end

"""
    get_repartition_plan(A::HPCMatrix{T,B}, p::Vector{Int}) where {T,B}

Get a memoized DenseRepartitionPlan for repartitioning `A` to row partition `p`.
The plan is cached based on the structural hash of A and the target partition hash.
"""
function get_repartition_plan(A::HPCMatrix{T,B}, p::Vector{Int}) where {T,B}
    target_hash = compute_partition_hash(p)
    key = (A.structural_hash, target_hash, T, B)
    if haskey(_repartition_plan_cache, key)
        return _repartition_plan_cache[key]::DenseRepartitionPlan{T}
    end
    plan = DenseRepartitionPlan(A, p)
    _repartition_plan_cache[key] = plan
    return plan
end

"""
    repartition(A::HPCMatrix{T}, p::Vector{Int}) where T

Redistribute a HPCMatrix to a new row partition `p`.
The col_partition remains unchanged.

The partition `p` must be a valid partition vector of length `nranks + 1` with
`p[1] == 1` and `p[end] == size(A, 1) + 1`.

Returns a new HPCMatrix with the same data but `row_partition == p`.

# Example
```julia
A = HPCMatrix(randn(6, 4))  # uniform partition
new_partition = [1, 2, 4, 5, 7]  # 1, 2, 1, 2 rows per rank
A_repart = repartition(A, new_partition)
```
"""
function repartition(A::HPCMatrix{T}, p::Vector{Int}) where T
    # Fast path: partition unchanged (identity check first, then element-wise)
    if A.row_partition === p || A.row_partition == p
        return A
    end

    plan = get_repartition_plan(A, p)
    return execute_plan!(plan, A)
end

# ============================================================================
# Scalar multiplication for HPCMatrix (GPU-compatible)
# ============================================================================

"""
    Base.:*(α::Number, A::HPCMatrix{T,B}) where {T,B<:HPCBackend}

Scalar-matrix multiplication. GPU-compatible: uses the underlying array's
multiplication which dispatches to GPU kernels.
"""
function Base.:*(α::Number, A::HPCMatrix{T,B}) where {T,B<:HPCBackend}
    new_A = T(α) * A.A
    HPCMatrix{T,B}(A.structural_hash, A.row_partition, A.col_partition, new_A, A.backend)
end

"""
    Base.:*(A::HPCMatrix{T,B}, α::Number) where {T,B<:HPCBackend}

Matrix-scalar multiplication. GPU-compatible.
"""
function Base.:*(A::HPCMatrix{T,B}, α::Number) where {T,B<:HPCBackend}
    new_A = A.A * T(α)
    HPCMatrix{T,B}(A.structural_hash, A.row_partition, A.col_partition, new_A, A.backend)
end

"""
    Base.:/(A::HPCMatrix{T,B}, α::Number) where {T,B<:HPCBackend}

Matrix-scalar division. GPU-compatible.
"""
function Base.:/(A::HPCMatrix{T,B}, α::Number) where {T,B<:HPCBackend}
    new_A = A.A / T(α)
    HPCMatrix{T,B}(A.structural_hash, A.row_partition, A.col_partition, new_A, A.backend)
end

# Broadcasting: scalar .* matrix - redirect to scalar * matrix
Base.broadcasted(::typeof(*), α::Number, A::HPCMatrix) = α * A
Base.broadcasted(::typeof(*), A::HPCMatrix, α::Number) = A * α
Base.broadcasted(::typeof(/), A::HPCMatrix, α::Number) = A / α

# Handle HPCMatrix in HPCVector broadcasts by extracting the underlying local matrix data
# This enables broadcasting HPCVector .* HPCMatrix on GPU without passing the wrapper
# (wrapper contains non-bitstype fields that can't be passed to GPU kernels)
_prepare_broadcast_arg(m::HPCMatrix, ref_partition, comm) = m.A
