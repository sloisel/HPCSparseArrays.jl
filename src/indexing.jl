# Indexing operations for distributed types
# Communication tags: 40 (index request), 41 (value response)

"""
    _invalidate_cached_transpose!(A::HPCSparseMatrix)

Invalidate the cached transpose of A bidirectionally.
If A has a cached transpose B, then B.cached_transpose is also cleared.
"""
function _invalidate_cached_transpose!(A::HPCSparseMatrix)
    if A.cached_transpose !== nothing
        A.cached_transpose.cached_transpose = nothing
    end
    A.cached_transpose = nothing
end

# ============================================================================
# Scalar indexing REMOVED for all MPI types to prevent MPI desync.
# Generic AbstractArray fallbacks (sum, iterate, etc.) use scalar indexing
# which causes ranks to diverge. Only slice/range operations are allowed.
# ============================================================================

# ============================================================================
# Range Indexing for HPCVector
# ============================================================================

"""
    _compute_subpartition(partition::Vector{Int}, rng::UnitRange{Int})

Compute a new partition for a subrange of a distributed object.

Given a partition over global indices and a range `rng`, returns a new partition
where each rank's portion is the intersection of its original portion with `rng`,
mapped to local indices starting at 1.

This computation is local (no communication needed) since all ranks have the same partition.
"""
function _compute_subpartition(partition::Vector{Int}, rng::UnitRange{Int})
    nranks = length(partition) - 1
    new_partition = Vector{Int}(undef, nranks + 1)
    new_partition[1] = 1

    for r in 1:nranks
        # Rank r owns global indices partition[r]:(partition[r+1]-1)
        rank_start = partition[r]
        rank_end = partition[r + 1] - 1

        # Intersection with rng
        intersect_start = max(rank_start, first(rng))
        intersect_end = min(rank_end, last(rng))

        if intersect_start <= intersect_end
            count = intersect_end - intersect_start + 1
        else
            count = 0
        end

        new_partition[r + 1] = new_partition[r] + count
    end

    return new_partition
end

"""
    Base.getindex(v::HPCVector{T}, rng::UnitRange{Int}) where T

Extract a subvector `v[rng]` from a distributed vector, returning a new HPCVector.

This is a collective operation - all ranks must call it with the same range.
The result has a partition derived from `v.partition` such that each rank
extracts only its local portion (no data communication, only hash computation).

# Example
```julia
v = HPCVector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
w = v[3:6]  # Returns HPCVector with elements [3.0, 4.0, 5.0, 6.0]
```
"""
function Base.getindex(v::HPCVector{T,B}, rng::UnitRange{Int}) where {T, B<:HPCBackend}
    backend = v.backend
    comm = backend.comm
    device = backend.device
    rank = comm_rank(comm)

    n = length(v)
    if first(rng) < 1 || last(rng) > n
        error("HPCVector range out of bounds: $rng, length=$n")
    end

    if isempty(rng)
        # Empty range - return empty HPCVector
        new_partition = ones(Int, comm_size(comm) + 1)
        hash = compute_partition_hash(new_partition)
        return HPCVector{T,B}(hash, new_partition, similar(v.v, 0), backend)
    end

    # Compute new partition (local computation, no communication)
    new_partition = _compute_subpartition(v.partition, rng)

    # Extract local portion
    my_start = v.partition[rank + 1]
    my_end = v.partition[rank + 2] - 1

    # Intersection of my range with rng
    intersect_start = max(my_start, first(rng))
    intersect_end = min(my_end, last(rng))

    if intersect_start <= intersect_end
        # Convert to local indices in v.v
        local_start = intersect_start - my_start + 1
        local_end = intersect_end - my_start + 1
        local_v = v.v[local_start:local_end]
    else
        local_v = similar(v.v, 0)
    end

    # Compute hash (requires Allgather for consistency)
    hash = compute_partition_hash(new_partition)

    return HPCVector{T,B}(hash, new_partition, local_v, backend)
end

"""
    Base.setindex!(v::HPCVector{T}, vals, rng::UnitRange{Int}) where T

Set elements `v[rng] = vals` in a distributed vector.

This is a collective operation - all ranks must call it with the same range.
`vals` can be:
- A scalar (broadcast to all positions)
- A HPCVector with compatible length
- A regular Vector (must have length equal to the range)

Only ranks that own elements in the range modify their local data.

# Example
```julia
v = HPCVector([1.0, 2.0, 3.0, 4.0])
v[2:3] = 0.0  # Set elements 2 and 3 to zero
v[2:3] = [5.0, 6.0]  # Set elements 2 and 3 to 5.0 and 6.0
```
"""
function Base.setindex!(v::HPCVector{T,B}, val::Number, rng::UnitRange{Int}) where {T, B<:HPCBackend}
    backend = v.backend
    comm = backend.comm
    rank = comm_rank(comm)

    n = length(v)
    if first(rng) < 1 || last(rng) > n
        error("HPCVector range out of bounds: $rng, length=$n")
    end

    if isempty(rng)
        return val
    end

    # Find intersection with my portion
    my_start = v.partition[rank + 1]
    my_end = v.partition[rank + 2] - 1

    intersect_start = max(my_start, first(rng))
    intersect_end = min(my_end, last(rng))

    if intersect_start <= intersect_end
        # Convert to local indices
        local_start = intersect_start - my_start + 1
        local_end = intersect_end - my_start + 1
        v.v[local_start:local_end] .= convert(T, val)
    end

    return val
end

function Base.setindex!(v::HPCVector{T,B}, vals::AbstractVector, rng::UnitRange{Int}) where {T, B<:HPCBackend}
    backend = v.backend
    comm = backend.comm
    rank = comm_rank(comm)

    n = length(v)
    if first(rng) < 1 || last(rng) > n
        error("HPCVector range out of bounds: $rng, length=$n")
    end

    if length(vals) != length(rng)
        error("HPCVector setindex!: length mismatch, got $(length(vals)) values for range of length $(length(rng))")
    end

    if isempty(rng)
        return vals
    end

    # Find intersection with my portion
    my_start = v.partition[rank + 1]
    my_end = v.partition[rank + 2] - 1

    intersect_start = max(my_start, first(rng))
    intersect_end = min(my_end, last(rng))

    if intersect_start <= intersect_end
        # Convert to local indices in v.v
        local_start = intersect_start - my_start + 1
        local_end = intersect_end - my_start + 1
        # Indices into vals
        vals_start = intersect_start - first(rng) + 1
        vals_end = intersect_end - first(rng) + 1
        v.v[local_start:local_end] .= convert.(T, vals[vals_start:vals_end])
    end

    return vals
end

"""
    Base.setindex!(v::HPCVector{T}, src::HPCVector, rng::UnitRange{Int}) where T

Set elements `v[rng] = src` where `src` is a distributed HPCVector.

This is a collective operation - all ranks must call it with the same `rng` and `src`.
Each rank only updates the elements it owns that fall within `rng`.

If the source partition matches the target partition induced by `rng`, a direct
local copy is performed. Otherwise, a communication plan redistributes the source values.

# Example
```julia
v = HPCVector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
src = HPCVector([10.0, 20.0, 30.0])
v[2:4] = src  # Each rank only writes to elements it owns
```
"""
function Base.setindex!(v::HPCVector{T,B}, src::HPCVector, rng::UnitRange{Int}) where {T, B<:HPCBackend}
    backend = v.backend
    comm = backend.comm
    rank = comm_rank(comm)

    n = length(v)
    if first(rng) < 1 || last(rng) > n
        error("HPCVector range out of bounds: $rng, length=$n")
    end

    if length(src) != length(rng)
        error("HPCVector setindex!: length mismatch, got $(length(src)) values for range of length $(length(rng))")
    end

    if isempty(rng)
        return src
    end

    # Compute my owned range intersection with rng
    my_start = v.partition[rank + 1]
    my_end = v.partition[rank + 2] - 1

    intersect_start = max(my_start, first(rng))
    intersect_end = min(my_end, last(rng))

    # Compute the partition of the target range
    target_partition = _compute_subpartition(v.partition, rng)

    # Check if source partition matches target partition
    if src.partition == target_partition
        # Direct local copy - each rank copies its local portion
        if intersect_start <= intersect_end
            local_start = intersect_start - my_start + 1
            local_end = intersect_end - my_start + 1
            v.v[local_start:local_end] .= convert.(T, src.v)
        end
    else
        # Need to align src to target_partition
        plan = VectorPlan(target_partition, src)
        aligned = execute_plan!(plan, src)

        if intersect_start <= intersect_end
            local_start = intersect_start - my_start + 1
            local_end = intersect_end - my_start + 1
            v.v[local_start:local_end] .= convert.(T, aligned)
        end
    end

    return src
end

# ============================================================================
# Range Indexing for HPCMatrix
# ============================================================================

"""
    Base.getindex(A::HPCMatrix{T}, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where T

Extract a submatrix `A[row_rng, col_rng]` from a distributed dense matrix, returning a new HPCMatrix.

This is a collective operation - all ranks must call it with the same ranges.
The result has a row partition derived from `A.row_partition` such that each rank
extracts only its local portion (no data communication, only hash computation).

# Example
```julia
A = HPCMatrix(reshape(1.0:12.0, 4, 3))
B = A[2:3, 1:2]  # Returns HPCMatrix submatrix
```
"""
function Base.getindex(A::HPCMatrix{T,B}, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where {T, B<:HPCBackend}
    backend = A.backend
    comm = backend.comm
    device = backend.device
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    m, n = size(A)
    if first(row_rng) < 1 || last(row_rng) > m
        error("HPCMatrix row range out of bounds: $row_rng, nrows=$m")
    end
    if first(col_rng) < 1 || last(col_rng) > n
        error("HPCMatrix column range out of bounds: $col_rng, ncols=$n")
    end

    new_nrows = length(row_rng)
    new_ncols = length(col_rng)

    if isempty(row_rng) || isempty(col_rng)
        # Empty range - return empty HPCMatrix with correct dimensions
        new_row_partition = uniform_partition(new_nrows, nranks)
        new_col_partition = uniform_partition(new_ncols, nranks)
        my_local_rows = new_row_partition[rank + 2] - new_row_partition[rank + 1]
        hash = compute_dense_structural_hash(new_row_partition, new_col_partition, (new_nrows, new_ncols), comm)
        return HPCMatrix{T,B}(hash, new_row_partition, new_col_partition, similar(A.A, my_local_rows, new_ncols), backend)
    end

    # Compute new row partition (local computation, no communication)
    new_row_partition = _compute_subpartition(A.row_partition, row_rng)

    # Compute new column partition (standard even distribution for the submatrix column count)
    new_col_partition = uniform_partition(new_ncols, nranks)

    # Extract local portion
    my_row_start = A.row_partition[rank + 1]
    my_row_end = A.row_partition[rank + 2] - 1

    # Intersection of my row range with row_rng
    intersect_start = max(my_row_start, first(row_rng))
    intersect_end = min(my_row_end, last(row_rng))

    if intersect_start <= intersect_end
        # Convert to local row indices in A.A
        local_row_start = intersect_start - my_row_start + 1
        local_row_end = intersect_end - my_row_start + 1
        local_A = A.A[local_row_start:local_row_end, col_rng]
    else
        local_A = similar(A.A, 0, new_ncols)
    end

    # Compute hash (requires Allgather for consistency)
    hash = compute_dense_structural_hash(new_row_partition, new_col_partition, size(local_A), comm)

    return HPCMatrix{T,B}(hash, new_row_partition, new_col_partition, local_A, backend)
end

# Convenience: A[row_rng, :] - all columns
function Base.getindex(A::HPCMatrix{T}, row_rng::UnitRange{Int}, ::Colon) where T
    return A[row_rng, 1:size(A, 2)]
end

# Convenience: A[:, col_rng] - all rows
function Base.getindex(A::HPCMatrix{T}, ::Colon, col_rng::UnitRange{Int}) where T
    return A[1:size(A, 1), col_rng]
end

# Convenience: A[:, :] - full copy
function Base.getindex(A::HPCMatrix{T}, ::Colon, ::Colon) where T
    return A[1:size(A, 1), 1:size(A, 2)]
end

"""
    Base.getindex(A::HPCMatrix{T}, ::Colon, k::Integer) where T

Extract column k from a distributed dense matrix as a HPCVector.

This is a collective operation - all ranks must call it.
Each rank extracts its local portion of the column.

# Example
```julia
A = HPCMatrix(reshape(1.0:12.0, 4, 3))
v = A[:, 2]  # Get second column as HPCVector
```
"""
function Base.getindex(A::HPCMatrix{T}, ::Colon, k::Integer) where T
    m, n = size(A)
    if k < 1 || k > n
        error("HPCMatrix column index out of bounds: k=$k, ncols=$n")
    end
    # Extract local portion of column k
    local_col = A.A[:, k]
    return HPCVector_local(local_col, A.backend)
end

"""
    Base.setindex!(A::HPCMatrix{T}, val, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where T

Set elements `A[row_rng, col_rng] = val` in a distributed dense matrix.

This is a collective operation - all ranks must call it with the same ranges.
`val` can be a scalar (broadcast to all positions) or a matrix of matching size.

# Example
```julia
A = HPCMatrix(reshape(1.0:12.0, 4, 3))
A[2:3, 1:2] = 0.0  # Set submatrix to zeros
A[2:3, 1:2] = [5.0 6.0; 7.0 8.0]  # Set submatrix to specific values
```
"""
function Base.setindex!(A::HPCMatrix{T,B}, val::Number, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where {T, B<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)

    m, n = size(A)
    if first(row_rng) < 1 || last(row_rng) > m
        error("HPCMatrix row range out of bounds: $row_rng, nrows=$m")
    end
    if first(col_rng) < 1 || last(col_rng) > n
        error("HPCMatrix column range out of bounds: $col_rng, ncols=$n")
    end

    if isempty(row_rng) || isempty(col_rng)
        return val
    end

    # Find intersection with my row portion
    my_row_start = A.row_partition[rank + 1]
    my_row_end = A.row_partition[rank + 2] - 1

    intersect_start = max(my_row_start, first(row_rng))
    intersect_end = min(my_row_end, last(row_rng))

    if intersect_start <= intersect_end
        local_row_start = intersect_start - my_row_start + 1
        local_row_end = intersect_end - my_row_start + 1
        A.A[local_row_start:local_row_end, col_rng] .= convert(T, val)
    end

    return val
end

function Base.setindex!(A::HPCMatrix{T,B}, vals::AbstractMatrix, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where {T, B<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)

    m, n = size(A)
    if first(row_rng) < 1 || last(row_rng) > m
        error("HPCMatrix row range out of bounds: $row_rng, nrows=$m")
    end
    if first(col_rng) < 1 || last(col_rng) > n
        error("HPCMatrix column range out of bounds: $col_rng, ncols=$n")
    end

    if size(vals) != (length(row_rng), length(col_rng))
        error("HPCMatrix setindex!: size mismatch, got $(size(vals)) for range of size ($(length(row_rng)), $(length(col_rng)))")
    end

    if isempty(row_rng) || isempty(col_rng)
        return vals
    end

    # Find intersection with my row portion
    my_row_start = A.row_partition[rank + 1]
    my_row_end = A.row_partition[rank + 2] - 1

    intersect_start = max(my_row_start, first(row_rng))
    intersect_end = min(my_row_end, last(row_rng))

    if intersect_start <= intersect_end
        local_row_start = intersect_start - my_row_start + 1
        local_row_end = intersect_end - my_row_start + 1
        # Indices into vals
        vals_row_start = intersect_start - first(row_rng) + 1
        vals_row_end = intersect_end - first(row_rng) + 1
        A.A[local_row_start:local_row_end, col_rng] .= convert.(T, vals[vals_row_start:vals_row_end, :])
    end

    return vals
end

"""
    Base.setindex!(A::HPCMatrix{T}, src::HPCMatrix, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where T

Set elements `A[row_rng, col_rng] = src` where `src` is a distributed HPCMatrix.

This is a collective operation - all ranks must call it with the same arguments.
Each rank only updates the rows it owns that fall within `row_rng`.

If the source row partition matches the target partition induced by `row_rng`, a direct
local copy is performed. Otherwise, point-to-point communication redistributes the source rows.

# Example
```julia
A = HPCMatrix(zeros(6, 4))
src = HPCMatrix(ones(3, 2))
A[2:4, 1:2] = src  # Each rank only writes to rows it owns
```
"""
function Base.setindex!(A::HPCMatrix{T,B}, src::HPCMatrix, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where {T, B<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    m, n = size(A)
    if first(row_rng) < 1 || last(row_rng) > m
        error("HPCMatrix row range out of bounds: $row_rng, nrows=$m")
    end
    if first(col_rng) < 1 || last(col_rng) > n
        error("HPCMatrix column range out of bounds: $col_rng, ncols=$n")
    end

    if size(src) != (length(row_rng), length(col_rng))
        error("HPCMatrix setindex!: size mismatch, got $(size(src)) for range of size ($(length(row_rng)), $(length(col_rng)))")
    end

    if isempty(row_rng) || isempty(col_rng)
        return src
    end

    ncols_src = length(col_rng)

    # Compute my owned row intersection with row_rng
    my_row_start = A.row_partition[rank + 1]
    my_row_end = A.row_partition[rank + 2] - 1

    intersect_start = max(my_row_start, first(row_rng))
    intersect_end = min(my_row_end, last(row_rng))

    # Compute the partition of the target range
    target_row_partition = _compute_subpartition(A.row_partition, row_rng)

    # Check if source partition matches target partition
    if src.row_partition == target_row_partition
        # Direct local copy - each rank copies its local portion
        if intersect_start <= intersect_end
            local_row_start = intersect_start - my_row_start + 1
            local_row_end = intersect_end - my_row_start + 1
            A.A[local_row_start:local_row_end, col_rng] .= convert.(T, src.A[:, 1:ncols_src])
        end
    else
        # Partitions don't match - need communication to redistribute rows
        # For each row I need (in target_row_partition), find which rank owns it in src

        # Rows I need in global src indexing (1-based in src)
        my_target_start = target_row_partition[rank + 1]
        my_target_end = target_row_partition[rank + 2] - 1
        num_rows_needed = max(0, my_target_end - my_target_start + 1)

        # Build receive plan: which ranks will send me data
        recv_counts = zeros(Int, nranks)
        recv_row_ranges = Vector{UnitRange{Int}}(undef, nranks)  # global src indices from each rank

        for global_src_row in my_target_start:my_target_end
            src_owner = searchsortedlast(src.row_partition, global_src_row) - 1
            if src_owner >= nranks
                src_owner = nranks - 1
            end
            recv_counts[src_owner + 1] += 1
        end

        # Compute contiguous ranges from each rank
        for r in 0:(nranks-1)
            if recv_counts[r + 1] > 0
                # Find the range of global src indices that rank r owns and that I need
                src_r_start = src.row_partition[r + 1]
                src_r_end = src.row_partition[r + 2] - 1
                range_start = max(my_target_start, src_r_start)
                range_end = min(my_target_end, src_r_end)
                recv_row_ranges[r + 1] = range_start:range_end
            else
                recv_row_ranges[r + 1] = 1:0  # empty
            end
        end

        # Build send plan: which ranks need data from me
        send_counts = zeros(Int, nranks)
        send_row_ranges = Vector{UnitRange{Int}}(undef, nranks)

        my_src_start = src.row_partition[rank + 1]
        my_src_end = src.row_partition[rank + 2] - 1

        for r in 0:(nranks-1)
            # Rank r needs rows in target_row_partition[r+1]:target_row_partition[r+2]-1
            r_target_start = target_row_partition[r + 1]
            r_target_end = target_row_partition[r + 2] - 1

            # Intersection with rows I own in src
            range_start = max(r_target_start, my_src_start)
            range_end = min(r_target_end, my_src_end)

            if range_start <= range_end
                send_counts[r + 1] = range_end - range_start + 1
                send_row_ranges[r + 1] = range_start:range_end
            else
                send_row_ranges[r + 1] = 1:0  # empty
            end
        end

        # Post receives
        recv_reqs = MPI.Request[]
        recv_bufs = Dict{Int, Matrix{T}}()
        for r in 0:(nranks-1)
            if recv_counts[r + 1] > 0 && r != rank
                recv_bufs[r] = Matrix{T}(undef, recv_counts[r + 1], ncols_src)
                push!(recv_reqs, comm_irecv!(comm, recv_bufs[r], r, 60))
            end
        end

        # Send data
        send_reqs = MPI.Request[]
        send_bufs = Dict{Int, Matrix{T}}()
        for r in 0:(nranks-1)
            if send_counts[r + 1] > 0 && r != rank
                rng = send_row_ranges[r + 1]
                local_start = first(rng) - my_src_start + 1
                local_end = last(rng) - my_src_start + 1
                send_bufs[r] = convert.(T, src.A[local_start:local_end, 1:ncols_src])
                push!(send_reqs, comm_isend(comm, send_bufs[r], r, 60))
            end
        end

        # Wait for receives
        comm_waitall(comm, recv_reqs)

        # Assemble the aligned local matrix
        if num_rows_needed > 0
            aligned = Matrix{T}(undef, num_rows_needed, ncols_src)

            for r in 0:(nranks-1)
                if recv_counts[r + 1] > 0
                    rng = recv_row_ranges[r + 1]
                    # Destination indices in aligned
                    dst_start = first(rng) - my_target_start + 1
                    dst_end = last(rng) - my_target_start + 1

                    if r == rank
                        # Local copy from src
                        local_start = first(rng) - my_src_start + 1
                        local_end = last(rng) - my_src_start + 1
                        aligned[dst_start:dst_end, :] .= src.A[local_start:local_end, 1:ncols_src]
                    else
                        # From received buffer
                        aligned[dst_start:dst_end, :] .= recv_bufs[r]
                    end
                end
            end

            # Write to A - only rows I own and that fall in row_rng
            if intersect_start <= intersect_end
                local_row_start = intersect_start - my_row_start + 1
                local_row_end = intersect_end - my_row_start + 1
                A.A[local_row_start:local_row_end, col_rng] .= convert.(T, aligned)
            end
        end

        # Wait for sends
        comm_waitall(comm, send_reqs)
    end

    return src
end

# Convenience methods for setindex! with Colon
function Base.setindex!(A::HPCMatrix{T}, val, row_rng::UnitRange{Int}, ::Colon) where T
    return setindex!(A, val, row_rng, 1:size(A, 2))
end

function Base.setindex!(A::HPCMatrix{T}, val, ::Colon, col_rng::UnitRange{Int}) where T
    return setindex!(A, val, 1:size(A, 1), col_rng)
end

# ============================================================================
# Range Indexing for HPCSparseMatrix
# ============================================================================

"""
    Base.getindex(A::HPCSparseMatrix{T}, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where T

Extract a submatrix `A[row_rng, col_rng]` from a distributed sparse matrix, returning a new HPCSparseMatrix.

This is a collective operation - all ranks must call it with the same ranges.
The result has a row partition derived from `A.row_partition` such that each rank
extracts only its local portion (no data communication, only hash computation).

# Example
```julia
A = HPCSparseMatrix{Float64}(sprand(10, 10, 0.3))
B = A[3:7, 2:8]  # Returns HPCSparseMatrix submatrix
```
"""
function Base.getindex(A::HPCSparseMatrix{T,Ti,Bk}, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where {T,Ti,Bk<:HPCBackend}
    backend = A.backend
    comm = backend.comm
    device = backend.device
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    m, n = size(A)
    if first(row_rng) < 1 || last(row_rng) > m
        error("HPCSparseMatrix row range out of bounds: $row_rng, nrows=$m")
    end
    if first(col_rng) < 1 || last(col_rng) > n
        error("HPCSparseMatrix column range out of bounds: $col_rng, ncols=$n")
    end

    new_nrows = length(row_rng)
    new_ncols = length(col_rng)

    if isempty(row_rng) || isempty(col_rng)
        # Empty range - return empty HPCSparseMatrix with correct dimensions
        new_row_partition = uniform_partition(new_nrows, nranks)
        new_col_partition = uniform_partition(new_ncols, nranks)
        my_local_rows = new_row_partition[rank + 2] - new_row_partition[rank + 1]
        # SparseMatrixCSC(ncols, nrows, colptr, rowval, nzval) - transposed storage
        empty_AT = SparseMatrixCSC{T,Ti}(new_ncols, my_local_rows, ones(Ti, my_local_rows + 1), Ti[], T[])
        hash = compute_structural_hash(new_row_partition, Int[], empty_AT, comm)
        # Convert to target backend
        empty_nzval = _values_to_backend(empty_AT.nzval, A.nzval)
        rowptr_target = _to_target_device(empty_AT.colptr, device)
        colval_target = _to_target_device(empty_AT.rowval, device)
        return HPCSparseMatrix{T,Ti,Bk}(hash, new_row_partition, new_col_partition, Int[],
                                   empty_AT.colptr, empty_AT.rowval, empty_nzval,
                                   my_local_rows, 0, nothing, nothing, rowptr_target, colval_target, backend)
    end

    # Compute new row partition (local computation, no communication)
    new_row_partition = _compute_subpartition(A.row_partition, row_rng)

    # Compute new column partition (standard even distribution)
    new_col_partition = uniform_partition(new_ncols, nranks)

    # Extract local portion
    my_row_start = A.row_partition[rank + 1]
    my_row_end = A.row_partition[rank + 2] - 1

    # Intersection of my row range with row_rng
    intersect_start = max(my_row_start, first(row_rng))
    intersect_end = min(my_row_end, last(row_rng))

    if intersect_start <= intersect_end
        # Number of local rows in the result
        local_nrows = intersect_end - intersect_start + 1

        # Local row indices (in A.A) that we're extracting
        local_row_start = intersect_start - my_row_start + 1
        local_row_end = intersect_end - my_row_start + 1

        # A.A is transpose(AT) where AT is CSC with columns = local rows
        AT = _get_csc(A)  # SparseMatrixCSC with columns = local rows

        # Build new sparse structure for the extracted rows
        # AT has columns for each local row, rows indexed by compressed col_indices
        new_colptr = Vector{Ti}(undef, local_nrows + 1)
        new_colptr[1] = 1

        # First pass: count entries per extracted row that fall in col_rng
        # Map col_rng to local col_indices
        col_rng_start = first(col_rng)
        col_rng_end = last(col_rng)

        # Find which A.col_indices fall in col_rng
        col_mask = (A.col_indices .>= col_rng_start) .& (A.col_indices .<= col_rng_end)
        new_col_indices = A.col_indices[col_mask] .- (col_rng_start - 1)  # Shift to 1-based for new matrix

        # Build mapping from old local col index to new local col index
        old_to_new_col = zeros(Int, length(A.col_indices))
        new_idx = 1
        for (old_idx, in_range) in enumerate(col_mask)
            if in_range
                old_to_new_col[old_idx] = new_idx
                new_idx += 1
            end
        end

        # Count entries and collect data
        rowval_list = Ti[]
        nzval_list = T[]

        for local_row in local_row_start:local_row_end
            col_start = AT.colptr[local_row]
            col_end = AT.colptr[local_row + 1] - 1

            count = 0
            for k in col_start:col_end
                old_col_idx = AT.rowval[k]
                new_col_idx = old_to_new_col[old_col_idx]
                if new_col_idx > 0
                    push!(rowval_list, new_col_idx)
                    push!(nzval_list, AT.nzval[k])
                    count += 1
                end
            end
            new_colptr[local_row - local_row_start + 2] = new_colptr[local_row - local_row_start + 1] + count
        end

        # Sort entries within each column by row index
        for local_row in 1:local_nrows
            start_idx = new_colptr[local_row]
            end_idx = new_colptr[local_row + 1] - 1
            if start_idx <= end_idx
                perm = sortperm(view(rowval_list, start_idx:end_idx))
                rowval_list[start_idx:end_idx] = rowval_list[start_idx:end_idx][perm]
                nzval_list[start_idx:end_idx] = nzval_list[start_idx:end_idx][perm]
            end
        end

        # Recompute col_indices to be just the columns that actually appear
        # rowval_list contains positions into new_col_indices
        if !isempty(rowval_list)
            unique_positions = sort(unique(rowval_list))
            # Map positions to compressed indices: unique_positions is sorted, use binary search
            compressed_rowval = Ti[searchsortedfirst(unique_positions, r) for r in rowval_list]
            # final_col_indices maps compressed index to global column in result
            # new_col_indices contains the shifted global column indices
            final_col_indices = new_col_indices[unique_positions]
        else
            compressed_rowval = Ti[]
            final_col_indices = Int[]
        end

        new_AT = SparseMatrixCSC{T,Ti}(length(final_col_indices), local_nrows, new_colptr, compressed_rowval, nzval_list)
    else
        # No local rows in range
        local_nrows = 0
        new_AT = SparseMatrixCSC{T,Ti}(0, 0, Ti[1], Ti[], T[])
        final_col_indices = Int[]
    end

    # Compute hash (requires Allgather for consistency)
    hash = compute_structural_hash(new_row_partition, final_col_indices, new_AT.colptr, new_AT.rowval, comm)

    # Convert to target backend (no-op for CPU, copy for GPU)
    result_nzval = _values_to_backend(new_AT.nzval, A.nzval)
    rowptr_target = _to_target_device(new_AT.colptr, device)
    colval_target = _to_target_device(new_AT.rowval, device)

    return HPCSparseMatrix{T,Ti,Bk}(hash, new_row_partition, new_col_partition, final_col_indices,
                               new_AT.colptr, new_AT.rowval, result_nzval, local_nrows,
                               length(final_col_indices), nothing, nothing, rowptr_target, colval_target, backend)
end

# Convenience: A[row_rng, :] - all columns
function Base.getindex(A::HPCSparseMatrix{T}, row_rng::UnitRange{Int}, ::Colon) where T
    return A[row_rng, 1:size(A, 2)]
end

# Convenience: A[:, col_rng] - all rows
function Base.getindex(A::HPCSparseMatrix{T}, ::Colon, col_rng::UnitRange{Int}) where T
    return A[1:size(A, 1), col_rng]
end

# Convenience: A[:, :] - full copy
function Base.getindex(A::HPCSparseMatrix{T}, ::Colon, ::Colon) where T
    return A[1:size(A, 1), 1:size(A, 2)]
end

"""
    Base.getindex(A::HPCSparseMatrix{T,Ti,AV}, ::Colon, k::Integer) where {T,Ti,AV}

Extract column k from a distributed sparse matrix as a HPCVector.

This is a collective operation - all ranks must call it.
Each rank extracts its local portion of the column.
The returned HPCVector uses the same backend (CPU/GPU) as the input sparse matrix.

# Example
```julia
A = HPCSparseMatrix{Float64}(sprand(10, 5, 0.3))
v = A[:, 2]  # Get second column as HPCVector
```
"""
function Base.getindex(A::HPCSparseMatrix{T,Ti,Bk}, ::Colon, k::Integer) where {T,Ti,Bk<:HPCBackend}
    m, n = size(A)
    if k < 1 || k > n
        error("HPCSparseMatrix column index out of bounds: k=$k, ncols=$n")
    end

    backend = A.backend
    comm = backend.comm
    rank = comm_rank(comm)

    # Get local row range
    my_row_start = A.row_partition[rank + 1]
    my_row_end = A.row_partition[rank + 2] - 1
    local_nrows = my_row_end - my_row_start + 1

    # Check if column k is in our col_indices
    local_col_idx = searchsortedfirst(A.col_indices, k)
    has_col = local_col_idx <= length(A.col_indices) && A.col_indices[local_col_idx] == k

    # Build local portion of column on CPU (uses scalar indexing, then transfer to GPU if needed)
    local_col_cpu = zeros(T, local_nrows)

    if has_col && local_nrows > 0
        # _get_csc(A) is CSC with shape (length(col_indices), local_nrows)
        # Row indices in _get_csc(A).rowval are local column indices
        # Column indices in _get_csc(A) are local row indices
        parent = _get_csc(A)
        for local_row in 1:local_nrows
            # Iterate over nonzeros in this row (stored as column in parent)
            for idx in parent.colptr[local_row]:(parent.colptr[local_row+1]-1)
                if parent.rowval[idx] == local_col_idx
                    local_col_cpu[local_row] = parent.nzval[idx]
                    break
                end
            end
        end
    end

    # Convert to same backend as input sparse matrix
    local_col = _values_to_backend(local_col_cpu, A.nzval)

    return HPCVector_local(local_col, A.backend)
end

"""
    Base.setindex!(A::HPCSparseMatrix{T}, val::Number, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where T

Set elements `A[row_rng, col_rng] = val` in a distributed sparse matrix.

This is a collective operation - all ranks must call it with the same ranges.
Only existing structural nonzeros within the range are modified.
If val is zero, existing entries become explicit zeros (structure is preserved).

# Example
```julia
A = HPCSparseMatrix{Float64}(sprand(10, 10, 0.3))
A[2:4, 3:5] = 0.0  # Set all structural nonzeros in range to zero
A[2:4, 3:5] = 2.0  # Set all structural nonzeros in range to 2.0
```
"""
function Base.setindex!(A::HPCSparseMatrix{T,Ti,Bk}, val::Number, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where {T,Ti,Bk<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)

    m, n = size(A)
    if first(row_rng) < 1 || last(row_rng) > m
        error("HPCSparseMatrix row range out of bounds: $row_rng, nrows=$m")
    end
    if first(col_rng) < 1 || last(col_rng) > n
        error("HPCSparseMatrix column range out of bounds: $col_rng, ncols=$n")
    end

    if isempty(row_rng) || isempty(col_rng)
        return val
    end

    # Find intersection with my row portion
    my_row_start = A.row_partition[rank + 1]
    my_row_end = A.row_partition[rank + 2] - 1

    intersect_start = max(my_row_start, first(row_rng))
    intersect_end = min(my_row_end, last(row_rng))

    if intersect_start <= intersect_end
        local_row_start = intersect_start - my_row_start + 1
        local_row_end = intersect_end - my_row_start + 1

        AT = _get_csc(A)
        col_rng_start = first(col_rng)
        col_rng_end = last(col_rng)

        # Iterate over local rows in the range
        for local_row in local_row_start:local_row_end
            col_start = AT.colptr[local_row]
            col_end = AT.colptr[local_row + 1] - 1

            for k in col_start:col_end
                local_col_idx = AT.rowval[k]
                global_col = A.col_indices[local_col_idx]
                if global_col >= col_rng_start && global_col <= col_rng_end
                    AT.nzval[k] = convert(T, val)
                end
            end
        end
    end

    # Invalidate cached transpose bidirectionally (values changed)
    _invalidate_cached_transpose!(A)

    return val
end

"""
    Base.setindex!(A::HPCSparseMatrix{T}, src::HPCSparseMatrix{T}, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where T

Set elements `A[row_rng, col_rng] = src` where `src` is a distributed HPCSparseMatrix.

This is a collective operation - all ranks must call it with the same arguments.
Each rank only updates the rows it owns that fall within `row_rng`.

The source sparse matrix values replace the corresponding region in A. This is a
structural modification - new nonzeros from src are added, and the sparsity pattern
of A in the target region is replaced by src's pattern.

After the operation:
- col_indices may expand if src contains columns not in A
- The internal CSC storage is rebuilt for affected rows
- structural_hash is recomputed (collective)
- cached_transpose is invalidated
- Plan caches referencing the old hash are cleaned

# Example
```julia
A = HPCSparseMatrix{Float64}(spzeros(6, 6))
src = HPCSparseMatrix{Float64}(sparse([1, 2], [1, 2], [1.0, 2.0], 3, 3))
A[2:4, 1:3] = src  # Each rank only writes to rows it owns
```
"""
function Base.setindex!(A::HPCSparseMatrix{T,Ti,Bk}, src::HPCSparseMatrix{T,Ti,Bk}, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where {T,Ti,Bk<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    m, n = size(A)
    if first(row_rng) < 1 || last(row_rng) > m
        error("HPCSparseMatrix row range out of bounds: $row_rng, nrows=$m")
    end
    if first(col_rng) < 1 || last(col_rng) > n
        error("HPCSparseMatrix column range out of bounds: $col_rng, ncols=$n")
    end

    if size(src) != (length(row_rng), length(col_rng))
        error("HPCSparseMatrix setindex!: size mismatch, got $(size(src)) for range of size ($(length(row_rng)), $(length(col_rng)))")
    end

    if isempty(row_rng) || isempty(col_rng)
        return src
    end

    # Compute my owned row intersection with row_rng
    my_row_start = A.row_partition[rank + 1]
    my_row_end = A.row_partition[rank + 2] - 1

    intersect_start = max(my_row_start, first(row_rng))
    intersect_end = min(my_row_end, last(row_rng))

    # Compute the partition of the target range (how src rows map to A rows)
    target_row_partition = _compute_subpartition(A.row_partition, row_rng)

    # Column offset for translating src column indices to A column indices
    col_offset = first(col_rng) - 1

    # Rows I need in global src indexing (1-based in src)
    my_target_start = target_row_partition[rank + 1]
    my_target_end = target_row_partition[rank + 2] - 1
    num_rows_needed = max(0, my_target_end - my_target_start + 1)

    # Check if partitions match (fast path)
    partitions_match = (src.row_partition == target_row_partition)

    # Build insertions from local or received src data
    insertions = Vector{Tuple{Int,Int,T}}()

    if partitions_match
        # Fast path: direct local extraction
        if num_rows_needed > 0 && intersect_start <= intersect_end
            src_AT = _get_csc(src)
            src_my_start = src.row_partition[rank + 1]

            for src_local_row in 1:num_rows_needed
                src_global_row = src_my_start + src_local_row - 1
                A_global_row = first(row_rng) + src_global_row - 1

                # Only process if this A row is owned by us
                if A_global_row >= intersect_start && A_global_row <= intersect_end
                    # Extract entries from src for this row
                    for k in src_AT.colptr[src_local_row]:(src_AT.colptr[src_local_row+1]-1)
                        src_local_col = src_AT.rowval[k]
                        src_global_col = src.col_indices[src_local_col]
                        A_global_col = src_global_col + col_offset
                        if A_global_col >= first(col_rng) && A_global_col <= last(col_rng)
                            push!(insertions, (A_global_row, A_global_col, src_AT.nzval[k]))
                        end
                    end
                end
            end
        end
    else
        # Slow path: need communication
        # Build receive plan
        recv_counts = zeros(Int, nranks)
        for global_src_row in my_target_start:my_target_end
            src_owner = searchsortedlast(src.row_partition, global_src_row) - 1
            if src_owner >= nranks
                src_owner = nranks - 1
            end
            recv_counts[src_owner + 1] += 1
        end

        # Compute contiguous row ranges from each rank
        recv_row_ranges = Vector{UnitRange{Int}}(undef, nranks)
        for r in 0:(nranks-1)
            if recv_counts[r + 1] > 0
                src_r_start = src.row_partition[r + 1]
                src_r_end = src.row_partition[r + 2] - 1
                range_start = max(my_target_start, src_r_start)
                range_end = min(my_target_end, src_r_end)
                recv_row_ranges[r + 1] = range_start:range_end
            else
                recv_row_ranges[r + 1] = 1:0
            end
        end

        # Build send plan
        my_src_start = src.row_partition[rank + 1]
        my_src_end = src.row_partition[rank + 2] - 1

        send_row_ranges = Vector{UnitRange{Int}}(undef, nranks)
        for r in 0:(nranks-1)
            r_target_start = target_row_partition[r + 1]
            r_target_end = target_row_partition[r + 2] - 1
            range_start = max(r_target_start, my_src_start)
            range_end = min(r_target_end, my_src_end)
            if range_start <= range_end
                send_row_ranges[r + 1] = range_start:range_end
            else
                send_row_ranges[r + 1] = 1:0
            end
        end

        # Exchange sparse data: for each row, send (num_entries, col_indices, values)
        # First exchange the number of nonzeros per row
        send_nnz_per_row = Dict{Int, Vector{Int}}()
        send_col_indices = Dict{Int, Vector{Int}}()
        send_values = Dict{Int, Vector{T}}()

        src_AT = _get_csc(src)

        for r in 0:(nranks-1)
            rng = send_row_ranges[r + 1]
            if !isempty(rng) && r != rank
                nnz_list = Int[]
                cols_list = Int[]
                vals_list = T[]
                for src_global_row in rng
                    src_local_row = src_global_row - my_src_start + 1
                    k_start = src_AT.colptr[src_local_row]
                    k_end = src_AT.colptr[src_local_row + 1] - 1
                    row_nnz = 0
                    for k in k_start:k_end
                        src_local_col = src_AT.rowval[k]
                        src_global_col = src.col_indices[src_local_col]
                        A_global_col = src_global_col + col_offset
                        if A_global_col >= first(col_rng) && A_global_col <= last(col_rng)
                            push!(cols_list, A_global_col)
                            push!(vals_list, src_AT.nzval[k])
                            row_nnz += 1
                        end
                    end
                    push!(nnz_list, row_nnz)
                end
                send_nnz_per_row[r] = nnz_list
                send_col_indices[r] = cols_list
                send_values[r] = vals_list
            end
        end

        # Post receives for nnz counts
        recv_reqs_nnz = MPI.Request[]
        recv_nnz_per_row = Dict{Int, Vector{Int}}()
        for r in 0:(nranks-1)
            if recv_counts[r + 1] > 0 && r != rank
                recv_nnz_per_row[r] = Vector{Int}(undef, recv_counts[r + 1])
                push!(recv_reqs_nnz, comm_irecv!(comm, recv_nnz_per_row[r], r, 70))
            end
        end

        # Send nnz counts
        send_reqs_nnz = MPI.Request[]
        for r in 0:(nranks-1)
            if haskey(send_nnz_per_row, r)
                push!(send_reqs_nnz, comm_isend(comm, send_nnz_per_row[r], r, 70))
            end
        end

        comm_waitall(comm, recv_reqs_nnz)

        # Post receives for col indices and values
        recv_reqs_data = MPI.Request[]
        recv_col_indices = Dict{Int, Vector{Int}}()
        recv_values = Dict{Int, Vector{T}}()
        for r in 0:(nranks-1)
            if recv_counts[r + 1] > 0 && r != rank
                total_nnz = sum(recv_nnz_per_row[r])
                if total_nnz > 0
                    recv_col_indices[r] = Vector{Int}(undef, total_nnz)
                    recv_values[r] = Vector{T}(undef, total_nnz)
                    push!(recv_reqs_data, comm_irecv!(comm, recv_col_indices[r], r, 71))
                    push!(recv_reqs_data, comm_irecv!(comm, recv_values[r], r, 72))
                end
            end
        end

        # Send col indices and values
        send_reqs_data = MPI.Request[]
        for r in 0:(nranks-1)
            if haskey(send_col_indices, r) && !isempty(send_col_indices[r])
                push!(send_reqs_data, comm_isend(comm, send_col_indices[r], r, 71))
                push!(send_reqs_data, comm_isend(comm, send_values[r], r, 72))
            end
        end

        comm_waitall(comm, send_reqs_nnz)
        comm_waitall(comm, recv_reqs_data)

        # Build insertions from received data and local data
        for r in 0:(nranks-1)
            if recv_counts[r + 1] > 0
                rng = recv_row_ranges[r + 1]
                if r == rank
                    # Local data
                    for src_global_row in rng
                        A_global_row = first(row_rng) + src_global_row - 1
                        if A_global_row >= intersect_start && A_global_row <= intersect_end
                            src_local_row = src_global_row - my_src_start + 1
                            for k in src_AT.colptr[src_local_row]:(src_AT.colptr[src_local_row+1]-1)
                                src_local_col = src_AT.rowval[k]
                                src_global_col = src.col_indices[src_local_col]
                                A_global_col = src_global_col + col_offset
                                if A_global_col >= first(col_rng) && A_global_col <= last(col_rng)
                                    push!(insertions, (A_global_row, A_global_col, src_AT.nzval[k]))
                                end
                            end
                        end
                    end
                elseif haskey(recv_nnz_per_row, r)
                    # Received data
                    nnz_list = recv_nnz_per_row[r]
                    if haskey(recv_col_indices, r)
                        cols = recv_col_indices[r]
                        vals = recv_values[r]
                        data_idx = 1
                        for (row_idx, src_global_row) in enumerate(rng)
                            A_global_row = first(row_rng) + src_global_row - 1
                            row_nnz = nnz_list[row_idx]
                            if A_global_row >= intersect_start && A_global_row <= intersect_end
                                for _ in 1:row_nnz
                                    push!(insertions, (A_global_row, cols[data_idx], vals[data_idx]))
                                    data_idx += 1
                                end
                            else
                                data_idx += row_nnz
                            end
                        end
                    end
                end
            end
        end

        comm_waitall(comm, send_reqs_data)
    end

    # First, zero out existing entries in the target region (within owned rows)
    # This ensures src's sparsity pattern replaces the old one
    if intersect_start <= intersect_end
        AT = _get_csc(A)
        local_row_start = intersect_start - my_row_start + 1
        local_row_end = intersect_end - my_row_start + 1
        col_rng_start = first(col_rng)
        col_rng_end = last(col_rng)

        for local_row in local_row_start:local_row_end
            for k in AT.colptr[local_row]:(AT.colptr[local_row+1]-1)
                local_col_idx = AT.rowval[k]
                if local_col_idx <= length(A.col_indices)
                    global_col = A.col_indices[local_col_idx]
                    if global_col >= col_rng_start && global_col <= col_rng_end
                        AT.nzval[k] = zero(T)
                    end
                end
            end
        end
    end

    # Apply insertions using the helper function
    if !isempty(insertions)
        row_offset = A.row_partition[rank + 1]
        new_AT, new_col_indices = _rebuild_AT_with_insertions(
            _get_csc(A), A.col_indices, insertions, row_offset
        )
        A.col_indices = new_col_indices
        # Update explicit CSR arrays from the new CSC
        A.rowptr = new_AT.colptr
        A.colval = new_AT.rowval
        A.nzval = new_AT.nzval
        A.ncols_compressed = length(new_col_indices)
    end

    # Recompute structural hash after modification
    A.structural_hash = compute_structural_hash(A.row_partition, A.col_indices, A.rowptr, A.colval, A.backend.comm)

    # Invalidate cached transpose bidirectionally
    _invalidate_cached_transpose!(A)

    return src
end

# Convenience methods for setindex! with Colon
function Base.setindex!(A::HPCSparseMatrix{T}, val, row_rng::UnitRange{Int}, ::Colon) where T
    return setindex!(A, val, row_rng, 1:size(A, 2))
end

function Base.setindex!(A::HPCSparseMatrix{T}, val, ::Colon, col_rng::UnitRange{Int}) where T
    return setindex!(A, val, 1:size(A, 1), col_rng)
end

function Base.setindex!(A::HPCSparseMatrix{T}, val, ::Colon, ::Colon) where T
    return setindex!(A, val, 1:size(A, 1), 1:size(A, 2))
end

# Also add full colon setindex! for HPCMatrix
function Base.setindex!(A::HPCMatrix{T}, val, ::Colon, ::Colon) where T
    return setindex!(A, val, 1:size(A, 1), 1:size(A, 2))
end

# ============================================================================
# HPCVector Indexing with HPCVector indices
# ============================================================================

"""
    Base.getindex(v::HPCVector{T}, idx::HPCVector{Int}) where T

Extract elements `v[idx]` where `idx` is a distributed HPCVector of integer indices.

This is a collective operation - all ranks must call it with the same `idx`.
Each rank requests the values at its local indices `idx.v`, which may be owned
by different ranks in `v`. Communication is used to gather the requested values.

The result is a HPCVector with the same partition as `idx`.

# Example
```julia
v = HPCVector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
idx = HPCVector([3, 1, 5, 2])
result = v[idx]  # Returns HPCVector with values [3.0, 1.0, 5.0, 2.0]
```
"""
function Base.getindex(v::HPCVector{T,B}, idx::HPCVector{Int}) where {T,B<:HPCBackend}
    comm = v.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    n = length(v)

    # Validate indices (local check - each rank checks its own indices)
    for i in idx.v
        if i < 1 || i > n
            error("HPCVector index out of bounds: $i, length=$n")
        end
    end

    # My local indices into v (global indices)
    local_idx = idx.v
    n_local = length(local_idx)

    # Group local indices by which rank owns them in v
    needed_from = [Tuple{Int,Int}[] for _ in 1:nranks]  # (global_v_idx, local_result_idx)
    for (result_idx, v_global_idx) in enumerate(local_idx)
        owner = searchsortedlast(v.partition, v_global_idx) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(needed_from[owner + 1], (v_global_idx, result_idx))
    end

    # Exchange counts via Alltoall
    send_counts = Int32[length(needed_from[r + 1]) for r in 0:(nranks-1)]
    recv_counts = comm_alltoall(comm, MPI.UBuffer(send_counts, 1))

    # Send requested indices to each owner rank
    struct_send_reqs = MPI.Request[]
    struct_send_bufs = Dict{Int, Vector{Int}}()
    recv_perm_map = Dict{Int, Vector{Int}}()  # Maps rank -> destination indices in result

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            indices = [t[1] for t in needed_from[r + 1]]
            dst_indices = [t[2] for t in needed_from[r + 1]]
            recv_perm_map[r] = dst_indices
            struct_send_bufs[r] = indices
            push!(struct_send_reqs, comm_isend(comm, indices, r, 80))
        end
    end

    # Receive index requests from other ranks
    struct_recv_bufs = Dict{Int, Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(struct_recv_reqs, comm_irecv!(comm, buf, r, 80))
            struct_recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, struct_recv_reqs)
    comm_waitall(comm, struct_send_reqs)

    # Prepare to send values
    my_v_start = v.partition[rank + 1]

    # Post receives for values
    recv_reqs = MPI.Request[]
    recv_bufs = Dict{Int, Vector{T}}()
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            recv_bufs[r] = Vector{T}(undef, send_counts[r + 1])
            push!(recv_reqs, comm_irecv!(comm, recv_bufs[r], r, 81))
        end
    end

    # Send values to requesters
    send_reqs = MPI.Request[]
    send_bufs = Dict{Int, Vector{T}}()
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            global_indices = struct_recv_bufs[r]
            vals = Vector{T}(undef, length(global_indices))
            for (k, g_idx) in enumerate(global_indices)
                local_idx_in_v = g_idx - my_v_start + 1
                vals[k] = v.v[local_idx_in_v]
            end
            send_bufs[r] = vals
            push!(send_reqs, comm_isend(comm, vals, r, 81))
        end
    end

    comm_waitall(comm, recv_reqs)

    # Assemble result
    result_v = Vector{T}(undef, n_local)

    # Fill from local data (indices I own in v)
    for (v_global_idx, result_idx) in needed_from[rank + 1]
        local_idx_in_v = v_global_idx - my_v_start + 1
        result_v[result_idx] = v.v[local_idx_in_v]
    end

    # Fill from received data
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            dst_indices = recv_perm_map[r]
            vals = recv_bufs[r]
            for (k, dst_idx) in enumerate(dst_indices)
                result_v[dst_idx] = vals[k]
            end
        end
    end

    comm_waitall(comm, send_reqs)

    # Result has same partition as idx, so same hash
    # Convert result to target device if needed
    result_v_backend = _convert_array(result_v, v.backend.device)
    return HPCVector{T,B}(idx.structural_hash, idx.partition, result_v_backend, v.backend)
end

# ============================================================================
# HPCMatrix Indexing with HPCVector indices
# ============================================================================

"""
    Base.getindex(A::HPCMatrix{T}, row_idx::HPCVector{Int}, col_idx::HPCVector{Int}) where T

Extract a submatrix `A[row_idx, col_idx]` where indices are distributed HPCVector{Int}.

This is a collective operation - all ranks must call it with the same `row_idx` and `col_idx`.
The result is a HPCMatrix of size `(length(row_idx), length(col_idx))`.

Each rank computes its local portion of the result matrix based on `row_idx`'s partition.
Communication is used to gather row data from ranks that own the requested rows.

# Example
```julia
A = HPCMatrix(reshape(1.0:12.0, 4, 3))
row_idx = HPCVector([2, 4, 1])
col_idx = HPCVector([3, 1])
result = A[row_idx, col_idx]  # Returns HPCMatrix submatrix
```
"""
function Base.getindex(A::HPCMatrix{T,B}, row_idx::HPCVector{Int}, col_idx::HPCVector{Int}) where {T,B<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    m, n = size(A)

    # Gather col_idx to all ranks (columns are not distributed, all ranks need all column indices)
    col_indices = _gather_vector_to_all(col_idx, comm)

    # Validate column indices
    for j in col_indices
        if j < 1 || j > n
            error("HPCMatrix column index out of bounds: $j, ncols=$n")
        end
    end

    ncols_result = length(col_indices)
    nrows_result = length(row_idx)

    # Result partition follows row_idx partition
    result_row_partition = row_idx.partition

    # My local row indices (global indices into A)
    my_row_indices = row_idx.v
    n_local_rows = length(my_row_indices)

    # Validate row indices
    for i in my_row_indices
        if i < 1 || i > m
            error("HPCMatrix row index out of bounds: $i, nrows=$m")
        end
    end

    # Group local row indices by which rank owns them in A
    needed_from = [Tuple{Int,Int}[] for _ in 1:nranks]  # (global_A_row, local_result_row)
    for (result_row, A_global_row) in enumerate(my_row_indices)
        owner = searchsortedlast(A.row_partition, A_global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(needed_from[owner + 1], (A_global_row, result_row))
    end

    # Exchange counts via Alltoall
    send_counts = Int32[length(needed_from[r + 1]) for r in 0:(nranks-1)]
    recv_counts = comm_alltoall(comm, MPI.UBuffer(send_counts, 1))

    # Send requested row indices to each owner rank
    struct_send_reqs = MPI.Request[]
    struct_send_bufs = Dict{Int, Vector{Int}}()
    recv_perm_map = Dict{Int, Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            indices = [t[1] for t in needed_from[r + 1]]
            dst_indices = [t[2] for t in needed_from[r + 1]]
            recv_perm_map[r] = dst_indices
            struct_send_bufs[r] = indices
            push!(struct_send_reqs, comm_isend(comm, indices, r, 82))
        end
    end

    # Receive index requests from other ranks
    struct_recv_bufs = Dict{Int, Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(struct_recv_reqs, comm_irecv!(comm, buf, r, 82))
            struct_recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, struct_recv_reqs)
    comm_waitall(comm, struct_send_reqs)

    # Prepare to send row data
    my_A_row_start = A.row_partition[rank + 1]

    # Post receives for row data (each row has ncols_result elements)
    recv_reqs = MPI.Request[]
    recv_bufs = Dict{Int, Matrix{T}}()
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            recv_bufs[r] = Matrix{T}(undef, send_counts[r + 1], ncols_result)
            push!(recv_reqs, comm_irecv!(comm, recv_bufs[r], r, 83))
        end
    end

    # Send row data to requesters
    send_reqs = MPI.Request[]
    send_bufs = Dict{Int, Matrix{T}}()
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            global_row_indices = struct_recv_bufs[r]
            nrows_to_send = length(global_row_indices)
            data = Matrix{T}(undef, nrows_to_send, ncols_result)
            for (k, g_row) in enumerate(global_row_indices)
                local_row_in_A = g_row - my_A_row_start + 1
                for (c, g_col) in enumerate(col_indices)
                    data[k, c] = A.A[local_row_in_A, g_col]
                end
            end
            send_bufs[r] = data
            push!(send_reqs, comm_isend(comm, data, r, 83))
        end
    end

    comm_waitall(comm, recv_reqs)

    # Assemble result
    result_A = Matrix{T}(undef, n_local_rows, ncols_result)

    # Fill from local data (rows I own in A)
    for (A_global_row, result_row) in needed_from[rank + 1]
        local_row_in_A = A_global_row - my_A_row_start + 1
        for (c, g_col) in enumerate(col_indices)
            result_A[result_row, c] = A.A[local_row_in_A, g_col]
        end
    end

    # Fill from received data
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            dst_indices = recv_perm_map[r]
            data = recv_bufs[r]
            for (k, dst_row) in enumerate(dst_indices)
                result_A[dst_row, :] .= data[k, :]
            end
        end
    end

    comm_waitall(comm, send_reqs)

    # Compute column partition for result
    result_col_partition = uniform_partition(ncols_result, nranks)

    # Compute hash for result
    hash = compute_dense_structural_hash(result_row_partition, result_col_partition, size(result_A), comm)

    # Convert result to target device if needed (no-op for CPU, copies to GPU for GPU backends)
    result_A_backend = _convert_array(result_A, A.backend.device)
    return HPCMatrix{T,B}(hash, result_row_partition, result_col_partition, result_A_backend, A.backend)
end

# ============================================================================
# HPCSparseMatrix Indexing with HPCVector indices
# ============================================================================

"""
    Base.getindex(A::HPCSparseMatrix{T}, row_idx::HPCVector{Int}, col_idx::HPCVector{Int}) where T

Extract a submatrix `A[row_idx, col_idx]` where indices are distributed HPCVector{Int}.

This is a collective operation - all ranks must call it with the same `row_idx` and `col_idx`.
The result is a HPCMatrix (dense) of size `(length(row_idx), length(col_idx))`.

Note: The result is dense because arbitrary indexing typically doesn't preserve
useful sparsity patterns. For large sparse matrices with structured indexing,
consider using range-based indexing instead.

# Example
```julia
A = HPCSparseMatrix{Float64}(sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 4, 4))
row_idx = HPCVector([2, 4, 1])
col_idx = HPCVector([3, 1])
result = A[row_idx, col_idx]  # Returns HPCMatrix (dense) submatrix
```
"""
function Base.getindex(A::HPCSparseMatrix{T,Ti,Bk}, row_idx::HPCVector{Int}, col_idx::HPCVector{Int}) where {T,Ti,Bk<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    m, n = size(A)

    # Gather col_idx to all ranks (columns are not distributed)
    col_indices = _gather_vector_to_all(col_idx, comm)

    # Validate column indices
    for j in col_indices
        if j < 1 || j > n
            error("HPCSparseMatrix column index out of bounds: $j, ncols=$n")
        end
    end

    ncols_result = length(col_indices)
    nrows_result = length(row_idx)

    # Build column lookup: global_col -> result_col_position
    col_to_result = Dict{Int, Int}()
    for (pos, col) in enumerate(col_indices)
        col_to_result[col] = pos
    end

    # Result partition follows row_idx partition
    result_row_partition = row_idx.partition

    # My local row indices (global indices into A)
    my_row_indices = row_idx.v
    n_local_rows = length(my_row_indices)

    # Validate row indices
    for i in my_row_indices
        if i < 1 || i > m
            error("HPCSparseMatrix row index out of bounds: $i, nrows=$m")
        end
    end

    # Group local row indices by which rank owns them in A
    needed_from = [Tuple{Int,Int}[] for _ in 1:nranks]
    for (result_row, A_global_row) in enumerate(my_row_indices)
        owner = searchsortedlast(A.row_partition, A_global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(needed_from[owner + 1], (A_global_row, result_row))
    end

    # Exchange counts via Alltoall
    send_counts = Int32[length(needed_from[r + 1]) for r in 0:(nranks-1)]
    recv_counts = comm_alltoall(comm, MPI.UBuffer(send_counts, 1))

    # Send requested row indices to each owner rank
    struct_send_reqs = MPI.Request[]
    struct_send_bufs = Dict{Int, Vector{Int}}()
    recv_perm_map = Dict{Int, Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            indices = [t[1] for t in needed_from[r + 1]]
            dst_indices = [t[2] for t in needed_from[r + 1]]
            recv_perm_map[r] = dst_indices
            struct_send_bufs[r] = indices
            push!(struct_send_reqs, comm_isend(comm, indices, r, 84))
        end
    end

    # Receive index requests from other ranks
    struct_recv_bufs = Dict{Int, Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(struct_recv_reqs, comm_irecv!(comm, buf, r, 84))
            struct_recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, struct_recv_reqs)
    comm_waitall(comm, struct_send_reqs)

    # Prepare to send row data
    my_A_row_start = A.row_partition[rank + 1]
    AT = _get_csc(A)

    # Helper function to extract a row from sparse matrix
    function extract_sparse_row(local_row::Int, cols::Vector{Int}, col_lookup::Dict{Int,Int})
        row_data = zeros(T, length(cols))
        for k in AT.colptr[local_row]:(AT.colptr[local_row+1]-1)
            local_col_idx = AT.rowval[k]
            if local_col_idx <= length(A.col_indices)
                global_col = A.col_indices[local_col_idx]
                if haskey(col_lookup, global_col)
                    result_col = col_lookup[global_col]
                    row_data[result_col] = AT.nzval[k]
                end
            end
        end
        return row_data
    end

    # Post receives for row data
    recv_reqs = MPI.Request[]
    recv_bufs = Dict{Int, Matrix{T}}()
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            recv_bufs[r] = Matrix{T}(undef, send_counts[r + 1], ncols_result)
            push!(recv_reqs, comm_irecv!(comm, recv_bufs[r], r, 85))
        end
    end

    # Send row data to requesters
    send_reqs = MPI.Request[]
    send_bufs = Dict{Int, Matrix{T}}()
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            global_row_indices = struct_recv_bufs[r]
            nrows_to_send = length(global_row_indices)
            data = Matrix{T}(undef, nrows_to_send, ncols_result)
            for (k, g_row) in enumerate(global_row_indices)
                local_row_in_A = g_row - my_A_row_start + 1
                data[k, :] .= extract_sparse_row(local_row_in_A, col_indices, col_to_result)
            end
            send_bufs[r] = data
            push!(send_reqs, comm_isend(comm, data, r, 85))
        end
    end

    comm_waitall(comm, recv_reqs)

    # Assemble result
    result_A = zeros(T, n_local_rows, ncols_result)

    # Fill from local data (rows I own in A)
    for (A_global_row, result_row) in needed_from[rank + 1]
        local_row_in_A = A_global_row - my_A_row_start + 1
        result_A[result_row, :] .= extract_sparse_row(local_row_in_A, col_indices, col_to_result)
    end

    # Fill from received data
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            dst_indices = recv_perm_map[r]
            data = recv_bufs[r]
            for (k, dst_row) in enumerate(dst_indices)
                result_A[dst_row, :] .= data[k, :]
            end
        end
    end

    comm_waitall(comm, send_reqs)

    # Compute column partition for result
    result_col_partition = uniform_partition(ncols_result, nranks)

    # Compute hash for result
    hash = compute_dense_structural_hash(result_row_partition, result_col_partition, size(result_A), comm)

    # Convert result to target device if needed (no-op for CPU, copies to GPU for GPU backends)
    result_A_backend = _convert_array(result_A, A.backend.device)
    return HPCMatrix{T,Bk}(hash, result_row_partition, result_col_partition, result_A_backend, A.backend)
end

# Helper function to gather a HPCVector to all ranks (generic version for any element type)
function _gather_vector_to_all(v::HPCVector{T}, comm::AbstractComm) where T
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    # Gather counts
    local_count = Int32(length(v.v))
    counts = Vector{Int32}(undef, nranks)
    comm_allgather!(comm, Ref(local_count), MPI.UBuffer(counts, 1))

    # Compute displacements
    displs = Vector{Int32}(undef, nranks)
    displs[1] = 0
    for i in 2:nranks
        displs[i] = displs[i-1] + counts[i-1]
    end

    total = sum(counts)
    result = Vector{T}(undef, total)

    # Allgatherv
    comm_allgatherv!(comm, v.v, MPI.VBuffer(result, counts, displs))

    return result
end

# ============================================================================
# HPCVector setindex! with HPCVector indices
# ============================================================================

"""
    Base.setindex!(v::HPCVector{T}, src::HPCVector{T}, idx::HPCVector{Int}) where T

Set elements `v[idx] = src` where `idx` is a distributed HPCVector of integer indices
and `src` is a distributed HPCVector of values.

This is a collective operation - all ranks must call it with the same `idx` and `src`.
The `src` and `idx` must have the same partition (same length and distribution).
Each `src[k]` is assigned to `v[idx[k]]`.

Communication is used to send values from the ranks that own them in `src` to the
ranks that own the destination positions in `v`.

# Example
```julia
v = HPCVector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
idx = HPCVector([3, 1, 5, 2])
src = HPCVector([30.0, 10.0, 50.0, 20.0])
v[idx] = src  # Sets v[3]=30, v[1]=10, v[5]=50, v[2]=20
```
"""
function Base.setindex!(v::HPCVector{T,B}, src::HPCVector{T,B}, idx::HPCVector{Int}) where {T,B<:HPCBackend}
    comm = v.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    n = length(v)

    # Validate that src and idx have the same partition
    if src.partition != idx.partition
        error("HPCVector setindex!: src and idx must have the same partition")
    end

    # Validate indices (local check)
    for i in idx.v
        if i < 1 || i > n
            error("HPCVector index out of bounds: $i, length=$n")
        end
    end

    # My local indices and values
    local_idx = idx.v
    local_src = src.v
    n_local = length(local_idx)

    # Group (index, value) pairs by which rank owns the destination in v
    send_to = [Tuple{Int,T}[] for _ in 1:nranks]  # (global_v_idx, value)
    for k in 1:n_local
        v_global_idx = local_idx[k]
        value = local_src[k]
        owner = searchsortedlast(v.partition, v_global_idx) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(send_to[owner + 1], (v_global_idx, value))
    end

    # Exchange counts via Alltoall
    send_counts = Int32[length(send_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = comm_alltoall(comm, MPI.UBuffer(send_counts, 1))

    # Send indices to destination ranks
    struct_send_reqs = MPI.Request[]
    struct_send_bufs = Dict{Int, Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            indices = [t[1] for t in send_to[r + 1]]
            struct_send_bufs[r] = indices
            push!(struct_send_reqs, comm_isend(comm, indices, r, 90))
        end
    end

    # Receive indices from source ranks
    struct_recv_bufs = Dict{Int, Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(struct_recv_reqs, comm_irecv!(comm, buf, r, 90))
            struct_recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, struct_recv_reqs)
    comm_waitall(comm, struct_send_reqs)

    # Send values to destination ranks
    send_reqs = MPI.Request[]
    send_bufs = Dict{Int, Vector{T}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            values = [t[2] for t in send_to[r + 1]]
            send_bufs[r] = values
            push!(send_reqs, comm_isend(comm, values, r, 91))
        end
    end

    # Receive values from source ranks
    recv_reqs = MPI.Request[]
    recv_bufs = Dict{Int, Vector{T}}()

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{T}(undef, recv_counts[r + 1])
            push!(recv_reqs, comm_irecv!(comm, buf, r, 91))
            recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, recv_reqs)

    # Apply local assignments (from my own send_to[rank+1])
    my_v_start = v.partition[rank + 1]
    for (v_global_idx, value) in send_to[rank + 1]
        local_idx_in_v = v_global_idx - my_v_start + 1
        v.v[local_idx_in_v] = value
    end

    # Apply received assignments
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = struct_recv_bufs[r]
            values = recv_bufs[r]
            for (k, v_global_idx) in enumerate(indices)
                local_idx_in_v = v_global_idx - my_v_start + 1
                v.v[local_idx_in_v] = values[k]
            end
        end
    end

    comm_waitall(comm, send_reqs)

    return src
end

# ============================================================================
# HPCMatrix setindex! with HPCVector indices
# ============================================================================

"""
    Base.setindex!(A::HPCMatrix{T}, src::HPCMatrix{T}, row_idx::HPCVector{Int}, col_idx::HPCVector{Int}) where T

Set elements `A[row_idx, col_idx] = src` where indices are distributed HPCVector{Int}
and `src` is a distributed HPCMatrix of values.

This is a collective operation - all ranks must call it with the same arguments.
The `src` must have size `(length(row_idx), length(col_idx))` and its row partition
must match `row_idx`'s partition.

Each `src[i, j]` is assigned to `A[row_idx[i], col_idx[j]]`.

# Example
```julia
A = HPCMatrix(zeros(6, 4))
row_idx = HPCVector([2, 4, 1])
col_idx = HPCVector([3, 1])
src = HPCMatrix(ones(3, 2))
A[row_idx, col_idx] = src  # Sets A[2,3]=1, A[2,1]=1, A[4,3]=1, etc.
```
"""
function Base.setindex!(A::HPCMatrix{T,B}, src::HPCMatrix{T,B}, row_idx::HPCVector{Int}, col_idx::HPCVector{Int}) where {T,B<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    m, n = size(A)

    # Gather col_idx to all ranks
    col_indices = _gather_vector_to_all(col_idx, comm)
    ncols_src = length(col_indices)

    # Validate dimensions
    if size(src) != (length(row_idx), ncols_src)
        error("HPCMatrix setindex!: src size $(size(src)) doesn't match index dimensions ($(length(row_idx)), $ncols_src)")
    end

    # Validate that src row partition matches row_idx partition
    if src.row_partition != row_idx.partition
        error("HPCMatrix setindex!: src row partition must match row_idx partition")
    end

    # Validate column indices
    for j in col_indices
        if j < 1 || j > n
            error("HPCMatrix column index out of bounds: $j, ncols=$n")
        end
    end

    # Validate row indices (local check)
    for i in row_idx.v
        if i < 1 || i > m
            error("HPCMatrix row index out of bounds: $i, nrows=$m")
        end
    end

    # My local row indices and corresponding src rows
    local_row_idx = row_idx.v
    n_local_rows = length(local_row_idx)

    # Group rows by which rank owns the destination in A
    # Each entry: (global_A_row, local_src_row)
    send_to = [Tuple{Int,Int}[] for _ in 1:nranks]
    for local_src_row in 1:n_local_rows
        A_global_row = local_row_idx[local_src_row]
        owner = searchsortedlast(A.row_partition, A_global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(send_to[owner + 1], (A_global_row, local_src_row))
    end

    # Exchange counts via Alltoall
    send_counts = Int32[length(send_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = comm_alltoall(comm, MPI.UBuffer(send_counts, 1))

    # Send row indices to destination ranks
    struct_send_reqs = MPI.Request[]
    struct_send_bufs = Dict{Int, Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            indices = [t[1] for t in send_to[r + 1]]
            struct_send_bufs[r] = indices
            push!(struct_send_reqs, comm_isend(comm, indices, r, 92))
        end
    end

    # Receive row indices from source ranks
    struct_recv_bufs = Dict{Int, Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(struct_recv_reqs, comm_irecv!(comm, buf, r, 92))
            struct_recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, struct_recv_reqs)
    comm_waitall(comm, struct_send_reqs)

    # Send row data to destination ranks
    send_reqs = MPI.Request[]
    send_bufs = Dict{Int, Matrix{T}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            nrows_to_send = send_counts[r + 1]
            data = Matrix{T}(undef, nrows_to_send, ncols_src)
            for (k, (_, local_src_row)) in enumerate(send_to[r + 1])
                data[k, :] .= src.A[local_src_row, 1:ncols_src]
            end
            send_bufs[r] = data
            push!(send_reqs, comm_isend(comm, data, r, 93))
        end
    end

    # Receive row data from source ranks
    recv_reqs = MPI.Request[]
    recv_bufs = Dict{Int, Matrix{T}}()

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Matrix{T}(undef, recv_counts[r + 1], ncols_src)
            push!(recv_reqs, comm_irecv!(comm, buf, r, 93))
            recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, recv_reqs)

    # Apply local assignments
    my_A_row_start = A.row_partition[rank + 1]
    for (A_global_row, local_src_row) in send_to[rank + 1]
        local_row_in_A = A_global_row - my_A_row_start + 1
        for (c, g_col) in enumerate(col_indices)
            A.A[local_row_in_A, g_col] = src.A[local_src_row, c]
        end
    end

    # Apply received assignments
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = struct_recv_bufs[r]
            data = recv_bufs[r]
            for (k, A_global_row) in enumerate(indices)
                local_row_in_A = A_global_row - my_A_row_start + 1
                for (c, g_col) in enumerate(col_indices)
                    A.A[local_row_in_A, g_col] = data[k, c]
                end
            end
        end
    end

    comm_waitall(comm, send_reqs)

    return src
end

# ============================================================================
# HPCSparseMatrix setindex! with HPCVector indices
# ============================================================================

"""
    Base.setindex!(A::HPCSparseMatrix{T}, src::HPCMatrix{T}, row_idx::HPCVector{Int}, col_idx::HPCVector{Int}) where T

Set elements `A[row_idx, col_idx] = src` where indices are distributed HPCVector{Int}
and `src` is a distributed HPCMatrix of values.

This is a collective operation - all ranks must call it with the same arguments.
The `src` must have size `(length(row_idx), length(col_idx))` and its row partition
must match `row_idx`'s partition.

This is a structural modification - new nonzeros from src are added to A's sparsity
pattern. After the operation, structural_hash is recomputed and caches are invalidated.

# Example
```julia
A = HPCSparseMatrix{Float64}(spzeros(6, 6))
row_idx = HPCVector([2, 4, 1])
col_idx = HPCVector([3, 1])
src = HPCMatrix(ones(3, 2))
A[row_idx, col_idx] = src  # Adds nonzeros at specified positions
```
"""
function Base.setindex!(A::HPCSparseMatrix{T,Ti,Bk}, src::HPCMatrix{T}, row_idx::HPCVector{Int}, col_idx::HPCVector{Int}) where {T,Ti,Bk<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    m, n = size(A)

    # Gather col_idx to all ranks
    col_indices = _gather_vector_to_all(col_idx, comm)
    ncols_src = length(col_indices)

    # Validate dimensions
    if size(src) != (length(row_idx), ncols_src)
        error("HPCSparseMatrix setindex!: src size $(size(src)) doesn't match index dimensions ($(length(row_idx)), $ncols_src)")
    end

    # Validate that src row partition matches row_idx partition
    if src.row_partition != row_idx.partition
        error("HPCSparseMatrix setindex!: src row partition must match row_idx partition")
    end

    # Validate column indices
    for j in col_indices
        if j < 1 || j > n
            error("HPCSparseMatrix column index out of bounds: $j, ncols=$n")
        end
    end

    # Validate row indices (local check)
    for i in row_idx.v
        if i < 1 || i > m
            error("HPCSparseMatrix row index out of bounds: $i, nrows=$m")
        end
    end

    # My local row indices and corresponding src rows
    local_row_idx = row_idx.v
    n_local_rows = length(local_row_idx)

    # Group rows by which rank owns the destination in A
    send_to = [Tuple{Int,Int}[] for _ in 1:nranks]
    for local_src_row in 1:n_local_rows
        A_global_row = local_row_idx[local_src_row]
        owner = searchsortedlast(A.row_partition, A_global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(send_to[owner + 1], (A_global_row, local_src_row))
    end

    # Exchange counts via Alltoall
    send_counts = Int32[length(send_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = comm_alltoall(comm, MPI.UBuffer(send_counts, 1))

    # Send row indices to destination ranks
    struct_send_reqs = MPI.Request[]
    struct_send_bufs = Dict{Int, Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            indices = [t[1] for t in send_to[r + 1]]
            struct_send_bufs[r] = indices
            push!(struct_send_reqs, comm_isend(comm, indices, r, 94))
        end
    end

    # Receive row indices from source ranks
    struct_recv_bufs = Dict{Int, Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(struct_recv_reqs, comm_irecv!(comm, buf, r, 94))
            struct_recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, struct_recv_reqs)
    comm_waitall(comm, struct_send_reqs)

    # Send row data to destination ranks
    send_reqs = MPI.Request[]
    send_bufs = Dict{Int, Matrix{T}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            nrows_to_send = send_counts[r + 1]
            data = Matrix{T}(undef, nrows_to_send, ncols_src)
            for (k, (_, local_src_row)) in enumerate(send_to[r + 1])
                data[k, :] .= src.A[local_src_row, 1:ncols_src]
            end
            send_bufs[r] = data
            push!(send_reqs, comm_isend(comm, data, r, 95))
        end
    end

    # Receive row data from source ranks
    recv_reqs = MPI.Request[]
    recv_bufs = Dict{Int, Matrix{T}}()

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Matrix{T}(undef, recv_counts[r + 1], ncols_src)
            push!(recv_reqs, comm_irecv!(comm, buf, r, 95))
            recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, recv_reqs)

    # Build insertions for structural modification
    insertions = Vector{Tuple{Int,Int,T}}()
    my_A_row_start = A.row_partition[rank + 1]

    # From local assignments
    for (A_global_row, local_src_row) in send_to[rank + 1]
        for (c, g_col) in enumerate(col_indices)
            val = src.A[local_src_row, c]
            push!(insertions, (A_global_row, g_col, val))
        end
    end

    # From received assignments
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = struct_recv_bufs[r]
            data = recv_bufs[r]
            for (k, A_global_row) in enumerate(indices)
                for (c, g_col) in enumerate(col_indices)
                    val = data[k, c]
                    push!(insertions, (A_global_row, g_col, val))
                end
            end
        end
    end

    comm_waitall(comm, send_reqs)

    # Apply insertions using the helper function
    if !isempty(insertions)
        row_offset = A.row_partition[rank + 1]
        new_AT, new_col_indices = _rebuild_AT_with_insertions(
            _get_csc(A), A.col_indices, insertions, row_offset
        )
        A.col_indices = new_col_indices
        # Update explicit CSR arrays from the new CSC
        A.rowptr = new_AT.colptr
        A.colval = new_AT.rowval
        A.nzval = new_AT.nzval
        A.ncols_compressed = length(new_col_indices)
    end

    # Recompute structural hash after modification
    A.structural_hash = compute_structural_hash(A.row_partition, A.col_indices, A.rowptr, A.colval, A.backend.comm)

    # Invalidate cached transpose bidirectionally
    _invalidate_cached_transpose!(A)

    return src
end

# ============================================================================
# Mixed indexing: HPCVector with ranges, Colon, and scalars
# ============================================================================

"""
    Base.getindex(A::HPCMatrix{T}, row_idx::HPCVector{Int}, col_rng::UnitRange{Int}) where T

Get submatrix with rows selected by HPCVector and columns by range.
Returns a HPCMatrix with row partition matching row_idx.partition.
"""
function Base.getindex(A::HPCMatrix{T,B}, row_idx::HPCVector{Int}, col_rng::UnitRange{Int}) where {T,B<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    m, n = size(A)
    ncols_result = length(col_rng)

    # Validate column range
    if first(col_rng) < 1 || last(col_rng) > n
        error("HPCMatrix column range out of bounds: $col_rng, ncols=$n")
    end

    # Validate row indices (local check)
    for i in row_idx.v
        if i < 1 || i > m
            error("HPCMatrix row index out of bounds: $i, nrows=$m")
        end
    end

    local_row_indices = row_idx.v
    n_local_result_rows = length(local_row_indices)

    # Group requests by source rank
    requests_to = [Int[] for _ in 1:nranks]
    local_positions = [Int[] for _ in 1:nranks]

    for (pos, global_row) in enumerate(local_row_indices)
        owner = searchsortedlast(A.row_partition, global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(requests_to[owner + 1], global_row)
        push!(local_positions[owner + 1], pos)
    end

    # Exchange request counts
    send_counts = Int32[length(requests_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = comm_alltoall(comm, MPI.UBuffer(send_counts, 1))

    # Send row indices we need
    send_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            push!(send_reqs, comm_isend(comm, requests_to[r + 1], r, 100))
        end
    end

    # Receive row indices others need from us
    recv_bufs = Dict{Int, Vector{Int}}()
    recv_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(recv_reqs, comm_irecv!(comm, buf, r, 100))
            recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, recv_reqs)
    comm_waitall(comm, send_reqs)

    # Send requested row data
    data_send_reqs = MPI.Request[]
    data_send_bufs = Dict{Int, Matrix{T}}()
    my_A_start = A.row_partition[rank + 1]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = recv_bufs[r]
            data = Matrix{T}(undef, length(indices), ncols_result)
            for (k, global_row) in enumerate(indices)
                local_row = global_row - my_A_start + 1
                data[k, :] .= A.A[local_row, col_rng]
            end
            data_send_bufs[r] = data
            push!(data_send_reqs, comm_isend(comm, data, r, 101))
        end
    end

    # Receive row data
    data_recv_bufs = Dict{Int, Matrix{T}}()
    data_recv_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            buf = Matrix{T}(undef, send_counts[r + 1], ncols_result)
            push!(data_recv_reqs, comm_irecv!(comm, buf, r, 101))
            data_recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, data_recv_reqs)

    # Build result matrix
    result_local = Matrix{T}(undef, n_local_result_rows, ncols_result)

    # Handle local rows (from self)
    for (k, global_row) in enumerate(requests_to[rank + 1])
        local_row = global_row - my_A_start + 1
        pos = local_positions[rank + 1][k]
        result_local[pos, :] .= A.A[local_row, col_rng]
    end

    # Handle received rows
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            data = data_recv_bufs[r]
            for (k, pos) in enumerate(local_positions[r + 1])
                result_local[pos, :] .= data[k, :]
            end
        end
    end

    comm_waitall(comm, data_send_reqs)

    return HPCMatrix_local(result_local, A.backend)
end

"""
    Base.getindex(A::HPCMatrix{T}, row_idx::HPCVector{Int}, ::Colon) where T

Get submatrix with rows selected by HPCVector and all columns.
"""
function Base.getindex(A::HPCMatrix{T}, row_idx::HPCVector{Int}, ::Colon) where T
    return A[row_idx, 1:size(A, 2)]
end

"""
    Base.getindex(A::HPCMatrix{T}, row_rng::UnitRange{Int}, col_idx::HPCVector{Int}) where T

Get submatrix with rows selected by range and columns by HPCVector.
Returns a HPCMatrix with standard row partition for the given row range size.
"""
function Base.getindex(A::HPCMatrix{T,B}, row_rng::UnitRange{Int}, col_idx::HPCVector{Int}) where {T,B<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    m, n = size(A)
    nrows_result = length(row_rng)

    # Validate row range
    if first(row_rng) < 1 || last(row_rng) > m
        error("HPCMatrix row range out of bounds: $row_rng, nrows=$m")
    end

    # Gather col_idx to all ranks (columns are not distributed)
    col_indices = _gather_vector_to_all(col_idx, comm)
    ncols_result = length(col_indices)

    # Validate column indices
    for j in col_indices
        if j < 1 || j > n
            error("HPCMatrix column index out of bounds: $j, ncols=$n")
        end
    end

    # Create result partition for row range
    result_partition = uniform_partition(nrows_result, nranks)

    # Determine which rows I need
    my_result_start = result_partition[rank + 1]
    my_result_end = result_partition[rank + 2] - 1
    n_local_result_rows = my_result_end - my_result_start + 1

    if n_local_result_rows <= 0
        result_local = Matrix{T}(undef, 0, ncols_result)
        return HPCMatrix_local(result_local, A.backend)
    end

    # Global rows I need from A
    global_rows_needed = [first(row_rng) + my_result_start + i - 2 for i in 1:n_local_result_rows]

    # Group by source rank in A
    requests_to = [Int[] for _ in 1:nranks]
    local_positions = [Int[] for _ in 1:nranks]

    for (pos, global_row) in enumerate(global_rows_needed)
        owner = searchsortedlast(A.row_partition, global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(requests_to[owner + 1], global_row)
        push!(local_positions[owner + 1], pos)
    end

    # Exchange counts
    send_counts = Int32[length(requests_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = comm_alltoall(comm, MPI.UBuffer(send_counts, 1))

    # Send row indices
    send_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            push!(send_reqs, comm_isend(comm, requests_to[r + 1], r, 102))
        end
    end

    # Receive row indices
    recv_bufs = Dict{Int, Vector{Int}}()
    recv_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(recv_reqs, comm_irecv!(comm, buf, r, 102))
            recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, recv_reqs)
    comm_waitall(comm, send_reqs)

    # Send row data (only the selected columns)
    data_send_reqs = MPI.Request[]
    data_send_bufs = Dict{Int, Matrix{T}}()
    my_A_start = A.row_partition[rank + 1]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = recv_bufs[r]
            data = Matrix{T}(undef, length(indices), ncols_result)
            for (k, global_row) in enumerate(indices)
                local_row = global_row - my_A_start + 1
                for (c, col) in enumerate(col_indices)
                    data[k, c] = A.A[local_row, col]
                end
            end
            data_send_bufs[r] = data
            push!(data_send_reqs, comm_isend(comm, data, r, 103))
        end
    end

    # Receive row data
    data_recv_bufs = Dict{Int, Matrix{T}}()
    data_recv_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            buf = Matrix{T}(undef, send_counts[r + 1], ncols_result)
            push!(data_recv_reqs, comm_irecv!(comm, buf, r, 103))
            data_recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, data_recv_reqs)

    # Build result
    result_local = Matrix{T}(undef, n_local_result_rows, ncols_result)

    # Local rows
    for (k, global_row) in enumerate(requests_to[rank + 1])
        local_row = global_row - my_A_start + 1
        pos = local_positions[rank + 1][k]
        for (c, col) in enumerate(col_indices)
            result_local[pos, c] = A.A[local_row, col]
        end
    end

    # Received rows
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            data = data_recv_bufs[r]
            for (k, pos) in enumerate(local_positions[r + 1])
                result_local[pos, :] .= data[k, :]
            end
        end
    end

    comm_waitall(comm, data_send_reqs)

    return HPCMatrix_local(result_local, A.backend)
end

"""
    Base.getindex(A::HPCMatrix{T}, ::Colon, col_idx::HPCVector{Int}) where T

Get submatrix with all rows and columns selected by HPCVector.
"""
function Base.getindex(A::HPCMatrix{T}, ::Colon, col_idx::HPCVector{Int}) where T
    return A[1:size(A, 1), col_idx]
end

"""
    Base.getindex(A::HPCMatrix{T}, row_idx::HPCVector{Int}, j::Int) where T

Get column vector with rows selected by HPCVector and single column j.
Returns a HPCVector with partition matching row_idx.partition.
"""
function Base.getindex(A::HPCMatrix{T,B}, row_idx::HPCVector{Int}, j::Int) where {T,B<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    m, n = size(A)

    # Validate column
    if j < 1 || j > n
        error("HPCMatrix column index out of bounds: $j, ncols=$n")
    end

    # Validate row indices
    for i in row_idx.v
        if i < 1 || i > m
            error("HPCMatrix row index out of bounds: $i, nrows=$m")
        end
    end

    local_row_indices = row_idx.v
    n_local = length(local_row_indices)

    # Group by source rank
    requests_to = [Int[] for _ in 1:nranks]
    local_positions = [Int[] for _ in 1:nranks]

    for (pos, global_row) in enumerate(local_row_indices)
        owner = searchsortedlast(A.row_partition, global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(requests_to[owner + 1], global_row)
        push!(local_positions[owner + 1], pos)
    end

    # Exchange counts
    send_counts = Int32[length(requests_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = comm_alltoall(comm, MPI.UBuffer(send_counts, 1))

    # Send row indices
    send_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            push!(send_reqs, comm_isend(comm, requests_to[r + 1], r, 104))
        end
    end

    # Receive row indices
    recv_bufs = Dict{Int, Vector{Int}}()
    recv_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(recv_reqs, comm_irecv!(comm, buf, r, 104))
            recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, recv_reqs)
    comm_waitall(comm, send_reqs)

    # Send values
    val_send_reqs = MPI.Request[]
    val_send_bufs = Dict{Int, Vector{T}}()
    my_A_start = A.row_partition[rank + 1]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = recv_bufs[r]
            vals = Vector{T}(undef, length(indices))
            for (k, global_row) in enumerate(indices)
                local_row = global_row - my_A_start + 1
                vals[k] = A.A[local_row, j]
            end
            val_send_bufs[r] = vals
            push!(val_send_reqs, comm_isend(comm, vals, r, 105))
        end
    end

    # Receive values
    val_recv_bufs = Dict{Int, Vector{T}}()
    val_recv_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            buf = Vector{T}(undef, send_counts[r + 1])
            push!(val_recv_reqs, comm_irecv!(comm, buf, r, 105))
            val_recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, val_recv_reqs)

    # Build result
    result_local = Vector{T}(undef, n_local)

    # Local values
    for (k, global_row) in enumerate(requests_to[rank + 1])
        local_row = global_row - my_A_start + 1
        pos = local_positions[rank + 1][k]
        result_local[pos] = A.A[local_row, j]
    end

    # Received values
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            vals = val_recv_bufs[r]
            for (k, pos) in enumerate(local_positions[r + 1])
                result_local[pos] = vals[k]
            end
        end
    end

    comm_waitall(comm, val_send_reqs)

    return HPCVector_local(result_local, A.backend)
end

"""
    Base.getindex(A::HPCMatrix{T}, i::Int, col_idx::HPCVector{Int}) where T

Get row vector with single row i and columns selected by HPCVector.
Returns a HPCVector with partition matching col_idx.partition.
"""
function Base.getindex(A::HPCMatrix{T,B}, i::Int, col_idx::HPCVector{Int}) where {T,B<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    m, n = size(A)

    # Validate row
    if i < 1 || i > m
        error("HPCMatrix row index out of bounds: $i, nrows=$m")
    end

    # Gather col_idx to all ranks
    col_indices = _gather_vector_to_all(col_idx, comm)

    # Validate column indices
    for j in col_indices
        if j < 1 || j > n
            error("HPCMatrix column index out of bounds: $j, ncols=$n")
        end
    end

    # Determine which rank owns row i
    owner = searchsortedlast(A.row_partition, i) - 1
    if owner >= nranks
        owner = nranks - 1
    end

    # Owner extracts the row and broadcasts
    if rank == owner
        my_A_start = A.row_partition[rank + 1]
        local_row = i - my_A_start + 1
        row_data = [A.A[local_row, j] for j in col_indices]
    else
        row_data = Vector{T}(undef, length(col_indices))
    end

    comm_bcast!(comm, row_data, owner)

    # Each rank extracts its local portion based on col_idx.partition
    my_start = col_idx.partition[rank + 1]
    my_end = col_idx.partition[rank + 2] - 1
    n_local = my_end - my_start + 1

    if n_local > 0
        result_local = row_data[my_start:my_end]
    else
        result_local = T[]
    end

    return HPCVector_local(result_local, A.backend)
end

# HPCSparseMatrix mixed indexing methods

"""
    Base.getindex(A::HPCSparseMatrix{T}, row_idx::HPCVector{Int}, col_rng::UnitRange{Int}) where T

Get submatrix with rows selected by HPCVector and columns by range.
Returns a dense HPCMatrix with row partition matching row_idx.partition.
"""
function Base.getindex(A::HPCSparseMatrix{T,Ti,Bk}, row_idx::HPCVector{Int}, col_rng::UnitRange{Int}) where {T,Ti,Bk<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    m, n = size(A)
    ncols_result = length(col_rng)

    # Validate column range
    if first(col_rng) < 1 || last(col_rng) > n
        error("HPCSparseMatrix column range out of bounds: $col_rng, ncols=$n")
    end

    # Validate row indices
    for i in row_idx.v
        if i < 1 || i > m
            error("HPCSparseMatrix row index out of bounds: $i, nrows=$m")
        end
    end

    local_row_indices = row_idx.v
    n_local_result_rows = length(local_row_indices)

    # Group requests by source rank
    requests_to = [Int[] for _ in 1:nranks]
    local_positions = [Int[] for _ in 1:nranks]

    for (pos, global_row) in enumerate(local_row_indices)
        owner = searchsortedlast(A.row_partition, global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(requests_to[owner + 1], global_row)
        push!(local_positions[owner + 1], pos)
    end

    # Exchange counts
    send_counts = Int32[length(requests_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = comm_alltoall(comm, MPI.UBuffer(send_counts, 1))

    # Send row indices
    send_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            push!(send_reqs, comm_isend(comm, requests_to[r + 1], r, 106))
        end
    end

    # Receive row indices
    recv_bufs = Dict{Int, Vector{Int}}()
    recv_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(recv_reqs, comm_irecv!(comm, buf, r, 106))
            recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, recv_reqs)
    comm_waitall(comm, send_reqs)

    # Send row data
    data_send_reqs = MPI.Request[]
    data_send_bufs = Dict{Int, Matrix{T}}()
    my_A_start = A.row_partition[rank + 1]
    local_A = _get_csc(A)
    col_indices = A.col_indices

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = recv_bufs[r]
            data = zeros(T, length(indices), ncols_result)
            for (k, global_row) in enumerate(indices)
                local_row = global_row - my_A_start + 1
                for (local_j, global_j) in enumerate(col_indices)
                    if global_j in col_rng
                        result_col = global_j - first(col_rng) + 1
                        data[k, result_col] = local_A[local_j, local_row]
                    end
                end
            end
            data_send_bufs[r] = data
            push!(data_send_reqs, comm_isend(comm, data, r, 107))
        end
    end

    # Receive row data
    data_recv_bufs = Dict{Int, Matrix{T}}()
    data_recv_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            buf = Matrix{T}(undef, send_counts[r + 1], ncols_result)
            push!(data_recv_reqs, comm_irecv!(comm, buf, r, 107))
            data_recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, data_recv_reqs)

    # Build result
    result_local = zeros(T, n_local_result_rows, ncols_result)

    # Local rows
    for (k, global_row) in enumerate(requests_to[rank + 1])
        local_row = global_row - my_A_start + 1
        pos = local_positions[rank + 1][k]
        for (local_j, global_j) in enumerate(col_indices)
            if global_j in col_rng
                result_col = global_j - first(col_rng) + 1
                result_local[pos, result_col] = local_A[local_j, local_row]
            end
        end
    end

    # Received rows
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            data = data_recv_bufs[r]
            for (k, pos) in enumerate(local_positions[r + 1])
                result_local[pos, :] .= data[k, :]
            end
        end
    end

    comm_waitall(comm, data_send_reqs)

    return HPCMatrix_local(result_local, A.backend)
end

"""
    Base.getindex(A::HPCSparseMatrix{T}, row_idx::HPCVector{Int}, ::Colon) where T

Get submatrix with rows selected by HPCVector and all columns.
"""
function Base.getindex(A::HPCSparseMatrix{T}, row_idx::HPCVector{Int}, ::Colon) where T
    return A[row_idx, 1:size(A, 2)]
end

"""
    Base.getindex(A::HPCSparseMatrix{T}, row_rng::UnitRange{Int}, col_idx::HPCVector{Int}) where T

Get submatrix with rows by range and columns by HPCVector.
Returns a dense HPCMatrix.
"""
function Base.getindex(A::HPCSparseMatrix{T,Ti,Bk}, row_rng::UnitRange{Int}, col_idx::HPCVector{Int}) where {T,Ti,Bk<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    m, n = size(A)
    nrows_result = length(row_rng)

    # Validate row range
    if first(row_rng) < 1 || last(row_rng) > m
        error("HPCSparseMatrix row range out of bounds: $row_rng, nrows=$m")
    end

    # Gather col_idx to all ranks
    col_indices_result = _gather_vector_to_all(col_idx, comm)
    ncols_result = length(col_indices_result)

    # Validate column indices
    for j in col_indices_result
        if j < 1 || j > n
            error("HPCSparseMatrix column index out of bounds: $j, ncols=$n")
        end
    end

    # Create result partition
    result_partition = uniform_partition(nrows_result, nranks)
    my_result_start = result_partition[rank + 1]
    my_result_end = result_partition[rank + 2] - 1
    n_local_result_rows = my_result_end - my_result_start + 1

    if n_local_result_rows <= 0
        result_local = Matrix{T}(undef, 0, ncols_result)
        return HPCMatrix_local(result_local, A.backend)
    end

    # Global rows I need
    global_rows_needed = [first(row_rng) + my_result_start + i - 2 for i in 1:n_local_result_rows]

    # Group by source rank
    requests_to = [Int[] for _ in 1:nranks]
    local_positions = [Int[] for _ in 1:nranks]

    for (pos, global_row) in enumerate(global_rows_needed)
        owner = searchsortedlast(A.row_partition, global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(requests_to[owner + 1], global_row)
        push!(local_positions[owner + 1], pos)
    end

    # Exchange counts
    send_counts = Int32[length(requests_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = comm_alltoall(comm, MPI.UBuffer(send_counts, 1))

    # Send/receive row indices
    send_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            push!(send_reqs, comm_isend(comm, requests_to[r + 1], r, 108))
        end
    end

    recv_bufs = Dict{Int, Vector{Int}}()
    recv_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(recv_reqs, comm_irecv!(comm, buf, r, 108))
            recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, recv_reqs)
    comm_waitall(comm, send_reqs)

    # Send row data
    data_send_reqs = MPI.Request[]
    data_send_bufs = Dict{Int, Matrix{T}}()
    my_A_start = A.row_partition[rank + 1]
    local_A = _get_csc(A)
    col_indices = A.col_indices

    # Build column index lookup
    col_idx_map = Dict{Int, Int}()
    for (c, j) in enumerate(col_indices_result)
        col_idx_map[j] = c
    end

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = recv_bufs[r]
            data = zeros(T, length(indices), ncols_result)
            for (k, global_row) in enumerate(indices)
                local_row = global_row - my_A_start + 1
                for (local_j, global_j) in enumerate(col_indices)
                    if haskey(col_idx_map, global_j)
                        result_col = col_idx_map[global_j]
                        data[k, result_col] = local_A[local_j, local_row]
                    end
                end
            end
            data_send_bufs[r] = data
            push!(data_send_reqs, comm_isend(comm, data, r, 109))
        end
    end

    # Receive row data
    data_recv_bufs = Dict{Int, Matrix{T}}()
    data_recv_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            buf = Matrix{T}(undef, send_counts[r + 1], ncols_result)
            push!(data_recv_reqs, comm_irecv!(comm, buf, r, 109))
            data_recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, data_recv_reqs)

    # Build result
    result_local = zeros(T, n_local_result_rows, ncols_result)

    # Local rows
    for (k, global_row) in enumerate(requests_to[rank + 1])
        local_row = global_row - my_A_start + 1
        pos = local_positions[rank + 1][k]
        for (local_j, global_j) in enumerate(col_indices)
            if haskey(col_idx_map, global_j)
                result_col = col_idx_map[global_j]
                result_local[pos, result_col] = local_A[local_j, local_row]
            end
        end
    end

    # Received rows
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            data = data_recv_bufs[r]
            for (k, pos) in enumerate(local_positions[r + 1])
                result_local[pos, :] .= data[k, :]
            end
        end
    end

    comm_waitall(comm, data_send_reqs)

    return HPCMatrix_local(result_local, A.backend)
end

"""
    Base.getindex(A::HPCSparseMatrix{T}, ::Colon, col_idx::HPCVector{Int}) where T

Get submatrix with all rows and columns by HPCVector.
"""
function Base.getindex(A::HPCSparseMatrix{T}, ::Colon, col_idx::HPCVector{Int}) where T
    return A[1:size(A, 1), col_idx]
end

"""
    Base.getindex(A::HPCSparseMatrix{T}, row_idx::HPCVector{Int}, j::Int) where T

Get column vector with rows by HPCVector and single column j.
Returns a HPCVector.
"""
function Base.getindex(A::HPCSparseMatrix{T,Ti,Bk}, row_idx::HPCVector{Int}, j::Int) where {T,Ti,Bk<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    m, n = size(A)

    # Validate
    if j < 1 || j > n
        error("HPCSparseMatrix column index out of bounds: $j, ncols=$n")
    end
    for i in row_idx.v
        if i < 1 || i > m
            error("HPCSparseMatrix row index out of bounds: $i, nrows=$m")
        end
    end

    local_row_indices = row_idx.v
    n_local = length(local_row_indices)

    # Group by source rank
    requests_to = [Int[] for _ in 1:nranks]
    local_positions = [Int[] for _ in 1:nranks]

    for (pos, global_row) in enumerate(local_row_indices)
        owner = searchsortedlast(A.row_partition, global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(requests_to[owner + 1], global_row)
        push!(local_positions[owner + 1], pos)
    end

    # Exchange counts
    send_counts = Int32[length(requests_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = comm_alltoall(comm, MPI.UBuffer(send_counts, 1))

    # Send/receive row indices
    send_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            push!(send_reqs, comm_isend(comm, requests_to[r + 1], r, 110))
        end
    end

    recv_bufs = Dict{Int, Vector{Int}}()
    recv_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(recv_reqs, comm_irecv!(comm, buf, r, 110))
            recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, recv_reqs)
    comm_waitall(comm, send_reqs)

    # Send values
    val_send_reqs = MPI.Request[]
    val_send_bufs = Dict{Int, Vector{T}}()
    my_A_start = A.row_partition[rank + 1]
    local_A = _get_csc(A)
    col_indices = A.col_indices
    local_j_idx = findfirst(==(j), col_indices)

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = recv_bufs[r]
            vals = zeros(T, length(indices))
            if local_j_idx !== nothing
                for (k, global_row) in enumerate(indices)
                    local_row = global_row - my_A_start + 1
                    vals[k] = local_A[local_j_idx, local_row]
                end
            end
            val_send_bufs[r] = vals
            push!(val_send_reqs, comm_isend(comm, vals, r, 111))
        end
    end

    # Receive values
    val_recv_bufs = Dict{Int, Vector{T}}()
    val_recv_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            buf = Vector{T}(undef, send_counts[r + 1])
            push!(val_recv_reqs, comm_irecv!(comm, buf, r, 111))
            val_recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, val_recv_reqs)

    # Build result
    result_local = zeros(T, n_local)

    # Local values
    if local_j_idx !== nothing
        for (k, global_row) in enumerate(requests_to[rank + 1])
            local_row = global_row - my_A_start + 1
            pos = local_positions[rank + 1][k]
            result_local[pos] = local_A[local_j_idx, local_row]
        end
    end

    # Received values
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            vals = val_recv_bufs[r]
            for (k, pos) in enumerate(local_positions[r + 1])
                result_local[pos] = vals[k]
            end
        end
    end

    comm_waitall(comm, val_send_reqs)

    return HPCVector_local(result_local, A.backend)
end

"""
    Base.getindex(A::HPCSparseMatrix{T}, i::Int, col_idx::HPCVector{Int}) where T

Get row vector with single row i and columns by HPCVector.
Returns a HPCVector.
"""
function Base.getindex(A::HPCSparseMatrix{T,Ti,Bk}, i::Int, col_idx::HPCVector{Int}) where {T,Ti,Bk<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    m, n = size(A)

    # Validate
    if i < 1 || i > m
        error("HPCSparseMatrix row index out of bounds: $i, nrows=$m")
    end

    # Gather col_idx
    col_indices_result = _gather_vector_to_all(col_idx, comm)

    for j in col_indices_result
        if j < 1 || j > n
            error("HPCSparseMatrix column index out of bounds: $j, ncols=$n")
        end
    end

    # Determine owner of row i
    owner = searchsortedlast(A.row_partition, i) - 1
    if owner >= nranks
        owner = nranks - 1
    end

    # Owner extracts row and broadcasts
    if rank == owner
        my_A_start = A.row_partition[rank + 1]
        local_row = i - my_A_start + 1
        local_A = _get_csc(A)
        col_indices = A.col_indices

        row_data = zeros(T, length(col_indices_result))
        # col_indices_result is sorted, use binary search instead of Dict
        for (local_j, global_j) in enumerate(col_indices)
            idx = searchsortedfirst(col_indices_result, global_j)
            if idx <= length(col_indices_result) && col_indices_result[idx] == global_j
                row_data[idx] = local_A[local_j, local_row]
            end
        end
    else
        row_data = Vector{T}(undef, length(col_indices_result))
    end

    comm_bcast!(comm, row_data, owner)

    # Extract local portion
    my_start = col_idx.partition[rank + 1]
    my_end = col_idx.partition[rank + 2] - 1
    n_local = my_end - my_start + 1

    if n_local > 0
        result_local = row_data[my_start:my_end]
    else
        result_local = T[]
    end

    return HPCVector_local(result_local, A.backend)
end

# ============================================================================
# Mixed setindex!: HPCVector with ranges, Colon, and scalars
# ============================================================================

"""
    Base.setindex!(A::HPCMatrix{T}, src::HPCMatrix{T}, row_idx::HPCVector{Int}, col_rng::UnitRange{Int}) where T

Set `A[row_idx, col_rng] = src` where rows are selected by HPCVector and columns by range.
The `src` must have row partition matching `row_idx.partition` and column count matching `length(col_rng)`.
"""
function Base.setindex!(A::HPCMatrix{T,B}, src::HPCMatrix{T,B}, row_idx::HPCVector{Int}, col_rng::UnitRange{Int}) where {T,B<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    m, n = size(A)
    ncols_src = length(col_rng)

    # Validate column range
    if first(col_rng) < 1 || last(col_rng) > n
        error("HPCMatrix column range out of bounds: $col_rng, ncols=$n")
    end

    # Validate src dimensions
    if size(src, 2) != ncols_src
        error("HPCMatrix setindex!: src columns ($(size(src, 2))) must match range length ($ncols_src)")
    end
    if src.row_partition != row_idx.partition
        error("HPCMatrix setindex!: src row partition must match row_idx partition")
    end

    # Validate row indices (local check)
    for i in row_idx.v
        if i < 1 || i > m
            error("HPCMatrix row index out of bounds: $i, nrows=$m")
        end
    end

    local_row_indices = row_idx.v
    n_local_src_rows = length(local_row_indices)

    # Group data by destination rank
    send_to = [Tuple{Int, Vector{T}}[] for _ in 1:nranks]  # (global_row, row_data)

    for (pos, global_row) in enumerate(local_row_indices)
        owner = searchsortedlast(A.row_partition, global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        row_data = Vector{T}(src.A[pos, :])
        push!(send_to[owner + 1], (global_row, row_data))
    end

    # Exchange counts via Alltoall
    send_counts = Int32[length(send_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = comm_alltoall(comm, MPI.UBuffer(send_counts, 1))

    # Send row indices to destination ranks
    struct_send_reqs = MPI.Request[]
    struct_send_bufs = Dict{Int, Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            indices = [t[1] for t in send_to[r + 1]]
            struct_send_bufs[r] = indices
            push!(struct_send_reqs, comm_isend(comm, indices, r, 110))
        end
    end

    # Receive row indices from source ranks
    struct_recv_bufs = Dict{Int, Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(struct_recv_reqs, comm_irecv!(comm, buf, r, 110))
            struct_recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, struct_recv_reqs)
    comm_waitall(comm, struct_send_reqs)

    # Send row data to destination ranks
    data_send_reqs = MPI.Request[]
    data_send_bufs = Dict{Int, Matrix{T}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            n_rows = length(send_to[r + 1])
            data = Matrix{T}(undef, n_rows, ncols_src)
            for (k, (_, row_data)) in enumerate(send_to[r + 1])
                data[k, :] .= row_data
            end
            data_send_bufs[r] = data
            push!(data_send_reqs, comm_isend(comm, data, r, 111))
        end
    end

    # Receive row data from source ranks
    data_recv_bufs = Dict{Int, Matrix{T}}()
    data_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Matrix{T}(undef, recv_counts[r + 1], ncols_src)
            push!(data_recv_reqs, comm_irecv!(comm, buf, r, 111))
            data_recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, data_recv_reqs)

    # Apply local assignments (from my own send_to[rank+1])
    my_A_start = A.row_partition[rank + 1]
    for (global_row, row_data) in send_to[rank + 1]
        local_row = global_row - my_A_start + 1
        A.A[local_row, col_rng] .= row_data
    end

    # Apply received assignments
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = struct_recv_bufs[r]
            data = data_recv_bufs[r]
            for (k, global_row) in enumerate(indices)
                local_row = global_row - my_A_start + 1
                A.A[local_row, col_rng] .= data[k, :]
            end
        end
    end

    comm_waitall(comm, data_send_reqs)

    return src
end

"""
    Base.setindex!(A::HPCMatrix{T}, src::HPCMatrix{T}, row_idx::HPCVector{Int}, ::Colon) where T

Set `A[row_idx, :] = src` where rows are selected by HPCVector and all columns.
"""
function Base.setindex!(A::HPCMatrix{T}, src::HPCMatrix{T}, row_idx::HPCVector{Int}, ::Colon) where T
    A[row_idx, 1:size(A, 2)] = src
    return src
end

"""
    Base.setindex!(A::HPCMatrix{T}, src::HPCMatrix{T}, row_rng::UnitRange{Int}, col_idx::HPCVector{Int}) where T

Set `A[row_rng, col_idx] = src` where rows are selected by range and columns by HPCVector.
The `src` must have matching dimensions.
"""
function Base.setindex!(A::HPCMatrix{T,B}, src::HPCMatrix{T,B}, row_rng::UnitRange{Int}, col_idx::HPCVector{Int}) where {T,B<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    m, n = size(A)
    nrows_src = length(row_rng)

    # Validate row range
    if first(row_rng) < 1 || last(row_rng) > m
        error("HPCMatrix row range out of bounds: $row_rng, nrows=$m")
    end

    # Gather col_idx to all ranks (columns are not distributed)
    col_indices = _gather_vector_to_all(col_idx, comm)
    ncols_src = length(col_indices)

    # Validate column indices
    for j in col_indices
        if j < 1 || j > n
            error("HPCMatrix column index out of bounds: $j, ncols=$n")
        end
    end

    # Validate src dimensions
    if size(src, 1) != nrows_src || size(src, 2) != ncols_src
        error("HPCMatrix setindex!: src size ($(size(src))) must match ($nrows_src, $ncols_src)")
    end

    # Compute which rows of src I own vs which rows of A's row_rng I own
    src_partition = src.row_partition
    my_src_start = src_partition[rank + 1]
    my_src_end = src_partition[rank + 2] - 1

    # Map src row indices to global row indices in A
    # src row i corresponds to A row row_rng[i]

    # Group data by destination rank in A
    send_to = [Tuple{Int, Vector{T}}[] for _ in 1:nranks]  # (global_row_in_A, row_data)

    for src_row in my_src_start:my_src_end
        global_row_in_A = row_rng[src_row]
        owner = searchsortedlast(A.row_partition, global_row_in_A) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        local_src_row = src_row - my_src_start + 1
        row_data = Vector{T}(src.A[local_src_row, :])
        push!(send_to[owner + 1], (global_row_in_A, row_data))
    end

    # Exchange counts via Alltoall
    send_counts = Int32[length(send_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = comm_alltoall(comm, MPI.UBuffer(send_counts, 1))

    # Send row indices to destination ranks
    struct_send_reqs = MPI.Request[]
    struct_send_bufs = Dict{Int, Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            indices = [t[1] for t in send_to[r + 1]]
            struct_send_bufs[r] = indices
            push!(struct_send_reqs, comm_isend(comm, indices, r, 112))
        end
    end

    # Receive row indices from source ranks
    struct_recv_bufs = Dict{Int, Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(struct_recv_reqs, comm_irecv!(comm, buf, r, 112))
            struct_recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, struct_recv_reqs)
    comm_waitall(comm, struct_send_reqs)

    # Send row data to destination ranks
    data_send_reqs = MPI.Request[]
    data_send_bufs = Dict{Int, Matrix{T}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            n_rows = length(send_to[r + 1])
            data = Matrix{T}(undef, n_rows, ncols_src)
            for (k, (_, row_data)) in enumerate(send_to[r + 1])
                data[k, :] .= row_data
            end
            data_send_bufs[r] = data
            push!(data_send_reqs, comm_isend(comm, data, r, 113))
        end
    end

    # Receive row data from source ranks
    data_recv_bufs = Dict{Int, Matrix{T}}()
    data_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Matrix{T}(undef, recv_counts[r + 1], ncols_src)
            push!(data_recv_reqs, comm_irecv!(comm, buf, r, 113))
            data_recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, data_recv_reqs)

    # Apply local assignments (from my own send_to[rank+1])
    my_A_start = A.row_partition[rank + 1]
    for (global_row_in_A, row_data) in send_to[rank + 1]
        local_row = global_row_in_A - my_A_start + 1
        for (c, global_col) in enumerate(col_indices)
            A.A[local_row, global_col] = row_data[c]
        end
    end

    # Apply received assignments
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = struct_recv_bufs[r]
            data = data_recv_bufs[r]
            for (k, global_row_in_A) in enumerate(indices)
                local_row = global_row_in_A - my_A_start + 1
                for (c, global_col) in enumerate(col_indices)
                    A.A[local_row, global_col] = data[k, c]
                end
            end
        end
    end

    comm_waitall(comm, data_send_reqs)

    return src
end

"""
    Base.setindex!(A::HPCMatrix{T}, src::HPCMatrix{T}, ::Colon, col_idx::HPCVector{Int}) where T

Set `A[:, col_idx] = src` where all rows and columns selected by HPCVector.
"""
function Base.setindex!(A::HPCMatrix{T}, src::HPCMatrix{T}, ::Colon, col_idx::HPCVector{Int}) where T
    A[1:size(A, 1), col_idx] = src
    return src
end

"""
    Base.setindex!(A::HPCMatrix{T}, src::HPCVector{T}, row_idx::HPCVector{Int}, j::Integer) where T

Set `A[row_idx, j] = src` where rows are selected by HPCVector and a single column.
The `src` must have partition matching `row_idx.partition`.
"""
function Base.setindex!(A::HPCMatrix{T,B}, src::HPCVector{T,B}, row_idx::HPCVector{Int}, j::Integer) where {T,B<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    m, n = size(A)

    # Validate column index
    if j < 1 || j > n
        error("HPCMatrix column index out of bounds: $j, ncols=$n")
    end

    # Validate partitions match
    if src.partition != row_idx.partition
        error("HPCMatrix setindex!: src partition must match row_idx partition")
    end

    # Validate row indices (local check)
    for i in row_idx.v
        if i < 1 || i > m
            error("HPCMatrix row index out of bounds: $i, nrows=$m")
        end
    end

    local_row_indices = row_idx.v
    local_src = src.v

    # Group data by destination rank
    send_to = [Tuple{Int, T}[] for _ in 1:nranks]  # (global_row, value)

    for (pos, global_row) in enumerate(local_row_indices)
        owner = searchsortedlast(A.row_partition, global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(send_to[owner + 1], (global_row, local_src[pos]))
    end

    # Exchange counts via Alltoall
    send_counts = Int32[length(send_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = comm_alltoall(comm, MPI.UBuffer(send_counts, 1))

    # Send row indices to destination ranks
    struct_send_reqs = MPI.Request[]
    struct_send_bufs = Dict{Int, Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            indices = [t[1] for t in send_to[r + 1]]
            struct_send_bufs[r] = indices
            push!(struct_send_reqs, comm_isend(comm, indices, r, 114))
        end
    end

    # Receive row indices from source ranks
    struct_recv_bufs = Dict{Int, Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(struct_recv_reqs, comm_irecv!(comm, buf, r, 114))
            struct_recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, struct_recv_reqs)
    comm_waitall(comm, struct_send_reqs)

    # Send values to destination ranks
    data_send_reqs = MPI.Request[]
    data_send_bufs = Dict{Int, Vector{T}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            values = [t[2] for t in send_to[r + 1]]
            data_send_bufs[r] = values
            push!(data_send_reqs, comm_isend(comm, values, r, 115))
        end
    end

    # Receive values from source ranks
    data_recv_bufs = Dict{Int, Vector{T}}()
    data_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{T}(undef, recv_counts[r + 1])
            push!(data_recv_reqs, comm_irecv!(comm, buf, r, 115))
            data_recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, data_recv_reqs)

    # Apply local assignments (from my own send_to[rank+1])
    my_A_start = A.row_partition[rank + 1]
    for (global_row, value) in send_to[rank + 1]
        local_row = global_row - my_A_start + 1
        A.A[local_row, j] = value
    end

    # Apply received assignments
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = struct_recv_bufs[r]
            values = data_recv_bufs[r]
            for (k, global_row) in enumerate(indices)
                local_row = global_row - my_A_start + 1
                A.A[local_row, j] = values[k]
            end
        end
    end

    comm_waitall(comm, data_send_reqs)

    return src
end

"""
    Base.setindex!(A::HPCMatrix{T}, src::HPCVector{T}, i::Integer, col_idx::HPCVector{Int}) where T

Set `A[i, col_idx] = src` where a single row and columns selected by HPCVector.
The `src` must have partition matching `col_idx.partition`.
"""
function Base.setindex!(A::HPCMatrix{T,B}, src::HPCVector{T,B}, i::Integer, col_idx::HPCVector{Int}) where {T,B<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    m, n = size(A)

    # Validate row index
    if i < 1 || i > m
        error("HPCMatrix row index out of bounds: $i, nrows=$m")
    end

    # Validate partitions match
    if src.partition != col_idx.partition
        error("HPCMatrix setindex!: src partition must match col_idx partition")
    end

    # Validate column indices (local check)
    for j in col_idx.v
        if j < 1 || j > n
            error("HPCMatrix column index out of bounds: $j, ncols=$n")
        end
    end

    # Find owner of row i
    owner = searchsortedlast(A.row_partition, i) - 1
    if owner >= nranks
        owner = nranks - 1
    end

    # Gather col_idx and src to owner rank
    col_indices = _gather_vector_to_all(col_idx, comm)
    src_values = _gather_vector_to_all(src, comm)

    # Only owner updates
    if rank == owner
        my_A_start = A.row_partition[rank + 1]
        local_row = i - my_A_start + 1
        for (k, global_col) in enumerate(col_indices)
            A.A[local_row, global_col] = src_values[k]
        end
    end

    return src
end

# ============================================================================
# HPCSparseMatrix mixed setindex!
# ============================================================================

"""
    Base.setindex!(A::HPCSparseMatrix{T}, src::HPCMatrix{T}, row_idx::HPCVector{Int}, col_rng::UnitRange{Int}) where T

Set `A[row_idx, col_rng] = src` where rows are selected by HPCVector and columns by range.
The `src` must have row partition matching `row_idx.partition` and column count matching `length(col_rng)`.
"""
function Base.setindex!(A::HPCSparseMatrix{T,Ti,Bk}, src::HPCMatrix{T}, row_idx::HPCVector{Int}, col_rng::UnitRange{Int}) where {T,Ti,Bk<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    m, n = size(A)
    ncols_src = length(col_rng)

    # Validate column range
    if first(col_rng) < 1 || last(col_rng) > n
        error("HPCSparseMatrix column range out of bounds: $col_rng, ncols=$n")
    end

    # Validate src dimensions
    if size(src, 2) != ncols_src
        error("HPCSparseMatrix setindex!: src columns ($(size(src, 2))) must match range length ($ncols_src)")
    end
    if src.row_partition != row_idx.partition
        error("HPCSparseMatrix setindex!: src row partition must match row_idx partition")
    end

    # Validate row indices (local check)
    for i in row_idx.v
        if i < 1 || i > m
            error("HPCSparseMatrix row index out of bounds: $i, nrows=$m")
        end
    end

    local_row_indices = row_idx.v
    n_local_src_rows = length(local_row_indices)

    # Group rows by which rank owns the destination in A
    send_to = [Tuple{Int,Int}[] for _ in 1:nranks]
    for local_src_row in 1:n_local_src_rows
        A_global_row = local_row_indices[local_src_row]
        owner = searchsortedlast(A.row_partition, A_global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(send_to[owner + 1], (A_global_row, local_src_row))
    end

    # Exchange counts via Alltoall
    send_counts = Int32[length(send_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = comm_alltoall(comm, MPI.UBuffer(send_counts, 1))

    # Send row indices to destination ranks
    struct_send_reqs = MPI.Request[]
    struct_send_bufs = Dict{Int, Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            indices = [t[1] for t in send_to[r + 1]]
            struct_send_bufs[r] = indices
            push!(struct_send_reqs, comm_isend(comm, indices, r, 120))
        end
    end

    # Receive row indices from source ranks
    struct_recv_bufs = Dict{Int, Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(struct_recv_reqs, comm_irecv!(comm, buf, r, 120))
            struct_recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, struct_recv_reqs)
    comm_waitall(comm, struct_send_reqs)

    # Send row data to destination ranks
    send_reqs = MPI.Request[]
    send_bufs = Dict{Int, Matrix{T}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            nrows_to_send = send_counts[r + 1]
            data = Matrix{T}(undef, nrows_to_send, ncols_src)
            for (k, (_, local_src_row)) in enumerate(send_to[r + 1])
                data[k, :] .= src.A[local_src_row, 1:ncols_src]
            end
            send_bufs[r] = data
            push!(send_reqs, comm_isend(comm, data, r, 121))
        end
    end

    # Receive row data from source ranks
    recv_reqs = MPI.Request[]
    recv_bufs = Dict{Int, Matrix{T}}()

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Matrix{T}(undef, recv_counts[r + 1], ncols_src)
            push!(recv_reqs, comm_irecv!(comm, buf, r, 121))
            recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, recv_reqs)

    # Build insertions for structural modification
    insertions = Vector{Tuple{Int,Int,T}}()
    col_indices = collect(col_rng)

    # From local assignments
    for (A_global_row, local_src_row) in send_to[rank + 1]
        for (c, g_col) in enumerate(col_indices)
            val = src.A[local_src_row, c]
            push!(insertions, (A_global_row, g_col, val))
        end
    end

    # From received assignments
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = struct_recv_bufs[r]
            data = recv_bufs[r]
            for (k, A_global_row) in enumerate(indices)
                for (c, g_col) in enumerate(col_indices)
                    val = data[k, c]
                    push!(insertions, (A_global_row, g_col, val))
                end
            end
        end
    end

    comm_waitall(comm, send_reqs)

    # Apply insertions using the helper function
    if !isempty(insertions)
        row_offset = A.row_partition[rank + 1]
        new_AT, new_col_indices = _rebuild_AT_with_insertions(
            _get_csc(A), A.col_indices, insertions, row_offset
        )
        A.col_indices = new_col_indices
        # Update explicit CSR arrays from the new CSC
        A.rowptr = new_AT.colptr
        A.colval = new_AT.rowval
        A.nzval = new_AT.nzval
        A.ncols_compressed = length(new_col_indices)
    end

    # Recompute structural hash after modification
    A.structural_hash = compute_structural_hash(A.row_partition, A.col_indices, A.rowptr, A.colval, A.backend.comm)

    # Invalidate cached transpose bidirectionally
    _invalidate_cached_transpose!(A)

    return src
end

"""
    Base.setindex!(A::HPCSparseMatrix{T}, src::HPCMatrix{T}, row_idx::HPCVector{Int}, ::Colon) where T

Set `A[row_idx, :] = src` where rows are selected by HPCVector and all columns.
"""
function Base.setindex!(A::HPCSparseMatrix{T}, src::HPCMatrix{T}, row_idx::HPCVector{Int}, ::Colon) where T
    A[row_idx, 1:size(A, 2)] = src
    return src
end

"""
    Base.setindex!(A::HPCSparseMatrix{T}, src::HPCMatrix{T}, row_rng::UnitRange{Int}, col_idx::HPCVector{Int}) where T

Set `A[row_rng, col_idx] = src` where rows are selected by range and columns by HPCVector.
"""
function Base.setindex!(A::HPCSparseMatrix{T,Ti,Bk}, src::HPCMatrix{T}, row_rng::UnitRange{Int}, col_idx::HPCVector{Int}) where {T,Ti,Bk<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    m, n = size(A)
    nrows_src = length(row_rng)

    # Validate row range
    if first(row_rng) < 1 || last(row_rng) > m
        error("HPCSparseMatrix row range out of bounds: $row_rng, nrows=$m")
    end

    # Gather col_idx to all ranks (columns are not distributed)
    col_indices = _gather_vector_to_all(col_idx, comm)
    ncols_src = length(col_indices)

    # Validate column indices
    for j in col_indices
        if j < 1 || j > n
            error("HPCSparseMatrix column index out of bounds: $j, ncols=$n")
        end
    end

    # Validate src dimensions
    if size(src, 1) != nrows_src || size(src, 2) != ncols_src
        error("HPCSparseMatrix setindex!: src size ($(size(src))) must match ($nrows_src, $ncols_src)")
    end

    # Compute which rows of src I own vs which rows of A's row_rng I own
    src_partition = src.row_partition
    my_src_start = src_partition[rank + 1]
    my_src_end = src_partition[rank + 2] - 1

    # Group data by destination rank in A
    send_to = [Tuple{Int, Int}[] for _ in 1:nranks]  # (global_row_in_A, local_src_row)

    for src_row in my_src_start:my_src_end
        global_row_in_A = row_rng[src_row]
        owner = searchsortedlast(A.row_partition, global_row_in_A) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        local_src_row = src_row - my_src_start + 1
        push!(send_to[owner + 1], (global_row_in_A, local_src_row))
    end

    # Exchange counts via Alltoall
    send_counts = Int32[length(send_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = comm_alltoall(comm, MPI.UBuffer(send_counts, 1))

    # Send row indices to destination ranks
    struct_send_reqs = MPI.Request[]
    struct_send_bufs = Dict{Int, Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            indices = [t[1] for t in send_to[r + 1]]
            struct_send_bufs[r] = indices
            push!(struct_send_reqs, comm_isend(comm, indices, r, 122))
        end
    end

    # Receive row indices from source ranks
    struct_recv_bufs = Dict{Int, Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(struct_recv_reqs, comm_irecv!(comm, buf, r, 122))
            struct_recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, struct_recv_reqs)
    comm_waitall(comm, struct_send_reqs)

    # Send row data to destination ranks
    send_reqs = MPI.Request[]
    send_bufs = Dict{Int, Matrix{T}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            n_rows = length(send_to[r + 1])
            data = Matrix{T}(undef, n_rows, ncols_src)
            for (k, (_, local_src_row)) in enumerate(send_to[r + 1])
                data[k, :] .= src.A[local_src_row, :]
            end
            send_bufs[r] = data
            push!(send_reqs, comm_isend(comm, data, r, 123))
        end
    end

    # Receive row data from source ranks
    recv_bufs = Dict{Int, Matrix{T}}()
    recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Matrix{T}(undef, recv_counts[r + 1], ncols_src)
            push!(recv_reqs, comm_irecv!(comm, buf, r, 123))
            recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, recv_reqs)

    # Build insertions for structural modification
    insertions = Vector{Tuple{Int,Int,T}}()

    # From local assignments
    for (A_global_row, local_src_row) in send_to[rank + 1]
        for (c, g_col) in enumerate(col_indices)
            val = src.A[local_src_row, c]
            push!(insertions, (A_global_row, g_col, val))
        end
    end

    # From received assignments
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = struct_recv_bufs[r]
            data = recv_bufs[r]
            for (k, A_global_row) in enumerate(indices)
                for (c, g_col) in enumerate(col_indices)
                    val = data[k, c]
                    push!(insertions, (A_global_row, g_col, val))
                end
            end
        end
    end

    comm_waitall(comm, send_reqs)

    # Apply insertions using the helper function
    if !isempty(insertions)
        row_offset = A.row_partition[rank + 1]
        new_AT, new_col_indices = _rebuild_AT_with_insertions(
            _get_csc(A), A.col_indices, insertions, row_offset
        )
        A.col_indices = new_col_indices
        # Update explicit CSR arrays from the new CSC
        A.rowptr = new_AT.colptr
        A.colval = new_AT.rowval
        A.nzval = new_AT.nzval
        A.ncols_compressed = length(new_col_indices)
    end

    # Recompute structural hash after modification
    A.structural_hash = compute_structural_hash(A.row_partition, A.col_indices, A.rowptr, A.colval, A.backend.comm)

    # Invalidate cached transpose bidirectionally
    _invalidate_cached_transpose!(A)

    return src
end

"""
    Base.setindex!(A::HPCSparseMatrix{T}, src::HPCMatrix{T}, ::Colon, col_idx::HPCVector{Int}) where T

Set `A[:, col_idx] = src` where all rows and columns selected by HPCVector.
"""
function Base.setindex!(A::HPCSparseMatrix{T}, src::HPCMatrix{T}, ::Colon, col_idx::HPCVector{Int}) where T
    A[1:size(A, 1), col_idx] = src
    return src
end

"""
    Base.setindex!(A::HPCSparseMatrix{T}, src::HPCVector{T}, row_idx::HPCVector{Int}, j::Integer) where T

Set `A[row_idx, j] = src` where rows are selected by HPCVector and a single column.
The `src` must have partition matching `row_idx.partition`.
"""
function Base.setindex!(A::HPCSparseMatrix{T,Ti,Bk}, src::HPCVector{T}, row_idx::HPCVector{Int}, j::Integer) where {T,Ti,Bk<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    m, n = size(A)

    # Validate column index
    if j < 1 || j > n
        error("HPCSparseMatrix column index out of bounds: $j, ncols=$n")
    end

    # Validate partitions match
    if src.partition != row_idx.partition
        error("HPCSparseMatrix setindex!: src partition must match row_idx partition")
    end

    # Validate row indices (local check)
    for i in row_idx.v
        if i < 1 || i > m
            error("HPCSparseMatrix row index out of bounds: $i, nrows=$m")
        end
    end

    local_row_indices = row_idx.v
    local_src = src.v

    # Group data by destination rank
    send_to = [Tuple{Int, T}[] for _ in 1:nranks]  # (global_row, value)

    for (pos, global_row) in enumerate(local_row_indices)
        owner = searchsortedlast(A.row_partition, global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(send_to[owner + 1], (global_row, local_src[pos]))
    end

    # Exchange counts via Alltoall
    send_counts = Int32[length(send_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = comm_alltoall(comm, MPI.UBuffer(send_counts, 1))

    # Send row indices to destination ranks
    struct_send_reqs = MPI.Request[]
    struct_send_bufs = Dict{Int, Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            indices = [t[1] for t in send_to[r + 1]]
            struct_send_bufs[r] = indices
            push!(struct_send_reqs, comm_isend(comm, indices, r, 124))
        end
    end

    # Receive row indices from source ranks
    struct_recv_bufs = Dict{Int, Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(struct_recv_reqs, comm_irecv!(comm, buf, r, 124))
            struct_recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, struct_recv_reqs)
    comm_waitall(comm, struct_send_reqs)

    # Send values to destination ranks
    data_send_reqs = MPI.Request[]
    data_send_bufs = Dict{Int, Vector{T}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            values = [t[2] for t in send_to[r + 1]]
            data_send_bufs[r] = values
            push!(data_send_reqs, comm_isend(comm, values, r, 125))
        end
    end

    # Receive values from source ranks
    data_recv_bufs = Dict{Int, Vector{T}}()
    data_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{T}(undef, recv_counts[r + 1])
            push!(data_recv_reqs, comm_irecv!(comm, buf, r, 125))
            data_recv_bufs[r] = buf
        end
    end

    comm_waitall(comm, data_recv_reqs)

    # Build insertions for structural modification
    insertions = Vector{Tuple{Int,Int,T}}()

    # From local assignments
    for (global_row, value) in send_to[rank + 1]
        push!(insertions, (global_row, j, value))
    end

    # From received assignments
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = struct_recv_bufs[r]
            values = data_recv_bufs[r]
            for (k, global_row) in enumerate(indices)
                push!(insertions, (global_row, j, values[k]))
            end
        end
    end

    comm_waitall(comm, data_send_reqs)

    # Apply insertions using the helper function
    if !isempty(insertions)
        row_offset = A.row_partition[rank + 1]
        new_AT, new_col_indices = _rebuild_AT_with_insertions(
            _get_csc(A), A.col_indices, insertions, row_offset
        )
        A.col_indices = new_col_indices
        # Update explicit CSR arrays from the new CSC
        A.rowptr = new_AT.colptr
        A.colval = new_AT.rowval
        A.nzval = new_AT.nzval
        A.ncols_compressed = length(new_col_indices)
    end

    # Recompute structural hash after modification
    A.structural_hash = compute_structural_hash(A.row_partition, A.col_indices, A.rowptr, A.colval, A.backend.comm)

    # Invalidate cached transpose bidirectionally
    _invalidate_cached_transpose!(A)

    return src
end

"""
    Base.setindex!(A::HPCSparseMatrix{T}, src::HPCVector{T}, i::Integer, col_idx::HPCVector{Int}) where T

Set `A[i, col_idx] = src` where a single row and columns selected by HPCVector.
The `src` must have partition matching `col_idx.partition`.
"""
function Base.setindex!(A::HPCSparseMatrix{T,Ti,Bk}, src::HPCVector{T}, i::Integer, col_idx::HPCVector{Int}) where {T,Ti,Bk<:HPCBackend}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    m, n = size(A)

    # Validate row index
    if i < 1 || i > m
        error("HPCSparseMatrix row index out of bounds: $i, nrows=$m")
    end

    # Validate partitions match
    if src.partition != col_idx.partition
        error("HPCSparseMatrix setindex!: src partition must match col_idx partition")
    end

    # Validate column indices (local check)
    for j in col_idx.v
        if j < 1 || j > n
            error("HPCSparseMatrix column index out of bounds: $j, ncols=$n")
        end
    end

    # Find owner of row i
    owner = searchsortedlast(A.row_partition, i) - 1
    if owner >= nranks
        owner = nranks - 1
    end

    # Gather col_idx and src to all ranks (needed because only owner updates)
    col_indices = _gather_vector_to_all(col_idx, comm)
    src_values = _gather_vector_to_all(src, comm)

    # Only owner applies insertions
    if rank == owner
        insertions = Vector{Tuple{Int,Int,T}}()
        for (k, global_col) in enumerate(col_indices)
            push!(insertions, (i, global_col, src_values[k]))
        end

        if !isempty(insertions)
            row_offset = A.row_partition[rank + 1]
            new_AT, new_col_indices = _rebuild_AT_with_insertions(
                _get_csc(A), A.col_indices, insertions, row_offset
            )
            A.col_indices = new_col_indices
            # Update explicit CSR arrays from the new CSC
            A.rowptr = new_AT.colptr
            A.colval = new_AT.rowval
            A.nzval = new_AT.nzval
            A.ncols_compressed = length(new_col_indices)
        end
    end

    # Recompute structural hash after modification
    A.structural_hash = compute_structural_hash(A.row_partition, A.col_indices, A.rowptr, A.colval, A.backend.comm)

    # Invalidate cached transpose bidirectionally
    _invalidate_cached_transpose!(A)

    return src
end
