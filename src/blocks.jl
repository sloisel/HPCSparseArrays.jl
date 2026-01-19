# ============================================================================
# Block Matrix Primitives - cat, hcat, vcat for distributed matrices
# ============================================================================

# Note: uniform_partition is defined in HPCSparseArrays.jl (included before this file)

# ============================================================================
# Core cat implementation for HPCSparseMatrix
# ============================================================================

"""
    Base.cat(As::HPCSparseMatrix...; dims)

Concatenate HPCSparseMatrix matrices.

# Arguments
- `dims::Integer`: Concatenate along dimension `dims` (1=vertical, 2=horizontal)
- `dims::Tuple{Integer,Integer}`: Arrange matrices in a grid of shape `dims=(nrows, ncols)`

# Examples
```julia
cat(A, B; dims=1)           # vcat: stack A on top of B
cat(A, B; dims=2)           # hcat: place A left of B
cat(A, B, C, D; dims=(2,2)) # 2×2 block matrix [A B; C D]
```

This is a distributed implementation that only gathers the rows each rank needs
for its local output, rather than gathering all data to all ranks.
"""
function Base.cat(As::HPCSparseMatrix{T,Ti,Bk}...; dims) where {T,Ti,Bk<:HPCBackend}
    isempty(As) && error("cat requires at least one matrix")
    length(As) == 1 && return copy(As[1])

    # Normalize dims to (nblock_rows, nblock_cols)
    if dims isa Integer
        if dims == 1
            # vcat: n matrices stacked vertically
            block_shape = (length(As), 1)
        elseif dims == 2
            # hcat: n matrices side by side
            block_shape = (1, length(As))
        else
            error("For dims::Integer, only dims=1 (vcat) or dims=2 (hcat) supported")
        end
    elseif dims isa Tuple{Integer, Integer}
        block_shape = dims
    else
        error("dims must be Integer or Tuple{Integer,Integer}")
    end

    nblock_rows, nblock_cols = block_shape
    nblock_rows * nblock_cols == length(As) || error("Number of matrices ($(length(As))) must equal prod(dims) = $(nblock_rows * nblock_cols)")

    # Arrange matrices into grid (row-major order)
    # blocks[i,j] is the matrix at block position (i,j)
    # Use permutedims instead of ' to avoid transposing the matrix elements
    blocks = permutedims(reshape(collect(As), (nblock_cols, nblock_rows)), (2, 1))

    # Validate block sizes: blocks in same row must have same #rows, same column must have same #cols
    block_row_heights = [size(blocks[i, 1], 1) for i in 1:nblock_rows]
    block_col_widths = [size(blocks[1, j], 2) for j in 1:nblock_cols]

    for i in 1:nblock_rows
        for j in 1:nblock_cols
            m, n = size(blocks[i, j])
            m == block_row_heights[i] || error("Block ($i,$j) has $m rows but expected $(block_row_heights[i])")
            n == block_col_widths[j] || error("Block ($i,$j) has $n cols but expected $(block_col_widths[j])")
        end
    end

    # Compute total dimensions
    total_rows = sum(block_row_heights)
    total_cols = sum(block_col_widths)

    # Row and column offsets for each block row/column
    row_offsets = [0; cumsum(block_row_heights[1:end-1])]
    col_offsets = [0; cumsum(block_col_widths[1:end-1])]

    backend = As[1].backend
    comm = backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    # Step 1: Compute output row partition
    rows_per_rank = div(total_rows, nranks)
    row_remainder = mod(total_rows, nranks)
    output_row_partition = Vector{Int}(undef, nranks + 1)
    output_row_partition[1] = 1
    for r in 1:nranks
        extra = r <= row_remainder ? 1 : 0
        output_row_partition[r+1] = output_row_partition[r] + rows_per_rank + extra
    end

    my_out_row_start = output_row_partition[rank+1]
    my_out_row_end = output_row_partition[rank+2] - 1
    local_nrows = my_out_row_end - my_out_row_start + 1

    # Step 2: For each block row that overlaps our output, gather needed rows
    # Ti is from the function's type parameter (matches the input matrices)
    local_I = Ti[]
    local_J = Ti[]
    local_V = T[]

    for bi in 1:nblock_rows
        block_row_start = row_offsets[bi] + 1
        block_row_end = row_offsets[bi] + block_row_heights[bi]

        # Determine rows we need from this block row (in block-local coordinates)
        # NOTE: ALL ranks must call _gather_rows_from_sparse for EVERY block to
        # participate in MPI collectives. Pass empty array if no overlap.
        has_overlap = !(block_row_end < my_out_row_start || block_row_start > my_out_row_end)

        if has_overlap
            first_row_in_block = max(1, my_out_row_start - row_offsets[bi])
            last_row_in_block = min(block_row_heights[bi], my_out_row_end - row_offsets[bi])
            rows_needed = collect(first_row_in_block:last_row_in_block)
        else
            rows_needed = Int[]
        end

        # Get rows from each block in this block row
        for bj in 1:nblock_cols
            A = blocks[bi, bj]
            col_offset = col_offsets[bj]

            # Gather the needed rows from this block (all ranks must call this!)
            triplets = _gather_rows_from_sparse(A, rows_needed)

            # Add triplets with offsets applied
            # Use let block to capture loop variables and avoid boxing overhead
            let row_off = row_offsets[bi], out_start = my_out_row_start, col_off = col_offset
                for (row_in_block, col_in_block, val) in triplets
                    push!(local_I, row_off + row_in_block - out_start + 1)
                    push!(local_J, col_off + col_in_block)
                    push!(local_V, val)
                end
            end
        end
    end

    # Step 3: Build M^T directly as CSC (swap I↔J), then wrap in lazy transpose for CSR
    # This avoids an unnecessary physical transpose operation
    AT_local = isempty(local_I) ?
        SparseMatrixCSC{T,Ti}(total_cols, local_nrows, ones(Ti, local_nrows + 1), Ti[], T[]) :
        SparseMatrixCSC{T,Ti}(sparse(local_J, local_I, local_V, total_cols, local_nrows))

    result = HPCSparseMatrix_local(transpose(AT_local), backend)

    # Convert to target backend (no-op for CPU, copies for GPU)
    return to_backend(result, backend)
end

# ============================================================================
# hcat and vcat call cat
# ============================================================================

"""
    Base.hcat(As::HPCSparseMatrix...)

Horizontally concatenate HPCSparseMatrix matrices. Equivalent to `cat(As...; dims=2)`.
"""
Base.hcat(As::HPCSparseMatrix...) = cat(As...; dims=2)

"""
    Base.vcat(As::HPCSparseMatrix...)

Vertically concatenate HPCSparseMatrix matrices. Equivalent to `cat(As...; dims=1)`.
"""
Base.vcat(As::HPCSparseMatrix...) = cat(As...; dims=1)

# ============================================================================
# cat for HPCMatrix (dense)
# ============================================================================

"""
    Base.cat(As::HPCMatrix...; dims)

Concatenate HPCMatrix matrices. Same interface as HPCSparseMatrix version.

This is a distributed implementation that only gathers the rows each rank needs
for its local output, rather than gathering all data to all ranks.
"""
function Base.cat(As::HPCMatrix{T,B}...; dims) where {T, B<:HPCBackend}
    isempty(As) && error("cat requires at least one matrix")
    length(As) == 1 && return copy(As[1])

    backend = As[1].backend

    # Normalize dims
    if dims isa Integer
        if dims == 1
            block_shape = (length(As), 1)
        elseif dims == 2
            block_shape = (1, length(As))
        else
            error("For dims::Integer, only dims=1 or dims=2 supported")
        end
    elseif dims isa Tuple{Integer, Integer}
        block_shape = dims
    else
        error("dims must be Integer or Tuple{Integer,Integer}")
    end

    nblock_rows, nblock_cols = block_shape
    nblock_rows * nblock_cols == length(As) || error("Number of matrices must equal prod(dims)")

    # Use permutedims instead of ' to avoid transposing the matrix elements
    blocks = permutedims(reshape(collect(As), (nblock_cols, nblock_rows)), (2, 1))

    # Validate sizes
    block_row_heights = [size(blocks[i, 1], 1) for i in 1:nblock_rows]
    block_col_widths = [size(blocks[1, j], 2) for j in 1:nblock_cols]

    for i in 1:nblock_rows, j in 1:nblock_cols
        m, n = size(blocks[i, j])
        m == block_row_heights[i] || error("Block ($i,$j) row size mismatch")
        n == block_col_widths[j] || error("Block ($i,$j) col size mismatch")
    end

    total_rows = sum(block_row_heights)
    total_cols = sum(block_col_widths)
    row_offsets = [0; cumsum(block_row_heights[1:end-1])]
    col_offsets = [0; cumsum(block_col_widths[1:end-1])]

    comm = backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    # Step 1: Compute output row partition
    rows_per_rank = div(total_rows, nranks)
    row_remainder = mod(total_rows, nranks)
    output_row_partition = Vector{Int}(undef, nranks + 1)
    output_row_partition[1] = 1
    for r in 1:nranks
        extra = r <= row_remainder ? 1 : 0
        output_row_partition[r+1] = output_row_partition[r] + rows_per_rank + extra
    end

    my_out_row_start = output_row_partition[rank+1]
    my_out_row_end = output_row_partition[rank+2] - 1
    local_nrows = my_out_row_end - my_out_row_start + 1

    # Step 2: Allocate local output matrix
    local_matrix = Matrix{T}(undef, local_nrows, total_cols)

    # Step 3: For each block row, gather needed rows (all ranks must participate)
    for bi in 1:nblock_rows
        block_row_start = row_offsets[bi] + 1
        block_row_end = row_offsets[bi] + block_row_heights[bi]

        # Determine rows we need from this block row (in block-local coordinates)
        # NOTE: ALL ranks must call _gather_dense_rows for EVERY block to
        # participate in MPI collectives. Pass empty array if no overlap.
        has_overlap = !(block_row_end < my_out_row_start || block_row_start > my_out_row_end)

        if has_overlap
            first_row_in_block = max(1, my_out_row_start - row_offsets[bi])
            last_row_in_block = min(block_row_heights[bi], my_out_row_end - row_offsets[bi])
            rows_needed = collect(first_row_in_block:last_row_in_block)
        else
            rows_needed = Int[]
        end

        # Get rows from each block in this block row
        for bj in 1:nblock_cols
            A = blocks[bi, bj]
            col_start = col_offsets[bj] + 1
            col_end = col_offsets[bj] + block_col_widths[bj]

            # Gather the needed rows from this block (all ranks must call this!)
            gathered_rows = _gather_dense_rows(A, rows_needed)

            # Place into local matrix (only if we have overlap)
            # Use block copy instead of element-by-element loop to avoid boxing overhead
            if has_overlap && !isempty(rows_needed)
                first_local_row = row_offsets[bi] + first(rows_needed) - my_out_row_start + 1
                last_local_row = row_offsets[bi] + last(rows_needed) - my_out_row_start + 1
                local_matrix[first_local_row:last_local_row, col_start:col_end] = gathered_rows
            end
        end
    end

    # Step 4: Create HPCMatrix from local data
    result = HPCMatrix_local(local_matrix, backend)

    # Convert to target backend (no-op for CPU, copies for GPU)
    return to_backend(result, backend)
end

Base.hcat(As::HPCMatrix...) = cat(As...; dims=2)
Base.vcat(As::HPCMatrix...) = cat(As...; dims=1)

# ============================================================================
# cat for HPCVector
# ============================================================================

"""
    Base.cat(vs::HPCVector...; dims)

Concatenate HPCVector vectors.
- `dims=1`: vertical concatenation (produces longer HPCVector)
- `dims=2`: horizontal concatenation (produces HPCMatrix with vectors as columns)
"""
function Base.cat(vs::HPCVector{T,AV}...; dims) where {T,AV}
    isempty(vs) && error("cat requires at least one vector")

    if dims isa Integer
        if dims == 1
            return _vcat_vectors(vs...)
        elseif dims == 2
            return _hcat_vectors(vs...)
        else
            error("For HPCVector, only dims=1 or dims=2 supported")
        end
    elseif dims isa Tuple{Integer, Integer}
        if dims == (1, 1) && length(vs) == 1
            return copy(vs[1])
        elseif dims[1] == 1
            # Single row of vectors -> hcat
            return _hcat_vectors(vs...)
        elseif dims[2] == 1
            # Single column of vectors -> vcat
            return _vcat_vectors(vs...)
        else
            error("HPCVector cat with dims=$dims not supported (vectors are 1D)")
        end
    else
        error("dims must be Integer or Tuple{Integer,Integer}")
    end
end

"""
    _vcat_target_partition(output_partition::Vector{Int}, offset::Int, vec_len::Int) -> Vector{Int}

Compute the target partition for repartitioning a vector for use in vcat.

The vector starts at position (offset + 1) in the output. Each rank needs elements
from this vector that fall within its output range.
"""
function _vcat_target_partition(output_partition::Vector{Int}, offset::Int, vec_len::Int)
    nranks = length(output_partition) - 1
    target = Vector{Int}(undef, nranks + 1)

    for r in 0:(nranks-1)
        # Rank r owns output indices [output_partition[r+1], output_partition[r+2]-1]
        # From this vector (at offset), it needs indices starting at:
        # max(1, output_partition[r+1] - offset)
        target[r+1] = clamp(output_partition[r+1] - offset, 1, vec_len + 1)
    end
    target[nranks+1] = vec_len + 1

    return target
end

"""
    _vcat_vectors(vs::HPCVector{T}...) where T

Vertically concatenate HPCVector vectors.

Uses `repartition` to redistribute each input vector's elements to the ranks that
need them for the output. This provides plan caching and a fast path when partitions
already align (no communication needed).
"""
function _vcat_vectors(vs::HPCVector{T,B}...) where {T, B<:HPCBackend}
    length(vs) == 1 && return copy(vs[1])

    backend = vs[1].backend
    comm = backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    # Step 1: Compute total length and offsets
    lengths = [length(v) for v in vs]
    total_length = sum(lengths)
    offsets = [0; cumsum(lengths[1:end-1])]

    # Step 2: Compute output partition
    output_partition = uniform_partition(total_length, nranks)

    my_out_start = output_partition[rank+1]
    my_out_end = output_partition[rank+2] - 1
    local_len = my_out_end - my_out_start + 1

    # Step 3: Allocate local output vector (same type as input: CPU or GPU)
    local_v = similar(vs[1].v, local_len)

    # Step 4: For each input vector, repartition and copy elements
    for (vec_idx, v) in enumerate(vs)
        vec_len = lengths[vec_idx]
        offset = offsets[vec_idx]
        vec_start = offset + 1
        vec_end = offset + vec_len

        # Compute target partition for this vector and repartition
        target = _vcat_target_partition(output_partition, offset, vec_len)
        v_repart = repartition(v, target)
        my_v_start = v_repart.partition[rank+1]

        # Check if this vector contributes to my output range
        has_overlap = !(vec_end < my_out_start || vec_start > my_out_end)

        if has_overlap
            # Copy elements from repartitioned vector to output
            first_in_vec = max(1, my_out_start - offset)
            last_in_vec = min(vec_len, my_out_end - offset)
            n_copy = last_in_vec - first_in_vec + 1

            # Use copyto! instead of element-by-element loop to avoid boxing overhead
            dst_start = offset + first_in_vec - my_out_start + 1
            src_start = first_in_vec - my_v_start + 1
            copyto!(local_v, dst_start, v_repart.v, src_start, n_copy)
        end
    end

    return HPCVector_local(local_v, backend)
end

function _hcat_vectors(vs::HPCVector{T,B}...) where {T, B<:HPCBackend}
    backend = vs[1].backend
    comm = backend.comm
    nranks = comm_size(comm)

    # Verify all vectors have same length and partition
    n = length(vs[1])
    row_partition = vs[1].partition
    for (i, v) in enumerate(vs)
        length(v) == n || error("All vectors must have same length for hcat")
        v.partition == row_partition || error("All vectors must have same partition for hcat")
    end

    # Stack local vectors as columns (preserves GPU type)
    local_matrix = hcat([v.v for v in vs]...)
    ncols = length(vs)

    # Compute col partition
    col_partition = uniform_partition(ncols, nranks)

    # Compute structural hash
    structural_hash = compute_dense_structural_hash(row_partition, col_partition, size(local_matrix), comm)

    return HPCMatrix{T,B}(structural_hash, row_partition, col_partition, local_matrix, backend)
end

Base.hcat(vs::HPCVector...) = cat(vs...; dims=2)
Base.vcat(vs::HPCVector...) = cat(vs...; dims=1)

# ============================================================================
# blockdiag for HPCSparseMatrix
# ============================================================================

"""
    blockdiag(As::HPCSparseMatrix...)

Create a block diagonal matrix from the input matrices.

```
blockdiag(A, B, C) = [A 0 0]
                     [0 B 0]
                     [0 0 C]
```

Returns a HPCSparseMatrix.

This is a distributed implementation that only gathers the rows each rank needs
for its local output, rather than gathering all data to all ranks.
"""
function blockdiag(As::HPCSparseMatrix{T,Ti,Bk}...) where {T,Ti,Bk<:HPCBackend}
    isempty(As) && error("blockdiag requires at least one matrix")
    length(As) == 1 && return copy(As[1])

    backend = As[1].backend
    comm = backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    # Step 1: Compute dimensions and offsets (all ranks compute same values)
    row_sizes = [size(A, 1) for A in As]
    col_sizes = [size(A, 2) for A in As]
    total_rows = sum(row_sizes)
    total_cols = sum(col_sizes)
    row_offsets = [0; cumsum(row_sizes[1:end-1])]
    col_offsets = [0; cumsum(col_sizes[1:end-1])]

    # Step 2: Compute output row partition
    rows_per_rank = div(total_rows, nranks)
    row_remainder = mod(total_rows, nranks)
    output_row_partition = Vector{Int}(undef, nranks + 1)
    output_row_partition[1] = 1
    for r in 1:nranks
        extra = r <= row_remainder ? 1 : 0
        output_row_partition[r+1] = output_row_partition[r] + rows_per_rank + extra
    end

    my_out_row_start = output_row_partition[rank+1]
    my_out_row_end = output_row_partition[rank+2] - 1
    local_nrows = my_out_row_end - my_out_row_start + 1

    # Step 3: For each block, determine if it overlaps our output rows
    # and gather the needed rows
    # Ti is from the function's type parameter (matches the input matrices)
    local_I = Ti[]
    local_J = Ti[]
    local_V = T[]

    for (k, A) in enumerate(As)
        block_row_start = row_offsets[k] + 1
        block_row_end = row_offsets[k] + row_sizes[k]
        col_offset = col_offsets[k]

        # Determine rows we need from this block (in block-local coordinates)
        # NOTE: ALL ranks must call _gather_rows_from_sparse for EVERY block to
        # participate in MPI collectives. Pass empty array if no overlap.
        has_overlap = !(block_row_end < my_out_row_start || block_row_start > my_out_row_end)

        if has_overlap
            first_row_in_block = max(1, my_out_row_start - row_offsets[k])
            last_row_in_block = min(row_sizes[k], my_out_row_end - row_offsets[k])
            rows_needed = collect(first_row_in_block:last_row_in_block)
        else
            rows_needed = Int[]
        end

        # Gather these rows from block A (all ranks must call this!)
        triplets = _gather_rows_from_sparse(A, rows_needed)

        # Add triplets with offsets applied
        # Use let block to capture loop variables and avoid boxing overhead
        let row_off = row_offsets[k], out_start = my_out_row_start, col_off = col_offset
            for (row_in_block, col_in_block, val) in triplets
                push!(local_I, row_off + row_in_block - out_start + 1)
                push!(local_J, col_off + col_in_block)
                push!(local_V, val)
            end
        end
    end

    # Step 4: Build M^T directly as CSC (swap I↔J), then wrap in lazy transpose for CSR
    # This avoids an unnecessary physical transpose operation
    AT_local = isempty(local_I) ?
        SparseMatrixCSC{T,Ti}(total_cols, local_nrows, ones(Ti, local_nrows + 1), Ti[], T[]) :
        SparseMatrixCSC{T,Ti}(sparse(local_J, local_I, local_V, total_cols, local_nrows))

    result = HPCSparseMatrix_local(transpose(AT_local), backend)

    # Convert to target backend (no-op for CPU, copies for GPU)
    return to_backend(result, backend)
end
