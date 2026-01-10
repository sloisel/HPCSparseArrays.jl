# HPCVector type and vector operations

using Adapt
using KernelAbstractions

"""
    HPCVector{T, B<:HPCBackend}

A distributed dense vector partitioned across MPI ranks.

# Type Parameters
- `T`: Element type (e.g., `Float64`, `ComplexF64`)
- `B<:HPCBackend`: Backend configuration (device, communication, solver)

# Fields
- `structural_hash::Blake3Hash`: 256-bit Blake3 hash of the partition
- `partition::Vector{Int}`: Partition boundaries, length = nranks + 1 (always on CPU)
- `v::AbstractVector{T}`: Local vector elements owned by this rank
- `backend::B`: The HPC backend configuration
"""
struct HPCVector{T, B<:HPCBackend} <: AbstractVector{T}
    structural_hash::Blake3Hash
    partition::Vector{Int}
    v::AbstractVector{T}
    backend::B
    # Inner constructor that takes all arguments
    function HPCVector{T,B}(hash::Blake3Hash, partition::Vector{Int}, v::AbstractVector{T}, backend::B) where {T, B<:HPCBackend}
        new{T,B}(hash, partition, v, backend)
    end
end

# Type aliases for common backend configurations
const HPCVector_CPU{T} = HPCVector{T, HPCBackend{DeviceCPU, CommMPI, SolverMUMPS}}
const HPCVector_CPU_Serial{T} = HPCVector{T, HPCBackend{DeviceCPU, CommSerial, SolverMUMPS}}

# Convenience constructor that infers B from the backend type
function HPCVector{T}(hash::Blake3Hash, partition::Vector{Int}, v::AbstractVector{T}, backend::B) where {T, B<:HPCBackend}
    HPCVector{T,B}(hash, partition, v, backend)
end

# Constructor that infers T from the vector
function HPCVector(hash::Blake3Hash, partition::Vector{Int}, v::AbstractVector{T}, backend::B) where {T, B<:HPCBackend}
    HPCVector{T,B}(hash, partition, v, backend)
end

# Get the KernelAbstractions backend for a HPCVector (for GPU kernels)
function get_backend(v::HPCVector)
    return KernelAbstractions.get_backend(v.v)
end

# Get the HPCBackend for a HPCVector
get_hpc_backend(v::HPCVector) = v.backend

"""
    HPCVector_local(v_local::AbstractVector{T}, backend::HPCBackend) where T

Create a HPCVector from a local vector on each rank.

Unlike `HPCVector(v_global, backend)` which takes a global vector and partitions it,
this constructor takes only the local portion of the vector that each rank owns.
The partition is computed by gathering the local sizes from all ranks.

# Arguments
- `v_local`: Local vector portion owned by this rank
- `backend`: The HPCBackend configuration (determines communication)

# Example
```julia
backend = backend_cpu_mpi(MPI.COMM_WORLD)
# Rank 0 has [1.0, 2.0], Rank 1 has [3.0, 4.0, 5.0]
v = HPCVector_local([1.0, 2.0], backend)  # on rank 0
v = HPCVector_local([3.0, 4.0, 5.0], backend)  # on rank 1
# Result: distributed vector [1.0, 2.0, 3.0, 4.0, 5.0] with partition [1, 3, 6]
```
"""
function HPCVector_local(v_local::AbstractVector{T}, backend::B) where {T, B<:HPCBackend}
    nranks = comm_size(backend.comm)

    # Gather local sizes from all ranks
    local_size = Int32(length(v_local))
    all_sizes = comm_allgather(backend.comm, local_size)

    # Build partition from sizes
    partition = Vector{Int}(undef, nranks + 1)
    partition[1] = 1
    for r in 1:nranks
        partition[r+1] = partition[r] + all_sizes[r]
    end

    hash = compute_partition_hash(partition)
    # Convert to target device (handles CPU→GPU, GPU→CPU, and same-device cases)
    local_v = _convert_array(v_local, backend.device)
    return HPCVector{T,B}(hash, partition, local_v, backend)
end

"""
    HPCVector(v_global::Vector{T}, backend::HPCBackend; partition=uniform_partition(...)) where T

Create a HPCVector from a global vector, partitioning it across MPI ranks.

Each rank extracts only its local portion from `v_global`, so:

- **Simple usage**: Pass identical `v_global` to all ranks
- **Efficient usage**: Pass a vector with correct `length(v_global)` on all ranks,
  but only populate the elements that each rank owns (other elements are ignored)

# Arguments
- `v_global`: Global vector (identical on all ranks, or at least with local elements populated)
- `backend`: The HPCBackend configuration

# Keyword Arguments
- `partition::Vector{Int}`: Partition boundaries (default: `uniform_partition(length(v_global), nranks)`)

Use `uniform_partition(n, nranks)` to compute custom partitions.

To create a GPU-backed HPCVector, use `adapt(ka_backend, v)` after creation,
or use `HPCVector_local` with a GPU array.
"""
function HPCVector(v_global::Vector{T}, backend::B;
                   partition::Vector{Int}=uniform_partition(length(v_global), comm_size(backend.comm))) where {T, B<:HPCBackend}
    rank = comm_rank(backend.comm)
    local_range = partition[rank + 1]:(partition[rank + 2] - 1)
    local_v_cpu = v_global[local_range]
    # Convert to target device (no-op for CPU, copies to GPU for GPU backends)
    local_v = _convert_array(local_v_cpu, backend.device)

    hash = compute_partition_hash(partition)
    return HPCVector{T,B}(hash, partition, local_v, backend)
end

# Adapt.jl support for converting HPCVector between KernelAbstractions backends (GPU/CPU)
# Note: This adapts the array storage but preserves the HPCBackend
function Adapt.adapt_structure(to, v::HPCVector{T,B}) where {T,B}
    new_v = adapt(to, v.v)
    return HPCVector{T,B}(v.structural_hash, v.partition, new_v, v.backend)
end

# Helper to create output vector with same storage type as reference
# STRICT: Only allows same-type matching. Mismatched types will error.
# This prevents accidental CPU↔GPU mixing.
_create_output_like(::Vector{T}, result::Vector{T}) where T = result
_create_output_like(::AV, result::AV) where {T,AV<:AbstractVector{T}} = result
# Note: No catch-all method for (GPU, CPU) - this is intentional to catch bugs.
# MPI staging (CPU→GPU) is inlined where needed with explicit ternary checks.

# Helper to get CPU copy of vector data (no-op for CPU, copy for GPU)
# Used for MPI communication which always requires CPU buffers
_ensure_cpu(v::Vector) = v
_ensure_cpu(v::AbstractVector) = Array(v)

# Helper to determine if GPU gather kernel should be used
# CPU arrays use loop-based gather; GPU arrays use kernel-based gather
_use_gpu_gather(::Vector) = false
_use_gpu_gather(::AbstractVector) = true

# Helper to copy only a range from vector to CPU (view for CPU, copy range for GPU)
# Used by execute_plan! to avoid copying entire GPU arrays
_copy_range_to_cpu(v::Vector, range::UnitRange) = view(v, range)
_copy_range_to_cpu(v::AbstractVector, range::UnitRange) = Array(view(v, range))

# Helper to copy CPU data to destination buffer (no-op if same, copy if different)
# Used after MPI communication to transfer results back to GPU if needed
_copy_to_output!(dst::Vector{T}, src::Vector{T}) where T = (dst === src || copyto!(dst, src); dst)
_copy_to_output!(dst::AV, src::Vector{T}) where {T,AV<:AbstractVector{T}} = copyto!(dst, src)

# ============================================================================
# GPU Gather Kernel
# ============================================================================

"""
Gather kernel: gathered[dst[i]] = x[src[i]] for i = 1:length(src)
Used to avoid copying entire GPU arrays for VectorPlan execution.
"""
@kernel function _gather_kernel!(gathered, x, src_indices, dst_indices)
    i = @index(Global)
    @inbounds gathered[dst_indices[i]] = x[src_indices[i]]
end

"""
    _execute_gather_gpu!(gathered, x, src_indices_gpu, dst_indices_gpu)

Execute GPU gather using a KernelAbstractions kernel.
"""
function _execute_gather_gpu!(gathered::AbstractVector{T}, x::AbstractVector{T},
                               src_indices::AbstractVector, dst_indices::AbstractVector) where T
    n = length(src_indices)
    if n == 0
        return gathered
    end
    backend = KernelAbstractions.get_backend(gathered)
    kernel = _gather_kernel!(backend)
    kernel(gathered, x, src_indices, dst_indices; ndrange=n)
    return gathered
end

"""
    _to_gpu_indices(gpu_array, cpu_indices)

Convert CPU index array to same device as specified by the device type.
Falls back to CPU if device is CPU.
"""
_to_gpu_indices(::DeviceCPU, indices::Vector{Int}) = indices
_to_gpu_indices(device::AbstractDevice, indices::Vector{Int}) = _to_target_device(indices, device)

"""
    VectorPlan{T,AV}

A communication plan for gathering vector elements needed for A * x.

# Type Parameters
- `T`: Element type
- `AV<:AbstractVector{T}`: Storage type for gathered buffer (matches input HPCVector)

# Fields
- `send_rank_ids::Vector{Int}`: Ranks we send elements to (0-indexed)
- `send_indices::Vector{Vector{Int}}`: For each rank, local indices to send
- `send_bufs::Vector{Vector{T}}`: Pre-allocated CPU send buffers (for MPI)
- `send_reqs::Vector{MPI.Request}`: Pre-allocated send request handles
- `recv_rank_ids::Vector{Int}`: Ranks we receive elements from (0-indexed)
- `recv_bufs::Vector{Vector{T}}`: Pre-allocated CPU receive buffers (for MPI)
- `recv_reqs::Vector{MPI.Request}`: Pre-allocated receive request handles
- `recv_perm::Vector{Vector{Int}}`: For each recv rank, indices into gathered
- `local_src_indices::Vector{Int}`: Source indices for local copy (into x.v)
- `local_dst_indices::Vector{Int}`: Destination indices for local copy (into gathered)
- `gathered::AV`: Pre-allocated buffer for gathered elements (same type as input)
- `gathered_cpu::Vector{T}`: CPU staging buffer (used when AV is GPU array)
"""
mutable struct VectorPlan{T,AV<:AbstractVector{T}}
    send_rank_ids::Vector{Int}
    send_indices::Vector{Vector{Int}}
    send_bufs::Vector{Vector{T}}      # Always CPU for MPI
    send_reqs::Vector{Any}            # MPI.Request or nothing for CommSerial
    recv_rank_ids::Vector{Int}
    recv_bufs::Vector{Vector{T}}      # Always CPU for MPI
    recv_reqs::Vector{Any}            # MPI.Request or nothing for CommSerial
    recv_perm::Vector{Vector{Int}}
    local_src_indices::Vector{Int}
    local_dst_indices::Vector{Int}
    gathered::AV                       # Same type as input vector
    gathered_cpu::Vector{T}            # CPU staging buffer
    # Cached partition hash for result vector (computed lazily on first use)
    result_partition_hash::OptionalBlake3Hash
    result_partition::Union{Nothing, Vector{Int}}
    # Cached GPU buffers for SpMV (lazily allocated on first use)
    cached_gathered_target::Union{Nothing, AbstractVector{T}}  # Gathered in matrix backend
    cached_y_local::Union{Nothing, AbstractVector{T}}          # Result buffer
    # Cached GPU index arrays for gather kernel (lazily allocated on first use)
    cached_local_src_gpu::Union{Nothing, AbstractVector{Int}}
    cached_local_dst_gpu::Union{Nothing, AbstractVector{Int}}
end

# Type alias for CPU-backed VectorPlan (backwards compatible)
const VectorPlan_CPU{T} = VectorPlan{T,Vector{T}}

"""
    VectorPlan(target_partition::Vector{Int}, source::HPCVector{T,AV}) where {T,AV}

Create a communication plan to gather elements from `source` according to `target_partition`.
This allows binary operations between vectors with different partitions.

After executing, `plan.gathered` contains `source[target_partition[rank+1]:target_partition[rank+2]-1]`.

The gathered buffer will have the same storage type as the source vector (CPU or GPU).
MPI communication always uses CPU staging buffers.
"""
function VectorPlan(target_partition::Vector{Int}, source::HPCVector{T,B}) where {T,B<:HPCBackend}
    # Get actual array type from the vector for VectorPlan type parameter
    AV = typeof(source.v)
    comm = source.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    # Indices this rank needs from source (contiguous range)
    my_start = target_partition[rank+1]
    my_end = target_partition[rank+2] - 1
    col_indices = collect(my_start:my_end)
    n_gathered = length(col_indices)

    my_x_start = source.partition[rank+1]

    # Step 1: Group col_indices by owner rank in source's partition
    needed_from = [Tuple{Int,Int}[] for _ in 1:nranks]
    for (dst_idx, global_idx) in enumerate(col_indices)
        owner = searchsortedlast(source.partition, global_idx) - 1
        # Clamp to handle edge case where index equals last partition boundary
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
    struct_send_reqs = Any[]
    recv_rank_ids = Int[]
    recv_perm_map = Dict{Int,Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r+1] > 0 && r != rank
            push!(recv_rank_ids, r)
            indices = [t[1] for t in needed_from[r+1]]
            dst_indices = [t[2] for t in needed_from[r+1]]
            recv_perm_map[r] = dst_indices
            struct_send_bufs[r] = indices
            req = comm_isend(comm, indices, r, 22)
            push!(struct_send_reqs, req)
        end
    end

    # Step 4: Receive requests from other ranks
    send_rank_ids = Int[]
    struct_recv_bufs = Dict{Int,Vector{Int}}()
    struct_recv_reqs = Any[]

    for r in 0:(nranks-1)
        if recv_counts[r+1] > 0 && r != rank
            push!(send_rank_ids, r)
            buf = Vector{Int}(undef, recv_counts[r+1])
            req = comm_irecv!(comm, buf, r, 22)
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

    # Step 6: Handle local elements (elements we own in source)
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
    AV = typeof(source.v)
    send_bufs = [Vector{T}(undef, length(inds)) for inds in send_indices_final]
    recv_bufs = [Vector{T}(undef, send_counts[r+1]) for r in recv_rank_ids]
    send_reqs = Vector{Any}(undef, length(send_rank_ids))
    recv_reqs = Vector{Any}(undef, length(recv_rank_ids))

    # CPU staging buffer (always needed for MPI)
    gathered_cpu = Vector{T}(undef, n_gathered)

    # Gathered buffer matches source vector's storage type
    # Use similar() to create same type, or adapt for GPU arrays
    gathered = similar(source.v, n_gathered)

    return VectorPlan{T,AV}(
        send_rank_ids, send_indices_final, send_bufs, send_reqs,
        recv_rank_ids, recv_bufs, recv_reqs, recv_perm_final,
        local_src_indices, local_dst_indices, gathered, gathered_cpu,
        nothing, nothing,  # result_partition_hash, result_partition (computed lazily)
        nothing, nothing,  # cached_gathered_target, cached_y_local (for SpMV)
        nothing, nothing   # cached_local_src_gpu, cached_local_dst_gpu (for GPU gather)
    )
end

"""
    execute_plan!(plan::VectorPlan{T,AV}, x::HPCVector{T,B}) where {T,AV,B}

Execute a vector communication plan to gather elements from x.
Returns plan.gathered containing x[A.col_indices] for the associated matrix A.

Works uniformly for CPU and GPU vectors - MPI communication always uses CPU
buffers, with automatic staging for GPU arrays.

GPU optimization: When no MPI communication is needed (all data is local),
uses a GPU gather kernel to avoid copying entire arrays to CPU.
"""
function execute_plan!(plan::VectorPlan{T,AV}, x::HPCVector{T,B}) where {T,AV,B}
    comm = x.backend.comm

    # Check if we can use GPU-optimized path (no MPI communication needed)
    no_mpi_needed = isempty(plan.send_rank_ids) && isempty(plan.recv_rank_ids)
    device = x.backend.device

    if no_mpi_needed && _use_gpu_gather(x.v) && !isempty(plan.local_src_indices)
        # GPU path: Use gather kernel directly, no CPU round-trip
        # Cache GPU index arrays on first use
        if plan.cached_local_src_gpu === nothing
            plan.cached_local_src_gpu = _to_gpu_indices(device, plan.local_src_indices)
        end
        if plan.cached_local_dst_gpu === nothing
            plan.cached_local_dst_gpu = _to_gpu_indices(device, plan.local_dst_indices)
        end

        # Execute GPU gather directly into plan.gathered
        _execute_gather_gpu!(plan.gathered, x.v,
                            plan.cached_local_src_gpu, plan.cached_local_dst_gpu)

        # Note: gathered_cpu is NOT populated here for performance.
        # Callers that need CPU data should use ensure_gathered_cpu!(plan) below.

        return plan.gathered
    end

    # CPU path: Required for MPI communication or CPU vectors
    # Get CPU data for MPI operations (no-op for CPU, copy for GPU)
    x_cpu = _ensure_cpu(x.v)

    # Step 1: Copy local values to CPU staging buffer
    @inbounds for i in eachindex(plan.local_src_indices, plan.local_dst_indices)
        plan.gathered_cpu[plan.local_dst_indices[i]] = x_cpu[plan.local_src_indices[i]]
    end

    # Step 2: Fill send buffers and send
    @inbounds for i in eachindex(plan.send_rank_ids)
        r = plan.send_rank_ids[i]
        send_idx = plan.send_indices[i]
        buf = plan.send_bufs[i]
        for k in eachindex(send_idx)
            buf[k] = x_cpu[send_idx[k]]
        end
        plan.send_reqs[i] = comm_isend(comm, buf, r, 21)
    end

    # Step 3: Receive values to CPU buffers
    @inbounds for i in eachindex(plan.recv_rank_ids)
        plan.recv_reqs[i] = comm_irecv!(comm, plan.recv_bufs[i], plan.recv_rank_ids[i], 21)
    end

    comm_waitall(comm, plan.recv_reqs)

    # Step 4: Scatter received values into CPU staging buffer
    @inbounds for i in eachindex(plan.recv_rank_ids)
        perm = plan.recv_perm[i]
        buf = plan.recv_bufs[i]
        for k in eachindex(perm)
            plan.gathered_cpu[perm[k]] = buf[k]
        end
    end

    comm_waitall(comm, plan.send_reqs)

    # Step 5: Copy to output buffer (no-op if CPU, copy if GPU)
    _copy_to_output!(plan.gathered, plan.gathered_cpu)

    return plan.gathered
end


# ============================================================================
# VectorRepartitionPlan: Repartition a HPCVector to a new partition
# ============================================================================

"""
    VectorRepartitionPlan{T}

Communication plan for repartitioning a HPCVector to a new partition.

# Fields
- `send_rank_ids::Vector{Int}`: Ranks we send elements to (0-indexed)
- `send_ranges::Vector{UnitRange{Int}}`: For each rank, range of local indices to send
- `send_bufs::Vector{Vector{T}}`: Pre-allocated send buffers
- `send_reqs::Vector{Any}`: Pre-allocated send request handles (MPI.Request or nothing for CommSerial)
- `recv_rank_ids::Vector{Int}`: Ranks we receive elements from (0-indexed)
- `recv_counts::Vector{Int}`: Number of elements to receive from each rank
- `recv_bufs::Vector{Vector{T}}`: Pre-allocated receive buffers
- `recv_reqs::Vector{Any}`: Pre-allocated receive request handles (MPI.Request or nothing for CommSerial)
- `recv_offsets::Vector{Int}`: Offset into result for each recv rank
- `local_src_range::UnitRange{Int}`: Source range for local copy
- `local_dst_offset::Int`: Destination offset for local copy
- `result_partition::Vector{Int}`: Target partition (copy of p)
- `result_partition_hash::Blake3Hash`: Hash of target partition
- `result_local_size::Int`: Number of elements this rank owns after repartition
"""
mutable struct VectorRepartitionPlan{T}
    send_rank_ids::Vector{Int}
    send_ranges::Vector{UnitRange{Int}}
    send_bufs::Vector{Vector{T}}
    send_reqs::Vector{Any}
    recv_rank_ids::Vector{Int}
    recv_counts::Vector{Int}
    recv_bufs::Vector{Vector{T}}
    recv_reqs::Vector{Any}
    recv_offsets::Vector{Int}
    local_src_range::UnitRange{Int}
    local_dst_offset::Int
    result_partition::Vector{Int}
    result_partition_hash::Blake3Hash
    result_local_size::Int
end

"""
    VectorRepartitionPlan(x::HPCVector{T,B}, p::Vector{Int}) where {T,B}

Create a communication plan to repartition `x` to have partition `p`.

The plan computes:
1. Which elements to send to each rank based on partition overlap
2. Which elements to receive from each rank
3. Pre-allocates all buffers for allocation-free execution
4. Computes the result partition hash eagerly
"""
function VectorRepartitionPlan(x::HPCVector{T,B}, p::Vector{Int}) where {T,B}
    comm = x.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    # Source partition info
    src_start = x.partition[rank+1]
    src_end = x.partition[rank+2] - 1

    # Target partition info
    dst_start = p[rank+1]
    dst_end = p[rank+2] - 1
    result_local_size = max(0, dst_end - dst_start + 1)

    # Compute result hash eagerly
    result_partition_hash = compute_partition_hash(p)

    # Step 1: Determine which elements we send to each rank
    # For each destination rank r, compute overlap of our elements [src_start, src_end]
    # with rank r's target range [p[r+1], p[r+2]-1]
    send_ranges_map = Dict{Int, UnitRange{Int}}()
    for r in 0:(nranks-1)
        r_start = p[r+1]
        r_end = p[r+2] - 1
        if r_end < r_start
            continue  # rank r has no elements in target partition
        end
        # Intersection of our elements with rank r's target
        overlap_start = max(src_start, r_start)
        overlap_end = min(src_end, r_end)
        if overlap_start <= overlap_end
            # Convert to local indices in x.v
            local_start = overlap_start - src_start + 1
            local_end = overlap_end - src_start + 1
            send_ranges_map[r] = local_start:local_end
        end
    end

    # Step 2: Exchange counts via Alltoall
    send_counts = Int32[haskey(send_ranges_map, r) ? length(send_ranges_map[r]) : 0 for r in 0:(nranks-1)]
    recv_counts_raw = comm_alltoall(comm, MPI.UBuffer(send_counts, 1))

    # Step 3: Build send/recv structures
    send_rank_ids = Int[]
    send_ranges = UnitRange{Int}[]
    recv_rank_ids = Int[]
    recv_counts = Int[]
    recv_offsets = Int[]

    local_src_range = 1:0  # empty range
    local_dst_offset = 0

    # Handle local copy separately
    if haskey(send_ranges_map, rank)
        local_src_range = send_ranges_map[rank]
        # Compute destination offset: where do these elements go in the result?
        # The elements at global indices [src_start + local_src_range.start - 1, ...]
        # go to local indices starting at (global_start - dst_start + 1)
        global_start = src_start + local_src_range.start - 1
        local_dst_offset = global_start - dst_start + 1
    end

    # Build send arrays (excluding local)
    for r in 0:(nranks-1)
        if haskey(send_ranges_map, r) && r != rank
            push!(send_rank_ids, r)
            push!(send_ranges, send_ranges_map[r])
        end
    end

    # Build recv arrays (excluding local)
    # For each rank r that sends to us, compute where their data goes in our result
    for r in 0:(nranks-1)
        if recv_counts_raw[r+1] > 0 && r != rank
            push!(recv_rank_ids, r)
            push!(recv_counts, recv_counts_raw[r+1])

            # Elements from rank r: their source range is [x.partition[r+1], x.partition[r+2]-1]
            # intersected with our target range [dst_start, dst_end]
            r_src_start = x.partition[r+1]
            r_src_end = x.partition[r+2] - 1
            overlap_start = max(r_src_start, dst_start)
            # The offset in our result where these elements go
            offset = overlap_start - dst_start + 1
            push!(recv_offsets, offset)
        end
    end

    # Pre-allocate buffers
    send_bufs = [Vector{T}(undef, length(r)) for r in send_ranges]
    recv_bufs = [Vector{T}(undef, c) for c in recv_counts]
    send_reqs = Vector{Any}(undef, length(send_rank_ids))
    recv_reqs = Vector{Any}(undef, length(recv_rank_ids))

    return VectorRepartitionPlan{T}(
        send_rank_ids, send_ranges, send_bufs, send_reqs,
        recv_rank_ids, recv_counts, recv_bufs, recv_reqs, recv_offsets,
        local_src_range, local_dst_offset,
        copy(p), result_partition_hash, result_local_size
    )
end

"""
    execute_plan!(plan::VectorRepartitionPlan{T}, x::HPCVector{T,B}) where {T,B}

Execute a vector repartition plan to redistribute elements from x to a new partition.
Returns a new HPCVector with the target partition, preserving the array type (CPU/GPU) and backend.
"""
function execute_plan!(plan::VectorRepartitionPlan{T}, x::HPCVector{T,B}) where {T,B}
    comm = x.backend.comm

    # Get CPU version of input for MPI operations
    x_cpu = _ensure_cpu(x.v)

    # Allocate CPU result buffer (MPI requires CPU)
    result_cpu = Vector{T}(undef, plan.result_local_size)

    # Step 1: Local copy (from CPU staging)
    if !isempty(plan.local_src_range)
        @inbounds for (i, src_i) in enumerate(plan.local_src_range)
            result_cpu[plan.local_dst_offset + i - 1] = x_cpu[src_i]
        end
    end

    # Step 2: Fill send buffers and send (from CPU staging)
    @inbounds for i in eachindex(plan.send_rank_ids)
        r = plan.send_rank_ids[i]
        range = plan.send_ranges[i]
        buf = plan.send_bufs[i]
        for (k, src_k) in enumerate(range)
            buf[k] = x_cpu[src_k]
        end
        plan.send_reqs[i] = comm_isend(comm, buf, r, 92)
    end

    # Step 3: Post receives
    @inbounds for i in eachindex(plan.recv_rank_ids)
        plan.recv_reqs[i] = comm_irecv!(comm, plan.recv_bufs[i], plan.recv_rank_ids[i], 92)
    end

    comm_waitall(comm, plan.recv_reqs)

    # Step 4: Scatter received values into result
    @inbounds for i in eachindex(plan.recv_rank_ids)
        offset = plan.recv_offsets[i]
        buf = plan.recv_bufs[i]
        for k in eachindex(buf)
            result_cpu[offset + k - 1] = buf[k]
        end
    end

    comm_waitall(comm, plan.send_reqs)

    # Copy result back to target array type using dispatch (no type checks)
    result_v = _values_to_backend(result_cpu, x.v)

    return HPCVector{T,B}(plan.result_partition_hash, plan.result_partition, result_v, x.backend)
end

"""
    get_repartition_plan(x::HPCVector{T}, p::Vector{Int}) where T

Get a memoized VectorRepartitionPlan for repartitioning `x` to partition `p`.
The plan is cached based on the structural hash of x and the target partition hash.
"""
function get_repartition_plan(x::HPCVector{T}, p::Vector{Int}) where T
    target_hash = compute_partition_hash(p)
    key = (x.structural_hash, target_hash, T)
    if haskey(_repartition_plan_cache, key)
        return _repartition_plan_cache[key]::VectorRepartitionPlan{T}
    end
    plan = VectorRepartitionPlan(x, p)
    _repartition_plan_cache[key] = plan
    return plan
end

"""
    repartition(x::HPCVector{T}, p::Vector{Int}) where T

Redistribute a HPCVector to a new partition `p`.

The partition `p` must be a valid partition vector of length `nranks + 1` with
`p[1] == 1` and `p[end] == length(x) + 1`.

Returns a new HPCVector with the same data but `partition == p`.

# Example
```julia
v = HPCVector([1.0, 2.0, 3.0, 4.0])  # uniform partition
new_partition = [1, 2, 5]  # rank 0 gets 1 element, rank 1 gets 3
v_repart = repartition(v, new_partition)
```
"""
function repartition(x::HPCVector{T}, p::Vector{Int}) where T
    # Fast path: partition unchanged (identity check first, then element-wise)
    if x.partition === p || x.partition == p
        return x
    end

    plan = get_repartition_plan(x, p)
    return execute_plan!(plan, x)
end

# Vector operations: conj, transpose, adjoint

"""
    Base.conj(v::HPCVector{T,B}) where {T,B}

Return a new HPCVector with conjugated values.
"""
function Base.conj(v::HPCVector{T,B}) where {T,B}
    return HPCVector{T,B}(v.structural_hash, v.partition, conj.(v.v), v.backend)
end

"""
    Base.transpose(v::HPCVector{T}) where T

Return a lazy transpose wrapper around v.
"""
Base.transpose(v::HPCVector{T}) where T = Transpose(v)

"""
    Base.adjoint(v::HPCVector{T}) where T

Return transpose(conj(v)), i.e., the conjugate transpose.
The conj(v) is materialized.
"""
Base.adjoint(v::HPCVector{T}) where T = transpose(conj(v))

# Vector norms and reductions

"""
    LinearAlgebra.norm(v::HPCVector{T,B}, p::Real=2) where {T,B}

Compute the p-norm of the distributed vector v.
- `p=2` (default): Euclidean norm (sqrt of sum of squared absolute values)
- `p=1`: Sum of absolute values
- `p=Inf`: Maximum absolute value
"""
function LinearAlgebra.norm(v::HPCVector{T,B}, p::Real=2) where {T,B}
    comm = v.backend.comm

    if p == 2
        # Use BLAS-optimized local norm, then reduce
        local_nrm = isempty(v.v) ? zero(real(T)) : norm(v.v)
        local_sum = local_nrm * local_nrm
        global_sum = comm_allreduce(comm, local_sum, +)
        return sqrt(global_sum)
    elseif p == 1
        # Use BLAS-optimized local norm(v, 1) = asum
        local_sum = isempty(v.v) ? zero(real(T)) : norm(v.v, 1)
        return comm_allreduce(comm, local_sum, +)
    elseif p == Inf
        local_max = isempty(v.v) ? zero(real(T)) : norm(v.v, Inf)
        return comm_allreduce(comm, local_max, max)
    else
        # General p-norm - no BLAS optimization available
        local_sum = isempty(v.v) ? zero(real(T)) : sum(x -> abs(x)^p, v.v)
        global_sum = comm_allreduce(comm, local_sum, +)
        return global_sum^(1 / p)
    end
end

"""
    LinearAlgebra.dot(x::HPCVector{T,B}, y::HPCVector{T,B}) where {T,B}

Compute the dot product of two distributed vectors.

This is a collective operation. If the vectors have different partitions,
the second vector is redistributed to match the first vector's partition.

# Example
```julia
backend = backend_cpu_mpi(MPI.COMM_WORLD)
x = HPCVector(rand(10), backend)
y = HPCVector(rand(10), backend)
d = dot(x, y)
```
"""
function LinearAlgebra.dot(x::HPCVector{T,B}, y::HPCVector{T,B}) where {T,B}
    assert_backends_compatible(x.backend, y.backend)
    comm = x.backend.comm

    # If partitions match (compare hashes for efficiency), use local dot product directly
    if x.structural_hash == y.structural_hash
        local_dot = dot(x.v, y.v)
        return comm_allreduce(comm, local_dot, +)
    else
        # Redistribute y to match x's partition using repartition
        y_aligned = repartition(y, x.partition)
        local_dot = dot(x.v, y_aligned.v)
        return comm_allreduce(comm, local_dot, +)
    end
end

"""
    Base.maximum(v::HPCVector{T,B}) where {T,B}

Compute the maximum element of the distributed vector.
"""
function Base.maximum(v::HPCVector{T,B}) where {T,B}
    comm = v.backend.comm
    local_max = isempty(v.v) ? typemin(real(T)) : maximum(real, v.v)
    return comm_allreduce(comm, local_max, max)
end

"""
    Base.minimum(v::HPCVector{T,B}) where {T,B}

Compute the minimum element of the distributed vector.
"""
function Base.minimum(v::HPCVector{T,B}) where {T,B}
    comm = v.backend.comm
    local_min = isempty(v.v) ? typemax(real(T)) : minimum(real, v.v)
    return comm_allreduce(comm, local_min, min)
end

"""
    Base.sum(v::HPCVector{T,B}) where {T,B}

Compute the sum of all elements in the distributed vector.
"""
function Base.sum(v::HPCVector{T,B}) where {T,B}
    comm = v.backend.comm
    # Use native sum without init for better performance; handle empty with ternary
    local_sum = isempty(v.v) ? zero(T) : sum(v.v)
    return comm_allreduce(comm, local_sum, +)
end

"""
    Base.prod(v::HPCVector{T,B}) where {T,B}

Compute the product of all elements in the distributed vector.
"""
function Base.prod(v::HPCVector{T,B}) where {T,B}
    comm = v.backend.comm
    # Use native prod without init for better performance; handle empty with ternary
    local_prod = isempty(v.v) ? one(T) : prod(v.v)
    return comm_allreduce(comm, local_prod, *)
end

# Vector addition and subtraction

"""
    Base.:+(u::HPCVector{T}, v::HPCVector{T}) where T

Add two distributed vectors. If partitions differ, v is aligned to u's partition.
The result has u's partition. Both vectors must have the same backend.
"""
function Base.:+(u::HPCVector{T,B}, v::HPCVector{T,B}) where {T,B}
    assert_backends_compatible(u.backend, v.backend)
    if u.structural_hash == v.structural_hash
        return HPCVector{T,B}(u.structural_hash, u.partition, u.v .+ v.v, u.backend)
    else
        # Align v to u's partition using repartition
        v_aligned = repartition(v, u.partition)
        return HPCVector{T,B}(u.structural_hash, u.partition, u.v .+ v_aligned.v, u.backend)
    end
end

"""
    Base.:-(u::HPCVector{T}, v::HPCVector{T}) where T

Subtract two distributed vectors. If partitions differ, v is aligned to u's partition.
The result has u's partition. Both vectors must have the same backend.
"""
function Base.:-(u::HPCVector{T,B}, v::HPCVector{T,B}) where {T,B}
    assert_backends_compatible(u.backend, v.backend)
    if u.structural_hash == v.structural_hash
        return HPCVector{T,B}(u.structural_hash, u.partition, u.v .- v.v, u.backend)
    else
        # Align v to u's partition using repartition
        v_aligned = repartition(v, u.partition)
        return HPCVector{T,B}(u.structural_hash, u.partition, u.v .- v_aligned.v, u.backend)
    end
end

"""
    Base.:-(v::HPCVector{T,B}) where {T,B}

Negate a distributed vector.
"""
function Base.:-(v::HPCVector{T,B}) where {T,B}
    return HPCVector{T,B}(v.structural_hash, v.partition, .-v.v, v.backend)
end

# Mixed transpose addition/subtraction
# transpose(u) +/- transpose(v) works, aligning v to u's partition if needed

"""
    Base.:+(ut::Transpose{<:Any, HPCVector{T}}, vt::Transpose{<:Any, HPCVector{T}}) where T

Add two transposed vectors. If partitions differ, vt is aligned to ut's partition.
Returns a transposed HPCVector.
"""
function Base.:+(ut::Transpose{<:Any, HPCVector{T}}, vt::Transpose{<:Any, HPCVector{T}}) where T
    return transpose(ut.parent + vt.parent)
end

"""
    Base.:-(ut::Transpose{<:Any, HPCVector{T}}, vt::Transpose{<:Any, HPCVector{T}}) where T

Subtract two transposed vectors. If partitions differ, vt is aligned to ut's partition.
Returns a transposed HPCVector.
"""
function Base.:-(ut::Transpose{<:Any, HPCVector{T}}, vt::Transpose{<:Any, HPCVector{T}}) where T
    return transpose(ut.parent - vt.parent)
end

"""
    Base.:-(vt::Transpose{<:Any, HPCVector{T}}) where T

Negate a transposed vector. Returns a transposed HPCVector.
"""
function Base.:-(vt::Transpose{<:Any, HPCVector{T}}) where T
    return transpose(-vt.parent)
end

# Scalar multiplication for HPCVector

"""
    Base.:*(a::Number, v::HPCVector{T,B}) where {T,B}

Scalar times vector.
"""
function Base.:*(a::Number, v::HPCVector{T,B}) where {T,B}
    RT = promote_type(typeof(a), T)
    return HPCVector{RT,B}(v.structural_hash, v.partition, RT.(a .* v.v), v.backend)
end

"""
    Base.:*(v::HPCVector{T,B}, a::Number) where {T,B}

Vector times scalar.
"""
Base.:*(v::HPCVector{T,B}, a::Number) where {T,B} = a * v

"""
    Base.:/(v::HPCVector{T,B}, a::Number) where {T,B}

Vector divided by scalar.
"""
function Base.:/(v::HPCVector{T,B}, a::Number) where {T,B}
    RT = promote_type(typeof(a), T)
    return HPCVector{RT,B}(v.structural_hash, v.partition, RT.(v.v ./ a), v.backend)
end

# Scalar multiplication for transposed HPCVector

"""
    Base.:*(a::Number, vt::Transpose{<:Any, HPCVector{T}}) where T

Scalar times transposed vector.
"""
Base.:*(a::Number, vt::Transpose{<:Any, HPCVector{T}}) where T = transpose(a * vt.parent)

"""
    Base.:*(vt::Transpose{<:Any, HPCVector{T}}, a::Number) where T

Transposed vector times scalar.
"""
Base.:*(vt::Transpose{<:Any, HPCVector{T}}, a::Number) where T = transpose(vt.parent * a)

"""
    Base.:/(vt::Transpose{<:Any, HPCVector{T}}, a::Number) where T

Transposed vector divided by scalar.
"""
Base.:/(vt::Transpose{<:Any, HPCVector{T}}, a::Number) where T = transpose(vt.parent / a)

# Vector size and eltype

"""
    Base.length(v::HPCVector)

Return the total length of the distributed vector.
"""
Base.length(v::HPCVector) = v.partition[end] - 1

"""
    Base.size(v::HPCVector)

Return the size of the distributed vector as a tuple.
"""
Base.size(v::HPCVector) = (length(v),)

Base.size(v::HPCVector, d::Integer) = d == 1 ? length(v) : 1

Base.eltype(::HPCVector{T}) where T = T
Base.eltype(::Type{HPCVector{T}}) where T = T

# ============================================================================
# Extended HPCVector API - Element-wise Operations
# ============================================================================

"""
    Base.abs(v::HPCVector{T,B}) where {T,B}

Return a new HPCVector with absolute values of all elements.
"""
function Base.abs(v::HPCVector{T,B}) where {T,B}
    RT = real(T)
    return HPCVector{RT,B}(v.structural_hash, v.partition, abs.(v.v), v.backend)
end

"""
    Base.abs2(v::HPCVector{T,B}) where {T,B}

Return a new HPCVector with squared absolute values of all elements.
"""
function Base.abs2(v::HPCVector{T,B}) where {T,B}
    RT = real(T)
    return HPCVector{RT,B}(v.structural_hash, v.partition, abs2.(v.v), v.backend)
end

"""
    Base.real(v::HPCVector{T,B}) where {T,B}

Return a new HPCVector containing the real parts of all elements.
"""
function Base.real(v::HPCVector{T,B}) where {T,B}
    RT = real(T)
    return HPCVector{RT,B}(v.structural_hash, v.partition, real.(v.v), v.backend)
end

"""
    Base.imag(v::HPCVector{T,B}) where {T,B}

Return a new HPCVector containing the imaginary parts of all elements.
"""
function Base.imag(v::HPCVector{T,B}) where {T,B}
    RT = real(T)
    return HPCVector{RT,B}(v.structural_hash, v.partition, imag.(v.v), v.backend)
end

"""
    Base.copy(v::HPCVector{T,B}) where {T,B}

Create a deep copy of the distributed vector.
"""
function Base.copy(v::HPCVector{T,B}) where {T,B}
    return HPCVector{T,B}(v.structural_hash, copy(v.partition), copy(v.v), v.backend)
end

"""
    mean(v::HPCVector{T}) where T

Compute the mean of all elements in the distributed vector.
"""
function mean(v::HPCVector{T}) where T
    return sum(v) / length(v)
end

# ============================================================================
# Broadcasting Support for HPCVector
# ============================================================================

import Base.Broadcast: BroadcastStyle, Broadcasted, DefaultArrayStyle, AbstractArrayStyle
import Base.Broadcast: broadcasted, materialize, instantiate, broadcastable

"""
    HPCVectorStyle <: AbstractArrayStyle{1}

Custom broadcast style for HPCVector that ensures broadcast operations
return HPCVector results and handle distributed data correctly.
"""
struct HPCVectorStyle <: AbstractArrayStyle{1} end

# HPCVector uses HPCVectorStyle
Base.BroadcastStyle(::Type{<:HPCVector}) = HPCVectorStyle()

# HPCVector is its own broadcastable representation (don't try to iterate)
Base.Broadcast.broadcastable(v::HPCVector) = v

# Define axes for HPCVector (needed for broadcast)
Base.axes(v::HPCVector) = (Base.OneTo(length(v)),)

# HPCVectorStyle wins over DefaultArrayStyle for scalars and regular arrays
Base.BroadcastStyle(::HPCVectorStyle, ::DefaultArrayStyle{0}) = HPCVectorStyle()
Base.BroadcastStyle(::HPCVectorStyle, ::DefaultArrayStyle{N}) where N = HPCVectorStyle()

# Two HPCVector => HPCVectorStyle
Base.BroadcastStyle(::HPCVectorStyle, ::HPCVectorStyle) = HPCVectorStyle()

"""
    _find_vectormpi(args...)

Find the first HPCVector in a tuple of broadcast arguments.
Recursively searches through nested Broadcasted objects.
"""
_find_vectormpi(v::HPCVector, args...) = v
function _find_vectormpi(bc::Broadcasted, args...)
    # Search in nested Broadcasted
    result = _find_vectormpi(bc.args...)
    if result !== nothing
        return result
    end
    return _find_vectormpi(args...)
end
_find_vectormpi(::Any, args...) = _find_vectormpi(args...)
_find_vectormpi() = nothing

"""
    _find_all_vectormpi(args...)

Find all HPCVector arguments and return them as a tuple.
"""
_find_all_vectormpi(args::Tuple) = _find_all_vectormpi_impl(args...)
_find_all_vectormpi_impl() = ()
_find_all_vectormpi_impl(v::HPCVector, args...) = (v, _find_all_vectormpi_impl(args...)...)
_find_all_vectormpi_impl(::Any, args...) = _find_all_vectormpi_impl(args...)

"""
    _prepare_broadcast_arg(arg, ref_partition, comm)

Prepare a broadcast argument for local computation.
- HPCVector with same partition: return local vector
- HPCVector with different partition: align to ref_partition
- Nested Broadcasted: recursively prepare and materialize
- Scalar or other: return as-is
"""
function _prepare_broadcast_arg(v::HPCVector, ref_partition, backend)
    # Use identity check first (fast), then element-wise comparison
    if v.partition === ref_partition || v.partition == ref_partition
        return v.v
    else
        # Align to reference partition using repartition (uses v.backend for communication)
        return repartition(v, ref_partition).v
    end
end

# Handle nested Broadcasted objects by recursively preparing their arguments
function _prepare_broadcast_arg(bc::Broadcasted{HPCVectorStyle}, ref_partition, backend)
    # Recursively prepare nested arguments
    prepared_args = map(arg -> _prepare_broadcast_arg(arg, ref_partition, backend), bc.args)
    # Return a new Broadcasted with prepared (local) arguments
    return Broadcasted{Nothing}(bc.f, prepared_args)
end

# Handle Broadcasted with other styles (e.g., scalar operations nested)
function _prepare_broadcast_arg(bc::Broadcasted, ref_partition, backend)
    # Recursively prepare nested arguments
    prepared_args = map(arg -> _prepare_broadcast_arg(arg, ref_partition, backend), bc.args)
    # Return a new Broadcasted with prepared arguments
    return Broadcasted{Nothing}(bc.f, prepared_args)
end

# Handle Base.RefValue (used in literal_pow for things like x.^2)
_prepare_broadcast_arg(r::Base.RefValue, ref_partition, backend) = r

_prepare_broadcast_arg(x, ref_partition, backend) = x

"""
    Base.similar(bc::Broadcasted{HPCVectorStyle}, ::Type{ElType}) where ElType

Allocate output array for HPCVector broadcast.
Preserves the storage type (CPU/GPU) of the first HPCVector in the broadcast.
"""
function Base.similar(bc::Broadcasted{HPCVectorStyle}, ::Type{ElType}) where ElType
    # Find a HPCVector to get partition info, array type, and backend
    v = _find_vectormpi(bc.args...)
    if v === nothing
        error("No HPCVector found in broadcast arguments")
    end
    # Create output with same partition, storage type, and backend
    # similar() preserves the array type (CPU Vector or GPU MtlVector)
    new_v = similar(v.v, ElType, length(v.v))
    B = typeof(v.backend)
    return HPCVector{ElType,B}(v.structural_hash, v.partition, new_v, v.backend)
end

"""
    Base.copyto!(dest::HPCVector, bc::Broadcasted{HPCVectorStyle})

Execute the broadcast operation and store results in dest.
"""
function Base.copyto!(dest::HPCVector, bc::Broadcasted{HPCVectorStyle})
    # Use backend from destination for communication
    backend = dest.backend

    # Find all HPCVector arguments
    all_vmpi = _find_all_vectormpi(bc.args)

    # Use the destination's partition as reference
    ref_partition = dest.partition

    # Prepare all arguments (align HPCVector to ref_partition, pass others through)
    prepared_args = map(arg -> _prepare_broadcast_arg(arg, ref_partition, backend), bc.args)

    # Perform local broadcast
    local_bc = Broadcasted{Nothing}(bc.f, prepared_args, axes(dest.v))
    copyto!(dest.v, local_bc)

    return dest
end

# Convenience: allow broadcast assignment to existing HPCVector
function Base.materialize!(dest::HPCVector, bc::Broadcasted{HPCVectorStyle})
    return copyto!(dest, instantiate(bc))
end
