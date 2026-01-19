"""
    HPCSparseArraysCUDAExt

Extension module for CUDA GPU support in HPCSparseArrays.
Provides:
- DeviceCUDA backend support for CuArray-backed distributed arrays
- cuDSS sparse direct solver with NCCL backend for multi-GPU factorization

Requires: `using CUDA, NCCL_jll, CUDSS_jll` before loading HPCSparseArrays.
(Uses NCCL_jll directly instead of NCCL.jl to avoid finalizer-induced MPI desync.)
"""
module HPCSparseArraysCUDAExt

using HPCSparseArrays
using CUDA
using NCCL_jll
using CUDSS_jll
using Adapt
using MPI
using SparseArrays
using LinearAlgebra

# ============================================================================
# Part 1: Core CUDA Support (cu/cpu conversions, backend helpers)
# ============================================================================

# Import backend types
using HPCSparseArrays: HPCBackend, DeviceCPU, DeviceCUDA, DeviceMetal,
                        CommSerial, CommMPI, AbstractComm, AbstractDevice,
                        SolverMUMPS, AbstractSolverCuDSS,
                        comm_rank, comm_size,
                        eltype_backend, indextype_backend

# Type aliases for convenience (with T and Ti type parameters)
const CuBackend{T,Ti,C,S} = HPCSparseArrays.HPCBackend{T, Ti, HPCSparseArrays.DeviceCUDA, C, S}
const CPUBackend{T,Ti,C,S} = HPCSparseArrays.HPCBackend{T, Ti, HPCSparseArrays.DeviceCPU, C, S}

# ============================================================================
# cuDSS Solver Types
# ============================================================================

"""
    SolverCuDSS <: AbstractSolverCuDSS

cuDSS sparse direct solver for CUDA GPUs.
- With CommSerial: Single-GPU cuDSS
- With CommMPI: Multi-GPU Multi-Node (MGMN) cuDSS with NCCL backend
"""
struct SolverCuDSS <: HPCSparseArrays.AbstractSolverCuDSS end

# Type alias for cuDSS-specific backends (constrains solver type to SolverCuDSS)
const CuDSSBackend{T,Ti,C} = HPCSparseArrays.HPCBackend{T, Ti, HPCSparseArrays.DeviceCUDA, C, SolverCuDSS}

# ============================================================================
# Pre-constructed Backend Constants (Deprecated)
# ============================================================================
#
# These use default types (Float64, Int) for backward compatibility.
# New code should use the factory functions with explicit type parameters.

"""
    BACKEND_CUDA_SERIAL

Pre-constructed CUDA backend with serial communication and cuDSS solver.
Uses Float64 element type and Int index type.

!!! warning "Deprecated"
    Use `backend_cuda_serial(T, Ti)` instead for explicit type control.
"""
const BACKEND_CUDA_SERIAL = HPCSparseArrays.HPCBackend{Float64,Int,DeviceCUDA,CommSerial,SolverCuDSS}(
    DeviceCUDA(), CommSerial(), SolverCuDSS())

"""
    BACKEND_CUDA_MPI

Pre-constructed CUDA backend with MPI communication (using COMM_WORLD) and cuDSS solver.
Uses Float64 element type and Int index type.

!!! warning "Deprecated"
    Use `backend_cuda_mpi(T, Ti)` instead for explicit type control.
"""
const BACKEND_CUDA_MPI = HPCSparseArrays.HPCBackend{Float64,Int,DeviceCUDA,CommMPI,SolverCuDSS}(
    DeviceCUDA(), CommMPI(MPI.COMM_WORLD), SolverCuDSS())

# ============================================================================
# Backend Factory Functions
# ============================================================================

"""
    backend_cuda_serial(::Type{T}=Float64, ::Type{Ti}=Int) where {T,Ti} -> HPCBackend

Create a CUDA backend with serial communication and cuDSS solver.

# Arguments
- `T`: Element type for array values (default: Float64)
- `Ti`: Index type for sparse matrix indices (default: Int)
"""
function HPCSparseArrays.backend_cuda_serial(::Type{T}=Float64, ::Type{Ti}=Int) where {T,Ti<:Integer}
    return HPCSparseArrays.HPCBackend{T,Ti,DeviceCUDA,CommSerial,SolverCuDSS}(
        DeviceCUDA(), CommSerial(), SolverCuDSS())
end

"""
    backend_cuda_mpi(::Type{T}=Float64, ::Type{Ti}=Int; comm=MPI.COMM_WORLD) where {T,Ti} -> HPCBackend

Create a CUDA GPU backend with MPI communication and cuDSS solver (MGMN mode).

# Arguments
- `T`: Element type for array values (default: Float64)
- `Ti`: Index type for sparse matrix indices (default: Int)
- `comm`: MPI communicator (default: MPI.COMM_WORLD)
"""
function HPCSparseArrays.backend_cuda_mpi(::Type{T}=Float64, ::Type{Ti}=Int; comm::MPI.Comm=MPI.COMM_WORLD) where {T,Ti<:Integer}
    return HPCSparseArrays.HPCBackend{T,Ti,DeviceCUDA,CommMPI,SolverCuDSS}(
        DeviceCUDA(), CommMPI(comm), SolverCuDSS())
end

# Legacy overload for backward compatibility (comm as positional argument)
function HPCSparseArrays.backend_cuda_mpi(comm::MPI.Comm)
    return HPCSparseArrays.backend_cuda_mpi(Float64, Int; comm=comm)
end


# ============================================================================
# _convert_array methods for CUDA
# ============================================================================

# CPU → CUDA: copy to GPU
HPCSparseArrays._convert_array(v::Vector, ::HPCSparseArrays.DeviceCUDA) = CuVector(v)
HPCSparseArrays._convert_array(A::Matrix, ::HPCSparseArrays.DeviceCUDA) = CuMatrix(A)

# CUDA → CUDA: identity (no copy)
HPCSparseArrays._convert_array(v::CuVector, ::HPCSparseArrays.DeviceCUDA) = v
HPCSparseArrays._convert_array(A::CuMatrix, ::HPCSparseArrays.DeviceCUDA) = A

# ============================================================================
# Backend helper functions
# ============================================================================

"""
    _zeros_device(::DeviceCUDA, ::Type{T}, dims...) where T

Create a zero CuVector/CuMatrix of the specified dimensions on CUDA device.
Used by Base.zeros(HPCVector, backend, n) etc.
"""
HPCSparseArrays._zeros_device(::HPCSparseArrays.DeviceCUDA, ::Type{T}, dims...) where T = CUDA.zeros(T, dims...)

"""
    _index_array_type(::DeviceCUDA, ::Type{Ti}) where Ti

Map DeviceCUDA to CuVector{Ti} index array type.
Used by MatrixPlan to store symbolic index arrays on GPU.
"""
HPCSparseArrays._index_array_type(::HPCSparseArrays.DeviceCUDA, ::Type{Ti}) where Ti = CuVector{Ti}

"""
    _to_target_device(v::Vector{Ti}, ::DeviceCUDA) where Ti

Convert a CPU index vector to CUDA GPU.
Used by HPCSparseMatrix constructors to create GPU structure arrays.
"""
HPCSparseArrays._to_target_device(v::Vector{Ti}, ::HPCSparseArrays.DeviceCUDA) where Ti = CuVector(v)

"""
    _array_to_device(v::Vector{T}, ::DeviceCUDA) where T

Convert a CPU vector to a CUDA GPU vector.
Used by MUMPS factorization for round-trip GPU conversion during solve.
"""
function HPCSparseArrays._array_to_device(v::Vector{T}, ::HPCSparseArrays.DeviceCUDA) where T
    return CuVector(v)
end

"""
    _convert_vector_to_device(v::HPCSparseArrays.HPCVector, ::DeviceCUDA)

Convert a HPCVector to CUDA GPU device.
Used by MUMPS factorization for GPU reconstruction after solve.
"""
function HPCSparseArrays._convert_vector_to_device(v::HPCSparseArrays.HPCVector{T,B}, device::HPCSparseArrays.DeviceCUDA) where {T, B}
    # Create CUDA backend preserving T, Ti, comm, and solver from source backend
    Ti = indextype_backend(B)
    C = typeof(v.backend.comm)
    S = typeof(v.backend.solver)
    cuda_backend = HPCSparseArrays.HPCBackend{T,Ti,DeviceCUDA,C,S}(device, v.backend.comm, v.backend.solver)
    return HPCSparseArrays.to_backend(v, cuda_backend)
end

# ============================================================================
# Part 2: cuDSS Constants and ccall Wrappers
# ============================================================================

const cudssHandle_t = Ptr{Cvoid}
const cudssConfig_t = Ptr{Cvoid}
const cudssData_t = Ptr{Cvoid}
const cudssMatrix_t = Ptr{Cvoid}

# Status codes
const CUDSS_STATUS_SUCCESS = UInt32(0)

# Data parameters (cudssDataParam_t enum values)
const CUDSS_DATA_INFO = UInt32(0)
const CUDSS_DATA_LU_NNZ = UInt32(1)
const CUDSS_DATA_NPIVOTS = UInt32(2)
const CUDSS_DATA_INERTIA = UInt32(3)
const CUDSS_DATA_PERM_REORDER_ROW = UInt32(4)
const CUDSS_DATA_PERM_REORDER_COL = UInt32(5)
const CUDSS_DATA_PERM_ROW = UInt32(6)
const CUDSS_DATA_PERM_COL = UInt32(7)
const CUDSS_DATA_DIAG = UInt32(8)
const CUDSS_DATA_USER_PERM = UInt32(9)
const CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN = UInt32(10)
const CUDSS_DATA_COMM = UInt32(11)
const CUDSS_DATA_MEMORY_ESTIMATES = UInt32(12)
const CUDSS_DATA_USER_ELIMINATION_TREE = UInt32(20)
const CUDSS_DATA_ELIMINATION_TREE = UInt32(21)

# Phases (can be OR'd together)
const CUDSS_PHASE_ANALYSIS = Cint(3)  # REORDERING | SYMBOLIC
const CUDSS_PHASE_FACTORIZATION = Cint(4)
const CUDSS_PHASE_SOLVE = Cint(1008)

# Matrix types
const CUDSS_MTYPE_GENERAL = UInt32(0)
const CUDSS_MTYPE_SYMMETRIC = UInt32(1)
const CUDSS_MTYPE_SPD = UInt32(3)

# Matrix view types
const CUDSS_MVIEW_FULL = UInt32(0)
const CUDSS_MVIEW_LOWER = UInt32(1)

# Index base
const CUDSS_BASE_ZERO = UInt32(0)
const CUDSS_BASE_ONE = UInt32(1)

# Layout
const CUDSS_LAYOUT_COL_MAJOR = UInt32(0)

# CUDA data type mapping
_cuda_data_type(::Type{Float32}) = UInt32(0)   # CUDA_R_32F
_cuda_data_type(::Type{Float64}) = UInt32(1)   # CUDA_R_64F
_cuda_data_type(::Type{Int32}) = UInt32(10)    # CUDA_R_32I
_cuda_data_type(::Type{Int64}) = UInt32(24)    # CUDA_R_64I

# Low-level ccall wrappers (no finalizers, explicit error handling)

function _cudss_create(handle_ref::Ref{cudssHandle_t})
    status = @ccall libcudss.cudssCreate(handle_ref::Ptr{cudssHandle_t})::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssCreate failed with status $status")
    return nothing
end

function _cudss_destroy(handle::cudssHandle_t)
    status = @ccall libcudss.cudssDestroy(handle::cudssHandle_t)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssDestroy failed with status $status")
    return nothing
end

function _cudss_set_comm_layer(handle::cudssHandle_t, lib_path::String)
    status = @ccall libcudss.cudssSetCommLayer(handle::cudssHandle_t,
                                                lib_path::Cstring)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssSetCommLayer failed with status $status")
    return nothing
end

function _cudss_config_create(config_ref::Ref{cudssConfig_t})
    status = @ccall libcudss.cudssConfigCreate(config_ref::Ptr{cudssConfig_t})::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssConfigCreate failed with status $status")
    return nothing
end

function _cudss_config_destroy(config::cudssConfig_t)
    status = @ccall libcudss.cudssConfigDestroy(config::cudssConfig_t)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssConfigDestroy failed with status $status")
    return nothing
end

function _cudss_data_create(handle::cudssHandle_t, data_ref::Ref{cudssData_t})
    status = @ccall libcudss.cudssDataCreate(handle::cudssHandle_t,
                                              data_ref::Ptr{cudssData_t})::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssDataCreate failed with status $status")
    return nothing
end

function _cudss_data_destroy(handle::cudssHandle_t, data::cudssData_t)
    status = @ccall libcudss.cudssDataDestroy(handle::cudssHandle_t,
                                               data::cudssData_t)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssDataDestroy failed with status $status")
    return nothing
end

function _cudss_data_set(handle::cudssHandle_t, data::cudssData_t,
                         param::UInt32, value::Ptr{Cvoid}, size::Csize_t)
    status = @ccall libcudss.cudssDataSet(handle::cudssHandle_t,
                                           data::cudssData_t,
                                           param::UInt32,
                                           value::Ptr{Cvoid},
                                           size::Csize_t)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssDataSet failed with status $status")
    return nothing
end

function _cudss_data_get(handle::cudssHandle_t, data::cudssData_t,
                         param::UInt32, value::Ptr{Cvoid}, size::Csize_t)
    size_written = Ref{Csize_t}(0)
    status = @ccall libcudss.cudssDataGet(handle::cudssHandle_t,
                                           data::cudssData_t,
                                           param::UInt32,
                                           value::Ptr{Cvoid},
                                           size::Csize_t,
                                           size_written::Ptr{Csize_t})::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssDataGet (param=$param) failed with status $status")
    return size_written[]
end

function _cudss_matrix_create_csr(matrix_ref::Ref{cudssMatrix_t},
                                   nrows::Int64, ncols::Int64, nnz::Int64,
                                   row_offsets::CuPtr{Cvoid}, row_end::CuPtr{Cvoid},
                                   col_indices::CuPtr{Cvoid}, values::CuPtr{Cvoid},
                                   index_type::UInt32, value_type::UInt32,
                                   mtype::UInt32, mview::UInt32, index_base::UInt32)
    status = @ccall libcudss.cudssMatrixCreateCsr(
        matrix_ref::Ptr{cudssMatrix_t},
        nrows::Int64, ncols::Int64, nnz::Int64,
        row_offsets::CuPtr{Cvoid}, row_end::CuPtr{Cvoid},
        col_indices::CuPtr{Cvoid}, values::CuPtr{Cvoid},
        index_type::UInt32, value_type::UInt32,
        mtype::UInt32, mview::UInt32, index_base::UInt32)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssMatrixCreateCsr failed with status $status")
    return nothing
end

function _cudss_matrix_create_dn(matrix_ref::Ref{cudssMatrix_t},
                                  nrows::Int64, ncols::Int64, ld::Int64,
                                  values::CuPtr{Cvoid}, value_type::UInt32, layout::UInt32)
    status = @ccall libcudss.cudssMatrixCreateDn(
        matrix_ref::Ptr{cudssMatrix_t},
        nrows::Int64, ncols::Int64, ld::Int64,
        values::CuPtr{Cvoid}, value_type::UInt32, layout::UInt32)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssMatrixCreateDn failed with status $status")
    return nothing
end

function _cudss_matrix_destroy(matrix::cudssMatrix_t)
    status = @ccall libcudss.cudssMatrixDestroy(matrix::cudssMatrix_t)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssMatrixDestroy failed with status $status")
    return nothing
end

function _cudss_matrix_set_distribution_row1d(matrix::cudssMatrix_t,
                                               first_row::Int64, last_row::Int64)
    status = @ccall libcudss.cudssMatrixSetDistributionRow1d(
        matrix::cudssMatrix_t, first_row::Int64, last_row::Int64)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssMatrixSetDistributionRow1d failed with status $status")
    return nothing
end

function _cudss_execute(handle::cudssHandle_t, phase::Cint,
                        config::cudssConfig_t, data::cudssData_t,
                        matrix::cudssMatrix_t, solution::cudssMatrix_t, rhs::cudssMatrix_t)
    status = @ccall libcudss.cudssExecute(
        handle::cudssHandle_t, phase::Cint,
        config::cudssConfig_t, data::cudssData_t,
        matrix::cudssMatrix_t, solution::cudssMatrix_t, rhs::cudssMatrix_t)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssExecute (phase=$phase) failed with status $status")
    return nothing
end

# ============================================================================
# Part 3: NCCL Low-Level Bindings and Bootstrap
# ============================================================================

# NCCL low-level bindings (no finalizers - communicator lives until process exit)
const NCCL_UNIQUE_ID_BYTES = 128

struct ncclUniqueId
    internal::NTuple{NCCL_UNIQUE_ID_BYTES, UInt8}
end
ncclUniqueId() = ncclUniqueId(ntuple(_ -> UInt8(0), NCCL_UNIQUE_ID_BYTES))

const ncclComm_t = Ptr{Cvoid}
const ncclSuccess = 0

# Global cache: one NCCL communicator per MPI communicator, never destroyed
# Process exit handles cleanup automatically (like unclosed files)
const _nccl_comm_cache = Dict{UInt64, ncclComm_t}()

function _nccl_get_unique_id()
    id = Ref(ncclUniqueId())
    status = @ccall libnccl.ncclGetUniqueId(id::Ptr{ncclUniqueId})::Cint
    status == ncclSuccess || error("ncclGetUniqueId failed: $status")
    return id[]
end

function _nccl_comm_init_rank(nranks::Int, rank::Int, unique_id::ncclUniqueId)
    comm = Ref{ncclComm_t}(C_NULL)
    status = @ccall libnccl.ncclCommInitRank(
        comm::Ptr{ncclComm_t}, nranks::Cint, unique_id::ncclUniqueId, rank::Cint
    )::Cint
    status == ncclSuccess || error("ncclCommInitRank failed: $status")
    return comm[]
end

"""
    _get_nccl_comm(mpi_comm::MPI.Comm) -> ncclComm_t

Get or create an NCCL communicator for the given MPI communicator.
Communicators are cached globally and never destroyed - process exit handles cleanup.
This avoids all finalizer-related MPI desync issues.
"""
function _get_nccl_comm(mpi_comm::MPI.Comm)
    # Use MPI comm handle as cache key
    key = UInt64(mpi_comm.val)

    if haskey(_nccl_comm_cache, key)
        return _nccl_comm_cache[key]
    end

    # Create new NCCL comm from MPI (collective operation)
    rank = MPI.Comm_rank(mpi_comm)
    nranks = MPI.Comm_size(mpi_comm)

    # Rank 0 generates the unique ID
    if rank == 0
        unique_id = _nccl_get_unique_id()
        unique_id_bytes = collect(reinterpret(UInt8, [unique_id.internal]))
    else
        unique_id_bytes = zeros(UInt8, NCCL_UNIQUE_ID_BYTES)
    end

    # Broadcast the unique ID from rank 0 to all ranks
    MPI.Bcast!(unique_id_bytes, 0, mpi_comm)

    # Reconstruct ncclUniqueId from bytes
    internal_tuple = NTuple{NCCL_UNIQUE_ID_BYTES, UInt8}(unique_id_bytes)
    unique_id = ncclUniqueId(internal_tuple)

    # Create NCCL communicator (no finalizer - lives until process exit)
    nccl_comm = _nccl_comm_init_rank(nranks, rank, unique_id)

    _nccl_comm_cache[key] = nccl_comm
    return nccl_comm
end

# ============================================================================
# Part 4: cuDSS Factorization and Analysis Caching
# ============================================================================
#
# Analysis caching strategy:
# - After first analysis, extract permutation + elimination tree via cudssDataGet
# - Cache these by structural hash
# - For subsequent lu(A) with same structure:
#   - Create NEW data object (gets its own L/U storage)
#   - Set cached perm/tree via cudssDataSet (skips expensive reordering)
#   - Run factorization
# - Each factorization has independent L/U factors
# - F \ b is solve-only (no refactorization)
#
# Nothing is ever destroyed - process exit handles cleanup (like NCCL comms).

"""
    CuDSSAnalysisCache

Cached analysis results (permutation + elimination tree) for a sparsity structure.
Can be reused across multiple factorizations with the same structure.
Each factorization gets its own cuDSS data object with independent L/U storage.
"""
struct CuDSSAnalysisCache
    # Permutation vectors (device pointers, size n each)
    perm_row::CuVector{Int64}
    perm_col::CuVector{Int64}
    # Elimination tree (device pointer)
    elimination_tree::CuVector{Int64}
    # NCCL comm (stored to avoid dict lookup)
    nccl_comm::ncclComm_t
    # Global matrix dimension
    n::Int64
end

# Cache: (structural_hash, symmetric, element_type) -> CuDSSAnalysisCache
const _cudss_analysis_cache = Dict{Tuple{NTuple{32,UInt8}, Bool, DataType}, CuDSSAnalysisCache}()

# Backslash cache: reuse cuDSS data objects across A\b calls with same structure.
# Key: (structural_hash, symmetric, element_type)
# Value: CuDSSFactorizationMPI (never destroyed - process exit handles cleanup)
#
# This is the key optimization: for matrices with the same sparsity pattern,
# we skip analysis and only refactorize with new values.
const _cudss_backslash_cache = Dict{Tuple{NTuple{32,UInt8}, Bool, DataType}, Any}()

# Global cuDSS handle + config cache - one per process, shared across all factorizations
# Creating many handles leads to memory corruption in cuDSS
const _cudss_handle_ref = Ref{cudssHandle_t}(C_NULL)
const _cudss_config_ref = Ref{cudssConfig_t}(C_NULL)

"""
    _get_cudss_handle_and_config() -> (cudssHandle_t, cudssConfig_t)

Get or create cuDSS handle and config for this process. Both are cached globally.
"""
function _get_cudss_handle_and_config()
    if _cudss_handle_ref[] == C_NULL
        # Create handle
        handle_ref = Ref{cudssHandle_t}(C_NULL)
        _cudss_create(handle_ref)
        _cudss_handle_ref[] = handle_ref[]

        # Set NCCL communication layer
        comm_lib = joinpath(CUDSS_jll.artifact_dir, "lib", "libcudss_commlayer_nccl.so")
        if !isfile(comm_lib)
            error("NCCL communication layer not found at $comm_lib")
        end
        _cudss_set_comm_layer(_cudss_handle_ref[], comm_lib)

        # Create config (reused across all factorizations)
        config_ref = Ref{cudssConfig_t}(C_NULL)
        _cudss_config_create(config_ref)
        _cudss_config_ref[] = config_ref[]
    end
    return _cudss_handle_ref[], _cudss_config_ref[]
end

"""
    CuDSSFactorizationMPI{T, B}

Distributed cuDSS factorization with NCCL backend.
Created by `lu(A)` - owns its cuDSS data object with completed analysis + factorization.
Use `F \\ b` for solve-only (no refactorization needed).

Resources:
- handle, config: Global (cached per-process, not destroyed)
- data, matrix, solution, rhs: Per-factorization (destroyed in finalize!)
"""
mutable struct CuDSSFactorizationMPI{T, B<:HPCSparseArrays.HPCBackend}
    # cuDSS handles (global, NOT owned - do not destroy)
    handle::cudssHandle_t
    config::cudssConfig_t
    # Per-factorization resources (owned - destroy in finalize!)
    data::cudssData_t
    matrix::cudssMatrix_t
    solution::cudssMatrix_t
    rhs::cudssMatrix_t
    # GPU arrays
    row_offsets::CuVector{Int32}
    col_indices::CuVector{Int32}
    values::CuVector{T}
    x_gpu::CuVector{T}
    b_gpu::CuVector{T}
    # Storage for NCCL comm pointer (prevents GC)
    nccl_comm_storage::Vector{Ptr{Nothing}}
    # NCCL comm (stored directly to avoid dict lookup on solve)
    nccl_comm::ncclComm_t
    # Metadata
    n::Int  # Global matrix dimension
    local_nrows::Int
    first_row::Int  # 0-based
    last_row::Int   # 0-based, inclusive
    symmetric::Bool
    row_partition::Vector{Int}
    # HPCBackend for comm access
    backend::B
end

# ============================================================================
# Part 5: lu/ldlt and solve interface
# ============================================================================

"""
    lu(A::HPCSparseMatrix{T,Ti,<:CuDSSBackend})

Compute LU factorization of a GPU sparse matrix using cuDSS.
Returns a CuDSSFactorizationMPI that can be used with `F \\ b`.

If a previous factorization with the same sparsity structure exists,
the cached analysis (permutation + elimination tree) is reused,
skipping the expensive reordering phase.

Note: This method is specific to cuDSS backends (SolverCuDSS).
"""
function LinearAlgebra.lu(A::HPCSparseArrays.HPCSparseMatrix{T,Ti,<:CuDSSBackend}) where {T,Ti}
    return _create_cudss_factorization(A, false)
end

"""
    ldlt(A::HPCSparseMatrix{T,Ti,<:CuDSSBackend})

Compute LDLT factorization of a symmetric positive definite GPU sparse matrix.
Returns a CuDSSFactorizationMPI that can be used with `F \\ b`.

If a previous factorization with the same sparsity structure exists,
the cached analysis (permutation + elimination tree) is reused.

Note: This method is specific to cuDSS backends (SolverCuDSS).
"""
function LinearAlgebra.ldlt(A::HPCSparseArrays.HPCSparseMatrix{T,Ti,<:CuDSSBackend}) where {T,Ti}
    return _create_cudss_factorization(A, true)
end

"""
Internal: Create cuDSS factorization with analysis caching.
"""
function _create_cudss_factorization(A::HPCSparseArrays.HPCSparseMatrix{T,Ti,B}, symmetric::Bool) where {T,Ti,B}
    comm = A.backend.comm
    rank = comm_rank(comm)
    nranks = comm_size(comm)

    # Get MPI.Comm for NCCL bootstrapping (requires actual MPI communicator)
    mpi_comm = _get_mpi_comm_for_nccl(comm)

    # Assign GPU to rank
    num_gpus = length(CUDA.devices())
    gpu_id = mod(rank, num_gpus)
    CUDA.device!(gpu_id)

    # Get structural hash (computed eagerly in constructor)
    structural_hash = A.structural_hash
    cache_key = (structural_hash, symmetric, T)

    # Get matrix dimensions
    n = A.row_partition[end] - 1  # Global dimension (1-indexed partition)
    local_nrows = A.nrows_local
    first_row = A.row_partition[rank + 1] - 1  # Convert to 0-based
    last_row = A.row_partition[rank + 2] - 2   # 0-based, inclusive

    # Convert to CSR format with 0-based indices for cuDSS
    row_offsets_cpu = Int32.(A.rowptr .- 1)
    row_offsets = CuVector{Int32}(row_offsets_cpu)

    # A.colval contains LOCAL indices into col_indices
    # Need GLOBAL column indices for cuDSS
    col_indices_global_cpu = Int32.(A.col_indices[A.colval] .- 1)  # 0-based global
    col_indices_gpu = CuVector{Int32}(col_indices_global_cpu)

    # Values
    values_cpu = HPCSparseArrays._ensure_cpu(A.nzval)
    values_gpu = CuVector{T}(values_cpu)
    nnz_local = length(values_gpu)

    # Allocate solution and RHS vectors
    x_gpu = CUDA.zeros(T, local_nrows)
    b_gpu = CUDA.zeros(T, local_nrows)

    # Get or create cached NCCL communicator (requires MPI.Comm)
    nccl_comm = _get_nccl_comm(mpi_comm)

    # Get cached cuDSS handle + config (shared across all factorizations)
    handle, config = _get_cudss_handle_and_config()

    # Create per-factorization data object (holds L/U factors)
    data_ref = Ref{cudssData_t}(C_NULL)
    _cudss_data_create(handle, data_ref)
    data = data_ref[]

    # Set NCCL communicator
    nccl_comm_storage = Vector{Ptr{Nothing}}(undef, 1)
    nccl_comm_storage[1] = Ptr{Nothing}(nccl_comm)
    _cudss_data_set(handle, data, CUDSS_DATA_COMM,
                    Ptr{Cvoid}(pointer(nccl_comm_storage)), Csize_t(sizeof(Ptr{Nothing})))

    # Create sparse matrix wrapper
    mtype = symmetric ? CUDSS_MTYPE_SPD : CUDSS_MTYPE_GENERAL
    matrix_ref = Ref{cudssMatrix_t}(C_NULL)
    _cudss_matrix_create_csr(matrix_ref,
        Int64(n), Int64(n), Int64(nnz_local),
        reinterpret(CuPtr{Cvoid}, pointer(row_offsets)),
        CuPtr{Cvoid}(0),
        reinterpret(CuPtr{Cvoid}, pointer(col_indices_gpu)),
        reinterpret(CuPtr{Cvoid}, pointer(values_gpu)),
        _cuda_data_type(Int32), _cuda_data_type(T),
        mtype, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO)
    matrix = matrix_ref[]
    _cudss_matrix_set_distribution_row1d(matrix, Int64(first_row), Int64(last_row))

    # Create dense wrappers for x and b
    solution_ref = Ref{cudssMatrix_t}(C_NULL)
    _cudss_matrix_create_dn(solution_ref,
        Int64(local_nrows), Int64(1), Int64(local_nrows),
        reinterpret(CuPtr{Cvoid}, pointer(x_gpu)),
        _cuda_data_type(T), CUDSS_LAYOUT_COL_MAJOR)
    solution = solution_ref[]
    _cudss_matrix_set_distribution_row1d(solution, Int64(first_row), Int64(last_row))

    rhs_ref = Ref{cudssMatrix_t}(C_NULL)
    _cudss_matrix_create_dn(rhs_ref,
        Int64(local_nrows), Int64(1), Int64(local_nrows),
        reinterpret(CuPtr{Cvoid}, pointer(b_gpu)),
        _cuda_data_type(T), CUDSS_LAYOUT_COL_MAJOR)
    rhs = rhs_ref[]
    _cudss_matrix_set_distribution_row1d(rhs, Int64(first_row), Int64(last_row))

    # Run analysis phase
    # Note: Analysis caching is disabled in MGMN mode because cudssDataGet for
    # PERM_REORDER_ROW/COL returns INVALID_VALUE. Each factorization does full analysis.
    _cudss_execute(handle, CUDSS_PHASE_ANALYSIS, config, data, matrix, solution, rhs)

    # Run numeric factorization
    _cudss_execute(handle, CUDSS_PHASE_FACTORIZATION, config, data, matrix, solution, rhs)

    # Create factorization object
    F = CuDSSFactorizationMPI{T,B}(
        handle, config, data, matrix, solution, rhs,
        row_offsets, col_indices_gpu, values_gpu, x_gpu, b_gpu,
        nccl_comm_storage, nccl_comm,
        n, local_nrows, first_row, last_row,
        symmetric, copy(A.row_partition),
        A.backend
    )

    return F
end

# Helper to extract MPI.Comm from AbstractComm for NCCL bootstrapping
_get_mpi_comm_for_nccl(c::HPCSparseArrays.CommMPI) = c.comm
_get_mpi_comm_for_nccl(::HPCSparseArrays.CommSerial) = error("cuDSS MGMN mode requires MPI communication (CommMPI), not CommSerial")

"""
    solve(F::CuDSSFactorizationMPI{T,B}, b::HPCVector{T,<:CuDSSBackend}) where {T,B}

Solve the linear system using the cuDSS factorization.
This is solve-only - no refactorization is performed.

Note: This method is specific to cuDSS backends (SolverCuDSS).
"""
function HPCSparseArrays.solve(F::CuDSSFactorizationMPI{T,B}, b::HPCSparseArrays.HPCVector{T,<:CuDSSBackend}) where {T,B}
    comm = F.backend.comm

    # Copy b directly to RHS buffer (GPU to GPU)
    copyto!(F.b_gpu, b.v)

    # Execute solve phase only (collective operation)
    _cudss_execute(F.handle, CUDSS_PHASE_SOLVE, F.config, F.data, F.matrix, F.solution, F.rhs)

    # Return GPU vector (copy from internal buffer) with backend
    return HPCSparseArrays.HPCVector{T,B}(b.structural_hash, b.partition, copy(F.x_gpu), F.backend)
end

"""
    \\(F::CuDSSFactorizationMPI{T,B}, b::HPCVector{T,<:CuDSSBackend}) where {T,B}

Solve the linear system using backslash notation (solve-only, no refactorization).

Note: This method is specific to cuDSS backends (SolverCuDSS).
"""
function Base.:\(F::CuDSSFactorizationMPI{T,B}, b::HPCSparseArrays.HPCVector{T,<:CuDSSBackend}) where {T,B}
    return HPCSparseArrays.solve(F, b)
end

"""
    finalize!(F::CuDSSFactorizationMPI)

Destroy per-factorization cuDSS resources. Must be called collectively on all ranks.
Only destroys: data (L/U factors), matrix wrappers.
Does NOT destroy: handle, config (global, cached per-process).
"""
function HPCSparseArrays.finalize!(F::CuDSSFactorizationMPI)
    # Destroy data object (holds L/U factors - collective operation in MGMN mode)
    if F.data != C_NULL
        _cudss_data_destroy(F.handle, F.data)
        F.data = C_NULL
    end

    # Destroy matrix wrappers (thin wrappers, cheap to destroy)
    if F.matrix != C_NULL
        _cudss_matrix_destroy(F.matrix)
        F.matrix = C_NULL
    end
    if F.solution != C_NULL
        _cudss_matrix_destroy(F.solution)
        F.solution = C_NULL
    end
    if F.rhs != C_NULL
        _cudss_matrix_destroy(F.rhs)
        F.rhs = C_NULL
    end

    return nothing
end

# ============================================================================
# Part 6: Cached Backslash for GPU Sparse Matrices
# ============================================================================
#
# Strategy: Reuse cuDSS data objects across A\b calls with the same sparsity pattern.
# - First call (cache miss): full analysis + factorization
# - Subsequent calls (cache hit): update values + refactorize only (skip analysis!)
#
# This works because:
# 1. After analysis, the cuDSS data object has the symbolic factorization
# 2. We can call CUDSS_PHASE_FACTORIZATION again with updated values
# 3. The cudss matrix wrapper points to our values buffer - we update it in place

"""
    _refactorize_and_solve!(F::CuDSSFactorizationMPI{T,B}, A::HPCSparseMatrix{T,Ti,B}, b::HPCVector{T,B}) where {T,Ti,B<:CuDSSBackend}

Update the values in a cached factorization, refactorize (skip analysis), and solve.
Returns the solution vector.

Note: This method is specific to cuDSS backends (SolverCuDSS).
"""
function _refactorize_and_solve!(F::CuDSSFactorizationMPI{T,B},
                                  A::HPCSparseArrays.HPCSparseMatrix{T,Ti,B},
                                  b::HPCSparseArrays.HPCVector{T,B}) where {T,Ti,B<:CuDSSBackend}
    comm = F.backend.comm

    # Update values in the GPU buffer (the cudss matrix wrapper points to this)
    # A.colval contains local indices, A.nzval contains values
    values_cpu = HPCSparseArrays._ensure_cpu(A.nzval)
    copyto!(F.values, values_cpu)

    # Copy RHS to buffer
    copyto!(F.b_gpu, b.v)

    # Refactorize (skip analysis - the symbolic factorization is already done)
    _cudss_execute(F.handle, CUDSS_PHASE_FACTORIZATION, F.config, F.data, F.matrix, F.solution, F.rhs)

    # Solve
    _cudss_execute(F.handle, CUDSS_PHASE_SOLVE, F.config, F.data, F.matrix, F.solution, F.rhs)

    # Return GPU vector (copy from internal buffer) with backend
    return HPCSparseArrays.HPCVector{T, B}(b.structural_hash, b.partition, copy(F.x_gpu), b.backend)
end

"""
    \\(A::HPCSparseMatrix{T,Ti,B}, b::HPCVector{T,B}) where {T,Ti,B<:CuDSSBackend}

Solve A*x = b using cuDSS with analysis caching.

First call for a given sparsity pattern: full analysis + factorization.
Subsequent calls with same pattern: refactorize only (skip expensive analysis).

The cuDSS data object is cached globally and reused - never destroyed.

Note: This method is specific to cuDSS backends (SolverCuDSS).
"""
function Base.:\(A::HPCSparseArrays.HPCSparseMatrix{T,Ti,B},
                 b::HPCSparseArrays.HPCVector{T,B}) where {T,Ti,B<:CuDSSBackend}
    structural_hash = A.structural_hash
    cache_key = (structural_hash, false, T)  # false = not symmetric (LU)

    if haskey(_cudss_backslash_cache, cache_key)
        # Cache hit: refactorize and solve (skip analysis!)
        F = _cudss_backslash_cache[cache_key]::CuDSSFactorizationMPI{T,B}
        return _refactorize_and_solve!(F, A, b)
    else
        # Cache miss: create full factorization
        F = _create_cudss_factorization(A, false)
        _cudss_backslash_cache[cache_key] = F

        # Solve (collective operation)
        copyto!(F.b_gpu, b.v)
        _cudss_execute(F.handle, CUDSS_PHASE_SOLVE, F.config, F.data, F.matrix, F.solution, F.rhs)

        return HPCSparseArrays.HPCVector{T, B}(b.structural_hash, b.partition, copy(F.x_gpu), b.backend)
    end
end

"""
    \\(A::Symmetric{T,<:HPCSparseMatrix{T,Ti,B}}, b::HPCVector{T,B}) where {T,Ti,B<:CuDSSBackend}

Solve A*x = b for a symmetric matrix using LDLT with analysis caching.

Note: This method is specific to cuDSS backends (SolverCuDSS).
"""
function Base.:\(A::Symmetric{T,<:HPCSparseArrays.HPCSparseMatrix{T,Ti,B}},
                 b::HPCSparseArrays.HPCVector{T,B}) where {T,Ti,B<:CuDSSBackend}
    A_inner = parent(A)
    structural_hash = A_inner.structural_hash
    cache_key = (structural_hash, true, T)  # true = symmetric (LDLT)

    if haskey(_cudss_backslash_cache, cache_key)
        # Cache hit: refactorize and solve (skip analysis!)
        F = _cudss_backslash_cache[cache_key]::CuDSSFactorizationMPI{T,B}
        return _refactorize_and_solve!(F, A_inner, b)
    else
        # Cache miss: create full factorization
        F = _create_cudss_factorization(A_inner, true)
        _cudss_backslash_cache[cache_key] = F

        # Solve (collective operation)
        copyto!(F.b_gpu, b.v)
        _cudss_execute(F.handle, CUDSS_PHASE_SOLVE, F.config, F.data, F.matrix, F.solution, F.rhs)

        return HPCSparseArrays.HPCVector{T, B}(b.structural_hash, b.partition, copy(F.x_gpu), b.backend)
    end
end

# ============================================================================
# GPU map_rows_gpu implementation via CUDA kernels
# ============================================================================

using StaticArrays

"""
    _map_rows_gpu_kernel(f, arg1::CuMatrix, rest::CuMatrix...)

GPU-accelerated row-wise map for CUDA arrays.
"""
function HPCSparseArrays._map_rows_gpu_kernel(f, arg1::CuMatrix{T}, rest::CuMatrix...) where T
    n = size(arg1, 1)

    # Get output size by evaluating f on first row
    arg1_row1 = Array(view(arg1, 1:1, :))[1, :]
    first_rows = (SVector{size(arg1,2),T}(arg1_row1...),)
    for m in rest
        m_row1 = Array(view(m, 1:1, :))[1, :]
        first_rows = (first_rows..., SVector{size(m,2),T}(m_row1...))
    end
    sample_out = f(first_rows...)

    if sample_out isa SVector
        out_cols = length(sample_out)
    elseif sample_out isa SMatrix
        out_cols = length(sample_out)
    else
        out_cols = 1
    end

    output = CUDA.zeros(T, n, out_cols)
    _cuda_map_rows_kernel_dispatch(f, output, arg1, rest...)

    return output
end

function _cuda_map_rows_kernel_dispatch(f, output::CuMatrix{T}, arg1::CuMatrix{T}) where T
    n = size(arg1, 1)
    ncols1 = size(arg1, 2)
    out_cols = size(output, 2)

    kernel = @cuda launch=false _cuda_map_rows_kernel_1arg(f, output, arg1, Val(ncols1), Val(out_cols))
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)
    kernel(f, output, arg1, Val(ncols1), Val(out_cols); threads=threads, blocks=blocks)
end

function _cuda_map_rows_kernel_dispatch(f, output::CuMatrix{T}, arg1::CuMatrix{T}, arg2::CuMatrix{T}) where T
    n = size(arg1, 1)
    ncols1 = size(arg1, 2)
    ncols2 = size(arg2, 2)
    out_cols = size(output, 2)

    kernel = @cuda launch=false _cuda_map_rows_kernel_2args(f, output, arg1, arg2, Val(ncols1), Val(ncols2), Val(out_cols))
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)
    kernel(f, output, arg1, arg2, Val(ncols1), Val(ncols2), Val(out_cols); threads=threads, blocks=blocks)
end

# CUDA kernels
function _cuda_map_rows_kernel_1arg(f, output, arg1, ::Val{NC1}, ::Val{OCols}) where {NC1, OCols}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    n = size(arg1, 1)
    if i <= n
        T = eltype(arg1)
        row1 = SVector{NC1,T}(ntuple(j -> @inbounds(arg1[i,j]), Val(NC1)))
        result = f(row1)
        _cuda_write_result!(output, i, result, Val(OCols))
    end
    return nothing
end

function _cuda_map_rows_kernel_2args(f, output, arg1, arg2, ::Val{NC1}, ::Val{NC2}, ::Val{OCols}) where {NC1, NC2, OCols}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    n = size(arg1, 1)
    if i <= n
        T = eltype(arg1)
        row1 = SVector{NC1,T}(ntuple(j -> @inbounds(arg1[i,j]), Val(NC1)))
        row2 = SVector{NC2,T}(ntuple(j -> @inbounds(arg2[i,j]), Val(NC2)))
        result = f(row1, row2)
        _cuda_write_result!(output, i, result, Val(OCols))
    end
    return nothing
end

@inline function _cuda_write_result!(output, i, result::Number, ::Val{1})
    @inbounds output[i, 1] = result
    return nothing
end

@inline function _cuda_write_result!(output, i, result::SVector{N,T}, ::Val{N}) where {N,T}
    for j in 1:N
        @inbounds output[i, j] = result[j]
    end
    return nothing
end

@inline function _cuda_write_result!(output, i, result::SMatrix{M,N,T}, ::Val{MN}) where {M,N,T,MN}
    for j in 1:MN
        @inbounds output[i, j] = result[j]
    end
    return nothing
end

# ============================================================================
# Part 7: cuSPARSE Sparse Matrix-Matrix Multiplication
# ============================================================================

using CUDA.CUSPARSE

"""
    *(A::HPCSparseMatrix{T,Ti,B}, B_mat::HPCSparseMatrix{T,Ti,B}) where {T,Ti,B<:CuBackend}

Multiply two distributed sparse matrices using cuSPARSE on GPU.

MPI communication gathers the required rows of B (via MatrixPlan), then
cuSPARSE performs the sparse-sparse multiplication on GPU.

Uses pre-computed structure from MatrixPlan with GPU-computed values.
"""
function Base.:*(A::HPCSparseArrays.HPCSparseMatrix{T,Ti,B},
                 B_mat::HPCSparseArrays.HPCSparseMatrix{T,Ti,B}) where {T,Ti,B<:CuBackend}
    # Step 1: MPI Gather - fetches remote B rows to plan.AT (CPU)
    # This is the only unavoidable CPU landing - MPI communication requires it
    plan = HPCSparseArrays.MatrixPlan(A, B_mat)
    HPCSparseArrays.execute_plan!(plan, B_mat)

    # Step 2: Marshal plan.AT to GPU (one transfer of gathered B data)
    AT_gpu = CuSparseMatrixCSC(plan.AT)

    # Step 3: Build A on GPU directly (NO copy of A.nzval - it's already GPU)
    # _get_csc(A) uses local indices A.colval directly, so we do the same
    # This matches the CSC format: (ncols_compressed, nrows_local)
    A_gpu = CuSparseMatrixCSC{T,Ti}(
        CuVector{Ti}(A.rowptr),  # colPtr - small structure
        CuVector{Ti}(A.colval),  # rowVal - local indices, small structure
        A.nzval,                 # nzVal - ALREADY ON GPU - no copy!
        (A.ncols_compressed, A.nrows_local)
    )

    # Step 4: cuSPARSE multiply - all on GPU
    CT_gpu = AT_gpu * A_gpu

    # Step 5: Build result using pre-computed structure from plan (matches CPU path)
    # Both Julia and cuSPARSE produce sorted CSC/CSR output, so structures match
    nrows_local = size(CT_gpu, 2)  # CT columns = C rows
    ncols_compressed = length(plan.result_col_indices)

    # Use cached GPU arrays from plan, or create them on first use (matches CPU path)
    if plan.cached_rowptr_target === nothing
        plan.cached_rowptr_target = CuVector{Ti}(plan.result_colptr)
    end
    if plan.cached_colval_target === nothing
        plan.cached_colval_target = CuVector{Ti}(plan.result_colval)
    end

    return HPCSparseArrays.HPCSparseMatrix{T,Ti,B}(
        plan.result_structural_hash,  # Use cached hash from plan
        plan.result_row_partition, plan.result_col_partition,
        plan.result_col_indices, plan.result_colptr, plan.result_colval,
        CT_gpu.nzVal,  # STAYS ON GPU - no copy!
        nrows_local, ncols_compressed, nothing, nothing,
        plan.cached_rowptr_target, plan.cached_colval_target, A.backend)
end

end # module
