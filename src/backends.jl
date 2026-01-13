# ============================================================================
# HPCBackend Abstraction Layer
# ============================================================================
#
# This module defines the HPCBackend type hierarchy that encapsulates:
# - Device: Where data lives (CPU, Metal, CUDA)
# - Comm: How processes communicate (MPI vs Serial)
# - Solver: How linear systems are solved (MUMPS, cuDSS variants)
#
# The HPCBackend type unifies these concerns into a single type parameter for
# distributed array types (HPCVector, HPCMatrix, HPCSparseMatrix).

# ============================================================================
# Device Types
# ============================================================================

"""
    AbstractDevice

Abstract base type for compute devices (CPU, GPU backends).
"""
abstract type AbstractDevice end

"""
    DeviceCPU <: AbstractDevice

CPU device - uses standard Julia arrays (Vector, Matrix).
"""
struct DeviceCPU <: AbstractDevice end

"""
    DeviceMetal <: AbstractDevice

Metal GPU device (macOS) - uses Metal.jl arrays (MtlVector, MtlMatrix).
Defined here as a placeholder; actual Metal support is in the Metal extension.
"""
struct DeviceMetal <: AbstractDevice end

"""
    DeviceCUDA <: AbstractDevice

CUDA GPU device - uses CUDA.jl arrays (CuVector, CuMatrix).
Defined here as a placeholder; actual CUDA support is in the CUDA extension.
"""
struct DeviceCUDA <: AbstractDevice end

# ============================================================================
# Comm Types
# ============================================================================

"""
    AbstractComm

Abstract base type for communication backends.
"""
abstract type AbstractComm end

"""
    CommSerial <: AbstractComm

Serial (single-process) communication - all collective operations become no-ops.
"""
struct CommSerial <: AbstractComm end

"""
    CommMPI <: AbstractComm

MPI-based distributed communication.

# Fields
- `comm::MPI.Comm`: The MPI communicator to use for all operations.
"""
struct CommMPI <: AbstractComm
    comm::MPI.Comm
end

# ============================================================================
# Solver Types
# ============================================================================

"""
    AbstractSolver

Abstract base type for linear solvers.
"""
abstract type AbstractSolver end

"""
    SolverMUMPS <: AbstractSolver

MUMPS sparse direct solver. Works on CPU with MPI or serial communication.
"""
struct SolverMUMPS <: AbstractSolver end

"""
    AbstractSolverCuDSS <: AbstractSolver

Abstract base type for cuDSS (CUDA Direct Sparse Solver) variants.
Concrete types are defined in the CUDA extension.
"""
abstract type AbstractSolverCuDSS <: AbstractSolver end

# Note: SolverCuDSS_Single, SolverCuDSS_MG, SolverCuDSS_MGMN are defined
# in ext/HPCLinearAlgebraCUDAExt.jl since they depend on CUDA types.

# ============================================================================
# HPCBackend Type
# ============================================================================

"""
    HPCBackend{T, Ti<:Integer, D<:AbstractDevice, C<:AbstractComm, S<:AbstractSolver}

Unified backend type that encapsulates element type, index type, device, communication,
and solver configuration.

# Type Parameters
- `T`: Element type for array values (e.g., Float64, ComplexF64)
- `Ti`: Index type for sparse matrix indices (e.g., Int32, Int64)
- `D`: Device type (DeviceCPU, DeviceMetal, DeviceCUDA)
- `C`: Communication type (CommSerial, CommMPI)
- `S`: Solver type (SolverMUMPS, or cuDSS variants)

# Fields
- `device::D`: The compute device
- `comm::C`: The communication backend
- `solver::S`: The linear solver

# Example
```julia
# CPU with MPI and MUMPS, Float64 values and Int32 indices
backend = backend_cpu_mpi(Float64, Int32)

# CPU serial (single process) with default types (Float64, Int)
backend = backend_cpu_serial()
```
"""
struct HPCBackend{T, Ti<:Integer, D<:AbstractDevice, C<:AbstractComm, S<:AbstractSolver}
    device::D
    comm::C
    solver::S
end

# ============================================================================
# Type Extraction Helpers
# ============================================================================

"""
    eltype_backend(::Type{<:HPCBackend{T}}) where T -> Type
    eltype_backend(b::HPCBackend) -> Type

Extract the element type T from an HPCBackend type or instance.
"""
eltype_backend(::Type{<:HPCBackend{T}}) where {T} = T
eltype_backend(b::HPCBackend) = eltype_backend(typeof(b))

"""
    indextype_backend(::Type{<:HPCBackend{T,Ti}}) where {T,Ti} -> Type
    indextype_backend(b::HPCBackend) -> Type

Extract the index type Ti from an HPCBackend type or instance.
"""
indextype_backend(::Type{<:HPCBackend{T,Ti}}) where {T,Ti} = Ti
indextype_backend(b::HPCBackend) = indextype_backend(typeof(b))

# Type aliases for convenience
const HPCBackendCPU{T,Ti,C,S} = HPCBackend{T,Ti,DeviceCPU,C,S}
const HPCBackendMetal{T,Ti,C,S} = HPCBackend{T,Ti,DeviceMetal,C,S}
const HPCBackendCUDA{T,Ti,C,S} = HPCBackend{T,Ti,DeviceCUDA,C,S}

# Common configuration aliases
const HPCBackend_CPU_MPI{T,Ti} = HPCBackend{T,Ti,DeviceCPU,CommMPI,SolverMUMPS}
const HPCBackend_CPU_Serial{T,Ti} = HPCBackend{T,Ti,DeviceCPU,CommSerial,SolverMUMPS}

# ============================================================================
# Device → Array Type Mapping
# ============================================================================

"""
    array_type(device::AbstractDevice, ::Type{T}) where T

Return the appropriate vector type for the given device and element type.
"""
array_type(::DeviceCPU, ::Type{T}) where T = Vector{T}

# Metal and CUDA array types are defined in their respective extensions
# These are placeholder methods that will be extended

"""
    matrix_type(device::AbstractDevice, ::Type{T}) where T

Return the appropriate matrix type for the given device and element type.
"""
matrix_type(::DeviceCPU, ::Type{T}) where T = Matrix{T}

# ============================================================================
# Communication Primitives
# ============================================================================
#
# These functions abstract MPI operations so that CommSerial can provide
# no-op implementations while CommMPI calls the actual MPI functions.

"""
    comm_rank(comm::AbstractComm) -> Int

Return the rank of this process (0 for CommSerial).
"""
comm_rank(::CommSerial) = 0
comm_rank(c::CommMPI) = MPI.Comm_rank(c.comm)

"""
    comm_size(comm::AbstractComm) -> Int

Return the total number of processes (1 for CommSerial).
"""
comm_size(::CommSerial) = 1
comm_size(c::CommMPI) = MPI.Comm_size(c.comm)

"""
    comm_allgather(comm::AbstractComm, data)

Gather data from all processes to all processes.
For CommSerial, returns the data directly (since there's only one process).
For vector data, this mimics MPI.Allgather which concatenates all ranks' data into a flat vector.
"""
comm_allgather(::CommSerial, data::AbstractVector) = copy(data)  # Single rank: data is already the "concatenation"
comm_allgather(::CommSerial, data) = [data]  # Scalar data: wrap in vector
comm_allgather(c::CommMPI, data) = MPI.Allgather(data, c.comm)

"""
    comm_allgatherv!(comm::AbstractComm, sendbuf, recvbuf::MPI.VBuffer)

Variable-length allgather. For CommSerial, copies sendbuf to recvbuf.
"""
function comm_allgatherv!(::CommSerial, sendbuf, recvbuf)
    copyto!(recvbuf.data, sendbuf)
    return recvbuf.data
end
comm_allgatherv!(c::CommMPI, sendbuf, recvbuf) = MPI.Allgatherv!(sendbuf, recvbuf, c.comm)

"""
    comm_allgather!(comm::AbstractComm, sendbuf, recvbuf::MPI.UBuffer)

In-place allgather with uniform buffer. For CommSerial, copies sendbuf to recvbuf.
"""
function comm_allgather!(::CommSerial, sendbuf, recvbuf::MPI.UBuffer)
    recvbuf.data[1] = sendbuf[]
    return recvbuf.data
end
comm_allgather!(c::CommMPI, sendbuf, recvbuf) = MPI.Allgather!(sendbuf, recvbuf, c.comm)

"""
    comm_bcast!(comm::AbstractComm, buf, root::Int)

Broadcast buffer from root to all processes. For CommSerial, this is a no-op.
"""
comm_bcast!(::CommSerial, buf, root::Int) = buf
comm_bcast!(c::CommMPI, buf, root::Int) = MPI.Bcast!(buf, root, c.comm)

"""
    _julia_op_to_mpi(op) -> MPI.Op

Convert a Julia operator to an MPI operator.
"""
_julia_op_to_mpi(::typeof(+)) = MPI.SUM
_julia_op_to_mpi(::typeof(max)) = MPI.MAX
_julia_op_to_mpi(::typeof(min)) = MPI.MIN
_julia_op_to_mpi(::typeof(*)) = MPI.PROD
_julia_op_to_mpi(op::MPI.Op) = op  # Already an MPI op

"""
    comm_allreduce(comm::AbstractComm, data, op)

Reduce data across all processes. For CommSerial, returns data unchanged.
Accepts Julia operators (+, max, min, *) or MPI operators (MPI.SUM, MPI.MAX, etc.).
"""
comm_allreduce(::CommSerial, data, op) = data
comm_allreduce(c::CommMPI, data, op) = MPI.Allreduce(data, _julia_op_to_mpi(op), c.comm)

"""
    comm_alltoall(comm::AbstractComm, sendbuf::MPI.UBuffer)

All-to-all exchange. For CommSerial, returns a copy of the send data.
"""
comm_alltoall(::CommSerial, sendbuf::MPI.UBuffer) = copy(sendbuf.data)
comm_alltoall(c::CommMPI, sendbuf) = MPI.Alltoall(sendbuf, c.comm)

"""
    comm_alltoallv!(comm::AbstractComm, sendbuf::MPI.VBuffer, recvbuf::MPI.VBuffer)

Variable-length all-to-all exchange.
"""
function comm_alltoallv!(::CommSerial, sendbuf::MPI.VBuffer, recvbuf::MPI.VBuffer)
    copyto!(recvbuf.data, 1, sendbuf.data, 1, length(sendbuf.data))
    return recvbuf.data
end
comm_alltoallv!(c::CommMPI, sendbuf, recvbuf) = MPI.Alltoallv!(sendbuf, recvbuf, c.comm)

"""
    comm_isend(comm::AbstractComm, data, dest::Int, tag::Int) -> request

Non-blocking send. For CommSerial, this is a no-op (returns nothing).
"""
comm_isend(::CommSerial, data, dest::Int, tag::Int) = nothing
comm_isend(c::CommMPI, data, dest::Int, tag::Int) = MPI.Isend(data, c.comm; dest=dest, tag=tag)

"""
    comm_irecv!(comm::AbstractComm, buf, source::Int, tag::Int) -> request

Non-blocking receive. For CommSerial, this is a no-op (returns nothing).
"""
comm_irecv!(::CommSerial, buf, source::Int, tag::Int) = nothing
comm_irecv!(c::CommMPI, buf, source::Int, tag::Int) = MPI.Irecv!(buf, c.comm; source=source, tag=tag)

"""
    comm_waitall(comm::AbstractComm, requests)

Wait for all requests to complete. For CommSerial, this is a no-op.
"""
comm_waitall(::CommSerial, requests) = nothing
function comm_waitall(::CommMPI, requests)
    # Filter out nothing entries and convert to Vector{MPI.Request}
    # (nothing entries can appear if some requests are skipped, e.g., rank == source)
    valid_requests = MPI.Request[r for r in requests if r !== nothing]
    if !isempty(valid_requests)
        MPI.Waitall(valid_requests)
    end
end

# ============================================================================
# HPCBackend Factory Functions
# ============================================================================

"""
    backend_cpu_serial(::Type{T}=Float64, ::Type{Ti}=Int) where {T,Ti} -> HPCBackend

Create a CPU backend with serial (single-process) communication and MUMPS solver.

# Arguments
- `T`: Element type for array values (default: Float64)
- `Ti`: Index type for sparse matrix indices (default: Int)

# Example
```julia
backend = backend_cpu_serial()                    # Float64, Int
backend = backend_cpu_serial(Float32, Int32)      # Float32, Int32
```
"""
function backend_cpu_serial(::Type{T}=Float64, ::Type{Ti}=Int) where {T,Ti<:Integer}
    return HPCBackend{T,Ti,DeviceCPU,CommSerial,SolverMUMPS}(DeviceCPU(), CommSerial(), SolverMUMPS())
end

"""
    backend_cpu_mpi(::Type{T}=Float64, ::Type{Ti}=Int; comm=MPI.COMM_WORLD) where {T,Ti} -> HPCBackend

Create a CPU backend with MPI communication and MUMPS solver.

# Arguments
- `T`: Element type for array values (default: Float64)
- `Ti`: Index type for sparse matrix indices (default: Int)
- `comm`: MPI communicator (default: MPI.COMM_WORLD)

# Example
```julia
backend = backend_cpu_mpi()                       # Float64, Int, COMM_WORLD
backend = backend_cpu_mpi(Float64, Int32)         # Float64, Int32, COMM_WORLD
backend = backend_cpu_mpi(Float64, Int32; comm=my_comm)  # Custom communicator
```
"""
function backend_cpu_mpi(::Type{T}=Float64, ::Type{Ti}=Int; comm::MPI.Comm=MPI.COMM_WORLD) where {T,Ti<:Integer}
    return HPCBackend{T,Ti,DeviceCPU,CommMPI,SolverMUMPS}(DeviceCPU(), CommMPI(comm), SolverMUMPS())
end

# Legacy overload for backward compatibility (comm as positional argument)
function backend_cpu_mpi(comm::MPI.Comm)
    return backend_cpu_mpi(Float64, Int; comm=comm)
end

# ============================================================================
# Pre-constructed Backend Constants (Deprecated)
# ============================================================================
#
# These constants use default types (Float64, Int) for backward compatibility.
# New code should use the factory functions with explicit type parameters.

"""
    BACKEND_CPU_SERIAL

Pre-constructed CPU backend with serial communication, Float64 values, and Int indices.

!!! warning "Deprecated"
    Use `backend_cpu_serial(T, Ti)` instead for explicit type control.
"""
const BACKEND_CPU_SERIAL = HPCBackend{Float64,Int,DeviceCPU,CommSerial,SolverMUMPS}(
    DeviceCPU(), CommSerial(), SolverMUMPS())

"""
    BACKEND_CPU_MPI

Pre-constructed CPU backend with MPI communication, Float64 values, and Int indices.

!!! warning "Deprecated"
    Use `backend_cpu_mpi(T, Ti)` instead for explicit type control.
"""
const BACKEND_CPU_MPI = HPCBackend{Float64,Int,DeviceCPU,CommMPI,SolverMUMPS}(
    DeviceCPU(), CommMPI(MPI.COMM_WORLD), SolverMUMPS())

# GPU backend factory functions are defined in the extensions
# These are placeholder declarations so extensions can add methods

"""
    backend_metal_mpi(comm::MPI.Comm) -> HPCBackend

Create a Metal GPU backend with MPI communication.
Requires the Metal extension to be loaded.
"""
function backend_metal_mpi end

"""
    backend_cuda_serial() -> HPCBackend

Create a CUDA GPU backend with serial (single-process) communication and cuDSS solver.
Requires the CUDA extension to be loaded.
"""
function backend_cuda_serial end

"""
    backend_cuda_mpi(comm::MPI.Comm) -> HPCBackend

Create a CUDA GPU backend with MPI communication and cuDSS solver.
Requires the CUDA extension to be loaded.
"""
function backend_cuda_mpi end

# ============================================================================
# HPCBackend Comparison
# ============================================================================

"""
    backends_compatible(b1::HPCBackend, b2::HPCBackend) -> Bool

Check if two backends are compatible for operations between their arrays.
HPCBackends must have the same device and comm to be compatible.
"""
function backends_compatible(b1::HPCBackend, b2::HPCBackend)
    # Device must match
    typeof(b1.device) != typeof(b2.device) && return false
    # Comm must match (for MPI, the actual communicator must be the same)
    typeof(b1.comm) != typeof(b2.comm) && return false
    if b1.comm isa CommMPI && b2.comm isa CommMPI
        b1.comm.comm != b2.comm.comm && return false
    end
    return true
end

"""
    assert_backends_compatible(b1::HPCBackend, b2::HPCBackend)

Assert that two backends are compatible, throwing an error if not.
"""
function assert_backends_compatible(b1::HPCBackend, b2::HPCBackend)
    if !backends_compatible(b1, b2)
        error("Incompatible backends: $(typeof(b1)) vs $(typeof(b2))")
    end
end

"""
    retype_backend(backend::HPCBackend, ::Type{Tnew}) -> HPCBackend

Create a new backend with a different element type, preserving all other parameters.
Useful for operations that change element type (e.g., abs on complex matrices returns real).

# Arguments
- `backend`: The original backend
- `Tnew`: The new element type

# Example
```julia
complex_backend = backend_cpu_mpi(ComplexF64)
real_backend = retype_backend(complex_backend, Float64)
```
"""
function retype_backend(backend::HPCBackend{T,Ti,D,C,S}, ::Type{Tnew}) where {T,Ti,D,C,S,Tnew}
    if T === Tnew
        return backend  # No change needed
    end
    return HPCBackend{Tnew,Ti,D,C,S}(backend.device, backend.comm, backend.solver)
end
