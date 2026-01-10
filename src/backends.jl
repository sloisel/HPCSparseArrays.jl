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
# distributed array types (VectorMPI, MatrixMPI, SparseMatrixMPI).

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
# in ext/LinearAlgebraMPICUDAExt.jl since they depend on CUDA types.

# ============================================================================
# HPCBackend Type
# ============================================================================

"""
    HPCBackend{D<:AbstractDevice, C<:AbstractComm, S<:AbstractSolver}

Unified backend type that encapsulates device, communication, and solver configuration.

# Type Parameters
- `D`: Device type (DeviceCPU, DeviceMetal, DeviceCUDA)
- `C`: Communication type (CommSerial, CommMPI)
- `S`: Solver type (SolverMUMPS, or cuDSS variants)

# Fields
- `device::D`: The compute device
- `comm::C`: The communication backend
- `solver::S`: The linear solver

# Example
```julia
# CPU with MPI and MUMPS
backend = HPCBackend(DeviceCPU(), CommMPI(MPI.COMM_WORLD), SolverMUMPS())

# CPU serial (single process)
backend = HPCBackend(DeviceCPU(), CommSerial(), SolverMUMPS())
```
"""
struct HPCBackend{D<:AbstractDevice, C<:AbstractComm, S<:AbstractSolver}
    device::D
    comm::C
    solver::S
end

# Type aliases for convenience
const HPCBackendCPU{C,S} = HPCBackend{DeviceCPU,C,S}
const HPCBackendMetal{C,S} = HPCBackend{DeviceMetal,C,S}
const HPCBackendCUDA{C,S} = HPCBackend{DeviceCUDA,C,S}

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

"""
    comm_barrier(comm::AbstractComm)

Synchronization barrier. For CommSerial, this is a no-op.
"""
comm_barrier(::CommSerial) = nothing
comm_barrier(c::CommMPI) = MPI.Barrier(c.comm)

# ============================================================================
# HPCBackend Factory Functions
# ============================================================================

"""
    backend_cpu_serial() -> HPCBackend

Create a CPU backend with serial (single-process) communication and MUMPS solver.
"""
function backend_cpu_serial()
    return HPCBackend(DeviceCPU(), CommSerial(), SolverMUMPS())
end

"""
    backend_cpu_mpi(comm::MPI.Comm) -> HPCBackend

Create a CPU backend with MPI communication and MUMPS solver.

# Arguments
- `comm::MPI.Comm`: The MPI communicator to use.
"""
function backend_cpu_mpi(comm::MPI.Comm)
    return HPCBackend(DeviceCPU(), CommMPI(comm), SolverMUMPS())
end

# GPU backend factory functions are defined in the extensions
# These are placeholder declarations so extensions can add methods

"""
    backend_metal_mpi(comm::MPI.Comm) -> HPCBackend

Create a Metal GPU backend with MPI communication.
Requires the Metal extension to be loaded.
"""
function backend_metal_mpi end

"""
    backend_cuda_mpi(comm::MPI.Comm) -> HPCBackend

Create a CUDA GPU backend with MPI communication.
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
