"""
    LinearAlgebraMPIMetalExt

Extension module for Metal GPU support in LinearAlgebraMPI.
Provides constructors and operations for MtlArray-backed distributed arrays.
"""
module LinearAlgebraMPIMetalExt

using LinearAlgebraMPI
using Metal
using Adapt

# Import HPCBackend types for type-based dispatch
using LinearAlgebraMPI: HPCBackend, DeviceCPU, DeviceMetal, DeviceCUDA,
                        CommSerial, CommMPI, AbstractComm, AbstractDevice,
                        SolverMUMPS

# Backend type aliases for Metal
const MtlBackend{C,S} = LinearAlgebraMPI.HPCBackend{LinearAlgebraMPI.DeviceMetal, C, S}
const CPUBackend{C,S} = LinearAlgebraMPI.HPCBackend{LinearAlgebraMPI.DeviceCPU, C, S}

"""
    backend_metal_mpi(comm::MPI.Comm) -> HPCBackend

Create a Metal GPU backend with MPI communication and MUMPS solver.
Metal doesn't have a native sparse direct solver, so MUMPS is used (data staged via CPU).
"""
function LinearAlgebraMPI.backend_metal_mpi(comm)
    return LinearAlgebraMPI.HPCBackend(DeviceMetal(), CommMPI(comm), SolverMUMPS())
end

# ============================================================================
# _convert_array methods for Metal
# ============================================================================

# CPU → Metal: copy to GPU
LinearAlgebraMPI._convert_array(v::Vector, ::LinearAlgebraMPI.DeviceMetal) = MtlVector(v)
LinearAlgebraMPI._convert_array(A::Matrix, ::LinearAlgebraMPI.DeviceMetal) = MtlMatrix(A)

# Metal → Metal: identity (no copy)
LinearAlgebraMPI._convert_array(v::MtlVector, ::LinearAlgebraMPI.DeviceMetal) = v
LinearAlgebraMPI._convert_array(A::MtlMatrix, ::LinearAlgebraMPI.DeviceMetal) = A

# ============================================================================
# MUMPS Factorization Support
# ============================================================================

"""
    _array_to_device(v::Vector{T}, ::LinearAlgebraMPI.DeviceMetal) where T

Convert a CPU vector to a Metal GPU vector.
Used by MUMPS factorization for round-trip GPU conversion during solve.
"""
function LinearAlgebraMPI._array_to_device(v::Vector{T}, ::LinearAlgebraMPI.DeviceMetal) where T
    return MtlVector(v)
end

"""
    _convert_vector_to_device(v::LinearAlgebraMPI.VectorMPI, ::LinearAlgebraMPI.DeviceMetal)

Convert a CPU VectorMPI to GPU (Metal) backend.
Used by MUMPS factorization for GPU reconstruction after solve.
"""
function LinearAlgebraMPI._convert_vector_to_device(v::LinearAlgebraMPI.VectorMPI, device::LinearAlgebraMPI.DeviceMetal)
    # Create Metal backend preserving comm and solver
    metal_backend = LinearAlgebraMPI.HPCBackend(device, v.backend.comm, v.backend.solver)
    return LinearAlgebraMPI.to_backend(v, metal_backend)
end

# ============================================================================
# Base.zeros Support
# ============================================================================

"""
    _zeros_device(::LinearAlgebraMPI.DeviceMetal, ::Type{T}, dims...) where T

Create a zero MtlVector/MtlMatrix of the specified dimensions on Metal device.
Used by Base.zeros(VectorMPI, backend, n) etc.
"""
LinearAlgebraMPI._zeros_device(::LinearAlgebraMPI.DeviceMetal, ::Type{T}, dims...) where T = Metal.zeros(T, dims...)

# ============================================================================
# MatrixPlan Index Array Support
# ============================================================================

"""
    _index_array_type(::LinearAlgebraMPI.DeviceMetal, ::Type{Ti}) where Ti

Map DeviceMetal to MtlVector{Ti} index array type.
Used by MatrixPlan to store symbolic index arrays on GPU.
"""
LinearAlgebraMPI._index_array_type(::LinearAlgebraMPI.DeviceMetal, ::Type{Ti}) where Ti = MtlVector{Ti}

"""
    _to_target_device(v::Vector{Ti}, ::LinearAlgebraMPI.DeviceMetal) where Ti

Convert a CPU index vector to Metal GPU.
Used by SparseMatrixMPI constructors to create GPU structure arrays.
"""
LinearAlgebraMPI._to_target_device(v::Vector{Ti}, ::LinearAlgebraMPI.DeviceMetal) where Ti = MtlVector(v)

# ============================================================================
# GPU map_rows_gpu implementation via Metal kernels
# ============================================================================

using StaticArrays

"""
    _map_rows_gpu_kernel(f, arg1::MtlMatrix, rest::MtlMatrix...)

GPU-accelerated row-wise map for Metal arrays.
Each thread processes one row, applying `f` to the corresponding rows of all input matrices.
Returns a Metal matrix with the same number of rows.
"""
function LinearAlgebraMPI._map_rows_gpu_kernel(f, arg1::MtlMatrix{T}, rest::MtlMatrix...) where T
    n = size(arg1, 1)

    # Get output size by evaluating f on first row (copy to CPU to avoid scalar indexing)
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
        out_cols = length(sample_out)  # Flatten matrix output
    else
        out_cols = 1  # Scalar output
    end

    # Allocate output
    output = Metal.zeros(T, n, out_cols)

    # Create kernel
    _map_rows_kernel_dispatch(f, output, arg1, rest...)

    return output
end

"""
Dispatch to appropriate kernel based on number of arguments.
"""
function _map_rows_kernel_dispatch(f, output::MtlMatrix{T}, arg1::MtlMatrix{T}) where T
    n = size(arg1, 1)
    ncols1 = size(arg1, 2)
    out_cols = size(output, 2)

    kernel = @metal launch=false _map_rows_kernel_1arg(f, output, arg1, Val(ncols1), Val(out_cols))
    threads = min(n, 256)
    groups = cld(n, threads)
    kernel(f, output, arg1, Val(ncols1), Val(out_cols); threads=threads, groups=groups)
    Metal.synchronize()
end

function _map_rows_kernel_dispatch(f, output::MtlMatrix{T}, arg1::MtlMatrix{T}, arg2::MtlMatrix{T}) where T
    n = size(arg1, 1)
    ncols1 = size(arg1, 2)
    ncols2 = size(arg2, 2)
    out_cols = size(output, 2)

    kernel = @metal launch=false _map_rows_kernel_2args(f, output, arg1, arg2, Val(ncols1), Val(ncols2), Val(out_cols))
    threads = min(n, 256)
    groups = cld(n, threads)
    kernel(f, output, arg1, arg2, Val(ncols1), Val(ncols2), Val(out_cols); threads=threads, groups=groups)
    Metal.synchronize()
end

function _map_rows_kernel_dispatch(f, output::MtlMatrix{T}, arg1::MtlMatrix{T}, arg2::MtlMatrix{T}, arg3::MtlMatrix{T}) where T
    n = size(arg1, 1)
    ncols1 = size(arg1, 2)
    ncols2 = size(arg2, 2)
    ncols3 = size(arg3, 2)
    out_cols = size(output, 2)

    kernel = @metal launch=false _map_rows_kernel_3args(f, output, arg1, arg2, arg3, Val(ncols1), Val(ncols2), Val(ncols3), Val(out_cols))
    threads = min(n, 256)
    groups = cld(n, threads)
    kernel(f, output, arg1, arg2, arg3, Val(ncols1), Val(ncols2), Val(ncols3), Val(out_cols); threads=threads, groups=groups)
    Metal.synchronize()
end

function _map_rows_kernel_dispatch(f, output::MtlMatrix{T}, arg1::MtlMatrix{T}, arg2::MtlMatrix{T}, arg3::MtlMatrix{T}, arg4::MtlMatrix{T}) where T
    n = size(arg1, 1)
    ncols1 = size(arg1, 2)
    ncols2 = size(arg2, 2)
    ncols3 = size(arg3, 2)
    ncols4 = size(arg4, 2)
    out_cols = size(output, 2)

    kernel = @metal launch=false _map_rows_kernel_4args(f, output, arg1, arg2, arg3, arg4, Val(ncols1), Val(ncols2), Val(ncols3), Val(ncols4), Val(out_cols))
    threads = min(n, 256)
    groups = cld(n, threads)
    kernel(f, output, arg1, arg2, arg3, arg4, Val(ncols1), Val(ncols2), Val(ncols3), Val(ncols4), Val(out_cols); threads=threads, groups=groups)
    Metal.synchronize()
end

function _map_rows_kernel_dispatch(f, output::MtlMatrix{T}, arg1::MtlMatrix{T}, arg2::MtlMatrix{T}, arg3::MtlMatrix{T}, arg4::MtlMatrix{T}, arg5::MtlMatrix{T}) where T
    n = size(arg1, 1)
    ncols1 = size(arg1, 2)
    ncols2 = size(arg2, 2)
    ncols3 = size(arg3, 2)
    ncols4 = size(arg4, 2)
    ncols5 = size(arg5, 2)
    out_cols = size(output, 2)

    kernel = @metal launch=false _map_rows_kernel_5args(f, output, arg1, arg2, arg3, arg4, arg5, Val(ncols1), Val(ncols2), Val(ncols3), Val(ncols4), Val(ncols5), Val(out_cols))
    threads = min(n, 256)
    groups = cld(n, threads)
    kernel(f, output, arg1, arg2, arg3, arg4, arg5, Val(ncols1), Val(ncols2), Val(ncols3), Val(ncols4), Val(ncols5), Val(out_cols); threads=threads, groups=groups)
    Metal.synchronize()
end

# ============================================================================
# Metal kernels
# ============================================================================

function _map_rows_kernel_1arg(f, output, arg1, ::Val{NC1}, ::Val{OCols}) where {NC1, OCols}
    i = thread_position_in_grid_1d()
    n = size(arg1, 1)
    if i <= n
        T = eltype(arg1)
        row1 = SVector{NC1,T}(ntuple(j -> @inbounds(arg1[i,j]), Val(NC1)))
        result = f(row1)
        _write_result!(output, i, result, Val(OCols))
    end
    return nothing
end

function _map_rows_kernel_2args(f, output, arg1, arg2, ::Val{NC1}, ::Val{NC2}, ::Val{OCols}) where {NC1, NC2, OCols}
    i = thread_position_in_grid_1d()
    n = size(arg1, 1)
    if i <= n
        T = eltype(arg1)
        row1 = SVector{NC1,T}(ntuple(j -> @inbounds(arg1[i,j]), Val(NC1)))
        row2 = SVector{NC2,T}(ntuple(j -> @inbounds(arg2[i,j]), Val(NC2)))
        result = f(row1, row2)
        _write_result!(output, i, result, Val(OCols))
    end
    return nothing
end

function _map_rows_kernel_3args(f, output, arg1, arg2, arg3, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{OCols}) where {NC1, NC2, NC3, OCols}
    i = thread_position_in_grid_1d()
    n = size(arg1, 1)
    if i <= n
        T = eltype(arg1)
        row1 = SVector{NC1,T}(ntuple(j -> @inbounds(arg1[i,j]), Val(NC1)))
        row2 = SVector{NC2,T}(ntuple(j -> @inbounds(arg2[i,j]), Val(NC2)))
        row3 = SVector{NC3,T}(ntuple(j -> @inbounds(arg3[i,j]), Val(NC3)))
        result = f(row1, row2, row3)
        _write_result!(output, i, result, Val(OCols))
    end
    return nothing
end

function _map_rows_kernel_4args(f, output, arg1, arg2, arg3, arg4, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{NC4}, ::Val{OCols}) where {NC1, NC2, NC3, NC4, OCols}
    i = thread_position_in_grid_1d()
    n = size(arg1, 1)
    if i <= n
        T = eltype(arg1)
        row1 = SVector{NC1,T}(ntuple(j -> @inbounds(arg1[i,j]), Val(NC1)))
        row2 = SVector{NC2,T}(ntuple(j -> @inbounds(arg2[i,j]), Val(NC2)))
        row3 = SVector{NC3,T}(ntuple(j -> @inbounds(arg3[i,j]), Val(NC3)))
        row4 = SVector{NC4,T}(ntuple(j -> @inbounds(arg4[i,j]), Val(NC4)))
        result = f(row1, row2, row3, row4)
        _write_result!(output, i, result, Val(OCols))
    end
    return nothing
end

function _map_rows_kernel_5args(f, output, arg1, arg2, arg3, arg4, arg5, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{NC4}, ::Val{NC5}, ::Val{OCols}) where {NC1, NC2, NC3, NC4, NC5, OCols}
    i = thread_position_in_grid_1d()
    n = size(arg1, 1)
    if i <= n
        T = eltype(arg1)
        row1 = SVector{NC1,T}(ntuple(j -> @inbounds(arg1[i,j]), Val(NC1)))
        row2 = SVector{NC2,T}(ntuple(j -> @inbounds(arg2[i,j]), Val(NC2)))
        row3 = SVector{NC3,T}(ntuple(j -> @inbounds(arg3[i,j]), Val(NC3)))
        row4 = SVector{NC4,T}(ntuple(j -> @inbounds(arg4[i,j]), Val(NC4)))
        row5 = SVector{NC5,T}(ntuple(j -> @inbounds(arg5[i,j]), Val(NC5)))
        result = f(row1, row2, row3, row4, row5)
        _write_result!(output, i, result, Val(OCols))
    end
    return nothing
end

# Helper to write result (scalar, SVector, or SMatrix) to output row
@inline function _write_result!(output, i, result::Number, ::Val{1})
    @inbounds output[i, 1] = result
    return nothing
end

@inline function _write_result!(output, i, result::SVector{N,T}, ::Val{N}) where {N,T}
    for j in 1:N
        @inbounds output[i, j] = result[j]
    end
    return nothing
end

@inline function _write_result!(output, i, result::SMatrix{M,N,T}, ::Val{MN}) where {M,N,T,MN}
    for j in 1:MN
        @inbounds output[i, j] = result[j]
    end
    return nothing
end

end # module
