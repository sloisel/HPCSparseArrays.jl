# Shared test utilities for parameterized testing
# This module provides test configurations for CPU and GPU backends

module TestUtils

using SparseArrays
using MPI

# Detect Metal availability BEFORE loading HPCLinearAlgebra
# (Metal must be loaded first for GPU detection to work)
const METAL_AVAILABLE = try
    using Metal
    Metal.functional()
catch e
    false
end

if METAL_AVAILABLE
    @info "Metal is available for GPU tests"
end

# Detect CUDA availability BEFORE loading HPCLinearAlgebra
# (CUDA/NCCL/CUDSS_jll must be loaded first for GPU extension to work)
const CUDA_AVAILABLE = try
    using CUDA
    using NCCL
    using CUDSS_jll
    CUDA.functional()
catch e
    false
end

if CUDA_AVAILABLE
    @info "CUDA is available for GPU tests"
end

# Import HPCLinearAlgebra after GPU checks
using HPCLinearAlgebra

# ============================================================================
# Backend configurations for parameterized testing
# ============================================================================

# Backend getters - return pre-constructed singletons (available after MPI.Init)
function get_cpu_backend()
    HPCLinearAlgebra.BACKEND_CPU_MPI
end

function get_metal_backend()
    @assert METAL_AVAILABLE "Metal is not available"
    HPCLinearAlgebra.backend_metal_mpi(MPI.COMM_WORLD)  # Metal still uses function with comm
end

function get_cuda_backend()
    @assert CUDA_AVAILABLE "CUDA is not available"
    HPCLinearAlgebra.backend_cuda_mpi()  # Zero-arg returns singleton
end

# Backend configurations: (ScalarType, backend_getter, backend_name)
# The backend_getter is a function that returns the backend (deferred to after MPI.Init)
const CPU_CONFIGS = [
    (Float64, get_cpu_backend, "CPU"),
    (ComplexF64, get_cpu_backend, "CPU")
]

const GPU_CONFIGS = begin
    configs = Tuple{Type, Function, String}[]
    if METAL_AVAILABLE
        push!(configs, (Float32, get_metal_backend, "Metal"))
        # ComplexF32 skipped - Julia's complex ops use Float64 internally, unsupported on Metal
    end
    if CUDA_AVAILABLE
        push!(configs, (Float32, get_cuda_backend, "CUDA"))
        push!(configs, (Float64, get_cuda_backend, "CUDA"))  # CUDA supports Float64
    end
    configs
end

const ALL_CONFIGS = [CPU_CONFIGS; GPU_CONFIGS]

# For operations that only work on CPU (like MUMPS factorization)
const CPU_ONLY_CONFIGS = CPU_CONFIGS

"""
    tridiagonal_matrix(T, n=8)

Generate a deterministic tridiagonal test matrix of type T.
"""
function tridiagonal_matrix(::Type{T}, n::Int=8) where T
    I = [1:n; 1:n-1; 2:n]
    J = [1:n; 2:n; 1:n-1]
    V = if T <: Complex
        T.([2.0*ones(n); -0.5*ones(n-1); -0.5*ones(n-1)]) .+
        im .* T.([0.1*ones(n); 0.2*ones(n-1); -0.2*ones(n-1)])
    else
        T.([2.0*ones(n); -0.5*ones(n-1); -0.5*ones(n-1)])
    end
    sparse(I, J, V, n, n)
end

"""
    dense_matrix(T, m, n)

Generate a deterministic dense test matrix of type T.
"""
function dense_matrix(::Type{T}, m::Int, n::Int) where T
    A = zeros(T, m, n)
    for i in 1:m, j in 1:n
        if T <: Complex
            A[i, j] = T(i + j) + im * T(i - j)
        else
            A[i, j] = T(i + j)
        end
    end
    A
end

"""
    test_vector(T, n=8)

Generate a deterministic test vector of type T.
"""
function test_vector(::Type{T}, n::Int=8) where T
    if T <: Complex
        T.(1:n) .+ im .* T.(n:-1:1)
    else
        T.(collect(1.0:n))
    end
end

"""
    test_vector_pair(T, n=8)

Generate two deterministic test vectors for addition/subtraction tests.
"""
function test_vector_pair(::Type{T}, n::Int=8) where T
    u = test_vector(T, n)
    v = if T <: Complex
        T.(n:-1:1) .+ im .* T.(1:n)
    else
        T.(collect(Float64(n):-1.0:1))
    end
    u, v
end

"""
    tolerance(T)

Return appropriate tolerance for type T.
Float32 tolerance is looser (1e-4) to accommodate accumulated errors
in matrix operations like transpose(A) * B.
"""
tolerance(::Type{Float64}) = 1e-10
tolerance(::Type{ComplexF64}) = 1e-10
tolerance(::Type{Float32}) = 1e-4
tolerance(::Type{ComplexF32}) = 1e-4

"""
    assert_uniform(x; atol=0, rtol=0, name="value")

Assert that scalar/small value `x` is identical across all MPI ranks.
Uses Allgather to collect values from all ranks and compares them.
Throws an error if values differ, without desynchronizing the cluster.

For exact values (Int, Bool, Type): `assert_uniform(x)`
For Float comparisons: `assert_uniform(x, atol=tol)` or `assert_uniform(x, rtol=tol)`
"""
function assert_uniform(x; atol=0, rtol=0, name="value")
    comm = MPI.COMM_WORLD
    nranks = MPI.Comm_size(comm)

    # Gather all values to all ranks
    all_values = MPI.Allgather(x, comm)

    # Check uniformity
    ref = all_values[1]
    for i in 2:nranks
        if atol == 0 && rtol == 0
            # Exact equality
            if all_values[i] != ref
                error("assert_uniform failed for '$name': rank 0 has $ref, rank $(i-1) has $(all_values[i])")
            end
        else
            # Approximate equality for floats
            if !isapprox(all_values[i], ref; atol=atol, rtol=rtol)
                error("assert_uniform failed for '$name': rank 0 has $ref, rank $(i-1) has $(all_values[i])")
            end
        end
    end
    return x
end

"""
    cpu_version(backend::HPCBackend) -> HPCBackend

Create a CPU version of the given backend, preserving comm but using MUMPS solver.
Use with `to_backend(x, cpu_version(x.backend))` to convert GPU data to CPU for comparison.

Note: Always uses SolverMUMPS since GPU solvers (cuDSS) don't work on CPU.
The solver choice only matters for lu()/ldlt(), not for data comparison.
"""
function cpu_version(backend::HPCLinearAlgebra.HPCBackend)
    HPCLinearAlgebra.HPCBackend(
        HPCLinearAlgebra.DeviceCPU(), backend.comm, HPCLinearAlgebra.SolverMUMPS()
    )
end

"""
    local_values(v::HPCVector)

Get local values as a CPU array for comparison.
Works for both CPU and GPU vectors by checking the backend device type.
"""
function local_values(v::HPCVector)
    # Check if data is on GPU by looking at backend device type
    if v.backend.device isa HPCLinearAlgebra.DeviceCPU
        return v.v
    else
        # GPU case: convert to Array
        return Array(v.v)
    end
end

# ============================================================================
# Type assertions for catching GPU/CPU type mismatches
# ============================================================================

# Detect concrete Metal storage type at module load time
# This allows us to assert exact types, not just subtypes
if METAL_AVAILABLE
    # Create small test arrays to capture the concrete types
    # MtlVector{T, S} is actually MtlArray{T, 1, S} so parameters are (T, ndims, Storage)
    const _MTL_STORAGE = let
        v = Metal.MtlVector{Float32}(undef, 1)
        typeof(v).parameters[3]  # Extract storage type (e.g., Metal.PrivateStorage)
    end
    @info "Detected Metal storage type: $_MTL_STORAGE"
end

# Detect concrete CUDA storage type at module load time
if CUDA_AVAILABLE
    # CuVector{T} is CuArray{T, 1, CUDA.DeviceMemory}
    const _CUDA_STORAGE = let
        v = CUDA.CuVector{Float32}(undef, 1)
        typeof(v).parameters[3]  # Extract storage type
    end
    @info "Detected CUDA storage type: $_CUDA_STORAGE"
end

"""
    assert_type(x, T)

Assert that typeof(x) === T exactly and return x.
This catches any type mismatch, including GPU/CPU mismatches.
Throws TypeError if the check fails.

Example:
    result = assert_type(A * x, VT)  # Throws if typeof(result) !== VT
"""
function assert_type(x, ::Type{T}) where T
    typeof(x) === T || throw(TypeError(:assert_type, "", T, x))
    x
end

"""
    expected_types(T, backend)

Returns (VectorType, SparseType, DenseType) for type assertions in tests.
Returns fully concrete types for exact matching.

Example:
    for (T, get_backend, backend_name) in ALL_CONFIGS
        backend = get_backend()
        VT, ST, MT = expected_types(T, backend)
        result = assert_type(A * x, VT)  # Assert exact type match
    end
"""
function expected_types(::Type{T}, backend::HPCLinearAlgebra.HPCBackend) where T
    BK = typeof(backend)
    (HPCVector{T, BK},
     HPCSparseMatrix{T, Int, BK},
     HPCMatrix{T, BK})
end

export METAL_AVAILABLE, CUDA_AVAILABLE, CPU_CONFIGS, GPU_CONFIGS, ALL_CONFIGS, CPU_ONLY_CONFIGS
export tridiagonal_matrix, dense_matrix, test_vector, test_vector_pair
export tolerance, cpu_version, local_values, expected_types, assert_type, assert_uniform
export get_cpu_backend, get_metal_backend, get_cuda_backend

end # module
