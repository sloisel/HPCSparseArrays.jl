# User Guide

This guide covers the essential workflows for using HPCSparseArrays.jl.

## Core Types

HPCSparseArrays provides three distributed types:

| Type | Description | Storage |
|------|-------------|---------|
| `HPCVector{T,B}` | Distributed vector | Row-partitioned |
| `HPCMatrix{T,B}` | Distributed dense matrix | Row-partitioned |
| `HPCSparseMatrix{T,Ti,B}` | Distributed sparse matrix | Row-partitioned CSR |

The type parameters are:
- `T`: Element type (`Float64`, `Float32`, `ComplexF64`, etc.)
- `B<:HPCBackend`: Backend configuration (device, communication, solver)
- `Ti`: Index type for sparse matrices (typically `Int`)

### Backends

The `HPCBackend{Device, Comm, Solver}` type encapsulates three concerns:
- **Device**: Where data lives (`DeviceCPU`, `DeviceMetal`, `DeviceCUDA`)
- **Comm**: How ranks communicate (`CommSerial`, `CommMPI`)
- **Solver**: Sparse direct solver (`SolverMUMPS`, cuDSS variants)

Pre-constructed backends for common use cases:
- `BACKEND_CPU_SERIAL`: CPU with no MPI (single-process)
- `BACKEND_CPU_MPI`: CPU with MPI communication (most common)

GPU backends are created via factory functions after loading the GPU package:
- `backend_metal_mpi(comm)`: Metal GPU with MPI
- `backend_cuda_mpi(comm)`: CUDA GPU with MPI

All types are row-partitioned across MPI ranks, meaning each rank owns a contiguous range of rows.

### Internal Storage: CSR Format

Internally, `HPCSparseMatrix` stores local rows in CSR (Compressed Sparse Row) format using the `SparseMatrixCSR` type. This enables efficient row-wise iteration for a row-partitioned distributed matrix.

In Julia, `SparseMatrixCSR{T,Ti}` is a type alias for `Transpose{T, SparseMatrixCSC{T,Ti}}`. You don't need to worry about this for normal usage - it's handled automatically.

## Creating Distributed Types

### From Native Julia Types

```julia
using MPI
MPI.Init()
using HPCSparseArrays
using SparseArrays

# Use the default MPI backend
backend = BACKEND_CPU_MPI

# Create from native types (data is distributed automatically)
v = HPCVector(randn(100), backend)
A = HPCMatrix(randn(50, 30), backend)
S = HPCSparseMatrix(sprandn(100, 100, 0.1), backend)
```

### Local Constructors

For performance-critical code, use local constructors that avoid global communication:

```julia
# Create from local data (each rank provides its own rows)
v_local = HPCVector_local(my_local_vector)
A_local = HPCMatrix_local(my_local_matrix)
S_local = HPCSparseMatrix_local(my_local_sparse)
```

### Efficient Local-Only Construction

For large matrices, avoid replicating data across all ranks by only populating each rank's local portion:

```julia
backend = BACKEND_CPU_MPI
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Global dimensions
m, n = 1000, 1000

# Compute which rows this rank owns
rows_per_rank = div(m, nranks)
remainder = mod(m, nranks)
my_row_start = 1 + rank * rows_per_rank + min(rank, remainder)
my_row_end = my_row_start + rows_per_rank - 1 + (rank < remainder ? 1 : 0)

# Create sparse matrix with correct size, but only populate local rows
I, J, V = Int[], Int[], Float64[]
for i in my_row_start:my_row_end
    # Example: tridiagonal matrix
    if i > 1
        push!(I, i); push!(J, i-1); push!(V, -1.0)
    end
    push!(I, i); push!(J, i); push!(V, 2.0)
    if i < m
        push!(I, i); push!(J, i+1); push!(V, -1.0)
    end
end
A = sparse(I, J, V, m, n)

# The constructor extracts only local rows - other rows are ignored
Adist = HPCSparseMatrix(A, backend)
```

## Basic Operations

### Vector Operations

```julia
backend = BACKEND_CPU_MPI
v = HPCVector(randn(100), backend)
w = HPCVector(randn(100), backend)

# Arithmetic (backend is inferred from operands)
u = v + w
u = v - w
u = 2.0 * v
u = v * 2.0

# Linear algebra
n = norm(v)
d = dot(v, w)
c = conj(v)
```

### Matrix-Vector Products

```julia
backend = BACKEND_CPU_MPI
A = HPCMatrix(randn(50, 100), backend)
v = HPCVector(randn(100), backend)

# Matrix-vector multiply
y = A * v
```

### Sparse Operations

```julia
using SparseArrays

backend = BACKEND_CPU_MPI
A = HPCSparseMatrix(sprandn(100, 100, 0.1), backend)
v = HPCVector(randn(100), backend)

# Matrix-vector multiply
y = A * v

# Matrix-matrix multiply
B = HPCSparseMatrix(sprandn(100, 100, 0.1), backend)
C = A * B
```

## Solving Linear Systems

### Direct Solve with Backslash

```julia
using MPI
MPI.Init()
using HPCSparseArrays
using SparseArrays

backend = BACKEND_CPU_MPI

# Create a well-conditioned sparse matrix
A = HPCSparseMatrix(sprandn(100, 100, 0.1) + 10I, backend)
b = HPCVector(randn(100), backend)

# Solve A * x = b
x = A \ b
```

### Symmetric Systems (Faster)

For symmetric matrices, wrap with `Symmetric` to use faster LDLT factorization:

```julia
using LinearAlgebra

backend = BACKEND_CPU_MPI

# Create symmetric positive definite matrix
A_base = HPCSparseMatrix(sprandn(100, 100, 0.1), backend)
A_spd = A_base + HPCSparseMatrix(transpose(A_base), backend) +
        HPCSparseMatrix(sparse(10.0I, 100, 100), backend)

b = HPCVector(randn(100), backend)

# Use Symmetric wrapper for faster solve
x = Symmetric(A_spd) \ b
```

### Reusing Factorizations

For repeated solves with the same matrix, compute the factorization once:

```julia
using LinearAlgebra

# LU factorization
F = lu(A)
x1 = F \ b1
x2 = F \ b2
finalize!(F)  # Clean up MUMPS resources

# LDLT factorization (symmetric matrices)
F = ldlt(A_spd)
x = F \ b
finalize!(F)
```

## Threading

HPCSparseArrays uses two threading mechanisms for the MUMPS sparse direct solver:

1. **OpenMP threads** (`OMP_NUM_THREADS`) - Affects MUMPS algorithm-level parallelism
2. **BLAS threads** (`OPENBLAS_NUM_THREADS`) - Affects dense matrix operations in both Julia and MUMPS

### MUMPS Solver Threading

HPCSparseArrays uses the MUMPS (MUltifrontal Massively Parallel Solver) library for sparse direct solves via `lu()` and `ldlt()`. MUMPS has two independent threading mechanisms that can be tuned for performance.

**OpenMP threads (`OMP_NUM_THREADS`)**
- Controls MUMPS's algorithm-level parallelism
- The multifrontal method builds an elimination tree of "frontal matrices"
- OpenMP threads process independent subtrees in parallel
- This is coarse-grained: different threads work on different parts of the matrix

**BLAS threads (`OPENBLAS_NUM_THREADS`)**
- Controls parallelism inside dense matrix operations
- When MUMPS factors a frontal matrix, it calls BLAS routines (DGEMM, etc.)
- OpenBLAS can parallelize these dense operations
- This is fine-grained: threads cooperate on the same dense block

**Note on BLAS libraries**: Julia and MUMPS use separate OpenBLAS libraries (`libopenblas64_.dylib` for Julia's ILP64 interface, `libopenblas.dylib` for MUMPS's LP64 interface). Both libraries read `OPENBLAS_NUM_THREADS` at initialization, so this environment variable affects both.

### Recommended Configuration

For behavior that closely matches Julia's built-in sparse solver (UMFPACK):

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=<number_of_cores>
```

This configuration uses only BLAS-level threading, which is the same strategy Julia's built-in solver uses.

### Performance Comparison (Single-Rank)

The following table compares MUMPS (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=10`) against Julia's built-in sparse solver (also using the same settings) on a 2D Laplacian problem. **This is a single-rank comparison** to establish baseline overhead; multi-rank MPI parallelism provides additional speedup. Benchmarks were run on a 2025 M4 MacBook Pro with 10 CPU cores:

| n | Julia (ms) | MUMPS (ms) | Ratio |
|---|------------|------------|-------|
| 9 | 0.004 | 0.024 | 6.1x |
| 100 | 0.020 | 0.044 | 2.3x |
| 961 | 0.226 | 0.276 | 1.2x |
| 10,000 | 3.99 | 3.76 | 0.9x |
| 99,856 | 48.6 | 44.8 | 0.9x |
| 1,000,000 | 597 | 550 | 0.9x |

Key observations:
- At small problem sizes, MUMPS has initialization overhead (~0.02ms)
- At large problem sizes (n ≥ 10,000), MUMPS is **faster** than Julia's built-in solver
- Cached symbolic analysis and vectorized value copying minimize repeated factorization overhead

### Default Behavior

For optimal performance, set threading environment variables **before starting Julia**:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=10  # or your number of CPU cores
julia your_script.jl
```

Environment variables must be set before starting Julia because OpenBLAS creates its thread pool during library initialization. HPCSparseArrays attempts to set sensible defaults programmatically, but this may not always take effect if the thread pool is already initialized.

You can also add these to your shell profile (`.bashrc`, `.zshrc`, etc.) or Julia's `startup.jl`:

```julia
# In ~/.julia/config/startup.jl
ENV["OMP_NUM_THREADS"] = "1"
ENV["OPENBLAS_NUM_THREADS"] = string(Sys.CPU_THREADS)
```

### Advanced: Combined OMP and BLAS Threading

For some problems, combining OMP and BLAS threading can be faster:

```bash
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
julia your_script.jl
```

This MUMPS configuration (OMP=4, BLAS=4) achieved 14% faster performance than Julia's built-in solver on a 1M DOF 2D Laplacian in testing. However, the optimal configuration depends on your specific problem structure and hardware.

**Important caveat**: `OPENBLAS_NUM_THREADS` is a process-wide setting that affects both MUMPS and Julia's built-in sparse solver (UMFPACK). If you set `OPENBLAS_NUM_THREADS=4` to optimize MUMPS, Julia's built-in solver will also be limited to 4 BLAS threads.

## Row-wise Operations with map_rows

The `map_rows` function applies a function to corresponding rows across distributed arrays:

```julia
backend = BACKEND_CPU_MPI
A = HPCMatrix(randn(50, 10), backend)

# Compute row norms (backend is inferred from A)
norms = map_rows(row -> norm(row), A)  # Returns HPCVector

# Compute row sums and products
stats = map_rows(row -> [sum(row), prod(row)]', A)  # Returns HPCMatrix

# Combine multiple inputs
v = HPCVector(randn(50), backend)
weighted = map_rows((row, w) -> sum(row) * w[1], A, v)
```

### Result Types

| `f` returns | Result type |
|-------------|-------------|
| Scalar | `HPCVector` |
| Column vector | `HPCVector` (concatenated) |
| Row vector (`v'`) | `HPCMatrix` |
| Matrix | `HPCMatrix` |

## Type Conversions

### Gathering to Native Types

Convert distributed types back to native Julia arrays (gathers data to all ranks):

```julia
backend = BACKEND_CPU_MPI

v_mpi = HPCVector(randn(100), backend)
v_native = Vector(v_mpi)  # Full vector on all ranks

A_mpi = HPCMatrix(randn(50, 30), backend)
A_native = Matrix(A_mpi)  # Full matrix on all ranks

S_mpi = HPCSparseMatrix(sprandn(100, 100, 0.1), backend)
S_native = SparseMatrixCSC(S_mpi)  # Full sparse matrix
```

## IO and Output

### Printing from Rank 0

Use `io0()` to print from rank 0 only:

```julia
println(io0(), "This prints once from rank 0!")

# Custom rank selection
println(io0(r=Set([0, 1])), "Hello from ranks 0 and 1!")
```

### MPI Rank Information

```julia
using MPI

rank = MPI.Comm_rank(MPI.COMM_WORLD)   # Current rank (0 to nranks-1)
nranks = MPI.Comm_size(MPI.COMM_WORLD) # Total number of ranks
```

## Repartitioning

Redistribute data to match a different partition:

```julia
backend = BACKEND_CPU_MPI
v = HPCVector(randn(100), backend)

# Get current partition
old_partition = v.partition

# Create new partition
new_partition = uniform_partition(100, MPI.Comm_size(MPI.COMM_WORLD))

# Repartition
v_new = repartition(v, new_partition)
```

## GPU Support

HPCSparseArrays supports GPU acceleration via Metal.jl (macOS) or CUDA.jl (Linux/Windows). GPU support is optional and loaded as package extensions.

### Setup

Load the GPU package **before** MPI for proper detection:

```julia
# For Metal (macOS)
using Metal
using MPI
MPI.Init()
using HPCSparseArrays

# For CUDA (Linux/Windows)
using CUDA, NCCL, CUDSS_jll
using MPI
MPI.Init()
using HPCSparseArrays
```

### Creating GPU Backends

Use factory functions to create GPU backends:

```julia
comm = MPI.COMM_WORLD

# Metal backend (macOS)
metal_backend = backend_metal_mpi(comm)

# CUDA backend (Linux/Windows)
cuda_backend = backend_cuda_mpi(comm)
```

### Converting Between Backends

Use `to_backend(obj, backend)` to convert between CPU and GPU:

```julia
comm = MPI.COMM_WORLD
cpu_backend = BACKEND_CPU_MPI
metal_backend = backend_metal_mpi(comm)

# Create CPU vector
x_cpu = HPCVector(rand(Float32, 1000), cpu_backend)

# Convert to GPU (Metal)
x_gpu = to_backend(x_cpu, metal_backend)  # Returns HPCVector with MtlVector storage

# GPU operations work transparently
y_gpu = x_gpu + x_gpu  # Native GPU addition
z_gpu = 2.0f0 * x_gpu  # Native GPU scalar multiply

# Convert back to CPU
y_cpu = to_backend(y_gpu, cpu_backend)
```

For CUDA:

```julia
comm = MPI.COMM_WORLD
cpu_backend = BACKEND_CPU_MPI
cuda_backend = backend_cuda_mpi(comm)

# Create CPU vector
x_cpu = HPCVector(rand(Float64, 1000), cpu_backend)

# Convert to GPU (CUDA)
x_gpu = to_backend(x_cpu, cuda_backend)   # Returns HPCVector with CuVector storage

# GPU operations work transparently
y_gpu = x_gpu + x_gpu  # Native GPU addition
z_gpu = 2.0 * x_gpu    # Native GPU scalar multiply
```

### How It Works

The backend's device type determines where data lives:

| Backend Device | Storage | Operations |
|----------------|---------|------------|
| `DeviceCPU()` | `Vector`/`Matrix` | Native CPU |
| `DeviceMetal()` | `MtlVector`/`MtlMatrix` | Native GPU for vector ops |
| `DeviceCUDA()` | `CuVector`/`CuMatrix` | Native GPU for vector ops |

### MPI Communication

MPI always uses CPU buffers (no GPU-aware MPI). GPU data is automatically staged through CPU:

1. GPU vector data copied to CPU staging buffer
2. MPI communication on CPU buffers
3. Results copied back to GPU

This is handled transparently - you just use the same operations.

### Sparse Matrix Operations

Sparse matrices (`HPCSparseMatrix`) remain on CPU. When multiplying with GPU vectors:

```julia
cuda_backend = backend_cuda_mpi(MPI.COMM_WORLD)
A = HPCSparseMatrix(sprand(100, 100, 0.1), cuda_backend)
x_gpu = HPCVector(rand(100), cuda_backend)

# Sparse multiply: x gathered via CPU, multiply on CPU, result copied to GPU
y_gpu = A * x_gpu  # Returns HPCVector with CuVector storage
```

### Supported GPU Operations

| Operation | Metal | CUDA |
|-----------|-------|------|
| `v + w`, `v - w` | Native | Native |
| `α * v` (scalar) | Native | Native |
| `A * x` (sparse) | CPU staging | CPU staging |
| `A * x` (dense) | CPU staging | CPU staging |
| `transpose(A) * x` | CPU staging | CPU staging |
| Broadcasting (`abs.(v)`) | Native | Native |
| `cudss_lu(A)`, `cudss_ldlt(A)` | N/A | Multi-GPU native |

### Element Types

- **Metal**: Requires `Float32` (no `Float64` support)
- **CUDA**: Supports both `Float32` and `Float64`

## cuDSS Multi-GPU Solver (CUDA only)

For multi-GPU distributed sparse direct solves, HPCSparseArrays provides `CuDSSFactorizationMPI` using NVIDIA's cuDSS library with NCCL inter-GPU communication.

### Basic Usage

```julia
using CUDA, NCCL, CUDSS_jll
using MPI
MPI.Init()
using HPCSparseArrays

# Each MPI rank should use a different GPU
CUDA.device!(MPI.Comm_rank(MPI.COMM_WORLD) % length(CUDA.devices()))

# Create CUDA backend
cuda_backend = backend_cuda_mpi(MPI.COMM_WORLD)

# Create distributed sparse matrix (symmetric positive definite)
A = HPCSparseMatrix(make_spd_matrix(1000), cuda_backend)
b = HPCVector(rand(1000), cuda_backend)

# Multi-GPU LDLT factorization
F = cudss_ldlt(A)
x = F \ b
finalize!(F)  # Required: clean up cuDSS resources
```

### Available Factorizations

```julia
# For symmetric positive definite matrices
F = cudss_ldlt(A)

# For general (non-symmetric) matrices
F = cudss_lu(A)
```

### Important Notes

- **GPU assignment**: Each MPI rank should use a different GPU. Use `CUDA.device!()` to assign.
- **NCCL bootstrap**: The NCCL communicator is created automatically from MPI - no manual setup needed.
- **Resource cleanup**: Always call `finalize!(F)` when done. This prevents MPI desynchronization during garbage collection.
- **Requirements**: cuDSS 0.4+ with MGMN (Multi-GPU Multi-Node) support.
- **NCCL P2P**: On some clusters, you may need to set `NCCL_P2P_DISABLE=1` to avoid hangs with cross-NUMA GPU topologies.

### Known cuDSS Bug (as of January 2026)

cuDSS MGMN mode crashes with `status=5` on certain sparse matrix patterns, notably narrow-bandwidth matrices like tridiagonals. This bug has been reported to NVIDIA.

**Symptoms**: The analysis phase fails with status code 5 for matrices that have few nonzeros per row when distributed across multiple GPUs.

**Workaround**: Adding explicit numerical zeros to widen the bandwidth allows cuDSS to succeed. For example, if your matrix is tridiagonal, storing it with additional zero entries in each row (making it appear as a wider-band matrix) works around the bug. The 2D Laplacian (5-point stencil) works correctly.

**Minimal reproducer**: See `bug/cudss_mgmn_tridiag_bug.cu` in the repository for a C/CUDA test case demonstrating the issue.

## Cache Management

HPCSparseArrays caches communication plans for efficiency. Clear caches when needed:

```julia
clear_plan_cache!()  # Clears all plan caches including MUMPS analysis cache
```

## MPI Collective Operations

!!! warning "All Operations Are Collective"
    Most HPCSparseArrays functions are MPI collective operations. All ranks must:
    - Call the function together
    - Use the same parameters
    - Avoid conditional execution based on rank

**Correct:**
```julia
# All ranks execute this together
x = A \ b
```

**Incorrect (causes deadlock):**
```julia
if rank == 0
    x = A \ b  # Only rank 0 calls - DEADLOCK!
end
```

## Next Steps

- See [Examples](@ref) for detailed code examples
- See the [API Reference](@ref) for detailed function documentation
