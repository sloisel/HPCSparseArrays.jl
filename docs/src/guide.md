# User Guide

This guide covers the essential workflows for using LinearAlgebraMPI.jl.

## Core Types

LinearAlgebraMPI provides three distributed types:

| Type | Description | Storage |
|------|-------------|---------|
| `VectorMPI{T}` | Distributed vector | Row-partitioned |
| `MatrixMPI{T}` | Distributed dense matrix | Row-partitioned |
| `SparseMatrixMPI{T,Ti}` | Distributed sparse matrix | Row-partitioned CSR |

All types are row-partitioned across MPI ranks, meaning each rank owns a contiguous range of rows.

### Internal Storage: CSR Format

Internally, `SparseMatrixMPI` stores local rows in CSR (Compressed Sparse Row) format using the `SparseMatrixCSR` type. This enables efficient row-wise iteration for a row-partitioned distributed matrix.

In Julia, `SparseMatrixCSR{T,Ti}` is a type alias for `Transpose{T, SparseMatrixCSC{T,Ti}}`. You don't need to worry about this for normal usage - it's handled automatically.

## Creating Distributed Types

### From Native Julia Types

```julia
using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays

# Create from native types (data is distributed automatically)
v = VectorMPI(randn(100))
A = MatrixMPI(randn(50, 30))
S = SparseMatrixMPI{Float64}(sprandn(100, 100, 0.1))
```

### Local Constructors

For performance-critical code, use local constructors that avoid global communication:

```julia
# Create from local data (each rank provides its own rows)
v_local = VectorMPI_local(my_local_vector)
A_local = MatrixMPI_local(my_local_matrix)
S_local = SparseMatrixMPI_local(my_local_sparse)
```

### Efficient Local-Only Construction

For large matrices, avoid replicating data across all ranks by only populating each rank's local portion:

```julia
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
Adist = SparseMatrixMPI{Float64}(A)
```

## Basic Operations

### Vector Operations

```julia
v = VectorMPI(randn(100))
w = VectorMPI(randn(100))

# Arithmetic
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
A = MatrixMPI(randn(50, 100))
v = VectorMPI(randn(100))

# Matrix-vector multiply
y = A * v
```

### Sparse Operations

```julia
using SparseArrays

A = SparseMatrixMPI{Float64}(sprandn(100, 100, 0.1))
v = VectorMPI(randn(100))

# Matrix-vector multiply
y = A * v

# Matrix-matrix multiply
B = SparseMatrixMPI{Float64}(sprandn(100, 100, 0.1))
C = A * B
```

## Solving Linear Systems

### Direct Solve with Backslash

```julia
using SparseArrays

# Create a well-conditioned sparse matrix
A = SparseMatrixMPI{Float64}(sprandn(100, 100, 0.1) + 10I)
b = VectorMPI(randn(100))

# Solve A * x = b
x = A \ b
```

### Symmetric Systems (Faster)

For symmetric matrices, wrap with `Symmetric` to use faster LDLT factorization:

```julia
using LinearAlgebra

# Create symmetric positive definite matrix
A_base = SparseMatrixMPI{Float64}(sprandn(100, 100, 0.1))
A_spd = A_base + SparseMatrixMPI(transpose(A_base)) + 
        SparseMatrixMPI{Float64}(sparse(10.0I, 100, 100))

b = VectorMPI(randn(100))

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

### MUMPS Solver Threading

LinearAlgebraMPI uses MUMPS for sparse direct solves. MUMPS has two threading mechanisms:

- **OpenMP threads (`OMP_NUM_THREADS`)**: Controls algorithm-level parallelism in the multifrontal method
- **BLAS threads (`OPENBLAS_NUM_THREADS`)**: Controls parallelism inside dense matrix operations

For optimal performance matching Julia's built-in solver:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=<number_of_cores>
mpiexec -n 4 julia --project my_program.jl
```

You can also set these in Julia's `startup.jl`:

```julia
# In ~/.julia/config/startup.jl
ENV["OMP_NUM_THREADS"] = "1"
ENV["OPENBLAS_NUM_THREADS"] = string(Sys.CPU_THREADS)
```

!!! tip "Performance"
    At large problem sizes (n ≥ 100,000), MUMPS matches Julia's built-in sparse solver speed.

## Row-wise Operations with map_rows

The `map_rows` function applies a function to corresponding rows across distributed arrays:

```julia
A = MatrixMPI(randn(50, 10))

# Compute row norms
norms = map_rows(row -> norm(row), A)  # Returns VectorMPI

# Compute row sums and products
stats = map_rows(row -> [sum(row), prod(row)]', A)  # Returns MatrixMPI

# Combine multiple inputs
v = VectorMPI(randn(50))
weighted = map_rows((row, w) -> sum(row) * w[1], A, v)
```

### Result Types

| `f` returns | Result type |
|-------------|-------------|
| Scalar | `VectorMPI` |
| Column vector | `VectorMPI` (concatenated) |
| Row vector (`v'`) | `MatrixMPI` |
| Matrix | `MatrixMPI` |

## Type Conversions

### Gathering to Native Types

Convert distributed types back to native Julia arrays (gathers data to all ranks):

```julia
v_mpi = VectorMPI(randn(100))
v_native = Vector(v_mpi)  # Full vector on all ranks

A_mpi = MatrixMPI(randn(50, 30))
A_native = Matrix(A_mpi)  # Full matrix on all ranks

S_mpi = SparseMatrixMPI{Float64}(sprandn(100, 100, 0.1))
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
v = VectorMPI(randn(100))

# Get current partition
old_partition = v.partition

# Create new partition
new_partition = uniform_partition(100, MPI.Comm_size(MPI.COMM_WORLD))

# Repartition
v_new = repartition(v, new_partition)
```

## Cache Management

LinearAlgebraMPI caches communication plans for efficiency. Clear caches when needed:

```julia
clear_plan_cache!()  # Clears all plan caches including MUMPS analysis cache
```

## MPI Collective Operations

!!! warning "All Operations Are Collective"
    Most LinearAlgebraMPI functions are MPI collective operations. All ranks must:
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
