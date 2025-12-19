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

- See the [API Reference](@ref) for detailed function documentation
