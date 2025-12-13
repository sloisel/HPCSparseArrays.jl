# Getting Started

This guide will walk you through the basics of using LinearAlgebraMPI.jl for distributed sparse matrix computations.

## Prerequisites

Before using LinearAlgebraMPI.jl, ensure you have:

1. A working MPI installation (OpenMPI, MPICH, or Intel MPI)
2. MPI.jl configured to use your MPI installation

You can verify your MPI setup with:

```julia
using MPI
MPI.Init()
println("Rank $(MPI.Comm_rank(MPI.COMM_WORLD)) of $(MPI.Comm_size(MPI.COMM_WORLD))")
MPI.Finalize()
```

Run with:
```bash
mpiexec -n 4 julia --project=. your_script.jl
```

## Creating Distributed Matrices

### From a Global Sparse Matrix

The most common way to create a distributed matrix is from an existing `SparseMatrixCSC`:

```julia
using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays

# Create a sparse matrix - MUST be identical on all ranks
n = 100
A = spdiagm(0 => 2.0*ones(n), 1 => -ones(n-1), -1 => -ones(n-1))

# Distribute across MPI ranks
Adist = SparseMatrixMPI{Float64}(A)
```

**Important**: All MPI ranks must have the same matrix **size** when constructing distributed types. However, each rank only extracts its own local rows, so the actual **data** only needs to be correct for each rank's portion.

### Understanding Row Partitioning

The matrix is partitioned roughly equally by rows. For example, with 4 ranks and a 100x100 matrix:

- Rank 0: rows 1-25
- Rank 1: rows 26-50
- Rank 2: rows 51-75
- Rank 3: rows 76-100

### Efficient Local-Only Construction

For large matrices, you can avoid replicating data across all ranks by only populating each rank's local portion:

```julia
using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays

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

# Create a sparse matrix with correct size, but only populate local rows
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

This pattern is useful when:
- The global matrix is too large to fit in memory on each rank
- You're generating matrix entries programmatically
- You want to minimize memory usage during construction

## Basic Operations

### Matrix Multiplication

```julia
# Both matrices must be distributed
Adist = SparseMatrixMPI{Float64}(A)
Bdist = SparseMatrixMPI{Float64}(B)

# Multiply
Cdist = Adist * Bdist
```

The multiplication automatically handles the necessary communication between ranks.

### Addition and Subtraction

```julia
Cdist = Adist + Bdist
Ddist = Adist - Bdist
```

If A and B have different row partitions, B's rows are redistributed to match A's partition.

### Scalar Multiplication

```julia
Cdist = 2.5 * Adist
Cdist = Adist * 2.5  # Equivalent
```

### Transpose

```julia
# Transpose is lazy (no communication until needed)
At = transpose(Adist)

# Use in multiplication - automatically materializes when needed
Cdist = At * Bdist
```

### Adjoint (Conjugate Transpose)

For complex matrices:

```julia
Adist = SparseMatrixMPI{ComplexF64}(A)
Aadj = Adist'  # Conjugate transpose (lazy)
```

### Computing Norms

```julia
# Frobenius norm (default)
f_norm = norm(Adist)

# 1-norm (sum of absolute values)
one_norm = norm(Adist, 1)

# Infinity norm (maximum absolute value)
inf_norm = norm(Adist, Inf)

# General p-norm
p_norm = norm(Adist, 3)

# Operator norms
col_norm = opnorm(Adist, 1)   # Max column sum
row_norm = opnorm(Adist, Inf) # Max row sum
```

## Running MPI Programs

### Command Line

```bash
mpiexec -n 4 julia --project=. my_program.jl
```

### Program Structure

A typical LinearAlgebraMPI.jl program follows this pattern:

```julia
using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra

# Create matrices (identical on all ranks)
A = create_my_matrix()  # Your matrix creation function
B = create_my_matrix()

# Distribute
Adist = SparseMatrixMPI{Float64}(A)
Bdist = SparseMatrixMPI{Float64}(B)

# Compute
Cdist = Adist * Bdist

# Get results (e.g., norm is computed globally)
result_norm = norm(Cdist)

println(io0(), "Result norm: $result_norm")

MPI.Finalize()
```

## Performance Tips

### Reuse Communication Plans

For repeated operations with the same sparsity pattern, LinearAlgebraMPI.jl automatically caches communication plans:

```julia
# First multiplication creates and caches the plan
C1 = Adist * Bdist

# Subsequent multiplications with same A, B reuse the cached plan
C2 = Adist * Bdist  # Uses cached plan - much faster
```

### Clear Cache When Done

If you're done with a set of matrices and want to free memory:

```julia
clear_plan_cache!()
```

### Use Deterministic Test Data

For testing with the simple "replicate everywhere" pattern, avoid random matrices since they'll differ across ranks:

```julia
# Bad - different random values on each rank
A = sprand(100, 100, 0.01)

# Good - deterministic formula, same on all ranks
I = [1:100; 1:99; 2:100]
J = [1:100; 2:100; 1:99]
V = [2.0*ones(100); -0.5*ones(99); -0.5*ones(99)]
A = sparse(I, J, V, 100, 100)
```

Alternatively, use the [local-only construction pattern](#Efficient-Local-Only-Construction) where each rank generates only its own rows.

## Next Steps

- See [Examples](@ref) for more detailed usage examples
- Read the [API Reference](@ref) for complete function documentation
