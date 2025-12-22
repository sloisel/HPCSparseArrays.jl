# LinearAlgebraMPI.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sloisel.github.io/LinearAlgebraMPI.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sloisel.github.io/LinearAlgebraMPI.jl/dev/)
[![Build Status](https://github.com/sloisel/LinearAlgebraMPI.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sloisel/LinearAlgebraMPI.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/sloisel/LinearAlgebraMPI.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/sloisel/LinearAlgebraMPI.jl)

**Author:** S. Loisel

Distributed sparse matrix and vector operations using MPI for Julia. This package provides efficient parallel linear algebra operations across multiple MPI ranks.

## Features

- **Distributed sparse matrices** (`SparseMatrixMPI{T}`) with row-partitioning across MPI ranks
- **Distributed dense vectors** (`VectorMPI{T}`) with flexible partitioning
- **Matrix-matrix multiplication** (`A * B`) with memoized communication plans
- **Matrix-vector multiplication** (`A * x`, `mul!(y, A, x)`)
- **Sparse direct solvers**: LU and LDLT factorization using MUMPS
- **Lazy transpose** with optimized multiplication rules
- **Matrix addition/subtraction** (`A + B`, `A - B`)
- **Vector operations**: norms, reductions, arithmetic with automatic partition alignment
- Support for both `Float64` and `ComplexF64` element types
- **GPU acceleration** via Metal.jl (macOS) with automatic CPU staging for MPI

## Installation

```julia
using Pkg
Pkg.add("LinearAlgebraMPI")
```

## Quick Start

```julia
using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays

# Create a sparse matrix (must be identical on all ranks)
A_global = sprand(1000, 1000, 0.01)
A = SparseMatrixMPI{Float64}(A_global)

# Create a vector
x_global = rand(1000)
x = VectorMPI(x_global)

# Matrix-vector multiplication
y = A * x

# Matrix-matrix multiplication
B_global = sprand(1000, 500, 0.01)
B = SparseMatrixMPI{Float64}(B_global)
C = A * B

# Transpose operations
At = transpose(A)
D = At * B  # Materializes transpose as needed

# Solve linear systems
using LinearAlgebra
A_sym = A + transpose(A) + 10I  # Make symmetric positive definite
A_sym_dist = SparseMatrixMPI{Float64}(A_sym)
F = ldlt(A_sym_dist)  # LDLT factorization
x_sol = solve(F, y)   # Solve A_sym * x_sol = y
```

## GPU Support (Metal)

LinearAlgebraMPI supports GPU acceleration on macOS via Metal.jl. GPU support is optional - Metal.jl is loaded as a weak dependency.

### Converting between CPU and GPU

```julia
using Metal  # Load Metal BEFORE MPI for GPU detection
using MPI
MPI.Init()
using LinearAlgebraMPI

# Create CPU vectors/matrices as usual
x_cpu = VectorMPI(Float32.(rand(1000)))
A = SparseMatrixMPI{Float32}(sprand(Float32, 1000, 1000, 0.01))

# Convert to GPU
x_gpu = mtl(x_cpu)  # Returns VectorMPI with MtlVector storage

# GPU operations work transparently
y_gpu = A * x_gpu   # Sparse A*x with GPU vector (CPU staging for computation)
z_gpu = x_gpu + x_gpu  # Vector addition on GPU

# Convert back to CPU
y_cpu = cpu(y_gpu)
```

### Creating GPU vectors directly

```julia
using Metal

# Create GPU vector from local data
local_data = MtlVector(Float32.(rand(100)))
v_gpu = VectorMPI_local(local_data)
```

### How it works

- **Vectors**: `VectorMPI{T,AV}` where `AV` is `Vector{T}` (CPU) or `MtlVector{T}` (GPU)
- **Matrices**: Currently remain on CPU; GPU vectors are staged through CPU for matrix operations
- **MPI communication**: Always uses CPU buffers (no Metal-aware MPI exists)
- **Element type**: Metal requires `Float32` (no `Float64` support)

### Supported GPU operations

| Operation | GPU Support |
|-----------|-------------|
| `v + w`, `v - w` | Native GPU |
| `α * v` (scalar) | Native GPU |
| `A * x` (sparse) | CPU staging |
| `A * x` (dense) | CPU staging |
| `transpose(A) * x` | CPU staging |
| Broadcasting (`abs.(v)`) | Native GPU |

## Running with MPI

```bash
mpiexec -n 4 julia your_script.jl
```

## Documentation

For detailed documentation, see the [stable docs](https://sloisel.github.io/LinearAlgebraMPI.jl/stable/) or [dev docs](https://sloisel.github.io/LinearAlgebraMPI.jl/dev/).
