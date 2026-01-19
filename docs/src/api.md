# API Reference

This page provides detailed documentation for all exported types and functions in HPCSparseArrays.jl.

!!! note "MPI Collective Operations"
    Unless otherwise noted, all functions are MPI collective operations. Every MPI rank must call these functions together.

## Distributed Types

### HPCVector

```@docs
HPCVector
```

### HPCMatrix

```@docs
HPCMatrix
```

### HPCSparseMatrix

```@docs
HPCSparseMatrix
```

### SparseMatrixCSR

```@docs
SparseMatrixCSR
```

## Local Constructors

These constructors create distributed types from local data without global communication.

```@docs
HPCVector_local
HPCMatrix_local
HPCSparseMatrix_local
```

## Row-wise Operations

```@docs
map_rows
map_rows_gpu
```

## Linear System Solvers

```@docs
solve
solve!
finalize!
```

## Partition Utilities

```@docs
uniform_partition
repartition
```

## Cache Management

```@docs
clear_plan_cache!
clear_mumps_analysis_cache!
```

## IO Utilities

```@docs
io0
```

## Backend Types

```@docs
HPCBackend
DeviceCPU
DeviceMetal
DeviceCUDA
CommSerial
CommMPI
SolverMUMPS
BACKEND_CPU_MPI
BACKEND_CPU_SERIAL
backend_metal_mpi
backend_cuda_mpi
to_backend
```

## Type Mappings

### Native to Distributed Conversions

| Native Type | Distributed Type | Description |
|-------------|------------------|-------------|
| `Vector{T}` | `HPCVector{T,B}` | Distributed vector |
| `Matrix{T}` | `HPCMatrix{T,B}` | Distributed dense matrix |
| `SparseMatrixCSC{T,Ti}` | `HPCSparseMatrix{T,Ti,B}` | Distributed sparse matrix |

The `B<:HPCBackend` type parameter specifies the backend configuration (device, communication, solver). Use pre-constructed backends like `BACKEND_CPU_MPI` or factory functions like `backend_cuda_mpi(comm)`.

### Distributed to Native Conversions

| Distributed Type | Native Type | Function |
|------------------|-------------|----------|
| `HPCVector{T,B}` | `Vector{T}` | `Vector(v)` |
| `HPCMatrix{T,B}` | `Matrix{T}` | `Matrix(A)` |
| `HPCSparseMatrix{T,Ti,B}` | `SparseMatrixCSC{T,Ti}` | `SparseMatrixCSC(A)` |

## Supported Operations

### HPCVector Operations

- Arithmetic: `+`, `-`, `*` (scalar)
- Linear algebra: `norm`, `dot`, `conj`
- Indexing: `v[i]` (global index)
- Conversion: `Vector(v)`

### HPCMatrix Operations

- Arithmetic: `*` (scalar), matrix-vector product
- Transpose: `transpose(A)`
- Indexing: `A[i, j]` (global indices)
- Conversion: `Matrix(A)`

### HPCSparseMatrix Operations

- Arithmetic: `+`, `-`, `*` (scalar, matrix-vector, matrix-matrix)
- Transpose: `transpose(A)`
- Linear solve: `A \ b`, `Symmetric(A) \ b`
- Utilities: `nnz`, `norm`, `issymmetric`
- Conversion: `SparseMatrixCSC(A)`

## Factorization Types

HPCSparseArrays uses MUMPS for sparse direct solves:

- `lu(A)`: LU factorization (general matrices)
- `ldlt(A)`: LDLT factorization (symmetric matrices, faster)

Both return factorization objects that support:
- `F \ b`: Solve with factorization
- `finalize!(F)`: Release MUMPS resources
