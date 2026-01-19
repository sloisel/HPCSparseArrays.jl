```@meta
CurrentModule = HPCSparseArrays
```

```@eval
using Markdown
using Pkg
using HPCSparseArrays
v = string(pkgversion(HPCSparseArrays))
md"# HPCSparseArrays.jl $v"
```

**Pure Julia distributed linear algebra with MPI.**

## Overview

HPCSparseArrays.jl provides distributed matrix and vector types for parallel computing with MPI. It offers a pure Julia implementation of distributed linear algebra, with MUMPS for sparse direct solves.

## Key Features

- **Distributed Types**: `HPCVector`, `HPCMatrix`, and `HPCSparseMatrix` for row-partitioned distributed storage
- **MUMPS Solver**: Direct solves using MUMPS for sparse linear systems
- **Row-wise Operations**: `map_rows` for efficient distributed row operations
- **Seamless Integration**: Works with standard Julia linear algebra operations
- **Plan Caching**: Efficient repeated operations through memoized communication plans

## Quick Example

```julia
using MPI
MPI.Init()
using HPCSparseArrays
using SparseArrays

# Use the default MPI backend
backend = BACKEND_CPU_MPI

# Create distributed sparse matrix
A = HPCSparseMatrix(sprandn(100, 100, 0.1) + 10I, backend)
b = HPCVector(randn(100), backend)

# Solve linear system
x = A \ b

# Row-wise operations
norms = map_rows(row -> norm(row), HPCMatrix(randn(50, 10), backend))

println(io0(), "Solution computed!")
```

**Run with MPI:**

```bash
mpiexec -n 4 julia --project example.jl
```

## Documentation Contents

```@contents
Pages = ["installation.md", "guide.md", "examples.md", "api.md"]
Depth = 2
```

## Related Packages

- **[MultiGridBarrierMPI.jl](https://github.com/sloisel/MultiGridBarrierMPI.jl)**: Multigrid barrier methods using HPCSparseArrays
- **[MultiGridBarrier.jl](https://github.com/sloisel/MultiGridBarrier.jl)**: Core multigrid barrier method implementation
- **MPI.jl**: Julia MPI bindings

## Requirements

- Julia 1.11 or later
- MPI installation (OpenMPI, MPICH, or Intel MPI)
- MUMPS for sparse direct solves (bundled via MUMPS.jl)

## License

This package is licensed under the MIT License.
