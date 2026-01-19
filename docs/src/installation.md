# Installation

## Prerequisites

### MPI

HPCSparseArrays.jl requires an MPI implementation. When you install the package, Julia automatically provides `MPI.jl` with `MPI_jll` (bundled MPI implementation).

For HPC environments, you may want to configure MPI.jl to use your system's MPI installation. See the [MPI.jl documentation](https://juliaparallel.org/MPI.jl/stable/configuration/) for details.

### MUMPS

The package uses MUMPS for sparse direct solves. MUMPS is typically available through your system's package manager or HPC module system.

## Package Installation

### From GitHub

```julia
using Pkg
Pkg.add(url="https://github.com/sloisel/HPCSparseArrays.jl")
```

### Development Installation

```bash
git clone https://github.com/sloisel/HPCSparseArrays.jl
cd HPCSparseArrays.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Verification

Test your installation:

```bash
cd HPCSparseArrays.jl
julia --project -e 'using Pkg; Pkg.test()'
```

The test harness automatically spawns MPI processes for each test file.

## Initialization Pattern

!!! tip "Initialization Pattern"
    Initialize MPI before using HPCSparseArrays:

```julia
using MPI
MPI.Init()
using HPCSparseArrays
# Now you can use the package with BACKEND_CPU_MPI
```

## Running MPI Programs

Create a script file (e.g., `my_program.jl`):

```julia
using MPI
MPI.Init()
using HPCSparseArrays
using SparseArrays

# Use the default MPI backend
backend = BACKEND_CPU_MPI

# Create distributed matrix
A = HPCSparseMatrix(sprandn(100, 100, 0.1) + 10I, backend)
b = HPCVector(randn(100), backend)

# Solve
x = A \ b

println(io0(), "Solution computed!")
```

Run with MPI:

```bash
mpiexec -n 4 julia --project my_program.jl
```

## Troubleshooting

### MPI Issues

If you see MPI-related errors, try rebuilding MPI.jl:

```julia
using Pkg; Pkg.build("MPI")
```

### MUMPS Issues

If MUMPS fails to load, ensure it's properly installed on your system.

## Next Steps

Once installed, proceed to the [User Guide](@ref) to learn how to use the package.
