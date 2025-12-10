# API Reference

This page documents the public API of LinearAlgebraMPI.jl.

## Types

### SparseMatrixMPI

```@docs
SparseMatrixMPI
```

The main distributed sparse matrix type. Rows are partitioned across MPI ranks.

**Fields:**
- `structural_hash::NTuple{32,UInt8}`: 256-bit Blake3 hash of the structural pattern
- `row_partition::Vector{Int}`: Row partition boundaries (length = nranks + 1)
- `col_partition::Vector{Int}`: Column partition boundaries (length = nranks + 1)
- `col_indices::Vector{Int}`: Column indices that appear in the local part
- `AT::SparseMatrixCSC{T,Int}`: Transpose of local rows

### MatrixPlan

```@docs
MatrixPlan
```

A communication plan for gathering rows from an `SparseMatrixMPI` for matrix multiplication.

### TransposePlan

```@docs
TransposePlan
```

A communication plan for computing the transpose of an `SparseMatrixMPI`.

### VectorMPI

```@docs
VectorMPI
```

A distributed vector type. Elements are partitioned across MPI ranks.

**Fields:**
- `structural_hash::NTuple{32,UInt8}`: 256-bit Blake3 hash of the partition pattern
- `partition::Vector{Int}`: Element partition boundaries (length = nranks + 1)
- `v::Vector{T}`: Local elements owned by this rank

### VectorPlan

```@docs
VectorPlan
```

A communication plan for gathering vector elements for matrix-vector multiplication or vector alignment.

## Constructors

### SparseMatrixMPI Constructor

Create a distributed sparse matrix from a global sparse matrix.

**Signature:**
```julia
SparseMatrixMPI{T}(A::SparseMatrixCSC{T,Int}) where T
```

**Arguments:**
- `A::SparseMatrixCSC{T,Int}`: The global sparse matrix (must be identical on all ranks)

**Returns:**
- `SparseMatrixMPI{T}`: The distributed matrix

**Example:**
```julia
using SparseArrays
A = sprand(100, 100, 0.01)
Adist = SparseMatrixMPI{Float64}(A)
```

### MatrixPlan Constructors

Create communication plans for matrix operations.

**Signatures:**
```julia
MatrixPlan(row_indices::Vector{Int}, B::SparseMatrixMPI{T}) where T
MatrixPlan(A::SparseMatrixMPI{T}, B::SparseMatrixMPI{T}) where T
```

The second form creates a memoized plan for `A * B` based on structural hashes.

### TransposePlan Constructor

Create a communication plan for transpose.

**Signature:**
```julia
TransposePlan(A::SparseMatrixMPI{T}) where T
```

### VectorMPI Constructor

Create a distributed vector from a global vector.

**Signature:**
```julia
VectorMPI(v_global::Vector{T}, comm::MPI.Comm=MPI.COMM_WORLD) where T
```

**Arguments:**
- `v_global::Vector{T}`: The global vector (must be identical on all ranks)
- `comm::MPI.Comm`: MPI communicator (optional, defaults to COMM_WORLD)

**Returns:**
- `VectorMPI{T}`: The distributed vector

**Example:**
```julia
v = collect(1.0:100.0)
vdist = VectorMPI(v)
```

### VectorPlan Constructors

Create communication plans for vector operations.

**Signatures:**
```julia
VectorPlan(A::SparseMatrixMPI{T}, x::VectorMPI{T}) where T
VectorPlan(target_partition::Vector{Int}, source::VectorMPI{T}) where T
```

The first form creates a plan for gathering `x[A.col_indices]` for matrix-vector multiplication.
The second form creates a plan for aligning a source vector to a target partition.

## Executing Plans

```@docs
execute_plan!
```

Execute a pre-computed communication plan.

## Matrix Operations

### Multiplication

```julia
*(A::SparseMatrixMPI{T}, B::SparseMatrixMPI{T}) where T
```

Multiply two distributed sparse matrices. Communication plans are automatically cached.

**Example:**
```julia
C = A * B
```

### Addition

```julia
+(A::SparseMatrixMPI{T}, B::SparseMatrixMPI{T}) where T
```

Add two distributed sparse matrices. The result has A's row partition.

**Example:**
```julia
C = A + B
```

### Subtraction

```julia
-(A::SparseMatrixMPI{T}, B::SparseMatrixMPI{T}) where T
```

Subtract two distributed sparse matrices. The result has A's row partition.

**Example:**
```julia
C = A - B
```

### Scalar Multiplication

```julia
*(a::Number, A::SparseMatrixMPI{T}) where T
*(A::SparseMatrixMPI{T}, a::Number) where T
```

Multiply a distributed matrix by a scalar.

**Example:**
```julia
B = 2.0 * A
B = A * 2.0  # Equivalent
```

### Transpose

Return a lazy transpose wrapper. The transpose is not computed until needed.

**Signature:**
```julia
transpose(A::SparseMatrixMPI{T}) where T
```

**Example:**
```julia
At = transpose(A)  # Lazy, no communication
```

To materialize:
```julia
plan = TransposePlan(A)
At_materialized = execute_plan!(plan, A)
```

### Conjugate

Return a new `SparseMatrixMPI` with conjugated values.

**Signature:**
```julia
conj(A::SparseMatrixMPI{T}) where T
```

**Example:**
```julia
Aconj = conj(A)
```

### Adjoint

Return the conjugate transpose (lazy).

**Signature:**
```julia
adjoint(A::SparseMatrixMPI{T}) where T
```

**Example:**
```julia
Aadj = A'  # Equivalent to transpose(conj(A))
```

## Lazy Transpose Operations

The following operations work with lazy transposes:

```julia
# transpose(A) * transpose(B) = transpose(B * A)
transpose(A) * transpose(B)

# transpose(A) * B - materializes transpose(A) first
transpose(A) * B

# A * transpose(B) - materializes transpose(B) first
A * transpose(B)

# Scalar times lazy transpose
a * transpose(A)  # Returns transpose(a * A)
transpose(A) * a  # Returns transpose(a * A)
```

## Vector Operations

### Matrix-Vector Multiplication

```julia
*(A::SparseMatrixMPI{T}, x::VectorMPI{T}) where T
mul!(y::VectorMPI{T}, A::SparseMatrixMPI{T}, x::VectorMPI{T}) where T
```

Multiply a distributed sparse matrix by a distributed vector. The result vector has `A.row_partition` as its partition.

**Example:**
```julia
y = A * x
mul!(y, A, x)  # In-place version
```

### Vector-Matrix Multiplication

```julia
*(vt::Transpose{<:Any, VectorMPI{T}}, A::SparseMatrixMPI{T}) where T
```

Multiply a transposed vector by a matrix: `transpose(v) * A = transpose(transpose(A) * v)`.

**Example:**
```julia
wt = transpose(v) * A
```

### Vector Addition and Subtraction

```julia
+(u::VectorMPI{T}, v::VectorMPI{T}) where T
-(u::VectorMPI{T}, v::VectorMPI{T}) where T
-(v::VectorMPI{T}) where T
```

Add or subtract distributed vectors. If partitions differ, the second operand is automatically aligned to match the first operand's partition.

**Example:**
```julia
w = u + v  # Result has u's partition
w = u - v
w = -v
```

### Vector Scalar Multiplication

```julia
*(a::Number, v::VectorMPI{T}) where T
*(v::VectorMPI{T}, a::Number) where T
/(v::VectorMPI{T}, a::Number) where T
```

Multiply or divide a distributed vector by a scalar.

**Example:**
```julia
w = 2.0 * v
w = v * 2.0
w = v / 2.0
```

### Vector Transpose and Adjoint

```julia
transpose(v::VectorMPI{T}) where T
conj(v::VectorMPI{T}) where T
adjoint(v::VectorMPI{T}) where T
```

Return transpose (lazy wrapper), conjugate (new VectorMPI), or adjoint (transpose of conjugate).

**Example:**
```julia
vt = transpose(v)  # Lazy
vc = conj(v)       # New vector with conjugated values
va = v'            # Equivalent to transpose(conj(v))
```

### Vector Norms

```julia
norm(v::VectorMPI, p=2)
```

Compute the p-norm of a distributed vector.

**Arguments:**
- `p=2` (default): Euclidean norm
- `p=1`: Sum of absolute values
- `p=Inf`: Maximum absolute value
- Other p: General p-norm

### Vector Reductions

```julia
sum(v::VectorMPI)
prod(v::VectorMPI)
maximum(v::VectorMPI)
minimum(v::VectorMPI)
```

Compute reductions across all elements of a distributed vector.

### Vector Properties

```julia
length(v::VectorMPI) -> Int
size(v::VectorMPI) -> (Int,)
eltype(v::VectorMPI{T}) -> T
```

Return the global length, size tuple, or element type.

## Norms

### Element-wise Norms

Compute the p-norm of A treated as a vector of elements.

**Signature:**
```julia
norm(A::SparseMatrixMPI, p=2)
```

**Arguments:**
- `p=2` (default): Frobenius norm
- `p=1`: Sum of absolute values
- `p=Inf`: Maximum absolute value
- Other p: General p-norm

### Operator Norms

Compute the induced operator norm.

**Signature:**
```julia
opnorm(A::SparseMatrixMPI, p=1)
```

**Arguments:**
- `p=1`: Maximum absolute column sum
- `p=Inf`: Maximum absolute row sum

**Note:** `opnorm(A, 2)` (spectral norm) is not implemented.

## Matrix Properties

### Size

```julia
size(A::SparseMatrixMPI) -> (Int, Int)
size(A::SparseMatrixMPI, d::Integer) -> Int
```

Return the global size of the distributed matrix.

### Element Type

```julia
eltype(A::SparseMatrixMPI{T}) -> T
```

Return the element type of the matrix.

## Cache Management

```@docs
clear_plan_cache!
```

Clear all memoized plan caches.

**Example:**
```julia
clear_plan_cache!()
```

## Type Aliases

```julia
const TransposedSparseMatrixMPI{T} = Transpose{T, SparseMatrixMPI{T}}
```

Type alias for lazy transpose of `SparseMatrixMPI`.

## Full API Index

```@index
```
