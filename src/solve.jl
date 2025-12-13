"""
Distributed solve routines for LU and LDLT factorizations.

## LU Factorization Solve

The factorization gives: L * U = Pr_elim * Ap_elim where:
- Ap = P' * A * P is the AMD-reordered matrix (P is fill-reducing permutation)
- Ap_elim = Ap reordered to elimination order via elim_to_global
- Pr_elim is the row permutation from pivoting, in elimination order indices
- L and U are stored in elimination order indices (1 to n)

To solve A * x = b:
1. Apply AMD permutation: bp = P' * b = b[perm]
2. Reorder to elimination order: b_elim[k] = bp[elim_to_global[k]]
3. Apply pivot permutation: c[k] = b_elim[row_perm_elim[k]] where row_perm_elim maps elim indices
4. Forward solve: L * y = c (both in elimination order)
5. Backward solve: U * z = y (both in elimination order)
6. Reorder from elimination to global: zp[elim_to_global[k]] = z[k]
7. Apply inverse AMD permutation: x = P * zp = zp[invperm]

## LDLT Factorization Solve

Given L * D * L^T * x = b where:
- L is unit lower triangular (distributed)
- D is block diagonal with 1×1 and 2×2 blocks

Solve by:
1. Apply fill-reducing permutation
2. Apply symmetric pivot permutation
3. Forward solve: L * y = b
4. Diagonal solve: D * z = y (handling 2×2 blocks)
5. Backward solve: L^T * x = z
6. Apply inverse permutations
"""

using MPI
using SparseArrays

# ============================================================================
# LU Factorization Solve
# ============================================================================

"""
    solve(F::LUFactorizationMPI{T}, b::VectorMPI{T}) -> VectorMPI{T}

Solve the linear system A*x = b using the precomputed LU factorization.
"""
function solve(F::LUFactorizationMPI{T}, b::VectorMPI{T}) where T
    x = VectorMPI(zeros(T, F.symbolic.n); partition=b.partition)
    solve!(x, F, b)
    return x
end

"""
    solve!(x::VectorMPI{T}, F::LUFactorizationMPI{T}, b::VectorMPI{T})

Solve A*x = b in-place using LU factorization.

Uses MUMPS-style distributed solve that keeps factors distributed
and only communicates at subtree boundaries.
"""
function solve!(x::VectorMPI{T}, F::LUFactorizationMPI{T}, b::VectorMPI{T}) where T
    distributed_solve_lu!(x, F, b)
    return x
end

"""
    Base.:\\(F::LUFactorizationMPI{T}, b::VectorMPI{T}) where T

Solve A*x = b using the backslash operator.
"""
function Base.:\(F::LUFactorizationMPI{T}, b::VectorMPI{T}) where T
    return solve(F, b)
end

# ============================================================================
# LDLT Factorization Solve
# ============================================================================

"""
    solve(F::LDLTFactorizationMPI{T}, b::VectorMPI{T}) -> VectorMPI{T}

Solve the linear system A*x = b using the precomputed LDLT factorization.
"""
function solve(F::LDLTFactorizationMPI{T}, b::VectorMPI{T}) where T
    x = VectorMPI(zeros(T, F.symbolic.n); partition=b.partition)
    solve!(x, F, b)
    return x
end

"""
    solve!(x::VectorMPI{T}, F::LDLTFactorizationMPI{T}, b::VectorMPI{T})

Solve A*x = b in-place using LDLT factorization.

Uses MUMPS-style distributed solve that keeps factors distributed
and only communicates at subtree boundaries.
"""
function solve!(x::VectorMPI{T}, F::LDLTFactorizationMPI{T}, b::VectorMPI{T}) where T
    distributed_solve_ldlt!(x, F, b)
    return x
end

"""
    Base.:\\(F::LDLTFactorizationMPI{T}, b::VectorMPI{T}) where T

Solve A*x = b using the backslash operator.
"""
function Base.:\(F::LDLTFactorizationMPI{T}, b::VectorMPI{T}) where T
    return solve(F, b)
end

# ============================================================================
# Transpose Solve for LDLT (for symmetric matrices, transpose(A) = A)
# ============================================================================

"""
    solve_transpose(F::LDLTFactorizationMPI{T}, b::VectorMPI{T}) -> VectorMPI{T}

Solve transpose(A)*x = b. For symmetric matrices, transpose(A) = A,
so this is equivalent to solve(F, b).
"""
function solve_transpose(F::LDLTFactorizationMPI{T}, b::VectorMPI{T}) where T
    return solve(F, b)
end

# Wrapper type for transpose of LDLT factorization
struct TransposeLDLT{T}
    parent::LDLTFactorizationMPI{T}
end

Base.transpose(F::LDLTFactorizationMPI{T}) where T = TransposeLDLT{T}(F)

function Base.:\(Ft::TransposeLDLT{T}, b::VectorMPI{T}) where T
    return solve_transpose(Ft.parent, b)
end
