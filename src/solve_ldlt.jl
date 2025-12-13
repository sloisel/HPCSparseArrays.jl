"""
Distributed solve routines for LDLT factorization with Bunch-Kaufman pivoting.

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
