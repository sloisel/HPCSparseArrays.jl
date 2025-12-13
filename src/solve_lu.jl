"""
Distributed solve routines for LU factorization.

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
"""

using MPI
using SparseArrays

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
