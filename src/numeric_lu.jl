"""
Numerical LU factorization using the distributed multifrontal method.

Main algorithm:
1. Process supernodes in postorder (leaves to root)
2. For each supernode owned by this rank:
   a. Initialize frontal matrix from distributed sparse entries
   b. Extend-add contributions from children (all local since subtrees are on same rank)
   c. Partial factorization with pivoting
   d. Extract L and U factors to local storage
   e. Store update matrix for parent supernode
"""

using MPI
using SparseArrays

"""
    lu(A::SparseMatrixMPI{T}; reuse_symbolic=true) -> LUFactorizationMPI{T}

Compute LU factorization of a distributed sparse matrix.

Uses the multifrontal method with:
- AMD fill-reducing ordering
- Partial pivoting for numerical stability
- MUMPS-style subtree-to-rank mapping
- MUMPS-style distributed matrix input (each rank only provides its local portion)

If `reuse_symbolic=true`, caches and reuses symbolic factorization for matrices
with the same sparsity pattern.
"""
function LinearAlgebra.lu(A::SparseMatrixMPI{T}; reuse_symbolic::Bool=true) where T
    # Get or compute symbolic factorization
    symbolic = reuse_symbolic ?
        get_symbolic_factorization(A; symmetric=false) :
        compute_symbolic_factorization(A; symmetric=false)

    # Perform numerical factorization
    return numerical_factorization_lu(A, symbolic)
end

"""
    numerical_factorization_lu(A, symbolic) -> LUFactorizationMPI{T}

Perform the numerical LU factorization.

Uses MUMPS-style distributed matrix input where each rank only provides its
local portion of the matrix.

Note: The current supernode assignment places complete subtrees on single ranks,
so all parent-child communication is local (no MPI communication during factorization).
"""
function numerical_factorization_lu(A::SparseMatrixMPI{T},
                                    symbolic::SymbolicFactorization) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    n = symbolic.n
    nsupernodes = length(symbolic.supernodes)

    # Use distributed input plan
    input_plan = get_or_create_input_plan(symbolic, A)
    execute_input_plan!(input_plan, A)

    # Storage for update matrices (keyed by supernode index)
    updates = Dict{Int, Matrix{T}}()
    update_rows = Dict{Int, Vector{Int}}()
    update_cols = Dict{Int, Vector{Int}}()  # Column indices (may differ from rows after pivoting)

    # Accumulators for L and U in COO format (each rank stores its portion)
    L_I = Int[]
    L_J = Int[]
    L_V = T[]
    U_I = Int[]
    U_J = Int[]
    U_V = T[]

    # Track row permutation from pivoting
    # Each rank contributes its portion, then we Allreduce
    row_perm_local = zeros(Int, n)

    # Process supernodes in postorder
    for sidx in symbolic.snode_postorder
        snode = symbolic.supernodes[sidx]
        info = symbolic.frontal_info[sidx]
        snode_owner = symbolic.snode_owner[sidx]

        if rank == snode_owner
            # This rank owns this supernode

            # 1. Initialize frontal matrix from distributed input
            F = initialize_frontal_distributed(input_plan, A, snode, info)

            # 2. Extend-add update matrices from children
            # Note: All children are on the same rank (subtrees assigned to single ranks)
            for child_sidx in symbolic.snode_children[sidx]
                if haskey(updates, child_sidx)
                    extend_add!(F, updates[child_sidx], update_rows[child_sidx], update_cols[child_sidx])
                    delete!(updates, child_sidx)
                    delete!(update_rows, child_sidx)
                    delete!(update_cols, child_sidx)
                end
            end

            # 3. Partial factorization with pivoting
            partial_factor!(F)

            # 4. Record row permutation
            base_elim = symbolic.global_to_elim[F.col_indices[1]]
            for k = 1:info.nfs
                elim_pos = base_elim + k - 1
                row_perm_local[elim_pos] = F.row_indices[k]
            end

            # 5. Extract L and U entries
            extract_LU!(F, snode, symbolic.global_to_elim, L_I, L_J, L_V, U_I, U_J, U_V)

            # 6. Store update matrix for parent (parent is on same rank)
            parent_sidx = symbolic.snode_parent[sidx]
            if parent_sidx != 0
                nfs = info.nfs
                nrows = length(info.row_indices)
                if nrows > nfs
                    updates[sidx] = copy(F.F[nfs+1:nrows, nfs+1:nrows])
                    # After pivoting: row_indices is permuted, col_indices is not
                    update_rows[sidx] = copy(F.row_indices[nfs+1:nrows])
                    update_cols[sidx] = copy(F.col_indices[nfs+1:nrows])
                end
            end
        end

        # Synchronize between supernodes to ensure correct ordering
        MPI.Barrier(comm)
    end

    # Build local sparse L and U
    L_local = sparse(L_I, L_J, L_V, n, n)
    U_local = sparse(U_I, U_J, U_V, n, n)

    # Gather global row permutation via Allreduce (max works since uninit are 0)
    row_perm = MPI.Allreduce(row_perm_local, MPI.MAX, comm)

    # Build inverse row permutation
    inv_row_perm = zeros(Int, n)
    for k = 1:n
        if row_perm[k] != 0
            inv_row_perm[row_perm[k]] = k
        end
    end

    return LUFactorizationMPI{T}(symbolic, L_local, U_local, row_perm, inv_row_perm)
end
