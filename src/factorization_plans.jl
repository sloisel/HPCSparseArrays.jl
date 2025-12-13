"""
Communication plans for distributed multifrontal factorization.

Note: The current implementation gathers the entire matrix upfront and assigns
complete subtrees to single ranks, so no inter-rank communication occurs during
factorization. This file provides the caching infrastructure for factorization plans.
"""

using MPI
using SparseArrays

# ============================================================================
# Factorization Plans Collection
# ============================================================================

"""
    FactorizationPlans{T}

Placeholder for factorization communication plans.

Currently, the factorization gathers the entire matrix upfront and assigns
complete subtrees to single ranks, so no inter-rank communication is needed.
This struct exists for future extensibility and to maintain the caching interface.
"""
struct FactorizationPlans{T}
    structural_hash::Blake3Hash
end

# ============================================================================
# Plan Creation
# ============================================================================

"""
    create_factorization_plans(A::SparseMatrixMPI{T}, symbolic::SymbolicFactorization) -> FactorizationPlans{T}

Create factorization plans (currently just stores structural hash for caching).
"""
function create_factorization_plans(A::SparseMatrixMPI{T}, symbolic::SymbolicFactorization) where T
    return FactorizationPlans{T}(symbolic.structural_hash)
end

# ============================================================================
# Plan Caching
# ============================================================================

"""
Global cache for factorization plans, keyed by (structural hash, element type).
"""
const _factorization_plan_cache = Dict{Tuple{Blake3Hash, DataType}, Any}()

"""
    get_factorization_plans(A::SparseMatrixMPI{T}, symbolic::SymbolicFactorization) -> FactorizationPlans{T}

Get factorization plans, using cache if available.
"""
function get_factorization_plans(A::SparseMatrixMPI{T}, symbolic::SymbolicFactorization) where T
    key = (symbolic.structural_hash, T)
    if haskey(_factorization_plan_cache, key)
        return _factorization_plan_cache[key]::FactorizationPlans{T}
    end
    plans = create_factorization_plans(A, symbolic)
    _factorization_plan_cache[key] = plans
    return plans
end

"""
    clear_factorization_plan_cache!()

Clear the factorization plan cache.
"""
function clear_factorization_plan_cache!()
    empty!(_factorization_plan_cache)
end
