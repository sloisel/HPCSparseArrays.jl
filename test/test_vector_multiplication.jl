# MPI test for matrix-vector multiplication
# This file is executed under mpiexec by runtests.jl
# Parameterized over scalar types and backends (CPU/GPU)

# Check Metal availability BEFORE loading MPI
const METAL_AVAILABLE = try
    using Metal
    Metal.functional()
catch e
    false
end

using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra
using Test

include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

include(joinpath(@__DIR__, "test_utils.jl"))
using .TestUtils

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

ts = @testset QuietTestSet "Matrix-Vector Multiplication" begin

for (T, to_backend, backend_name) in TestUtils.ALL_CONFIGS
    TOL = TestUtils.tolerance(T)
    VT, ST, MT = TestUtils.expected_types(T, to_backend)

    println(io0(), "[test] Matrix-vector multiplication ($T, $backend_name)")

    # Create deterministic test matrix (same on all ranks)
    n = 8
    A = TestUtils.tridiagonal_matrix(T, n)

    # Vector x: simple sequence
    x_global = TestUtils.test_vector(T, n)

    # Create distributed versions
    Adist = to_backend(SparseMatrixMPI{T}(A))
    xdist = to_backend(VectorMPI(x_global))

    # Compute distributed product
    ydist = assert_type(Adist * xdist, VT)

    # Reference result
    y_ref = A * x_global

    # Check local portion matches
    my_start = Adist.row_partition[rank+1]
    my_end = Adist.row_partition[rank+2] - 1
    local_ref = y_ref[my_start:my_end]

    local_y = TestUtils.local_values(ydist)
    err = maximum(abs.(local_y .- local_ref))
    @test err < TOL


    println(io0(), "[test] Matrix-vector multiplication in-place ($T, $backend_name)")

    n = 8
    A = TestUtils.tridiagonal_matrix(T, n)
    x_global = TestUtils.test_vector(T, n)

    Adist = to_backend(SparseMatrixMPI{T}(A))
    xdist = to_backend(VectorMPI(x_global))

    # Create output vector with matching partition
    ydist = to_backend(VectorMPI(zeros(T, n)))

    # In-place multiplication
    LinearAlgebra.mul!(ydist, Adist, xdist)

    # Reference result
    y_ref = A * x_global

    my_start = Adist.row_partition[rank+1]
    my_end = Adist.row_partition[rank+2] - 1
    local_ref = y_ref[my_start:my_end]

    local_y = TestUtils.local_values(ydist)
    err = maximum(abs.(local_y .- local_ref))
    @test err < TOL


    println(io0(), "[test] Non-square matrix-vector multiplication ($T, $backend_name)")

    # A is 6x8, x is length 8, y should be length 6
    m, k = 6, 8
    I_A = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
    J_A = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2]
    V_A = T <: Complex ? T.(1:length(I_A)) .+ im .* T.(length(I_A):-1:1) : T.(1:length(I_A))
    A = sparse(I_A, J_A, V_A, m, k)

    x_global = TestUtils.test_vector(T, k)

    Adist = to_backend(SparseMatrixMPI{T}(A))
    xdist = to_backend(VectorMPI(x_global))

    ydist = assert_type(Adist * xdist, VT)
    y_ref = A * x_global

    my_start = Adist.row_partition[rank+1]
    my_end = Adist.row_partition[rank+2] - 1
    local_ref = y_ref[my_start:my_end]

    local_y = TestUtils.local_values(ydist)
    err = maximum(abs.(local_y .- local_ref); init=zero(real(T)))
    @test err < TOL


    # Vector transpose and adjoint (meaningful for complex types, but works for real too)
    if T <: Complex
        println(io0(), "[test] Vector transpose and adjoint ($T, $backend_name)")

        n = 8
        A = TestUtils.tridiagonal_matrix(T, n)
        x_global = TestUtils.test_vector(T, n)

        Adist = to_backend(SparseMatrixMPI{T}(A))
        xdist = to_backend(VectorMPI(x_global))

        # Test conj(v)
        xconj = assert_type(conj(xdist), VT)
        xconj_ref = conj.(x_global)
        my_start = xdist.partition[rank+1]
        my_end = xdist.partition[rank+2] - 1
        local_xconj = TestUtils.local_values(xconj)
        err_conj = maximum(abs.(local_xconj .- xconj_ref[my_start:my_end]))
        @test err_conj < TOL

        # Test transpose(v) * A = transpose(transpose(A) * v)
        yt = transpose(xdist) * Adist
        y_ref = transpose(transpose(A) * x_global)

        my_col_start = Adist.col_partition[rank+1]
        my_col_end = Adist.col_partition[rank+2] - 1
        local_ref = collect(y_ref)[my_col_start:my_col_end]
        local_yt = TestUtils.local_values(yt.parent)
        err_transpose = maximum(abs.(local_yt .- local_ref))
        @test err_transpose < TOL

        # Test adjoint: v' * A = transpose(conj(v)) * A = transpose(transpose(A) * conj(v))
        yt_adj = xdist' * Adist
        y_adj_ref = x_global' * A

        local_adj_ref = collect(y_adj_ref)[my_col_start:my_col_end]
        local_yt_adj = TestUtils.local_values(yt_adj.parent)
        err_adjoint = maximum(abs.(local_yt_adj .- local_adj_ref))
        @test err_adjoint < TOL
    end


    println(io0(), "[test] Vector norms ($T, $backend_name)")

    n = 10
    x_global = TestUtils.test_vector(T, n)
    xdist = to_backend(VectorMPI(x_global))

    # For GPU, norm requires CPU conversion
    xdist_cpu = TestUtils.to_cpu(xdist)

    # 2-norm
    norm2 = assert_uniform(norm(xdist_cpu), name="norm2")
    norm2_ref = norm(x_global)
    @test abs(norm2 - norm2_ref) < TOL

    # 1-norm
    norm1 = assert_uniform(norm(xdist_cpu, 1), name="norm1")
    norm1_ref = norm(x_global, 1)
    @test abs(norm1 - norm1_ref) < TOL

    # Inf-norm
    norminf = assert_uniform(norm(xdist_cpu, Inf), name="norminf")
    norminf_ref = norm(x_global, Inf)
    @test abs(norminf - norminf_ref) < TOL

    # 3-norm (general p)
    norm3 = assert_uniform(norm(xdist_cpu, 3), name="norm3")
    norm3_ref = norm(x_global, 3)
    @test abs(norm3 - norm3_ref) < TOL

    # Non-integer p-norm (p = 1.5)
    norm15 = assert_uniform(norm(xdist_cpu, 1.5), name="norm15")
    norm15_ref = norm(x_global, 1.5)
    @test abs(norm15 - norm15_ref) < TOL


    println(io0(), "[test] Vector reductions ($T, $backend_name)")

    n = 8
    # Use real values for reductions (prod can overflow with complex)
    x_global_real = real(T).(collect(1.0:n))
    xdist = to_backend(VectorMPI(x_global_real))
    xdist_cpu = TestUtils.to_cpu(xdist)

    # sum
    s = assert_uniform(sum(xdist_cpu), name="sum")
    s_ref = sum(x_global_real)
    @test abs(s - s_ref) < TOL

    # prod
    p = assert_uniform(prod(xdist_cpu), name="prod")
    p_ref = prod(x_global_real)
    @test abs(p - p_ref) < TOL

    # maximum
    mx = assert_uniform(maximum(xdist_cpu), name="maximum")
    mx_ref = maximum(x_global_real)
    @test abs(mx - mx_ref) < TOL

    # minimum
    mn = assert_uniform(minimum(xdist_cpu), name="minimum")
    mn_ref = minimum(x_global_real)
    @test abs(mn - mn_ref) < TOL


    println(io0(), "[test] Vector addition and subtraction ($T, $backend_name)")

    n = 8
    u_global, v_global = TestUtils.test_vector_pair(T, n)

    udist = to_backend(VectorMPI(u_global))
    vdist = to_backend(VectorMPI(v_global))

    my_start = udist.partition[rank+1]
    my_end = udist.partition[rank+2] - 1

    # u + v
    wdist = assert_type(udist + vdist, VT)
    w_ref = u_global + v_global
    local_w = TestUtils.local_values(wdist)
    err_add = maximum(abs.(local_w .- w_ref[my_start:my_end]))
    @test err_add < TOL

    # u - v
    wdist = assert_type(udist - vdist, VT)
    w_ref = u_global - v_global
    local_w = TestUtils.local_values(wdist)
    err_sub = maximum(abs.(local_w .- w_ref[my_start:my_end]))
    @test err_sub < TOL

    # -v
    wdist = assert_type(-vdist, VT)
    w_ref = -v_global
    local_w = TestUtils.local_values(wdist)
    err_neg = maximum(abs.(local_w .- w_ref[my_start:my_end]))
    @test err_neg < TOL


    println(io0(), "[test] Scalar multiplication ($T, $backend_name)")

    n = 8
    v_global = TestUtils.test_vector(T, n)
    vdist = to_backend(VectorMPI(v_global))
    a = T <: Complex ? T(3.5 + 0.5im) : T(3.5)

    my_start = vdist.partition[rank+1]
    my_end = vdist.partition[rank+2] - 1

    # a * v
    wdist = assert_type(a * vdist, VT)
    w_ref = a * v_global
    local_w = TestUtils.local_values(wdist)
    err_av = maximum(abs.(local_w .- w_ref[my_start:my_end]))
    @test err_av < TOL

    # v * a
    wdist = assert_type(vdist * a, VT)
    local_w = TestUtils.local_values(wdist)
    err_va = maximum(abs.(local_w .- w_ref[my_start:my_end]))
    @test err_va < TOL

    # v / a
    wdist = assert_type(vdist / a, VT)
    w_ref = v_global / a
    local_w = TestUtils.local_values(wdist)
    err_div = maximum(abs.(local_w .- w_ref[my_start:my_end]))
    @test err_div < TOL

    # a * transpose(v)
    wt = a * transpose(vdist)
    w_ref = a * v_global
    local_wt = TestUtils.local_values(wt.parent)
    err_avt = maximum(abs.(local_wt .- w_ref[my_start:my_end]))
    @test err_avt < TOL

    # transpose(v) * a
    wt = transpose(vdist) * a
    local_wt = TestUtils.local_values(wt.parent)
    err_vta = maximum(abs.(local_wt .- w_ref[my_start:my_end]))
    @test err_vta < TOL

    # transpose(v) / a
    wt = transpose(vdist) / a
    w_ref = v_global / a
    local_wt = TestUtils.local_values(wt.parent)
    err_vtdiv = maximum(abs.(local_wt .- w_ref[my_start:my_end]))
    @test err_vtdiv < TOL

end  # for (T, to_backend, backend_name)


# Tests that don't need to be parameterized (type-agnostic)

println(io0(), "[test] Vector operations with different partitions")

# Skip this test if running with fewer than 2 ranks (need different partitions)
if nranks < 2
    @test true  # Pass trivially
else
    # Clear cache to avoid interference from previous tests
    LinearAlgebraMPI.clear_plan_cache!()

    # Use Float64 for this test (partition logic is type-independent)
    T = Float64
    TOL = TestUtils.tolerance(T)

    # Use a size that works well with nranks
    n = 3 * nranks
    u_global, v_global = TestUtils.test_vector_pair(T, n)

    # Create u with default partition
    udist = VectorMPI(u_global)

    # Create v with a different (custom) partition
    custom_partition = Vector{Int}(undef, nranks + 1)
    custom_partition[1] = 1
    for r in 1:nranks
        extra = r == 1 ? 2 : (r <= nranks÷2 ? 3 : 4)
        remaining = n - custom_partition[r] + 1
        remaining_ranks = nranks - r + 1
        alloc = min(extra, remaining - (remaining_ranks - 1))
        custom_partition[r+1] = custom_partition[r] + max(1, alloc)
    end
    custom_partition[end] = n + 1

    v_hash = LinearAlgebraMPI.compute_partition_hash(custom_partition)
    local_v_range = custom_partition[rank+1]:(custom_partition[rank+2]-1)
    vdist = VectorMPI{T}(v_hash, copy(custom_partition), v_global[local_v_range])

    # Verify partitions are different
    @test udist.partition != vdist.partition

    # u + v (result should have u's partition)
    wdist = udist + vdist
    w_ref = u_global + v_global
    my_start = udist.partition[rank+1]
    my_end = udist.partition[rank+2] - 1
    err_add = maximum(abs.(wdist.v .- w_ref[my_start:my_end]))
    @test err_add < TOL
    @test wdist.partition == udist.partition

    # u - v
    wdist = udist - vdist
    w_ref = u_global - v_global
    err_sub = maximum(abs.(wdist.v .- w_ref[my_start:my_end]))
    @test err_sub < TOL

    # v + u (result should have v's partition)
    wdist2 = vdist + udist
    w_ref2 = v_global + u_global
    my_v_start = vdist.partition[rank+1]
    my_v_end = vdist.partition[rank+2] - 1
    err_add2 = maximum(abs.(wdist2.v .- w_ref2[my_v_start:my_v_end]))
    @test err_add2 < TOL
    @test wdist2.partition == vdist.partition

    # transpose(u) + transpose(v) with different partitions
    wt = transpose(udist) + transpose(vdist)
    w_add_ref = u_global + v_global
    err_tadd = maximum(abs.(wt.parent.v .- w_add_ref[my_start:my_end]))
    @test err_tadd < TOL
end


println(io0(), "[test] Vector size and eltype")

n = 8
v_global = collect(1.0:n)
vdist = VectorMPI(v_global)

@test assert_uniform(length(vdist), name="length") == n
@test assert_uniform(size(vdist, 1), name="size1") == n
@test eltype(vdist) == Float64
@test eltype(VectorMPI{ComplexF64}) == ComplexF64


end  # QuietTestSet

# Aggregate counts across ranks
local_counts = [
    get(ts.counts, :pass, 0),
    get(ts.counts, :fail, 0),
    get(ts.counts, :error, 0),
    get(ts.counts, :broken, 0),
    get(ts.counts, :skip, 0),
]
global_counts = similar(local_counts)
MPI.Allreduce!(local_counts, global_counts, +, comm)

println(io0(), "Test Summary: Matrix-Vector Multiplication | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")

MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
