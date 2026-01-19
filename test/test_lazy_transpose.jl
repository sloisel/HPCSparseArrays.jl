# MPI test for lazy transpose operations
# This file is executed under mpiexec by runtests.jl
# Parameterized over scalar types and backends (CPU and GPU)

# Check Metal availability BEFORE loading MPI
const METAL_AVAILABLE = try
    using Metal
    Metal.functional()
catch e
    false
end

using Test
using MPI
using SparseArrays
using LinearAlgebra: Transpose, norm, opnorm

MPI.Init()

# Check CUDA availability BEFORE loading HPCSparseArrays
const CUDA_AVAILABLE = try
    using CUDA
    CUDA.device!(MPI.Comm_rank(MPI.COMM_WORLD) % length(CUDA.devices()))
    using NCCL_jll
    using CUDSS_jll
    CUDA.functional()
catch e
    false
end

using HPCSparseArrays

include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

include(joinpath(@__DIR__, "test_utils.jl"))
using .TestUtils

comm = MPI.COMM_WORLD

ts = @testset QuietTestSet "Lazy Transpose" begin

for (T, get_backend, backend_name) in TestUtils.ALL_CONFIGS
    TOL = TestUtils.tolerance(T)
    backend = get_backend(T)
    cpu_backend = TestUtils.cpu_version(backend)
    VT, ST, MT = TestUtils.expected_types(T, backend)

    println(io0(), "[test] transpose(A) * transpose(B) = transpose(B * A) ($T, $backend_name)")

    # C is 8x6, D is 6x8
    # C' is 6x8, D' is 8x6
    # C' * D' should be 6x6, and equal to (D * C)'
    m, n, p = 8, 6, 6
    I_C = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3]
    J_C = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
    V_C = T <: Complex ? T.(1:length(I_C)) .+ im .* T.(length(I_C):-1:1) : T.(1:length(I_C))
    C = sparse(I_C, J_C, V_C, m, n)

    I_D = [1, 2, 3, 4, 5, 6, 1, 2]
    J_D = [1, 2, 3, 4, 5, 6, 7, 8]
    V_D = T <: Complex ? T.(1:length(I_D)) .+ im .* T.(length(I_D):-1:1) : T.(1:length(I_D))
    D = sparse(I_D, J_D, V_D, p, m)

    Cdist = assert_type(HPCSparseMatrix(C, backend), ST)
    Ddist = assert_type(HPCSparseMatrix(D, backend), ST)

    # Compute transpose(C) * transpose(D) using lazy method
    result_lazy = transpose(Cdist) * transpose(Ddist)

    # Materialize the result (internal API)
    plan = HPCSparseArrays.TransposePlan(result_lazy.parent)
    result_dist = assert_type(HPCSparseArrays.execute_plan!(plan, result_lazy.parent), ST)

    # Reference: transpose(D * C)
    ref = sparse(transpose(D * C))
    ref_dist = assert_type(HPCSparseMatrix(ref, backend), ST)

    result_dist_cpu = to_backend(result_dist, cpu_backend)
    ref_dist_cpu = to_backend(ref_dist, cpu_backend)
    err = assert_uniform(norm(result_dist_cpu - ref_dist_cpu, Inf), name="lazy_trans_trans_err")
    @test err < TOL


    println(io0(), "[test] transpose(A) * B materialize left ($T, $backend_name)")

    # A is 8x6, so A' is 6x8
    # B is 8x10, so A' * B is 6x10
    m, n, p = 8, 6, 10
    I_A = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3, 5, 7]
    J_A = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
    V_A = T <: Complex ? T.(1:length(I_A)) .+ im .* T.(length(I_A):-1:1) : T.(1:length(I_A))
    A = sparse(I_A, J_A, V_A, m, n)

    I_B = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3, 5, 7]
    J_B = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2]
    V_B = T <: Complex ? T.(1:length(I_B)) .+ im .* T.(length(I_B):-1:1) : T.(1:length(I_B))
    B = sparse(I_B, J_B, V_B, m, p)

    Adist = assert_type(HPCSparseMatrix(A, backend), ST)
    Bdist = assert_type(HPCSparseMatrix(B, backend), ST)

    result_dist = assert_type(transpose(Adist) * Bdist, ST)
    ref = sparse(transpose(A)) * B
    ref_dist = assert_type(HPCSparseMatrix(ref, backend), ST)

    result_dist_cpu = to_backend(result_dist, cpu_backend)
    ref_dist_cpu = to_backend(ref_dist, cpu_backend)
    err = assert_uniform(norm(result_dist_cpu - ref_dist_cpu, Inf), name="trans_left_err")
    @test err < TOL


    println(io0(), "[test] A * transpose(B) materialize right ($T, $backend_name)")

    # A is 8x10, B is 6x10, so B' is 10x6
    # A * B' is 8x6
    m, n, p = 8, 10, 6
    I_A = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3]
    J_A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    V_A = T <: Complex ? T.(1:length(I_A)) .+ im .* T.(length(I_A):-1:1) : T.(1:length(I_A))
    A = sparse(I_A, J_A, V_A, m, n)

    I_B = [1, 2, 3, 4, 5, 6, 1, 2]
    J_B = [1, 2, 3, 4, 5, 6, 7, 8]
    V_B = T <: Complex ? T.(1:length(I_B)) .+ im .* T.(length(I_B):-1:1) : T.(1:length(I_B))
    B = sparse(I_B, J_B, V_B, p, n)

    Adist = assert_type(HPCSparseMatrix(A, backend), ST)
    Bdist = assert_type(HPCSparseMatrix(B, backend), ST)

    result_dist = assert_type(Adist * transpose(Bdist), ST)
    ref = A * sparse(transpose(B))
    ref_dist = assert_type(HPCSparseMatrix(ref, backend), ST)

    result_dist_cpu = to_backend(result_dist, cpu_backend)
    ref_dist_cpu = to_backend(ref_dist, cpu_backend)
    err = assert_uniform(norm(result_dist_cpu - ref_dist_cpu, Inf), name="trans_right_err")
    @test err < TOL


    if T <: Complex
        println(io0(), "[test] Adjoint conjugate transpose ($T, $backend_name)")

        m, n = 6, 8
        I_A = [1, 2, 3, 4, 5, 6, 1, 3]
        J_A = [1, 2, 3, 4, 5, 6, 7, 8]
        V_A = T.(1:length(I_A)) .+ im .* T.(length(I_A):-1:1)
        A = sparse(I_A, J_A, V_A, m, n)

        Adist = assert_type(HPCSparseMatrix(A, backend), ST)

        # A' = conj(A)^T
        Aadj = Adist'
        @test Aadj isa Transpose

        # Materialize and compare (internal API)
        plan = HPCSparseArrays.TransposePlan(Aadj.parent)
        result_dist = assert_type(HPCSparseArrays.execute_plan!(plan, Aadj.parent), ST)
        ref = sparse(A')
        ref_dist = assert_type(HPCSparseMatrix(ref, backend), ST)

        result_dist_cpu = to_backend(result_dist, cpu_backend)
        ref_dist_cpu = to_backend(ref_dist, cpu_backend)
        err = assert_uniform(norm(result_dist_cpu - ref_dist_cpu, Inf), name="adjoint_err")
        @test err < TOL


        println(io0(), "[test] conj(A) ($T, $backend_name)")

        m, n = 6, 8
        I_A = [1, 2, 3, 4, 5, 6, 1, 3]
        J_A = [1, 2, 3, 4, 5, 6, 7, 8]
        V_A = T.(1:length(I_A)) .+ im .* T.(length(I_A):-1:1)
        A = sparse(I_A, J_A, V_A, m, n)

        Adist = assert_type(HPCSparseMatrix(A, backend), ST)
        result_dist = assert_type(conj(Adist), ST)
        ref = conj(A)
        ref_dist = assert_type(HPCSparseMatrix(ref, backend), ST)

        result_dist_cpu = to_backend(result_dist, cpu_backend)
        ref_dist_cpu = to_backend(ref_dist, cpu_backend)
        err = assert_uniform(norm(result_dist_cpu - ref_dist_cpu, Inf), name="conj_err")
        @test err < TOL
    end


    println(io0(), "[test] Scalar multiplication ($T, $backend_name)")

    m, n = 6, 8
    I_A = [1, 2, 3, 4, 5, 6, 1, 3]
    J_A = [1, 2, 3, 4, 5, 6, 7, 8]
    V_A = T <: Complex ? T.(1:length(I_A)) .+ im .* T.(length(I_A):-1:1) : T.(1:length(I_A))
    A = sparse(I_A, J_A, V_A, m, n)

    Adist = assert_type(HPCSparseMatrix(A, backend), ST)

    # Test a * A
    a = T <: Complex ? T(2.5 + 0.5im) : T(2.5)
    result_dist = assert_type(a * Adist, ST)
    ref_dist = assert_type(HPCSparseMatrix(a * A, backend), ST)
    result_dist_cpu = to_backend(result_dist, cpu_backend)
    ref_dist_cpu = to_backend(ref_dist, cpu_backend)
    err1 = assert_uniform(norm(result_dist_cpu - ref_dist_cpu, Inf), name="scalar_mul_aA_err")
    @test err1 < TOL

    # Test A * a
    result_dist = assert_type(Adist * a, ST)
    result_dist_cpu = to_backend(result_dist, cpu_backend)
    err2 = assert_uniform(norm(result_dist_cpu - ref_dist_cpu, Inf), name="scalar_mul_Aa_err")
    @test err2 < TOL

    # Test a * transpose(A) (internal API)
    At = transpose(Adist)
    result_lazy = a * At
    plan = HPCSparseArrays.TransposePlan(result_lazy.parent)
    result_dist = assert_type(HPCSparseArrays.execute_plan!(plan, result_lazy.parent), ST)
    ref = sparse(transpose(a * A))
    ref_dist = assert_type(HPCSparseMatrix(ref, backend), ST)
    result_dist_cpu = to_backend(result_dist, cpu_backend)
    ref_dist_cpu = to_backend(ref_dist, cpu_backend)
    err3 = assert_uniform(norm(result_dist_cpu - ref_dist_cpu, Inf), name="scalar_mul_aAt_err")
    @test err3 < TOL

    # Test transpose(A) * a
    result_lazy = At * a
    result_dist = assert_type(HPCSparseArrays.execute_plan!(plan, result_lazy.parent), ST)
    result_dist_cpu = to_backend(result_dist, cpu_backend)
    err4 = assert_uniform(norm(result_dist_cpu - ref_dist_cpu, Inf), name="scalar_mul_Ata_err")
    @test err4 < TOL


    println(io0(), "[test] norm ($T, $backend_name)")

    m, n = 6, 8
    I_A = [1, 2, 3, 4, 5, 6, 1, 3]
    J_A = [1, 2, 3, 4, 5, 6, 7, 8]
    V_A = T <: Complex ? T.(1:length(I_A)) .+ im .* T.(length(I_A):-1:1) : T.(1:length(I_A))
    A = sparse(I_A, J_A, V_A, m, n)

    Adist = assert_type(HPCSparseMatrix(A, backend), ST)
    Adist_cpu = to_backend(Adist, cpu_backend)

    err1 = assert_uniform(abs(norm(Adist_cpu) - norm(A)), name="norm_2_err")
    err2 = assert_uniform(abs(norm(Adist_cpu, 1) - norm(A, 1)), name="norm_1_err")
    err3 = assert_uniform(abs(norm(Adist_cpu, Inf) - norm(A, Inf)), name="norm_inf_err")
    err4 = assert_uniform(abs(norm(Adist_cpu, 3) - norm(A, 3)), name="norm_3_err")

    @test err1 < TOL
    @test err2 < TOL
    @test err3 < TOL
    @test err4 < TOL


    println(io0(), "[test] opnorm ($T, $backend_name)")

    m, n = 6, 8
    I_A = [1, 2, 3, 4, 5, 6, 1, 3, 2, 4]
    J_A = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3]
    V_A = T <: Complex ? T.(1:length(I_A)) .+ im .* T.(length(I_A):-1:1) : T.(1:length(I_A))
    A = sparse(I_A, J_A, V_A, m, n)

    Adist = assert_type(HPCSparseMatrix(A, backend), ST)
    Adist_cpu = to_backend(Adist, cpu_backend)

    err1 = assert_uniform(abs(opnorm(Adist_cpu, 1) - opnorm(A, 1)), name="opnorm_1_err")
    err2 = assert_uniform(abs(opnorm(Adist_cpu, Inf) - opnorm(A, Inf)), name="opnorm_inf_err")

    @test err1 < TOL
    @test err2 < TOL

end  # for (T, get_backend, backend_name)

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

println(io0(), "Test Summary: Lazy Transpose | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")

MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
