# Exploratory ≥3-GPU distributed tests. NOT included from runtests.jl or test/mdla.jl
# — see test/multigpu/README.md for why and for how to run them.
#
# What 2 partitions cannot prove: with 3 slabs the MIDDLE partition carries
# `Interface` ghosts on BOTH cut faces, owned by two DIFFERENT neighbors. That is
# the first configuration where the ghost section of MDLA's `local_x` holds two
# owners' planes at once, so it is the first real test of the owner-ascending,
# plane-ascending ordering rule `_slab_ghost_layout` replays from MDLA's
# `_compute_ghost_topology`. `test/partitioning.jl` proves this CPU-side against
# emulated exchange semantics; here it runs on real `scatter!`/`reduce!`.
#
# Assumes the caller has loaded MatrixFreeOperators, Test, LinearAlgebra, Random,
# StaticArrays, Adapt and Krylov (same contract as test/mdla_gpu.jl).
using CUDA
using MultiDeviceLinearAlgebra

const MDLA_EXT_MP = Base.get_extension(MatrixFreeOperators, :MatrixFreeOperatorsMDLAExt)
NGPUS_MP = length(CUDA.devices())

mp_grid(cutbc) = CartesianGrid(
    ((0.0, 2π), (0.0, 1.0)), (16, 18); bc=((Periodic(), Periodic()), cutbc)
)

@testset "MDLA 3-partition (≥3 GPUs)" begin
    if NGPUS_MP >= 3
        @testset "middle slab is doubly-Interface" begin
            for cutbc in ((Dirichlet(), Neumann()), (Periodic(), Periodic()))
                parts = partition_grid(mp_grid(cutbc), 3)
                @test length(parts) == 3
                # the point of 3 partitions: the middle slab is cut on both faces
                mid = boundary_conditions(parts[2])[2]
                @test mid[1] isa MatrixFreeOperators.Interface
                @test mid[2] isa MatrixFreeOperators.Interface
                # and it requests ghosts from two distinct owners
                gg, plans = MatrixFreeOperators._slab_ghost_layout(mp_grid(cutbc), parts)
                @test length(plans[2]) == 2
            end
        end

        @testset "3-partition forward parity" begin
            rng = Random.MersenneTwister(53)
            for cutbc in ((Dirichlet(), Neumann()), (Periodic(), Periodic()))
                g = mp_grid(cutbc)
                L = laplacian(g)
                n = prod(local_size(g))
                xflat = rand(rng, n)

                P1 = prepare_distributed(L, 1)
                y1 = MultiDeviceVector(zeros(n), P1.spec)
                mul!(y1, P1, MultiDeviceVector(copy(xflat), P1.spec))

                P3 = prepare_distributed(L, 3)
                y3 = MultiDeviceVector(zeros(n), P3.spec)
                mul!(y3, P3, MultiDeviceVector(copy(xflat), P3.spec))
                # bitwise: same device arithmetic, per-cell-independent kernels
                @test gather(y3) == gather(y1)

                y0 = rand(rng, n)
                yαβ = MultiDeviceVector(copy(y0), P3.spec)
                mul!(yαβ, P3, MultiDeviceVector(copy(xflat), P3.spec), 2.5, 0.5)
                @test gather(yαβ) ≈ 2.5 .* gather(y1) .+ 0.5 .* y0
            end
        end

        @testset "3-partition adjoint identity and CPU parity" begin
            rng = Random.MersenneTwister(59)
            for cutbc in ((Dirichlet(), Neumann()), (Periodic(), Periodic()))
                g = mp_grid(cutbc)
                n = prod(local_size(g))
                for L in (laplacian(g), 0.5 * laplacian(g) + 2.0 * identity_op())
                    P = prepare_distributed(L, 3)
                    xflat = rand(rng, n)
                    yflat = rand(rng, n)
                    x = MultiDeviceVector(copy(xflat), P.spec)
                    y = MultiDeviceVector(copy(yflat), P.spec)

                    Lx = MultiDeviceVector(zeros(n), P.spec)
                    mul!(Lx, P, x)
                    x̄ = MultiDeviceVector(zeros(n), P.spec)
                    MDLA_EXT_MP._mul_adjoint!(x̄, P, y)
                    @test isapprox(dot(Lx, y), dot(x, x̄); rtol=1e-12)

                    # CPU global adjoint parity — catches a mis-ordered two-owner ghost section
                    ȳg = scalar_field(g)
                    flat_to_interior!(ȳg, yflat)
                    x̄g = scalar_field(g)
                    apply_adjoint!(x̄g, L, ȳg, g)
                    @test isapprox(gather(x̄), flatten(x̄g); rtol=1e-12)
                end
            end
        end

        @testset "3-partition Krylov.cg parity" begin
            g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (24, 26))
            L = -1.0 * laplacian(g)   # SPD under homogeneous Dirichlet
            f = set!(scalar_field(g), x -> sin(π * x[1]) * sin(π * x[2]))
            bflat = flatten(f)
            n = length(bflat)

            u_cpu, stats_cpu = Krylov.cg(prepare(L), bflat; atol=1e-10, rtol=1e-10)
            @test stats_cpu.solved

            P = prepare_distributed(L, 3)
            b = MultiDeviceVector(copy(bflat), P.spec)
            u, stats = Krylov.cg(P, b; atol=1e-10, rtol=1e-10)
            @test stats.solved
            @test isapprox(gather(u), u_cpu; rtol=1e-8)
        end
    else
        @test_skip "MDLA 3-partition — needs ≥ 3 CUDA devices"
    end
end
