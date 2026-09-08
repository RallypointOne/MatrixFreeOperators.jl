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

        # The mid-tree exchange (issue #31 slice 2a) has the same two-owner
        # property as the root one, and this is the only configuration that
        # exercises it: the middle slab's intermediate ghost section carries
        # planes from two different owners.
        @testset "3-partition Composed forward parity" begin
            rng = Random.MersenneTwister(61)
            for cutbc in ((Dirichlet(), Neumann()), (Periodic(), Periodic()))
                g = mp_grid(cutbc)
                n = prod(local_size(g))
                xflat = rand(rng, n)
                for L in (
                    laplacian(g) * laplacian(g),
                    derivative(g, 1) * derivative(g, 2),
                    adjoint(derivative(g, 2)) * derivative(g, 2),
                )
                    P1 = prepare_distributed(L, 1)
                    y1 = MultiDeviceVector(zeros(n), P1.spec)
                    mul!(y1, P1, MultiDeviceVector(copy(xflat), P1.spec))

                    P3 = prepare_distributed(L, 3)
                    y3 = MultiDeviceVector(zeros(n), P3.spec)
                    mul!(y3, P3, MultiDeviceVector(copy(xflat), P3.spec))
                    @test gather(y3) == gather(y1)
                end
            end
        end

        @testset "3-partition Composed adjoint identity" begin
            rng = Random.MersenneTwister(67)
            for cutbc in ((Dirichlet(), Neumann()), (Periodic(), Periodic()))
                g = mp_grid(cutbc)
                n = prod(local_size(g))
                for L in (laplacian(g) * laplacian(g), adjoint(derivative(g, 1)) + laplacian(g))
                    P = prepare_distributed(L, 3)
                    x = MultiDeviceVector(rand(rng, n), P.spec)
                    y = MultiDeviceVector(rand(rng, n), P.spec)
                    Lx = MultiDeviceVector(zeros(n), P.spec)
                    mul!(Lx, P, x)
                    x̄ = MultiDeviceVector(zeros(n), P.spec)
                    MDLA_EXT_MP._mul_adjoint!(x̄, P, y)
                    @test isapprox(dot(Lx, y), dot(x, x̄); rtol=1e-12)
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

        #------------------------------------------------------------# Slice 2b

        # The middle slab is the first place a coefficient slice can be right at
        # one seam and wrong at the other. Coefficient varies along the cut.
        mp_coeff(x) = 1.5 + x[2] + 0.3 * x[1] * x[2] + 0.2 * x[2]^2

        @testset "3-partition Field coefficient forward parity" begin
            rng = Random.MersenneTwister(71)
            for cutbc in ((Dirichlet(), Neumann()), (Periodic(), Periodic()))
                g = mp_grid(cutbc)
                n = prod(local_size(g))
                xflat = rand(rng, n)
                κ = set!(scalar_field(g), mp_coeff)
                for L in (
                    scaling(κ),
                    laplacian(g) * scaling(κ),
                    scaling(κ) * laplacian(g),
                    derivative(g, 1) * scaling(κ) * derivative(g, 1),
                )
                    P1 = prepare_distributed(L, 1)
                    y1 = MultiDeviceVector(zeros(n), P1.spec)
                    mul!(y1, P1, MultiDeviceVector(copy(xflat), P1.spec))

                    P3 = prepare_distributed(L, 3)
                    y3 = MultiDeviceVector(zeros(n), P3.spec)
                    mul!(y3, P3, MultiDeviceVector(copy(xflat), P3.spec))
                    @test gather(y3) == gather(y1)
                end
            end
        end

        @testset "3-partition Field coefficient adjoint identity" begin
            rng = Random.MersenneTwister(73)
            for cutbc in ((Dirichlet(), Neumann()), (Periodic(), Periodic()))
                g = mp_grid(cutbc)
                n = prod(local_size(g))
                κ = set!(scalar_field(g), mp_coeff)
                for L in (scaling(κ) * laplacian(g), laplacian(g) * scaling(κ))
                    P = prepare_distributed(L, 3)
                    x = MultiDeviceVector(rand(rng, n), P.spec)
                    y = MultiDeviceVector(rand(rng, n), P.spec)
                    Lx = MultiDeviceVector(zeros(n), P.spec)
                    mul!(Lx, P, x)
                    x̄ = MultiDeviceVector(zeros(n), P.spec)
                    MDLA_EXT_MP._mul_adjoint!(x̄, P, y)
                    @test isapprox(dot(Lx, y), dot(x, x̄); rtol=1e-12)
                end
            end
        end

        # The middle slab lifts ONLY through its transverse physical faces — its
        # cut faces are both Interface. A lift that leaked a cut-dimension BC value
        # onto an Interface face shows up here and nowhere with 2 partitions.
        @testset "3-partition boundary_rhs parity" begin
            for cut in ((Dirichlet(2.5), Neumann(0.4)), (Periodic(), Periodic()))
                g = CartesianGrid(
                    ((0.0, 2π), (0.0, 1.0)), (16, 18);
                    bc=((Dirichlet(0.75), Neumann(-1.25)), cut),
                )
                κ = set!(scalar_field(g), mp_coeff)
                D1 = derivative(g, 1)
                for L in (
                    laplacian(g),
                    laplacian(g) * laplacian(g),
                    scaling(κ) * laplacian(g),
                    adjoint(D1) + laplacian(g),
                    laplacian(g) + adjoint(D1),
                )
                    P3 = prepare_distributed(L, 3)
                    @test gather(boundary_rhs(P3)) == flatten(boundary_rhs(L, g))
                end
            end
        end

        @testset "3-partition inhomogeneous RHS assembled distributed" begin
            g = CartesianGrid(
                ((0.3, 1.7), (-1.1, 2.9)), (24, 26);
                bc=((Dirichlet(0.5), Dirichlet(-0.25)), (Dirichlet(1.0), Dirichlet(-0.5))),
            )
            L = -1.0 * laplacian(g)
            fun = x -> sin(3x[1]) * exp(-x[2]) + 0.25x[1] * x[2]
            bflat = flatten(set!(scalar_field(g), fun)) .- flatten(boundary_rhs(L, g))
            u_cpu, stats_cpu = Krylov.cg(prepare(L), bflat; atol=1e-10, rtol=1e-10)
            @test stats_cpu.solved

            niters = Int[]
            for nd in (1, 3)
                P = prepare_distributed(L, nd)
                b = assemble_rhs(P, fun)
                @test gather(b) == bflat
                u, stats = Krylov.cg(P, b; atol=1e-10, rtol=1e-10)
                @test stats.solved
                @test isapprox(gather(u), u_cpu; rtol=1e-8)
                push!(niters, stats.niter)
            end
            @test allequal(niters)
        end

        #------------------------------------------------------------# Slice 2c

        # The diffusion leaf reads κ across BOTH cut faces of the middle slab, each
        # owned by a different neighbour — the one configuration where a padded
        # window right at one seam and wrong at the other is visible.
        @testset "3-partition diffusion coefficient upload" begin
            for cutbc in ((Dirichlet(), Neumann()), (Periodic(), Periodic()))
                g = mp_grid(cutbc)
                κ = set!(scalar_field(g), mp_coeff)
                Dg = diffusion(g, κ)
                P = prepare_distributed(laplacian(g) * Dg, 3)
                locals = partition_grid(g, 3)
                for d in 1:3
                    # Composed(laplacian, diffusion) ⇒ the inner factor `b` is the leaf
                    Dd = P.tree.b.ops[d]
                    @test Dd isa MatrixFreeOperators.Diffusion
                    @test Dd.κ.data isa CuArray
                    @test Array(Dd.κ.data) ==
                        MatrixFreeOperators._slab_field(Dg.κ, locals[d]).data
                    # every ghost carries a value: a neighbour's κ at each cut, the
                    # even mirror or the wrap at a physical face
                    @test !any(iszero, Array(Dd.κ.data))
                end
            end
        end

        @testset "3-partition diffusion forward parity" begin
            rng = Random.MersenneTwister(79)
            for cutbc in ((Dirichlet(), Neumann()), (Periodic(), Periodic()))
                g = mp_grid(cutbc)
                n = prod(local_size(g))
                xflat = rand(rng, n)
                κ = set!(scalar_field(g), mp_coeff)
                for L in (
                    diffusion(g, κ),
                    diffusion(g, κ; averaging=HarmonicMean()),
                    laplacian(g) * diffusion(g, κ),
                    2.0 * diffusion(g, κ) + identity_op(),
                )
                    P1 = prepare_distributed(L, 1)
                    y1 = MultiDeviceVector(zeros(n), P1.spec)
                    mul!(y1, P1, MultiDeviceVector(copy(xflat), P1.spec))

                    P3 = prepare_distributed(L, 3)
                    y3 = MultiDeviceVector(zeros(n), P3.spec)
                    mul!(y3, P3, MultiDeviceVector(copy(xflat), P3.spec))
                    @test gather(y3) == gather(y1)
                end
            end
        end

        @testset "3-partition diffusion adjoint identity" begin
            rng = Random.MersenneTwister(83)
            for cutbc in ((Dirichlet(), Neumann()), (Periodic(), Periodic()))
                g = mp_grid(cutbc)
                n = prod(local_size(g))
                κ = set!(scalar_field(g), mp_coeff)
                for L in (diffusion(g, κ), laplacian(g) * diffusion(g, κ))
                    P = prepare_distributed(L, 3)
                    x = MultiDeviceVector(rand(rng, n), P.spec)
                    y = MultiDeviceVector(rand(rng, n), P.spec)
                    Lx = MultiDeviceVector(zeros(n), P.spec)
                    mul!(Lx, P, x)
                    x̄ = MultiDeviceVector(zeros(n), P.spec)
                    MDLA_EXT_MP._mul_adjoint!(x̄, P, y)
                    @test isapprox(dot(Lx, y), dot(x, x̄); rtol=1e-12)
                    if L isa MatrixFreeOperators.Diffusion
                        # A real κ makes the leaf its own transpose, so the middle
                        # slab's two-owner gather must reproduce the forward action.
                        Ly = MultiDeviceVector(zeros(n), P.spec)
                        mul!(Ly, P, y)
                        @test isapprox(gather(x̄), gather(Ly); rtol=1e-12)
                    end
                end
            end
        end
    else
        @test_skip "MDLA 3-partition — needs ≥ 3 CUDA devices"
    end
end
