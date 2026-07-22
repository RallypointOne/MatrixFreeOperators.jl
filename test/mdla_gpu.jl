# Included by mdla.jl only when MFO_TEST_MDLA=true with CUDA.jl and
# MultiDeviceLinearAlgebra.jl available (dev environment — neither lives in
# test/Project.toml). Multi-partition testsets need ≥ 2 physical GPUs (MDLA
# enforces unique device IDs per partition) and are skipped below that.
using CUDA
using MultiDeviceLinearAlgebra

const MDLA_EXT = Base.get_extension(MatrixFreeOperators, :MatrixFreeOperatorsMDLAExt)
NGPUS_MDLA = length(CUDA.devices())

function mdla_grid(cutbc)
    return CartesianGrid(
        ((0.0, 2π), (0.0, 1.0)), (16, 18);
        bc=((Periodic(), Periodic()), cutbc),
    )
end

@testset "extension resolves" begin
    @test MDLA_EXT !== nothing
end

@testset "guards" begin
    g = mdla_grid((Dirichlet(), Dirichlet()))
    κ = set!(scalar_field(g), x -> 1 + x[1] / 7)
    @test_throws ArgumentError prepare_distributed(laplacian(g) * laplacian(g), 1)
    @test_throws ArgumentError prepare_distributed(adjoint(derivative(g, 1)), 1)
    @test_throws ArgumentError prepare_distributed(MatrixFreeOperators.gradient(g), 1)
    @test_throws ArgumentError prepare_distributed(scaling(κ) + laplacian(g), 1)
    @test_throws ArgumentError prepare_distributed(laplacian(g), NGPUS_MDLA + 1)
end

@testset "single-partition parity vs single-GPU prepare" begin
    rng = Random.MersenneTwister(31)
    for cutbc in ((Dirichlet(), Neumann()), (Periodic(), Periodic()))
        g = mdla_grid(cutbc)
        L = laplacian(g)
        n = prod(local_size(g))
        xflat = rand(rng, n)

        gg = Adapt.adapt(CuArray, g)
        Pg = prepare(L, scalar_field(gg))
        y_ref = CUDA.zeros(Float64, n)
        mul!(y_ref, Pg, CuVector(xflat))

        P1 = prepare_distributed(L, 1)
        @test size(P1) == (n, n) && eltype(P1) == Float64
        x1 = MultiDeviceVector(xflat, P1.spec)
        y1 = MultiDeviceVector(zeros(n), P1.spec)
        mul!(y1, P1, x1)
        @test gather(y1) == Array(y_ref)

        y0 = rand(rng, n)
        yαβ = MultiDeviceVector(copy(y0), P1.spec)
        mul!(yαβ, P1, x1, 2.5, 0.5)
        @test gather(yαβ) ≈ 2.5 .* Array(y_ref) .+ 0.5 .* y0
    end
end

@testset "2-partition forward parity" begin
    if NGPUS_MDLA >= 2
        rng = Random.MersenneTwister(37)
        for cutbc in ((Dirichlet(), Neumann()), (Periodic(), Periodic()))
            g = mdla_grid(cutbc)
            L = laplacian(g)
            n = prod(local_size(g))
            xflat = rand(rng, n)

            P1 = prepare_distributed(L, 1)
            y1 = MultiDeviceVector(zeros(n), P1.spec)
            mul!(y1, P1, MultiDeviceVector(xflat, P1.spec))

            P2 = prepare_distributed(L, 2)
            y2 = MultiDeviceVector(zeros(n), P2.spec)
            mul!(y2, P2, MultiDeviceVector(xflat, P2.spec))
            @test gather(y2) == gather(y1)
        end
    else
        @test_skip "2-partition forward parity — needs ≥ 2 CUDA devices"
    end
end

@testset "distributed adjoint: identity and transpose structure" begin
    if NGPUS_MDLA >= 2
        rng = Random.MersenneTwister(41)
        for cutbc in ((Dirichlet(), Neumann()), (Periodic(), Periodic()))
            g = mdla_grid(cutbc)
            n = prod(local_size(g))
            for L in (laplacian(g), 0.5 * laplacian(g) + 2.0 * identity_op())
                P = prepare_distributed(L, 2)
                xflat = rand(rng, n)
                yflat = rand(rng, n)
                x = MultiDeviceVector(xflat, P.spec)
                y = MultiDeviceVector(yflat, P.spec)
                Lx = MultiDeviceVector(zeros(n), P.spec)
                mul!(Lx, P, x)
                x̄ = MultiDeviceVector(zeros(n), P.spec)
                MDLA_EXT._mul_adjoint!(x̄, P, y)
                @test isapprox(dot(Lx, y), dot(x, x̄); rtol=1e-12)

                # symmetric BCs ⇒ globally self-adjoint: Lᵀy must match Ly
                Ly = MultiDeviceVector(zeros(n), P.spec)
                mul!(Ly, P, MultiDeviceVector(yflat, P.spec))
                @test isapprox(gather(x̄), gather(Ly); rtol=1e-12)

                # CPU global adjoint parity
                ȳg = scalar_field(g)
                flat_to_interior!(ȳg, yflat)
                x̄g = scalar_field(g)
                apply_adjoint!(x̄g, L, ȳg, g)
                @test isapprox(gather(x̄), flatten(x̄g); rtol=1e-12)
            end
        end

        # dense transpose structure on a tiny grid, on real scatter!/reduce!
        gt = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (3, 4);
            bc=((Dirichlet(), Dirichlet()), (Dirichlet(), Neumann())),
        )
        Lt = laplacian(gt)
        Pt = prepare_distributed(Lt, 2)
        nt = prod(local_size(gt))
        A_fwd = zeros(nt, nt)
        A_adj = zeros(nt, nt)
        e = zeros(nt)
        for j in 1:nt
            fill!(e, 0)
            e[j] = 1
            col = MultiDeviceVector(zeros(nt), Pt.spec)
            mul!(col, Pt, MultiDeviceVector(copy(e), Pt.spec))
            A_fwd[:, j] .= gather(col)
            colᵀ = MultiDeviceVector(zeros(nt), Pt.spec)
            MDLA_EXT._mul_adjoint!(colᵀ, Pt, MultiDeviceVector(copy(e), Pt.spec))
            A_adj[:, j] .= gather(colᵀ)
        end
        @test A_fwd == materialize(prepare(Lt))
        @test A_adj ≈ A_fwd'
    else
        @test_skip "distributed adjoint — needs ≥ 2 CUDA devices"
    end
end

@testset "Krylov.cg on distributed Poisson" begin
    g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (24, 26))
    L = -1.0 * laplacian(g)   # SPD under homogeneous Dirichlet
    f = set!(scalar_field(g), x -> sin(π * x[1]) * sin(π * x[2]))
    bflat = flatten(f)
    n = length(bflat)

    u_cpu, stats_cpu = Krylov.cg(prepare(L), bflat; atol=1e-10, rtol=1e-10)
    @test stats_cpu.solved

    niters = Int[]
    for nd in 1:min(NGPUS_MDLA, 2)
        P = prepare_distributed(L, nd)
        b = MultiDeviceVector(copy(bflat), P.spec)
        u, stats = Krylov.cg(P, b; atol=1e-10, rtol=1e-10)
        @test stats.solved
        @test isapprox(gather(u), u_cpu; rtol=1e-8)
        push!(niters, stats.niter)
    end
    @test allequal(niters)
    if NGPUS_MDLA < 2
        @test_skip "2-partition CG parity — needs ≥ 2 CUDA devices"
    end
end
