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

# Composed and AdjointOp used to be rejected here; slice 2a implements the
# mid-tree exchange and reduction behind those guards, so they now go through —
# see the composed/adjoint testsets below. What remains rejected is rejected
# because the failure mode would be a silently wrong answer, not an error.
@testset "guards" begin
    g = mdla_grid((Dirichlet(), Dirichlet()))
    gc = coarsen(g)
    v = set!(vector_field(g), x -> SVector(1.0, 0.0))
    @test_throws ArgumentError prepare_distributed(MatrixFreeOperators.gradient(g), 1)
    @test_throws ArgumentError prepare_distributed(MatrixFreeOperators.divergence(g), 1)
    @test_throws ArgumentError prepare_distributed(advection(g, v) + laplacian(g), 1)
    @test_throws ArgumentError prepare_distributed(restriction(g, gc), 1)
    # both factors distributable, but on different grids
    @test_throws ArgumentError prepare_distributed(laplacian(gc) * restriction(g, gc), 1)
    @test_throws ArgumentError prepare_distributed(laplacian(g), NGPUS_MDLA + 1)
    # a coefficient on some OTHER grid: each leaf is fine alone, only the tree shows it
    κc = set!(scalar_field(gc), x -> 1 + x[1] / 7)
    @test_throws ArgumentError prepare_distributed(laplacian(g) + scaling(κc), 1)
end

@testset "newly distributable operators are accepted" begin
    g = mdla_grid((Dirichlet(), Dirichlet()))
    κ = set!(scalar_field(g), x -> 1 + x[1] / 7)
    for L in (
        laplacian(g) * laplacian(g),
        adjoint(derivative(g, 1)),
        adjoint(derivative(g, 1)) + laplacian(g),
        MatrixFreeOperators.AdjointOp(derivative(g, 1) * derivative(g, 2)),
        # slice 2b
        scaling(κ),
        scaling(κ) + laplacian(g),
        laplacian(g) * scaling(κ),
        adjoint(derivative(g, 1) * scaling(κ)),
    )
        @test prepare_distributed(L, 1) isa MDLA_EXT.MDLAPreparedOperator
    end
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

#--------------------------------------------------------------------------------# Slice 2a: mid-tree exchange and reduction

# Helper: dense forward/adjoint matrices through the distributed path.
function mdla_materialize(P, n)
    A_fwd, A_adj, e = zeros(n, n), zeros(n, n), zeros(n)
    for j in 1:n
        fill!(e, 0)
        e[j] = 1
        col = MultiDeviceVector(zeros(n), P.spec)
        mul!(col, P, MultiDeviceVector(copy(e), P.spec))
        A_fwd[:, j] .= gather(col)
        colᵀ = MultiDeviceVector(zeros(n), P.spec)
        MDLA_EXT._mul_adjoint!(colᵀ, P, MultiDeviceVector(copy(e), P.spec))
        A_adj[:, j] .= gather(colᵀ)
    end
    return A_fwd, A_adj
end

# The mid-tree exchange is the whole point of slice 2a: without it the
# intermediate's cut-plane ghosts are stale zeros and this parity fails near
# every seam. Bitwise, like the slice-1 forward parity — same device arithmetic.
@testset "composed: 2-partition forward parity" begin
    if NGPUS_MDLA >= 2
        rng = Random.MersenneTwister(43)
        for cutbc in ((Dirichlet(), Neumann()), (Periodic(), Periodic()))
            g = mdla_grid(cutbc)
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

                P2 = prepare_distributed(L, 2)
                y2 = MultiDeviceVector(zeros(n), P2.spec)
                mul!(y2, P2, MultiDeviceVector(copy(xflat), P2.spec))
                @test gather(y2) == gather(y1)
            end
        end
    else
        @test_skip "composed forward parity — needs ≥ 2 CUDA devices"
    end
end

@testset "composed: adjoint identity and transpose structure" begin
    if NGPUS_MDLA >= 2
        rng = Random.MersenneTwister(47)
        for cutbc in ((Dirichlet(), Neumann()), (Periodic(), Periodic()))
            g = mdla_grid(cutbc)
            n = prod(local_size(g))
            for L in (laplacian(g) * laplacian(g), derivative(g, 1) * derivative(g, 2))
                P = prepare_distributed(L, 2)
                x = MultiDeviceVector(rand(rng, n), P.spec)
                y = MultiDeviceVector(rand(rng, n), P.spec)
                Lx = MultiDeviceVector(zeros(n), P.spec)
                mul!(Lx, P, x)
                x̄ = MultiDeviceVector(zeros(n), P.spec)
                MDLA_EXT._mul_adjoint!(x̄, P, y)
                @test isapprox(dot(Lx, y), dot(x, x̄); rtol=1e-12)
            end
        end

        # The literal statement of the transpose argument, on real scatter!/reduce!
        # with a mid-tree exchange in the middle of it.
        gt = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (3, 6);
            bc=((Dirichlet(), Dirichlet()), (Dirichlet(), Neumann())),
        )
        nt = prod(local_size(gt))
        for Lt in (laplacian(gt) * laplacian(gt), adjoint(derivative(gt, 2)) * derivative(gt, 2))
            A_fwd, A_adj = mdla_materialize(prepare_distributed(Lt, 2), nt)
            @test A_fwd ≈ materialize(prepare(MatrixFreeOperators._push_adjoints(Lt)))
            @test A_adj ≈ A_fwd'
        end
    else
        @test_skip "composed adjoint — needs ≥ 2 CUDA devices"
    end
end

@testset "AdjointOp as a distributed forward operator" begin
    if NGPUS_MDLA >= 2
        rng = Random.MersenneTwister(53)
        g = mdla_grid((Dirichlet(), Neumann()))
        n = prod(local_size(g))
        xflat = rand(rng, n)
        L = adjoint(derivative(g, 1))

        # CPU reference on the undivided grid
        xg = flat_to_interior!(scalar_field(g), xflat)
        ref = flatten(apply(L, xg))

        P2 = prepare_distributed(L, 2)
        y2 = MultiDeviceVector(zeros(n), P2.spec)
        mul!(y2, P2, MultiDeviceVector(copy(xflat), P2.spec))
        @test isapprox(gather(y2), ref; rtol=1e-12)

        # dense transpose structure through the node's mid-tree reduction
        gt = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (3, 6);
            bc=((Dirichlet(), Dirichlet()), (Dirichlet(), Neumann())),
        )
        nt = prod(local_size(gt))
        A_fwd, A_adj = mdla_materialize(prepare_distributed(adjoint(derivative(gt, 1)), 2), nt)
        @test A_fwd ≈ materialize(prepare(derivative(gt, 1)))'
        @test A_adj ≈ A_fwd'
    else
        @test_skip "AdjointOp forward — needs ≥ 2 CUDA devices"
    end
end

# The adjoint node's gather zeroes its input's ghosts, so it works on its own
# copy. Without that, the exchanged ghosts the Laplacian sibling needs would be
# destroyed whenever the adjoint term comes first — silently, and only in one
# term order.
@testset "adjoint sibling under Added: both term orders agree" begin
    if NGPUS_MDLA >= 2
        rng = Random.MersenneTwister(59)
        g = mdla_grid((Dirichlet(), Neumann()))
        n = prod(local_size(g))
        xflat = rand(rng, n)
        A = adjoint(derivative(g, 1))
        B = laplacian(g)
        ref = flatten(apply(A + B, flat_to_interior!(scalar_field(g), xflat)))
        results = map((A + B, B + A)) do L
            P = prepare_distributed(L, 2)
            y = MultiDeviceVector(zeros(n), P.spec)
            mul!(y, P, MultiDeviceVector(copy(xflat), P.spec))
            gather(y)
        end
        @test isapprox(results[1], ref; rtol=1e-12)
        @test isapprox(results[2], ref; rtol=1e-12)
        @test isapprox(results[1], results[2]; rtol=1e-12)
    else
        @test_skip "adjoint sibling ordering — needs ≥ 2 CUDA devices"
    end
end

# DᵀD + I is SPD, so it is a valid CG system that also exercises both a mid-tree
# exchange and a mid-tree reduction inside every matvec.
@testset "Krylov.cg on a distributed composed system" begin
    g = CartesianGrid(
        ((0.0, 1.0), (0.0, 1.0)), (24, 26);
        bc=((Dirichlet(), Dirichlet()), (Dirichlet(), Dirichlet())),
    )
    L = adjoint(derivative(g, 2)) * derivative(g, 2) + 1.0 * identity_op()
    f = set!(scalar_field(g), x -> sin(π * x[1]) * sin(π * x[2]))
    bflat = flatten(f)

    u_cpu, stats_cpu = Krylov.cg(
        prepare(MatrixFreeOperators._push_adjoints(L)), bflat; atol=1e-10, rtol=1e-10
    )
    @test stats_cpu.solved

    niters = Int[]
    for nd in 1:min(NGPUS_MDLA, 2)
        P = prepare_distributed(L, nd)
        u, stats = Krylov.cg(P, MultiDeviceVector(copy(bflat), P.spec); atol=1e-10, rtol=1e-10)
        @test stats.solved
        @test isapprox(gather(u), u_cpu; rtol=1e-8)
        push!(niters, stats.niter)
    end
    @test allequal(niters)
    if NGPUS_MDLA < 2
        @test_skip "2-partition composed CG parity — needs ≥ 2 CUDA devices"
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

#--------------------------------------------------------------------------------# Slice 2b: Field coefficients, boundary lift, distributed RHS

# The coefficient VARIES ALONG THE CUT DIMENSION (dim 2) and is asymmetric about
# it. A coefficient constant in the cut dimension could not catch a slice shifted
# by the halo width, which is the likeliest way `_slab_field` goes wrong.
mdla_coeff(x) = 1.5 + x[2] + 0.3 * x[1] * x[2] + 0.2 * x[2]^2

# `prepare_distributed` slices the coefficient on the host and uploads the slab.
# If it ever uploaded the global field instead, the shapes would still broadcast on
# partition 1 and only the later partitions would be wrong — so assert the shape.
@testset "the coefficient is sliced and uploaded per partition" begin
    if NGPUS_MDLA >= 2
        g = mdla_grid((Dirichlet(), Neumann()))
        κ = set!(scalar_field(g), mdla_coeff)
        P = prepare_distributed(laplacian(g) * scaling(κ), 2)
        locals = partition_grid(g, 2)
        for d in 1:2
            # Composed(laplacian, scaling) ⇒ the inner factor `b` is the scaling
            Sd = P.tree.b.ops[d]
            @test Sd isa MatrixFreeOperators.ScalingOp
            κd = Sd.coeff
            @test κd.data isa CuArray
            @test size(κd.data) == MatrixFreeOperators.padded_size(locals[d])
            @test Array(interior(κd)) == collect(view(interior(κ), locals[d].local_range...))
        end
    else
        @test_skip "per-partition coefficient upload — needs ≥ 2 CUDA devices"
    end
end

@testset "Field coefficient: 2-partition forward parity" begin
    if NGPUS_MDLA >= 2
        rng = Random.MersenneTwister(61)
        for cutbc in ((Dirichlet(), Neumann()), (Periodic(), Periodic()))
            g = mdla_grid(cutbc)
            n = prod(local_size(g))
            xflat = rand(rng, n)
            κ = set!(scalar_field(g), mdla_coeff)
            for L in (
                scaling(κ),
                laplacian(g) * scaling(κ),
                scaling(κ) * laplacian(g),
                2.0 * scaling(κ) + laplacian(g),
                derivative(g, 1) * scaling(κ) * derivative(g, 1),
            )
                P1 = prepare_distributed(L, 1)
                y1 = MultiDeviceVector(zeros(n), P1.spec)
                mul!(y1, P1, MultiDeviceVector(copy(xflat), P1.spec))

                P2 = prepare_distributed(L, 2)
                y2 = MultiDeviceVector(zeros(n), P2.spec)
                mul!(y2, P2, MultiDeviceVector(copy(xflat), P2.spec))
                @test gather(y2) == gather(y1)
            end
        end
    else
        @test_skip "Field coefficient forward parity — needs ≥ 2 CUDA devices"
    end
end

@testset "Field coefficient: adjoint identity and transpose structure" begin
    if NGPUS_MDLA >= 2
        rng = Random.MersenneTwister(67)
        g = mdla_grid((Dirichlet(), Neumann()))
        n = prod(local_size(g))
        κ = set!(scalar_field(g), mdla_coeff)
        x, y = rand(rng, n), rand(rng, n)
        for L in (scaling(κ) * laplacian(g), laplacian(g) * scaling(κ))
            P = prepare_distributed(L, 2)
            Lx = MultiDeviceVector(zeros(n), P.spec)
            mul!(Lx, P, MultiDeviceVector(copy(x), P.spec))
            Ly = MultiDeviceVector(zeros(n), P.spec)
            MDLA_EXT._mul_adjoint!(Ly, P, MultiDeviceVector(copy(y), P.spec))
            @test dot(gather(Lx), y) ≈ dot(x, gather(Ly)) rtol = 1e-12
        end

        # Dense structure on a small grid, against the single-device reference.
        gs = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (3, 6);
            bc=((Dirichlet(), Dirichlet()), (Dirichlet(), Neumann())))
        ns = prod(local_size(gs))
        κs = set!(scalar_field(gs), mdla_coeff)
        for Lt in (scaling(κs) * laplacian(gs), adjoint(derivative(gs, 1) * scaling(κs)))
            P = prepare_distributed(Lt, 2)
            A_fwd, A_adj = mdla_materialize(P, ns)
            @test A_fwd ≈ materialize(prepare(MatrixFreeOperators._push_adjoints(Lt)))
            @test A_adj ≈ A_fwd'
        end
    else
        @test_skip "Field coefficient adjoint — needs ≥ 2 CUDA devices"
    end
end

# The lift assembles per slab: Interface faces contribute nothing, so only the end
# slabs lift through the cut dimension. A different value on every face, so a slab
# that applied a cut BC to its Interface face cannot pass.
@testset "distributed boundary_rhs parity" begin
    for cut in ((Dirichlet(2.5), Neumann(0.4)), (Periodic(), Periodic()))
        g = CartesianGrid(
            ((0.0, 2π), (0.0, 1.0)), (16, 18);
            bc=((Dirichlet(0.75), Neumann(-1.25)), cut),
        )
        κ = set!(scalar_field(g), mdla_coeff)
        D1 = derivative(g, 1)
        for L in (
            laplacian(g),
            -1.5 * laplacian(g),
            laplacian(g) * laplacian(g),
            scaling(κ) * laplacian(g),
            adjoint(D1) + laplacian(g),
            laplacian(g) + adjoint(D1),
        )
            ref = flatten(boundary_rhs(L, g))
            for nd in 1:min(NGPUS_MDLA, 2)
                P = prepare_distributed(L, nd)
                @test gather(boundary_rhs(P)) == ref
            end
        end
    end
    if NGPUS_MDLA < 2
        @test_skip "2-partition boundary_rhs parity — needs ≥ 2 CUDA devices"
    end
end

# The whole point of the slice: assemble `f - b` without a global-sized array on
# any one device, and get the same system — hence the same iteration count — the
# single-device path assembles globally. An iteration count that moves with the
# partition count means the RHS is not partition-independent.
@testset "Krylov.cg with an inhomogeneous RHS assembled distributed" begin
    g = CartesianGrid(
        ((0.0, 1.0), (0.0, 1.0)), (24, 26);
        bc=((Dirichlet(0.5), Dirichlet(-0.25)), (Dirichlet(1.0), Dirichlet(-0.5))),
    )
    κ = set!(scalar_field(g), mdla_coeff)
    fun = x -> sin(π * x[1]) * sin(π * x[2]) + 0.3x[2]
    for L in (-1.0 * laplacian(g), -1.0 * (scaling(κ) * laplacian(g)))
        bflat = flatten(set!(scalar_field(g), fun)) .- flatten(boundary_rhs(L, g))
        u_cpu, stats_cpu = Krylov.cg(prepare(L), bflat; atol=1e-10, rtol=1e-10)
        @test stats_cpu.solved

        niters = Int[]
        for nd in 1:min(NGPUS_MDLA, 2)
            P = prepare_distributed(L, nd)
            b = assemble_rhs(P, fun)
            @test gather(b) == bflat            # bitwise, thanks to global-index cell_center
            u, stats = Krylov.cg(P, b; atol=1e-10, rtol=1e-10)
            @test stats.solved
            @test isapprox(gather(u), u_cpu; rtol=1e-8)
            push!(niters, stats.niter)
        end
        @test allequal(niters)
    end
    if NGPUS_MDLA < 2
        @test_skip "2-partition inhomogeneous CG parity — needs ≥ 2 CUDA devices"
    end
end

# The escape hatch: build the source term yourself on the grids local_grids
# reports, and hand the fields to assemble_rhs.
@testset "assemble_rhs from per-partition fields" begin
    if NGPUS_MDLA >= 2
        g = CartesianGrid(
            ((0.3, 1.7), (-1.1, 2.9)), (16, 18);
            bc=((Dirichlet(0.75), Neumann(-1.25)), (Dirichlet(2.5), Neumann(0.4))),
        )
        L = laplacian(g)
        fun = x -> sin(3x[1]) * exp(-x[2]) + 0.25x[1] * x[2]
        ref = flatten(set!(scalar_field(g), fun)) .- flatten(boundary_rhs(L, g))
        P = prepare_distributed(L, 2)
        grids = local_grids(P)
        @test length(grids) == 2
        fields = [set!(scalar_field(lg), fun) for lg in grids]
        @test gather(assemble_rhs(P, fields)) == ref
        @test_throws ArgumentError assemble_rhs(P, fields[1:1])
    else
        @test_skip "assemble_rhs from fields — needs ≥ 2 CUDA devices"
    end
end
