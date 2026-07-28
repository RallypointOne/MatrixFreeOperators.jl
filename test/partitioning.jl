# Emulation of the MDLA exchange semantics on plain Vectors: ghost sections are
# filled by direct global indexing (definitionally what scatter! delivers) and
# the adjoint reduction runs the two-phase reduce!(+) contract — every owner
# copy first, then all ghost contributions accumulated. This proves the
# distributed apply and its adjoint CPU-only, before any GPU code exists.

function dist_setup(g, nparts)
    parts = partition_grid(g, nparts)
    ghost_globals, plans = MatrixFreeOperators._slab_ghost_layout(g, parts)
    return parts, ghost_globals, plans
end

owned_flat_range(g, lg) = MatrixFreeOperators._owned_flat_range(g, lg)

# The plane-view geometry is core's (src/partitioning.jl), shared verbatim with
# the MDLA extension — re-deriving it here would let the two silently decouple.
halo_plane_view(f::Field, plane::Int) = MatrixFreeOperators._halo_plane_view(f, plane)

function dist_apply_emulated(L, g, parts, ghost_globals, plans, xflat)
    T = eltype(xflat)
    yflat = similar(xflat)
    for (p, lg) in enumerate(parts)
        xpad = scalar_field(lg, T)
        ypad = scalar_field(lg, T)
        owned = owned_flat_range(g, lg)
        flat_to_interior!(xpad, view(xflat, owned))
        ghost = xflat[ghost_globals[p]]
        for (rng, plane) in plans[p]
            dst = halo_plane_view(xpad, plane)
            dst .= reshape(view(ghost, rng), size(dst))
        end
        apply!(ypad, L, xpad, lg)
        interior_to_flat!(view(yflat, owned), ypad)
    end
    return yflat
end

function dist_adjoint_emulated(L, g, parts, ghost_globals, plans, ȳflat)
    T = eltype(ȳflat)
    x̄flat = zeros(T, length(ȳflat))
    contribs = Vector{Vector{T}}(undef, length(parts))
    for (p, lg) in enumerate(parts)
        ȳpad = scalar_field(lg, T)
        x̄pad = scalar_field(lg, T)
        flat_to_interior!(ȳpad, view(ȳflat, owned_flat_range(g, lg)))
        apply_adjoint!(x̄pad, L, ȳpad, lg)
        interior_to_flat!(view(x̄flat, owned_flat_range(g, lg)), x̄pad)
        contrib = Vector{T}(undef, length(ghost_globals[p]))
        for (rng, plane) in plans[p]
            contrib[rng] .= vec(halo_plane_view(x̄pad, plane))
        end
        contribs[p] = contrib
    end
    for p in eachindex(parts)
        x̄flat[ghost_globals[p]] .+= contribs[p]
    end
    return x̄flat
end

@testset "Grid partitioning" begin
    @testset "slab geometry" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 3.0)), (5, 8);
            bc=((Periodic(), Periodic()), (Dirichlet(), Neumann())),
        )
        parts = partition_grid(g, 3)
        @test length(parts) == 3
        @test [local_size(p)[2] for p in parts] == [3, 3, 2]
        @test all(p -> local_size(p)[1] == 5, parts)
        @test all(p -> spacing(p) == spacing(g), parts)
        @test all(p -> halo_width(p) == halo_width(g), parts)
        @test parts[1].extent[2][1] == g.extent[2][1]
        @test parts[3].extent[2][2] == g.extent[2][2]
        @test parts[1].extent[2][2] == parts[2].extent[2][1]
        @test [p.local_range[2] for p in parts] == [1:3, 4:6, 7:8]
        @test all(p -> p.local_range[1] == 1:5, parts)
        Interface = MatrixFreeOperators.Interface
        @test boundary_conditions(parts[1])[2] isa Tuple{Dirichlet,Interface}
        @test boundary_conditions(parts[2])[2] isa Tuple{Interface,Interface}
        @test boundary_conditions(parts[3])[2] isa Tuple{Interface,Neumann}
        @test all(p -> boundary_conditions(p)[1] isa Tuple{Periodic,Periodic}, parts)

        gp = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (4, 8);
            bc=((Dirichlet(), Dirichlet()), (Periodic(), Periodic())))
        pparts = partition_grid(gp, 2)
        @test all(p -> boundary_conditions(p)[2] isa Tuple{Interface,Interface}, pparts)

        @test partition_grid(g, 1) == [g] && partition_grid(g, 1)[1] === g
    end

    @testset "validation" begin
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (4, 4))
        @test_throws ArgumentError partition_grid(g, 0)
        @test_throws ArgumentError partition_grid(g, 5)
        gp = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (4, 6);
            bc=((Dirichlet(), Dirichlet()), (Periodic(), Periodic())), halo=(1, 2))
        @test_throws ArgumentError partition_grid(gp, 2)  # periodic slabs need ≥ 2h planes
        # ...but the same grid cut three ways is fine: each slab holds exactly h
        # planes, and its low and high ghosts now come from distinct owners — so
        # the requests stay duplicate-free, which is all the 2h rule protected
        gp3, gp3gg, _ = dist_setup(gp, 3)
        @test [local_size(p)[2] for p in gp3] == [2, 2, 2]
        @test all(p -> allunique(gp3gg[p]), 1:3)
        @test all(p -> isempty(intersect(gp3gg[p], owned_flat_range(gp, gp3[p]))), 1:3)
        gd = CartesianGrid{2,Float64,typeof(g.bc),typeof(g.device),Symbol}(
            g.extent, g.spacing, g.size, g.halo, g.bc, g.device, g.local_range, :topo
        )
        @test_throws ArgumentError partition_grid(gd, 2)
    end

    @testset "flat plane layout matches flatten" begin
        rng = Random.MersenneTwister(11)
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (3, 4))
        u = scalar_field(g)
        interior(u) .= rand(rng, local_size(g)...)
        uflat = flatten(u)
        ui = collect(interior(u))
        for z in 1:4
            r = MatrixFreeOperators._plane_flat_range(local_size(g), z, 1)
            @test uflat[r] == vec(ui[:, z])
        end
        v = vector_field(g)
        interior(v) .= SVector.(rand(rng, local_size(g)...), rand(rng, local_size(g)...))
        vflat = flatten(v)
        vi = collect(interior(v))
        for z in 1:4
            r = MatrixFreeOperators._plane_flat_range(local_size(g), z, 2)
            @test vflat[r] == collect(reinterpret(Float64, vec(vi[:, z])))
        end
    end

    @testset "ghost layout invariants ($(cutbc isa Periodic ? "periodic" : "physical") cut, h=$h)" for cutbc in (
            Dirichlet(), Periodic()
        ), h in (1, 2)
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (3, 12);
            bc=((Dirichlet(), Dirichlet()), (cutbc, cutbc isa Periodic ? Periodic() : Neumann())),
            halo=(1, h),
        )
        parts, gg, plans = dist_setup(g, 3)
        m = 3
        ntotal = prod(local_size(g))
        periodic = cutbc isa Periodic
        plane_of(i) = fld1(i, m)
        owner_of_plane(z) = findfirst(p -> z in parts[p].local_range[2], eachindex(parts))
        for p in 1:3
            nfaces = periodic ? 2 : (p == 1 || p == 3 ? 1 : 2)
            @test length(gg[p]) == nfaces * h * m
            @test all(i -> 1 <= i <= ntotal, gg[p])
            @test allunique(gg[p])
            @test isempty(intersect(gg[p], owned_flat_range(g, parts[p])))
            @test issorted(owner_of_plane.(plane_of.(gg[p])))
            @test first(plans[p][1][1]) == 1
            @test last(plans[p][end][1]) == length(gg[p])
            @test all(k -> first(plans[p][k][1]) == last(plans[p][k - 1][1]) + 1,
                2:length(plans[p]))
            len = local_size(parts[p])[2]
            expected_planes = Int[]
            (periodic || p > 1) && append!(expected_planes, 1:h)
            (periodic || p < 3) && append!(expected_planes, (h + len + 1):(h + len + h))
            @test sort(last.(plans[p])) == expected_planes
        end
        if periodic
            # partition 1's wrapped low ghosts (padded planes 1:h) come from the last slab
            lowidx = reduce(
                vcat, [collect(gg[1][rng]) for (rng, plane) in plans[1] if plane <= h]
            )
            @test all(i -> plane_of(i) in parts[3].local_range[2], lowidx)
        end
        g1parts, g1gg, g1plans = dist_setup(g, 1)
        @test isempty(g1gg[1]) && isempty(g1plans[1])
    end

    @testset "forward parity vs single grid ($(nameof(typeof(cutlo))) cut, nparts=$np)" for (cutlo, cuthi) in (
            (Dirichlet(), Neumann()), (Periodic(), Periodic())
        ), np in (2, 3)
        rng = Random.MersenneTwister(3)
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 2.0)), (4, 9);
            bc=((Periodic(), Periodic()), (cutlo, cuthi)),
        )
        L = laplacian(g)
        parts, gg, plans = dist_setup(g, np)
        xflat = rand(rng, prod(local_size(g)))
        xg = scalar_field(g)
        flat_to_interior!(xg, xflat)
        yref = flatten(apply(L, copy(xg)))
        @test dist_apply_emulated(L, g, parts, gg, plans, xflat) == yref
    end

    @testset "slab apply through the prepared mul! boundary" begin
        # Guards the core↔ext contract the MDLA extension leans on: mul! writes
        # only the interior, so Interface ghosts staged into P.xpad beforehand
        # survive the local sweep. A zero_ghosts! added to mul! breaks this.
        seed = Random.MersenneTwister(29)
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 2.0)), (4, 9);
            bc=((Periodic(), Periodic()), (Dirichlet(), Neumann())),
        )
        L = laplacian(g)
        parts, gg, plans = dist_setup(g, 3)
        xflat = rand(seed, prod(local_size(g)))
        xg = scalar_field(g)
        flat_to_interior!(xg, xflat)
        yref = flatten(apply(L, copy(xg)))

        # Stage the ghosts once per slab, then drive the flat Krylov boundary.
        function slab_mul!(out, α...)
            for (p, lg) in enumerate(parts)
                P = prepare(L, scalar_field(lg))
                owned = owned_flat_range(g, lg)
                ghost = xflat[gg[p]]
                for (sec, plane) in plans[p]
                    dst = halo_plane_view(P.xpad, plane)
                    dst .= reshape(view(ghost, sec), size(dst))
                end
                mul!(view(out, owned), P, view(xflat, owned), α...)
            end
            return out
        end

        @test slab_mul!(similar(xflat)) == yref

        y0 = rand(seed, length(xflat))
        @test slab_mul!(copy(y0), 2.5, 0.5) ≈ 2.5 .* yref .+ 0.5 .* y0
    end

    @testset "3-D forward parity" begin
        rng = Random.MersenneTwister(5)
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), (4, 3, 6);
            bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann()), (Periodic(), Periodic())),
        )
        L = laplacian(g)
        parts, gg, plans = dist_setup(g, 2)
        xflat = rand(rng, prod(local_size(g)))
        xg = scalar_field(g)
        flat_to_interior!(xg, xflat)
        @test dist_apply_emulated(L, g, parts, gg, plans, xflat) ==
            flatten(apply(L, copy(xg)))
    end

    @testset "adjoint identity ⟨Lx,y⟩ = ⟨x,Lᵀy⟩ across partitions ($(nameof(typeof(cutlo))) cut, nparts=$np)" for (cutlo, cuthi) in (
            (Dirichlet(), Neumann()), (Periodic(), Periodic())
        ), np in (2, 3)
        rng = Random.MersenneTwister(17)
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 2.0)), (4, 9);
            bc=((Periodic(), Periodic()), (cutlo, cuthi)),
        )
        parts, gg, plans = dist_setup(g, np)
        n = prod(local_size(g))
        for L in (laplacian(g), 0.5 * laplacian(g) + 2.0 * identity_op(), -1.5 * laplacian(g))
            xflat = rand(rng, n)
            yflat = rand(rng, n)
            Lx = dist_apply_emulated(L, g, parts, gg, plans, xflat)
            Lty = dist_adjoint_emulated(L, g, parts, gg, plans, yflat)
            @test isapprox(dot(Lx, yflat), dot(xflat, Lty); rtol=1e-13)
            # symmetric BCs ⇒ L is self-adjoint globally: the distributed adjoint
            # must reproduce the distributed forward action on the same input
            @test isapprox(Lty, dist_apply_emulated(L, g, parts, gg, plans, yflat);
                rtol=1e-13)
            ȳg = scalar_field(g)
            flat_to_interior!(ȳg, yflat)
            x̄g = scalar_field(g)
            apply_adjoint!(x̄g, L, ȳg, g)
            @test isapprox(Lty, flatten(x̄g); rtol=1e-13)
        end
    end

    @testset "3-D adjoint identity" begin
        rng = Random.MersenneTwister(23)
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), (4, 3, 6);
            bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann()), (Periodic(), Periodic())),
        )
        L = laplacian(g)
        parts, gg, plans = dist_setup(g, 2)
        n = prod(local_size(g))
        xflat = rand(rng, n)
        yflat = rand(rng, n)
        Lx = dist_apply_emulated(L, g, parts, gg, plans, xflat)
        Lty = dist_adjoint_emulated(L, g, parts, gg, plans, yflat)
        @test isapprox(dot(Lx, yflat), dot(xflat, Lty); rtol=1e-13)
    end

    @testset "dense structure: distributed == global, adjoint == transpose" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (3, 4);
            bc=((Dirichlet(), Dirichlet()), (Dirichlet(), Neumann())),
        )
        L = laplacian(g)
        parts, gg, plans = dist_setup(g, 2)
        n = prod(local_size(g))
        A_fwd = zeros(n, n)
        A_adj = zeros(n, n)
        e = zeros(n)
        for j in 1:n
            fill!(e, 0)
            e[j] = 1
            A_fwd[:, j] .= dist_apply_emulated(L, g, parts, gg, plans, e)
            A_adj[:, j] .= dist_adjoint_emulated(L, g, parts, gg, plans, e)
        end
        @test A_fwd == materialize(prepare(L))
        @test A_adj ≈ A_fwd'
    end

    # The distributability whitelist is the whole defense against a silently
    # wrong distributed answer, so it is tested here — on CPU, in CI — not only
    # behind the GPU gate in test/mdla_gpu.jl.
    @testset "distributability guards" begin
        distributable = MatrixFreeOperators._distributable
        check = MatrixFreeOperators._check_distributable

        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8))
        gc = coarsen(g)

        @testset "accepted" begin
            @test distributable(laplacian(g))
            @test distributable(identity_op())
            @test distributable(scaling(2.5))
            @test distributable(-1.5 * laplacian(g))
            @test distributable(0.5 * laplacian(g) + 2.0 * identity_op())
            L = 0.5 * laplacian(g) + 2.0 * identity_op()
            @test check(L) === L      # passes the operator through, does not throw
        end

        @testset "rejected: field-valued parameters" begin
            κ = set!(scalar_field(g), x -> 1 + x[1])
            @test !distributable(scaling(κ))
            @test !distributable(scaling(κ) + laplacian(g))
            v = set!(vector_field(g), x -> SVector(1.0, 0.0))
            @test !distributable(advection(g, v))
            # the message must name the reason, not just the type
            err = try
                check(scaling(κ))
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("global grid", err.msg)
        end

        @testset "rejected: transfer operators span two grids" begin
            @test !distributable(restriction(g, gc))
            @test !distributable(prolongation(gc, g))
            err = try
                check(restriction(g, gc))
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("partitioning", err.msg)
        end

        @testset "rejected: rank changers are non-square" begin
            @test !distributable(gradient(g))
            @test !distributable(divergence(g))
            err = try
                check(gradient(g))
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("non-square", err.msg)
        end

        # The error points at the offending node, not merely at the tree root.
        @testset "message names the culprit inside a tree" begin
            κ = set!(scalar_field(g), x -> 1 + x[1])
            err = try
                check(laplacian(g) + 2.0 * scaling(κ))
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("ScalingOp", err.msg)
        end
    end
end
