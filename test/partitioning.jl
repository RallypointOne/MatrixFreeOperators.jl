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

# Global-grid field carrying a flat interior vector — the single-device reference.
_field(g, xflat) = flat_to_interior!(scalar_field(g, eltype(xflat)), xflat)

# The plane-view geometry is core's (src/partitioning.jl), shared verbatim with
# the MDLA extension — re-deriving it here would let the two silently decouple.
halo_plane_view(f::Field, plane::Int) = MatrixFreeOperators._halo_plane_view(f, plane)

# Every leaf of one kind in a (possibly prepared) tree, so the slice-2b/2c
# testsets can reach in and poke at what `_slab_op` built. One walk driven by a
# predicate: a combinator forgotten here is forgotten for every leaf kind at
# once, rather than making one walk return `()` so its poison loop passes
# vacuously.
let M = MatrixFreeOperators
    global _leaves
    _leaves(L::Union{M.Scaled,M.AdjointOp,M.PreparedAdjoint}, keep) = _leaves(L.op, keep)
    _leaves(L::Union{M.Added,M.Composed,M.PreparedComposed}, keep) =
        (_leaves(L.a, keep)..., _leaves(L.b, keep)...)
    _leaves(L::M.AbstractOperator, keep) = keep(L) ? (L,) : ()
end
_scaling_leaves(L) = _leaves(L, Base.Fix2(isa, MatrixFreeOperators.ScalingOp{<:Field}))
_diffusion_leaves(L) = _leaves(L, Base.Fix2(isa, MatrixFreeOperators.Diffusion))

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

#--------------------------------------------------------------------------------# CPU backend for the distributed walk

# The four backend primitives (src/distributed.jl), implemented over plain
# Vectors. A scatter is direct global indexing — definitionally what a correct
# exchange delivers, given the ordering contract — and a reduction runs the
# two-phase contract explicitly. These drive the REAL core walk, so the CPU proof
# covers the same code the MDLA extension runs, not a second copy of it.

struct EmuCtx{P}
    parts::P
end

MatrixFreeOperators._dist_map!(f, ctx::EmuCtx) = (foreach(f, eachindex(ctx.parts)); nothing)

struct EmuExchange{T}
    ghost_globals::Vector{Vector{Int}}
    plans::Vector{Vector{Tuple{UnitRange{Int},Int}}}
    owned::Vector{UnitRange{Int}}
    stage::Vector{T}              # global flat vector
    local_x::Vector{Vector{T}}    # per-partition [owned | ghost]
end

function emu_exchange(g, parts, proto)
    nc = ncomponents(proto)
    gg, plans = MatrixFreeOperators._slab_ghost_layout(g, parts; ncomp=nc)
    owned = [MatrixFreeOperators._owned_flat_range(g, lg; ncomp=nc) for lg in parts]
    T = MatrixFreeOperators._scalar_eltype(eltype(proto.data))
    local_x = [Vector{T}(undef, length(owned[p]) + length(gg[p])) for p in eachindex(parts)]
    return EmuExchange(gg, plans, owned, Vector{T}(undef, prod(local_size(g)) * nc), local_x)
end

function MatrixFreeOperators._dist_scatter!(X::EmuExchange, fields, ::EmuCtx)
    for p in eachindex(fields)
        interior_to_flat!(view(X.stage, X.owned[p]), fields[p])
    end
    for p in eachindex(fields)
        nown = length(X.owned[p])
        X.local_x[p][1:nown] .= view(X.stage, X.owned[p])
        X.local_x[p][(nown + 1):end] .= view(X.stage, X.ghost_globals[p])
        MatrixFreeOperators._unpack_ghosts!(fields[p], X.local_x[p], nown, X.plans[p])
    end
    return fields
end

function MatrixFreeOperators._dist_reduce!(X::EmuExchange, fields, ::EmuCtx)
    for p in eachindex(fields)
        MatrixFreeOperators._pack_local_x!(
            X.local_x[p], fields[p], length(X.owned[p]), X.plans[p]
        )
    end
    fill!(X.stage, zero(eltype(X.stage)))
    # Phase 1: every owner's own contribution. Phase 2: all neighbour
    # contributions. Splitting them is essential — fused, an owner's copy would
    # overwrite a neighbour's already-accumulated share.
    for p in eachindex(fields)
        X.stage[X.owned[p]] .= view(X.local_x[p], 1:length(X.owned[p]))
    end
    for p in eachindex(fields)
        nown = length(X.owned[p])
        X.stage[X.ghost_globals[p]] .+= view(X.local_x[p], (nown + 1):length(X.local_x[p]))
    end
    for p in eachindex(fields)
        flat_to_interior!(fields[p], view(X.stage, X.owned[p]))
        MatrixFreeOperators.zero_ghosts!(fields[p])
    end
    return fields
end

# CPU twin of prepare_distributed: same normalization, same guards, same
# localization, same tree.
function dist_prepare(L0, g, nparts)
    L = MatrixFreeOperators._push_adjoints(L0)
    MatrixFreeOperators._check_distributable(L)
    MatrixFreeOperators._check_one_grid(L, g)
    T = eltype(spacing(g))
    parts = partition_grid(g, nparts)
    # Guards first, on the GLOBAL tree, then localize — see `_slab_op`.
    prepared = [
        prepare(MatrixFreeOperators._slab_op(L, lg), scalar_field(lg, T)) for lg in parts
    ]
    owned = [MatrixFreeOperators._owned_flat_range(g, lg) for lg in parts]
    ctx = EmuCtx(parts)
    tree = MatrixFreeOperators._dist_tree(
        [p.op for p in prepared],
        AbstractField[p.xpad for p in prepared],
        proto -> emu_exchange(g, parts, proto),
    )
    root = emu_exchange(g, parts, first(prepared).xpad)
    xs = AbstractField[p.xpad for p in prepared]
    ys = AbstractField[p.ypad for p in prepared]
    return (; L, parts, prepared, owned, ctx, tree, root, xs, ys)
end

# Root drivers — the CPU twin of the extension's mul! / _mul_adjoint!. α/β are
# applied at the flat boundary, as PreparedOperator.mul! does, so they never
# cross a collective step.
function dist_mul!(yflat, D, xflat, α=true, β=false)
    for p in eachindex(D.parts)
        flat_to_interior!(D.xs[p], view(xflat, D.owned[p]))
    end
    MatrixFreeOperators._dist_scatter!(D.root, D.xs, D.ctx)
    MatrixFreeOperators._dist_capply!(D.ys, D.tree, D.xs, D.ctx, true, false)
    for p in eachindex(D.parts)
        interior_to_flat!(view(yflat, D.owned[p]), D.ys[p], α, β)
    end
    return yflat
end
dist_mul(D, xflat) = dist_mul!(similar(xflat), D, xflat)

function dist_adjoint!(x̄flat, D, ȳflat)
    for p in eachindex(D.parts)
        flat_to_interior!(D.ys[p], view(ȳflat, D.owned[p]))
    end
    MatrixFreeOperators._dist_adjoint_segment!(D.xs, D.tree, D.ys, D.ctx)
    MatrixFreeOperators._dist_reduce!(D.root, D.xs, D.ctx)
    for p in eachindex(D.parts)
        interior_to_flat!(view(x̄flat, D.owned[p]), D.xs[p])
    end
    return x̄flat
end
dist_adjoint(D, ȳflat) = dist_adjoint!(similar(ȳflat), D, ȳflat)

# CPU twin of the extension's boundary_rhs(P): assemble the lift per slab and copy
# the owned interiors out. `zs` is transient here exactly as it is there.
function dist_boundary_rhs(D, T=Float64)
    zs = MatrixFreeOperators._dist_lift_scratch(D.xs, D.ctx)
    MatrixFreeOperators._dist_boundary_rhs!(D.ys, D.tree, zs, D.ctx, true, false)
    b = Vector{T}(undef, sum(length, D.owned))
    for p in eachindex(D.parts)
        interior_to_flat!(view(b, D.owned[p]), D.ys[p])
    end
    return b
end

# CPU twin of the extension's set!(::MultiDeviceVector, P, fun).
function dist_set(D, fun, T=Float64)
    MatrixFreeOperators._dist_set!(D.xs, fun, D.ctx)
    x = Vector{T}(undef, sum(length, D.owned))
    for p in eachindex(D.parts)
        interior_to_flat!(view(x, D.owned[p]), D.xs[p])
    end
    return x
end

# Dense forward/adjoint matrices through the distributed path.
function dist_materialize(D, n)
    A_fwd, A_adj, e = zeros(n, n), zeros(n, n), zeros(n)
    for j in 1:n
        fill!(e, 0)
        e[j] = 1
        A_fwd[:, j] .= dist_mul(D, e)
        A_adj[:, j] .= dist_adjoint(D, e)
    end
    return A_fwd, A_adj
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
        # A slab keeps the GLOBAL extent; its position lives in local_range alone.
        # That is what makes cell_center bitwise equal on a slab and on the uncut
        # grid — see "slab coordinates are global-index exact" below.
        @test all(p -> p.extent == g.extent, parts)
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

    # Coordinates, not just spacing. A slab that derived its own origin would
    # evaluate lo + (z0)h + (i-0.5)h against the global lo + (z-0.5)h — two
    # roundings versus one, drifting by an ulp. That is invisible in a stencil
    # apply (which reads only spacing) but makes any coordinate-assembled RHS
    # depend on the partition count. `===` on purpose: `≈` would pass while the
    # bug is present.
    @testset "slab coordinates are global-index exact" begin
        # An origin and spacing chosen so lo + k*h is NOT exact in Float64.
        for (ext, sz) in (
            ((( 0.3, 1.7), (-1.1, 2.9)), (5, 9)),
            (((-0.7, 0.9), ( 0.1, 1.3), (2.2, 5.8)), (3, 4, 6)),
        )
            g = CartesianGrid(ext, sz)
            N = length(sz)
            for np in (2, 3)
                parts = partition_grid(g, np)
                for lp in parts
                    off = first(lp.local_range[N]) - 1
                    for I in interior(lp)
                        J = CartesianIndex(ntuple(d -> d == N ? I[d] + off : I[d], N))
                        @test cell_center(lp, I) === cell_center(g, J)
                    end
                end
            end
        end

        # ...and the consequence that matters: a slab-local `set!` reproduces the
        # global one bit for bit, so a distributed RHS is partition-independent.
        g = CartesianGrid(((0.3, 1.7), (-1.1, 2.9)), (5, 9))
        fun = x -> sin(3x[1]) * exp(-x[2]) + 0.25x[1] * x[2]
        ref = set!(scalar_field(g), fun)
        for np in (2, 3)
            for lp in partition_grid(g, np)
                loc = set!(scalar_field(lp), fun)
                @test collect(interior(loc)) ==
                    collect(view(interior(ref), lp.local_range...))
            end
        end
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

    #----------------------------------------------------------------# The distributed tree walk

    cutbcs = ((Dirichlet(), Neumann()), (Periodic(), Periodic()))
    gridof(cut, sz=(4, 6)) =
        CartesianGrid(((0.0, 1.0), (0.0, 2.0)), sz; bc=((Dirichlet(), Dirichlet()), cut))

    # The walk must reproduce the independent emulation on the slice-1 whitelist,
    # or it has regressed the path that was already proven.
    @testset "walk reproduces the emulated exchange on slice-1 operators" begin
        rng = MersenneTwister(20260728)
        for cut in cutbcs, np in (2, 3)
            g = gridof(cut)
            parts, gg, plans = dist_setup(g, np)
            n = prod(local_size(g))
            x, y = rand(rng, n), rand(rng, n)
            for L in (laplacian(g), 0.5 * laplacian(g) + 2.0 * identity_op(), -1.5 * laplacian(g))
                D = dist_prepare(L, g, np)
                @test dist_mul(D, x) == dist_apply_emulated(L, g, parts, gg, plans, x)
                @test dist_adjoint(D, y) ≈ dist_adjoint_emulated(L, g, parts, gg, plans, y)
            end
        end
    end

    @testset "forward parity: Composed" begin
        rng = MersenneTwister(20260729)
        for cut in cutbcs, np in (2, 3)
            g = gridof(cut)
            n = prod(local_size(g))
            x = rand(rng, n)
            composed = (
                laplacian(g) * laplacian(g),
                derivative(g, 1) * derivative(g, 2),
                laplacian(g) * (2.0 * identity_op() + laplacian(g)),
            )
            for L in composed
                @test dist_mul(dist_prepare(L, g, np), x) == flatten(apply(L, _field(g, x)))
            end
        end
    end

    # Negative control: without the mid-tree exchange the intermediate's cut-plane
    # ghosts are stale zeros. If suppressing it changed nothing, the parity test
    # above would be passing for the wrong reason.
    @testset "the mid-tree exchange is load-bearing" begin
        g = gridof((Dirichlet(), Neumann()))
        n = prod(local_size(g))
        x = rand(MersenneTwister(3), n)
        L = laplacian(g) * laplacian(g)
        D = dist_prepare(L, g, 2)
        @test D.tree.xch !== nothing
        # A *fresh* prepare for the suppressed variant: sharing D's buffers would
        # let the intermediate keep the ghosts D's own scatter just wrote, and the
        # control would pass for the wrong reason.
        D2 = dist_prepare(L, g, 2)
        Dsup = (;
            D2...,
            tree=MatrixFreeOperators.DistComposed(D2.tree.a, D2.tree.b, D2.tree.tmps, nothing),
        )
        @test dist_mul(D, x) != dist_mul(Dsup, x)
    end

    @testset "adjoint identity across partitions: Composed" begin
        rng = MersenneTwister(20260730)
        for cut in cutbcs, np in (2, 3)
            g = gridof(cut)
            n = prod(local_size(g))
            x, y = rand(rng, n), rand(rng, n)
            for L in (laplacian(g) * laplacian(g), derivative(g, 1) * derivative(g, 2))
                D = dist_prepare(L, g, np)
                @test isapprox(dot(dist_mul(D, x), y), dot(x, dist_adjoint(D, y)); rtol=1e-12)
            end
        end
    end

    # The literal statement of the transpose argument, on real exchange semantics.
    @testset "dense structure: Composed and AdjointOp" begin
        for cut in cutbcs, np in (2, 3)
            g = gridof(cut, (3, 6))
            n = prod(local_size(g))
            for L in (
                laplacian(g) * laplacian(g),
                derivative(g, 1) * derivative(g, 2),
                adjoint(derivative(g, 1)),
                adjoint(derivative(g, 2)) * derivative(g, 2),
            )
                D = dist_prepare(L, g, np)
                A_fwd, A_adj = dist_materialize(D, n)
                @test A_fwd ≈ materialize(prepare(D.L))
                @test A_adj ≈ A_fwd'
            end
        end
    end

    @testset "AdjointOp leaf as a distributed forward operator" begin
        rng = MersenneTwister(20260731)
        for cut in cutbcs, np in (2, 3)
            g = gridof(cut)
            n = prod(local_size(g))
            x = rand(rng, n)
            L = adjoint(derivative(g, 1))
            D = dist_prepare(L, g, np)
            @test dist_mul(D, x) ≈ flatten(apply(L, _field(g, x)))
        end
    end

    # The forward adjoint node's gather zeroes its input's ghosts. Sharing the
    # enclosing segment's field would destroy exchanged ghosts a sibling still
    # needs, making the answer depend on which term of the Added comes first.
    # Verified load-bearing: dropping DistAdjoint.ins fails only the A + B order.
    @testset "adjoint node under Added: both term orders agree" begin
        rng = MersenneTwister(20260801)
        for cut in cutbcs, np in (2, 3)
            g = gridof(cut)
            n = prod(local_size(g))
            x = rand(rng, n)
            A = adjoint(derivative(g, 1))
            B = laplacian(g)
            ref = flatten(apply(A + B, _field(g, x)))
            first_ = dist_mul(dist_prepare(A + B, g, np), x)
            second = dist_mul(dist_prepare(B + A, g, np), x)
            @test first_ ≈ ref
            @test second ≈ ref
            @test first_ ≈ second
        end
    end

    # Ghosts must be cleared once per adjoint segment: per leaf would wipe the
    # first Added term before it reaches the reduction; never would let the
    # previous call's ghosts leak into the accumulating sibling. Verified
    # load-bearing: moving the clear per-leaf fails this testset.
    @testset "adjoint segment ghost clearing" begin
        rng = MersenneTwister(20260802)
        for cut in cutbcs, np in (2, 3)
            g = gridof(cut)
            n = prod(local_size(g))
            y = rand(rng, n)
            for L in (
                identity_op() + laplacian(g),
                laplacian(g) + identity_op(),
                laplacian(g) + laplacian(g),
            )
                D = dist_prepare(L, g, np)
                x̄g = flatten(apply_adjoint!(scalar_field(g), D.L, _field(g, y), g))
                first_ = dist_adjoint(D, y)
                @test first_ ≈ x̄g
                # Reusing the same prepared operator must be stateless.
                @test dist_adjoint(D, y) == first_
                @test dist_mul(D, y) == dist_mul(D, y)
            end
        end
    end

    # `_dist_reduce!` clears ghosts as part of placing interiors. No current
    # consumer depends on it — the segment clear above already covers every path,
    # and a stencil adjoint zeroes its own input — but it is part of the
    # primitive's documented contract that the MDLA backend must also honour, so
    # pin it here rather than leaving the next consumer to discover it.
    @testset "reduce leaves ghosts cleared" begin
        g = gridof((Dirichlet(), Neumann()))
        D = dist_prepare(laplacian(g), g, 2)
        for f in D.xs
            fill!(f.data, 7.0)
        end
        MatrixFreeOperators._dist_reduce!(D.root, D.xs, D.ctx)
        for f in D.xs
            ghosts = copy(f.data)
            ghosts[interior(f.grid)] .= 0
            @test all(iszero, ghosts)
        end
    end

    # Three partitions: the middle slab is cut on both faces, so a mid-tree
    # reduction that double-counts shows up as a factor-2 error at the seams.
    @testset "three-partition Composed dense structure" begin
        g = gridof((Dirichlet(), Neumann()), (3, 9))
        n = prod(local_size(g))
        for L in (laplacian(g) * laplacian(g), adjoint(derivative(g, 2)) * derivative(g, 2))
            D = dist_prepare(L, g, 3)
            A_fwd, A_adj = dist_materialize(D, n)
            @test A_fwd ≈ materialize(prepare(D.L))
            @test A_adj ≈ A_fwd'
        end
    end

    @testset "α/β through the walk" begin
        rng = MersenneTwister(20260803)
        g = gridof((Dirichlet(), Neumann()))
        n = prod(local_size(g))
        x, y0 = rand(rng, n), rand(rng, n)
        for L in (laplacian(g) * laplacian(g), adjoint(derivative(g, 1)) + laplacian(g))
            D = dist_prepare(L, g, 2)
            base = dist_mul(D, x)
            out = copy(y0)
            dist_mul!(out, D, x, 2.5, 0.5)
            @test out ≈ 2.5 .* base .+ 0.5 .* y0
        end
    end

    # A PreparedAdjoint over a composition would run aᵀ then bᵀ with no reduction
    # between them. Normalization (the same _push_adjoints prepare applies) rewrites
    # it to Composed(bᵀ, aᵀ), which the walk handles node by node.
    @testset "AdjointOp over a composite is normalized" begin
        g = gridof((Dirichlet(), Neumann()), (3, 6))
        n = prod(local_size(g))
        inner = derivative(g, 1) * derivative(g, 2)
        L = MatrixFreeOperators.AdjointOp(inner)
        pushed = MatrixFreeOperators._push_adjoints(L)
        @test pushed isa Composed
        @test !(pushed isa MatrixFreeOperators.AdjointOp)
        D = dist_prepare(L, g, 2)
        A_fwd, _ = dist_materialize(D, n)
        @test A_fwd ≈ materialize(prepare(inner))'
    end

    #----------------------------------------------------------------# Field coefficients (2b)

    # Every coefficient here VARIES ALONG THE CUT DIMENSION and is asymmetric
    # about it. A coefficient constant in the cut dimension cannot catch a
    # halo-shifted or reversed slice, which is the likeliest bug in `_slab_field`
    # (unit-tested cell by cell in the diffusion section below, where the ghosts
    # it carries are first read).
    coeff_fun(x) = 1.5 + x[2] + 0.3 * x[1] * x[2] + 0.2 * x[2]^2

    # The claim that lets a ScalingOp coefficient be partitioned with NO exchange
    # of its own: it is read pointwise at the cell being written, so its ghosts
    # are never consulted. (Diffusion's ARE — see the slice-2c negative control
    # below.) Poison them and demand the answer not move.
    @testset "a localized ScalingOp coefficient's ghosts are never read" begin
        g = gridof((Dirichlet(), Neumann()))
        n = prod(local_size(g))
        x = rand(MersenneTwister(11), n)
        κ = set!(scalar_field(g), coeff_fun)
        for L in (scaling(κ), laplacian(g) * scaling(κ), scaling(κ) * laplacian(g))
            D = dist_prepare(L, g, 2)
            clean = dist_mul(D, x)
            for pre in D.prepared
                for S in _scaling_leaves(pre.op)
                    saved = copy(interior(S.coeff))
                    fill!(S.coeff.data, NaN)      # poison every ghost
                    interior(S.coeff) .= saved
                end
            end
            @test dist_mul(D, x) == clean
        end
    end

    @testset "forward parity: Field coefficient" begin
        rng = MersenneTwister(20260729)
        for cut in cutbcs, np in (2, 3)
            g = gridof(cut)
            n = prod(local_size(g))
            x = rand(rng, n)
            κ = set!(scalar_field(g), coeff_fun)
            ops = (
                scaling(κ),
                laplacian(g) * scaling(κ),
                scaling(κ) * laplacian(g),
                2.0 * scaling(κ) + laplacian(g),
                derivative(g, 1) * scaling(κ) * derivative(g, 1),
            )
            for L in ops
                @test dist_mul(dist_prepare(L, g, np), x) == flatten(apply(L, _field(g, x)))
            end
        end
    end

    # Negative control for the slice window. Shifting it by one plane is the
    # halo-off-by-h bug in miniature: still a valid coefficient everywhere, still
    # the right shapes, and wrong on every partition after the first.
    @testset "the coefficient slice window is load-bearing" begin
        g = gridof((Dirichlet(), Neumann()))
        n = prod(local_size(g))
        x = rand(MersenneTwister(13), n)
        κ = set!(scalar_field(g), coeff_fun)
        D = dist_prepare(laplacian(g) * scaling(κ), g, 2)
        good = dist_mul(D, x)
        @test good == flatten(apply(laplacian(g) * scaling(κ), _field(g, x)))
        Dbad = dist_prepare(laplacian(g) * scaling(κ), g, 2)
        for (p, lp) in enumerate(Dbad.parts)
            p == 1 && continue                       # partition 1's window is right
            shifted = ntuple(d -> d == 2 ? lp.local_range[d] .- 1 : lp.local_range[d], 2)
            for S in _scaling_leaves(Dbad.prepared[p].op)
                interior(S.coeff) .= view(interior(κ), shifted...)
            end
        end
        @test dist_mul(Dbad, x) != good
    end

    @testset "adjoint identity and dense structure: Field coefficient" begin
        rng = MersenneTwister(20260730)
        for cut in cutbcs, np in (2, 3)
            g = gridof(cut)
            n = prod(local_size(g))
            κ = set!(scalar_field(g), coeff_fun)
            x, y = rand(rng, n), rand(rng, n)
            for L in (
                scaling(κ) * laplacian(g),
                laplacian(g) * scaling(κ),
                adjoint(derivative(g, 1) * scaling(κ)),
                scaling(κ) + laplacian(g),
            )
                D = dist_prepare(L, g, np)
                @test dot(dist_mul(D, x), y) ≈ dot(x, dist_adjoint(D, y)) rtol = 1e-13
                A_fwd, A_adj = dist_materialize(D, n)
                @test A_fwd ≈ materialize(prepare(D.L))
                @test A_adj ≈ A_fwd'
            end
        end
    end

    #----------------------------------------------------------------# Boundary lift (2b)

    # A DIFFERENT inhomogeneous value on every face. A slab that applied a
    # cut-dimension BC to its Interface face, or swapped low for high, would still
    # produce a plausible lift under a symmetric choice; it cannot under this one.
    inhom_grid(cut, sz=(4, 6)) = CartesianGrid(
        ((0.0, 1.0), (0.0, 2.0)), sz;
        bc=((Dirichlet(0.75), Neumann(-1.25)), cut),
    )
    inhom_cuts = (
        (Dirichlet(2.5), Neumann(0.4)),      # physical cut: only the end slabs lift
        (Periodic(), Periodic()),            # periodic cut: transverse faces only
    )

    # The premise the whole leaf-level lift rests on: a slab's inhomogeneous ghost
    # field is the restriction of the global one, with no exchange. Compared at
    # transverse-INTERIOR positions only — corner ghosts legitimately differ,
    # because the dimension-N pass that would overwrite them globally is a no-op on
    # an Interface face, and no whitelisted stencil reads a corner.
    @testset "the inhomogeneous ghost field is exact per slab" begin
        for cut in inhom_cuts, np in (2, 3)
            g = inhom_grid(cut)
            zg = scalar_field(g)
            MatrixFreeOperators.fill_bc_inhomogeneous!(zg.data, g)
            D = dist_prepare(laplacian(g), g, np)
            zs = MatrixFreeOperators._dist_lift_scratch(D.xs, D.ctx)
            h1, n1 = halo_width(g)[1], local_size(g)[1]
            tr = (h1 + 1):(h1 + n1)
            for (p, lp) in enumerate(D.parts)
                off = first(lp.local_range[2]) - 1
                for i in axes(zs[p].data, 2)
                    @test view(zs[p].data, tr, i) == view(zg.data, tr, i + off)
                end
            end
        end
    end

    @testset "boundary_rhs parity" begin
        for cut in inhom_cuts, np in (1, 2, 3)
            g = inhom_grid(cut)
            κ = set!(scalar_field(g), coeff_fun)
            D1 = derivative(g, 1)
            ops = (
                laplacian(g),
                -1.5 * laplacian(g),
                laplacian(g) + 2.0 * identity_op(),
                laplacian(g) * laplacian(g),
                derivative(g, 1) * derivative(g, 2),
                scaling(κ) * laplacian(g),
                laplacian(g) * scaling(κ),
                # A diagonal factor between two stencils: the OUTER Composed's
                # `_reads_ghosts` is false, so its lift must skip the exchange while
                # the inner one still fires — the gating, not just the exchange.
                derivative(g, 1) * scaling(κ) * derivative(g, 1),
                # both term orders: a DistAdjoint's zero lift has to WRITE its zero
                # when it runs first, and stay a no-op when it accumulates
                adjoint(D1) + laplacian(g),
                laplacian(g) + adjoint(D1),
                # slice 2c: the lift reads κ at the wall ghost, where the even
                # mirror gives the one-sided face coefficient — so a slab's lift
                # is only exact if its coefficient ghosts survived localization.
                diffusion(g, κ),
                laplacian(g) * diffusion(g, κ),
            )
            for L in ops
                @test dist_boundary_rhs(dist_prepare(L, g, np)) ==
                    flatten(boundary_rhs(L, g))
            end
        end

        # 3-D, where the transverse faces of every slab contribute and the cut
        # dimension is the last one.
        g3 = CartesianGrid(
            ((0.0, 1.0), (0.0, 2.0), (0.0, 1.5)), (3, 4, 6);
            bc=(
                (Dirichlet(0.75), Neumann(-1.25)),
                (Neumann(0.3), Dirichlet(-2.0)),
                (Dirichlet(2.5), Neumann(0.4)),
            ),
        )
        for np in (2, 3), L in (laplacian(g3), laplacian(g3) * laplacian(g3))
            @test dist_boundary_rhs(dist_prepare(L, g3, np)) ==
                flatten(boundary_rhs(L, g3))
        end

        # A homogeneous problem lifts to exactly zero — the case that must not
        # start costing anything now that there is a walk for it.
        gh = gridof((Dirichlet(), Neumann()))
        @test all(iszero, dist_boundary_rhs(dist_prepare(laplacian(gh), gh, 2)))
    end

    # Negative control, mirroring "the mid-tree exchange is load-bearing": the
    # inner lift `b_b` is a real field with nonzero interior near the physical
    # boundary, so the outer factor reads its cut-plane ghosts. Without the
    # exchange those are stale, and the parity test above would be passing for the
    # wrong reason on a grid whose intermediate happened to vanish at the cut.
    @testset "the lift's mid-tree exchange is load-bearing" begin
        g = inhom_grid((Dirichlet(2.5), Neumann(0.4)))
        L = laplacian(g) * laplacian(g)
        D = dist_prepare(L, g, 2)
        @test D.tree.xch !== nothing
        D2 = dist_prepare(L, g, 2)
        Dsup = (;
            D2...,
            tree=MatrixFreeOperators.DistComposed(D2.tree.a, D2.tree.b, D2.tree.tmps, nothing),
        )
        @test dist_boundary_rhs(D) != dist_boundary_rhs(Dsup)
    end

    # The lift reuses each Composed node's own `tmps`/`xch` rather than allocating
    # a second set. That is safe only because the two never run concurrently — pin
    # it, because a violation would surface as stale physical ghosts on an
    # intermediate, i.e. a wrong answer on the *next* solve rather than this one.
    @testset "lift and mul! do not corrupt each other" begin
        for cut in inhom_cuts
            g = inhom_grid(cut)
            n = prod(local_size(g))
            x = rand(MersenneTwister(17), n)
            L = laplacian(g) * laplacian(g)
            D = dist_prepare(L, g, 2)
            y0, b0 = dist_mul(D, x), dist_boundary_rhs(D)
            @test dist_mul(D, x) == y0        # a lift in between changes nothing
            @test dist_boundary_rhs(D) == b0  # ...and neither does a mul!
            @test dist_boundary_rhs(D) == b0  # the lift is idempotent on its own
        end
    end

    # The end the whole slice exists for: assemble `f - b` slab-locally and get the
    # same linear system the single-device path assembles globally.
    @testset "a full inhomogeneous RHS assembles slab-locally" begin
        for cut in inhom_cuts, np in (2, 3)
            g = inhom_grid(cut, (8, 12))
            κ = set!(scalar_field(g), coeff_fun)
            fun = x -> sin(3x[1]) * exp(-x[2]) + 0.25x[1] * x[2]
            for L in (laplacian(g), scaling(κ) * laplacian(g))
                D = dist_prepare(L, g, np)
                ref = flatten(set!(scalar_field(g), fun)) .- flatten(boundary_rhs(L, g))
                @test dist_set(D, fun) .- dist_boundary_rhs(D) == ref
            end
        end
    end

    #----------------------------------------------------------------# Compact diffusion leaf (2c)

    # A coefficient the compact flux form can actually be wrong about: strictly
    # POSITIVE, so HarmonicMean is defined, and varying across the cut so that
    # κ differs on the two sides of every cut face. A κ constant in the cut
    # dimension would average to the same face value from either side and would
    # hide a zeroed ghost entirely — see the negative control below.
    diff_coeff_fun(x) = 1.5 + sum(d -> d * x[d]^2, eachindex(x)) + 0.4 * prod(x)

    # Every grid rank, because `partition_grid` always cuts dimension N — so
    # "a cut in each dimension" is reached by varying the rank, not the axis.
    diff_exts = (
        (((0.0, 1.0),), (8,)),
        (((0.0, 1.0), (0.0, 2.0)), (4, 6)),
        (((0.0, 1.0), (0.0, 1.0), (0.0, 2.0)), (3, 3, 6)),
    )
    diff_grid(ext, sz, cut) = CartesianGrid(
        ext, sz; bc=ntuple(d -> d == length(sz) ? cut : (Dirichlet(), Dirichlet()), length(sz))
    )

    # `_slab_field` slices the PADDED window for every coefficient, so the oracle
    # covers the ghosts too. Two sources, because the two consumers differ: a
    # plain `set!` field is what `ScalingOp` hands in (ghosts zero, never read),
    # and a `diffusion`-extended κ is what the leaf hands in, whose ghosts ARE
    # read and must all carry a value — the neighbour's κ at a cut, the even
    # mirror at a wall, the wrap under Periodic.
    @testset "_slab_field restricts the padded coefficient exactly" begin
        for (ext, sz) in diff_exts, cut in ((Dirichlet(), Neumann()), (Periodic(), Periodic()))
            g = diff_grid(ext, sz, cut)
            N = length(sz)
            plain = set!(scalar_field(g), diff_coeff_fun)
            extended = diffusion(g, plain).κ
            for (κ, ghosts_read) in ((plain, false), (extended, true)), np in (1, 2, 3)
                parts = partition_grid(g, np)
                for lp in parts
                    lκ = MatrixFreeOperators._slab_field(κ, lp)
                    @test lκ.grid === lp
                    @test size(lκ.data) == MatrixFreeOperators.padded_size(lp)
                    # Slab padded index p is global padded first(local_range)-1+p,
                    # in every dimension. Spelled out cell by cell rather than as
                    # the same `ntuple` the implementation uses, so a wrong window
                    # cannot agree with a wrong oracle.
                    for I in CartesianIndices(lκ.data)
                        J = CartesianIndex(
                            ntuple(d -> first(lp.local_range[d]) - 1 + I[d], N)
                        )
                        @test lκ.data[I] == κ.data[J]
                    end
                    # ...and for the extended κ no ghost is left at zero.
                    ghosts_read && @test !any(iszero, lκ.data)
                end
                # ...and the slabs tile the global interior with no gap or overlap
                rebuilt = similar(interior(κ))
                for lp in parts
                    view(rebuilt, lp.local_range...) .=
                        interior(MatrixFreeOperators._slab_field(κ, lp))
                end
                @test rebuilt == interior(κ)
            end
        end
    end

    @testset "forward parity, adjoint identity, and dense structure: Diffusion" begin
        rng = MersenneTwister(20260814)
        for (ext, sz) in diff_exts,
            cut in ((Dirichlet(), Neumann()), (Periodic(), Periodic())),
            avg in (ArithmeticMean(), HarmonicMean()),
            np in (2, 3)

            g = diff_grid(ext, sz, cut)
            n = prod(local_size(g))
            κ = set!(scalar_field(g), diff_coeff_fun)
            Dop = diffusion(g, κ; averaging=avg)
            x, y = rand(rng, n), rand(rng, n)
            # `laplacian(g) + Dop` puts the leaf SECOND under the Added, so the slab
            # reduction folds a Diffusion contribution accumulated with β = true — the
            # blending branch of its Interface transpose (issue #77).
            for L in (
                Dop, laplacian(g) * Dop, Dop * laplacian(g), 2.0 * Dop + identity_op(),
                laplacian(g) + Dop,
            )
                D = dist_prepare(L, g, np)
                @test dist_mul(D, x) == flatten(apply(L, _field(g, x)))
                @test dot(dist_mul(D, x), y) ≈ dot(x, dist_adjoint(D, y)) rtol = 1e-12
                A_fwd, A_adj = dist_materialize(D, n)
                @test A_fwd ≈ materialize(prepare(D.L))
                @test A_adj ≈ A_fwd'
            end
            # A real κ makes the leaf its own transpose globally, so the
            # distributed adjoint must reproduce the distributed forward action —
            # a claim the `Interface` gather path has to earn, since it is a
            # different code path from `apply!` (`src/operators/diffusion.jl`).
            Ds = dist_prepare(Dop, g, np)
            @test dist_adjoint(Ds, y) ≈ dist_mul(Ds, y) rtol = 1e-13
        end
    end

    # The claim this whole slice rests on: the cut-plane ghosts of a localized κ
    # ARE read, so they must carry the neighbour's values. Zero them — what an
    # interior-only slice would have left — and demand the answer move. Verified
    # load-bearing: the slice-2b `_slab_field`, which copied the interior and
    # zeroed the ghosts, fails this testset and the dense-parity one above.
    @testset "a localized diffusion coefficient's cut-plane ghosts ARE read" begin
        for (ext, sz) in diff_exts,
            cut in ((Dirichlet(), Neumann()), (Periodic(), Periodic()))

            g = diff_grid(ext, sz, cut)
            n = prod(local_size(g))
            x = rand(MersenneTwister(23), n)
            κ = set!(scalar_field(g), diff_coeff_fun)
            for L in (diffusion(g, κ), laplacian(g) * diffusion(g, κ))
                D = dist_prepare(L, g, 2)
                clean = dist_mul(D, x)
                # The slab's Interface planes are exactly what the prepared
                # exchange carries as `plans[p]` (the "ghost layout invariants"
                # testset proves that set), so take them from there rather than
                # spelling the padded-plane index map out a third time.
                for p in eachindex(D.parts)
                    @test !isempty(D.root.plans[p])   # a 2-way cut always has one
                    for Dl in _diffusion_leaves(D.prepared[p].op),
                        (_, pl) in D.root.plans[p]

                        fill!(halo_plane_view(Dl.κ, pl), 0)
                    end
                end
                @test dist_mul(D, x) != clean
            end
        end
    end

    # #56's invariant: κ is constant through a solve, so localizing it costs a
    # slice at prepare time and nothing per apply. The exchange structure must
    # therefore be bit-for-bit the Laplacian's — same gating rule, same nodes.
    @testset "Diffusion adds no per-apply exchange" begin
        g = gridof((Dirichlet(), Neumann()))
        κ = set!(scalar_field(g), diff_coeff_fun)
        Dop = diffusion(g, κ)
        # A bare stencil leaf: no mid-tree node exists to hold an exchange.
        @test dist_prepare(Dop, g, 2).tree isa MatrixFreeOperators.DistLeaf
        @test dist_prepare(laplacian(g), g, 2).tree isa MatrixFreeOperators.DistLeaf
        # ...and inside a composition it gates identically to a Laplacian: an
        # exchange when the OUTER factor reads ghosts, none when it is diagonal.
        for (dif, lap) in (
            (laplacian(g) * Dop, laplacian(g) * laplacian(g)),
            (Dop * scaling(κ), laplacian(g) * scaling(κ)),
            (scaling(κ) * Dop, scaling(κ) * laplacian(g)),
        )
            @test (dist_prepare(dif, g, 2).tree.xch === nothing) ==
                (dist_prepare(lap, g, 2).tree.xch === nothing)
        end
    end

    # Steady-state work must not scale with the grid: a per-call scratch
    # allocation inside the walk would show as a ~4x jump when cells quadruple.
    @testset "walk allocations do not scale with grid size" begin
        function steady(sz, mk, drive!)
            g = CartesianGrid(
                ((0.0, 1.0), (0.0, 1.0)), sz;
                bc=((Dirichlet(), Dirichlet()), (Dirichlet(), Neumann())),
            )
            n = prod(sz)
            x = rand(MersenneTwister(7), n)
            D = dist_prepare(mk(g), g, 2)
            out = similar(x)
            drive!(out, D, x)             # warm up
            return @allocated drive!(out, D, x)
        end
        for mk in (
            g -> laplacian(g) * laplacian(g),
            # A localized leaf dispatches dynamically; that must stay O(1), not O(cells).
            g -> laplacian(g) * scaling(set!(scalar_field(g), coeff_fun)),
            # ...including the one whose localization copies a padded coefficient:
            # that copy belongs to prepare, and must not reappear per apply.
            g -> laplacian(g) * diffusion(g, set!(scalar_field(g), diff_coeff_fun)),
        )
            small, large = steady((16, 16), mk, dist_mul!), steady((32, 32), mk, dist_mul!)
            @test large < 2 * small
        end

        # The adjoint walk, where `Added` sends β = true into a leaf and the gather
        # has to accumulate (issue #33). One padded slab of a 32² grid cut in two is
        # ~4.9 kB, and the pre-fix gather allocated one per accumulating call per
        # partition: `laplacian + laplacian` measured 6.4 kB → 14.0 kB across the 4x
        # cell jump, against 3.3 kB → 3.6 kB after. The residual is the emulated
        # exchange's staging, which is O(surface) and so grows a little on its own —
        # hence a bound on the *delta*, which is what an O(cells) leak moves.
        for mk in (
            g -> laplacian(g) + laplacian(g),
            g -> laplacian(g) + derivative(g, 1),
            g -> (laplacian(g) * laplacian(g)) + laplacian(g),
            # ...and the slab diffusion leaf, whose adjoint is the interior stencil
            # plus the ghost-plane gather over a padded κ (issue #77) rather than the
            # self-adjoint shortcut.
            g -> diffusion(g, set!(scalar_field(g), diff_coeff_fun)) + laplacian(g),
            # ...in both Added slots: as `node.b` the leaf accumulates with β = true.
            g -> laplacian(g) + diffusion(g, set!(scalar_field(g), diff_coeff_fun)),
        )
            small = steady((16, 16), mk, dist_adjoint!)
            large = steady((32, 32), mk, dist_adjoint!)
            @test large - small < 1024
            @test large < 1.5 * small
        end
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
            @test distributable(derivative(g, 1))
            @test distributable(identity_op())
            @test distributable(scaling(2.5))
            @test distributable(-1.5 * laplacian(g))
            @test distributable(0.5 * laplacian(g) + 2.0 * identity_op())
            @test distributable(laplacian(g) * laplacian(g))
            @test distributable(adjoint(derivative(g, 1)))
            L = 0.5 * laplacian(g) + 2.0 * identity_op()
            @test check(L) === L      # passes the operator through, does not throw

            # slice 2b: a real coefficient field on an undistributed CartesianGrid
            # is sliceable onto the slabs, so it joins the whitelist.
            κ = set!(scalar_field(g), x -> 1 + x[1])
            @test distributable(scaling(κ))
            @test distributable(scaling(κ) + laplacian(g))
            @test distributable(laplacian(g) * scaling(κ))
            @test distributable(adjoint(derivative(g, 1) * scaling(κ)))

            # slice 2c: the compact diffusion leaf joins on the same coefficient
            # terms. Its face averaging reads κ across the cut, which
            # `_slab_field`'s padded window supplies at localization time — no exchange, so
            # nothing further to require of it here.
            κp = set!(scalar_field(g), x -> 1 + x[1] + x[2])
            for avg in (ArithmeticMean(), HarmonicMean())
                @test distributable(diffusion(g, κp; averaging=avg))
            end
            @test distributable(laplacian(g) * diffusion(g, κp))
            @test distributable(2.0 * diffusion(g, κp) + identity_op())
            @test distributable(adjoint(diffusion(g, κp) * derivative(g, 1)))
        end

        @testset "rejected: field-valued parameters" begin
            v = set!(vector_field(g), x -> SVector(1.0, 0.0))
            @test !distributable(advection(g, v))
            # the message must name the reason, not just the type
            err = try
                check(advection(g, v))
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("global grid", err.msg)

            # A complex coefficient stays rejected: the distributed vectors are
            # real (typed from the grid spacing), so its product has nowhere to
            # land — the guard names the cause instead of an InexactError.
            κc = Field(ComplexF64.(ones(size(scalar_field(g).data))), g)
            @test !distributable(scaling(κc))
            errc = try
                check(scaling(κc))
            catch e
                e
            end
            @test errc isa ArgumentError
            @test occursin("real-eltype", errc.msg)

            # ...and rejected for Diffusion for exactly the same reason.
            @test !distributable(diffusion(g, κc))
            errd = try
                check(diffusion(g, κc))
            catch e
                e
            end
            @test errd isa ArgumentError
            @test occursin("Diffusion", errd.msg)
            @test occursin("real-eltype", errd.msg)
        end

        # A coefficient on a *different* grid is only visible from the tree, not
        # from any single operator — without this check it would be sliced onto
        # slabs of a grid it does not live on and answer with plausible numbers.
        @testset "rejected: a coefficient on another grid" begin
            check1 = MatrixFreeOperators._check_one_grid
            same = MatrixFreeOperators._same_grid
            g2 = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8))   # separately built
            @test same(g2, g)
            @test !same(g, gc)                                     # different resolution
            # The two properties that make _same_grid worth having: a slab never
            # compares equal to the grid it was cut from (local_range differs), and
            # a device-adapted twin does (device is not compared).
            @test all(p -> !same(p, g), partition_grid(g, 2))
            @test same(Adapt.adapt(Array, g), g)

            κg = set!(scalar_field(g), x -> 1 + x[1])
            @test check1(laplacian(g) + scaling(κg), g) isa MatrixFreeOperators.Added
            @test check1(laplacian(g) + scaling(κg), g2) isa MatrixFreeOperators.Added

            κc = set!(scalar_field(gc), x -> 1 + x[1])
            @test distributable(laplacian(g) + scaling(κc))        # each leaf is fine alone
            err = try
                check1(laplacian(g) + scaling(κc), g)
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("ScalingOp", err.msg)
            @test occursin("different grid", err.msg)

            # Same trap for the diffusion leaf, and it is `_check_one_grid` that
            # has to catch it: `_slab_field` copies the coefficient's OWN
            # ghosts, so a κ carrying another grid's ghosts is the one input that
            # would slice into plausible numbers rather than an error.
            errd = try
                check1(laplacian(g) + diffusion(gc, κc), g)
            catch e
                e
            end
            @test errd isa ArgumentError
            @test occursin("Diffusion", errd.msg)
            @test occursin("different grid", errd.msg)

            # The public constructor already refuses a κ on another grid, so the
            # trap is the inner one: `Diffusion(g, κ, avg)` with κ living
            # elsewhere. Its `operator_grid` is `g`, so the generic mismatch
            # check is blind to κ's grid, and `_slab_field` would then
            # window κ by a `local_range` that means nothing on it — a wall
            # ghost filled from κ's interior on a larger grid (a plausible
            # face coefficient, no error), or a `BoundsError` on a smaller one.
            for szκ in ((12, 10), (4, 6))
                gκ = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), szκ)
                κκ = set!(scalar_field(gκ), x -> 1 + x[1])
                Dbad = MatrixFreeOperators.Diffusion(g, κκ, ArithmeticMean())
                @test distributable(Dbad)          # the leaf alone cannot tell
                for L in (Dbad, laplacian(g) + Dbad)
                    errk = try
                        check1(L, g)
                    catch e
                        e
                    end
                    @test errk isa ArgumentError
                    @test occursin("Diffusion", errk.msg)
                    @test occursin("different grid", errk.msg)
                end
                @test_throws ArgumentError dist_prepare(Dbad, g, 2)
            end
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

        # A composition of two distributable factors on DIFFERENT grids is still
        # rejected — the intermediate would need a second, consistent partitioning.
        @testset "rejected: Composed across grids" begin
            L = laplacian(gc) * restriction(g, gc)
            @test !distributable(L)
            @test_throws ArgumentError check(L)
        end

        # The error points at the offending node, not merely at the tree root.
        @testset "message names the culprit inside a tree" begin
            v = set!(vector_field(g), x -> SVector(1.0, 0.0))
            err = try
                check(laplacian(g) + 2.0 * advection(g, v))
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("Advection", err.msg)
        end
    end
end
