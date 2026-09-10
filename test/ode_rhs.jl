# Explicit time stepping. The idiom is field-level `apply!` — `du = L(u)` on
# `Field`s, no flat vectors — because the flat `mul!` boundary exists for Krylov
# and pays for a `flat_to_interior!` / `interior_to_flat!` copy pair on every
# call (issue #87). `mul!` still works as an RHS, and the last testset keeps it
# honest, but the documented path is `apply!(du, L, u)` on the operator or
# `apply!(du, P, u)` on a prepared one.

# Classical RK4 over flat vectors: the same closure shape an ODE integrator takes.
function rk4!(f!, u, dt, nsteps)
    k1 = similar(u)
    k2 = similar(u)
    k3 = similar(u)
    k4 = similar(u)
    utmp = similar(u)
    for _ in 1:nsteps
        f!(k1, u)
        utmp .= u .+ (dt / 2) .* k1
        f!(k2, utmp)
        utmp .= u .+ (dt / 2) .* k2
        f!(k3, utmp)
        utmp .= u .+ dt .* k3
        f!(k4, utmp)
        u .+= (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end
    return u
end

# The same RK4 over Fields: stages are fields, the axpys act on interiors, and the
# RHS is `apply!` at field level. Ghosts of the stage fields are scratch.
function rk4_field!(f!, u::Field, dt, nsteps)
    k1 = similar(u)
    k2 = similar(u)
    k3 = similar(u)
    k4 = similar(u)
    utmp = similar(u)
    ui, ti = interior(u), interior(utmp)
    for _ in 1:nsteps
        f!(k1, u)
        ti .= ui .+ (dt / 2) .* interior(k1)
        f!(k2, utmp)
        ti .= ui .+ (dt / 2) .* interior(k2)
        f!(k3, utmp)
        ti .= ui .+ dt .* interior(k3)
        f!(k4, utmp)
        ui .+= (dt / 6) .* (interior(k1) .+ 2 .* interior(k2) .+ 2 .* interior(k3) .+ interior(k4))
    end
    return u
end

# Allocation probes behind a function barrier, so @allocated charges the call and
# not type-unstable global access.
function alloc_apply(du, L, u)
    apply!(du, L, u)
    apply!(du, L, u)
    return @allocated apply!(du, L, u)
end

# Allocation policy for the stepping paths.
#
# These paths allocate nothing *per cell*, but they are not bit-for-bit zero on
# every platform. Each leaf stencil wraps its non-broadcast arguments in `Ref`
# before the fused broadcast, and each ghost fill builds `view` slabs; whether
# LLVM elides those heap objects is a codegen decision, not a contract. Measured
# on macOS/aarch64 every probe below reports 0 B. On Linux x86_64 the same probes
# report a few tens of bytes for the 1D/2D leaf and Composed trees, of order a
# thousand for the 3D Float32 tree, and a couple of thousand for the per-leaf
# forest sweeps. That residual is fixed per operator node, per spatial dimension
# and per forest leaf — it never scales with the number of interior cells.
#
# So do NOT re-tighten these to `== 0`; that pins a platform constant. Assert the
# portable property instead, in two halves that are both load-bearing:
#   1. the same operator on a larger grid allocates the *same* number of bytes.
#      This is what catches a reintroduced per-cell allocation — a `Core.Box` in
#      a stencil costs 16 B/cell and on grids this small (48 and 120 cells) it
#      would sail under any absolute cap loose enough to hold the real residual;
#      see the boxed-neighbour-index note in test/diffusion.jl.
#   2. that fixed residual stays small in absolute terms, so a grid-independent
#      blow-up is still caught.
# The pure (unprepared) Composed tree below is the positive control: it really
# does allocate its intermediate every call, and its byte count grows with the
# grid, which is what proves half (1) has teeth on this machine too.

# Build the same operator/field trio at two sizes and measure both. `build(s)`
# returns the `(du, L, u)` trio `alloc_apply` takes, at scale `s`.
function alloc_grid_pair(build, small, large)
    return alloc_apply(build(small)...), alloc_apply(build(large)...)
end

_grid_1d(n) = CartesianGrid(((0.0, 2π),), (n,); bc=((Periodic(), Periodic()),))

function build_1d_leaf(n)
    g = _grid_1d(n)
    u = set!(scalar_field(g), x -> sin(x[1]))
    return similar(u), laplacian(g), u
end

function build_1d_added(n)
    g = _grid_1d(n)
    u = set!(scalar_field(g), x -> sin(x[1]))
    A = laplacian(g) + 2.0 * derivative(g, 1; order=1)          # Added(leaf, Scaled)
    return similar(u), A, u
end

# `divergence * scaling * gradient` on an 8s × 6s grid, prepared or pure.
function build_2d(s::Int, prepared::Bool)
    g = CartesianGrid(
        ((0.0, 1.0), (0.0, 1.0)), (8s, 6s);
        bc=((Dirichlet(), Dirichlet()), (Periodic(), Periodic())),
    )
    κ = set!(scalar_field(g), x -> 1 + x[1])
    K = divergence(g) * scaling(κ) * MatrixFreeOperators.gradient(g)
    u = scalar_field(g)
    interior(u) .= rand(Random.MersenneTwister(87 + s), local_size(g)...)
    return similar(u), prepared ? prepare(K, scalar_field(g)) : K, u
end

build_2d_prepared(s) = build_2d(s, true)
build_2d_pure(s) = build_2d(s, false)

# 3D Float32 Composed + Scaled + Added under one prepare, on a 6s × 5s × 4s grid.
function build_3d_prepared(s::Int)
    g = CartesianGrid(
        ((0.0f0, 1.0f0), (0.0f0, 2.0f0), (0.0f0, 1.0f0)), (6s, 5s, 4s);
        bc=((Dirichlet(), Dirichlet()), (Periodic(), Periodic()), (Neumann(), Neumann())),
    )
    κ = set!(scalar_field(g), x -> 1.0f0 + x[1] * x[3])
    K = divergence(g) * scaling(κ) * MatrixFreeOperators.gradient(g) + 0.5f0 * laplacian(g)
    u = scalar_field(g)
    interior(u) .= rand(Random.MersenneTwister(87 + s), Float32, local_size(g)...)
    return similar(u), prepare(K, scalar_field(g)), u
end

# The forest probe varies the leaf *size* at a fixed leaf count, so the per-leaf
# residual is held constant and only the cell count moves. Returns the leaf count
# alongside the two measurements so the caller can assert the count really is fixed.
function forest_step_alloc(n::Int, bs::Int)
    bc = ((Dirichlet(), Dirichlet()), (Neumann(), Neumann()))
    g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (n, n); bc=bc)
    bf = BlockForest(g; blocksize=(bs, bs), maxlevel=2)
    uf = set!(scalar_field(bf), x -> sinpi(x[1]) * cospi(2x[2]) + 0.3 * x[1])
    Sf = laplacian(bf) + adjoint(derivative(bf, 1; order=1))
    P = prepare(Sf)
    up = pack(uf)
    Pp = prepare(Sf, up)
    return (MatrixFreeOperators.nleaves(bf),
            alloc_apply(similar(uf), P, uf),
            alloc_apply(similar(up), Pp, up))
end

@testset "Explicit time stepping" begin
    @testset "field-level apply! RHS (1D periodic heat equation, RK4)" begin
        g = CartesianGrid(((0.0, 2π),), (64,); bc=((Periodic(), Periodic()),))
        L = laplacian(g)
        f!(du, u) = apply!(du, L, u)

        u = set!(scalar_field(g), x -> sin(x[1]))
        dt = 0.005
        nsteps = 200
        rk4_field!(f!, u, dt, nsteps)

        t = dt * nsteps
        uref = exp(-t) .* interior(set!(scalar_field(g), x -> sin(x[1])))
        err = maximum(abs, interior(u) .- uref)
        @info "field-level RK4 heat equation" t err
        @test err < 1e-3

        # A leaf needs no prepare to step, nor does a +/scalar combination of
        # leaves. Per the allocation policy above `alloc_grid_pair`, assert that
        # the byte count does not grow with the grid, plus a small absolute cap —
        # never an exact zero, which is true only on some platforms.
        a_leaf_small, a_leaf_large = alloc_grid_pair(build_1d_leaf, 64, 1024)
        a_added_small, a_added_large = alloc_grid_pair(build_1d_added, 64, 1024)
        @info "1D stepping allocations (leaf, Added(leaf, Scaled))" a_leaf_small a_leaf_large a_added_small a_added_large
        @test a_leaf_small == a_leaf_large
        @test a_added_small == a_added_large
        @test a_leaf_small ≤ 512
        @test a_added_small ≤ 512
    end

    @testset "inhomogeneous BCs: the explicit RHS is L(u) + boundary_rhs" begin
        # apply! is the homogeneous linear part, so on Dirichlet(1), Dirichlet(2)
        # the heat equation stepped as du = Δu relaxes to zero; the documented
        # idiom adds the lift once per stage and relaxes to the steady state 1 + x.
        g = CartesianGrid(((0.0, 1.0),), (16,); bc=((Dirichlet(1.0), Dirichlet(2.0)),))
        L = laplacian(g)
        b = boundary_rhs(L, g)
        @test any(!iszero, interior(b))
        dt = 0.4 * spacing(g)[1]^2
        nsteps = 4000                                 # t = 6.25 ≫ 1/π², fully relaxed
        exact = interior(set!(scalar_field(g), x -> 1 + x[1]))

        u = scalar_field(g)
        du = similar(u)
        for _ in 1:nsteps
            apply!(du, L, u)
            interior(du) .+= interior(b)
            interior(u) .+= dt .* interior(du)
        end
        err_lift = maximum(abs, interior(u) .- exact)
        @info "explicit steady state with the boundary lift" err_lift
        @test err_lift < 1e-10

        # the prepared entry point is the same homogeneous part: same idiom, same answer
        P = prepare(L)
        up = scalar_field(g)
        for _ in 1:nsteps
            apply!(du, P, up)
            interior(du) .+= interior(b)
            interior(up) .+= dt .* interior(du)
        end
        @test interior(up) == interior(u)

        # without the lift, the same loop integrates the homogeneous problem
        u0 = scalar_field(g)
        for _ in 1:nsteps
            apply!(du, L, u0)
            interior(u0) .+= dt .* interior(du)
        end
        @test maximum(abs, interior(u0)) < 1e-10
    end

    @testset "apply!(du, P, u) on a prepared composed tree" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (8, 6);
            bc=((Dirichlet(), Dirichlet()), (Periodic(), Periodic())),
        )
        κ = set!(scalar_field(g), x -> 1 + x[1])
        K = divergence(g) * scaling(κ) * MatrixFreeOperators.gradient(g)   # Composed
        P = prepare(K, scalar_field(g))

        rng = Random.MersenneTwister(87)
        u = scalar_field(g)
        interior(u) .= rand(rng, local_size(g)...)
        u_before = copy(interior(u))
        du = similar(u)

        # Same numbers as the flat boundary and as the pure path — only the copies differ.
        @test apply!(du, P, u) === du
        y = similar(flatten(u))
        mul!(y, P, flatten(u))
        @test flatten(du) == y
        @test flatten(du) ≈ flatten(apply(K, u))
        @test interior(u) == u_before              # the state is never mutated by the RHS

        # α/β accumulate into the interior of du exactly like mul!'s axpby.
        du0 = rand(rng, local_size(g)...)
        interior(du) .= du0
        apply!(du, P, u, 2.5, -0.5)
        @test interior(du) ≈ 2.5 .* reshape(y, local_size(g)) .- 0.5 .* du0

        # The prepared path reuses the Composed intermediate; the pure path on the
        # same tree allocates it every call — that is why prepare is still worth it
        # for explicit stepping of a *-composed tree, even though mul! is not.
        @inferred apply!(du, P, u, true, false)
        a_prep_small, a_prep_large = alloc_grid_pair(build_2d_prepared, 1, 4)
        a_pure_small, a_pure_large = alloc_grid_pair(build_2d_pure, 1, 4)
        @info "explicit-stepping allocations on a Composed tree" a_prep_small a_prep_large a_pure_small a_pure_large
        @test a_prep_small == a_prep_large
        @test a_prep_small ≤ 512
        # Positive control. The pure tree allocates its Composed intermediate on
        # every call, so its cost scales with the cell count. That is both why
        # prepare is still worth it for explicit stepping of a *-composed tree and
        # the demonstration that the grid-independence assertion above can fail:
        # a per-cell allocation reintroduced into the prepared path would break
        # `a_prep_small == a_prep_large` exactly the way it breaks here.
        @test a_pure_small > 0
        @test a_pure_large > a_pure_small

        # 3D Float32, mixed BCs, Composed + Scaled + Added under one prepare. The
        # default call is bitwise mul!; the α/β call is only ≈ because the tree
        # pushes α/β into each node (Added applies them per branch) while mul!
        # applies them once in interior_to_flat!'s axpby, so the rounding order
        # differs at the eps level. eltype must survive as Float32 throughout.
        g3 = CartesianGrid(
            ((0.0f0, 1.0f0), (0.0f0, 2.0f0), (0.0f0, 1.0f0)), (6, 5, 4);
            bc=((Dirichlet(), Dirichlet()), (Periodic(), Periodic()), (Neumann(), Neumann())),
        )
        κ3 = set!(scalar_field(g3), x -> 1.0f0 + x[1] * x[3])
        K3 = divergence(g3) * scaling(κ3) * MatrixFreeOperators.gradient(g3) + 0.5f0 * laplacian(g3)
        P3 = prepare(K3, scalar_field(g3))
        u3 = scalar_field(g3)
        interior(u3) .= rand(rng, Float32, local_size(g3)...)
        du3 = similar(u3)
        @test apply!(du3, P3, u3) === du3
        @test eltype(du3) === Float32
        y3 = similar(flatten(u3))
        mul!(y3, P3, flatten(u3))
        @test eltype(y3) === Float32
        @test flatten(du3) == y3
        @test flatten(du3) ≈ flatten(apply(K3, u3))
        du30 = rand(rng, Float32, local_size(g3)...)
        interior(du3) .= du30
        apply!(du3, P3, u3, 2.5f0, -0.5f0)
        @test interior(du3) ≈ 2.5f0 .* reshape(y3, local_size(g3)) .- 0.5f0 .* du30
        a_3d_small, a_3d_large = alloc_grid_pair(build_3d_prepared, 1, 2)
        @info "explicit-stepping allocations on a 3D Float32 tree" a_3d_small a_3d_large
        @test a_3d_small == a_3d_large
        # The 3D residual is the largest of the single-grid probes (six ghost
        # planes, two `_dimslice` views each), so its cap is looser than the 512 B
        # the 1D/2D probes carry. It is still grid-independent.
        @test a_3d_small ≤ 4096
        # adjoint through the prepared field-level path: ⟨K u, v⟩ = ⟨u, Kᵀ v⟩
        P3t = prepare(adjoint(K3), scalar_field(g3))
        v3 = scalar_field(g3)
        interior(v3) .= rand(rng, Float32, local_size(g3)...)
        Ku = apply!(similar(u3), P3, u3)
        Ktv = apply!(similar(v3), P3t, v3)
        @test dot(flatten(Ku), flatten(v3)) ≈ dot(flatten(u3), flatten(Ktv))
    end

    @testset "apply!(du, P, u) on a prepared forest" begin
        MFO = MatrixFreeOperators
        bc = ((Dirichlet(), Dirichlet()), (Neumann(), Neumann()))
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=bc)
        bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
        uf = set!(scalar_field(bf), x -> sinpi(x[1]) * cospi(2x[2]) + 0.3 * x[1])
        Df = derivative(bf, 1; order=1)
        Sf = laplacian(bf) + adjoint(Df)           # Added(leaf, PreparedAdjoint) after prepare
        P = prepare(Sf)
        @test P isa PreparedForest

        du = similar(uf)
        @test apply!(du, P, uf) === du
        v = flatten(uf)
        y = similar(v)
        mul!(y, P, v)
        @test flatten(du) == y
        @test flatten(du) ≈ flatten(Sf * copy(uf))

        # accumulating form drives the prepared adjoint scratch, as mul!'s axpby does
        du2 = copy(du)
        apply!(du2, P, uf, 2.0, 3.0)
        @test flatten(du2) ≈ 2.0 .* y .+ 3.0 .* flatten(du)

        # The packed prototype routes prepare to the forest-native kernel sweeps
        # and packed coefficients; the same entry point must give the same numbers.
        up = pack(uf)
        Pp = prepare(Sf, up)
        @test Pp isa PreparedForest
        dp = similar(up)
        @test apply!(dp, Pp, up) === dp
        @test dp isa PackedBlockField
        @test flatten(dp) == y
        yp = similar(v)
        mul!(yp, Pp, flatten(up))
        @test flatten(dp) == yp
        dp2 = copy(dp)
        apply!(dp2, Pp, up, 2.0, 3.0)
        @test flatten(dp2) ≈ 2.0 .* y .+ 3.0 .* flatten(dp)
        # Per-leaf forest sweeps are the one path documented to allocate, and the
        # residual is per leaf by construction, so a constant bound is the wrong
        # shape: use the house `1000 * nleaves` convention (test/forest_prepare.jl,
        # test/forest_packed.jl, test/amr_driver.jl).
        alloc_bound(nl) = 1000 * nl
        a_block = alloc_apply(similar(uf), P, uf)
        a_packed = alloc_apply(similar(up), Pp, up)
        @info "explicit-stepping allocations on a prepared forest" a_block a_packed
        @test a_block ≤ alloc_bound(MFO.nleaves(bf))
        @test a_packed ≤ alloc_bound(MFO.nleaves(bf))

        # Grid independence in forest form: hold the leaf *count* fixed and grow
        # the leaf. 16×16 with blocksize 4 and 32×32 with blocksize 8 are both 16
        # leaves, of 16 and 64 cells respectively, so anything per-cell shows up
        # as a difference while the per-leaf residual cancels.
        nl_small, ab_small, ap_small = forest_step_alloc(16, 4)
        nl_large, ab_large, ap_large = forest_step_alloc(32, 8)
        @info "forest stepping allocations at a fixed leaf count" nl_small nl_large ab_small ab_large ap_small ap_large
        @test nl_small == nl_large == MFO.nleaves(bf)
        @test ab_small == ab_large
        @test ap_small == ap_large

        # tied to the forest generation, exactly like mul!
        refine!(bf, _ -> true)
        @test_throws ArgumentError apply!(du, P, uf)
        @test_throws ArgumentError apply!(dp, Pp, up)
    end

    @testset "apply!(du, P, u) refuses fields off the prototype" begin
        # This entry point hands a user field straight to the prepared tree, so a
        # field on another grid would run as a hybrid: halo_update! and the stencil
        # spacing from P.grid, apply_bc! from x.grid. Same shape, different BCs is
        # the silent case; it must throw rather than return a plausible answer.
        gd = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 6))
        gn = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (8, 6);
            bc=((Neumann(), Neumann()), (Neumann(), Neumann())),
        )
        P = prepare(laplacian(gd))
        u = set!(scalar_field(gd), x -> x[1] + x[2]^2)
        du = similar(u)
        @test apply!(du, P, u) === du

        un = Field(copy(u.data), gn)
        err = try apply!(similar(un), P, un); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("x lives on a different grid", err.msg)
        @test_throws ArgumentError apply!(similar(un), P, u)     # y off the grid
        # a structurally identical CartesianGrid is the same grid (isbits ===)
        u2 = Field(copy(u.data), CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 6)))
        @test apply!(similar(u2), P, u2) |> flatten == flatten(du)

        # same grid, wrong element type: the prepared scratch is Float64
        u32 = Field(Float32.(u.data), gd)
        err32 = try apply!(similar(u32), P, u32); nothing catch e; e end
        @test err32 isa ArgumentError
        @test occursin("Float32", err32.msg) && occursin("Float64", err32.msg)
        @test_throws ArgumentError apply!(similar(u32), P, u)    # y of the wrong eltype
        # rank-changing output: a scalar y for a Gradient's SVector output
        Pg = prepare(MatrixFreeOperators.gradient(gd), scalar_field(gd))
        @test_throws ArgumentError apply!(similar(u), Pg, u)
        gu = similar(vector_field(gd))
        @test apply!(gu, Pg, u) === gu

        # forest form: a second forest over the same extent is a different grid
        bc = ((Dirichlet(), Dirichlet()), (Neumann(), Neumann()))
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=bc)
        bf = BlockForest(g; blocksize=(4, 4), maxlevel=1)
        bf2 = BlockForest(g; blocksize=(4, 4), maxlevel=1)
        Pf = prepare(laplacian(bf))
        uf = set!(scalar_field(bf), x -> x[1])
        @test apply!(similar(uf), Pf, uf) isa BlockField
        uf2 = set!(scalar_field(bf2), x -> x[1])
        @test_throws ArgumentError apply!(similar(uf2), Pf, uf2)
        @test_throws ArgumentError apply!(similar(uf2), Pf, uf)
    end

    @testset "flat mul! RHS still works (the Krylov boundary as an RHS)" begin
        g = CartesianGrid(((0.0, 2π),), (64,); bc=((Periodic(), Periodic()),))
        P = prepare(laplacian(g))
        f!(du, u) = mul!(du, P, u)

        u = flatten(set!(scalar_field(g), x -> sin(x[1])))
        dt = 0.005
        nsteps = 200
        rk4!(f!, u, dt, nsteps)

        t = dt * nsteps
        uref = exp(-t) .* flatten(set!(scalar_field(g), x -> sin(x[1])))
        @test maximum(abs, u .- uref) < 1e-3
    end
end
