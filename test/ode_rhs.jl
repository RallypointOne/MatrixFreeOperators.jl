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

        # A leaf needs no prepare to step allocation-free, nor does a +/scalar
        # combination of leaves — exactly zero bytes, as the docstrings promise.
        du = similar(u)
        @test alloc_apply(du, L, u) == 0
        A = laplacian(g) + 2.0 * derivative(g, 1; order=1)          # Added(leaf, Scaled)
        @test alloc_apply(du, A, u) == 0
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
        a_prepared = alloc_apply(du, P, u)
        a_pure = alloc_apply(du, K, u)
        @info "explicit-stepping allocations on a Composed tree" a_prepared a_pure
        @test a_prepared == 0
        @test a_pure > 0

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
        @test alloc_apply(du3, P3, u3) == 0
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
        # per-leaf forest sweeps are the one path documented to allocate; keep a
        # loose bound here and report the measured value
        a_block = alloc_apply(similar(uf), P, uf)
        a_packed = alloc_apply(similar(up), Pp, up)
        @info "explicit-stepping allocations on a prepared forest" a_block a_packed
        @test a_block ≤ 512
        @test a_packed ≤ 512

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
