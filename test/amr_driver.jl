@testset "AMR driver (regrid! + transfer)" begin
    MFO = MatrixFreeOperators
    lin = x -> 1 + 2x[1] - 3x[2]

    dirbc = ((Dirichlet(), Dirichlet()), (Dirichlet(), Dirichlet()))
    mixbc = ((Dirichlet(), Dirichlet()), (Neumann(), Neumann()))
    perbc = ((Periodic(), Periodic()), (Periodic(), Periodic()))

    # Volume-weighted L1 distance between a transferred field and the exact
    # function sampled on the current leaf set — the forest error norm.
    function l1_error(u, bf, fun)
        e = 0.0
        for i in 1:MFO.nleaves(bf)
            key = bf.forest.leaves[i]
            sp = MFO._leaf_spacing(bf, key.level)
            lg = MFO.leaf_grid(bf, i)
            ref = set!(scalar_field(lg), fun)
            r = collect(interior(MFO.block(u, i))) .- collect(interior(ref))
            e += sum(abs, r) * prod(sp)
        end
        return e
    end

    # Max-norm distance to the exact function — for transfers that must be exact.
    function max_error(u, bf, fun)
        maximum(1:MFO.nleaves(bf)) do i
            lg = MFO.leaf_grid(bf, i)
            ref = set!(similar(MFO.block(u, i)), fun)
            maximum(abs, interior(MFO.block(u, i)) .- interior(ref))
        end
    end

    @testset "marking and topology semantics" begin
        base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=dirbc)
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=2)   # 4×4 roots
        u = set!(scalar_field(bf), x -> x[1])

        # refine the right column (blocks whose data exceeds the threshold)
        u = regrid!(u; refine=b -> maximum(interior(b)) > 0.75)
        @test MFO.nleaves(bf) == 16 - 4 + 16
        for key in bf.forest.leaves
            ext = MFO._leaf_extent(bf, key)
            @test (key.level == 1) == (ext[1][1] >= 0.75)
        end

        # partially marked families do not coarsen; fully marked families do
        u2 = regrid!(u; refine=Returns(false), coarsen=b -> b.grid.extent[1][1] > 0.8)
        @test u2 === u                                  # 2 of 4 children marked: no-op
        u = regrid!(u; refine=Returns(false), coarsen=Returns(true))
        @test MFO.nleaves(bf) == 16                     # complete families collapse
        @test bf.forest.uniform[]

        # refine wins: the refine-marked child splits, so its family cannot coarsen
        u = regrid!(u; refine=b -> b.grid.extent[1][1] < 0.2 && b.grid.extent[2][1] < 0.2)
        n1 = MFO.nleaves(bf)
        @test n1 == 16 - 1 + 4
        corner = b -> b.grid.extent[1][1] < 0.1 && b.grid.extent[2][1] < 0.1
        u = regrid!(u; refine=corner, coarsen=Returns(true))
        @test maximum(k -> k.level, bf.forest.leaves) == 2   # corner child split again
        @test !bf.forest.uniform[]

        # maxlevel saturation: marks at maxlevel are ignored (may still be a no-op)
        bf0 = BlockForest(base; blocksize=(4, 4), maxlevel=0)
        u0 = scalar_field(bf0)
        @test regrid!(u0; refine=Returns(true)) === u0
        @test bf0.forest.generation[] == 1               # only the constructor commit
    end

    @testset "no-op regrid preserves fields and prepared operators" begin
        base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8))
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=1)
        u = set!(scalar_field(bf), lin)
        w = scalar_field(bf)
        P = prepare(laplacian(bf), u)
        gen = bf.forest.generation[]
        out = regrid!(u, w; refine=Returns(false), coarsen=Returns(false))
        @test out === (u, w)
        @test bf.forest.generation[] == gen
        v = flatten(u)
        @test mul!(similar(v), P, v) isa Vector      # prepared operator still valid
    end

    @testset "transfer exactness on linears" begin
        for bc in (dirbc, mixbc, perbc)
            base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=bc)
            bf = BlockForest(base; blocksize=(4, 4), maxlevel=2)
            u = set!(scalar_field(bf), lin)

            # refine a band touching the physical boundary: one-sided edge taps and
            # boundary-adjacent blocks are exercised for every BC kind
            u = regrid!(u; refine=b -> b.grid.extent[1][1] < 0.5)
            @test !bf.forest.uniform[]
            @test max_error(u, bf, lin) < 1e-13

            # coarsen back to uniform: conservative child mean is exact on linears
            u = regrid!(u; refine=Returns(false), coarsen=Returns(true))
            @test bf.forest.uniform[]
            @test max_error(u, bf, lin) < 1e-13
        end
    end

    @testset "balance-cascade leaves are transferred (key-based, not mark-based)" begin
        base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=dirbc)
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=2)
        u = set!(scalar_field(bf), lin)
        u = regrid!(u; refine=b -> b.grid.extent[1][1] < 0.25 && b.grid.extent[2][1] < 0.25)
        gen1 = bf.forest.generation[]
        # corner → level 2 forces balance! to refine unmarked neighbors to level 1
        u = regrid!(u; refine=b -> b.grid.extent[1][2] < 0.26 && b.grid.extent[2][2] < 0.26)
        @test bf.forest.generation[] > gen1
        @test maximum(k -> k.level, bf.forest.leaves) == 2
        @test length(unique(map(k -> k.level, bf.forest.leaves))) == 3   # cascade happened
        @test max_error(u, bf, lin) < 1e-13
    end

    @testset "vector fields and varargs transfer consistently" begin
        base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=dirbc)
        vfun = x -> SVector(x[1] - 2x[2], 1 + x[2])

        bf = BlockForest(base; blocksize=(4, 4), maxlevel=1)
        u = set!(scalar_field(bf), lin)
        w = set!(vector_field(bf), vfun)
        band = b -> b.grid.extent[1][1] < 0.5
        u2, w2 = regrid!(u, w; refine=band)
        @test u2 isa BlockField && w2 isa BlockField
        werr = maximum(1:MFO.nleaves(bf)) do i
            ref = set!(similar(MFO.block(w2, i)), vfun)
            maximum(norm, interior(MFO.block(w2, i)) .- interior(ref))
        end
        @test werr < 1e-13

        # the same regrid of a lone field gives bit-identical blocks
        bf3 = BlockForest(base; blocksize=(4, 4), maxlevel=1)
        u3 = set!(scalar_field(bf3), lin)
        u3 = regrid!(u3; refine=band)
        @test all(i -> u3.blocks[i] == u2.blocks[i], 1:MFO.nleaves(bf3))
    end

    @testset "transfer converges at 2nd order (volume-weighted L1)" begin
        fun = x -> sin(x[1]) * sin(x[2])
        function refine_transfer_error(ncells)
            base = CartesianGrid(((0.0, 2π), (0.0, 2π)), (ncells, ncells); bc=perbc)
            bf = BlockForest(base; blocksize=(4, 4), maxlevel=1)
            u = set!(scalar_field(bf), fun)
            u = regrid!(u; refine=b -> b.grid.extent[1][1] < 1.6)
            @test !bf.forest.uniform[]
            return l1_error(u, bf, fun)
        end
        function coarsen_transfer_error(ncells)
            base = CartesianGrid(((0.0, 2π), (0.0, 2π)), (ncells, ncells); bc=perbc)
            bf = BlockForest(base; blocksize=(4, 4), maxlevel=1)
            ind = set!(scalar_field(bf), x -> x[1] < 1.6 ? 1.0 : 0.0)
            regrid!(ind; refine=b -> maximum(interior(b)) > 0.5)
            u = set!(scalar_field(bf), fun)
            u = regrid!(u; refine=Returns(false), coarsen=Returns(true))
            @test bf.forest.uniform[]
            return l1_error(u, bf, fun)
        end
        for err in (refine_transfer_error, coarsen_transfer_error)
            e16, e32 = err(16), err(32)
            order = log2(e16 / e32)
            @test e32 < e16
            @test order ≥ 1.9
        end
    end

    @testset "staleness and argument errors" begin
        base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=dirbc)
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=1)
        u = set!(scalar_field(bf), lin)
        P = prepare(laplacian(bf), u)
        v = flatten(u)
        u2 = regrid!(u; refine=b -> b.grid.extent[1][1] < 0.5)
        @test_throws ArgumentError flatten(u)                    # stale field
        @test_throws ArgumentError mul!(similar(v), P, v)        # stale prepared op
        @test_throws ArgumentError regrid!(u; refine=Returns(false))  # stale input
        @test flatten(u2) isa Vector                             # returned field is live

        other = BlockForest(base; blocksize=(4, 4), maxlevel=1)
        @test_throws ArgumentError regrid!(u2, scalar_field(other); refine=Returns(false))

        base9 = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (9, 9); bc=dirbc)
        bf9 = BlockForest(base9; blocksize=(3, 3), maxlevel=1)   # odd blocksize
        @test_throws ArgumentError regrid!(scalar_field(bf9); refine=Returns(false))
    end

    @testset "adjoint identity on a driver-adapted forest" begin
        rng = Random.MersenneTwister(29)
        base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=mixbc)
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=3)
        ind = set!(scalar_field(bf), x -> x[1] < 0.5 && x[2] < 0.5 ? 1.0 : 0.0)
        ind = regrid!(ind; refine=b -> maximum(interior(b)) > 0.5)
        ind2 = set!(scalar_field(bf), x -> x[1] < 0.2 && x[2] < 0.2 ? 1.0 : 0.0)
        regrid!(ind2; refine=b -> maximum(interior(b)) > 0.5)    # levels 0–2
        @test maximum(k -> k.level, bf.forest.leaves) == 2

        for L in (laplacian(bf), derivative(bf, 1; order=1))
            x = scalar_field(bf)
            y = scalar_field(bf)
            for i in 1:MFO.nleaves(bf)
                interior(MFO.block(x, i)) .= rand(rng, bf.blocksize...)
                interior(MFO.block(y, i)) .= rand(rng, bf.blocksize...)
            end
            Lx = apply(L, copy(x))
            Lty = apply_adjoint!(scalar_field(bf), L, copy(y), bf)
            ip1 = sum(
                i -> dot(
                    collect(interior(MFO.block(Lx, i))),
                    collect(interior(MFO.block(y, i))),
                ),
                1:MFO.nleaves(bf),
            )
            ip2 = sum(
                i -> dot(
                    collect(interior(MFO.block(x, i))),
                    collect(interior(MFO.block(Lty, i))),
                ),
                1:MFO.nleaves(bf),
            )
            @test ip1 ≈ ip2
        end

        G = MFO.gradient(bf)
        u = scalar_field(bf)
        w = vector_field(bf)
        for i in 1:MFO.nleaves(bf)
            interior(MFO.block(u, i)) .= rand(rng, bf.blocksize...)
            interior(MFO.block(w, i)) .=
                SVector.(rand(rng, bf.blocksize...), rand(rng, bf.blocksize...))
        end
        Gu = apply(G, copy(u))
        Gtw = apply_adjoint!(scalar_field(bf), G, copy(w), bf)
        ip1 = sum(
            i -> dot(collect(interior(MFO.block(Gu, i))), collect(interior(MFO.block(w, i)))),
            1:MFO.nleaves(bf),
        )
        ip2 = sum(
            i -> dot(collect(interior(MFO.block(u, i))), collect(interior(MFO.block(Gtw, i)))),
            1:MFO.nleaves(bf),
        )
        @test ip1 ≈ ip2
    end

    @testset "Δ action keeps 2nd order on a driver-adapted forest (L1)" begin
        function l1_action_error(ncells)
            base = CartesianGrid(((0.0, 2π), (0.0, 2π)), (ncells, ncells); bc=perbc)
            bf = BlockForest(base; blocksize=(4, 4), maxlevel=2)
            ind = set!(scalar_field(bf), x -> x[1] < 1.6 ? 1.0 : 0.0)
            regrid!(ind; refine=b -> maximum(interior(b)) > 0.5)
            @test !bf.forest.uniform[]
            u = set!(scalar_field(bf), x -> sin(x[1]) * sin(x[2]))
            Lu = laplacian(bf) * u
            e = 0.0
            for i in 1:MFO.nleaves(bf)
                sp = MFO._leaf_spacing(bf, bf.forest.leaves[i].level)
                r = collect(interior(MFO.block(Lu, i))) .+
                    2 .* collect(interior(MFO.block(u, i)))
                e += sum(abs, r) * prod(sp)
            end
            return e
        end
        e_coarse = l1_action_error(16)
        e_fine = l1_action_error(32)
        order = log2(e_coarse / e_fine)
        @test e_fine < e_coarse
        @test order ≥ 1.9
    end

    @testset "acceptance: adaptive Poisson solve tracks the solution feature" begin
        # −Δu = f on (0,1)², homogeneous Dirichlet, manufactured Gaussian bump
        # u* = exp(−r²/s) at (0.7, 0.3): negligible (≈1e-8) at the boundary. The
        # forest Laplacian is nonsymmetric on an adapted forest ⇒ GMRES, not CG.
        c = (0.7, 0.3)
        s = 0.005
        uexact = x -> exp(-((x[1] - c[1])^2 + (x[2] - c[2])^2) / s)
        f = x -> (4 / s - 4 * ((x[1] - c[1])^2 + (x[2] - c[2])^2) / s^2) * uexact(x)

        base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (32, 32); bc=dirbc)
        bf = BlockForest(base; blocksize=(8, 8), maxlevel=2)
        u = scalar_field(bf)

        errs = Float64[]
        for cycle in 1:3
            P = prepare(laplacian(bf), u)
            rhs = .-flatten(set!(scalar_field(bf), f))   # Δu = −f
            sol, stats = Krylov.gmres(P, rhs; rtol=1e-10)
            @test stats.solved
            flat_to_interior!(u, sol)
            push!(errs, l1_error(u, bf, uexact))
            cycle == 3 && break
            η = MFO.gradient(bf) * u
            η, u = regrid!(η, u; refine=b -> maximum(norm, interior(b)) > 1.0)
        end

        # refinement localizes at the bump ...
        @test maximum(k -> k.level, bf.forest.leaves) == 2
        for key in bf.forest.leaves
            if key.level == 2
                center = MFO.leaf_center(bf, key)
                @test hypot(center[1] - c[1], center[2] - c[2]) < 0.3
            end
        end
        # ... with far fewer DOFs than the uniformly-fine grid
        @test MFO.nleaves(bf) < (16 * 4^2) ÷ 2           # 16 roots at 4ˡᵉᵛᵉˡ leaves each
        # ... and the error drops every cycle
        @test errs[2] < errs[1] && errs[3] < errs[2]
        @test errs[3] < errs[1] / 4
    end

    @testset "re-prepared mul! stays within the steady-state allocation budget" begin
        base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=mixbc)
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=1)
        u = set!(scalar_field(bf), lin)
        u = regrid!(u; refine=b -> b.grid.extent[1][1] < 0.5)
        P = prepare(laplacian(bf), u)
        v = flatten(u)
        out = similar(v)
        function alloc_mul(P, out, v)
            mul!(out, P, v)
            mul!(out, P, v)
            return @allocated mul!(out, P, v)
        end
        @test alloc_mul(P, out, v) ≤ 1000 * MFO.nleaves(bf)   # forest_prepare.jl bound
    end

    @testset "invalid transfer policies fail before regridding" begin
        base = CartesianGrid(((0.0, 1.0),), (8,))
        bf = BlockForest(base; blocksize=(4,), maxlevel=1)
        u = set!(scalar_field(bf), x -> x[1])
        p = pack(u)
        generation = bf.forest.generation[]
        keys = copy(bf.forest.leaves)
        values = flatten(u)

        @testset "policy $(repr(invalid))" for invalid in (Conservative, :conservative, nothing)
            @test_throws TypeError scalar_field(bf; transfer=invalid)
            @test_throws TypeError vector_field(bf; transfer=invalid)
            @test_throws MethodError with_transfer(u, invalid)
            @test_throws MethodError with_transfer(p, invalid)
            @test_throws TypeError BlockField{Center,typeof(invalid)}(u.blocks, bf)
            @test_throws TypeError PackedBlockField{Center,typeof(invalid)}(p.data, p.levels, bf)
            @test bf.forest.generation[] == generation
            @test bf.forest.leaves == keys
            @test flatten(u) == values
        end

        transferred = regrid!(with_transfer(u, Conservative()); refine=Returns(true))
        @test MFO.nleaves(bf) > length(keys)
        @test MFO._transfer_policy(transferred) === Conservative()
    end

    @testset "derived fields preserve policy and generation: $pol" for pol in (
        Interpolated(), Conservative(), SlopeLimited()
    )
        base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8))
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=1)
        u = set!(vector_field(bf; transfer=pol), x -> SVector(x[1], 2x[2]))
        p = pack(u)
        generation = u.generation

        @testset "stale=$stale" for stale in (false, true)
            stale && refine!(bf, Returns(true))
            b = @inferred BlockField{Center,typeof(pol)}(u.blocks, bf, generation)
            packed = @inferred PackedBlockField{Center,typeof(pol)}(p.data, p.levels, bf, generation)
            @test b.blocks === u.blocks
            @test packed.data === p.data
            @test packed.levels === p.levels
            @test b.generation == packed.generation == generation

            @testset "layout=$(nameof(typeof(f)))" for f in (b, packed)
                @testset "$name" for (name, derived, expected_type) in (
                    ("copy", copy(f), eltype(f)),
                    ("similar", similar(f), eltype(f)),
                    ("similar Float32", similar(f, SVector{2,Float32}), SVector{2,Float32}),
                    ("component", component(f, 1), Float64),
                    ("with_transfer", with_transfer(f, pol), eltype(f)),
                    ("adapt", Adapt.adapt(Array, f), eltype(f)),
                )
                    @test MFO._transfer_policy(derived) === pol
                    @test derived.generation == generation
                    @test eltype(derived) == expected_type
                    if stale
                        @test_throws ArgumentError MFO.block(derived, 1)
                    else
                        @test MFO._require_current(derived) === nothing
                        if name in ("copy", "with_transfer", "adapt")
                            @test flatten(derived) == flatten(f)
                        end
                    end
                end
            end
        end
    end

    @testset "minmod preserves small slopes: $T" for T in (Float32, Float64)
        tiny = T === Float32 ? T(1e-25) : T(1e-200)
        @test MFO._minmod(tiny, 2tiny) === tiny
        @test MFO._minmod(2tiny, tiny) === tiny
        @test MFO._minmod(-tiny, -2tiny) === -tiny
        @test iszero(MFO._minmod(tiny, -tiny))
        @test iszero(MFO._minmod(zero(T), tiny))
        @test iszero(MFO._minmod(-tiny, zero(T)))
        @test MFO._minmod(SVector(tiny, -tiny), SVector(2tiny, tiny)) == SVector(tiny, zero(T))

        # Changing field units must not turn a sloped reconstruction into injection.
        base = CartesianGrid(((zero(T), one(T)),), (8,))
        bf = BlockForest(base; blocksize=(4,), maxlevel=1)
        u = set!(scalar_field(bf, T; transfer=SlopeLimited()), x -> x[1])
        v = set!(scalar_field(bf, T; transfer=SlopeLimited()), x -> tiny * x[1])
        u, v = regrid!(u, v; refine=Returns(true))
        @test isapprox(flatten(v) ./ tiny, flatten(u); rtol=8eps(T))
    end

    # Volume-weighted mass Σ V·u and its scale Σ V·|u| — the conserved quantity
    # the transfer policies are about (the forest_diffusion.jl defect idiom).
    function field_mass(u, bf)
        total = 0.0
        scale = 0.0
        for i in 1:MFO.nleaves(bf)
            V = prod(MFO._leaf_spacing(bf, bf.forest.leaves[i].level))
            ui = collect(interior(MFO.block(u, i)))
            total += V * sum(ui)
            scale += V * sum(abs, ui)
        end
        return total, scale
    end

    # Curvature everywhere, so the Interpolated stencil's ⅛·δ²u child-mean defect
    # cannot hide (it vanishes on linears).
    curved = x -> sin(3 * x[1]) * cos(2 * x[2]) + x[1]^2 + 0.5 * x[2]

    @testset "Σ V·u across regrid: conservative policies hold, Interpolated does not" begin
        for (name, pol) in (("Conservative", Conservative()), ("SlopeLimited", SlopeLimited()))
            base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=mixbc)
            bf = BlockForest(base; blocksize=(4, 4), maxlevel=2)
            u = set!(scalar_field(bf; transfer=pol), curved)
            m0, s0 = field_mass(u, bf)
            # refine a corner, then its inside — the second regrid forces balance!
            # to refine leaves the criteria never marked (issue #59's acceptance:
            # conservation must include balance-induced refinements)
            u = regrid!(u; refine=b -> b.grid.extent[1][1] < 0.25 && b.grid.extent[2][1] < 0.25)
            m1, s1 = field_mass(u, bf)
            @test abs(m1 - m0) ≤ 1e3 * eps() * s0
            gen1 = bf.forest.generation[]
            u = regrid!(u; refine=b -> b.grid.extent[1][2] < 0.26 && b.grid.extent[2][2] < 0.26)
            @test bf.forest.generation[] > gen1
            @test length(unique(map(k -> k.level, bf.forest.leaves))) == 3   # cascade fired
            m2, _ = field_mass(u, bf)
            @test abs(m2 - m0) ≤ 1e3 * eps() * s0
            # policy survived both transfers — the returned fields, not the input,
            # carry it (a silent reset would conserve once and then stop)
            @test MFO._transfer_policy(u) === pol
            # coarsen everything back: the 2⁻ᴺ child mean is conservative for
            # every policy, and the round trip must still hold the total
            u = regrid!(u; refine=Returns(false), coarsen=Returns(true))
            u = regrid!(u; refine=Returns(false), coarsen=Returns(true))
            m3, _ = field_mass(u, bf)
            @test abs(m3 - m0) ≤ 1e3 * eps() * s0
        end

        # negative control: the linear-exact default interpolates with side-biased
        # slopes and must NOT conserve on curved data — the documented property
        # that makes conservation an explicit opt-in, and the proof this testset
        # has teeth
        base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=mixbc)
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=2)
        u = set!(scalar_field(bf), curved)
        m0, s0 = field_mass(u, bf)
        u = regrid!(u; refine=b -> b.grid.extent[1][1] < 0.25 && b.grid.extent[2][1] < 0.25)
        m1, _ = field_mass(u, bf)
        @test abs(m1 - m0) > 1e6 * eps() * s0
    end

    # The reconstruction is written over Val(N); 2D cannot tell a per-axis index
    # slip from a correct one on the axis it never varies, and the 2ᴺ-children
    # telescoping has to hold with N = 1 (one child pair) and N = 3 (eight
    # children) alike. Same sequence as above — refine, cascade, coarsen back —
    # with dimension-generic predicates and a curved function in every axis.
    @testset "Σ V·u across regrid holds in 1D and 3D" begin
        cases = (
            (
                CartesianGrid(((0.0, 1.0),), (16,); bc=((Dirichlet(), Neumann()),)),
                (4,),
                x -> sin(3 * x[1]) + x[1]^2,
            ),
            (
                CartesianGrid(
                    ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), (8, 8, 8);
                    bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann()), (Periodic(), Periodic())),
                ),
                (4, 4, 4),
                x -> sin(3 * x[1]) * cos(2 * x[2]) + x[3]^2 + 0.5 * x[2],
            ),
        )
        corner(b) = all(e -> e[1] < 0.25, b.grid.extent)
        inner(b) = all(e -> e[2] < 0.26, b.grid.extent)
        for (base, bs, fun) in cases, pol in (Conservative(), SlopeLimited())
            bf = BlockForest(base; blocksize=bs, maxlevel=2)
            u = set!(scalar_field(bf; transfer=pol), fun)
            nl0 = MFO.nleaves(bf)
            m0, s0 = field_mass(u, bf)
            u = regrid!(u; refine=corner)
            @test MFO.nleaves(bf) > nl0
            m1, _ = field_mass(u, bf)
            @test abs(m1 - m0) ≤ 1e3 * eps() * s0
            u = regrid!(u; refine=inner)
            @test length(unique(map(k -> k.level, bf.forest.leaves))) == 3   # cascade fired
            m2, _ = field_mass(u, bf)
            @test abs(m2 - m0) ≤ 1e3 * eps() * s0
            @test MFO._transfer_policy(u) === pol
            u = regrid!(u; refine=Returns(false), coarsen=Returns(true))
            u = regrid!(u; refine=Returns(false), coarsen=Returns(true))
            @test MFO.nleaves(bf) == nl0
            m3, _ = field_mass(u, bf)
            @test abs(m3 - m0) ≤ 1e3 * eps() * s0

            # negative control in the same dimension
            bfi = BlockForest(base; blocksize=bs, maxlevel=2)
            ui = set!(scalar_field(bfi), fun)
            mi0, si0 = field_mass(ui, bfi)
            ui = regrid!(ui; refine=corner)
            mi1, _ = field_mass(ui, bfi)
            @test abs(mi1 - mi0) > 1e6 * eps() * si0
        end
    end

    @testset "conservative policies: exactness, bounds, vectors, round-trips" begin
        base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=dirbc)

        # Conservative is exact on linears everywhere (centered and one-sided
        # slopes both reproduce a linear), like the default
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=1)
        u = set!(scalar_field(bf; transfer=Conservative()), lin)
        u = regrid!(u; refine=b -> b.grid.extent[1][1] < 0.5)
        @test max_error(u, bf, lin) < 1e-13

        # SlopeLimited introduces no new extrema: a steep front stays within the
        # source's global bounds (minmod interior, zero slope at block edges)
        bf2 = BlockForest(base; blocksize=(4, 4), maxlevel=1)
        step = x -> x[1] < 0.4 ? 0.0 : 1.0
        v = set!(scalar_field(bf2; transfer=SlopeLimited()), step)
        lo, hi = extrema(
            reduce(vcat, [vec(collect(interior(MFO.block(v, i)))) for i in 1:MFO.nleaves(bf2)])
        )
        v = regrid!(v; refine=b -> true)
        vals = reduce(vcat, [vec(collect(interior(MFO.block(v, i)))) for i in 1:MFO.nleaves(bf2)])
        @test minimum(vals) ≥ lo - 1e-14
        @test maximum(vals) ≤ hi + 1e-14
        m_step, s_step = field_mass(v, bf2)   # and it still conserved
        # (mass computed post-refine equals pre-refine: recompute the reference)
        bf2b = BlockForest(base; blocksize=(4, 4), maxlevel=1)
        v0 = set!(scalar_field(bf2b), step)
        m0_step, _ = field_mass(v0, bf2b)
        @test abs(m_step - m0_step) ≤ 1e3 * eps() * s_step

        # SVector state conserves componentwise
        bf3 = BlockForest(base; blocksize=(4, 4), maxlevel=1)
        w = set!(
            vector_field(bf3; transfer=Conservative()),
            x -> SVector(sin(3 * x[1]) + x[2]^2, cos(2 * x[2]) - x[1]^2),
        )
        msum(f, bfx) = sum(
            i -> prod(MFO._leaf_spacing(bfx, bfx.forest.leaves[i].level)) .*
                 sum(collect(interior(MFO.block(f, i)))),
            1:MFO.nleaves(bfx),
        )
        mv0 = msum(w, bf3)
        w = regrid!(w; refine=b -> b.grid.extent[1][1] < 0.5)
        mv1 = msum(w, bf3)
        @test maximum(abs, mv1 - mv0) ≤ 1e-12

        # the policy rides pack/unpack and with_transfer
        bf4 = BlockForest(base; blocksize=(4, 4), maxlevel=1)
        c = scalar_field(bf4; transfer=Conservative())
        @test MFO._transfer_policy(unpack(pack(c))) === Conservative()
        @test MFO._transfer_policy(with_transfer(scalar_field(bf4), SlopeLimited())) ===
            SlopeLimited()
        @test MFO._transfer_policy(scalar_field(bf4)) === Interpolated()
        # shared storage: with_transfer copies nothing
        plain = scalar_field(bf4)
        @test with_transfer(plain, Conservative()).blocks === plain.blocks
    end

    @testset "Conservative transfer converges at 2nd order (volume-weighted L1)" begin
        smooth = x -> sin(2π * x[1]) * cos(2π * x[2])
        function refine_l1(ncells)
            base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (ncells, ncells); bc=perbc)
            bf = BlockForest(base; blocksize=(4, 4), maxlevel=1)
            u = set!(scalar_field(bf; transfer=Conservative()), smooth)
            u = regrid!(u; refine=b -> b.grid.extent[1][1] < 0.5)
            return l1_error(u, bf, smooth)
        end
        e16 = refine_l1(16)
        e32 = refine_l1(32)
        @test e32 < e16
        @test log2(e16 / e32) ≥ 1.9
    end
end
