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
end
