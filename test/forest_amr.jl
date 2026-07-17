@testset "Coarse–fine interfaces (non-uniform forest)" begin
    MFO = MatrixFreeOperators

    # Ghost-cell center along one dim for padded index p (h = 1): lo + (p − 1.5)·sp.
    ghost_center(ext, sp, idx) = ntuple(k -> ext[k][1] + (idx[k] - 1.5) * sp[k], length(idx))

    # Walk every coarse–fine face and hand each interface-ghost cell (tangential
    # interior band) to `check(value, center)` — topology recomputed independently
    # of the schedule.
    function foreach_cf_ghost(check, u, bf)
        n = bf.blocksize
        for (i, K) in enumerate(bf.forest.leaves)
            ext = MFO._leaf_extent(bf, K)
            sp = MFO._leaf_spacing(bf, K.level)
            for d in 1:2, side in (-1, 1)
                nbr = MFO.face_neighbor(bf.forest, K, d, side)
                (nbr === nothing || MFO.is_leaf(bf.forest, nbr)) && continue
                gp = side == -1 ? 1 : n[d] + 2
                t = d == 1 ? 2 : 1
                for jt in 2:(n[t] + 1)
                    idx = d == 1 ? (gp, jt) : (jt, gp)
                    check(u.blocks[i][idx...], ghost_center(ext, sp, idx))
                end
            end
        end
    end

    # Quadratic exactness is the property that guarantees 2nd order: every weight
    # set (normal parabola, centered/shifted tangential stencils, flux-matching
    # restriction) reproduces a full quadratic — including cross terms — exactly.
    quad = x -> 1 + 2x[1] - x[2] + 3x[1]^2 - x[1] * x[2] + 0.5x[2]^2

    @testset "quadratic exactness of the coarse–fine ghost fill" begin
        base = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (16, 16);
            bc=((Dirichlet(), Dirichlet()), (Dirichlet(), Dirichlet())),
        )
        # interior refined region, and a region touching the physical boundary
        # (an accidental BC-ghost read would break exactness loudly)
        for predicate in (
            x -> 0.25 < x[1] < 0.75 && 0.25 < x[2] < 0.75,
            x -> x[1] < 0.25 && x[2] < 0.25,
        )
            bf = BlockForest(base; blocksize=(4, 4), maxlevel=2)
            refine!(bf, predicate)
            @test !bf.forest.uniform[]
            u = set!(scalar_field(bf), quad)
            halo_update!(u, bf)
            nchecked = 0
            foreach_cf_ghost(u, bf) do val, center
                @test val ≈ quad(center) atol = 1e-12
                nchecked += 1
            end
            @test nchecked > 0
        end
    end

    @testset "quadratic exactness across three levels (2:1 graded)" begin
        base = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (16, 16);
            bc=((Neumann(), Neumann()), (Dirichlet(), Dirichlet())),
        )
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=3)
        refine!(bf, x -> x[1] < 0.5 && x[2] < 0.5)
        refine!(bf, x -> x[1] < 0.2 && x[2] < 0.2)   # levels 0–2, balance! keeps 2:1
        @test minimum(k -> k.level, bf.forest.leaves) == 0
        @test maximum(k -> k.level, bf.forest.leaves) == 2
        u = set!(scalar_field(bf), quad)
        halo_update!(u, bf)
        foreach_cf_ghost(u, bf) do val, center
            @test val ≈ quad(center) atol = 1e-12
        end
    end

    @testset "Δ(sin·sin) keeps 2nd order across refinement interfaces (L1 action)" begin
        # Volume-weighted L1 of (Δ_h u + 2u) on a periodic domain with a fixed
        # refined physical band; the band edge is a coarse–fine interface and the
        # wrap face is a wrapped coarse–fine interface. Deliberately not max-norm:
        # the pointwise action error at interface-adjacent cells is O(h) by design
        # for this scheme family — 2nd order holds in (volume-weighted) L1 of the
        # action and in the solution.
        function l1_action_error(ncells)
            bcp = ((Periodic(), Periodic()), (Periodic(), Periodic()))
            base = CartesianGrid(((0.0, 2π), (0.0, 2π)), (ncells, ncells); bc=bcp)
            bf = BlockForest(base; blocksize=(4, 4), maxlevel=2)
            # refine the fixed physical band x < π/2 (whole root columns at both
            # resolutions): CF interface at x = π/2 and a wrapped one at x = 0
            refine!(bf, x -> x[1] < 1.6)
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

    @testset "adjoint identity ⟨Lx,y⟩ = ⟨x,Lᵀy⟩ on a non-uniform forest" begin
        rng = Random.MersenneTwister(13)
        bcs = [
            ((Periodic(), Periodic()), (Periodic(), Periodic())),
            ((Dirichlet(), Dirichlet()), (Dirichlet(), Dirichlet())),
            ((Neumann(), Neumann()), (Neumann(), Neumann())),
            ((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
        ]
        for bc in bcs
            base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=bc)
            bf = BlockForest(base; blocksize=(4, 4), maxlevel=3)
            refine!(bf, x -> x[1] < 0.5 && x[2] < 0.5)
            refine!(bf, x -> x[1] < 0.2 && x[2] < 0.2)   # levels 0–2
            ops = (
                laplacian(bf),                  # self-adjoint only on a uniform forest
                derivative(bf, 1; order=1),
                derivative(bf, 2; order=2),
            )
            for L in ops
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
            # rank-changer with a vector cotangent across a coarse–fine interface
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
                i -> dot(
                    collect(interior(MFO.block(Gu, i))), collect(interior(MFO.block(w, i)))
                ),
                1:MFO.nleaves(bf),
            )
            ip2 = sum(
                i -> dot(
                    collect(interior(MFO.block(u, i))), collect(interior(MFO.block(Gtw, i)))
                ),
                1:MFO.nleaves(bf),
            )
            @test ip1 ≈ ip2
        end
    end

    @testset "self-adjointness is grid-aware; prepared transpose is structural" begin
        base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8))
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=2)
        @test isselfadjoint(laplacian(bf))               # uniform: halo copy is symmetric
        refine!(bf, x -> x[1] < 0.5 && x[2] < 0.5)
        L = laplacian(bf)
        @test !isselfadjoint(L)                          # CF coupling breaks the symmetry
        @test adjoint(L) isa AdjointOp                   # must not fold to L
        @test !isselfadjoint(derivative(bf, 2; order=2))
        A = materialize(prepare(L))
        At = materialize(prepare(adjoint(L)))
        @test At ≈ A'                                    # declared transpose, exactly
        @test !(A ≈ A')                                  # and L itself is not symmetric
    end

    @testset "block-geometry constraints degrade to an error" begin
        base9 = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (9, 9))
        bf9 = BlockForest(base9; blocksize=(3, 3), maxlevel=2)   # odd blocksize
        refine!(bf9, x -> x[1] < 0.34 && x[2] < 0.34)
        u9 = scalar_field(bf9)
        @test_throws ArgumentError halo_update!(u9, bf9)
        base4 = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (4, 4))
        bf2 = BlockForest(base4; blocksize=(2, 2), maxlevel=2)   # blocksize < 4
        refine!(bf2, x -> x[1] < 0.5 && x[2] < 0.5)
        u2 = scalar_field(bf2)
        @test_throws ArgumentError halo_update!(u2, bf2)
        # uniform forests keep the looser v1 constraints
        bf2u = BlockForest(base4; blocksize=(2, 2), maxlevel=2)
        @test laplacian(bf2u) * set!(scalar_field(bf2u), x -> x[1]) isa BlockField
    end
end
