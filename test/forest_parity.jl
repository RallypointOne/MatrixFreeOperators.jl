# An operator that declares nothing — the trait defaults it must fall back to.
struct NoTraitOp85 <: AbstractOperator end

@testset "Forest operator parity" begin
    MFO = MatrixFreeOperators

    # Reconstruct a uniform-level forest scalar field into the equivalent dense array
    # (block coords × blocksize give each block's slice of the full grid).
    function reconstruct(f, dims)
        bf = f.grid
        b = bf.blocksize
        full = zeros(eltype(f), dims)
        for i in 1:MFO.nleaves(bf)
            key = bf.forest.leaves[i]
            idx = ntuple(d -> (key.coords[d] * b[d]) .+ (1:b[d]), length(dims))
            full[idx...] .= collect(interior(MFO.block(f, i)))
        end
        return full
    end

    bcs = [
        ((Periodic(), Periodic()), (Periodic(), Periodic())),
        ((Dirichlet(), Dirichlet()), (Dirichlet(), Dirichlet())),
        ((Neumann(), Neumann()), (Neumann(), Neumann())),
        ((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
    ]
    fun = x -> sinpi(x[1]) * cospi(2x[2]) + 0.3 * x[1]
    leaf_ops = (
        laplacian,
        g -> derivative(g, 1; order=1),
        g -> derivative(g, 2; order=1),
        g -> derivative(g, 1; order=2),
    )

    @testset "level-0 forward parity vs equal-resolution single grid" begin
        for bc in bcs
            g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8); bc=bc)
            bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)   # level 0 ⇒ same (8, 8)
            u = set!(scalar_field(g), fun)
            uf = set!(scalar_field(bf), fun)
            for makeL in leaf_ops
                ref = collect(interior(makeL(g) * u))
                rec = reconstruct(makeL(bf) * uf, (8, 8))
                @test rec == ref                                # bit-identical
            end
        end
    end

    @testset "uniformly-refined forward parity vs double-resolution grid" begin
        for bc in bcs
            g16 = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=bc)
            base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8); bc=bc)
            bf = BlockForest(base; blocksize=(4, 4), maxlevel=3)
            refine!(bf, _ -> true)                              # uniform level 1 = 16×16
            @test all(k -> k.level == 1, bf.forest.leaves)
            u = set!(scalar_field(g16), fun)
            uf = set!(scalar_field(bf), fun)
            for makeL in leaf_ops
                ref = collect(interior(makeL(g16) * u))
                rec = reconstruct(makeL(bf) * uf, (16, 16))
                @test rec == ref
            end
        end
    end

    @testset "adjoint identity ⟨Lx,y⟩ = ⟨x,Lᵀy⟩ on the forest" begin
        rng = Random.MersenneTwister(5)
        for bc in bcs
            g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8); bc=bc)
            bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
            ops = (
                laplacian(bf),
                derivative(bf, 1; order=1),     # non-self-adjoint
                derivative(bf, 2; order=2),     # self-adjoint
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
                        collect(interior(MFO.block(Lx, i))), collect(interior(MFO.block(y, i)))
                    ),
                    1:MFO.nleaves(bf),
                )
                ip2 = sum(
                    i -> dot(
                        collect(interior(MFO.block(x, i))), collect(interior(MFO.block(Lty, i)))
                    ),
                    1:MFO.nleaves(bf),
                )
                @test ip1 ≈ ip2
            end
        end
    end

    @testset "prepared operator: dense symmetry + mul! round-trip" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (8, 8);
            bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
        )
        bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
        A = prepare(laplacian(bf))
        @test size(A) == (64, 64)                   # nleaves·prod(blocksize) = 4·16
        M = materialize(A)
        @test M ≈ M'                                # forest Laplacian is self-adjoint
        uf = set!(scalar_field(bf), fun)
        v = flatten(uf)
        out = similar(v)
        mul!(out, A, v)
        @test out == flatten(laplacian(bf) * uf)
    end

    @testset "lazy adjoint (AdjointOp) folds across blocks" begin
        for bc in bcs
            g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8); bc=bc)
            bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
            u = set!(scalar_field(g), fun)
            uf = set!(scalar_field(bf), fun)
            D = derivative(g, 1; order=1)            # order 1 ⇒ adjoint is an AdjointOp
            Df = derivative(bf, 1; order=1)
            @test Df' isa AdjointOp
            # algebra path (L' * x) must match the single grid — accumulation order
            # differs at block faces, so ≈ rather than ==
            @test reconstruct(Df' * copy(uf), (8, 8)) ≈ collect(interior(D' * copy(u)))
            # double adjoint routes back to the forward action
            Dtt = apply_adjoint!(scalar_field(bf), Df', copy(uf), bf)
            @test reconstruct(Dtt, (8, 8)) ≈ collect(interior(D * copy(u)))
            # prepared path (PreparedAdjoint twin), β = 0 and β ≠ 0
            A = prepare(Df')
            v = flatten(uf)
            out = similar(v)
            mul!(out, A, v)
            @test out ≈ flatten(Df' * copy(uf))
            ref = 2.0 .* flatten(Df' * copy(uf)) .+ 3.0 .* v
            out2 = copy(v)
            mul!(out2, A, v, 2.0, 3.0)
            @test out2 ≈ ref
            # nested inside a combinator: (Δ + Dᵀ) must fold across blocks too
            S = laplacian(g) + D'
            Sf = laplacian(bf) + Df'
            @test reconstruct(Sf * copy(uf), (8, 8)) ≈ collect(interior(S * copy(u)))
            As = prepare(Sf)
            outs = similar(v)
            mul!(outs, As, v)
            @test outs ≈ flatten(Sf * copy(uf))
        end
    end

    @testset "Composed on a forest: parity, prepared path, composition law" begin
        for bc in bcs
            g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8); bc=bc)
            bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
            u = set!(scalar_field(g), fun)
            uf = set!(scalar_field(bf), fun)
            L2g = laplacian(g) * laplacian(g)
            L2f = laplacian(bf) * laplacian(bf)
            @test reconstruct(L2f * uf, (8, 8)) == collect(interior(L2g * u))
            # nested inside Added/Scaled, still at the forest level
            Sg = laplacian(g) + 2.0 * (derivative(g, 1; order=1) * laplacian(g))
            Sf = laplacian(bf) + 2.0 * (derivative(bf, 1; order=1) * laplacian(bf))
            @test reconstruct(Sf * uf, (8, 8)) == collect(interior(Sg * u))
        end
        # prepared mul! matches the un-prepared path, and the adjoint identity holds
        bf = BlockForest(
            CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8)); blocksize=(4, 4), maxlevel=2
        )
        uf = set!(scalar_field(bf), fun)
        L2 = laplacian(bf) * laplacian(bf)
        A = prepare(L2, uf)
        v = flatten(uf)
        out = similar(v)
        mul!(out, A, v)
        @test out == flatten(L2 * copy(uf))
        M = materialize(A)
        @test M ≈ M'                                # Δ² is self-adjoint on a uniform forest
        Dc = derivative(bf, 1; order=1) * laplacian(bf)
        Ac = materialize(prepare(Dc, uf))
        Act = materialize(prepare(adjoint(Dc), uf))
        @test Act ≈ Ac'
        # rank-changing composition: the intermediate is a vector BlockField whose
        # inter-block exchange must reproduce the single-grid wide Laplacian exactly
        g8 = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8))
        u8 = set!(scalar_field(g8), fun)
        wide = apply(divergence(g8), apply(MFO.gradient(g8), u8))
        DG = divergence(bf) * MFO.gradient(bf)
        @test reconstruct(DG * copy(uf), (8, 8)) == collect(interior(wide))
    end

    @testset "rank-changers: gradient/divergence parity + adjoint identity" begin
        vfun = x -> SVector(sinpi(x[1]) + 0.2 * x[2], cospi(x[2]) - x[1])
        for bc in bcs
            g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8); bc=bc)
            bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
            u = set!(scalar_field(g), fun)
            uf = set!(scalar_field(bf), fun)
            @test reconstruct(MFO.gradient(bf) * uf, (8, 8)) ==
                collect(interior(MFO.gradient(g) * u))
            w = set!(vector_field(g), vfun)
            wf = set!(vector_field(bf), vfun)
            @test reconstruct(divergence(bf) * wf, (8, 8)) ==
                collect(interior(divergence(g) * w))
        end
        # ⟨∇u, w⟩ = ⟨u, ∇ᵀw⟩ with a vector cotangent, across blocks
        rng = Random.MersenneTwister(7)
        bf = BlockForest(CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8)); blocksize=(4, 4), maxlevel=2)
        G = MFO.gradient(bf)
        u = scalar_field(bf)
        w = vector_field(bf)
        for i in 1:MFO.nleaves(bf)
            interior(MFO.block(u, i)) .= rand(rng, bf.blocksize...)
            interior(MFO.block(w, i)) .= SVector.(rand(rng, bf.blocksize...), rand(rng, bf.blocksize...))
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

    @testset "boundary_rhs parity vs single grid" begin
        bc = ((Dirichlet(2.0), Dirichlet(-1.0)), (Neumann(0.5), Dirichlet()))
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8); bc=bc)
        bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
        b = boundary_rhs(laplacian(g), g)
        bfor = boundary_rhs(laplacian(bf), bf)
        @test bfor isa BlockField
        @test reconstruct(bfor, (8, 8)) == collect(interior(b))
        # combinators lift per leaf too
        Ls = 2.0 * laplacian(g) + derivative(g, 1; order=1)
        Lf = 2.0 * laplacian(bf) + derivative(bf, 1; order=1)
        @test reconstruct(boundary_rhs(Lf, bf), (8, 8)) == collect(interior(boundary_rhs(Ls, g)))
        # refined forest: Neumann offsets scale with the leaf's own (halved) spacing
        g16 = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=bc)
        bfr = BlockForest(g; blocksize=(4, 4), maxlevel=2)
        refine!(bfr, _ -> true)                        # uniform level 1 = 16×16
        @test reconstruct(boundary_rhs(laplacian(bfr), bfr), (16, 16)) ==
            collect(interior(boundary_rhs(laplacian(g16), g16)))
    end
    @testset "Added shares one exchange per operand set (issue #85)" begin
        # Interiors of two forest fields agree bit-for-bit (either layout).
        interiors_equal(a, b) = all(
            i -> interior(MFO.block(a, i)) == interior(MFO.block(b, i)),
            1:MFO.nleaves(a.grid),
        )
        max_interior_diff(a, b) = maximum(
            i -> maximum(abs, interior(MFO.block(a, i)) .- interior(MFO.block(b, i))),
            1:MFO.nleaves(a.grid),
        )
        # The un-shared reference: every operand applied through its own full forest
        # action (exchange + BC pass + sweep), accumulating — what the sum did before.
        function per_operand!(y, L::Added, x, g, α, β)
            per_operand!(y, L.a, x, g, α, β)
            per_operand!(y, L.b, x, g, α, true)
            return y
        end
        per_operand!(y, L, x, g, α, β) = apply!(y, L, x, g, α, β)

        @testset "trait table" begin
            g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8))
            bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
            κ = set!(scalar_field(bf), x -> 1 + x[1])
            vel = set!(vector_field(bf), x -> SVector(1.0, 0.5))
            for L in (
                laplacian(bf), derivative(bf, 1; order=1), derivative(bf, 2; order=2),
                MFO.gradient(bf), divergence(bf), scaling(κ), scaling(2.0), identity_op(),
                advection(bf, vel), advection(bf, SelfAdvection()),
            )
                @test shares_exchange(L)
                @test shares_exchange(2.0 * L)
            end
            @test shares_exchange(laplacian(bf) + 0.5 * derivative(bf, 1; order=2))
            @test shares_exchange((laplacian(bf) + scaling(κ)) + identity_op())
            # combinators that must never share: their forest action is not one sweep
            @test !shares_exchange(laplacian(bf) * laplacian(bf))
            @test !shares_exchange(MFO.AdjointOp(derivative(bf, 1; order=1)))
            @test !shares_exchange(derivative(bf, 1; order=1)')
            @test !shares_exchange(laplacian(bf) + laplacian(bf) * laplacian(bf))
            @test !shares_exchange(laplacian(bf) + derivative(bf, 1; order=1)')
            # the default is the weaker claim (NoTraitOp85 is declared at file top level)
            @test !shares_exchange(NoTraitOp85())
            @test !shares_exchange(laplacian(bf) + NoTraitOp85())
            # Diffusion is grid-aware: its coarse–fine ghost rewrite forbids sharing on
            # an adapted forest, and a regrid flips the answer live
            D = diffusion(bf, κ)
            @test shares_exchange(D)
            refine!(bf, x -> x[1] < 0.5)
            @test !bf.forest.uniform[]
            @test !shares_exchange(D)
            @test !shares_exchange(D + laplacian(bf))
        end

        bc = ((Dirichlet(), Dirichlet()), (Neumann(), Neumann()))
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=bc)
        # A 3D Float32 root with a periodic axis: the packed kernels and the periodic
        # wrap are the layouts most likely to diverge from the per-operand reference,
        # and Float32 checks the shared sweep keeps the field's eltype.
        g3 = CartesianGrid(
            ((0.0f0, 1.0f0), (0.0f0, 1.0f0), (0.0f0, 1.0f0)), (8, 8, 8);
            bc=((Periodic(), Periodic()), (Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
        )
        fun3 = x -> sinpi(2x[1]) * cospi(2x[2]) + 0.3f0 * x[3]^2
        # (label, root grid, blocksize, test function); refining the origin corner
        # twice gives levels 0–2, 2:1 balanced, in either dimension — and in 3D the
        # corner's coarse–fine faces sit on the periodic wrap
        cases = (
            ("2D Float64", g, (4, 4), fun),
            ("3D Float32 periodic", g3, (4, 4, 4), fun3),
        )
        for (label, groot, bs, f) in cases, refined in (false, true)
            bf = BlockForest(groot; blocksize=bs, maxlevel=3)
            if refined
                refine!(bf, x -> all(<(0.5), x))
                refine!(bf, x -> all(<(0.2), x))   # levels 0–2, 2:1 balanced
                balance!(bf)
                @test !bf.forest.uniform[]
            end
            u = set!(scalar_field(bf), f)
            T = eltype(u)
            @test T === eltype(groot.spacing)
            N = length(bs)
            # the Niederer-style anisotropic operator: an Added of two Scaled leaves
            aniso = T(0.13) * laplacian(bf) + T(0.7) * derivative(bf, 1; order=2)
            three = (aniso + T(0.3) * derivative(bf, N; order=1)) + identity_op()
            @test eltype(aniso * u) === T

            # The packed row counts through storage/`block` forwarding only (the
            # counter is not a PackedBlockField, so dispatch lands on the
            # AbstractBlockField sweep, which is also the CPU fallback of the packed
            # overrides); the packed overrides themselves are covered by the
            # unwrapped parity checks in forest_packed.jl / forest_prepare.jl.
            @testset "exchange count and bit-parity, $label, refined=$refined, $(nameof(typeof(x)))" for x in (u, pack(u))
                # baseline: one leaf costs one exchange and one BC pass
                @test count_exchanges!(similar(x), laplacian(bf), x, bf) == (1, 1)
                # a two-leaf sum: one, not two
                y = similar(x)
                @test count_exchanges!(y, aniso, x, bf) == (1, 1)
                ref = per_operand!(similar(x), aniso, x, bf, true, false)
                @test interiors_equal(y, ref)      # same sweeps on the same ghosts
                # three leaves, nested Added: still one
                y3 = similar(x)
                @test count_exchanges!(y3, three, x, bf) == (1, 1)
                @test interiors_equal(y3, per_operand!(similar(x), three, x, bf, true, false))
                # the accumulating form and scaling stay exact
                yacc = copy(ref)
                @test count_exchanges!(yacc, aniso, x, bf, T(2), T(-1)) == (1, 1)
                @test interiors_equal(yacc, per_operand!(copy(ref), aniso, x, bf, T(2), T(-1)))
                # Scaled over the sum shares too
                ys = similar(x)
                @test count_exchanges!(ys, T(3) * aniso, x, bf) == (1, 1)
                @test interiors_equal(ys, per_operand!(similar(x), T(3) * aniso, x, bf, true, false))
                # a diagonal operand rides along on the shared exchange
                κ = set!(scalar_field(bf), z -> 1 + z[1] * z[2])
                κx = x isa PackedBlockField ? pack(κ) : κ
                mixed = laplacian(bf) + scaling(κx)
                ym = similar(x)
                @test count_exchanges!(ym, mixed, x, bf) == (1, 1)
                @test interiors_equal(ym, per_operand!(similar(x), mixed, x, bf, true, false))
            end

            @testset "non-shareable operands still exchange per operand, $label, refined=$refined" begin
                x = u
                # Composed operand: its intermediate needs an exchange of its own; the
                # inner factor exchanges x, the outer one exchanges the intermediate
                LC = laplacian(bf) + T(0.5) * (derivative(bf, 1; order=1) * laplacian(bf))
                @test !shares_exchange(LC)
                yc = similar(x)
                @test count_exchanges!(yc, LC, x, bf) == (2, 2)
                @test interiors_equal(yc, per_operand!(similar(x), LC, x, bf, true, false))
                # AdjointOp operand: one exchange for the leaf, a gather + fold for the
                # adjoint (which exchanges nothing on x)
                LA = laplacian(bf) + derivative(bf, 1; order=1)'
                @test !shares_exchange(LA)
                ya = similar(x)
                @test count_exchanges!(ya, LA, x, bf) == (1, 1)
                @test interiors_equal(ya, per_operand!(similar(x), LA, x, bf, true, false))
            end
        end

        @testset "Diffusion: shares on a uniform forest, opts out on an adapted one" begin
            g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=bc)
            κ_fun = z -> 1 + z[1] + 0.5 * z[2]^2      # variable κ: rewrite ≠ restriction
            # uniform: cfflux is empty, so the sum shares one exchange, bit-exactly
            bfu = BlockForest(g; blocksize=(4, 4), maxlevel=3)
            u = set!(scalar_field(bfu), fun)
            Du = diffusion(bfu, set!(scalar_field(bfu), κ_fun)) + laplacian(bfu)
            @test shares_exchange(Du)
            yu = similar(u)
            @test count_exchanges!(yu, Du, u, bfu) == (1, 1)
            @test interiors_equal(yu, per_operand!(similar(u), Du, u, bfu, true, false))
            # adapted: the coarse–fine flux rewrite overwrites x's coarse-side ghosts
            bfr = BlockForest(g; blocksize=(4, 4), maxlevel=3)
            refine!(bfr, x -> x[1] < 0.5 && x[2] < 0.5)
            balance!(bfr)
            ur = set!(scalar_field(bfr), fun)
            Dr = diffusion(bfr, set!(scalar_field(bfr), κ_fun))
            Lr = Dr + laplacian(bfr)
            @test !shares_exchange(Lr)
            yr = similar(ur)
            @test count_exchanges!(yr, Lr, ur, bfr) == (2, 2)
            ref = per_operand!(similar(ur), Lr, ur, bfr, true, false)
            @test interiors_equal(yr, ref)
            # Why the trait must be false: sweeping the Laplacian on the exchange the
            # Diffusion sweep already rewrote gives a different (wrong) answer — the
            # flaw a wrongly-true declaration would introduce.
            xw = copy(ur)
            yw = similar(ur)
            MFO.halo_update!(xw, bfr)
            MFO.apply_bc!(xw, bfr)
            MFO._forest_sweep!(yw, Dr, xw, bfr, true, false)
            MFO._forest_sweep!(yw, laplacian(bfr), xw, bfr, true, true)
            @test max_interior_diff(yw, ref) > 1e-6
            # ...and the other order is fine only by accident of ordering, which the
            # trait deliberately does not exploit
            @test !shares_exchange(laplacian(bfr) + Dr)
        end

        @testset "adjoint of a sum: self-adjoint shortcut shares, general path unchanged" begin
            g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=bc)
            rng = Random.MersenneTwister(85)
            randomize!(f) = (foreach(i -> (interior(MFO.block(f, i)) .= rand(rng, eltype(f), f.grid.blocksize...)), 1:MFO.nleaves(f.grid)); f)
            ipdot(a, b) = sum(
                i -> dot(collect(interior(MFO.block(a, i))), collect(interior(MFO.block(b, i)))),
                1:MFO.nleaves(a.grid),
            )
            # (label, root grid, blocksize, identity tolerance): the Float32 row
            # accumulates its inner products in single precision, so its tolerance
            # is loosened to match (measured ~2.5e-7 relative on this forest)
            adj_cases = (
                ("2D Float64", g, (4, 4), 1e-10),
                ("3D Float32 periodic", g3, (4, 4, 4), 1e-5),
            )
            for (label, groot, bs, tol) in adj_cases, refined in (false, true)
                bf = BlockForest(groot; blocksize=bs, maxlevel=3)
                if refined
                    refine!(bf, x -> all(<(0.5), x))
                    balance!(bf)
                end
                T = eltype(groot.spacing)
                sym = T(0.13) * laplacian(bf) + T(0.7) * derivative(bf, 1; order=2)
                skew = T(0.13) * laplacian(bf) + T(0.7) * derivative(bf, 1; order=1)
                @test isselfadjoint(sym) == !refined
                @testset "$label, refined=$refined, $(isselfadjoint(L) ? "self-adjoint" : "skew")" for L in (sym, skew)
                    x = randomize!(scalar_field(bf))
                    y = randomize!(scalar_field(bf))
                    Lx = apply!(scalar_field(bf), L, copy(x), bf)
                    c = CountingBlockField(copy(y))
                    Lty = apply_adjoint!(scalar_field(bf), L, c, bf)
                    # the self-adjoint sum rides one shared forward exchange (it used to
                    # take one per operand); otherwise each operand transposes on its
                    # own — a gather + fold that exchanges nothing, or, for a leaf that
                    # is itself self-adjoint, its single forward exchange
                    @test c.exchanges[] ≤ 1
                    isselfadjoint(L) && @test c.exchanges[] == 1
                    @test eltype(Lty) === T
                    ip1 = ipdot(Lx, y)
                    ip2 = ipdot(x, Lty)
                    @test abs(ip1 - ip2) ≤ tol * max(one(T), abs(ip1), abs(ip2))
                end
            end
        end
    end
end
