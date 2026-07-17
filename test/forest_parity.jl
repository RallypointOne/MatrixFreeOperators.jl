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
    end
end
