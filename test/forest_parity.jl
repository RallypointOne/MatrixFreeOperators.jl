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
end
