@testset "Packed forest operators (single-launch kernels)" begin
    MFO = MatrixFreeOperators
    fun = x -> sinpi(x[1]) * cospi(2x[2]) + 0.3 * x[1]
    vfun = x -> SVector(sinpi(x[1]) + 0.2 * x[2], cospi(x[2]) - x[1])

    bcs = [
        ((Periodic(), Periodic()), (Periodic(), Periodic())),
        ((Dirichlet(), Dirichlet()), (Dirichlet(), Dirichlet())),
        ((Neumann(), Neumann()), (Neumann(), Neumann())),
        ((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
    ]

    parity(a, b) = all(
        i -> interior(MFO.block(a, i)) == interior(MFO.block(b, i)),
        1:MFO.nleaves(a.grid),
    )

    @testset "forward bit-parity vs the per-leaf reference path" begin
        for bc in bcs
            g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8); bc=bc)
            bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
            uf = set!(scalar_field(bf), fun)
            # Laplacian runs the single-launch kernel; derivative runs the fallback sweep.
            for makeL in (laplacian, g -> derivative(g, 1; order=1))
                L = makeL(bf)
                @test parity(L * pack(uf), L * uf)
            end
            # Non-uniform forest: the kernel derives per-leaf spacing from the levels SoA.
            refine!(bf, x -> x[1] < 0.5)
            ur = set!(scalar_field(bf), fun)
            @test parity(laplacian(bf) * pack(ur), laplacian(bf) * ur)
        end
    end

    @testset "adjoints: kernel via self-adjoint shortcut, fallback transpose" begin
        rng = Random.MersenneTwister(11)
        for bc in bcs
            g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8); bc=bc)
            bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
            uf = set!(scalar_field(bf), fun)
            # Uniform forest: isselfadjoint(Laplacian) is live-true, so the adjoint
            # action IS the kernel sweep.
            At = apply_adjoint!(similar(pack(uf)), laplacian(bf), pack(uf), bf)
            Ar = apply_adjoint!(scalar_field(bf), laplacian(bf), copy(uf), bf)
            @test parity(At, Ar)

            # Refined forest: grid-aware isselfadjoint flips false; packed runs the
            # per-leaf transpose-gather fallback + fold_bc! + halo_update_adjoint!
            # on packed storage. Verify the adjoint identity ⟨Lx,y⟩ = ⟨x,Lᵀy⟩.
            refine!(bf, x -> x[1] < 0.5)
            x = pack(scalar_field(bf))
            y = pack(scalar_field(bf))
            for i in 1:MFO.nleaves(bf)
                interior(MFO.block(x, i)) .= rand(rng, bf.blocksize...)
                interior(MFO.block(y, i)) .= rand(rng, bf.blocksize...)
            end
            L = laplacian(bf)
            Lx = apply(L, copy(x))
            Lty = apply_adjoint!(similar(x), L, copy(y), bf)
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

    @testset "combinators and rank-changers on packed storage" begin
        for bc in bcs
            g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8); bc=bc)
            bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
            uf = set!(scalar_field(bf), fun)
            p = pack(uf)
            S = 2.0 * laplacian(bf) + adjoint(derivative(bf, 1; order=1))
            @test parity(S * copy(p), S * copy(uf))
            DG = divergence(bf) * MFO.gradient(bf)   # packed SVector intermediate
            @test parity(DG * copy(p), DG * copy(uf))
            @test parity(MFO.gradient(bf) * p, MFO.gradient(bf) * uf)
            w = set!(vector_field(bf), vfun)
            @test parity(divergence(bf) * pack(w), divergence(bf) * w)
        end
    end

    @testset "boundary_rhs parity on packed" begin
        bc = ((Dirichlet(2.0), Dirichlet(-1.0)), (Neumann(0.5), Dirichlet()))
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8); bc=bc)
        bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
        proto = pack(scalar_field(bf))
        bref = boundary_rhs(laplacian(bf), scalar_field(bf))
        bpk = boundary_rhs(laplacian(bf), proto)
        @test bpk isa PackedBlockField
        @test parity(bpk, bref)
        Lc = 2.0 * laplacian(bf) + derivative(bf, 1; order=1)
        @test parity(boundary_rhs(Lc, proto), boundary_rhs(Lc, scalar_field(bf)))
    end

    @testset "prepared packed mul!: bit-parity, type stability, generation guard" begin
        bc = ((Dirichlet(), Dirichlet()), (Neumann(), Neumann()))
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=bc)
        bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
        uf = set!(scalar_field(bf), fun)
        v = flatten(uf)
        out = similar(v)
        A = prepare(laplacian(bf), pack(uf))
        @test A isa PreparedForest
        @test A.xpad isa PackedBlockField               # packed prototype ⇒ packed scratch
        mul!(out, A, v)
        ref = similar(v)
        mul!(ref, prepare(laplacian(bf), uf), v)
        @test out == ref

        S = laplacian(bf) * laplacian(bf) + adjoint(derivative(bf, 1; order=1))
        As = prepare(S, pack(uf))
        refs = similar(v)
        mul!(refs, prepare(S, uf), v)
        mul!(out, As, v)
        @test out ≈ refs
        out2 = copy(v)
        ref2 = copy(v)
        mul!(out2, As, v, 2.0, 3.0)
        mul!(ref2, prepare(S, uf), v, 2.0, 3.0)
        @test out2 ≈ ref2

        infer(P) = @inferred MFO._forest_capply!(P.ypad, P.op, P.xpad, P, true, false)
        @test infer(A) === A.ypad

        refine!(bf, _ -> true)
        @test_throws ArgumentError mul!(out, A, v)      # stale prepared operator
    end

    @testset "single-launch mul! allocations are forest-size independent" begin
        bc = ((Dirichlet(), Dirichlet()), (Neumann(), Neumann()))
        # DCE-proof: consume the output so the measured mul! cannot be elided.
        function alloc_mul(P, out, v)
            mul!(out, P, v)
            mul!(out, P, v)
            a = @allocated mul!(out, P, v)
            return a, sum(out)
        end
        allocs = map((16, 32)) do n
            gn = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (n, n); bc=bc)
            bfn = BlockForest(gn; blocksize=(4, 4), maxlevel=2)
            un = set!(scalar_field(bfn), fun)
            vn = flatten(un)
            P = prepare(laplacian(bfn), pack(un))
            a, s = alloc_mul(P, similar(vn), vn)
            @test isfinite(s)
            a
        end
        # The whole point of the packed sweep (#7 closure): one launch, so the cost
        # cannot scale with nleaves (16 vs 64 leaves here). The absolute bound is the
        # KA CPU-launch constant (~480 B measured), kept loose for version drift.
        @test allocs[1] == allocs[2]
        @test allocs[1] ≤ 4096
    end
end
