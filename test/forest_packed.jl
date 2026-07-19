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

    ipdot(a, b) = sum(
        i -> dot(
            collect(interior(MFO.block(a, i))), collect(interior(MFO.block(b, i)))
        ),
        1:MFO.nleaves(a.grid),
    )
    gfun = x -> cospi(x[1]) * x[2] + 0.1 * x[2]^2

    @testset "forward bit-parity vs the per-leaf reference path" begin
        for bc in bcs
            g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8); bc=bc)
            bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
            uf = set!(scalar_field(bf), fun)
            for makeL in (laplacian, g -> derivative(g, 1; order=1))
                L = makeL(bf)
                @test parity(L * pack(uf), L * uf)
            end
            refine!(bf, x -> x[1] < 0.5)
            ur = set!(scalar_field(bf), fun)
            @test parity(laplacian(bf) * pack(ur), laplacian(bf) * ur)
        end
    end

    @testset "forest-native kernel: direct launch bit-parity (CPU backend)" begin
        # The public API routes non-GPU backends to the fallback sweep, so the kernel
        # body is exercised here by direct launch — same coverage MFO_TEST_GPU gets
        # through the API on CUDA.
        for refined in (false, true)
            g = CartesianGrid(
                ((0.0, 1.0), (0.0, 1.0)), (8, 8);
                bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
            )
            bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
            refined && refine!(bf, x -> x[1] < 0.5)   # kernel reads per-leaf levels
            u = set!(scalar_field(bf), fun)
            x = pack(u)
            MFO.halo_update!(x, bf)
            MFO.apply_bc!(x, bf)
            ref = MFO._forest_sweep_leaves!(similar(x), laplacian(bf), x, bf, 2.0, false)
            y = MFO._zero_all!(similar(x))
            kernel! = MFO._lap_forest_kernel!(KernelAbstractions.CPU())
            kernel!(
                y.data, x.data, x.levels, bf.spacing0, bf.halo, 2.0, false;
                ndrange=(bf.blocksize..., MFO.nleaves(bf)),
            )
            @test parity(y, ref)
            # accumulating form: β ≠ 0 blends into existing y
            y2 = MFO._zero_all!(similar(x))
            for i in 1:MFO.nleaves(bf)
                interior(MFO.block(y2, i)) .= 1.0
            end
            ref2 = copy(y2)
            MFO._forest_sweep_leaves!(ref2, laplacian(bf), x, bf, 2.0, 3.0)
            kernel!(
                y2.data, x.data, x.levels, bf.spacing0, bf.halo, 2.0, 3.0;
                ndrange=(bf.blocksize..., MFO.nleaves(bf)),
            )
            @test parity(y2, ref2)
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

    @testset "packed mul! allocations: parity with the reference path (CPU)" begin
        bc = ((Dirichlet(), Dirichlet()), (Neumann(), Neumann()))
        # DCE-proof: consume the output so the measured mul! cannot be elided.
        function alloc_mul(P, out, v)
            mul!(out, P, v)
            mul!(out, P, v)
            a = @allocated mul!(out, P, v)
            return a, sum(out)
        end
        # Under Pkg.test's --check-bounds=yes BOTH per-leaf sweeps allocate per leaf
        # (reference ~48 B/leaf, packed ~96 B/leaf on 1.12); under default flags both
        # are exactly 0 B. So the CPU fallback gets the same loose per-leaf bound the
        # reference path uses (forest_prepare.jl); the kernel's size-independence is
        # asserted by direct launch below.
        alloc_bound(nl) = 1000 * nl
        for n in (16, 32)
            gn = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (n, n); bc=bc)
            bfn = BlockForest(gn; blocksize=(4, 4), maxlevel=2)
            un = set!(scalar_field(bfn), fun)
            vn = flatten(un)
            P = prepare(laplacian(bfn), pack(un))
            a, s = alloc_mul(P, similar(vn), vn)
            @test isfinite(s)
            @test a ≤ alloc_bound(MFO.nleaves(bfn))
        end
    end

    @testset "single-launch kernel allocations are forest-size independent" begin
        bc = ((Dirichlet(), Dirichlet()), (Neumann(), Neumann()))
        # The #7 closure claim, asserted where it holds — the kernel itself: one
        # launch whose cost is a fixed KA constant, flat in nleaves (16 vs 64).
        # DCE-proof: consume the swept output.
        function alloc_launch(kernel!, y, x, bf)
            args = (y.data, x.data, x.levels, bf.spacing0, bf.halo, true, false)
            nd = (bf.blocksize..., MFO.nleaves(bf))
            kernel!(args...; ndrange=nd)
            kernel!(args...; ndrange=nd)
            a = @allocated kernel!(args...; ndrange=nd)
            return a, sum(sum(interior(MFO.block(y, i))) for i in 1:MFO.nleaves(bf))
        end
        kernel! = MFO._lap_forest_kernel!(KernelAbstractions.CPU())
        allocs = map((16, 32)) do n
            gn = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (n, n); bc=bc)
            bfn = BlockForest(gn; blocksize=(4, 4), maxlevel=2)
            x = pack(set!(scalar_field(bfn), fun))
            MFO.halo_update!(x, bfn)
            MFO.apply_bc!(x, bfn)
            a, s = alloc_launch(kernel!, MFO._zero_all!(similar(x)), x, bfn)
            @test isfinite(s)
            a
        end
        @test allocs[1] == allocs[2]
        @test allocs[1] ≤ 4096
    end

    @testset "forward bit-parity: part-2 operators (uniform + refined)" begin
        for bc in bcs, refined in (false, true)
            g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8); bc=bc)
            bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
            refined && refine!(bf, x -> x[1] < 0.5)
            uf = set!(scalar_field(bf), fun)
            κ = set!(scalar_field(bf), x -> 1 + x[1]^2 + 0.5 * x[2])
            vel = set!(vector_field(bf), vfun)
            for makeL in (
                g -> derivative(g, 2; order=2),
                MFO.gradient,
                _ -> scaling(2.5),
                _ -> scaling(pack(κ)),
                _ -> scaling(κ),              # BlockField coeff on packed x: fallback
                g -> advection(g, pack(vel)),
                g -> advection(g, vel),       # BlockField velocity on packed x: fallback
                _ -> identity_op(),
            )
                L = makeL(bf)
                @test parity(L * pack(uf), L * uf)
            end
            w = set!(vector_field(bf), vfun)
            @test parity(divergence(bf) * pack(w), divergence(bf) * w)
            A = advection(bf, SelfAdvection())
            @test parity(apply(A, pack(w)), apply(A, w))
        end
    end

    @testset "part-2 kernels: direct launch bit-parity (CPU backend)" begin
        cpu = KernelAbstractions.CPU()
        for refined in (false, true)
            g = CartesianGrid(
                ((0.0, 1.0), (0.0, 1.0)), (8, 8);
                bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
            )
            bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
            refined && refine!(bf, x -> x[1] < 0.5)
            nd = (bf.blocksize..., MFO.nleaves(bf))
            x = pack(set!(scalar_field(bf), fun))
            MFO.halo_update!(x, bf)
            MFO.apply_bc!(x, bf)
            xw = pack(set!(vector_field(bf), vfun))
            MFO.halo_update!(xw, bf)
            MFO.apply_bc!(xw, bf)
            κp = pack(set!(scalar_field(bf), x -> 1 + x[1]^2))
            velp = pack(set!(vector_field(bf), vfun))
            cases = (
                (
                    derivative(bf, 1; order=1), x,
                    (y, α, β) -> MFO._deriv_forest_kernel!(cpu)(
                        y.data, x.data, x.levels, bf.spacing0, bf.halo, 1, 1, α, β;
                        ndrange=nd,
                    ),
                ),
                (
                    derivative(bf, 2; order=2), x,
                    (y, α, β) -> MFO._deriv_forest_kernel!(cpu)(
                        y.data, x.data, x.levels, bf.spacing0, bf.halo, 2, 2, α, β;
                        ndrange=nd,
                    ),
                ),
                (
                    MFO.gradient(bf), x,
                    (y, α, β) -> MFO._grad_forest_kernel!(cpu)(
                        y.data, x.data, x.levels, bf.spacing0, bf.halo, α, β; ndrange=nd
                    ),
                ),
                (
                    divergence(bf), xw,
                    (y, α, β) -> MFO._div_forest_kernel!(cpu)(
                        y.data, xw.data, xw.levels, bf.spacing0, bf.halo, α, β; ndrange=nd
                    ),
                ),
                (
                    advection(bf, velp), x,
                    (y, α, β) -> MFO._adv_forest_kernel!(cpu)(
                        y.data, x.data, velp.data, x.levels, bf.spacing0, bf.halo, α, β;
                        ndrange=nd,
                    ),
                ),
                (
                    advection(bf, SelfAdvection()), xw,
                    (y, α, β) -> MFO._adv_forest_kernel!(cpu)(
                        y.data, xw.data, xw.data, xw.levels, bf.spacing0, bf.halo, α, β;
                        ndrange=nd,
                    ),
                ),
                (
                    scaling(2.5), x,
                    (y, α, β) -> MFO._scale_forest_kernel!(cpu)(
                        y.data, x.data, 2.5, bf.halo, α, β; ndrange=nd
                    ),
                ),
                (
                    scaling(κp), x,
                    (y, α, β) -> MFO._scalefield_forest_kernel!(cpu)(
                        y.data, x.data, κp.data, bf.halo, α, β; ndrange=nd
                    ),
                ),
                (
                    identity_op(), x,
                    (y, α, β) -> MFO._scale_forest_kernel!(cpu)(
                        y.data, x.data, true, bf.halo, α, β; ndrange=nd
                    ),
                ),
            )
            for (L, xin, launch) in cases
                y = MFO._zero_all!(MFO.allocate_output(L, xin))
                launch(y, 2.0, false)
                ref = MFO._forest_sweep_leaves!(
                    MFO._zero_all!(MFO.allocate_output(L, xin)), L, xin, bf, 2.0, false
                )
                @test parity(y, ref)
                # accumulating form blends into the (nontrivial) β = 0 result
                y2 = copy(y)
                ref2 = copy(ref)
                launch(y2, 2.0, 3.0)
                MFO._forest_sweep_leaves!(ref2, L, xin, bf, 2.0, 3.0)
                @test parity(y2, ref2)
            end
        end
    end

    @testset "adjoint kernels: direct launch pre-fold parity (full padded arrays)" begin
        cpu = KernelAbstractions.CPU()
        for refined in (false, true)
            g = CartesianGrid(
                ((0.0, 1.0), (0.0, 1.0)), (8, 8);
                bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
            )
            bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
            refined && refine!(bf, x -> x[1] < 0.5)
            ndp = (bf.blocksize .+ 2 .* bf.halo..., MFO.nleaves(bf))
            ȳs = pack(set!(scalar_field(bf), fun))     # scalar cotangent
            ȳv = pack(set!(vector_field(bf), vfun))    # vector cotangent
            cases = (
                (
                    laplacian(bf), ȳs,
                    (x̄, ȳ, α) -> MFO._lap_adjoint_forest_kernel!(cpu)(
                        x̄.data, ȳ.data, ȳ.levels, bf.spacing0, α; ndrange=ndp
                    ),
                ),
                (
                    derivative(bf, 1; order=1), ȳs,
                    (x̄, ȳ, α) -> MFO._deriv_adjoint_forest_kernel!(cpu)(
                        x̄.data, ȳ.data, ȳ.levels, bf.spacing0, 1, 1, α; ndrange=ndp
                    ),
                ),
                (
                    derivative(bf, 2; order=2), ȳs,
                    (x̄, ȳ, α) -> MFO._deriv_adjoint_forest_kernel!(cpu)(
                        x̄.data, ȳ.data, ȳ.levels, bf.spacing0, 2, 2, α; ndrange=ndp
                    ),
                ),
                (
                    MFO.gradient(bf), ȳv,
                    (x̄, ȳ, α) -> MFO._grad_adjoint_forest_kernel!(cpu)(
                        x̄.data, ȳ.data, ȳ.levels, bf.spacing0, α; ndrange=ndp
                    ),
                ),
                (
                    divergence(bf), ȳs,
                    (x̄, ȳ, α) -> MFO._div_adjoint_forest_kernel!(cpu)(
                        x̄.data, ȳ.data, ȳ.levels, bf.spacing0, α; ndrange=ndp
                    ),
                ),
            )
            for (L, ȳ, launch) in cases
                ȳk = copy(ȳ)
                ȳr = copy(ȳ)
                x̄k = MFO.allocate_input(L, ȳk)
                x̄r = MFO.allocate_input(L, ȳr)
                MFO.zero_ghosts!(ȳk)    # the seam zeroes before launching
                launch(x̄k, ȳk, 2.0)
                MFO._forest_adjoint_sweep_leaves!(x̄r, L, ȳr, bf, 2.0)
                # full arrays: the ghost cotangents are the point of the padded ndrange
                @test x̄k.data == x̄r.data
            end
        end
    end

    @testset "adjoint identity ⟨Lx,y⟩ = ⟨x,Lᵀy⟩: part-2 operators" begin
        for bc in bcs, refined in (false, true)
            g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8); bc=bc)
            bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
            refined && refine!(bf, x -> x[1] < 0.5)
            vel = set!(vector_field(bf), vfun)
            xs = pack(set!(scalar_field(bf), fun))
            ys = pack(set!(scalar_field(bf), gfun))
            yv = pack(set!(vector_field(bf), x -> SVector(gfun(x), fun(x))))
            xv = pack(set!(vector_field(bf), vfun))
            for L in (derivative(bf, 1; order=1), derivative(bf, 2; order=2))
                Lx = apply(L, copy(xs))
                Lty = apply_adjoint!(similar(xs), L, copy(ys), bf)
                @test ipdot(Lx, ys) ≈ ipdot(xs, Lty)
            end
            # advection adjoint is declared algebraically: Σ_d D_dᵀ ∘ scaling(v_d)
            A = advection(bf, pack(vel))
            Ax = apply(A, copy(xs))
            Aty = apply(adjoint(A), copy(ys))
            @test ipdot(Ax, ys) ≈ ipdot(xs, Aty)
            # rank-changers: mixed-rank inner products
            G = MFO.gradient(bf)
            Gx = apply(G, copy(xs))
            Gty = apply_adjoint!(similar(xs), G, copy(yv), bf)
            @test ipdot(Gx, yv) ≈ ipdot(xs, Gty)
            D = divergence(bf)
            Dx = apply(D, copy(xv))
            Dty = apply_adjoint!(similar(xv), D, copy(ys), bf)
            @test ipdot(Dx, ys) ≈ ipdot(xv, Dty)
        end
    end

    @testset "diagonal adjoints: complex coefficients skip the ghost fold" begin
        for refined in (false, true)
            g = CartesianGrid(
                ((0.0, 1.0), (0.0, 1.0)), (8, 8);
                bc=((Dirichlet(), Dirichlet()), (Dirichlet(), Dirichlet())),
            )
            bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
            refined && refine!(bf, x -> x[1] < 0.5)
            κc = set!(scalar_field(bf, ComplexF64), x -> (1 + x[1]) + im * x[2])
            xs = pack(set!(scalar_field(bf, ComplexF64), x -> fun(x) + 0.5im * x[1]))
            ys = pack(set!(scalar_field(bf, ComplexF64), x -> gfun(x) - im * x[2]))
            for S in (scaling(κc), scaling(pack(κc)), scaling(1.5 + 2.0im))
                Sx = apply(S, copy(xs))
                Sty = apply_adjoint!(similar(xs), S, copy(ys), bf)
                @test ipdot(Sx, ys) ≈ ipdot(xs, Sty)
            end
        end
    end

    @testset "prepared coefficient normalization and composite mul!" begin
        bc = ((Dirichlet(), Dirichlet()), (Neumann(), Neumann()))
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=bc)
        bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
        uf = set!(scalar_field(bf), fun)
        κ = set!(scalar_field(bf), x -> 1 + x[1]^2 + 0.5 * x[2])
        vel = set!(vector_field(bf), vfun)
        v = flatten(uf)

        K = divergence(bf) * scaling(κ) * MFO.gradient(bf)
        A = prepare(K, pack(uf))
        # prepare packed the BlockField coefficient (layout normalization)
        @test A.op.a.b.coeff isa PackedBlockField
        out = similar(v)
        ref = similar(v)
        mul!(out, A, v)
        mul!(ref, prepare(K, uf), v)
        @test out == ref
        out2 = copy(v)
        ref2 = copy(v)
        mul!(out2, A, v, 2.0, 3.0)
        mul!(ref2, prepare(K, uf), v, 2.0, 3.0)
        @test out2 == ref2

        Aa = prepare(advection(bf, vel), pack(uf))
        @test Aa.op.velocity isa PackedBlockField
        mul!(out, Aa, v)
        mul!(ref, prepare(advection(bf, vel), uf), v)
        @test out == ref

        infer(P) = @inferred MFO._forest_capply!(P.ypad, P.op, P.xpad, P, true, false)
        @test infer(A) === A.ypad
        @test infer(Aa) === Aa.ypad

        # steady-state composite mul!: same loose per-leaf bound as the reference
        # path (exactly 0 B under default flags; --check-bounds=yes allocates per
        # leaf in both sweeps — see the alloc testsets above)
        function alloc_mul(P, out_, v_)
            mul!(out_, P, v_)
            mul!(out_, P, v_)
            a = @allocated mul!(out_, P, v_)
            return a, sum(out_)
        end
        a, s = alloc_mul(A, similar(v), v)
        @test isfinite(s)
        @test a ≤ 2000 * MFO.nleaves(bf)
    end

    @testset "coefficient staleness after regrid" begin
        bc = ((Dirichlet(), Dirichlet()), (Dirichlet(), Dirichlet()))
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8); bc=bc)
        bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
        κp = pack(set!(scalar_field(bf), x -> 1 + x[1]))
        velp = pack(set!(vector_field(bf), vfun))
        S = scaling(κp)
        A = advection(bf, velp)
        refine!(bf, _ -> true)
        xfresh = pack(set!(scalar_field(bf), fun))
        @test_throws ArgumentError apply(S, xfresh)   # stale packed coefficient
        @test_throws ArgumentError apply(A, xfresh)   # stale packed velocity

        bf2 = BlockForest(g; blocksize=(4, 4), maxlevel=2)
        κ2 = set!(scalar_field(bf2), x -> 1 + x[1])
        u2 = set!(scalar_field(bf2), fun)
        P = prepare(divergence(bf2) * scaling(κ2) * MFO.gradient(bf2), pack(u2))
        v2 = flatten(u2)
        refine!(bf2, _ -> true)
        @test_throws ArgumentError mul!(similar(v2), P, v2)   # stale prepared operator
    end

    @testset "part-2 single-launch allocations are forest-size independent" begin
        bc = ((Dirichlet(), Dirichlet()), (Neumann(), Neumann()))
        cpu = KernelAbstractions.CPU()
        # DCE-proof: consume the swept output (see the Laplacian testset above).
        function alloc_adv(kernel!, y, x, velp, bfn)
            args = (y.data, x.data, velp.data, x.levels, bfn.spacing0, bfn.halo, true, false)
            nd = (bfn.blocksize..., MFO.nleaves(bfn))
            kernel!(args...; ndrange=nd)
            kernel!(args...; ndrange=nd)
            a = @allocated kernel!(args...; ndrange=nd)
            return a, sum(sum(interior(MFO.block(y, i))) for i in 1:MFO.nleaves(bfn))
        end
        function alloc_dadj(kernel!, x̄, ȳ, bfn)
            args = (x̄.data, ȳ.data, ȳ.levels, bfn.spacing0, 1, 1, true)
            nd = (bfn.blocksize .+ 2 .* bfn.halo..., MFO.nleaves(bfn))
            kernel!(args...; ndrange=nd)
            kernel!(args...; ndrange=nd)
            a = @allocated kernel!(args...; ndrange=nd)
            return a, sum(sum(MFO._block_array(x̄, i)) for i in 1:MFO.nleaves(bfn))
        end
        adv! = MFO._adv_forest_kernel!(cpu)
        dadj! = MFO._deriv_adjoint_forest_kernel!(cpu)
        allocs = map((16, 32)) do n
            gn = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (n, n); bc=bc)
            bfn = BlockForest(gn; blocksize=(4, 4), maxlevel=2)
            x = pack(set!(scalar_field(bfn), fun))
            MFO.halo_update!(x, bfn)
            MFO.apply_bc!(x, bfn)
            velp = pack(set!(vector_field(bfn), vfun))
            a1, s1 = alloc_adv(adv!, MFO._zero_all!(similar(x)), x, velp, bfn)
            @test isfinite(s1)
            ȳ = copy(x)
            MFO.zero_ghosts!(ȳ)
            a2, s2 = alloc_dadj(dadj!, similar(x), ȳ, bfn)
            @test isfinite(s2)
            (a1, a2)
        end
        @test allocs[1][1] == allocs[2][1]
        @test allocs[1][1] ≤ 4096
        @test allocs[1][2] == allocs[2][2]
        @test allocs[1][2] ≤ 4096
    end
end
