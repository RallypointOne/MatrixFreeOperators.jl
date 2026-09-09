# Included by device.jl only when MFO_TEST_GPU=true and CUDA.jl is available.
using CUDA

CUDA.allowscalar(false)

@testset "CUDA parity vs CPU" begin
    g = CartesianGrid(
        ((0.0, 2π), (0.0, 1.0)), (32, 24);
        bc=((Periodic(), Periodic()), (Dirichlet(), Neumann())),
    )
    κ = set!(scalar_field(g), x -> 1 + x[1] / 7)
    u = set!(scalar_field(g), x -> sin(x[1]) * x[2])

    @testset "operator action parity" begin
        for L in (laplacian(g), derivative(g, 2), scaling(κ) - laplacian(g))
            y_cpu = collect(interior(apply(L, copy(u))))
            Lg = Adapt.adapt(CuArray, L)
            ug = Adapt.adapt(CuArray, u)
            y_gpu = Array(collect(interior(apply(Lg, ug))))
            @test y_gpu ≈ y_cpu
        end
    end

    @testset "rank-changers parity" begin
        Gg = Adapt.adapt(CuArray, MatrixFreeOperators.gradient(g))
        ug = Adapt.adapt(CuArray, u)
        ∇u_gpu = apply(Gg, ug)
        ∇u_cpu = apply(MatrixFreeOperators.gradient(g), copy(u))
        @test Array(∇u_gpu.data) ≈ ∇u_cpu.data
    end

    @testset "BlockForest action parity" begin
        MFO = MatrixFreeOperators
        base = CartesianGrid(
            ((0.0, 2π), (0.0, 1.0)), (16, 16);
            bc=((Periodic(), Periodic()), (Dirichlet(), Dirichlet())),
        )
        bf = BlockForest(base; blocksize=(8, 8), maxlevel=2)
        uf = set!(scalar_field(bf), x -> sin(x[1]) * x[2])
        for L in (laplacian(bf), derivative(bf, 1; order=1))
            y_cpu = [
                collect(interior(MFO.block(apply(L, copy(uf)), i))) for i in 1:MFO.nleaves(bf)
            ]
            Lg = Adapt.adapt(CuArray, L)
            ug = Adapt.adapt(CuArray, uf)
            yg = apply(Lg, ug)
            for i in 1:MFO.nleaves(bf)
                @test Array(collect(interior(MFO.block(yg, i)))) ≈ y_cpu[i]
            end
        end
    end

    @testset "BlockForest coarse–fine action parity (non-uniform)" begin
        MFO = MatrixFreeOperators
        base = CartesianGrid(
            ((0.0, 2π), (0.0, 1.0)), (16, 16);
            bc=((Periodic(), Periodic()), (Dirichlet(), Dirichlet())),
        )
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=2)
        refine!(bf, x -> x[1] < π && x[2] < 0.5)       # mixed levels: CF ghost fills run
        @test !bf.forest.uniform[]
        uf = set!(scalar_field(bf), x -> sin(x[1]) * x[2])
        for L in (laplacian(bf), derivative(bf, 1; order=1))
            y_cpu = [
                collect(interior(MFO.block(apply(L, copy(uf)), i))) for i in 1:MFO.nleaves(bf)
            ]
            Lg = Adapt.adapt(CuArray, L)
            ug = Adapt.adapt(CuArray, uf)
            yg = apply(Lg, ug)
            for i in 1:MFO.nleaves(bf)
                @test Array(collect(interior(MFO.block(yg, i)))) ≈ y_cpu[i]
            end
        end
    end

    @testset "BlockForest flat/Krylov path parity" begin
        MFO = MatrixFreeOperators
        base = CartesianGrid(
            ((0.0, 2π), (0.0, 1.0)), (16, 16);
            bc=((Periodic(), Periodic()), (Dirichlet(), Dirichlet())),
        )
        bf = BlockForest(base; blocksize=(8, 8), maxlevel=2)
        uf = set!(scalar_field(bf), x -> sin(x[1]) * x[2])
        ug = Adapt.adapt(CuArray, uf)
        vg = flatten(ug)                       # device vector, no scalar indexing
        @test vg isa CuArray
        v = flatten(uf)
        @test Array(vg) ≈ v
        A = prepare(laplacian(bf), uf)
        Ag = prepare(Adapt.adapt(CuArray, laplacian(bf)), ug)
        out = similar(v)
        mul!(out, A, v)
        outg = similar(vg)
        mul!(outg, Ag, vg)
        @test Array(outg) ≈ out
    end

    @testset "PackedBlockField single-launch kernel parity" begin
        MFO = MatrixFreeOperators
        base = CartesianGrid(
            ((0.0, 2π), (0.0, 1.0)), (16, 16);
            bc=((Periodic(), Periodic()), (Dirichlet(), Dirichlet())),
        )
        bf = BlockForest(base; blocksize=(8, 8), maxlevel=2)
        uf = set!(scalar_field(bf), x -> sin(x[1]) * x[2])
        p = pack(uf)
        pg = Adapt.adapt(CuArray, p)
        @test pg.data isa CuArray
        @test pg.levels isa CuArray

        # un-prepared apply: halo/BC view sweeps + the forest-native kernel launch
        Lg = Adapt.adapt(CuArray, laplacian(bf))
        yg = apply(Lg, copy(pg))
        y = apply(laplacian(bf), copy(p))
        @test Array(yg.data) ≈ y.data

        # prepared flat path: kernel launch ordered before the flat copy-out
        v = flatten(p)
        A = prepare(laplacian(bf), p)
        out = similar(v)
        mul!(out, A, v)
        vg2 = flatten(Adapt.adapt(CuArray, pack(uf)))
        Ag = prepare(Lg, Adapt.adapt(CuArray, pack(uf)))
        outg = similar(vg2)
        mul!(outg, Ag, vg2)
        @test Array(outg) ≈ out

        # refined forest: per-leaf levels SoA feeds the kernel on device
        refine!(bf, x -> x[1] < π)
        ur = set!(scalar_field(bf), x -> sin(x[1]) * x[2])
        pr = pack(ur)
        yr = apply(laplacian(bf), copy(pr))
        prg = Adapt.adapt(CuArray, pr)
        yrg = apply(Adapt.adapt(CuArray, laplacian(bf)), copy(prg))
        @test Array(yrg.data) ≈ yr.data
    end

    @testset "part-2 packed kernels on device" begin
        MFO = MatrixFreeOperators
        base = CartesianGrid(
            ((0.0, 2π), (0.0, 1.0)), (16, 16);
            bc=((Periodic(), Periodic()), (Dirichlet(), Dirichlet())),
        )
        sfun = x -> sin(x[1]) * x[2]
        wfun = x -> SVector(sin(x[1]) + x[2], cos(x[1]) - 0.5 * x[2])
        for refined in (false, true)
            bf = BlockForest(base; blocksize=(8, 8), maxlevel=2)
            refined && refine!(bf, x -> x[1] < π)
            u = set!(scalar_field(bf), sfun)
            w = set!(vector_field(bf), wfun)
            κp = pack(set!(scalar_field(bf), x -> 1 + x[2]^2))
            velp = pack(set!(vector_field(bf), wfun))
            p = pack(u)
            pw = pack(w)
            pg = Adapt.adapt(CuArray, p)
            pwg = Adapt.adapt(CuArray, pw)

            # un-prepared public path: every part-2 forward kernel vs its CPU result
            for L in (
                derivative(bf, 1; order=1),
                derivative(bf, 2; order=2),
                MFO.gradient(bf),
                advection(bf, velp),
                scaling(2.5),
                scaling(κp),
                identity_op(),
            )
                y = apply(L, copy(p))
                yg = apply(Adapt.adapt(CuArray, L), copy(pg))
                @test Array(yg.data) ≈ y.data
            end
            yd = apply(divergence(bf), copy(pw))
            ydg = apply(Adapt.adapt(CuArray, divergence(bf)), copy(pwg))
            @test Array(ydg.data) ≈ yd.data

            # prepared flat path through a coefficient composite (normalized packed
            # coefficient rides the adapted tree)
            K = divergence(bf) * scaling(κp) * MFO.gradient(bf)
            v = flatten(p)
            out = similar(v)
            mul!(out, prepare(K, p), v)
            pg2 = Adapt.adapt(CuArray, pack(u))
            vg = flatten(pg2)
            outg = similar(vg)
            mul!(outg, prepare(Adapt.adapt(CuArray, K), pg2), vg)
            @test Array(outg) ≈ out

            # declared adjoint transpose-gather kernels + fold path on device
            ys = pack(set!(scalar_field(bf), x -> cos(x[1]) + x[2]^2))
            for L in (derivative(bf, 1; order=1), MFO.gradient(bf))
                ȳ = L isa MFO.Gradient ? pack(set!(vector_field(bf), wfun)) : ys
                x̄ = apply_adjoint!(MFO.allocate_input(L, ȳ), L, copy(ȳ), bf)
                x̄g = apply_adjoint!(
                    MFO.allocate_input(Adapt.adapt(CuArray, L), Adapt.adapt(CuArray, ȳ)),
                    Adapt.adapt(CuArray, L),
                    Adapt.adapt(CuArray, copy(ȳ)),
                    pg.grid,
                )
                @test Array(x̄g.data) ≈ x̄.data
            end

            # compact flux-form diffusion: packed-κ forward and adjoint kernels,
            # both averaging policies; on the refined forest the coarse–fine flux
            # rewrite runs as device-view broadcasts ahead of the launch
            for avg in (ArithmeticMean(), HarmonicMean())
                D = diffusion(bf, set!(scalar_field(bf), x -> 1 + x[2]^2); averaging=avg)
                Dp = MFO.Diffusion(bf, pack(D.κ), D.avg)
                Dg = Adapt.adapt(CuArray, Dp)
                y = apply(Dp, copy(p))
                yg = apply(Dg, copy(pg))
                @test Array(yg.data) ≈ y.data
                x̄ = apply_adjoint!(MFO.allocate_input(Dp, ys), Dp, copy(ys), bf)
                x̄g = apply_adjoint!(
                    MFO.allocate_input(Dg, Adapt.adapt(CuArray, ys)),
                    Dg,
                    Adapt.adapt(CuArray, copy(ys)),
                    pg.grid,
                )
                @test Array(x̄g.data) ≈ x̄.data
            end
        end
    end

    @testset "part-3 batched exchange/BC + flat broadcasts on device" begin
        MFO = MatrixFreeOperators
        base = CartesianGrid(
            ((0.0, 2π), (0.0, 1.0)), (16, 16);
            bc=((Periodic(), Periodic()), (Dirichlet(), Dirichlet())),
        )
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=2)
        refine!(bf, x -> x[1] < π)
        u = set!(scalar_field(bf), x -> sin(x[1]) * x[2])
        p = pack(u)
        # host reference exchange + BC
        pr = copy(p)
        MFO.halo_update!(pr, bf)
        MFO.apply_bc!(pr, bf)
        # kernelized path through the public API on the adapted twin
        pg = Adapt.adapt(CuArray, copy(p))
        MFO.halo_update!(pg, pg.grid)
        MFO.apply_bc!(pg, pg.grid)
        back = Array(pg.data)
        h, n = bf.halo, bf.blocksize
        for l in 1:MFO.nleaves(bf), I in CartesianIndices(MFO._block_array(pr, l))
            out = count(d -> !(h[d] < I[d] <= h[d] + n[d]), 1:2)
            out >= 2 && continue    # copy corners: documented divergence
            @test back[I, l] ≈ MFO._block_array(pr, l)[I]
        end
        # prepared mul! (single-broadcast flat transfers) parity, incl. axpby
        v = flatten(p)
        out = similar(v)
        mul!(out, prepare(laplacian(bf), p), v)
        pg2 = Adapt.adapt(CuArray, pack(u))
        vg = flatten(pg2)
        outg = similar(vg)
        Ag = prepare(Adapt.adapt(CuArray, laplacian(bf)), pg2)
        mul!(outg, Ag, vg)
        @test Array(outg) ≈ out
        out2 = copy(v)
        outg2 = CuArray(copy(v))
        mul!(out2, prepare(laplacian(bf), p), v, 2.0, 3.0)
        mul!(outg2, Ag, vg, 2.0, 3.0)
        @test Array(outg2) ≈ out2
        # regrid invalidates the device schedule: fresh generation rebuilds
        refine!(bf, x -> x[2] < 0.5)
        u2 = set!(scalar_field(bf), x -> sin(x[1]) * x[2])
        pr2 = pack(u2)
        MFO.halo_update!(pr2, bf)
        pg3 = Adapt.adapt(CuArray, pack(u2))
        MFO.halo_update!(pg3, pg3.grid)
        back3 = Array(pg3.data)
        for l in 1:MFO.nleaves(bf), I in CartesianIndices(MFO._block_array(pr2, l))
            out3 = count(d -> !(h[d] < I[d] <= h[d] + n[d]), 1:2)
            out3 >= 2 && continue
            @test back3[I, l] ≈ MFO._block_array(pr2, l)[I]
        end
    end

    @testset "Krylov cg parity" begin
        σ = set!(scalar_field(g), x -> 1 + x[2])
        K = scaling(σ) - laplacian(g)
        f = set!(scalar_field(g), x -> sin(x[1]))

        P_cpu = prepare(K, scalar_field(g))
        b_cpu = flatten(f)
        u_cpu, stats_cpu = Krylov.cg(P_cpu, b_cpu)
        @test stats_cpu.solved

        Kg = Adapt.adapt(CuArray, K)
        xg = Adapt.adapt(CuArray, scalar_field(g))
        P_gpu = prepare(Kg, xg)
        b_gpu = CuArray(b_cpu)
        u_gpu, stats_gpu = Krylov.cg(P_gpu, b_gpu)
        @test stats_gpu.solved
        @test Array(u_gpu) ≈ u_cpu rtol = 1e-6
    end

    @testset "AMR regrid! driver parity" begin
        MFO = MatrixFreeOperators
        dirbc = ((Dirichlet(), Dirichlet()), (Dirichlet(), Dirichlet()))
        mk() = BlockForest(
            CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (32, 32); bc=dirbc);
            blocksize=(8, 8), maxlevel=2,
        )
        c = (0.7, 0.3)
        s = 0.005
        bump = x -> exp(-((x[1] - c[1])^2 + (x[2] - c[2])^2) / s)
        rhsf = x -> (4 / s - 4 * ((x[1] - c[1])^2 + (x[2] - c[2])^2) / s^2) * bump(x)
        crit = b -> maximum(abs, interior(b)) > 0.1          # device reduction only

        # Adapt shares the forest/schedule Refs and a regrid mutates them, so
        # parity needs two independent forests; the GPU side adapts a twin whose
        # CPU original is used only to build the initial data.
        bf_cpu = mk()
        u_cpu = set!(scalar_field(bf_cpu), bump)
        u_gpu = Adapt.adapt(CuArray, set!(scalar_field(mk()), bump))
        bf_gpu = u_gpu.grid
        @test first(u_gpu.blocks) isa CuArray

        u_cpu = regrid!(u_cpu; refine=crit)
        u_gpu = regrid!(u_gpu; refine=crit)
        @test !bf_cpu.forest.uniform[]
        @test bf_gpu.forest.leaves == bf_cpu.forest.leaves   # identical marks + balance
        for i in 1:MFO.nleaves(bf_cpu)
            @test Array(collect(interior(MFO.block(u_gpu, i)))) ≈
                collect(interior(MFO.block(u_cpu, i)))
        end

        # one adaptive Krylov cycle per device: solve → refine → transfer → re-prepare → solve
        function cycle!(u, bf, rhs)
            P = prepare(laplacian(bf), u)
            sol, stats = Krylov.gmres(P, rhs; rtol=1e-10)    # nonsymmetric on adapted forest
            @test stats.solved
            flat_to_interior!(u, sol)
            return sol
        end
        rhs = .-flatten(set!(scalar_field(bf_cpu), rhsf))    # identical topology ⇒ same layout
        @test Array(cycle!(u_gpu, bf_gpu, CuArray(rhs))) ≈ cycle!(u_cpu, bf_cpu, rhs) rtol = 1e-6

        crit2 = b -> maximum(abs, interior(b)) > 0.5
        u_cpu = regrid!(u_cpu; refine=crit2)
        u_gpu = regrid!(u_gpu; refine=crit2)                 # prolongs solved data on device
        @test bf_gpu.forest.leaves == bf_cpu.forest.leaves
        rhs2 = .-flatten(set!(scalar_field(bf_cpu), rhsf))
        @test Array(cycle!(u_gpu, bf_gpu, CuArray(rhs2))) ≈ cycle!(u_cpu, bf_cpu, rhs2) rtol = 1e-6

        # coarsen everything back: the conservative child-mean path on device
        u_cpu = regrid!(u_cpu; refine=Returns(false), coarsen=Returns(true))
        u_gpu = regrid!(u_gpu; refine=Returns(false), coarsen=Returns(true))
        @test bf_gpu.forest.leaves == bf_cpu.forest.leaves
        for i in 1:MFO.nleaves(bf_cpu)
            @test Array(collect(interior(MFO.block(u_gpu, i)))) ≈
                collect(interior(MFO.block(u_cpu, i))) rtol = 1e-6
        end
    end
end
