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
