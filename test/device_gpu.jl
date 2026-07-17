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
end
