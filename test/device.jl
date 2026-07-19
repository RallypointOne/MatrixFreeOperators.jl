@testset "Device transfer (Adapt)" begin
    @testset "CPU round trip preserves structure and behavior" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (6, 5);
            bc=((Dirichlet(), Neumann()), (Periodic(), Periodic())),
        )
        κ = set!(scalar_field(g), x -> 1 + x[1])
        v = set!(vector_field(g), x -> SVector(x[1], 1.0))
        u = set!(scalar_field(g), x -> sin(x[1]) * x[2])

        K = divergence(g) * scaling(κ) * MatrixFreeOperators.gradient(g)
        ops = (
            laplacian(g),
            derivative(g, 1),
            MatrixFreeOperators.gradient(g),
            divergence(g),
            scaling(κ),
            identity_op(),
            advection(g, v),
            advection(g, SelfAdvection()),
            adjoint(derivative(g, 1)),
            2.5 * laplacian(g) + scaling(κ),
            K,
            linearize(advection(g, v), u),
        )
        for L in ops
            L2 = Adapt.adapt(Array, L)
            @test typeof(L2) === typeof(L)
        end

        f = Adapt.adapt(Array, u)
        @test f.data == u.data
        @test KernelAbstractions.get_backend(f.grid) == KernelAbstractions.CPU()

        K2 = Adapt.adapt(Array, K)
        @test collect(interior(apply(K2, copy(u)))) ≈ collect(interior(apply(K, copy(u))))
    end

    @testset "BlockForest CPU round trip" begin
        MFO = MatrixFreeOperators
        base = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (8, 8);
            bc=((Dirichlet(), Dirichlet()), (Periodic(), Periodic())),
        )
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=2)
        uf = set!(scalar_field(bf), x -> sin(x[1]) * x[2])
        L = laplacian(bf)

        L2 = Adapt.adapt(Array, L)
        @test typeof(L2) === typeof(L)

        uf2 = Adapt.adapt(Array, uf)
        @test uf2 isa BlockField
        @test all(i -> uf2.blocks[i] == uf.blocks[i], 1:MFO.nleaves(bf))
        @test KernelAbstractions.get_backend(uf2.grid) == KernelAbstractions.CPU()

        y2 = reduce(vcat, collect(interior(MFO.block(apply(L2, copy(uf2)), i))) for i in 1:MFO.nleaves(bf))
        y1 = reduce(vcat, collect(interior(MFO.block(apply(L, copy(uf)), i))) for i in 1:MFO.nleaves(bf))
        @test y2 ≈ y1

        p = pack(uf)
        p2 = Adapt.adapt(Array, p)
        @test p2 isa PackedBlockField
        @test p2.data == p.data
        @test collect(p2.levels) == collect(p.levels)
        @test KernelAbstractions.get_backend(p2.grid) == KernelAbstractions.CPU()
        yp = apply(Adapt.adapt(Array, L), copy(p2))
        @test all(
            i -> collect(interior(MFO.block(yp, i))) ≈ collect(interior(MFO.block(apply(L, copy(uf)), i))),
            1:MFO.nleaves(bf),
        )
    end

    @testset "GPU parity" begin
        if get(ENV, "MFO_TEST_GPU", "") == "true" && Base.find_package("CUDA") !== nothing
            include("device_gpu.jl")
        else
            @test_skip "GPU parity — run with MFO_TEST_GPU=true and CUDA.jl available"
        end
    end
end
