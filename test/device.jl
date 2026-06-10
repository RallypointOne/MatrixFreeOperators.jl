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

    @testset "GPU parity" begin
        if get(ENV, "MFO_TEST_GPU", "") == "true" && Base.find_package("CUDA") !== nothing
            include("device_gpu.jl")
        else
            @test_skip "GPU parity — run with MFO_TEST_GPU=true and CUDA.jl available"
        end
    end
end
