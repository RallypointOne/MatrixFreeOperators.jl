@testset "Prepared operators and Krylov interop" begin
    @testset "size, eltype, and error stubs" begin
        g = CartesianGrid(((0.0, 1.0),), (32,))
        L = laplacian(g)
        P = prepare(L)
        @test size(P) == (32, 32)
        @test size(P, 1) == 32
        @test eltype(P) === Float64

        Pv = prepare(L, vector_field(g))
        @test size(Pv) == (32, 32)

        g2 = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (4, 4))
        Pv2 = prepare(laplacian(g2), vector_field(g2))
        @test size(Pv2) == (32, 32)
        @test eltype(Pv2) === Float64

        x = rand(32)
        y = similar(x)
        @test_throws ArgumentError mul!(y, L, x)
        @test_throws ArgumentError mul!(y, L, x, 1.0, 0.0)
        @test_throws ArgumentError prepare(DummyNonlinearOp(), scalar_field(g))
    end

    @testset "5-arg mul! axpby semantics" begin
        g = CartesianGrid(((0.0, 1.0),), (8,); bc=((Neumann(), Dirichlet()),))
        P = prepare(laplacian(g))
        A = materialize(P)
        rng = Random.MersenneTwister(5)
        x = rand(rng, 8)
        y0 = rand(rng, 8)
        y = copy(y0)
        mul!(y, P, x, 2.5, -0.5)
        @test y ≈ 2.5 .* (A * x) .- 0.5 .* y0
        mul!(y, P, x)
        @test y ≈ A * x
    end

    @testset "mul! never mutates the solver's vectors" begin
        g = CartesianGrid(((0.0, 1.0),), (8,))
        P = prepare(laplacian(g))
        x = rand(8)
        x_before = copy(x)
        y = similar(x)
        mul!(y, P, x)
        @test x == x_before
    end

    @testset "SVector flat boundary matches componentwise scalar path" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (4, 3);
            bc=((Periodic(), Periodic()), (Dirichlet(), Dirichlet())),
        )
        Pv = prepare(laplacian(g), vector_field(g))
        Ps = prepare(laplacian(g), scalar_field(g))
        As = materialize(Ps)
        rng = Random.MersenneTwister(9)
        n = prod(local_size(g))
        xv = rand(rng, 2 * n)
        yv = similar(xv)
        mul!(yv, Pv, xv)
        x1 = xv[1:2:end]
        x2 = xv[2:2:end]
        @test yv[1:2:end] ≈ As * x1
        @test yv[2:2:end] ≈ As * x2
    end

    @testset "Krylov Poisson solve (homogeneous Dirichlet)" begin
        function poisson_error(n)
            g = CartesianGrid(((0.0, 1.0),), (n,))
            A = prepare(laplacian(g))
            f = set!(scalar_field(g), x -> π^2 * sin(π * x[1]))
            b = -flatten(f)
            u, stats = Krylov.minres(A, b)
            @test stats.solved
            ustar = flatten(set!(scalar_field(g), x -> sin(π * x[1])))
            return maximum(abs, u .- ustar)
        end
        e32 = poisson_error(32)
        e64 = poisson_error(64)
        @test e32 < 0.01
        @test e32 / e64 > 3
    end

    @testset "steady-state allocations of prepared mul!" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (16, 16);
            bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
        )
        P = prepare(laplacian(g))
        x = rand(256)
        y = similar(x)
        mul!(y, P, x)
        mul!(y, P, x)
        alloc = @allocated mul!(y, P, x)
        @test alloc ≤ 512
    end
end
