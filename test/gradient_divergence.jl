function periodic_grid_2d(n)
    return CartesianGrid(
        ((0.0, 2π), (0.0, 2π)), (n, n);
        bc=((Periodic(), Periodic()), (Periodic(), Periodic())),
    )
end

@testset "Gradient and Divergence (rank-changers)" begin
    @testset "gradient analytic action" begin
        g = periodic_grid_2d(32)
        u = set!(scalar_field(g), x -> sin(x[1]) * sin(x[2]))
        ∇u = MatrixFreeOperators.gradient(g) * u
        @test eltype(∇u) === SVector{2,Float64}
        ref = set!(vector_field(g), x -> SVector(cos(x[1]) * sin(x[2]), sin(x[1]) * cos(x[2])))
        @test maximum(norm.(collect(interior(∇u)) .- collect(interior(ref)))) < 0.01
    end

    @testset "divergence analytic action" begin
        g = periodic_grid_2d(32)
        v = set!(vector_field(g), x -> SVector(sin(x[1]) * cos(x[2]), cos(x[1]) * sin(x[2])))
        divv = divergence(g) * v
        @test eltype(divv) === Float64
        ref = set!(scalar_field(g), x -> 2 * cos(x[1]) * cos(x[2]))
        @test maximum(abs, collect(interior(divv)) .- collect(interior(ref))) < 0.02
    end

    @testset "div ∘ grad agrees with laplacian (both vs analytic)" begin
        g = periodic_grid_2d(48)
        u = set!(scalar_field(g), x -> sin(x[1]) * sin(x[2]))
        wide = apply(divergence(g), apply(MatrixFreeOperators.gradient(g), u))
        compact = laplacian(g) * u
        ref = -2 .* collect(interior(set!(scalar_field(g), x -> sin(x[1]) * sin(x[2]))))
        @test maximum(abs, collect(interior(wide)) .- ref) < 0.02
        @test maximum(abs, collect(interior(compact)) .- ref) < 0.02
    end

    @testset "rectangular sizes and eltype checks" begin
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (4, 3))
        G = MatrixFreeOperators.gradient(g)
        D = divergence(g)
        @test size(G) == (24, 12)
        @test size(D) == (12, 24)
        @test_throws ArgumentError apply(G, vector_field(g))
        @test_throws ArgumentError apply(D, scalar_field(g))
    end

    @testset "dense transpose through the flat boundary under $(nameof(typeof(bc)))" for bc in (
        Periodic(), Dirichlet(), Neumann()
    )
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (3, 4); bc=((bc, bc), (bc, bc)))
        G = MatrixFreeOperators.gradient(g)
        A = materialize(prepare(G, scalar_field(g)))
        At = materialize(prepare(adjoint(G), vector_field(g)))
        @test size(A) == (24, 12) && size(At) == (12, 24)
        @test At ≈ A'

        D = divergence(g)
        B = materialize(prepare(D, vector_field(g)))
        Bt = materialize(prepare(adjoint(D), scalar_field(g)))
        @test size(B) == (12, 24) && size(Bt) == (24, 12)
        @test Bt ≈ B'
    end

    @testset "adjoint identity through fields" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (5, 4);
            bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
        )
        rng = Random.MersenneTwister(33)
        G = MatrixFreeOperators.gradient(g)
        x = scalar_field(g)
        y = vector_field(g)
        interior(x) .= rand(rng, local_size(g)...)
        interior(y) .= [SVector(rand(rng), rand(rng)) for _ in 1:5, _ in 1:4]
        Gx = apply(G, copy(x))
        Gty = apply(adjoint(G), copy(y))
        @test sum(dot.(collect(interior(Gx)), collect(interior(y)))) ≈
            dot(collect(interior(x)), collect(interior(Gty)))
    end
end
