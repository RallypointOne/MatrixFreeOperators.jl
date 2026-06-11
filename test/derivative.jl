function deriv_periodic_error(n::Int, order::Int)
    g = CartesianGrid(((0.0, 2π),), (n,); bc=((Periodic(), Periodic()),))
    u = set!(scalar_field(g), x -> sin(x[1]))
    y = derivative(g, 1; order) * u
    ref = set!(scalar_field(g), order == 1 ? (x -> cos(x[1])) : (x -> -sin(x[1])))
    return maximum(abs, collect(interior(y)) .- collect(interior(ref)))
end

@testset "Derivative" begin
    @testset "validation" begin
        g = CartesianGrid(((0.0, 1.0),), (4,))
        @test_throws ArgumentError derivative(g, 2)
        @test_throws ArgumentError derivative(g, 0)
        @test_throws ArgumentError derivative(g, 1; order=3)
    end

    @testset "analytic action and convergence (order=$o)" for o in (1, 2)
        e32 = deriv_periodic_error(32, o)
        e64 = deriv_periodic_error(64, o)
        @test log2(e32 / e64) ≥ 1.9
    end

    @testset "acts along the requested dimension" begin
        g = CartesianGrid(
            ((0.0, 2π), (0.0, 2π)), (32, 32);
            bc=((Periodic(), Periodic()), (Periodic(), Periodic())),
        )
        u = set!(scalar_field(g), x -> sin(x[2]))
        dy = derivative(g, 2) * u
        ref = set!(scalar_field(g), x -> cos(x[2]))
        @test maximum(abs, collect(interior(dy)) .- collect(interior(ref))) < 0.01
        dx = derivative(g, 1) * u
        @test maximum(abs, collect(interior(dx))) < 1e-12
    end

    @testset "dense transpose: matrix(D') == matrix(D)' under $(nameof(typeof(bc)))" for bc in (
        Periodic(), Dirichlet(), Neumann()
    )
        g = CartesianGrid(((0.0, 1.0),), (6,); bc=((bc, bc),))
        for o in (1, 2)
            D = derivative(g, 1; order=o)
            A = materialize(prepare(D))
            At = materialize(prepare(adjoint(D)))
            @test At ≈ A'
        end
    end

    @testset "traits and declared adjoints" begin
        g = CartesianGrid(((0.0, 1.0),), (6,))
        D1 = derivative(g, 1)
        D2 = derivative(g, 1; order=2)
        @test islinear(D1) && isconstant(D1)
        @test !isselfadjoint(D1)
        @test isselfadjoint(D2)
        @test adjoint(D2) === D2
        @test adjoint(D1) isa AdjointOp
        @test MatrixFreeOperators.adjoint_operator(adjoint(D1)) === D1
    end

    @testset "adjoint identity on mixed-BC 2-D grid" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (5, 4);
            bc=((Dirichlet(), Neumann()), (Periodic(), Periodic())),
        )
        rng = Random.MersenneTwister(21)
        for dims in (1, 2), o in (1, 2)
            D = derivative(g, dims; order=o)
            x = scalar_field(g)
            y = scalar_field(g)
            interior(x) .= rand(rng, local_size(g)...)
            interior(y) .= rand(rng, local_size(g)...)
            Lx = apply(D, copy(x))
            Lty = apply(adjoint(D), copy(y))
            @test dot(collect(interior(Lx)), collect(interior(y))) ≈
                dot(collect(interior(x)), collect(interior(Lty)))
        end
    end
end
