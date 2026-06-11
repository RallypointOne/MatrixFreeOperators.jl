@testset "linearize / linearize!" begin
    @testset "Jacobian of a linear operator is the operator" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (6, 5);
            bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
        )
        rng = Random.MersenneTwister(41)
        for F in (laplacian(g), advection(g, set!(vector_field(g), x -> SVector(x[1], 1.0))))
            u0 = scalar_field(g)
            v = scalar_field(g)
            interior(u0) .= rand(rng, local_size(g)...)
            interior(v) .= rand(rng, local_size(g)...)
            J = linearize(F, u0)
            @test islinear(J)
            Jv = apply(J, copy(v))
            Fv = apply(F, copy(v))
            @test collect(interior(Jv)) ≈ collect(interior(Fv)) rtol = 1e-6
        end
    end

    @testset "self-advection JVP matches directional finite difference" begin
        g = CartesianGrid(((0.0, 2π),), (32,); bc=((Periodic(), Periodic()),))
        F = advection(g, SelfAdvection())
        u0 = set!(vector_field(g), x -> SVector(sin(x[1])))
        v = set!(vector_field(g), x -> SVector(cos(2 * x[1])))
        J = linearize(F, u0)
        Jv = apply(J, copy(v))

        # analytic: ∂/∂ε (u+εv)·∇(u+εv)|₀ = v·∇u + u·∇v
        ref = set!(
            vector_field(g),
            x -> SVector(cos(2 * x[1]) * cos(x[1]) - 2 * sin(x[1]) * sin(2 * x[1])),
        )
        @test maximum(norm.(collect(interior(Jv)) .- collect(interior(ref)))) < 0.05

        ε = 1e-6
        up = copy(u0)
        um = copy(u0)
        up.data .= u0.data .+ ε .* v.data
        um.data .= u0.data .- ε .* v.data
        fd = (collect(interior(apply(F, up))) .- collect(interior(apply(F, um)))) ./ (2 * ε)
        @test maximum(norm.(collect(interior(Jv)) .- fd)) < 1e-4
    end

    @testset "linearize! refresh equals fresh linearize" begin
        g = CartesianGrid(((0.0, 2π),), (16,); bc=((Periodic(), Periodic()),))
        F = advection(g, SelfAdvection())
        u0 = set!(vector_field(g), x -> SVector(sin(x[1])))
        u1 = set!(vector_field(g), x -> SVector(cos(x[1])))
        v = set!(vector_field(g), x -> SVector(sin(2 * x[1])))

        J = linearize(F, u0)
        linearize!(J, u1)
        fresh = linearize(F, u1)
        @test collect(interior(apply(J, copy(v)))) ≈
            collect(interior(apply(fresh, copy(v))))
    end

    @testset "prepared Jacobian drives Krylov (implicit-Euler JFNK system)" begin
        g = CartesianGrid(((0.0, 2π),), (24,); bc=((Periodic(), Periodic()),))
        F = advection(g, SelfAdvection())
        u0 = set!(vector_field(g), x -> SVector(2 + sin(x[1]) / 4))
        J = linearize(F, u0)
        @test_throws ArgumentError adjoint(J)
        @test size(J) == (24, 24)

        v = set!(vector_field(g), x -> SVector(cos(x[1])))
        Pj = prepare(J, u0)
        jv = similar(flatten(v))
        mul!(jv, Pj, flatten(v))
        @test jv ≈ flatten(apply(J, copy(v))) rtol = 1e-6

        dt = 0.01
        A = identity_op() - dt * J          # the implicit-stepping Newton–Krylov system
        @test islinear(A)
        P = prepare(A, u0)
        b = rand(Random.MersenneTwister(55), 24)
        x, stats = Krylov.gmres(P, b)
        @test stats.solved
        r = similar(b)
        mul!(r, P, x)
        @test maximum(abs, r .- b) < 1e-6
    end
end
