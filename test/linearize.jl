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

    @testset "EnzymeJVP: exact JVP and a real transpose" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (6, 5);
            bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
        )
        rng = Random.MersenneTwister(93)
        vel = set!(vector_field(g), x -> SVector(x[1], 1.0))

        @testset "Jacobian of the linear $(name) is the operator, exactly" for (name, F) in
                                                                              (
            ("Laplacian", laplacian(g)), ("Advection", advection(g, vel))
        )
            u0 = scalar_field(g)
            interior(u0) .= rand(rng, local_size(g)...)
            J = linearize(F, u0, EnzymeJVP())
            @test islinear(J)
            x = scalar_field(g)
            interior(x) .= rand(rng, local_size(g)...)
            # Forward-mode AD of a linear map returns the map itself — no ε, so this
            # is exact rather than the 1e-6 the finite-difference JVP can manage.
            @test collect(interior(apply(J, copy(x)))) ≈
                collect(interior(apply(F, copy(x)))) rtol = 1e-14

            # The transpose the finite-difference Jacobian cannot provide.
            y = scalar_field(g)
            interior(y) .= rand(rng, local_size(g)...)
            lhs = dot(collect(interior(apply(J, copy(x)))), collect(interior(y)))
            rhs = dot(collect(interior(x)), collect(interior(apply(adjoint(J), copy(y)))))
            @test lhs ≈ rhs rtol = 1e-12
        end

        @testset "nonlinear self-advection: JVP vs FD, and the adjoint identity" begin
            g1 = CartesianGrid(((0.0, 2π),), (16,); bc=((Periodic(), Periodic()),))
            F = advection(g1, SelfAdvection())
            u0 = set!(vector_field(g1), x -> SVector(2 + sin(x[1])))
            J = linearize(F, u0, EnzymeJVP())

            x = vector_field(g1)
            interior(x) .= [SVector(randn(rng)) for _ in 1:16]
            Jx = collect(interior(apply(J, copy(x))))

            ε = 1e-6
            up = copy(u0)
            up.data .= u0.data .+ ε .* x.data
            um = copy(u0)
            um.data .= u0.data .- ε .* x.data
            fd = (collect(interior(apply(F, up))) .- collect(interior(apply(F, um)))) ./ (2ε)
            @test maximum(norm.(Jx .- fd)) < 1e-6

            # ⟨Jx, y⟩ = ⟨x, Jᵀy⟩ for the frozen Jacobian of a nonlinear operator.
            for _ in 1:3
                y = vector_field(g1)
                interior(y) .= [SVector(randn(rng)) for _ in 1:16]
                lhs = sum(dot.(collect(interior(apply(J, copy(x)))), collect(interior(y))))
                rhs = sum(
                    dot.(
                        collect(interior(x)),
                        collect(interior(apply(adjoint(J), copy(y)))),
                    )
                )
                @test lhs ≈ rhs rtol = 1e-10
            end
        end

        @testset "linearize! refresh, and the unloaded-backend error" begin
            g1 = CartesianGrid(((0.0, 2π),), (16,); bc=((Periodic(), Periodic()),))
            F = advection(g1, SelfAdvection())
            u0 = set!(vector_field(g1), x -> SVector(sin(x[1])))
            u1 = set!(vector_field(g1), x -> SVector(cos(x[1])))
            v = set!(vector_field(g1), x -> SVector(sin(2 * x[1])))
            J = linearize(F, u0, EnzymeJVP())
            linearize!(J, u1)
            fresh = linearize(F, u1, EnzymeJVP())
            @test collect(interior(apply(J, copy(v)))) ≈
                collect(interior(apply(fresh, copy(v))))

            # A backend with no loaded implementation must say so, not fall back.
            struct _UnloadedJVP <: MatrixFreeOperators.AbstractJVPBackend end
            @test_throws ArgumentError linearize(F, u0, _UnloadedJVP())
        end
    end
end
