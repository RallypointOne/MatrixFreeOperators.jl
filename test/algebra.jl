@testset "Operator algebra" begin
    g = periodic_grid_2d(24)
    u = set!(scalar_field(g), x -> sin(x[1]) * sin(x[2]))
    v = set!(vector_field(g), x -> SVector(sin(x[2]), cos(x[1])))
    A = laplacian(g)
    B = advection(g, v)
    ints(f) = collect(interior(f))

    @testset "algebra identities (§12.3)" begin
        @test ints((A * B) * u) ≈ ints(A * (B * u))
        @test ints((A + B) * u) ≈ ints(A * u) .+ ints(B * u)
        @test ints((A - B) * u) ≈ ints(A * u) .- ints(B * u)
        @test ints((2.5 * A) * u) ≈ 2.5 .* ints(A * u)
        @test ints((A * 2.5) * u) ≈ 2.5 .* ints(A * u)
        @test ints((A / 2) * u) ≈ ints(A * u) ./ 2
        @test ints((-A) * u) ≈ -ints(A * u)
    end

    @testset "adjoint propagation" begin
        @test adjoint(A * B) isa Composed
        rng = Random.MersenneTwister(13)
        x = scalar_field(g)
        y = scalar_field(g)
        interior(x) .= rand(rng, local_size(g)...)
        interior(y) .= rand(rng, local_size(g)...)
        lhs = apply(adjoint(A * B), copy(y))
        rhs = apply(adjoint(B), apply(adjoint(A), copy(y)))
        @test ints(lhs) ≈ ints(rhs)
        @test dot(ints((A * B) * x), ints(y)) ≈ dot(ints(x), ints(lhs))
        @test dot(ints((A + B) * x), ints(y)) ≈ dot(ints(x), ints(apply(adjoint(A + B), copy(y))))
        @test dot(ints((3 * A) * x), ints(y)) ≈ dot(ints(x), ints(apply(adjoint(3 * A), copy(y))))
    end

    @testset "trait propagation truth table" begin
        S = scaling(2.0)
        I = identity_op()
        @test islinear(A + B) && islinear(A * B) && islinear(2 * A)
        @test isconstant(A + B) && isconstant(A * B)
        @test isdiagonal(S * I) && isdiagonal(S + I) && isdiagonal(2 * S)
        @test !isdiagonal(A * S)
        @test isselfadjoint(A + S)
        @test isselfadjoint(2.0 * A)
        @test !isselfadjoint(S * I)       # never compositional, even for commuting diagonals
        @test !isselfadjoint(A * B)

        NL = advection(g, SelfAdvection())
        @test !islinear(A + NL)
        @test !islinear(A * NL)
        @test !islinear(2 * NL)
        @test_throws ArgumentError adjoint(A + NL)
        @test_throws ArgumentError adjoint(A * NL)
        @test_throws ArgumentError prepare(A * NL, vector_field(g))
    end

    @testset "variable-coefficient diffusion ∇·(κ∇u) stress test" begin
        function vc_error(n)
            gn = periodic_grid_2d(n)
            κ = set!(scalar_field(gn), x -> 2 + cos(x[1]))
            un = set!(scalar_field(gn), x -> sin(x[1]) * sin(x[2]))
            K = divergence(gn) * scaling(κ) * MatrixFreeOperators.gradient(gn)
            y = K * un
            ref = set!(
                scalar_field(gn),
                x ->
                    -2 * (2 + cos(x[1])) * sin(x[1]) * sin(x[2]) -
                    sin(x[1]) * cos(x[1]) * sin(x[2]),
            )
            return maximum(abs, collect(interior(y)) .- collect(interior(ref)))
        end
        e24 = vc_error(24)
        e48 = vc_error(48)
        @test log2(e24 / e48) ≥ 1.8

        gn = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (4, 3);
            bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
        )
        κ = set!(scalar_field(gn), x -> 1 + x[1] * x[2])
        K = divergence(gn) * scaling(κ) * MatrixFreeOperators.gradient(gn)
        M = materialize(prepare(K, scalar_field(gn)))
        Mt = materialize(prepare(adjoint(K), scalar_field(gn)))
        @test Mt ≈ M'
    end

    @testset "prescribed-advection adjoint (algebra expression)" begin
        gn = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (4, 4);
            bc=((Periodic(), Periodic()), (Dirichlet(), Neumann())),
        )
        vn = set!(vector_field(gn), x -> SVector(1 + x[1], x[2] - 2))
        Adv = advection(gn, vn)
        M = materialize(prepare(Adv, scalar_field(gn)))
        Mt = materialize(prepare(adjoint(Adv), scalar_field(gn)))
        @test Mt ≈ M'
    end

    @testset "IdentityOp composes and resolves size from siblings" begin
        H = A - 4 * identity_op()
        @test ints(H * u) ≈ ints(A * u) .- 4 .* ints(u)
        @test size(H) == size(A)
        @test size(identity_op() * A) == size(A)
        @test_throws ArgumentError size(identity_op() * identity_op())
        @test isselfadjoint(H)
        @test sprint(show, H) == "(Laplacian + (-4 * IdentityOp))"
    end
end
