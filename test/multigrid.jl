@testset "Multigrid" begin
    @testset "operator_diagonal" begin
        # Laplacian under all-Periodic: a uniform Number, matching the dense diagonal
        for D in 1:3
            gp = CartesianGrid(
                ntuple(_ -> (0.0, 1.0), D),
                ntuple(_ -> 6, D);
                bc=ntuple(_ -> (Periodic(), Periodic()), D),
            )
            d = operator_diagonal(laplacian(gp))
            @test d isa Number
            @test all(diag(materialize(prepare(laplacian(gp)))) .≈ d)
        end

        # Dirichlet/Neumann faces adjust the boundary cells: exact vs dense diagonal
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 2.0)),
            (4, 6);
            bc=((Dirichlet(), Neumann()), (Periodic(), Periodic())),
        )
        L = laplacian(g)
        d = operator_diagonal(L)
        @test d isa Field
        @test flatten(d) ≈ diag(materialize(prepare(L)))

        g1 = CartesianGrid(((0.0, 1.0),), (5,); bc=((Dirichlet(), Dirichlet()),))
        @test flatten(operator_diagonal(laplacian(g1))) ≈
            diag(materialize(prepare(laplacian(g1))))

        # ScalingOp / IdentityOp leaves
        @test operator_diagonal(scaling(2.5)) == 2.5
        κ = set!(scalar_field(g), x -> 1 + x[1]^2)
        @test operator_diagonal(scaling(κ)) === κ
        @test operator_diagonal(identity_op()) === true

        # combinators: Scaled, Added, Composed-of-diagonals, AdjointOp
        gp = CartesianGrid(
            ((0.0, 1.0),), (6,); bc=((Periodic(), Periodic()),)
        )
        @test operator_diagonal(3 * laplacian(gp)) ≈ 3 * operator_diagonal(laplacian(gp))
        σ = set!(scalar_field(g), x -> x[1] + x[2])
        M = scaling(σ) - laplacian(g)
        @test flatten(operator_diagonal(M)) ≈ diag(materialize(prepare(M)))
        @test operator_diagonal(scaling(2.0) * scaling(κ)) isa Field
        @test flatten(operator_diagonal(scaling(2.0) * scaling(κ))) ≈ 2 .* flatten(κ)
        @test operator_diagonal(adjoint(scaling(2 - 3im))) == 2 + 3im

        # non-diagonal trees degrade to errors, never wrong diagonals
        @test_throws ArgumentError operator_diagonal(gradient(g))
        @test_throws ArgumentError operator_diagonal(laplacian(g) * laplacian(g))
    end
end
