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

    @testset "prolongation action" begin
        # 1D Dirichlet: wall child = (3/4)u₁ + (1/4)(-u₁) = u₁/2; interior children
        # of a linear coarse profile are exact
        gf = CartesianGrid(((0.0, 1.0),), (8,))
        gc = coarsen(gf)
        P = prolongation(gc, gf)
        u = set!(scalar_field(gc), x -> 2 * x[1] + 1)
        y = P * u
        uc = collect(interior(u))
        yf = collect(interior(y))
        @test yf[1] ≈ uc[1] / 2
        @test yf[8] ≈ uc[4] / 2
        for f in 2:7
            xf = 0.0625 + (f - 1) * 0.125
            @test yf[f] ≈ 2 * xf + 1
        end

        # Neumann wall child mirrors: (3/4)u₁ + (1/4)u₁ = u₁; constants preserved
        gfn = CartesianGrid(((0.0, 1.0),), (8,); bc=((Neumann(), Neumann()),))
        un = set!(scalar_field(coarsen(gfn)), x -> 3.5 + 0 * x[1])
        @test all(collect(interior(prolongation(coarsen(gfn), gfn) * un)) .≈ 3.5)

        # P·1 = 1 under Periodic and Neumann, D ∈ 1:3
        for D in 1:3, bc in (Periodic(), Neumann())
            gD = CartesianGrid(
                ntuple(_ -> (0.0, 1.0), D),
                ntuple(_ -> 8, D);
                bc=ntuple(_ -> (bc, bc), D),
            )
            gcD = coarsen(gD)
            ones_c = set!(scalar_field(gcD), _ -> 1.0)
            @test all(collect(interior(prolongation(gcD, gD) * ones_c)) .≈ 1.0)
        end
    end

    @testset "restriction action" begin
        # 1D interior full weighting (1/8, 3/8, 3/8, 1/8) against a manual sum
        gf = CartesianGrid(((0.0, 1.0),), (8,))
        gc = coarsen(gf)
        R = restriction(gf, gc)
        u = scalar_field(gf)
        rand!(interior(u))
        uf = collect(interior(u))
        rc = collect(interior(R * u))
        for c in 2:3
            @test rc[c] ≈
                (uf[2c - 2] + 3 * uf[2c - 1] + 3 * uf[2c] + uf[2c + 1]) / 8
        end

        # R·1 = 1 under Periodic and Neumann, D ∈ 1:3
        for D in 1:3, bc in (Periodic(), Neumann())
            gD = CartesianGrid(
                ntuple(_ -> (0.0, 1.0), D),
                ntuple(_ -> 8, D);
                bc=ntuple(_ -> (bc, bc), D),
            )
            ones_f = set!(scalar_field(gD), _ -> 1.0)
            @test all(collect(interior(restriction(gD) * ones_f)) .≈ 1.0)
        end
    end

    @testset "transfer adjoint identity" begin
        bcs = (
            (Periodic(), Periodic()),
            (Dirichlet(), Dirichlet()),
            (Neumann(), Neumann()),
            (Dirichlet(), Neumann()),
        )
        for D in 1:2, bc in bcs
            gf = CartesianGrid(
                ntuple(_ -> (0.0, 1.0), D),
                ntuple(_ -> 8, D);
                bc=ntuple(_ -> bc, D),
            )
            gc = coarsen(gf)
            R = restriction(gf, gc)
            P = prolongation(gc, gf)
            Rd = materialize(prepare(R))
            Pd = materialize(prepare(P))
            # R = 2⁻ᴺ·Pᵀ exactly, and the declared adjoints match the dense transposes
            @test Rd ≈ 2.0^(-D) .* Pd'
            @test materialize(prepare(adjoint(R), scalar_field(gc))) ≈ Rd'
            @test materialize(prepare(adjoint(P), scalar_field(gf))) ≈ Pd'

            # random-field dot-product identity through the in-place kernels
            x = scalar_field(gc)
            y = scalar_field(gf)
            rand!(interior(x))
            rand!(interior(y))
            x̄ = scalar_field(gc)
            apply_adjoint!(x̄, P, y, gf)
            @test dot(flatten(P * x), flatten(y)) ≈ dot(flatten(x), flatten(x̄))
        end

        # mixed-BC 2D case
        gf = CartesianGrid(
            ((0.0, 1.0), (0.0, 2.0)),
            (8, 12);
            bc=((Dirichlet(), Neumann()), (Periodic(), Periodic())),
        )
        gc = coarsen(gf)
        @test materialize(prepare(restriction(gf, gc))) ≈
            0.25 .* materialize(prepare(prolongation(gc, gf)))'
    end

    @testset "transfer plumbing" begin
        gf = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8))
        gc = coarsen(gf)
        R = restriction(gf, gc)
        P = prolongation(gc, gf)
        @test size(R) == (16, 64)
        @test size(P) == (64, 16)
        @test islinear(R) && isconstant(R) && islinear(P) && isconstant(P)

        # composed rectangular chain: prepared RAP matches the dense triple product
        A = laplacian(gf)
        RAP = prepare(R * A * P, scalar_field(gc))
        @test size(RAP) == (16, 16)
        G = materialize(RAP)
        @test G ≈
            materialize(prepare(R)) * materialize(prepare(A)) * materialize(prepare(P))

        # affine lifts: coarse Dirichlet(a) puts a/2 on wall-child rows; R has none
        a = 3.0
        gfa = CartesianGrid(((0.0, 1.0),), (8,); bc=((Dirichlet(a), Dirichlet(a)),))
        gca = coarsen(gfa)
        bP = boundary_rhs(prolongation(gca, gfa), scalar_field(gca))
        @test collect(interior(bP))[1] ≈ a / 2
        @test collect(interior(bP))[8] ≈ a / 2
        @test all(iszero, collect(interior(bP))[2:7])
        @test all(iszero, boundary_rhs(restriction(gfa, gca), scalar_field(gfa)).data)

        # validation errors
        godd = CartesianGrid(((0.0, 1.0),), (6,))
        @test_throws ArgumentError restriction(gf, godd)
        gext = CartesianGrid(((0.0, 2.0), (0.0, 1.0)), (4, 4))
        @test_throws ArgumentError prolongation(gext, gf)
        gbc = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (4, 4); bc=ntuple(_ -> (Neumann(), Neumann()), 2)
        )
        @test_throws ArgumentError prolongation(gbc, gf)
    end

    @testset "Galerkin coarse operator check" begin
        # RAP is a *different* consistent coarse operator than rediscretization for
        # cell-centered transfers — never assert equality, assert structure + O(h²)
        # agreement of actions on smooth fields.
        function galerkin_gap(n)
            gf = CartesianGrid(
                ((0.0, 1.0),), (2n,); bc=((Periodic(), Periodic()),)
            )
            gc = coarsen(gf)
            G = materialize(
                prepare(
                    restriction(gf, gc) * laplacian(gf) * prolongation(gc, gf),
                    scalar_field(gc),
                ),
            )
            Ac = materialize(prepare(laplacian(gc)))
            u = [sinpi(2 * (i - 0.5) / n) for i in 1:n]
            return G, Ac, maximum(abs, (G - Ac) * u)
        end
        G8, Ac8, gap8 = galerkin_gap(8)
        G16, _, gap16 = galerkin_gap(16)
        @test G8 ≈ G8'
        @test maximum(abs, G8 * ones(8)) < 1e-12
        @test maximum(abs, Ac8 * ones(8)) < 1e-12
        @test gap8 / gap16 > 3.0

        # Dirichlet: Galerkin preserves negative definiteness of Δ
        gfd = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8))
        gcd = coarsen(gfd)
        Gd = materialize(
            prepare(
                restriction(gfd, gcd) * laplacian(gfd) * prolongation(gcd, gfd),
                scalar_field(gcd),
            ),
        )
        @test eigmax(Symmetric(Gd)) < 0
    end
end
