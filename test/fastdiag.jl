@testset "Fast-diagonalization direct solver" begin
    rng = Random.MersenneTwister(2718)

    # Dense (αI + βΔ) on the same discrete operator the solver inverts.
    function dense_op(g, α, β)
        A = β .* materialize(prepare(laplacian(g), scalar_field(g)))
        α == 0 || (A += α * I)
        return A
    end

    @testset "inverse identity vs materialized operator" begin
        grids = [
            CartesianGrid(((0.0, 1.0),), (16,)),
            CartesianGrid(((0.0, 1.0),), (12,); bc=((Neumann(), Dirichlet()),)),
            CartesianGrid(
                ((0.0, 1.0), (0.0, 2.0)), (8, 6);
                bc=((Periodic(), Periodic()), (Dirichlet(), Dirichlet())),
            ),
            CartesianGrid(
                ((0.0, 1.0), (0.0, 1.0)), (8, 7);
                bc=((Dirichlet(), Neumann()), (Neumann(), Dirichlet())),
            ),
            CartesianGrid(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), (6, 5, 4)),
        ]
        # (0,1): Poisson (every grid above has a Dirichlet face ⇒ nonsingular).
        # (1,-1), (3,-0.5): shifted SPD (αI − |β|Δ ⪰ αI ≻ 0), nonsingular for any BC.
        for g in grids, (α, β) in ((0.0, 1.0), (1.0, -1.0), (3.0, -0.5))
            n = prod(local_size(g))
            S = fast_diag_solver(g; α=α, β=β)
            A = dense_op(g, α, β)
            f = rand(rng, n)
            u = S \ f
            @test isapprox(A * u, f; rtol=1e-6, atol=1e-9)          # forward residual
            x = rand(rng, n)
            @test isapprox(S \ (A * x), x; rtol=1e-6, atol=1e-9)    # round trip
        end
    end

    @testset "Kronecker-sum structure (2D)" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.5)), (6, 5);
            bc=((Dirichlet(), Neumann()), (Periodic(), Periodic())),
        )
        Dx = MatrixFreeOperators._assemble_1d_laplacian(g, 1, Float64)
        Dy = MatrixFreeOperators._assemble_1d_laplacian(g, 2, Float64)
        nx, ny = local_size(g)
        Ix = Matrix{Float64}(I, nx, nx)
        Iy = Matrix{Float64}(I, ny, ny)
        A = materialize(prepare(laplacian(g), scalar_field(g)))   # acts on x-fastest vec
        @test kron(Iy, Dx) + kron(Dy, Ix) ≈ A
    end

    @testset "analytic Poisson convergence (Dirichlet)" begin
        function err(n)
            g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (n, n))
            S = fast_poisson(g)
            ustar(x) = sin(π * x[1]) * sin(π * x[2])
            f = set!(scalar_field(g), x -> -2π^2 * ustar(x))      # Δu = f
            u = S \ flatten(f)
            return maximum(abs, u .- flatten(set!(scalar_field(g), ustar)))
        end
        e16, e32 = err(16), err(32)
        @test e16 < 0.02
        @test e16 / e32 > 3                                       # ~second order
    end

    @testset "inhomogeneous Dirichlet via boundary_rhs lift" begin
        a, c = 0.7, -0.3
        g = CartesianGrid(((0.0, 1.0),), (64,); bc=((Dirichlet(a), Dirichlet(c)),))
        L = laplacian(g)
        S = fast_poisson(g)
        f = set!(scalar_field(g), x -> π^2 * sin(π * x[1]))
        rhs = -flatten(f) .- flatten(boundary_rhs(L, g))          # Δu = -f
        u = S \ rhs
        ustar = flatten(
            set!(scalar_field(g), x -> sin(π * x[1]) + (1 - x[1]) * a + x[1] * c)
        )
        @test maximum(abs, u .- ustar) < 0.01
    end

    @testset "singular pure-Neumann Poisson (compatible RHS)" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (12, 12);
            bc=((Neumann(), Neumann()), (Neumann(), Neumann())),
        )
        S = fast_poisson(g)
        A = materialize(prepare(laplacian(g), scalar_field(g)))
        f = rand(rng, prod(local_size(g)))
        f .-= sum(f) / length(f)                                  # compatible: zero mean
        u = S \ f
        @test abs(sum(u)) < 1e-9                                  # min-norm ⇒ zero-mean solution
        @test maximum(abs, A * u .- f) < 1e-8                     # solves the compatible system
    end

    @testset "preconditioner for variable-coefficient cg" begin
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (32, 32))
        σ = set!(scalar_field(g), x -> 1 + 9 * x[1] * x[2])       # mean ≈ 3.25
        K = scaling(σ) - laplacian(g)                             # SPD: σu − Δu
        P = prepare(K, scalar_field(g))
        b = rand(rng, prod(local_size(g)))
        S = fast_diag_solver(g; α=3.25, β=-1.0)                   # ≈ (σ̄I − Δ)⁻¹
        u_pre, st_pre = Krylov.cg(P, b; M=S)
        u_raw, st_raw = Krylov.cg(P, b)
        @test st_pre.solved && st_raw.solved
        @test st_pre.niter < st_raw.niter                        # preconditioning helps
        @test isapprox(u_pre, u_raw; rtol=1e-5)                  # same system, same answer
    end

    @testset "implicit diffusion step ≡ Krylov on the same operator" begin
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (24, 24))
        ν, dt = 1.0, 0.05
        u0 = flatten(set!(scalar_field(g), x -> sin(π * x[1]) * sin(π * x[2])))
        S = fast_diag_solver(g; α=1.0, β=-ν * dt)                 # (I − νΔt·Δ)
        u_fd = S \ u0
        Limp = identity_op() - (ν * dt) * laplacian(g)
        u_kry, st = Krylov.cg(prepare(Limp, scalar_field(g)), u0)
        @test st.solved
        @test maximum(abs, u_fd .- u_kry) < 1e-6
    end
end
