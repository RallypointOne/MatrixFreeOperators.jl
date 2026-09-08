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

    @testset "prepared composed tree ≡ pure path (div ∘ κ ∘ grad)" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (8, 6);
            bc=((Dirichlet(), Dirichlet()), (Periodic(), Periodic())),
        )
        κ = set!(scalar_field(g), x -> 1 + x[1])
        K = divergence(g) * scaling(κ) * MatrixFreeOperators.gradient(g)
        P = prepare(K, scalar_field(g))
        @test P.op isa MatrixFreeOperators.PreparedComposed
        @test eltype(P.op.tmp) === SVector{2,Float64}    # rank-changing internal buffer

        rng = Random.MersenneTwister(17)
        x = rand(rng, 48)
        y = similar(x)
        mul!(y, P, x)
        xf = scalar_field(g)
        flat_to_interior!(xf, x)
        @test y ≈ flatten(apply(K, xf))

        @inferred MatrixFreeOperators.apply!(P.ypad, P.op, P.xpad, P.grid, true, false)
        mul!(y, P, x)
        alloc = @allocated mul!(y, P, x)
        @test alloc ≤ 512
    end

    @testset "prepared adjoint tree (PreparedAdjoint coverage)" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (5, 4);
            bc=((Dirichlet(), Neumann()), (Periodic(), Periodic())),
        )
        v = set!(vector_field(g), x -> SVector(1 + x[1], x[2]))
        Adv = advection(g, v)
        P = prepare(Adv, scalar_field(g))
        Pt = prepare(adjoint(Adv), scalar_field(g))
        A = materialize(P)
        At = materialize(Pt)
        @test At ≈ A'

        rng = Random.MersenneTwister(23)
        x = rand(rng, 20)
        y0 = rand(rng, 20)
        y = copy(y0)
        mul!(y, Pt, x, 1.5, 2.0)
        @test y ≈ 1.5 .* (A' * x) .+ 2.0 .* y0

        mul!(y, Pt, x)
        alloc = @allocated mul!(y, Pt, x)
        @test alloc ≤ 512
    end

    # A user-written AdjointOp over a composition has no prepared twin of its own —
    # PreparedComposed carries one intermediate shaped for the forward pass — so
    # prepare must normalize it down to the leaves (Composed(bᵀ, aᵀ)) before the
    # walk. The rank-changing case is the sharp one: a buffer allocated from the
    # adjoint's input prototype (a scalar) has the wrong element type for divᵀ's
    # vector output.
    @testset "prepare normalizes AdjointOp over a composite" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (5, 4);
            bc=((Dirichlet(), Neumann()), (Periodic(), Periodic())),
        )
        inner = derivative(g, 1) * derivative(g, 2)
        P = prepare(MatrixFreeOperators.AdjointOp(inner))
        @test !(P.op isa MatrixFreeOperators.PreparedAdjoint)   # rewritten, not wrapped
        @test P.op isa MatrixFreeOperators.PreparedComposed
        @test materialize(P) ≈ materialize(prepare(inner))'

        κ = set!(scalar_field(g), x -> 1 + x[1] * x[2])
        K = divergence(g) * scaling(κ)                           # vector → scalar
        Kt = prepare(MatrixFreeOperators.AdjointOp(K), scalar_field(g))
        B = materialize(prepare(K, vector_field(g)))
        @test size(Kt) == (40, 20)
        @test materialize(Kt) ≈ B'

        rng = Random.MersenneTwister(5)
        x = rand(rng, 20)
        y0 = rand(rng, 40)
        y = copy(y0)
        mul!(y, Kt, x, 1.5, 2.0)                                 # β ≠ 0 through the rewritten tree
        @test y ≈ 1.5 .* (B' * x) .+ 2.0 .* y0
        mul!(y, Kt, x)
        @test (@allocated mul!(y, Kt, x)) ≤ 512
    end

    # The composed wide-stencil div∘κ∘grad is deliberately NOT used as a cg system:
    # on collocated grids it is neither symmetric nor definite (odd-even
    # decoupling — see the design doc §8.3 caveat). The SPD variable-coefficient
    # system below is built from self-adjoint pieces instead.
    @testset "Krylov cg: variable-coefficient reaction-diffusion -Δu + σu = f" begin
        function helmholtz_error(n)
            g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (n, n))
            σf(x) = 1 + x[1] * x[2]
            ustar(x) = sin(π * x[1]) * sin(π * x[2])
            f(x) = 2 * π^2 * ustar(x) + σf(x) * ustar(x)
            σ = set!(scalar_field(g), σf)
            K = scaling(σ) - laplacian(g)
            @test isselfadjoint(K)
            P = prepare(K, scalar_field(g))
            b = flatten(set!(scalar_field(g), f))
            u, stats = Krylov.cg(P, b)
            @test stats.solved
            return maximum(abs, u .- flatten(set!(scalar_field(g), ustar)))
        end
        e16 = helmholtz_error(16)
        e32 = helmholtz_error(32)
        @test e16 < 0.05
        @test e16 / e32 > 3
    end

    @testset "inhomogeneous Dirichlet via boundary_rhs lift" begin
        a, c = 0.7, -0.3
        g = CartesianGrid(((0.0, 1.0),), (64,); bc=((Dirichlet(a), Dirichlet(c)),))
        L = laplacian(g)

        zero_in = scalar_field(g)
        @test all(iszero, collect(interior(apply(L, zero_in))))      # islinear ⇒ L(0) = 0

        b = boundary_rhs(L, g)
        f = set!(scalar_field(g), x -> π^2 * sin(π * x[1]))
        rhs = -flatten(f) .- flatten(b)                               # Δu = -f  ⇒  A·u = -f - b
        P = prepare(L)
        u, stats = Krylov.minres(P, rhs)
        @test stats.solved
        ustar = flatten(
            set!(scalar_field(g), x -> sin(π * x[1]) + (1 - x[1]) * a + x[1] * c)
        )
        @test maximum(abs, u .- ustar) < 0.01
    end

    @testset "boundary_rhs through combinators" begin
        g = CartesianGrid(((0.0, 1.0),), (8,); bc=((Dirichlet(2.0), Neumann(1.0)),))
        L = laplacian(g)
        S = scaling(set!(scalar_field(g), x -> 1 + x[1]))
        bL = collect(interior(boundary_rhs(L, g)))
        @test collect(interior(boundary_rhs(3 * L, g))) ≈ 3 .* bL
        @test collect(interior(boundary_rhs(L + L, g))) ≈ 2 .* bL
        @test collect(interior(boundary_rhs(S * L, g))) ≈
            collect(interior(S.coeff)) .* bL                          # S has zero lift
        @test_throws ArgumentError boundary_rhs(advection(g, SelfAdvection()), g)
    end
end
