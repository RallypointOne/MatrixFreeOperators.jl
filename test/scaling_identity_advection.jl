@testset "ScalingOp, IdentityOp, Advection" begin
    @testset "scaling by a Number" begin
        g = CartesianGrid(((0.0, 1.0),), (6,))
        u = set!(x -> x[1], scalar_field(g))
        S = scaling(2.5)
        @test collect(interior(S * u)) ≈ 2.5 .* collect(interior(u))
        @test islinear(S) && isconstant(S) && isdiagonal(S) && isselfadjoint(S)
        @test adjoint(S) === S
        @test MatrixFreeOperators.operator_grid(S) === nothing
    end

    @testset "scaling by a coefficient field" begin
        g = CartesianGrid(((0.0, 1.0),), (6,))
        κ = set!(x -> 1 + x[1]^2, scalar_field(g))
        u = set!(x -> sin(x[1]), scalar_field(g))
        S = scaling(κ)
        @test collect(interior(S * u)) ≈ collect(interior(κ)) .* collect(interior(u))
        @test isselfadjoint(S) && adjoint(S) === S
        @test MatrixFreeOperators.operator_grid(S) === g

        v = set!(x -> SVector(x[1]), vector_field(g))
        Sv = S * v
        @test getindex.(collect(interior(Sv)), 1) ≈
            collect(interior(κ)) .* getindex.(collect(interior(v)), 1)

        @test_throws ArgumentError scaling(vector_field(g))
    end

    @testset "identity_op" begin
        g = CartesianGrid(((0.0, 1.0),), (6,))
        u = set!(x -> x[1]^3, scalar_field(g))
        I = identity_op()
        @test collect(interior(I * u)) == collect(interior(u))
        @test islinear(I) && isselfadjoint(I) && isdiagonal(I)
        @test adjoint(I) === I
        z = set!(x -> 1.0, scalar_field(g))
        MatrixFreeOperators.apply!(z, I, u, g, 2.0, -1.0)
        @test collect(interior(z)) ≈ 2 .* collect(interior(u)) .- 1
    end

    @testset "prescribed advection: analytic action and convergence" begin
        function adv_error(n)
            g = CartesianGrid(
                ((0.0, 2π), (0.0, 2π)), (n, n);
                bc=((Periodic(), Periodic()), (Periodic(), Periodic())),
            )
            v = set!(x -> SVector(sin(x[2]), cos(x[1])), vector_field(g))
            u = set!(x -> sin(x[1]) * sin(x[2]), scalar_field(g))
            A = advection(g, v)
            @test islinear(A) && isconstant(A)
            y = A * u
            ref = set!(
                x ->
                    sin(x[2]) * cos(x[1]) * sin(x[2]) + cos(x[1]) * sin(x[1]) * cos(x[2]),
                scalar_field(g),
            )
            return maximum(abs, collect(interior(y)) .- collect(interior(ref)))
        end
        e32 = adv_error(32)
        e64 = adv_error(64)
        @test log2(e32 / e64) ≥ 1.9
    end

    @testset "advection validation" begin
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (4, 4))
        @test_throws ArgumentError advection(g, scalar_field(g))
        g1 = CartesianGrid(((0.0, 1.0),), (4,))
        @test_throws ArgumentError advection(g, vector_field(g1))
    end

    @testset "self-advection is nonlinear and computes u·∇u" begin
        g = CartesianGrid(((0.0, 2π),), (64,); bc=((Periodic(), Periodic()),))
        A = advection(g, SelfAdvection())
        @test !islinear(A)
        @test_throws ArgumentError adjoint(A)

        u = set!(x -> SVector(sin(x[1])), vector_field(g))
        y = A * u
        ref = set!(x -> SVector(sin(x[1]) * cos(x[1])), vector_field(g))
        @test maximum(norm.(collect(interior(y)) .- collect(interior(ref)))) < 0.01

        @test_throws ArgumentError apply(A, scalar_field(g))
    end
end

@testset "ScalingOp (function-of-fields scaling)" begin
    g = CartesianGrid(
        ((0.0, 1.0), (0.0, 1.0)), (5, 4);
        bc=((Dirichlet(1.0), Dirichlet(2.0)), (Neumann(), Neumann())),
    )
    κ = set!(x -> 1 + x[1]^2, scalar_field(g))
    σ = set!(x -> 2 + x[2], scalar_field(g))
    u = set!(x -> sin(x[1] + 0.3) * cos(x[2]), scalar_field(g))
    f(a, b) = exp(a) * b
    S = scaling(f, κ, σ)
    coeff = f.(collect(interior(κ)), collect(interior(σ)))

    @testset "construction and traits" begin
        @test S isa ScalingOp && !(S isa CoeffScaling)
        @test scaling(2.0) isa CoeffScaling && scaling(κ) isa CoeffScaling
        @test islinear(S) && isconstant(S) && isdiagonal(S) && isselfadjoint(S)
        @test shares_exchange(S)
        @test adjoint(S) === S
        @test MatrixFreeOperators.operator_grid(S) === g
        @test MatrixFreeOperators.operator_grid(scaling(*, 2.0, κ)) === g
        @test MatrixFreeOperators.operator_grid(scaling(*, 2.0, 3.0)) === nothing
        @test size(S) == (20, 20)
    end

    @testset "argument checks" begin
        v = set!(x -> SVector(x[1], 1 + x[2]), vector_field(g))
        g2 = CartesianGrid(((0.0, 5.0), (0.0, 1.0)), (5, 4))
        @test_throws ArgumentError scaling(() -> 2.0)                  # no arguments
        @test_throws ArgumentError scaling([1.0, 2.0])                 # not a function of fields
        @test_throws ArgumentError scaling(*, κ, [1.0, 2.0])           # array argument
        @test_throws ArgumentError scaling(*, κ, scalar_field(g2))     # mismatched grids
        # still accepted: Number arguments, vector-eltype arguments
        @test scaling(*, 2.0, 3.0) isa ScalingOp
        @test scaling(norm, v) isa ScalingOp
    end

    @testset "action" begin
        @test collect(interior(S * u)) ≈ coeff .* collect(interior(u))
        # Number arguments broadcast as scalars
        @test collect(interior(scaling(*, 3.0, κ) * u)) ≈ 3 .* collect(interior(κ)) .* collect(interior(u))
        # the accumulating form
        z = set!(x -> 1.0, scalar_field(g))
        MatrixFreeOperators.apply!(z, S, u, g, 2.0, -1.0)
        @test collect(interior(z)) ≈ 2 .* coeff .* collect(interior(u)) .- 1
        # vector-valued x scales componentwise
        v = set!(x -> SVector(x[1], 1 + x[2]), vector_field(g))
        Sv = S * v
        @test getindex.(collect(interior(Sv)), 2) ≈ coeff .* getindex.(collect(interior(v)), 2)
        # vector-eltype arguments reach f whole
        @test collect(interior(scaling(norm, v) * u)) ≈ norm.(collect(interior(v))) .* collect(interior(u))
        # agrees with the materialized CoeffScaling
        c = scalar_field(g)
        interior(c) .= coeff
        @test collect(interior(S * u)) ≈ collect(interior(scaling(c) * u))
    end

    @testset "adjoint identity and structure" begin
        rng = Random.MersenneTwister(2718)
        x = scalar_field(g)
        y = scalar_field(g)
        interior(x) .= rand(rng, 5, 4)
        interior(y) .= rand(rng, 5, 4)
        @test dot(collect(interior(S * x)), collect(interior(y))) ≈
            dot(collect(interior(x)), collect(interior(adjoint(S) * y)))
        A = materialize(prepare(S, scalar_field(g)))
        @test A ≈ Diagonal(vec(coeff))
    end

    @testset "composition" begin
        c = scalar_field(g)
        interior(c) .= coeff
        L = laplacian(g) - S
        Lref = laplacian(g) - scaling(c)
        @test collect(interior(L * u)) ≈ collect(interior(Lref * u))
        @test materialize(prepare(L, scalar_field(g))) ≈ materialize(prepare(Lref, scalar_field(g)))
        # product of two function scalings is the scaling by the product
        @test collect(interior(scaling(exp, κ) * (scaling(identity, σ) * u))) ≈ collect(interior(S * u))
        @test collect(interior((2.0 * S) * u)) ≈ 2 .* coeff .* collect(interior(u))
    end

    @testset "boundary_rhs" begin
        # pointwise on the interior: inhomogeneous boundary data never reaches it
        @test all(iszero, interior(boundary_rhs(S, g)))
        @test collect(interior(boundary_rhs(laplacian(g) - S, g))) ≈
            collect(interior(boundary_rhs(laplacian(g), g)))
    end

    @testset "operator_diagonal" begin
        d = operator_diagonal(S)
        @test d isa Field && collect(interior(d)) ≈ coeff
        @test operator_diagonal(scaling(*, 2.0, 3.0)) == 6.0
        c = scalar_field(g)
        interior(c) .= coeff
        @test collect(interior(operator_diagonal(laplacian(g) - S))) ≈
            collect(interior(operator_diagonal(laplacian(g) - scaling(c))))
    end

    @testset "forest" begin
        gf = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8))
        bf = BlockForest(gf; blocksize=(4, 4), maxlevel=2)
        κb = set!(x -> 1 + x[1]^2, scalar_field(bf))
        σb = set!(x -> 2 + x[2], scalar_field(bf))
        cb = set!(x -> f(1 + x[1]^2, 2 + x[2]), scalar_field(bf))
        ub = set!(x -> sin(x[1] + 0.3) * cos(x[2]), scalar_field(bf))
        Sb = scaling(f, κb, σb)
        @test MatrixFreeOperators.operator_grid(Sb) === bf
        @test max_interior_diff(Sb * ub, scaling(cb) * ub) < 1e-12
        @test max_interior_diff((laplacian(bf) - Sb) * ub, (laplacian(bf) - scaling(cb)) * ub) < 1e-12
        @test_throws ArgumentError operator_diagonal(Sb)
    end

    # Enzyme is not covered yet: locally (Julia 1.13) `DI.AutoEnzyme` fails with an
    # EnzymeInternalError for a general ScalingOp, but identically for CoeffScaling through
    # the same loss, so the failure is not specific to this operator. Unverified on
    # the CI Julia versions; add an Enzyme check here once that is sorted out.
    @testset "AD gradients (Mooncake) vs finite differences" begin
        DI = DifferentiationInterface
        backend = DI.AutoMooncake(; config=nothing)
        w = rand(Random.MersenneTwister(31), 5, 4)
        # operator parameter: a field argument of f
        loss_κ(κd) = sum(w .* interior(scaling(f, Field(κd, g), σ) * u))
        κd = copy(κ.data)
        @test DI.gradient(loss_κ, backend, κd) ≈ fd_gradient(loss_κ, κd) rtol = 1e-6
        # the field the operator is applied to
        loss_x(xd) = sum(w .* interior(S * Field(xd, g)))
        xd = copy(u.data)
        @test DI.gradient(loss_x, backend, xd) ≈ fd_gradient(loss_x, xd) rtol = 1e-6
    end
end
