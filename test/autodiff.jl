# Top-level loss functions: Enzyme needs typed arguments, not non-const globals.
ad_field_loss(xdata, w, L, g) = sum(w .* interior(apply(L, Field(xdata, g))))

function ad_kappa_loss(κdata, udata, w, g)
    K = divergence(g) * scaling(Field(κdata, g)) * MatrixFreeOperators.gradient(g)
    return sum(w .* interior(apply(K, Field(copy(udata), g))))
end

ad_selfadv_loss(udata, w, F, g) = sum(dot.(w, interior(apply(F, Field(udata, g)))))

# Forest reference path: flat interior vector in, weighted flat action out. The
# halo-exchange schedule must be pre-warmed so the loss only reads the cache.
function ad_forest_loss(v, w, L, bf)
    u = scalar_field(bf)
    flat_to_interior!(u, v)
    return sum(w .* flatten(apply(L, u)))
end

@testset "Automatic differentiation (Enzyme + Mooncake, no custom rules)" begin
    g = CartesianGrid(
        ((0.0, 1.0), (0.0, 1.0)), (5, 4);
        bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
    )
    rng = Random.MersenneTwister(61)
    w = rand(rng, local_size(g)...)
    κ0 = set!(scalar_field(g), x -> 1 + x[1] * x[2])

    @testset "field gradient of $(name)" for (name, L) in (
        ("Laplacian", laplacian(g)),
        ("Derivative", derivative(g, 1)),
        ("div∘κ∘grad", divergence(g) * scaling(κ0) * MatrixFreeOperators.gradient(g)),
    )
        x = rand(rng, padded_size(g)...)
        dx = zero(x)
        # set_runtime_activity: the Const coefficient field inside L flows into the
        # active output buffer, which static activity analysis cannot prove safe.
        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            ad_field_loss,
            Enzyme.Active,
            Enzyme.Duplicated(x, dx),
            Enzyme.Const(w),
            Enzyme.Const(L),
            Enzyme.Const(g),
        )
        fd = fd_gradient(xd -> ad_field_loss(xd, w, L, g), x)
        @test dx ≈ fd atol = 1e-5

        # adjoint = pullback for linear operators: ∂loss/∂x = Lᵀ·w̃
        w̃ = scalar_field(g)
        interior(w̃) .= w
        lt = apply(adjoint(L), w̃)
        @test dx[interior(g)] ≈ collect(interior(lt)) rtol = 1e-10
        ghost_mask = trues(padded_size(g))
        ghost_mask[interior(g)] .= false
        @test all(iszero, dx[ghost_mask])

        cache = Mooncake.prepare_gradient_cache(ad_field_loss, x, w, L, g)
        _, grads = Mooncake.value_and_gradient!!(cache, ad_field_loss, x, w, L, g)
        @test grads[2] ≈ dx rtol = 1e-10
    end

    @testset "parameter gradient w.r.t. coefficient field κ (Decision B)" begin
        u = set!(scalar_field(g), x -> sin(3 * x[1]) * x[2])
        κ = 1.0 .+ rand(rng, padded_size(g)...)
        dκ = zero(κ)
        Enzyme.autodiff(
            Enzyme.Reverse,
            ad_kappa_loss,
            Enzyme.Active,
            Enzyme.Duplicated(κ, dκ),
            Enzyme.Const(u.data),
            Enzyme.Const(w),
            Enzyme.Const(g),
        )
        fd = fd_gradient(κd -> ad_kappa_loss(κd, u.data, w, g), κ)
        @test any(!iszero, fd)
        @test dκ ≈ fd atol = 1e-5

        cache = Mooncake.prepare_gradient_cache(ad_kappa_loss, κ, u.data, w, g)
        _, grads = Mooncake.value_and_gradient!!(cache, ad_kappa_loss, κ, u.data, w, g)
        @test grads[2] ≈ dκ rtol = 1e-10
    end

    @testset "field gradient through the forest reference path (refined=$(refined))" for refined in (false, true)
        gf = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (8, 8);
            bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
        )
        bf = BlockForest(gf; blocksize=(4, 4), maxlevel=2)
        refined && refine!(bf, x -> x[1] < 0.5)   # coarse–fine interp/restrict in the tape
        MatrixFreeOperators._exchange_schedule(bf)
        L = laplacian(bf)
        n = length(flatten(scalar_field(bf)))
        v = rand(rng, n)
        wf = rand(rng, n)

        # Ground truth: the declared adjoint action (exact for a linear operator,
        # incl. the cross-block ghost fold and its coarse–fine transpose when
        # refined), sanity-checked against finite differences.
        w̃ = scalar_field(bf)
        flat_to_interior!(w̃, wf)
        lt = flatten(apply_adjoint!(scalar_field(bf), L, w̃, bf))
        fd = fd_gradient(vd -> ad_forest_loss(vd, wf, L, bf), v)
        @test fd ≈ lt atol = 1e-5

        # Enzyme hits EnzymeNoTypeError inside _run_fills! (the coarse–fine fill
        # sweep) on Julia 1.10 — refined forests only; see issue #26.
        if !refined || VERSION >= v"1.11"
            dv = zero(v)
            Enzyme.autodiff(
                Enzyme.set_runtime_activity(Enzyme.Reverse),
                ad_forest_loss,
                Enzyme.Active,
                Enzyme.Duplicated(v, dv),
                Enzyme.Const(wf),
                Enzyme.Const(L),
                Enzyme.Const(bf),
            )
            @test dv ≈ lt rtol = 1e-8
        else
            @test_skip "Enzyme refined-forest gradient — EnzymeNoTypeError on Julia 1.10"
        end

        cache = Mooncake.prepare_gradient_cache(ad_forest_loss, v, wf, L, bf)
        _, grads = Mooncake.value_and_gradient!!(cache, ad_forest_loss, v, wf, L, bf)
        @test grads[2] ≈ lt rtol = 1e-8
    end

    @testset "gradient through the nonlinear leaf u·∇u" begin
        g1 = CartesianGrid(((0.0, 2π),), (16,); bc=((Periodic(), Periodic()),))
        F = advection(g1, SelfAdvection())
        u = set!(vector_field(g1), x -> SVector(2 + sin(x[1])))
        wv = [SVector(rand(rng)) for _ in 1:16]

        du = zero(u.data)
        Enzyme.autodiff(
            Enzyme.Reverse,
            ad_selfadv_loss,
            Enzyme.Active,
            Enzyme.Duplicated(copy(u.data), du),
            Enzyme.Const(wv),
            Enzyme.Const(F),
            Enzyme.Const(g1),
        )

        # directional FD checks: ⟨∇loss, v⟩ ≈ d/dε loss(u + εv)
        for trial in 1:3
            v = [SVector(randn(rng)) for _ in 1:18]
            ε = 1e-6
            lp = ad_selfadv_loss(u.data .+ ε .* v, wv, F, g1)
            lm = ad_selfadv_loss(u.data .- ε .* v, wv, F, g1)
            @test sum(dot.(du, v)) ≈ (lp - lm) / (2 * ε) atol = 1e-5
        end
    end
end
