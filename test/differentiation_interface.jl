# DifferentiationInterface is the frontend the docs and examples use: one backend
# object instead of Enzyme's annotation vocabulary. It is deliberately NOT a
# dependency of the core package — custom rules cannot be routed through it
# (DESIGN.md §6) — so this file exists to keep the *documented* pattern honest.
# If it breaks, the docs are wrong.

const DI = DifferentiationInterface

# Extra arguments ride along as DI `Constant` contexts, which is what
# `Enzyme.Const` becomes at this layer.
di_field_loss(xdata, w, L, g) = sum(w .* interior(apply(L, Field(xdata, g))))

function di_kappa_loss(κdata, udata, w, g)
    K = divergence(g) * scaling(Field(κdata, g)) * MatrixFreeOperators.gradient(g)
    return sum(w .* interior(apply(K, Field(copy(udata), g))))
end

@testset "DifferentiationInterface frontend (AutoEnzyme)" begin
    # Runtime activity: the Const coefficient field inside L flows into the active
    # output buffer, which static activity analysis cannot prove safe. This is the
    # same annotation the raw-Enzyme path needs, expressed once on the backend.
    backend = DI.AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))

    g = CartesianGrid(
        ((0.0, 1.0), (0.0, 1.0)), (5, 4);
        bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
    )
    rng = Random.MersenneTwister(1907)
    w = rand(rng, local_size(g)...)
    κ0 = set!(scalar_field(g), x -> 1 + x[1] * x[2])

    @testset "field gradient matches the declared adjoint: $(name)" for (name, L) in (
        ("Laplacian", laplacian(g)),
        ("div∘κ∘grad", divergence(g) * scaling(κ0) * MatrixFreeOperators.gradient(g)),
    )
        x = rand(rng, padded_size(g)...)
        dx = DI.gradient(
            di_field_loss, backend, x, DI.Constant(w), DI.Constant(L), DI.Constant(g)
        )

        # Same oracle as the raw-Enzyme suite: for a linear operator the pullback
        # is the declared adjoint, to machine precision.
        w̃ = scalar_field(g)
        interior(w̃) .= w
        lt = apply(adjoint(L), w̃)
        @test dx[interior(g)] ≈ collect(interior(lt)) rtol = 1e-10

        ghost_mask = trues(padded_size(g))
        ghost_mask[interior(g)] .= false
        @test all(iszero, dx[ghost_mask])

        # The prepared form is what a real optimization loop calls repeatedly.
        prep = DI.prepare_gradient(
            di_field_loss, backend, x, DI.Constant(w), DI.Constant(L), DI.Constant(g)
        )
        val, dx2 = DI.value_and_gradient(
            di_field_loss, prep, backend, x, DI.Constant(w), DI.Constant(L), DI.Constant(g)
        )
        @test val ≈ di_field_loss(x, w, L, g)
        @test dx2 ≈ dx
    end

    @testset "parameter gradient w.r.t. the coefficient field κ" begin
        u = set!(scalar_field(g), x -> sin(3 * x[1]) * x[2])
        κ = 1.0 .+ rand(rng, padded_size(g)...)
        dκ = DI.gradient(
            di_kappa_loss,
            backend,
            κ,
            DI.Constant(u.data),
            DI.Constant(w),
            DI.Constant(g),
        )
        fd = fd_gradient(κd -> di_kappa_loss(κd, u.data, w, g), κ)
        @test any(!iszero, fd)
        @test dκ ≈ fd atol = 1e-5
    end

    @testset "forest gradient through the DI frontend" begin
        gf = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (8, 8);
            bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
        )
        bf = BlockForest(gf; blocksize=(4, 4), maxlevel=2)
        refine!(bf, x -> x[1] < 0.5)
        MatrixFreeOperators._exchange_schedule(bf)
        L = laplacian(bf)
        n = length(flatten(scalar_field(bf)))
        v = rand(rng, n)
        wf = rand(rng, n)

        function di_forest_loss(vv, ww, LL, bff)
            u = scalar_field(bff)
            flat_to_interior!(u, vv)
            return sum(ww .* flatten(apply(LL, u)))
        end

        dv = DI.gradient(
            di_forest_loss, backend, v, DI.Constant(wf), DI.Constant(L), DI.Constant(bf)
        )
        w̃ = scalar_field(bf)
        flat_to_interior!(w̃, wf)
        lt = flatten(apply_adjoint!(scalar_field(bf), L, w̃, bf))
        @test dv ≈ lt rtol = 1e-8
    end
end
