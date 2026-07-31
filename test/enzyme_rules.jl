# A rule that silently stops dispatching is the dangerous failure mode: every
# numerical assertion still passes, via the tape, and the only symptom is that the
# thing the rules exist to avoid — taping the coarse–fine descriptor sweep and the
# exchange kernels — quietly comes back. So each testset checks *both* that the
# gradient is right and that the rule fired.

const ENZ_EXT = Base.get_extension(MatrixFreeOperators, :MatrixFreeOperatorsEnzymeCoreExt)

er_field_loss(xdata, w, L, g) = sum(w .* interior(apply(L, Field(xdata, g))))

function er_forest_loss(v, w, L, bf)
    u = scalar_field(bf)
    flat_to_interior!(u, v)
    return sum(w .* flatten(apply(L, u)))
end

function er_kappa_loss(κdata, udata, w, g)
    K = divergence(g) * scaling(Field(κdata, g)) * MatrixFreeOperators.gradient(g)
    return sum(w .* interior(apply(K, Field(copy(udata), g))))
end

@testset "Enzyme custom rules (declared transposes, not the tape)" begin
    @test ENZ_EXT !== nothing

    rng = Random.MersenneTwister(2604)

    @testset "forest sweeps run under rules (refined=$(refined))" for refined in
                                                                     (false, true)
        gf = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (8, 8);
            bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
        )
        bf = BlockForest(gf; blocksize=(4, 4), maxlevel=2)
        refined && refine!(bf, x -> x[1] < 0.5)
        MatrixFreeOperators._exchange_schedule(bf)
        L = laplacian(bf)
        n = length(flatten(scalar_field(bf)))
        v = rand(rng, n)
        wf = rand(rng, n)

        # Ground truth: the declared adjoint action, exact for a linear operator
        # including the coarse–fine transpose when refined.
        w̃ = scalar_field(bf)
        flat_to_interior!(w̃, wf)
        lt = flatten(apply_adjoint!(scalar_field(bf), L, w̃, bf))

        dv = zero(v)
        before = ENZ_EXT.rule_hits()
        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            er_forest_loss,
            Enzyme.Active,
            Enzyme.Duplicated(v, dv),
            Enzyme.Const(wf),
            Enzyme.Const(L),
            Enzyme.Const(bf),
        )
        after = ENZ_EXT.rule_hits()

        # The whole point: `_exchange_storage!` ran under a rule, so the GhostFill
        # descriptor sweep never reached Enzyme's type analysis (issue #26).
        @test after.exchange > before.exchange
        @test after.bc > before.bc
        @test dv ≈ lt rtol = 1e-8

        fd = fd_gradient(vd -> er_forest_loss(vd, wf, L, bf), v)
        @test dv ≈ fd atol = 1e-5
    end

    @testset "batch width 2 exercises the multi-shadow branch" begin
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

        w̃ = scalar_field(bf)
        flat_to_interior!(w̃, wf)
        lt = flatten(apply_adjoint!(scalar_field(bf), L, w̃, bf))

        dv1 = zero(v)
        dv2 = zero(v)
        before = ENZ_EXT.rule_hits()
        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            er_forest_loss,
            Enzyme.BatchDuplicated(v, (dv1, dv2)),
            Enzyme.Const(wf),
            Enzyme.Const(L),
            Enzyme.Const(bf),
        )
        @test ENZ_EXT.rule_hits().exchange > before.exchange
        # Both shadows are seeded with the same cotangent, so both must reproduce
        # the width-1 gradient.
        @test dv1 ≈ lt rtol = 1e-8
        @test dv2 ≈ dv1
    end

    @testset "adjoint direction stays on rules ((Hᵀ)ᵀ = H)" begin
        gf = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (8, 8);
            bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
        )
        bf = BlockForest(gf; blocksize=(4, 4), maxlevel=2)
        refine!(bf, x -> x[1] < 0.5)
        MatrixFreeOperators._exchange_schedule(bf)
        # Applying this runs the transpose sweeps in the primal, so the reverse pass
        # must come back through the forward sweeps.
        L = adjoint(laplacian(bf))
        n = length(flatten(scalar_field(bf)))
        v = rand(rng, n)
        wf = rand(rng, n)

        dv = zero(v)
        before = ENZ_EXT.rule_hits()
        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            er_forest_loss,
            Enzyme.Active,
            Enzyme.Duplicated(v, dv),
            Enzyme.Const(wf),
            Enzyme.Const(L),
            Enzyme.Const(bf),
        )
        after = ENZ_EXT.rule_hits()
        @test after.exchange_adjoint > before.exchange_adjoint
        fd = fd_gradient(vd -> er_forest_loss(vd, wf, L, bf), v)
        @test dv ≈ fd atol = 1e-5
    end

    @testset "single-grid gradients are unchanged (no rule on that path)" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (5, 4);
            bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
        )
        w = rand(rng, local_size(g)...)
        κ0 = set!(scalar_field(g), x -> 1 + x[1] * x[2])

        @testset "$(name)" for (name, L) in (
            ("Laplacian", laplacian(g)),
            ("div∘κ∘grad", divergence(g) * scaling(κ0) * MatrixFreeOperators.gradient(g)),
        )
            x = rand(rng, padded_size(g)...)
            dx = zero(x)
            before = ENZ_EXT.rule_hits()
            Enzyme.autodiff(
                Enzyme.set_runtime_activity(Enzyme.Reverse),
                er_field_loss,
                Enzyme.Active,
                Enzyme.Duplicated(x, dx),
                Enzyme.Const(w),
                Enzyme.Const(L),
                Enzyme.Const(g),
            )
            # No forest storage is involved, so no rule may fire here.
            @test ENZ_EXT.rule_hits() == before

            w̃ = scalar_field(g)
            interior(w̃) .= w
            lt = apply(adjoint(L), w̃)
            @test dx[interior(g)] ≈ collect(interior(lt)) rtol = 1e-10
            ghost_mask = trues(padded_size(g))
            ghost_mask[interior(g)] .= false
            @test all(iszero, dx[ghost_mask])
        end
    end

    @testset "parameter gradients still tape (rules must not swallow them)" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (5, 4);
            bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
        )
        w = rand(rng, local_size(g)...)
        u = set!(scalar_field(g), x -> sin(3 * x[1]) * x[2])
        κ = 1.0 .+ rand(rng, padded_size(g)...)
        dκ = zero(κ)
        Enzyme.autodiff(
            Enzyme.Reverse,
            er_kappa_loss,
            Enzyme.Active,
            Enzyme.Duplicated(κ, dκ),
            Enzyme.Const(u.data),
            Enzyme.Const(w),
            Enzyme.Const(g),
        )
        fd = fd_gradient(κd -> er_kappa_loss(κd, u.data, w, g), κ)
        @test any(!iszero, fd)
        @test dκ ≈ fd atol = 1e-5
    end
end
