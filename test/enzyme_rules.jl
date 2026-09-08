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

function er_diffusion_kappa_loss(κdata, udata, w, g)
    D = diffusion(g, Field(κdata, g); check=false)
    return sum(w .* interior(apply(D, Field(copy(udata), g))))
end

function er_forest_diffusion_kappa_loss(κv, v, w, bf)
    κ = scalar_field(bf)
    flat_to_interior!(κ, κv)
    D = diffusion(bf, κ; check=false)
    u = scalar_field(bf)
    flat_to_interior!(u, v)
    return sum(w .* flatten(apply(D, u)))
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

    @testset "composed operators: repeated exchange of one field" begin
        # The rules apply the transpose *in place* (x̄ ← Hᵀx̄) rather than
        # accumulating, which is right only because the exchange overwrites the same
        # storage it reads. Combinators are where that gets stressed: each operand of
        # an Added re-exchanges the same input field, and a Composed exchanges a
        # freshly allocated intermediate as well. If the in-place semantics were
        # wrong, the second exchange's cotangent would be lost or double-counted.
        gf = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (8, 8);
            bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
        )
        bf = BlockForest(gf; blocksize=(4, 4), maxlevel=2)
        refine!(bf, x -> x[1] < 0.5)
        MatrixFreeOperators._exchange_schedule(bf)
        n = length(flatten(scalar_field(bf)))
        v = rand(rng, n)
        wf = rand(rng, n)

        @testset "$(name)" for (name, L, nexchange) in (
            ("Added: Δ + 2Δ", laplacian(bf) + 2 * laplacian(bf), 2),
            ("Added: Δ + ∂x", laplacian(bf) + derivative(bf, 1), 2),
            ("Composed: Δ∘Δ", laplacian(bf) * laplacian(bf), 2),
        )
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
            # One rule invocation per operand exchange — proof the combinator really
            # did re-exchange, so this test is not vacuous.
            @test ENZ_EXT.rule_hits().exchange - before.exchange == nexchange

            # Relative, not absolute: Δ∘Δ scales like h⁻⁴, so these gradients run to
            # ~1e6 and any absolute tolerance is meaningless. 1e-6 is the central
            # difference's own accuracy — the AD/FD agreement is ~1e-9 relative.
            fd = fd_gradient(vd -> er_forest_loss(vd, wf, L, bf), v)
            @test dv ≈ fd rtol = 1e-6
        end
    end

    @testset "forest Diffusion: exchange count at the Laplacian baseline, κ taped" begin
        gf = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (8, 8);
            bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
        )
        bf = BlockForest(gf; blocksize=(4, 4), maxlevel=2)
        refine!(bf, x -> x[1] < 0.5)
        MatrixFreeOperators._exchange_schedule(bf)
        n = length(flatten(scalar_field(bf)))
        v = rand(rng, n)
        wf = rand(rng, n)
        D = diffusion(bf, set!(scalar_field(bf), x -> 1.2 + 0.8 * x[1]^2 + 0.5 * x[2]))

        # The measurable form of "no additional halo exchange per application": one
        # exchange-rule invocation per apply, exactly the Laplacian's count — the
        # coarse–fine flux rewrite reuses the exchanged ghosts, it never adds a pass.
        lastdv = zero(v)
        for L in (laplacian(bf), D)
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
            @test after.exchange - before.exchange == 1
            @test after.bc - before.bc == 1
            lastdv = dv
        end
        # and the gradient the rules helped produce is the declared adjoint's,
        # seam transpose included
        w̃ = scalar_field(bf)
        flat_to_interior!(w̃, wf)
        @test lastdv ≈ flatten(apply_adjoint!(scalar_field(bf), D, w̃, bf)) rtol = 1e-8

        # κ active: the halo rules still fire only for the solution exchange — the
        # κ path (coefficient exchange + rewrite weights) stays on the tape, so its
        # gradient is nonzero and matches finite differences instead of being
        # silently zeroed by a rule's empty derivative slots.
        κv = 1.0 .+ rand(rng, n)
        dκ = zero(κv)
        before = ENZ_EXT.rule_hits()
        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            er_forest_diffusion_kappa_loss,
            Enzyme.Active,
            Enzyme.Duplicated(κv, dκ),
            Enzyme.Const(v),
            Enzyme.Const(wf),
            Enzyme.Const(bf),
        )
        @test ENZ_EXT.rule_hits().exchange - before.exchange == 1
        fdκ = fd_gradient(κd -> er_forest_diffusion_kappa_loss(κd, v, wf, bf), κv)
        @test any(!iszero, fdκ)
        @test dκ ≈ fdκ atol = 1e-5
    end

    @testset "shadow extraction covers the annotation lattice" begin
        # A rule must handle every annotation its signature claims: Enzyme marks the
        # call site as ruled with activity erased, so an uncovered annotation
        # surfaces as a runtime MethodError rather than a fallback to taping.
        #
        # This is a unit test rather than an end-to-end one on purpose. Batched
        # reverse mode over a scalar loss is not reachable through
        # `Enzyme.autodiff` — it requires the thunk API with an explicit NTuple of
        # seeds — so an end-to-end version would be testing Enzyme's calling
        # convention, not this package's rules.
        a = [1.0, 2.0]
        b = [3.0, 4.0]
        c = [5.0, 6.0]
        @test ENZ_EXT._shadows(Enzyme.Const(a)) == ()
        @test ENZ_EXT._shadows(Enzyme.Duplicated(a, b)) == (b,)
        @test ENZ_EXT._shadows(Enzyme.DuplicatedNoNeed(a, b)) == (b,)
        @test ENZ_EXT._shadows(Enzyme.BatchDuplicated(a, (b, c))) == (b, c)
        @test ENZ_EXT._shadows(Enzyme.BatchDuplicatedNoNeed(a, (b, c))) == (b, c)
        # Mixed* should not arise for pure-pointer storage, but an uncovered
        # annotation is a runtime MethodError, not a fallback — so cover it.
        @test ENZ_EXT._shadows(Enzyme.MixedDuplicated(a, Ref(b))) == (b,)
        @test ENZ_EXT._shadows(Enzyme.BatchMixedDuplicated(a, (Ref(b), Ref(c)))) == (b, c)
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
            ("Diffusion", diffusion(g, κ0)),
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

        # Same for the fused leaf: its coefficient accessor is pure reads and index
        # arithmetic, so the whole κ path stays on the tape with no rule in sight.
        dκd = zero(κ)
        before = ENZ_EXT.rule_hits()
        # Runtime activity: constructing the leaf inside the differentiated region
        # stores the Const grid into the active coefficient Field. It does not affect
        # rule dispatch, so the rule_hits assertion below still means what it says.
        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            er_diffusion_kappa_loss,
            Enzyme.Active,
            Enzyme.Duplicated(κ, dκd),
            Enzyme.Const(u.data),
            Enzyme.Const(w),
            Enzyme.Const(g),
        )
        @test ENZ_EXT.rule_hits() == before
        fdd = fd_gradient(κd -> er_diffusion_kappa_loss(κd, u.data, w, g), κ)
        @test any(!iszero, fdd)
        @test dκd ≈ fdd atol = 1e-5
    end
end
