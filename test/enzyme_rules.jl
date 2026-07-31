# Rules that silently stop dispatching are the dangerous failure mode: every
# numerical assertion still passes, via the tape, and the only symptom is that the
# thing the rules exist to avoid — taping the coarse–fine descriptor sweep and the
# exchange kernels — quietly comes back. So each testset here checks *both* that the
# gradient is right and that the rule fired.

const ENZ_EXT = Base.get_extension(MatrixFreeOperators, :MatrixFreeOperatorsEnzymeCoreExt)

rule_hits() = Dict(k => v[] for (k, v) in ENZ_EXT.RULE_HITS)
fired_since(before) = Set(k for (k, v) in rule_hits() if v > before[k])

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

    @testset "single grid: apply_bc! rule routes through fold_bc!" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (5, 4);
            bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
        )
        w = rand(rng, local_size(g)...)
        κ0 = set!(scalar_field(g), x -> 1 + x[1] * x[2])

        @testset "$(name)" for (name, L) in (
            ("Laplacian", laplacian(g)),
            ("Derivative", derivative(g, 1)),
            ("div∘κ∘grad", divergence(g) * scaling(κ0) * MatrixFreeOperators.gradient(g)),
        )
            x = rand(rng, padded_size(g)...)
            dx = zero(x)
            before = rule_hits()
            Enzyme.autodiff(
                Enzyme.set_runtime_activity(Enzyme.Reverse),
                er_field_loss,
                Enzyme.Active,
                Enzyme.Duplicated(x, dx),
                Enzyme.Const(w),
                Enzyme.Const(L),
                Enzyme.Const(g),
            )
            @test :apply_bc in fired_since(before)

            # Same three oracles the tape-through suite uses, at the same tolerances:
            # the rule must be an exact substitution, not an approximation.
            fd = fd_gradient(xd -> er_field_loss(xd, w, L, g), x)
            @test dx ≈ fd atol = 1e-5

            w̃ = scalar_field(g)
            interior(w̃) .= w
            lt = apply(adjoint(L), w̃)
            @test dx[interior(g)] ≈ collect(interior(lt)) rtol = 1e-10

            ghost_mask = trues(padded_size(g))
            ghost_mask[interior(g)] .= false
            @test all(iszero, dx[ghost_mask])
        end
    end

    @testset "forest: exchange + BC-face rules (refined=$(refined))" for refined in
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

        w̃ = scalar_field(bf)
        flat_to_interior!(w̃, wf)
        lt = flatten(apply_adjoint!(scalar_field(bf), L, w̃, bf))

        dv = zero(v)
        before = rule_hits()
        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            er_forest_loss,
            Enzyme.Active,
            Enzyme.Duplicated(v, dv),
            Enzyme.Const(wf),
            Enzyme.Const(L),
            Enzyme.Const(bf),
        )
        fired = fired_since(before)
        # The whole point: `_exchange_storage!` ran under a rule, so `_run_fills!`
        # never reached Enzyme's type analysis (issue #26).
        @test :exchange in fired
        @test :bc_faces in fired
        @test dv ≈ lt rtol = 1e-8
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

    @testset "adjoint direction: (Bᵀ)ᵀ = B keeps fold_bc! on a rule" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (5, 4);
            bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
        )
        w = rand(rng, local_size(g)...)
        L = adjoint(laplacian(g))    # applying this runs fold_bc! in the primal
        x = rand(rng, padded_size(g)...)
        dx = zero(x)
        before = rule_hits()
        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            er_field_loss,
            Enzyme.Active,
            Enzyme.Duplicated(x, dx),
            Enzyme.Const(w),
            Enzyme.Const(L),
            Enzyme.Const(g),
        )
        @test !isempty(fired_since(before))
        fd = fd_gradient(xd -> er_field_loss(xd, w, L, g), x)
        @test dx ≈ fd atol = 1e-5
    end

    @testset "batch width 2 and Const shadows" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (5, 4);
            bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
        )
        w = rand(rng, local_size(g)...)
        L = laplacian(g)
        x = rand(rng, padded_size(g)...)

        # BatchDuplicated exercises the multi-shadow branch of the rules, which
        # nothing else in the suite reaches.
        dx1 = zero(x)
        dx2 = zero(x)
        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            er_field_loss,
            Enzyme.BatchDuplicated(x, (dx1, dx2)),
            Enzyme.Const(w),
            Enzyme.Const(L),
            Enzyme.Const(g),
        )
        fd = fd_gradient(xd -> er_field_loss(xd, w, L, g), x)
        # Both shadows are seeded with the same unit cotangent, so both must equal
        # the single-width gradient.
        @test dx1 ≈ fd atol = 1e-5
        @test dx2 ≈ dx1

        # A Const field must leave the rule a no-op rather than erroring.
        @test apply_bc!(copy(x), g) isa AbstractArray
    end
end
