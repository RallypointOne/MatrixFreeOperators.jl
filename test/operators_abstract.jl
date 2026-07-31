struct DummyNonlinearOp <: MatrixFreeOperators.AbstractOperator end

struct DummyDoubleOp{G} <: MatrixFreeOperators.AbstractOperator
    grid::G
end
MatrixFreeOperators.islinear(::DummyDoubleOp) = true
MatrixFreeOperators.isselfadjoint(::DummyDoubleOp) = true
MatrixFreeOperators.operator_grid(L::DummyDoubleOp) = L.grid
function MatrixFreeOperators.apply!(
    y::Field, ::DummyDoubleOp, x::Field, g::AbstractGrid, α, β
)
    if iszero(β)
        interior(y) .= α .* 2 .* interior(x)
    else
        interior(y) .= α .* 2 .* interior(x) .+ β .* interior(y)
    end
    return y
end
function MatrixFreeOperators.apply_adjoint!(
    x̄::Field, L::DummyDoubleOp, ȳ::Field, g::AbstractGrid, α, β
)
    return MatrixFreeOperators.apply!(x̄, L, ȳ, g, α, β)
end

# Fill every cell of a padded field, ghosts included — the point of the
# accumulating-gather tests below is what happens to a running total's ghosts.
_gather_randel(rng, ::Type{T}) where {T<:Number} = rand(rng, T)
_gather_randel(rng, ::Type{SVector{N,T}}) where {N,T} =
    SVector{N,T}(ntuple(_ -> rand(rng, T), Val(N)))
function _gather_randfill!(f::Field, rng)
    T = eltype(f.data)
    for i in eachindex(f.data)
        f.data[i] = _gather_randel(rng, T)
    end
    return f
end

@testset "Operator abstraction" begin
    g = CartesianGrid(((0.0, 1.0),), (4,))
    u = set!(scalar_field(g), x -> x[1])

    @testset "trait defaults make the weak claim" begin
        L = DummyNonlinearOp()
        @test !MatrixFreeOperators.islinear(L)
        @test !MatrixFreeOperators.isconstant(L)
        @test !MatrixFreeOperators.isselfadjoint(L)
        @test !MatrixFreeOperators.isdiagonal(L)
    end

    @testset "adjoint gated on islinear" begin
        err = try
            adjoint(DummyNonlinearOp())
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("linearize", err.msg)
    end

    @testset "apply matches apply! and applies α, β" begin
        L = DummyDoubleOp(g)
        y = apply(L, u)
        @test collect(interior(y)) ≈ 2 .* collect(interior(u))
        @test all(iszero, y.data[[1, end]])
        @test collect(interior(L(u))) == collect(interior(y))
        @test collect(interior(L * u)) == collect(interior(y))

        z = set!(scalar_field(g), x -> 1.0)
        MatrixFreeOperators.apply!(z, L, u, g, 3.0, 2.0)
        @test collect(interior(z)) ≈ 6 .* collect(interior(u)) .+ 2
    end

    @testset "AdjointOp wrapper" begin
        L = DummyDoubleOp(g)
        A = MatrixFreeOperators.AdjointOp(L)
        @test MatrixFreeOperators.islinear(A)
        @test MatrixFreeOperators.adjoint_operator(A) === L
        y = apply(A, u)
        @test collect(interior(y)) ≈ 2 .* collect(interior(u))
        @test sprint(show, A) == "adjoint(DummyDoubleOp)"
    end

    @testset "size and eltype" begin
        L = DummyDoubleOp(g)
        @test size(L) == (4, 4)
        @test eltype(L) === Float64
        @test_throws ArgumentError size(DummyNonlinearOp())
    end

    # adjoint_gather! accumulates in place (issue #33): no per-call scratch, so the
    # α/β blend runs over the whole padded array and the foldable ghost slabs are
    # cleared before fold_bc! sees them. What must hold on every cell anything
    # reads — interiors and Interface slabs — is x̄ = α·Lᵀȳ + β·x̄₀.
    @testset "accumulating adjoint gather" begin
        Interface = MatrixFreeOperators.Interface   # internal — the BC of slab / leaf grids
        zero_bc_ghosts! = MatrixFreeOperators.zero_bc_ghosts!
        rng = Random.MersenneTwister(33)

        # One cut face per dimension: Laplacian and order-2 Derivative shortcut to
        # the forward action on an all-physical grid, so only an interface-bearing
        # grid exercises their gather.
        gi = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)),
            (8, 6);
            bc=((Interface(), Dirichlet()), (Neumann(), Interface())),
        )
        # Transfer operators reject Interface faces (_validate_transfer), and they
        # gather unconditionally anyway — so they get an all-physical pair, where
        # every ghost slab is foldable and only the interior is live.
        gp = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)),
            (8, 6);
            bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
        )
        gpc = coarsen(gp)

        # run!(out, in, α, β) — `out`/`in` prototypes, and the grid `out` lives on
        # (Prolongation/Restriction land on the coarse grid).
        cases = (
            ("Laplacian", () -> scalar_field(gi), () -> scalar_field(gi), gi,
                (o, i, α, β) -> apply_adjoint!(o, laplacian(gi), i, gi, α, β)),
            ("Derivative order 1", () -> scalar_field(gi), () -> scalar_field(gi), gi,
                (o, i, α, β) -> apply_adjoint!(o, derivative(gi, 1), i, gi, α, β)),
            ("Derivative order 2", () -> scalar_field(gi), () -> scalar_field(gi), gi,
                (o, i, α, β) -> apply_adjoint!(o, derivative(gi, 2; order=2), i, gi, α, β)),
            ("Gradient (rank-reducing adjoint)", () -> scalar_field(gi),
                () -> vector_field(gi), gi,
                (o, i, α, β) -> apply_adjoint!(o, gradient(gi), i, gi, α, β)),
            ("Divergence (rank-raising adjoint)", () -> vector_field(gi),
                () -> scalar_field(gi), gi,
                (o, i, α, β) -> apply_adjoint!(o, divergence(gi), i, gi, α, β)),
            ("Prolongation", () -> scalar_field(gpc), () -> scalar_field(gp), gpc,
                (o, i, α, β) -> apply_adjoint!(o, prolongation(gpc, gp), i, gp, α, β)),
            # Restriction's FORWARD action is a scaled Pᵀ gather, so it inherits the
            # same contract.
            ("Restriction (forward)", () -> scalar_field(gpc), () -> scalar_field(gp), gpc,
                (o, i, α, β) -> apply!(o, restriction(gp, gpc), i, gp, α, β)),
            # How β = true actually arrives in practice: Added's second term.
            ("Added", () -> scalar_field(gi), () -> scalar_field(gi), gi,
                (o, i, α, β) -> apply_adjoint!(o, laplacian(gi) + derivative(gi, 1), i, gi, α, β)),
        )

        α, β = 2.5, -1.5
        for (name, mkout, mkin, gout, run!) in cases
            @testset "$name" begin
                x̄₀ = _gather_randfill!(mkout(), rng)
                ȳ = _gather_randfill!(mkin(), rng)

                clean = mkout()
                fill!(clean.data, zero(eltype(clean.data)))
                run!(clean, copy(ȳ), true, false)

                x̄ = copy(x̄₀)
                run!(x̄, copy(ȳ), α, β)

                # true on every cell a consumer reads: interiors plus the Interface
                # slabs the distributed reduction and halo_update_adjoint! collect.
                live = zero_bc_ghosts!(fill!(similar(x̄.data, Bool), true), gout)
                expected = α .* clean.data .+ β .* x̄₀.data
                @test all(isapprox.(x̄.data[live], expected[live]))
                # foldable ghosts: folded away, not left holding β·junk
                @test all(iszero, x̄.data[.!live])
                @test any(!iszero, x̄₀.data[.!live])   # the junk was really there
            end
        end
    end

    @testset "accumulating adjoint gather does not allocate" begin
        Interface = MatrixFreeOperators.Interface
        # Big enough that a leaked padded scratch (~34 kB) could not hide under the
        # bound: the pre-fix code allocated one full padded array per call.
        ga = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)),
            (64, 64);
            bc=((Interface(), Dirichlet()), (Neumann(), Interface())),
        )
        function alloc_adjoint(L, x̄, ȳ, g)
            apply_adjoint!(x̄, L, ȳ, g, 1.5, 2.0)
            apply_adjoint!(x̄, L, ȳ, g, 1.5, 2.0)
            a = @allocated apply_adjoint!(x̄, L, ȳ, g, 1.5, 2.0)
            return a, sum(interior(x̄))   # DCE-proof: consume the output
        end
        for L in (laplacian(ga), derivative(ga, 1), laplacian(ga) + derivative(ga, 2; order=2))
            ȳ = set!(scalar_field(ga), x -> sinpi(x[1]) * exp(-x[2]))
            a, s = alloc_adjoint(L, scalar_field(ga), ȳ, ga)
            @test isfinite(s) && !iszero(s)
            @test a ≤ 512
        end
    end
end
