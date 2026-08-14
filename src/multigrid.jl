#--------------------------------------------------------------------------------# Smoother configurations

"""
    Jacobi(ω=2//3, sweeps=2)

Weighted-Jacobi smoother configuration for [`MultigridPreconditioner`](@ref):
`x ← x + ω·D⁻¹·(b - A·x)` repeated `sweeps` times, with `D` extracted once per
level via [`operator_diagonal`](@ref). Applied identically before and after
each coarse-grid correction, which keeps the V-cycle symmetric.
"""
struct Jacobi{T<:Real}
    ω::T
    sweeps::Int
end
Jacobi(ω::Real=2//3) = Jacobi(ω, 2)

"""
    Chebyshev(degree=3)

Chebyshev polynomial smoother configuration for
[`MultigridPreconditioner`](@ref): a fixed degree-`degree` polynomial in
`D⁻¹·A` targeting the upper part of the spectrum `[λmax/4, λmax]`, with `λmax`
estimated by power iteration at construction. Needs only repeated `apply!` plus
[`operator_diagonal`](@ref); fixed coefficients keep the V-cycle symmetric.
"""
struct Chebyshev
    degree::Int
end
Chebyshev() = Chebyshev(3)

#--------------------------------------------------------------------------------# Per-level smoother state

_di(d::Number) = d
_di(d::Field) = interior(d)

_inv_diag(d::Number) = inv(d)
function _inv_diag(d::Field)
    invd = scalar_field(d.grid, typeof(inv(one(eltype(d.data)))))
    interior(invd) .= inv.(interior(d))
    return invd
end

struct JacobiState{T<:Real,D}
    ω::T
    sweeps::Int
    invdiag::D
end

struct ChebyshevState{T<:Real,D,F<:Field}
    degree::Int
    invdiag::D
    θ::T
    δ::T
    d::F
end

function _smoother_state(cfg::Jacobi, Lc::AbstractOperator, A, g::CartesianGrid)
    return JacobiState(cfg.ω, cfg.sweeps, _inv_diag(operator_diagonal(Lc)))
end

_checkerboard(I::CartesianIndex) = iseven(sum(Tuple(I))) ? 1 : -1

function _smoother_state(cfg::Chebyshev, Lc::AbstractOperator, A, g::CartesianGrid)
    invd = _inv_diag(operator_diagonal(Lc))
    T = eltype(spacing(g))
    v = scalar_field(g, T)
    w = scalar_field(g, T)
    interior(v) .= _checkerboard.(interior(g))
    λ = zero(T)
    for _ in 1:10
        apply!(w, A, v, g)
        interior(w) .= _di(invd) .* interior(w)
        λ = abs(dot(interior(v), interior(w)) / dot(interior(v), interior(v)))
        nw = norm(interior(w))
        iszero(nw) && break
        interior(v) .= interior(w) ./ nw
    end
    λ > 0 || throw(
        ArgumentError(
            "Chebyshev setup: power iteration found no positive eigenvalue estimate for D⁻¹A",
        ),
    )
    hi = T(11) / 10 * λ
    lo = hi / 4
    return ChebyshevState(cfg.degree, invd, (hi + lo) / 2, (hi - lo) / 2, scalar_field(g, T))
end

#--------------------------------------------------------------------------------# Level data and smoothing sweeps

struct MGLevel{O<:AbstractOperator,RO<:Restriction,PO<:Prolongation,G<:CartesianGrid,F<:Field,S}
    A::O          # prepared rediscretized operator on this level
    R::RO         # this level -> next-coarser
    P::PO         # next-coarser -> this level
    g::G
    x::F          # correction
    b::F          # level right-hand side
    r::F          # residual scratch
    t::F          # spare scratch
    smoother::S
end

struct MGCoarsest{G<:CartesianGrid,F<:Field,T<:Number,LU}
    g::G
    x::F
    b::F
    fact::LU
    bflat::Vector{T}
end

_zero_interior!(f::Field) = (fill!(interior(f), zero(eltype(f.data))); f)

_smooth!(lvl::MGLevel, fromzero::Bool) = _smooth!(lvl.smoother, lvl, fromzero)

function _smooth!(s::JacobiState, lvl::MGLevel, fromzero::Bool)
    invd = _di(s.invdiag)
    ω = s.ω
    start = 1
    if fromzero
        interior(lvl.x) .= ω .* invd .* interior(lvl.b)
        start = 2
    end
    for _ in start:s.sweeps
        apply!(lvl.r, lvl.A, lvl.x, lvl.g)
        interior(lvl.x) .+= ω .* invd .* (interior(lvl.b) .- interior(lvl.r))
    end
    return nothing
end

# Classic first-kind Chebyshev recurrence on the smoothing range [lo, hi] of
# σ(D⁻¹A): θ = (hi+lo)/2, δ = (hi-lo)/2, σ₁ = θ/δ.
function _smooth!(s::ChebyshevState, lvl::MGLevel, fromzero::Bool)
    invd = _di(s.invdiag)
    θ, δ = s.θ, s.δ
    σ1 = θ / δ
    ρ = δ / θ
    if fromzero
        _zero_interior!(lvl.x)
        interior(lvl.r) .= interior(lvl.b)
    else
        apply!(lvl.r, lvl.A, lvl.x, lvl.g)
        interior(lvl.r) .= interior(lvl.b) .- interior(lvl.r)
    end
    interior(s.d) .= invd .* interior(lvl.r) ./ θ
    for k in 1:s.degree
        interior(lvl.x) .+= interior(s.d)
        k == s.degree && break
        apply!(lvl.t, lvl.A, s.d, lvl.g)
        interior(lvl.r) .-= interior(lvl.t)
        ρnew = inv(2 * σ1 - ρ)
        interior(s.d) .=
            (ρnew * ρ) .* interior(s.d) .+ (2 * ρnew / δ) .* invd .* interior(lvl.r)
        ρ = ρnew
    end
    return nothing
end

#--------------------------------------------------------------------------------# V-cycle

# Tuple recursion (Base.tail idiom): fully specialized per level, allocation-free.
# Every level's pre-smooth starts from x = 0, so no explicit zeroing is needed —
# the fromzero sweep overwrites x.
function _vcycle!(levels::Tuple)
    lvl = first(levels)
    rest = Base.tail(levels)
    _smooth!(lvl, true)
    apply!(lvl.r, lvl.A, lvl.x, lvl.g)
    interior(lvl.r) .= interior(lvl.b) .- interior(lvl.r)
    nxt = first(rest)
    apply!(nxt.b, lvl.R, lvl.r, lvl.g)
    _vcycle!(rest)
    apply!(lvl.x, lvl.P, nxt.x, lvl.g, true, true)
    _smooth!(lvl, false)
    return nothing
end

function _vcycle!(levels::Tuple{<:MGCoarsest})
    c = first(levels)
    interior_to_flat!(c.bflat, c.b)
    ldiv!(c.fact, c.bflat)
    flat_to_interior!(c.x, c.bflat)
    return nothing
end

#--------------------------------------------------------------------------------# Rediscretization walk

_rediscretize(::Laplacian, gc::CartesianGrid) = Laplacian(gc)
_rediscretize(L::Derivative, gc::CartesianGrid) = Derivative(gc, L.dim, L.order)
_rediscretize(::Gradient, gc::CartesianGrid) = Gradient(gc)
_rediscretize(::Divergence, gc::CartesianGrid) = Divergence(gc)
_rediscretize(L::IdentityOp, ::CartesianGrid) = L
_rediscretize(L::ScalingOp{<:Number}, ::CartesianGrid) = L
function _rediscretize(L::ScalingOp{<:Field}, gc::CartesianGrid)
    return ScalingOp(_average_to_coarse(L.coeff, gc))
end
# The coarse coefficient goes through the child mean, not Restriction — see
# _average_to_coarse. check=false: the fine κ was already validated at construction, and
# both means map positive values to positive values.
function _rediscretize(L::Diffusion, gc::CartesianGrid)
    return diffusion(gc, _average_to_coarse(L.κ, gc); averaging=L.avg, check=false)
end
_rediscretize(L::Scaled, gc::CartesianGrid) = Scaled(_rediscretize(L.op, gc), L.α)
function _rediscretize(L::Added, gc::CartesianGrid)
    return Added(_rediscretize(L.a, gc), _rediscretize(L.b, gc))
end
function _rediscretize(L::Composed, gc::CartesianGrid)
    return Composed(_rediscretize(L.a, gc), _rediscretize(L.b, gc))
end
_rediscretize(L::AbstractOperator, ::CartesianGrid) = throw(
    ArgumentError(
        "multigrid rediscretization does not support $(nameof(typeof(L))); supported " *
        "leaves: Laplacian, Derivative, Gradient, Divergence, ScalingOp, Diffusion, IdentityOp",
    ),
)

# Plain 2^N-child arithmetic mean of a coefficient field onto the coarse grid.
# Deliberately NOT restriction: R folds Dirichlet's -1 mirror through the
# boundary, which would corrupt a material coefficient at walls.
@inline function _child_mean(
    u::AbstractArray{T,N},
    J::CartesianIndex{N},
    hf::NTuple{N,Int},
    hc::NTuple{N,Int},
    scale,
) where {T,N}
    F0 = ntuple(d -> 2 * (J[d] - hc[d]) - 1 + hf[d], Val(N))
    acc = zero(T)
    @inbounds for t in CartesianIndices(ntuple(_ -> 0:1, Val(N)))
        acc += u[CartesianIndex(ntuple(d -> F0[d] + t[d], Val(N)))]
    end
    return scale * acc
end

function _average_to_coarse(f::Field, gc::CartesianGrid{N}) where {N}
    c = _field_like(f, gc)
    hf = halo_width(f.grid)
    hc = halo_width(gc)
    scale = inv(eltype(spacing(gc))(2)^N)
    interior(c) .= _child_mean.(Ref(f.data), interior(gc), Ref(hf), Ref(hc), scale)
    return c
end

#--------------------------------------------------------------------------------# Level construction

function _mg_grids(g::CartesianGrid, levels)
    grids = [g]
    if levels === :auto
        while all(iseven, local_size(last(grids))) &&
            prod(local_size(last(grids))) > 64 &&
            length(grids) < 10
            push!(grids, coarsen(last(grids)))
        end
        length(grids) > 1 || throw(
            ArgumentError(
                "levels=:auto found no coarsenable level below grid size " *
                "$(local_size(g)); pass levels=n explicitly",
            ),
        )
    elseif levels isa Integer
        levels >= 2 || throw(ArgumentError("levels must be ≥ 2, got $levels"))
        for _ in 2:levels
            push!(grids, coarsen(last(grids)))
        end
    else
        throw(ArgumentError("levels must be :auto or an Integer ≥ 2"))
    end
    return grids
end

function _build_levels(L::AbstractOperator, grids::Vector, smoother, i::Int)
    g = grids[i]
    i == length(grids) && return (_build_coarsest(L, g),)
    proto = scalar_field(g)
    A = _prepare_tree(L, proto)
    lvl = MGLevel(
        A,
        restriction(g, grids[i + 1]),
        prolongation(grids[i + 1], g),
        g,
        scalar_field(g),
        scalar_field(g),
        scalar_field(g),
        scalar_field(g),
        _smoother_state(smoother, L, A, g),
    )
    return (lvl, _build_levels(_rediscretize(L, grids[i + 1]), grids, smoother, i + 1)...)
end

function _build_coarsest(L::AbstractOperator, g::CartesianGrid)
    proto = scalar_field(g)
    A = _prepare_tree(L, proto)
    T = eltype(spacing(g))
    n = prod(local_size(g))
    Ad = Matrix{T}(undef, n, n)
    e = scalar_field(g)
    y = scalar_field(g)
    bflat = Vector{T}(undef, n)
    for j in 1:n
        fill!(e.data, zero(T))
        fill!(bflat, zero(T))
        bflat[j] = one(T)
        flat_to_interior!(e, bflat)
        apply!(y, A, e, g)
        interior_to_flat!(view(Ad, :, j), y)
    end
    fact = lu(Ad; check=false)
    dU = abs.(diag(fact.factors))
    if !issuccess(fact) || minimum(dU) <= sqrt(eps(T)) * maximum(dU)
        throw(
            ArgumentError(
                "the operator is singular at the coarsest multigrid level — an " *
                "all-Neumann/Periodic Poisson problem has a constant nullspace; add " *
                "a Dirichlet face or a zeroth-order term",
            ),
        )
    end
    return MGCoarsest(g, scalar_field(g), scalar_field(g), fact, bflat)
end

#--------------------------------------------------------------------------------# Preconditioner

"""
    MultigridPreconditioner(L; smoother=Jacobi(), cycle=:V, levels=:auto)

Geometric-multigrid V-cycle preconditioner for a linear operator on a uniform
[`CartesianGrid`](@ref). Each `mul!(z, M, r)` runs one V-cycle from a zero
initial guess, so `M` acts as a fixed linear approximation of `L⁻¹` — pass it
as the preconditioner `M` to a Krylov solver.

Levels are built by [`coarsen`](@ref)-ing the grid (`levels=:auto` stops at ≤64
DOFs, odd cell counts, or 10 levels; pass an `Integer` to force a depth); each
level rediscretizes `L` on its grid, transfers residuals/corrections with
[`restriction`](@ref)/[`prolongation`](@ref), smooths with `smoother`
([`Jacobi`](@ref) or [`Chebyshev`](@ref)) symmetrically before and after the
coarse correction, and solves the coarsest level with a dense LU factored at
construction. For a symmetric `L` the cycle is symmetric, so `Krylov.cg`
applies.

Stateful and single-threaded, like [`prepare`](@ref) — construct one per
concurrent solve.

### Examples

```julia
g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (64, 64))
L = -laplacian(g)
A = prepare(L)
M = MultigridPreconditioner(L)
b = flatten(set!(scalar_field(g), x -> sinpi(x[1]) * sinpi(x[2])))
u, stats = Krylov.cg(A, b; M)
```

See also: [`MultigridSolver`](@ref), [`operator_diagonal`](@ref).
"""
struct MultigridPreconditioner{Lv<:Tuple}
    levels::Lv
end

function MultigridPreconditioner(
    L::AbstractOperator; smoother=Jacobi(), cycle::Symbol=:V, levels=:auto
)
    islinear(L) || throw(
        ArgumentError(
            "multigrid requires a linear operator; linearize nonlinear operators first",
        ),
    )
    g = operator_grid(L)
    g isa CartesianGrid || throw(
        ArgumentError(
            "geometric multigrid supports uniform CartesianGrid operators only; " *
            "multigrid on a BlockForest is not yet supported",
        ),
    )
    cycle === :V ||
        throw(ArgumentError("only cycle=:V is supported in v1; :W and :F follow"))
    return MultigridPreconditioner(_build_levels(L, _mg_grids(g, levels), smoother, 1))
end

function Base.size(M::MultigridPreconditioner)
    n = prod(local_size(first(M.levels).g))
    return (n, n)
end
Base.size(M::MultigridPreconditioner, d::Integer) = size(M)[d]
Base.eltype(M::MultigridPreconditioner) = eltype(spacing(first(M.levels).g))

function LinearAlgebra.mul!(
    z::AbstractVector, M::MultigridPreconditioner, r::AbstractVector, α::Number, β::Number
)
    top = first(M.levels)
    flat_to_interior!(top.b, r)
    _vcycle!(M.levels)
    interior_to_flat!(z, top.x, α, β)
    return z
end
function LinearAlgebra.mul!(z::AbstractVector, M::MultigridPreconditioner, r::AbstractVector)
    return mul!(z, M, r, true, false)
end

_smoother_name(::JacobiState) = "Jacobi"
_smoother_name(::ChebyshevState) = "Chebyshev"

function Base.show(io::IO, M::MultigridPreconditioner)
    top = first(M.levels)
    bottom = last(M.levels)
    return print(
        io,
        "MultigridPreconditioner($(length(M.levels)) levels: ",
        join(local_size(top.g), "×"),
        " → ",
        join(local_size(bottom.g), "×"),
        ", ",
        _smoother_name(top.smoother),
        ")",
    )
end

#--------------------------------------------------------------------------------# Standalone solver

"""
    MultigridSolver(L; smoother=Jacobi(), cycle=:V, levels=:auto, maxiter=200)

Standalone multigrid solver: the stationary iteration `u ← u + M·(b - A·u)`
with `M` a [`MultigridPreconditioner`](@ref) of `L` and `A = prepare(L)`. Run
it with [`solve`](@ref).
"""
struct MultigridSolver{M<:MultigridPreconditioner,P<:PreparedOperator,T<:Number}
    M::M
    A::P
    r::Vector{T}
    z::Vector{T}
    maxiter::Int
end

function MultigridSolver(
    L::AbstractOperator; smoother=Jacobi(), cycle::Symbol=:V, levels=:auto, maxiter::Int=200
)
    M = MultigridPreconditioner(L; smoother, cycle, levels)
    A = prepare(L)
    T = eltype(A)
    n = size(A, 2)
    return MultigridSolver(M, A, Vector{T}(undef, n), Vector{T}(undef, n), maxiter)
end

"""
    solve(s::MultigridSolver, b::AbstractVector; rtol=1e-8, atol=0) -> u

Solve `A·u = b` by multigrid V-cycle iteration on flat interior-DOF vectors
(the [`flatten`](@ref) layout), from a zero initial guess, until
`‖b - A·u‖ ≤ max(rtol·‖b‖, atol)` or `maxiter` cycles. Warns and returns the
current iterate if the tolerance is not reached.

Note: other packages (CommonSolve/SciML) also export a `solve`; qualify as
`MatrixFreeOperators.solve` when both are loaded.

### Examples

```julia
u = solve(MultigridSolver(-laplacian(g)), flatten(f); rtol=1e-10)
```
"""
function solve(s::MultigridSolver, b::AbstractVector; rtol::Real=1e-8, atol::Real=zero(rtol))
    T = eltype(s.A)
    u = zeros(T, length(b))
    tol = max(rtol * norm(b), atol)
    for _ in 1:s.maxiter
        mul!(s.r, s.A, u)
        s.r .= b .- s.r
        norm(s.r) <= tol && return u
        mul!(s.z, s.M, s.r)
        u .+= s.z
    end
    mul!(s.r, s.A, u)
    s.r .= b .- s.r
    @warn "multigrid solve did not reach the requested tolerance in $(s.maxiter) cycles" residual =
        norm(s.r) tol
    return u
end
