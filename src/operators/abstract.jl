#--------------------------------------------------------------------------------# Operator abstraction

"""
    AbstractOperator

Supertype of all matrix-free operators. A concrete operator provides:

- `apply!(y, L, x, grid, α, β)` — the action `y = α·L(x) + β·y`, writing the
  interior of `y` (ghost layers of `x` are scratch: filled by `halo_update!` and
  [`apply_bc!`](@ref) before the stencil reads neighbors)
- traits [`islinear`](@ref), [`isconstant`](@ref), [`isselfadjoint`](@ref),
  [`isdiagonal`](@ref) (default `false` — leaves opt in)
- for linear operators: an adjoint via `adjoint_operator` and
  [`apply_adjoint!`](@ref)

Operators compose lazily with `+`, `-`, `*`, and scalar scaling, and are applied
with [`apply`](@ref), `L(x)`, or `L * x`. For Krylov solvers, wrap with
[`prepare`](@ref) to get an allocation-free `mul!`.
"""
abstract type AbstractOperator end

#--------------------------------------------------------------------------------# Traits

"""
    islinear(L::AbstractOperator) -> Bool

Whether `L` is a linear map of its input field. Defaults to `false`; linear leaves
opt in. Gates `adjoint` and use as a Krylov linear map — see [`linearize`](@ref)
for the nonlinear path.
"""
islinear(::AbstractOperator) = false

"""
    isconstant(L::AbstractOperator) -> Bool

Whether the operator's parameters and coefficients are time-invariant. Defaults to
`false`; leaves opt in.
"""
isconstant(::AbstractOperator) = false

"""
    isselfadjoint(L::AbstractOperator) -> Bool

Whether `⟨L*x, y⟩ = ⟨x, L*y⟩` holds, *including* boundary contributions. Defaults
to `false` — boundary conditions break self-adjointness easily, so leaves opt in
only when their BC handling makes it exact.
"""
isselfadjoint(::AbstractOperator) = false

# Whether the grid's inter-block ghost coupling is symmetric — the extra condition
# a stencil leaf's self-adjointness claim needs on a composite grid. Same-level
# halo copies couple both neighbors symmetrically; coarse–fine interpolation does
# not, so a non-uniform forest breaks self-adjointness even for the Laplacian
# (queried live: a regrid can flip it, and stencil leaves hold only the grid).
_selfadjoint_grid(::AbstractGrid) = true
_selfadjoint_grid(g::BlockForest) = g.forest.uniform[]

"""
    isdiagonal(L::AbstractOperator) -> Bool

Whether `L` acts pointwise (a diagonal operator). Enables cheap Jacobi-type
smoothers. Defaults to `false`.
"""
isdiagonal(::AbstractOperator) = false

"""
    operator_diagonal(L::AbstractOperator) -> Number | Field

Exact diagonal of `L` as a linear map over interior DOFs, including boundary
contributions — a `Number` when the diagonal is uniform, a scalar
[`Field`](@ref) (ghost entries zero) otherwise. Powers the multigrid smoothers.
There is no generic fallback: leaves declare their diagonal explicitly, so a
missing declaration errors instead of degrading to a wrong diagonal.
"""
operator_diagonal(L::AbstractOperator) = throw(
    ArgumentError(
        "operator_diagonal has no method for $(nameof(typeof(L))); declare one to enable Jacobi/Chebyshev smoothing",
    ),
)

#--------------------------------------------------------------------------------# Action

"""
    apply!(y::AbstractField, L::AbstractOperator, x::AbstractField, g::AbstractGrid, α=true, β=false) -> y

Apply the operator in place: `y = α·L(x) + β·y` on the interior of `y`. Ghost
layers of `x` are treated as scratch (overwritten with halo/BC fills); ghost
layers of `y` are left untouched by the accumulating form. The three-argument
form takes the grid from `x`.

This is the explicit time-stepping path; `prepare` + `mul!` is the Krylov
boundary and stages through flat vectors. `*`-composed and adjoint trees
allocate their intermediates here — [`prepare`](@ref) those once and call
`apply!(du, P, u)` instead. Like every `apply!` this is the homogeneous linear
part: with inhomogeneous BCs the RHS is `L(u) + b`, `b = boundary_rhs(L, g)`.

### Examples

```julia
L = laplacian(g)
du = similar(u)
apply!(du, L, u)                        # du = Δu
interior(u) .+= dt .* interior(du)      # one forward-Euler step
```

See also: [`apply`](@ref), [`apply_adjoint!`](@ref), [`prepare`](@ref).
"""
function apply!(y::AbstractField, L::AbstractOperator, x::AbstractField, g::AbstractGrid)
    return apply!(y, L, x, g, true, false)
end
apply!(y::AbstractField, L::AbstractOperator, x::AbstractField) = apply!(y, L, x, x.grid)

"""
    apply(L::AbstractOperator, x::AbstractField) -> AbstractField

Allocating application `L(x)`. The pure path used by autodiff; hot loops should
use in-place [`apply!`](@ref) instead (explicit stepping), or [`prepare`](@ref) +
`mul!` at the Krylov boundary.

### Examples

```julia
g = CartesianGrid(((0.0, 2π),), (64,); bc=((Periodic(), Periodic()),))
u = set!(scalar_field(g), x -> sin(x[1]))
Δu = apply(laplacian(g), u)        # equivalently laplacian(g)(u) or laplacian(g) * u
```
"""
function apply(L::AbstractOperator, x::AbstractField)
    y = allocate_output(L, x)
    apply!(y, L, x, x.grid)
    return y
end

(L::AbstractOperator)(x::AbstractField) = apply(L, x)
Base.:*(L::AbstractOperator, x::AbstractField) = apply(L, x)

# Output field for L(x): same shape/eltype as x by default; ghost layers zeroed so
# every package-produced field has deterministic ghosts. Rank-changing leaves
# override to switch the element type. Generic over Field / BlockField.
function allocate_output(::AbstractOperator, x::AbstractField)
    y = similar(x)
    zero_ghosts!(y)
    return y
end

# Input-shaped field for L (the output shape of its adjoint). Rank-changers override.
function allocate_input(::AbstractOperator, y::AbstractField)
    x = similar(y)
    zero_ghosts!(x)
    return x
end

#--------------------------------------------------------------------------------# Adjoint protocol

"""
    adjoint(L::AbstractOperator) -> AbstractOperator

Lazy adjoint of a *linear* operator. Throws an `ArgumentError` for nonlinear
operators — adjoints of nonlinear maps are not defined; use
[`linearize`](@ref) and take the adjoint of the resulting Jacobian operator.

Every leaf declares its adjoint explicitly (there is no self-adjoint default);
combinators propagate adjoints automatically.
"""
function Base.adjoint(L::AbstractOperator)
    if !islinear(L)
        throw(
            ArgumentError(
                "adjoint is only defined for linear operators, and islinear($(nameof(typeof(L)))) " *
                "is false; for nonlinear operators take the adjoint of linearize(L, u0) instead",
            ),
        )
    end
    return adjoint_operator(L)
end

# The declared adjoint of a linear operator, bypassing the islinear gate. Leaves
# override when a cheaper expression than the lazy wrapper exists (e.g. themselves,
# when exactly self-adjoint).
adjoint_operator(L::AbstractOperator) = AdjointOp(L)

"""
    apply_adjoint!(x̄::AbstractField, L::AbstractOperator, ȳ::AbstractField, g::AbstractGrid, α=true, β=false) -> x̄

Apply the adjoint of a linear operator in place: `x̄ = α·Lᵀ(ȳ) + β·x̄`. Ghost
layers of `ȳ` are treated as scratch (zeroed — only interior values are adjoint
inputs, matching the flat Krylov boundary).
"""
function apply_adjoint!(x̄::AbstractField, L::AbstractOperator, ȳ::AbstractField, g::AbstractGrid)
    return apply_adjoint!(x̄, L, ȳ, g, true, false)
end

function apply_adjoint!(::Field, L::AbstractOperator, ::Field, ::AbstractGrid, α, β)
    throw(ArgumentError("no adjoint action declared for $(nameof(typeof(L)))"))
end

"""
    AdjointOp(L)

Lazy adjoint wrapper: applying it calls [`apply_adjoint!`](@ref) of the wrapped
operator. Produced by `adjoint(L)` for leaves without a cheaper adjoint expression.
"""
struct AdjointOp{O<:AbstractOperator} <: AbstractOperator
    op::O
end

islinear(L::AdjointOp) = islinear(L.op)
isconstant(L::AdjointOp) = isconstant(L.op)
isselfadjoint(L::AdjointOp) = isselfadjoint(L.op)
isdiagonal(L::AdjointOp) = isdiagonal(L.op)
adjoint_operator(L::AdjointOp) = L.op
operator_grid(L::AdjointOp) = operator_grid(L.op)
allocate_output(L::AdjointOp, x::AbstractField) = allocate_input(L.op, x)
allocate_input(L::AdjointOp, y::AbstractField) = allocate_output(L.op, y)

function apply!(y::Field, L::AdjointOp, x::Field, g::AbstractGrid, α, β)
    return apply_adjoint!(y, L.op, x, g, α, β)
end
function apply_adjoint!(x̄::Field, L::AdjointOp, ȳ::Field, g::AbstractGrid, α, β)
    return apply!(x̄, L.op, ȳ, g, α, β)
end

Adapt.adapt_structure(to, L::AdjointOp) = AdjointOp(Adapt.adapt(to, L.op))

#--------------------------------------------------------------------------------# Generic adjoint gather engine

# Bounds-masked read: out-of-range neighbors contribute zero. Used by adjoint
# gathers sweeping all padded cells, where stencil offsets step outside the array.
@inline function _maskedget(u::AbstractArray{T,N}, J::CartesianIndex{N}) where {T,N}
    return checkbounds(Bool, u, J) ? @inbounds(u[J]) : zero(T)
end

# Unit CartesianIndex along dimension d.
@inline function _unitindex(::Val{N}, d::Int) where {N}
    return CartesianIndex(ntuple(i -> i == d ? 1 : 0, Val(N)))
end

# Mechanical exact transpose of a stencil leaf: with forward action
# y_int = S·(P·x) (P = homogeneous BC fill, S = stencil into the interior), the
# adjoint is x̄ = Pᵀ·Sᵀ·ȳ_int. `gather(ȳdata, J)` must compute the flipped-stencil
# sum Σ_o w_o·ȳ[J−o] (bounds-masked); ghosts of ȳ are zeroed so only interior
# values enter, and fold_bc! applies Pᵀ. Both branches are allocation-free.
function adjoint_gather!(x̄::Field, ȳ::Field, gather::F, α::Number, β::Number) where {F}
    zero_ghosts!(ȳ)
    if iszero(β)
        x̄.data .= gather.(Ref(ȳ.data), CartesianIndices(x̄.data))
        fold_bc!(x̄)
        isone(α) || (x̄.data .*= α)
    else
        # Accumulate in place. fold_bc! must see this call's ghost contribution
        # alone — folding the running total's would double-count — so the
        # foldable slabs are cleared first. Nothing is lost: fold_bc! zeroes
        # them on exit anyway and no consumer reads a physical-BC ghost (the
        # flat boundary takes interiors; halo_update_adjoint!/the distributed
        # reduction take Interface slabs). Interface slabs are deliberately NOT
        # cleared — they carry the neighbour-owned cotangents a sibling under
        # the same Added accumulates into, and they are why the blend spans the
        # whole padded array rather than just the interior.
        zero_bc_ghosts!(x̄)
        x̄.data .= α .* gather.(Ref(ȳ.data), CartesianIndices(x̄.data)) .+ β .* x̄.data
        fold_bc!(x̄)
    end
    return x̄
end

#--------------------------------------------------------------------------------# Size, eltype, show

# Grid an operator is bound to; `nothing` for grid-free operators (resolved from a
# composition sibling or the applied-to field).
operator_grid(::AbstractOperator) = nothing

"""
    size(L::AbstractOperator) -> (Int, Int)

Dimensions of `L` as a linear map over flat interior-DOF vectors, assuming a
scalar input field. Multi-component sizes are determined when the operator is
bound to a concrete field by [`prepare`](@ref).
"""
function Base.size(L::AbstractOperator)
    g = _require_grid(L)
    n = prod(local_size(g))
    return (n, n)
end

function Base.eltype(L::AbstractOperator)
    g = _require_grid(L)
    return eltype(spacing(g))
end

function _require_grid(L::AbstractOperator)
    g = operator_grid(L)
    g === nothing && throw(ArgumentError("operator $(nameof(typeof(L))) is not bound to a grid"))
    return g
end

Base.show(io::IO, L::AbstractOperator) = print(io, nameof(typeof(L)))
Base.show(io::IO, L::AdjointOp) = (print(io, "adjoint("); show(io, L.op); print(io, ")"))
