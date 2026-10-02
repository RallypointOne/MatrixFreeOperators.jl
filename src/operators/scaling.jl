#--------------------------------------------------------------------------------# Scaling (pointwise coefficient)

# For a general `func` we impose self-adjointness, since otherwise the adjoint is hard to
# implement.
"""
    ScalingOp(func, args)

Pointwise multiplication by a coefficient computed on the fly as
`func(args...)`, where each of `args` is a `Number` or a [`Field`](@ref) read
pointwise at the cell being written. The fields in `args` are differentiable
operator parameters (material coefficients for inverse problems). Construct with
[`scaling`](@ref).

Scaling by a single coefficient `κ` is the special case `func === identity`,
`args === (κ,)` — see [`CoeffScaling`](@ref). For any other `func`, `func` must
return a **real scalar**: the operator declares itself self-adjoint
unconditionally.
"""
struct ScalingOp{F,A} <: AbstractOperator
    func::F
    args::A
end

"""
    CoeffScaling{C}

Alias for a [`ScalingOp`](@ref) that multiplies pointwise by a single coefficient of
type `C` — a `Number` or a scalar-eltype [`Field`](@ref) `κ(x)`. The diagonal,
parameter-carrying leaf: its coefficient field is a differentiable operator parameter.
Construct with [`scaling`](@ref)`(κ)` or `CoeffScaling(κ)`.
"""
const CoeffScaling{C} = ScalingOp{typeof(identity),Tuple{C}}

CoeffScaling(κ) = ScalingOp(identity, (κ,))

# The coefficient of a single-coefficient scaling.
_coeff(S::CoeffScaling) = S.args[1]

"""
    scaling(κ) -> ScalingOp

Build a pointwise scaling operator `x ↦ κ ⊙ x` from a `Number` or a scalar-eltype
coefficient field ([`Field`](@ref) on a single grid, [`BlockField`](@ref)/
[`PackedBlockField`](@ref) on a forest — the coefficient's layout should match
the field the operator is applied to for the forest-native kernel to engage).
Diagonal and (for real coefficients) self-adjoint —
the natural Jacobi-smoother target.

!!! note "For ∇·(κ∇u), reach for `diffusion`"
    [`diffusion`](@ref)`(g, κ)` is the compact flux form: exactly symmetric,
    with an `operator_diagonal`, and ~3× faster.
    `divergence(g) * scaling(κ) * gradient(g)` also builds a valid ∇·(κ∇u) and
    is a good demonstration that the algebra composes, but it chains two
    centered differences: the resulting stencil reaches `u[I±2δ]` and never
    `u[I±δ]`, so the grid decouples into interleaved sublattices and κ at a cell
    never enters that cell's own equation — which makes a κ-inversion fit
    disjoint halves of the data.

### Examples

```julia
g = CartesianGrid(((0.0, 1.0),), (64,))
κ = set!(x -> 1 + x[1]^2, scalar_field(g))
H = laplacian(g) - scaling(κ)                    # Helmholtz-type: Δu − κu
```

See also: [`diffusion`](@ref), [`gradient`](@ref), [`divergence`](@ref),
[`identity_op`](@ref).
"""
scaling(κ::Number) = CoeffScaling(κ)
function scaling(κ::Field)
    eltype(κ.data) <: Number || throw(
        ArgumentError(
            "scaling coefficient must be a Number or a scalar-eltype Field, got eltype $(eltype(κ.data))",
        ),
    )
    return CoeffScaling(κ)
end
function scaling(κ::AbstractBlockField)
    eltype(κ) <: Number || throw(
        ArgumentError(
            "scaling coefficient must be a Number or a scalar-eltype field, got eltype $(eltype(κ))",
        ),
    )
    return CoeffScaling(κ)
end

isselfadjoint(S::CoeffScaling{<:Number}) = isreal(_coeff(S))
isselfadjoint(S::CoeffScaling{<:Field}) = eltype(_coeff(S).data) <: Real
# Sound on refined forests too: a diagonal operator has no cross-block coupling.
isselfadjoint(S::CoeffScaling{<:AbstractBlockField}) = eltype(_coeff(S)) <: Real

operator_grid(::CoeffScaling{<:Number}) = nothing
operator_grid(S::CoeffScaling{<:AbstractField}) = _coeff(S).grid

adjoint_operator(S::CoeffScaling{<:Real}) = S
adjoint_operator(S::CoeffScaling{<:Number}) = CoeffScaling(conj(_coeff(S)))
function adjoint_operator(S::CoeffScaling{<:Field})
    eltype(_coeff(S).data) <: Real && return S
    κ = _coeff(S)
    return CoeffScaling(Field(conj.(κ.data), κ.grid))
end
function adjoint_operator(S::CoeffScaling{<:AbstractBlockField})
    eltype(_coeff(S)) <: Real && return S
    κ = _coeff(S)
    c = similar(κ)
    for i in 1:nleaves(κ.grid)
        _block_array(c, i) .= conj.(_block_array(κ, i))
    end
    return CoeffScaling(c)
end

_coeff_values(c::Number) = c
_coeff_values(c::Field) = interior(c)

function apply!(y::Field, S::CoeffScaling, x::Field, ::AbstractGrid, α, β)
    κ = _coeff_values(_coeff(S))
    yi = interior(y)
    if iszero(β)
        yi .= α .* κ .* interior(x)
    else
        yi .= α .* κ .* interior(x) .+ β .* yi
    end
    return y
end

function apply_adjoint!(x̄::Field, S::CoeffScaling, ȳ::Field, g::AbstractGrid, α, β)
    return apply!(x̄, adjoint_operator(S), ȳ, g, α, β)
end

#--------------------------------------------------------------------------------# Scaling (function of fields)

"""
    scaling(f, args...) -> ScalingOp

Build a pointwise scaling operator whose coefficient is
evaluated from `args` — each a `Number` or a [`Field`](@ref) on the grid of the
field the operator is applied to — inside the action's broadcast, so no
coefficient field is materialized. Field arguments may have any element type
(e.g. a vector field passed to `norm`); `f` must return a real scalar. At least
one argument is required, and all field arguments must share a grid.

Diagonal and self-adjoint. On a [`BlockForest`](@ref), pass
[`BlockField`](@ref) arguments.

### Examples

```julia
g = CartesianGrid(((0.0, 1.0),), (64,))
κ = set!(x -> 1 + x[1]^2, scalar_field(g))
T = set!(x -> 300 + 10x[1], scalar_field(g))
S = scaling((κ, T) -> κ * exp(-1 / T), κ, T)     # Arrhenius-type coefficient
H = laplacian(g) - S
```

See also: [`ScalingOp`](@ref), [`CoeffScaling`](@ref).
"""
function scaling(𝒻::F, args...) where {F}
    isempty(args) && throw(
        ArgumentError(
            "scaling(f, args...) needs at least one argument; for a constant coefficient use scaling(κ::Number)",
        ),
    )
    foreach(_check_scaling_arg, args)
    grids = map(a -> a.grid, filter(a -> a isa AbstractField, args))
    (isempty(grids) || all(g -> _same_grid(g, first(grids)), grids)) || throw(
        ArgumentError("scaling function arguments must all be defined on the same grid"),
    )
    return ScalingOp(𝒻, args)
end

_check_scaling_arg(::Union{Number,Field,AbstractBlockField}) = nothing
_check_scaling_arg(a) = throw(
    ArgumentError("scaling function arguments must be Numbers or fields, got $(typeof(a))"),
)


islinear(::ScalingOp) = true
isconstant(::ScalingOp) = true
isdiagonal(::ScalingOp) = true
shares_exchange(::ScalingOp) = true   # pointwise — reads no ghosts
isselfadjoint(S::ScalingOp) = true

adjoint_operator(S::ScalingOp) = S


_maybe_interior(x::Number) = x  # not sure why you'd do this, but we support it
_maybe_interior(ϕ::Field) = interior(ϕ)

function apply!(y::Field, S::ScalingOp, x::Field, ::AbstractGrid, α, β)
    yi = interior(y)
    args = map(_maybe_interior, S.args)
    if iszero(β)
        yi .= α .* S.func.(args...) .* interior(x)
    else
        yi .= α .* S.func.(args...) .* interior(x) .+ β .* yi
    end
    return y
end

function apply_adjoint!(y::Field, S::ScalingOp, x::Field, g::AbstractGrid, α, β)
    return apply!(y, S, x, g, α, β)
end

Adapt.adapt_structure(to, S::ScalingOp) = ScalingOp(Adapt.adapt(to, S.func), Adapt.adapt(to, S.args))

# Pointwise: reads no ghosts, so the raw sweep is the ordinary action.
function _apply_raw!(y::Field, S::ScalingOp, x::Field, g::AbstractGrid, α, β)
    return apply!(y, S, x, g, α, β)
end

_arg_grid(::Any) = nothing
_arg_grid(a::AbstractField) = a.grid

# The first field argument's grid; `nothing` when every argument is a Number.
operator_grid(S::ScalingOp) = foldl((g, a) -> _first_grid(g, _arg_grid(a)), S.args; init=nothing)

# Unlike CoeffScaling's, this does not alias operator state: the coefficient is only
# ever evaluated inside the action's broadcast, so it is materialized here.
function operator_diagonal(S::ScalingOp)
    any(a -> a isa AbstractBlockField, S.args) && throw(
        ArgumentError("operator_diagonal of a ScalingOp with block-field arguments is not supported"),
    )
    g = operator_grid(S)
    g === nothing && return S.func(S.args...)
    c = S.func.(map(_maybe_interior, S.args)...)
    d = scalar_field(g, eltype(c))
    interior(d) .= c
    return d
end

# Per-leaf slicing for forests, as for CoeffScaling: block-field arguments become the
# leaf's block view; Numbers pass through.
_leaf_arg(a, i, lg) = a
_leaf_arg(a::AbstractBlockField, i, lg) = block(a, i, lg)
@inline _leaf_op(S::ScalingOp, i, lg) = ScalingOp(S.func, map(a -> _leaf_arg(a, i, lg), S.args))

#--------------------------------------------------------------------------------# Identity

"""
    IdentityOp()

Grid-free identity operator. Resolves its size and element type from a
composition sibling or the field it is applied to. Construct with
[`identity_op`](@ref).
"""
struct IdentityOp <: AbstractOperator end

"""
    identity_op() -> IdentityOp

Build the identity operator, e.g. for shifted systems `A - λ*identity_op()`.

### Examples

```julia
g = CartesianGrid(((0.0, 1.0),), (64,))
H = laplacian(g) - 4 * identity_op()             # Helmholtz-style shift
```
"""
identity_op() = IdentityOp()

islinear(::IdentityOp) = true
isconstant(::IdentityOp) = true
isselfadjoint(::IdentityOp) = true
isdiagonal(::IdentityOp) = true
shares_exchange(::IdentityOp) = true
adjoint_operator(L::IdentityOp) = L

function apply!(y::Field, ::IdentityOp, x::Field, ::AbstractGrid, α, β)
    yi = interior(y)
    if iszero(β)
        yi .= α .* interior(x)
    else
        yi .= α .* interior(x) .+ β .* yi
    end
    return y
end

function apply_adjoint!(x̄::Field, L::IdentityOp, ȳ::Field, g::AbstractGrid, α, β)
    return apply!(x̄, L, ȳ, g, α, β)
end

# Pointwise: reads no ghosts, so the raw sweep is the ordinary action.
function _apply_raw!(y::Field, L::IdentityOp, x::Field, g::AbstractGrid, α, β)
    return apply!(y, L, x, g, α, β)
end
