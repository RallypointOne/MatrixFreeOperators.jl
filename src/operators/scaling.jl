#--------------------------------------------------------------------------------# Scaling (pointwise coefficient)

"""
    ScalingOp(coeff)

Pointwise multiplication by a coefficient — a `Number` or a scalar-eltype
[`Field`](@ref) `κ(x)`. The diagonal, parameter-carrying leaf: its coefficient
field is a differentiable operator parameter (material coefficients for inverse
problems). Construct with [`scaling`](@ref).
"""
struct ScalingOp{F} <: AbstractOperator
    coeff::F
end

"""
    scaling(κ) -> ScalingOp

Build a pointwise scaling operator `x ↦ κ ⊙ x` from a `Number` or a scalar-eltype
coefficient field ([`Field`](@ref) on a single grid, [`BlockField`](@ref)/
[`PackedBlockField`](@ref) on a forest — the coefficient's layout should match
the field the operator is applied to for the forest-native kernel to engage).
Diagonal and (for real coefficients) self-adjoint —
the natural Jacobi-smoother target. Variable-coefficient diffusion composes as
`divergence(g) * scaling(κ) * gradient(g)`.

### Examples

```julia
g = CartesianGrid(((0.0, 1.0),), (64,))
κ = set!(scalar_field(g), x -> 1 + x[1]^2)
K = divergence(g) * scaling(κ) * gradient(g)     # ∇·(κ∇u)
```

See also: [`gradient`](@ref), [`divergence`](@ref), [`identity_op`](@ref).
"""
scaling(κ::Number) = ScalingOp(κ)
function scaling(κ::Field)
    eltype(κ.data) <: Number || throw(
        ArgumentError(
            "scaling coefficient must be a Number or a scalar-eltype Field, got eltype $(eltype(κ.data))",
        ),
    )
    return ScalingOp(κ)
end
function scaling(κ::AbstractBlockField)
    eltype(κ) <: Number || throw(
        ArgumentError(
            "scaling coefficient must be a Number or a scalar-eltype field, got eltype $(eltype(κ))",
        ),
    )
    return ScalingOp(κ)
end

islinear(::ScalingOp) = true
isconstant(::ScalingOp) = true
isdiagonal(::ScalingOp) = true
isselfadjoint(S::ScalingOp{<:Number}) = isreal(S.coeff)
isselfadjoint(S::ScalingOp{<:Field}) = eltype(S.coeff.data) <: Real
# Sound on refined forests too: a diagonal operator has no cross-block coupling.
isselfadjoint(S::ScalingOp{<:AbstractBlockField}) = eltype(S.coeff) <: Real

operator_grid(::ScalingOp{<:Number}) = nothing
operator_grid(S::ScalingOp{<:AbstractField}) = S.coeff.grid

adjoint_operator(S::ScalingOp{<:Real}) = S
adjoint_operator(S::ScalingOp{<:Number}) = ScalingOp(conj(S.coeff))
function adjoint_operator(S::ScalingOp{<:Field})
    eltype(S.coeff.data) <: Real && return S
    κ = S.coeff
    return ScalingOp(Field(conj.(κ.data), κ.grid))
end
function adjoint_operator(S::ScalingOp{<:AbstractBlockField})
    eltype(S.coeff) <: Real && return S
    κ = S.coeff
    c = similar(κ)
    for i in 1:nleaves(κ.grid)
        _block_array(c, i) .= conj.(_block_array(κ, i))
    end
    return ScalingOp(c)
end

_coeff_values(c::Number) = c
_coeff_values(c::Field) = interior(c)

function apply!(y::Field, S::ScalingOp, x::Field, ::AbstractGrid, α, β)
    κ = _coeff_values(S.coeff)
    yi = interior(y)
    if iszero(β)
        yi .= α .* κ .* interior(x)
    else
        yi .= α .* κ .* interior(x) .+ β .* yi
    end
    return y
end

function apply_adjoint!(x̄::Field, S::ScalingOp, ȳ::Field, g::AbstractGrid, α, β)
    return apply!(x̄, adjoint_operator(S), ȳ, g, α, β)
end

# Pointwise: reads no ghosts, so the raw sweep is the ordinary action.
function _apply_raw!(y::Field, S::ScalingOp, x::Field, g::AbstractGrid, α, β)
    return apply!(y, S, x, g, α, β)
end

Adapt.adapt_structure(to, S::ScalingOp) = ScalingOp(Adapt.adapt(to, S.coeff))

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
