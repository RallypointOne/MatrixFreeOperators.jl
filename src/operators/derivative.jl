#--------------------------------------------------------------------------------# Derivative

"""
    derivative_stencil(u, I::CartesianIndex, dim, order, inv_h) -> (uc, du)

Per-cell second-order central derivative stencil along dimension `dim`: returns
the center value `uc` and, for `order == 1`, `du = (u[I+δ] - u[I-δ]) / 2Δ` or, for
`order == 2`, `du = (u[I-δ] - 2uc + u[I+δ]) / Δ²`, where `inv_h = 1/Δ`. The single
stencil body shared by the built-in [`Derivative`](@ref) leaf and custom fused
operators. Ghost layers of `u` must be filled before calling.

### Examples

```julia
uc, du = derivative_stencil(u.data, CartesianIndex(3, 3), 1, 1, inv(spacing(g)[1]))
```
"""
@inline function derivative_stencil(
    u::AbstractArray{<:Any,N}, I::CartesianIndex{N}, dim::Int, order::Int, inv_h
) where {N}
    uc = @inbounds u[I]
    δ = _unitindex(Val(N), dim)
    du = if order == 1
        @inbounds (u[I + δ] - u[I - δ]) * (inv_h / 2)
    else
        @inbounds (u[I - δ] - 2 * uc + u[I + δ]) * inv_h^2
    end
    return (uc, du)
end

@inline _deriv_at(u, I, dim, order, inv_h) = derivative_stencil(u, I, dim, order, inv_h)[2]

# Adjoint gathers: flipped stencil weights, bounds-masked. Order 1 is
# antisymmetric (sign flip); order 2 is symmetric (its own flip).
@inline function _deriv_adjoint_gather(
    ȳ::AbstractArray{<:Any,N}, J::CartesianIndex{N}, dim::Int, order::Int, inv_h
) where {N}
    δ = _unitindex(Val(N), dim)
    if order == 1
        return (_maskedget(ȳ, J - δ) - _maskedget(ȳ, J + δ)) * (inv_h / 2)
    else
        return (_maskedget(ȳ, J - δ) - 2 * _maskedget(ȳ, J) + _maskedget(ȳ, J + δ)) *
               inv_h^2
    end
end

"""
    Derivative(grid, dim, order)

Matrix-free partial-derivative leaf `∂^order/∂x_dim^order`. Construct with
[`derivative`](@ref).
"""
struct Derivative{G<:AbstractGrid} <: AbstractOperator
    grid::G
    dim::Int
    order::Int
end

"""
    derivative(g::AbstractGrid, dim::Integer; order=1) -> Derivative

Build a matrix-free partial derivative `∂/∂x_dim` (or `∂²/∂x_dim²` with
`order=2`) bound to `g`, discretized with second-order central differences. Acts
componentwise on any element type.

### Examples

```julia
g = CartesianGrid(((0.0, 2π),), (64,); bc=((Periodic(), Periodic()),))
u = set!(scalar_field(g), x -> sin(x[1]))
∂u = derivative(g, 1) * u        # ≈ cos
```

See also: [`laplacian`](@ref), [`gradient`](@ref), [`derivative_stencil`](@ref).
"""
function derivative(g::AbstractGrid, dim::Integer; order::Integer=1)
    1 <= dim <= dimension(g) ||
        throw(ArgumentError("dimension $dim out of range for a $(dimension(g))-D grid"))
    order in (1, 2) || throw(ArgumentError("derivative order must be 1 or 2, got $order"))
    return Derivative(g, Int(dim), Int(order))
end

islinear(::Derivative) = true
isconstant(::Derivative) = true
isselfadjoint(L::Derivative) = L.order == 2 && _selfadjoint_grid(L.grid)
shares_exchange(::Derivative) = true
operator_grid(L::Derivative) = L.grid
adjoint_operator(L::Derivative) = isselfadjoint(L) ? L : AdjointOp(L)

function apply!(y::Field, L::Derivative, x::Field, g::AbstractGrid, α, β)
    halo_update!(x, g)
    apply_bc!(x)
    return _apply_raw!(y, L, x, g, α, β)
end

function _apply_raw!(y::Field, L::Derivative, x::Field, g::AbstractGrid, α, β)
    inv_h = _inv_spacing(g)[L.dim]
    yi = interior(y)
    if iszero(β)
        yi .= α .* _deriv_at.(Ref(x.data), interior(g), L.dim, L.order, inv_h)
    else
        yi .= α .* _deriv_at.(Ref(x.data), interior(g), L.dim, L.order, inv_h) .+ β .* yi
    end
    return y
end

function apply_adjoint!(x̄::Field, L::Derivative, ȳ::Field, g::AbstractGrid, α, β)
    # order 2 is self-adjoint only on an all-physical-BC grid — a forest leaf's
    # Interface ghosts need the mechanical transpose (see the Laplacian adjoint).
    L.order == 2 && !_has_interface(g) && return apply!(x̄, L, ȳ, g, α, β)
    inv_h = _inv_spacing(g)[L.dim]
    gather = let dim = L.dim, order = L.order, inv_h = inv_h
        (u, J) -> _deriv_adjoint_gather(u, J, dim, order, inv_h)
    end
    return adjoint_gather!(x̄, ȳ, gather, α, β)
end

function Adapt.adapt_structure(to, L::Derivative)
    return Derivative(Adapt.adapt(to, L.grid), L.dim, L.order)
end
