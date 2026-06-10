#--------------------------------------------------------------------------------# Laplacian

"""
    laplacian_stencil(u, I::CartesianIndex, inv_h2::NTuple) -> (uc, lap)

Per-cell second-order central Laplacian stencil: returns the center value `uc`
and the Laplacian `lap = Σ_d (u[I-δd] - 2uc + u[I+δd]) / Δd²`. This is the single
stencil body — the built-in [`Laplacian`](@ref) leaf calls it, and custom fused
operators (§4a of the design) must reuse it so the numerical definition never
forks. Ghost layers of `u` must be filled before calling.

### Examples

```julia
uc, lap = laplacian_stencil(u.data, CartesianIndex(2, 2), inv.(spacing(g) .^ 2))
```
"""
@inline function laplacian_stencil(
    u::AbstractArray{<:Any,N}, I::CartesianIndex{N}, inv_h2::NTuple{N}
) where {N}
    uc = @inbounds u[I]
    terms = ntuple(Val(N)) do d
        δ = _unitindex(Val(N), d)
        @inbounds (u[I - δ] - 2 * uc + u[I + δ]) * inv_h2[d]
    end
    return (uc, sum(terms))
end

@inline _lap_at(u, I, inv_h2) = laplacian_stencil(u, I, inv_h2)[2]

"""
    Laplacian(grid)

Matrix-free Laplacian (∇²) leaf bound to `grid`. Construct with
[`laplacian`](@ref).
"""
struct Laplacian{G<:AbstractGrid} <: AbstractOperator
    grid::G
end

"""
    laplacian(g::AbstractGrid) -> Laplacian

Build a matrix-free Laplacian operator (∇²) bound to `g`, discretized with
second-order central differences. Acts componentwise on any element type, so the
same operator serves scalar and `SVector`-valued fields. Exactly self-adjoint
under the homogeneous ghost fills of all built-in boundary conditions.

### Examples

```julia
g = CartesianGrid(((0.0, 2π),), (64,); bc=((Periodic(), Periodic()),))
u = set!(scalar_field(g), x -> sin(x[1]))
Δu = laplacian(g) * u            # ≈ -u
```

See also: [`gradient`](@ref), [`divergence`](@ref), [`laplacian_stencil`](@ref).
"""
laplacian(g::AbstractGrid) = Laplacian(g)

islinear(::Laplacian) = true
isconstant(::Laplacian) = true
isselfadjoint(::Laplacian) = true
operator_grid(L::Laplacian) = L.grid
adjoint_operator(L::Laplacian) = L

function apply!(y::Field, L::Laplacian, x::Field, g::AbstractGrid, α, β)
    halo_update!(x, g)
    apply_bc!(x)
    inv_h2 = _inv_spacing2(g)
    yi = interior(y)
    if iszero(β)
        yi .= α .* _lap_at.(Ref(x.data), interior(g), Ref(inv_h2))
    else
        yi .= α .* _lap_at.(Ref(x.data), interior(g), Ref(inv_h2)) .+ β .* yi
    end
    return y
end

function apply_adjoint!(x̄::Field, L::Laplacian, ȳ::Field, g::AbstractGrid, α, β)
    return apply!(x̄, L, ȳ, g, α, β)
end

Adapt.adapt_structure(to, L::Laplacian) = Laplacian(Adapt.adapt(to, L.grid))
