#--------------------------------------------------------------------------------# Divergence

@inline function _div_at(
    u::AbstractArray{<:Any,N}, I::CartesianIndex{N}, inv_h::NTuple{N}
) where {N}
    terms = ntuple(Val(N)) do d
        δ = _unitindex(Val(N), d)
        @inbounds (u[I + δ][d] - u[I - δ][d]) * (inv_h[d] / 2)
    end
    return sum(terms)
end

# Adjoint gather: the scalar cotangent scatters into component d through the
# flipped centered stencil of dimension d.
@inline function _div_adjoint_gather(
    ȳ::AbstractArray{<:Any,N}, J::CartesianIndex{N}, inv_h::NTuple{N}
) where {N}
    return SVector(
        ntuple(Val(N)) do d
            δ = _unitindex(Val(N), d)
            (_maskedget(ȳ, J - δ) - _maskedget(ȳ, J + δ)) * (inv_h[d] / 2)
        end,
    )
end

"""
    Divergence(grid)

Matrix-free divergence leaf — a rank-changer mapping an `SVector{N}`-valued field
to a scalar field. Construct with [`divergence`](@ref).
"""
struct Divergence{G<:AbstractGrid} <: AbstractOperator
    grid::G
end

"""
    divergence(g::AbstractGrid) -> Divergence

Build a matrix-free divergence operator (∇⋅) bound to `g`, discretized with
second-order central differences. Maps an `SVector{N}`-valued vector field to a
scalar field — one of the two rank-changing leaves (with [`gradient`](@ref)).
Variable-coefficient diffusion can be assembled as
`divergence(g) * scaling(κ) * gradient(g)`, but prefer the compact flux-form leaf
[`diffusion`](@ref) — chaining two centered differences doubles the stencil width.

### Examples

```julia
g = CartesianGrid(((0.0, 2π), (0.0, 2π)), (32, 32);
                  bc=((Periodic(), Periodic()), (Periodic(), Periodic())))
v = set!(vector_field(g), x -> SVector(sin(x[1]), cos(x[2])))
divv = divergence(g) * v
```

See also: [`diffusion`](@ref), [`gradient`](@ref), [`scaling`](@ref).
"""
divergence(g::AbstractGrid) = Divergence(g)

islinear(::Divergence) = true
isconstant(::Divergence) = true
operator_grid(L::Divergence) = L.grid

function Base.size(L::Divergence)
    n = prod(local_size(L.grid))
    return (n, dimension(L.grid) * n)
end

function allocate_output(::Divergence, x::AbstractField)
    T = eltype(x)
    T <: SVector || throw(ArgumentError("divergence expects a vector field, got eltype $T"))
    y = similar(x, _scalar_eltype(T))
    zero_ghosts!(y)
    return y
end

function allocate_input(::Divergence, y::AbstractField)
    x = similar(y, SVector{dimension(y.grid),eltype(y)})
    zero_ghosts!(x)
    return x
end

function apply!(y::Field, L::Divergence, x::Field, g::AbstractGrid, α, β)
    halo_update!(x, g)
    apply_bc!(x)
    return _apply_raw!(y, L, x, g, α, β)
end

function _apply_raw!(y::Field, ::Divergence, x::Field, g::AbstractGrid, α, β)
    inv_h = _inv_spacing(g)
    yi = interior(y)
    if iszero(β)
        yi .= α .* _div_at.(Ref(x.data), interior(g), Ref(inv_h))
    else
        yi .= α .* _div_at.(Ref(x.data), interior(g), Ref(inv_h)) .+ β .* yi
    end
    return y
end

function apply_adjoint!(x̄::Field, L::Divergence, ȳ::Field, g::AbstractGrid, α, β)
    gather = let inv_h = _inv_spacing(g)
        (u, J) -> _div_adjoint_gather(u, J, inv_h)
    end
    return adjoint_gather!(x̄, ȳ, gather, α, β)
end

Adapt.adapt_structure(to, L::Divergence) = Divergence(Adapt.adapt(to, L.grid))
