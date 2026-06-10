#--------------------------------------------------------------------------------# Gradient

@inline function _grad_at(
    u::AbstractArray{<:Any,N}, I::CartesianIndex{N}, inv_h::NTuple{N}
) where {N}
    return SVector(
        ntuple(Val(N)) do d
            δ = _unitindex(Val(N), d)
            @inbounds (u[I + δ] - u[I - δ]) * (inv_h[d] / 2)
        end,
    )
end

# Adjoint gather: component d of the cotangent enters through the flipped
# centered stencil of dimension d, summed over dimensions.
@inline function _grad_adjoint_gather(
    ȳ::AbstractArray{<:Any,N}, J::CartesianIndex{N}, inv_h::NTuple{N}
) where {N}
    terms = ntuple(Val(N)) do d
        δ = _unitindex(Val(N), d)
        (_maskedget(ȳ, J - δ)[d] - _maskedget(ȳ, J + δ)[d]) * (inv_h[d] / 2)
    end
    return sum(terms)
end

"""
    Gradient(grid)

Matrix-free gradient leaf — a rank-changer mapping a scalar field to an
`SVector{N}`-valued field. Construct with [`gradient`](@ref).
"""
struct Gradient{G<:AbstractGrid} <: AbstractOperator
    grid::G
end

"""
    gradient(g::AbstractGrid) -> Gradient

Build a matrix-free gradient operator (∇) bound to `g`, discretized with
second-order central differences. Maps a scalar field to an `SVector{N}`-valued
vector field — one of the two rank-changing leaves (with [`divergence`](@ref)).

!!! note
    `gradient` is also exported by Enzyme; qualify as
    `MatrixFreeOperators.gradient` when both are loaded.

### Examples

```julia
g = CartesianGrid(((0.0, 2π), (0.0, 2π)), (32, 32);
                  bc=((Periodic(), Periodic()), (Periodic(), Periodic())))
u = set!(scalar_field(g), x -> sin(x[1]) * sin(x[2]))
∇u = gradient(g) * u             # SVector{2}-valued Field
```

See also: [`divergence`](@ref), [`laplacian`](@ref).
"""
gradient(g::AbstractGrid) = Gradient(g)

islinear(::Gradient) = true
isconstant(::Gradient) = true
operator_grid(L::Gradient) = L.grid

function Base.size(L::Gradient)
    n = prod(local_size(L.grid))
    return (dimension(L.grid) * n, n)
end

function allocate_output(::Gradient, x::Field{Loc}) where {Loc}
    T = eltype(x.data)
    T <: Number || throw(ArgumentError("gradient expects a scalar field, got eltype $T"))
    N = dimension(x.grid)
    y = Field{Loc}(similar(x.data, SVector{N,T}), x.grid)
    zero_ghosts!(y)
    return y
end

function allocate_input(::Gradient, y::Field{Loc}) where {Loc}
    x = Field{Loc}(similar(y.data, _scalar_eltype(eltype(y.data))), y.grid)
    zero_ghosts!(x)
    return x
end

function apply!(y::Field, L::Gradient, x::Field, g::AbstractGrid, α, β)
    halo_update!(x, g)
    apply_bc!(x)
    inv_h = _inv_spacing(g)
    yi = interior(y)
    if iszero(β)
        yi .= α .* _grad_at.(Ref(x.data), interior(g), Ref(inv_h))
    else
        yi .= α .* _grad_at.(Ref(x.data), interior(g), Ref(inv_h)) .+ β .* yi
    end
    return y
end

function apply_adjoint!(x̄::Field, L::Gradient, ȳ::Field, g::AbstractGrid, α, β)
    gather = let inv_h = _inv_spacing(g)
        (u, J) -> _grad_adjoint_gather(u, J, inv_h)
    end
    return adjoint_gather!(x̄, ȳ, gather, α, β)
end

Adapt.adapt_structure(to, L::Gradient) = Gradient(Adapt.adapt(to, L.grid))
