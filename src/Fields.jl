#--------------------------------------------------------------------------------# Fields

"""
    Center

Location trait for cell-centered (collocated) fields — the only location in v1.
Staggered layouts add face locations as new trait types.
"""
struct Center end

"""
    AbstractField

Supertype for fields the operator algebra and solver boundary act on. [`Field`](@ref)
is the single-grid case; [`BlockField`](@ref) is the block-structured (forest) case.
The shared contract is `similar`/`zero_ghosts!` and the flat-vector boundary
(`flatten`/`flat_to_interior!`/`interior_to_flat!`/`flat_length`); operators are
otherwise written against single-grid `Field`s and reused per block.
"""
abstract type AbstractField end

"""
    Field(data, grid)
    Field{L}(data, grid)

Halo-padded field on `grid` with location trait `L` (default [`Center`](@ref)).
`data` must have size [`padded_size`](@ref)`(grid)`. The element type carries the
tensor rank: a scalar field stores numbers, a vector field stores `SVector`s — see
[`scalar_field`](@ref) and [`vector_field`](@ref).

### Examples

```julia
g = CartesianGrid(((0.0, 1.0),), (64,))
u = Field(zeros(padded_size(g)), g)
```
"""
struct Field{L,A<:AbstractArray,G<:AbstractGrid} <: AbstractField
    data::A
    grid::G

    function Field{L}(data::A, grid::G) where {L,A<:AbstractArray,G<:AbstractGrid}
        if size(data) != padded_size(grid)
            throw(
                DimensionMismatch(
                    "field data size $(size(data)) must equal padded grid size $(padded_size(grid))",
                ),
            )
        end
        return new{L,A,G}(data, grid)
    end
end
Field(data::AbstractArray, grid::AbstractGrid) = Field{Center}(data, grid)

"""
    scalar_field(g::AbstractGrid, T=eltype(spacing(g))) -> Field

Allocate a zeroed cell-centered scalar field on the grid's device.

### Examples

```julia
g = CartesianGrid(((0.0, 1.0),), (64,))
u = scalar_field(g)
v = scalar_field(g, Float32)
```

See also: [`vector_field`](@ref), [`set!`](@ref).
"""
function scalar_field(g::AbstractGrid, ::Type{T}=eltype(spacing(g))) where {T<:Number}
    data = KernelAbstractions.zeros(KernelAbstractions.get_backend(g), T, padded_size(g)...)
    return Field(data, g)
end

"""
    vector_field(g::AbstractGrid{N}, T=eltype(spacing(g))) -> Field

Allocate a zeroed cell-centered vector field with element type `SVector{N,T}`.
Operators written generically over the element type act componentwise on it.

### Examples

```julia
g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (32, 32))
v = vector_field(g)        # eltype SVector{2,Float64}
```

See also: [`scalar_field`](@ref), [`component`](@ref).
"""
function vector_field(g::AbstractGrid{N}, ::Type{T}=eltype(spacing(g))) where {N,T<:Number}
    data = KernelAbstractions.zeros(
        KernelAbstractions.get_backend(g), SVector{N,T}, padded_size(g)...
    )
    return Field(data, g)
end

#--------------------------------------------------------------------------------# Field interface

"""
    interior(f::Field) -> SubArray

View of the interior (owned, non-halo) cells of the field.
"""
interior(f::Field) = view(f.data, interior(f.grid))

"""
    set!(f::Field, fun) -> f

Set the interior of `f` to `fun(x)` evaluated at cell centers, where `x` is the
`SVector` of physical coordinates.

### Examples

```julia
g = CartesianGrid(((0.0, 2π),), (64,))
u = set!(scalar_field(g), x -> sin(x[1]))
```
"""
function set!(f::Field, fun::F) where {F}
    interior(f) .= fun.(cell_center.(Ref(f.grid), interior(f.grid)))
    return f
end

"""
    ncomponents(f::Field) -> Int

Number of components of the field's element type: 1 for scalar fields, `N` for
`SVector{N}`-valued fields.
"""
ncomponents(f::Field) = _ncomponents(eltype(f.data))
_ncomponents(::Type{<:Number}) = 1
_ncomponents(::Type{SVector{M,T}}) where {M,T} = M

_scalar_eltype(::Type{T}) where {T<:Number} = T
_scalar_eltype(::Type{SVector{M,T}}) where {M,T} = T

"""
    component(f::Field, d::Integer) -> Field

Extract component `d` of a vector field as a new (allocated) scalar field.
"""
function component(f::Field{L}, d::Integer) where {L}
    1 <= d <= ncomponents(f) ||
        throw(ArgumentError("component $d out of range for $(ncomponents(f)) components"))
    return Field{L}(getindex.(f.data, d), f.grid)
end
component(f::Field{L,<:AbstractArray{<:Number}}, d::Integer) where {L} =
    d == 1 ? Field{L}(copy(f.data), f.grid) : throw(ArgumentError("scalar field has only component 1"))

Base.eltype(f::Field) = eltype(f.data)
Base.similar(f::Field{L}) where {L} = Field{L}(similar(f.data), f.grid)
Base.similar(f::Field{L}, ::Type{E}) where {L,E} = Field{L}(similar(f.data, E), f.grid)
Base.copy(f::Field{L}) where {L} = Field{L}(copy(f.data), f.grid)

apply_bc!(f::Field) = (apply_bc!(f.data, f.grid); f)
fold_bc!(f::Field) = (fold_bc!(f.data, f.grid); f)
zero_ghosts!(f::Field) = (zero_ghosts!(f.data, f.grid); f)

function Adapt.adapt_structure(to, f::Field{L}) where {L}
    return Field{L}(Adapt.adapt(to, f.data), Adapt.adapt(to, f.grid))
end

#--------------------------------------------------------------------------------# Flat-vector boundary (interior DOFs only)

# Reshape a flat interior vector of scalars into the field's element type without
# copying: identity for scalar eltypes, reinterpret for SVector eltypes.
_as_eltype(::Type{E}, v::AbstractVector{E}, dims) where {E} = reshape(v, dims)
function _as_eltype(::Type{SVector{M,T}}, v::AbstractVector{T}, dims) where {M,T}
    return reshape(reinterpret(SVector{M,T}, v), dims)
end

"""
    flatten(f::Field) -> AbstractVector

Copy the interior of `f` into a flat vector of scalars — the Krylov-facing
representation. Spans interior DOFs only (ghost cells are never solver unknowns);
`SVector` elements are flattened component-fastest.

See also: [`flat_to_interior!`](@ref), [`interior_to_flat!`](@ref).
"""
function flatten(f::Field)
    data_int = f.data[interior(f.grid)]
    return _flat_vector(vec(data_int))
end
_flat_vector(v::AbstractVector{<:Number}) = v
_flat_vector(v::AbstractVector{SVector{M,T}}) where {M,T} = copy(vec(reinterpret(T, v)))

"""
    flat_length(f::AbstractField) -> Int

Number of interior scalar DOFs in the flat (Krylov) representation of `f`:
`prod(local_size) * ncomponents`, summed over blocks for a block field.
"""
flat_length(f::Field) = prod(local_size(f.grid)) * ncomponents(f)

"""
    flat_to_interior!(f::Field, v::AbstractVector) -> f

Copy the flat interior vector `v` (as produced by [`flatten`](@ref)) into the
interior of `f`. Ghost cells are untouched.
"""
function flat_to_interior!(f::Field, v::AbstractVector)
    interior(f) .= _as_eltype(eltype(f.data), v, local_size(f.grid))
    return f
end

"""
    interior_to_flat!(v::AbstractVector, f::Field, α=true, β=false) -> v

Fused axpby copy-out of the field interior into the flat vector:
`v = α * interior(f) + β * v` in one broadcast.
"""
function interior_to_flat!(v::AbstractVector, f::Field, α::Number=true, β::Number=false)
    vi = _as_eltype(eltype(f.data), v, local_size(f.grid))
    if iszero(β)
        vi .= α .* interior(f)
    else
        vi .= α .* interior(f) .+ β .* vi
    end
    return v
end
