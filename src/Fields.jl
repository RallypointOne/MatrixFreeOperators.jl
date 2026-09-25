#--------------------------------------------------------------------------------# Fields

"""
    Center

Location trait for cell-centered (collocated) fields — the only location in v1.
Staggered layouts add face locations as new trait types.
"""
struct Center end

#--------------------------------------------------------------------------------# Regrid-transfer policies

"""
    Interpolated()

Regrid-transfer policy trait (the default): refined leaves are filled by the
linear-exact per-dimension interpolation of the old parent. Second-order and
exact on linears, but **not** mean-preserving — the two children of an interior
parent cell use side-biased slopes, leaving the child mean off by `⅛·δ²u` per
dimension. The right transfer for smooth, non-conserved fields (geometric
indicators, coefficients). Attach a policy at construction
([`scalar_field`](@ref)/[`vector_field`](@ref)'s `transfer` keyword) or with
[`with_transfer`](@ref); [`regrid!`](@ref) resolves it per field at transfer
time, so nothing on an operator hot path ever consults it.

See also: [`Conservative`](@ref), [`SlopeLimited`](@ref).
"""
struct Interpolated end

"""
    Conservative()

Mean-preserving regrid-transfer policy: refined leaves are filled by the
cell-conservative linear reconstruction `u_child = u_parent + Σ_d ξ_d·σ_d`
(`ξ_d = ∓¼`), one shared slope per parent cell per dimension — centered in the
block interior, one-sided difference at parent-block edges. The volume-weighted
mean of the `2ᴺ` children equals the parent exactly for *any* slope, including
at block and physical boundaries, so `Σ V·u` is preserved to roundoff across
[`regrid!`](@ref) (coarsening already is: the `2⁻ᴺ` child mean). Exact on
linears and second-order like [`Interpolated`](@ref). The policy for conserved
state on smooth solutions.

See also: [`SlopeLimited`](@ref), [`with_transfer`](@ref).
"""
struct Conservative end

"""
    SlopeLimited()

Mean-preserving *and* bounds-preserving regrid-transfer policy: the same
cell-conservative reconstruction as [`Conservative`](@ref) with the per-dim
slope minmod-limited, and dropped to zero at parent-block edges (the transfer
is interior-only, so no second slope exists there to limit against). Children
never leave the hull of the parent's neighborhood — no new extrema across a
regrid — at the price of first-order transfer at extrema and block edges. Note
the edge cost scales with the block: every parent cell on a block face is
injected, so a `4×4` block reconstructs only its inner `2×2` and larger blocks
shrink that fraction. The policy for conserved state with steep fronts or
discontinuities.

See also: [`Conservative`](@ref), [`with_transfer`](@ref).
"""
struct SlopeLimited end

const RegridTransferPolicy = Union{Interpolated,Conservative,SlopeLimited}

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

getgrid(ϕ::AbstractField) = ϕ.grid

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
    set!(fun, ϕ::AbstractField) -> ϕ

Set the interior of `ϕ` to `fun(x)` evaluated at cell centers, where `x` is the
`SVector` of physical coordinates. Ghost cells are left untouched. On a forest
field the sweep runs block by block, so `ϕ` must be current with its forest
(see [`regrid!`](@ref)).

`fun` runs on the field's device, so it must be device-compatible: plain
arithmetic on the coordinate `SVector`, with no captured host arrays. The
function comes first so the `do`-block form reads naturally.

### Examples

```julia
g = CartesianGrid(((0.0, 2π),), (64,))
u = set!(x -> sin(x[1]), scalar_field(g))
v = set!(scalar_field(g)) do x
    exp(-x[1]^2)
end
```

See also: [`op!`](@ref), [`cell_center`](@ref).
"""
function set!(f::F, ϕ::Field) where {F}
    g = getgrid(ϕ)
    AK.map!(interior(ϕ), interior(g), AK.get_backend(g)) do idx
        x = cell_center(g, idx)
        f(x)
    end
    return ϕ
end

"""
    op!(fun, ϕ::AbstractField, ϕs::AbstractField...; check=true) -> ϕ

Pointwise update in place: set every interior cell of `ϕ` to
`fun(x, ϕ[I], ϕs[1][I], ϕs[2][I], …)`, where `x` is the `SVector` of the cell's
physical coordinates and the remaining arguments are the current values of `ϕ`
and of each field in `ϕs` at that cell. Ghost cells are left untouched. Like
[`set!`](@ref), `fun` runs on the device and the function comes first so the
`do`-block form reads naturally.

With `check=true` (the default) every field is first verified to live on the
same grid as `ϕ` via [`check_compatible`](@ref); pass `check=false` only from a
caller that has already checked. On forest fields the check runs once on the
whole forest and the sweep then runs block by block.

### Examples

```julia
g = CartesianGrid(((0.0, 1.0),), (64,))
u = set!(x -> sin(x[1]), scalar_field(g))
v = set!(x -> cos(x[1]), scalar_field(g))
op!((x, a, b) -> a + x[1] * b, u, v)     # u ← u + x⋅v, cell by cell
```

See also: [`set!`](@ref), [`compatible`](@ref).
"""
function op!(f::F, ϕ::Field, ϕs::Field...; check::Bool=true) where {F}
    check && check_compatible(ϕ, ϕs...)
    g = getgrid(ϕ)
    idxs = CartesianIndices(g)
    ϕint = interior(ϕ)
    ϕsint = map(interior, ϕs)
    AK.foreachindex(ϕint, AK.get_backend(g)) do j
        idx = idxs[j]
        x = cell_center(g, idx)
        ϕint[j] = f(x, ϕint[idx], map(φ -> φ[idx], ϕsint)...)
    end
    return ϕ
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
zero_bc_ghosts!(f::Field) = (zero_bc_ghosts!(f.data, f.grid); f)

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
    flat_to_interior!(f::Field, v::AbstractVector, α=true, β=false) -> f

Fused axpby copy-in of the flat interior vector `v` (as produced by
[`flatten`](@ref)) into the interior of `f`: `interior(f) = α * v + β * interior(f)`
in one broadcast. Ghost cells are untouched — the mirror of
[`interior_to_flat!`](@ref), and the reason the distributed path can stage
exchanged ghosts before a local apply.
"""
function flat_to_interior!(f::Field, v::AbstractVector, α::Number=true, β::Number=false)
    vi = _as_eltype(eltype(f.data), v, local_size(f.grid))
    fi = interior(f)
    if iszero(β)
        fi .= α .* vi
    else
        fi .= α .* vi .+ β .* fi
    end
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

block(ϕ::Field, ::Integer, lg=ϕ.grid) = ϕ
_block_array(ϕ::Field, ::Integer) = ϕ.data
_require_current(::Field) = nothing

# A `Field` is exactly as compatible as its grid.
@inline _field_mismatch(a::Field, b::Field) = _grid_mismatch(a.grid, b.grid)
@inline _field_layout_mismatch(a::Field, b::Field) = _layout_mismatch(a.grid, b.grid)

# `Field` vs. forest field, or two forest fields of different storage layout
# (`BlockField` vs. `PackedBlockField`): the per-block sweep can still pair them
# through `_block_array`, so layout is *not* refused on the field type — only the
# grid decides.  (Change these to `:type` if a function needs identical storage.)
_field_mismatch(::AbstractField, ::AbstractField) = :type
_field_layout_mismatch(::AbstractField, ::AbstractField) = :type

"""
    compatible(a::AbstractField, b::AbstractField...) -> Bool

Whether every field lives on a grid `==` to `a`'s grid. Forest fields must
also be current — allocated on the forest's present leaf set (see
[`regrid!`](@ref)); a stale field is never compatible with anything.

### Examples

```julia
g = CartesianGrid(((0.0, 1.0),), (64,))
u = scalar_field(g); v = scalar_field(g)
compatible(u, v)                       # true — same grid object, one `===`
compatible(u, scalar_field(coarsen(g)))  # false
```

See also: [`check_compatible`](@ref), [`same_layout`](@ref).
"""
@inline compatible(::AbstractField) = true
@inline compatible(a::AbstractField, b::AbstractField, rest::AbstractField...) =
    isnothing(_field_mismatch(a, b)) && compatible(a, rest...)

"""
    check_compatible(a::AbstractField, b::AbstractField...) -> nothing
    check_compatible(a::AbstractGrid, b::AbstractGrid...) -> nothing

Throw an `ArgumentError` naming the first property on which any argument differs
from `a` (grid `==`, plus regrid currency for forest fields).
Return `nothing` otherwise. The guard for a multi-field function:

```julia
function fma!(y::AbstractField, α, x::AbstractField, z::AbstractField)
    check_compatible(y, x, z)
    ...
end
```

The identity fast path makes this free when all fields share one grid object;
the throw is out of line, so the check inlines into the caller.

See also: [`check_layout`](@ref), [`compatible`](@ref).
"""
# Recursion over the argument tuple rather than `foreach` with a closure: the
# closure form allocates ~1 KB per call from three fields up under
# `--check-bounds=yes`; the recursive form measures 0 B for Field and BlockField.
@inline check_compatible(::AbstractField) = nothing
@inline function check_compatible(a::AbstractField, b::AbstractField, rest::AbstractField...)
    _check_pair(_field_mismatch(a, b), a, b, "grid")
    return check_compatible(a, rest...)
end

"""
    check_layout(a::AbstractField, b::AbstractField...) -> nothing
    check_layout(a::AbstractGrid, b::AbstractGrid...) -> nothing

The [`same_layout`](@ref) counterpart of [`check_compatible`](@ref): throw an
`ArgumentError` unless every argument has `a`'s padded storage shape (and, for
forest fields, is current). For pointwise kernels that never read spacing or
boundary conditions.
"""
@inline check_layout(::AbstractField) = nothing
@inline function check_layout(a::AbstractField, b::AbstractField, rest::AbstractField...)
    _check_pair(_field_layout_mismatch(a, b), a, b, "layout")
    return check_layout(a, rest...)
end

nleaves(ϕ::Field) = nleaves(getgrid(ϕ))
