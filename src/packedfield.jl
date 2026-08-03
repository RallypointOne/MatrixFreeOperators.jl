#--------------------------------------------------------------------------------# PackedBlockField (packed contiguous storage)

"""
    PackedBlockField{L}(data, levels, grid)

Packed twin of [`BlockField`](@ref): every leaf's halo-padded block stored in one
contiguous `(blocksize .+ 2halo ..., nleaves)` array (`data`, leaves in Morton
order along the trailing dimension), with the per-leaf refinement levels in a
device-resident vector (`levels`) — the geometry SoA forest-native kernels read;
spacing derives from the forest's root spacing and the level. Behind the shared
`AbstractBlockField` interface every operator runs on packed storage via the
per-leaf fallback sweep (each leaf is a trailing-dim view), and operators with a
forest-native kernel sweep all leaves in a single launch. Convert with
[`pack`](@ref) / [`unpack`](@ref); `BlockField` remains the reference layout for
AD, Reactant, and [`regrid!`](@ref) (re-`pack` after a regrid).

See also: [`block`](@ref), [`flatten`](@ref).
"""
struct PackedBlockField{
    L,A<:AbstractArray,V<:AbstractVector{Int},G<:BlockForest
} <: AbstractBlockField
    data::A
    levels::V
    grid::G
    generation::Int
end
PackedBlockField{L}(data::AbstractArray, levels::AbstractVector{Int}, grid::BlockForest) where {L} =
    PackedBlockField{L,typeof(data),typeof(levels),typeof(grid)}(
        data, levels, grid, grid.forest.generation[]
    )

# The i-th leaf of a packed parent array as an N-dim view — shared by the field
# accessors and the forest-native kernel bodies.
@inline function _leaf_slice(data::AbstractArray{<:Any,M}, leaf::Integer) where {M}
    return view(data, ntuple(_ -> Colon(), Val(M - 1))..., leaf)
end

@inline _leaf_array(store, ::PackedLayout, i::Integer) = _leaf_slice(store, i)
@inline _leaf_view(store, ::PackedLayout, i::Integer, ranges) = view(store, ranges..., i)

_storage(f::PackedBlockField) = f.data
_layout(::PackedBlockField) = PackedLayout()
_block_array(f::PackedBlockField, i::Integer) = _leaf_array(f.data, PackedLayout(), i)
_block_view(f::PackedBlockField, i::Integer, ranges) = _leaf_view(f.data, PackedLayout(), i, ranges)
_flat_similar(f::PackedBlockField, ::Type{T}, len::Int) where {T} = similar(f.data, T, len)

function block(f::PackedBlockField{L}, i::Integer, leaf_grid) where {L}
    _require_current(f)
    return Field{L}(_block_array(f, i), leaf_grid)
end

Base.eltype(::PackedBlockField{L,A}) where {L,A} = eltype(A)

function component(f::PackedBlockField{L}, d::Integer) where {L}
    1 <= d <= ncomponents(f) ||
        throw(ArgumentError("component $d out of range for $(ncomponents(f)) components"))
    data = getindex.(f.data, d)
    return PackedBlockField{L,typeof(data),typeof(f.levels),typeof(f.grid)}(
        data, f.levels, f.grid, f.generation
    )
end

# Derived fields inherit the source's generation (same rule as BlockField).
Base.similar(f::PackedBlockField{L,A,V,G}) where {L,A,V,G} =
    PackedBlockField{L,A,V,G}(similar(f.data), f.levels, f.grid, f.generation)
function Base.similar(f::PackedBlockField{L,A,V,G}, ::Type{E}) where {L,A,V,G,E}
    data = similar(f.data, E)
    return PackedBlockField{L,typeof(data),V,G}(data, f.levels, f.grid, f.generation)
end
Base.copy(f::PackedBlockField{L,A,V,G}) where {L,A,V,G} =
    PackedBlockField{L,A,V,G}(copy(f.data), f.levels, f.grid, f.generation)

_zero_all!(f::PackedBlockField) = (fill!(f.data, zero(eltype(f))); f)

# Whole-forest interior view: the flat vector's layout (leaf-major contiguous
# block ranges, column-major interior, component-fastest) is exactly a reshape
# against this view, so the flat boundary is ONE broadcast instead of nleaves
# per-leaf copies.
_interior_view(f::PackedBlockField) =
    _interior_view_data(f.data, f.grid.halo, f.grid.blocksize)
function _interior_view_data(
    data::AbstractArray{<:Any,M}, h::NTuple{N,Int}, n::NTuple{N,Int}
) where {M,N}
    rs = ntuple(d -> (h[d] + 1):(h[d] + n[d]), Val(N))
    return view(data, rs..., Colon())
end

# GPU-gated single-broadcast flat transfers (CPU keeps the per-leaf loops, the
# same execution-mode-per-backend rule as the kernel sweeps). Exact axpby
# contract of the per-leaf primitive, including β = 0 never reading v.
function flat_to_interior!(f::PackedBlockField, v::AbstractVector)
    KernelAbstractions.get_backend(f.data) isa KernelAbstractions.GPU ||
        return _flat_to_interior_leaves!(f, v)
    _require_current(f)
    dims = (f.grid.blocksize..., nleaves(f.grid))
    _interior_view(f) .= _as_eltype(eltype(f), v, dims)
    return f
end

function interior_to_flat!(
    v::AbstractVector, f::PackedBlockField, α::Number=true, β::Number=false
)
    KernelAbstractions.get_backend(f.data) isa KernelAbstractions.GPU ||
        return _interior_to_flat_leaves!(v, f, α, β)
    _require_current(f)
    dims = (f.grid.blocksize..., nleaves(f.grid))
    vi = _as_eltype(eltype(f), v, dims)
    xi = _interior_view(f)
    if iszero(β)
        vi .= α .* xi
    else
        vi .= α .* xi .+ β .* vi
    end
    return v
end

function Adapt.adapt_structure(to, f::PackedBlockField{L}) where {L}
    data = Adapt.adapt(to, f.data)
    levels = Adapt.adapt(to, f.levels)
    grid = Adapt.adapt(to, f.grid)
    return PackedBlockField{L,typeof(data),typeof(levels),typeof(grid)}(
        data, levels, grid, f.generation
    )
end

# Per-leaf refinement levels on the forest's backend, in Morton (storage) order.
function _leaf_levels(bf::BlockForest)
    host = [key.level for key in bf.forest.leaves]
    backend = KernelAbstractions.get_backend(bf)
    levels = KernelAbstractions.allocate(backend, Int, length(host))
    copyto!(levels, host)
    return levels
end

"""
    pack(f::BlockField) -> PackedBlockField

Copy `f` into packed contiguous storage: one `(blocksize .+ 2halo ..., nleaves)`
array on the same device, leaves in the same Morton order. The packed field is
what the forest-native kernel sweeps consume; `prepare` on a packed prototype
yields packed scratch, so the prepared `mul!` runs the single-launch path.
Coefficient fields ([`scaling`](@ref), [`advection`](@ref)) follow the same
layout rule: their kernels engage when the coefficient is packed too — `prepare`
packs `BlockField` coefficients under a packed prototype automatically, and any
remaining layout mismatch degrades to the per-leaf reference sweep.

### Examples

```julia
u  = set!(scalar_field(bf), x -> sin(π * x[1]))
P  = prepare(laplacian(bf), pack(u))
```

See also: [`unpack`](@ref), [`PackedBlockField`](@ref).
"""
function pack(f::BlockField{L}) where {L}
    _require_current(f)
    bf = f.grid
    psize = bf.blocksize .+ 2 .* bf.halo
    data = similar(first(f.blocks), eltype(f), (psize..., nleaves(bf)))
    p = PackedBlockField{L}(data, _leaf_levels(bf), bf)
    for i in 1:nleaves(bf)
        _block_array(p, i) .= f.blocks[i]
    end
    return p
end

"""
    unpack(f::PackedBlockField) -> BlockField

Copy packed storage back into the vector-of-blocks reference layout — the
[`BlockField`](@ref) that AD, Reactant, and [`regrid!`](@ref) consume.

See also: [`pack`](@ref).
"""
function unpack(f::PackedBlockField{L}) where {L}
    _require_current(f)
    bf = f.grid
    psize = bf.blocksize .+ 2 .* bf.halo
    blocks = [similar(f.data, eltype(f), psize) for _ in 1:nleaves(bf)]
    for i in 1:nleaves(bf)
        blocks[i] .= _block_array(f, i)
    end
    return BlockField{L}(blocks, bf)
end
