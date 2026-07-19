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

_block_array(f::PackedBlockField, i::Integer) = _leaf_slice(f.data, i)
_block_view(f::PackedBlockField, i::Integer, ranges) = view(f.data, ranges..., i)
_flat_similar(f::PackedBlockField, ::Type{T}, len::Int) where {T} = similar(f.data, T, len)

function block(f::PackedBlockField{L}, i::Integer, leaf_grid) where {L}
    _require_current(f)
    return Field{L}(_block_array(f, i), leaf_grid)
end

Base.eltype(::PackedBlockField{L,A}) where {L,A} = eltype(A)

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
