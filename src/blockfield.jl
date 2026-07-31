#--------------------------------------------------------------------------------# BlockField (vector-of-blocks)

"""
    AbstractBlockField <: AbstractField

Supertype of the forest field storage layouts: the vector-of-blocks reference
[`BlockField`](@ref) and the packed [`PackedBlockField`](@ref). Both expose the
same per-leaf interface — `block`, `_block_array` (the `i`-th leaf's padded
array), `_block_view` (a view into it by ranges) — plus `grid`/`generation`
fields, so the halo/BC descriptor sweeps, flat-vector boundary, and the per-leaf
fallback operator sweep run on either layout; only allocation, `pack`/`unpack`,
and the forest-native kernel sweeps dispatch on the concrete type.
"""
abstract type AbstractBlockField <: AbstractField end

"""
    BlockLayout

Supertype of the zero-size tags naming a block-storage layout: [`BlocksLayout`](@ref)
for the vector-of-blocks [`BlockField`](@ref), [`PackedLayout`](@ref) for the
contiguous [`PackedBlockField`](@ref).

The tag exists so the descriptor sweeps can be written against **raw storage**
(`_storage(f)`) plus a layout tag instead of against the field struct. That split is
what makes them a legal Enzyme custom-rule seam: from Julia 1.12 a rule argument may
not mix GC-tracked pointers with inline floats (EnzymeAD/Enzyme.jl#2707), which a
field does through its embedded grid, while a bare block vector or packed array does
not. Dispatch and specialization are unchanged — the tag is a singleton, so
`_leaf_array`/`_leaf_view` still resolve to one concrete `SubArray` per layout.
"""
abstract type BlockLayout end

"""
    BlocksLayout()

Layout tag for vector-of-blocks storage: `store[i]` is leaf `i`'s padded array.
"""
struct BlocksLayout <: BlockLayout end

"""
    PackedLayout()

Layout tag for packed storage: leaf `i` is the `i`-th slice along the trailing
dimension of one contiguous array.
"""
struct PackedLayout <: BlockLayout end

"""
    _storage(f::AbstractBlockField)

The field's raw block storage, stripped of grid and generation metadata — the
argument the descriptor sweeps and their Enzyme rules take. Paired with
[`_layout`](@ref).
"""
function _storage end

"""
    _layout(f::AbstractBlockField) -> BlockLayout

The field's storage layout tag. See [`BlockLayout`](@ref).
"""
function _layout end

# Storage-level twins of `_block_array`/`_block_view`, indexing raw storage by
# layout. The field-level accessors are defined in terms of these, so the two can
# never drift apart.
@inline _leaf_array(store, ::BlocksLayout, i::Integer) = store[i]
@inline _leaf_view(store, ::BlocksLayout, i::Integer, ranges) = view(store[i], ranges...)

"""
    BlockField{L}(blocks, grid)
    BlockField(blocks, grid)

Field over a [`BlockForest`](@ref): one halo-padded array per leaf block, indexed
in the forest's Morton (storage) order, with location trait `L` (default
[`Center`](@ref)). The vector-of-blocks layout is the simplest correct storage; a
packed contiguous buffer can replace it later without touching operators. Block
storage is tied to the leaf set at allocation time — the forest's regrid
generation is stamped into the field, and any use after a `refine!`/`coarsen!`
that changed the leaf set throws; allocate a fresh field after a regrid.

See also: [`scalar_field`](@ref), [`vector_field`](@ref), [`block`](@ref).
"""
struct BlockField{L,A<:AbstractArray,G<:BlockForest} <: AbstractBlockField
    blocks::Vector{A}
    grid::G
    generation::Int
end
BlockField{L}(blocks::Vector{<:AbstractArray}, grid::BlockForest) where {L} =
    BlockField{L,eltype(blocks),typeof(grid)}(blocks, grid, grid.forest.generation[])
BlockField(blocks::Vector{<:AbstractArray}, grid::BlockForest) = BlockField{Center}(blocks, grid)

# Per-layout storage accessors behind which everything else is layout-agnostic.
_storage(f::BlockField) = f.blocks
_layout(::BlockField) = BlocksLayout()
_block_array(f::BlockField, i::Integer) = _leaf_array(f.blocks, BlocksLayout(), i)
_block_view(f::BlockField, i::Integer, ranges) = _leaf_view(f.blocks, BlocksLayout(), i, ranges)
_flat_similar(f::BlockField, ::Type{T}, len::Int) where {T} = similar(first(f.blocks), T, len)

# Regrid guard: block storage is tied to the leaf set the field was allocated on.
function _require_current(f::AbstractBlockField)
    f.generation == f.grid.forest.generation[] || throw(
        ArgumentError(
            "this $(nameof(typeof(f))) was allocated before the forest was regridded " *
            "(refine!/coarsen!/balance!); allocate a fresh field on the current forest",
        ),
    )
    return nothing
end

function scalar_field(bf::BlockForest, ::Type{T}=eltype(bf.spacing0)) where {T<:Number}
    backend = KernelAbstractions.get_backend(bf)
    psize = bf.blocksize .+ 2 .* bf.halo
    blocks = [KernelAbstractions.zeros(backend, T, psize...) for _ in 1:nleaves(bf)]
    return BlockField(blocks, bf)
end

function vector_field(bf::BlockForest{N}, ::Type{T}=eltype(bf.spacing0)) where {N,T<:Number}
    backend = KernelAbstractions.get_backend(bf)
    psize = bf.blocksize .+ 2 .* bf.halo
    blocks = [KernelAbstractions.zeros(backend, SVector{N,T}, psize...) for _ in 1:nleaves(bf)]
    return BlockField(blocks, bf)
end

"""
    block(f::AbstractBlockField, i::Integer) -> Field
    block(f::AbstractBlockField, i::Integer, leaf_grid) -> Field

The `i`-th leaf as an ordinary [`Field`](@ref) sharing storage with `f` (no copy),
on its leaf [`CartesianGrid`](@ref). This is what per-block operators consume; the
3-arg form takes an already-computed `leaf_grid` (rebuilding one is cheap — every
leaf shares one concrete all-`Interface` grid type).
"""
function block(f::BlockField{L}, i::Integer, leaf_grid) where {L}
    _require_current(f)
    return Field{L}(f.blocks[i], leaf_grid)
end
block(f::AbstractBlockField, i::Integer) = block(f, i, leaf_grid(f.grid, i))

Base.eltype(::BlockField{L,A}) where {L,A} = eltype(A)
ncomponents(f::AbstractBlockField) = _ncomponents(eltype(f))

function component(f::BlockField{L}, d::Integer) where {L}
    1 <= d <= ncomponents(f) ||
        throw(ArgumentError("component $d out of range for $(ncomponents(f)) components"))
    blocks = [getindex.(b, d) for b in f.blocks]
    return BlockField{L,eltype(blocks),typeof(f.grid)}(blocks, f.grid, f.generation)
end

# Derived fields inherit the source's generation: a copy of a stale field is
# equally stale — stamping the current generation would bless wrong-size storage.
Base.similar(f::BlockField{L,A,G}) where {L,A,G} =
    BlockField{L,A,G}([similar(b) for b in f.blocks], f.grid, f.generation)
function Base.similar(f::BlockField{L}, ::Type{E}) where {L,E}
    blocks = [similar(b, E) for b in f.blocks]
    return BlockField{L,eltype(blocks),typeof(f.grid)}(blocks, f.grid, f.generation)
end
Base.copy(f::BlockField{L,A,G}) where {L,A,G} =
    BlockField{L,A,G}([copy(b) for b in f.blocks], f.grid, f.generation)

function set!(f::AbstractBlockField, fun::F) where {F}
    for i in 1:nleaves(f.grid)
        set!(block(f, i), fun)
    end
    return f
end

function zero_ghosts!(f::AbstractBlockField)
    for i in 1:nleaves(f.grid)
        zero_ghosts!(block(f, i))
    end
    return f
end

# Zero every entry (interior and ghosts) of every block.
function _zero_all!(f::AbstractBlockField)
    for i in 1:nleaves(f.grid)
        fill!(_block_array(f, i), zero(eltype(f)))
    end
    return f
end

function Adapt.adapt_structure(to, f::BlockField{L}) where {L}
    blocks = [Adapt.adapt(to, b) for b in f.blocks]
    grid = Adapt.adapt(to, f.grid)
    return BlockField{L,eltype(blocks),typeof(grid)}(blocks, grid, f.generation)
end

#--------------------------------------------------------------------------------# Flat-vector boundary (per-block, Morton order)

flat_length(f::AbstractBlockField) = nleaves(f.grid) * prod(f.grid.blocksize) * ncomponents(f)

# DOFs contributed by one block (uniform across blocks: same blocksize & ncomp).
_block_dofs(f::AbstractBlockField) = prod(f.grid.blocksize) * ncomponents(f)
_block_range(f::AbstractBlockField, i::Integer) =
    ((i - 1) * _block_dofs(f) + 1):(i * _block_dofs(f))

function flatten(f::AbstractBlockField)
    # Allocate on the field's device (matching the single-grid flatten) so the
    # flat/Krylov path stays device-generic.
    v = _flat_similar(f, _scalar_eltype(eltype(f)), flat_length(f))
    return interior_to_flat!(v, f)
end

flat_to_interior!(f::AbstractBlockField, v::AbstractVector) = _flat_to_interior_leaves!(f, v)

interior_to_flat!(v::AbstractVector, f::AbstractBlockField, α::Number=true, β::Number=false) =
    _interior_to_flat_leaves!(v, f, α, β)

# Per-leaf reference bodies — shared by the AbstractBlockField methods above and
# the non-GPU branch of the packed overrides (packedfield.jl).
function _flat_to_interior_leaves!(f::AbstractBlockField, v::AbstractVector)
    for i in 1:nleaves(f.grid)
        flat_to_interior!(block(f, i), view(v, _block_range(f, i)))
    end
    return f
end

function _interior_to_flat_leaves!(v::AbstractVector, f::AbstractBlockField, α::Number, β::Number)
    for i in 1:nleaves(f.grid)
        interior_to_flat!(view(v, _block_range(f, i)), block(f, i), α, β)
    end
    return v
end
