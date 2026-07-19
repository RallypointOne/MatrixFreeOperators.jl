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
_block_array(f::BlockField, i::Integer) = f.blocks[i]
_block_view(f::BlockField, i::Integer, ranges) = view(f.blocks[i], ranges...)
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
    for i in 1:nleaves(f.grid)
        interior_to_flat!(view(v, _block_range(f, i)), block(f, i))
    end
    return v
end

function flat_to_interior!(f::AbstractBlockField, v::AbstractVector)
    for i in 1:nleaves(f.grid)
        flat_to_interior!(block(f, i), view(v, _block_range(f, i)))
    end
    return f
end

function interior_to_flat!(
    v::AbstractVector, f::AbstractBlockField, α::Number=true, β::Number=false
)
    for i in 1:nleaves(f.grid)
        interior_to_flat!(view(v, _block_range(f, i)), block(f, i), α, β)
    end
    return v
end
