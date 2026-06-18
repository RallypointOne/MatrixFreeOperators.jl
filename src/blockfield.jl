#--------------------------------------------------------------------------------# BlockField (vector-of-blocks)

"""
    BlockField{L}(blocks, grid)
    BlockField(blocks, grid)

Field over a [`BlockForest`](@ref): one halo-padded array per leaf block, indexed
in the forest's Morton (storage) order, with location trait `L` (default
[`Center`](@ref)). The vector-of-blocks layout is the simplest correct storage; a
packed contiguous buffer can replace it later without touching operators. Block
storage is tied to the leaf set at allocation time — re-allocate after a regrid.

See also: [`scalar_field`](@ref), [`vector_field`](@ref), [`block`](@ref).
"""
struct BlockField{L,A<:AbstractArray,G<:BlockForest} <: AbstractField
    blocks::Vector{A}
    grid::G
end
BlockField{L}(blocks::Vector{<:AbstractArray}, grid::BlockForest) where {L} =
    BlockField{L,eltype(blocks),typeof(grid)}(blocks, grid)
BlockField(blocks::Vector{<:AbstractArray}, grid::BlockForest) = BlockField{Center}(blocks, grid)

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
    block(f::BlockField, i::Integer) -> Field
    block(f::BlockField, i::Integer, leaf_grid) -> Field

The `i`-th leaf as an ordinary [`Field`](@ref) sharing storage with `f` (no copy),
on its leaf [`CartesianGrid`](@ref). This is what per-block operators consume; pass
a precomputed `leaf_grid` to avoid rebuilding it.
"""
block(f::BlockField{L}, i::Integer, leaf_grid) where {L} = Field{L}(f.blocks[i], leaf_grid)
block(f::BlockField, i::Integer) = block(f, i, leaf_grid(f.grid, i))

Base.eltype(::BlockField{L,A}) where {L,A} = eltype(A)
ncomponents(f::BlockField) = _ncomponents(eltype(f))

Base.similar(f::BlockField{L}) where {L} = BlockField{L}([similar(b) for b in f.blocks], f.grid)
Base.copy(f::BlockField{L}) where {L} = BlockField{L}([copy(b) for b in f.blocks], f.grid)

function set!(f::BlockField, fun::F) where {F}
    for i in 1:nleaves(f.grid)
        set!(block(f, i), fun)
    end
    return f
end

function zero_ghosts!(f::BlockField)
    for i in 1:nleaves(f.grid)
        zero_ghosts!(block(f, i))
    end
    return f
end

function Adapt.adapt_structure(to, f::BlockField{L}) where {L}
    blocks = [Adapt.adapt(to, b) for b in f.blocks]
    return BlockField{L}(blocks, Adapt.adapt(to, f.grid))
end

#--------------------------------------------------------------------------------# Flat-vector boundary (per-block, Morton order)

flat_length(f::BlockField) = nleaves(f.grid) * prod(f.grid.blocksize) * ncomponents(f)

# DOFs contributed by one block (uniform across blocks: same blocksize & ncomp).
_block_dofs(f::BlockField) = prod(f.grid.blocksize) * ncomponents(f)
_block_range(f::BlockField, i::Integer) = ((i - 1) * _block_dofs(f) + 1):(i * _block_dofs(f))

function flatten(f::BlockField)
    v = Vector{_scalar_eltype(eltype(f))}(undef, flat_length(f))
    for i in 1:nleaves(f.grid)
        interior_to_flat!(view(v, _block_range(f, i)), block(f, i))
    end
    return v
end

function flat_to_interior!(f::BlockField, v::AbstractVector)
    for i in 1:nleaves(f.grid)
        flat_to_interior!(block(f, i), view(v, _block_range(f, i)))
    end
    return f
end

function interior_to_flat!(v::AbstractVector, f::BlockField, α::Number=true, β::Number=false)
    for i in 1:nleaves(f.grid)
        interior_to_flat!(view(v, _block_range(f, i)), block(f, i), α, β)
    end
    return v
end
