#--------------------------------------------------------------------------------# Inter-block halo exchange

# Slab geometry of a face copy along dimension D (h = halo width, n = interior count
# per block). The same-level copy reuses the periodic source indices: this block's
# low-ghost slab [1:h] mirrors the neighbor's high-interior slab [n+1:n+h], and the
# high-ghost slab [h+n+1:2h+n] mirrors the neighbor's low-interior slab [h+1:2h].
# The transverse extent is the full slab (Colon); corner cells it touches are not
# read by axis-aligned (Laplacian/Derivative) stencils.
@inline function _face_slabs(::Val{D}, side::Int, h::Int, n::Int) where {D}
    if side == -1
        return (1:h, (n + 1):(n + h))                  # (this ghost, neighbor source)
    else
        return ((h + n + 1):(2h + n), (h + 1):(2h))
    end
end

"""
    halo_update!(x::BlockField, g::BlockForest) -> x

Fill each leaf block's interface ghosts from its neighbors. For every leaf face
with a same-level neighbor (including periodic wrap), copy the neighbor's
boundary-interior slab into this block's ghost slab. Domain-boundary faces are left
to the per-leaf `apply_bc!`. Must run once over the whole forest before any stencil
sweep. Coarse–fine interfaces (non-uniform forests) are filled in a later phase;
they are skipped here.
"""
function halo_update!(x::BlockField, g::BlockForest{N}) where {N}
    forest = g.forest
    h = g.halo
    n = g.blocksize
    for (i, K) in enumerate(forest.leaves)
        this_block = x.blocks[i]
        for d in 1:N, side in (-1, 1)
            nbr = face_neighbor(forest, K, d, side)
            nbr === nothing && continue                # domain boundary → apply_bc!
            is_leaf(forest, nbr) || continue           # coarse–fine interface → later phase
            nbr_block = x.blocks[leaf_index(forest, nbr)]
            ghost, source = _face_slabs(Val(d), side, h[d], n[d])
            _dimslice(this_block, Val(d), ghost) .= _dimslice(nbr_block, Val(d), source)
        end
    end
    return x
end

"""
    halo_update_adjoint!(x::BlockField, g::BlockForest) -> x

Exact discrete adjoint of [`halo_update!`](@ref): scatter-add each interface-ghost
contribution into the neighbor block's interior source cell, then zero the ghost.
Mirrors [`fold_bc!`](@ref) across block faces — the transpose half needed for the
adjoint identity on a forest.
"""
function halo_update_adjoint!(x::BlockField, g::BlockForest{N}) where {N}
    forest = g.forest
    h = g.halo
    n = g.blocksize
    for (i, K) in enumerate(forest.leaves)
        this_block = x.blocks[i]
        for d in 1:N, side in (-1, 1)
            nbr = face_neighbor(forest, K, d, side)
            nbr === nothing && continue
            is_leaf(forest, nbr) || continue
            nbr_block = x.blocks[leaf_index(forest, nbr)]
            ghost, source = _face_slabs(Val(d), side, h[d], n[d])
            dst = _dimslice(this_block, Val(d), ghost)
            _dimslice(nbr_block, Val(d), source) .+= dst
            fill!(dst, zero(eltype(this_block)))
        end
    end
    return x
end
