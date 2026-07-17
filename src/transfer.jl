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
sweep. Coarse–fine interfaces are a later phase — a non-uniform forest throws.
"""
function halo_update!(x::BlockField, g::BlockForest{N}) where {N}
    _require_uniform(g)
    _require_current(x)
    forest = g.forest
    for (i, K) in enumerate(forest.leaves)
        _halo_faces!(x, forest, K, i, g.halo, g.blocksize, Val(1))
    end
    return x
end

# Compile-time dimension recursion: a runtime `d` in `_dimslice(block, Val(d), r)`
# boxes the SubArray type and forces dynamic dispatch in this hot loop (see the
# _dimslice note in boundaries.jl). Recursing on `Val{D}` keeps D a constant, so the
# face copy stays type-stable and allocation-free — the same idiom as `_apply_bc_dims!`.
function _halo_faces!(x::BlockField, forest::Forest{N}, K, i, h, n, ::Val{D}) where {N,D}
    D > N && return nothing
    this_block = x.blocks[i]
    for side in (-1, 1)
        nbr = face_neighbor(forest, K, Val(D), side)   # nothing at a domain boundary → apply_bc!
        if nbr !== nothing && is_leaf(forest, nbr)     # is_leaf always holds while uniform is enforced
            nbr_block = x.blocks[leaf_index(forest, nbr)]
            ghost, source = _face_slabs(Val(D), side, h[D], n[D])
            _dimslice(this_block, Val(D), ghost) .= _dimslice(nbr_block, Val(D), source)
        end
    end
    return _halo_faces!(x, forest, K, i, h, n, Val(D + 1))
end

"""
    halo_update_adjoint!(x::BlockField, g::BlockForest) -> x

Exact discrete adjoint of [`halo_update!`](@ref): scatter-add each interface-ghost
contribution into the neighbor block's interior source cell, then zero the ghost.
Mirrors [`fold_bc!`](@ref) across block faces — the transpose half needed for the
adjoint identity on a forest.

Exactness relies on per-leaf adjoints leaving corner ghosts exactly zero (true for
axis-aligned stencils, whose transposes never scatter there): face slabs span the
full transverse extent, so a nonzero corner ghost would propagate through two
sequential face scatters into a diagonal neighbor's interior. A future
cross-derivative leaf needs a corner-aware exchange.
"""
function halo_update_adjoint!(x::BlockField, g::BlockForest{N}) where {N}
    _require_uniform(g)
    _require_current(x)
    forest = g.forest
    for (i, K) in enumerate(forest.leaves)
        _halo_faces_adjoint!(x, forest, K, i, g.halo, g.blocksize, Val(1))
    end
    return x
end

# Exact transpose of _halo_faces!; same compile-time Val{D} recursion for the same
# allocation-free reason.
function _halo_faces_adjoint!(x::BlockField, forest::Forest{N}, K, i, h, n, ::Val{D}) where {N,D}
    D > N && return nothing
    this_block = x.blocks[i]
    for side in (-1, 1)
        nbr = face_neighbor(forest, K, Val(D), side)
        if nbr !== nothing && is_leaf(forest, nbr)
            nbr_block = x.blocks[leaf_index(forest, nbr)]
            ghost, source = _face_slabs(Val(D), side, h[D], n[D])
            dst = _dimslice(this_block, Val(D), ghost)
            _dimslice(nbr_block, Val(D), source) .+= dst
            fill!(dst, zero(eltype(this_block)))
        end
    end
    return _halo_faces_adjoint!(x, forest, K, i, h, n, Val(D + 1))
end
