#--------------------------------------------------------------------------------# Inter-block halo exchange

# Slab geometry of a face copy along a dimension (h = halo width, n = interior count
# per block, both for that dimension). The same-level copy reuses the periodic source
# indices: this block's low-ghost slab [1:h] mirrors the neighbor's high-interior
# slab [n+1:n+h], and the high-ghost slab [h+n+1:2h+n] mirrors the neighbor's
# low-interior slab [h+1:2h]. Build-time only — the schedule stores the resolved
# ranges, so the hot path never recomputes this.
@inline function _face_slabs(side::Int, h::Int, n::Int)
    if side == -1
        return (1:h, (n + 1):(n + h))                  # (this ghost, neighbor source)
    else
        return ((h + n + 1):(2h + n), (h + 1):(2h))
    end
end

# Full padded box for a face slab: range `r` along dim `d`, the full padded extent
# transversely (matching the retired `_dimslice` Colon). The corner cells it spans
# are not read by axis-aligned (Laplacian/Derivative) stencils — see the
# `halo_update_adjoint!` caveat.
_face_box(d::Int, r::UnitRange{Int}, h::NTuple{N,Int}, n::NTuple{N,Int}) where {N} =
    ntuple(t -> t == d ? r : (1:(n[t] + 2h[t])), Val(N))

# Build the same-level exchange schedule: for every leaf face with a same-level
# neighbor (including periodic wrap), one CopyDescriptor. Domain-boundary faces
# (`face_neighbor === nothing`) emit nothing — they are `apply_bc!` territory. The
# iteration order (Morton leaf, dim, side) is exactly the retired `Val{D}` sweep's
# execution order, so each ghost slab is the `dst` of exactly one descriptor and the
# run order is bit-identical to the previous implementation.
function _build_exchange_schedule(bf::BlockForest{N}) where {N}
    _require_uniform(bf)     # Part 2 fills the coarse–fine branch and lifts this
    forest, h, n = bf.forest, bf.halo, bf.blocksize
    copies = CopyDescriptor{N}[]
    for (i, K) in enumerate(forest.leaves), d in 1:N, side in (-1, 1)
        nbr = face_neighbor(forest, K, d, side)
        nbr === nothing && continue
        if is_leaf(forest, nbr)
            ghost, source = _face_slabs(side, h[d], n[d])
            push!(
                copies,
                CopyDescriptor{N}(
                    leaf_index(forest, nbr), i,
                    _face_box(d, source, h, n), _face_box(d, ghost, h, n),
                ),
            )
            # else — coarse–fine face; unreachable while uniform is enforced. Part 2:
            #   leaf_covering(forest, nbr) !== nothing → coarse neighbor (K is finer)
            #     → coarse→fine quadratic ghost-interpolation descriptor
            #   otherwise nbr is a refined node (K is coarser)
            #     → fine→coarse restriction descriptor from nbr's face children
        end
    end
    return ExchangeSchedule{N}(copies, forest.generation[])
end

# Per-generation cache accessor: keyed by the live forest generation, so a schedule
# built before a regrid can never be used — it rebuilds instead. The user-facing
# staleness throws come from `_require_current` (fields) and the PreparedForest
# generation guard (prepared operators). Single-threaded per solve by package
# contract; the build is deterministic, so a benign race writes identical schedules.
function _exchange_schedule(bf::BlockForest)
    sched = bf.schedule[]
    sched.generation == bf.forest.generation[] && return sched
    sched = _build_exchange_schedule(bf)
    bf.schedule[] = sched
    return sched
end

"""
    halo_update!(x::BlockField, g::BlockForest) -> x

Fill each leaf block's interface ghosts from its neighbors. For every leaf face
with a same-level neighbor (including periodic wrap), copy the neighbor's
boundary-interior slab into this block's ghost slab. Domain-boundary faces are left
to the per-leaf `apply_bc!`. Must run once over the whole forest before any stencil
sweep. Coarse–fine interfaces are a later phase — a non-uniform forest throws.
"""
function halo_update!(x::BlockField, g::BlockForest)
    _require_uniform(g)
    _require_current(x)
    _run_copies!(x.blocks, _exchange_schedule(g).copies)
    return x
end

# Function barrier: specializing on the concrete block-array type `A` makes each view
# one concrete SubArray, so the loop is dispatch- and allocation-free. `view .= view`
# broadcasts on GPU arrays without scalar indexing.
function _run_copies!(blocks::Vector{A}, copies::Vector{CopyDescriptor{N}}) where {A,N}
    for c in copies
        view(blocks[c.dst], c.dst_ranges...) .= view(blocks[c.src], c.src_ranges...)
    end
    return nothing
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
function halo_update_adjoint!(x::BlockField, g::BlockForest)
    _require_uniform(g)
    _require_current(x)
    _run_copies_adjoint!(x.blocks, _exchange_schedule(g).copies)
    return x
end

# Exact transpose of _run_copies!: same descriptors, roles transposed, ghost zeroed
# after folding. Each ghost slab is a `dst` exactly once, so the fold-then-zero
# touches every ghost region exactly once regardless of iteration order.
function _run_copies_adjoint!(blocks::Vector{A}, copies::Vector{CopyDescriptor{N}}) where {A,N}
    for c in copies
        ghost = view(blocks[c.dst], c.dst_ranges...)
        view(blocks[c.src], c.src_ranges...) .+= ghost
        fill!(ghost, zero(eltype(A)))
    end
    return nothing
end
