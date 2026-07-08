#--------------------------------------------------------------------------------# Forest topology

"""
    LeafKey{N}

Identity of one block in a [`BlockForest`](@ref): its refinement `level` and its
integer block `coords` at that level. At level `ℓ` a dimension is tiled by
`nroot * 2^ℓ` blocks, so `coords[d] ∈ 0:(nroot[d]*2^ℓ - 1)`. Geometry (extent,
spacing) is *recomputed* from the key — never stored per block.
"""
struct LeafKey{N}
    level::Int
    coords::NTuple{N,Int}
end

"""
    parent_key(key::LeafKey) -> LeafKey

The level-`(ℓ-1)` block that contains `key`. Undefined at level 0.
"""
parent_key(key::LeafKey{N}) where {N} = LeafKey(key.level - 1, ntuple(d -> key.coords[d] >> 1, Val(N)))

"""
    children(key::LeafKey) -> Vector{LeafKey}

The `2ᴺ` level-`(ℓ+1)` blocks tiling `key`.
"""
function children(key::LeafKey{N}) where {N}
    base = ntuple(d -> key.coords[d] << 1, Val(N))
    offsets = CartesianIndices(ntuple(_ -> 0:1, Val(N)))
    return [LeafKey(key.level + 1, ntuple(d -> base[d] + o[d], Val(N))) for o in offsets]
end

#--------------------------------------------------------------------------------# Forest

"""
    Forest{N}

Serial, pure-Julia block-structured AMR topology: a forest of `2ᴺ`-trees whose
leaves tile the domain with no overlap. Stores the leaf set in Morton (Z-order)
order plus a key→index map for `O(1)` lookup; the tree *shape* is recomputed from
the keys (no parent/child pointers). Backs [`BlockForest`](@ref); the geometry and
field storage live there.

Fields: `nroot` (root tiling = base ncells ÷ blocksize), `periodic` (per-dimension,
controls neighbor wrap), `maxlevel`, `leaves` (Morton-sorted), `index`, `uniform`
(all leaves on one level — no coarse–fine faces), `generation` (bumped by every
regrid that changes the leaf set; fields are tied to the generation they were
allocated on).
"""
struct Forest{N}
    nroot::NTuple{N,Int}
    periodic::NTuple{N,Bool}
    maxlevel::Int
    leaves::Vector{LeafKey{N}}
    index::Dict{LeafKey{N},Int}
    uniform::Base.RefValue{Bool}
    generation::Base.RefValue{Int}
end

function Forest(nroot::NTuple{N,Int}, periodic::NTuple{N,Bool}, maxlevel::Int) where {N}
    N * _morton_bits(nroot, maxlevel) <= 8 * sizeof(UInt) || throw(
        ArgumentError(
            "nroot $nroot with maxlevel $maxlevel exceeds the $(8 * sizeof(UInt))-bit " *
            "Morton key capacity",
        ),
    )
    forest = Forest{N}(
        nroot, periodic, maxlevel, LeafKey{N}[], Dict{LeafKey{N},Int}(), Ref(true), Ref(0)
    )
    roots = [LeafKey(0, ntuple(d -> I[d] - 1, Val(N))) for I in CartesianIndices(nroot)]
    return _set_leaves!(forest, roots)
end

nleaves(forest::Forest) = length(forest.leaves)
is_leaf(forest::Forest, key::LeafKey) = haskey(forest.index, key)
leaf_index(forest::Forest, key::LeafKey) = forest.index[key]

function Base.show(io::IO, forest::Forest{N}) where {N}
    if isempty(forest.leaves)
        print(io, "Forest{$N}(nroot=$(forest.nroot), maxlevel=$(forest.maxlevel), 0 leaves)")
    else
        lo, hi = extrema(k -> k.level, forest.leaves)
        print(
            io,
            "Forest{$N}(nroot=$(forest.nroot), maxlevel=$(forest.maxlevel), ",
            "$(nleaves(forest)) leaves, levels $lo:$hi)",
        )
    end
end

# Number of bits per coordinate needed to Morton-encode the finest resolution.
function _morton_bits(nroot::NTuple{N,Int}, maxlevel::Int) where {N}
    maxfine = maximum(nroot) << maxlevel
    return maxfine <= 1 ? 1 : (8 * sizeof(Int) - leading_zeros(maxfine - 1))
end
_morton_bits(forest::Forest) = _morton_bits(forest.nroot, forest.maxlevel)

# Lower-corner block index at the finest level — the position a leaf maps to for
# Morton ordering (a leaf and its descendant are never both present, so unique).
_fine_coords(key::LeafKey{N}, maxlevel::Int) where {N} =
    ntuple(d -> key.coords[d] << (maxlevel - key.level), Val(N))

# Z-order interleave of `coords` (dim 1 least significant within each bit group).
function morton(coords::NTuple{N,Int}, nbits::Int) where {N}
    code = zero(UInt)
    for b in 0:(nbits - 1), d in 1:N
        bit = (UInt(coords[d]) >> b) & one(UInt)
        code |= bit << (b * N + (d - 1))
    end
    return code
end

# Commit a new leaf set: sort into Morton order, rebuild the key→index map, and
# refresh the uniformity flag. A set identical to the current one is a no-op so
# predicate-miss regrids do not invalidate existing fields.
function _set_leaves!(forest::Forest{N}, keys) where {N}
    nbits = _morton_bits(forest)
    maxlevel = forest.maxlevel
    sorted = sort!(vec(collect(LeafKey{N}, keys)); by=k -> morton(_fine_coords(k, maxlevel), nbits))
    sorted == forest.leaves && return forest
    empty!(forest.leaves)
    append!(forest.leaves, sorted)
    empty!(forest.index)
    for (i, k) in enumerate(sorted)
        forest.index[k] = i
    end
    forest.uniform[] = isempty(sorted) || all(k -> k.level == sorted[1].level, sorted)
    forest.generation[] += 1
    return forest
end

#--------------------------------------------------------------------------------# Neighbor queries

"""
    face_neighbor(forest, key, dim, side) -> LeafKey or nothing

The same-level block adjacent to `key` across the face in dimension `dim`
(`side = -1` low, `+1` high). Wraps for a periodic dimension; returns `nothing` at
a non-periodic domain boundary. The returned key may or may not be an actual leaf
— use [`leaf_covering`](@ref) to resolve which leaf covers its region.
"""
# Neighbor block coordinate along `dim` at `side` (±1), or `nothing` past a
# non-periodic domain boundary.
@inline function _neighbor_coord(forest::Forest, key::LeafKey, dim::Int, side::Int)
    nblocks = forest.nroot[dim] << key.level
    c = key.coords[dim] + side
    if c < 0 || c >= nblocks
        forest.periodic[dim] || return nothing
        return mod(c, nblocks)
    end
    return c
end

function face_neighbor(forest::Forest{N}, key::LeafKey{N}, dim::Int, side::Int) where {N}
    c = _neighbor_coord(forest, key, dim, side)
    c === nothing && return nothing
    return LeafKey(key.level, ntuple(d -> d == dim ? c : key.coords[d], Val(N)))
end

# Compile-time-dimension variant for the hot halo sweeps ([`halo_update!`](@ref)):
# with `D` a constant the coord rebuild carries no capturing closure over a runtime
# dimension, so it allocates nothing.
function face_neighbor(forest::Forest{N}, key::LeafKey{N}, ::Val{D}, side::Int) where {N,D}
    c = _neighbor_coord(forest, key, D, side)
    c === nothing && return nothing
    return LeafKey(key.level, ntuple(d -> d == D ? c : key.coords[d], Val(N)))
end

"""
    leaf_covering(forest, key) -> LeafKey or nothing

The leaf that covers `key`'s region: `key` itself if it is a leaf, else its
nearest leaf ancestor (a coarser block). Returns `nothing` when `key`'s region is
covered by *finer* leaves (i.e. `key` is a refined, internal node).
"""
function leaf_covering(forest::Forest{N}, key::LeafKey{N}) where {N}
    k = key
    while true
        is_leaf(forest, k) && return k
        k.level == 0 && return nothing
        k = parent_key(k)
    end
end

#--------------------------------------------------------------------------------# Refine / coarsen / balance

"""
    refine!(forest::Forest, should_refine) -> forest

Replace every leaf `key` (below `maxlevel`) for which `should_refine(key)` is true
with its `2ᴺ` children, then re-establish 2:1 balance via [`balance!`](@ref).
"""
function refine!(forest::Forest{N}, should_refine) where {N}
    keys = Set(forest.leaves)
    for key in forest.leaves
        if key.level < forest.maxlevel && should_refine(key)
            delete!(keys, key)
            for c in children(key)
                push!(keys, c)
            end
        end
    end
    _set_leaves!(forest, keys)
    return balance!(forest)
end

"""
    coarsen!(forest::Forest, should_coarsen) -> forest

Replace each complete family of `2ᴺ` sibling leaves — all present and all
satisfying `should_coarsen(key)` — with their parent, then re-balance. Families
that are incomplete or only partially flagged are left untouched.
"""
function coarsen!(forest::Forest{N}, should_coarsen) where {N}
    families = Dict{LeafKey{N},Vector{LeafKey{N}}}()
    for key in forest.leaves
        key.level == 0 && continue
        push!(get!(families, parent_key(key), LeafKey{N}[]), key)
    end
    keys = Set(forest.leaves)
    for (p, sibs) in families
        (length(sibs) == 2^N && all(should_coarsen, sibs)) || continue
        for s in sibs
            delete!(keys, s)
        end
        push!(keys, p)
    end
    _set_leaves!(forest, keys)
    return balance!(forest)
end

"""
    balance!(forest::Forest) -> forest

Enforce the 2:1 balance invariant across faces: no leaf has a face-adjacent leaf
more than one level finer. Iterates to a fixpoint, refining the coarser leaf of
any violating pair. This is what reduces every coarse–fine interface to the three
tractable cases (same level / one-coarser / one-finer).
"""
function balance!(forest::Forest{N}) where {N}
    while true
        to_refine = Set{LeafKey{N}}()
        for F in forest.leaves, dim in 1:N, side in (-1, 1)
            nbr = face_neighbor(forest, F, dim, side)
            nbr === nothing && continue
            cover = leaf_covering(forest, nbr)
            # cover.level ≤ F.level − 2 < maxlevel, so refining it never exceeds maxlevel.
            if cover !== nothing && F.level - cover.level >= 2
                push!(to_refine, cover)
            end
        end
        isempty(to_refine) && break
        keys = Set(forest.leaves)
        for k in to_refine
            delete!(keys, k)
            for c in children(k)
                push!(keys, c)
            end
        end
        _set_leaves!(forest, keys)
    end
    return forest
end
