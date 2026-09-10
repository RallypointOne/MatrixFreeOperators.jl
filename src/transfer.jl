#--------------------------------------------------------------------------------# Inter-block halo exchange

# Slab geometry of a same-level face copy along a dimension (h = halo width, n =
# interior count per block, both for that dimension). The copy reuses the periodic
# source indices: this block's low-ghost slab [1:h] mirrors the neighbor's
# high-interior slab [n+1:n+h], and the high-ghost slab [h+n+1:2h+n] mirrors the
# neighbor's low-interior slab [h+1:2h]. Build-time only — the schedule stores the
# resolved ranges, so the hot path never recomputes this.
@inline function _face_slabs(side::Int, h::Int, n::Int)
    if side == -1
        return (1:h, (n + 1):(n + h))                  # (this ghost, neighbor source)
    else
        return ((h + n + 1):(2h + n), (h + 1):(2h))
    end
end

# Full padded box for a same-level face slab: range `r` along dim `d`, the full
# padded extent transversely (matching the retired `_dimslice` Colon). The corner
# cells it spans are not read by axis-aligned (Laplacian/Derivative) stencils.
_face_box(d::Int, r::UnitRange{Int}, h::NTuple{N,Int}, n::NTuple{N,Int}) where {N} =
    ntuple(t -> t == d ? r : (1:(n[t] + 2h[t])), Val(N))

#--------------------------------------------------------------------------------# Coarse–fine descriptor emission

# The quadratic coarse–fine scheme fills exactly one ghost layer and needs the
# coarse tangential taps and both coarse normal layers to be interior, which
# bounds the block geometry. Uniform forests keep the looser v1 constraints.
function _validate_coarse_fine(bf::BlockForest{N}) where {N}
    all(==(1), bf.halo) || throw(
        ArgumentError(
            "coarse–fine interface interpolation requires halo width 1 in every " *
            "dimension, got $(bf.halo)",
        ),
    )
    ok = N == 1 ? all(>=(2), bf.blocksize) : all(n -> n >= 4 && iseven(n), bf.blocksize)
    ok || throw(
        ArgumentError(
            "coarse–fine interface interpolation requires blocksize ≥ " *
            "$(N == 1 ? "2" : "4 and even") in every dimension, got $(bf.blocksize)",
        ),
    )
    return nothing
end

_throw_unbalanced(K) = throw(
    ArgumentError(
        "forest is not 2:1 balanced at leaf $K; run balance! after manual topology " *
        "edits (refine!/coarsen! re-balance automatically)",
    ),
)

# One run of same-parity fine ghost columns in a single tangential dimension of a
# coarse→fine face, with its coarse column range and 3-point tangential stencil.
# The `coarse` range holds the padded center-column J of each fine column in the
# run; `offs`/`w` are the node offsets and weights about J.
struct _TangClass{T}
    fine::StepRange{Int,Int}
    coarse::UnitRange{Int}
    offs::NTuple{3,Int}
    w::NTuple{3,T}
end

# Partition one tangential dimension of a coarse→fine face into descriptor
# classes. q is the fine block's quadrant bit within the coarse neighbor
# (0 = low half, 1 = high half): fine ghost column j (1-based, padded j+1) maps
# to coarse column J = (q·nt + j + 1) >> 1 (padded J+1). The two fine columns at
# the coarse block's tangential extreme share its first/last column and take the
# one-sided (shifted) stencil; all interior columns take the centered one.
function _tangential_classes(nt::Int, q::Int, ::Type{T}) where {T}
    ξm, ξp = -T(1) / 4, T(1) / 4
    half = nt >> 1
    ctr = (-1, 0, 1)
    if q == 0
        lo = (0, 1, 2)
        return [
            _TangClass{T}(2:2:2, 2:2, lo, _cf_tangential_weights(lo, ξm)),
            _TangClass{T}(3:2:3, 2:2, lo, _cf_tangential_weights(lo, ξp)),
            _TangClass{T}(4:2:nt, 3:(half + 1), ctr, _cf_tangential_weights(ctr, ξm)),
            _TangClass{T}(5:2:(nt + 1), 3:(half + 1), ctr, _cf_tangential_weights(ctr, ξp)),
        ]
    else
        hi = (-2, -1, 0)
        return [
            _TangClass{T}(2:2:(nt - 2), (half + 2):nt, ctr, _cf_tangential_weights(ctr, ξm)),
            _TangClass{T}(3:2:(nt - 1), (half + 2):nt, ctr, _cf_tangential_weights(ctr, ξp)),
            _TangClass{T}(nt:2:nt, (nt + 1):(nt + 1), hi, _cf_tangential_weights(hi, ξm)),
            _TangClass{T}((nt + 1):2:(nt + 1), (nt + 1):(nt + 1), hi, _cf_tangential_weights(hi, ξp)),
        ]
    end
end

# Assemble an N-dimensional range box from a normal-dimension range and one range
# per tangential dimension. Build-time only.
function _cf_box(::Val{N}, d::Int, dr::StepRange{Int,Int}, tdims, tranges) where {N}
    return ntuple(Val(N)) do k
        k == d ? dr : tranges[findfirst(==(k), tdims)]
    end
end

_step1(r::UnitRange{Int}) = first(r):1:last(r)

# Coarse→fine (Martin–Cartwright quadratic interpolation): fill the fine leaf K's
# ghost layer on face (d, side) from the coarse neighbor `cover` and K's own
# first interior layer — g = 5/21·u_own + 5/6·U₁ + (−1/14)·U₂, with U₁/U₂ first
# interpolated to the ghost's tangential position (tensor-product 3-point
# quadratics at ξ = ±1/4). One descriptor per tangential class combination;
# every term reads block interiors only, so the fill is order-independent and
# never touches BC ghosts.
function _emit_interp!(
    interp::Vector{GhostFill{N}}, terms::Vector{SlabTerm{N,T}}, bf::BlockForest{N,T},
    i::Int, K::LeafKey{N}, d::Int, side::Int, cover::LeafKey{N},
) where {N,T}
    forest, n = bf.forest, bf.blocksize
    ci = leaf_index(forest, cover)
    nd = n[d]
    g_n = side == -1 ? 1 : nd + 2            # K's ghost layer
    f_n = side == -1 ? 2 : nd + 1            # K's own first interior layer
    U1_n = side == -1 ? nd + 1 : 2           # coarse first interior layer at the face
    U2_n = side == -1 ? nd : 3               # coarse second interior layer
    w_own, w_U1, w_U2 = _cf_normal_weights(T)
    tdims = Tuple(filter(!=(d), ntuple(identity, Val(N))))
    classlists = map(t -> _tangential_classes(n[t], K.coords[t] & 1, T), tdims)
    for combo in Iterators.product(classlists...)
        fine_t = map(c -> c.fine, combo)
        tfirst = length(terms) + 1
        push!(terms, SlabTerm{N,T}(i, _cf_box(Val(N), d, f_n:1:f_n, tdims, fine_t), w_own))
        for (layer_n, w_layer) in ((U1_n, w_U1), (U2_n, w_U2))
            for taps in Iterators.product(ntuple(_ -> (1, 2, 3), length(tdims))...)
                w = w_layer
                for (k, c) in zip(taps, combo)
                    w *= c.w[k]
                end
                coarse_t = map(taps, combo) do k, c
                    (first(c.coarse) + c.offs[k]):1:(last(c.coarse) + c.offs[k])
                end
                push!(
                    terms,
                    SlabTerm{N,T}(
                        ci, _cf_box(Val(N), d, layer_n:1:layer_n, tdims, coarse_t), w
                    ),
                )
            end
        end
        push!(
            interp,
            _close_fill(
                i, _cf_box(Val(N), d, g_n:1:g_n, tdims, fine_t),
                tfirst, length(terms), _ninterp_terms(Val(N)),
            ),
        )
    end
    return nothing
end

# Close one fill over the terms pushed onto the phase buffer since `tfirst`. The
# row width is a function of N alone (schedule.jl); a mismatch is an emitter bug,
# not a topology case, so it throws rather than recording a ragged row.
function _close_fill(
    dst_block::Int, dst_ranges::NTuple{N,StepRange{Int,Int}},
    tfirst::Int, tlast::Int, nexpected::Int,
) where {N}
    nterms = tlast - tfirst + 1
    nterms == nexpected || throw(
        AssertionError(
            "coarse–fine fill emitted $nterms terms, expected $nexpected for N = $N",
        ),
    )
    return GhostFill{N}(dst_block, dst_ranges, tfirst, tlast)
end

# Fine→coarse (flux-matching restriction): fill the coarse leaf K's ghost layer
# on face (d, side) so its stencil flux through the interface equals the mean of
# the fine-grid fluxes — g = u₁ + 2/2^(N−1) · Σ (u_f1 − g_f) over the fine
# columns under each coarse ghost cell. Reads the abutting fine children's first
# interior layer and their interpolation-filled ghosts (⇒ restriction runs after
# interpolation). One descriptor per abutting fine child; the children tile K's
# ghost slab disjointly. A plain 2^N volume average would leave O(1) truncation
# at the interface (1st-order solutions) — rejected.
function _emit_restrict!(
    restrict::Vector{GhostFill{N}}, terms::Vector{SlabTerm{N,T}},
    cfflux::Vector{CFFluxDescriptor{N}}, bf::BlockForest{N,T},
    i::Int, K::LeafKey{N}, d::Int, side::Int, nbr::LeafKey{N},
) where {N,T}
    forest, n = bf.forest, bf.blocksize
    nd = n[d]
    gC_n = side == -1 ? 1 : nd + 2           # K's ghost layer
    u1_n = side == -1 ? 2 : nd + 1           # K's own first interior layer
    uf_n = side == -1 ? nd + 1 : 2           # fine child's interior layer facing K
    gf_n = side == -1 ? nd + 2 : 1           # fine child's interp-filled ghost facing K
    tdims = Tuple(filter(!=(d), ntuple(identity, Val(N))))
    wf = _cf_flux_weight(Val(N), T)
    facing = side == 1 ? 0 : 1               # child d-bit on the face shared with K
    for child in children(nbr)
        (child.coords[d] & 1) == facing || continue
        is_leaf(forest, child) || _throw_unbalanced(K)
        cj = leaf_index(forest, child)
        dst_t = map(tdims) do t
            lo = 2 + (child.coords[t] & 1) * (n[t] >> 1)
            lo:1:(lo + (n[t] >> 1) - 1)
        end
        tfirst = length(terms) + 1
        push!(terms, SlabTerm{N,T}(i, _cf_box(Val(N), d, u1_n:1:u1_n, tdims, dst_t), one(T)))
        for parities in Iterators.product(ntuple(_ -> (0, 1), length(tdims))...)
            fine_t = map(tdims, parities) do t, p
                (2 + p):2:(n[t] + p)
            end
            push!(terms, SlabTerm{N,T}(cj, _cf_box(Val(N), d, uf_n:1:uf_n, tdims, fine_t), wf))
            push!(terms, SlabTerm{N,T}(cj, _cf_box(Val(N), d, gf_n:1:gf_n, tdims, fine_t), -wf))
        end
        push!(
            restrict,
            _close_fill(
                i, _cf_box(Val(N), d, gC_n:1:gC_n, tdims, dst_t),
                tfirst, length(terms), _nrestrict_terms(Val(N)),
            ),
        )
        # The same child walk also records the weight-free coarse–fine-face topology
        # the Diffusion coarse-ghost rewrite consumes (see CFFluxDescriptor) — one
        # emission site, so the two dst/u1 boxes can never disagree.
        push!(
            cfflux,
            CFFluxDescriptor{N}(
                Int32(i), Int32(cj), Int32(d), Int32(uf_n), Int32(gf_n),
                _cf_box(Val(N), d, gC_n:1:gC_n, tdims, dst_t),
                _cf_box(Val(N), d, u1_n:1:u1_n, tdims, dst_t),
            ),
        )
    end
    return nothing
end

#--------------------------------------------------------------------------------# Schedule build + cache

# Build the exchange schedule: for every leaf face, classify against the 2:1
# balance's three cases — same-level neighbor (copy), one-coarser neighbor
# (coarse→fine interpolation), one-finer neighbors (fine→coarse restriction).
# Domain-boundary faces (`face_neighbor === nothing`) land in the per-(dim, side)
# `bcfaces` lists driving the forest-level physical-BC passes below. Each ghost
# region is the dst of exactly one descriptor;
# same-level iteration order (Morton leaf, dim, side) matches the retired `Val{D}`
# sweep's execution order, so uniform-forest behavior is bit-identical.
function _build_exchange_schedule(bf::BlockForest{N,T}) where {N,T}
    forest, h, n = bf.forest, bf.halo, bf.blocksize
    forest.uniform[] || _validate_coarse_fine(bf)
    copies = CopyDescriptor{N}[]
    interp, interp_terms = GhostFill{N}[], SlabTerm{N,T}[]
    restrict, restrict_terms = GhostFill{N}[], SlabTerm{N,T}[]
    cfflux = CFFluxDescriptor{N}[]
    bcfaces = ntuple(_ -> (Int[], Int[]), Val(N))
    for (i, K) in enumerate(forest.leaves), d in 1:N, side in (-1, 1)
        nbr = face_neighbor(forest, K, d, side)
        if nbr === nothing
            push!(bcfaces[d][side == -1 ? 1 : 2], i)
            continue
        end
        if is_leaf(forest, nbr)
            ghost, source = _face_slabs(side, h[d], n[d])
            push!(
                copies,
                CopyDescriptor{N}(
                    leaf_index(forest, nbr), i,
                    _face_box(d, source, h, n), _face_box(d, ghost, h, n),
                ),
            )
        else
            cover = leaf_covering(forest, nbr)
            if cover !== nothing                    # K is the finer side
                cover.level == K.level - 1 || _throw_unbalanced(K)
                _emit_interp!(interp, interp_terms, bf, i, K, d, side, cover)
            else                                    # K is the coarser side
                _emit_restrict!(restrict, restrict_terms, cfflux, bf, i, K, d, side, nbr)
            end
        end
    end
    return ExchangeSchedule{N,T}(
        copies, interp, interp_terms, restrict, restrict_terms, cfflux, bcfaces,
        forest.generation[],
    )
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

#--------------------------------------------------------------------------------# Halo sweeps

"""
    halo_update!(x::AbstractBlockField, g::BlockForest) -> x

Fill each leaf block's interface ghosts from its neighbors, in three phases over
the precomputed per-generation [`ExchangeSchedule`](@ref): same-level slab copies,
then coarse→fine quadratic interpolation, then fine→coarse flux-matching
restriction (which reads the interpolation-filled fine ghosts). Domain-boundary
faces are left to the forest-level `apply_bc!` face pass. Must run once over the whole forest
before any stencil sweep.

On a [`PackedBlockField`](@ref) with a GPU backend the phases run as batched
descriptor kernels over the flattened device schedule — bit-identical to the
reference loops except corner ghost cells of the copy phase (the batched
per-dim launches let the last dim win where the host order interleaves), which
no axis-aligned stencil reads and adjoints require to be zero.
"""
function halo_update!(x::AbstractBlockField, g::BlockForest)
    _require_current(x)
    _run_exchange!(x, g, _exchange_schedule(g))
    return x
end

# Forward-exchange execution seam — device layouts override per (layout, backend)
# with batched descriptor kernels (transfer_kernels.jl); the reference runs
# per-descriptor broadcasts.
_run_exchange!(x::AbstractBlockField, g::BlockForest, sched::ExchangeSchedule) =
    _run_exchange_host!(x, sched)

_run_exchange_host!(x::AbstractBlockField, sched::ExchangeSchedule) =
    _exchange_storage!(_storage(x), _layout(x), sched)

"""
    _exchange_storage!(store, lay::BlockLayout, sched::ExchangeSchedule)

Run the three forward exchange phases over raw block storage. This is the **Enzyme
rule seam** for the inter-block halo: `store` is a bare block vector or packed
array, `lay` a singleton, and `sched` a `Const` descriptor struct — none of them
mixes GC-tracked pointers with inline floats, which a rule argument may not do from
Julia 1.12 on (EnzymeAD/Enzyme.jl#2707). Keeping the seam here rather than at
`halo_update!` is also what lets one rule cover both storage layouts.

Restriction runs last because it reads the interpolation-filled fine ghosts; no
other cross-phase dependency exists.
"""
function _exchange_storage!(store, lay::BlockLayout, sched::ExchangeSchedule)
    _run_copies!(store, lay, sched.copies)
    _run_fills!(store, lay, sched.interp, sched.interp_terms)
    _run_fills!(store, lay, sched.restrict, sched.restrict_terms)
    return nothing
end

# Function barrier: specializing on the concrete storage type and layout singleton
# makes each `_leaf_view` one concrete SubArray, so the loop is dispatch- and
# allocation-free. `view .= view` broadcasts on GPU arrays without scalar indexing.
function _run_copies!(store, lay::BlockLayout, copies::Vector{CopyDescriptor{N}}) where {N}
    for c in copies
        _leaf_view(store, lay, c.dst, c.dst_ranges) .= _leaf_view(store, lay, c.src, c.src_ranges)
    end
    return nothing
end

# `fills` and `terms` are the phase's flat isbits buffers: each fill names its CSR
# row `tfirst:tlast` into `terms`, so no per-fill record wider than a dst box is
# ever copied (96 bytes in 3D against 1752 for a 19-term row held inline). That
# is a structural choice, not a measured one — this loop times the same either
# way, the broadcasts dominating the descriptor walk. Seed with term 1,
# accumulate in term order — the arithmetic the device CSR kernel reproduces
# term-for-term.
function _run_fills!(
    store, lay::BlockLayout, fills::Vector{GhostFill{N}}, terms::Vector{SlabTerm{N,T}}
) where {N,T}
    for f in fills
        dst = _leaf_view(store, lay, f.dst_block, f.dst_ranges)
        t1 = terms[f.tfirst]
        dst .= t1.weight .* _leaf_view(store, lay, t1.block, t1.ranges)
        for k in (f.tfirst + 1):f.tlast
            tk = terms[k]
            dst .+= tk.weight .* _leaf_view(store, lay, tk.block, tk.ranges)
        end
    end
    return nothing
end

"""
    halo_update_adjoint!(x::AbstractBlockField, g::BlockForest) -> x

Exact discrete adjoint of [`halo_update!`](@ref): the same schedule with every
descriptor's roles transposed — each ghost region is scatter-added into its
source cells with the forward weights, then zeroed — with the phases, and the
descriptors within each phase, run in exact reverse order (the transpose of a
composition is the reversed composition of transposes). Mirrors
[`fold_bc!`](@ref) across block faces — the transpose half needed for the
adjoint identity on a forest.

Per-leaf stencil adjoints must leave corner ghosts exactly zero (true for
axis-aligned stencils, whose transposes never scatter there): same-level face
slabs span the full transverse extent, so a nonzero corner cotangent would flow
into a diagonal neighbor's interior. A future cross-derivative leaf needs a
corner-aware exchange.
"""
function halo_update_adjoint!(x::AbstractBlockField, g::BlockForest)
    _require_current(x)
    _exchange_storage_adjoint!(_storage(x), _layout(x), _exchange_schedule(g))
    return x
end

"""
    _exchange_storage_adjoint!(store, lay::BlockLayout, sched::ExchangeSchedule)

Exact transpose of [`_exchange_storage!`](@ref) over raw block storage, and the
reverse body of its Enzyme rule. Phases and, within each phase, descriptors run in
exact reverse order — the transpose of a composition is the reversed composition of
transposes.
"""
function _exchange_storage_adjoint!(store, lay::BlockLayout, sched::ExchangeSchedule)
    _run_fills_adjoint!(store, lay, sched.restrict, sched.restrict_terms)
    _run_fills_adjoint!(store, lay, sched.interp, sched.interp_terms)
    _run_copies_adjoint!(store, lay, sched.copies)
    return nothing
end

# Transposed reverse-order run: fold each ghost slab into its source, zero it.
function _run_copies_adjoint!(
    store, lay::BlockLayout, copies::Vector{CopyDescriptor{N}}
) where {N}
    for c in Iterators.reverse(copies)
        ghost = _leaf_view(store, lay, c.dst, c.dst_ranges)
        _leaf_view(store, lay, c.src, c.src_ranges) .+= ghost
        fill!(ghost, zero(eltype(ghost)))
    end
    return nothing
end

# Fills in reverse, terms within a fill in forward order — the scatter-adds of
# one fill collide on shared source cells, so the term order is part of the
# bit-exact contract.
function _run_fills_adjoint!(
    store, lay::BlockLayout, fills::Vector{GhostFill{N}}, terms::Vector{SlabTerm{N,T}}
) where {N,T}
    for f in Iterators.reverse(fills)
        dst = _leaf_view(store, lay, f.dst_block, f.dst_ranges)
        for k in f.tfirst:f.tlast
            tk = terms[k]
            _leaf_view(store, lay, tk.block, tk.ranges) .+= tk.weight .* dst
        end
        fill!(dst, zero(eltype(dst)))
    end
    return nothing
end

#--------------------------------------------------------------------------------# Forest-level physical-BC passes

# Physical-BC ghost work on a forest, driven by the schedule's bcfaces lists
# instead of per-leaf grid BCs (leaf grids are all-Interface). Reusing the
# single-grid per-face primitives (_fill_ghost!/_fold_ghost!/_offset_ghost!) keeps
# layer indexing, corner ordering, and signs identical to the single-grid sweeps.

"""
    apply_bc!(x::AbstractBlockField, g::BlockForest) -> x

Fill the physical domain-boundary ghosts of every boundary-touching leaf from the
per-generation face lists (see [`ExchangeSchedule`](@ref)) — the forest
counterpart of the single-grid homogeneous [`apply_bc!`](@ref). Dimensions fill
in order `1:N`, so corner ghosts are consistent ghost-of-ghost values. Runs after
[`halo_update!`](@ref) and before the per-leaf stencils.
"""
function apply_bc!(x::AbstractBlockField, g::BlockForest)
    _require_current(x)
    _run_bc!(x, g, _exchange_schedule(g))
    return x
end

# Physical-BC execution seam, mirroring _run_exchange!.
_run_bc!(x::AbstractBlockField, g::BlockForest, sched::ExchangeSchedule) =
    _run_bc_host!(x, g, sched)

_run_bc_host!(x::AbstractBlockField, g::BlockForest, sched::ExchangeSchedule) =
    _bc_storage!(_storage(x), _layout(x), g.bc, sched.bcfaces, g.halo, g.blocksize)

"""
    _bc_storage!(store, lay::BlockLayout, bcs, faces, halo, sz)

Forest physical-BC ghost fill over raw block storage — the Enzyme rule seam for the
face pass, split out of the field struct for the same Julia 1.12 reason as
[`_exchange_storage!`](@ref). The grid arrives as its already-separated isbits
pieces (`bcs`, `halo`, `sz`) plus the per-generation face lists.
"""
function _bc_storage!(store, lay::BlockLayout, bcs::Tuple, faces::Tuple, halo::Tuple, sz::Tuple)
    _fill_bcfaces_dims!(store, lay, bcs, faces, halo, sz, Val(1))
    return nothing
end

function _fill_bcfaces_dims!(
    store, lay::BlockLayout, bcs::Tuple, faces::Tuple, halo::Tuple, sz::Tuple, ::Val{D}
) where {D}
    lo, hi = first(bcs)
    flo, fhi = first(faces)
    h, n = first(halo), first(sz)
    for k in 1:h
        for i in flo
            _fill_ghost!(
                _leaf_array(store, lay, i), Val(D), h + 1 - k, lo, _source_low(lo, h, n, k)
            )
        end
        for i in fhi
            _fill_ghost!(
                _leaf_array(store, lay, i), Val(D), h + n + k, hi, _source_high(hi, h, n, k)
            )
        end
    end
    return _fill_bcfaces_dims!(
        store, lay, Base.tail(bcs), Base.tail(faces), Base.tail(halo), Base.tail(sz), Val(D + 1)
    )
end
_fill_bcfaces_dims!(_, ::BlockLayout, ::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}, ::Val) =
    nothing

"""
    fold_bc!(x̄::AbstractBlockField, g::BlockForest) -> x̄

Exact discrete adjoint of the forest-level [`apply_bc!`](@ref): fold each physical
ghost back into its mirror source with the same sign, then zero it — dimensions in
reverse order `N:1`, transposing the fill. Runs after the per-leaf adjoint gathers
(which leave ghost cotangents in place) and before [`halo_update_adjoint!`](@ref).
"""
function fold_bc!(x̄::AbstractBlockField, g::BlockForest)
    _require_current(x̄)
    sched = _exchange_schedule(g)
    _bc_storage_adjoint!(_storage(x̄), _layout(x̄), g.bc, sched.bcfaces, g.halo, g.blocksize)
    return x̄
end

"""
    _bc_storage_adjoint!(store, lay::BlockLayout, bcs, faces, halo, sz)

Exact transpose of [`_bc_storage!`](@ref) over raw block storage, and the reverse
body of its Enzyme rule.
"""
function _bc_storage_adjoint!(
    store, lay::BlockLayout, bcs::Tuple, faces::Tuple, halo::Tuple, sz::Tuple
)
    _fold_bcfaces_dims!(store, lay, bcs, faces, halo, sz, Val(1))
    return nothing
end

function _fold_bcfaces_dims!(
    store, lay::BlockLayout, bcs::Tuple, faces::Tuple, halo::Tuple, sz::Tuple, ::Val{D}
) where {D}
    _fold_bcfaces_dims!(
        store, lay, Base.tail(bcs), Base.tail(faces), Base.tail(halo), Base.tail(sz), Val(D + 1)
    )
    lo, hi = first(bcs)
    flo, fhi = first(faces)
    h, n = first(halo), first(sz)
    for k in 1:h
        for i in flo
            _fold_ghost!(
                _leaf_array(store, lay, i), Val(D), h + 1 - k, lo, _source_low(lo, h, n, k)
            )
        end
        for i in fhi
            _fold_ghost!(
                _leaf_array(store, lay, i), Val(D), h + n + k, hi, _source_high(hi, h, n, k)
            )
        end
    end
    return nothing
end
_fold_bcfaces_dims!(_, ::BlockLayout, ::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}, ::Val) =
    nothing

# Forest counterpart of the single-grid fill_bc_inhomogeneous!: write the affine
# ghost offsets of every physical boundary face into a ZEROED BlockField. Δ is the
# leaf's own spacing — a refined boundary leaf has halved Δ in the Neumann
# (2k-1)·Δ·flux offsets.
function fill_bc_inhomogeneous!(z::AbstractBlockField, g::BlockForest)
    _require_current(z)
    sched = _exchange_schedule(g)
    _offset_bcfaces_dims!(z, g, g.bc, sched.bcfaces, Val(1))
    return z
end

function _offset_bcfaces_dims!(
    z::AbstractBlockField, bf::BlockForest, bcs::Tuple, faces::Tuple, ::Val{D}
) where {D}
    lo, hi = first(bcs)
    flo, fhi = first(faces)
    h, n = bf.halo[D], bf.blocksize[D]
    keys = bf.forest.leaves
    for k in 1:h
        for i in flo
            Δ = _leaf_spacing(bf, keys[i].level)[D]
            _offset_ghost!(_block_array(z, i), Val(D), h + 1 - k, lo, Δ, k)
        end
        for i in fhi
            Δ = _leaf_spacing(bf, keys[i].level)[D]
            _offset_ghost!(_block_array(z, i), Val(D), h + n + k, hi, Δ, k)
        end
    end
    return _offset_bcfaces_dims!(z, bf, Base.tail(bcs), Base.tail(faces), Val(D + 1))
end
_offset_bcfaces_dims!(::AbstractBlockField, ::BlockForest, ::Tuple{}, ::Tuple{}, ::Val) = nothing
