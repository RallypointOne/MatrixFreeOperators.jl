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
                terms, i, _cf_box(Val(N), d, g_n:1:g_n, tdims, fine_t),
                tfirst, _ninterp_terms(Val(N)),
            ),
        )
    end
    return nothing
end

# Close one fill over the terms pushed onto the phase buffer since `tfirst`, and
# check the three invariants the sweeps and the device kernel index by (see the
# `GhostFill` docstring). All three are emitter bugs rather than topology cases, so
# they throw at schedule build — never as an out-of-bounds read, a shape error
# inside a broadcast, or (for the disjointness one) a silently wrong answer:
#
#   1. the row is `nexpected` terms wide (a function of N alone, schedule.jl),
#   2. every term window has the dst slab's shape, cell for cell,
#   3. every term window is disjoint from the dst slab.
#
# (3) is what lets the fused gather read all K windows while writing the dst in one
# broadcast: the term views ride an immutable `Ref` wrapper, which takes them out of
# Base's `broadcast_unalias` machinery, so nothing else would notice a dst that
# aliased a source. No emitter produces one — dst boxes are ghost layers and the
# terms read interiors — but the fusion is what makes it load-bearing.
function _close_fill(
    terms::Vector{SlabTerm{N,T}}, dst_block::Int,
    dst_ranges::NTuple{N,StepRange{Int,Int}}, tfirst::Int, nexpected::Int,
) where {N,T}
    tlast = length(terms)
    nterms = tlast - tfirst + 1
    nterms == nexpected || throw(
        AssertionError(
            "coarse–fine fill emitted $nterms terms, expected $nexpected for N = $N",
        ),
    )
    len = length.(dst_ranges)
    for k in tfirst:tlast
        t = terms[k]
        length.(t.ranges) == len || throw(
            AssertionError(
                "coarse–fine fill term $(k - tfirst + 1) reads a $(length.(t.ranges)) " *
                "window but the dst slab is $len; every term must match it cell for cell",
            ),
        )
        _boxes_overlap(dst_block, dst_ranges, t.block, t.ranges) && throw(
            AssertionError(
                "coarse–fine fill term $(k - tfirst + 1) reads block $(t.block) " *
                "$(t.ranges), which overlaps the dst slab $dst_block $dst_ranges; a " *
                "fill's sources must be disjoint from the cells it writes",
            ),
        )
    end
    return GhostFill{N}(dst_block, dst_ranges, tfirst, tlast)
end

# Do two index boxes of the same storage share a cell? Build-time only.
function _boxes_overlap(
    b1::Int, r1::NTuple{N,StepRange{Int,Int}}, b2::Int, r2::NTuple{N,StepRange{Int,Int}}
) where {N}
    b1 == b2 || return false
    for d in 1:N
        isempty(intersect(r1[d], r2[d])) && return false
    end
    return true
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
                terms, i, _cf_box(Val(N), d, gC_n:1:gC_n, tdims, dst_t),
                tfirst, _nrestrict_terms(Val(N)),
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
function _exchange_storage!(store, lay::BlockLayout, sched::ExchangeSchedule{N}) where {N}
    _run_copies!(store, lay, sched.copies)
    _run_fills!(store, lay, sched.interp, sched.interp_terms, _interp_width(Val(N)))
    _run_fills!(store, lay, sched.restrict, sched.restrict_terms, _restrict_width(Val(N)))
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

# The phase row widths as compile-time values. Both are functions of N alone
# (schedule.jl), so `Val(N)` in, `Val(K)` out: dispatching the gather on K
# specializes it over two values per dimension, not over a dynamic loop bound.
@inline _interp_width(::Val{N}) where {N} = Val(_ninterp_terms(Val(N)))
@inline _restrict_width(::Val{N}) where {N} = Val(_nrestrict_terms(Val(N)))

# One fused gather pass per fill: `dst[I] = Σₖ wₖ · srcₖ[I]` over the fill's K term
# windows, which all have the dst slab's shape (`_close_fill` enforces it). The
# retired form ran one `dst .= / .+=` broadcast per term — 7 per fill in 2D, 19 in
# 3D — each re-reading and re-writing the same handful of dst cells and each paying
# broadcast setup for them, so the fill phases dominated the exchange on a refined
# forest (#49/#84). Here the K term views are built once per fill and the weighted
# sum is evaluated per cell inside a single broadcast over the dst indices, the same
# `f.(CartesianIndices(dst), Ref(arrays))` shape the stencil leaves use, so the body
# stays array-level and device-agnostic.
#
# `fills`/`terms` stay the phase's flat CSR buffers — the fill record is its dst box
# plus a `[tfirst, tlast]` row (96 bytes in 3D against 1752 for a 19-term row held
# inline) — and the K views are built by recursion over `Val(K)` off `terms[tfirst +
# k - 1]`, closure-free: an `ntuple(Val(K)) do k` over two arrays stops inlining and
# loses the vectorization (`_diff_axes` in operators/diffusion.jl is the same trap).
#
# Term and accumulation order are unchanged: seed with term 1, add term k to the
# running sum, plain `*`/`+` and no fma, so `((w₁s₁ + w₂s₂) + w₃s₃) + …` is what the
# per-term loop computed and what the device CSR kernel (`_fill_kernel!`) reproduces
# term-for-term. It is therefore bit-identical to the retired loop whenever the
# field eltype IS the schedule's weight type `T`. When it is not — a Float32 field
# on a Float64 forest — the retired loop rounded its partial sum into the field's
# eltype K−1 times where this keeps the promoted accumulator and rounds once: about
# one ulp on the affected cells, and in the device kernel's direction, since
# `_fill_kernel!` accumulates the same promoted way. The fusion removes that
# pre-existing host/device divergence rather than introducing one (#84).
function _run_fills!(
    store, lay::BlockLayout, fills::Vector{GhostFill{N}}, terms::Vector{SlabTerm{N,T}},
    ::Val{K},
) where {N,T,K}
    for f in fills
        f.tlast - f.tfirst + 1 == K && checkbounds(Bool, terms, f.tfirst:f.tlast) ||
            _throw_row_width(f, K, length(terms))
        dst = _leaf_view(store, lay, f.dst_block, f.dst_ranges)
        srcs = _term_views(store, lay, terms, f.tfirst, Val(K))
        ws = _term_weights(terms, f.tfirst, Val(K))
        dst .= _gather_at.(CartesianIndices(dst), _AsScalar(srcs), _AsScalar(ws))
    end
    return nothing
end

# The row width licenses the `@inbounds` walk down `terms`, so it is checked per
# fill rather than trusted — once, outside the broadcast.
@noinline _throw_row_width(f::GhostFill, K::Int, nterms::Int) = throw(
    AssertionError(
        "ghost fill row $(f.tfirst):$(f.tlast) is not $K terms inside a $nterms-term " *
        "buffer — the phase's row width is fixed by the dimension",
    ),
)

# Immutable stand-in for `Ref` as a broadcast scalar. `Ref(x)` is a mutable
# `RefValue`: once the gather body is too big to inline into `copyto!` (it is, at 19
# term views) the Ref escapes and is heap-allocated per fill — the whole tuple of
# views, on every fill of every exchange, inside the Enzyme rule seam that must not
# allocate. An immutable `Ref` subtype rides every 0-dimensional `Ref`
# specialization of Base.Broadcast and stays on the stack.
#
# The Adapt rule keeps the wrapped views convertible for a device broadcast: a
# refined `BlockField` of GPU arrays is the one path that reaches this host body on
# a GPU (packed fields take the batched CSR kernels in transfer_kernels.jl, and a
# vector of device arrays is not something a kernel can index). It ships K device
# `SubArray`s as kernel parameters — ~2.4 KB for a 19-term 3D interpolation fill
# against CUDA's 4 KB parameter budget on pre-sm_90 hardware, so it fits, but with
# under 2× of headroom and none to spare for a wider stencil. Untested on GPU here.
struct _AsScalar{X} <: Ref{X}
    x::X
end
@inline Base.getindex(s::_AsScalar) = s.x
Adapt.adapt_structure(to, s::_AsScalar) = _AsScalar(Adapt.adapt(to, s.x))

# The K source windows of one fill as a tuple of concrete SubArrays, and its K
# weights as a tuple — both by recursion down the CSR row from `i` (closure-free;
# see the note above).
@inline _term_views(store, lay::BlockLayout, terms, i::Int, ::Val{0}) = ()
@inline function _term_views(store, lay::BlockLayout, terms, i::Int, ::Val{K}) where {K}
    t = @inbounds terms[i]
    return (
        _leaf_view(store, lay, t.block, t.ranges),
        _term_views(store, lay, terms, i + 1, Val(K - 1))...,
    )
end

@inline _term_weights(terms, i::Int, ::Val{0}) = ()
@inline _term_weights(terms, i::Int, ::Val{K}) where {K} =
    (@inbounds(terms[i].weight), _term_weights(terms, i + 1, Val(K - 1))...)

# Per-cell body of the fused gather: the K-term weighted sum at the dst-local index
# I, left-associated in term order. `@inbounds` is licensed by the shape invariant —
# every term window has the dst slab's shape, and I ranges over that slab.
@inline function _gather_at(
    I::CartesianIndex, srcs::Tuple{Vararg{AbstractArray,K}}, ws::Tuple{Vararg{Number,K}}
) where {K}
    return _gather_terms(I, srcs, ws, Val(K))
end
@inline _gather_terms(I::CartesianIndex, srcs::Tuple, ws::Tuple, ::Val{1}) =
    @inbounds ws[1] * srcs[1][I]
@inline _gather_terms(I::CartesianIndex, srcs::Tuple, ws::Tuple, ::Val{k}) where {k} =
    @inbounds _gather_terms(I, srcs, ws, Val(k - 1)) + ws[k] * srcs[k][I]

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
#
# Deliberately still one scatter-add broadcast per term, unlike the fused forward
# gather, and the slower half of the exchange because of it. Those collisions are
# exactly what blocks the transposition: a dst-centric single pass would accumulate
# into a shared source cell in dst-cell order instead of term order and change the
# roundoff. The bit-identical fused form is a source-centric transposed CSR — a
# second descriptor set built at schedule time, deferred to issue #94. This runs off
# the mul! hot path, in apply_adjoint! and the reverse rule only.
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
