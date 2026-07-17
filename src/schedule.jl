#--------------------------------------------------------------------------------# Exchange schedule

"""
    CopyDescriptor{N}

One same-level face copy of the inter-block halo exchange: block `dst`'s ghost
slab `dst_ranges` is assigned from block `src`'s boundary-interior slab
`src_ranges`. The adjoint runs the same descriptor with the roles transposed —
the ghost slab is scatter-added into the source slab, then zeroed.
"""
struct CopyDescriptor{N}
    src::Int
    dst::Int
    src_ranges::NTuple{N,UnitRange{Int}}
    dst_ranges::NTuple{N,UnitRange{Int}}
end

"""
    SlabTerm{N,T}

One weighted gather term of a coarse–fine ghost fill: a slab window of one source
block and a scalar weight. Step-2 ranges encode the 2:1 fine↔coarse column
parity — every fine column of one parity maps to a consecutive run of coarse
columns.
"""
struct SlabTerm{N,T}
    block::Int
    ranges::NTuple{N,StepRange{Int,Int}}
    weight::T
end

"""
    GhostFill{N,T}

One coarse–fine ghost sub-slab fill: `dst .= Σₖ wₖ · srcₖ` over `terms`. The same
concrete shape serves both directions of a 2:1 interface — coarse→fine quadratic
interpolation and fine→coarse flux-matching restriction. Each dst region is
written by exactly one descriptor, so the adjoint is the same-weight scatter-add
into every term source followed by zeroing the dst — forward and adjoint share
their weights by construction.
"""
struct GhostFill{N,T}
    dst_block::Int
    dst_ranges::NTuple{N,StepRange{Int,Int}}
    terms::Vector{SlabTerm{N,T}}
end

"""
    ExchangeSchedule{N,T}

Halo-exchange plan for one forest generation, replacing every runtime topology
query in [`halo_update!`](@ref) with flat homogeneous descriptor vectors:
`copies` (same-level faces), `interp` (coarse→fine quadratic ghost
interpolation), `restrict` (fine→coarse flux-matching restriction). The forward
sweep runs `copies → interp → restrict` — restriction reads the
interpolation-filled fine ghosts, and no other cross-phase dependency exists.
The adjoint runs the phases, and each phase's descriptors, in exact reverse
order, making it the exact transpose of the forward composition. Each ghost
region is the dst of exactly one descriptor across all three vectors. The
descriptor list is also the send/recv list a future distributed backend
consumes.

Built by `_build_exchange_schedule` and cached on the grid per regrid generation
by `_exchange_schedule`; a stale schedule is unusable by construction because the
cache keys on the live `forest.generation[]`.
"""
struct ExchangeSchedule{N,T}
    copies::Vector{CopyDescriptor{N}}
    interp::Vector{GhostFill{N,T}}
    restrict::Vector{GhostFill{N,T}}
    generation::Int
end

# Sentinel: generation -1 never matches a live forest generation (construction
# already bumps it to ≥ 1), so the first _exchange_schedule fetch always builds.
_empty_schedule(::Val{N}, ::Type{T}) where {N,T} =
    ExchangeSchedule{N,T}(CopyDescriptor{N}[], GhostFill{N,T}[], GhostFill{N,T}[], -1)

#--------------------------------------------------------------------------------# Coarse–fine transfer weights

# Quadratic Lagrange weights on nodes (a, b, c) evaluated at ξ. Quadratic
# exactness of every weight set below is what preserves 2nd-order convergence
# across refinement interfaces; these ARE the multigrid transfer stencils
# (Restriction/Prolongation extract them — issue #11).
function _lagrange3(a, b, c, ξ)
    return (
        (ξ - b) * (ξ - c) / ((a - b) * (a - c)),
        (ξ - a) * (ξ - c) / ((b - a) * (b - c)),
        (ξ - a) * (ξ - b) / ((c - a) * (c - b)),
    )
end

# Normal-direction parabola of the coarse→fine ghost fill (Martin–Cartwright), in
# coarse-cell units with the interface at 0: through the fine block's own first
# interior cell (−1/4) and the coarse neighbor's first two interior layers
# (1/2, 3/2), evaluated at the fine ghost center (1/4) → (5/21, 5/6, −1/14).
_cf_normal_weights(::Type{T}) where {T} = _lagrange3(-T(1) / 4, T(1) / 2, T(3) / 2, T(1) / 4)

# Tangential 3-point quadratic at ξ = ±1/4 (a fine column sits a quarter coarse
# cell off its coarse column's center). `offs` are the coarse node offsets:
# (−1, 0, 1) centered in the interior; shifted one-sided at the coarse block's
# tangential extremes so coarse–fine fills only ever read block interiors.
_cf_tangential_weights(offs::NTuple{3,Int}, ξ::T) where {T} =
    _lagrange3(T(offs[1]), T(offs[2]), T(offs[3]), ξ)
