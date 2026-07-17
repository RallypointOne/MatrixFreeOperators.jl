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
    ExchangeSchedule{N}

Halo-exchange plan for one forest generation: a flat homogeneous vector of
[`CopyDescriptor`](@ref)s replacing every runtime topology query in
[`halo_update!`](@ref). Each ghost slab appears as the `dst` of exactly one
descriptor, in the fixed order (Morton leaf, dim, side) — the adjoint's
fold-then-zero relies on this. Coarse–fine interpolation/restriction descriptors
extend the schedule as separate homogeneous vectors (later phase), so every sweep
loop stays concretely typed. The descriptor list is also the send/recv list a
future distributed backend consumes.

Built by `_build_exchange_schedule` and cached on the grid per regrid generation
by `_exchange_schedule`; a stale schedule is unusable by construction because the
cache keys on the live `forest.generation[]`.
"""
struct ExchangeSchedule{N}
    copies::Vector{CopyDescriptor{N}}
    generation::Int
end

# Sentinel: generation -1 never matches a live forest generation (construction
# already bumps it to ≥ 1), so the first _exchange_schedule fetch always builds.
_empty_schedule(::Val{N}) where {N} = ExchangeSchedule{N}(CopyDescriptor{N}[], -1)
