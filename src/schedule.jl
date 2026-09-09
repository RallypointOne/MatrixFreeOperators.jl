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
    GhostFill{N,T,K}

One coarse–fine ghost sub-slab fill: `dst .= Σₖ wₖ · srcₖ` over the `K` `terms`.
The same concrete shape serves both directions of a 2:1 interface — coarse→fine
quadratic interpolation and fine→coarse flux-matching restriction. Each dst
region is written by exactly one descriptor, so the adjoint is the same-weight
scatter-add into every term source followed by zeroing the dst — forward and
adjoint share their weights by construction.

`K` is fixed per fill kind by `N` alone, never by topology (see
[`_ninterp_terms`](@ref) / [`_nrestrict_terms`](@ref)), so `terms` is an
`NTuple` and the descriptor is fully isbits: a `Vector{GhostFill}` is one flat
inline buffer, the term loop unrolls statically, and the record is a shape
Enzyme's type analysis can see through (issue #26 territory otherwise).
"""
struct GhostFill{N,T,K}
    dst_block::Int
    dst_ranges::NTuple{N,StepRange{Int,Int}}
    terms::NTuple{K,SlabTerm{N,T}}
end

"""
    _ninterp_terms(::Val{N}) -> 1 + 2·3^(N-1)
    _nrestrict_terms(::Val{N}) -> 1 + 2^N

Term count of every coarse→fine interpolation / fine→coarse restriction
[`GhostFill`](@ref), a function of the dimension only. Interpolation reads the
fine block's own first interior cell plus a 3-point tangential tensor stencil on
each of two coarse layers; restriction reads the coarse first interior layer plus
one (interior, ghost) pair per tangential fine-column parity combination. Both are
fixed by the emitters' `Iterators.product` loops (`_emit_interp!` /
`_emit_restrict!` in transfer.jl), which never branch on topology — a face fill at
a domain boundary or a coarse tangential extreme only changes *which* cells the
terms read (one-sided stencils), never how many. Checked at emission.
"""
_ninterp_terms(::Val{N}) where {N} = 1 + 2 * 3^(N - 1)
_nrestrict_terms(::Val{N}) where {N} = 1 + 2^N

"""
    CFFluxDescriptor{N}

One (coarse face × abutting fine child) record of a 2:1 refinement interface —
the topology input of the `Diffusion` coarse-ghost flux rewrite
(`_cf_flux_rewrite!` in operators/diffusion.jl). Emitted by `_emit_restrict!`
from the same child walk as the solution restriction, so the index arithmetic
cannot fork; empty on a uniform forest. Deliberately carries **no weights**: the
κ-dependent face factors are formed per application from the operator's own
coefficient storage, keeping κ on the AD tape (a κ-dependent `SlabTerm.weight`
would be hidden by the halo Enzyme rules, which report zero derivative for
schedule weights — the coefficient-gradient failure mode the package forbids).
Fully isbits, so a loop over `Vector{CFFluxDescriptor}` is a shape Enzyme can
type-analyze inside a differentiated apply (the issue-#26 constraint; `GhostFill`
has since been made isbits too, #42). The per-parity fine boxes are reconstructed
at use from `d`, the layer indices, and the blocksize.
"""
struct CFFluxDescriptor{N}
    coarse::Int32
    fine::Int32
    d::Int32                              # face-normal dimension
    uf_n::Int32                           # fine interior layer facing the coarse block
    gf_n::Int32                           # fine interp-filled ghost layer facing it
    gC::NTuple{N,StepRange{Int,Int}}      # coarse ghost sub-slab (the rewrite dst)
    u1::NTuple{N,StepRange{Int,Int}}      # coarse first-interior box, same tangential
end

"""
    ExchangeSchedule{N,T,KI,KR}

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
consumes. `cfflux` is not a sweep phase: it is the weight-free
coarse–fine-face topology list ([`CFFluxDescriptor`](@ref)) the `Diffusion`
leaf's κ-weighted coarse-ghost rewrite consumes after the exchange, riding the
same build pass and generation key rather than adding a second schedule. `bcfaces` lists, per dimension and (low, high) side, the leaves whose
face lies on the physical domain boundary — the input of the forest-level
physical-BC passes ([`apply_bc!`](@ref)/[`fold_bc!`](@ref)/
`fill_bc_inhomogeneous!` on a [`BlockField`](@ref)); periodic dimensions stay
empty, their wrap being halo-exchange territory.

Every descriptor vector has an isbits eltype — `KI`/`KR` are the fixed
interpolation/restriction term counts (`_ninterp_terms`/`_nrestrict_terms`), so the
sweeps walk flat inline buffers with no per-term pointer chase. The concrete
schedule type is a function of `(N, T)` alone ([`_schedule_type`](@ref)), which is
what lets a [`BlockForest`](@ref) hold it in a concretely typed `Ref`.

Built by `_build_exchange_schedule` and cached on the grid per regrid generation
by `_exchange_schedule`; a stale schedule is unusable by construction because the
cache keys on the live `forest.generation[]`.
"""
struct ExchangeSchedule{N,T,KI,KR}
    copies::Vector{CopyDescriptor{N}}
    interp::Vector{GhostFill{N,T,KI}}
    restrict::Vector{GhostFill{N,T,KR}}
    cfflux::Vector{CFFluxDescriptor{N}}
    bcfaces::NTuple{N,NTuple{2,Vector{Int}}}
    generation::Int
end

"""
    _schedule_type(::Val{N}, ::Type{T}) -> Type{ExchangeSchedule{N,T,KI,KR}}

The one concrete [`ExchangeSchedule`](@ref) type a forest of dimension `N` and
element type `T` ever builds — the term counts are dimension-fixed, so the type
parameters are derivable rather than discovered per build.
"""
_schedule_type(::Val{N}, ::Type{T}) where {N,T} =
    ExchangeSchedule{N,T,_ninterp_terms(Val(N)),_nrestrict_terms(Val(N))}

# Sentinel: generation -1 never matches a live forest generation (construction
# already bumps it to ≥ 1), so the first _exchange_schedule fetch always builds.
_empty_schedule(::Val{N}, ::Type{T}) where {N,T} = ExchangeSchedule(
    CopyDescriptor{N}[],
    GhostFill{N,T,_ninterp_terms(Val(N))}[],
    GhostFill{N,T,_nrestrict_terms(Val(N))}[],
    CFFluxDescriptor{N}[],
    ntuple(_ -> (Int[], Int[]), Val(N)), -1,
)

#--------------------------------------------------------------------------------# Device schedule (flattened descriptor SoA)

# Flattened, device-resident twin of an ExchangeSchedule for the batched
# single-launch exchange of packed fields on GPU backends (transfer_kernels.jl).
# Isbits AoS records in device vectors — Int32 indices (descriptor metadata is
# bandwidth), ranges flattened to first/step/len scalars, and the
# GhostFill.terms tuples concatenated per phase in host fill order with each
# fill carrying its embedded CSR row [tfirst, tlast] (fixed-width rows now that
# K is dimension-fixed; the CSR shape stays so the kernel needs no K
# parameter). Copy boxes need no shape fields: for normal dim d the box is
# h[d] × the full padded transverse extent, recomputed at launch. Built lazily by _device_schedule, cached on the forest
# keyed on generation AND backend; _NoDeviceSchedule is the unbuilt sentinel
# behind the abstract-eltype Ref (only GPU paths ever touch it).
abstract type _AbstractDeviceSchedule end
struct _NoDeviceSchedule <: _AbstractDeviceSchedule end

struct _DevCopy{N}
    src::Int32
    dst::Int32
    src_first::NTuple{N,Int32}
    dst_first::NTuple{N,Int32}
end

struct _DevFill{N}
    dst::Int32
    first::NTuple{N,Int32}
    step::NTuple{N,Int32}
    len::NTuple{N,Int32}
    tfirst::Int32
    tlast::Int32
end

struct _DevTerm{N,T}
    block::Int32
    first::NTuple{N,Int32}
    step::NTuple{N,Int32}
    weight::T
end

struct _DeviceSchedule{N,T,VC,VF,VT,VI} <: _AbstractDeviceSchedule
    copies::VC                       # sorted by normal dim, host order within a dim
    copy_offsets::Vector{Int}        # host, length N+1: dim d = offsets[d]+1:offsets[d+1]
    interp::VF
    interp_terms::VT
    interp_maxcells::Int             # host: bounds-mask ndrange for the fill kernel
    restrict::VF
    restrict_terms::VT
    restrict_maxcells::Int
    bcfaces::NTuple{N,NTuple{2,VI}}  # device Int32 leaf lists per (dim, side)
    generation::Int
end

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

# Fine→coarse flux-matching weight 2/2^(N−1): with H = 2h, equating the coarse
# face flux to the area-weighted sum of the abutting fine fluxes leaves the
# factor h^(N−2)/H^(N−2) = 2^(2−N). Single-sourced so the solution restriction
# (_emit_restrict!) and the Diffusion κ-weighted coarse-ghost rewrite
# (_cf_flux_rewrite!) cannot drift apart — constant κ must reduce one to the other.
_cf_flux_weight(::Val{N}, ::Type{T}) where {N,T} = T(2) / (1 << (N - 1))
