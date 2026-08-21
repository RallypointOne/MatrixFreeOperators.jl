#--------------------------------------------------------------------------------# AMR driver: indicator-driven regrid + solution transfer

"""
    regrid!(u::BlockField, more::BlockField...; refine, coarsen=Returns(false))
        -> u′ | (u′, more′...)

Adapt the forest to the current solution and carry the field(s) across the regrid.
Evaluates the criteria on every leaf of `u` — each receives the leaf block as an
ordinary [`Field`](@ref) (data plus leaf grid), so a criterion is typically a
reduction like `b -> maximum(abs, interior(b)) > τ` — then edits the topology in a
single pass (refine-marked leaves split, complete fully-marked sibling families
coarsen, refine wins conflicts), re-establishes 2:1 balance, and returns freshly
allocated field(s) on the new leaf set. Refine marks at `maxlevel` and coarsen
marks on level-0 or incomplete families are silently ignored, matching
[`refine!`](@ref)/[`coarsen!`](@ref).

Solution transfer is interior-only and keyed by leaf identity: an unchanged leaf
is copied, a refined leaf is filled by the field's regrid-transfer policy from
its old parent, and a coarsened leaf by the conservative `2⁻ᴺ` mean of its old
children. The policy is per field, resolved from the field's type at transfer
time: [`Interpolated`](@ref) (the default) is the linear-exact per-dimension
interpolation — second-order, right for indicators and coefficients, **not**
mean-preserving; [`Conservative`](@ref) and [`SlopeLimited`](@ref) use the
cell-conservative reconstruction, preserving `Σ V·u` to roundoff across every
regrid — balance-induced refinements included. Conservation is deliberately not
a default: mark conserved state explicitly (the `transfer` keyword of
[`scalar_field`](@ref)/[`vector_field`](@ref), or [`with_transfer`](@ref)) so
indicators, coefficients, and state do not silently share one policy. Ghost
cells of the returned fields are zero — they are scratch, filled by
`halo_update!`/`apply_bc!` on the next operator application — so boundary data
re-enters through the next solve.

Additional fields passed as `more` ride the same transfer (results are returned in
input order). Because a regrid invalidates every field allocated on the old leaf
set, any field needed afterwards — coefficients, or a precomputed indicator —
must be passed here; compute indicator fields *before* calling.

When the marks change nothing, the forest generation is untouched and the *input*
field(s) are returned unchanged — existing [`prepare`](@ref)d operators remain
valid. After a real regrid, stale fields and `PreparedForest`s throw on use;
re-run [`prepare`](@ref) on the returned field.

### Examples

```julia
base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (32, 32);
                     bc=((Dirichlet(), Dirichlet()), (Dirichlet(), Dirichlet())))
bf   = BlockForest(base; blocksize=(8, 8), maxlevel=3)
u    = scalar_field(bf)

# canonical adaptive solve loop
for cycle in 1:ncycles
    L = laplacian(bf)
    P = prepare(L, u)
    b = flatten(f) .- flatten(boundary_rhs(L, u))
    sol, stats = Krylov.cg(P, b)
    flat_to_interior!(u, sol)

    η = magnitude_of(gradient(bf) * u)          # indicator, computed pre-regrid
    η, u = regrid!(η, u; refine  = b -> maximum(interior(b)) > τ,
                         coarsen = b -> maximum(interior(b)) < τ / 10)
end
```

See also: [`refine!`](@ref), [`coarsen!`](@ref), [`prepare`](@ref).
"""
function regrid!(
    u::BlockField, more::BlockField...; refine::FR, coarsen::FC=Returns(false)
) where {FR,FC}
    bf = u.grid
    _require_current(u)
    for f in more
        f.grid === bf ||
            throw(ArgumentError("all fields passed to regrid! must share u's BlockForest"))
        _require_current(f)
    end
    all(iseven, bf.blocksize) || throw(
        ArgumentError(
            "regrid! solution transfer requires an even blocksize in every dimension, " *
            "got $(bf.blocksize)",
        ),
    )
    refine_marks, coarsen_marks = _regrid_marks(u, bf, refine, coarsen)
    forest = bf.forest
    gen0 = forest.generation[]
    # Snapshot before mutating: `_set_leaves!` empties `forest.index` in place and
    # re-sorts the leaf vector, so the key→index map must be copied and transfer
    # must be keyed by LeafKey, never by (unstable) integer position. Block arrays
    # are captured by reference — topology edits never touch field storage.
    old_index = copy(forest.index)
    fields = (u, more...)
    old_blocks = map(f -> f.blocks, fields)
    _regrid_topology!(forest, refine_marks, coarsen_marks)
    forest.generation[] == gen0 && return isempty(more) ? u : fields
    out = map(fields, old_blocks) do f, blocks
        _transfer!(_fresh_field(f, bf), blocks, old_index, bf)
    end
    return isempty(more) ? out[1] : out
end

# Solution transfer is vector-of-blocks only; a packed field anywhere in the call
# degrades to an error, never a wrong result.
function regrid!(::AbstractBlockField, ::AbstractBlockField...; kwargs...)
    throw(
        ArgumentError(
            "regrid! operates on the reference BlockField layout; unpack packed " *
            "fields, regrid, then re-pack",
        ),
    )
end

# Evaluate the per-leaf criteria on the current leaf set. Coarsen evaluation is
# skipped for refine-marked leaves (refine wins) and for level-0 leaves (nothing
# to coarsen into), so the two mark sets are disjoint by construction.
function _regrid_marks(
    u::BlockField, bf::BlockForest{N}, refine::FR, coarsen::FC
) where {N,FR,FC}
    forest = bf.forest
    refine_marks = Set{LeafKey{N}}()
    coarsen_marks = Set{LeafKey{N}}()
    for (i, key) in enumerate(forest.leaves)
        b = block(u, i, leaf_grid(bf, i))
        if refine(b)
            push!(refine_marks, key)
        elseif key.level > 0 && coarsen(b)
            push!(coarsen_marks, key)
        end
    end
    return refine_marks, coarsen_marks
end

# One combined topology pass: split refine-marked leaves, collapse complete
# fully-coarsen-marked sibling families — completeness judged against the same
# leaf set the criteria saw — then a single `_set_leaves!` commit and one
# `balance!`. Mirrors the bodies of `refine!`/`coarsen!` but avoids the two
# intermediate regrids (and generation bumps) sequential calls would cost.
function _regrid_topology!(
    forest::Forest{N}, refine_marks::Set{LeafKey{N}}, coarsen_marks::Set{LeafKey{N}}
) where {N}
    keys = Set(forest.leaves)
    for key in forest.leaves
        if key in refine_marks && key.level < forest.maxlevel
            delete!(keys, key)
            for c in children(key)
                push!(keys, c)
            end
        end
    end
    families = Dict{LeafKey{N},Vector{LeafKey{N}}}()
    for key in forest.leaves
        key in coarsen_marks || continue
        push!(get!(families, parent_key(key), LeafKey{N}[]), key)
    end
    for (p, sibs) in families
        # A refine-marked sibling is never coarsen-marked, so an incomplete count
        # already covers the refine-wins conflict rule.
        length(sibs) == 1 << N || continue
        for s in sibs
            delete!(keys, s)
        end
        push!(keys, p)
    end
    _set_leaves!(forest, keys)
    balance!(forest)
    return forest
end

# A zeroed field matching `f`'s location, transfer policy, eltype, and device on
# the current leaf set; the BlockField constructor stamps the post-regrid
# generation. The policy must survive this round-trip — a returned field with a
# silently-reset policy would conserve on the first regrid and not the second.
function _fresh_field(f::BlockField{L,P}, bf::BlockForest) where {L,P}
    backend = KernelAbstractions.get_backend(bf)
    psize = bf.blocksize .+ 2 .* bf.halo
    blocks = [KernelAbstractions.zeros(backend, eltype(f), psize...) for _ in 1:nleaves(bf)]
    return BlockField{L,P}(blocks, bf)
end

# Fill every new leaf from the snapshot by key relation: same key → copy, old
# parent → prolong, old children → average. Marks never enter here — `balance!`
# refines leaves that were never marked, and can undo a coarsen (same-key copy).
# A single regrid! pass cannot move any region by more than one level: the marks
# were evaluated on a 2:1-balanced leaf set, marked leaves move exactly one
# level, and the balance fixpoint only refines a leaf whose neighborhood already
# changed by one level, so no leaf ends more than one level from the leaf that
# covered its region. The fallthrough guards that argument.
function _transfer!(
    new::BlockField, old_blocks::Vector{A}, old_index::Dict{LeafKey{N},Int},
    bf::BlockForest{N},
) where {A<:AbstractArray,N}
    n, h = bf.blocksize, bf.halo
    # Resolved from the field's type once per field, here at transfer time — the
    # per-field policy the map over the varargs tuple specializes on, so no
    # operator path (and no per-leaf branch) ever consults it.
    pol = _transfer_policy(new)
    for (i, K) in enumerate(bf.forest.leaves)
        dst = new.blocks[i]
        oi = get(old_index, K, 0)
        if oi > 0
            _transfer_copy!(dst, old_blocks[oi], h, n)
            continue
        end
        pi = K.level > 0 ? get(old_index, parent_key(K), 0) : 0
        if pi > 0
            q = ntuple(d -> K.coords[d] & 1, Val(N))
            _transfer_prolong!(pol, dst, old_blocks[pi], q, h, n)
            continue
        end
        kids = children(K)
        if all(c -> haskey(old_index, c), kids)
            for c in kids
                q = ntuple(d -> c.coords[d] & 1, Val(N))
                _transfer_average!(dst, old_blocks[old_index[c]], q, h, n)
            end
            continue
        end
        error(
            "regrid! transfer found no pre-regrid leaf within one level of $K; " *
            "a single regrid cannot change any region by more than one level — " *
            "this is a bug in the regrid driver",
        )
    end
    return new
end

function _transfer_copy!(
    dst::AbstractArray{T,N}, src::AbstractArray{T,N}, h::NTuple{N,Int}, n::NTuple{N,Int}
) where {T,N}
    ir = ntuple(d -> (h[d] + 1):(h[d] + n[d]), Val(N))
    view(dst, ir...) .= view(src, ir...)
    return nothing
end

# Value of the per-dimension linear interpolant of the old parent block at one
# fine destination cell. `F` is the fine index in the parent's doubled child
# space, so the covering parent cell and the ξ = ∓1/4 side follow from its
# parity for either octant. Where the (1/4)-neighbor tap would leave the parent
# interior — destination cells on the parent-block edge — the tap flips to
# one-sided extrapolation (5/4, −1/4) from the inside neighbor, exact on
# linears, keeping the transfer interior-only (old ghosts are never valid).
@inline function _transfer_prolong_at(
    u::AbstractArray{T,N}, I::CartesianIndex{N}, q::NTuple{N,Int}, h::NTuple{N,Int},
    n::NTuple{N,Int}, w::NTuple{2,W},
) where {T,N,W}
    taps = ntuple(Val(N)) do d
        F = q[d] * n[d] + (I[d] - h[d])
        c = (F + 1) >> 1
        s = isodd(F) ? -1 : 1
        cn = c + s
        1 <= cn <= n[d] ? (c, cn, w[1], w[2]) : (c, c - s, one(W) + w[2], -w[2])
    end
    acc = zero(W) * zero(T)
    @inbounds for t in CartesianIndices(ntuple(_ -> 0:1, Val(N)))
        wt = prod(ntuple(d -> t[d] == 0 ? taps[d][3] : taps[d][4], Val(N)))
        J = CartesianIndex(ntuple(d -> h[d] + (t[d] == 0 ? taps[d][1] : taps[d][2]), Val(N)))
        acc += wt * u[J]
    end
    return acc
end

function _transfer_prolong!(
    ::Interpolated, dst::AbstractArray{T,N}, src::AbstractArray{T,N}, q::NTuple{N,Int},
    h::NTuple{N,Int}, n::NTuple{N,Int},
) where {T,N}
    w = _prolong_weights(_scalar_eltype(T))
    ir = CartesianIndices(ntuple(d -> (h[d] + 1):(h[d] + n[d]), Val(N)))
    view(dst, ir) .= _transfer_prolong_at.(Ref(src), ir, Ref(q), Ref(h), Ref(n), Ref(w))
    return nothing
end

# Scalar minmod, applied componentwise to SVector eltypes: the slope both
# children share, clamped to zero across an extremum.
@inline function _minmod(a::T, b::T) where {T<:Number}
    return ifelse(a * b > zero(a * b), ifelse(abs(a) <= abs(b), a, b), zero(a))
end
@inline _minmod(a::SVector, b::SVector) = _minmod.(a, b)

# Per-dim slope of the cell-conservative reconstruction. The Interpolated
# stencil's conservation defect is its side-biased slopes — the two children of
# an interior parent cell each lean on their own neighbor, leaving the child
# mean off by ⅛·δ²u per dim. One SHARED slope per parent cell fixes that for
# any slope value, which is also what makes the limited variant free.
@inline _interior_slope(::Conservative, um, uc, up) = (up - um) / 2
@inline _interior_slope(::SlopeLimited, um, uc, up) = _minmod(up - uc, uc - um)
# Parent-block edges (the transfer is interior-only, so only the inside neighbor
# exists): the one-sided difference is exact on linears and mean-preserving like
# any shared slope; the limited policy drops it to zero — with a single candidate
# there is nothing to limit against, and boundedness is its contract.
@inline _edge_slope(::Conservative, diff) = diff
@inline _edge_slope(::SlopeLimited, diff) = zero(diff)

# Cell-conservative linear reconstruction at one fine destination cell:
# u_child = u_parent + Σ_d ξ_d·σ_d with ξ_d = ∓1/4 by child parity. The mean of
# the 2ᴺ children telescopes Σ_d σ_d·mean(ξ_d) = 0 exactly for ANY σ, including
# at block and physical boundaries — conservation is structural, not a weight
# identity. Additive (2N+1 taps), so exact on linears, not multilinears; the
# named σ_d is the seam the limiter clamps.
@inline function _transfer_ccl_at(
    pol, u::AbstractArray{T,N}, I::CartesianIndex{N}, q::NTuple{N,Int}, h::NTuple{N,Int},
    n::NTuple{N,Int},
) where {T,N}
    W = _scalar_eltype(T)
    F = ntuple(d -> q[d] * n[d] + (I[d] - h[d]), Val(N))
    c = ntuple(d -> (F[d] + 1) >> 1, Val(N))
    Jc = CartesianIndex(ntuple(d -> h[d] + c[d], Val(N)))
    uc = @inbounds u[Jc]
    acc = uc
    @inbounds for d in 1:N
        ξ = isodd(F[d]) ? -W(1) / 4 : W(1) / 4
        δ = _unitindex(Val(N), d)
        σ = if c[d] == 1
            _edge_slope(pol, u[Jc + δ] - uc)
        elseif c[d] == n[d]
            _edge_slope(pol, uc - u[Jc - δ])
        else
            _interior_slope(pol, u[Jc - δ], uc, u[Jc + δ])
        end
        acc += ξ * σ
    end
    return acc
end

function _transfer_prolong!(
    pol::Union{Conservative,SlopeLimited}, dst::AbstractArray{T,N},
    src::AbstractArray{T,N}, q::NTuple{N,Int}, h::NTuple{N,Int}, n::NTuple{N,Int},
) where {T,N}
    ir = CartesianIndices(ntuple(d -> (h[d] + 1):(h[d] + n[d]), Val(N)))
    view(dst, ir) .= _transfer_ccl_at.(Ref(pol), Ref(src), ir, Ref(q), Ref(h), Ref(n))
    return nothing
end

# Conservative coarsening of one old child block into its octant `q` of the new
# parent: each destination cell is the 2⁻ᴺ mean of the 2ᴺ fine cells it covers,
# accumulated as one strided-view broadcast per fine parity.
function _transfer_average!(
    dst::AbstractArray{T,N}, src::AbstractArray{T,N}, q::NTuple{N,Int}, h::NTuple{N,Int},
    n::NTuple{N,Int},
) where {T,N}
    half = ntuple(d -> n[d] >> 1, Val(N))
    dstv = view(
        dst,
        ntuple(d -> (h[d] + q[d] * half[d] + 1):(h[d] + q[d] * half[d] + half[d]), Val(N))...,
    )
    w = _scalar_eltype(T)(1) / (1 << N)
    fill!(dstv, zero(T))
    for p in CartesianIndices(ntuple(_ -> 0:1, Val(N)))
        srcv = view(
            src, ntuple(d -> (h[d] + 1 + p[d]):2:(h[d] + n[d] - 1 + p[d]), Val(N))...
        )
        dstv .+= w .* srcv
    end
    return nothing
end
