#--------------------------------------------------------------------------------# Distributability guards

# The whitelist behind prepare_distributed. It lives in core, not in the MDLA
# extension, because these guards are the entire defense against a silently
# wrong distributed answer — and an extension-only guard is only tested where a
# GPU is present, which CI is not. Everything here is CPU-reachable.
#
# The rule is the package's "traits default to the weaker claim" invariant: an
# operator is distributable only if it says so. A forgotten declaration falls to
# the ::AbstractOperator fallback and throws, never returns a wrong answer.

"""
    _distributable(L::AbstractOperator) -> Bool

Whether `L`'s ghost needs are met by the slab exchange machinery (internal).

Stencil and pointwise leaves qualify; combinators propagate explicitly. A
field-valued parameter qualifies when it can be sliced onto the slabs — see
[`_partitionable_coeff`](@ref); one that cannot does not, because it stays bound
to the *global* grid and would silently broadcast against a slab interior.
Transfer operators do not, because their factors live on two different grids
that would each need their own consistent partitioning.
"""
_distributable(::Laplacian) = true
# Same shape as Laplacian — a constant-coefficient stencil that reads spacing and
# indices from the grid it is passed, and whose adjoint branches on
# _has_interface. It is also the only whitelisted leaf that is not self-adjoint,
# so it is what makes an AdjointOp node reachable at all.
_distributable(::Derivative) = true
_distributable(::IdentityOp) = true
_distributable(::ScalingOp{<:Number}) = true
_distributable(S::ScalingOp{<:Field}) = _partitionable_coeff(S.coeff)
# An AbstractBlockField coefficient belongs to a forest, not to a slab.
_distributable(::ScalingOp) = false
# Same coefficient requirement as ScalingOp, for the same two reasons — and the
# face averaging needs κ one cell PAST the cut as well, which the padded window
# `_slab_field` slices supplies at construction rather than by exchanging anything.
_distributable(D::Diffusion) = _partitionable_coeff(D.κ)
_distributable(L::Scaled) = _distributable(L.op)
_distributable(L::Added) = _distributable(L.a) && _distributable(L.b)
_distributable(L::AdjointOp) = _distributable(L.op)
_distributable(::AbstractOperator) = false

"""
    _partitionable_coeff(κ) -> Bool

Whether a field-valued operator parameter can be sliced onto slabs (internal).

Requires an undistributed `CartesianGrid` — the layout [`_slab_field`](@ref)
slices — and a **real** element type. Real only
because `adjoint_operator(::ScalingOp{<:Field})` and `_conj_op(::Diffusion)`
build `conj.(κ.data)` on every call (`src/operators/scaling.jl`,
`src/operators/diffusion.jl`), which in the distributed adjoint would allocate a
full coefficient array per partition per Krylov iteration.
"""
function _partitionable_coeff(κ::Field)
    κ.grid isa CartesianGrid || return false
    κ.grid.topology === nothing || return false
    return _scalar_eltype(eltype(κ.data)) <: Real
end
_partitionable_coeff(::Any) = false

# A composition additionally needs both factors on the SAME grid: its
# intermediate is exchanged on the slabs of that one partitioning, and a
# transfer chain's two grids would each need their own consistent cut.
function _distributable(L::Composed)
    (_distributable(L.a) && _distributable(L.b)) || return false
    ga, gb = operator_grid(L.a), operator_grid(L.b)
    return ga === nothing || gb === nothing || _same_grid(ga, gb)
end

"""
    _same_grid(a::AbstractGrid, b::AbstractGrid) -> Bool

Whether two grids describe the same discretization (internal).

Structural rather than `===`, for two reasons — neither of which is "`===` is
wrong today". A `CartesianGrid` is currently isbits, so `===` *is* value equality;
that is a property of the current field set, not a guarantee. The moment
`topology` carries a real object (DESIGN §9's distributed/AMR seam) `===` silently
becomes identity again, and a guard that quietly changes meaning is exactly what
this file exists to prevent.

Second, `device` is deliberately **not** compared: a host grid and its
device-adapted twin describe the same discretization, and a guard that runs on the
host tree should not care which. `local_range` **is** compared, and that is what
keeps a slab from ever comparing equal to the grid it was cut from.
"""
_same_grid(a::AbstractGrid, b::AbstractGrid) = a === b
function _same_grid(a::CartesianGrid{N}, b::CartesianGrid{N}) where {N}
    a === b && return true
    return a.extent == b.extent &&
           a.spacing == b.spacing &&
           a.size == b.size &&
           a.halo == b.halo &&
           a.bc === b.bc &&
           a.local_range == b.local_range &&
           a.topology === b.topology
end

# Why each rejected case is rejected, for an error message that names the reason
# rather than just the type. Anything without an entry gets the generic tail.
_undistributable_reason(::Restriction) =
    "transfer operators compose grids that would each need their own consistent partitioning"
_undistributable_reason(::Prolongation) =
    "transfer operators compose grids that would each need their own consistent partitioning"
_undistributable_reason(S::ScalingOp) =
    "its coefficient must be a Number or a real-eltype Field on an undistributed " *
    "CartesianGrid, so it can be sliced onto the slabs"
# Reaching this means the coefficient failed `_partitionable_coeff`: the cut
# itself is no longer a reason, since `_slab_field`'s padded window carries κ
# across it.
_undistributable_reason(::Diffusion) =
    "its coefficient must be a real-eltype Field on an undistributed CartesianGrid, " *
    "so it can be sliced onto the slabs"
_undistributable_reason(::Advection) =
    "the velocity field is bound to the global grid; it must be partitioned onto the slabs first"
_undistributable_reason(::Gradient) =
    "a rank-changing operator is non-square, so it needs two partition specs; use it inside a composition whose result is scalar"
_undistributable_reason(::Divergence) =
    "a rank-changing operator is non-square, so it needs two partition specs; use it inside a composition whose result is scalar"
_undistributable_reason(L::Composed) =
    _distributable(L.a) && _distributable(L.b) ?
    "its factors live on different grids, which would each need their own consistent partitioning" :
    nothing
_undistributable_reason(::AbstractOperator) = nothing

# Walk to the first undistributable node so the message points at the actual
# culprit rather than at the root of a large tree.
_first_undistributable(L::Scaled) = _first_undistributable(L.op)
_first_undistributable(L::AdjointOp) = _first_undistributable(L.op)
function _first_undistributable(L::Added)
    a = _first_undistributable(L.a)
    return a === nothing ? _first_undistributable(L.b) : a
end
function _first_undistributable(L::Composed)
    _distributable(L) && return nothing
    a = _first_undistributable(L.a)
    a === nothing || return a
    b = _first_undistributable(L.b)
    # Both factors fine individually ⇒ the composition itself is the culprit
    # (mismatched grids), so report the node rather than descending past it.
    return b === nothing ? L : b
end
_first_undistributable(L::AbstractOperator) = _distributable(L) ? nothing : L

function _check_distributable(L::AbstractOperator)
    _distributable(L) && return L
    culprit = something(_first_undistributable(L), L)
    reason = _undistributable_reason(culprit)
    detail = reason === nothing ? "" : " — $reason"
    throw(
        ArgumentError(
            "prepare_distributed cannot distribute $(nameof(typeof(culprit)))$detail. " *
            "Supported: Laplacian, Derivative, IdentityOp, ScalingOp with a Number or " *
            "real-eltype Field coefficient, Diffusion with a real-eltype Field " *
            "coefficient, and their Scaled/Added/Composed/adjoint combinations on one " *
            "grid; got $(sprint(show, L)).",
        ),
    )
end

"""
    _check_one_grid(L::AbstractOperator, g::AbstractGrid) -> L

Check that every grid-bearing operator in `L` lives on `g`, the grid being
partitioned (internal).

`_distributable` cannot express this: it is a predicate on one operator, while a
mismatched pair is only visible from the tree. `_distributable(::Composed)`
covers a composition's two factors; this covers the rest, so that
`Added(laplacian(g), scaling(κ))` with `κ` on some other grid throws instead of
slicing the coefficient onto slabs of a grid it does not live on.
"""
function _check_one_grid(L::AbstractOperator, g::AbstractGrid)
    culprit = _grid_mismatch(L, g)
    culprit === nothing && return L
    throw(
        ArgumentError(
            "prepare_distributed needs every grid-bearing operator on the grid it " *
            "partitions, but $(nameof(typeof(culprit))) is on a different grid. " *
            "Rebuild it on the same grid (matching extent, spacing, size, halo, and " *
            "boundary conditions); got $(sprint(show, L)).",
        ),
    )
end

_grid_mismatch(L::Scaled, g) = _grid_mismatch(L.op, g)
_grid_mismatch(L::AdjointOp, g) = _grid_mismatch(L.op, g)
# A Diffusion reports its OWN grid as `operator_grid`, so the generic method below
# never sees the coefficient's — and the inner constructor accepts a κ from
# anywhere. Both must match: `_slab_field` windows κ by the slab's
# `local_range`, which on another grid lands a wall ghost on κ's interior (a
# larger grid) or past its end (a smaller one). `ScalingOp` needs no such method
# because its `operator_grid` *is* the coefficient's grid.
function _grid_mismatch(D::Diffusion, g)
    return (_same_grid(D.grid, g) && _same_grid(D.κ.grid, g)) ? nothing : D
end
function _grid_mismatch(L::Added, g)
    a = _grid_mismatch(L.a, g)
    return a === nothing ? _grid_mismatch(L.b, g) : a
end
function _grid_mismatch(L::Composed, g)
    a = _grid_mismatch(L.a, g)
    return a === nothing ? _grid_mismatch(L.b, g) : a
end
function _grid_mismatch(L::AbstractOperator, g)
    lg = operator_grid(L)
    return (lg === nothing || _same_grid(lg, g)) ? nothing : L
end

#--------------------------------------------------------------------------------# Adjoint normalization

"""
    _push_adjoints(L::AbstractOperator) -> AbstractOperator

Push `AdjointOp` nodes down toward the leaves, exactly as `Base.adjoint` does
(`adjoint_operator`, `src/operators/algebra.jl`), and return the rewritten tree
(internal).

`prepare` does not recurse into an `AdjointOp` — `_prepare_tree` turns the whole
subtree into a single `PreparedAdjoint` — so `AdjointOp(Composed(a, b))` would
run a per-partition `aᵀ` then `bᵀ` with **no reduction in between**, dropping the
intermediate's `Interface` cotangents on the floor. Silently wrong, and exactly
the failure mode the distributability guards exist to prevent.

Rewriting first turns it into `Composed(bᵀ, aᵀ)`, which the distributed walk
handles node by node. `Base.adjoint` never builds the nested form, but a user can
write `AdjointOp(A * B)` directly.
"""
function _push_adjoints(L::AdjointOp)
    inner = _push_adjoints(L.op)
    a = adjoint_operator(inner)
    # A leaf with no cheaper adjoint reports AdjointOp(inner) — the fixed point.
    # Recursing on it unguarded would not terminate.
    return (a isa AdjointOp && a.op === inner) ? a : _push_adjoints(a)
end
_push_adjoints(L::Added) = Added(_push_adjoints(L.a), _push_adjoints(L.b))
_push_adjoints(L::Scaled) = Scaled(_push_adjoints(L.op), L.α)
_push_adjoints(L::Composed) = Composed(_push_adjoints(L.a), _push_adjoints(L.b))
_push_adjoints(L::AbstractOperator) = L

#--------------------------------------------------------------------------------# Slab localization

"""
    _slab_op(L::AbstractOperator, lg::AbstractGrid) -> AbstractOperator

Rebuild `L` with every field-valued operator parameter restricted to the slab
`lg` (internal).

The slab analogue of `_leaf_op` (`src/operators/forest.jl`), which solves the
same problem one level down for a block forest. Leaves carrying no field
parameter come back **unchanged and identical** — they read the grid they are
*passed*, never the one they hold — so every partition keeps sharing one object
and [`DistLeaf`](@ref) stays concretely typed for them.

Runs per partition, *after* the distributability guards: those check the global
tree, where a coefficient's `operator_grid` still agrees with its siblings'. A
localized `ScalingOp` reports the slab grid while a sibling `Laplacian` still
reports the global one, so re-checking a localized tree would reject it. Check
once, before rewriting.
"""
_slab_op(L::AbstractOperator, ::AbstractGrid) = L
_slab_op(S::ScalingOp{<:Field}, lg::AbstractGrid) = ScalingOp(_slab_field(S.coeff, lg))
# The INNER constructor, deliberately: `diffusion(g, κ)` refuses Interface faces,
# and this is the seam its docstring reserves for supplying cross-block
# coefficient ghosts. The validation it skips already ran on the global operator.
_slab_op(D::Diffusion, lg::AbstractGrid) = Diffusion(lg, _slab_field(D.κ, lg), D.avg)
_slab_op(L::Added, lg::AbstractGrid) = Added(_slab_op(L.a, lg), _slab_op(L.b, lg))
_slab_op(L::Scaled, lg::AbstractGrid) = Scaled(_slab_op(L.op, lg), L.α)
_slab_op(L::Composed, lg::AbstractGrid) = Composed(_slab_op(L.a, lg), _slab_op(L.b, lg))
_slab_op(L::AdjointOp, lg::AbstractGrid) = AdjointOp(_slab_op(L.op, lg))

"""
    _slab_field(f::Field, lg::AbstractGrid) -> Field

The slab window of a global field, **ghosts included**, as a field on `lg`
(internal).

A *restriction*, never a re-evaluation — slicing is exact by construction and
needs nothing from the caller. Slab padded index `p` is global padded index
`first(local_range[d]) - 1 + p`, so one padded window is the whole job, and it
lands every ghost on the value it should hold with no per-face logic:

  - an `Interface` ghost maps onto a global *interior* plane — the neighbour's
    value;
  - a physical ghost on an end slab maps onto a global ghost plane, verbatim;
  - a periodic cut maps onto the global ghost planes too, so a wrap-around
    neighbour needs no special case;
  - a transverse dimension has `local_range == 1:n`, so the window is that
    dimension's whole padded extent.

The window is `first(zr):(last(zr) + 2h) ⊆ 1:(n + 2h)`, so it never needs
clamping. What the ghosts *mean* is the consumer's business, and the two
consumers differ. `ScalingOp` reads its coefficient **pointwise at the cell
being written** — `_coeff_values(c::Field)` is `interior(c)` — so its ghosts are
never consulted and may hold anything. `Diffusion` averages κ to faces, so it
reads κ one cell past every face of the interior, a partition cut included, where
that cell belongs to the neighbour; it needs the `Interface` ghosts to be the
neighbour's κ and the physical ghosts to be the extension `diffusion` already
applied to the global κ — an even mirror at a wall, the wrap under `Periodic`.
Both arrive in the same window, which is why neither consumer costs any
communication of its own. The one precondition — that `f`'s ghosts are already
extended for `f.grid` — is `diffusion`'s to supply, through `_extended_coeff`;
what `_check_one_grid` enforces is that `f.grid` *is* the grid being partitioned,
so the window is never taken with a `local_range` that means nothing on `f`.

Its pullback, should distributed AD ever arrive, is the transpose of the window —
a scatter-add of each partition's `∂/∂κ` into the global coefficient, `Interface`
ghosts landing on the neighbour's interior cells and physical ghosts on the global
ghost planes for `fill_coefficient_ghosts!`'s own transpose to fold. Not
implemented: what blocks distributed AD is the transport, not this rewrite.
"""
function _slab_field(f::Field{L}, lg::AbstractGrid{N}) where {L,N}
    f.grid === lg && return f
    h = halo_width(lg)
    win = ntuple(Val(N)) do d
        r = lg.local_range[d] .- (first(f.grid.local_range[d]) - 1)
        first(r):(last(r) + 2 * h[d])
    end
    return Field{L}(copy(view(f.data, win...)), lg)
end

#--------------------------------------------------------------------------------# Backend primitives (no methods in core)

# The four seams a distributed backend fills in, following the prepare_distributed
# pattern: declared here, implemented by whoever owns the transport. The MDLA
# extension implements them with scatter!/reduce! over MultiDeviceVectors; the CPU
# proof in test/partitioning.jl implements them with plain-Vector global indexing.
# Core therefore never sees a backend type, and both implementations drive the
# *same* walk instead of each emulating it.

"""
    _dist_map!(f, ctx)

Run `f(p)` for every partition `p`, on that partition's device, and return once
all have completed (internal; backend-supplied). The device-context idiom
(`CUDA.device!` inside `@sync`/`@async`, or a plain loop on CPU) is the backend's
business, so core cannot write this loop itself.
"""
function _dist_map! end

"""
    _dist_scatter!(xch, fields, ctx)

Fill the [`Interface`](@ref) halo slabs of every `fields[p]` from its neighbours'
interiors (internal; backend-supplied).

Contract — on entry each `fields[p]` interior is valid; on exit its interior is
unchanged and its `Interface` ghosts hold the neighbour-owned values. Other ghost
cells (physical-BC faces and corners) are left alone: the next `apply!` refills
them via `apply_bc!`. Implementations stage through
[`_unpack_ghosts!`](@ref).

The exact transpose of [`_dist_reduce!`](@ref); the distributed adjoint identity
⟨Lx,y⟩ = ⟨x,Lᵀy⟩ rests on that pairing.
"""
function _dist_scatter! end

"""
    _dist_reduce!(xch, fields, ctx)

Accumulate every `fields[p]`'s [`Interface`](@ref) ghost cotangents into the
partitions that own those cells, then leave each `fields[p]` holding the summed
owned cotangents in its interior and **zeros in its ghosts** (internal;
backend-supplied).

Contract — on entry each `fields[p]` holds interior cotangents plus the
neighbour-owned contributions `fold_bc!` migrated into its `Interface` slabs; on
exit interiors are complete and ghosts are cleared. Clearing is not cosmetic: a
following pointwise leaf writes interiors only, so a leftover contribution would
be packed into the *next* reduction and double-counted at every cut plane.
Implementations stage through [`_pack_local_x!`](@ref).

The exact transpose of [`_dist_scatter!`](@ref). The two differ only on ghost
cells outside the `Interface` slabs (physical-BC faces and corners), which every
consumer annihilates anyway — `adjoint_gather!` opens with `zero_ghosts!` and
`fold_bc!` zeroes what it folds — so the pairing holds on the space that matters.
"""
function _dist_reduce! end

#--------------------------------------------------------------------------------# Distributed operator tree

# The distributed twin of the prepared tree. `prepare` already hoisted every
# buffer a node needs (`PreparedComposed.tmp`, `PreparedAdjoint.scratch`), once
# per partition; these nodes gather those per-partition buffers into one
# structure so the walk visits each structural position exactly once and fires a
# *collective* exchange between the local applies.
#
# This is the distributed analogue of `_forest_capply!`/`_forest_capply_adjoint!`
# in linalg.jl, which solves the same mid-tree-exchange problem for a block
# forest by recursing at the forest level.
#
# Node types are device-independent and concrete: `_prepare_tree` leaves leaf
# operators untouched, and leaves read the grid they are *passed* rather than the
# one they carry, so every partition shares the same leaf object. Only the
# buffers differ, and those are held in per-partition vectors whose element type
# is necessarily abstract (slab grids differ in their BC type parameters by
# construction). One dynamic dispatch per node per partition against a full grid
# sweep is not a cost worth chasing.

abstract type DistNode end

# One operator per partition. They are the SAME object for every leaf that
# carries no field parameter, so `identity.` in `_dist_tree` narrows the vector to
# a concrete element type and `ops[p]` still dispatches statically — the slice-1
# and slice-2a hot path is unchanged. Only a leaf localized by `_slab_op` gets an
# abstract element type (slab grids differ in their BC type parameters by
# construction), which costs one dynamic dispatch per node per partition against a
# full grid sweep — the same trade already taken for the buffer vectors below.
struct DistLeaf{V<:AbstractVector} <: DistNode
    ops::V
end

struct DistAdded{A<:DistNode,B<:DistNode} <: DistNode
    a::A
    b::B
end

struct DistScaled{A<:DistNode,T<:Number} <: DistNode
    node::A
    α::T
end

# `tmps` are the per-partition intermediates; `xch` is the exchange for them, or
# `nothing` when the outer factor reads no ghosts (then the forward scatter and
# the adjoint reduce are both no-ops — see `_reads_ghosts`).
struct DistComposed{A<:DistNode,B<:DistNode,F<:AbstractVector,X} <: DistNode
    a::A
    b::B
    tmps::F
    xch::X
end

# `ins` is this node's own copy of its input. It is not an optimization: the
# adjoint gather opens with `zero_ghosts!` on its input (`adjoint_gather!` in
# operators/abstract.jl), so handing it the enclosing segment's field would
# destroy exchanged ghosts a *sibling* under the same `Added` still needs —
# making the result depend on term order, silently. `outs` is the gather target
# (`PreparedAdjoint.scratch`).
struct DistAdjoint{V<:AbstractVector,FI<:AbstractVector,FO<:AbstractVector,X} <: DistNode
    ops::V
    ins::FI
    outs::FO
    xch::X
end

"""
    _reads_ghosts(node::DistNode) -> Bool

Whether applying `node` forward consumes its input's [`Interface`](@ref) ghosts
(internal).

Used to skip an exchange that would move no information — and it must gate the
forward scatter and the adjoint reduce **together**. The two are equivalent:
reading a neighbour's value forward is exactly what writes a contribution to that
neighbour's cotangent in the transpose. Gating one direction and not the other
breaks the adjoint identity.
"""
# `first` is enough: isdiagonal is a type-level trait, and localization varies a
# leaf's grid type parameters, never its trait.
_reads_ghosts(node::DistLeaf) = !isdiagonal(first(node.ops))
_reads_ghosts(node::DistAdded) = _reads_ghosts(node.a) || _reads_ghosts(node.b)
_reads_ghosts(node::DistScaled) = _reads_ghosts(node.node)
_reads_ghosts(node::DistComposed) = _reads_ghosts(node.b)   # `b` consumes the input
# The forward action is an adjoint gather, which zeroes its input's ghosts and
# reads interiors only; its transpose is a forward `apply!`, which writes
# interiors only. Neither touches the enclosing intermediate's ghosts.
_reads_ghosts(::DistAdjoint) = false

"""
    _dist_tree(nodes, protos, mkxch) -> DistNode

Build the distributed walk tree from the per-partition prepared operator trees
`nodes` (all structurally identical) and their per-partition input field
prototypes `protos` (internal). `mkxch(proto)` is the backend's exchange
constructor, called once per node that needs one.
"""
_dist_tree(nodes::AbstractVector, protos::AbstractVector, mkxch) =
    _dist_tree(first(nodes), nodes, protos, mkxch)

_dist_tree(::AbstractOperator, nodes, protos, mkxch) = DistLeaf(identity.(nodes))

_dist_tree(::Added, nodes, protos, mkxch) = DistAdded(
    _dist_tree([n.a for n in nodes], protos, mkxch),
    _dist_tree([n.b for n in nodes], protos, mkxch),
)

_dist_tree(proto::Scaled, nodes, protos, mkxch) =
    DistScaled(_dist_tree([n.op for n in nodes], protos, mkxch), proto.α)

function _dist_tree(::PreparedComposed, nodes, protos, mkxch)
    tmps = AbstractField[n.tmp for n in nodes]
    b = _dist_tree([n.b for n in nodes], protos, mkxch)
    a = _dist_tree([n.a for n in nodes], tmps, mkxch)
    xch = _reads_ghosts(a) ? mkxch(first(tmps)) : nothing
    return DistComposed(a, b, tmps, xch)
end

function _dist_tree(proto::PreparedAdjoint, nodes, protos, mkxch)
    ins = AbstractField[zero_ghosts!(similar(p)) for p in protos]
    outs = AbstractField[n.scratch for n in nodes]
    xch = isdiagonal(proto.op) ? nothing : mkxch(first(outs))
    return DistAdjoint(identity.([n.op for n in nodes]), ins, outs, xch)
end

# α/β blend of one field's interior into another's — the pattern PreparedAdjoint
# uses in linalg.jl, shared by the distributed nodes that must apply β *after* a
# collective step rather than through it.
function _blend_interior!(y::AbstractField, s::AbstractField, α::Number, β::Number)
    yi = interior(y)
    if iszero(β)
        yi .= α .* interior(s)
    else
        yi .= α .* interior(s) .+ β .* yi
    end
    return y
end

#--------------------------------------------------------------------------------# Forward action

"""
    _dist_capply!(ys, node, xs, ctx, α, β) -> ys

Apply `node` across every partition, exchanging ghosts wherever the tree needs
them (internal).

Contract — on entry every `xs[p]` has a valid interior **and** valid
[`Interface`](@ref) ghosts; on exit every `ys[p]` interior is valid and its
ghosts are undefined. Other ghost cells of an intermediate are left stale on
purpose: every ghost-reading leaf opens with `apply_bc!`, which refills the
physical-BC faces and the corners from the interior.

The transpose is [`_dist_capply_adjoint!`](@ref).
"""
function _dist_capply!(ys, node::DistLeaf, xs, ctx, α, β)
    _dist_map!(ctx) do p
        apply!(ys[p], node.ops[p], xs[p], xs[p].grid, α, β)
    end
    return ys
end

function _dist_capply!(ys, node::DistAdded, xs, ctx, α, β)
    _dist_capply!(ys, node.a, xs, ctx, α, β)
    _dist_capply!(ys, node.b, xs, ctx, α, true)
    return ys
end

_dist_capply!(ys, node::DistScaled, xs, ctx, α, β) =
    _dist_capply!(ys, node.node, xs, ctx, α * node.α, β)

# The mid-tree exchange, and the reason this walk exists: `b`'s result is only
# interior-valid, so `a` would read stale zeros across every cut plane.
function _dist_capply!(ys, node::DistComposed, xs, ctx, α, β)
    _dist_capply!(node.tmps, node.b, xs, ctx, true, false)
    node.xch === nothing || _dist_scatter!(node.xch, node.tmps, ctx)
    _dist_capply!(ys, node.a, node.tmps, ctx, α, β)
    return ys
end

# The mid-tree reduction. `α`/`β` are applied *after* it, never through it:
# a reduction overwrites the owned section before accumulating neighbour
# contributions, so a pending `β·y_old` staged into it would be destroyed.
function _dist_capply!(ys, node::DistAdjoint, xs, ctx, α, β)
    _dist_map!(ctx) do p
        interior(node.ins[p]) .= interior(xs[p])
        zero_ghosts!(node.outs[p])
        apply_adjoint!(node.outs[p], node.ops[p], node.ins[p], node.ins[p].grid)
    end
    node.xch === nothing || _dist_reduce!(node.xch, node.outs, ctx)
    _dist_map!(ctx) do p
        _blend_interior!(ys[p], node.outs[p], α, β)
    end
    return ys
end

#--------------------------------------------------------------------------------# Adjoint action

"""
    _dist_adjoint_segment!(x̄s, node, ȳs, ctx) -> x̄s

Run one adjoint segment — the span between two collective steps — clearing the
target's ghosts exactly once before the walk (internal).

Once, and here, is the only correct placement. Per leaf, the second term of an
`Added` would wipe the first term's `Interface` cotangents before they reach the
reduction. Omitted, a pointwise leaf's interior-only adjoint would leave the
*previous* call's ghosts in place for the accumulating (`β ≠ 0`) sibling to add
in.
"""
function _dist_adjoint_segment!(x̄s, node::DistNode, ȳs, ctx)
    _dist_map!(ctx) do p
        zero_ghosts!(x̄s[p])
    end
    _dist_capply_adjoint!(x̄s, node, ȳs, ctx, true, false)
    return x̄s
end

"""
    _dist_capply_adjoint!(x̄s, node, ȳs, ctx, α, β) -> x̄s

Transpose of [`_dist_capply!`](@ref): the same walk with every scatter replaced
by a reduction and vice versa (internal).

Contract — on entry every `ȳs[p]` interior is valid; on exit every `x̄s[p]` holds
interior cotangents plus the neighbour-owned contributions `fold_bc!` migrated
into its `Interface` slabs, awaiting the caller's reduction. Ghost clearing is
the segment's job, not this function's — see [`_dist_adjoint_segment!`](@ref).
"""
function _dist_capply_adjoint!(x̄s, node::DistLeaf, ȳs, ctx, α, β)
    # No isselfadjoint/isdiagonal shortcut here, deliberately, unlike the forest
    # walk in linalg.jl. `_selfadjoint_grid` is true for any CartesianGrid, so
    # the trait still claims self-adjointness on a slab whose cut faces are
    # Interface; the leaf ignores it and branches on `_has_interface` instead.
    # Taking the shortcut would route to the forward action, whose contract wants
    # scattered ghosts that the adjoint contract does not deliver — and mid-tree
    # it would flip a reduction into a scatter and lose the transpose.
    _dist_map!(ctx) do p
        apply_adjoint!(x̄s[p], node.ops[p], ȳs[p], ȳs[p].grid, α, β)
    end
    return x̄s
end

function _dist_capply_adjoint!(x̄s, node::DistAdded, ȳs, ctx, α, β)
    _dist_capply_adjoint!(x̄s, node.a, ȳs, ctx, α, β)
    _dist_capply_adjoint!(x̄s, node.b, ȳs, ctx, α, true)
    return x̄s
end

# conj, mirroring the forest adjoint in linalg.jl: the forward path scales by α,
# its transpose by conj(α).
_dist_capply_adjoint!(x̄s, node::DistScaled, ȳs, ctx, α, β) =
    _dist_capply_adjoint!(x̄s, node.node, ȳs, ctx, α * conj(node.α), β)

# (a∘b)ᵀ = bᵀ∘aᵀ. The mid reduction is the exact transpose of the forward
# mid scatter, and is gated by the same predicate so the pair stays matched.
function _dist_capply_adjoint!(x̄s, node::DistComposed, ȳs, ctx, α, β)
    _dist_adjoint_segment!(node.tmps, node.a, ȳs, ctx)
    node.xch === nothing || _dist_reduce!(node.xch, node.tmps, ctx)
    _dist_capply_adjoint!(x̄s, node.b, node.tmps, ctx, α, β)
    return x̄s
end

# Transpose of the forward node: its reduction becomes a scatter, and the
# wrapped operator runs forward. Writes interiors only, so it contributes
# nothing to the enclosing segment's ghosts — matching `_reads_ghosts`.
function _dist_capply_adjoint!(x̄s, node::DistAdjoint, ȳs, ctx, α, β)
    _dist_map!(ctx) do p
        interior(node.outs[p]) .= interior(ȳs[p])
    end
    node.xch === nothing || _dist_scatter!(node.xch, node.outs, ctx)
    _dist_map!(ctx) do p
        apply!(x̄s[p], node.ops[p], node.outs[p], node.outs[p].grid, α, β)
    end
    return x̄s
end

#--------------------------------------------------------------------------------# Boundary lift

"""
    _dist_lift_scratch(protos, ctx) -> Vector{AbstractField}

Per-partition zero field carrying the *inhomogeneous* ghost offsets of its own
slab (internal) — the `z` of [`boundary_rhs`](@ref), once per partition.

Read-only for the whole lift walk, and exact **without any exchange**.
`fill_bc_inhomogeneous!` is a no-op on [`Interface`](@ref) faces
(`src/boundaries.jl`) and otherwise depends only on `(bc, spacing, k)`, which a
slab shares with the global grid. So a slab's `z` is the restriction of the
global `z`: nonzero only in physical-boundary ghosts, and zero across every cut
plane — which is right, because the global `z` is zero at those cells too, they
being *interior* cells of the neighbouring partition. An interior slab's leaf
lift is therefore genuinely zero except through its transverse physical faces.

One caveat worth knowing rather than testing: `fill_bc_inhomogeneous!` sweeps
dimensions `1:N` writing full cross-dimensional slices, so a *corner* ghost keeps
whichever dimension wrote last — dimension `N` globally, but dimension 1 on a slab
whose dimension-`N` faces are `Interface`. Harmless here: every whitelisted leaf
steps `±1` along one axis at a time, so no stencil ever reads a corner ghost. A
diagonal or wider stencil would break that silently.
"""
function _dist_lift_scratch(protos, ctx)
    zs = Vector{AbstractField}(undef, length(protos))
    _dist_map!(ctx) do p
        z = similar(protos[p])
        fill!(z.data, zero(eltype(z.data)))
        fill_bc_inhomogeneous!(z.data, z.grid)
        zs[p] = z
    end
    return zs
end

"""
    _dist_boundary_rhs!(bs, node::DistNode, zs, ctx, α, β) -> bs

Assemble the boundary lift of `node` across every partition (internal): the same
recursion [`boundary_rhs`](@ref) runs over a single `Field`, walked over the
`DistNode` tree so a `Composed` node's `apply(a, b_b)` gets the mid-tree ghost
exchange it needs.

Contract — `zs` is the per-partition inhomogeneous-ghost field from
[`_dist_lift_scratch`](@ref) and is never written; on exit every `bs[p]` interior
holds `α·lift + β·bs[p]`.
"""
function _dist_boundary_rhs!(bs, node::DistLeaf, zs, ctx, α, β)
    # _apply_raw!, not apply!: apply! opens with halo_update!/apply_bc!, which
    # would overwrite the inhomogeneous ghost offsets this whole walk exists to
    # sweep. Same seam the single-device boundary_rhs uses.
    _dist_map!(ctx) do p
        _apply_raw!(bs[p], node.ops[p], zs[p], zs[p].grid, α, β)
    end
    return bs
end

function _dist_boundary_rhs!(bs, node::DistAdded, zs, ctx, α, β)
    _dist_boundary_rhs!(bs, node.a, zs, ctx, α, β)
    _dist_boundary_rhs!(bs, node.b, zs, ctx, α, true)
    return bs
end

# No conj here, unlike the adjoint walk above: this is a forward lift, matching
# `boundary_rhs(L::Scaled)` in linalg.jl.
_dist_boundary_rhs!(bs, node::DistScaled, zs, ctx, α, β) =
    _dist_boundary_rhs!(bs, node.node, zs, ctx, α * node.α, β)

# The adjoint action is built homogeneous (gather + fold, no ghost offsets), so
# its lift is identically zero — but β = 0 still has to WRITE that zero, or the
# caller's buffer keeps whatever the previous walk left in it.
function _dist_boundary_rhs!(bs, ::DistAdjoint, zs, ctx, α, β)
    iszero(β) && _dist_map!(ctx) do p
        fill!(interior(bs[p]), zero(eltype(bs[p].data)))
    end
    return bs
end

# Affine composition: a(b(x) + c_b) + c_a = (a∘b)(x) + a(c_b) + c_a. The middle
# term applies `a` to a REAL field, so `c_b`'s cut-plane ghosts have to be
# exchanged first — the slice-2a problem, gated by the same predicate so no
# unmatched exchange is introduced.
#
# Reusing this node's own `tmps`/`xch` is safe and deliberate: they are never
# shared with another node, `_dist_scatter!` fully overwrites what it stages, and
# the lift is assembled once per solve, never concurrently with a `mul!` — the
# same single-threaded contract `prepare` already carries. Afterwards `tmps` holds
# stale *physical* ghosts, which the walk's standing contract already covers:
# every ghost-reading leaf opens with `apply_bc!`.
function _dist_boundary_rhs!(bs, node::DistComposed, zs, ctx, α, β)
    _dist_boundary_rhs!(node.tmps, node.b, zs, ctx, true, false)
    node.xch === nothing || _dist_scatter!(node.xch, node.tmps, ctx)
    _dist_capply!(bs, node.a, node.tmps, ctx, α, β)
    _dist_boundary_rhs!(bs, node.a, zs, ctx, α, true)
    return bs
end

#--------------------------------------------------------------------------------# Distributed source terms

# set! per partition. Exact against the global set! because cell_center evaluates
# at the global cell index (Grids.jl) — that is what makes a slab-assembled
# right-hand side independent of the partition count.
function _dist_set!(fields, fun::F, ctx) where {F}
    _dist_map!(ctx) do p
        set!(fields[p], fun)
    end
    return fields
end
