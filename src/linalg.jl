#--------------------------------------------------------------------------------# Prepared operators (the Krylov boundary)

"""
    PreparedOperator

A linear operator bound to pre-allocated scratch fields, exposing the flat
`mul!`/`size`/`eltype` interface Krylov solvers need. Built with
[`prepare`](@ref); after warm-up, `mul!` runs without steady-state allocations on
single-grid [`Field`](@ref)s. The [`BlockField`](@ref) path uses the sibling
[`PreparedForest`](@ref).

The flat vectors span interior DOFs only — ghost cells are determined by
boundary conditions and `halo_update!`, never solver unknowns. Each `mul!` copies
the flat vector into a halo-padded scratch field, applies the operator, and
copies the interior back out fused with the `α`/`β` axpby, so the solver's
vectors are never mutated by halo or BC fills.

`mul!` writes only the interior of `xpad`, deliberately leaving ghost cells as it
found them. The distributed path ([`prepare_distributed`](@ref)) depends on that
same discipline throughout: it drives the operator tree itself so it can fill
[`Interface`](@ref) ghost slabs from a neighbor exchange between applies, and
anything that zeroed ghosts on the way in would silently discard exchanged data.

Those copies are the price of the Krylov boundary, not of the operator: a
prepared operator also takes fields directly through `apply!(du, P, u)`, the
explicit time-stepping path.

The traits [`islinear`](@ref), [`isconstant`](@ref), [`isselfadjoint`](@ref) and
[`isdiagonal`](@ref) are those of the operator handed to `prepare` — binding
buffers changes nothing about the map. A `PreparedOperator` is the solver
boundary rather than a node of the lazy algebra, so it is not an
[`AbstractOperator`](@ref): it does not enter `+`, `*`, `adjoint`, or `apply`;
compose first, then prepare.
"""
struct PreparedOperator{O<:AbstractOperator,G<:AbstractGrid,FX<:AbstractField,FY<:AbstractField}
    op::O
    grid::G
    xpad::FX
    ypad::FY
end

"""
    PreparedForest

Block-forest counterpart of [`PreparedOperator`](@ref): the flat `mul!`/`size`/
`eltype` boundary for an operator over a [`BlockForest`](@ref). Beyond the padded
scratch fields it carries a scratch [`BlockField`](@ref) for accumulating adjoint
sweeps (`adjscratch`), and `prepare` warms the grid's per-generation halo-exchange
schedule (see [`halo_update!`](@ref)), so the inter-block ghost copies and the
physical-BC face passes run as flat descriptor loops with no topology queries.
Every leaf grid is one concrete all-`Interface` type, so the leaf sweep is
type-stable and rebuilding a leaf grid allocates nothing; the residual cost is the
per-leaf stencil `apply!` itself (allocation-free only when inlined into a single
`mul!`); see the allocation-free-kernel follow-up. The prepared operator is tied
to the forest's regrid `generation`; a `refine!`/`coarsen!`/`balance!` after
`prepare` invalidates it and `mul!` throws — re-run `prepare` on the new forest.
"""
struct PreparedForest{
    O<:AbstractOperator,
    G<:BlockForest,
    FX<:AbstractBlockField,
    FY<:AbstractBlockField,
    S<:AbstractBlockField,
}
    op::O
    grid::G
    xpad::FX
    ypad::FY
    adjscratch::S
    generation::Int
end

const _AnyPrepared = Union{PreparedOperator,PreparedForest}

# A prepared operator is the same linear map as the tree it binds buffers to, so
# its traits are the tree's. Declared explicitly (there is no supertype to fall
# through to) — an undeclared trait must be a MethodError, never a silent
# default the wrapper did not earn.
islinear(P::_AnyPrepared) = islinear(P.op)
isconstant(P::_AnyPrepared) = isconstant(P.op)
isselfadjoint(P::_AnyPrepared) = isselfadjoint(P.op)
isdiagonal(P::_AnyPrepared) = isdiagonal(P.op)

"""
    prepare(L::AbstractOperator, x::Field) -> PreparedOperator
    prepare(L::AbstractOperator) -> PreparedOperator

Walk the operator tree once, allocating the scratch buffers every node needs, and
return a [`PreparedOperator`](@ref) — or a [`PreparedForest`](@ref) for a
[`BlockField`](@ref) prototype. `mul!` is then allocation-free in steady state on a
single grid; on a block forest the residual cost is the per-leaf stencil
apply. `x` is a prototype of the input field (contents are
ignored); the one-argument form assumes a scalar field on the operator's grid.

The prepared operator is stateful and single-threaded — prepare once per
concurrent solve. Requires `islinear(L)`; linearize nonlinear operators first
with [`linearize`](@ref).

`mul!` is the *solver* boundary: every call stages the flat vector into a
halo-padded field and back out again. Explicit integrators do not need that —
step at field level with [`apply!`](@ref) instead.

### Examples

```julia
g = CartesianGrid(((0.0, 1.0),), (64,))
A = prepare(laplacian(g))
b = flatten(set!(scalar_field(g), x -> sin(π * x[1])))
u, stats = Krylov.minres(A, b)

# explicit stepping stays at field level:
uf = set!(scalar_field(g), x -> sin(π * x[1]))
du = similar(uf)
dt = 0.4 * spacing(g)[1]^2              # forward-Euler bound
apply!(du, A, uf)                       # or apply!(du, laplacian(g), uf)
interior(uf) .+= dt .* interior(du)
```
"""
function prepare(L::AbstractOperator, x::AbstractField)
    if !islinear(L)
        throw(
            ArgumentError(
                "prepare requires a linear operator; for nonlinear operators prepare " *
                "the Jacobian linearize(L, u0) instead",
            ),
        )
    end
    xpad = similar(x)
    zero_ghosts!(xpad)
    ypad = allocate_output(L, x)
    return PreparedOperator(_prepare_tree(L, x), x.grid, xpad, ypad)
end
prepare(L::AbstractOperator) = prepare(L, scalar_field(_require_grid(L)))

# Forest prepare: same tree walk. adjscratch backs the accumulating adjoint sweep
# the un-prepared path allocates per call. A PackedBlockField prototype yields
# packed scratch (via similar), routing mul! to the forest-native kernel sweeps.
function prepare(L::AbstractOperator, x::AbstractBlockField)
    if !islinear(L)
        throw(
            ArgumentError(
                "prepare requires a linear operator; for nonlinear operators prepare " *
                "the Jacobian linearize(L, u0) instead",
            ),
        )
    end
    xpad = similar(x)
    zero_ghosts!(xpad)
    ypad = allocate_output(L, x)
    op = _prepare_tree(L, x)
    _exchange_schedule(x.grid)   # warm the halo-exchange cache (build once, not on first mul!)
    return PreparedForest(op, x.grid, xpad, ypad, similar(x), x.grid.forest.generation[])
end

"""
    prepare_distributed(L::AbstractOperator, nparts::Integer; devices=nothing)

Prepare `L` for a distributed multi-device solve: partition its grid into
`nparts` slabs ([`partition_grid`](@ref)), build one [`prepare`](@ref)d operator
per partition on its own device, and return a distributed prepared operator
exposing `mul!`/`size`/`eltype` over device-partitioned vectors, with ghost
slabs exchanged between partitions around each local apply.

Implemented by package extensions; the function has no methods until one is
loaded. The MDLA extension (load MultiDeviceLinearAlgebra.jl, CUDA.jl, and
Krylov.jl) maps slabs onto one CUDA device each — `devices` optionally picks
which (0-indexed, unique) — and returns an operator over MDLA
`MultiDeviceVector`s.

### Examples

```julia
using MultiDeviceLinearAlgebra, CUDA, Krylov
g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (256, 256))
P = prepare_distributed(laplacian(g), 2)
b = assemble_rhs(P, x -> sin(x[1]) * exp(-x[2]))
u, stats = Krylov.cg(P, b)
```

See also: [`assemble_rhs`](@ref), [`local_grids`](@ref), [`boundary_rhs`](@ref).
"""
function prepare_distributed end

"""
    assemble_rhs(P, f) -> distributed vector

Solve-ready right-hand side `f - boundary_rhs(P)` for a **distributed** prepared
operator `P` ([`prepare_distributed`](@ref)), assembled entirely
partition-locally — nothing global is materialized on one device.

`f` is either a function of physical coordinates or one [`Field`](@ref) per
partition on the grids [`local_grids`](@ref) reports. Implemented by package
extensions, like `prepare_distributed`; there is no single-device method, since
a serial right-hand side is the one-liner
`flatten(f) .- flatten(boundary_rhs(L, g))`.

This is the *whole* right-hand side, not a sibling of [`boundary_rhs`](@ref) —
it is the function that consumes one.

### Examples

```julia
P = prepare_distributed(laplacian(g), 2)
b = assemble_rhs(P, x -> sin(x[1]) * exp(-x[2]))
u, stats = Krylov.cg(P, b)
```

See also: [`local_grids`](@ref), [`boundary_rhs`](@ref).
"""
function assemble_rhs end

"""
    local_grids(P) -> Vector{<:AbstractGrid}

The grid each partition of a distributed prepared operator owns, in partition
order.

The escape hatch for source data [`assemble_rhs`](@ref) cannot build from a
coordinate function: allocate a [`Field`](@ref) on one of these, fill it however
you like, and pass the vector of fields to `assemble_rhs`. Each grid records
its span of the global grid in `local_range`, and [`cell_center`](@ref) on it
agrees bitwise with the uncut grid. Implemented by package extensions.
"""
function local_grids end

# Tree-walking buffer allocation: leaves pass through unchanged; Composed and
# AdjointOp nodes are replaced by buffer-carrying twins so steady-state mul! never
# allocates. AdjointOp nodes are first pushed down to the leaves (_push_adjoints
# below), so a PreparedAdjoint never wraps a combinator.
_prepare_tree(L::AbstractOperator, ::AbstractField) = L
# Coefficient layout normalization: a BlockField coefficient under a packed
# prototype is packed once here, so the prepared hot path dispatches to the
# forest-native kernel sweep instead of silently staying on the per-leaf
# fallback. Sound because these leaves are isconstant; staleness after a regrid
# is caught by the PreparedForest generation guard. (Packed coefficients under a
# BlockField prototype need no conversion — block views serve the fallback.)
_prepare_tree(S::ScalingOp{<:BlockField}, ::PackedBlockField) = ScalingOp(pack(S.coeff))
_prepare_tree(L::Advection{<:BlockForest,<:BlockField}, ::PackedBlockField) =
    Advection(L.grid, pack(L.velocity))
# `pack` copies padded storage verbatim, so the coefficient ghosts the leaf already carries
# survive — no re-exchange, and none would be legal here anyway (κ is `isconstant`).
_prepare_tree(D::Diffusion{<:BlockForest,<:BlockField}, ::PackedBlockField) =
    Diffusion(D.grid, pack(D.κ), D.avg)
_prepare_tree(L::Added, x::AbstractField) = Added(_prepare_tree(L.a, x), _prepare_tree(L.b, x))
_prepare_tree(L::Scaled, x::AbstractField) = Scaled(_prepare_tree(L.op, x), L.α)

function _prepare_tree(L::Composed, x::AbstractField)
    pb = _prepare_tree(L.b, x)
    tmp = allocate_output(L.b, x)
    pa = _prepare_tree(L.a, tmp)
    return PreparedComposed(pa, pb, tmp)
end
# Composed on a BlockField uses the generic method above: the PreparedComposed
# tmp is a whole BlockField, and _forest_capply! recurses at the forest level so
# the intermediate gets its inter-block exchange.

"""
    _push_adjoints(L::AbstractOperator) -> AbstractOperator

Push `AdjointOp` nodes down toward the leaves, exactly as `Base.adjoint` does
(`adjoint_operator`, `src/operators/algebra.jl`), and return the rewritten tree
(internal).

`Base.adjoint` never builds the nested form, but a user can write `AdjointOp(A * B)`
directly, and no prepared twin knows how to run the adjoint of a *composition*:
`PreparedComposed` carries one intermediate buffer shaped for the forward pass, and
the distributed walk would run a per-partition `aᵀ` then `bᵀ` with **no reduction
in between**, dropping the intermediate's `Interface` cotangents on the floor.
Rewriting first turns it into `Composed(bᵀ, aᵀ)`, which every walk handles node by
node; only leaves without a cheaper declared adjoint remain wrapped.
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

# Normalize first so the PreparedAdjoint only ever wraps a leaf (see
# _push_adjoints); a rewritten combinator tree re-enters the ordinary walk. Then
# recurse into the leaf so its coefficient layout is normalized too: without
# this, prepare(adjoint(D), packed) would keep a BlockField κ and the adjoint hot
# path would silently stay on the per-leaf fallback. That recursion is
# type-driven only (the coefficient methods above never read x's shape), so the
# prototype's adjoint-vs-forward shape is irrelevant for a leaf.
function _prepare_tree(L::AdjointOp, x::AbstractField)
    pushed = _push_adjoints(L)
    pushed isa AdjointOp || return _prepare_tree(pushed, x)
    return PreparedAdjoint(_prepare_tree(pushed.op, x), allocate_output(pushed, x))
end

# Composed twin holding its concretely-typed intermediate field.
struct PreparedComposed{A<:AbstractOperator,B<:AbstractOperator,F<:AbstractField} <: AbstractOperator
    a::A
    b::B
    tmp::F
end

# Same map as Composed(a, b), so the same trait propagation (algebra.jl): a twin
# that fell through to the `false` defaults would still be *safe*, but by
# accident rather than by declaration, and would hide a linear prepared tree from
# any consumer (SciML caching, the isconstant path) that asks.
islinear(L::PreparedComposed) = islinear(L.a) && islinear(L.b)
isconstant(L::PreparedComposed) = isconstant(L.a) && isconstant(L.b)
isdiagonal(L::PreparedComposed) = isdiagonal(L.a) && isdiagonal(L.b)
isselfadjoint(::PreparedComposed) = false       # not compositional — see Composed

# The inner factor sees the grid of the intermediate it consumes (transfer
# chains compose factors living on different grids).
function apply!(y::Field, L::PreparedComposed, x::Field, g::AbstractGrid, α, β)
    apply!(L.tmp, L.b, x, g)
    apply!(y, L.a, L.tmp, L.tmp.grid, α, β)
    return y
end

# AdjointOp twin: accumulating applications gather into the held scratch with
# β = 0 and blend only the interiors, so an adjoint node writes no ghost of the
# field it shares with its siblings. The distributed walk depends on exactly
# that (`_reads_ghosts(::DistAdjoint) = false`, distributed.jl) and reuses this
# scratch as `DistAdjoint.outs`.
struct PreparedAdjoint{O<:AbstractOperator,F<:AbstractField} <: AbstractOperator
    op::O
    scratch::F
end

# Same map as AdjointOp(op), so the same trait forwarding AND the same declared
# transpose (operators/abstract.jl): the twin's adjoint is the leaf it wraps.
# `adjoint_operator` must be declared with `isdiagonal` — the generic forest
# adjoint walk below turns a diagonal transpose into `adjoint_operator(L)`, and
# the AbstractOperator default would hand back AdjointOp(PreparedAdjoint(op)),
# whose apply is this node's transpose again: recursion without bound.
islinear(L::PreparedAdjoint) = islinear(L.op)
isconstant(L::PreparedAdjoint) = isconstant(L.op)
isselfadjoint(L::PreparedAdjoint) = isselfadjoint(L.op)
isdiagonal(L::PreparedAdjoint) = isdiagonal(L.op)
adjoint_operator(L::PreparedAdjoint) = L.op

function apply!(y::Field, L::PreparedAdjoint, x::Field, g::AbstractGrid, α, β)
    iszero(β) && return apply_adjoint!(y, L.op, x, g, α, β)
    apply_adjoint!(L.scratch, L.op, x, g)
    interior(y) .= α .* interior(L.scratch) .+ β .* interior(y)
    return y
end
# (opᵀ)ᵀ = op, as for AdjointOp.
function apply_adjoint!(x̄::Field, L::PreparedAdjoint, ȳ::Field, g::AbstractGrid, α, β)
    return apply!(x̄, L.op, ȳ, g, α, β)
end

#--------------------------------------------------------------------------------# Cached forest apply (the PreparedForest hot path)

# Cached counterpart of the un-prepared forest apply in operators/forest.jl, using
# the prepared scratch fields (leaf grids are one concrete all-Interface type, so
# rebuilding them in the sweep is type-stable and free). The combinator
# structure mirrors forest.jl exactly — Added/Scaled/AdjointOp recurse at the forest
# level so a nested adjoint still reaches halo_update_adjoint!'s cross-block fold,
# and an Added whose operands all `shares_exchange` fills the halo once — and
# additionally handles the PreparedAdjoint nodes _prepare_tree introduces (which,
# like PreparedComposed, never share an exchange). (The residual per-leaf cost is
# the stencil apply! itself; see the alloc-free-kernel note on PreparedForest.)
function _forest_capply!(
    y::AbstractBlockField, L::AbstractOperator, x::AbstractBlockField, P::PreparedForest, α, β
)
    return _forest_exchange_sweep!(y, L, x, P.grid, α, β)
end
function _forest_capply!(
    y::AbstractBlockField, L::Added, x::AbstractBlockField, P::PreparedForest, α, β
)
    shares_exchange(L) && return _forest_exchange_sweep!(y, L, x, P.grid, α, β)
    _forest_capply!(y, L.a, x, P, α, β)
    _forest_capply!(y, L.b, x, P, α, true)
    return y
end
_forest_capply!(y::AbstractBlockField, L::Scaled, x::AbstractBlockField, P::PreparedForest, α, β) =
    _forest_capply!(y, L.op, x, P, α * L.α, β)
_forest_capply!(
    y::AbstractBlockField, L::AdjointOp, x::AbstractBlockField, P::PreparedForest, α, β
) = _forest_capply_adjoint!(y, L.op, x, P, α, β)

# PreparedComposed: the held tmp is a BlockField; recursing at the forest level
# gives the intermediate its inter-block exchange. The adjoint of a∘b is bᵀ∘aᵀ,
# and tmp doubles as the cotangent intermediate (aᵀȳ has tmp's shape, and tmp is
# dead between applications).
function _forest_capply!(
    y::AbstractBlockField, L::PreparedComposed, x::AbstractBlockField, P::PreparedForest, α, β
)
    _forest_capply!(L.tmp, L.b, x, P, true, false)
    _forest_capply!(y, L.a, L.tmp, P, α, β)
    return y
end
function _forest_capply_adjoint!(
    x̄::AbstractBlockField, L::PreparedComposed, ȳ::AbstractBlockField, P::PreparedForest, α, β
)
    _forest_capply_adjoint!(L.tmp, L.a, ȳ, P, true, false)
    _forest_capply_adjoint!(x̄, L.b, L.tmp, P, α, β)
    return x̄
end

# PreparedAdjoint: same interior-only discipline as the ::Field method above —
# the accumulating form gathers into the node's own scratch, then blends per leaf.
function _forest_capply!(
    y::AbstractBlockField, L::PreparedAdjoint, x::AbstractBlockField, P::PreparedForest, α, β
)
    iszero(β) && return _forest_capply_adjoint!(y, L.op, x, P, α, β)
    _forest_capply_adjoint!(L.scratch, L.op, x, P, true, false)
    s = L.scratch
    for i in 1:nleaves(P.grid)
        lg = leaf_grid(P.grid, i)
        yi = interior(block(y, i, lg))
        yi .= α .* interior(block(s, i, lg)) .+ β .* yi
    end
    return y
end

# Cached forest adjoint action, mirroring apply_adjoint! in operators/forest.jl. The
# accumulating (β ≠ 0) branch uses the prepared adjscratch in place of similar(x̄).
function _forest_capply_adjoint!(
    x̄::AbstractBlockField, L::AbstractOperator, ȳ::AbstractBlockField, P::PreparedForest, α, β
)
    _require_current(x̄)
    _require_current(ȳ)
    # isselfadjoint is grid-aware (false on a non-uniform forest, whose coarse–fine
    # coupling breaks the halo symmetry), so this shortcut never skips a real transpose.
    isselfadjoint(L) && return _forest_capply!(x̄, L, ȳ, P, α, β)
    # Diagonal transposes are pointwise; the gather + fold below would fold x̄'s
    # ghost scratch into interiors (mirrors the un-prepared path in forest.jl).
    isdiagonal(L) && return _forest_capply!(x̄, adjoint_operator(L), ȳ, P, α, β)
    if iszero(β)
        _forest_adjoint_sweep!(x̄, L, ȳ, P.grid, α)
        fold_bc!(x̄, P.grid)
        halo_update_adjoint!(x̄, P.grid)
    else
        s = P.adjscratch
        _forest_adjoint_sweep!(s, L, ȳ, P.grid, true)
        fold_bc!(s, P.grid)
        halo_update_adjoint!(s, P.grid)
        for i in 1:nleaves(P.grid)
            lg = leaf_grid(P.grid, i)
            xi = interior(block(x̄, i, lg))
            xi .= α .* interior(block(s, i, lg)) .+ β .* xi
        end
    end
    return x̄
end
# Self-adjoint sums take the (exchange-sharing) forward path, as in forest.jl.
function _forest_capply_adjoint!(
    x̄::AbstractBlockField, L::Added, ȳ::AbstractBlockField, P::PreparedForest, α, β
)
    isselfadjoint(L) && return _forest_capply!(x̄, L, ȳ, P, α, β)
    _forest_capply_adjoint!(x̄, L.a, ȳ, P, α, β)
    _forest_capply_adjoint!(x̄, L.b, ȳ, P, α, true)
    return x̄
end
_forest_capply_adjoint!(
    x̄::AbstractBlockField, L::Scaled, ȳ::AbstractBlockField, P::PreparedForest, α, β
) = _forest_capply_adjoint!(x̄, L.op, ȳ, P, α * conj(L.α), β)
_forest_capply_adjoint!(
    x̄::AbstractBlockField, L::AdjointOp, ȳ::AbstractBlockField, P::PreparedForest, α, β
) = _forest_capply!(x̄, L.op, ȳ, P, α, β)
# The prepared twin transposes the same way; without this method a PreparedAdjoint
# would fall into the generic body above, whose sweep has no adjoint action for it.
_forest_capply_adjoint!(
    x̄::AbstractBlockField, L::PreparedAdjoint, ȳ::AbstractBlockField, P::PreparedForest, α, β
) = _forest_capply!(x̄, L.op, ȳ, P, α, β)

#--------------------------------------------------------------------------------# Boundary lift (linear/affine split)

"""
    boundary_rhs(L::AbstractOperator, g::AbstractGrid) -> AbstractField
    boundary_rhs(L::AbstractOperator, x_proto::AbstractField) -> AbstractField

Boundary lift of the affine split `L_full(x) = L(x) + b`: the contribution of
*inhomogeneous* boundary data (Dirichlet values, Neumann fluxes) that
[`apply!`](@ref) deliberately omits so that `L` stays linear (`L(0) = 0`).
Assemble once per solve and fold into the right-hand side: the discrete problem
`L_full(u) = f` becomes `L·u = f - b`. The grid form assumes a scalar input
field; pass a prototype field for vector inputs. On a [`BlockForest`](@ref) the
lift is assembled by the forest-level inhomogeneous face pass — only physical
domain faces contribute; [`Interface`](@ref) faces stay homogeneous.

### Examples

```julia
g = CartesianGrid(((0.0, 1.0),), (64,); bc=((Dirichlet(1.0), Dirichlet(2.0)),))
L = laplacian(g)
b = boundary_rhs(L, g)
rhs = flatten(f) .- flatten(b)        # solve  prepare(L) \\ rhs  with Krylov
```
"""
function boundary_rhs(L::AbstractOperator, g::AbstractGrid)
    return boundary_rhs(L, scalar_field(g))
end

function boundary_rhs(L::AbstractOperator, x_proto::Field)
    if !islinear(L)
        throw(
            ArgumentError(
                "boundary_rhs is the affine lift of a linear operator; $(nameof(typeof(L))) is nonlinear",
            ),
        )
    end
    g = x_proto.grid
    z = similar(x_proto)
    fill!(z.data, zero(eltype(z.data)))
    fill_bc_inhomogeneous!(z.data, g)
    b = allocate_output(L, x_proto)
    _apply_raw!(b, L, z, g, true, false)
    return b
end

function boundary_rhs(L::Added, x_proto::Field)
    ba = boundary_rhs(L.a, x_proto)
    bb = boundary_rhs(L.b, x_proto)
    ba.data .+= bb.data
    return ba
end

function boundary_rhs(L::Scaled, x_proto::Field)
    b = boundary_rhs(L.op, x_proto)
    b.data .*= L.α
    return b
end

# Affine composition: a(b(x) + c_b) + c_a = (a∘b)(x) + a(c_b) + c_a.
function boundary_rhs(L::Composed, x_proto::Field)
    bb = boundary_rhs(L.b, x_proto)
    lift = apply(L.a, bb)
    ba = boundary_rhs(L.a, bb)
    lift.data .+= ba.data
    return lift
end

# The adjoint action is built homogeneous (gather + fold, no ghost offsets), so
# its lift is identically zero.
function boundary_rhs(L::AdjointOp, x_proto::Field)
    b = allocate_output(L, x_proto)
    fill!(b.data, zero(eltype(b.data)))
    return b
end

# Forest lift: the forest-level inhomogeneous face pass writes the ghost offsets
# (Interface ghosts stay zero, so the lift is local to each block), then the raw
# stencil sweeps each leaf.
function boundary_rhs(L::AbstractOperator, x_proto::AbstractBlockField)
    if !islinear(L)
        throw(
            ArgumentError(
                "boundary_rhs is the affine lift of a linear operator; $(nameof(typeof(L))) is nonlinear",
            ),
        )
    end
    g = x_proto.grid
    z = _zero_all!(similar(x_proto))
    fill_bc_inhomogeneous!(z, g)
    b = allocate_output(L, x_proto)
    for i in 1:nleaves(g)
        lg = leaf_grid(g, i)
        _apply_raw!(block(b, i, lg), _leaf_op(L, i, lg), block(z, i, lg), lg, true, false)
    end
    return b
end

# The adjoint action is built homogeneous (gather + fold, no ghost offsets), so
# its lift is identically zero — mirrors the ::Field method above.
boundary_rhs(L::AdjointOp, x_proto::AbstractBlockField) =
    _zero_all!(allocate_output(L, x_proto))

# Forest combinator lifts recurse at the FOREST level (the ::Field methods above
# would run per leaf, where a nested Composed's `apply(L.a, b_b)` misses the
# inter-block exchange its lift field needs).
function boundary_rhs(L::Added, x_proto::AbstractBlockField)
    ba = boundary_rhs(L.a, x_proto)
    bb = boundary_rhs(L.b, x_proto)
    for i in 1:nleaves(ba.grid)
        _block_array(ba, i) .+= _block_array(bb, i)
    end
    return ba
end
function boundary_rhs(L::Scaled, x_proto::AbstractBlockField)
    b = boundary_rhs(L.op, x_proto)
    for i in 1:nleaves(b.grid)
        _block_array(b, i) .*= L.α
    end
    return b
end
# Affine composition on a forest: a(b(x) + c_b) + c_a — the middle term a(c_b)
# applies `a` to a real forest field, whose forest apply performs the exchange.
function boundary_rhs(L::Composed, x_proto::AbstractBlockField)
    bb = boundary_rhs(L.b, x_proto)
    lift = apply(L.a, bb)
    ba = boundary_rhs(L.a, bb)
    for i in 1:nleaves(lift.grid)
        _block_array(lift, i) .+= _block_array(ba, i)
    end
    return lift
end

# A PreparedForest is tied to the forest generation it was built on.
function _require_prepared_current(P::PreparedForest)
    P.generation == P.grid.forest.generation[] || throw(
        ArgumentError(
            "this PreparedForest was built before the forest was regridded " *
            "(refine!/coarsen!/balance!); re-run prepare on the current forest",
        ),
    )
    return nothing
end

# The field-level entry point hands user fields straight to the prepared tree, so a
# field off the prototype would run halo/spacing from P.grid but ghost fills from
# x.grid. Refuse it.
function _require_prepared_match(P::_AnyPrepared, x::AbstractField, y::AbstractField)
    x.grid === P.grid || throw(
        ArgumentError(
            "apply!(y, P, x): x lives on a different grid than the prototype prepare " *
            "was given; prepare the operator on x's grid or pass a field on P.grid",
        ),
    )
    y.grid === P.grid || throw(
        ArgumentError(
            "apply!(y, P, x): y lives on a different grid than the prototype prepare " *
            "was given; allocate y with similar on a field over P.grid",
        ),
    )
    eltype(x) === eltype(P.xpad) || throw(
        ArgumentError(
            "apply!(y, P, x): x has element type $(eltype(x)) but the prepared " *
            "operator was built for $(eltype(P.xpad))",
        ),
    )
    eltype(y) === eltype(P.ypad) || throw(
        ArgumentError(
            "apply!(y, P, x): y has element type $(eltype(y)) but the prepared " *
            "operator produces $(eltype(P.ypad))",
        ),
    )
    return nothing
end

function Base.size(P::_AnyPrepared)
    return (flat_length(P.ypad), flat_length(P.xpad))
end
Base.size(P::_AnyPrepared, d::Integer) = size(P)[d]
Base.eltype(P::_AnyPrepared) = _scalar_eltype(eltype(P.xpad))

function LinearAlgebra.mul!(
    y::AbstractVector, P::PreparedOperator, x::AbstractVector, α::Number, β::Number
)
    # Interior-only write, by contract: ghosts staged into P.xpad beforehand must
    # survive the sweep. Do not add zero_ghosts! here.
    flat_to_interior!(P.xpad, x)
    apply!(P.ypad, P.op, P.xpad, P.grid)
    interior_to_flat!(y, P.ypad, α, β)
    return y
end
function LinearAlgebra.mul!(y::AbstractVector, P::PreparedOperator, x::AbstractVector)
    return mul!(y, P, x, true, false)
end

function LinearAlgebra.mul!(
    y::AbstractVector, P::PreparedForest, x::AbstractVector, α::Number, β::Number
)
    _require_prepared_current(P)
    flat_to_interior!(P.xpad, x)
    _forest_capply!(P.ypad, P.op, P.xpad, P, true, false)
    interior_to_flat!(y, P.ypad, α, β)
    return y
end
function LinearAlgebra.mul!(y::AbstractVector, P::PreparedForest, x::AbstractVector)
    return mul!(y, P, x, true, false)
end

#--------------------------------------------------------------------------------# Field-level action of a prepared operator (the explicit-stepping path)

"""
    apply!(y::AbstractField, P::PreparedOperator, x::AbstractField, α=true, β=false) -> y
    apply!(y::AbstractField, P::PreparedForest, x::AbstractField, α=true, β=false) -> y

Apply a [`prepare`](@ref)d operator at field level: `y = α·P(x) + β·y` on the
interior of `y`, reusing the scratch `prepare` bound (so `*`-composed and adjoint
nodes do not reallocate) and skipping the flat staging copies `mul!` performs.
Ghost handling and the homogeneous-BC caveat are those of [`apply!`](@ref) on an
unprepared operator.

`prepare` once, then `apply!(du, P, u)` per explicit stage; keep `mul!` for
Krylov. `x` and `y` must be on `P.grid` with the prototype's element type, and
the forest form additionally rejects a regridded forest.
"""
function apply!(y::Field, P::PreparedOperator, x::Field, α::Number=true, β::Number=false)
    _require_prepared_match(P, x, y)
    return apply!(y, P.op, x, P.grid, α, β)
end
function apply!(
    y::AbstractBlockField, P::PreparedForest, x::AbstractBlockField, α::Number=true, β::Number=false
)
    _require_prepared_current(P)
    _require_prepared_match(P, x, y)
    return _forest_capply!(y, P.op, x, P, α, β)
end

function LinearAlgebra.mul!(::AbstractVector, L::AbstractOperator, ::AbstractVector)
    throw(
        ArgumentError(
            "mul! on a raw $(nameof(typeof(L))) would allocate scratch every call; wrap the " *
            "operator once with prepare(L, x) and hand the prepared operator to the solver",
        ),
    )
end
function LinearAlgebra.mul!(
    y::AbstractVector, L::AbstractOperator, x::AbstractVector, ::Number, ::Number
)
    return mul!(y, L, x)
end
