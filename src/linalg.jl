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

### Examples

```julia
g = CartesianGrid(((0.0, 1.0),), (64,))
A = prepare(laplacian(g))
b = flatten(set!(scalar_field(g), x -> sin(π * x[1])))
u, stats = Krylov.minres(A, b)

# explicit time stepping (OrdinaryDiffEq-style RHS closure):
f!(du, u, p, t) = mul!(du, A, u)
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

# Tree-walking buffer allocation: leaves pass through unchanged; Composed and
# AdjointOp nodes are replaced by buffer-carrying twins so steady-state mul! never
# allocates.
_prepare_tree(L::AbstractOperator, ::AbstractField) = L
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

function _prepare_tree(L::AdjointOp, x::AbstractField)
    return PreparedAdjoint(L.op, allocate_output(L, x))
end

# Composed twin holding its concretely-typed intermediate field.
struct PreparedComposed{A<:AbstractOperator,B<:AbstractOperator,F<:AbstractField} <: AbstractOperator
    a::A
    b::B
    tmp::F
end

# The inner factor sees the grid of the intermediate it consumes (transfer
# chains compose factors living on different grids).
function apply!(y::Field, L::PreparedComposed, x::Field, g::AbstractGrid, α, β)
    apply!(L.tmp, L.b, x, g)
    apply!(y, L.a, L.tmp, L.tmp.grid, α, β)
    return y
end

# AdjointOp twin: the leaf adjoint gather is allocation-free only for β = 0, so
# accumulating applications gather into the held scratch first.
struct PreparedAdjoint{O<:AbstractOperator,F<:AbstractField} <: AbstractOperator
    op::O
    scratch::F
end

function apply!(y::Field, L::PreparedAdjoint, x::Field, g::AbstractGrid, α, β)
    iszero(β) && return apply_adjoint!(y, L.op, x, g, α, β)
    apply_adjoint!(L.scratch, L.op, x, g)
    interior(y) .= α .* interior(L.scratch) .+ β .* interior(y)
    return y
end

#--------------------------------------------------------------------------------# Cached forest apply (the PreparedForest hot path)

# Cached counterpart of the un-prepared forest apply in operators/forest.jl, using
# the prepared scratch fields (leaf grids are one concrete all-Interface type, so
# rebuilding them in the sweep is type-stable and free). The combinator
# structure mirrors forest.jl exactly — Added/Scaled/AdjointOp recurse at the forest
# level so a nested adjoint still reaches halo_update_adjoint!'s cross-block fold —
# and additionally handles the PreparedAdjoint nodes _prepare_tree introduces. (The
# residual per-leaf cost is the stencil apply! itself; see the alloc-free-kernel note
# on PreparedForest.)
function _forest_capply!(
    y::AbstractBlockField, L::AbstractOperator, x::AbstractBlockField, P::PreparedForest, α, β
)
    _require_current(y)
    halo_update!(x, P.grid)
    apply_bc!(x, P.grid)
    return _forest_sweep!(y, L, x, P.grid, α, β)
end
function _forest_capply!(
    y::AbstractBlockField, L::Added, x::AbstractBlockField, P::PreparedForest, α, β
)
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

# PreparedAdjoint: the leaf adjoint gather is allocation-free only for β = 0, so the
# accumulating form gathers into the node's own scratch, then blends per leaf.
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
function _forest_capply_adjoint!(
    x̄::AbstractBlockField, L::Added, ȳ::AbstractBlockField, P::PreparedForest, α, β
)
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

function Base.size(P::_AnyPrepared)
    return (flat_length(P.ypad), flat_length(P.xpad))
end
Base.size(P::_AnyPrepared, d::Integer) = size(P)[d]
Base.eltype(P::_AnyPrepared) = _scalar_eltype(eltype(P.xpad))

function LinearAlgebra.mul!(
    y::AbstractVector, P::PreparedOperator, x::AbstractVector, α::Number, β::Number
)
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
    P.generation == P.grid.forest.generation[] || throw(
        ArgumentError(
            "this PreparedForest was built before the forest was regridded " *
            "(refine!/coarsen!/balance!); re-run prepare on the current forest",
        ),
    )
    xpad = P.xpad
    for i in 1:nleaves(P.grid)
        flat_to_interior!(block(xpad, i, leaf_grid(P.grid, i)), view(x, _block_range(xpad, i)))
    end
    _forest_capply!(P.ypad, P.op, xpad, P, true, false)
    ypad = P.ypad
    for i in 1:nleaves(P.grid)
        lg = leaf_grid(P.grid, i)
        interior_to_flat!(view(y, _block_range(ypad, i)), block(ypad, i, lg), α, β)
    end
    return y
end
function LinearAlgebra.mul!(y::AbstractVector, P::PreparedForest, x::AbstractVector)
    return mul!(y, P, x, true, false)
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
