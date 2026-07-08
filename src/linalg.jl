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
scratch fields it caches, at `prepare` time, the per-leaf `(index, leaf grid)` pairs
grouped by BC signature (`groups`) plus a scratch [`BlockField`](@ref) for
accumulating adjoint sweeps (`adjscratch`), so steady-state `mul!` never rebuilds a
leaf grid — which was the dominant per-application allocation. The residual cost is
the per-leaf stencil `apply!` itself (allocation-free only when inlined into a single
`mul!`); see the allocation-free-kernel follow-up. The cache is tied to the forest's
regrid `generation`; a `refine!`/`coarsen!`/`balance!` after `prepare` invalidates it
and `mul!` throws — re-run `prepare` on the new forest.
"""
struct PreparedForest{
    O<:AbstractOperator,G<:BlockForest,FX<:BlockField,FY<:BlockField,C<:Tuple,S<:BlockField
}
    op::O
    grid::G
    xpad::FX
    ypad::FY
    groups::C
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
single grid; on a block forest it rebuilds no leaf grid (its residual cost is the
per-leaf stencil apply). `x` is a prototype of the input field (contents are
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

# Forest prepare: same tree walk (Composed still errors via _prepare_tree), plus a
# per-leaf grid cache so mul! never rebuilds a leaf grid. adjscratch backs the
# accumulating adjoint sweep the un-prepared path allocates per call.
function prepare(L::AbstractOperator, x::BlockField)
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
    return PreparedForest(
        op, x.grid, xpad, ypad, _leaf_cache(x.grid), similar(x), x.grid.forest.generation[]
    )
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
# The intermediate would need an inter-block exchange — same deferral as the
# unprepared forest path.
_prepare_tree(L::Composed, ::BlockField) = _check_forest_supported(L)

function _prepare_tree(L::AdjointOp, x::AbstractField)
    return PreparedAdjoint(L.op, allocate_output(L, x))
end

# Composed twin holding its concretely-typed intermediate field.
struct PreparedComposed{A<:AbstractOperator,B<:AbstractOperator,F<:AbstractField} <: AbstractOperator
    a::A
    b::B
    tmp::F
end

function apply!(y::Field, L::PreparedComposed, x::Field, g::AbstractGrid, α, β)
    apply!(L.tmp, L.b, x, g)
    apply!(y, L.a, L.tmp, g, α, β)
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

# Cached counterpart of the un-prepared forest apply in operators/forest.jl: iterate
# the prepare-time leaf-grid cache (behind a function barrier, P.groups) instead of
# rebuilding each leaf grid — the dominant per-application allocation. The combinator
# structure mirrors forest.jl exactly — Added/Scaled/AdjointOp recurse at the forest
# level so a nested adjoint still reaches halo_update_adjoint!'s cross-block fold —
# and additionally handles the PreparedAdjoint nodes _prepare_tree introduces. (The
# residual per-leaf cost is the stencil apply! itself; see the alloc-free-kernel note
# on PreparedForest.)
function _forest_capply!(y::BlockField, L::AbstractOperator, x::BlockField, P::PreparedForest, α, β)
    _require_current(y)
    halo_update!(x, P.grid)
    _foreach_leaf(P.groups) do i, lg
        apply!(block(y, i, lg), L, block(x, i, lg), lg, α, β)
    end
    return y
end
function _forest_capply!(y::BlockField, L::Added, x::BlockField, P::PreparedForest, α, β)
    _forest_capply!(y, L.a, x, P, α, β)
    _forest_capply!(y, L.b, x, P, α, true)
    return y
end
_forest_capply!(y::BlockField, L::Scaled, x::BlockField, P::PreparedForest, α, β) =
    _forest_capply!(y, L.op, x, P, α * L.α, β)
_forest_capply!(y::BlockField, L::AdjointOp, x::BlockField, P::PreparedForest, α, β) =
    _forest_capply_adjoint!(y, L.op, x, P, α, β)

# PreparedAdjoint: the leaf adjoint gather is allocation-free only for β = 0, so the
# accumulating form gathers into the node's own scratch, then blends per leaf.
function _forest_capply!(y::BlockField, L::PreparedAdjoint, x::BlockField, P::PreparedForest, α, β)
    iszero(β) && return _forest_capply_adjoint!(y, L.op, x, P, α, β)
    _forest_capply_adjoint!(L.scratch, L.op, x, P, true, false)
    s = L.scratch
    _foreach_leaf(P.groups) do i, lg
        yi = interior(block(y, i, lg))
        yi .= α .* interior(block(s, i, lg)) .+ β .* yi
    end
    return y
end

# Cached forest adjoint action, mirroring apply_adjoint! in operators/forest.jl. The
# accumulating (β ≠ 0) branch uses the prepared adjscratch in place of similar(x̄).
function _forest_capply_adjoint!(
    x̄::BlockField, L::AbstractOperator, ȳ::BlockField, P::PreparedForest, α, β
)
    _require_uniform(P.grid)
    _require_current(x̄)
    _require_current(ȳ)
    isselfadjoint(L) && return _forest_capply!(x̄, L, ȳ, P, α, β)
    if iszero(β)
        _foreach_leaf(P.groups) do i, lg
            apply_adjoint!(block(x̄, i, lg), L, block(ȳ, i, lg), lg, α, false)
        end
        halo_update_adjoint!(x̄, P.grid)
    else
        s = P.adjscratch
        _foreach_leaf(P.groups) do i, lg
            apply_adjoint!(block(s, i, lg), L, block(ȳ, i, lg), lg, true, false)
        end
        halo_update_adjoint!(s, P.grid)
        _foreach_leaf(P.groups) do i, lg
            xi = interior(block(x̄, i, lg))
            xi .= α .* interior(block(s, i, lg)) .+ β .* xi
        end
    end
    return x̄
end
function _forest_capply_adjoint!(x̄::BlockField, L::Added, ȳ::BlockField, P::PreparedForest, α, β)
    _forest_capply_adjoint!(x̄, L.a, ȳ, P, α, β)
    _forest_capply_adjoint!(x̄, L.b, ȳ, P, α, true)
    return x̄
end
_forest_capply_adjoint!(x̄::BlockField, L::Scaled, ȳ::BlockField, P::PreparedForest, α, β) =
    _forest_capply_adjoint!(x̄, L.op, ȳ, P, α * conj(L.α), β)
_forest_capply_adjoint!(x̄::BlockField, L::AdjointOp, ȳ::BlockField, P::PreparedForest, α, β) =
    _forest_capply!(x̄, L.op, ȳ, P, α, β)

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
lift assembles per leaf block — only physical domain faces contribute
([`Interface`](@ref) faces are homogeneous by construction).

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

# Forest lift: the inhomogeneous fill and the stencil are local to each block
# (Interface ghosts stay zero), so the single-grid lift runs per leaf unchanged.
function boundary_rhs(L::AbstractOperator, x_proto::BlockField)
    _check_forest_supported(L)
    b = allocate_output(L, x_proto)
    g = x_proto.grid
    for i in 1:nleaves(g)
        lg = leaf_grid(g, i)
        bi = boundary_rhs(L, block(x_proto, i, lg))
        block(b, i, lg).data .= bi.data
    end
    return b
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
    _foreach_leaf(P.groups) do i, lg
        flat_to_interior!(block(xpad, i, lg), view(x, _block_range(xpad, i)))
    end
    _forest_capply!(P.ypad, P.op, xpad, P, true, false)
    ypad = P.ypad
    _foreach_leaf(P.groups) do i, lg
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
