#--------------------------------------------------------------------------------# Prepared operators (the Krylov boundary)

"""
    PreparedOperator

A linear operator bound to pre-allocated scratch fields, exposing the flat
`mul!`/`size`/`eltype` interface Krylov solvers need. Built with
[`prepare`](@ref); after warm-up, `mul!` runs without steady-state allocations.

The flat vectors span interior DOFs only — ghost cells are determined by
boundary conditions and `halo_update!`, never solver unknowns. Each `mul!` copies
the flat vector into a halo-padded scratch field, applies the operator, and
copies the interior back out fused with the `α`/`β` axpby, so the solver's
vectors are never mutated by halo or BC fills.
"""
struct PreparedOperator{O<:AbstractOperator,G<:AbstractGrid,FX<:Field,FY<:Field}
    op::O
    grid::G
    xpad::FX
    ypad::FY
end

"""
    prepare(L::AbstractOperator, x::Field) -> PreparedOperator
    prepare(L::AbstractOperator) -> PreparedOperator

Walk the operator tree once, allocating the scratch buffers every node needs, and
return a [`PreparedOperator`](@ref) whose `mul!` is allocation-free in steady
state. `x` is a prototype of the input field (contents are ignored); the
one-argument form assumes a scalar field on the operator's grid.

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
function prepare(L::AbstractOperator, x::Field)
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

# Tree-walking buffer allocation: leaves pass through unchanged; Composed and
# AdjointOp nodes are replaced by buffer-carrying twins so steady-state mul! never
# allocates.
_prepare_tree(L::AbstractOperator, ::Field) = L
_prepare_tree(L::Added, x::Field) = Added(_prepare_tree(L.a, x), _prepare_tree(L.b, x))
_prepare_tree(L::Scaled, x::Field) = Scaled(_prepare_tree(L.op, x), L.α)

function _prepare_tree(L::Composed, x::Field)
    pb = _prepare_tree(L.b, x)
    tmp = allocate_output(L.b, x)
    pa = _prepare_tree(L.a, tmp)
    return PreparedComposed(pa, pb, tmp)
end

function _prepare_tree(L::AdjointOp, x::Field)
    return PreparedAdjoint(L.op, allocate_output(L, x))
end

# Composed twin holding its concretely-typed intermediate field.
struct PreparedComposed{A<:AbstractOperator,B<:AbstractOperator,F<:Field} <: AbstractOperator
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
struct PreparedAdjoint{O<:AbstractOperator,F<:Field} <: AbstractOperator
    op::O
    scratch::F
end

function apply!(y::Field, L::PreparedAdjoint, x::Field, g::AbstractGrid, α, β)
    iszero(β) && return apply_adjoint!(y, L.op, x, g, α, β)
    apply_adjoint!(L.scratch, L.op, x, g)
    interior(y) .= α .* interior(L.scratch) .+ β .* interior(y)
    return y
end

#--------------------------------------------------------------------------------# Boundary lift (linear/affine split)

"""
    boundary_rhs(L::AbstractOperator, g::AbstractGrid) -> Field
    boundary_rhs(L::AbstractOperator, x_proto::Field) -> Field

Boundary lift of the affine split `L_full(x) = L(x) + b`: the contribution of
*inhomogeneous* boundary data (Dirichlet values, Neumann fluxes) that
[`apply!`](@ref) deliberately omits so that `L` stays linear (`L(0) = 0`).
Assemble once per solve and fold into the right-hand side: the discrete problem
`L_full(u) = f` becomes `L·u = f - b`. The grid form assumes a scalar input
field; pass a prototype field for vector inputs.

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

function Base.size(P::PreparedOperator)
    n = prod(local_size(P.grid))
    return (n * ncomponents(P.ypad), n * ncomponents(P.xpad))
end
Base.size(P::PreparedOperator, d::Integer) = size(P)[d]
Base.eltype(P::PreparedOperator) = _scalar_eltype(eltype(P.xpad))

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
