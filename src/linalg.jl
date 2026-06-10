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

# Tree-walking buffer allocation: leaves pass through unchanged. Combinators that
# need intermediate storage override this to return buffer-carrying twins.
_prepare_tree(L::AbstractOperator, ::Field) = L

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
