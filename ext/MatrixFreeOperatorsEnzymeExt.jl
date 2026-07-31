"""
An exact, AD-powered matrix-free Jacobian.

The core [`LinearizedOp`](@ref) evaluates its Jacobian–vector product as a central
finite difference. That is dependency-free, which is why it is the default, but it
costs two operator applications per product, is accurate only to about `√eps` — the
step `ε ~ sqrt(eps)*(1+‖u₀‖)/‖v‖` and the `1/2ε` division amplify roundoff — and it
has no transpose, so `adjoint` of a finite-difference Jacobian throws.

This extension supplies the operator DESIGN.md always described: `apply!` is a
forward-mode JVP, exact to machine precision in one operator application, and
`apply_adjoint!` is a reverse-mode VJP, so `adjoint(J)` is a real operator and
transpose-needing Krylov methods work against a Jacobian-free Newton–Krylov
Jacobian.

Both directions differentiate the *pure* `apply!` path — array-level broadcasts over
halo-padded fields, exactly what Decision A exists to make differentiable — with the
frozen state, its shadow, and the output scratch all owned by the operator, so a
Newton–Krylov inner loop allocates nothing per product beyond what Enzyme itself
needs.
"""
module MatrixFreeOperatorsEnzymeExt

using MatrixFreeOperators
using Enzyme

import MatrixFreeOperators:
    AbstractField, AbstractGrid, AbstractOperator, EnzymeJVP, Field, _first_grid,
    _scalar_eltype, adjoint_operator, allocate_input, allocate_output, apply!,
    apply_adjoint!, interior, islinear, linearize, linearize!, ncomponents,
    operator_grid, zero_ghosts!

using Adapt: Adapt
using LinearAlgebra: norm

"""
    EnzymeLinearizedOp

Matrix-free Jacobian `J = ∂F/∂u` frozen at a state `u₀`, with the product evaluated
by Enzyme. Built by `linearize(F, u₀, EnzymeJVP())`.

Unlike the finite-difference [`LinearizedOp`](@ref) this operator declares an
adjoint: `apply_adjoint!` is the reverse-mode VJP `v ↦ (∂F/∂u)ᵀ|_{u₀}·v`.
"""
struct EnzymeLinearizedOp{O<:AbstractOperator,U<:Field,FO<:Field} <: AbstractOperator
    op::O
    u0::U        # frozen linearization state (owned copy)
    ustate::U    # scratch: working copy of u₀ (apply! overwrites ghosts)
    ushadow::U   # scratch: seeded with v forward, accumulates Jᵀȳ in reverse
    fout::FO     # scratch: primal output of F
    fshadow::FO  # scratch: output shadow — the JVP out, the cotangent seed in
end

function linearize(F::AbstractOperator, u0::Field, ::EnzymeJVP)
    u0c = copy(u0)
    return EnzymeLinearizedOp(
        F,
        u0c,
        similar(u0c),
        similar(u0c),
        allocate_output(F, u0c),
        allocate_output(F, u0c),
    )
end

function linearize!(J::EnzymeLinearizedOp, u::Field)
    copyto!(J.u0.data, u.data)
    return J
end

islinear(::EnzymeLinearizedOp) = true
operator_grid(J::EnzymeLinearizedOp) = _first_grid(operator_grid(J.op), J.u0.grid)
allocate_output(J::EnzymeLinearizedOp, x::AbstractField) = allocate_output(J.op, x)
allocate_input(J::EnzymeLinearizedOp, y::AbstractField) = allocate_input(J.op, y)

function Base.size(J::EnzymeLinearizedOp)
    n = prod(MatrixFreeOperators.local_size(J.u0.grid))
    return (n * ncomponents(J.fout), n * ncomponents(J.u0))
end

# Reset the differentiation scratch: a fresh copy of the frozen state (apply!
# overwrites its input's ghosts), and zeroed primal/shadow outputs so nothing from
# the previous product leaks into this one.
@inline function _reset!(J::EnzymeLinearizedOp)
    copyto!(J.ustate.data, J.u0.data)
    fill!(J.fout.data, zero(eltype(J.fout.data)))
    fill!(J.fshadow.data, zero(eltype(J.fshadow.data)))
    return nothing
end

"""
    apply!(y, J::EnzymeLinearizedOp, v, g, α, β)

Forward-mode JVP: `y = α·(∂F/∂u)|_{u₀}·v + β·y`. Seeds the input shadow with `v` and
pushes it through the operator, so the result is the exact directional derivative —
no step size, no `1/2ε` cancellation.
"""
function apply!(y::Field, J::EnzymeLinearizedOp, v::Field, g::AbstractGrid, α, β)
    _reset!(J)
    copyto!(J.ushadow.data, v.data)
    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Forward),
        apply!,
        Enzyme.Const,
        Enzyme.Duplicated(J.fout, J.fshadow),
        Enzyme.Const(J.op),
        Enzyme.Duplicated(J.ustate, J.ushadow),
        Enzyme.Const(g),
    )
    yi = interior(y)
    if iszero(β)
        yi .= α .* interior(J.fshadow)
    else
        yi .= α .* interior(J.fshadow) .+ β .* yi
    end
    return y
end

"""
    apply_adjoint!(x̄, J::EnzymeLinearizedOp, ȳ, g, α, β)

Reverse-mode VJP: `x̄ = α·(∂F/∂u)ᵀ|_{u₀}·ȳ + β·x̄`. Ghost layers of `ȳ` are treated as
scratch and zeroed before seeding, matching the adjoint contract — only interior
values are adjoint inputs.
"""
function apply_adjoint!(x̄::Field, J::EnzymeLinearizedOp, ȳ::Field, g::AbstractGrid, α, β)
    _reset!(J)
    fill!(J.ushadow.data, zero(eltype(J.ushadow.data)))
    copyto!(J.fshadow.data, ȳ.data)
    zero_ghosts!(J.fshadow)
    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        apply!,
        Enzyme.Const,
        Enzyme.Duplicated(J.fout, J.fshadow),
        Enzyme.Const(J.op),
        Enzyme.Duplicated(J.ustate, J.ushadow),
        Enzyme.Const(g),
    )
    xi = interior(x̄)
    if iszero(β)
        xi .= α .* interior(J.ushadow)
    else
        xi .= α .* interior(J.ushadow) .+ β .* xi
    end
    return x̄
end

# The declared adjoint is the reverse-mode VJP above, so the lazy wrapper suffices.
adjoint_operator(J::EnzymeLinearizedOp) = MatrixFreeOperators.AdjointOp(J)

function Adapt.adapt_structure(to, J::EnzymeLinearizedOp)
    return EnzymeLinearizedOp(
        Adapt.adapt(to, J.op),
        Adapt.adapt(to, J.u0),
        Adapt.adapt(to, J.ustate),
        Adapt.adapt(to, J.ushadow),
        Adapt.adapt(to, J.fout),
        Adapt.adapt(to, J.fshadow),
    )
end

end
