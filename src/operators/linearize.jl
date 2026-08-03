#--------------------------------------------------------------------------------# Linearization (matrix-free Jacobian)

"""
    AbstractJVPBackend

How a linearized operator evaluates its Jacobian–vector product. Selecting the
backend is deliberately explicit rather than ambient: the backends differ in
accuracy and in which operations they support, so which one you get must not depend
on whether some package happens to be loaded.

- [`FiniteDifferenceJVP`](@ref) — the dependency-free default.
- [`EnzymeJVP`](@ref) — exact, and the preferred choice; needs `using Enzyme`.
"""
abstract type AbstractJVPBackend end

"""
    FiniteDifferenceJVP()

Evaluate the JVP by a central finite difference, `(F(u₀+εv) - F(u₀-εv))/2ε`. The
default, because it needs no dependencies — but it costs two operator applications
per product, is accurate only to about `√eps`, and has no transpose, so
`adjoint` of the resulting Jacobian throws. Prefer [`EnzymeJVP`](@ref).
"""
struct FiniteDifferenceJVP <: AbstractJVPBackend end

"""
    EnzymeJVP()

Evaluate the JVP by forward-mode automatic differentiation, and the transpose by
reverse mode. Exact to machine precision, one operator application per product, and
— unlike [`FiniteDifferenceJVP`](@ref) — it supplies a real `adjoint`, so
transpose-needing Krylov methods work on a Jacobian-free Newton–Krylov operator.

Provided by the Enzyme extension: `using Enzyme` before calling
`linearize(F, u₀, EnzymeJVP())`.
"""
struct EnzymeJVP <: AbstractJVPBackend end

"""
    LinearizedOp

Matrix-free Jacobian `J = ∂F/∂u` of an operator `F`, frozen at a state `u₀`. Its
action is the Jacobian–vector product `J·v = ∂/∂ε F(u₀ + εv)|₀`, evaluated by a
central finite-difference JVP. Linear by construction — the operator handed to
Krylov in Jacobian-free Newton–Krylov. Build with [`linearize`](@ref); refresh the
frozen state in place with [`linearize!`](@ref).
"""
struct LinearizedOp{O<:AbstractOperator,U<:Field,FO<:Field} <: AbstractOperator
    op::O
    u0::U          # frozen linearization state (owned copy)
    shifted::U     # scratch for u₀ ± εv
    fplus::FO      # scratch for F(u₀ + εv)
    fminus::FO     # scratch for F(u₀ - εv)
end

"""
    linearize(F::AbstractOperator, u0::Field, backend=FiniteDifferenceJVP())

Linearize the (possibly nonlinear) operator `F` at the state `u0`, returning a
*linear* matrix-free Jacobian operator whose `apply!`/`mul!` is the JVP
`∂/∂ε F(u0 + εv)|₀`. Owns a copy of `u0`; see [`linearize!`](@ref) for in-place
refresh inside Newton–Krylov loops.

`backend` selects how the product is evaluated — see [`AbstractJVPBackend`](@ref).
The default [`FiniteDifferenceJVP`](@ref) keeps the core dependency-free;
[`EnzymeJVP`](@ref) is exact and additionally provides the transpose.

### Examples

```julia
g = CartesianGrid(((0.0, 2π),), (64,); bc=((Periodic(), Periodic()),))
F = advection(g, SelfAdvection())            # nonlinear u·∇u
u0 = set!(vector_field(g), x -> SVector(sin(x[1])))
J = linearize(F, u0)                         # linear: v ↦ (∂F/∂u)|_{u0} · v
P = prepare(J, u0)                           # Krylov-ready JFNK Jacobian

using Enzyme                                 # exact JVP, and adjoint(J) works
Jad = linearize(F, u0, EnzymeJVP())
```

See also: [`prepare`](@ref), [`apply`](@ref).
"""
linearize(F::AbstractOperator, u0::Field) = linearize(F, u0, FiniteDifferenceJVP())

function linearize(F::AbstractOperator, u0::Field, ::FiniteDifferenceJVP)
    u0c = copy(u0)
    return LinearizedOp(
        F, u0c, similar(u0c), allocate_output(F, u0c), allocate_output(F, u0c)
    )
end

# Backends whose implementation lives in an extension land here until it loads.
function linearize(::AbstractOperator, ::Field, backend::AbstractJVPBackend)
    throw(
        ArgumentError(
            "no linearize method for JVP backend $(nameof(typeof(backend))); " *
            "EnzymeJVP is provided by the Enzyme extension — run `using Enzyme` first",
        ),
    )
end

"""
    linearize!(J::LinearizedOp, u::Field) -> J

Refresh the frozen linearization state of `J` to `u` in place, reusing the
operator and its scratch fields. Safe inside Newton–Krylov loops because Krylov
solves are never differentiated through — sensitivities come from
implicit-function-theorem adjoints on the solution, not the iteration.
"""
function linearize!(J::LinearizedOp, u::Field)
    copyto!(J.u0.data, u.data)
    return J
end

islinear(::LinearizedOp) = true
operator_grid(J::LinearizedOp) = _first_grid(operator_grid(J.op), J.u0.grid)
allocate_output(J::LinearizedOp, x::Field) = allocate_output(J.op, x)
allocate_input(J::LinearizedOp, y::Field) = allocate_input(J.op, y)

function Base.size(J::LinearizedOp)
    n = prod(local_size(J.u0.grid))
    return (n * ncomponents(J.fminus), n * ncomponents(J.u0))
end

# Central-difference JVP: (F(u₀+εv) - F(u₀-εv)) / 2ε with ε scaled to the state
# and direction magnitudes.
function apply!(y::Field, J::LinearizedOp, v::Field, g::AbstractGrid, α, β)
    T = _scalar_eltype(eltype(J.u0.data))
    normv = norm(v.data)
    ε = iszero(normv) ? sqrt(eps(T)) : sqrt(eps(T)) * (1 + norm(J.u0.data)) / normv
    J.shifted.data .= J.u0.data .+ ε .* v.data
    apply!(J.fplus, J.op, J.shifted, g)
    J.shifted.data .= J.u0.data .- ε .* v.data
    apply!(J.fminus, J.op, J.shifted, g)
    c = α / (2 * ε)
    if iszero(β)
        interior(y) .= c .* (interior(J.fplus) .- interior(J.fminus))
    else
        interior(y) .= c .* (interior(J.fplus) .- interior(J.fminus)) .+ β .* interior(y)
    end
    return y
end

function adjoint_operator(::LinearizedOp)
    throw(
        ArgumentError(
            "the adjoint of a finite-difference JVP operator requires reverse-mode AD " *
            "and is not provided; differentiate the underlying operator instead",
        ),
    )
end

function Adapt.adapt_structure(to, J::LinearizedOp)
    return LinearizedOp(
        Adapt.adapt(to, J.op),
        Adapt.adapt(to, J.u0),
        Adapt.adapt(to, J.shifted),
        Adapt.adapt(to, J.fplus),
        Adapt.adapt(to, J.fminus),
    )
end
