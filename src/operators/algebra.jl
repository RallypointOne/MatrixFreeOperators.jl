#--------------------------------------------------------------------------------# Lazy combinators

"""
    Scaled(op, α)

Lazy scalar multiple `α·op`. Produced by `α * L`, `L * α`, `-L`, and `L / α`.
"""
struct Scaled{O<:AbstractOperator,T<:Number} <: AbstractOperator
    op::O
    α::T
end

"""
    Added(a, b)

Lazy operator sum `a + b`. Produced by `a + b` and `a - b`.
"""
struct Added{A<:AbstractOperator,B<:AbstractOperator} <: AbstractOperator
    a::A
    b::B
end

"""
    Composed(a, b)

Lazy operator composition `(a ∘ b)(x) = a(b(x))`. Produced by `a * b`.
"""
struct Composed{A<:AbstractOperator,B<:AbstractOperator} <: AbstractOperator
    a::A
    b::B
end

Base.:+(a::AbstractOperator, b::AbstractOperator) = Added(a, b)
Base.:-(a::AbstractOperator, b::AbstractOperator) = Added(a, -1 * b)
Base.:-(a::AbstractOperator) = -1 * a
Base.:*(a::AbstractOperator, b::AbstractOperator) = Composed(a, b)
Base.:*(α::Number, a::AbstractOperator) = Scaled(a, α)
Base.:*(a::AbstractOperator, α::Number) = α * a
Base.:*(α::Number, a::Scaled) = Scaled(a.op, α * a.α)
Base.:/(a::AbstractOperator, α::Number) = inv(α) * a

#--------------------------------------------------------------------------------# Action

function apply!(y::Field, L::Added, x::Field, g::AbstractGrid, α, β)
    apply!(y, L.a, x, g, α, β)
    apply!(y, L.b, x, g, α, true)
    return y
end

# Pure path: the intermediate is allocated per call (autodiff-friendly). The
# prepared path replaces Composed nodes with buffer-carrying twins — see linalg.jl.
# Each factor sees the grid of the field it consumes: the factors of a transfer
# chain (restriction/prolongation) live on different grids.
function apply!(y::Field, L::Composed, x::Field, g::AbstractGrid, α, β)
    tmp = allocate_output(L.b, x)
    apply!(tmp, L.b, x, g)
    apply!(y, L.a, tmp, tmp.grid, α, β)
    return y
end

apply!(y::Field, L::Scaled, x::Field, g::AbstractGrid, α, β) = apply!(y, L.op, x, g, α * L.α, β)

allocate_output(L::Added, x::AbstractField) = allocate_output(L.a, x)
allocate_output(L::Composed, x::AbstractField) = allocate_output(L.a, allocate_output(L.b, x))
allocate_output(L::Scaled, x::AbstractField) = allocate_output(L.op, x)
allocate_input(L::Added, y::AbstractField) = allocate_input(L.a, y)
allocate_input(L::Composed, y::AbstractField) = allocate_input(L.b, allocate_input(L.a, y))
allocate_input(L::Scaled, y::AbstractField) = allocate_input(L.op, y)

#--------------------------------------------------------------------------------# Adjoint propagation

adjoint_operator(L::Added) = Added(adjoint_operator(L.a), adjoint_operator(L.b))
adjoint_operator(L::Composed) = Composed(adjoint_operator(L.b), adjoint_operator(L.a))
adjoint_operator(L::Scaled) = Scaled(adjoint_operator(L.op), conj(L.α))

function apply_adjoint!(x̄::Field, L::Added, ȳ::Field, g::AbstractGrid, α, β)
    apply_adjoint!(x̄, L.a, ȳ, g, α, β)
    apply_adjoint!(x̄, L.b, ȳ, g, α, true)
    return x̄
end

function apply_adjoint!(x̄::Field, L::Composed, ȳ::Field, g::AbstractGrid, α, β)
    tmp = allocate_input(L.a, ȳ)
    apply_adjoint!(tmp, L.a, ȳ, ȳ.grid)
    apply_adjoint!(x̄, L.b, tmp, tmp.grid, α, β)
    return x̄
end

function apply_adjoint!(x̄::Field, L::Scaled, ȳ::Field, g::AbstractGrid, α, β)
    return apply_adjoint!(x̄, L.op, ȳ, g, α * conj(L.α), β)
end

#--------------------------------------------------------------------------------# Trait propagation

islinear(L::Added) = islinear(L.a) && islinear(L.b)
islinear(L::Composed) = islinear(L.a) && islinear(L.b)
islinear(L::Scaled) = islinear(L.op)
isconstant(L::Added) = isconstant(L.a) && isconstant(L.b)
isconstant(L::Composed) = isconstant(L.a) && isconstant(L.b)
isconstant(L::Scaled) = isconstant(L.op)
isdiagonal(L::Added) = isdiagonal(L.a) && isdiagonal(L.b)
isdiagonal(L::Composed) = isdiagonal(L.a) && isdiagonal(L.b)
isdiagonal(L::Scaled) = isdiagonal(L.op)
isselfadjoint(L::Added) = isselfadjoint(L.a) && isselfadjoint(L.b)
# NOT compositional: A, B self-adjoint does not make AB self-adjoint (they would
# have to commute), so Composed never claims it.
isselfadjoint(::Composed) = false
isselfadjoint(L::Scaled) = isselfadjoint(L.op) && isreal(L.α)
shares_exchange(L::Added) = shares_exchange(L.a) && shares_exchange(L.b)
shares_exchange(L::Scaled) = shares_exchange(L.op)
# The intermediate b(x) is a fresh field that needs its own exchange before a
# reads its ghosts, so a composition can never run on a sibling's exchange.
shares_exchange(::Composed) = false

#--------------------------------------------------------------------------------# Grid resolution, size, show, Adapt

_first_grid(a, b) = a === nothing ? b : a
operator_grid(L::Added) = _first_grid(operator_grid(L.a), operator_grid(L.b))
operator_grid(L::Composed) = _first_grid(operator_grid(L.a), operator_grid(L.b))
operator_grid(L::Scaled) = operator_grid(L.op)

Base.size(L::Added) = operator_grid(L.a) === nothing ? size(L.b) : size(L.a)
function Base.size(L::Composed)
    ga = operator_grid(L.a)
    gb = operator_grid(L.b)
    ga === nothing && gb === nothing &&
        throw(ArgumentError("composed operator is not bound to a grid"))
    ga === nothing && return size(L.b)
    gb === nothing && return size(L.a)
    return (size(L.a)[1], size(L.b)[2])
end
Base.size(L::Scaled) = size(L.op)

function Base.show(io::IO, L::Added)
    print(io, "(")
    show(io, L.a)
    print(io, " + ")
    show(io, L.b)
    return print(io, ")")
end
function Base.show(io::IO, L::Composed)
    print(io, "(")
    show(io, L.a)
    print(io, " ∘ ")
    show(io, L.b)
    return print(io, ")")
end
function Base.show(io::IO, L::Scaled)
    print(io, "(", L.α, " * ")
    show(io, L.op)
    return print(io, ")")
end

Adapt.adapt_structure(to, L::Scaled) = Scaled(Adapt.adapt(to, L.op), L.α)
Adapt.adapt_structure(to, L::Added) = Added(Adapt.adapt(to, L.a), Adapt.adapt(to, L.b))
function Adapt.adapt_structure(to, L::Composed)
    return Composed(Adapt.adapt(to, L.a), Adapt.adapt(to, L.b))
end
