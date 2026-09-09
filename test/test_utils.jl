# Central finite-difference gradient of a scalar loss w.r.t. every array entry —
# small arrays only; the reference all AD gradients are checked against.
function fd_gradient(loss, x::AbstractArray; ε=1e-6)
    grad = zero(x)
    for i in eachindex(x)
        xp = copy(x)
        xp[i] += ε
        xm = copy(x)
        xm[i] -= ε
        grad[i] = (loss(xp) - loss(xm)) / (2 * ε)
    end
    return grad
end

# Materialize a prepared operator as a dense matrix by applying it to basis
# vectors — small grids only; used to verify symmetry/transpose structure exactly.
function materialize(P)
    m, n = size(P)
    A = zeros(eltype(P), m, n)
    e = zeros(eltype(P), n)
    y = zeros(eltype(P), m)
    for j in 1:n
        fill!(e, 0)
        e[j] = 1
        mul!(y, P, e)
        A[:, j] .= y
    end
    return A
end

# Exchange-counting view of a forest field. An AbstractBlockField that forwards
# storage, layout, and per-leaf access to the wrapped field, and intercepts only
# the two execution seams every forest action passes through — the inter-block
# halo exchange (`_run_exchange!`) and the physical-BC face pass (`_run_bc!`) —
# to tally them. Wrap the *input* of an `apply!`/`_forest_capply!` to assert how
# many exchanges one operator application performs. `similar` unwraps, so a
# composition's intermediate (and an allocating `apply`'s output) is a plain,
# uncounted field: the tally is exchanges *on the wrapped input* only.
struct CountingBlockField{
    F<:MatrixFreeOperators.AbstractBlockField,G<:MatrixFreeOperators.BlockForest
} <: MatrixFreeOperators.AbstractBlockField
    inner::F
    grid::G
    generation::Int
    exchanges::Base.RefValue{Int}
    bcfills::Base.RefValue{Int}
end
CountingBlockField(f::MatrixFreeOperators.AbstractBlockField) =
    CountingBlockField(f, f.grid, f.generation, Ref(0), Ref(0))

MatrixFreeOperators._storage(c::CountingBlockField) = MatrixFreeOperators._storage(c.inner)
MatrixFreeOperators._layout(c::CountingBlockField) = MatrixFreeOperators._layout(c.inner)
MatrixFreeOperators._block_array(c::CountingBlockField, i::Integer) =
    MatrixFreeOperators._block_array(c.inner, i)
MatrixFreeOperators._block_view(c::CountingBlockField, i::Integer, ranges) =
    MatrixFreeOperators._block_view(c.inner, i, ranges)
MatrixFreeOperators.block(c::CountingBlockField, i::Integer, leaf_grid) =
    MatrixFreeOperators.block(c.inner, i, leaf_grid)
Base.eltype(c::CountingBlockField) = eltype(c.inner)
Base.similar(c::CountingBlockField) = similar(c.inner)
Base.similar(c::CountingBlockField, ::Type{E}) where {E} = similar(c.inner, E)

function MatrixFreeOperators._run_exchange!(
    c::CountingBlockField,
    g::MatrixFreeOperators.BlockForest,
    sched::MatrixFreeOperators.ExchangeSchedule,
)
    c.exchanges[] += 1
    return MatrixFreeOperators._run_exchange!(c.inner, g, sched)
end
function MatrixFreeOperators._run_bc!(
    c::CountingBlockField,
    g::MatrixFreeOperators.BlockForest,
    sched::MatrixFreeOperators.ExchangeSchedule,
)
    c.bcfills[] += 1
    return MatrixFreeOperators._run_bc!(c.inner, g, sched)
end

# Number of halo exchanges `apply!(y, L, x, g)` performs, plus the result — the
# count is reset per call so it reads as "exchanges per apply".
function count_exchanges!(y, L, x, g, α=true, β=false)
    c = CountingBlockField(x)
    apply!(y, L, c, g, α, β)
    return c.exchanges[], c.bcfills[]
end
