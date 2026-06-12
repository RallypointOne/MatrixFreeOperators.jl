"""
Reactant-traceable leaf bodies. The default Laplacian body broadcasts
`_lap_at` over `CartesianIndices`, hiding the traced array behind a `Ref` —
Reactant then tries (and fails) to promote `CartesianIndex` to a tensor, and
the per-cell `u[I ± δ]` gathers would be opaque to XLA regardless. The same
stencil written as shifted interior views is pure slicing + broadcast, which
traces and fuses. Term and reduction order mirror `laplacian_stencil` so the
two paths agree bit-for-bit.
"""
module MatrixFreeOperatorsReactantExt

using MatrixFreeOperators, Reactant
import MatrixFreeOperators: Field, Laplacian, AbstractGrid, interior

_shifted(u, rs::NTuple{N,<:AbstractUnitRange}, d::Int, s::Int) where {N} =
    view(u, ntuple(i -> i == d ? rs[i] .+ s : rs[i], Val(N))...)

function MatrixFreeOperators._apply_raw!(
    y::Field{<:Any,<:Reactant.TracedRArray},
    ::Laplacian,
    x::Field{<:Any,<:Reactant.TracedRArray},
    g::AbstractGrid{N},
    α,
    β,
) where {N}
    inv_h2 = MatrixFreeOperators._inv_spacing2(g)
    rs = interior(g).indices
    u = x.data
    c = view(u, rs...)
    terms = ntuple(Val(N)) do d
        (_shifted(u, rs, d, -1) .- 2 .* c .+ _shifted(u, rs, d, 1)) .* inv_h2[d]
    end
    lap = reduce((a, b) -> a .+ b, terms)
    yi = interior(y)
    if iszero(β)
        yi .= α .* lap
    else
        yi .= α .* lap .+ β .* yi
    end
    return y
end

end
