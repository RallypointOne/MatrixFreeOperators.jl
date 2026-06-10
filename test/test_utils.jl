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
