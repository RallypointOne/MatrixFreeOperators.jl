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
