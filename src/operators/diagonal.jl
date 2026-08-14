#--------------------------------------------------------------------------------# Diagonal arithmetic over Number | Field

# Setup-time helpers combining diagonals in either representation. Fields are
# never mutated; results are fresh Fields with zero ghosts.

_diag_scale(α::Number, d::Number) = α * d
_diag_scale(α::Number, d::Field) = Field(α .* d.data, d.grid)

_diag_conj(d::Number) = conj(d)
_diag_conj(d::Field) = Field(conj.(d.data), d.grid)

_diag_add(a::Number, b::Number) = a + b
function _diag_add(a::Number, b::Field)
    d = scalar_field(b.grid, promote_type(typeof(a), eltype(b.data)))
    interior(d) .= a .+ interior(b)
    return d
end
_diag_add(a::Field, b::Number) = _diag_add(b, a)
function _diag_add(a::Field, b::Field)
    a.grid === b.grid ||
        throw(ArgumentError("cannot combine diagonals bound to different grids"))
    return Field(a.data .+ b.data, a.grid)
end

_diag_mul(a::Number, b) = _diag_scale(a, b)
_diag_mul(a::Field, b::Number) = _diag_scale(b, a)
function _diag_mul(a::Field, b::Field)
    a.grid === b.grid ||
        throw(ArgumentError("cannot combine diagonals bound to different grids"))
    return Field(a.data .* b.data, a.grid)
end

#--------------------------------------------------------------------------------# Leaf diagonals

# Laplacian: -2·Σ_d h_d⁻² at interior cells. Dirichlet/Neumann faces mirror the
# boundary cell into ghost layer 1 with sign ∓1, so the stencil's ghost read
# feeds back into its own diagonal: the face-adjacent cell gains
# _bc_sign(bc)·h_d⁻² in that dimension. Periodic ghost fills read a *different*
# cell (the far side), so they contribute off-diagonal only.
function operator_diagonal(L::Laplacian)
    g = L.grid
    g isa CartesianGrid || throw(
        ArgumentError(
            "operator_diagonal(::Laplacian) supports CartesianGrid only, got $(nameof(typeof(g)))",
        ),
    )
    _has_interface(g) && throw(
        ArgumentError(
            "operator_diagonal(::Laplacian) does not support Interface faces (forest leaves)",
        ),
    )
    inv_h2 = _inv_spacing2(g)
    c = -2 * sum(inv_h2)
    bcs = boundary_conditions(g)
    all(pair -> pair[1] isa Periodic, bcs) && return c
    d = scalar_field(g)
    di = interior(d)
    di .= c
    _diag_bc_adjust!(di, g, Val(1), bcs, inv_h2)
    return d
end

function _diag_bc_adjust!(di, g, ::Val{D}, bcs::Tuple, inv_h2) where {D}
    lo, hi = first(bcs)
    n = local_size(g)[D]
    _diag_bc_face!(di, Val(D), 1:1, lo, inv_h2[D])
    _diag_bc_face!(di, Val(D), n:n, hi, inv_h2[D])
    return _diag_bc_adjust!(di, g, Val(D + 1), Base.tail(bcs), inv_h2)
end
_diag_bc_adjust!(di, g, ::Val, ::Tuple{}, inv_h2) = nothing

_diag_bc_face!(di, ::Val, r, ::Periodic, w) = nothing
function _diag_bc_face!(di, dim::Val, r, bc::AbstractBC, w)
    slab = _dimslice(di, dim, r)
    slab .+= _bc_sign(bc) * w
    return nothing
end

# Diffusion: every face carries its own coefficient, so unlike the Laplacian there is no
# uniform value to correct — the whole row is assembled from the same face average the
# action uses, which is what keeps a Jacobi diagonal from drifting away from the operator.
#
# A face's diagonal weight is s_f − 1, where s_f = ∂u[neighbour slot]/∂u_I: an interior or
# periodic neighbour is an independent unknown (s_f = 0, weight −1), Dirichlet mirrors
# with −1 (weight −2), Neumann mirrors with +1 (weight 0), and a periodic dimension of a
# *single* cell wraps onto the cell itself (s_f = +1, weight 0). That last row is the one
# a naive "periodic faces need no correction" rule gets wrong.
@inline function _diff_diag_axis(
    κ, avg, I, κc, w, wlo::Int, whi::Int, lo, hi, ::Val{N}, ::Val{D}
) where {N,D}
    δ = _unitindex(Val(N), D)
    wm = ifelse(I[D] == lo[D], wlo, -1)
    wp = ifelse(I[D] == hi[D], whi, -1)
    @inbounds (wp * avg(κc, κ[I + δ]) + wm * avg(κc, κ[I - δ])) * w
end

@inline function _diff_diag_axes(
    κ, avg, I, κc, inv_h2, wbc, lo, hi, vn::Val{N}, ::Val{1}
) where {N}
    return _diff_diag_axis(
        κ, avg, I, κc, inv_h2[1], wbc[1][1], wbc[1][2], lo, hi, vn, Val(1)
    )
end
@inline function _diff_diag_axes(
    κ, avg, I, κc, inv_h2, wbc, lo, hi, vn::Val{N}, ::Val{D}
) where {N,D}
    return _diff_diag_axes(κ, avg, I, κc, inv_h2, wbc, lo, hi, vn, Val(D - 1)) +
           _diff_diag_axis(
        κ, avg, I, κc, inv_h2[D], wbc[D][1], wbc[D][2], lo, hi, vn, Val(D)
    )
end

@inline function _diff_diag_at(
    κ::AbstractArray{<:Any,N}, I::CartesianIndex{N}, inv_h2, wbc, lo, hi, avg
) where {N}
    κc = @inbounds κ[I]
    return _diff_diag_axes(κ, avg, I, κc, inv_h2, wbc, lo, hi, Val(N), Val(N))
end

_diff_diag_weight(::Dirichlet, ::Int) = -2
_diff_diag_weight(::Neumann, ::Int) = 0
_diff_diag_weight(::Periodic, n::Int) = n == 1 ? 0 : -1

function operator_diagonal(D::Diffusion)
    g = D.grid
    g isa CartesianGrid || throw(
        ArgumentError(
            "operator_diagonal(::Diffusion) supports CartesianGrid only, got $(nameof(typeof(g)))",
        ),
    )
    _has_interface(g) && throw(
        ArgumentError(
            "operator_diagonal(::Diffusion) does not support Interface faces (forest leaves)",
        ),
    )
    N = dimension(g)
    inv_h2 = _inv_spacing2(g)
    bcs = boundary_conditions(g)
    n = local_size(g)
    h = halo_width(g)
    lo = ntuple(d -> h[d] + 1, Val(N))
    hi = ntuple(d -> h[d] + n[d], Val(N))
    wbc = ntuple(
        d -> (_diff_diag_weight(bcs[d][1], n[d]), _diff_diag_weight(bcs[d][2], n[d])), Val(N)
    )
    # Always a Field: κ varies, so the Laplacian's uniform-diagonal Number collapse does
    # not apply here even when every face is periodic.
    d = scalar_field(g, promote_type(eltype(D.κ.data), eltype(spacing(g))))
    interior(d) .=
        _diff_diag_at.(
            Ref(D.κ.data), interior(g), Ref(inv_h2), Ref(wbc), Ref(lo), Ref(hi), Ref(D.avg)
        )
    return d
end

# May alias operator state — treat as read-only.
operator_diagonal(S::ScalingOp) = S.coeff

operator_diagonal(::IdentityOp) = true

#--------------------------------------------------------------------------------# Combinator propagation

operator_diagonal(L::Scaled) = _diag_scale(L.α, operator_diagonal(L.op))
operator_diagonal(L::Added) = _diag_add(operator_diagonal(L.a), operator_diagonal(L.b))
function operator_diagonal(L::Composed)
    isdiagonal(L) || throw(
        ArgumentError(
            "the diagonal of a composition is only extractable when both factors are diagonal",
        ),
    )
    return _diag_mul(operator_diagonal(L.a), operator_diagonal(L.b))
end
operator_diagonal(L::AdjointOp) = _diag_conj(operator_diagonal(L.op))
