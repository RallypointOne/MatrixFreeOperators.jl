#--------------------------------------------------------------------------------# Fast-diagonalization direct solver

"""
    FastDiagSolver

Direct solver for a separable constant-coefficient elliptic operator `αI + βΔ` on a
uniform [`CartesianGrid`](@ref), built by [`fast_diag_solver`](@ref). Inverts the
*homogeneous* discrete operator exactly (no iteration) by fast diagonalization
(Lynch–Rice–Thomas 1964): the Laplacian is the Kronecker sum
`Δ = Σ_d I⊗…⊗D_d⊗…⊗I` of the per-axis 1-D second-derivative matrices `D_d`, each
eigendecomposed once as `D_d = S_d Λ_d S_dᵀ`. A solve is then a per-axis transform,
a pointwise division by `α + β·(λ_i+λ_j+…)`, and the inverse transforms.

Apply the inverse with `\\`/`ldiv!`; the same object also serves as a Krylov
*preconditioner*, where it is applied via `mul!` — so `mul!` and `ldiv!` coincide
(both apply `A⁻¹`), unlike a [`PreparedOperator`](@ref) whose `mul!` applies the
forward operator. Inhomogeneous boundary data folds into the right-hand side through
[`boundary_rhs`](@ref) exactly as in the Krylov path. For a singular operator (pure
Neumann or fully periodic Poisson, `α=0`) the constant null mode is projected out,
yielding the minimum-norm (zero-mean) solution; the right-hand side must satisfy the
usual compatibility condition.
"""
struct FastDiagSolver{T,N,M<:AbstractMatrix{T},A<:AbstractArray{T,N},G<:CartesianGrid{N}}
    grid::G
    Sfwd::NTuple{N,M}      # per-axis right-multiply factor for the forward transform (Sᵀ_d action)
    Sinv::NTuple{N,M}      # per-axis right-multiply factor for the inverse transform (S_d action)
    inv_denom::A           # reciprocal eigenvalue tensor, null modes zeroed
    buf1::A
    buf2::A
    sz::NTuple{N,Int}
end

"""
    fast_diag_solver(g::CartesianGrid; α=0, β=1) -> FastDiagSolver

Build a [`FastDiagSolver`](@ref) for `αI + βΔ` on `g`. Defaults solve the Poisson
problem `Δu = f`. Common shifts:

- `α=1, β=-ν*Δt` — backward-Euler implicit diffusion `(I − νΔt·Δ)uⁿ⁺¹ = uⁿ`,
  unconditionally stable (the denominator `1 + νΔt·|λ| > 0` never vanishes).
- `α=σ, β=-1` — screened Poisson / shifted Helmholtz `(σI − Δ)u = f`.

The per-axis 1-D matrices are assembled from the *actual* [`laplacian`](@ref) leaf
(including its boundary closure), so the solver inverts the same discrete operator
[`prepare`](@ref) builds for Krylov. Eigendecomposition happens once at construction;
reuse the solver across many right-hand sides (e.g. every time step).

### Examples

```julia
g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (64, 64))
S = fast_diag_solver(g)                       # direct Poisson
f = set!(scalar_field(g), x -> 2π^2 * sin(π*x[1]) * sin(π*x[2]))
u = S \\ flatten(f)                            # exact discrete solve, no iteration
```

See also: [`fast_poisson`](@ref), [`prepare`](@ref), [`boundary_rhs`](@ref).
"""
function fast_diag_solver(g::CartesianGrid{N,T}; α::Real=0, β::Real=1) where {N,T}
    αT, βT = T(α), T(β)
    sz = local_size(g)
    Smats = ntuple(Val(N)) do d
        F = LinearAlgebra.eigen(LinearAlgebra.Symmetric(_assemble_1d_laplacian(g, d, T)))
        (F.values, F.vectors)
    end
    λsum = zeros(T, sz...)
    for d in 1:N
        shp = ntuple(k -> k == d ? sz[d] : 1, Val(N))
        λsum .+= reshape(Smats[d][1], shp)
    end
    denom = αT .+ βT .* λsum
    tol = sqrt(eps(T)) * maximum(abs, denom)
    inv_denom = map(x -> abs(x) < tol ? zero(T) : inv(x), denom)

    backend = KernelAbstractions.get_backend(g)
    Sfwd = ntuple(d -> _to_device(backend, Smats[d][2]), Val(N))
    Sinv = ntuple(d -> _to_device(backend, permutedims(Smats[d][2])), Val(N))
    return FastDiagSolver(
        g, Sfwd, Sinv, _to_device(backend, inv_denom),
        KernelAbstractions.zeros(backend, T, sz...),
        KernelAbstractions.zeros(backend, T, sz...), sz,
    )
end

"""
    fast_poisson(g::CartesianGrid) -> FastDiagSolver

Direct Poisson solver `Δu = f` — `fast_diag_solver(g; α=0, β=1)`.
"""
fast_poisson(g::CartesianGrid) = fast_diag_solver(g; α=0, β=1)

# Dense N×N 1-D Laplacian along axis `d`, assembled from the real operator (prepare +
# basis-vector sweep) so the eigendata matches the leaf's boundary closure exactly.
function _assemble_1d_laplacian(g::CartesianGrid, d::Integer, ::Type{T}) where {T}
    n = local_size(g)[d]
    g1 = CartesianGrid(
        (g.extent[d],), (n,);
        bc=(boundary_conditions(g)[d],), halo=(halo_width(g)[d],),
    )
    P = prepare(laplacian(g1), scalar_field(g1, T))
    D = Matrix{T}(undef, n, n)
    e = zeros(T, n)
    y = Vector{T}(undef, n)
    for j in 1:n
        fill!(e, zero(T))
        e[j] = one(T)
        mul!(y, P, e)
        @views D[:, j] .= y
    end
    return D
end

function _to_device(backend, A::AbstractArray{T}) where {T}
    dev = KernelAbstractions.zeros(backend, T, size(A)...)
    copyto!(dev, A)
    return dev
end

#--------------------------------------------------------------------------------# Mode-d transforms and solve

# Mode-`d` product `Y = (Rᵀ) ×_d A`: apply `Rᵀ` along axis `d` of the interior array.
# Reshaping to (L, n_d, R) groups the faster/slower axes so each line solve is a dense
# right-multiply by `R` — the paper's multi-RHS reshape, device-agnostic (mul! only).
function _ttm!(Y, A, R::AbstractMatrix, sz::NTuple{N,Int}, d::Int) where {N}
    L = d == 1 ? 1 : prod(ntuple(k -> sz[k], d - 1))
    nd = sz[d]
    Rr = d == N ? 1 : prod(ntuple(k -> sz[d + k], N - d))
    Yr = reshape(Y, L, nd, Rr)
    Ar = reshape(A, L, nd, Rr)
    for r in 1:Rr
        @views mul!(Yr[:, :, r], Ar[:, :, r], R)
    end
    return Y
end

# Apply the per-axis factors `mats[d]` along every dimension, ping-ponging between
# `dst` and `scratch`; result lands in `dst`.
function _apply_all_modes!(dst, src, mats::NTuple{N}, sz::NTuple{N,Int}, scratch) where {N}
    from = src
    for d in 1:N
        to = isodd(d) ? dst : scratch
        _ttm!(to, from, mats[d], sz, d)
        from = to
    end
    from === dst || copyto!(dst, from)
    return dst
end

function _solve!(u_arr, f_arr, P::FastDiagSolver)
    _apply_all_modes!(P.buf1, f_arr, P.Sfwd, P.sz, P.buf2)
    P.buf1 .*= P.inv_denom
    _apply_all_modes!(u_arr, P.buf1, P.Sinv, P.sz, P.buf2)
    return u_arr
end

#--------------------------------------------------------------------------------# Solver interface

Base.eltype(::FastDiagSolver{T}) where {T} = T
Base.size(P::FastDiagSolver) = (prod(P.sz), prod(P.sz))
Base.size(P::FastDiagSolver, i::Integer) = i <= 2 ? prod(P.sz) : 1

function _check_length(P::FastDiagSolver, v::AbstractVector)
    n = prod(P.sz)
    length(v) == n ||
        throw(DimensionMismatch("vector length $(length(v)) ≠ $n interior DOFs of the solver grid"))
    return n
end

"""
    ldiv!(u, P::FastDiagSolver, f) -> u
    ldiv!(P::FastDiagSolver, x) -> x

Solve `(αI + βΔ) u = f` into `u` (or in place into `x`) by fast diagonalization.
"""
function LinearAlgebra.ldiv!(u::AbstractVector, P::FastDiagSolver, f::AbstractVector)
    _check_length(P, f)
    _check_length(P, u)
    _solve!(reshape(u, P.sz), reshape(f, P.sz), P)
    return u
end
function LinearAlgebra.ldiv!(P::FastDiagSolver, x::AbstractVector)
    _check_length(P, x)
    xa = reshape(x, P.sz)
    _solve!(xa, xa, P)
    return x
end

Base.:\(P::FastDiagSolver, f::AbstractVector) = LinearAlgebra.ldiv!(similar(f), P, f)

# As a Krylov preconditioner the solver applies A⁻¹, so mul! ≡ ldiv!.
LinearAlgebra.mul!(y::AbstractVector, P::FastDiagSolver, x::AbstractVector) =
    LinearAlgebra.ldiv!(y, P, x)
