#--------------------------------------------------------------------------------# Face-coefficient averaging policies

"""
    ArithmeticMean()

Arithmetic face averaging `κ_f = (κ_a + κ_b)/2`. The default for [`diffusion`](@ref):
second-order for smooth coefficients, and imposes no sign restriction on `κ`.
"""
struct ArithmeticMean end

"""
    HarmonicMean()

Harmonic face averaging `κ_f = 2κ_aκ_b/(κ_a + κ_b)`. Conserves flux across a jump in
`κ`, so it is the right choice for piecewise-constant or discontinuous media. Requires
`κ > 0` — the expression is singular at `κ_a = -κ_b` and `NaN` at `(0, 0)`.

Note for inverse problems: the harmonic mean is dominated by the *smaller* of its two
arguments, so `∂κ_f/∂κ_large → 0` as the contrast grows. That weakens identifiability of
the high-`κ` side of an interface. It is a real property of the discretization, not a bug.
"""
struct HarmonicMean end

@inline (::ArithmeticMean)(a, b) = (a + b) / 2
@inline (::HarmonicMean)(a, b) = 2 * a * b / (a + b)

#--------------------------------------------------------------------------------# Coefficient ghost extension

"""
    fill_coefficient_ghosts!(data, g::AbstractGrid) -> data

Extend a *coefficient* array into its ghost layers: periodic wrap where the dimension is
periodic, even (symmetric) mirror at every physical wall, and `Interface` faces left
untouched because those ghosts hold a neighbour's coefficient and are an external input.

Deliberately **not** [`apply_bc!`](@ref), which would antisymmetrize the coefficient at a
Dirichlet wall — the same hazard `_average_to_coarse` avoids by not using `Restriction`.
The even mirror is what makes a wall face self-average to the adjacent cell's value, which
is the one-sided face coefficient the finite-volume balance calls for.
"""
function fill_coefficient_ghosts!(data::AbstractArray{<:Any,N}, g::AbstractGrid{N}) where {N}
    _coeff_ghost_dims!(data, g, Val(1), boundary_conditions(g))
    return data
end

function _coeff_ghost_dims!(data, g, ::Val{D}, bcs::Tuple) where {D}
    lo, hi = first(bcs)
    h = halo_width(g)[D]
    n = local_size(g)[D]
    for k in 1:h
        _copy_coeff_ghost!(data, Val(D), h + 1 - k, lo, _source_low(lo, h, n, k))
        _copy_coeff_ghost!(data, Val(D), h + n + k, hi, _source_high(hi, h, n, k))
    end
    return _coeff_ghost_dims!(data, g, Val(D + 1), Base.tail(bcs))
end
_coeff_ghost_dims!(data, g, ::Val, ::Tuple{}) = nothing

_copy_coeff_ghost!(data, ::Val, ghost::Int, ::Interface, source::Int) = nothing
function _copy_coeff_ghost!(data, dim::Val, ghost::Int, ::AbstractBC, source::Int)
    dst = _dimslice(data, dim, ghost:ghost)
    src = _dimslice(data, dim, source:source)
    dst .= src
    return nothing
end

#--------------------------------------------------------------------------------# Stencil

# Dimensions are unrolled by RECURSION, not by an `ntuple(Val(N)) do d` closure. With two
# arrays in play the closure stops inlining and the broadcast loses vectorization
# entirely — measured 260 µs vs 33 µs for one 256² sweep, an 8× cliff with identical
# numerics. Keep this closure-free.
@inline function _diff_axis(u, κ, avg, I, uc, κc, w, ::Val{N}, ::Val{D}) where {N,D}
    δ = _unitindex(Val(N), D)
    Ip = I + δ
    Im = I - δ
    @inbounds (avg(κc, κ[Ip]) * (u[Ip] - uc) - avg(κc, κ[Im]) * (uc - u[Im])) * w
end

@inline function _diff_axes(u, κ, avg, I, uc, κc, inv_h2, vn::Val{N}, ::Val{1}) where {N}
    return _diff_axis(u, κ, avg, I, uc, κc, inv_h2[1], vn, Val(1))
end
@inline function _diff_axes(u, κ, avg, I, uc, κc, inv_h2, vn::Val{N}, ::Val{D}) where {N,D}
    return _diff_axes(u, κ, avg, I, uc, κc, inv_h2, vn, Val(D - 1)) +
           _diff_axis(u, κ, avg, I, uc, κc, inv_h2[D], vn, Val(D))
end

"""
    diffusion_stencil(u, κ, I::CartesianIndex, inv_h2::NTuple, avg) -> (uc, div)

Per-cell compact flux-form stencil: returns the center value `uc` and

    div = Σ_d [ κ_{I+½δ}·(u[I+δ] − uc) − κ_{I−½δ}·(uc − u[I−δ]) ] / Δd²

the finite-volume balance of the face fluxes `κ∇u`, with `κ_{I±½δ} = avg(κ[I], κ[I±δ])`.
This is the single stencil body — the built-in [`Diffusion`](@ref) leaf calls it, and
custom fused operators (§4a of the design) must reuse it so the numerical definition never
forks. Ghost layers of **both** `u` and `κ` must be filled first: `u` by
[`apply_bc!`](@ref), `κ` by [`fill_coefficient_ghosts!`](@ref).

Every equation couples `κ` and `u` at *adjacent* cells, which is what the wide
`divergence ∘ scaling ∘ gradient` composition does not do — see [`diffusion`](@ref).

### Examples

```julia
uc, div = diffusion_stencil(u.data, κ.data, CartesianIndex(2, 2),
                            inv.(spacing(g) .^ 2), ArithmeticMean())
```
"""
@inline function diffusion_stencil(
    u::AbstractArray{<:Any,N},
    κ::AbstractArray{<:Any,N},
    I::CartesianIndex{N},
    inv_h2::NTuple{N},
    avg,
) where {N}
    uc = @inbounds u[I]
    κc = @inbounds κ[I]
    return (uc, _diff_axes(u, κ, avg, I, uc, κc, inv_h2, Val(N), Val(N)))
end

@inline function _diff_at(u, κ, I, inv_h2, avg)
    return diffusion_stencil(u, κ, I, inv_h2, avg)[2]
end

# Transpose gather. The operator is symmetric face by face, so the flipped stencil is the
# forward stencil — bounds-masked, and evaluated at ghost cells too, where it produces
# exactly the weight the abutting interior row carries into that ghost. A masked κ read
# beyond the array only ever multiplies a masked (zero) cotangent.
@inline function _diff_adjoint_axis(ȳ, κ, avg, J, yc, κc, w, ::Val{N}, ::Val{D}) where {N,D}
    δ = _unitindex(Val(N), D)
    Jp = J + δ
    Jm = J - δ
    return (
        avg(κc, _maskedget(κ, Jp)) * (_maskedget(ȳ, Jp) - yc) -
        avg(κc, _maskedget(κ, Jm)) * (yc - _maskedget(ȳ, Jm))
    ) * w
end

@inline function _diff_adjoint_axes(ȳ, κ, avg, J, yc, κc, inv_h2, vn::Val{N}, ::Val{1}) where {N}
    return _diff_adjoint_axis(ȳ, κ, avg, J, yc, κc, inv_h2[1], vn, Val(1))
end
@inline function _diff_adjoint_axes(
    ȳ, κ, avg, J, yc, κc, inv_h2, vn::Val{N}, ::Val{D}
) where {N,D}
    return _diff_adjoint_axes(ȳ, κ, avg, J, yc, κc, inv_h2, vn, Val(D - 1)) +
           _diff_adjoint_axis(ȳ, κ, avg, J, yc, κc, inv_h2[D], vn, Val(D))
end

@inline function _diff_adjoint_gather(
    ȳ::AbstractArray{<:Any,N},
    κ::AbstractArray{<:Any,N},
    J::CartesianIndex{N},
    inv_h2::NTuple{N},
    avg,
) where {N}
    yc = _maskedget(ȳ, J)
    κc = _maskedget(κ, J)
    return _diff_adjoint_axes(ȳ, κ, avg, J, yc, κc, inv_h2, Val(N), Val(N))
end

#--------------------------------------------------------------------------------# Leaf

"""
    Diffusion(grid, κ, avg)

Compact flux-form variable-coefficient diffusion `∇·(κ∇u)`. Construct with
[`diffusion`](@ref), which is what extends `κ` into its ghost layers; the inner
constructor takes `κ` as given and is the seam the forest and distributed paths will use
to supply cross-block coefficient ghosts themselves.
"""
struct Diffusion{G<:AbstractGrid,K,AV} <: AbstractOperator
    grid::G
    κ::K
    avg::AV
end

"""
    diffusion(g::AbstractGrid, κ::Field; averaging=ArithmeticMean(), check=true) -> Diffusion

Build a matrix-free variable-coefficient diffusion operator `∇·(κ∇u)` on `g`, discretized
in **compact flux form**: fluxes `q_{i+½} = κ_{i+½}(u_{i+1} − u_i)/Δ` live on faces, with
`κ` averaged to faces by `averaging`, and the cell balance is `(q_{i+½} − q_{i−½})/Δ`.

Prefer this to the algebraic composition `divergence(g) * scaling(κ) * gradient(g)`. That
composition chains two centered first differences and so carries a *wide* 2Δ stencil: the
equation at cell `I` samples the flux only at neighbours `I±e`, never at `I`, and no
equation ever couples `κ` at two adjacent cells. For a parameter inversion in `κ` this
decouples the even and odd `(i+j)`-parity sublattices exactly — they are fit to disjoint
halves of the data and tied together only by the regularizer. The compact form couples
adjacent cells by construction, so the checkerboard mode is not in the null space.

The compact form is also **exactly symmetric** for real `κ` under all built-in boundary
conditions (the composed form is not), and it has an [`operator_diagonal`](@ref) (the
composed form cannot), which is what makes it usable with the multigrid smoothers.

With `κ ≡ c` constant this reduces to `c * laplacian(g)` up to floating-point operation
order.

Boundary treatment is exact under the package's homogeneous ghost fills, which reflect
about the *face*: a Neumann face carries zero flux, so its face coefficient is irrelevant;
a Dirichlet face carries `κ_I(0 − u_I)/(Δ/2)`, which the one-sided face coefficient
`κ_f = κ_I` reproduces exactly. That one-sided value is O(Δ) accurate in `κ` at a wall
while interior faces are O(Δ²), so wall-adjacent `κ` cells have a different sensitivity
structure than interior ones — worth knowing when inverting for `κ`.

`κ` must be a scalar-eltype [`Field`](@ref) matching `g`. The operator stores its **own**
copy with ghost layers extended by [`fill_coefficient_ghosts!`](@ref), so whatever `κ`
holds in its ghosts is ignored — and so mutating `κ` after construction does not affect
the operator. Pass `check=false` to skip the `κ > 0` validation that
[`HarmonicMean`](@ref) requires, which is what an optimization loop wants: the operator is
rebuilt every objective evaluation and the check would run inside the differentiated
region.

Building the leaf inside that region — which is exactly what an inversion loop does —
needs `Enzyme.set_runtime_activity(Enzyme.Reverse)`: the operator stores the `Const` grid
alongside the active coefficient, and static activity analysis cannot clear that store
before Julia 1.12. See `examples/inverse_diffusion.jl`, which sets it once on the backend
object.

### Examples

```julia
g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (64, 64))
κ = set!(scalar_field(g), x -> 1 + x[1]^2)
L = diffusion(g, κ)                                  # ∇·(κ∇u), compact 5-point
Lj = diffusion(g, κ; averaging=HarmonicMean())       # flux-conserving across jumps
```

See also: [`laplacian`](@ref), [`scaling`](@ref), [`diffusion_stencil`](@ref).
"""
function diffusion(g::AbstractGrid, κ::Field; averaging=ArithmeticMean(), check::Bool=true)
    g isa CartesianGrid || throw(
        ArgumentError(
            "diffusion supports CartesianGrid only, got $(nameof(typeof(g))); " *
            "BlockForest support is tracked in issue #54",
        ),
    )
    _has_interface(g) && throw(
        ArgumentError(
            "diffusion does not support Interface faces yet (forest leaves and partition " *
            "slabs); the coefficient has no cross-block ghost values to average — see issue #54",
        ),
    )
    eltype(κ.data) <: Number || throw(
        ArgumentError(
            "diffusion coefficient must be a scalar-eltype Field, got eltype $(eltype(κ.data))",
        ),
    )
    padded_size(κ.grid) == padded_size(g) || throw(
        ArgumentError(
            "diffusion coefficient is sized for $(padded_size(κ.grid)) but the grid is " *
            "$(padded_size(g)); κ must live on the same grid as the operator",
        ),
    )
    if check && averaging isa HarmonicMean
        all(>(zero(eltype(κ.data))), interior(κ)) || throw(
            ArgumentError(
                "HarmonicMean requires κ > 0 everywhere (2κaκb/(κa+κb) is singular " *
                "otherwise); pass check=false to skip this validation",
            ),
        )
    end
    return Diffusion(g, _extended_coeff(κ, g), averaging)
end

# The leaf's own ghost-extended copy. Built out of place so the caller's coefficient is
# never mutated and the whole thing stays on an AD tape.
function _extended_coeff(κ::Field, g::AbstractGrid)
    data = copy(κ.data)
    fill_coefficient_ghosts!(data, g)
    return Field(data, g)
end

islinear(::Diffusion) = true
isconstant(::Diffusion) = true
isdiagonal(::Diffusion) = false
isselfadjoint(D::Diffusion) = _selfadjoint_grid(D.grid) && eltype(D.κ.data) <: Real
operator_grid(D::Diffusion) = D.grid

# Complex κ makes the operator symmetric but not Hermitian, and both means commute with
# conjugation — so the conjugated-coefficient leaf is the exact adjoint, cheaper than the
# lazy wrapper. Conjugation commutes with the ghost extension, so no refill is needed.
function _conj_op(D::Diffusion)
    eltype(D.κ.data) <: Real && return D
    return Diffusion(D.grid, Field(conj.(D.κ.data), D.κ.grid), D.avg)
end

function adjoint_operator(D::Diffusion)
    _selfadjoint_grid(D.grid) || return AdjointOp(D)
    return _conj_op(D)
end

function apply!(y::Field, D::Diffusion, x::Field, g::AbstractGrid, α, β)
    halo_update!(x, g)
    apply_bc!(x)
    return _apply_raw!(y, D, x, g, α, β)
end

# No BC prologue: boundary_rhs feeds this the inhomogeneous ghost offsets alone, and
# apply_bc! would overwrite them.
function _apply_raw!(y::Field, D::Diffusion, x::Field, g::AbstractGrid, α, β)
    inv_h2 = _inv_spacing2(g)
    κ = D.κ.data
    avg = D.avg
    yi = interior(y)
    if iszero(β)
        yi .= α .* _diff_at.(Ref(x.data), Ref(κ), interior(g), Ref(inv_h2), Ref(avg))
    else
        yi .=
            α .* _diff_at.(Ref(x.data), Ref(κ), interior(g), Ref(inv_h2), Ref(avg)) .+
            β .* yi
    end
    return y
end

function apply_adjoint!(x̄::Field, D::Diffusion, ȳ::Field, g::AbstractGrid, α, β)
    # On an all-physical-BC grid the homogeneous fill makes the interior→interior map
    # symmetric, so the forward action of the conjugated leaf IS the adjoint. A forest
    # leaf's Interface ghosts are external inputs: the mechanical transpose must scatter
    # cotangents into them for halo_update_adjoint! to fold across blocks.
    _has_interface(g) || return apply!(x̄, _conj_op(D), ȳ, g, α, β)
    inv_h2 = _inv_spacing2(g)
    κ = _conj_op(D).κ.data
    avg = D.avg
    gather = let κ = κ, inv_h2 = inv_h2, avg = avg
        (u, J) -> _diff_adjoint_gather(u, κ, J, inv_h2, avg)
    end
    return adjoint_gather!(x̄, ȳ, gather, α, β)
end

function Adapt.adapt_structure(to, D::Diffusion)
    return Diffusion(Adapt.adapt(to, D.grid), Adapt.adapt(to, D.κ), D.avg)
end

Base.show(io::IO, D::Diffusion) = print(io, "Diffusion(", nameof(typeof(D.avg)), ")")
