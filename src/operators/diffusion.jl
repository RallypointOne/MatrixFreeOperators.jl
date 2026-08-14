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

#--------------------------------------------------------------------------------# Coefficient ghosts across a forest

"""
    fill_coefficient_ghosts!(κ::AbstractBlockField, bf::BlockForest) -> κ

Extend a *coefficient* field into every block's ghost layers: same-level face copies,
coarse→fine injection, fine→coarse volume averaging, and an even mirror at physical walls.
The forest counterpart of the single-grid method above, run once by [`diffusion`](@ref) at
construction so the per-application exchange count is unchanged from the `Laplacian`
baseline. A forest has no global coefficient array to window, so unlike a partition slab
(`_slab_coeff_field`) this is a real exchange — legitimate only because κ is constant
through a solve.

Deliberately **not** [`halo_update!`](@ref), whose coarse–fine phases are tuned for a
*solution*: the Martin–Cartwright quadratic interpolant assumes a smoothness a material
coefficient need not have, and the flux-matching restriction is a stencil on `u`, not a
volume average. Deliberately not [`apply_bc!`](@ref) either, which antisymmetrizes at a
Dirichlet wall and would negate the coefficient there — the same hazard the single-grid
method and `_average_to_coarse` document.

Written as a plain topology walk rather than through the cached [`ExchangeSchedule`](@ref)
descriptors on purpose. It must stay differentiable with respect to κ, and the `GhostFill`
descriptor loop is precisely the shape the halo Enzyme rules exist to hide (issue #26);
those rules report `nothing` for every derivative slot, so routing a coefficient through
them would silently zero its gradient. Recomputing topology here is free — this runs once
per operator, never per application.
"""
function fill_coefficient_ghosts!(κ::AbstractBlockField, bf::BlockForest{N}) where {N}
    _require_current(κ)
    forest = bf.forest
    forest.uniform[] || _validate_coarse_fine(bf)
    store, lay = _storage(κ), _layout(κ)
    h, n = bf.halo, bf.blocksize
    # Dimensions outermost and in order 1:N, so a corner ghost ends up a consistent
    # ghost-of-ghost value — the ordering the single-grid method and `apply_bc!` use.
    for d in 1:N, (i, K) in enumerate(forest.leaves), side in (-1, 1)
        nbr = face_neighbor(forest, K, d, side)
        if nbr === nothing
            _coeff_mirror!(store, lay, i, d, side, h, n)
            continue
        end
        if is_leaf(forest, nbr)
            _coeff_face_copy!(store, lay, leaf_index(forest, nbr), i, d, side, h, n)
            continue
        end
        # A coarse–fine fill spans the tangential *interiors* only, matching the solution
        # path. Mirroring first leaves this plane's tangential fringe holding the block's
        # own value rather than a zero — which `HarmonicMean` would turn into a NaN in the
        # adjoint gather, since 2·0·0/(0+0) is not 0. No stencil reads those cells.
        _coeff_mirror!(store, lay, i, d, side, h, n)
        cover = leaf_covering(forest, nbr)
        if cover !== nothing                      # this leaf is the finer side
            cover.level == K.level - 1 || _throw_unbalanced(K)
            _coeff_inject!(store, lay, bf, i, K, leaf_index(forest, cover), d, side)
        else                                      # this leaf is the coarser side
            _coeff_average!(store, lay, bf, i, K, nbr, d, side)
        end
    end
    return κ
end

# Even mirror of a block's own interior into one ghost face: the physical-wall fill, and
# the seed a coarse–fine fill's tangential fringe keeps.
function _coeff_mirror!(
    store, lay::BlockLayout, i::Int, d::Int, side::Int, h::NTuple{N,Int}, n::NTuple{N,Int}
) where {N}
    hd, nd = h[d], n[d]
    for k in 1:hd
        ghost, source = side == -1 ? (hd + 1 - k, hd + k) : (hd + nd + k, hd + nd + 1 - k)
        _leaf_view(store, lay, i, _face_box(d, ghost:ghost, h, n)) .=
            _leaf_view(store, lay, i, _face_box(d, source:source, h, n))
    end
    return nothing
end

# Same-level face: an exact copy, identical to the solution path's `_run_copies!` — the one
# phase where a coefficient and a solution want the same thing.
function _coeff_face_copy!(
    store, lay::BlockLayout, src::Int, dst::Int, d::Int, side::Int,
    h::NTuple{N,Int}, n::NTuple{N,Int},
) where {N}
    ghost, source = _face_slabs(side, h[d], n[d])
    _leaf_view(store, lay, dst, _face_box(d, ghost, h, n)) .=
        _leaf_view(store, lay, src, _face_box(d, source, h, n))
    return nothing
end

# The two same-parity runs of fine ghost columns sharing a coarse column range: fine ghost
# column j (padded j+1) lies inside coarse column (q·nt + j + 1) >> 1 (padded +1), so each
# coarse column covers exactly one even and one odd fine column. The same 2:1 mapping
# `_tangential_classes` encodes, minus the tangential quadratic a coefficient does not want.
@inline function _coeff_tang_classes(nt::Int, q::Int)
    half = nt >> 1
    coarse = q == 0 ? (2:(half + 1)) : ((half + 2):(nt + 1))
    return ((2:2:nt, coarse), (3:2:(nt + 1), coarse))
end

# Coarse→fine: the fine ghost cell lies wholly inside one coarse cell, so the coarse value
# already IS the volume average of κ over the ghost's footprint. Inject it.
function _coeff_inject!(
    store, lay::BlockLayout, bf::BlockForest{N}, i::Int, K::LeafKey{N}, ci::Int,
    d::Int, side::Int,
) where {N}
    n = bf.blocksize
    nd = n[d]
    g_n = side == -1 ? 1 : nd + 2                # this leaf's ghost layer
    U1_n = side == -1 ? nd + 1 : 2               # coarse first interior layer at the face
    tdims = Tuple(filter(!=(d), ntuple(identity, Val(N))))
    classlists = map(t -> _coeff_tang_classes(n[t], K.coords[t] & 1), tdims)
    for combo in Iterators.product(classlists...)
        fine_t, coarse_t = map(first, combo), map(last, combo)
        _leaf_view(store, lay, i, _cf_box(Val(N), d, g_n:1:g_n, tdims, fine_t)) .=
            _leaf_view(store, lay, ci, _cf_box(Val(N), d, U1_n:1:U1_n, tdims, coarse_t))
    end
    return nothing
end

# Fine→coarse: the coarse ghost cell's footprint is exactly 2ᴺ fine cells — two layers deep
# normal to the face, 2^(N−1) across it — so its volume average is their plain mean. That is
# the policy `_average_to_coarse`/`_child_mean` already use to coarsen a coefficient, and it
# is emphatically not the solution path's flux-matching restriction.
function _coeff_average!(
    store, lay::BlockLayout, bf::BlockForest{N}, i::Int, K::LeafKey{N}, nbr::LeafKey{N},
    d::Int, side::Int,
) where {N}
    forest, n = bf.forest, bf.blocksize
    nd = n[d]
    gC_n = side == -1 ? 1 : nd + 2                        # this leaf's ghost layer
    layers = side == -1 ? (nd + 1, nd) : (2, 3)           # the two fine layers beneath it
    tdims = Tuple(filter(!=(d), ntuple(identity, Val(N))))
    facing = side == 1 ? 0 : 1                            # child d-bit on the shared face
    for child in children(nbr)
        (child.coords[d] & 1) == facing || continue
        is_leaf(forest, child) || _throw_unbalanced(K)
        cj = leaf_index(forest, child)
        dst_t = map(tdims) do t
            lo = 2 + (child.coords[t] & 1) * (n[t] >> 1)
            lo:1:(lo + (n[t] >> 1) - 1)
        end
        dst = _leaf_view(store, lay, i, _cf_box(Val(N), d, gC_n:1:gC_n, tdims, dst_t))
        w = one(eltype(dst)) / (1 << N)
        fill!(dst, zero(eltype(dst)))
        for layer in layers,
            parities in Iterators.product(ntuple(_ -> (0, 1), length(tdims))...)

            fine_t = map(tdims, parities) do t, p
                (2 + p):2:(n[t] + p)
            end
            dst .+= w .* _leaf_view(store, lay, cj, _cf_box(Val(N), d, layer:1:layer, tdims, fine_t))
        end
    end
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
constructor takes `κ` as given and is the seam for supplying cross-block coefficient
ghosts from outside: `_slab_op` (`src/distributed.jl`) already builds through it, with a κ
sliced from the global one *including* its ghosts, and the forest path (issue #58) will.
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

Prefer this to the algebraic composition `divergence(g) * scaling(κ) * gradient(g)`. That composition chains two centered first differences and carries a *wide* 2Δ stencil: interior rows sample fluxes at `I±e` and reach solution values at `I±2e`, skipping adjacent solution cells. The compact form couples adjacent solution cells and removes this odd–even decoupling in `u`.

Coefficient identifiability is a separate question. With [`ArithmeticMean`](@ref), a checkerboard perturbation `δκ[i,j] = ε(-1)^(i+j)` cancels at every interior face. On periodic grids with even cell counts in every dimension, or with homogeneous Neumann walls, it leaves the entire operator unchanged for every `u`. Multiple excitations cannot resolve that ambiguity. Dirichlet wall coefficients use the adjacent cell's κ and can break it; the successful Dirichlet inversion in `examples/inverse_diffusion.jl` does not establish unique recovery for other boundary conditions or data.

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
            "diffusion supports CartesianGrid and BlockForest, got $(nameof(typeof(g))); " *
            "on a BlockForest the coefficient must be a BlockField on the same forest",
        ),
    )
    _has_interface(g) && throw(
        ArgumentError(
            "diffusion cannot build directly on a grid with Interface faces: κ's " *
            "cross-block ghosts are an external input, and this entry point has only the " *
            "one grid to fill them from. On a partition slab, build it on the undistributed " *
            "grid and pass that to prepare_distributed, which slices κ with its ghosts onto " *
            "each slab. On a BlockForest leaf, build it on the whole forest with a " *
            "BlockField coefficient, which exchanges them",
        ),
    )
    eltype(κ.data) <: Number || throw(
        ArgumentError(
            "diffusion coefficient must be a scalar-eltype Field, got eltype $(eltype(κ.data))",
        ),
    )
    # Equal padded sizes can hide different interior/halo layouts. Reuse the
    # discretization comparison so copying κ cannot reinterpret ghosts as cells.
    _same_grid(κ.grid, g) || throw(
        ArgumentError(
            "diffusion coefficient must live on the same grid as the operator " *
            "(matching extent, spacing, size, halo, boundary conditions, local range, " *
            "and topology)",
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

"""
    diffusion(bf::BlockForest, κ::AbstractBlockField; averaging=ArithmeticMean(), check=true) -> Diffusion

Build the same compact flux-form operator over a block-structured forest. Semantics,
boundary treatment, and the constant-κ reduction to `c * laplacian(bf)` are exactly the
single-grid method's; only the coefficient's ghost fill differs, because a forest's blocks
are separate arrays. `κ` must be a scalar-eltype block field on **the same forest** as
`bf` — `_leaf_op` slices it by this forest's leaf indices, so a coefficient bound to a
different topology would alias the wrong leaves.

The operator stores its own copy with every ghost layer filled by
[`fill_coefficient_ghosts!`](@ref), so mutating `κ` afterwards does not affect it, and the
exchange is paid once here rather than per application.

On a **non-uniform** forest coarse–fine coupling is not symmetric, so `isselfadjoint` is
false (via `_selfadjoint_grid`) and the adjoint runs the declared transpose gather rather
than the forward action. `operator_diagonal` is unavailable on forest leaves, matching
[`laplacian`](@ref).

### Examples

```julia
bf = BlockForest(CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16)); blocksize=(8, 8), maxlevel=2)
κ = set!(scalar_field(bf), x -> 1 + x[1]^2)
L = diffusion(bf, κ)
```
"""
function diffusion(
    bf::BlockForest, κ::AbstractBlockField; averaging=ArithmeticMean(), check::Bool=true
)
    eltype(κ) <: Number || throw(
        ArgumentError(
            "diffusion coefficient must be a scalar-eltype field, got eltype $(eltype(κ))"
        ),
    )
    # Topology identity, not wrapper identity: adaptation clones the BlockForest but shares
    # the forest and its generation Ref, so `===` on the wrapper would reject device twins.
    κ.grid.forest === bf.forest || throw(
        ArgumentError(
            "diffusion coefficient must be a field on the same forest as the operator"
        ),
    )
    if check && averaging isa HarmonicMean
        _all_positive(κ) || throw(
            ArgumentError(
                "HarmonicMean requires κ > 0 everywhere (2κaκb/(κa+κb) is singular " *
                "otherwise); pass check=false to skip this validation",
            ),
        )
    end
    return Diffusion(bf, _extended_coeff(κ, bf), averaging)
end

function _all_positive(κ::AbstractBlockField)
    z = zero(eltype(κ))
    return all(i -> all(>(z), interior(block(κ, i))), 1:nleaves(κ.grid))
end

# The leaf's own ghost-extended copy. Built out of place so the caller's coefficient is
# never mutated and the whole thing stays on an AD tape.
function _extended_coeff(κ::Field, g::AbstractGrid)
    data = copy(κ.data)
    fill_coefficient_ghosts!(data, g)
    return Field(data, g)
end

_extended_coeff(κ::AbstractBlockField, bf::BlockForest) =
    fill_coefficient_ghosts!(copy(κ), bf)

islinear(::Diffusion) = true
isconstant(::Diffusion) = true
isdiagonal(::Diffusion) = false
isselfadjoint(D::Diffusion) = _selfadjoint_grid(D.grid) && eltype(D.κ) <: Real
operator_grid(D::Diffusion) = D.grid

# Complex κ makes the operator symmetric but not Hermitian, and both means commute with
# conjugation — so the conjugated-coefficient leaf is the exact adjoint, cheaper than the
# lazy wrapper. Conjugation commutes with the ghost extension, so no refill is needed.
function _conj_op(D::Diffusion)
    eltype(D.κ) <: Real && return D
    return Diffusion(D.grid, _conj_coeff(D.κ), D.avg)
end

_conj_coeff(κ::Field) = Field(conj.(κ.data), κ.grid)
# Layout-agnostic, so it serves BlockField and PackedBlockField alike.
function _conj_coeff(κ::AbstractBlockField)
    c = similar(κ)
    for i in 1:nleaves(κ.grid)
        _block_array(c, i) .= conj.(_block_array(κ, i))
    end
    return c
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
