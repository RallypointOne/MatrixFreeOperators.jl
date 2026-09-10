#--------------------------------------------------------------------------------# Operators on a BlockForest

# The forest action reuses the single-grid operator path per leaf-block unchanged:
# halo_update! fills inter-block ghosts ONCE over the whole forest (via the
# per-generation exchange schedule), the forest-level apply_bc! face pass fills
# physical-boundary ghosts, then each leaf runs the ordinary apply! (whose own
# halo_update! and apply_bc! are no-ops on the all-Interface leaf
# CartesianGrid). Combinators (Added, Scaled, Composed,
# AdjointOp) recurse at the FOREST level — never per leaf — so a nested adjoint
# always reaches the forest adjoint action and its cross-block fold, and a
# Composed intermediate is a whole BlockField whose operand application performs
# its own inter-block exchange — except that an Added whose operands all
# `shares_exchange` fills one exchange for the set (below).

"""
    apply!(y::AbstractBlockField, L::AbstractOperator, x::AbstractBlockField, g::BlockForest, α=true, β=false) -> y

Apply `L` over a block-structured forest: exchange inter-block halos once, then run
the stencil sweep over every block.
"""
function apply!(
    y::AbstractBlockField, L::AbstractOperator, x::AbstractBlockField, g::BlockForest, α, β
)
    return _forest_exchange_sweep!(y, L, x, g, α, β)
end

# The forest action itself: fill x's ghosts, then sweep. Shared by the apply! above,
# the prepared hot path (linalg.jl), and the exchange-sharing Added below.
function _forest_exchange_sweep!(
    y::AbstractBlockField, L::AbstractOperator, x::AbstractBlockField, g::BlockForest, α, β
)
    _require_current(y)
    halo_update!(x, g)                  # also checks x's generation
    apply_bc!(x, g)                     # physical faces — leaf grids are all-Interface
    return _forest_sweep!(y, L, x, g, α, β)
end

# The stencil sweep behind the halo/BC fills — the dispatch seam forest-native
# kernels override per (operator, packed layout). The reference sweep runs the
# ordinary single-grid apply! per leaf; packed fields hit it too, via block views,
# and kernel overrides fall back to it on non-GPU backends (broadcast fusion beats
# KA CPU codegen).
function _forest_sweep!(
    y::AbstractBlockField, L::AbstractOperator, x::AbstractBlockField, g::BlockForest, α, β
)
    return _forest_sweep_leaves!(y, L, x, g, α, β)
end

function _forest_sweep_leaves!(
    y::AbstractBlockField, L::AbstractOperator, x::AbstractBlockField, g::BlockForest, α, β
)
    for i in 1:nleaves(g)
        lg = leaf_grid(g, i)
        apply!(block(y, i, lg), _leaf_op(L, i, lg), block(x, i, lg), lg, α, β)
    end
    return y
end

# Per-leaf slicing of coefficient-carrying leaves: the reference sweeps see
# ordinary single-grid operators whose coefficient is the leaf's block view.
# block() re-checks the coefficient field's generation on every slice.
@inline _leaf_op(L::AbstractOperator, i, lg) = L
@inline _leaf_op(S::ScalingOp{<:AbstractBlockField}, i, lg) = ScalingOp(block(S.coeff, i, lg))
@inline function _leaf_op(A::Advection{<:BlockForest,<:AbstractBlockField}, i, lg)
    return Advection(lg, block(A.velocity, i, lg))
end
# The INNER constructor, like `_slab_op`: the block view already carries the cross-block
# coefficient ghosts `fill_coefficient_ghosts!` filled at construction, which is exactly
# what the public entry point has nothing to fill and therefore refuses to guess.
@inline function _leaf_op(D::Diffusion{<:BlockForest,<:AbstractBlockField}, i, lg)
    return Diffusion(lg, block(D.κ, i, lg), D.avg)
end

# Diffusion's forest sweeps carry the conservative coarse–fine seam: the forward
# rewrites every coarse-side CF ghost to the κ-weighted authoritative flux value
# (`_cf_flux_rewrite!`) before the per-leaf reference sweep, and the adjoint runs
# the rewrite's exact transpose after the per-leaf gathers — still ahead of the
# fold_bc!/halo_update_adjoint! passes its scatter feeds. Hooking the sweep seam
# covers the unprepared path here and the PreparedForest hot path (linalg.jl),
# both β branches included; empty `cfflux` makes both no-ops on a uniform forest.
# A stage-4 packed kernel override for Diffusion must keep (or fuse) the rewrite.
function _forest_sweep!(
    y::AbstractBlockField, D::Diffusion{<:BlockForest,<:AbstractBlockField},
    x::AbstractBlockField, g::BlockForest, α, β,
)
    _require_current(D.κ)    # the rewrite reads raw κ storage ahead of block()'s guard
    _cf_flux_rewrite!(
        _storage(x), _layout(x), _storage(D.κ), _layout(D.κ), D.avg,
        _exchange_schedule(g).cfflux, g.blocksize,
    )
    return _forest_sweep_leaves!(y, D, x, g, α, β)
end

function _forest_adjoint_sweep!(
    x̄::AbstractBlockField, D::Diffusion{<:BlockForest,<:AbstractBlockField},
    ȳ::AbstractBlockField, g::BlockForest, α,
)
    _require_current(D.κ)    # mirrors the forward: fail before touching raw storage
    _forest_adjoint_sweep_leaves!(x̄, D, ȳ, g, α)
    _cf_flux_rewrite_adjoint!(
        _storage(x̄), _layout(x̄), _storage(D.κ), _layout(D.κ), D.avg,
        _exchange_schedule(g).cfflux, g.blocksize,
    )
    return x̄
end

# Combinators recurse at the forest level (mirroring their Field methods in
# algebra.jl): running a whole Added/Scaled tree per leaf would route any nested
# AdjointOp through the per-leaf adjoint, silently dropping its interface-ghost
# fold. Operands that all `shares_exchange` sweep back to back on one exchange;
# otherwise each runs its own full forest action, which is what a Composed
# intermediate, a nested AdjointOp, and Diffusion's coarse–fine rewrite need.
function apply!(y::AbstractBlockField, L::Added, x::AbstractBlockField, g::BlockForest, α, β)
    shares_exchange(L) && return _forest_exchange_sweep!(y, L, x, g, α, β)
    apply!(y, L.a, x, g, α, β)
    apply!(y, L.b, x, g, α, true)
    return y
end
function apply!(y::AbstractBlockField, L::Scaled, x::AbstractBlockField, g::BlockForest, α, β)
    return apply!(y, L.op, x, g, α * L.α, β)
end

# Sweep recursion behind a shared exchange — reachable only through the
# `shares_exchange` gates, since an un-gated operand can rewrite x's ghosts.
function _forest_sweep!(
    y::AbstractBlockField, L::Added, x::AbstractBlockField, g::BlockForest, α, β
)
    _forest_sweep!(y, L.a, x, g, α, β)
    _forest_sweep!(y, L.b, x, g, α, true)
    return y
end
function _forest_sweep!(
    y::AbstractBlockField, L::Scaled, x::AbstractBlockField, g::BlockForest, α, β
)
    return _forest_sweep!(y, L.op, x, g, α * L.α, β)
end
# The intermediate is a whole BlockField (allocated per call, like the single-grid
# pure path): the outer operand's forest apply! performs its own inter-block
# exchange on it, which is what a per-leaf Composed application would miss.
function apply!(y::AbstractBlockField, L::Composed, x::AbstractBlockField, g::BlockForest, α, β)
    tmp = allocate_output(L.b, x)
    apply!(tmp, L.b, x, g)
    apply!(y, L.a, tmp, g, α, β)
    return y
end

# Lazy adjoint wrappers route through the forest adjoint action so interface-ghost
# contributions are folded across blocks by halo_update_adjoint! (the per-leaf
# AdjointOp path would silently drop them).
function apply!(y::AbstractBlockField, L::AdjointOp, x::AbstractBlockField, g::BlockForest, α, β)
    return apply_adjoint!(y, L.op, x, g, α, β)
end
function apply_adjoint!(
    x̄::AbstractBlockField, L::AdjointOp, ȳ::AbstractBlockField, g::BlockForest, α, β
)
    return apply!(x̄, L.op, ȳ, g, α, β)
end

# A self-adjoint sum is its own adjoint, so it takes the (exchange-sharing) forward
# path. The per-operand transposes stay separate: each needs its own
# fold_bc!/halo_update_adjoint!, since a diagonal operand's transpose writes
# interiors only and leaves ghost scratch a shared fold would spread to neighbors.
function apply_adjoint!(
    x̄::AbstractBlockField, L::Added, ȳ::AbstractBlockField, g::BlockForest, α, β
)
    isselfadjoint(L) && return apply!(x̄, L, ȳ, g, α, β)
    apply_adjoint!(x̄, L.a, ȳ, g, α, β)
    apply_adjoint!(x̄, L.b, ȳ, g, α, true)
    return x̄
end
function apply_adjoint!(
    x̄::AbstractBlockField, L::Scaled, ȳ::AbstractBlockField, g::BlockForest, α, β
)
    return apply_adjoint!(x̄, L.op, ȳ, g, α * conj(L.α), β)
end
function apply_adjoint!(
    x̄::AbstractBlockField, L::Composed, ȳ::AbstractBlockField, g::BlockForest, α, β
)
    tmp = allocate_input(L.a, ȳ)
    apply_adjoint!(tmp, L.a, ȳ, g)
    apply_adjoint!(x̄, L.b, tmp, g, α, β)
    return x̄
end

"""
    apply_adjoint!(x̄::AbstractBlockField, L, ȳ::AbstractBlockField, g::BlockForest, α=true, β=false) -> x̄

Adjoint action over the forest. For a self-adjoint operator (e.g. the Laplacian on
a uniform forest, whose same-level halo copy couples both neighbors symmetrically)
this is the forward action. Otherwise it is the exact transpose of
stencil ∘ BC fill ∘ halo exchange: per-leaf stencil-transpose gathers (leaving all
ghost cotangents in place), the forest-level [`fold_bc!`](@ref) folding physical
ghosts, then [`halo_update_adjoint!`](@ref) folding interface ghosts into the
neighbor interiors.
"""
function apply_adjoint!(
    x̄::AbstractBlockField, L::AbstractOperator, ȳ::AbstractBlockField, g::BlockForest, α, β
)
    _require_current(x̄)
    _require_current(ȳ)
    # Sound on a non-uniform forest too: isselfadjoint is grid-aware (false once
    # coarse–fine coupling breaks the halo symmetry), so this only fires when the
    # forward action IS the adjoint.
    isselfadjoint(L) && return apply!(x̄, L, ȳ, g, α, β)
    # Diagonal transposes are pointwise (no cross-block coupling) and their
    # per-leaf adjoint writes interiors only — the gather + fold machinery below
    # would fold x̄'s ghost scratch into interiors, so it must be skipped.
    isdiagonal(L) && return apply!(x̄, adjoint_operator(L), ȳ, g, α, β)
    if iszero(β)
        _forest_adjoint_sweep!(x̄, L, ȳ, g, α)
        fold_bc!(x̄, g)
        halo_update_adjoint!(x̄, g)
    else
        s = similar(x̄)
        _forest_adjoint_sweep!(s, L, ȳ, g, true)
        fold_bc!(s, g)
        halo_update_adjoint!(s, g)
        for i in 1:nleaves(g)
            lg = leaf_grid(g, i)
            xi = interior(block(x̄, i, lg))
            xi .= α .* interior(block(s, i, lg)) .+ β .* xi
        end
    end
    return x̄
end

# The adjoint stencil sweep behind the fold path — the dispatch seam packed
# adjoint kernels override per (operator, layout), mirroring _forest_sweep!.
# Overwrite-only contract (β = 0 gather into the target; callers blend): the
# per-leaf reference runs the ordinary apply_adjoint!, whose leaf-level
# fold_bc! is a no-op on the all-Interface leaf grids.
function _forest_adjoint_sweep!(
    x̄::AbstractBlockField, L::AbstractOperator, ȳ::AbstractBlockField, g::BlockForest, α
)
    return _forest_adjoint_sweep_leaves!(x̄, L, ȳ, g, α)
end

function _forest_adjoint_sweep_leaves!(
    x̄::AbstractBlockField, L::AbstractOperator, ȳ::AbstractBlockField, g::BlockForest, α
)
    for i in 1:nleaves(g)
        lg = leaf_grid(g, i)
        apply_adjoint!(block(x̄, i, lg), _leaf_op(L, i, lg), block(ȳ, i, lg), lg, α, false)
    end
    return x̄
end
