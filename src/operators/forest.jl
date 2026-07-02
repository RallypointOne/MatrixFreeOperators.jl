#--------------------------------------------------------------------------------# Operators on a BlockForest

# The forest action reuses the single-grid operator path per leaf-block unchanged:
# halo_update! fills inter-block ghosts ONCE over the whole forest, then each leaf
# runs the ordinary apply! (whose own halo_update! is a no-op on the leaf
# CartesianGrid; apply_bc! fills only physical-boundary faces, Interface faces
# being left as halo_update! filled them). Combinators (Added, Scaled, AdjointOp)
# recurse at the FOREST level — never per leaf — so a nested adjoint always
# reaches the forest adjoint action and its cross-block fold; Composed throws
# until its intermediate field gets an inter-block exchange (coarse–fine phase).

# Composed needs an inter-block halo exchange on its intermediate field. Erroring
# — wherever it sits in the tree — upholds the design invariant: degrade to an
# error, never a wrong result.
_check_forest_supported(::AbstractOperator) = nothing
_check_forest_supported(L::Scaled) = _check_forest_supported(L.op)
_check_forest_supported(L::AdjointOp) = _check_forest_supported(L.op)
function _check_forest_supported(L::Added)
    _check_forest_supported(L.a)
    return _check_forest_supported(L.b)
end
function _check_forest_supported(::Composed)
    throw(
        ArgumentError(
            "Composed operators are not supported on a BlockForest yet: the " *
            "intermediate field needs an inter-block halo exchange (coarse–fine " *
            "phase); apply the factors separately per application instead",
        ),
    )
end

"""
    apply!(y::BlockField, L::AbstractOperator, x::BlockField, g::BlockForest, α=true, β=false) -> y

Apply `L` over a block-structured forest: exchange inter-block halos once, then run
the per-leaf stencil on every block.
"""
function apply!(y::BlockField, L::AbstractOperator, x::BlockField, g::BlockForest, α, β)
    _check_forest_supported(L)
    _require_current(y)
    halo_update!(x, g)                  # also checks uniformity and x's generation
    for i in 1:nleaves(g)
        lg = leaf_grid(g, i)
        apply!(block(y, i, lg), L, block(x, i, lg), lg, α, β)
    end
    return y
end

# Combinators recurse at the forest level (mirroring their Field methods in
# algebra.jl): running a whole Added/Scaled tree per leaf would route any nested
# AdjointOp through the per-leaf adjoint, silently dropping its interface-ghost
# fold. Each operand re-exchanges halos — idempotent, since exchanges read only
# interiors, which the operand applications never mutate.
function apply!(y::BlockField, L::Added, x::BlockField, g::BlockForest, α, β)
    apply!(y, L.a, x, g, α, β)
    apply!(y, L.b, x, g, α, true)
    return y
end
function apply!(y::BlockField, L::Scaled, x::BlockField, g::BlockForest, α, β)
    return apply!(y, L.op, x, g, α * L.α, β)
end

# Lazy adjoint wrappers route through the forest adjoint action so interface-ghost
# contributions are folded across blocks by halo_update_adjoint! (the per-leaf
# AdjointOp path would silently drop them).
function apply!(y::BlockField, L::AdjointOp, x::BlockField, g::BlockForest, α, β)
    return apply_adjoint!(y, L.op, x, g, α, β)
end
function apply_adjoint!(x̄::BlockField, L::AdjointOp, ȳ::BlockField, g::BlockForest, α, β)
    return apply!(x̄, L.op, ȳ, g, α, β)
end

function apply_adjoint!(x̄::BlockField, L::Added, ȳ::BlockField, g::BlockForest, α, β)
    apply_adjoint!(x̄, L.a, ȳ, g, α, β)
    apply_adjoint!(x̄, L.b, ȳ, g, α, true)
    return x̄
end
function apply_adjoint!(x̄::BlockField, L::Scaled, ȳ::BlockField, g::BlockForest, α, β)
    return apply_adjoint!(x̄, L.op, ȳ, g, α * conj(L.α), β)
end

"""
    apply_adjoint!(x̄::BlockField, L, ȳ::BlockField, g::BlockForest, α=true, β=false) -> x̄

Adjoint action over the forest. For a self-adjoint operator (e.g. the Laplacian on
a uniform forest, whose same-level halo copy couples both neighbors symmetrically)
this is the forward action. Otherwise it is the exact transpose: the per-leaf
adjoint (stencil transpose + `fold_bc!`, leaving interface-ghost contributions in
place) followed by [`halo_update_adjoint!`](@ref) folding those into the neighbor
interiors.
"""
function apply_adjoint!(x̄::BlockField, L::AbstractOperator, ȳ::BlockField, g::BlockForest, α, β)
    _check_forest_supported(L)
    _require_uniform(g)
    _require_current(x̄)
    _require_current(ȳ)
    isselfadjoint(L) && return apply!(x̄, L, ȳ, g, α, β)
    if iszero(β)
        for i in 1:nleaves(g)
            lg = leaf_grid(g, i)
            apply_adjoint!(block(x̄, i, lg), L, block(ȳ, i, lg), lg, α, false)
        end
        halo_update_adjoint!(x̄, g)
    else
        s = similar(x̄)
        for i in 1:nleaves(g)
            lg = leaf_grid(g, i)
            apply_adjoint!(block(s, i, lg), L, block(ȳ, i, lg), lg, true, false)
        end
        halo_update_adjoint!(s, g)
        for i in 1:nleaves(g)
            lg = leaf_grid(g, i)
            xi = interior(block(x̄, i, lg))
            xi .= α .* interior(block(s, i, lg)) .+ β .* xi
        end
    end
    return x̄
end
