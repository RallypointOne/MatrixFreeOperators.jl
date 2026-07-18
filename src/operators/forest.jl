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
# its own inter-block exchange.

"""
    apply!(y::BlockField, L::AbstractOperator, x::BlockField, g::BlockForest, α=true, β=false) -> y

Apply `L` over a block-structured forest: exchange inter-block halos once, then run
the per-leaf stencil on every block.
"""
function apply!(y::BlockField, L::AbstractOperator, x::BlockField, g::BlockForest, α, β)
    _require_current(y)
    halo_update!(x, g)                  # also checks x's generation
    apply_bc!(x, g)                     # physical faces — leaf grids are all-Interface
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
# The intermediate is a whole BlockField (allocated per call, like the single-grid
# pure path): the outer operand's forest apply! performs its own inter-block
# exchange on it, which is what a per-leaf Composed application would miss.
function apply!(y::BlockField, L::Composed, x::BlockField, g::BlockForest, α, β)
    tmp = allocate_output(L.b, x)
    apply!(tmp, L.b, x, g)
    apply!(y, L.a, tmp, g, α, β)
    return y
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
function apply_adjoint!(x̄::BlockField, L::Composed, ȳ::BlockField, g::BlockForest, α, β)
    tmp = allocate_input(L.a, ȳ)
    apply_adjoint!(tmp, L.a, ȳ, g)
    apply_adjoint!(x̄, L.b, tmp, g, α, β)
    return x̄
end

"""
    apply_adjoint!(x̄::BlockField, L, ȳ::BlockField, g::BlockForest, α=true, β=false) -> x̄

Adjoint action over the forest. For a self-adjoint operator (e.g. the Laplacian on
a uniform forest, whose same-level halo copy couples both neighbors symmetrically)
this is the forward action. Otherwise it is the exact transpose of
stencil ∘ BC fill ∘ halo exchange: per-leaf stencil-transpose gathers (leaving all
ghost cotangents in place), the forest-level [`fold_bc!`](@ref) folding physical
ghosts, then [`halo_update_adjoint!`](@ref) folding interface ghosts into the
neighbor interiors.
"""
function apply_adjoint!(x̄::BlockField, L::AbstractOperator, ȳ::BlockField, g::BlockForest, α, β)
    _require_current(x̄)
    _require_current(ȳ)
    # Sound on a non-uniform forest too: isselfadjoint is grid-aware (false once
    # coarse–fine coupling breaks the halo symmetry), so this only fires when the
    # forward action IS the adjoint.
    isselfadjoint(L) && return apply!(x̄, L, ȳ, g, α, β)
    if iszero(β)
        for i in 1:nleaves(g)
            lg = leaf_grid(g, i)
            apply_adjoint!(block(x̄, i, lg), L, block(ȳ, i, lg), lg, α, false)
        end
        fold_bc!(x̄, g)
        halo_update_adjoint!(x̄, g)
    else
        s = similar(x̄)
        for i in 1:nleaves(g)
            lg = leaf_grid(g, i)
            apply_adjoint!(block(s, i, lg), L, block(ȳ, i, lg), lg, true, false)
        end
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
