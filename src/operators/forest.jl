#--------------------------------------------------------------------------------# Operators on a BlockForest

# The forest action reuses the single-grid operator path per leaf-block unchanged:
# halo_update! fills inter-block ghosts ONCE over the whole forest, then each leaf
# runs the ordinary apply! (whose own halo_update! is a no-op on the leaf
# CartesianGrid; apply_bc! fills only physical-boundary faces, Interface faces
# being left as halo_update! filled them). Supports leaf operators and same-input
# combinators (Scaled, Added); Composed needs inter-block exchange on its
# intermediate field and is handled in a later phase.

"""
    apply!(y::BlockField, L::AbstractOperator, x::BlockField, g::BlockForest, α=true, β=false) -> y

Apply `L` over a block-structured forest: exchange inter-block halos once, then run
the per-leaf stencil on every block.
"""
function apply!(y::BlockField, L::AbstractOperator, x::BlockField, g::BlockForest, α, β)
    halo_update!(x, g)
    for i in 1:nleaves(g)
        lg = leaf_grid(g, i)
        apply!(block(y, i, lg), L, block(x, i, lg), lg, α, β)
    end
    return y
end
function apply!(y::BlockField, L::AbstractOperator, x::BlockField, g::BlockForest)
    return apply!(y, L, x, g, true, false)
end
apply!(y::BlockField, L::AbstractOperator, x::BlockField) = apply!(y, L, x, x.grid)

function apply(L::AbstractOperator, x::BlockField)
    y = allocate_output(L, x)
    apply!(y, L, x, x.grid)
    return y
end
(L::AbstractOperator)(x::BlockField) = apply(L, x)
Base.:*(L::AbstractOperator, x::BlockField) = apply(L, x)

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
function apply_adjoint!(x̄::BlockField, L::AbstractOperator, ȳ::BlockField, g::BlockForest)
    return apply_adjoint!(x̄, L, ȳ, g, true, false)
end
