#--------------------------------------------------------------------------------# Forest-native kernels (packed sweeps)

# Single-launch stencil sweeps over PackedBlockField storage: one
# KernelAbstractions kernel over ndrange (blocksize..., nleaves) replaces the
# per-leaf reference loop in _forest_sweep! — on GPU backends only, where the
# per-leaf launch cost is the problem; non-GPU backends route to
# _forest_sweep_leaves!, whose fused broadcasts outperform KA CPU codegen.
# Kernel bodies reuse the per-cell stencil functions of the broadcast path —
# the numerical definition never forks — and recompute per-leaf geometry from
# the levels SoA. Adjoints stay declared, never AD-through-kernel: the
# grid-aware isselfadjoint shortcut reaches the kernel on uniform forests, and
# non-uniform adjoints run the per-leaf transpose-gather fallback.

# Bit-identical to _inv_spacing2(leaf_grid(bf, i)): _leaf_spacing divides the
# root spacing by 1 << level, _inv_spacing2 inverts its square.
@inline function _leaf_inv_h2(spacing0::NTuple{N}, ℓ::Integer) where {N}
    return ntuple(d -> inv((spacing0[d] / (1 << ℓ))^2), Val(N))
end

# First N entries of the global (cell..., leaf) index, offset into the halo frame.
@inline function _halo_cell(idx::NTuple, h::NTuple{N,Int}) where {N}
    return CartesianIndex(ntuple(d -> idx[d] + h[d], Val(N)))
end

@kernel function _lap_forest_kernel!(y, @Const(x), @Const(levels), spacing0, h, α, β)
    idx = @index(Global, NTuple)
    leaf = idx[end]
    I = _halo_cell(idx, h)
    ℓ = @inbounds levels[leaf]
    v = α * _lap_at(_leaf_slice(x, leaf), I, _leaf_inv_h2(spacing0, ℓ))
    # β = 0 must overwrite, never read y — fresh scratch may hold NaN poison.
    @inbounds y[I, leaf] = iszero(β) ? v : muladd(β, y[I, leaf], v)
end

function _forest_sweep!(
    y::PackedBlockField, L::Laplacian, x::PackedBlockField, g::BlockForest, α, β
)
    backend = KernelAbstractions.get_backend(g)
    # Broadcast fusion beats KA CPU codegen for this stencil (~1.6× measured); the
    # single launch pays only where per-leaf launches are the cost (GPU backends).
    backend isa KernelAbstractions.GPU || return _forest_sweep_leaves!(y, L, x, g, α, β)
    kernel! = _lap_forest_kernel!(backend)
    kernel!(
        y.data, x.data, x.levels, g.spacing0, g.halo, α, β;
        ndrange=(g.blocksize..., nleaves(g)),
    )
    return y
end
