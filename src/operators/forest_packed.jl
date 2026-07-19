#--------------------------------------------------------------------------------# Forest-native kernels (packed sweeps)

# Single-launch stencil sweeps over PackedBlockField storage: one
# KernelAbstractions kernel over ndrange (blocksize..., nleaves) replaces the
# per-leaf reference loop in _forest_sweep! — on GPU backends only, where the
# per-leaf launch cost is the problem; non-GPU backends route to
# _forest_sweep_leaves!, whose fused broadcasts outperform KA CPU codegen.
# Kernel bodies reuse the per-cell stencil functions of the broadcast path —
# the numerical definition never forks — and recompute per-leaf geometry from
# the levels SoA. Adjoints stay declared, never AD-through-kernel: the
# grid-aware isselfadjoint shortcut reaches the forward kernel on uniform
# forests, and non-uniform adjoints run declared transpose-gather kernels
# (reusing the _*_adjoint_gather stencils) behind the _forest_adjoint_sweep!
# seam — over the full padded extent, so ghost cotangents are written for the
# callers' fold_bc!/halo_update_adjoint! transposes to fold.

# Bit-identical to _inv_spacing2(leaf_grid(bf, i)): _leaf_spacing divides the
# root spacing by 1 << level, _inv_spacing2 inverts its square.
@inline function _leaf_inv_h2(spacing0::NTuple{N}, ℓ::Integer) where {N}
    return ntuple(d -> inv((spacing0[d] / (1 << ℓ))^2), Val(N))
end

# Bit-identical to _inv_spacing(leaf_grid(bf, i)).
@inline function _leaf_inv_h(spacing0::NTuple{N}, ℓ::Integer) where {N}
    return ntuple(d -> inv(spacing0[d] / (1 << ℓ)), Val(N))
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

@kernel function _deriv_forest_kernel!(
    y, @Const(x), @Const(levels), spacing0, h, dim, order, α, β
)
    idx = @index(Global, NTuple)
    leaf = idx[end]
    I = _halo_cell(idx, h)
    ℓ = @inbounds levels[leaf]
    v = α * _deriv_at(_leaf_slice(x, leaf), I, dim, order, _leaf_inv_h(spacing0, ℓ)[dim])
    @inbounds y[I, leaf] = iszero(β) ? v : muladd(β, y[I, leaf], v)
end

function _forest_sweep!(
    y::PackedBlockField, L::Derivative, x::PackedBlockField, g::BlockForest, α, β
)
    backend = KernelAbstractions.get_backend(g)
    backend isa KernelAbstractions.GPU || return _forest_sweep_leaves!(y, L, x, g, α, β)
    kernel! = _deriv_forest_kernel!(backend)
    kernel!(
        y.data, x.data, x.levels, g.spacing0, g.halo, L.dim, L.order, α, β;
        ndrange=(g.blocksize..., nleaves(g)),
    )
    return y
end

@kernel function _grad_forest_kernel!(y, @Const(x), @Const(levels), spacing0, h, α, β)
    idx = @index(Global, NTuple)
    leaf = idx[end]
    I = _halo_cell(idx, h)
    ℓ = @inbounds levels[leaf]
    v = α * _grad_at(_leaf_slice(x, leaf), I, _leaf_inv_h(spacing0, ℓ))
    # SVector blend written out (no muladd): identical arithmetic, robust on device.
    @inbounds y[I, leaf] = iszero(β) ? v : β * y[I, leaf] + v
end

function _forest_sweep!(
    y::PackedBlockField, L::Gradient, x::PackedBlockField, g::BlockForest, α, β
)
    backend = KernelAbstractions.get_backend(g)
    backend isa KernelAbstractions.GPU || return _forest_sweep_leaves!(y, L, x, g, α, β)
    kernel! = _grad_forest_kernel!(backend)
    kernel!(
        y.data, x.data, x.levels, g.spacing0, g.halo, α, β;
        ndrange=(g.blocksize..., nleaves(g)),
    )
    return y
end

@kernel function _div_forest_kernel!(y, @Const(x), @Const(levels), spacing0, h, α, β)
    idx = @index(Global, NTuple)
    leaf = idx[end]
    I = _halo_cell(idx, h)
    ℓ = @inbounds levels[leaf]
    v = α * _div_at(_leaf_slice(x, leaf), I, _leaf_inv_h(spacing0, ℓ))
    @inbounds y[I, leaf] = iszero(β) ? v : muladd(β, y[I, leaf], v)
end

function _forest_sweep!(
    y::PackedBlockField, L::Divergence, x::PackedBlockField, g::BlockForest, α, β
)
    backend = KernelAbstractions.get_backend(g)
    backend isa KernelAbstractions.GPU || return _forest_sweep_leaves!(y, L, x, g, α, β)
    kernel! = _div_forest_kernel!(backend)
    kernel!(
        y.data, x.data, x.levels, g.spacing0, g.halo, α, β;
        ndrange=(g.blocksize..., nleaves(g)),
    )
    return y
end

@kernel function _adv_forest_kernel!(
    y, @Const(x), @Const(vel), @Const(levels), spacing0, h, α, β
)
    idx = @index(Global, NTuple)
    leaf = idx[end]
    I = _halo_cell(idx, h)
    ℓ = @inbounds levels[leaf]
    a = α * _adv_at(_leaf_slice(x, leaf), _leaf_slice(vel, leaf), I, _leaf_inv_h(spacing0, ℓ))
    @inbounds y[I, leaf] = iszero(β) ? a : muladd(β, y[I, leaf], a)
end

function _forest_sweep!(
    y::PackedBlockField,
    L::Advection{<:BlockForest,<:PackedBlockField},
    x::PackedBlockField,
    g::BlockForest,
    α,
    β,
)
    backend = KernelAbstractions.get_backend(g)
    backend isa KernelAbstractions.GPU || return _forest_sweep_leaves!(y, L, x, g, α, β)
    _require_current(L.velocity)
    kernel! = _adv_forest_kernel!(backend)
    kernel!(
        y.data, x.data, L.velocity.data, x.levels, g.spacing0, g.halo, α, β;
        ndrange=(g.blocksize..., nleaves(g)),
    )
    return y
end

function _forest_sweep!(
    y::PackedBlockField,
    L::Advection{<:BlockForest,SelfAdvection},
    x::PackedBlockField,
    g::BlockForest,
    α,
    β,
)
    backend = KernelAbstractions.get_backend(g)
    backend isa KernelAbstractions.GPU || return _forest_sweep_leaves!(y, L, x, g, α, β)
    eltype(x) <: SVector ||
        throw(ArgumentError("self-advection u·∇u requires an SVector-valued field"))
    kernel! = _adv_forest_kernel!(backend)
    # x.data aliased as state and velocity — both read-only in the kernel.
    kernel!(
        y.data, x.data, x.data, x.levels, g.spacing0, g.halo, α, β;
        ndrange=(g.blocksize..., nleaves(g)),
    )
    return y
end

# Pointwise sweeps still go through a kernel, not a whole-array broadcast: apply!
# writes the interior of y only (ghosts of y stay deterministic, and a packed
# coefficient's ghosts are unspecified), and β = 0 must never read y's scratch.
@kernel function _scale_forest_kernel!(y, @Const(x), κ, h, α, β)
    idx = @index(Global, NTuple)
    leaf = idx[end]
    I = _halo_cell(idx, h)
    xv = @inbounds x[I, leaf]
    v = α * κ * xv
    @inbounds y[I, leaf] = iszero(β) ? v : muladd(β, y[I, leaf], v)
end

@kernel function _scalefield_forest_kernel!(y, @Const(x), @Const(κ), h, α, β)
    idx = @index(Global, NTuple)
    leaf = idx[end]
    I = _halo_cell(idx, h)
    v = α * @inbounds(κ[I, leaf]) * @inbounds(x[I, leaf])
    @inbounds y[I, leaf] = iszero(β) ? v : muladd(β, y[I, leaf], v)
end

function _forest_sweep!(
    y::PackedBlockField, S::ScalingOp{<:Number}, x::PackedBlockField, g::BlockForest, α, β
)
    backend = KernelAbstractions.get_backend(g)
    backend isa KernelAbstractions.GPU || return _forest_sweep_leaves!(y, S, x, g, α, β)
    kernel! = _scale_forest_kernel!(backend)
    kernel!(y.data, x.data, S.coeff, g.halo, α, β; ndrange=(g.blocksize..., nleaves(g)))
    return y
end

function _forest_sweep!(
    y::PackedBlockField,
    S::ScalingOp{<:PackedBlockField},
    x::PackedBlockField,
    g::BlockForest,
    α,
    β,
)
    backend = KernelAbstractions.get_backend(g)
    backend isa KernelAbstractions.GPU || return _forest_sweep_leaves!(y, S, x, g, α, β)
    # @inbounds in the kernel would turn a foreign coefficient into UB, not an error.
    S.coeff.grid === g ||
        throw(ArgumentError("scaling coefficient must be a field on the same forest"))
    _require_current(S.coeff)
    kernel! = _scalefield_forest_kernel!(backend)
    kernel!(
        y.data, x.data, S.coeff.data, g.halo, α, β; ndrange=(g.blocksize..., nleaves(g))
    )
    return y
end

function _forest_sweep!(
    y::PackedBlockField, L::IdentityOp, x::PackedBlockField, g::BlockForest, α, β
)
    backend = KernelAbstractions.get_backend(g)
    backend isa KernelAbstractions.GPU || return _forest_sweep_leaves!(y, L, x, g, α, β)
    kernel! = _scale_forest_kernel!(backend)
    # κ = true: α * true * x ≡ α * x bit-exactly.
    kernel!(y.data, x.data, true, g.halo, α, β; ndrange=(g.blocksize..., nleaves(g)))
    return y
end

#--------------------------------------------------------------------------------# Adjoint transpose-gather kernels (packed sweeps)

# Single-launch transposes of the stencil sweeps, mirroring the per-leaf
# adjoint_gather! contract: ȳ's ghosts are zeroed (interior-only cotangents),
# every padded cell of x̄ is overwritten (ghost cotangents feed the callers'
# fold_bc!/halo_update_adjoint!), and the leaf-level fold_bc! is omitted — a
# no-op on the all-Interface leaf grids. Unlike the forward kernels the ndrange
# spans the full padded extent and the index is used directly (no halo offset);
# the seam contract is overwrite-only (no β — callers blend). α folds into the
# gather in-kernel; the reference post-multiplies, bit-equal by commutativity.

@kernel function _lap_adjoint_forest_kernel!(x̄, @Const(ȳ), @Const(levels), spacing0, α)
    idx = @index(Global, NTuple)
    leaf = idx[end]
    J = CartesianIndex(Base.front(idx))
    ℓ = @inbounds levels[leaf]
    v = α * _lap_adjoint_gather(_leaf_slice(ȳ, leaf), J, _leaf_inv_h2(spacing0, ℓ))
    @inbounds x̄[J, leaf] = v
end

@kernel function _deriv_adjoint_forest_kernel!(
    x̄, @Const(ȳ), @Const(levels), spacing0, dim, order, α
)
    idx = @index(Global, NTuple)
    leaf = idx[end]
    J = CartesianIndex(Base.front(idx))
    ℓ = @inbounds levels[leaf]
    v =
        α * _deriv_adjoint_gather(
            _leaf_slice(ȳ, leaf), J, dim, order, _leaf_inv_h(spacing0, ℓ)[dim]
        )
    @inbounds x̄[J, leaf] = v
end

@kernel function _grad_adjoint_forest_kernel!(x̄, @Const(ȳ), @Const(levels), spacing0, α)
    idx = @index(Global, NTuple)
    leaf = idx[end]
    J = CartesianIndex(Base.front(idx))
    ℓ = @inbounds levels[leaf]
    v = α * _grad_adjoint_gather(_leaf_slice(ȳ, leaf), J, _leaf_inv_h(spacing0, ℓ))
    @inbounds x̄[J, leaf] = v
end

@kernel function _div_adjoint_forest_kernel!(x̄, @Const(ȳ), @Const(levels), spacing0, α)
    idx = @index(Global, NTuple)
    leaf = idx[end]
    J = CartesianIndex(Base.front(idx))
    ℓ = @inbounds levels[leaf]
    v = α * _div_adjoint_gather(_leaf_slice(ȳ, leaf), J, _leaf_inv_h(spacing0, ℓ))
    @inbounds x̄[J, leaf] = v
end

function _forest_adjoint_sweep!(
    x̄::PackedBlockField, L::Laplacian, ȳ::PackedBlockField, g::BlockForest, α
)
    backend = KernelAbstractions.get_backend(g)
    backend isa KernelAbstractions.GPU ||
        return _forest_adjoint_sweep_leaves!(x̄, L, ȳ, g, α)
    zero_ghosts!(ȳ)
    kernel! = _lap_adjoint_forest_kernel!(backend)
    kernel!(
        x̄.data, ȳ.data, ȳ.levels, g.spacing0, α;
        ndrange=(g.blocksize .+ 2 .* g.halo..., nleaves(g)),
    )
    return x̄
end

function _forest_adjoint_sweep!(
    x̄::PackedBlockField, L::Derivative, ȳ::PackedBlockField, g::BlockForest, α
)
    backend = KernelAbstractions.get_backend(g)
    backend isa KernelAbstractions.GPU ||
        return _forest_adjoint_sweep_leaves!(x̄, L, ȳ, g, α)
    zero_ghosts!(ȳ)
    kernel! = _deriv_adjoint_forest_kernel!(backend)
    kernel!(
        x̄.data, ȳ.data, ȳ.levels, g.spacing0, L.dim, L.order, α;
        ndrange=(g.blocksize .+ 2 .* g.halo..., nleaves(g)),
    )
    return x̄
end

function _forest_adjoint_sweep!(
    x̄::PackedBlockField, L::Gradient, ȳ::PackedBlockField, g::BlockForest, α
)
    backend = KernelAbstractions.get_backend(g)
    backend isa KernelAbstractions.GPU ||
        return _forest_adjoint_sweep_leaves!(x̄, L, ȳ, g, α)
    zero_ghosts!(ȳ)
    kernel! = _grad_adjoint_forest_kernel!(backend)
    kernel!(
        x̄.data, ȳ.data, ȳ.levels, g.spacing0, α;
        ndrange=(g.blocksize .+ 2 .* g.halo..., nleaves(g)),
    )
    return x̄
end

function _forest_adjoint_sweep!(
    x̄::PackedBlockField, L::Divergence, ȳ::PackedBlockField, g::BlockForest, α
)
    backend = KernelAbstractions.get_backend(g)
    backend isa KernelAbstractions.GPU ||
        return _forest_adjoint_sweep_leaves!(x̄, L, ȳ, g, α)
    zero_ghosts!(ȳ)
    kernel! = _div_adjoint_forest_kernel!(backend)
    kernel!(
        x̄.data, ȳ.data, ȳ.levels, g.spacing0, α;
        ndrange=(g.blocksize .+ 2 .* g.halo..., nleaves(g)),
    )
    return x̄
end
