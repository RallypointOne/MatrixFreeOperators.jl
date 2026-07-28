# MDLA composition: partition_grid slabs become one CUDA device each, the
# Interface ghost slabs are exchanged with MDLA scatter!/reduce!, and the flat
# Krylov boundary becomes a MultiDeviceVector (DESIGN.md §7/§9). CUDA-only by
# MDLA's nature; MFO core stays backend-agnostic.
module MatrixFreeOperatorsMDLAExt

using MatrixFreeOperators, MultiDeviceLinearAlgebra, CUDA, Krylov
using LinearAlgebra: LinearAlgebra, mul!
import Adapt
import MatrixFreeOperators:
    AbstractGrid, AbstractOperator, Field, IdentityOp, Laplacian, PreparedOperator,
    ScalingOp, Scaled, Added, _owned_flat_range, _require_grid, _slab_ghost_layout,
    zero_ghosts!, prepare_distributed
import MultiDeviceLinearAlgebra: _empty_mdv

#--------------------------------------------------------------------------------# Distributed prepared operator

"""
    MDLAPreparedOperator

Distributed counterpart of [`PreparedOperator`](@ref): one prepared operator per
slab partition, each on its own CUDA device, plus the operator-owned MDLA
`GhostExchange` whose `local_x = [owned | ghost]` sections are unpacked into the
partitions' `Interface` halo slabs before every local apply. Built with
[`prepare_distributed`](@ref); `mul!`/`size`/`eltype` operate on MDLA
`MultiDeviceVector`s partitioned by `spec` (ghost-free — the exchange lives on
the operator, so Krylov workspace vectors carry no communication buffers).

Stateful and single-threaded, like [`PreparedOperator`](@ref) and for the same
reason: `mul!` stages through the operator-owned `ghost.local_x` and each
partition's `xpad`, so concurrent solves need one `prepare_distributed` each.
"""
struct MDLAPreparedOperator{T,S<:PartitionSpec,GE<:GhostExchange{T}}
    parts::Vector{PreparedOperator}   # heterogeneous local-grid BC type params
    spec::S
    ghost::GE
    plans::Vector{Vector{Tuple{UnitRange{Int},Int}}}
end

Base.size(P::MDLAPreparedOperator) = (P.spec.len, P.spec.len)
Base.size(P::MDLAPreparedOperator, d::Integer) = size(P)[d]
Base.eltype(::MDLAPreparedOperator{T}) where {T} = T

# The slice-1 whitelist: constant-coefficient pointwise leaves and stencil
# leaves whose ghost needs are one halo exchange. Composed would need a
# mid-tree exchange for its intermediate (its Interface ghosts would be stale
# zeros — silently wrong), AdjointOp a mid-tree reduction, and Field
# coefficients their own partitioning; all deferred, all rejected loudly.
_distributable(::Laplacian) = true
_distributable(::IdentityOp) = true
_distributable(S::ScalingOp) = S.coeff isa Number
_distributable(L::Scaled) = _distributable(L.op)
_distributable(L::Added) = _distributable(L.a) && _distributable(L.b)
_distributable(::AbstractOperator) = false

function prepare_distributed(L::AbstractOperator, nparts::Integer; devices=nothing)
    _distributable(L) || throw(
        ArgumentError(
            "prepare_distributed supports Laplacian, IdentityOp, number-coefficient " *
            "ScalingOp, and their Scaled/Added combinations; got $(sprint(show, L)). " *
            "Composed, adjoints, and Field coefficients are not yet distributable.",
        ),
    )
    g = _require_grid(L)
    g isa CartesianGrid ||
        throw(ArgumentError("prepare_distributed requires a CartesianGrid, got $(nameof(typeof(g)))"))
    ndev = devices === nothing ? length(CUDA.devices()) : length(devices)
    if nparts > ndev
        throw(
            ArgumentError(
                "$nparts partitions need $nparts distinct CUDA devices, " *
                "$ndev $(devices === nothing ? "available" : "given in `devices`")",
            ),
        )
    end
    T = eltype(spacing(g))
    locals = partition_grid(g, nparts)
    ghost_globals, plans = _slab_ghost_layout(g, locals)
    ranges = [_owned_flat_range(g, lg) for lg in locals]
    spec = devices === nothing ? PartitionSpec(ranges) : PartitionSpec(ranges; devices)
    ghost = GhostExchange(ghost_globals, spec, T)
    parts = Vector{PreparedOperator}(undef, nparts)
    @sync for d in 1:nparts
        @async begin
            CUDA.device!(device_id(spec, d))
            lg = Adapt.adapt(CuArray, locals[d])
            parts[d] = prepare(L, scalar_field(lg, T))
        end
    end
    return MDLAPreparedOperator{T,typeof(spec),typeof(ghost)}(parts, spec, ghost, plans)
end

#--------------------------------------------------------------------------------# Forward action

# View of one cut-dimension halo plane at transverse-interior positions.
function _halo_plane_view(f::Field, plane::Int)
    return _halo_plane_view(f.data, f.grid, plane)
end
function _halo_plane_view(data, g::AbstractGrid{N}, plane::Int) where {N}
    h = halo_width(g)
    n = local_size(g)
    idx = ntuple(d -> d == N ? (plane:plane) : ((h[d] + 1):(h[d] + n[d])), Val(N))
    return view(data, idx...)
end

# Copy the ghost section of local_x (laid out per the _slab_ghost_layout plans)
# into the Interface halo slabs of the partition's padded scratch field.
function _unpack_ghosts!(xpad::Field, local_x, nowned::Int, plans)
    for (rng, plane) in plans
        dst = _halo_plane_view(xpad, plane)
        dst .= reshape(view(local_x, nowned .+ rng), size(dst))
    end
    return xpad
end

function LinearAlgebra.mul!(
    y::MultiDeviceVector{T}, P::MDLAPreparedOperator{T}, x::MultiDeviceVector{T},
    α::Number, β::Number,
) where {T}
    scatter!(x, P.ghost, P.spec)
    @sync for d in 1:P.spec.ndevices
        @async begin
            CUDA.device!(device_id(P.spec, d))
            nowned = length(P.spec.ranges[d])
            _unpack_ghosts!(P.parts[d].xpad, P.ghost.local_x[d], nowned, P.plans[d])
            mul!(y.partitions[d], P.parts[d], x.partitions[d], α, β)
        end
    end
    return y
end
function LinearAlgebra.mul!(
    y::MultiDeviceVector{T}, P::MDLAPreparedOperator{T}, x::MultiDeviceVector{T}
) where {T}
    return mul!(y, P, x, true, false)
end

#--------------------------------------------------------------------------------# Adjoint action

# Pack a partition's adjoint result into its local_x = [owned | ghost] section:
# interior cotangents first, then the Interface halo slabs holding the
# neighbor-owned contributions fold_bc! migrated there.
function _pack_local_x!(local_x, x̄pad::Field, nowned::Int, plans)
    interior_to_flat!(view(local_x, 1:nowned), x̄pad)
    for (rng, plane) in plans
        src = _halo_plane_view(x̄pad, plane)
        reshape(view(local_x, nowned .+ rng), size(src)) .= src
    end
    return local_x
end

# Distributed adjoint: per-partition mechanical transpose (apply_adjoint! on an
# Interface-faced grid leaves neighbor-owned cotangents in the ghost slabs),
# then reduce!(+) accumulates them into their owners — the exact transpose of
# the scatter!-then-apply forward action. Internal: slice 1 exposes it to tests
# only; prepare_distributed rejects AdjointOp so the public boundary stays
# honest about what is supported.
function _mul_adjoint!(
    x̄::MultiDeviceVector{T}, P::MDLAPreparedOperator{T}, ȳ::MultiDeviceVector{T}
) where {T}
    @sync for d in 1:P.spec.ndevices
        @async begin
            CUDA.device!(device_id(P.spec, d))
            part = P.parts[d]
            flat_to_interior!(part.ypad, ȳ.partitions[d])
            zero_ghosts!(part.xpad)   # pointwise adjoints write interior-only; stale slabs must not reach reduce!
            apply_adjoint!(part.xpad, part.op, part.ypad, part.grid)
            _pack_local_x!(P.ghost.local_x[d], part.xpad, length(P.spec.ranges[d]), P.plans[d])
        end
    end
    reduce!(x̄, P.ghost, P.spec, +)
    return x̄
end

#--------------------------------------------------------------------------------# Krylov workspace

function Krylov.CgWorkspace(P::MDLAPreparedOperator{T}, b::MultiDeviceVector{T}) where {T}
    b_empty = _empty_mdv(b)
    kc = Krylov.KrylovConstructor(b; vm_empty=b_empty)
    return Krylov.CgWorkspace(kc)
end

end # module
