# MDLA composition: partition_grid slabs become one CUDA device each, the
# Interface ghost slabs are exchanged with MDLA scatter!/reduce!, and the flat
# Krylov boundary becomes a MultiDeviceVector (DESIGN.md §7/§9). CUDA-only by
# MDLA's nature; MFO core stays backend-agnostic.
module MatrixFreeOperatorsMDLAExt

using MatrixFreeOperators, MultiDeviceLinearAlgebra, CUDA, Krylov
using LinearAlgebra: LinearAlgebra, mul!
import Adapt
import MatrixFreeOperators:
    AbstractGrid, AbstractOperator, Field, PreparedOperator,
    _check_distributable, _owned_flat_range, _pack_local_x!, _require_grid,
    _slab_ghost_layout, _unpack_ghosts!, zero_ghosts!, prepare_distributed
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

function prepare_distributed(L::AbstractOperator, nparts::Integer; devices=nothing)
    _check_distributable(L)
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
