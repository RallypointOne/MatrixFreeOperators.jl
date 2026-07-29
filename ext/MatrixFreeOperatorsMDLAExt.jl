# MDLA composition: partition_grid slabs become one CUDA device each, the
# Interface ghost slabs are exchanged with MDLA scatter!/reduce!, and the flat
# Krylov boundary becomes a MultiDeviceVector (DESIGN.md §7/§9). CUDA-only by
# MDLA's nature; MFO core stays backend-agnostic.
#
# The operator tree is walked by core (src/distributed.jl); this file supplies
# the three backend primitives that walk calls — a per-device map and the
# scatter/reduce pair — so a mid-tree exchange fires wherever a Composed
# intermediate or an adjoint node needs one.
module MatrixFreeOperatorsMDLAExt

using MatrixFreeOperators, MultiDeviceLinearAlgebra, CUDA, Krylov
using LinearAlgebra: LinearAlgebra, mul!
import Adapt
import MatrixFreeOperators:
    AbstractField, AbstractGrid, AbstractOperator, DistNode, Field, PreparedOperator,
    _check_distributable, _check_one_grid, _dist_adjoint_segment!, _dist_boundary_rhs!,
    _dist_capply!, _dist_lift_scratch, _dist_map!, _dist_reduce!, _dist_scatter!,
    _dist_set!, _dist_tree, _owned_flat_range, _pack_local_x!, _push_adjoints,
    _require_grid, _slab_ghost_layout, _slab_op, _unpack_ghosts!, zero_ghosts!,
    boundary_rhs, prepare_distributed
import MultiDeviceLinearAlgebra: _empty_mdv, copy_exchange

#--------------------------------------------------------------------------------# Backend primitives

# Per-device execution context. MDLA is task-parallel: CUDA.jl gives each Julia
# task its own stream, and CUDA.device! sets the task-local device.
struct MDLAContext{S<:PartitionSpec}
    spec::S
end

function _dist_map!(f, ctx::MDLAContext)
    @sync for d in 1:ctx.spec.ndevices
        @async begin
            CUDA.device!(device_id(ctx.spec, d))
            f(d)
        end
    end
    return nothing
end

"""
    MDLAExchange

One ghost exchange: an MDLA `GhostExchange` whose per-device
`local_x = [owned | ghost]` sections stage the transfer, the `PartitionSpec` it
was built against, the unpack plans from `_slab_ghost_layout`, and a flat
staging `MultiDeviceVector`.

One of these per node that needs an exchange — the root, plus every `Composed`
intermediate and adjoint node. They never share buffers: `GhostExchange` is
stateful, and a mid-tree exchange fires while the root's `local_x` still holds
the values that produced the current intermediate.
"""
struct MDLAExchange{T,S<:PartitionSpec,GE<:GhostExchange{T},V}
    ghost::GE
    spec::S
    plans::Vector{Vector{Tuple{UnitRange{Int},Int}}}
    stage::V
end

# scatter!/reduce! are themselves host-orchestrated (@sync over devices), so they
# run *between* the per-device phases, never inside one.

function _dist_scatter!(X::MDLAExchange, fields, ctx::MDLAContext)
    _dist_map!(ctx) do d
        interior_to_flat!(X.stage.partitions[d], fields[d])
    end
    scatter!(X.stage, X.ghost, X.spec)
    _dist_map!(ctx) do d
        _unpack_ghosts!(fields[d], X.ghost.local_x[d], length(X.spec.ranges[d]), X.plans[d])
    end
    return fields
end

function _dist_reduce!(X::MDLAExchange, fields, ctx::MDLAContext)
    _dist_map!(ctx) do d
        _pack_local_x!(X.ghost.local_x[d], fields[d], length(X.spec.ranges[d]), X.plans[d])
    end
    reduce!(X.stage, X.ghost, X.spec, +)
    _dist_map!(ctx) do d
        flat_to_interior!(fields[d], X.stage.partitions[d])
        zero_ghosts!(fields[d])
    end
    return fields
end

# Root forms. The solver's vector already holds exactly the owned interiors, so
# the root scatters/reduces it directly instead of staging a copy — keeping the
# hot path as cheap as the pre-walk implementation.
function _root_scatter!(X::MDLAExchange, fields, ctx::MDLAContext, x::MultiDeviceVector)
    scatter!(x, X.ghost, X.spec)
    _dist_map!(ctx) do d
        flat_to_interior!(fields[d], x.partitions[d])
        _unpack_ghosts!(fields[d], X.ghost.local_x[d], length(X.spec.ranges[d]), X.plans[d])
    end
    return fields
end

function _root_reduce!(X::MDLAExchange, fields, ctx::MDLAContext, x̄::MultiDeviceVector)
    _dist_map!(ctx) do d
        _pack_local_x!(X.ghost.local_x[d], fields[d], length(X.spec.ranges[d]), X.plans[d])
    end
    reduce!(x̄, X.ghost, X.spec, +)
    return x̄
end

#--------------------------------------------------------------------------------# Distributed prepared operator

"""
    MDLAPreparedOperator

Distributed counterpart of [`PreparedOperator`](@ref): one prepared operator per
slab partition, each on its own CUDA device, plus the distributed walk tree and
the operator-owned [`MDLAExchange`](@ref)s whose `Interface` halo slabs are
filled before every local apply. Built with [`prepare_distributed`](@ref);
`mul!`/`size`/`eltype` operate on MDLA `MultiDeviceVector`s partitioned by `spec`
(ghost-free — the exchanges live on the operator, so Krylov workspace vectors
carry no communication buffers).

Stateful and single-threaded, like [`PreparedOperator`](@ref) and for the same
reason: `mul!` stages through each exchange's `local_x` and each partition's
scratch fields, so concurrent solves need one `prepare_distributed` each.
"""
struct MDLAPreparedOperator{T,S<:PartitionSpec,C<:MDLAContext,X<:MDLAExchange,N<:DistNode,V}
    parts::Vector{PreparedOperator}   # heterogeneous local-grid BC type params
    spec::S
    ctx::C
    root::X
    tree::N
    xpads::V
    ypads::V
end

Base.size(P::MDLAPreparedOperator) = (P.spec.len, P.spec.len)
Base.size(P::MDLAPreparedOperator, d::Integer) = size(P)[d]
Base.eltype(::MDLAPreparedOperator{T}) where {T} = T

# A node's exchange. Scalar intermediates share the root's topology exactly, so
# copy_exchange clones it with fresh buffers and skips re-probing device P2P.
# Rank-changing intermediates would need their own ncomp-scaled spec and layout;
# the distributability guards reject them, so this cannot be reached in slice 2a.
function _node_exchange(root::MDLAExchange{T}, proto::AbstractField) where {T}
    ncomponents(proto) == 1 || throw(
        ArgumentError(
            "distributed intermediates must be scalar fields; got $(ncomponents(proto)) " *
            "components. Rank-changing intermediates need their own partition spec " *
            "and ghost layout (not yet implemented).",
        ),
    )
    return MDLAExchange(
        copy_exchange(root.ghost, root.spec),
        root.spec,
        root.plans,
        MultiDeviceVector{T}(undef, root.spec),
    )
end

function prepare_distributed(L0::AbstractOperator, nparts::Integer; devices=nothing)
    # Normalize adjoints down to the leaves first: prepare does not recurse into
    # an AdjointOp, so AdjointOp(A*B) would otherwise skip its mid-tree reduction.
    L = _push_adjoints(L0)
    # Both guards run ONCE, here, on the global tree — before `_slab_op` rewrites
    # it. That ordering is load-bearing, not incidental: a localized ScalingOp
    # reports the slab grid from `operator_grid` while a sibling Laplacian still
    # reports the global one, so re-checking a localized tree would reject it.
    _check_distributable(L)
    g = _require_grid(L)
    g isa CartesianGrid ||
        throw(ArgumentError("prepare_distributed requires a CartesianGrid, got $(nameof(typeof(g)))"))
    _check_one_grid(L, g)
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
    root = MDLAExchange(
        GhostExchange(ghost_globals, spec, T), spec, plans,
        MultiDeviceVector{T}(undef, spec),
    )
    ctx = MDLAContext(spec)
    # Field parameters go to the HOST before being sliced. A coefficient the user
    # already moved to device 0 would make the slice below a cross-device copy —
    # the one operation MDLA's P2P probe exists to guard, and the one that returns
    # silent zeros on IOMMU-affected hosts. Slice on the host, upload the slab.
    Lh = Adapt.adapt(Array, L)
    parts = Vector{PreparedOperator}(undef, nparts)
    @sync for d in 1:nparts
        @async begin
            CUDA.device!(device_id(spec, d))
            lg = Adapt.adapt(CuArray, locals[d])
            # The adapted coefficient's grid is an adapted twin of locals[d], not
            # `===` lg. Nothing past prepare compares grid identity, and `interior`
            # only needs the shape, which matches by construction.
            parts[d] = prepare(Adapt.adapt(CuArray, _slab_op(Lh, locals[d])), scalar_field(lg, T))
        end
    end
    xpads = AbstractField[p.xpad for p in parts]
    ypads = AbstractField[p.ypad for p in parts]
    tree = _dist_tree([p.op for p in parts], xpads, proto -> _node_exchange(root, proto))
    return MDLAPreparedOperator{
        T,typeof(spec),typeof(ctx),typeof(root),typeof(tree),typeof(xpads)
    }(
        parts, spec, ctx, root, tree, xpads, ypads
    )
end

#--------------------------------------------------------------------------------# Forward action

function LinearAlgebra.mul!(
    y::MultiDeviceVector{T}, P::MDLAPreparedOperator{T}, x::MultiDeviceVector{T},
    α::Number, β::Number,
) where {T}
    _root_scatter!(P.root, P.xpads, P.ctx, x)
    # α/β are applied at the flat boundary, as PreparedOperator.mul! does, so they
    # never cross a collective step inside the walk.
    _dist_capply!(P.ypads, P.tree, P.xpads, P.ctx, true, false)
    _dist_map!(P.ctx) do d
        interior_to_flat!(y.partitions[d], P.ypads[d], α, β)
    end
    return y
end
function LinearAlgebra.mul!(
    y::MultiDeviceVector{T}, P::MDLAPreparedOperator{T}, x::MultiDeviceVector{T}
) where {T}
    return mul!(y, P, x, true, false)
end

#--------------------------------------------------------------------------------# Adjoint action

# Distributed adjoint: the walk's transpose (every scatter becomes a reduction and
# vice versa), closed by the root reduce!(+) that accumulates each partition's
# Interface cotangents into their owners. Internal — an adjoint operator is
# distributed by handing prepare_distributed the adjoint itself, which
# _push_adjoints normalizes into the tree; this entry point exists for tests that
# want the transpose of a given prepared operator.
function _mul_adjoint!(
    x̄::MultiDeviceVector{T}, P::MDLAPreparedOperator{T}, ȳ::MultiDeviceVector{T}
) where {T}
    _dist_map!(P.ctx) do d
        flat_to_interior!(P.ypads[d], ȳ.partitions[d])
    end
    _dist_adjoint_segment!(P.xpads, P.tree, P.ypads, P.ctx)
    return _root_reduce!(P.root, P.xpads, P.ctx, x̄)
end

#--------------------------------------------------------------------------------# Right-hand side assembly

"""
    boundary_rhs(P::MDLAPreparedOperator) -> MultiDeviceVector

Distributed boundary lift of `P`'s operator — the `b` of the affine split
`L_full(x) = L(x) + b`, assembled slab-locally.

Every partition assembles the lift of its own slab: `Interface` faces contribute
nothing, so only the slabs owning a physical face lift through the cut dimension,
while transverse physical faces contribute on all of them. A `Composed` node's
intermediate lift crosses cut planes through the same mid-tree exchange the
forward walk uses. Nothing global is ever materialized.

Returns a fresh `MultiDeviceVector` on `P.spec`, so `rhs = f .- boundary_rhs(P)`
stays partition-local. Shares `P`'s walk buffers, exactly as `mul!` does — do not
call it concurrently with a solve on the same prepared operator.

### Examples

```julia
P = prepare_distributed(laplacian(g), 2)
b = set!(MultiDeviceVector{Float64}(undef, P.spec), P, x -> sin(x[1]))
b .-= boundary_rhs(P)
u, stats = Krylov.cg(P, b)
```

See also: [`distributed_rhs`](@ref), [`prepare_distributed`](@ref).
"""
function boundary_rhs(P::MDLAPreparedOperator{T}) where {T}
    # zs is transient on purpose: the lift is a once-per-solve assembly, and a
    # permanent padded field per device is memory a homogeneous problem shouldn't pay.
    zs = _dist_lift_scratch(P.xpads, P.ctx)
    _dist_boundary_rhs!(P.ypads, P.tree, zs, P.ctx, true, false)
    b = MultiDeviceVector{T}(undef, P.spec)
    _dist_map!(P.ctx) do d
        interior_to_flat!(b.partitions[d], P.ypads[d])
    end
    return b
end

"""
    local_grids(P::MDLAPreparedOperator) -> Vector

The slab grid each partition owns, in partition order, **on the host**.

The escape hatch for building distributed data this module has no helper for:
allocate a [`Field`](@ref) on one of these, fill it however you like, and hand the
vector of fields to [`distributed_rhs`](@ref). Each grid records its span of the
global grid in `local_range`, and [`cell_center`](@ref) on it agrees bitwise with
the uncut grid.

Host grids on purpose. A field allocated on a device grid would land on whichever
device happened to be current, and `distributed_rhs` would then read it from a
*different* device — the cross-device copy MDLA's P2P probe exists to guard, and
the one that returns silent zeros on IOMMU-affected hosts. Build on the host;
`distributed_rhs` uploads each slab inside its own partition's device context.
"""
MatrixFreeOperators.local_grids(P::MDLAPreparedOperator) =
    [Adapt.adapt(Array, p.grid) for p in P.parts]

"""
    set!(x::MultiDeviceVector, P::MDLAPreparedOperator, fun) -> x

Fill `x` with `fun(coords)` evaluated slab-locally on each partition.

The distributed twin of `set!(::Field, fun)`, and exact: `cell_center` evaluates
at the global cell index, so this is bit-for-bit `MultiDeviceVector(flatten(set!(scalar_field(g), fun)), P.spec)`
without ever building the global field. Uses `P`'s input scratch, so the same
concurrency caveat as `mul!` applies.
"""
function MatrixFreeOperators.set!(
    x::MultiDeviceVector{T}, P::MDLAPreparedOperator{T}, fun
) where {T}
    _dist_set!(P.xpads, fun, P.ctx)
    _dist_map!(P.ctx) do d
        interior_to_flat!(x.partitions[d], P.xpads[d])
    end
    return x
end

"""
    distributed_rhs(P::MDLAPreparedOperator, f) -> MultiDeviceVector

The solve-ready right-hand side `f - boundary_rhs(P)`, assembled entirely
slab-locally.

`f` is either a function of physical coordinates or one [`Field`](@ref) per
partition on the grids [`local_grids`](@ref) reports. Equivalent to — and bitwise
equal to — the single-device recipe
`MultiDeviceVector(flatten(f) .- flatten(boundary_rhs(L, g)), P.spec)`, without
materializing the global right-hand side on one device.

### Examples

```julia
P = prepare_distributed(laplacian(g), 2)
b = distributed_rhs(P, x -> sin(x[1]) * exp(-x[2]))
u, stats = Krylov.cg(P, b)
```
"""
function MatrixFreeOperators.distributed_rhs(P::MDLAPreparedOperator{T}, f) where {T}
    x = MultiDeviceVector{T}(undef, P.spec)
    _source!(x, P, f)
    b = boundary_rhs(P)
    _dist_map!(P.ctx) do d
        x.partitions[d] .-= b.partitions[d]
    end
    return x
end

_source!(x, P::MDLAPreparedOperator, fun) = MatrixFreeOperators.set!(x, P, fun)
function _source!(x, P::MDLAPreparedOperator, fields::AbstractVector)
    length(fields) == length(P.parts) || throw(
        ArgumentError(
            "distributed_rhs got $(length(fields)) source fields for " *
            "$(length(P.parts)) partitions; pass one per partition, on the grids " *
            "local_grids(P) reports",
        ),
    )
    for (d, f) in enumerate(fields)
        flat_length(f) == length(P.spec.ranges[d]) || throw(
            ArgumentError(
                "distributed_rhs source field $d has $(flat_length(f)) interior DOFs " *
                "but partition $d owns $(length(P.spec.ranges[d])); build it on " *
                "local_grids(P)[$d]",
            ),
        )
    end
    # Adapt inside the partition's own device context, so a host field is uploaded
    # to the device that will read it rather than copied across devices.
    _dist_map!(P.ctx) do d
        interior_to_flat!(x.partitions[d], Adapt.adapt(CuArray, fields[d]))
    end
    return x
end

#--------------------------------------------------------------------------------# Krylov workspace

function Krylov.CgWorkspace(P::MDLAPreparedOperator{T}, b::MultiDeviceVector{T}) where {T}
    b_empty = _empty_mdv(b)
    kc = Krylov.KrylovConstructor(b; vm_empty=b_empty)
    return Krylov.CgWorkspace(kc)
end

end # module
