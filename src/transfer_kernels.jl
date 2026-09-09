#--------------------------------------------------------------------------------# Device exchange/BC kernels (packed layout, GPU backends)

# Batched execution of the halo exchange and physical-BC face pass for
# PackedBlockField on GPU backends: the per-descriptor broadcast loops of
# transfer.jl become a constant number of kernel launches over the flattened
# _DeviceSchedule (schedule.jl) — N copy launches (one per normal dim,
# ascending), one CSR fill launch each for interp and restrict, and ≤ 2·N·h BC
# face launches. Same-stream ordering sequences the phases exactly like the
# host loops; every kernel reproduces the host arithmetic term-for-term (seed
# with term 1, accumulate in CSR order, plain *+, no fma), so results are
# bit-identical to the reference path except copy-phase CORNER ghost cells,
# where the last dim wins instead of the interleaved (leaf, dim, side) host
# order — cells axis-aligned stencils never read and adjoints require to be
# zero (transfer.jl halo_update_adjoint! docstring).
#
# The ADJOINT exchange stays on the host descriptor loops on all backends: its
# scatter-adds collide on shared source cells (overlapping tangential-stencil
# reads), so a kernelization needs atomics (bit-nondeterministic) or a second
# transposed source-centric CSR — the sanctioned future path if it is ever
# needed. It only runs in apply_adjoint! on non-self-adjoint (refined)
# forests, off the mul! hot path. fold_bc! (race-free per leaf) and the
# setup-only fill_bc_inhomogeneous! stay host for the same
# cost-benefit reason.

# Upload a host vector of isbits records to the backend.
function _to_device(backend, host::Vector{S}) where {S}
    dev = KernelAbstractions.allocate(backend, S, length(host))
    isempty(host) || copyto!(dev, host)
    return dev
end

# The normal dim of a face-copy box: the unique dim whose dst slab is not the
# full padded extent (the h[d]-layer ghost band).
function _copy_normal_dim(c::CopyDescriptor{N}, psize::NTuple{N,Int}) where {N}
    for d in 1:N
        length(c.dst_ranges[d]) != psize[d] && return d
    end
    return N   # unreachable for h ≥ 1 slabs; keeps the compiler total
end

_dev_fill(f::GhostFill{N}, tfirst::Int, tlast::Int) where {N} = _DevFill{N}(
    Int32(f.dst_block),
    Int32.(first.(f.dst_ranges)),
    Int32.(step.(f.dst_ranges)),
    Int32.(length.(f.dst_ranges)),
    Int32(tfirst), Int32(tlast),
)

_dev_term(t::SlabTerm{N,T}) where {N,T} = _DevTerm{N,T}(
    Int32(t.block), Int32.(first.(t.ranges)), Int32.(step.(t.ranges)), t.weight
)

# CSR-flatten one fill phase: fills in host order, all terms concatenated in
# host fill/term order (the accumulation order the kernel must reproduce). Every
# row is K wide, so the term buffer is sized up front and row i is
# (i-1)K+1 : iK — the explicit [tfirst, tlast] on each fill is kept so the kernel
# stays K-agnostic.
function _flatten_fills(fills::Vector{GhostFill{N,T,K}}) where {N,T,K}
    devfills = Vector{_DevFill{N}}(undef, length(fills))
    terms = Vector{_DevTerm{N,T}}(undef, K * length(fills))
    maxcells = 0
    for (i, f) in enumerate(fills)
        tfirst = (i - 1) * K + 1
        for (k, t) in enumerate(f.terms)
            terms[tfirst + k - 1] = _dev_term(t)
        end
        devfills[i] = _dev_fill(f, tfirst, i * K)
        maxcells = max(maxcells, prod(length.(f.dst_ranges)))
    end
    return devfills, terms, maxcells
end

function _flatten_schedule(
    sched::ExchangeSchedule{N,T}, g::BlockForest{N,T}, backend
) where {N,T}
    psize = g.blocksize .+ 2 .* g.halo
    # copies bucketed by normal dim, host order preserved within a dim
    devcopies = _DevCopy{N}[]
    copy_offsets = zeros(Int, N + 1)
    for d in 1:N
        for c in sched.copies
            _copy_normal_dim(c, psize) == d || continue
            push!(
                devcopies,
                _DevCopy{N}(
                    Int32(c.src), Int32(c.dst),
                    Int32.(first.(c.src_ranges)), Int32.(first.(c.dst_ranges)),
                ),
            )
        end
        copy_offsets[d + 1] = length(devcopies)
    end
    interp, interp_terms, interp_maxcells = _flatten_fills(sched.interp)
    restrict, restrict_terms, restrict_maxcells = _flatten_fills(sched.restrict)
    bcfaces = ntuple(
        d -> (
            _to_device(backend, Int32.(sched.bcfaces[d][1])),
            _to_device(backend, Int32.(sched.bcfaces[d][2])),
        ),
        Val(N),
    )
    dc = _to_device(backend, devcopies)
    di = _to_device(backend, interp)
    dit = _to_device(backend, interp_terms)
    dr = _to_device(backend, restrict)
    drt = _to_device(backend, restrict_terms)
    return _DeviceSchedule{N,T,typeof(dc),typeof(di),typeof(dit),typeof(bcfaces[1][1])}(
        dc, copy_offsets, di, dit, interp_maxcells, dr, drt, restrict_maxcells,
        bcfaces, sched.generation,
    )
end

# Cache accessor: rebuilt when the generation moves OR the cached copy lives on
# a different backend (the Ref is shared across Adapt twins). One dynamic
# dispatch on the abstract Ref eltype — GPU path only, amortized against the
# launches it feeds.
function _device_schedule(
    g::BlockForest{N,T}, sched::ExchangeSchedule{N,T}, backend
) where {N,T}
    ds = g.schedule_device[]
    if ds isa _DeviceSchedule{N,T} &&
        ds.generation == sched.generation &&
        KernelAbstractions.get_backend(ds.interp_terms) == backend
        return ds
    end
    dsn = _flatten_schedule(sched, g, backend)
    g.schedule_device[] = dsn
    return dsn
end

#--------------------------------------------------------------------------------# Kernels

@kernel function _copy_kernel!(data, @Const(copies))
    idx = @index(Global, NTuple)
    c = @inbounds copies[idx[end]]
    o = Base.front(idx)   # 1-based offsets within the (uniform) box shape
    @inbounds data[(c.dst_first .+ o .- 1)..., Int(c.dst)] =
        data[(c.src_first .+ o .- 1)..., Int(c.src)]
end

# Mixed-radix decode of a 0-based linear cell index into 0-based per-dim
# offsets, first dim fastest (column-major, matching broadcast element order).
@inline _decode_offsets(lin::Int, ::Tuple{}) = ()
@inline function _decode_offsets(lin::Int, len::Tuple)
    d, r = divrem(lin, Int(first(len)))
    return (r, _decode_offsets(d, Base.tail(len))...)
end

@kernel function _fill_kernel!(data, @Const(fills), @Const(terms))
    j, f = @index(Global, NTuple)
    fl = @inbounds fills[f]
    if j <= prod(Int.(fl.len))
        o = _decode_offsets(j - 1, fl.len)
        t = @inbounds terms[Int(fl.tfirst)]
        acc = t.weight * @inbounds(data[(Int.(t.first) .+ o .* Int.(t.step))..., Int(t.block)])
        for k in (Int(fl.tfirst) + 1):Int(fl.tlast)
            tk = @inbounds terms[k]
            acc += tk.weight *
                   @inbounds(data[(Int.(tk.first) .+ o .* Int.(tk.step))..., Int(tk.block)])
        end
        @inbounds data[(Int.(fl.first) .+ o .* Int.(fl.step))..., Int(fl.dst)] = acc
    end
end

# Insert the fixed dim-D coordinate into an (N-1)-tuple of transverse coords.
@inline function _insert_dim(::Val{D}, t::NTuple{M,Int}, v::Int) where {D,M}
    return ntuple(i -> i < D ? t[i] : (i == D ? v : t[i - 1]), Val(M + 1))
end

@kernel function _bc_face_kernel!(data, @Const(faces), sign, ghost::Int, source::Int, ::Val{D}) where {D}
    idx = @index(Global, NTuple)   # (transverse padded extents..., face index)
    leaf = @inbounds faces[idx[end]]
    t = Base.front(idx)
    @inbounds data[_insert_dim(Val(D), t, ghost)..., Int(leaf)] =
        sign * data[_insert_dim(Val(D), t, source)..., Int(leaf)]
end

#--------------------------------------------------------------------------------# Launch drivers + seam overrides

function _run_copies_device!(data, ds::_DeviceSchedule{N}, g::BlockForest{N}, backend) where {N}
    psize = g.blocksize .+ 2 .* g.halo
    kernel! = _copy_kernel!(backend)
    # Ascending dims, one launch each: within a dim dst ghost bands are disjoint
    # and never alias the interior reads; across dims corners overlap, so the
    # launches stay sequential (same-stream) — last dim wins, the block-local
    # outcome of the host order.
    for d in 1:N
        lo, hi = ds.copy_offsets[d] + 1, ds.copy_offsets[d + 1]
        hi < lo && continue
        shape = ntuple(t -> t == d ? g.halo[d] : psize[t], Val(N))
        kernel!(data, view(ds.copies, lo:hi); ndrange=(shape..., hi - lo + 1))
    end
    return nothing
end

function _run_fills_device!(data, fills, terms, maxcells::Int, backend)
    isempty(fills) && return nothing
    _fill_kernel!(backend)(data, fills, terms; ndrange=(maxcells, length(fills)))
    return nothing
end

function _run_exchange!(x::PackedBlockField, g::BlockForest, sched::ExchangeSchedule)
    backend = KernelAbstractions.get_backend(x.data)
    # Gate before the schedule_device Ref load: the CPU path must never pay the
    # abstract-Ref dispatch (protects the 0-alloc CPU mul! claim).
    backend isa KernelAbstractions.GPU || return _run_exchange_host!(x, sched)
    ds = _device_schedule(g, sched, backend)
    _run_copies_device!(x.data, ds, g, backend)
    _run_fills_device!(x.data, ds.interp, ds.interp_terms, ds.interp_maxcells, backend)
    _run_fills_device!(x.data, ds.restrict, ds.restrict_terms, ds.restrict_maxcells, backend)
    return nothing
end

function _run_bc!(x::PackedBlockField, g::BlockForest, sched::ExchangeSchedule)
    backend = KernelAbstractions.get_backend(x.data)
    backend isa KernelAbstractions.GPU || return _run_bc_host!(x, g, sched)
    ds = _device_schedule(g, sched, backend)
    _bc_faces_launch!(
        _bc_face_kernel!(backend), x.data, g.bc, ds.bcfaces, g.halo, g.blocksize,
        g.blocksize .+ 2 .* g.halo, Val(1),
    )
    return nothing
end

# Host-driven Val(D) recursion issuing one launch per (dim, side, layer) with a
# nonempty face list — the kernelized twin of _fill_bcfaces_dims!. The BC
# semantics collapse to two scalars per launch (sign and mirrored source
# layer), computed by the same boundaries.jl functions the host path uses;
# ascending dims + sequential same-stream launches reproduce the host's
# ghost-of-ghost corners bit-exactly. Periodic dims have empty face lists.
function _bc_faces_launch!(
    kernel!, data, bcs::Tuple, faces::Tuple, halo::Tuple, sz::Tuple, psize, ::Val{D}
) where {D}
    lo, hi = first(bcs)
    flo, fhi = first(faces)
    h, n = first(halo), first(sz)
    tshape = ntuple(i -> i < D ? psize[i] : psize[i + 1], Val(length(psize) - 1))
    for k in 1:h
        if !isempty(flo)
            kernel!(
                data, flo, _bc_sign(lo), h + 1 - k, _source_low(lo, h, n, k), Val(D);
                ndrange=(tshape..., length(flo)),
            )
        end
        if !isempty(fhi)
            kernel!(
                data, fhi, _bc_sign(hi), h + n + k, _source_high(hi, h, n, k), Val(D);
                ndrange=(tshape..., length(fhi)),
            )
        end
    end
    return _bc_faces_launch!(
        kernel!, data, Base.tail(bcs), Base.tail(faces), Base.tail(halo), Base.tail(sz),
        psize, Val(D + 1),
    )
end
_bc_faces_launch!(kernel!, data, ::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}, psize, ::Val) =
    nothing
