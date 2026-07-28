#--------------------------------------------------------------------------------# Grid partitioning (distributed seam)

"""
    partition_grid(g::CartesianGrid{N}, nparts::Integer) -> Vector{CartesianGrid{N}}

Split `g` into `nparts` slab partitions along dimension `N` — the memory-contiguous
dimension, so each partition owns a contiguous range of global flat interior DOFs.
Cell counts split evenly with the remainder going to the first partitions.

Each local grid keeps the global spacing verbatim, gets [`Interface`](@ref) faces
on partition cuts (both cut-dimension faces on every partition when the global
cut-dimension BC is [`Periodic`](@ref)), and records its global plane range in
`local_range`. `Interface` ghost slabs are filled by a distributed exchange
(e.g. the MDLA extension), never by `apply_bc!`.

Every slab must have at least `halo` planes along the cut dimension (twice that
for a periodic cut into exactly two partitions, where both of a partition's
ghost stacks come from the same neighbor), so each ghost slab has a single owner
and ghost requests are duplicate-free. `partition_grid(g, 1)` returns `[g]`
unchanged.

### Examples

```julia
g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (32, 32))
parts = partition_grid(g, 2)    # two 32×16 slabs cut along dimension 2
```

See also: [`halo_update!`](@ref), [`prepare`](@ref).
"""
function partition_grid(g::CartesianGrid{N,T}, nparts::Integer) where {N,T}
    g.topology === nothing ||
        throw(ArgumentError("partition_grid requires an undistributed grid"))
    nparts >= 1 || throw(ArgumentError("nparts must be ≥ 1, got $nparts"))
    nparts == 1 && return [g]
    n = local_size(g)[N]
    h = halo_width(g)[N]
    periodic = boundary_conditions(g)[N][1] isa Periodic
    # A periodic 2-partition cut is the only case where one partition's low and
    # high ghosts come from the SAME owner; 2h keeps those stacks disjoint. From
    # three partitions up they have different owners, so h suffices.
    minplanes = periodic && nparts == 2 ? 2 * h : h
    ranges = _slab_ranges(n, nparts)
    all(r -> length(r) >= minplanes, ranges) || throw(
        ArgumentError(
            "every slab needs ≥ $minplanes planes along dimension $N (halo $h" *
            "$(periodic ? ", periodic" : "")), got slab sizes $(length.(ranges)) " *
            "from $n planes across $nparts partitions",
        ),
    )
    return [_slab_grid(g, ranges[p], p, nparts, periodic) for p in 1:nparts]
end

# Even split of n planes with the remainder going to the first partitions (the
# same formula as MDLA's compute_partition_ranges, replicated to avoid the dep).
function _slab_ranges(n::Int, nparts::Int)
    base, rem = divrem(n, nparts)
    ranges = Vector{UnitRange{Int}}(undef, nparts)
    start = 1
    for p in 1:nparts
        len = base + (p <= rem ? 1 : 0)
        ranges[p] = start:(start + len - 1)
        start += len
    end
    return ranges
end

# One slab-local grid. Spacing is copied verbatim from the global grid — never
# recomputed from the slab extent, where division could drift by an ulp and break
# bit parity with the single-device apply. Outer extent endpoints are reused
# exactly; interior cut points are derived from the global origin and spacing.
#
# Bit parity covers spacing, and so the stencil weights — not coordinates:
# cell_center on a slab evaluates lo + (first(zr)-1)h + (i-0.5)h against the
# global lo + (z-0.5)h, which can differ in the last ulp. Nothing in slice 1
# evaluates coordinates on a slab (the RHS is assembled on the global grid), but
# distributed boundary_rhs/set! (#31) will need to account for it.
function _slab_grid(
    g::CartesianGrid{N,T}, zr::UnitRange{Int}, p::Int, nparts::Int, periodic::Bool
) where {N,T}
    lo, hi = g.extent[N]
    slab_lo = first(zr) == 1 ? lo : lo + T(first(zr) - 1) * g.spacing[N]
    slab_hi = last(zr) == local_size(g)[N] ? hi : lo + T(last(zr)) * g.spacing[N]
    extent = ntuple(d -> d == N ? (slab_lo, slab_hi) : g.extent[d], Val(N))
    sz = ntuple(d -> d == N ? length(zr) : local_size(g)[d], Val(N))
    lowbc = periodic || p > 1 ? Interface() : boundary_conditions(g)[N][1]
    highbc = periodic || p < nparts ? Interface() : boundary_conditions(g)[N][2]
    bc = ntuple(d -> d == N ? (lowbc, highbc) : boundary_conditions(g)[d], Val(N))
    local_range = ntuple(d -> d == N ? zr : g.local_range[d], Val(N))
    return CartesianGrid{N,T,typeof(bc),typeof(g.device),Nothing}(
        extent, g.spacing, sz, g.halo, bc, g.device, local_range, nothing
    )
end

#--------------------------------------------------------------------------------# Ghost layout (flat-index side of the exchange)

# Global flat scalar indices of interior plane z along dimension N: column-major
# over interior cells, component-fastest for SVector eltypes — the layout of
# `flatten`. Each plane is one contiguous flat range.
function _plane_flat_range(gsize::NTuple{N,Int}, z::Int, ncomp::Int) where {N}
    m = prod(Base.front(gsize)) * ncomp
    return ((z - 1) * m + 1):(z * m)
end

# Global flat interior indices owned by slab `lg` of global grid `g`: the
# contiguous span of its cut-dimension planes, in `flatten` layout.
function _owned_flat_range(g::AbstractGrid{N}, lg::AbstractGrid{N}; ncomp::Int=1) where {N}
    zr = lg.local_range[N]
    m = prod(Base.front(local_size(g))) * ncomp
    return ((first(zr) - 1) * m + 1):(last(zr) * m)
end

"""
    _slab_ghost_layout(g, parts; ncomp=1) -> (ghost_globals, plans)

Per-partition ghost requests and unpack plans for the slab partitioning `parts`
of the global grid `g` (internal; consumed by the MDLA extension).

- `ghost_globals[p]::Vector{Int}`: global flat interior indices partition `p`
  needs as ghosts, grouped by owning partition in ascending order (plane-ascending
  within an owner) so a ghost exchange that lays its ghost section out
  neighbor-ascending, request-order within neighbor — MDLA's rule — reproduces
  exactly this order.
- `plans[p]`: one `(range, plane)` pair per requested ghost plane, where `range`
  indexes into partition `p`'s ghost section and `plane` is the padded
  cut-dimension index of the halo slab it fills. Ranges tile the ghost section
  contiguously in emission order.
"""
function _slab_ghost_layout(
    g::CartesianGrid{N}, parts::AbstractVector; ncomp::Int=1
) where {N}
    nparts = length(parts)
    ghost_globals = [Int[] for _ in 1:nparts]
    plans = [Tuple{UnitRange{Int},Int}[] for _ in 1:nparts]
    nparts == 1 && return ghost_globals, plans
    n = local_size(g)[N]
    h = halo_width(g)[N]
    periodic = boundary_conditions(g)[N][1] isa Periodic
    owner_of = Vector{Int}(undef, n)
    for (p, lg) in enumerate(parts)
        owner_of[lg.local_range[N]] .= p
    end
    for p in 1:nparts
        zr = parts[p].local_range[N]
        cands = Tuple{Int,Int}[]
        for k in 1:h
            zlow = first(zr) - k
            if periodic
                push!(cands, (mod1(zlow, n), h + 1 - k))
            elseif zlow >= 1
                push!(cands, (zlow, h + 1 - k))
            end
            zhigh = last(zr) + k
            if periodic
                push!(cands, (mod1(zhigh, n), h + length(zr) + k))
            elseif zhigh <= n
                push!(cands, (zhigh, h + length(zr) + k))
            end
        end
        sort!(cands; by=c -> (owner_of[c[1]], c[1]))
        offset = 0
        for (z, plane) in cands
            r = _plane_flat_range(local_size(g), z, ncomp)
            append!(ghost_globals[p], r)
            push!(plans[p], ((offset + 1):(offset + length(r)), plane))
            offset += length(r)
        end
    end
    return ghost_globals, plans
end
