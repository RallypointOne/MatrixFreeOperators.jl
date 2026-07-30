#--------------------------------------------------------------------------------# Grid partitioning (distributed seam)

"""
    partition_grid(g::CartesianGrid{N}, nparts::Integer) -> Vector{CartesianGrid{N}}

Split `g` into `nparts` slab partitions along dimension `N` — the memory-contiguous
dimension, so each partition owns a contiguous range of global flat interior DOFs.
Cell counts split evenly with the remainder going to the first partitions.

Each local grid keeps the global spacing *and the global extent* verbatim — its
position is carried entirely by `local_range`, which records the global plane
range it owns, so [`cell_center`](@ref) agrees bitwise with the uncut grid. It
gets [`Interface`](@ref) faces on partition cuts (both cut-dimension faces on
every partition when the global cut-dimension BC is [`Periodic`](@ref)).
`Interface` ghost slabs are filled by a distributed exchange (e.g. the MDLA
extension), never by `apply_bc!`.

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
# bit parity with the single-device apply.
#
# The GLOBAL extent is kept verbatim too, and `local_range` alone says which part
# of it this slab owns: extent describes the domain, local_range the ownership.
# That is what makes `cell_center` (Grids.jl) bitwise equal on a slab and on the
# grid it was cut from — deriving a slab-local origin `lo + (first(zr)-1)h` and
# then adding `(i-0.5)h` rounds twice and drifts by an ulp, which would make a
# coordinate-assembled RHS depend on the partition count.
function _slab_grid(
    g::CartesianGrid{N,T}, zr::UnitRange{Int}, p::Int, nparts::Int, periodic::Bool
) where {N,T}
    extent = g.extent
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

#--------------------------------------------------------------------------------# Ghost staging (padded-field side of the exchange)

# The two halves of a ghost exchange, expressed on a partition's padded scratch
# field. A backend supplies the transport (MDLA scatter!/reduce!, or the plain
# global indexing the CPU tests use); these translate between its flat
# [owned | ghost] section and the Interface halo slabs, per the `plans` emitted
# by _slab_ghost_layout. Kept in core so the CPU proof and the MDLA extension
# stage ghosts through exactly the same code.

# View of one cut-dimension halo plane at transverse-interior positions — the
# slab a ghost chunk fills (forward) or is packed from (adjoint). Corner cells
# are deliberately excluded: apply_bc! fills them dimension-1-first and fold_bc!
# folds them back dimension-N-first, so they are never exchanged.
_halo_plane_view(f::Field, plane::Int) = _halo_plane_view(f.data, f.grid, plane)
function _halo_plane_view(data, g::AbstractGrid{N}, plane::Int) where {N}
    h = halo_width(g)
    n = local_size(g)
    idx = ntuple(d -> d == N ? (plane:plane) : ((h[d] + 1):(h[d] + n[d])), Val(N))
    return view(data, idx...)
end

"""
    _unpack_ghosts!(xpad, local_x, nowned, plans) -> xpad

Copy the ghost section of a partition's flat `[owned | ghost]` vector into the
[`Interface`](@ref) halo slabs of its padded scratch field (internal). The
inverse of [`_pack_local_x!`](@ref).
"""
function _unpack_ghosts!(xpad::Field, local_x, nowned::Int, plans)
    for (rng, plane) in plans
        dst = _halo_plane_view(xpad, plane)
        dst .= _as_eltype(eltype(xpad.data), view(local_x, nowned .+ rng), size(dst))
    end
    return xpad
end

"""
    _pack_local_x!(local_x, x̄pad, nowned, plans) -> local_x

Pack a partition's adjoint result into its flat `[owned | ghost]` vector
(internal): interior cotangents first, then the [`Interface`](@ref) halo slabs
holding the neighbor-owned contributions `fold_bc!` migrated there. The exact
transpose of [`_unpack_ghosts!`](@ref) — a ghost-reducing backend consumes this.
"""
function _pack_local_x!(local_x, x̄pad::Field, nowned::Int, plans)
    interior_to_flat!(view(local_x, 1:nowned), x̄pad)
    for (rng, plane) in plans
        src = _halo_plane_view(x̄pad, plane)
        _as_eltype(eltype(x̄pad.data), view(local_x, nowned .+ rng), size(src)) .= src
    end
    return local_x
end
