#--------------------------------------------------------------------------------# Grids

"""
    AbstractGrid{N}

Supertype for all `N`-dimensional grids. Operators are authored once against the
grid interface (`spacing`, `interior`, [`halo_update!`](@ref), …); single-device,
distributed, and adaptive grids differ only in what the grid object is and what
`halo_update!` does.
"""
abstract type AbstractGrid{N} end

"""
    CartesianGrid(extent, ncells; bc, halo, device)

Uniform cell-centered Cartesian grid.

# Arguments
- `extent::NTuple{N,Tuple{T,T}}`: physical `(min, max)` per dimension
- `ncells::NTuple{N,Int}`: interior cell counts per dimension

# Keyword Arguments
- `bc`: per-dimension `(low, high)` boundary-condition pairs (default: homogeneous
  [`Dirichlet`](@ref) on every face)
- `halo`: ghost-layer width per dimension (default: 1 per dimension)
- `device`: KernelAbstractions backend used for field allocation (default: `CPU()`)

Cell spacing is derived from `extent` and `ncells`. Cell centers sit at
`min + (i - 1/2)Δ` for interior cell `i`.

### Examples

```julia
g = CartesianGrid(((0.0, 1.0),), (64,))                       # 1-D, Dirichlet
g = CartesianGrid(((0.0, 1.0), (0.0, 2.0)), (32, 64);
                  bc = ((Periodic(), Periodic()),
                        (Neumann(), Neumann())))              # 2-D, mixed BCs
```

See also: [`spacing`](@ref), [`interior`](@ref), [`boundary_conditions`](@ref).
"""
struct CartesianGrid{N,T,BC<:Tuple,Dev,Topo} <: AbstractGrid{N}
    extent::NTuple{N,Tuple{T,T}}
    spacing::NTuple{N,T}
    size::NTuple{N,Int}
    halo::NTuple{N,Int}
    bc::BC
    device::Dev
    local_range::NTuple{N,UnitRange{Int}}
    topology::Topo
end

function CartesianGrid(
    extent::NTuple{N,Tuple{T,T}},
    ncells::NTuple{N,Int};
    bc::Tuple=ntuple(_ -> (Dirichlet(), Dirichlet()), Val(N)),
    halo::NTuple{N,Int}=ntuple(_ -> 1, Val(N)),
    device=KernelAbstractions.CPU(),
) where {N,T<:Real}
    all(>=(1), ncells) || throw(ArgumentError("ncells must be ≥ 1 per dimension, got $ncells"))
    all(>=(1), halo) || throw(ArgumentError("halo width must be ≥ 1 per dimension, got $halo"))
    for d in 1:N
        lo, hi = extent[d]
        lo < hi || throw(ArgumentError("extent must satisfy min < max in dimension $d, got ($lo, $hi)"))
    end
    _validate_bc(bc, Val(N))
    grid_spacing = ntuple(d -> (extent[d][2] - extent[d][1]) / ncells[d], Val(N))
    local_range = ntuple(d -> 1:ncells[d], Val(N))
    return CartesianGrid{N,T,typeof(bc),typeof(device),Nothing}(
        extent, grid_spacing, ncells, halo, bc, device, local_range, nothing
    )
end

function _validate_bc(bc::Tuple, ::Val{N}) where {N}
    length(bc) == N || throw(ArgumentError("bc must provide one (low, high) pair per dimension"))
    for d in 1:N
        pair = bc[d]
        pair isa Tuple{AbstractBC,AbstractBC} ||
            throw(ArgumentError("bc[$d] must be a (low, high) pair of AbstractBC, got $(typeof(pair))"))
        if (pair[1] isa Periodic) != (pair[2] isa Periodic)
            throw(ArgumentError("Periodic boundary conditions must be paired on both faces of dimension $d"))
        end
    end
    return nothing
end

#--------------------------------------------------------------------------------# Grid interface

"""
    dimension(g::AbstractGrid) -> Int

Spatial dimension of the grid.
"""
dimension(::AbstractGrid{N}) where {N} = N

"""
    spacing(g::AbstractGrid) -> NTuple{N}

Cell spacing per dimension.
"""
spacing(g::CartesianGrid) = g.spacing

"""
    local_size(g::AbstractGrid) -> NTuple{N,Int}

Interior (owned, non-halo) cell counts per dimension.
"""
local_size(g::CartesianGrid) = g.size

"""
    halo_width(g::AbstractGrid) -> NTuple{N,Int}

Ghost-layer width per dimension.
"""
halo_width(g::CartesianGrid) = g.halo

"""
    boundary_conditions(g::AbstractGrid) -> NTuple{N,Tuple}

Per-dimension `(low, high)` boundary-condition pairs.
"""
boundary_conditions(g::CartesianGrid) = g.bc

"""
    interior(g::AbstractGrid) -> CartesianIndices

Indices of the interior (owned, non-halo) cells in halo-padded index space.
"""
function interior(g::AbstractGrid{N}) where {N}
    return CartesianIndices(
        ntuple(d -> (halo_width(g)[d] + 1):(halo_width(g)[d] + local_size(g)[d]), Val(N))
    )
end

"""
    padded_size(g::AbstractGrid) -> NTuple{N,Int}

Array size per dimension including ghost layers on both faces.
"""
padded_size(g::AbstractGrid{N}) where {N} =
    ntuple(d -> local_size(g)[d] + 2 * halo_width(g)[d], Val(N))

"""
    cell_center(g::AbstractGrid, I::CartesianIndex) -> SVector

Physical coordinates of the center of cell `I` (in halo-padded index space).
"""
function cell_center(g::CartesianGrid{N,T}, I::CartesianIndex{N}) where {N,T}
    return SVector(
        ntuple(
            d -> g.extent[d][1] + (T(I[d] - g.halo[d]) - T(0.5)) * g.spacing[d], Val(N)
        )
    )
end

"""
    coarsen(g::CartesianGrid) -> CartesianGrid

The next-coarser grid in a 2:1 multigrid hierarchy: half the cells per
dimension over the same extent, so the spacing doubles. Boundary conditions,
halo width, and device carry over verbatim. Requires an even cell count in
every dimension.

### Examples

```julia
g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (64, 64))
gc = coarsen(g)          # 32×32, spacing doubled
```
"""
function coarsen(g::CartesianGrid{N}) where {N}
    all(iseven, local_size(g)) || throw(
        ArgumentError(
            "coarsen requires an even cell count in every dimension, got $(local_size(g))"
        ),
    )
    g.topology === nothing ||
        throw(ArgumentError("coarsen does not support distributed grids yet"))
    return CartesianGrid(
        g.extent, ntuple(d -> g.size[d] >> 1, Val(N)); bc=g.bc, halo=g.halo, device=g.device
    )
end

KernelAbstractions.get_backend(g::CartesianGrid) = g.device

_inv_spacing(g::AbstractGrid{N}) where {N} = ntuple(d -> inv(spacing(g)[d]), Val(N))
_inv_spacing2(g::AbstractGrid{N}) where {N} = ntuple(d -> inv(spacing(g)[d]^2), Val(N))

"""
    halo_update!(x, g::AbstractGrid) -> x

Fill ghost layers with neighbor data. The single distributed seam: a no-op on
single-device grids; distributed grids overload it to exchange halos. Operators
call this before any stencil that reads neighbor cells.
"""
halo_update!(x, ::AbstractGrid) = x

function Adapt.adapt_structure(to, g::CartesianGrid{N}) where {N}
    device = KernelAbstractions.get_backend(Adapt.adapt(to, similar(Vector{Bool}, 0)))
    return CartesianGrid{N,eltype(g.spacing),typeof(g.bc),typeof(device),typeof(g.topology)}(
        g.extent, g.spacing, g.size, g.halo, g.bc, device, g.local_range, g.topology
    )
end
