#--------------------------------------------------------------------------------# Boundary conditions

"""
    AbstractBC

Supertype for boundary conditions attached to grid faces.

`apply_bc!` enforces only the *homogeneous* part of a boundary condition, so every
linear operator satisfies `L(0) = 0`; inhomogeneous boundary data enters a solve
separately through [`boundary_rhs`](@ref).
"""
abstract type AbstractBC end

"""
    Periodic()

Periodic boundary condition. Must be paired on both faces of a dimension.
"""
struct Periodic <: AbstractBC end

"""
    Dirichlet(value=0)

Dirichlet boundary condition `u = value` at the face. The homogeneous part fills
ghost cells by antisymmetric mirroring (`ghost = -interior`); `value` contributes
only through [`boundary_rhs`](@ref).
"""
struct Dirichlet{T} <: AbstractBC
    value::T
end
Dirichlet() = Dirichlet(0)

"""
    Neumann(flux=0)

Neumann boundary condition `∂u/∂n = flux` at the face (outward normal). The
homogeneous part fills ghost cells by symmetric mirroring (`ghost = interior`);
`flux` contributes only through [`boundary_rhs`](@ref).
"""
struct Neumann{T} <: AbstractBC
    flux::T
end
Neumann() = Neumann(0)

# Sign of the homogeneous ghost fill relative to the mirrored interior cell.
_bc_sign(::Periodic) = 1
_bc_sign(::Dirichlet) = -1
_bc_sign(::Neumann) = 1

# Padded source index mirrored by ghost layer k (h = halo width, n = interior count).
_source_low(::Periodic, h::Int, n::Int, k::Int) = h + n + 1 - k
_source_low(::AbstractBC, h::Int, n::Int, k::Int) = h + k
_source_high(::Periodic, h::Int, n::Int, k::Int) = h + k
_source_high(::AbstractBC, h::Int, n::Int, k::Int) = h + n + 1 - k

#--------------------------------------------------------------------------------# Homogeneous ghost fill and its adjoint

"""
    apply_bc!(data, g::AbstractGrid) -> data

Fill all ghost layers of the halo-padded array `data` with the homogeneous part of
the grid's boundary conditions (periodic wrap, Dirichlet antisymmetric mirror,
Neumann symmetric mirror). Dimensions are filled in order `1:N`, so corner ghosts
are consistent ghost-of-ghost values.

See also: [`fold_bc!`](@ref).
"""
function apply_bc!(data::AbstractArray{<:Any,N}, g::AbstractGrid{N}) where {N}
    _apply_bc_dims!(data, g, 1, boundary_conditions(g))
    return data
end

function _apply_bc_dims!(data, g, d::Int, bcs::Tuple)
    lo, hi = first(bcs)
    h = halo_width(g)[d]
    n = local_size(g)[d]
    for k in 1:h
        _fill_ghost!(data, d, h + 1 - k, _source_low(lo, h, n, k), _bc_sign(lo))
        _fill_ghost!(data, d, h + n + k, _source_high(hi, h, n, k), _bc_sign(hi))
    end
    return _apply_bc_dims!(data, g, d + 1, Base.tail(bcs))
end
_apply_bc_dims!(data, g, d::Int, ::Tuple{}) = nothing

function _fill_ghost!(data, d::Int, ghost::Int, source::Int, sign::Int)
    dst = selectdim(data, d, ghost)
    src = selectdim(data, d, source)
    dst .= sign .* src
    return nothing
end

"""
    fold_bc!(data, g::AbstractGrid) -> data

Exact discrete adjoint of [`apply_bc!`](@ref): fold each ghost value back into its
mirror/wrap source cell with the same sign, then zero all ghost layers. Dimensions
are folded in reverse order `N:1`, transposing the fill order exactly.
"""
function fold_bc!(data::AbstractArray{<:Any,N}, g::AbstractGrid{N}) where {N}
    _fold_bc_dims!(data, g, 1, boundary_conditions(g))
    return data
end

function _fold_bc_dims!(data, g, d::Int, bcs::Tuple)
    _fold_bc_dims!(data, g, d + 1, Base.tail(bcs))
    lo, hi = first(bcs)
    h = halo_width(g)[d]
    n = local_size(g)[d]
    for k in 1:h
        _fold_ghost!(data, d, h + 1 - k, _source_low(lo, h, n, k), _bc_sign(lo))
        _fold_ghost!(data, d, h + n + k, _source_high(hi, h, n, k), _bc_sign(hi))
    end
    return nothing
end
_fold_bc_dims!(data, g, d::Int, ::Tuple{}) = nothing

function _fold_ghost!(data, d::Int, ghost::Int, source::Int, sign::Int)
    dst = selectdim(data, d, ghost)
    src = selectdim(data, d, source)
    src .+= sign .* dst
    fill!(dst, zero(eltype(data)))
    return nothing
end

"""
    zero_ghosts!(data, g::AbstractGrid) -> data

Set every ghost cell of the halo-padded array `data` to zero.
"""
function zero_ghosts!(data::AbstractArray{<:Any,N}, g::AbstractGrid{N}) where {N}
    for d in 1:N
        h = halo_width(g)[d]
        n = local_size(g)[d]
        fill!(selectdim(data, d, 1:h), zero(eltype(data)))
        fill!(selectdim(data, d, (h + n + 1):(n + 2 * h)), zero(eltype(data)))
    end
    return data
end

#--------------------------------------------------------------------------------# Inhomogeneous ghost offsets

# Writes the affine ghost offsets of the full (inhomogeneous) boundary fill into a
# ZEROED padded array: Dirichlet ghosts get 2·value, Neumann ghosts get
# (2k-1)·Δ·flux at layer k, periodic ghosts stay zero. Sweeping a raw stencil over
# the result yields the boundary lift `b` of the affine split L(x) = A·x + b; see
# `boundary_rhs`.
function fill_bc_inhomogeneous!(data::AbstractArray{<:Any,N}, g::AbstractGrid{N}) where {N}
    bcs = boundary_conditions(g)
    for d in 1:N
        lo, hi = bcs[d]
        h = halo_width(g)[d]
        n = local_size(g)[d]
        for k in 1:h
            _offset_ghost!(data, d, h + 1 - k, lo, spacing(g)[d], k)
            _offset_ghost!(data, d, h + n + k, hi, spacing(g)[d], k)
        end
    end
    return data
end

_offset_ghost!(data, d::Int, ghost::Int, ::Periodic, Δ, k::Int) = nothing
function _offset_ghost!(data, d::Int, ghost::Int, bc::Dirichlet, Δ, k::Int)
    iszero(bc.value) && return nothing
    dst = selectdim(data, d, ghost)
    dst .= 2 .* bc.value
    return nothing
end
function _offset_ghost!(data, d::Int, ghost::Int, bc::Neumann, Δ, k::Int)
    iszero(bc.flux) && return nothing
    dst = selectdim(data, d, ghost)
    dst .= (2 * k - 1) .* Δ .* bc.flux
    return nothing
end
