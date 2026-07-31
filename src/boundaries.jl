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

"""
    Interface()

Inter-block interface "boundary" carried on faces whose ghost values are
external inputs: every face of a [`BlockForest`](@ref) leaf grid, and the
partition-cut faces of a [`partition_grid`](@ref) slab. The homogeneous fill
(`apply_bc!`/`fold_bc!`) and the inhomogeneous lift skip `Interface` faces:
those ghosts are filled by [`halo_update!`](@ref) (forest) or a distributed
exchange (slabs), and physical-boundary ghosts by the forest-level face passes,
so per-leaf BC sweeps never touch a ghost slab. Internal (not exported):
`leaf_grid` and `partition_grid` place it — in a user grid's bc it is rejected
at `BlockForest` construction, and hand-placed on a plain `CartesianGrid` it
would leave ghosts silently unfilled.
"""
struct Interface <: AbstractBC end

# Whether any face is an Interface — i.e. the grid is a forest leaf whose
# cross-block ghost values are external inputs, so a stencil's adjoint must
# scatter cotangents into them instead of shortcutting through the forward
# action. Constant-folds: the BC types are grid type parameters.
_has_interface(g::AbstractGrid) =
    any(pair -> pair[1] isa Interface || pair[2] isa Interface, boundary_conditions(g))

# Sign of the homogeneous ghost fill relative to the mirrored interior cell.
_bc_sign(::Periodic) = 1
_bc_sign(::Dirichlet) = -1
_bc_sign(::Neumann) = 1

# Padded source index mirrored by ghost layer k (h = halo width, n = interior count).
_source_low(::Periodic, h::Int, n::Int, k::Int) = h + n + 1 - k
_source_low(::AbstractBC, h::Int, n::Int, k::Int) = h + k
_source_high(::Periodic, h::Int, n::Int, k::Int) = h + k
_source_high(::AbstractBC, h::Int, n::Int, k::Int) = h + n + 1 - k

# Forest face-pass capability gate: a physical BC must implement the fill/fold/lift
# primitives before it can back a BlockForest boundary — erroring at construction
# beats silently unfilled ghosts.
_require_face_bc(::Dirichlet) = nothing
_require_face_bc(::Neumann) = nothing
function _require_face_bc(bc::AbstractBC)
    throw(
        ArgumentError(
            "$(nameof(typeof(bc))) has no forest face-pass implementation; define " *
            "_bc_sign, _offset_ghost!, and _require_face_bc methods to support it " *
            "on a BlockForest",
        ),
    )
end

# Type-stable slab view along compile-time dimension D (selectdim with a runtime
# dimension boxes the view type, which would allocate inside hot mul! loops).
@inline function _dimslice(data::AbstractArray{<:Any,N}, ::Val{D}, r) where {N,D}
    return view(data, ntuple(d -> d == D ? r : Colon(), Val(N))...)
end

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
    _apply_bc_dims!(data, g, Val(1), boundary_conditions(g))
    return data
end

function _apply_bc_dims!(data, g, ::Val{D}, bcs::Tuple) where {D}
    lo, hi = first(bcs)
    h = halo_width(g)[D]
    n = local_size(g)[D]
    for k in 1:h
        _fill_ghost!(data, Val(D), h + 1 - k, lo, _source_low(lo, h, n, k))
        _fill_ghost!(data, Val(D), h + n + k, hi, _source_high(hi, h, n, k))
    end
    return _apply_bc_dims!(data, g, Val(D + 1), Base.tail(bcs))
end
_apply_bc_dims!(data, g, ::Val, ::Tuple{}) = nothing

# Interface faces are filled by halo_update!, not the homogeneous BC fill.
_fill_ghost!(data, ::Val, ghost::Int, ::Interface, source::Int) = nothing
function _fill_ghost!(data, dim::Val, ghost::Int, bc::AbstractBC, source::Int)
    dst = _dimslice(data, dim, ghost:ghost)
    src = _dimslice(data, dim, source:source)
    dst .= _bc_sign(bc) .* src
    return nothing
end

"""
    fold_bc!(data, g::AbstractGrid) -> data

Exact discrete adjoint of [`apply_bc!`](@ref): fold each ghost value back into its
mirror/wrap source cell with the same sign, then zero all ghost layers. Dimensions
are folded in reverse order `N:1`, transposing the fill order exactly.
"""
function fold_bc!(data::AbstractArray{<:Any,N}, g::AbstractGrid{N}) where {N}
    _fold_bc_dims!(data, g, Val(1), boundary_conditions(g))
    return data
end

function _fold_bc_dims!(data, g, ::Val{D}, bcs::Tuple) where {D}
    _fold_bc_dims!(data, g, Val(D + 1), Base.tail(bcs))
    lo, hi = first(bcs)
    h = halo_width(g)[D]
    n = local_size(g)[D]
    for k in 1:h
        _fold_ghost!(data, Val(D), h + 1 - k, lo, _source_low(lo, h, n, k))
        _fold_ghost!(data, Val(D), h + n + k, hi, _source_high(hi, h, n, k))
    end
    return nothing
end
_fold_bc_dims!(data, g, ::Val, ::Tuple{}) = nothing

# Interface faces are folded by halo_update_adjoint!, not the homogeneous adjoint.
_fold_ghost!(data, ::Val, ghost::Int, ::Interface, source::Int) = nothing
function _fold_ghost!(data, dim::Val, ghost::Int, bc::AbstractBC, source::Int)
    dst = _dimslice(data, dim, ghost:ghost)
    src = _dimslice(data, dim, source:source)
    src .+= _bc_sign(bc) .* dst
    fill!(dst, zero(eltype(data)))
    return nothing
end

"""
    zero_ghosts!(data, g::AbstractGrid) -> data

Set every ghost cell of the halo-padded array `data` to zero.
"""
function zero_ghosts!(data::AbstractArray{<:Any,N}, g::AbstractGrid{N}) where {N}
    hn = ntuple(d -> (halo_width(g)[d], local_size(g)[d]), Val(N))
    _zero_ghosts_dims!(data, Val(1), hn)
    return data
end

# Recurse on a shrinking `(halo, size)` pair tuple with an empty-tuple base case —
# the same idiom as `_apply_bc_dims!`. Terminating on `Base.tail`/`Tuple{}` (rather
# than a `D > N` guard over a grid forwarded unchanged) is what keeps Julia
# specializing every level, so the ghost-slab fills allocate nothing; the earlier
# grid-forwarding form boxed its argument each recursion — a per-leaf cost in the
# forest adjoint gather's zero_ghosts!.
function _zero_ghosts_dims!(data::AbstractArray{<:Any,N}, ::Val{D}, hn::Tuple) where {N,D}
    h, n = first(hn)
    fill!(_dimslice(data, Val(D), 1:h), zero(eltype(data)))
    fill!(_dimslice(data, Val(D), (h + n + 1):(n + 2 * h)), zero(eltype(data)))
    return _zero_ghosts_dims!(data, Val(D + 1), Base.tail(hn))
end
_zero_ghosts_dims!(::AbstractArray, ::Val, ::Tuple{}) = nothing

"""
    zero_bc_ghosts!(data, g::AbstractGrid) -> data

Zero exactly the ghost cells [`fold_bc!`](@ref) folds — every non-[`Interface`](@ref)
face, whole slabs, corners included. `Interface` faces are left untouched: they carry
neighbour-owned cotangents that belong to `halo_update_adjoint!` / the distributed
reduction, not to the homogeneous adjoint.

The complement of [`zero_ghosts!`](@ref), which clears every ghost including
`Interface`. Used by [`adjoint_gather!`](@ref) to drop a running total's stale
foldable ghosts before an accumulating gather, so `fold_bc!` sees only that call's
contribution.
"""
function zero_bc_ghosts!(data::AbstractArray{<:Any,N}, g::AbstractGrid{N}) where {N}
    hnb = ntuple(
        d -> (halo_width(g)[d], local_size(g)[d], boundary_conditions(g)[d]), Val(N)
    )
    _zero_bc_ghosts_dims!(data, Val(1), hnb)
    return data
end

# Shrinking-tuple recursion for the same reason `_zero_ghosts_dims!` uses one: a
# forwarded grid boxes at every level and allocates per call.
function _zero_bc_ghosts_dims!(data::AbstractArray{<:Any,N}, ::Val{D}, hnb::Tuple) where {N,D}
    h, n, (lo, hi) = first(hnb)
    for k in 1:h
        _zero_bc_ghost!(data, Val(D), h + 1 - k, lo)
        _zero_bc_ghost!(data, Val(D), h + n + k, hi)
    end
    return _zero_bc_ghosts_dims!(data, Val(D + 1), Base.tail(hnb))
end
_zero_bc_ghosts_dims!(::AbstractArray, ::Val, ::Tuple{}) = nothing

# Interface faces are folded by halo_update_adjoint!, so their slabs must survive.
_zero_bc_ghost!(::AbstractArray, ::Val, ::Int, ::Interface) = nothing
function _zero_bc_ghost!(data, dim::Val, ghost::Int, ::AbstractBC)
    fill!(_dimslice(data, dim, ghost:ghost), zero(eltype(data)))
    return nothing
end

#--------------------------------------------------------------------------------# Inhomogeneous ghost offsets

# Writes the affine ghost offsets of the full (inhomogeneous) boundary fill into a
# ZEROED padded array: Dirichlet ghosts get 2·value, Neumann ghosts get
# (2k-1)·Δ·flux at layer k, periodic ghosts stay zero. Sweeping a raw stencil over
# the result yields the boundary lift `b` of the affine split L(x) = A·x + b; see
# `boundary_rhs`.
function fill_bc_inhomogeneous!(data::AbstractArray{<:Any,N}, g::AbstractGrid{N}) where {N}
    _fill_inhomogeneous_dims!(data, g, Val(1), boundary_conditions(g))
    return data
end

function _fill_inhomogeneous_dims!(data, g, ::Val{D}, bcs::Tuple) where {D}
    lo, hi = first(bcs)
    h = halo_width(g)[D]
    n = local_size(g)[D]
    for k in 1:h
        _offset_ghost!(data, Val(D), h + 1 - k, lo, spacing(g)[D], k)
        _offset_ghost!(data, Val(D), h + n + k, hi, spacing(g)[D], k)
    end
    return _fill_inhomogeneous_dims!(data, g, Val(D + 1), Base.tail(bcs))
end
_fill_inhomogeneous_dims!(data, g, ::Val, ::Tuple{}) = nothing

_offset_ghost!(data, ::Val, ghost::Int, ::Periodic, Δ, k::Int) = nothing
_offset_ghost!(data, ::Val, ghost::Int, ::Interface, Δ, k::Int) = nothing
function _offset_ghost!(data, dim::Val, ghost::Int, bc::Dirichlet, Δ, k::Int)
    iszero(bc.value) && return nothing
    dst = _dimslice(data, dim, ghost:ghost)
    dst .= 2 .* bc.value
    return nothing
end
function _offset_ghost!(data, dim::Val, ghost::Int, bc::Neumann, Δ, k::Int)
    iszero(bc.flux) && return nothing
    dst = _dimslice(data, dim, ghost:ghost)
    dst .= (2 * k - 1) .* Δ .* bc.flux
    return nothing
end
