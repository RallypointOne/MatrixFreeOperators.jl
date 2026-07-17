#--------------------------------------------------------------------------------# Transfer validation and shared kernels

# The 2:1 cell-centered transfer pair (multigrid): prolongation interpolates a
# coarse field to the fine grid, restriction is its scaled transpose. One kernel
# pair serves both directions — `_prolong_at` is P's forward sweep and
# `_transfer_gather` is Pᵀ's gather — so the dot-product identity
# ⟨P·x, y⟩ = ⟨x, Pᵀ·y⟩ holds to machine precision by construction.

_same_bc_kind(::AbstractBC, ::AbstractBC) = false
_same_bc_kind(::Periodic, ::Periodic) = true
_same_bc_kind(::Dirichlet, ::Dirichlet) = true
_same_bc_kind(::Neumann, ::Neumann) = true

function _validate_transfer(fine::CartesianGrid{N}, coarse::CartesianGrid{N}) where {N}
    ntuple(d -> 2 * local_size(coarse)[d], Val(N)) == local_size(fine) || throw(
        ArgumentError(
            "transfer requires a 2:1 grid pair, got fine $(local_size(fine)) vs coarse $(local_size(coarse))",
        ),
    )
    fine.extent == coarse.extent ||
        throw(ArgumentError("transfer grids must share their extent"))
    (_has_interface(fine) || _has_interface(coarse)) && throw(
        ArgumentError(
            "transfer operators do not support Interface faces (forest leaves)"
        ),
    )
    for d in 1:N, side in 1:2
        _same_bc_kind(boundary_conditions(fine)[d][side], boundary_conditions(coarse)[d][side]) ||
            throw(
                ArgumentError(
                    "transfer grids must agree on the boundary-condition kind of every face",
                ),
            )
    end
    fine.device == coarse.device ||
        throw(ArgumentError("transfer grids must live on the same device"))
    return nothing
end

# Fine-interior value of the linear tensor-product interpolant: fine child f of
# coarse parent c = (f+1)>>1 sits at ξ = ∓1/4 of the parent cell, so each dim
# blends parent (3/4) with the neighbor on the child's side (1/4). Boundary
# children read coarse ghost layer 1, filled by apply_bc! before the sweep.
@inline function _prolong_at(
    u::AbstractArray{T,N},
    I::CartesianIndex{N},
    hf::NTuple{N,Int},
    hc::NTuple{N,Int},
    w::NTuple{2,W},
) where {T,N,W}
    f = ntuple(d -> I[d] - hf[d], Val(N))
    J0 = ntuple(d -> ((f[d] + 1) >> 1) + hc[d], Val(N))
    s = ntuple(d -> isodd(f[d]) ? -1 : 1, Val(N))
    acc = zero(W) * zero(T)
    @inbounds for t in CartesianIndices(ntuple(_ -> 0:1, Val(N)))
        wt = prod(ntuple(d -> w[t[d] + 1], Val(N)))
        acc += wt * u[CartesianIndex(ntuple(d -> J0[d] + t[d] * s[d], Val(N)))]
    end
    return acc
end

# Flipped-stencil gather for Pᵀ at a padded coarse index: coarse cell c is read
# by its two children (weight 3/4) and by the adjacent child of each neighbor
# (weight 1/4), i.e. fine taps (2c-2, 2c-1, 2c, 2c+1) with weights
# (1/4, 3/4, 3/4, 1/4) per dim. Reads are bounds-masked and fine ghosts are
# zeroed by adjoint_gather!, so only fine interior cotangents enter.
@inline function _transfer_gather(
    ȳ::AbstractArray{T,N},
    J::CartesianIndex{N},
    hf::NTuple{N,Int},
    hc::NTuple{N,Int},
    w4::NTuple{4,W},
) where {T,N,W}
    F0 = ntuple(d -> 2 * (J[d] - hc[d]) - 1 + hf[d], Val(N))
    acc = zero(W) * zero(T)
    @inbounds for t in CartesianIndices(ntuple(_ -> -1:2, Val(N)))
        wt = prod(ntuple(d -> w4[t[d] + 2], Val(N)))
        acc += wt * _maskedget(ȳ, CartesianIndex(ntuple(d -> F0[d] + t[d], Val(N))))
    end
    return acc
end

_prolong_weights(::Type{T}) where {T} = (T(3) / 4, T(1) / 4)
_gather_weights(::Type{T}) where {T} = (T(1) / 4, T(3) / 4, T(3) / 4, T(1) / 4)

# Zeroed field with f's element type on another grid's padded shape.
function _field_like(f::Field, g::AbstractGrid)
    data = KernelAbstractions.zeros(
        KernelAbstractions.get_backend(g), eltype(f.data), padded_size(g)...
    )
    return Field(data, g)
end

#--------------------------------------------------------------------------------# Prolongation

"""
    Prolongation(coarse, fine)

Coarse-to-fine transfer operator on a 2:1 cell-centered grid pair: per-dim
linear interpolation blending each fine cell's coarse parent (weight 3/4) with
the neighbor on the child's side (weight 1/4). Boundary children read the
coarse homogeneous ghost fill, so the interpolant is consistent with the grid's
boundary conditions (Dirichlet wall children get `u₁/2`, Neumann get `u₁`).
Adjoint of [`Restriction`](@ref) up to scaling: `Pᵀ = 2^N · R`. Construct with
[`prolongation`](@ref).
"""
struct Prolongation{GC<:CartesianGrid,GF<:CartesianGrid} <: AbstractOperator
    coarse::GC
    fine::GF
end

"""
    prolongation(coarse::CartesianGrid, fine::CartesianGrid) -> Prolongation

Build the coarse-to-fine linear interpolation operator for a 2:1 grid pair.
The grids must share extent, boundary-condition kinds, and device.

### Examples

```julia
gf = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (64, 64))
gc = coarsen(gf)
P = prolongation(gc, gf)      # coarse -> fine
R = restriction(gf, gc)       # fine -> coarse, R = 2⁻ᴺ·Pᵀ
```

See also: [`restriction`](@ref), [`coarsen`](@ref).
"""
function prolongation(coarse::CartesianGrid{N}, fine::CartesianGrid{N}) where {N}
    _validate_transfer(fine, coarse)
    return Prolongation(coarse, fine)
end

islinear(::Prolongation) = true
isconstant(::Prolongation) = true
operator_grid(L::Prolongation) = L.coarse

function Base.size(L::Prolongation)
    return (prod(local_size(L.fine)), prod(local_size(L.coarse)))
end

allocate_output(L::Prolongation, x::AbstractField) = _field_like(x, L.fine)
allocate_input(L::Prolongation, y::AbstractField) = _field_like(y, L.coarse)

function apply!(y::Field, L::Prolongation, x::Field, g::AbstractGrid, α, β)
    halo_update!(x, L.coarse)
    apply_bc!(x)
    return _apply_raw!(y, L, x, g, α, β)
end

function _apply_raw!(y::Field, L::Prolongation, x::Field, ::AbstractGrid, α, β)
    w = _prolong_weights(eltype(spacing(L.coarse)))
    hf = halo_width(L.fine)
    hc = halo_width(L.coarse)
    yi = interior(y)
    if iszero(β)
        yi .= α .* _prolong_at.(Ref(x.data), interior(L.fine), Ref(hf), Ref(hc), Ref(w))
    else
        yi .= α .* _prolong_at.(Ref(x.data), interior(L.fine), Ref(hf), Ref(hc), Ref(w)) .+
            β .* yi
    end
    return y
end

function apply_adjoint!(x̄::Field, L::Prolongation, ȳ::Field, g::AbstractGrid, α, β)
    gather = let hf = halo_width(L.fine),
        hc = halo_width(L.coarse),
        w4 = _gather_weights(eltype(spacing(L.coarse)))

        (u, J) -> _transfer_gather(u, J, hf, hc, w4)
    end
    return adjoint_gather!(x̄, ȳ, gather, α, β)
end

function adjoint_operator(L::Prolongation)
    T = eltype(spacing(L.coarse))
    return Scaled(Restriction(L.fine, L.coarse), T(2)^dimension(L.coarse))
end

function Adapt.adapt_structure(to, L::Prolongation)
    return Prolongation(Adapt.adapt(to, L.coarse), Adapt.adapt(to, L.fine))
end
