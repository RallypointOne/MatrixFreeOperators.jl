#--------------------------------------------------------------------------------# Restriction

"""
    Restriction(fine, coarse)

Fine-to-coarse transfer operator on a 2:1 cell-centered grid pair, defined as
the scaled transpose of [`Prolongation`](@ref): `R = 2⁻ᴺ·Pᵀ` — full weighting,
per-dim interior weights `(1/8, 3/8, 3/8, 1/8)`. Sharing P's kernels makes the
adjoint identity exact by construction. Input ghost layers are scratch (zeroed
before the gather), so only fine interior values contribute and the affine
boundary lift is identically zero. Construct with [`restriction`](@ref).
"""
struct Restriction{GF<:CartesianGrid,GC<:CartesianGrid} <: AbstractOperator
    fine::GF
    coarse::GC
end

"""
    restriction(fine::CartesianGrid, coarse::CartesianGrid=coarsen(fine)) -> Restriction

Build the fine-to-coarse full-weighting operator for a 2:1 grid pair. The grids
must share extent, boundary-condition kinds, and device.

### Examples

```julia
gf = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (64, 64))
R = restriction(gf)                  # to coarsen(gf)
P = prolongation(coarsen(gf), gf)    # its adjoint partner: Rᵀ = 2⁻ᴺ·P
```

See also: [`prolongation`](@ref), [`coarsen`](@ref).
"""
function restriction(fine::CartesianGrid{N}, coarse::CartesianGrid{N}) where {N}
    _validate_transfer(fine, coarse)
    return Restriction(fine, coarse)
end
restriction(fine::CartesianGrid) = restriction(fine, coarsen(fine))

islinear(::Restriction) = true
isconstant(::Restriction) = true
operator_grid(L::Restriction) = L.fine

function Base.size(L::Restriction)
    return (prod(local_size(L.coarse)), prod(local_size(L.fine)))
end

allocate_output(L::Restriction, x::AbstractField) = _field_like(x, L.coarse)
allocate_input(L::Restriction, y::AbstractField) = _field_like(y, L.fine)

_restrict_scale(L::Restriction) = eltype(spacing(L.fine))(2)^(-dimension(L.fine))

# The forward action IS the scaled Pᵀ gather (adjoint_gather! zeroes input
# ghosts and folds the coarse output through its BCs), so ⟨R·x, y⟩ = ⟨x, Rᵀ·y⟩
# holds exactly — which also means the forward path inherits the gather's ghost
# contract: β ≠ 0 accumulates the whole padded coarse field, physical-BC ghosts
# included, and leaves them folded away.
function apply!(y::Field, L::Restriction, x::Field, g::AbstractGrid, α, β)
    gather = let hf = halo_width(L.fine),
        hc = halo_width(L.coarse),
        w4 = _gather_weights(eltype(spacing(L.fine)))

        (u, J) -> _transfer_gather(u, J, hf, hc, w4)
    end
    return adjoint_gather!(y, x, gather, α * _restrict_scale(L), β)
end

function apply_adjoint!(x̄::Field, L::Restriction, ȳ::Field, g::AbstractGrid, α, β)
    P = Prolongation(L.coarse, L.fine)
    halo_update!(ȳ, L.coarse)
    apply_bc!(ȳ)
    return _apply_raw!(x̄, P, ȳ, g, α * _restrict_scale(L), β)
end

function adjoint_operator(L::Restriction)
    return Scaled(Prolongation(L.coarse, L.fine), _restrict_scale(L))
end

# Reads no meaningful ghost data (the gather engine zeroes input ghosts), so
# the raw sweep is the ordinary action — and boundary_rhs(R, ·) is exactly zero.
function _apply_raw!(y::Field, L::Restriction, x::Field, g::AbstractGrid, α, β)
    return apply!(y, L, x, g, α, β)
end

function Adapt.adapt_structure(to, L::Restriction)
    return Restriction(Adapt.adapt(to, L.fine), Adapt.adapt(to, L.coarse))
end
