#--------------------------------------------------------------------------------# Advection

"""
    SelfAdvection()

Velocity marker for [`advection`](@ref) selecting self-advection `u·∇u`: the
velocity is the advected state itself, making the operator nonlinear.
"""
struct SelfAdvection end

@inline function _adv_at(
    u::AbstractArray{<:Any,N}, v::AbstractArray{<:Any,N}, I::CartesianIndex{N}, inv_h::NTuple{N}
) where {N}
    terms = ntuple(Val(N)) do d
        δ = _unitindex(Val(N), d)
        @inbounds v[I][d] * ((u[I + δ] - u[I - δ]) * (inv_h[d] / 2))
    end
    return sum(terms)
end

"""
    Advection(grid, velocity)

Matrix-free advection leaf `v·∇`. Linear when the velocity is a prescribed
[`Field`](@ref) (passive transport — a field-valued coefficient does not make an
operator nonlinear); nonlinear when the velocity is [`SelfAdvection`](@ref)
(`u·∇u`). Construct with [`advection`](@ref).
"""
struct Advection{G<:AbstractGrid,V} <: AbstractOperator
    grid::G
    velocity::V
end

"""
    advection(g::AbstractGrid, velocity::Field) -> Advection
    advection(g::BlockForest, velocity::AbstractBlockField) -> Advection
    advection(g::AbstractGrid, ::SelfAdvection) -> Advection

Build a matrix-free advection operator `v·∇` bound to `g`, with second-order
central differences. A prescribed velocity must be an `SVector`-valued field with
one component per grid dimension — on a [`BlockForest`](@ref) a block field on
that same forest (matching the applied field's layout lets the forest-native
kernel engage); the resulting operator is linear in the
advected input and acts componentwise on scalar or vector inputs. With
[`SelfAdvection`](@ref) the operator computes `u·∇u` of a vector field and is
nonlinear — see [`linearize`](@ref) for its Jacobian.

### Examples

```julia
g = CartesianGrid(((0.0, 2π), (0.0, 2π)), (32, 32);
                  bc=((Periodic(), Periodic()), (Periodic(), Periodic())))
v = set!(vector_field(g), x -> SVector(1.0, 0.0))
A = advection(g, v)                  # linear: passive transport by v
B = advection(g, SelfAdvection())    # nonlinear: u·∇u
```
"""
function advection(g::AbstractGrid, velocity::Field)
    eltype(velocity.data) <: SVector || throw(
        ArgumentError(
            "prescribed advection velocity must be an SVector-valued field, got eltype $(eltype(velocity.data))",
        ),
    )
    ncomponents(velocity) == dimension(g) || throw(
        ArgumentError(
            "velocity has $(ncomponents(velocity)) components but the grid is $(dimension(g))-D",
        ),
    )
    return Advection(g, velocity)
end
advection(g::AbstractGrid, velocity::SelfAdvection) = Advection(g, velocity)
function advection(g::BlockForest, velocity::AbstractBlockField)
    eltype(velocity) <: SVector || throw(
        ArgumentError(
            "prescribed advection velocity must be an SVector-valued field, got eltype $(eltype(velocity))",
        ),
    )
    ncomponents(velocity) == dimension(g) || throw(
        ArgumentError(
            "velocity has $(ncomponents(velocity)) components but the grid is $(dimension(g))-D",
        ),
    )
    # _leaf_op slices the velocity by this forest's leaf indices — a velocity
    # bound to a different forest would alias the wrong leaves. Topology
    # identity, not wrapper identity: adapted twins share the forest.
    velocity.grid.forest === g.forest ||
        throw(ArgumentError("advection velocity must be a field on the same forest"))
    return Advection(g, velocity)
end

islinear(::Advection{<:AbstractGrid,<:AbstractField}) = true
islinear(::Advection{<:AbstractGrid,SelfAdvection}) = false
isconstant(::Advection{<:AbstractGrid,<:AbstractField}) = true
isconstant(::Advection{<:AbstractGrid,SelfAdvection}) = false
# One sweep reading x's (and the velocity's construction-filled) ghosts, both forms.
shares_exchange(::Advection) = true
operator_grid(L::Advection) = L.grid

# Adjoint of passive transport, expressed in the operator algebra: the mechanical
# transpose of Σ_d diag(v_d)·D_d is Σ_d D_dᵀ·diag(v_d).
function adjoint_operator(L::Advection{<:AbstractGrid,<:AbstractField})
    g = L.grid
    terms = ntuple(Val(dimension(g))) do d
        Composed(adjoint_operator(derivative(g, d)), scaling(component(L.velocity, d)))
    end
    return reduce(Added, terms)
end

function apply!(
    y::Field, L::Advection{<:AbstractGrid,<:Field}, x::Field, g::AbstractGrid, α, β
)
    halo_update!(x, g)
    apply_bc!(x)
    return _apply_raw!(y, L, x, g, α, β)
end

function _apply_raw!(
    y::Field, L::Advection{<:AbstractGrid,<:Field}, x::Field, g::AbstractGrid, α, β
)
    inv_h = _inv_spacing(g)
    yi = interior(y)
    if iszero(β)
        yi .= α .* _adv_at.(Ref(x.data), Ref(L.velocity.data), interior(g), Ref(inv_h))
    else
        yi .= α .* _adv_at.(Ref(x.data), Ref(L.velocity.data), interior(g), Ref(inv_h)) .+ β .* yi
    end
    return y
end

function apply!(
    y::Field, L::Advection{<:AbstractGrid,SelfAdvection}, x::Field, g::AbstractGrid, α, β
)
    eltype(x.data) <: SVector ||
        throw(ArgumentError("self-advection u·∇u requires an SVector-valued field"))
    halo_update!(x, g)
    apply_bc!(x)
    inv_h = _inv_spacing(g)
    yi = interior(y)
    if iszero(β)
        yi .= α .* _adv_at.(Ref(x.data), Ref(x.data), interior(g), Ref(inv_h))
    else
        yi .= α .* _adv_at.(Ref(x.data), Ref(x.data), interior(g), Ref(inv_h)) .+ β .* yi
    end
    return y
end

function Adapt.adapt_structure(to, L::Advection)
    return Advection(Adapt.adapt(to, L.grid), Adapt.adapt(to, L.velocity))
end
