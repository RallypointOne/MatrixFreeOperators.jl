"""
Enzyme reverse-mode rules that route homogeneous ghost fills through their
declared transposes instead of taping them.

`apply_bc!` is a **constant, parameter-free linear map** on a halo-padded array: it
overwrites ghost cells with signed copies of interior cells, and the signs come
from the grid's boundary-condition *types*, never from the differentiated data. For
such a map `x ← B·x`, reverse mode is exactly `x̄ ← Bᵀ·x̄` — and `fold_bc!` already
*is* `Bᵀ`: it folds each ghost cotangent back into its mirror/wrap source with the
same sign, zeroes the ghost layers, and runs dimensions in reverse order,
transposing the fill order exactly. So these rules carry an **empty tape**: nothing
from the forward pass is needed to run the reverse pass.

The payoff is that Enzyme never differentiates the slab-broadcast sweep — `2·N·h`
broadcasts per operator application — and, once the forest seam lands, never
differentiates the coarse–fine descriptor loops or the KernelAbstractions kernels
that DESIGN.md §6 says must not be taped through.

These rules cannot swallow a parameter gradient: `apply_bc!` reads no operator, let
alone a coefficient field, so a coefficient's cotangent never flows through here.
Gradients w.r.t. `ScalingOp`/`Advection` coefficient fields keep taping as before.

Julia 1.12 note: a custom rule cannot take an argument whose type mixes GC-tracked
pointers with inline floats — Enzyme's 1.12 calling-convention support rejects it
with a `CallingConventionMismatchError` (EnzymeAD/Enzyme.jl#2707). That is why the
seam is the **array-level** `apply_bc!(::AbstractArray, ::AbstractGrid)` rather than
the `Field`/`BlockField` wrappers: a bare padded array is a pure-pointer argument
and `CartesianGrid` is all-bits, so both survive. The `BlockForest` sweeps need the
same treatment through a storage-level seam.
"""
module MatrixFreeOperatorsEnzymeCoreExt

using MatrixFreeOperators
using EnzymeCore
using EnzymeCore.EnzymeRules

import MatrixFreeOperators:
    AbstractGrid, BlockLayout, ExchangeSchedule, _bc_storage!, _bc_storage_adjoint!,
    _exchange_storage!, _exchange_storage_adjoint!, apply_bc!, fold_bc!

# Rule-firing counters. Asserting that a rule *fired* is the only way to catch one
# that silently stopped dispatching — without it every test still passes, via the
# tape. One atomic add per differentiated call; the primal path never touches it.
const RULE_HITS = Dict{Symbol,Threads.Atomic{Int}}(
    :apply_bc => Threads.Atomic{Int}(0),
    :fold_bc => Threads.Atomic{Int}(0),
    :exchange => Threads.Atomic{Int}(0),
    :exchange_adjoint => Threads.Atomic{Int}(0),
    :bc_faces => Threads.Atomic{Int}(0),
    :bc_faces_adjoint => Threads.Atomic{Int}(0),
)

@inline _hit!(k::Symbol) = (Threads.atomic_add!(RULE_HITS[k], 1); nothing)

# Every shadow an annotation carries, uniformly across batch widths. `Const` carries
# none, so the reverse pass degrades to a no-op instead of a special case.
@inline _shadows(::Const) = ()
@inline _shadows(x::Union{Duplicated,DuplicatedNoNeed}) = (x.dval,)
@inline _shadows(x::MixedDuplicated) = (x.dval[],)
@inline _shadows(x::Union{BatchDuplicated,BatchDuplicatedNoNeed}) = x.dval
@inline _shadows(x::BatchMixedDuplicated) = map(r -> r[], x.dval)

# Always the primal grid: a shadow grid carries zeroed spacing, which would silently
# produce a wrong transpose.
@inline _gridval(g::Annotation{<:AbstractGrid}) = g.val

# Cotangent to hand back for the grid argument. Enzyme's activity analysis can
# promote a grid to `Active` (it has `Float64` fields), and a reverse rule must then
# return a value of the grid's own type. Zero is not a truncation here: these two
# functions depend on the grid only through boundary-condition *types*, halo widths,
# and sizes — integers and type parameters — never through `extent` or `spacing`. So
# the geometry cotangent genuinely is zero, and no global
# `inactive_type(::Type{<:AbstractGrid})` is needed (which would also zero the
# geometry gradients DESIGN.md Decision B wants to keep available elsewhere).
@inline _grid_cotangent(g::Active) = EnzymeCore.make_zero(g.val)
@inline _grid_cotangent(::Annotation) = nothing

@inline function _augmented(config, data, primal!)
    primal!()
    primal = EnzymeRules.needs_primal(config) ? data.val : nothing
    shadow = if EnzymeRules.needs_shadow(config) && !(data isa Const)
        data.dval
    else
        nothing
    end
    # Empty tape: the transpose reads only the cotangent buffer and the grid's
    # integer/type-level metadata, so no primal value must survive to the reverse.
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

#--------------------------------------------------------------------------------# apply_bc!

# The array-level method is what `apply_bc!(::Field)` and every leaf's `apply!`
# funnel into, so one rule here covers the whole single-grid path.
function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    func::Const{typeof(apply_bc!)},
    ::Type{<:Annotation},
    data::Annotation{<:AbstractArray},
    g::Annotation{<:AbstractGrid},
)
    _hit!(:apply_bc)
    return _augmented(config, data, () -> func.val(data.val, _gridval(g)))
end

function EnzymeRules.reverse(
    ::EnzymeRules.RevConfig,
    ::Const{typeof(apply_bc!)},
    ::Type{<:Annotation},
    ::Nothing,
    data::Annotation{<:AbstractArray},
    g::Annotation{<:AbstractGrid},
)
    gv = _gridval(g)
    for d̄ in _shadows(data)
        fold_bc!(d̄, gv)
    end
    return (nothing, _grid_cotangent(g))
end

#--------------------------------------------------------------------------------# fold_bc!

# (Bᵀ)ᵀ = B. Differentiating an adjoint application — `apply(adjoint(L), w)`, or a
# nested `PreparedAdjoint` — runs `fold_bc!` in the primal, and its transpose is the
# ordinary fill. Registering the symmetric partner keeps those paths on rules too.
function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    func::Const{typeof(fold_bc!)},
    ::Type{<:Annotation},
    data::Annotation{<:AbstractArray},
    g::Annotation{<:AbstractGrid},
)
    _hit!(:fold_bc)
    return _augmented(config, data, () -> func.val(data.val, _gridval(g)))
end

function EnzymeRules.reverse(
    ::EnzymeRules.RevConfig,
    ::Const{typeof(fold_bc!)},
    ::Type{<:Annotation},
    ::Nothing,
    data::Annotation{<:AbstractArray},
    g::Annotation{<:AbstractGrid},
)
    gv = _gridval(g)
    for d̄ in _shadows(data)
        apply_bc!(d̄, gv)
    end
    return (nothing, _grid_cotangent(g))
end

#--------------------------------------------------------------------------------# forest storage seams

# `_exchange_storage!` and `_bc_storage!` are the same story one level up: constant
# linear maps over a forest's raw block storage, with `_exchange_storage_adjoint!`
# and `_bc_storage_adjoint!` as their exact transposes. Taking them off the tape is
# what keeps the coarse–fine `GhostFill` descriptor sweep — a `Vector` of structs
# each holding another `Vector`, which is not isbits — out of Enzyme's type
# analysis, and what makes DESIGN.md §6's "never AD through a kernel" rule
# structural for the packed GPU exchange rather than incidental.
#
# Only `store` carries a cotangent. The descriptor arguments — an `ExchangeSchedule`
# or the grid's already-separated isbits pieces — are read as constants, but Enzyme
# can still hand the schedule over as `Duplicated` because its interpolation weights
# are `Float64`. Leaving that shadow untouched reports ∂L/∂weights = 0, which is
# right: those weights are fixed rationals of the 2:1 refinement ratio, not inputs.

for (fwd, rev, key_f, key_r) in (
    (:_exchange_storage!, :_exchange_storage_adjoint!, :exchange, :exchange_adjoint),
    (:_bc_storage!, :_bc_storage_adjoint!, :bc_faces, :bc_faces_adjoint),
)
    # Both directions get a rule: an adjoint operator application runs the transpose
    # in its primal, and (Hᵀ)ᵀ = H makes the forward sweep its reverse body.
    for (primal, transpose, key) in ((fwd, rev, key_f), (rev, fwd, key_r))
        @eval function EnzymeRules.augmented_primal(
            config::EnzymeRules.RevConfig,
            func::Const{typeof($primal)},
            ::Type{<:Annotation},
            store::Annotation,
            lay::Const{<:BlockLayout},
            desc::Vararg{Annotation},
        )
            _hit!($(QuoteNode(key)))
            func.val(store.val, lay.val, map(d -> d.val, desc)...)
            return EnzymeRules.AugmentedReturn(nothing, nothing, nothing)
        end

        @eval function EnzymeRules.reverse(
            ::EnzymeRules.RevConfig,
            ::Const{typeof($primal)},
            ::Type{<:Annotation},
            ::Nothing,
            store::Annotation,
            lay::Const{<:BlockLayout},
            desc::Vararg{Annotation},
        )
            dvals = map(d -> d.val, desc)
            for s̄ in _shadows(store)
                $transpose(s̄, lay.val, dvals...)
            end
            return ntuple(_ -> nothing, Val(2 + length(desc)))
        end
    end
end

end
