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

import MatrixFreeOperators: AbstractGrid, apply_bc!, fold_bc!

# Rule-firing counters. Asserting that a rule *fired* is the only way to catch one
# that silently stopped dispatching — without it every test still passes, via the
# tape. One atomic add per differentiated call; the primal path never touches it.
const RULE_HITS = Dict{Symbol,Threads.Atomic{Int}}(
    :apply_bc => Threads.Atomic{Int}(0),
    :fold_bc => Threads.Atomic{Int}(0),
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

end
