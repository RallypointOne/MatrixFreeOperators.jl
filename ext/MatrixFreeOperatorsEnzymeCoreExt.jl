"""
Enzyme reverse-mode rules that route the forest's ghost sweeps through their
declared transposes instead of taping them.

`_exchange_storage!` (inter-block halo) and `_bc_storage!` (physical-BC faces) are
**constant, parameter-free linear maps** over a forest's raw block storage: they
overwrite ghost cells with weighted sums of interior or already-filled ghost cells,
and the weights come from the grid's topology and boundary-condition types, never
from the differentiated data. For such a map `x ← H·x`, reverse mode is exactly
`x̄ ← Hᵀ·x̄` — and `_exchange_storage_adjoint!` / `_bc_storage_adjoint!` already *are*
`Hᵀ`, phase- and dimension-reversed, checked by the dot-product identity in the
suite. The rules therefore carry an **empty tape**: nothing from the forward pass
has to survive to the reverse pass.

Two payoffs. Enzyme never performs type analysis on the coarse–fine `GhostFill`
descriptor loop — a `Vector` of structs each holding another `Vector`, hence not
isbits, which is what trips `EnzymeNoTypeError` on some platforms (issue #26). And
on GPU backends the same sweeps run as KernelAbstractions kernels, which DESIGN.md
§6 says must never be differentiated through; a rule makes that structural rather
than incidental.

These rules cannot swallow a parameter gradient: neither function reads an
operator, let alone a coefficient field, so a coefficient's cotangent never flows
through here. Gradients w.r.t. `ScalingOp`/`Advection` coefficient fields keep
taping exactly as before.

Two implementation constraints, both learned the hard way and both load-bearing:

  * **Raw storage, never a field.** From Julia 1.12 a custom-rule argument may not
    mix GC-tracked pointers with inline floats (EnzymeAD/Enzyme.jl#2707), which a
    `BlockField` does through its embedded grid. Hence the storage seam: a bare
    block vector or packed array plus isbits descriptors.
  * **No allocation in a rule body.** Rule bodies are compiled into Enzyme's
    generated code; a dynamic dispatch or GC allocation there segfaulted on Linux
    x86_64 while running clean on macOS/aarch64. The hit counters below are
    therefore plain `const` atomics reached by static field access, and the bodies
    build no closures.

The single-grid `apply_bc!` deliberately has *no* rule. It is `2·N·halo` slab
broadcasts that Enzyme already tapes correctly and cheaply, so a rule there adds
risk for negligible gain — the descriptor sweeps are where taping actually hurts.
"""
module MatrixFreeOperatorsEnzymeCoreExt

using MatrixFreeOperators
using EnzymeCore
using EnzymeCore.EnzymeRules

import MatrixFreeOperators:
    BlockLayout, _bc_storage!, _bc_storage_adjoint!, _exchange_storage!,
    _exchange_storage_adjoint!

# Rule-firing counters. Asserting that a rule *fired* is the only way to catch one
# that silently stops dispatching — without it every numerical test still passes,
# via the tape, and the only symptom is the taping this module exists to prevent
# quietly coming back. Plain consts, not a Dict: a lookup inside a rule body
# allocates, and allocation there is what segfaulted under Enzyme codegen.
const EXCHANGE_HITS = Threads.Atomic{Int}(0)
const EXCHANGE_ADJOINT_HITS = Threads.Atomic{Int}(0)
const BC_HITS = Threads.Atomic{Int}(0)
const BC_ADJOINT_HITS = Threads.Atomic{Int}(0)

"""
    rule_hits() -> NamedTuple

How many times each rule's augmented forward pass has run. Used by the test suite
to assert a rule actually fired.
"""
rule_hits() = (
    exchange=EXCHANGE_HITS[],
    exchange_adjoint=EXCHANGE_ADJOINT_HITS[],
    bc=BC_HITS[],
    bc_adjoint=BC_ADJOINT_HITS[],
)

# Every shadow an annotation carries, uniformly across batch widths. `Const` carries
# none, so the reverse pass degrades to a no-op instead of a special case.
@inline _shadows(::Const) = ()
@inline _shadows(x::Union{Duplicated,DuplicatedNoNeed}) = (x.dval,)
@inline _shadows(x::Union{BatchDuplicated,BatchDuplicatedNoNeed}) = x.dval

# `store` is the only argument carrying a cotangent. The descriptors — an
# `ExchangeSchedule`, or the grid's already-separated isbits pieces — are read as
# constants, but Enzyme may still hand the schedule over as `Duplicated`, because
# its interpolation weights are `Float64`. Leaving that shadow untouched reports
# ∂L/∂weights = 0, which is right: those weights are fixed rationals of the 2:1
# refinement ratio and of the boundary-condition mirror signs, not model inputs.
for (primal, transpose, counter) in (
    (:_exchange_storage!, :_exchange_storage_adjoint!, :EXCHANGE_HITS),
    (:_exchange_storage_adjoint!, :_exchange_storage!, :EXCHANGE_ADJOINT_HITS),
    (:_bc_storage!, :_bc_storage_adjoint!, :BC_HITS),
    (:_bc_storage_adjoint!, :_bc_storage!, :BC_ADJOINT_HITS),
)
    # Both directions carry a rule: applying an adjoint operator runs the transpose
    # in its primal, and (Hᵀ)ᵀ = H makes the forward sweep its reverse body, so
    # those paths stay on rules too.
    @eval function EnzymeRules.augmented_primal(
        ::EnzymeRules.RevConfig,
        func::Const{typeof($primal)},
        ::Type{<:Annotation},
        store::Annotation,
        lay::Const{<:BlockLayout},
        desc::Vararg{Annotation},
    )
        Threads.atomic_add!($counter, 1)
        func.val(store.val, lay.val, map(_val, desc)...)
        # Empty tape, and no primal or shadow return: these sweeps return `nothing`.
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
        dvals = map(_val, desc)
        for s̄ in _shadows(store)
            $transpose(s̄, lay.val, dvals...)
        end
        return ntuple(_ -> nothing, Val(2 + length(desc)))
    end
end

@inline _val(a::Annotation) = a.val

end
