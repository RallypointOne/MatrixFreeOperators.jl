#--------------------------------------------------------------------------------# Distributability guards

# The whitelist behind prepare_distributed. It lives in core, not in the MDLA
# extension, because these guards are the entire defense against a silently
# wrong distributed answer — and an extension-only guard is only tested where a
# GPU is present, which CI is not. Everything here is CPU-reachable.
#
# The rule is the package's "traits default to the weaker claim" invariant: an
# operator is distributable only if it says so. A forgotten declaration falls to
# the ::AbstractOperator fallback and throws, never returns a wrong answer.

"""
    _distributable(L::AbstractOperator) -> Bool

Whether `L`'s ghost needs are met by the slab exchange machinery (internal).

Stencil and pointwise leaves qualify; combinators propagate explicitly. An
operator that reads a field-valued parameter does not, because that parameter is
bound to the *global* grid and would silently broadcast against a slab interior.
Transfer operators do not, because their factors live on two different grids
that would each need their own consistent partitioning.
"""
_distributable(::Laplacian) = true
_distributable(::IdentityOp) = true
_distributable(S::ScalingOp) = S.coeff isa Number
_distributable(L::Scaled) = _distributable(L.op)
_distributable(L::Added) = _distributable(L.a) && _distributable(L.b)
_distributable(::AbstractOperator) = false

# Why each rejected case is rejected, for an error message that names the reason
# rather than just the type. Anything without an entry gets the generic tail.
_undistributable_reason(::Restriction) =
    "transfer operators compose grids that would each need their own consistent partitioning"
_undistributable_reason(::Prolongation) =
    "transfer operators compose grids that would each need their own consistent partitioning"
_undistributable_reason(S::ScalingOp) =
    "a field-valued coefficient is bound to the global grid; it must be partitioned onto the slabs first"
_undistributable_reason(::Advection) =
    "the velocity field is bound to the global grid; it must be partitioned onto the slabs first"
_undistributable_reason(::Gradient) =
    "a rank-changing operator is non-square, so it needs two partition specs; use it inside a composition whose result is scalar"
_undistributable_reason(::Divergence) =
    "a rank-changing operator is non-square, so it needs two partition specs; use it inside a composition whose result is scalar"
_undistributable_reason(::AbstractOperator) = nothing

# Walk to the first undistributable node so the message points at the actual
# culprit rather than at the root of a large tree.
_first_undistributable(L::Scaled) = _first_undistributable(L.op)
function _first_undistributable(L::Added)
    a = _first_undistributable(L.a)
    return a === nothing ? _first_undistributable(L.b) : a
end
_first_undistributable(L::AbstractOperator) = _distributable(L) ? nothing : L

function _check_distributable(L::AbstractOperator)
    _distributable(L) && return L
    culprit = something(_first_undistributable(L), L)
    reason = _undistributable_reason(culprit)
    detail = reason === nothing ? "" : " — $reason"
    throw(
        ArgumentError(
            "prepare_distributed cannot distribute $(nameof(typeof(culprit)))$detail. " *
            "Supported: Laplacian, IdentityOp, number-coefficient ScalingOp, and their " *
            "Scaled/Added combinations; got $(sprint(show, L)).",
        ),
    )
end
