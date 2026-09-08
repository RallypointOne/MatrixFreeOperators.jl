# Design Document: A Matrix-Free Operator Package for PDEs on Structured Grids

> Status: design exploration — recommended architecture with clearly flagged
> open decisions. Relocate into the new package repo once it is scaffolded.

## Context

We want a new Julia package providing **matrix-free linear (and some nonlinear)
operators** for solving PDEs, scoped initially to **structured grids** but built
so any grid amenable to matrix-free application fits later. The package must, by
requirement:

1. Run on **any device (CPU, GPU, …) with little-to-no code change** — via
   KernelAbstractions.jl and/or Reactant.jl.
2. Support **efficient forward- and reverse-mode autodiff** — via Enzyme.jl
   and/or Mooncake.jl — including **gradients w.r.t. operator parameters**
   (material coefficients, geometry) for inverse problems / PDE-constrained
   optimization.
3. Support **adaptive grids (AMR)**.
4. Be **easily extendable to geometric multigrid**.
5. Expose a **composable operator API** in the spirit of RadialBasisFunctions.jl.
6. Not preclude **distributed-memory** execution (multi-GPU / multi-node) later.
7. **Compose with the user's MultiDeviceLinearAlgebra.jl (MDLA)** and use
   Krylov.jl + OrdinaryDiffEq.jl as the solver stack.

The research that backs the decisions below covered SciMLOperators.jl,
ParallelStencil.jl + ImplicitGlobalGrid.jl, KernelAbstractions.jl, Reactant.jl,
Enzyme.jl, Mooncake.jl, the distributed-array landscape (DistributedArrays.jl,
PencilArrays.jl, MPIHaloArrays.jl), the AMR landscape (Trixi.jl, p4est/t8code),
GeometricMultigrid.jl, Oceananigans.jl as a reference architecture, and the
user's own RadialBasisFunctions.jl and MultiDeviceLinearAlgebra.jl.

**Why a new package and not ParallelStencil.jl:** ParallelStencil is a
*kernel-authoring DSL*, not a composable operator library. It offers no
first-class operator objects (`L1*L2`, `L1+L2`, `adjoint(L)`), no autodiff
integration to speak of, no AMR, no multigrid abstractions, and it *requires*
its macros in user code. It is excellent at what it does (architecture-agnostic
stencil kernels, GPU-aware MPI halos via ImplicitGlobalGrid) and we should learn
from its backend dispatch and `@hide_communication` overlap, but it does not
meet requirements 2–5. We sit one layer above it.

---

## 1 — Decisions locked (with the user)

| # | Decision | Consequence |
|---|----------|-------------|
| **A. Authoring model** | **Array-level (broadcast/slicing) leaves by default; KernelAbstractions `@kernel` as a per-operator escape hatch.** | This is the only default that simultaneously delivers device-agnosticism, Reactant-traceability, and automatic AD (incl. parameter gradients). Hand-written kernels are reserved for hot stencils that need shared-memory tiling. On the forest hot path the escape hatch graduates into a forest-native kernel layer — one launch sweeping all leaves over packed storage (§5) — while array-level per-leaf leaves remain the authoring default and the permanent correctness/AD reference. |
| **B. AD scope** | **Gradients w.r.t. both the solution field AND operator parameters** (material coefficients, geometry). | Custom adjoint rules become an *optimization* for linear leaves, not the whole AD story. Parameter gradients must flow through real AD — which array-level leaves provide for free. |
| **C. Solver API** | **Own lean lazy operator algebra exposing `mul!`/`size`/`eltype`. Drop SciMLOperators as the core.** Target Krylov.jl + MDLA + OrdinaryDiffEq.jl. | No `(u,p,t)` convention or `cache_operator` ceremony. A thin *optional* SciML `jac_prototype` adapter is provided only for matrix-free **implicit** OrdinaryDiffEq stepping (see §7). |

These three reshape everything below; the rest of the open choices are flagged in
§10 for the user to decide, with recommendations.

---

## 2 — Architecture overview

Five layers, each independently understandable and testable:

```
┌────────────────────────────────────────────────────────────────┐
│ Solver interop:  Krylov.jl · MDLA · OrdinaryDiffEq (RHS adapter) │  ← ext/
├────────────────────────────────────────────────────────────────┤
│ Autodiff:  auto via Enzyme/Mooncake on array-level leaves        │  ← ext/ (rules)
│            + optional adjoint-routing rules for linear leaves    │
├────────────────────────────────────────────────────────────────┤
│ Operator algebra (CORE):  lazy leaves + Added/Composed/Scaled/   │
│   Adjoint · traits (linear/constant/self-adjoint) · adjoint(L)   │
│   · apply!(y,L,x,grid) · mul!/size/eltype                        │
├────────────────────────────────────────────────────────────────┤
│ Execution:  array-level (GPUArrays+Adapt) default · @kernel      │
│   escape hatch · get_backend dispatch · Reactant reachable       │
├────────────────────────────────────────────────────────────────┤
│ Grid + Field (FOUNDATION):  topology · spacing · halo width ·    │
│   local/global index ranges · BCs · device · halo_update! seam   │
└────────────────────────────────────────────────────────────────┘
```

Design invariant that makes the forward-looking requirements cheap: **operators
are authored once against the Grid's `halo_update!` seam and the Field's index
ranges.** Single-device, distributed, and (later) AMR differ only in what the
Grid object *is* and what `halo_update!` *does* — not in operator code.

---

## 3 — Core abstraction 1: Grid and Field (the foundation)

Everything keys off the grid. It must expose enough that distributed and AMR are
reachable without rewriting operators.

```julia
abstract type AbstractGrid{N} end          # N = spatial dimension

# v1 concrete type
struct CartesianGrid{N,T,BC,Dev,Topo} <: AbstractGrid{N}
    extent      :: NTuple{N,Tuple{T,T}}     # physical (min,max) per dim
    spacing     :: NTuple{N,T}              # Δx, Δy, … (uniform in v1) — DERIVED by the
                                            # constructor from extent+size (stored for hot
                                            # loops, never user-set: 3 fields, 2 DOFs)
    size        :: NTuple{N,Int}            # interior cell counts (LOCAL count)
    halo        :: NTuple{N,Int}            # ghost-layer width per dim
    bc          :: BC                       # boundary conditions per face
    device      :: Dev                      # KernelAbstractions backend
    # --- seams that stay no-op in v1, carry real data later ---
    local_range :: NTuple{N,UnitRange{Int}} # interior index range (== global in v1)
    topology    :: Topo                     # Nothing in v1 (keeps the struct isbits-
                                            # compatible for @kernel args / Adapt /
                                            # Reactant); neighbor ranks / hierarchy later
end
```

Grid interface (the contract operators rely on — designed as a small
convention-based interface so foreign grid types can opt in):

- `dimension(g)`, `spacing(g)`, `local_size(g)`, `halo_width(g)`
- `interior(g)` → `CartesianIndices` of owned (non-halo) cells
- `boundary_conditions(g)`
- `get_backend(g)` (KernelAbstractions backend; also derivable from a field)
- **`halo_update!(field, g)`** — the single distributed seam. v1: no-op.
  Later: MPI/MDLA/Reactant fill ghost layers. Operators call this before any
  stencil that reads neighbors.
- (AMR) `BlockForest` implements this same interface per leaf and adds
  `refine!(g, predicate)`, `coarsen!(g, predicate)`, `balance!(g)`, and `leaves(g)`
  (see §9 AMR). Each leaf is an ordinary all-`Interface` `CartesianGrid`;
  `halo_update!` does the inter-block coupling — same-level copies plus quadratic
  coarse–fine interpolation/restriction at refinement interfaces — and a
  forest-level `apply_bc!` face pass fills physical domain faces (see §9).

**Field:** keep it minimal — a device array plus a reference to its grid and a
*location* tag (so staggered grids are additive later):

```julia
struct Field{L,A<:AbstractArray,G<:AbstractGrid}
    data     :: A          # CPU Array, CuArray, ROCArray, … (device-agnostic)
    grid     :: G
    # L is a location trait: Center, XFace, YFace, … (v1: Center only)
end
```

v1 uses **collocated (cell-centered)** fields. Staggered finite-volume layouts
(Oceananigans-style, velocities on faces) are the location trait `L` — flagged in
§10 as a near-term extension, not v1.

**Vector & multi-component fields.** A field's *element type* carries its tensor
rank — there is no rank type parameter and no new struct field. A scalar field is
a `Field` over `Array{T}`; a vector field is a `Field` over `Array{SVector{N,T}}`;
a tensor field over `Array{SMatrix{…}}`. The backing array's `eltype` is the only
thing that changes.

This works because **operators are written generically over the element type — the
core design principle here.** Stencil bodies use plain componentwise / broadcast
arithmetic and never hard-code `SVector`. So the vector→vector case (an operator
mapping a vector field to a vector field) is the *easy* one: the same `apply!` body for `Laplacian` /
`Derivative` runs unchanged on an `SVector`-valued field, because `SVector`
arithmetic is componentwise. `SVector` (StaticArrays — already a core dep) is the
batteries-included default, but **any user-supplied isbits vector/tensor element
type that is closed under the stencil arithmetic Just Works with no operator
changes.**

The only leaves that touch components explicitly are the **rank-changers** already
in the algebra: `Gradient` builds an `SVector{N,T}`-valued output from a scalar
input, and `Divergence` contracts it back. These define the small contract a custom
element type implements to opt into rank changes (build-from-N-components /
extract-component); `SVector` satisfies it for free.

**Krylov interop** stays flat: `mul!` / `size` / `eltype` see a vector, obtained by
`reinterpret`-ing `Array{SVector{N,T}}` ↔ `Array{T}` (length `N·ncells`, `eltype T`).
This holds for any isbits fixed-size element type; an exotic custom type can supply
a flatten/unflatten adapter. The flat vector spans **interior DOFs only** — ghost
cells are determined by BCs / `halo_update!`, never solver unknowns — so
`size(L) = (N·n_int, N·n_int)`; the padded↔flat boundary lives in the prepared
path (§10.7). Collocated v1 represents a vector field as one
`SVector`-valued `Field`; the deferred staggered layout (§10.2) instead uses one
face-located scalar `Field` per component.

---

## 4 — Core abstraction 2: the composable operator algebra

This is the heart, and it mirrors RadialBasisFunctions.jl's *style* (lazy
operator symbols + an algebra) **without depending on or subtyping RBF** (RBF is
matrix-based, scattered-data, k-NN; we are matrix-free, structured-grid — see
§10 / §11 for why coupling is wrong).

```julia
abstract type AbstractOperator end

# ---- Leaves (lazy symbols; bound to a grid when instantiated) ----
struct Laplacian{G}   <: AbstractOperator; grid::G; end
struct Derivative{G}  <: AbstractOperator; grid::G; dim::Int; order::Int; end
struct Gradient{G}    <: AbstractOperator; grid::G; end
struct Divergence{G}  <: AbstractOperator; grid::G; end
struct Advection{G,V} <: AbstractOperator; grid::G; velocity::V; end
    # islinear dispatches on V: prescribed velocity (constant / Field) ⇒ LINEAR in the
    # advected input (passive transport); state-coupled (self-advection u·∇u) ⇒ nonlinear
struct ScalingOp{F}   <: AbstractOperator; coeff::F; end   # pointwise ×κ(x) — diagonal,
                                                           # the parameter-field leaf (Decision B)
struct IdentityOp     <: AbstractOperator; end   # grid-free: size/eltype resolved from a
                                                 # composition sibling or the applied-to field

# ---- Lazy combinators (closed under the algebra) ----
struct Scaled{O,T}   <: AbstractOperator; op::O; α::T; end
struct Added{A,B}    <: AbstractOperator; a::A; b::B; end
struct Composed{A,B} <: AbstractOperator; a::A; b::B; end   # (A∘B)x = A(Bx)
struct AdjointOp{O}  <: AbstractOperator; op::O; end

Base.:+(a::AbstractOperator, b::AbstractOperator) = Added(a, b)
Base.:*(a::AbstractOperator, b::AbstractOperator) = Composed(a, b)
Base.:*(α::Number, a::AbstractOperator)           = Scaled(a, α)
# …plus the rest of the closure set, all one-liners:
#   a*α = α*a · -a = Scaled(a, -1) · a-b = Added(a, Scaled(b, -1)) · a/α = Scaled(a, inv(α))
```

Each operator must provide:

**1. Action** — `apply!(y, L, x, grid)`. For leaves this is the array-level body
(Decision A). The matrix-free analogue of RBF's `mul!(y, W, x)`:

```julia
function apply!(y, L::Laplacian, x, g)        # 1-D illustration
    Δx² = spacing(g)[1]^2
    halo_update!(x, g)                        # no-op in v1; fills ghosts later
    @views @. y[2:end-1] = (x[1:end-2] - 2x[2:end-1] + x[3:end]) / Δx²
    apply_bc!(y, L, g)                        # BC handling — see §10
    return y
end
```

**2. Adjoint** — `adjoint(L)::AbstractOperator` returning a lazy adjoint, plus
`apply_adjoint!(x, L, y, grid)`. **Every operator declares its adjoint
explicitly. We do NOT default to self-adjoint** — boundary conditions break
self-adjointness even for the Laplacian (only periodic / symmetric-homogeneous
cases are self-adjoint). The algebra propagates adjoints automatically:

```julia
adjoint(L::Added)    = Added(adjoint(L.a), adjoint(L.b))
adjoint(L::Composed) = Composed(adjoint(L.b), adjoint(L.a))   # reversed order
adjoint(L::Scaled)   = Scaled(adjoint(L.op), conj(L.α))
adjoint(L::AdjointOp)= L.op
# leaves: declared per-operator, e.g. adjoint(L::Derivative) = a sign/shift-flipped Derivative
```

**3. Traits** (Holy traits — orthogonal capabilities, per the interface skill):

```julia
# Defaults make the WEAKER claim — leaves opt IN to strong properties
# (a forgotten declaration degrades to an error, never a wrong result):
islinear(::AbstractOperator)       = false    # linear leaves opt in (one line each)
isconstant(::AbstractOperator)     = false    # opt in when parameters/coeffs are time-invariant
isselfadjoint(::AbstractOperator)  = false    # opt-in ONLY when BCs make it true
isdiagonal(::AbstractOperator)     = false    # enables cheap Jacobi smoother (multigrid)

# Propagation through combinators — explicit, so no combinator falls back to the default:
islinear(L::Added)         = islinear(L.a) && islinear(L.b)     # same for Composed; isconstant likewise
islinear(L::Scaled)        = islinear(L.op)
isdiagonal(L::Composed)    = isdiagonal(L.a) && isdiagonal(L.b) # same for Added/Scaled
isselfadjoint(L::Composed) = false   # NOT compositional: A,B self-adjoint ⇏ AB self-adjoint
                                     # (needs A,B to commute); Scaled additionally needs real α
```

**4. LinearAlgebra interop** — `mul!(y,L,x)`, `mul!(y,L,x,α,β)`, `size(L)`,
`eltype(L)`. This is all Krylov.jl and MDLA need; no SciML dependency.

**Device transfer:** implement `Adapt.adapt_structure` for every operator so
`adapt(CuArray, L)` moves bound grids/coefficients to device — exactly RBF's
pattern.

> **Tensor rank (flagged, §10):** RBF parameterizes operators by added tensor
> rank `AbstractOperator{N}` (gradient adds a dim, divergence removes one). We
> avoid a rank parameter on the *operator* entirely: rank lives in the *field's
> element type* and operators stay generic over it (see "Vector & multi-component
> fields" in §Core abstraction 1; the `Gradient`/`Divergence` rank-changers are the
> only leaves that touch components).

### 4a — Custom (fused) operators

The lazy algebra above composes leaves for clarity, AD, and Reactant fusion — at the
cost of **one kernel launch per leaf** plus an intermediate buffer per `Composed` node
(§10.7). For a hot residual or Jacobian–vector product evaluated inside a Newton–Krylov
inner loop, that overhead is real. The sanctioned escape hatch is a **custom (fused)
operator**: a user-authored concrete `AbstractOperator` subtype whose `apply!`/`mul!`
fuses a differential stencil with its pointwise companions — a reaction term, a source, a
variable coefficient, a sparse coupling — into a **single pass** (array-level or a
`@kernel`, §5). It is not outside the algebra: it satisfies the same operator interface, so
it still composes with `+`/`*`, feeds Krylov via `mul!`, and plugs into the OrdinaryDiffEq
adapters (§7) like any built-in leaf.

Reach for a custom operator when the terms genuinely share a grid sweep — fusing them saves
a full read of the field and a launch — and the fused result is linear (or nonlinear) as a
unit. When the terms are independent and reusable, prefer composing built-in leaves and let
Reactant (§5 / §10.7) fuse the tree.

**The contract** a custom operator implements (the `AbstractOperator` interface, restated as
a checklist):

```
Required:
- apply!(y, L, x, grid) -> y          action; a fused leaf does stencil + pointwise in one pass
- eltype(L) -> T
- size(L) -> (n, n)                   interior DOF count (for linear ops used as Krylov maps)
- mul!(y, L, x) -> y                  linear ops only; the matrix-free action for Krylov
                                      (5-arg mul!(y,L,x,α,β) recommended — fuse the axpby)
Optional:
- adjoint(L) / apply_adjoint!(x,L,y,grid)   linear ops only; unlocks Lᵀ-needing Krylov + AD
- Adapt.adapt_structure(to, L)              device (GPU) transfer
- linearize!(L, u) -> L                     in-place refresh of a Jacobian op at state u
Traits (default false — opt in):
- islinear, isconstant, isselfadjoint, isdiagonal
Invariants:
- islinear(L) ⇒ L(0)=0: homogeneous BCs only (§10.6 linear/affine split)
- size(L) spans interior DOFs only (ghosts are never solver unknowns)
- a fused leaf reusing an exported stencil primitive must be numerically identical to the
  equivalent built-in-leaf composition (parity invariant — gate it with a test)
```

**Reusable stencil primitives (required, so fusion does not fork the stencil).** A custom
fused kernel must not re-derive a stencil that a built-in leaf already owns — that splits one
numerical definition into two that drift apart. The core therefore **exports its leaf stencil
bodies as `@inline` primitives callable inside a user `@kernel`**, and implements its own
built-in leaves *via the same primitives*. The first such primitive (driven by the motivating
example below) is a no-flux 7-point Laplacian that returns **both the center value and the
Laplacian** — a fused caller needs the center value for its pointwise term:

```julia
# exported; a flat, halo-free escape-hatch stencil for fused custom @kernels.
@inline laplacian_7pt_noflux(u, c, i, j, k, nx, ny, nz, sy, sz, ihx, ihy, ihz) -> (uc, lap)
```

Two hard guarantees on that primitive:

- It is **numerically identical** to the built-in `Laplacian{_,Neumann}` leaf under
  homogeneous no-flux — the leaf fills symmetric mirror ghosts and sweeps the halo-based
  `laplacian_stencil`, the primitive skips the missing face flux on a flat halo-free state, and
  the two encodings agree to floating-point tolerance (the parity invariant above, locked by
  test). They are deliberately *separate* bodies: the halo model stays the general default (it
  generalizes to Dirichlet, inhomogeneous, periodic + distributed, and free adjoints), while
  the flat primitive is the fused hot-loop specialization that avoids a halo round-trip.
- No-flux Neumann means **skip the missing face flux at boundary cells** — equivalent to the
  leaf's symmetric mirror ghost, the homogeneous form §10.6 requires.

**In-place linearization refresh.** A custom Jacobian operator (below) freezes a
linearization state `u`; a Newton–Krylov solve refreshes it every iteration. Reallocating the
operator tree per iteration is wasteful, so the core sanctions an **in-place**
`linearize!(L, u)` that overwrites the frozen state on a reusable operator. This is safe
precisely because Krylov solves are never differentiated *through* (§10.7) — the operator's
statefulness is invisible to AD, whose sensitivities come from implicit-function-theorem
adjoints on the *solution*, not the iteration. (Out-of-place `linearize(F, u₀)` of §10.8 is
the AD-facing spelling; `linearize!` is its hot-loop sibling.)

**Motivating example — `TissueJacobianOperator` (MicrovascularOxygenTransport).** The tissue
O₂ Jacobian–vector product fuses three terms over one grid sweep: the no-flux Laplacian
`D∇²v` (via the exported primitive above), a Michaelis–Menten reaction-derivative diagonal
`M'(u)/α · v`, and a perivascular self-coupling `−ΠᵀgΠ v / (Vcell·α)` (two sparse matvecs run
just before the kernel). It is a **linear** custom operator frozen at `u`, refreshed by
`linearize!`, with `mul!` as its action — exactly the `linearize(F, u₀)` Jacobian of §10.8
and the matrix-free `jac_prototype` of §7. It is the first external consumer of this package,
and the reason the custom-operator pattern and the exported stencil primitive exist. (If that
consumer later wants the §7 `jac_prototype` route, the optional `…SciMLExt.jl` adapter must
wrap any `AbstractOperator` exposing `mul!`/`size`/`eltype`.)

---

## 5 — Execution / device layer

- **Default (array-level):** leaf bodies are `@.`/broadcast/slicing over the
  field array. These run on `Array`, `CuArray`, `ROCArray`, `MtlArray`, … through
  Base broadcast + GPUArrays — the same device story RBF already gets from
  `get_backend` + `Adapt`. No per-backend code.
- **Escape hatch (`@kernel`):** a leaf may instead launch a KernelAbstractions
  kernel via `get_backend(x)` for hot/complex stencils (shared-memory tiling).
  Cost, made explicit to authors: such a leaf is **not Reactant-traceable** and
  **needs a hand-written adjoint + parameter-adjoint rule** (Decision B).
- **Backend inference:** infer from the input array via
  `KernelAbstractions.get_backend(field.data)`; the grid also carries a `device`
  for allocation. (Idiom verified in KA docs and Oceananigans.)
- **Reactant reachable, not foundational:** because the default leaves are
  array-level, Reactant can `@compile` whole operator applications later for XLA
  fusion + sharding + MLIR-AD with no operator rewrite. We do **not** build v1 on
  Reactant (youngest/riskiest; sharp edges on scalar code, dynamic shapes).
- **Forest-native kernels (packed phase, #15):** the forest hot path adds a third
  execution mode. `PackedBlockField` is a twin of `BlockField` behind a shared
  `AbstractBlockField` interface: all leaves in one contiguous
  `(blocksize .+ 2halo ..., nleaves)` array plus a per-leaf `levels` vector — the
  geometry SoA; per-leaf spacing/inv-h² are *recomputed in-kernel* from `spacing0`
  and the level, bit-identical to `leaf_grid` + `_inv_spacing2` (recompute > store;
  the workload is bandwidth-bound). Packed **storage** is a drop-in: `block(f, i)`
  is a trailing-dim view, so the descriptor loops, BC face passes, flat transfers,
  and the per-leaf fallback sweep run unchanged — every operator works on packed
  storage with no kernel written. The single-launch **sweep** is per-operator: a
  KernelAbstractions kernel over `ndrange = (blocksize..., nleaves)` (leaf = trailing
  index) whose body calls the *same* per-cell stencil function the broadcast path
  uses (`_lap_at`, `_deriv_at`, …) on a per-leaf view, so the numerical definition
  never forks. Migration is layered behind the `_forest_sweep!` dispatch seam:
  storage first, then one kernel override per operator. Kernel overrides engage on
  **GPU backends only** — per-leaf launch overhead is the problem they solve; on
  CPU backends fused broadcasts beat KA CPU codegen (~1.6× measured on the
  Laplacian), so packed fields route to the per-leaf reference sweep there — the
  same execution-mode-per-backend principle as the halo bullet above. Kernel AD
  policy follows
  the escape-hatch rule above: forest kernels get declared adjoints — transpose-gather
  kernels reusing the `_*_adjoint_gather` stencils plus the existing
  `fold_bc!`/`halo_update_adjoint!` transposes, verified by the dot-product
  identity — never AD-through-kernel. `BlockField` remains the correctness
  reference and the AD (Enzyme/Mooncake) + Reactant path. Coefficient fields
  follow the storage layout: an operator's auxiliary per-cell data (`ScalingOp`
  coefficients, `Advection` velocities) must live in the same layout as the
  field it is applied to for the kernel override to engage — packed × packed
  dispatches to the single-launch kernel; any mismatch degrades to the per-leaf
  reference sweep, never a wrong result. `prepare` normalizes the mismatch away:
  a `BlockField` coefficient under a packed prototype is `pack`ed once at
  prepare time (sound because these leaves declare `isconstant`), so the hot
  path cannot silently stay on the fallback; staleness after a regrid is caught
  by the existing generation guards on both the prepared and un-prepared paths.
  The exchange itself is kernelized for packed fields on GPU backends: the
  per-generation schedule flattens into a device-resident descriptor SoA
  (`_DeviceSchedule` — copies bucketed by normal dim, the nested
  `GhostFill.terms` CSR-flattened per phase, bcfaces as device leaf lists,
  cached keyed on generation and backend), executed as a constant number of
  launches whose arithmetic is bit-identical to the host loops except
  copy-phase corner ghosts (last dim wins; no axis-aligned stencil reads them).
  Flat transfers collapse to one broadcast per direction against the
  whole-forest interior view. The **adjoint** exchange, `fold_bc!`, and the
  setup-only inhomogeneous pass stay on the host descriptor loops on all
  backends: adjoint scatter-adds collide on shared source cells, so
  kernelizing them needs atomics (bit-nondeterministic) or a transposed
  source-centric CSR — the sanctioned future path if it is ever needed; they
  run only off the `mul!` hot path.

> **API to verify at implementation:** exact `KernelAbstractions.get_backend`,
> `allocate`, `@index`/`@kernel` signatures (KA ≈ v0.9.x); Reactant
> `@compile`/sharding API (fast-moving, v0.2.x) — describe strategy now, pin
> signatures when coding.

**Why not JACC.jl (evaluated, not adopted).** JACC.jl (ORNL's Kokkos/RAJA-style
portability layer: `parallel_for` / `parallel_reduce` over CUDA/AMDGPU/oneAPI/
Metal/threads) is an *alternative* to this KA + array-level layer, **not** a
complement, and it loses on the two decisions that define the package:

- **Backend model.** JACC fixes one backend **project-wide** via Preferences.jl
  (`set_backend` / `@init_backend`). Decision A infers the backend **per array**
  from `get_backend(field.data)` — that is what lets a single process run CPU and
  CUDA arrays together (device-parity tests, §verification) and what the
  multi-device MDLA path *requires* (partitions living on different devices). One
  global backend can express neither.
- **AD / Reactant.** JACC's function-as-argument kernels (`parallel_for`, portable
  `@atomic`) are the same imperative form as the `@kernel` escape hatch: **not**
  Reactant-traceable and needing **hand-written adjoints**. They buy nothing on
  Decision B (automatic field + parameter gradients) — the entire payoff of
  array-level leaves. Adopting JACC would also add a *second* portability framework
  plus its own array abstraction (`JACC.array` / `JACC.zeros`) on top of the chosen
  KA + GPUArrays + Adapt stack.

*Worth borrowing as lessons, not as a dependency:* (1) JACC's documented
small-kernel **launch overhead** vs Kokkos (LULESH) is empirical support for Open
Decision 7 — composed matrix-free operators launch one kernel **per leaf**, so
whole-tree fusion (Reactant) and, on forests, the packed single-launch sweep (#15)
are the fixes, not micro-optimizations; (2) its two-tier
API (simple productivity default + low-level escape hatch) independently converges
on Decision A's split; (3) its benchmark suite (XSBench / miniBUDE / LULESH vs
Kokkos/native) is a methodology reference for *later* checking array-level leaves
don't trail hand-written kernels by too much. *Not lessons for us:* portable
reductions are already covered by `LinearAlgebra.dot` + GPUArrays `mapreduce` (our
⟨Lx,y⟩ and Krylov norms); a Preferences-based "default device" would reintroduce
the global-state tension the per-grid `device::Dev` deliberately avoids.

---

## 6 — Autodiff design

The default path needs **no AD-specific code**: array-level leaves are ordinary
Julia array ops, which Enzyme.jl (reverse + forward, mutation-friendly, GPU) and
Mooncake.jl (reverse + forward, CPU today) differentiate directly — *including*
gradients w.r.t. parameters stored in operator structs (Decision B). This is the
single biggest payoff of Decision A.

**The "adjoint = derivative" shortcut — correctly scoped.** For a *linear*
operator L, the reverse-mode pullback w.r.t. its *input field* is exactly the
adjoint Lᵀ, and the forward pushforward is L itself. We exploit this as an
**optimization**, not the whole story:

- We register custom rules (Enzyme `EnzymeRules.augmented_primal`/`reverse`;
  Mooncake `rrule!!`) for linear leaves that route input-field reverse-mode
  through the already-declared `apply_adjoint!`, avoiding taping through the
  kernel. This is also **mandatory** for `@kernel`-authored leaves.
- **It does NOT cover:** (a) gradients w.r.t. operator *parameters* — those flow
  through ordinary AD; (b) *nonlinear* operators (e.g. self-advection `u·∇u`
  where the velocity is the advected state itself, nonlinear residuals) — those
  need JVP/VJP via AD. Both work automatically on array-level leaves. (Advection
  with a *prescribed* velocity field is linear in its input — a field-valued
  coefficient does not make an operator nonlinear.)

**Backends:** Enzyme is primary (GPU support; best mutation handling). Mooncake
is the CPU-friendly alternative (pure Julia; **no GPU support as of 2026-06**).
Provide both via package extensions; the custom-rule files live in
`ext/…EnzymeExt.jl` and `ext/…MooncakeExt.jl`, loaded only when the AD package is
present, so the **core has zero AD dependencies**. Do not route custom rules
through DifferentiationInterface.jl (it cannot carry custom Enzyme rules); DI may
be offered as a user-facing convenience for generic code.

> **API verified at implementation (2026-07-31).** EnzymeRules
> `augmented_primal(config, func, RT, args...)` → `AugmentedReturn(primal, shadow,
> tape)` and `reverse(config, func, RT, tape, args...)` → one slot per argument;
> `RevConfig{NeedsPrimal,NeedsShadow,Width,Overwritten,RuntimeActivity,StrongZero}`.
> Rules live in `EnzymeCore`, so the rules extension weak-depends on **EnzymeCore**
> (tiny, LLVM-free) and only the AD-*powered* Jacobian needs full `Enzyme`.
>
> Two constraints the design did not anticipate, both learned by hitting them:
>
> 1. **A rule argument may not mix GC-tracked pointers with inline floats.** From
>    Julia 1.12 that is a hard `CallingConventionMismatchError`
>    (EnzymeAD/Enzyme.jl#2707), and a `Field`/`BlockField`/`BlockForest` qualifies
>    through its embedded grid. Rule seams therefore take **raw storage** —
>    `_exchange_storage!`/`_bc_storage!` over a block vector or packed array plus
>    isbits descriptors — which is what `_storage`/`_layout`/`BlockLayout` exist
>    for. This also blocks the planned rules on `apply`/`apply!` and on
>    `mul!(::PreparedOperator)`: `apply` both takes *and returns* a `Field`.
> 2. **A rule body must not allocate.** Rule bodies are compiled into Enzyme's
>    generated code; a `Dict` lookup and a closure in an augmented-primal body
>    segfaulted on Linux x86_64 while running clean on macOS/aarch64.
>
> Corollary: the "custom rules for linear leaves" plan below is **deferred**, not
> abandoned — it is blocked on #2707, not on this package's design.

**Adjoint correctness is testable** and must be tested: the dot-product identity
⟨L x, y⟩ = ⟨x, Lᵀ y⟩ for random x, y, and AD gradients checked against
finite-difference / complex-step on small grids.

---

## 7 — Solver interop

**Krylov.jl + MultiDeviceLinearAlgebra:** Krylov needs only `mul!`, `size`,
`eltype` — which the algebra provides. To run distributed, implement
`mul!(y::MultiDeviceVector, L, x::MultiDeviceVector)` that (1) triggers MDLA's
halo/ghost exchange, (2) applies the leaf per partition with device context, (3)
writes the result partition-local. MDLA already supplies the distributed
primitives we compose with — `PartitionSpec`, `MultiDeviceVector`,
`GhostExchange`, `scatter!`/`reduce!`, and `mdla_solve` (Krylov-backed). All
names confirmed against MDLA v0.0.1 source; `scatter!`/`reduce!(+)` is a
verified exact adjoint pair in MDLA's own tests.

*Built (issue #16, slice 1):* `partition_grid` cuts a `CartesianGrid` into
last-dimension slabs with `Interface` cut faces (core, `src/partitioning.jl`,
adjoint proven CPU-side against emulated exchange semantics), and
`ext/MatrixFreeOperatorsMDLAExt.jl` implements `prepare_distributed` — one
`prepare`d operator per slab per CUDA device, the operator-owned
`GhostExchange` unpacked into `Interface` halo slabs before each local apply,
`reduce!(+)` for the distributed adjoint, and a `Krylov.CgWorkspace` hook.
Scope: scalar fields, `Laplacian`/`IdentityOp`/number-`ScalingOp` under
`Scaled`/`Added`.
CUDA-only by MDLA's nature (one partition per physical GPU — MDLA enforces
unique device IDs, no CPU mode), so distributed tests are env-gated
(`MFO_TEST_MDLA=true`, ≥ 2 GPUs for multi-partition testsets).

*Built (issue #31, slice 2a):* `Composed` and `AdjointOp`, i.e. the cases needing
an exchange **inside** the tree rather than only at its root. The operator tree
is now walked by core (`src/distributed.jl`) — a recursive walk mirroring the
block-forest `_forest_capply!`/`_forest_capply_adjoint!` pair, which solves the
same mid-tree-exchange problem — parameterized on three backend primitives
(`_dist_map!`, `_dist_scatter!`, `_dist_reduce!`). The MDLA extension supplies
them with `scatter!`/`reduce!`; `test/partitioning.jl` supplies them with
plain-`Vector` global indexing, so the CPU proof exercises the real walk rather
than re-emulating it. The distributability guards moved to core with the walk, so
CI runs them. `Derivative` joins the whitelist (same shape as `Laplacian`, and the
only whitelisted leaf that is not self-adjoint, hence the only way to reach an
`AdjointOp` node). Adjoints are normalized down to the leaves before `prepare`,
which does not recurse into an `AdjointOp` and would otherwise skip the mid-tree
reduction for `AdjointOp(A∘B)`.

*Built (issue #31, slice 2b):* `Field` coefficients and distributed `boundary_rhs`
— i.e. the whole solve, including its right-hand side, assembled slab-locally with
nothing global materialized on one device.

A coefficient needs no exchange of its own: `ScalingOp` reads it *pointwise at the
cell being written*, so `_slab_op`/`_slab_field` (core, `src/distributed.jl`) slice
its window onto each slab and nothing else is required. Slicing happens on the
host and the slab is uploaded, so a coefficient the user had already moved to a
device never becomes a cross-device copy. Only field-carrying leaves differ per
partition, so `DistLeaf`/`DistAdjoint` hold a per-partition operator vector that
`identity.` narrows — shared leaves stay concretely typed and statically
dispatched, and the slice-1/2a hot path is untouched. Complex coefficients stay
rejected — the distributed vectors are real, typed from the grid spacing, so a
complex product has nowhere to land and the guard turns an `InexactError` into a
named error — as does `Advection`.

`_dist_boundary_rhs!` is a third walk over the same `DistNode` tree, mirroring
`boundary_rhs`'s recursion. Leaf lifts are slab-local for free — the inhomogeneous
fill is a no-op on `Interface` faces, and that is exact rather than approximate,
since the global lift is zero at a cut plane too (inhomogeneous data lives only in
physical-boundary ghosts). Only a `Composed` lift needs the mid-tree exchange,
which it takes from the node it already belongs to.

Making that assembly *partition-independent* required one core change: a slab now
keeps the **global** `extent` and carries its position in `local_range` alone, and
`cell_center` evaluates at the global cell index. A slab-local origin rounds twice
and drifts by an ulp — invisible to a stencil, which reads only spacing, but enough
to make a coordinate-assembled RHS and therefore the Krylov iteration count depend
on `nparts`. It is bit-for-bit a no-op for undistributed grids and forest leaf
grids, where `first(local_range[d]) == 1`. The user-facing surface is
`boundary_rhs(P)`, `set!(::MultiDeviceVector, P, fun)`, `assemble_rhs(P, f)`, and
`local_grids(P)`. Only `prepare_distributed` carries a `distributed` qualifier,
because only it shadows a single-device function; everything downstream dispatches
on `P` and is named for what it computes, not for where it runs.

*Built (issue #57, slice 2c):* the compact flux-form `Diffusion` leaf on slabs —
the first whitelisted operator whose coefficient is read at a *neighbour* rather
than pointwise, so an interior-only slice with zeroed ghosts would not serve it.

It still costs no communication, and the reason is worth stating because it is
what makes stage 2 of #56 the cheap stage. `_slab_op` runs on the host holding
the **global** κ, whose ghosts `diffusion` already extended by an even mirror and
a periodic wrap at construction. Slab padded index `p` is global padded index
`first(local_range[d]) - 1 + p`, so one *padded* window — which is what
`_slab_field` now slices for every coefficient — lands every ghost on the value
it should hold with no per-face logic: an `Interface` ghost onto a global interior plane (the
neighbour's κ), a wall ghost onto the global mirror, a periodic cut onto the
global wrap. Widening the slice is a setup-time indexing change, not a transport,
so the per-apply exchange count is unchanged from the `Laplacian` baseline — the
"one halo exchange per application" invariant holds trivially rather than by
construction. Two preconditions make that legitimate rather than lucky. κ is
*constant through a solve*, so the setup-time slice is never owed again per
iteration. And the process running `_slab_op` holds the **global** κ —
`prepare_distributed` adapts the whole tree to the host before slicing — which is
a property of a single-node backend, not of the design; see *Owed by the next
backend* below.

There is one slice, not a pointwise one and a stencil one: `ScalingOp` takes the
same padded window, and because it reads its coefficient pointwise the ghosts it
now carries are inert (a CPU testset poisons them with `NaN`). The real-eltype
restriction in `_partitionable_coeff` carries over unchanged, for the same reason
— the distributed vectors are real. The leaf's mechanical transpose — the
`adjoint_gather!` branch `apply_adjoint!` takes on a slab's `Interface` faces,
scattering cotangents into ghosts for the slab reduction to fold — is reached only
through the test-facing `_mul_adjoint!`. A Krylov `mul!` never sees it:
`_push_adjoints` folds `adjoint(Diffusion)` to the conjugated leaf at setup, which
for real κ is the leaf itself, running forward.

*Deferred:* rank-changing intermediates (`Divergence ∘ Gradient` needs its own
`ncomp = N` spec and ghost layout — `_slab_ghost_layout` and `_owned_flat_range`
already take the kwarg), transfer chains (factors on two grids, each needing a
consistent cut), and distributed autodiff (blocked by the transport, not by the
localization rewrite — `_slab_field`'s pullback is the transpose gather). All
rejected loudly.

*Owed by the next backend:* the setup-time κ exchange. On ImplicitGlobalGrid
(§10.4's resolved next backend, #46) no rank holds the global κ, so `_slab_field`
has nothing to window and the cut-plane coefficient ghosts must be *exchanged*,
once, at prepare time. The seams already fit: build each rank's leaf through the
inner constructor from a rank-local κ whose physical ghosts
`fill_coefficient_ghosts!` has mirrored — it leaves `Interface` ghosts untouched —
then run one `_dist_scatter!`-shaped exchange over those `Interface` planes at
setup, and the per-apply exchange count stays at the `Laplacian` baseline.
Reusing `_slab_op(::Diffusion)` per rank with a rank-local κ would not do: it
throws on an out-of-range `view`, or with matching sizes silently leaves the
cut-plane ghosts at zero or the mirror — `κ_I/2` face coefficients under
`ArithmeticMean`, `0` under `HarmonicMean` — and no guard here would catch it.

**OrdinaryDiffEq.jl:** you do **not** need SciMLOperators to use it.
- *Explicit* solvers (RK4, SSPRK, …): provide a trivial RHS adapter
  `f!(du,u,p,t) = apply!(du, L, u, grid)` and hand it to `ODEProblem`.
- *Implicit/stiff* solvers (needed for diffusion): the idiomatic way to give
  OrdinaryDiffEq a **matrix-free Jacobian** is `ODEFunction(f; jac_prototype = J)`
  where `J` is a SciML-style lazy operator. Here `J` is the linear
  `linearize(F, u₀)` operator from §10.8 — its `mul!` is a central finite-difference
  JVP by default, or the forward-mode-AD JVP under `EnzymeJVP()`, so the matrix-free
  Jacobian comes from the AD layer rather than by hand. We ship a
  **thin optional SciMLOperators adapter** (a `FunctionOperator`-style wrapper
  exposing that `mul!`) *only* for this one hook — in `ext/…SciMLExt.jl`, not in
  the core, and not adopting the `(u,p,t)` model anywhere else.

---

## 8 — v1 scope (build this first — keep it tight)

1. `CartesianGrid{N}` — uniform, single device, collocated, with periodic +
   Dirichlet + Neumann BCs; `halo_update!` present as a no-op.
2. `Field{Center}` over device arrays.
3. Operator algebra: leaves `Laplacian`, `Derivative`, `Gradient`, `Divergence`, `ScalingOp`, `IdentityOp`, `Advection`; combinators `Added`, `Composed`, `Scaled`, `AdjointOp`; traits; **declared adjoints per leaf**; `apply!` + `apply_adjoint!`; `mul!`/`size`/`eltype`; `Adapt` support. `ScalingOp` (pointwise ×κ(x)) is the leaf that makes Decision B concrete: it carries a differentiable coefficient *field*, it is the genuine `isdiagonal` / `isselfadjoint` instance (Jacobi smoother target), and variable-coefficient diffusion falls out of the algebra as `Divergence ∘ ScalingOp(κ) ∘ Gradient` — a built-in composition stress-test. (Caveat: the composed form has a wider effective stencil and collocated odd-even quirks. **Resolved (2026-08-14): the fused `∇·(κ∇u)` leaf landed as `Diffusion`/`diffusion(g, κ)`** (issue #48), following the §4a custom-fused-leaf pattern with the exported `diffusion_stencil` primitive. It is the compact flux form — face-averaged κ, arithmetic or harmonic — and it is exactly symmetric for real κ, declares `operator_diagonal` (which the composed form cannot), and couples adjacent solution cells, removing odd–even decoupling in u. Coefficient identifiability remains separate: arithmetic averaging cancels a checkerboard perturbation of κ at every interior face, leaving the entire operator unchanged on periodic grids with even cell counts in every dimension or under homogeneous Neumann walls. Additional excitations cannot distinguish those coefficients; Dirichlet wall coefficients can break the ambiguity. The composition stays valid and stays the algebra stress-test; the leaf takes 115 µs against the composition's 337 µs for one 256² prepared `mul!`, and 1.7× the `Laplacian`'s 67.5 µs for 2× the memory traffic. `CartesianGrid` only for now — `BlockForest` and distributed slabs are staged in #54.)
4. Array-level authoring; device-agnostic via `get_backend`/`Adapt`; CI on CPU,
   and CUDA where available.
5. AD: works automatically (Enzyme + Mooncake) on array-level leaves for field +
   parameter gradients; optional adjoint-routing rules for linear leaves in
   extensions.
6. Solver interop: Krylov `mul!` path; OrdinaryDiffEq explicit RHS adapter.
7. Tests per operator: action vs analytic solution; adjoint via ⟨Lx,y⟩=⟨x,Lᵀy⟩;
   AD vs finite differences; CPU/GPU parity.

**Module / file layout (proposed):**

```
src/
  Grids.jl            # AbstractGrid, CartesianGrid, halo_update! (no-op), interior, BCs
  Fields.jl           # Field, location traits
  operators/
    abstract.jl       # AbstractOperator, traits, algebra (+,*,scale), adjoint propagation
    algebra.jl        # Added / Composed / Scaled / AdjointOp + their apply!/adjoint
    laplacian.jl      # leaf: apply!, apply_adjoint!, adjoint, traits
    derivative.jl     # leaf …
    gradient.jl       # leaf …
    divergence.jl     # leaf …
    scaling.jl        # leaf: pointwise ×κ(x); diagonal; the parameter-field leaf
    advection.jl      # nonlinear-capable leaf
  linalg.jl           # mul!/size/eltype, Krylov compatibility
  boundaries.jl       # BC representation + apply_bc! (see §10)
ext/
  <Pkg>EnzymeExt.jl   # custom adjoint-routing rules (linear leaves, kernel leaves)
  <Pkg>MooncakeExt.jl
  <Pkg>MDLAExt.jl     # mul!(::MultiDeviceVector, L, ::MultiDeviceVector) + halo via scatter!
  <Pkg>SciMLExt.jl    # thin jac_prototype adapter for implicit OrdinaryDiffEq
```

**Core dependencies (minimal):** `KernelAbstractions`, `Adapt`, `LinearAlgebra`,
`StaticArrays` (small per-cell tuples). Everything else — Enzyme, Mooncake, MDLA,
SciMLOperators, Reactant, OrdinaryDiffEq — is a **weakdep behind a package
extension**, keeping the core light and dependency-honest.

---

## 9 — Forward-looking seams (design now, implement later)

Each is one paragraph because the value of v1 is that these are *cheap*, not
pre-built.

- **Distributed memory.** *Correction to the original hope:* "just use
  DistributedArrays.jl and it works" is **not realistic** — DistributedArrays.jl
  has weak/absent halo support for stencils (long-standing, unresolved) and no
  GPU story. The seam is `halo_update!(field, grid)` plus the grid carrying
  local-vs-global index ranges and neighbor topology. Operators are written once
  against it. Concrete backends, in priority order: **(1) compose with your MDLA
  `GhostExchange`/`scatter!`** for multi-GPU (it is the closest existing model
  and it's yours); **(2) MPI + halo** via ImplicitGlobalGrid.jl for multi-node
  CPU+GPU; **(3) Reactant XLA sharding** for automatic
  partitioning. The seam supports all three; we do not hard-depend on any
  distributed-array type. (Pick-order **resolved** in §10.4: MDLA done, IGG is
  the multi-node backend, uniform `CartesianGrid` only.) *Not free:* an
  explicit `halo_update!` (MPI sends, `CUDA.device!` context switches) is **not**
  Reactant-traceable, while the Reactant sharding path wants the halo **implicit**
  (XLA inserts the communication during compilation). These are therefore
  *alternative execution modes* of the same leaf — chosen per backend — not one
  `apply!` body serving both at once.

- **Adaptive grids (AMR) — built end-to-end, including the packed-storage GPU
  phase (#15).** Decided against a
  cell-based octree (p4est/Trixi-style hanging nodes everywhere break the
  dense-array leaf stencils). Instead: a **forest of fixed-size leaf-blocks**
  (FLASH/PARAMESH/AMReX style). The domain is a forest of `2ᴺ`-trees of equal-cell
  blocks; refining a block replaces it with `2ᴺ` half-spacing children. Each leaf is
  an ordinary `CartesianGrid`, so every leaf operator runs **unchanged per block**;
  all adaptivity is confined to block faces. The load-bearing invariant is **2:1
  balance** (adjacent leaves within one level), which reduces every coarse–fine
  interface to three cases. The topology is a pure-Julia serial `Forest` (Morton
  keys, `refine!`/`coarsen!`/`balance!`, neighbor queries — no new deps);
  distributed/`P4estTopology` backends sit behind the `topology` + `halo_update!`
  seams. Storage is **vector-of-blocks** first (`BlockField`), with a packed twin
  (**`PackedBlockField`**: one contiguous `(blocksize .+ 2halo ..., nleaves)` buffer
  behind a shared `AbstractBlockField` interface, selected by dispatch via explicit
  `pack`/`unpack` — no flags, no implicit switching) as the GPU phase (#15). The
  original "packed drops in later without touching operators" claim splits into a
  two-level truth: packed *storage* is a drop-in — per-leaf trailing-dim views feed
  the unchanged descriptor/BC/flat machinery and the per-leaf fallback sweep — while
  the single-launch *sweep* is a per-operator forest-native kernel that reuses the
  factored per-cell stencil functions (§5, forest-native kernels).
  - **Built (Phase 0/1, reworked by #14):** `BlockForest` + topology + `BlockField`
    + the flat boundary; the forest action = `halo_update!` (inter-block exchange,
    once over the forest) → forest-level `apply_bc!` face pass (physical domain
    faces, from per-(dim, side) leaf-index lists on the per-generation schedule) →
    per-leaf stencil. Every leaf grid is the *same* concrete all-`Interface` type
    (`Interface` faces are skipped by the per-leaf BC sweeps), so `leaf_grid` is
    type-stable and isbits and the PR #8 BC-signature group cache is deleted (#14).
    The adjoint is Hᵀ∘Bᵀ∘Sᵀ: per-leaf stencil-transpose gathers, forest-level
    `fold_bc!`, then `halo_update_adjoint!` last. A BC type without a face-pass
    implementation is rejected at `BlockForest` construction — a missing capability
    degrades to an error, never a wrong result. Verified by
    bit-parity vs an equal-resolution single `CartesianGrid` and the adjoint identity
    (the same-level halo copy has a declared transpose `halo_update_adjoint!`,
    mirroring `apply_bc!`/`fold_bc!`). The exchange is driven by a per-generation
    **`ExchangeSchedule`** — flat homogeneous vectors of copy descriptors
    (`src`/`dst` block + slab ranges) plus the physical-face lists, built once per
    regrid generation and cached on the `BlockForest`, so `halo_update!` and the BC
    passes are dumb, type-stable loops of concrete
    `view .=` broadcasts with no per-application topology queries; the adjoint runs
    the same descriptors transposed (scatter-add then zero). The descriptor list is
    also the send/recv list a future distributed backend consumes.
  - **Built (coarse–fine phase):** operators apply across refinement levels. The
    schedule carries two more homogeneous descriptor vectors: coarse→fine ghost
    interpolation (Martin–Cartwright quadratic — normal parabola (5/21, 5/6, −1/14)
    through the fine block's first interior cell and two coarse layers, tensor-product
    3-point tangential quadratics at ξ = ±1/4, one-sided at coarse tangential extremes
    so fills read only block interiors) and fine→coarse flux-matching restriction
    (coarse ghost set so the coarse face flux equals the mean fine flux; a plain 2^N
    volume average leaves O(1) interface truncation ⇒ 1st-order solutions, rejected).
    Forward sweep: copies → interp → restrict (restriction reads interp-filled fine
    ghosts); the adjoint runs phases and descriptors in exact reverse order, making it
    the exact transpose by construction. Quadratic exactness of every weight set is
    unit-tested; the acceptance norm is the **volume-weighted L1 of the action**
    (pointwise interface action error is O(h) by design for this scheme family —
    Chombo/AMReX behavior; solutions and L1-action converge at 2nd order).
    **Self-adjointness is grid-aware:** coarse–fine coupling is nonsymmetric, so
    `isselfadjoint(laplacian(bf))` is `false` on a non-uniform forest (queried live)
    — the adjoint folds/shortcuts degrade to the declared transpose, never a wrong
    result. Requires halo width 1 and even blocksize ≥ 4 per dim (validated at
    schedule build; uniform forests keep the looser v1 constraints). `Composed`
    works on forests: its intermediate is a whole `BlockField` and every combinator
    (apply, prepared apply, adjoint, `boundary_rhs`) recurses at the forest level,
    so the intermediate gets its inter-block exchange — the per-leaf path would
    silently miss the cross-block coupling.
  - **Built (AMR driver, #13):** a single atomic **`regrid!(u, more...; refine,
    coarsen)`** — per-block criteria evaluated on a live field (each criterion
    receives the leaf as a `Field`; per-block marking is the native granularity),
    one combined topology pass (refine-marked leaves split, complete fully-marked
    sibling families coarsen with completeness judged on the leaf set the criteria
    saw, refine wins conflicts), a single `balance!`, then freshly allocated
    fields returned with the solution transferred. Marking, topology edit, and
    transfer are one indivisible transaction because a regrid instantly stales
    every old field behind the generation guard. Varargs is load-bearing, not
    convenience: any field needed after the regrid (coefficients, a precomputed
    indicator) must ride the same call. **Transfer is interior-only and keyed by
    `LeafKey`** (integer leaf indices are re-sorted every regrid): same key →
    copy; refined leaf → per-dim linear interpolation from the old parent
    ((3/4, 1/4), flipping to one-sided (5/4, −1/4) at parent-block edges so no
    ghost is ever read — old ghosts are not guaranteed valid, and no BC
    composition yields full-value inhomogeneous ghosts); coarsened leaf →
    conservative `2⁻ᴺ` child mean. A single pass cannot move any region by more
    than one level (marks are evaluated on a 2:1-balanced set; the balance
    cascade is bounded) — the transfer guards that invariant with an error, never
    a silent fallback. A no-op regrid (marks change nothing) returns the *input*
    fields unchanged and does not bump the generation, so prepared operators
    survive — the convergence signal for adaptive loops. The adaptive solve loop
    itself (solve → indicator → `regrid!` → re-`prepare`) is deliberately **user
    code** (`examples/adaptive_poisson.jl` is the canonical form; solver must be
    a nonsymmetric Krylov method on an adapted forest) — an `adaptive_solve`
    export would guess the interface from one use case (rule of three).
  - **Built (packed phase, #15):** staged like #10 — (1) `AbstractBlockField` +
    `PackedBlockField` + `pack`/`unpack` with the Laplacian forest kernel
    end-to-end, (2) the remaining operator kernels + adjoint
    transpose-gather kernels + the coefficient-field layout policy,
    (3) kernelized exchange/BC/flat passes (incl. `GhostFill.terms` CSR
    flattening; adjoint exchange deliberately stays on the host
    descriptor loops — see §5), (4) packed regrid — resolved as the documented
    re-pack contract, not code: `regrid!` of a packed field errors — regrid the
    reference field and re-`pack` (a missing capability degrades to an error,
    never a wrong result; regridding is host-side leaf surgery, so an automatic
    round-trip would only hide the device transfer). The per-leaf fallback
    guarantee held throughout: every operator works on packed storage from day
    one via `_forest_sweep!`'s reference loop. Closing metrics (first real-CUDA
    session, RTX 4000 Ada): the #7 residual is closed — prepared packed `mul!`
    host allocations hold at ~11 KB of CUDA launch bookkeeping from 256 to 4096
    leaves, no per-leaf term (the historical ~384 B/leaf per-leaf residual had
    already vanished with the all-Interface leaves, #14/#23, which let the
    per-leaf `apply!` fully specialize — per-leaf prepared `mul!` measures 0 B
    on CPU); the packed single-launch sweep runs 30–95× faster than the
    per-leaf device path at equal DOFs (60× at 4 M DOFs uniform, 95× refined),
    and the batched device exchange runs 230–1370× faster than the
    per-descriptor loop on the same device data (`benchmark/gpu.jl`). An nsys
    trace shows a constant 5 kernel launches per prepared `mul!` — packed
    sweep, two exchange copies, flat in/out broadcasts — identical at 64 and
    4096 leaves. The
    unsynchronized launch chain (same-task-stream FIFO, §5) was probed with 100
    unsynced `mul!` parity checks and adapted-forest Krylov solves on device —
    no explicit `synchronize` is needed.
  - **Extending to other tree structures (deliberately not abstracted yet).** There is
    no pluggable "swap the tree structure" interface, and that is the design, not an
    omission. A *fundamentally different* AMR (cell-octree, patch-based) would enter as a
    new `AbstractGrid{N}` subtype sibling to `BlockForest` — operators are insulated by
    the grid interface + `halo_update!` seam, so they would not change. `Forest`/`LeafKey`
    are intentionally left as concrete types: an `AbstractForest`/`AbstractTopology`
    abstraction is **deferred until a second topology backend (`P4estTopology`/distributed)
    actually exists** — abstracting from one implementation guesses the interface wrong
    (rule of three). Do not introduce it speculatively.

- **Geometric multigrid — built (uniform grids, v1 scope: issue #11).**
  Restriction R and prolongation P are **themselves operators** in the same algebra
  (`src/operators/prolongation.jl`/`restriction.jl` — the rank-changer template's
  first two-grid leaves: `operator_grid` is the *input* grid, `size` is rectangular).
  P is per-dim **linear** interpolation on the cell-centered 2:1 pair (3/4 parent,
  1/4 neighbor, boundary children read the coarse homogeneous ghost fill); R **is
  defined as** `2⁻ᴺ·Pᵀ` (full weighting) sharing one kernel pair, so the dot-product
  identity is exact by construction and `adjoint_operator` declares the partners.
  A deliberate deviation from the earlier plan to reuse the AMR `_lagrange3`
  quadratic weights: those kernels are interface-only ghost fills, and linear P +
  full-weighting R is the textbook cell-centered pair satisfying the transfer-order
  rule for 2nd-order PDEs — the quadratic weights remain the upgrade path.
  Coarse operators are **rediscretizations** on `coarsen(g)` via a tree walk
  (`ScalingOp` coefficient fields child-averaged, not restricted — R's Dirichlet
  fold would corrupt a material coefficient at walls). Note: cell-centered Galerkin
  `R·A·P` ≠ rediscretization even in the periodic interior (both O(h²)-consistent;
  tests assert structure + action agreement, never equality). Smoothers: weighted
  Jacobi and fixed-coefficient Chebyshev (power-iteration bounds at setup), both on
  the new `operator_diagonal` (exact incl. BC diagonal contributions
  `(-2 + bc_sign)·h⁻²`; `Number` when uniform, `Field` otherwise; no fallback —
  missing declarations error). The V-cycle (`src/multigrid.jl`) is symmetric
  (ν₁ = ν₂, R = c·Pᵀ, exact dense-LU coarsest solve) so
  `MultigridPreconditioner` is a fixed SPD linear map feeding `Krylov.cg` via
  `mul!`; `MultigridSolver`/`solve` wrap it as a stationary iteration.
  Deferred: `:W`/`:F` cycles, forest MG (the AMR hierarchy is the natural level
  structure), `Advection` rediscretization, GPU coarsest solve. References mined:
  GeometricMultigrid.jl, RestrictProlong.jl.

---

## 10 — Open decisions to flag (your call)

Presented one at a time with a recommendation; none block writing v1's spine.

1. **Package name.** Working title only. Candidates: `MatrixFreeOperators.jl`,
   `StencilOperators.jl`, `GridOperators.jl`, `MeshFreeOperators.jl`.
   *Recommendation:* `MatrixFreeOperators.jl` (says exactly what it is).

2. **Collocated vs staggered fields in v1.** Staggered (faces/centers) avoids
   odd-even decoupling for incompressible flow but adds bookkeeping.
   *Recommendation:* collocated v1; encode location as the `Field` trait `L` so
   staggered is purely additive. (Note, 2026-08-14: for *diffusion* the odd-even
   decoupling is now handled without staggering, by the compact flux-form
   `Diffusion` leaf — §8. A warning for whoever does implement staggered `G`/`D`:
   the compact operator factors as `L = −Bᵀ diag(κ_f) B`, but at a Dirichlet wall
   the row of `B` is `√2/Δ`, not `2/Δ` — the geometric mean of the one-sided
   gradient weight and the divergence weight. There the gradient and
   negative-divergence operators are *not* transposes of each other; only their
   product is symmetric. A staggered pair that assumes `D = −Gᵀ` will be
   asymmetric at Dirichlet walls.)

3. ~~**AD backend default.**~~ **Resolved (2026-07-31): Enzyme.** It is the
   documented default and preferred backend, and the only one the custom rules in
   `ext/…EnzymeCoreExt.jl` apply to. Mooncake stays a tested CPU-only cross-check:
   because EnzymeRules are invisible to it, it tapes through everything and is
   therefore a genuinely independent oracle rather than a second view of the same
   machinery. Mooncake still has no GPU support, and no Mooncake rules are
   written. DifferentiationInterface.jl is the recommended *frontend* — never a
   dependency, and never a route for rules (§6).

4. ~~**Distributed target to implement *first*.**~~ **Resolved (2026-08-03):
   MDLA first (done), ImplicitGlobalGrid.jl for multi-node next.** MDLA is
   implemented in `ext/MatrixFreeOperatorsMDLAExt.jl` behind the
   `_dist_scatter!`/`_dist_reduce!` seam (§9) and covers single-node multi-GPU,
   adjoint included. **The multi-node backend is ImplicitGlobalGrid.jl**, added
   as a further extension behind that same seam — not MPIHaloArrays, and not a
   dependency of the core. Reactant sharding stays opportunistic.

   *Why IGG:* it is production-proven GPU-aware MPI halo exchange over a
   Cartesian MPI topology for exactly our uniform `CartesianGrid` case,
   multi-node and CPU+GPU — the one capability MDLA structurally cannot provide
   (single node, CUDA-only). We take **its halo only**, underneath our seam.

   *Why the split is at "one node", precisely.* The two exchanges differ on both
   topology and decomposition, in opposite directions. MDLA's `GhostExchange` is
   **more** general in topology — an index-list SpMV halo built by
   `_compute_ghost_map` from a sparse matrix's column structure, so neighbors are
   whoever the sparsity says — while IGG only ever does structured Cartesian face
   exchange. But MDLA's `PartitionSpec` is contiguous ranges over a **flat** index
   space, so on a lexicographically flattened 3D grid it can only cut **slabs**
   along the slowest axis, whereas IGG does true 3D blocking. That is what bounds
   MDLA to one node: for `P` devices on an `n³` grid, slab halo traffic scales as
   `2Pn²` against `6n²P^(1/3)` for 3D blocks — a ratio of `P^(2/3)/3`, so ≈1.3× at
   `P=8` (noise), ≈5× at `P=64`, ≈33× at `P=1000`. Slabs also cap the device count
   at `nz` and want each slab thicker than the halo. Right for a node's worth of
   GPUs, hopeless past it. Note also that MDLA has **no MPI dependency at all**
   (CUDA/Krylov/LinearAlgebra/SparseArrays) — it is single-process by design, not
   MPI-made-easy, so growing it multi-node is the same work as adopting IGG.
   Corollary: if an *irregular* (non-Cartesian) distributed coupling is ever
   needed, MDLA's index-list model is the starting point, not IGG's.

   *Why not ParallelStencil (evaluated, not adopted).* PS is the kernel-authoring
   DSL that usually ships alongside IGG, and it is a poor fit here on three
   counts. `@init_parallel_stencil(backend, precision, ndims)` is module-level
   global state fixed at load, which is incompatible with keeping GPU support in
   a weak-dep extension. It offers no operator algebra, no linear-solve path (the
   ecosystem's scaling comes from pseudo-transient relaxation, which does not
   apply to an IVP where the transient is the answer), and no AMR. Our hot-stencil
   escape hatch is KernelAbstractions `@kernel` (§1.A), which has none of those
   problems. Adopting IGG's halo does **not** pull in PS.

   *Calibrate the expected payoff.* For the driving workload — reaction-diffusion
   where a large local ODE dominates each node (cardiac monodomain: ~65 states,
   ~255 `exp` per node per step) — only the scalar field crosses a boundary while
   the full state churns over the volume. For a 256³ subdomain that is roughly a
   2800:1 byte ratio, so even a naive blocking exchange costs low single-digit
   percent of a step. IGG is chosen for **correctness and multi-node CPU+GPU
   portability, not because the halo is a bottleneck.** Do not pre-invest in
   `@hide_communication`-style overlap; measure first.

   *Scope boundary.* IGG assumes one fixed uniform Cartesian decomposition, so it
   serves the uniform `CartesianGrid` multi-node case only. **Multi-node AMR
   (`BlockForest`) stays open** — the forest's own `halo_update!` is intra-forest
   and a distributed forest needs a different answer. See issue #46.

5. **How operators carry the grid.** Bind the grid into the leaf at construction
   (RBF-style: `laplacian(grid; …)` returns a bound operator) vs pass grid to
   `apply!` each call. *Recommendation:* bind at construction for ergonomics, but
   keep `apply!(y,L,x,grid)` taking explicit args so it stays AD- and
   Reactant-trace-friendly (closures over big structs hurt both). Slight
   redundancy; flagging it.

6. **Boundary-condition representation.** As grid metadata, as part of each
   operator, or as separate BC operators? This genuinely affects **adjoint
   correctness** (BCs are where self-adjointness breaks). *Recommendation:* model
   BCs as grid-attached data that operators consult in `apply_bc!`, and derive
   each operator's adjoint *including* its boundary contribution. RBF's Hermite
   boundary mechanism is a reference. This is the subtlest correctness area —
   worth a focused design pass before coding the leaves.

   **Hard requirement, independent of the representation chosen — the
   linear/affine split.** Inhomogeneous Dirichlet/Neumann data baked into the
   operator action makes it *affine*: `L(x) = A·x + b` with `b ≠ 0`, so
   `L(0) ≠ 0`. That breaks Krylov (which assumes a linear map — the correct
   formulation is `A·u = f − b` with `A` the homogeneous-BC part), falsifies the
   adjoint identity ⟨Lx,y⟩=⟨x,Lᵀy⟩ (verification item 2 silently depends on this
   split), and turns `islinear(L) = true` into false advertising. So:
   `apply!`/`apply_bc!` enforce **homogeneous** constraints only; inhomogeneous
   boundary data is exported separately as a lift vector, e.g.
   `boundary_rhs(L, g)`, assembled once per solve and folded into the RHS.

7. **Buffer / scratch model (the real cost of dropping `cache_operator`).**
   Matrix-free composition is not allocation-free: `Composed(A,B)` needs a
   temporary `tmp = B*x` before `A*tmp`, and Krylov's 5-arg
   `mul!(y,L,x,α,β) = α(L·x)+βy` needs a temporary for composed operators.
   Calling `similar(x)` per `mul!` inside a Krylov inner loop on GPU is exactly
   the allocation pattern that defeats the efficiency requirement (CUDA's pool
   softens but does not erase it; other backends vary). SciMLOperators'
   `cache_operator` ceremony exists precisely to solve this — we dropped the
   ceremony but inherited the problem, so we owe an explicit decision.
   *Recommendation — a two-path model (no `(u,p,t)` baggage):*
   - **Allocating pure path** for AD + Reactant: `*` / out-of-place `apply`
     allocate fresh outputs and fresh composed temporaries. AD *wants* this
     (Enzyme/Mooncake tape it; Reactant fuses the temporaries away entirely),
     and these paths are not in a hand-tuned hot loop.
   - **Prepared in-place path** for Krylov hot loops: a `prepare(L, x)` walks the
     operator tree once, allocates exactly the intermediate buffers each
     `Composed` node needs, and stores them in the prepared operator's fields;
     thereafter in-place `mul!` is **zero-allocation** in steady state. This
     operator is stateful/single-thread — acceptable, because Krylov solves are
     never differentiated *through* (sensitivities come from
     implicit-function-theorem adjoints on the *solution*, not the iteration).
     For concurrent solves or MDLA partitions, `prepare` per thread/partition
     (just buffer allocation). Leaf 5-arg `mul!` fuses `α·stencil(x)+β·y` into a
     single broadcast (one kernel launch, no temp).
   - **Interior-only vector space (the Krylov boundary).** `apply!` operates on
     halo-padded `Field`s; `mul!` is the flat boundary, and the flat vector spans
     **interior DOFs only** (ghosts are never solver unknowns), so
     `size(L) = (n_int·ncomp, n_int·ncomp)`. `prepare` therefore also owns one
     halo-padded scratch field per operator tree: `mul!` copies the flat vector
     in, runs `halo_update!` + the stencil on the scratch, and copies the
     interior back out. The copies are cheap (fusable with the α/β axpby) and
     keep `mul!`'s contract intact — `halo_update!` mutates the *scratch*, never
     Krylov's `x` (which Enzyme would otherwise observe as an input mutation). *Caveat to note in the doc:*
     composed matrix-free operators incur one kernel launch per leaf; whole-tree
     fusion is exactly what Reactant buys you — a point in its favor for
     composition-heavy operators. The other escapes from per-leaf launches are to
     hand-author a **custom fused leaf** (§4a) for the hot operator — the manual
     counterpart to Reactant's automatic tree fusion — or, on forests, the
     packed-storage single-launch sweep (#15), which replaces the per-leaf loop
     with one forest-native kernel (§5).

8. **Linear vs. nonlinear operators, and where the Krylov Jacobian comes from.**
   `adjoint(L)` and "use `L` as a linear map in Krylov" are meaningful **only for
   linear operators**; nothing should let `adjoint(Composed(nonlinear, linear))`
   return a silently-wrong result. The user wants nonlinear residuals + implicit
   solvers, which is the JFNK pattern: the operator handed to Krylov is the
   **Jacobian** `J = ∂F/∂u` linearized at a state `u₀` (a *linear* matrix-free
   operator), **not** the nonlinear `F`. *Recommendation — make the contract
   explicit in the type system:*
   - Nonlinear operators support `apply!` + AD but **not** `adjoint` / linear
     `mul!`; the `islinear` trait gates which methods are valid (error, don't
     silently misbehave, when an adjoint/linear-map is requested of a nonlinear
     op).
   - Provide `linearize(F, u₀)` (a.k.a. `jacobian(F, u₀)`) returning a **linear**
     operator whose `mul!(Jv, J, v)` is the JVP `∂/∂ε F(u₀+εv)|₀`. With Decision
     A (array-level leaves) this JVP is **free via forward-mode AD** — the
     matrix-free Jacobian falls straight out of the autodiff layer. (A
     finite-difference JVP is the classic fallback.) That linear `J` is what feeds
     Krylov and the `jac_prototype` hook in §7.

     *As built (2026-07-31):* the backend is an explicit argument, not ambient.
     `FiniteDifferenceJVP()` is the default — central-difference, two operator
     applications per product, ~`√eps` accurate, and **no transpose** (its
     `adjoint` throws). `EnzymeJVP()` is the preferred choice where Enzyme is
     available: exact, one application per product, and it supplies a real
     reverse-mode VJP so `adjoint(J)` works and transpose-needing Krylov methods
     can run against a JFNK Jacobian. Explicit rather than ambient because the two
     differ in *numbers* as well as capability, and that must not depend on which
     packages happen to be loaded.

9. **Vector-field memory layout.** Given the §Core-abstraction-1 decision that a
   field's element type carries its rank, what is the *concrete default layout* for
   a vector field: array-of-`SVector` (`Array{SVector{N,T}}`, matching the
   StaticArrays core dep) or planar (`Array{T}` with a trailing component
   dimension)? The driver is **Reactant/XLA**: it is not certain XLA traces
   `SVector`-element arrays cleanly (req 1 / Decision A lean hard on
   Reactant-traceability), and it may prefer the explicit component dimension.
   *Recommendation:* AoS `SVector` as the default — the operator-level genericity
   makes the *algebra* layout-agnostic, so this is purely a representation /
   Reactant concern, not an algebra one. **Verify `SVector`-array Reactant tracing
   at implementation;** if it doesn't trace cleanly, a planar layout (or a
   Reactant-only field adapter) is the fallback, with operators unchanged.

---

## 11 — Files to reuse / reference

**User's packages (mirror style / compose with — do NOT subtype):**

- RadialBasisFunctions.jl — mirror the lazy operator algebra + trait style:
  - `~/dev/RadialBasisFunctions/src/operators/operators.jl`
  - `~/dev/RadialBasisFunctions/src/operators/operator_algebra.jl`
  - `~/dev/RadialBasisFunctions/src/operators/operator_traits.jl`
  - *Why not subtype its `AbstractOperator{N}` or reuse `RadialBasisOperator`:*
    RBF is matrix-based (precomputed sparse weights), scattered-data, k-NN. Its
    algebra combines *sparse matrices*; a matrix-free op cannot join that path.
    If tight RBF cross-composition ever becomes a goal, factor a tiny shared
    *lazy-algebra interface*, don't inherit the matrix-based tree.
- MultiDeviceLinearAlgebra.jl — the distributed model we compose with (NOT
  DistributedArrays.jl):
  - `~/dev/MultiDeviceLinearAlgebra/src/partition.jl` (`PartitionSpec`)
  - `~/dev/MultiDeviceLinearAlgebra/src/vector.jl` (`MultiDeviceVector`)
  - `~/dev/MultiDeviceLinearAlgebra/src/ghost.jl` (`GhostExchange`, `scatter!`, `reduce!`)
  - `~/dev/MultiDeviceLinearAlgebra/src/matrix.jl`, `mul.jl`, `krylov_compat.jl`

**External references (learn from):** Oceananigans.jl (operator composability +
KA + staggered grids + multi-arch — the closest architectural sibling);
ParallelStencil.jl + ImplicitGlobalGrid.jl (halo exchange + `@hide_communication`
overlap to learn from); GeometricMultigrid.jl + RestrictProlong.jl (multigrid);
Trixi.jl / P4est.jl / T8code.jl (AMR); SciMLOperators.jl (adapter target only).

---

## 12 — Verification plan

End-to-end checks to run as v1 lands (do not run until implementation is
underway, per testing convention):

1. **Operator action** — Laplacian / gradient / divergence against analytic
   fields (e.g. Δ(sin kx) = −k² sin kx) on a refined sequence; check convergence
   order.
2. **Adjoint identity** — for random x, y on random grids: `⟨L*x, y⟩ ≈ ⟨x, L'*y⟩`
   to tolerance, for every leaf and for composed/added/scaled operators.
3. **Composition** — `(A*B)*x ≈ A*(B*x)`, `(A+B)*x ≈ A*x + B*x`, `(αA)*x ≈ α*(A*x)`,
   and `adjoint(A*B) == adjoint(B)*adjoint(A)` behaviorally.
4. **Autodiff** — Enzyme & Mooncake gradients of a scalar loss `sum(L*x)` w.r.t.
   **x and w.r.t. operator parameters** (e.g. a diffusion coefficient field),
   checked against finite differences / complex-step.
5. **Device parity** — identical results (to fp tolerance) for CPU vs CUDA arrays
   on the same problem; confirm no scalar-indexing fallbacks on GPU.
6. **Solver interop** — solve a Poisson problem with `Krylov.cg` using the
   operator's `mul!`; integrate a heat equation with OrdinaryDiffEq via the RHS
   adapter; compare to a known solution.
7. **(Seam smoke test)** — `halo_update!` no-op path returns unchanged field;
   stub a 2-partition MDLA `mul!` and verify it matches the single-device result.
   *Built out beyond the stub:* `test/partitioning.jl` proves 2-/3-partition
   forward parity (bitwise) and the distributed adjoint identity CPU-only via
   emulated exchange semantics; `test/mdla_gpu.jl` (gated on `MFO_TEST_MDLA`)
   re-proves both on real MDLA `scatter!`/`reduce!` plus distributed
   `Krylov.cg` parity against single-device CG.

---

## 13 — Next steps

1. You decide the §10 open items (or defer any to implementation).
2. Generate the package skeleton (per your `JuliaPackageTemplate.generate()` +
   move to `~/dev/` convention), wire weakdeps/extensions, and relocate this
   document into the repo as its design doc.
3. Implement v1 in the file layout above — starting with `Grids.jl` →
   `operators/abstract.jl` → one leaf (`laplacian.jl`) end-to-end (action +
   adjoint + AD + Krylov) as the vertical slice that proves the architecture
   before adding the rest.
