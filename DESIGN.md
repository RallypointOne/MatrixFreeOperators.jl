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

## Decisions locked (with the user)

| # | Decision | Consequence |
|---|----------|-------------|
| **A. Authoring model** | **Array-level (broadcast/slicing) leaves by default; KernelAbstractions `@kernel` as a per-operator escape hatch.** | This is the only default that simultaneously delivers device-agnosticism, Reactant-traceability, and automatic AD (incl. parameter gradients). Hand-written kernels are reserved for hot stencils that need shared-memory tiling. |
| **B. AD scope** | **Gradients w.r.t. both the solution field AND operator parameters** (material coefficients, geometry). | Custom adjoint rules become an *optimization* for linear leaves, not the whole AD story. Parameter gradients must flow through real AD — which array-level leaves provide for free. |
| **C. Solver API** | **Own lean lazy operator algebra exposing `mul!`/`size`/`eltype`. Drop SciMLOperators as the core.** Target Krylov.jl + MDLA + OrdinaryDiffEq.jl. | No `(u,p,t)` convention or `cache_operator` ceremony. A thin *optional* SciML `jac_prototype` adapter is provided only for matrix-free **implicit** OrdinaryDiffEq stepping (see §7). |

These three reshape everything below; the rest of the open choices are flagged in
§10 for the user to decide, with recommendations.

---

## Architecture overview

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

## Core abstraction 1 — Grid and Field (the foundation)

Everything keys off the grid. It must expose enough that distributed and AMR are
reachable without rewriting operators.

```julia
abstract type AbstractGrid{N} end          # N = spatial dimension

# v1 concrete type
struct CartesianGrid{N,T,BC,Dev} <: AbstractGrid{N}
    extent      :: NTuple{N,Tuple{T,T}}     # physical (min,max) per dim
    spacing     :: NTuple{N,T}              # Δx, Δy, … (uniform in v1)
    size        :: NTuple{N,Int}            # interior cell counts (LOCAL count)
    halo        :: NTuple{N,Int}            # ghost-layer width per dim
    bc          :: BC                       # boundary conditions per face
    device      :: Dev                      # KernelAbstractions backend
    # --- seams that stay no-op in v1, carry real data later ---
    local_range :: NTuple{N,UnitRange{Int}} # interior index range (== global in v1)
    topology    :: Any                      # neighbor ranks / hierarchy — Nothing in v1
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
- (AMR, phase 2) `coarsen(g)`, `refine(g, mask)`, `neighbors(g, cell)` returning
  level info + hanging-node flags.

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

---

## Core abstraction 2 — the composable operator algebra

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
struct Advection{G,V} <: AbstractOperator; grid::G; velocity::V; end   # nonlinear-capable
struct IdentityOp     <: AbstractOperator; end

# ---- Lazy combinators (closed under the algebra) ----
struct Scaled{O,T}   <: AbstractOperator; op::O; α::T; end
struct Added{A,B}    <: AbstractOperator; a::A; b::B; end
struct Composed{A,B} <: AbstractOperator; a::A; b::B; end   # (A∘B)x = A(Bx)
struct AdjointOp{O}  <: AbstractOperator; op::O; end

Base.:+(a::AbstractOperator, b::AbstractOperator) = Added(a, b)
Base.:*(a::AbstractOperator, b::AbstractOperator) = Composed(a, b)
Base.:*(α::Number, a::AbstractOperator)           = Scaled(a, α)
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
islinear(::AbstractOperator)       = true     # leaves opt out (e.g. Advection w/ field velocity)
isconstant(::AbstractOperator)     = true     # false if parameters/coeffs change in time
isselfadjoint(::AbstractOperator)  = false    # opt-in ONLY when BCs make it true
isdiagonal(::AbstractOperator)     = false    # enables cheap Jacobi smoother (multigrid)
```

**4. LinearAlgebra interop** — `mul!(y,L,x)`, `mul!(y,L,x,α,β)`, `size(L)`,
`eltype(L)`. This is all Krylov.jl and MDLA need; no SciML dependency.

**Device transfer:** implement `Adapt.adapt_structure` for every operator so
`adapt(CuArray, L)` moves bound grids/coefficients to device — exactly RBF's
pattern.

> **Tensor rank (flagged, §10):** RBF parameterizes operators by added tensor
> rank `AbstractOperator{N}` (gradient adds a dim, divergence removes one). v1
> can avoid this by representing vector/multi-component fields explicitly; adopt
> a rank parameter only if it proves necessary.

---

## Execution / device layer

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
whole-tree fusion (Reactant) is the fix, not a micro-optimization; (2) its two-tier
API (simple productivity default + low-level escape hatch) independently converges
on Decision A's split; (3) its benchmark suite (XSBench / miniBUDE / LULESH vs
Kokkos/native) is a methodology reference for *later* checking array-level leaves
don't trail hand-written kernels by too much. *Not lessons for us:* portable
reductions are already covered by `LinearAlgebra.dot` + GPUArrays `mapreduce` (our
⟨Lx,y⟩ and Krylov norms); a Preferences-based "default device" would reintroduce
the global-state tension the per-grid `device::Dev` deliberately avoids.

---

## Autodiff design

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
  through ordinary AD; (b) *nonlinear* operators (e.g. advection with a field
  velocity, nonlinear residuals) — those need JVP/VJP via AD. Both work
  automatically on array-level leaves.

**Backends:** Enzyme is primary (GPU support; best mutation handling). Mooncake
is the CPU-friendly alternative (pure Julia; **no GPU support as of 2026-06**).
Provide both via package extensions; the custom-rule files live in
`ext/…EnzymeExt.jl` and `ext/…MooncakeExt.jl`, loaded only when the AD package is
present, so the **core has zero AD dependencies**. Do not route custom rules
through DifferentiationInterface.jl (it cannot carry custom Enzyme rules); DI may
be offered as a user-facing convenience for generic code.

> **API to verify at implementation:** EnzymeRules `augmented_primal`/`reverse`
> exact signatures and `RevConfig`/activity types; Mooncake `rrule!!` / `CoDual`
> / `@from_rrule`. (One research subagent emitted a non-existent
> `EnzymeCore.register_primitive`; the correct mechanism is EnzymeRules — pin
> against Enzyme docs when coding.)

**Adjoint correctness is testable** and must be tested: the dot-product identity
⟨L x, y⟩ = ⟨x, Lᵀ y⟩ for random x, y, and AD gradients checked against
finite-difference / complex-step on small grids.

---

## Solver interop

**Krylov.jl + MultiDeviceLinearAlgebra:** Krylov needs only `mul!`, `size`,
`eltype` — which the algebra provides. To run distributed, implement
`mul!(y::MultiDeviceVector, L, x::MultiDeviceVector)` that (1) triggers MDLA's
halo/ghost exchange, (2) applies the leaf per partition with device context, (3)
writes the result partition-local. MDLA already supplies the distributed
primitives we compose with — `PartitionSpec`, `MultiDeviceVector`,
`GhostExchange`, `scatter!`/`reduce!`, and `mdla_solve` (Krylov-backed). Our
`halo_update!` seam maps onto MDLA `scatter!`. (Confirm these names against MDLA
source — see §11.)

**OrdinaryDiffEq.jl:** you do **not** need SciMLOperators to use it.
- *Explicit* solvers (RK4, SSPRK, …): provide a trivial RHS adapter
  `f!(du,u,p,t) = apply!(du, L, u, grid)` and hand it to `ODEProblem`.
- *Implicit/stiff* solvers (needed for diffusion): the idiomatic way to give
  OrdinaryDiffEq a **matrix-free Jacobian** is `ODEFunction(f; jac_prototype = J)`
  where `J` is a SciML-style lazy operator. Here `J` is the linear
  `linearize(F, u₀)` operator from §10.8 — its `mul!` is the forward-mode-AD JVP,
  so the matrix-free Jacobian comes from the AD layer, not by hand. We ship a
  **thin optional SciMLOperators adapter** (a `FunctionOperator`-style wrapper
  exposing that `mul!`) *only* for this one hook — in `ext/…SciMLExt.jl`, not in
  the core, and not adopting the `(u,p,t)` model anywhere else.

---

## v1 scope (build this first — keep it tight)

1. `CartesianGrid{N}` — uniform, single device, collocated, with periodic +
   Dirichlet + Neumann BCs; `halo_update!` present as a no-op.
2. `Field{Center}` over device arrays.
3. Operator algebra: leaves `Laplacian`, `Derivative`, `Gradient`, `Divergence`,
   `IdentityOp`, `Advection`; combinators `Added`, `Composed`, `Scaled`,
   `AdjointOp`; traits; **declared adjoints per leaf**; `apply!` + `apply_adjoint!`;
   `mul!`/`size`/`eltype`; `Adapt` support.
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

## Forward-looking seams (design now, implement later)

Each is one paragraph because the value of v1 is that these are *cheap*, not
pre-built.

- **Distributed memory.** *Correction to the original hope:* "just use
  DistributedArrays.jl and it works" is **not realistic** — DistributedArrays.jl
  has weak/absent halo support for stencils (long-standing, unresolved) and no
  GPU story. The seam is `halo_update!(field, grid)` plus the grid carrying
  local-vs-global index ranges and neighbor topology. Operators are written once
  against it. Concrete backends, in priority order: **(1) compose with your MDLA
  `GhostExchange`/`scatter!`** for multi-GPU (it is the closest existing model
  and it's yours); **(2) MPI + halo** via ImplicitGlobalGrid.jl / MPIHaloArrays.jl
  for multi-node CPU+GPU; **(3) Reactant XLA sharding** for automatic
  partitioning. The seam supports all three; we do not hard-depend on any
  distributed-array type. (See §10 for the pick-order decision.) *Not free:* an
  explicit `halo_update!` (MPI sends, `CUDA.device!` context switches) is **not**
  Reactant-traceable, while the Reactant sharding path wants the halo **implicit**
  (XLA inserts the communication during compilation). These are therefore
  *alternative execution modes* of the same leaf — chosen per backend — not one
  `apply!` body serving both at once.

- **Adaptive grids (AMR).** Tree-based AMR (forest of octrees) via p4est/t8code
  is the Julia ecosystem standard (Trixi.jl). The grid abstraction carries an
  optional `topology`/hierarchy; `neighbors(g, cell)` returns level info and
  hanging-node flags; near refinement interfaces, stencils need interpolation
  operators. The hard part — non-conforming/hanging-node stencils for high-order
  FD — is a phase-2 risk to prototype early on a small case.

- **Geometric multigrid.** Falls out of the algebra: restriction R and
  prolongation P are **themselves operators** in the same algebra; the grid
  exposes `coarsen(g)`; smoothers (Jacobi via `isdiagonal`, Chebyshev via repeated
  `apply!`) are built from the operator; the coarse operator is rediscretization
  (or Galerkin Rᵀ A P). References to mine: GeometricMultigrid.jl,
  RestrictProlong.jl. V/W/F-cycles compose on top.

---

## Open decisions to flag (your call)

Presented one at a time with a recommendation; none block writing v1's spine.

1. **Package name.** Working title only. Candidates: `MatrixFreeOperators.jl`,
   `StencilOperators.jl`, `GridOperators.jl`, `MeshFreeOperators.jl`.
   *Recommendation:* `MatrixFreeOperators.jl` (says exactly what it is).

2. **Collocated vs staggered fields in v1.** Staggered (faces/centers) avoids
   odd-even decoupling for incompressible flow but adds bookkeeping.
   *Recommendation:* collocated v1; encode location as the `Field` trait `L` so
   staggered is purely additive.

3. **AD backend default.** Enzyme (GPU + best mutation) vs Mooncake (CPU, pure
   Julia, gentler). *Recommendation:* ship both via extensions; document Enzyme
   as primary for GPU, Mooncake for CPU-only / where Enzyme struggles. Note
   Mooncake has no GPU support today.

4. **Distributed target to implement *first* (when we get there).** MDLA
   `GhostExchange` (multi-GPU, yours, CUDA-only today) vs MPI+halo
   (ImplicitGlobalGrid/MPIHaloArrays, multi-node, CPU+GPU) vs Reactant sharding.
   *Recommendation:* MDLA first (matches your stack and is multi-GPU now), MPI+halo
   for multi-node next, Reactant sharding opportunistically. The seam keeps all
   three open. **Does MDLA need to grow beyond CUDA (KernelAbstractions backends)
   for this?** — your call, since it's your package.

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
     single broadcast (one kernel launch, no temp). *Caveat to note in the doc:*
     composed matrix-free operators incur one kernel launch per leaf; whole-tree
     fusion is exactly what Reactant buys you — a point in its favor for
     composition-heavy operators.

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
     finite-difference JVP `(F(u₀+εv)−F(u₀))/ε` is the classic fallback.) That
     linear `J` is what feeds Krylov and the `jac_prototype` hook in §7.

---

## Files to reuse / reference

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

## Verification plan

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

---

## Next steps

1. You decide the §10 open items (or defer any to implementation).
2. Generate the package skeleton (per your `JuliaPackageTemplate.generate()` +
   move to `~/dev/` convention), wire weakdeps/extensions, and relocate this
   document into the repo as its design doc.
3. Implement v1 in the file layout above — starting with `Grids.jl` →
   `operators/abstract.jl` → one leaf (`laplacian.jl`) end-to-end (action +
   adjoint + AD + Krylov) as the vertical slice that proves the architecture
   before adding the rest.
