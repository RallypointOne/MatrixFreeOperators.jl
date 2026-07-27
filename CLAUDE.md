# MatrixFreeOperators.jl

Lazy, composable, matrix-free operator algebra for PDEs on structured grids — device-agnostic
(CPU/GPU) and differentiable (forward and reverse, including w.r.t. operator parameters).

**`DESIGN.md` is the authoritative design document — read it before any structural change.**
It holds the locked decisions, the full rationale behind the invariants below, and the
open/deferred choices (staggered grids, distributed backends, AMR, multigrid).

## Key files
- foundation: `src/Grids.jl`, `src/boundaries.jl`, `src/Fields.jl`
- operator leaves and combinators: `src/operators/`
- solver boundary (`prepare`, `PreparedOperator`, flat `mul!`): `src/linalg.jl`
- nonlinear → linear Jacobian: `src/operators/linearize.jl`
- AD / MDLA / Reactant integrations: `ext/`

## Invariants — violating these produces silently wrong results
- **Traits default to the weaker claim.** `islinear` / `isconstant` / `isselfadjoint` / `isdiagonal` default to `false`; leaves opt in, combinators propagate explicitly. A forgotten declaration must degrade to an error, never a wrong answer.
- **Adjoints are declared, never assumed.** Every linear leaf declares its adjoint *including boundary contributions* — BCs break self-adjointness even for the Laplacian. Check with the dot-product identity ⟨Lx,y⟩ = ⟨x,Lᵀy⟩.
- **Linear/affine split.** `apply!` / `apply_bc!` enforce homogeneous BCs only, so `islinear(L)` ⇒ `L(0) = 0`. Inhomogeneous boundary data goes out separately through `boundary_rhs` and is folded into the solve RHS.
- **Interior-only flat vectors.** Flat (Krylov) vectors span interior DOFs only. Ghost cells are scratch filled by `halo_update!` / `apply_bc!` and are never solver unknowns.
- **Element type carries tensor rank.** A vector field is a `Field` over `Array{SVector{N,T}}` — operators have no rank parameter. Only the rank-changers `Gradient` / `Divergence` touch components.
- **Nonlinear operators never masquerade as linear maps.** They support `apply!` and AD but not `adjoint` / `prepare`; `linearize(F, u₀)` yields the AD-powered Jacobian operator that feeds Krylov.

Two structural rules follow from these: leaf bodies stay array-level (broadcast/slicing) so they are
device-agnostic and AD-friendly with no per-backend code — KernelAbstractions `@kernel` is the
per-operator escape hatch, not the default. And `halo_update!` is a deliberate no-op seam in v1, so
distributed/AMR work later changes only the grid and that function, never operator code.

## Gotchas
- The core depends only on Adapt, KernelAbstractions, LinearAlgebra, and StaticArrays. AD, MDLA, SciML, and Reactant integrations belong in `ext/` — don't add them to `[deps]`.
- `prepare` is stateful and single-threaded: call it once per concurrent solve, not once globally.
- GPU parity tests are skipped unless `MFO_TEST_GPU=true` and CUDA.jl is available (`test/device.jl` gates `test/device_gpu.jl`) — a green suite does not mean GPU paths ran.
- Running one test file needs the test env, which `dev`s the package at `path=".."`:
  ```
  julia --project=test -e '
  using MatrixFreeOperators, Test, LinearAlgebra, Random, StaticArrays
  import Adapt, Enzyme, KernelAbstractions, Krylov, Mooncake
  include("test/test_utils.jl"); include("test/laplacian.jl")'
  ```

## Testing conventions
Every operator gets four checks: action vs. analytic solution, the adjoint identity, composition
laws, and AD gradients (field **and** parameters) against finite differences. `test/test_utils.jl`
provides `fd_gradient` and `materialize` (densify a prepared operator on small grids to check
structure exactly).
