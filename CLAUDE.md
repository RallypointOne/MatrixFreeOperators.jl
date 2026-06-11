# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Development

- Run tests: `julia --project -e 'using Pkg; Pkg.test()'`
- Run a single test file (the test env devs the package at `path=".."`):
  ```
  julia --project=test -e '
  using MatrixFreeOperators, Test, LinearAlgebra, Random, StaticArrays
  import Adapt, Enzyme, KernelAbstractions, Krylov, Mooncake
  include("test/test_utils.jl"); include("test/laplacian.jl")'
  ```
- GPU parity tests are skipped unless `MFO_TEST_GPU=true` is set and CUDA.jl is available (`test/device.jl` gates `test/device_gpu.jl`)
- Build docs: `quarto render docs`
- Quarto YAML reference: https://quarto.org/docs/reference/

# Architecture

`DESIGN.md` is the authoritative design document — read it before structural changes. It records the locked decisions, the invariants below, and the open/deferred choices (staggered grids, distributed backends, AMR, multigrid).

The package is a lazy, composable, matrix-free operator algebra for PDEs on structured grids, in three layers:

1. **Foundation** — `src/Grids.jl`, `src/boundaries.jl`, `src/Fields.jl`: `CartesianGrid` (uniform, collocated, halo-padded), BCs (`Periodic`/`Dirichlet`/`Neumann`), and `Field` (a device array + grid + location tag). `halo_update!` is a no-op seam in v1; distributed/AMR later change only what the grid is and what `halo_update!` does — never operator code.
2. **Operator algebra** — `src/operators/`: leaves (`Laplacian`, `Derivative`, `Gradient`, `Divergence`, `ScalingOp`, `IdentityOp`, `Advection`) bind a grid at construction; combinators (`Added`, `Composed`, `Scaled`, `AdjointOp`) close the algebra under `+`, `*`, scalar scaling, and `adjoint`. Leaf bodies are array-level (broadcast/slicing) by default so they are device-agnostic (GPUArrays + Adapt) and differentiable (Enzyme/Mooncake) with no per-backend or per-AD code; KernelAbstractions `@kernel` is the per-operator escape hatch.
3. **Solver boundary** — `src/linalg.jl`: `prepare(L, x)` walks the operator tree once, allocates all scratch buffers, and returns a `PreparedOperator` whose flat `mul!`/`size`/`eltype` is what Krylov.jl consumes — zero-allocation in steady state, stateful, single-threaded (prepare once per concurrent solve).

Invariants to preserve when adding or modifying operators:

- **Traits default to the weaker claim.** `islinear`/`isconstant`/`isselfadjoint`/`isdiagonal` default `false`; leaves opt in, combinators propagate explicitly. A forgotten declaration must degrade to an error, never a wrong result.
- **Adjoints are declared, never assumed.** Every linear leaf declares its adjoint including boundary contributions — BCs break self-adjointness even for the Laplacian. Verified by the dot-product identity ⟨Lx,y⟩ = ⟨x,Lᵀy⟩.
- **Linear/affine split.** `apply!`/`apply_bc!` enforce homogeneous BCs only, so `islinear(L)` ⇒ `L(0)=0`. Inhomogeneous boundary data is exported separately via `boundary_rhs` and folded into the solve RHS.
- **Interior-only flat vectors.** Flat (Krylov) vectors span interior DOFs only; ghost cells are scratch filled by `halo_update!`/`apply_bc!`, never solver unknowns. `apply!(y, L, x, grid, α, β)` writes the interior of `y`.
- **Element type carries tensor rank.** A vector field is a `Field` over `Array{SVector{N,T}}` — no rank parameter on operators. Stencil bodies stay generic over eltype; only the rank-changers `Gradient`/`Divergence` touch components.
- **Nonlinear operators never masquerade as linear maps.** They support `apply!` + AD but not `adjoint`/`prepare`; `linearize(F, u₀)` produces the linear Jacobian operator (AD-powered JVP) that feeds Krylov.

AD (Enzyme, Mooncake), MDLA, SciML, and Reactant integrations belong in `ext/` package extensions — the core depends only on Adapt, KernelAbstractions, LinearAlgebra, StaticArrays.

# Testing conventions

Every operator gets: action vs analytic solution, the adjoint identity, composition laws, and AD gradients (field **and** parameters) checked against finite differences. `test/test_utils.jl` provides `fd_gradient` (FD reference gradient) and `materialize` (densify a prepared operator on small grids to check structure exactly).

# Docs Sidebar

- `api.qmd` must always be the last item before the "Reference" section in `_quarto.yml`
- `api.qmd` lives in its own `part: "API"` to visually separate it from other doc pages
- `index.qmd` must always begin with `## Overview` and `## Quickstart` sections

# Style

- 4-space indentation
- Docstrings on all exports
- Use `### Examples` for inline docs examples
- Segment code sections with: "#" * repeat('-', 80) * "# " * "$section_title" on a single line

# Releases

- First released version should be v0.1.0
- Preflight: tests must pass and git status must be clean
- If current version has no git tag, release it as-is (don't bump)
- If current version is already tagged, bump based on commit log:
  - **Major**: major rewrites (ask user if major bump is ok)
  - **Minor**: new features, exports, or API additions
  - **Patch**: fixes, docs, refactoring, dependency updates (default)
- Commit message: `bump version for new release: {x} to {y}`
- Generate release notes from commits since last tag (group by features, fixes, etc.)
- Important: For major or minor version bumps, release notes must include the word "breaking"
- Register via:
  ```
  gh api repos/{owner}/{repo}/commits/{sha}/comments -f body='@JuliaRegistrator register

  Release notes:

  <release notes here>'
  ```
