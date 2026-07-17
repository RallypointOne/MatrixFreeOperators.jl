---
slug: packed-forest-kernels
created: 2026-07-17-1009
status: open
---

# Handoff: Packed-storage forest + forest-native kernels

## Goal / why this matters

Rewrite the forest data layout and apply path: replace the vector-of-arrays `BlockField`
with one packed `(blocksize .+ 2h ..., nleaves)` array and sweep **all leaves in a single
KernelAbstractions launch**, with per-leaf geometry in a small SoA table. Leaf `CartesianGrid`
objects vanish from the hot path entirely. This is the only design option that also
eliminates the residual ~384 B/leaf stencil-apply allocation (issue #7) and it is the real
GPU path — one kernel launch instead of nleaves. Ranked #2 in the design survey: the most
complete answer, priced as a project phase rather than a fix.

## Background & current state

After PR #8 (branch `amr-mg-issue5-alloc`, merged into `amr-mg`), forest `mul!` no longer
rebuilds leaf grids (prepare-time cache `_leaf_cache`/`_foreach_leaf`,
`src/blockfield.jl:76/88`, consumed by `PreparedForest`/`_forest_capply!`,
`src/linalg.jl:39/175`). The residual ~384 B/leaf is the per-leaf stencil `apply!` itself:
it is allocation-free only when inlined into a single `mul!` — a per-leaf call defeats that
inlining and its internal `interior(...)` views escape. Issue #7 tracks the
allocation-free-kernel rewrite; this handoff is the design that solves #7 *and* the GPU
batching problem in one move. The comment on `_leaf_cache` (`src/blockfield.jl:74`) already
notes the type-grouped layout "is also the block layout a packed GPU buffer would launch
over" — a deliberate breadcrumb toward this design.

## Key files / locations

- `src/blockfield.jl:17` — `BlockField{L,A,G}` (vector-of-blocks storage to replace or twin)
- `src/BlockForest.jl:85-149` — per-leaf geometry recompute (`_leaf_spacing`,
  `_leaf_extent`, `leaf_grid`) → becomes an SoA geometry table (origin, spacing per leaf)
- `src/operators/` — every leaf stencil body (`laplacian.jl`, `derivative.jl`,
  `gradient.jl`, `divergence.jl`, `advection.jl`, `scaling.jl`); per DESIGN.md,
  KernelAbstractions `@kernel` is the sanctioned per-operator escape hatch
- `src/operators/forest.jl` + `src/linalg.jl:175` — per-leaf sweeps to replace with launches
- `src/transfer.jl` — halo exchange becomes precomputed gather/scatter index maps (the
  array form of the halo-exchange-schedule handoff)
- `test/device.jl` / `test/device_gpu.jl` — GPU parity harness (gated on `MFO_TEST_GPU=true`)
- `DESIGN.md` — **authoritative**; this changes a locked layer-2 decision, so it must be
  revisited there first

## Decisions & conclusions

- This subsumes both sibling handoffs: homogeneous-leaves-bc-face-pass (a packed layout has
  no per-leaf grid types at all; BCs become face-list kernels) and halo-exchange-schedule
  (exchange = index-mapped copies inside or between launches).
- The honest cost: it abandons layer 2's core reuse story — "leaf bodies are array-level
  broadcast/slicing code reused unchanged per leaf, device-agnostic and AD-differentiable
  with no per-backend code". Each operator needs a forest-aware `@kernel` (or a new generic
  stencil abstraction that generates both single-grid and packed-forest bodies). That is
  why this ranked below the homogeneous-leaves option as a *fix* despite being the better
  *endgame*.
- Keep the flat Krylov boundary unchanged: `PreparedForest`'s `mul!` contract
  (interior-only flat vectors, generation guard, `src/linalg.jl:350`) survives; only the
  padded storage behind it and the sweep implementation change.
- AD is a first-class constraint, not an afterthought: Enzyme/Mooncake must still
  differentiate the kernels (Enzyme has KA support; verify per-operator), and adjoints
  remain declared, never assumed — the forest adjoint's cross-block ghost fold must be
  reimplemented as a scatter kernel and re-verified by the dot-product identity.
- Regridding invalidates the packed buffer layout ⇒ same generation-stamp pattern as
  `BlockField` (`src/blockfield.jl:20,28`) applies to the packed field type.

## What's left / next steps

1. Read `DESIGN.md` in full and issue #7 (including its notes on alloc-measurement
   gotchas); this is a design-doc change first, code second.
2. Decide: replace `BlockField` outright, or add a packed twin (e.g. `PackedBlockField`)
   behind the same `AbstractField` interface with `prepare` choosing it — the latter keeps
   the array-level reference path for correctness testing and AD fallback.
3. Design the geometry SoA (per-leaf origin, spacing, level) and the kernel indexing
   convention `(cell..., leaf)`; halo cells included in the packed extent.
4. Port one operator end-to-end first (Laplacian: forward + adjoint + boundary handling)
   and validate against the existing per-leaf path (`materialize` on small grids,
   dot-product identity, AD gradients vs `fd_gradient`) before touching the rest.
5. Port halo exchange as gather/scatter index maps; then the remaining operators.
6. Benchmark: single-launch forest apply vs current per-leaf path, CPU and GPU
   (`MFO_TEST_GPU=true`); confirm 0 B steady-state `mul!` — closing issue #7.
7. Update `DESIGN.md`, docstrings, and the PR #8-era comments that reference the residual.

## Gotchas / constraints

- **Do not start this as a refactor PR.** It is a phase with a design-doc decision gate;
  the repo's locked decisions live in `DESIGN.md` and changing layer 2's reuse story needs
  Kyle's sign-off there first.
- Mixed blocksizes/levels: packing assumes uniform block extents — fine while
  `_require_uniform` (`src/BlockForest.jl:70`) gates operators to single-level forests, but
  the layout choice should not paint coarse–fine into a corner (levels can pack per-level
  planes or pad).
- Per-leaf BC variation must not reintroduce per-leaf types: boundary handling belongs in
  face-list kernels (see the homogeneous-leaves handoff), not in per-leaf branches inside
  the stencil kernel.
- Alloc measurement: DCE false-zeros and `Profile.Allocs` blind spots bit this project
  before — measure behind function barriers as in `test/forest_prepare.jl:14-27`.
- Do not run tests unless told to (project convention); CI covers them, GPU parity only
  with `MFO_TEST_GPU=true` + CUDA.
