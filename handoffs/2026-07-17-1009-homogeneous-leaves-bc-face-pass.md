---
slug: homogeneous-leaves-bc-face-pass
created: 2026-07-17-1009
status: open
---

# Handoff: Homogeneous all-Interface leaf grids + forest-level BC face pass

## Goal / why this matters

Eliminate the root cause of the forest `mul!` type-instability — the per-leaf variation of
the BC type parameter of `CartesianGrid` — by making every leaf grid the *same* concrete
type (`Interface` on all faces) and moving physical-BC ghost filling into a forest-level
pass over precomputed face lists. This deletes the BC-signature group cache machinery that
PR #8 introduced as a workaround, and simplifies the forest layer permanently. Ranked #1
(best solution-per-cost) in the design survey that produced this handoff.

## Background & current state

The AMR forest applies operators leaf-by-leaf, where each leaf is an ordinary
`CartesianGrid`. `leaf_grid` (`src/BlockForest.jl:140`) is type-unstable because `leaf_bc`
(`src/BlockForest.jl:117`) mixes physical BCs and `Interface()` per face depending on the
leaf's runtime position — and BC is a **type parameter** of `CartesianGrid{N,T,BC,Dev,…}`.
Up to 3ᴺ distinct concrete grid types exist per forest (9 in 2D, 27 in 3D).

PR #8 (branch `amr-mg-issue5-alloc`, merged into `amr-mg`) worked around this with a
prepare-time cache grouped by concrete grid type: `_leaf_cache` / `_foreach_leaf`
(`src/blockfield.jl:76` / `:88`) iterated behind per-group function barriers by
`PreparedForest` / `_forest_capply!` (`src/linalg.jl:39` / `:175`). That fix is correct and
took forest `mul!` from ~1584 to ~384 B/leaf, but the group-cache + tuple-recursion idiom
is complexity this redesign would let you delete.

## Key files / locations

- `src/BlockForest.jl:117` — `leaf_bc` (the per-leaf BC mixing to remove)
- `src/BlockForest.jl:140` — `leaf_grid` (becomes type-stable: always all-`Interface`)
- `src/blockfield.jl:76,88` — `_leaf_cache`, `_foreach_leaf` (deletable once leaves are homogeneous)
- `src/linalg.jl:39,99,175` — `PreparedForest`, forest `prepare`, `_forest_capply!` (simplify:
  `groups` collapses to a plain leaf-grid vector or nothing)
- `src/operators/forest.jl:38` — un-prepared forest apply (same simplification)
- `src/boundaries.jl` — `apply_bc!` / `fold_bc!` / ghost-offset machinery (per-face pieces
  get forest-level callers)
- `src/linalg.jl:~320` — `boundary_rhs(L, ::BlockField)` (must use the face lists too)
- `DESIGN.md` — authoritative; records the "leaf runs single-grid code unchanged" story
  this changes. Read before starting; update after.

## Decisions & conclusions

- The 3ᴺ signature combinatorics collapse to (#physical BC types × faces): build, at forest
  construction or prepare time, small per-BC-type lists of `(leaf index, dim, side)` faces
  on the physical domain boundary. Each list is homogeneous; iterate each behind a function
  barrier (same trick as PR #8's `_foreach_leaf_group`, but with trivially few groups).
- This preserves open BC extensibility (dispatch on BC type per list) — the reason this
  option beat "make BC a runtime value" (which closes the BC set) in the ranking.
- This is how AMReX/p4est-style production AMR codes structure boundaries: homogeneous
  blocks, boundary operators as separate passes over face lists.
- Interaction with siblings: independent of the halo-exchange-schedule handoff (that one
  replaces the `Val{D}` halo idiom; this one replaces the leaf-grid cache). The
  packed-forest-kernels handoff subsumes both.

## What's left / next steps

1. Read `DESIGN.md` (forest/AMR sections) and PR #8 for full context.
2. Make `leaf_grid` return the all-`Interface` grid type unconditionally; build the
   physical-boundary face lists (grouped by BC type) on `BlockForest` or `PreparedForest`.
3. Add a forest-level `apply_bc!` pass (fill physical ghosts from the face lists) that runs
   after `halo_update!` and before the per-leaf stencil sweep, in both
   `src/operators/forest.jl` and the cached `_forest_capply!` path.
4. Move the adjoint counterpart: a forest-level `fold_bc!` pass in `apply_adjoint!` /
   `_forest_capply_adjoint!`, ordered correctly against `halo_update_adjoint!`.
5. Route `boundary_rhs(L, ::BlockField)` through the same face lists.
6. Delete `_leaf_cache` / `_foreach_leaf` and collapse `PreparedForest.groups`.
7. Tests: all of `test/forest_prepare.jl` must still pass (correctness vs un-prepared path,
   `@inferred` type-stability guard, generation guard); the adjoint dot-product identity
   ⟨Lx,y⟩ = ⟨x,Lᵀy⟩ on forests with mixed Dirichlet/Neumann BCs is the critical check.
8. Update `DESIGN.md`.

## Gotchas / constraints

- **Adjoint ordering is the subtle part.** Today the per-leaf adjoint runs stencil
  transpose + `fold_bc!` leaving interface-ghost contributions in place, then
  `halo_update_adjoint!` folds them across blocks (`src/operators/forest.jl:92`). With BCs
  hoisted out of the leaf, the forest-level `fold_bc!` must fold physical-ghost
  contributions at the right point relative to the interface fold — get this wrong and the
  dot-product identity fails only on boundary-touching blocks.
- Leaf `apply!` still calls its internal `apply_bc!`; with all-`Interface` faces that must
  be (and should be verified to be) a no-op, not an error.
- The repo invariant "a missing capability degrades to an error, never a wrong result"
  applies: if a BC type has no face-pass implementation, throw.
- Do not run tests unless told to (project convention); CI covers them.
