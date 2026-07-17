---
slug: halo-exchange-schedule
created: 2026-07-17-1009
status: open
---

# Handoff: Precomputed halo-exchange schedule

## Goal / why this matters

Replace the runtime topology queries in the forest halo sweeps with a prepare-time
**exchange schedule**: a flat vector of homogeneous copy descriptors
`(src_leaf, dst_leaf, src_ranges, dst_ranges)`. `halo_update!` becomes a dumb loop of
concretely-typed `view .=` copies — no `face_neighbor` calls, no `Val{D}` dimension
recursion, no per-face branch logic in the hot path. Beyond cleanliness, this is the
structure the coarse–fine interpolation phase (add interpolation descriptors) and a future
MPI backend (send/recv lists) will require anyway — the work is never wasted.

## Background & current state

PR #8 (branch `amr-mg-issue5-alloc`, merged into `amr-mg`) fixed the halo sweeps'
allocations with compile-time dimension recursion: `_halo_faces!` /
`_halo_faces_adjoint!` (`src/transfer.jl:40` / `:80`) recurse on `Val{D}` so
`_dimslice(block, Val(D), r)` stays type-stable, and `face_neighbor` gained a `Val{D}`
hot-path variant (`src/topology.jl:164`) so the neighbor coord-tuple rebuild carries no
capturing closure over a runtime dimension. That fix works, but it re-derives the same
topology answers (who is my neighbor across each face?) on **every application** — answers
that are fixed between regrids. A schedule computes them once per forest generation.

## Key files / locations

- `src/transfer.jl:26,40` — `halo_update!` + `_halo_faces!` (to be replaced by schedule loop)
- `src/transfer.jl:68,80` — `halo_update_adjoint!` + `_halo_faces_adjoint!` (transpose loop)
- `src/topology.jl:145,155,164` — `_neighbor_coord`, `face_neighbor` (Int and `Val{D}`
  variants); the `Val{D}` variant becomes deletable — `balance!` keeps the Int form
- `src/boundaries.jl:71` — `_dimslice` (`_face_slabs` nearby gives the ghost/source ranges)
- `src/linalg.jl:39` — `PreparedForest` (natural home for the schedule; already
  generation-tied via its `generation` field and the guard in `mul!`, `src/linalg.jl:350`)
- `src/topology.jl:51` — `Forest` (`generation::RefValue{Int}`, bumped by every regrid)
- `DESIGN.md` — check the halo/`halo_update!` seam story before changing it

## Decisions & conclusions

- Descriptor type: `(src::Int, dst::Int, src_ranges::NTuple{N,UnitRange{Int}},
  dst_ranges::NTuple{N,UnitRange{Int}})` — one concrete type for all faces of all
  dimensions, so a plain `Vector` of them iterates type-stably with no barriers or tricks.
  `view(block, ranges...)` with an `NTuple{N,UnitRange}` is one concrete `SubArray` type.
- The adjoint sweep (add-back + zero, `src/transfer.jl:80`) runs the same descriptors with
  src/dst roles transposed — one schedule serves both directions.
- Ranked #3 in the design survey: strictly better structure than the `Val{D}` idiom it
  replaces, but partial — it does not address the leaf-grid type-instability (see the
  homogeneous-leaves-bc-face-pass handoff; the two are independent and composable). The
  packed-forest-kernels handoff subsumes this one (its halo exchange is index-mapped
  gather/scatter — a schedule in array form).
- Periodic wrapping and domain-boundary faces are resolved at schedule build time
  (`_neighbor_coord` logic); domain-boundary faces simply produce no descriptor.

## What's left / next steps

1. Read PR #8's diff to `src/transfer.jl` / `src/topology.jl` and this doc's anchors.
2. Define the descriptor struct and a `_build_exchange_schedule(bf::BlockForest)` that
   loops leaves × dims × sides using the existing `_neighbor_coord` / `_face_slabs` logic.
3. Store it on `PreparedForest` (build in `prepare(L, x::BlockField)`,
   `src/linalg.jl:99`); the existing generation guard already invalidates it on regrid.
   Decide whether the un-prepared `apply!` path keeps the current `Val{D}` sweeps or gets a
   per-generation cached schedule on `BlockForest` (RefValue cache keyed by generation).
4. Rewrite `halo_update!` / `halo_update_adjoint!` (or schedule-taking variants) as loops
   over descriptors; delete `_halo_faces!` / `_halo_faces_adjoint!` and the `Val{D}`
   `face_neighbor` variant if nothing else uses them.
5. Tests: forest operator correctness + adjoint identity must be unchanged;
   `test/forest_prepare.jl`'s `@inferred` and allocation guards must still pass (the
   schedule loop should be trivially inferrable).
6. Update `DESIGN.md` if the `halo_update!` seam description changes.

## Gotchas / constraints

- **Generation-tie is mandatory.** A schedule built before a `refine!`/`coarsen!`/
  `balance!` silently copies wrong faces — it must be invalidated exactly like
  `PreparedForest` (throw, never wrong results). If cached on `BlockForest` itself, key the
  cache by `forest.generation[]`.
- `halo_update_adjoint!` also **zeros the ghost after folding** (`src/transfer.jl:88-90`) —
  the schedule loop must preserve that, and the fold order across descriptors must remain
  equivalent (current order is leaf-major; additive folds commute, but the ghost-zeroing
  makes order within one ghost region matter — each ghost slab is touched exactly once
  today, keep it that way).
- The `_require_uniform` gate (`src/BlockForest.jl:70`) still applies until coarse–fine
  lands; the schedule builder can assert same-level neighbors for now but should be shaped
  so interpolation descriptors slot in later (that is the point).
- Julia alloc-measurement gotchas (DCE false-zeros, `@allocated` at global scope) — see
  `test/forest_prepare.jl:14-27` for the established measurement pattern.
- Do not run tests unless told to (project convention); CI covers them.
