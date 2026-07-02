---
slug: mfo-mdla-composition
created: 2026-07-01-1054
status: open
---

# Handoff: Compose MatrixFreeOperators.jl with MDLA (the `MDLAExt.jl` extension)

## Goal / why this matters
Make MatrixFreeOperators (MFO) run distributed / multi-GPU by composing with the
user's own MultiDeviceLinearAlgebra.jl (MDLA) — the #1 distributed target in
`DESIGN.md` (§9, §10.4: *"MDLA first — matches your stack and is multi-GPU now"*).
The layered architecture was built so this touches **two seams and nothing in the
operator algebra**. This handoff captures the plan to turn that designed seam into
a concrete `ext/MatrixFreeOperatorsMDLAExt.jl` extension.

## Background & current state
- MFO is a lazy, matrix-free operator algebra on structured grids. `DESIGN.md` is
  authoritative — read §3 (Grid/Field seams), §7 (solver interop / MDLA), §9
  (distributed seam), §10.4 (distributed pick-order), §10.7 (prepare/scratch), §11
  (MDLA source file list), §12.7 (seam smoke test).
- `halo_update!(field, grid)` is currently a **no-op** (the single distributed seam).
- The flat Krylov vector is currently a plain interior-only `Vector`; `prepare(L, x)`
  → `PreparedOperator` exposes `mul!`/`size`/`eltype` for Krylov.
- `CartesianGrid` already carries the `local_range` and `topology` seam fields
  (`== global` / `Nothing` in v1) that a distributed grid will populate.
- No MDLA integration exists yet. This is greenfield extension work.

## Key files / locations
MFO (this repo):
- `DESIGN.md` §3, §7, §9, §10.4, §10.7, §11, §12.7 — the design of record.
- `src/Grids.jl` — `CartesianGrid`, `halo_update!` (no-op), `local_range`, `topology`.
- `src/linalg.jl` — `prepare`/`PreparedOperator`, flat `mul!`/`size`/`eltype`.
- `src/boundaries.jl` — `apply_bc!` / `fold_bc!` (adjoint of BC application; the
  pattern the distributed halo adjoint mirrors).
- Target new file: `ext/MatrixFreeOperatorsMDLAExt.jl` (+ weakdep wiring in
  `Project.toml`, mirroring the other `ext/` extensions).

MDLA (separate package, on disk — **API names below are from `DESIGN.md` §7/§11 and
are UNVERIFIED**; `DESIGN.md` itself says "confirm these names against MDLA source"):
- `~/dev/MultiDeviceLinearAlgebra/src/partition.jl` — `PartitionSpec`
- `~/dev/MultiDeviceLinearAlgebra/src/vector.jl` — `MultiDeviceVector`
- `~/dev/MultiDeviceLinearAlgebra/src/ghost.jl` — `GhostExchange`, `scatter!`, `reduce!`
- `~/dev/MultiDeviceLinearAlgebra/src/matrix.jl`, `mul.jl`, `krylov_compat.jl`
- (`~/dev` is a symlink to the iCloud `pro/dev`; `JULIA_PKG_DEVDIR`.)

## Decisions & conclusions
Composing with MDLA reduces to making **two seams speak MDLA**, plus a per-partition
`prepare` — the operator algebra (leaves, adjoints, traits, composition) does **not**
change:

1. **`halo_update!(field, grid)` → MDLA `GhostExchange`/`scatter!`.** Fills ghost
   layers from neighbor partitions. Every leaf already calls it before reading
   neighbors, so no leaf changes.
2. **Flat Krylov vector → `MultiDeviceVector`.** Entry point becomes
   `mul!(y::MultiDeviceVector, L, x::MultiDeviceVector)`: exchange ghosts → apply the
   leaf per partition (setting device context, e.g. `CUDA.device!`) → write
   partition-local. Krylov dot/norm run on `MultiDeviceVector` via MDLA's
   `krylov_compat`.
3. **`prepare` per partition.** §10.7 already sanctions per-thread/per-partition
   prepared operators — each partition owns its halo-padded scratch field + device
   context. Just buffer allocation; no new abstraction.
4. The `CartesianGrid` becomes a **local** grid per partition; `local_range` +
   `topology` carry the local index range and neighbor ranks. This is the
   "distributed differs only in what the Grid is and what `halo_update!` does"
   invariant (§3, §9).

## What's left / next steps
1. **Read the actual MDLA source** (`~/dev/MultiDeviceLinearAlgebra/src/`) and pin
   the real API — the names above are unverified. Confirm: `PartitionSpec`,
   `MultiDeviceVector`, `GhostExchange`, `scatter!`, `reduce!`, `mdla_solve`, and
   what `krylov_compat.jl` actually provides (dot/norm/axpy on `MultiDeviceVector`).
2. **Map the seams** in a short design note: exactly how `halo_update!` calls MDLA
   ghost exchange, and how `mul!(::MultiDeviceVector, L, ::MultiDeviceVector)` drives
   per-partition application with device context.
3. **Prove the distributed adjoint FIRST** (the real crux — see gotchas): show
   ⟨Lx,y⟩=⟨x,Lᵀy⟩ survives partitioning on a 2-partition Laplacian. The transpose of
   a ghost-`scatter!` is a ghost-`reduce!` (accumulate halo contributions back to
   owners) — MDLA exposing both is the adjoint pair; the AMR `halo_update_adjoint!`
   (§9) is the existing analogue.
4. **Minimal vertical slice:** distributed Laplacian solved with `Krylov.cg` on 2
   partitions, checked bit-parity against the single-device result (mirrors the
   §12.7 seam smoke test — currently stubs a 2-partition MDLA `mul!`).
5. Write `ext/MatrixFreeOperatorsMDLAExt.jl` + `Project.toml` weakdep wiring; add
   distributed tests behind an env gate like the existing GPU-parity gate
   (`MFO_TEST_GPU` in `test/device.jl`).

## Gotchas / constraints
- **Distributed adjoint correctness is where subtle bugs hide.** The scatter/reduce
  transpose pair must accumulate (not overwrite) halo contributions, or the adjoint
  identity silently fails. Do this before building anything else on top.
- **MDLA is CUDA-only today; MFO is backend-agnostic.** Open question flagged in
  §10.4: either the distributed path is CUDA-only for now, or MDLA grows
  KernelAbstractions-backend support so MFO's device-agnosticism survives
  distribution. This is a decision on the *MDLA* package's scope, and it's the
  user's call (their package).
- **MDLA path is imperative-halo, NOT Reactant.** Explicit `scatter!` + `CUDA.device!`
  context switches are not Reactant-traceable (§9). The MDLA path and the
  Reactant-sharding path are *alternative execution modes* of the same leaf — picking
  MDLA-first means the Reactant distributed story is a separate, later track. Don't
  try to make one `apply!` body serve both.
- **API names are unverified** — treat every MDLA symbol in this doc as a hypothesis
  until confirmed against source (step 1). `DESIGN.md` explicitly flags this.
- **Testing convention:** do not run tests unless explicitly told to. Write the tests
  (per-operator: action, adjoint identity, parity) but leave running them to the user.
- No secrets involved in this work; nothing to redact.
