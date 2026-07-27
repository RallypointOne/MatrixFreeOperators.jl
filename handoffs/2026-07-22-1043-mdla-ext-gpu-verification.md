---
slug: mdla-ext-gpu-verification
created: 2026-07-22-1043
status: open
---

# Handoff: run the MDLA distributed GPU verification (issue #16, steps 3–5)

## Goal / why this matters

Branch `feat/mdla-ext` implements issue #16 slice 1 — distributed multi-GPU
solves via a MultiDeviceLinearAlgebra.jl (MDLA) package extension. Everything
CPU-provable is already proven and green; what remains is executing the
env-gated GPU testsets on a machine with **≥ 2 NVIDIA GPUs**: 2-partition
forward parity, the distributed adjoint identity on real `scatter!`/`reduce!`,
and distributed `Krylov.cg` parity. You are on that machine; run them, fix
anything that flakes (fallbacks below), and report.

## Background & current state

- Branch: `feat/mdla-ext` (4 commits on top of `main`). If it isn't on this
  machine yet it must be pushed/pulled first — it exists locally on Kyle's Mac.
- Full CPU suite is green: 4657 pass, 0 fail, 3 broken (= the env-gated skips
  for GPU / MDLA / Reactant).
- `test/partitioning.jl` already proves, CPU-only via emulated exchange
  semantics: bitwise 2-/3-partition forward parity vs the single grid, the
  adjoint identity ⟨Lx,y⟩=⟨x,Lᵀy⟩ across partitions, and dense transpose
  structure. The GPU run re-proves these on real MDLA communication and adds
  distributed CG.
- MDLA is Kyle's package, unregistered, v0.0.1 — on Kyle's machines it lives at
  `~/dev/MultiDeviceLinearAlgebra`. It is CUDA-only, has no CPU mode, and
  enforces **unique physical GPU IDs per partition** (no 2-partitions-on-1-GPU).

## Key files / locations

- `ext/MatrixFreeOperatorsMDLAExt.jl` — the extension: `prepare_distributed`
  (one `prepare`d operator per `partition_grid` slab per CUDA device),
  `mul!(y::MultiDeviceVector, P, x::MultiDeviceVector[, α, β])`
  (= `scatter!` → per-device ghost unpack into Interface halo slabs → local
  apply), `_mul_adjoint!` (local mechanical transpose → pack `local_x` →
  `reduce!(+)`), `Krylov.CgWorkspace` hook.
- `test/mdla.jl` — the gate (`MFO_TEST_MDLA=true` + CUDA.jl + MDLA findable).
- `test/mdla_gpu.jl` — the testsets to make pass. Multi-partition sets
  self-skip below 2 GPUs; on this machine nothing should skip.
- `src/partitioning.jl` — `partition_grid` + `_slab_ghost_layout` (ghost
  request lists are owner-ascending, plane-ascending within owner — this
  deliberately replays MDLA's `local_x` ghost-section ordering rule from
  `_compute_ghost_topology`, MDLA `src/ghost.jl:7-72`).
- MDLA contracts the ext composes with: `src/ghost.jl:220` (`GhostExchange`
  ctor), `:320/:338` (`scatter!`), `:387/:411` (`reduce!`),
  `src/krylov_compat.jl` (`_empty_mdv`, `CgWorkspace`).
- Design of record: `DESIGN.md` §7 (updated with slice-1 scope), plan context
  in the issue #16 thread.

## Decisions & conclusions (don't relitigate)

- CUDA-only slice by design; MFO core stays backend-agnostic — only the ext is
  CUDA-bound.
- The operator algebra is untouched. Locality comes from
  `prepare(L, scalar_field(local_grid))`: `apply!` reads spacing from the
  passed grid and BCs from the field's grid, and `Interface` cut faces are
  skipped by `apply_bc!`/`fold_bc!`, so ghost slabs filled by the exchange
  survive the local sweep.
- `reduce!(x̄, ghost, spec, +)` is the exact transpose of `scatter!` (verified
  by MDLA's own round-trip test); the local `apply_adjoint!` on an
  Interface-faced grid leaves neighbor-owned cotangents in the ghost slabs,
  which is exactly what gets packed and reduced.
- Slice-1 whitelist: `Laplacian`, `IdentityOp`, number-coefficient `ScalingOp`,
  under `Scaled`/`Added`. `Composed`/`AdjointOp`/`Field` coefficients throw by
  design (mid-tree exchange/reduction not built) — do not "fix" the guards.

## What's left / next steps

1. Ensure this repo checkout is on `feat/mdla-ext` and MDLA is cloned locally
   (ask Kyle or check `~/dev/MultiDeviceLinearAlgebra`).
2. Build a throwaway env (gated deps deliberately stay OUT of
   `test/Project.toml`):

   ```
   julia --project=/tmp/mfo-gpu-env -e '
   using Pkg
   Pkg.develop(path=".")                                    # MFO repo root
   Pkg.develop(path="<path-to-MultiDeviceLinearAlgebra>")
   Pkg.add(["CUDA", "Krylov", "Test", "Random", "StaticArrays", "Adapt"])'
   ```

3. Run the MDLA testsets alone first (fast loop, skips the AD deps):

   ```
   MFO_TEST_MDLA=true julia --project=/tmp/mfo-gpu-env -e '
   using MatrixFreeOperators, Test, LinearAlgebra, Random, StaticArrays
   import Adapt, Krylov
   include("test/test_utils.jl"); include("test/mdla.jl")'
   ```

4. Expected: every testset in `test/mdla_gpu.jl` passes with zero skips
   (machine has ≥ 2 GPUs). Checkpoint order mirrors the plan: single-partition
   parity → 2-partition forward parity → adjoint identity + dense transpose →
   CG parity with equal iteration counts across ndev ∈ {1, 2}.
5. If fixes were needed, commit them to `feat/mdla-ext` (atomic,
   `fix(ext): ...` / `test(mdla): ...`), re-run, and report results. Pushing
   the branch / opening the PR against #16 is Kyle's call — report, don't
   assume.

## Gotchas / constraints

- **Known watch-item 1 — stream ordering.** The `scatter!`-then-`@async`
  pattern is inherited from MDLA's own `mul!` (MDLA `src/mul.jl`). If forward
  parity flakes nondeterministically, insert a per-device `CUDA.synchronize()`
  after `scatter!` in the ext's `mul!` and `_mul_adjoint!` (after the pack
  loop) — this was pre-agreed as the fallback, not a design change.
- **Known watch-item 2 — CG iteration counts.** `@test allequal(niters)` in
  the CG testset may flake: MDLA's `dot` is per-partition cuBLAS dots summed on
  host, which rounds differently than a fused dot. If it fails persistently,
  loosen to comparing solutions only (document in the test) — also pre-agreed.
- **Known watch-item 3 — `Krylov.cg(P, b)` dispatch.** The
  `Krylov.CgWorkspace(P, b)` hook mirrors MDLA's and is verified only
  transitively (Krylov 0.10). If `cg` doesn't route to it, compare against how
  MDLA's own `test_krylov.jl` calls it.
- The ext imports MDLA-internal `_empty_mdv` (`krylov_compat.jl:1`) — pinned by
  `compat MultiDeviceLinearAlgebra = "0.0.1"`. If MDLA on this machine is newer
  and the import fails, that is the reason.
- The forward parity asserts are **`==` (bitwise)** between distributed and
  single-GPU results — same device arithmetic, per-cell-independent kernels.
  Only distributed-vs-CPU and adjoint comparisons use `≈`. Don't blanket-loosen
  `==` to `≈`; a bitwise mismatch means a real indexing/exchange bug (check
  ghost plane ↔ `local_x` section mapping first).
- `test/device_gpu.jl` (gated by `MFO_TEST_GPU=true`) sets
  `CUDA.allowscalar(false)` globally; `test/mdla_gpu.jl` deliberately does not
  set it. If you run both gates in one session and MDLA internals hit scalar
  indexing, run the MDLA gate in its own session before concluding anything.
- Julia ≥ 1.10; repo path on Kyle's Mac contains spaces — always double-quote
  paths in shell commands.
- No secrets involved anywhere in this task.
