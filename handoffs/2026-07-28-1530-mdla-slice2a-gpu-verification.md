---
slug: mdla-slice2a-gpu-verification
created: 2026-07-28-1530
status: open
---

# Handoff: run the GPU verification for MDLA slice 2a (issue #31, Composed + AdjointOp)

## Goal / why this matters

Branch `feat/mdla-slice2a` implements the first half of issue #31: distributed `Composed` and
`AdjointOp`. Everything CPU-provable is proven and green in CI. What remains is executing the
env-gated GPU testsets on a machine with **≥ 2 NVIDIA GPUs** (and, for the `test/multigpu` set,
**≥ 3**) — the same drill as the slice-1 handoff
(`2026-07-22-1043-mdla-ext-gpu-verification.md`), which passed 34/34 + 23/23 on the 9× A30 host
`sasquatch`.

Nothing here is known-broken. The GPU testsets are **written but unrun**.

## What changed

`Composed(A, B)` applied `B` into an intermediate and then let `A` read that intermediate's
neighbours — but its `Interface` ghost slabs were never filled, so they were stale zeros and the
answer was wrong near every cut plane with nothing raised. `AdjointOp` was the transpose of the
same gap: `reduce!(+)` only ran at the root. Both were rejected loudly rather than answered
wrongly. This branch implements the machinery.

The single-device precedent was already in the repo — `_forest_capply!` /
`_forest_capply_adjoint!` (`src/linalg.jl`) recurse at the forest level so a `PreparedComposed`
intermediate gets its inter-block exchange. `src/distributed.jl` is the distributed twin.

**The walk lives in core**, parameterized on three backend primitives (`_dist_map!`,
`_dist_scatter!`, `_dist_reduce!`). The MDLA extension fills them with `scatter!`/`reduce!`;
`test/partitioning.jl` fills them with plain-`Vector` global indexing. So the CPU proof exercises
the *real* walk rather than a second copy of it — that is the main structural improvement over
slice 1, where the extension and the test each had their own `_halo_plane_view`.

Scope is deliberately the mid-tree half only. `Field` coefficients and distributed `boundary_rhs`
are slice 2b; #31 stays open until both land.

## Key files / locations

| File | What |
|---|---|
| `src/distributed.jl` (new) | the guards (moved out of the ext), `_push_adjoints`, the three backend primitives, the `DistNode` tree, `_dist_capply!` / `_dist_capply_adjoint!` |
| `src/partitioning.jl` | now also owns `_halo_plane_view` / `_unpack_ghosts!` / `_pack_local_x!`, moved out of the ext |
| `ext/MatrixFreeOperatorsMDLAExt.jl` | `MDLAExchange`, `MDLAContext`, the three primitives, root scatter/reduce fast paths; `mul!` and `_mul_adjoint!` are now thin drivers over the walk |
| `test/partitioning.jl` | the CPU proof of record — the CPU backend plus the slice-2a testsets |
| `test/mdla_gpu.jl` | **the GPU testsets to run** |
| `test/multigpu/mdla_3partition.jl` | **the ≥3-GPU testsets to run** |

## What's left / next steps

Run the two gated suites, exactly as in the slice-1 handoff (that document's "How to run" and its
P2P health check still apply verbatim — check the host's peer copies before trusting any result):

1. `test/mdla.jl` with `MFO_TEST_MDLA=true`, ≥ 2 GPUs.
2. `test/multigpu/mdla_3partition.jl`, ≥ 3 GPUs.

New testsets to watch, and what a failure in each would mean:

- **`composed: 2-partition forward parity`** — bitwise `==` against 1 partition. A mismatch means
  the mid-tree exchange is landing in the wrong place; check the intermediate's ghost plane ↔
  `local_x` section mapping first, exactly as in slice 1.
- **`composed: adjoint identity and transpose structure`** — `⟨Lx,y⟩ = ⟨x,Lᵀy⟩` plus
  `A_adj ≈ A_fwd'`. This is the literal statement of the transpose argument with a mid-tree
  exchange in the middle of it.
- **`AdjointOp as a distributed forward operator`** — the mid-tree *reduction* path.
- **`adjoint sibling under Added: both term orders agree`** — the order-dependence regression (see
  below). CPU-verified by mutation; this is its GPU mirror.
- **`Krylov.cg on a distributed composed system`** — `DᵀD + I` (SPD, min eigenvalue 3.46 on the
  test grid), iteration-count equality across partition counts.
- **`3-partition Composed forward parity` / `adjoint identity`** — the only configuration where the
  *mid-tree* exchange's ghost section carries planes from two different owners.

Two pre-agreed fallbacks if something does fail, both mirroring slice 1's (neither was needed
then):

- If bitwise composed parity fails but `≈` passes, do **not** loosen it blindly — a bitwise
  mismatch means a real indexing bug. Loosen only after confirming the cause is FP reassociation.
- If `allequal(niters)` fails on the composed CG system, check `dot` first: MDLA sums per-partition
  cuBLAS dots on the host, which rounds differently than a fused dot.

## Decisions & conclusions (don't relitigate)

Carried forward from slice 1: CUDA-only by MDLA's nature; the operator algebra is untouched;
locality comes from `prepare(L, scalar_field(local_grid))`; `reduce!(+)` is the exact transpose of
`scatter!`. New to this slice:

- **The walk is host-orchestrated, not a rendezvous.** Overloading `halo_update!` on a
  topology-carrying slab grid would make `Composed` work with zero algebra changes, since every
  leaf already calls it. Rejected: MDLA's `scatter!` is itself host-orchestrated
  (`@sync for d … @async`), so a rendezvous-style `halo_update!` would mean reimplementing it
  inside MFO and losing MDLA's P2P-health probe and host-staging fallback (MDLA #22).
- **`DistAdjoint` keeps its own input scratch, and this is not an optimization.** The adjoint
  gather opens with `zero_ghosts!` on its input (`adjoint_gather!`), so sharing the enclosing
  segment's field destroys exchanged ghosts a *sibling* under the same `Added` still needs —
  making the answer depend on term order, silently. Verified load-bearing by mutation: removing
  the scratch fails `adjoint(D₁) + laplacian` and leaves `laplacian + adjoint(D₁)` passing.
  **The buffer discipline is deliberately asymmetric** — the adjoint direction has no mirror of
  this bug, because there the node scatters onto ghosts that were zero, which is additive.
- **`zero_ghosts!` runs once per adjoint *segment*, never per leaf.** Per leaf, the second term of
  an `Added` wipes the first term's `Interface` cotangents before they reach the reduction. Also
  verified load-bearing by mutation.
- **No `isselfadjoint` / `isdiagonal` shortcut in the adjoint walk**, unlike the forest version:
  `_selfadjoint_grid` is `true` for any `CartesianGrid`, so the trait still claims
  self-adjointness on a slab whose cut faces are `Interface`. The leaf ignores the trait and
  branches on `_has_interface`; taking the shortcut would route to the forward walk and, mid-tree,
  flip a reduction into a scatter.
- **`_reads_ghosts` gates the forward scatter and the adjoint reduce together.** Gating one alone
  breaks the transpose.
- **`_push_adjoints` normalizes before the guards.** `prepare` does not recurse into an
  `AdjointOp`, so `AdjointOp(A*B)` would run `aᵀ` then `bᵀ` with no reduction between them.
  `Base.adjoint` never builds that form, but a user can write it directly.
- **`Derivative` joined the whitelist.** Same shape as `Laplacian`, and the only whitelisted leaf
  that is *not* self-adjoint — without it, no operator can produce an `AdjointOp` node at all, so
  the `AdjointOp` half of #31 would be untestable.
- **Scalar intermediates only.** Rank-changing intermediates (`Divergence ∘ Gradient`) need their
  own `ncomp = N` spec, ghost layout and exchange; `_slab_ghost_layout` and `_owned_flat_range`
  already take the `ncomp` kwarg, so 2b is a contained increment. Transfer chains stay rejected —
  their factors live on different grids that would each need a consistent cut.

## Verified so far (no GPU)

- **CPU suite: 4830 pass / 0 fail / 3 broken** (the three env-gated skips). Baseline before this
  work was 4657.
- `test/partitioning.jl` alone: 359 pass (was 212).
- The two subtlest hazards above were confirmed by **mutation testing** — deliberately
  reintroducing each bug makes a named testset fail, so those are genuine regression tests, not
  decoration.
- The extension **precompiles** and all three primitives attach (checked in a scratch env with
  MDLA + CUDA + Krylov dev'd in; CUDA loads on macOS but has no driver, so anything past the
  guards is unrun).
- `prepare_distributed`'s guards were exercised through the real entry point: `Composed`,
  `AdjointOp` and `AdjointOp(A*B)` now pass; rank-changing roots, `Field` coefficients and
  transfer operators are rejected with messages naming the reason.

## Gotchas / constraints

- `test/mdla_gpu.jl` deliberately does **not** set `CUDA.allowscalar(false)` while
  `test/device_gpu.jl` sets it globally — run the two gates in separate sessions.
- Neither CUDA nor MDLA is in `test/Project.toml`; the gated files need a hand-built env with both
  dev'd in.
- The extension now also reaches into MDLA's `copy_exchange` (used to clone the root ghost
  topology per node with fresh buffers, skipping a P2P re-probe). That is another non-public
  surface behind the exact-patch `MultiDeviceLinearAlgebra = "0.0.1"` pin; the Project.toml
  comment records it.
- Known, deliberately out of scope: `adjoint_gather!`'s `β ≠ 0` branch allocates
  `similar(x̄.data)` on **every call** (`src/operators/abstract.jl`), which fires for the second
  term of every `Added` in the adjoint walk, on every device. O(cells), not O(1). Pre-existing and
  single-device — `PreparedForest` already works around it with `adjscratch`. Filed separately;
  filed as #33; the allocation test in `test/partitioning.jl` pins the scaling behaviour rather than the
  absolute number.
