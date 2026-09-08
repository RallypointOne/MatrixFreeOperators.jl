---
slug: mdla-slice2a-gpu-verification
created: 2026-07-28-1530
updated: 2026-07-29
status: open
---

> **Update 2026-07-29 — this is now a slice 2a *and* 2b handoff.** Branch
> `feat/mdla-slice2b` landed the rest of #31 (`Field` coefficients, distributed
> `boundary_rhs`, global-index-exact slab coordinates) on top of 2a, and its GPU
> testsets went into the *same* two gated files. Slice 2a's sets were still unrun
> when 2b landed, so there is **one** GPU run to do, not two. Read the whole
> document, then the "Slice 2b" section at the bottom for the extra testsets.

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

---

# Slice 2b (added 2026-07-29): `Field` coefficients, distributed `boundary_rhs`

`feat/mdla-slice2b` closes the rest of #31. Same drill, same two gated files, same hardware
ladder. CPU side is green and mutation-verified; the GPU testsets are again **written but unrun**.

## What changed

Three things, and the third is the one to be careful about.

1. **`Field` coefficients.** `_slab_op` / `_slab_field` (`src/distributed.jl`) slice a coefficient's
   interior onto each slab. No exchange of its own is needed — `ScalingOp` reads the coefficient
   *pointwise at the cell being written*, which a CPU testset proves by poisoning every localized
   coefficient's ghosts with `NaN` and demanding the answer not move. Slicing happens on the **host**
   and the slab is then uploaded (`Adapt.adapt(Array, L)` before the per-device loop), so a
   coefficient the user had already moved to device 0 never becomes a cross-device copy.
   `DistLeaf`/`DistAdjoint` now hold a per-partition operator *vector*, narrowed with `identity.` so
   shared leaves stay concretely typed — the slice-1/2a hot path is unchanged, which the extended
   allocation-scaling test pins.
2. **Distributed `boundary_rhs`.** `_dist_boundary_rhs!` is a third walk over the same `DistNode`
   tree. Leaf lifts are slab-local for free (the inhomogeneous fill is a no-op on `Interface`); only
   a `Composed` lift needs the mid-tree exchange, and it reuses the node's own. New surface:
   `boundary_rhs(P)`, `set!(::MultiDeviceVector, P, fun)`, `assemble_rhs(P, f)`, `local_grids(P)`.
3. **Slab coordinates are now global-index exact.** A slab keeps the **global** `extent` and carries
   its position in `local_range` alone; `cell_center` evaluates at the global cell index. Without
   this, a slab-local `set!` rounds twice and drifts by an ulp, and the assembled RHS — hence the CG
   iteration count — depends on `nparts`. Verified non-vacuous: on the test extents, 2 cells per
   configuration actually differ under the old formula.

## New GPU testsets to watch, and what a failure in each would mean

`test/mdla_gpu.jl`:

- **`the coefficient is sliced and uploaded per partition`** — shapes and values of each
  partition's `CuArray` coefficient. A failure means the global field was uploaded instead of the
  slab, which would still broadcast on partition 1 and be wrong on the rest.
- **`Field coefficient: 2-partition forward parity`** — bitwise vs 1 partition. A mismatch means the
  slice window is off; check `lg.local_range` against the padded window first, and remember every
  coefficient in these tests varies **along the cut dimension** on purpose.
- **`Field coefficient: adjoint identity and transpose structure`** — ⟨Lx,y⟩ = ⟨x,Lᵀy⟩ plus dense
  structure with a localized coefficient in the middle of it.
- **`distributed boundary_rhs parity`** — bitwise vs `flatten(boundary_rhs(L, g))`, with a
  *different* inhomogeneous value on every face. A failure on the physical-cut grid but not the
  periodic one points at a slab applying a cut BC to its `Interface` face.
- **`Krylov.cg with an inhomogeneous RHS assembled distributed`** — the acceptance test. Asserts
  `gather(assemble_rhs(P, fun)) == bflat` bitwise *and* `allequal(niters)`. If the bitwise RHS
  check fails while `≈` passes, suspect the `cell_center` change, not the lift.
- **`assemble_rhs from per-partition fields`** — the `local_grids` escape hatch.

`test/multigpu/mdla_3partition.jl` adds 3-partition versions of all four (coefficient forward
parity, coefficient adjoint identity, `boundary_rhs` parity, inhomogeneous RHS + CG). The middle
slab is the only place a coefficient slice can be right at one seam and wrong at the other, and the
only place a lift can leak a cut-dimension BC value onto an `Interface` face.

## The one genuinely new GPU code path, and its pre-agreed fallback

`set!(::MultiDeviceVector, P, fun)` calls `set!(::Field, fun)` on a **device** field, which nothing
in this repo has done before — every existing GPU test builds a field on the CPU with `set!` and then
`Adapt.adapt`s it. So this is the one place where "CPU-green" carries less weight than usual.

It should work: `set!` is `interior(f) .= fun.(cell_center.(Ref(f.grid), interior(f.grid)))`, an
adapted `CartesianGrid` is isbits (BCs and `local_range` included), `cell_center` is pure arithmetic
returning an `SVector`, and `CartesianIndices` is a valid GPU broadcast argument. But it is unrun.

If it fails — a `Ref`/`Adapt` complaint, or a scalar-indexing error out of the broadcast — do **not**
reach for `CUDA.allowscalar`. The pre-agreed fallback is to assemble each slab on the host and
upload, which costs one slab-sized host buffer per call and is fine for a once-per-solve assembly
(the point of the slice is avoiding a *global* array, not any host memory at all):

```julia
hosts = local_grids(P)
_dist_map!(P.ctx) do d
    interior_to_flat!(x.partitions[d], Adapt.adapt(CuArray, set!(scalar_field(hosts[d], T), fun)))
end
```

Note the user's `fun` must be GPU-compatible under the current implementation (no captured host
arrays); the fallback removes that constraint too, so if a user hits it, that is the fix rather than
a bug.

## Decisions & conclusions (don't relitigate)

- **Guards run once, on the global tree, before `_slab_op`.** Load-bearing, not incidental: after
  localization a `ScalingOp` reports the slab grid from `operator_grid` while a sibling `Laplacian`
  still reports the global one, so re-checking a localized tree would reject it. Commented at the
  call site.
- **`_same_grid` is structural, not `===`.** Not because `===` is wrong today — a `CartesianGrid` is
  isbits, so `===` *is* value equality — but because that is a property of the current field set. The
  moment `topology` carries a real object, `===` silently becomes identity. It also deliberately
  ignores `device`, so a host guard accepts a device-adapted twin, and deliberately compares
  `local_range`, so a slab never equals its global grid.
- **`_check_one_grid` is separate from `_distributable`.** A coefficient on the wrong grid is only
  visible from the tree, not from any single operator, so it cannot be a `_distributable` method.
- **Complex `Field` coefficients stay rejected.** `adjoint_operator(::ScalingOp{<:Field})` builds
  `conj.(κ.data)` per call — a full array per partition per Krylov iteration.
- **`Advection` stays rejected**, deliberately, even though its velocity is also read pointwise.
  Out of scope for #31; the guard and its reason are unchanged.
- **`local_grids` returns HOST grids.** A field allocated on a device grid would land on whichever
  device happened to be current and then be read from a different one — the cross-device copy MDLA's
  P2P probe exists to guard. `assemble_rhs` adapts each slab inside its own device context.
- **`zs` is transient, not a field on `MDLAPreparedOperator`.** The lift is a once-per-solve
  assembly; a permanent padded field per device is memory a homogeneous problem should not pay.
- **The lift reuses each `Composed` node's own `tmps`/`xch`.** Safe because they are never shared
  between nodes and the lift never runs concurrently with a `mul!` — the same single-threaded
  contract `prepare` already carries. A CPU testset pins that a lift between two `mul!`s changes
  nothing and vice versa, because a violation would surface as stale physical ghosts on the *next*
  solve rather than this one.
- **Corner ghosts of the inhomogeneous field differ between a slab and the global grid**, because the
  dimension-`N` pass that would overwrite them is a no-op on an `Interface` face. Harmless: every
  whitelisted leaf steps ±1 along one axis at a time, so no stencil reads a corner. A diagonal or
  wider stencil would break this silently — noted in `_dist_lift_scratch`'s docstring, and the
  `zs`-exactness testset compares transverse-*interior* positions only for exactly this reason.

## Verified so far (no GPU)

- **CPU suite: 5317 pass / 0 fail / 3 broken** (the three env-gated skips). Baseline before 2b was
  4830.
- `test/partitioning.jl` alone: 846 pass (was 359 after 2a).
- Mutation-verified, so these are regression tests rather than decoration:
  - dropping the zero-write in `_dist_boundary_rhs!(::DistAdjoint)` fails **`boundary_rhs parity`**
    (6 assertions), and nothing else;
  - the coefficient slice window and the lift's mid-tree exchange each carry an in-suite negative
    control that asserts the answer *changes* when suppressed;
  - the `cell_center` fix was shown non-vacuous numerically before relying on it.
- The extension **precompiles** in a side env with MDLA + CUDA + Krylov dev'd in, all four new entry
  points attach, and the guards were exercised through the real `prepare_distributed`: field
  coefficients now pass, while `Advection`, a cross-grid coefficient, rank changers and transfer
  operators are rejected with messages naming the reason.

# Slice 2c (added 2026-09-08): the compact `Diffusion` leaf on slabs

`feat/distributed-diffusion-leaf` (PR #66) closes #57. Same drill, same gated files, same hardware ladder. CPU side was green on the PR's own commit `b41d731`; the GPU testsets are — for the third slice running — **written but unrun**. Three unexecuted GPU slices now stack (2a, 2b, 2c), and the PR's review (`handoffs/2026-09-08-1104-pr66-review-decisions.md`, finding 2) rated that the PR's most severe gap after the κ-grid guard. Schedule the run.

## What changed

One thing, in two halves. `_slab_field` (`src/distributed.jl`) now slices the **padded** window of a global coefficient for every field parameter — offset-correct, slab padded index `p` is global padded index `first(local_range[d]) - 1 + p` — so an `Interface` ghost lands on the neighbour's interior κ, a wall ghost on the even mirror `diffusion` already applied, a periodic cut on the wrap. `_slab_op(::Diffusion)` builds the slab leaf through the **inner** constructor from that window; `diffusion(g, κ)` itself still refuses `Interface` faces. No new exchange and no new backend seam: the upload is `Adapt.adapt(CuArray, Diffusion(lg, <padded window>, avg))`, the same shape as the ScalingOp upload one slice earlier.

The review's fixes on top of the PR commit: `_check_one_grid` now also checks κ's grid for a `Diffusion` (the inner constructor accepted a κ from anywhere), the two slice helpers became one, and the real-eltype rationale was corrected — see Decisions.

## New GPU testsets to watch, and what a failure in each would mean

`test/mdla_gpu.jl`:

- **`the diffusion coefficient is uploaded with its cut-plane ghosts`** — the whole padded slab against `_slab_field(Dg.κ, locals[d]).data`, plus `!any(iszero)`. The window is proven cell by cell on CPU, so a failure here is the upload: a shape, or a `view` copied on the wrong device.
- **`newly distributable operators are accepted`** — `diffusion(g, κ)`, its `HarmonicMean` twin, `laplacian * diffusion`, and `2.0 * diffusion + identity_op()` go through `prepare_distributed`. A throw here is a guard or a compile failure in the leaf's device broadcast, not numerics.
- **`Field coefficient: 2-partition forward parity`** — bitwise vs 1 partition for the same four. A mismatch on partition 2 only is the padded window off by `halo`; on both partitions, the device stencil.
- **`Field coefficient: adjoint identity and transpose structure`** — `diffusion(g, κ)` and `laplacian * diffusion` through `_mul_adjoint!`, plus dense structure on the small grid. This is the first GPU execution of `apply_adjoint!(::Diffusion)`'s `adjoint_gather!` branch with a coefficient array: a failure the CPU twin does not show is the masked gather on device.
- **`distributed boundary_rhs parity`** and **`Krylov.cg with an inhomogeneous RHS assembled distributed`** — `diffusion` and `laplacian * diffusion` entries in the lift, and `-diffusion` under both averagings as the SPD operator for CG. Bitwise RHS and `allequal(niters)`, as in 2b.

`test/multigpu/mdla_3partition.jl` (≥3 GPUs) gains **`3-partition diffusion coefficient upload`**, **`3-partition diffusion forward parity`**, and **`3-partition diffusion adjoint identity`**. The middle slab reads κ across both cut faces from two different owners: a padded window right at one seam and wrong at the other is visible here and nowhere with 2 partitions.

## The one genuinely new GPU code path

The leaf's forward stencil is a broadcast over **two** arrays (`x.data` and κ). `test/device.jl` lists `diffusion` in its device-parity ops behind `MFO_TEST_GPU`, so it may have compiled on CUDA before; nothing in CI has run it, and the slab adjoint — `adjoint_gather!` with a κ-capturing closure — has never run on a device at all. If either fails to compile on device, the fix is a KernelAbstractions `@kernel` for the leaf (DESIGN §1.A's escape hatch), not a change to the slicing.

## Decisions & conclusions (don't relitigate)

- **The slice-2b rationale for rejecting complex coefficients (above) was wrong**, and the review corrected it everywhere it was written. `_push_adjoints` folds `adjoint(ScalingOp)`/`adjoint(Diffusion)` to the conjugated leaf once at setup; nothing rebuilds `conj.(κ)` per Krylov iteration. The real blocker is that both backends type every slab and flat vector from `eltype(spacing(g))`, so a complex product has nowhere to land. The guard stays; the reason changed.
- **The slab adjoint is reached only through `_mul_adjoint!`.** A Krylov `mul!` gets the folded conjugate leaf, which for real κ runs forward. So the `adjoint_gather!` cost on slabs (~5.7× the forward stencil on CPU, the reviewer's measurement) is a test-path cost today — tracked in #77, not fixed in the PR.
- **One padded slice, not two.** `_slab_coeff_field` was merged into `_slab_field`; `ScalingOp` takes the padded window too and its ghosts are inert (NaN-poisoned in a CPU testset).
- **κ's grid is guarded.** `_grid_mismatch(::Diffusion, g)` checks both `D.grid` and `D.κ.grid`; the inner constructor is the only way to get them to differ, and a CPU testset does exactly that with κ on a larger and on a smaller grid.

## Verified so far (no GPU)

- The CPU suite was **not** re-run for the review fixes (standing instruction: no test runs unless asked). Every edited Julia file parses. The review's fixes add one CPU testset (κ on another grid), merge two into one, add one adjoint allocation entry, and reword comments.
- The reviewer verified the κ-grid hole by probe on `b41d731`: κ on 12×10 with the operator on 8×6 passed `prepare_distributed(D, 2)` and produced (37+38)/2 wall-face coefficients instead of 37 with no error; κ on 4×6 died with `BoundsError`. Both constructions now throw `ArgumentError` naming `Diffusion` and "different grid" at the guard.
