---
slug: pr66-review-decisions
created: 2026-09-08-1104
status: done
---

# Handoff: walk Kyle through the 15 review findings on PR #66

## Goal / why this matters

PR #66 (`feat(distributed): compact diffusion leaf on partition slabs`, branch `feat/distributed-diffusion-leaf` → `main`, single commit `b41d731` on top of `9556a7d`) was reviewed with `/code-review 66` at extra-high effort on 2026-09-08. Fifteen findings survived verification. None has been decided yet. The next agent's job is to present them to Kyle **one at a time** with a recommendation (his global CLAUDE.md rule: progress counter like "3/15", recommendation, then wait for his decision before the next), then apply the accepted fixes on the PR branch.

## Background & current state

- The review ran 10 finder angles, deduplicated 47 candidates, verified with 15 verifiers plus a gap sweep. Two candidates were refuted and are **not** in the list: a supposed GPU closure-capture crash (Adapt's `Function` rule handles it) and an `operator_diagonal` wording nit (it is the codebase's own generic term).
- Things that were traced end to end in both backends and check out, so do not re-review them: the padded-window arithmetic in `_slab_coeff_field`, the `Diffusion` struct field order, the `_push_adjoints` fold, the exchange gating, the `boundary_rhs` lift, and every test-harness reach-in.
- Finding 1 was presented to Kyle (1/15) and he parked the whole queue before answering. Start from finding 1.
- The session's working tree was on `feat/packed-diffusion-regrid`, **not** the PR branch. Check out `feat/distributed-diffusion-leaf` (or `gh pr checkout 66`) before editing anything. Line numbers below refer to commit `b41d731`.
- Local `main` may be behind; the review compared against `origin/main` at `9556a7d`.

## Key files / locations

- `src/distributed.jl` lines 38, 50, 285, 326 (guards, real-eltype rationale, `_slab_field`, `_slab_coeff_field` docstring)
- `src/operators/diffusion.jl` lines 169, 236, 332 (struct docstring, `_has_interface` error text, `apply_adjoint!`)
- `ext/` MDLA extension, line ~181 (slab vector eltype forced to `T = eltype(spacing(g)) <: Real`)
- `test/partitioning.jl` lines 42, 793, 1117, 1188, 1253, 1267
- `test/mdla_gpu.jl` lines 405, 418
- `test/multigpu/mdla_3partition.jl` (not extended by the PR)
- `DESIGN.md` lines 606, 620, 626, 659
- `docs/pages/distributed.qmd` ~182
- `handoffs/2026-07-28-1530-mdla-slice2a-gpu-verification.md` (prior GPU sets still unrun, no slice-2c section)

## Decisions & conclusions

The findings, ranked most severe first, each with the recommendation to present. Severity groupings: 1 is a correctness gap, 2 is a CI coverage hole, 3 is measured perf, 4 through 8 are docs/design accuracy, 9 through 14 are test quality, 15 is wording. Items 6, 7, and 15 are one-word or one-line doc fixes and can be offered as one bulk decision at the end if Kyle prefers.

### 1/15 — `src/distributed.jl:38`. Guards never check that κ lives on the operator's grid. **Recommend: fix.**

`_distributable(D::Diffusion) = _partitionable_coeff(D.κ)` and `_check_one_grid` (which only inspects `operator_grid(D) = D.grid`) never verify `D.κ.grid` matches `D.grid`. A `Diffusion` built through the exported inner constructor with κ on another grid passes every distributed guard. `ScalingOp` is immune because its `operator_grid` is the coefficient's grid. Verified by probe: κ on 12×10 with the operator on 8×6 passes `prepare_distributed(D, 2)`, then `_slab_coeff_field` windows κ with the wrong grid's `local_range`, fills the Dirichlet wall ghost from κ's interior (38.0 beside 37.0 instead of the even mirror 37.0), so every wall-face coefficient is (37+38)/2 instead of 37 with no error. With κ on a smaller 4×6 grid the same call dies with `BoundsError`. The `_slab_coeff_field` docstring at line 326 also wrongly credits `_check_one_grid` with enforcing the ghost-extension precondition that only `diffusion()`/`_extended_coeff` supplies.

Fix: `_grid_mismatch(D::Diffusion, g) = (_same_grid(D.grid, g) && _same_grid(D.κ.grid, g)) ? nothing : D` (the `_grid_mismatch` hook exists at lines 187–197 of the PR commit). Reword lines 325–327 to attribute ghost extension to `diffusion()`. Add a test that `prepare_distributed` rejects the mismatched-κ construction.

**Decision (2026-09-08): fix — done.** `_grid_mismatch(::Diffusion, g)` checks `D.κ.grid` too; docstring reworded; testset builds the mismatched leaf through the inner constructor on a larger and a smaller grid and expects the guard, plus `dist_prepare`, to throw. Commit e6072d2.

### 2/15 — `test/mdla_gpu.jl:405`. Every GPU assertion the PR adds is unreachable in CI. **Recommend: fix (schedule a run, extend the 3-partition file, add a slice-2c section to the existing GPU handoff).**

The new upload testset plus Diffusion entries in five existing testsets are gated on `MFO_TEST_MDLA=true` and `NGPUS_MDLA >= 2`, which no workflow under `.github/workflows/` sets and no runner satisfies. The author states they were not run locally. The prior slices' GPU sets are still unrun per the open `handoffs/2026-07-28-1530-mdla-slice2a-gpu-verification.md`, which gained no slice-2c section. `test/multigpu/mdla_3partition.jl` (whose own comment says the middle slab is where a coefficient slice can be right at one seam and wrong at the other) was not extended with `diffusion` at all. Consequence: a GPU-only fault in the new path (`Adapt.adapt(CuArray, Diffusion(lg, Field{L}(copy(view(...)), lg), avg))` mis-uploading the padded window, or the `_diff_at.(Ref(x.data), Ref(κ), ...)` broadcast failing to compile on device) ships to `main` green. Three unexecuted GPU slices now stack.

**Decision (2026-09-08): fix — done as far as this Mac allows.** `test/multigpu/mdla_3partition.jl` gains upload, forward-parity, and adjoint-identity testsets for `Diffusion`; the GPU handoff gains a slice-2c section naming every gated testset and what a failure in each means. The run itself still needs a ≥2-GPU (and ≥3-GPU) host. Commit 133c658.

### 3/15 — `src/operators/diffusion.jl:332`. Slab adjoint runs the generic masked gather at ~5.7× the forward stencil cost. **Recommend: discuss; likely fix as a follow-up issue rather than in this PR.**

On a partition slab `apply_adjoint!(::Diffusion)` always runs the inherited `adjoint_gather!` engine: a bounds-masked sweep of every padded cell doing 10 `_maskedget`/`checkbounds` reads per cell in 2-D over both ȳ and κ. Measured 30.8 µs per call on a 128×64 slab versus 5.4 µs for the slab's forward `apply!` and 10.9 µs for the full 128² self-adjoint shortcut. This PR's `_distributable`/`_slab_op` additions route Diffusion onto that branch for the first time. A prototype that runs the `@inbounds` forward stencil over `interior(g)`, the masked gather only over the 2N ghost planes, then `fold_bc!` and scales by α reproduces the engine's padded output bit for bit (0/8192 cells differ, α=1.0 and α=0.7) at 8.8 µs, 3.5× faster. Caveat: the path is reachable today only via `_mul_adjoint!`, which the ext labels as existing for tests, so this is an engine cost inherited by the adjoint tests rather than by Krylov `mul!`. Kyle's standing rule (memory: perf regressions are blockers on perf-adjacent PRs) applies, but this is a new path rather than a regression of an existing one.

**Decision (2026-09-08): defer — filed as #77.** Kyle's call; nothing changed in the PR. The issue carries the reviewer's numbers flagged as an in-session prototype to re-measure, the reachability caveat, and the `Val(D)`/Enzyme constraints.

### 4/15 — `src/distributed.jl:50`. The real-eltype restriction is justified with an allocation that is unreachable in production. **Recommend: fix the rationale in all five places.**

The docstring says `_conj_op(::Diffusion)` would "build `conj.(κ.data)` on every call ... per partition per Krylov iteration". But `_push_adjoints` folds `adjoint(Diffusion)`/`adjoint(ScalingOp{Field})` to a conjugated leaf once at setup on the all-physical global grid, no `mul!` on an adjoint-wrapped `MDLAPreparedOperator` exists, and both backends force the slab vector eltype to `T = eltype(spacing(g)) <: Real`, so a complex κ fails in the forward `apply!` with `InexactError` first. The wrong rationale is copied into: this docstring, `_undistributable_reason(::Diffusion)`, `docs/pages/distributed.qmd` ~182, `DESIGN.md` ~626, `test/partitioning.jl` ~1253/1267. DESIGN.md's companion claim that this is "the first production path to reach the leaf's mechanical transpose" is contradicted by the same trace (the transpose runs only under the test-facing `_mul_adjoint!`). The real blocker for a complex κ is the real vector eltype at ext line ~181.

**Decision (2026-09-08): fix — done.** Rationale corrected in the `_partitionable_coeff` docstring, both `_undistributable_reason` strings, the docs table, both DESIGN.md paragraphs (the slice-2b one carried the same claim), and the test comments; the "first production path" sentence now says the transpose is reached only by `_mul_adjoint!`. Commit cef7d6a.

### 5/15 — `src/operators/diffusion.jl:236`. The `_has_interface` error advises forest users into a dead end. **Recommend: fix the message.**

The rewritten error names "forest leaves and partition slabs" as triggers but its one piece of advice, "Build it on the undistributed grid and pass that to prepare_distributed", is valid only for slabs. A `leaf_grid` is a real `CartesianGrid` with `Interface()` on every face (so it reaches this branch), has no undistributed twin, and `prepare_distributed` rejects a `BlockForest` ("requires a CartesianGrid") and is a bare `function prepare_distributed end` stub without the MDLA extension loaded. Only the trailing "tracked in issue #54" clause applies to forest users. Split the advice by case.

**Decision (2026-09-08): fix — done.** Advice split by slab vs forest leaf; both messages now point forest support at #58, not #54. Commit 8a406c6.

### 6/15 — `src/operators/diffusion.jl:169`. Struct docstring claims a forest path that does not exist. **Recommend: fix (one-word tense change). Bulk-groupable.**

Docstring was changed from "the seam the forest and distributed paths will use" to present-tense "the seam the forest and distributed paths use to supply cross-block coefficient ghosts themselves". No forest path exists on this branch: `grep Diffusion` over `src/operators/forest.jl`, `forest_packed.jl`, `linalg.jl`, `BlockForest.jl`, `amr.jl` is empty, `_leaf_op` has no `Diffusion` overload, and `diffusion()` still throws "BlockForest support is tracked in issue #54" in two places. Suggested wording: "the forest path will use and `_slab_op` already uses".

**Decision (2026-09-08): fix — done.** Docstring says `_slab_op` already builds through the seam and the forest path (#58) will. Commit 8a406c6.

### 7/15 — `DESIGN.md:659`. Stale §8 scope note contradicts the PR's own new paragraph. **Recommend: fix (one line). Bulk-groupable.**

The unchanged §8 scope note still ends "`CartesianGrid` only for now — `BlockForest` and distributed slabs are staged in #54", contradicting the PR's new "*Built (issue #57, slice 2c):* the compact flux-form `Diffusion` leaf on slabs" paragraph 54 lines earlier. BlockForest is #58, not #54. The PR correctly updated the parallel "Distributable today" list in `docs/pages/distributed.qmd`; this line was missed.

**Decision (2026-09-08): fix — done.** Commit 74d9a1b.

### 8/15 — `DESIGN.md:620`. The no-communication claim names the wrong precondition. **Recommend: fix the Deferred paragraph to name the κ Interface exchange the next backend owes.**

The new paragraph records the no-communication claim as "legitimate only because κ is *constant through a solve*", but the actual precondition is that the host holds the global κ (`Lh = Adapt.adapt(Array, L)` then `_slab_coeff_field` windows `f.data` of the global coefficient). §10.4's resolved next backend is ImplicitGlobalGrid (multi-node, no rank holds global κ), on which `_slab_coeff_field` cannot run. Neither this hunk, the Deferred paragraph, nor issue #46 records the setup-time κ `Interface` exchange that backend owes. A future IGG implementer reusing `_slab_op(D::Diffusion, lg)` per rank with rank-local κ either crashes on an out-of-range `view` or silently gets zero/mirrored cut-plane ghosts (κ_I/2 face coefficients under `ArithmeticMean`, 0 under `HarmonicMean`). The general form already fits the seams: `Diffusion(lg, _extended_coeff(_slab_field(D.κ, lg), lg), D.avg)` (`fill_coefficient_ghosts!` leaves `Interface` ghosts untouched) plus one `_dist_scatter!`-shaped exchange at prepare time.

**Decision (2026-09-08): fix — done.** The built paragraph names both preconditions (κ constant through a solve; the host holds the global κ); a new *Owed by the next backend* paragraph in DESIGN.md §7 records the setup-time κ `Interface` exchange and the shape that fits the seams; the docs page's twin sentence is corrected; a comment on #46 points at it. Commit f4e083c.

### 9/15 — `src/distributed.jl:285`. `_slab_field` and `_slab_coeff_field` are near-duplicates whose only difference nothing consumes. **Recommend: discuss; lean toward merging into one padded slice.**

`_slab_field` (interior copy + `similar`/`fill!`/zeroed ghosts) and the new `_slab_coeff_field` (offset-correct padded `copy(view)`) compute the same owned window and differ only in ghost contents. `_coeff_values(c::Field) = interior(c)` is ScalingOp's only read, the NaN-poison testset (~793) proves ghost contents are unread end to end, no test asserts zero ghosts, the MDLA upload copies the whole padded field regardless, and the allocation test measures only per-apply work. Cost of keeping both: two functions, two 30-line docstrings, two testsets, and "narrowed, not broken" paragraphs in DESIGN.md and the qmd maintaining a zero-ghost invariant no consumer depends on. The next field-parameter leaf (Advection velocity per the docs table) must choose between two helpers. Also, `_slab_field`'s existing warning that a padded slice "would shift every partition after the first by `halo` planes" describes the naive un-offset slice, not the one the PR implements.

**Decision (2026-09-08): merge — done.** One padded `_slab_field` for both consumers; `_slab_coeff_field` removed; the two unit testsets folded into one cell-by-cell check over a plain field and a `diffusion`-extended κ; DESIGN.md and the docs no longer describe a zero-ghost invariant. Commit 8729f04.

### 10/15 — `test/partitioning.jl:1188`. Adjoint allocation-scaling list lacks a Diffusion entry. **Recommend: fix (one line).**

The PR added `laplacian(g) * diffusion(g, ...)` to the forward (`dist_mul!`) list of "walk allocations do not scale with grid size" but not to the adjoint (`dist_adjoint!`) list four lines below. Measured 0 B per call today at every size 16²–128² (β=0 and β=1), so this is a coverage gap not a regression. One entry such as `g -> diffusion(g, set!(scalar_field(g), diff_coeff_fun)) + laplacian(g)` closes it.

**Decision (2026-09-08): fix — done.** Commit b3845bd.

### 11/15 — `test/partitioning.jl:793`. Unscoped "ghosts are never read" comment and testset name now false for Diffusion. **Recommend: fix (rename/rescope to `ScalingOp`).**

The pre-existing comment and testset name ("it is read pointwise at the cell being written, so its ghosts are never consulted" / `@testset "a localized coefficient's ghosts are never read"`) assert a general invariant that this PR falsifies for Diffusion and directly negates 300 lines later in "a localized diffusion coefficient's cut-plane ghosts ARE read". The body walks only `_scaling_leaves`, so no assertion is wrong at runtime. The PR narrowed the same claim in `_slab_field`'s docstring and DESIGN.md but left this one unscoped.

**Decision (2026-09-08): fix — done.** Testset renamed to `a localized ScalingOp coefficient's ghosts are never read`, comment rescoped. Commit b3845bd.

### 12/15 — `test/mdla_gpu.jl:418`. GPU upload oracle retypes the implementation's window formula. **Recommend: fix (replace with the one-liner).**

The only test inspecting the uploaded slab coefficient builds its oracle as `win = ntuple(2) do dd; lr = locals[d].local_range[dd]; first(lr):(last(lr) + 2 * halo_width(locals[d])[dd]) end`, which is `_slab_coeff_field`'s own window formula retyped (the offset term vanishes because `mdla_grid` is unpartitioned). This contradicts its CPU twin's explicit rationale ("Spelled out cell by cell rather than as the same `ntuple` the implementation uses, so a wrong window cannot agree with a wrong oracle"), and its scaffold near-duplicates the ScalingOp upload testset at 381–399. Since the window is already proven cell by cell on CPU, the extension's own contribution (upload + shape) is tested exactly by `@test Array(Dd.κ.data) == MatrixFreeOperators._slab_coeff_field(Dg.κ, locals[d]).data`.

**Decision (2026-09-08): fix — done.** Oracle is `_slab_field(Dg.κ, locals[d]).data`. Commit 133c658.

### 13/15 — `test/partitioning.jl:42`. `_diffusion_leaves` is a structural copy of `_scaling_leaves`. **Recommend: fix (collapse to one predicate-driven walk).**

Seven identical combinator methods, one leaf method changed, justified by a comment claiming the differing field names `S.coeff` vs `D.κ` force separate functions. Neither walk touches either field (access happens at call sites 802, 847, 1122), and only the `Diffusion` and `PreparedComposed` methods are exercised by its single call site. A forgotten combinator in one copy makes that walk return `()` so the poison loops pass vacuously. A 5-line `_leaves(L, keep)` with `Union{Scaled,AdjointOp,PreparedAdjoint}` / `Union{Added,Composed,PreparedComposed}` methods plus `Base.Fix2(isa, T)` predicates replaces both with call sites unchanged.

**Decision (2026-09-08): fix — done.** `_leaves(L, keep)` with `Union` combinator methods and `Base.Fix2(isa, T)` predicates; call sites unchanged. Commit 7ff4532.

### 14/15 — `test/partitioning.jl:1117`. Ghost-poison testset hand-derives Interface planes the prepared operator already carries. **Recommend: fix.**

The testset hand-derives each slab's Interface ghost planes (`bc[1] isa Interface && append!(planes, 1:h)`; `bc[2] isa Interface && append!(planes, (h + nn + 1):(h + nn + h))`) when `D = dist_prepare(L, g, 2)` built five lines earlier already carries them as `D.root.plans[p]` from `_slab_ghost_layout`, and the "ghost layout invariants" testset (373–394) already proves that set equals this formula. The padded-plane index map is now spelled in three places (core, geometry test, this test), the decoupling the file's own `halo_plane_view` comment at 18–19 warns against. Replacement: `for Dl in _diffusion_leaves(...), (_, pl) in D.root.plans[p]; fill!(halo_plane_view(Dl.κ, pl), 0); end`.

**Decision (2026-09-08): fix — done.** Planes come from `D.root.plans[p]`. Commit b3845bd.

### 15/15 — `DESIGN.md:606`. "whitelisted" violates the inclusive-terminology rule. **Recommend: fix (one word, "allowlisted"). Bulk-groupable.**

PR-added prose "the first whitelisted operator whose coefficient is read at a *neighbour* rather". The term is already entrenched on main (13 pre-existing occurrences across src/, test/, DESIGN.md, docs/) and the PR's net change is +1/−1, so this is one word rather than a pattern the PR introduces. Optionally file a separate chore to sweep the 13 existing occurrences.

**Decision (2026-09-08): fix — done.** One word; the 13 pre-existing occurrences are untouched and no chore was filed. Commit 74d9a1b.

## What's left / next steps

All fifteen decided on 2026-09-08: fourteen applied on the PR branch (commit per finding, or per file for the one-liners) and finding 3 deferred to #77. Remaining after this doc:

1. The GPU-gated testsets (finding 2) still need a run on a ≥2-GPU host, and the 3-partition file on ≥3 — see the slice-2c section of `handoffs/2026-07-28-1530-mdla-slice2a-gpu-verification.md`. Three unexecuted GPU slices now stack.
2. The CPU suite was not run for these fixes (standing instruction). CI on the PR is the check.
3. Optional chore: sweep the 13 pre-existing "whitelist" occurrences (finding 15) — not filed.

## Gotchas / constraints

- **Do not dump all 15 at once.** Kyle's CLAUDE.md is explicit: one item at a time with a progress counter and a recommendation, wait for the decision. Trivial homogeneous items may be grouped.
- The review's output contract said not to call `ReportFindings`; present as text.
- Line numbers are for commit `b41d731`. If the PR has new commits, re-locate by symbol, not line.
- Finding 3's perf numbers came from a prototype the reviewer wrote in-session, not from anything on the branch. Re-measure before claiming the 3.5× in a commit message.
- Finding 2's GPU tests cannot be run on this Mac (no CUDA, `NGPUS_MDLA >= 2`). The existing GPU-verification handoff describes the runner setup that was never executed.
- Enzyme rule constraints and the two-array `ntuple` vectorization trap in the repo CLAUDE.md apply if finding 3 is implemented: unroll by recursion over `Val(D)`, not `ntuple(Val(N)) do d`.
- No secrets were involved in this session; nothing redacted.
