---
slug: paper-driven-mfo-improvements
created: 2026-07-21-0753
status: open
---

# Handoff: multigrid/preconditioner improvements mined from two papers

## Goal / why this matters

Two papers were read against this package to find improvements. The headline
conclusion: **`operator_diagonal` does not work on a `BlockForest`, so there is no
preconditioner of any kind available for an adapted-forest solve today** — users are on
unpreconditioned GMRES. Both papers independently point at this, and one of them
quantifies the cost. Fixing it is the keystone that also unblocks forest multigrid
(already named as deferred in `DESIGN.md:772`).

Nothing here has been implemented. This is a findings-and-scope brief; the work is
un-started.

## Background & current state

**The papers**

1. **Feder, Heltai, Kronbichler & Munch, *Matrix-free implementation of the non-nested
   multigrid method*** — arXiv:2412.10910 (deal.II). Non-nested geometric MG where each
   level is an independently meshed/partitioned triangulation. Transfer is nodal
   interpolation `P[i,j] = φⱼᶜᵒᵃʳˢᵉ(xᵢᶠⁱⁿᵉ)`, applied without assembly via a one-time
   geometric search plus tensor-product contraction of tabulated 1-D bases. Smoother is
   degree-3 Chebyshev in D⁻¹A with **λmax from 12 Lanczos iterations**. Converges in 3–6
   iterations independent of refinement level and polynomial degree.
2. **Sachetto Oliveira et al., *Performance evaluation of GPU parallelization, space-time
   adaptive algorithms, and their combination for simulating cardiac electrophysiology***
   — Int J Numer Meth Biomed Engng 2018;34:e2913. Monodomain via Godunov splitting
   (adaptive-Δt explicit Euler/Heun for ODEs, backward Euler + Jacobi-preconditioned CG
   for diffusion) on a block-adaptive hex mesh.
   Local copy at `~/Downloads/Numer Methods Biomed Eng - 2017 - Sachetto Oliveira - …pdf`.

**Numbers worth keeping** (they justify the ranking below):

| Source | Finding |
|---|---|
| Paper 2, Table 1 | Adaptivity degrades conditioning: avg CG iterations **15 (uniform) → 38 (adaptive, unpreconditioned) → 16 (adaptive + Jacobi)**. Jacobi was chosen *specifically* because it needs no rebuild after remeshing. |
| Paper 2, Table 3 | Speedups do **not** multiply. GPU-over-OpenMP: **6.5× fixed mesh → 3.0× (TA) → 2.4× (SA) → 2.1× (SA+TA)**. Cause: each refine/derefine ships state across the bus; per-cell adaptive Δt diverges warps. |
| Paper 2, Table 5 | Matrix reassembly after regrid = **7.0%** of runtime. Matrix-free removes this outright — a citable win for this package's premise. |
| Paper 1 | Non-nested transfers cost ~**10× more** than classical nested transfers; authors recommend nested transfers wherever a hierarchy exists. |

**Package state** (verified by reading, 2026-07-20): MG on uniform grids is built and
mature — `MultigridPreconditioner`/`MultigridSolver`, Jacobi + Chebyshev smoothers,
rediscretized level operators, P/R as first-class operators, dense-LU coarsest solve. AMR
(`BlockForest`, 2:1 balance, coarse–fine transfer, `regrid!`) is built end-to-end,
including the packed GPU phase. The gaps below are the delta.

## Key files / locations

- `src/operators/diagonal.jl:47` — `operator_diagonal(::Laplacian)` **throws on `Interface`
  faces**, i.e. on every forest leaf. The blocker.
- `src/operators/diagonal.jl:63–77` — `_diag_bc_adjust!` / `_diag_bc_face!`; the pattern a
  forest method should follow. Note `:72` — `Periodic` contributes off-diagonal only.
- `src/multigrid.jl:63–86` — `_smoother_state(::Chebyshev, …)`; **10 power iterations**,
  `hi = T(11)/10 * λ` at `:83`, `lo = hi/4`.
- `src/multigrid.jl:282` — `_build_coarsest`; `n` matvecs + dense `n×n` LU.
- `src/multigrid.jl:238` — `_mg_grids`; stops coarsening on odd cell counts.
- `src/multigrid.jl:364` — throws `"multigrid on a BlockForest is not yet supported"`.
- `src/schedule.jl` — `_cf_normal_weights` = Martin–Cartwright `(5/21, 5/6, −1/14)`; the
  `ExchangeSchedule`'s `interp` descriptors identify coarse–fine cells.
- `src/operators/prolongation.jl` / `restriction.jl` — `_validate_transfer` rejects
  `Interface` faces (blocks forest MG transfers).
- `src/Grids.jl:171` — `coarsen`, errors on odd cell counts.
- `examples/monodomain_amr.jl` — Aliev–Panfilov S1–S2 on an adaptive forest; fully explicit
  Euler at `DT = 0.006`, `REGRID_EVERY = 20`, CPU-only, not packed.
- `DESIGN.md:748–774` — the multigrid section and its deferred list.
- Fuller write-up (not committed): `~/.claude/plans/read-these-2-papers-concurrent-globe.md`

## Decisions & conclusions

- **Do NOT build non-nested transfer machinery** (geometric search, ArborX,
  reference-coordinate inversion). Paper 1's own conclusion rules it out when a nested
  hierarchy exists, and a `BlockForest` *is* nested. We inherit the cheap case; the
  existing transfer weights apply. This is the single most useful thing paper 1 settles.
- **Do NOT add a time-integration / operator-splitting / IMEX module.** `DESIGN.md` keeps
  time stepping out of the package deliberately (RHS closure + `jac_prototype` only) and
  the adaptive solve loop is deliberately user code. Paper 2's Godunov splitting belongs
  in `examples/monodomain_amr.jl` as an *example* change — and only after item 2 below,
  since the implicit diffusion solve needs a forest preconditioner to pay off.
- **Do NOT add a cell-model / reaction-operator abstraction** — one use case, rule of three.
- **Forest MG hierarchy design is genuinely open** and was deliberately not planned: MG
  levels could be the forest's own refinement levels (composite/FAC-style, levels covering
  different subdomains) or uniformly coarsened whole forests. Needs its own brainstorm;
  don't guess.
- The packed-forest `regrid!`-errors contract (`src/amr.jl:94`, unpack → regrid → re-pack)
  is the same host↔device round trip paper 2 measured. **The decision still looks correct**
  (regridding is host-side leaf surgery; an automatic round trip would hide the transfer).
  The gap is only that the cost is unmeasured here.

## What's left / next steps

Ranked; 1 is the keystone. Kyle asked to go through these one at a time to decide scope —
**that decision has not been made yet**, so confirm before implementing.

1. **`operator_diagonal` on `BlockForest` leaves.** Add the forest method; return a
   `BlockField`. Unblocks everything else and is paper 2's exact solver choice.
2. **Standalone Jacobi/Chebyshev preconditioner usable on a forest.** Falls out of (1).
   Today `Jacobi`/`Chebyshev` in `src/multigrid.jl` exist only as smoother configs consumed
   by `_smoother_state`; lift the diagonal-based application into something exposing
   `mul!`/`size`/`eltype` (the duck-typed contract `MultigridPreconditioner` already meets).
3. **Chebyshev λmax by Lanczos, not power iteration** (`src/multigrid.jl:63–86`). ~30 lines.
4. **Coarse-grid solver blowup** (`src/multigrid.jl:282`). Below a threshold keep dense LU;
   above it use a **fixed iteration count** of preconditioned CG.
5. **Forest multigrid** — design brainstorm first, not implementation.
6. *(optional)* Quantify regrid + re-pack as a fraction of an adaptive GPU solve in
   `benchmark/gpu.jl`; document a regrid-cadence guideline.

## Gotchas / constraints

- **The correctness trap in item 1.** The coarse→fine ghost fill's Martin–Cartwright
  parabola `(5/21, 5/6, −1/14)` has its **first weight reading the fine block's own first
  interior cell**, so it feeds back into that cell's own diagonal exactly the way a
  Dirichlet mirror does. Miss that term and the diagonal is silently wrong at every
  refinement boundary. **A uniform-forest test would still pass** — gate on a *refined*
  forest, comparing against `materialize`d columns (`test/test_utils.jl`) so it must match
  `diag(A)` exactly at interfaces.
- **Item 2, symmetry:** an adapted forest is nonsymmetric — `isselfadjoint(laplacian(bf))`
  is already `false`, queried live. Chebyshev is SPD-only; gate it and default to Jacobi on
  a non-uniform forest.
- **Item 4, fixed count not tolerance:** a tolerance-based coarse solve makes the
  preconditioner a non-fixed operator and breaks the `Krylov.cg` validity
  `MultigridPreconditioner`'s docstring currently promises.
- **Item 4 severity:** you cannot escape the blowup by passing `levels=` — `coarsen` errors
  on odd cell counts, so the coarsest level can't be reduced further. A 100³ grid stops at
  25³ = 15,625 DOFs → 15,625 matvecs and a ~1.95 GB dense matrix. 1000² hits the same
  number in 2D. Ordinary grid sizes, not exotic ones.
- **Item 3 failure mode is silent.** The Rayleigh quotient approaches λmax *from below* and
  converges slowly into a clustered spectrum; if `hi` undershoots, the Chebyshev polynomial
  amplifies the highest-frequency modes instead of damping them, showing up only as
  degraded MG convergence. Keep the existing checkerboard start vector — it's a good
  choice, rich in the modes that matter.
- Per repo convention: **do not run tests unless explicitly asked.** Benchmark CI
  (AirspeedVelocity, per-PR) shows ±15–20% spurious deltas on shared runners; treat a
  regression on these phases as a blocker but verify before believing it.
- No secrets are involved in this work; nothing was redacted from this doc.
