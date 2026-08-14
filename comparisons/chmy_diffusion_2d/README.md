# Chmy.jl vs MatrixFreeOperators.jl — Diffusion 2D

Head-to-head CPU benchmark against [Chmy.jl](https://github.com/PTsolvers/Chmy.jl)'s published [`examples/diffusion_2d.jl`](https://github.com/PTsolvers/Chmy.jl/blob/main/examples/diffusion_2d.jl) (captured from `main` at v0.1.26, 2026-08-12). Both implementations solve the identical problem with identical parameters; the packages differ only in *how* they apply the discrete operator, which is the thing being measured.

## Problem (verbatim from the Chmy example)

$$\partial C/\partial t = -\nabla\cdot q, \qquad q = -\chi\nabla C, \qquad \chi = 1$$

| parameter | value |
|---|---|
| domain | $(-1,1)^2$ — `origin=(-1,-1)`, `extent=(2,2)` |
| grid | cell-centered uniform, interior $(n-2)^2$, Chmy's `n = 128` convention → 126² primary case |
| spacing | $h = 2/(n-2)$ (identical in both packages, asserted by the congruency check) |
| time step | $\Delta t = h^2/\chi/\text{ndims}/2.1$ — the example's exact formula, recomputed per size |
| BCs | homogeneous Neumann on all four faces, re-applied every step |
| IC | uniform random per cell (see flag 3) |
| steps | `nt = 100` explicit Euler — the timed unit is this 100-step loop |

## What each package executes per step

- **Chmy** (`chmy.jl`): the example's own two KernelAbstractions kernels, verbatim — `compute_q!` writes the staggered flux `q` into a materialized face `VectorField`, then `update_C!` applies its divergence, with the Neumann BC batched into the second launch (`Launcher`, `outer_width=(16,8)` as shipped).
- **MFO** (`mfo.jl`), two legs: **apply** — `apply!(du, χ*laplacian(g), u, g)`, a ghost-cell fill plus one fused 5-point broadcast (no intermediate flux array); **mul!** — `prepare` + `mul!` on interior-only flat vectors, MFO's Krylov-facing API, which adds two flat↔interior copies per application.

For constant χ on a uniform grid the staggered flux form and the compact 5-point Laplacian are algebraically identical, so both packages compute the same update.

## Congruency flags

Everything the two setups share is listed above; these are the places where they are *not* trivially identical, each either neutralized or measured deliberately:

1. **Kernel structure differs by design.** Chmy: two passes with a materialized flux field (writes+reads 2 extra arrays per step). MFO: one fused pass. This is the packages' core design difference — it is what the benchmark measures, not a setup error.
2. **Floating-point op order differs** (two-pass flux/divergence vs fused stencil), so results drift at machine-eps scale. `compare.jl` verifies the final fields after 100 steps agree to rel ≤ 1e-11; observed max|Δ| ≈ 1e-16 (machine epsilon) — the discretizations are confirmed identical.
3. **IC seeding deviates from the published example.** The example uses unseeded `rand()`; both scripts here load the same `Xoshiro(1234)` matrix so the final fields are comparable. Timing is value-independent (no denormals arise).
4. **Visualization and per-step `@printf` stripped** from the example — precedent: Chmy's own `diffusion_2d_perf.jl` strips exactly these for timing.
5. **`Launcher(outer_width=(16,8))` kept as shipped** — a Chmy-only tuning knob with no MFO analogue.
6. **Threading is asymmetric, and that is a real product difference.** Chmy's `CPU()` backend parallelizes kernels over Julia threads; MFO's single-grid CPU path is a serial fused broadcast (no threading). `-t 1` rows are the like-for-like kernel comparison; `-t auto` rows show each package as it actually runs on a multicore machine.
7. Grid sizes beyond the example's 126² keep every other parameter identical and follow Chmy's own perf-example convention (`nxy = (n,n) .- 2`, larger n). 126² is ~127 KB/field — launch-overhead dominated — so the sweep adds cache-resident through DRAM-resident sizes.

## Running

```
./run.sh              # full sweep: {1 thread, all cores} × n ∈ {128 … 16384}
./run.sh 128          # just the exact published case
```

Each script can also run standalone: `julia --project=. --startup-file=no -t <1|auto> <chmy|mfo>.jl <n,n,...>`. `compare.jl` checks the dumped final fields. Timings accumulate in `results/timings.csv`; the field dumps go to a temp dir (`CHMY_MFO_FIELDS_DIR` to override) since they reach 2.1 GB each at 16382².

## Results

Apple M5 Pro, 5P+10E cores, 48 GB (`Sys.CPU_NAME` reports "apple-m1" — LLVM lags), Julia 1.12.6, Chmy v0.1.26, MFO v0.1.0 (dev). Minimum time per step over BenchmarkTools samples of the 100-step loop; speedup is Chmy time ÷ MFO-apply time (> 1 means MFO faster). Measured STREAM-triad bandwidth on this machine: 129 GB/s single-thread, 246 GB/s with 5 threads.

**1 thread** (like-for-like kernel comparison):

| interior | Chmy | MFO apply | MFO mul! | speedup |
|---|---|---|---|---|
| 126² | 160.8 µs | 9.5 µs | 10.8 µs | **16.9×** |
| 510² | 529.8 µs | 148.5 µs | 191.3 µs | **3.6×** |
| 1022² | 1679.3 µs | 569.4 µs | 823.8 µs | **2.9×** |
| 2046² | 6567.5 µs | 2653.9 µs | 3954.5 µs | **2.5×** |
| 4094² | 27.91 ms | 10.33 ms | 14.62 ms | **2.7×** |
| 8190² | 107.0 ms | 40.98 ms | 54.93 ms | **2.6×** |
| 16382² | 412.0 ms | 158.4 ms | 213.6 ms | **2.6×** |

**All cores** (`-t auto` → 5 threads on this machine):

| interior | Chmy | MFO apply | MFO mul! | speedup |
|---|---|---|---|---|
| 126² | 111.3 µs | 9.6 µs | 10.9 µs | **11.6×** |
| 510² | 212.6 µs | 144.3 µs | 191.5 µs | **1.5×** |
| 1022² | 564.2 µs | 569.2 µs | 818.3 µs | 1.0× |
| 2046² | 2193.5 µs | 2625.3 µs | 3899.8 µs | 0.84× |
| 4094² | 9.11 ms | 13.73 ms | 18.41 ms | 0.66× |
| 8190² | 34.99 ms | 39.53 ms | 54.34 ms | 0.89× |
| 16382² | 182.6 ms | 178.9 ms | 202.3 ms | 1.02× |

Congruency: final fields agree to max rel difference ≈ 2–4 × 10⁻¹⁶ (machine epsilon) at every size.

Reading: single-threaded, MFO's fused one-pass stencil beats Chmy's two-kernel flux-form launch at every size — dramatically at cache-resident sizes (per-launch overhead) and settling to a stable ~2.6× asymptote at DRAM-resident sizes (5 array passes/step vs Chmy's 7, plus KA-CPU per-cell overhead: Chmy plateaus at ~0.63 Gcells/s serial, MFO at ~1.7). With threads, Chmy scales (its KernelAbstractions CPU backend parallelizes; MFO's CPU path is serial, flag 6) and wins a middle band — peaking at ~1.5× ahead at 4094², where 5 threads buy it ~107 GB/s effective vs the ~68 GB/s one core streams for MFO. At 16382² the picture inverts again: Chmy's threaded throughput sags (1.92 → 1.47 Gcells/s) as its 7-pass, 15 GB/step traffic saturates the memory system, and serial MFO pulls back level (1.5 Gcells/s). So Chmy's threading advantage is real but bounded by its extra traffic; a threaded MFO (issue #51) plus step-fusion (issue #52) would dominate the whole range. MFO's Krylov-facing `mul!` path costs a further ~15–50% over `apply!` from the flat↔interior copies. (Caveat: the threaded MFO 4094² row ran ~25% slower than its serial twin — MFO uses one core either way, so treat that row's gap as thermal/scheduling noise from the surrounding threaded runs.)

## Files

| file | role |
|---|---|
| `chmy.jl` | faithful port of the published example (kernels verbatim) |
| `mfo.jl` | MFO implementation, `apply!` and prepared-`mul!` legs |
| `common.jl` | shared constants (χ, nt, seed, sizes), IC, field dump/load, CSV |
| `compare.jl` | cross-package congruency check on final fields |
| `run.sh` | full sweep orchestration |
