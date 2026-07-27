# Exploratory multi-GPU tests

Tests here need **≥ 3 physical NVIDIA GPUs** (MDLA enforces unique device IDs per
partition, so partitions cannot share a GPU). They are deliberately **not** included
from `runtests.jl` or `test/mdla.jl`: the standard suite must stay runnable on
machines with 0–2 GPUs, and wiring these into CI is a separate, deliberate decision.

They are written as ordinary `@testset`s so they can be lifted into the official
suite verbatim once there is a runner for them.

## What they cover

`mdla_3partition.jl` — the distributed MDLA path at 3 partitions. Two partitions can
never produce a slab that is cut on *both* faces; three can. That middle slab is the
first case where the ghost section of MDLA's `local_x` holds planes from **two
different owners** at once, which is the first real exercise of the owner-ascending /
plane-ascending ordering rule `_slab_ghost_layout` (`src/partitioning.jl`) replays
from MDLA's `_compute_ghost_topology`. `test/partitioning.jl` proves this CPU-side
against emulated exchange semantics; these run it on real `scatter!`/`reduce!`.

Covered: middle-slab topology, forward parity vs the 1-partition operator (bitwise
`==`), the `α`/`β` path, the adjoint identity `⟨Lx,y⟩ = ⟨x,Lᵀy⟩`, CPU
`apply_adjoint!` parity, and distributed `Krylov.cg` parity.

## Running

Build an env with the gated deps (they stay out of `test/Project.toml` on purpose):

```
julia --project=/tmp/mfo-gpu-env -e '
using Pkg
Pkg.develop(path=".")
Pkg.develop(path="<path-to-MultiDeviceLinearAlgebra>")
Pkg.add(["CUDA", "Krylov", "Test", "Random", "StaticArrays", "Adapt"])'
```

Then:

```
MFO_TEST_MDLA=true julia --project=/tmp/mfo-gpu-env -e '
using MatrixFreeOperators, Test, LinearAlgebra, Random, StaticArrays
import Adapt, Krylov
include("test/test_utils.jl"); include("test/multigpu/mdla_3partition.jl")'
```

Run this in its own session: `test/device_gpu.jl` sets `CUDA.allowscalar(false)`
globally and the MDLA tests deliberately do not.

## Known environment hazard: broken CUDA peer-to-peer

On some hosts a direct device-to-device `copyto!` **silently produces zeros** — no
error raised — even though `CUDA.can_access_peer` returns `true` and peer access is
already enabled. MDLA's `scatter!`/`reduce!` (`src/ghost.jl`) move ghost data with
exactly such peer copies, so on an affected host every ghost slab arrives as zeros
and every multi-partition test fails while single-partition tests pass.

Minimal check before trusting any failure here:

```julia
using CUDA
CUDA.device!(1); src = CuArray([10.0, 20.0, 30.0, 40.0]); CUDA.synchronize()
CUDA.device!(0); dst = CUDA.zeros(Float64, 4)
copyto!(dst, src); CUDA.synchronize()
Array(dst)   # [10,20,30,40] = healthy;  [0,0,0,0] = broken P2P
```

If it prints zeros, the failures are environmental, not MFO bugs — MDLA's own
`test/test_ghost_exchange.jl` will fail too.

**The fix is host-side.** This is almost always the IOMMU: PCIe P2P requires VT-d off or in
passthrough (`intel_iommu=off` or `iommu=pt`). Under translation, peer copies silently return
zeros or `nan` while `CUDA.can_access_peer` still reports `true` for every pair. Check ACS as
well. This exact fault was diagnosed and fixed on `sasquatch` in July 2026.

MDLA no longer relies on the host being correct: as of `8fddc9c`
(kylebeggs/MultiDeviceLinearAlgebra.jl#22) it probes each ordered device pair at `GhostExchange`
construction and, for any pair that fails the round-trip, falls back to host-staged transfers with
a one-time warning. So an affected host now yields correct numbers
plus a loud warning rather than silent zeros. That fallback is a safety net — fix the host.
