# GPU benchmark for the packed forest phase (#15): the packed single-launch sweep
# vs the per-leaf reference path at equal DOFs (uniform + refined), and the batched
# device exchange vs the per-descriptor loop on the same device data. Standalone —
# not part of the AirspeedVelocity SUITE, since CUDA cannot be a committed dep.
# Run on a CUDA box:
#   julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.add("CUDA")'  # local only
#   julia --project=benchmark benchmark/gpu.jl
using BenchmarkTools
using LinearAlgebra
using Printf
using CUDA
using MatrixFreeOperators
const MFO = MatrixFreeOperators
const Adapt = MFO.Adapt
CUDA.allowscalar(false)

function make_forest(n; refined=false)
    per = ((Periodic(), Periodic()), (Periodic(), Periodic()))
    g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (n, n); bc=per)
    bf = BlockForest(g; blocksize=(32, 32), maxlevel=2)
    refined && refine!(bf, x -> x[1] < 0.5)
    return bf
end

fmt(t) = t < 1e-3 ? @sprintf("%8.1f µs", 1e6 * t) : @sprintf("%8.2f ms", 1e3 * t)

println("device: $(CUDA.name(CUDA.device()))\n")

#--------------------------------------------------------------------------------# packed vs per-leaf mul!

println("laplacian mul! — CPU per-leaf / GPU per-leaf / GPU packed (min time)")
@printf("%-16s %8s %9s  %11s %11s %11s   %14s %s\n",
    "forest", "nleaves", "DOFs", "CPU/leaf", "GPU/leaf", "GPU packed", "packed speedup", "host alloc")
for (n, refined) in ((256, false), (512, false), (1024, false), (2048, false), (512, true))
    bf = make_forest(n; refined)
    u = set!(scalar_field(bf), x -> sin(4x[1]) + cos(3x[2]))
    L = laplacian(bf)
    x = flatten(u)
    y = similar(x)
    P = prepare(L, scalar_field(bf))

    Lg = Adapt.adapt(CuArray, L)
    Pl = prepare(Lg, Adapt.adapt(CuArray, scalar_field(bf)))    # per-leaf sweep on device
    Pp = prepare(Lg, Adapt.adapt(CuArray, pack(u)))             # single-launch packed sweep
    xg = CuArray(x)
    yl = similar(xg)
    yp = similar(xg)

    mul!(y, P, x)
    mul!(yl, Pl, xg)
    mul!(yp, Pp, xg)
    Array(yl) ≈ y && Array(yp) ≈ y || error("parity failure at n=$n refined=$refined")

    t_cpu = @belapsed mul!($y, $P, $x) seconds = 2
    t_leaf = @belapsed CUDA.@sync(mul!($yl, $Pl, $xg)) seconds = 2
    t_pack = @belapsed CUDA.@sync(mul!($yp, $Pp, $xg)) seconds = 2

    # host-side allocation of the packed device mul! — must stay flat in nleaves;
    # min-of-10 behind a function barrier rides out GC and lazy-init noise
    alloc_mul(P, out, v) = (mul!(out, P, v); minimum(_ -> @allocated(mul!(out, P, v)), 1:10))
    @printf("%-16s %8d %9d  %11s %11s %11s   %6.1f×        %7d B\n",
        refined ? "$(n)² refined" : "$(n)² uniform", MFO.nleaves(bf), length(x),
        fmt(t_cpu), fmt(t_leaf), fmt(t_pack), t_leaf / t_pack, alloc_mul(Pp, yp, xg))
end

#--------------------------------------------------------------------------------# halo_update!: batched kernels vs descriptor loop

println("\nhalo_update! on the device packed field — batched kernels vs per-descriptor loop")
@printf("%-16s %8s  %11s %11s   %s\n",
    "forest", "nleaves", "kernels", "loop", "kernel speedup")
for (n, refined) in ((512, false), (2048, false), (512, true))
    bf = make_forest(n; refined)
    u = set!(scalar_field(bf), x -> sin(4x[1]) + cos(3x[2]))
    pg = Adapt.adapt(CuArray, pack(u))
    sched = MFO._exchange_schedule(bf)
    halo_update!(pg, pg.grid)                        # warm the _device_schedule cache
    t_dev = @belapsed CUDA.@sync(halo_update!($pg, $(pg.grid))) seconds = 2
    t_loop = @belapsed CUDA.@sync(MFO._run_exchange_host!($pg, $sched)) seconds = 2
    @printf("%-16s %8d  %11s %11s   %6.1f×\n",
        refined ? "$(n)² refined" : "$(n)² uniform", MFO.nleaves(bf),
        fmt(t_dev), fmt(t_loop), t_loop / t_dev)
end
