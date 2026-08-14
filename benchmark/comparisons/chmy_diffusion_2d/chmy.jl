# Chmy.jl leg — faithful port of Chmy.jl's examples/diffusion_2d.jl (main, v0.1.26,
# captured 2026-08-12). Kernels, grid, Launcher, Δt line, and BC batching are verbatim.
# Deviations, all flagged in README.md:
#   - CairoMakie visualization and the per-step @printf removed (as Chmy's own
#     diffusion_2d_perf.jl does)
#   - the unseeded rand() IC replaced by a shared seeded matrix so the final field
#     can be cross-checked against the MFO leg
#   - grid size taken from ARGS for the size sweep; default n = 128 → nxy = 126²
#
# Run: julia --project=. --startup-file=no -t <1|auto> chmy.jl [128,512,...]

using Pkg; Pkg.activate(@__DIR__)
using Chmy, KernelAbstractions, BenchmarkTools, Statistics
include(joinpath(@__DIR__, "common.jl"))

@kernel inbounds = true function compute_q!(q, C, χ, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    q.x[I...] = -χ * ∂x(C, g, I...)
    q.y[I...] = -χ * ∂y(C, g, I...)
end

@kernel inbounds = true function update_C!(C, q, Δt, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    C[I...] -= Δt * divg(q, g, I...)
end

function steps!(arch, grid, launch, q, C, χ, Δt, nt, backend)
    for _ in 1:nt
        launch(arch, grid, compute_q! => (q, C, χ, grid))
        launch(arch, grid, update_C! => (C, q, Δt, grid); bc = batch(grid, C => Neumann(); exchange = C))
    end
    KernelAbstractions.synchronize(backend)
    return C
end

function init!(C, arch, grid, A)
    interior(C) .= A
    bc!(arch, grid, C => Neumann(); exchange = C)
    return C
end

function main(backend = CPU(); ns = parse_sizes())
    print_env_banner("chmy")
    for n in ns
        nxy = (n, n) .- 2
        arch   = Arch(backend)
        grid   = UniformGrid(arch; origin = (-1, -1), extent = (2, 2), dims = nxy)
        launch = Launcher(arch, grid; outer_width = (16, 8))
        χ  = CHI
        Δt = minimum(spacing(grid))^2 / χ / ndims(grid) / 2.1
        C  = Field(backend, grid, Center())
        q  = VectorField(backend, grid)
        A  = make_ic(nxy...)
        # correctness pass: exactly NT steps from the seeded IC → dump the final field
        init!(C, arch, grid, A)
        steps!(arch, grid, launch, q, C, χ, Δt, NT, backend)
        save_field("chmy", n, Array(interior(C)))
        # timing pass: the NT-step loop is the benchmark kernel, IC reset per sample
        b = @benchmark steps!($arch, $grid, $launch, $q, $C, $χ, $Δt, NT, $backend) setup =
            (init!($C, $arch, $grid, $A)) evals = 1 seconds = 20
        append_timing(; pkg = "chmy", leg = "kernels", n,
            tstep_min_ns = minimum(b).time / NT, tstep_median_ns = median(b).time / NT)
    end
end

main()
