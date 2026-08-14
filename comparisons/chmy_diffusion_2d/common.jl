# Shared parameters and helpers for the Chmy.jl vs MatrixFreeOperators.jl
# Diffusion 2D comparison. Everything physical/numerical mirrors Chmy.jl's
# examples/diffusion_2d.jl exactly — see README.md for the congruency notes.

using Printf, Random

# Chmy's convention: n = 128 → interior cells nxy = (n - 2, n - 2)
const DEFAULT_NS = [128, 512, 1024, 2048, 4096, 8192, 16384]
const CHI = 1.0    # χ in diffusion_2d.jl
const NT = 100     # nt in diffusion_2d.jl
const SEED = 1234  # replaces the example's unseeded rand() so both packages share one IC

const RESULTS_DIR = joinpath(@__DIR__, "results")
# Field dumps reach 2.1 GB each at 16382²; keep them out of the iCloud-synced repo.
const FIELDS_DIR = get(ENV, "CHMY_MFO_FIELDS_DIR",
    joinpath(tempdir(), "chmy_mfo_diffusion_2d"))

parse_sizes() = isempty(ARGS) ? DEFAULT_NS : parse.(Int, split(ARGS[1], ","))

make_ic(nx, ny) = rand(Xoshiro(SEED), nx, ny)

field_path(pkg, n) = joinpath(FIELDS_DIR, "$(pkg)_$(n).bin")

function save_field(pkg, n, A::Matrix{Float64})
    mkpath(FIELDS_DIR)
    open(field_path(pkg, n), "w") do io
        write(io, Int64(size(A, 1)), Int64(size(A, 2)))
        write(io, A)
    end
end

function load_field(pkg, n)
    open(field_path(pkg, n), "r") do io
        nx, ny = read(io, Int64), read(io, Int64)
        read!(io, Matrix{Float64}(undef, nx, ny))
    end
end

function append_timing(; pkg, leg, n, tstep_min_ns, tstep_median_ns)
    mkpath(RESULTS_DIR)
    csv = joinpath(RESULTS_DIR, "timings.csv")
    isfile(csv) || open(io -> println(io,
            "package,leg,n,interior,threads,tstep_min_ns,tstep_median_ns,mcups"), csv, "w")
    mcups = (n - 2)^2 / tstep_min_ns * 1e3  # million cell-updates per second
    open(csv, "a") do io
        @printf(io, "%s,%s,%d,%d,%d,%.1f,%.1f,%.1f\n",
            pkg, leg, n, n - 2, Threads.nthreads(), tstep_min_ns, tstep_median_ns, mcups)
    end
    @printf("  %-6s %-6s n=%4d (%4d²)  t/step min %10.1f µs  median %10.1f µs  %8.1f Mcells/s\n",
        pkg, leg, n, n - 2, tstep_min_ns / 1e3, tstep_median_ns / 1e3, mcups)
end

print_env_banner(pkg) = @printf("# %s — Julia %s, %d thread(s), %s, %s\n",
    pkg, string(VERSION), Threads.nthreads(), Sys.CPU_NAME, Sys.MACHINE)
