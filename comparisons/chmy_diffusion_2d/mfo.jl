# MatrixFreeOperators.jl leg — same PDE and discretization as chmy.jl:
# ∂C/∂t = χ∇²C on (-1,1)², cell-centered uniform grid, homogeneous Neumann,
# explicit Euler with Δt = h²/χ/ndims/2.1, NT = 100 steps, shared seeded IC.
# For constant χ on a uniform grid MFO's compact 5-point `laplacian` is
# algebraically identical to Chmy's staggered flux form (verified by compare.jl).
#
# Two legs:
#   apply — field-level apply!(du, χ*laplacian(g), u, g): ghost fill + one fused
#           broadcast; MFO's fastest repeated-application path
#   mul!  — prepare + mul! on interior-only flat vectors: the Krylov-facing API,
#           which adds two flat↔interior copies per application
#
# Run: julia --project=. --startup-file=no -t <1|auto> mfo.jl [128,512,...]

using Pkg; Pkg.activate(@__DIR__)
using MatrixFreeOperators, BenchmarkTools, LinearAlgebra, Statistics
include(joinpath(@__DIR__, "common.jl"))

function steps_apply!(u, du, L, g, dt, nt)
    ui, dui = interior(u), interior(du)
    for _ in 1:nt
        apply!(du, L, u, g)
        @. ui += dt * dui
    end
    return u
end

function steps_mul!(uf, duf, P, χdt, nt)
    for _ in 1:nt
        mul!(duf, P, uf)
        @. uf += χdt * duf
    end
    return uf
end

function main(; ns = parse_sizes())
    print_env_banner("mfo")
    for n in ns
        nxy = n - 2
        g = CartesianGrid(((-1.0, 1.0), (-1.0, 1.0)), (nxy, nxy);
            bc = ((Neumann(), Neumann()), (Neumann(), Neumann())))
        χ  = CHI
        Δt = minimum(spacing(g))^2 / χ / length(spacing(g)) / 2.1  # same formula as chmy.jl
        L  = χ * laplacian(g)
        u, du = scalar_field(g), scalar_field(g)
        A  = make_ic(nxy, nxy)
        # correctness pass: exactly NT steps from the seeded IC → dump the final field
        interior(u) .= A
        steps_apply!(u, du, L, g, Δt, NT)
        save_field("mfo", n, Matrix(interior(u)))
        # timing pass, leg 1: field-level apply!
        b = @benchmark steps_apply!($u, $du, $L, $g, $Δt, NT) setup =
            (copyto!(interior($u), $A)) evals = 1 seconds = 20
        append_timing(; pkg = "mfo", leg = "apply", n,
            tstep_min_ns = minimum(b).time / NT, tstep_median_ns = median(b).time / NT)
        # timing pass, leg 2: prepared flat-vector mul! (Krylov-facing API)
        P = prepare(laplacian(g), scalar_field(g))
        interior(u) .= A
        uf, duf = flatten(u), similar(flatten(u))
        u0f = copy(uf)
        b = @benchmark steps_mul!($uf, $duf, $P, $(χ * Δt), NT) setup =
            (copyto!($uf, $u0f)) evals = 1 seconds = 20
        append_timing(; pkg = "mfo", leg = "mul!", n,
            tstep_min_ns = minimum(b).time / NT, tstep_median_ns = median(b).time / NT)
    end
end

main()
