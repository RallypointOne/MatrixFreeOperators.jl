# 2D heat equation ∂u/∂t = α∇²u on (0,1)², homogeneous Dirichlet, explicit Euler.
#
# The stepper works at field level: `apply!(du, L, u)` is the RHS, and the Euler
# update touches `interior(u)`. `prepare` + `mul!` is the Krylov boundary — it
# copies the flat vector into and out of a halo-padded scratch field on every
# call, a cost an explicit integrator has no reason to pay.
#
# Run with: julia --project=examples examples/heat_equation.jl

using Pkg; Pkg.activate(@__DIR__)
using MatrixFreeOperators, BenchmarkTools, CairoMakie, LinearAlgebra, Printf, Random

n, α = 64, 0.0005
tspan = (0.0, 5.0)
g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (n, n))
L = laplacian(g)
# explicit-Euler stability bound, then shrink dt so the steps tile tspan exactly
dtmax = minimum(spacing(g))^2 / (8α)
nframes = 100
steps_per_frame = cld(ceil(Int, (tspan[2] - tspan[1]) / dtmax), nframes)
nsteps = nframes * steps_per_frame
dt = (tspan[2] - tspan[1]) / nsteps
# 20 Gaussian blobs at random centers; diameter ≈ 4σ drawn uniform in [0.1, 0.3]
blobs = [(rand(), rand(), 0.2 * (0.5 + rand())) for _ in 1:20]
u0 = set!(scalar_field(g), x ->
    sum(exp(-((x[1] - cx)^2 + (x[2] - cy)^2) / (s^2 / 8)) for (cx, cy, s) in blobs))
interior(u0) ./= maximum(interior(u0))

function step!(u, du, L, αdt)
    apply!(du, L, u)                        # du = ∇²u on the interior, no flat copies
    interior(u) .+= αdt .* interior(du)
    return nothing
end

Δ = spacing(g)
xs = range(0.5Δ[1], 1 - 0.5Δ[1]; length = n)
ys = range(0.5Δ[2], 1 - 0.5Δ[2]; length = n)
U = Observable(copy(interior(u0)))
tlabel = Observable(@sprintf "t = %.2f" tspan[1])
fig = Figure(size = (500, 400));
ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "y", title = tlabel, aspect = DataAspect())
hm = heatmap!(ax, xs, ys, U; colorrange = (0, 1))
Colorbar(fig[1, 2], hm)

u = copy(u0)
du = similar(u0)
figpath = joinpath(@__DIR__, "heat_equation.gif")
record(fig, figpath, 0:nframes; framerate = 30) do frame
    frame == 0 && return
    foreach(_ -> step!(u, du, L, α * dt), 1:steps_per_frame)
    U[] = copy(interior(u))
    tlabel[] = @sprintf "t = %.2f" tspan[1] + frame * steps_per_frame * dt
end
@printf "animation:         %s\n" figpath

t = @belapsed foreach(_ -> step!(u, du, $L, $(α * dt)), 1:$nsteps) setup =
    (u = copy($u0); du = similar($u0)) evals = 1
@printf "time/step:         %.3f µs\n" 1e6 * t / nsteps
