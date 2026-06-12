# 2D heat equation ∂u/∂t = α∇²u on (0,1)², homogeneous Dirichlet, explicit Euler.
#
# Run with: julia --project=examples examples/heat_equation.jl

using Pkg; Pkg.activate(@__DIR__)
using MatrixFreeOperators, BenchmarkTools, CairoMakie, LinearAlgebra, Printf, Random

n, α, nsteps = 64, 0.2, 500
g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (n, n))
P = prepare(laplacian(g), scalar_field(g))
dt = minimum(spacing(g))^2 / (8α)
Random.seed!(0)
# 20 Gaussian blobs at random centers; diameter ≈ 4σ drawn uniform in [0.1, 0.3]
blobs = [(rand(), rand(), 0.2 * (0.5 + rand())) for _ in 1:20]
u0 = flatten(set!(scalar_field(g), x ->
    sum(exp(-((x[1] - cx)^2 + (x[2] - cy)^2) / (s^2 / 8)) for (cx, cy, s) in blobs)))
u0 ./= maximum(u0)

function step!(u, du, P, αdt)
    mul!(du, P, u)
    u .+= αdt .* du
    return nothing
end

Δ = spacing(g)
xs = range(0.5Δ[1], 1 - 0.5Δ[1]; length = n)
ys = range(0.5Δ[2], 1 - 0.5Δ[2]; length = n)
U = Observable(reshape(copy(u0), n, n))
tlabel = Observable("t = 0.00")
fig = Figure(size = (500, 400))
ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "y", title = tlabel, aspect = DataAspect())
hm = heatmap!(ax, xs, ys, U; colorrange = (0, 1))
Colorbar(fig[1, 2], hm)

u = copy(u0)
du = zero(u0)
nframes = 100
steps_per_frame = nsteps ÷ nframes
figpath = joinpath(@__DIR__, "heat_equation.gif")
record(fig, figpath, 0:nframes; framerate = 30) do frame
    frame == 0 && return
    foreach(_ -> step!(u, du, P, α * dt), 1:steps_per_frame)
    U[] = reshape(copy(u), n, n)
    tlabel[] = @sprintf "t = %.2f" frame * steps_per_frame * dt
end
@printf "animation:         %s\n" figpath

t = @belapsed foreach(_ -> step!(u, du, $P, $(α * dt)), 1:$nsteps) setup =
    (u = copy($u0); du = zero($u0)) evals = 1
@printf "time/step:         %.3f µs\n" 1e6 * t / nsteps
