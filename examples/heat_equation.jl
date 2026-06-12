# 2D heat equation ∂u/∂t = α∇²u on (0,1)², homogeneous Dirichlet, explicit Euler.
#
# Run with: julia --project=examples examples/heat_equation.jl

using MatrixFreeOperators, BenchmarkTools, LinearAlgebra, Printf

n, α, nsteps = 64, 1e-2, 500
g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (n, n))
P = prepare(laplacian(g), scalar_field(g))
dt = minimum(spacing(g))^2 / (8α)
u0 = flatten(set!(scalar_field(g), x -> sinpi(x[1]) * sinpi(x[2])))

function step!(u, du, P, αdt)
    mul!(du, P, u)
    u .+= αdt .* du
    return nothing
end

u = copy(u0)
du = zero(u0)
foreach(_ -> step!(u, du, P, α * dt), 1:nsteps)

# u₀ is a Laplacian eigenmode, so the continuum solution is exp(-2π²αT)·u₀
uexact = exp(-2π^2 * α * nsteps * dt) .* u0
@printf "error vs analytic: %.3e\n" maximum(abs, u .- uexact)

t = @belapsed foreach(_ -> step!(u, du, $P, $(α * dt)), 1:$nsteps) setup =
    (u = copy($u0); du = zero($u0)) evals = 1
@printf "time/step:         %.3f µs\n" 1e6 * t / nsteps
