# 2D heat equation ∂u/∂t = α∇²u on (0,1)², homogeneous Dirichlet, explicit Euler.
# The same MatrixFreeOperators step is run plain and compiled with Reactant (XLA),
# then benchmarked against each other. Tracing uses the MatrixFreeOperatorsReactantExt
# shifted-view Laplacian body; results are bit-identical to the plain path.
#
# Each compiled call pays a host↔XLA round trip (~25 µs), so per-step calls are
# overhead-bound at small grids. Compiling the whole loop (@trace for) would remove
# that, but Reactant cannot yet carry the PreparedOperator's traced scratch fields
# through a stablehlo.while (NoFieldMatchError as of Reactant 0.2).
#
# Run with: julia --project=examples examples/heat_equation.jl

using MatrixFreeOperators, Reactant, BenchmarkTools, LinearAlgebra, Printf

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

#--------------------------------------------------------------------------------# Plain
u = copy(u0)
du = zero(u0)
foreach(_ -> step!(u, du, P, α * dt), 1:nsteps)

#--------------------------------------------------------------------------------# Reactant
Reactant.set_default_backend("cpu")
ur = Reactant.to_rarray(copy(u0))
dur = Reactant.to_rarray(zero(u0))
Pr = Reactant.to_rarray(P)
t_compile = @elapsed step_c = @compile step!(ur, dur, Pr, α * dt)
foreach(_ -> step_c(ur, dur, Pr, α * dt), 1:nsteps)

#--------------------------------------------------------------------------------# Results
# u₀ is a Laplacian eigenmode, so the continuum solution is exp(-2π²αT)·u₀
uexact = exp(-2π^2 * α * nsteps * dt) .* u0
@printf "error vs analytic   plain:    %.3e\n" maximum(abs, u .- uexact)
@printf "error vs analytic   reactant: %.3e\n" maximum(abs, Array(ur) .- uexact)
@printf "plain vs reactant:  %.3e\n\n" maximum(abs, u .- Array(ur))

t_plain = @belapsed foreach(_ -> step!(u, du, $P, $(α * dt)), 1:$nsteps) setup =
    (u = copy($u0); du = zero($u0)) evals = 1
t_react = @belapsed foreach(_ -> $step_c($ur, $dur, $Pr, $(α * dt)), 1:$nsteps) evals = 1
@printf "compile time:        %.2f s\n" t_compile
@printf "plain:               %.3f µs/step\n" 1e6 * t_plain / nsteps
@printf "reactant:            %.3f µs/step\n" 1e6 * t_react / nsteps
@printf "speedup:             %.1f×\n" t_plain / t_react
