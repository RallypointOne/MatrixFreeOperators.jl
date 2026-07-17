# 2D Poisson -∇²u = f on (0,1)², Dirichlet u = 1 on ∂Ω, manufactured solution
# u = 1 + eˣ·sin(πx)·sin(πy). Geometric-multigrid-preconditioned CG vs plain CG —
# the first example exercising the Krylov path.
#
# Run with: julia --project=examples examples/multigrid_poisson.jl

using Pkg; Pkg.activate(@__DIR__)
using MatrixFreeOperators, BenchmarkTools, CairoMakie, Krylov, LinearAlgebra, Printf

n = 256
g = CartesianGrid(
    ((0.0, 1.0), (0.0, 1.0)), (n, n); bc=ntuple(_ -> (Dirichlet(1.0), Dirichlet(1.0)), 2)
)
L = -1 * laplacian(g)                                    # SPD under Dirichlet
A = prepare(L)
uexact(x) = 1 + exp(x[1]) * sinpi(x[1]) * sinpi(x[2])
f(x) = -exp(x[1]) * sinpi(x[2]) * ((1 - 2 * pi^2) * sinpi(x[1]) + 2 * pi * cospi(x[1]))
# inhomogeneous boundary data enters the RHS through the affine lift
b = flatten(set!(scalar_field(g), f)) .- flatten(boundary_rhs(L, g))
uex = flatten(set!(scalar_field(g), uexact))

mg = MultigridPreconditioner(L)                          # Jacobi(2/3), levels=:auto
println(mg)
u_mg, stats_mg = Krylov.cg(A, b; M=mg, rtol=1e-8)
u_pl, stats_pl = Krylov.cg(A, b; rtol=1e-8)
@printf "MG-CG:      %4d iterations, max error %.2e\n" stats_mg.niter maximum(
    abs, u_mg .- uex
)
@printf "plain CG:   %4d iterations, max error %.2e\n" stats_pl.niter maximum(
    abs, u_pl .- uex
)

t_mg = @belapsed Krylov.cg($A, $b; M=$mg, rtol=1e-8) samples = 3 evals = 1
t_pl = @belapsed Krylov.cg($A, $b; rtol=1e-8) samples = 3 evals = 1
@printf "time:       MG-CG %.0f ms vs plain CG %.0f ms\n" 1e3 * t_mg 1e3 * t_pl

# ...or as a standalone multigrid iteration
u_mgs = solve(MultigridSolver(L), b; rtol=1e-8)
@printf "solve():    max error %.2e\n" maximum(abs, u_mgs .- uex)

Δ = spacing(g)
xs = range(0.5Δ[1], 1 - 0.5Δ[1]; length = n)
ys = range(0.5Δ[2], 1 - 0.5Δ[2]; length = n)
fig = Figure(size = (500, 400))
ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "y", title = "-∇²u = f", aspect = DataAspect())
hm = heatmap!(ax, xs, ys, reshape(u_mg, n, n))
Colorbar(fig[1, 2], hm)
figpath = joinpath(@__DIR__, "multigrid_poisson.png")
save(figpath, fig)
@printf "figure:     %s\n" figpath
