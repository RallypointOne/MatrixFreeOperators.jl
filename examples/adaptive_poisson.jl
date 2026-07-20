# 2D Poisson -∇²u = f on (0,1)², homogeneous Dirichlet, manufactured Gaussian bump
# u = exp(-‖x-c‖²/s) at c = (0.7, 0.3) — an adaptive solve on a BlockForest where
# refinement follows the solution feature: solve → gradient indicator → regrid!
# (solution transferred across the regrid) → re-prepare → solve.
#
# Run with: julia --project=examples examples/adaptive_poisson.jl

using Pkg; Pkg.activate(@__DIR__)
using MatrixFreeOperators, CairoMakie, Krylov, LinearAlgebra, Printf

const MFO = MatrixFreeOperators

c = (0.7, 0.3)
s = 0.005
uexact(x) = exp(-((x[1] - c[1])^2 + (x[2] - c[2])^2) / s)
f(x) = (4 / s - 4 * ((x[1] - c[1])^2 + (x[2] - c[2])^2) / s^2) * uexact(x)

base = CartesianGrid(
    ((0.0, 1.0), (0.0, 1.0)), (64, 64); bc=ntuple(_ -> (Dirichlet(), Dirichlet()), 2)
)
bf = BlockForest(base; blocksize=(8, 8), maxlevel=3)
u = scalar_field(bf)

function l1_error(u, bf)
    e = 0.0
    for i in 1:MFO.nleaves(bf)
        sp = MFO._leaf_spacing(bf, bf.forest.leaves[i].level)
        ref = set!(similar(MFO.block(u, i)), uexact)
        e += sum(abs, interior(MFO.block(u, i)) .- interior(ref)) * prod(sp)
    end
    return e
end

for cycle in 1:4
    P = prepare(laplacian(bf), u)
    rhs = .-flatten(set!(scalar_field(bf), f))          # Δu = -f; Dirichlet lift is zero
    sol, stats = Krylov.gmres(P, rhs; rtol=1e-10)       # nonsymmetric on an adapted forest
    flat_to_interior!(u, sol)
    lo, hi = extrema(k -> k.level, bf.forest.leaves)
    @printf "cycle %d: %3d leaves, levels %d–%d, %4d gmres iters, L1 error %.3e\n" cycle MFO.nleaves(
        bf
    ) lo hi stats.niter l1_error(u, bf)
    cycle == 4 && break
    η = MFO.gradient(bf) * u
    # refine well past the steep flank so coarse–fine interfaces land where the
    # solution is flat — an interface inside the feature costs more than it saves
    τ = 0.02 * maximum(i -> maximum(norm, interior(MFO.block(η, i))), 1:MFO.nleaves(bf))
    global η, u = regrid!(η, u; refine=b -> maximum(norm, interior(b)) > τ)
end

fig = Figure(size=(780, 660), fontsize=17)
ax = Axis(fig[1, 1]; xlabel="x", ylabel="y", aspect=DataAspect(),
          title="adaptive -∇²u = f: solution + leaf blocks by level")
hm = nothing
for (i, (key, lg)) in enumerate(leaves(bf))
    sp = spacing(lg)
    ext = lg.extent
    xs = range(ext[1][1] + 0.5sp[1], ext[1][2] - 0.5sp[1]; length=bf.blocksize[1])
    ys = range(ext[2][1] + 0.5sp[2], ext[2][2] - 0.5sp[2]; length=bf.blocksize[2])
    global hm = heatmap!(ax, xs, ys, collect(interior(MFO.block(u, i))); colorrange=(0, 1))
end
levelcolors = Makie.wong_colors()
# Thin the outlines with depth so the fine blocks do not mat over the solution.
for (key, lg) in leaves(bf)
    ext = lg.extent
    lines!(ax,
           [ext[1][1], ext[1][2], ext[1][2], ext[1][1], ext[1][1]],
           [ext[2][1], ext[2][1], ext[2][2], ext[2][2], ext[2][1]];
           color=levelcolors[key.level + 1], linewidth=max(0.7, 2.2 / 1.5^key.level))
end
Colorbar(fig[1, 2], hm; label="u")
maxlev = maximum(k -> k.level, bf.forest.leaves)
fig[2, 1:2] = Legend(fig,
                     [LineElement(color=levelcolors[l + 1], linewidth=3) for l in 0:maxlev],
                     ["level $l" for l in 0:maxlev], "leaf refinement";
                     orientation=:horizontal, framevisible=false, titleposition=:left)
figpath = joinpath(@__DIR__, "adaptive_poisson.png")
save(figpath, fig)
@printf "figure: %s\n" figpath
