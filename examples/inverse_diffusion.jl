# Inverse problem: recover the diffusion coefficient field κ(x,y) of
# ∇·(κ∇u) from a noisy observation of the response, by gradient descent.
#
# This is the example the whole design exists for. The gradient is with respect to
# an *operator parameter* — the coefficient field inside `scaling(κ)` — not with
# respect to the solution field, and nobody wrote an adjoint rule for it. Operator
# bodies are array-level broadcasts, so Enzyme differentiates straight through the
# composition `divergence(g) * scaling(κ) * gradient(g)`, boundary conditions and
# all. DifferentiationInterface is the frontend: one backend object, no annotation
# vocabulary at the call site.
#
# Run with: julia --project=examples examples/inverse_diffusion.jl

using Pkg; Pkg.activate(@__DIR__)
using MatrixFreeOperators, CairoMakie, LinearAlgebra, Printf, Random
import DifferentiationInterface as DI
import Enzyme

const MFO = MatrixFreeOperators   # `gradient` is both an operator here and DI's verb

n = 48
g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (n, n))       # homogeneous Dirichlet

# The truth we are trying to recover: a smooth background with a blob of high
# conductivity off-centre.
κ_true(x) = 1 + 1.5 * exp(-60 * ((x[1] - 0.65)^2 + (x[2] - 0.4)^2)) + 0.3 * x[1]

# The observation operator: apply ∇·(κ∇·) to a fixed excitation. Working from the
# operator action rather than a solve keeps the example about the gradient, not
# about differentiating through a Krylov loop.
#
# Everything the loss needs is an explicit argument — Enzyme wants typed arguments,
# not captured non-const globals, and at the DI layer the fixed ones become
# `Constant` contexts.
function response(κdata, udata, gg)
    K = divergence(gg) * scaling(Field(κdata, gg)) * MFO.gradient(gg)
    return interior(apply(K, Field(copy(udata), gg)))
end

# Recovering κ from one excitation is ill-posed: the forward map is linear in κ and
# blind wherever ∇u vanishes (here the domain centre and corners), so the misfit
# alone does not determine κ there. A small smoothness penalty picks the sensible
# member of that null space. λ is deliberately visible rather than tuned away.
const λ = 5e-4

function objective(κdata, obs, udata, gg)
    J = sum(abs2, response(κdata, udata, gg) .- obs) / length(obs)
    κi = interior(Field(κdata, gg))
    rough =
        sum(abs2, diff(κi; dims=1)) / length(κi) + sum(abs2, diff(κi; dims=2)) / length(κi)
    return J + λ * rough
end

u = set!(scalar_field(g), x -> sinpi(x[1]) * sinpi(x[2]))

rng = MersenneTwister(20260731)
κ★ = set!(scalar_field(g), κ_true)
data = collect(response(κ★.data, u.data, g))
data .+= 0.01 * maximum(abs, data) .* randn(rng, size(data))

backend = DI.AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))

κ = fill(1.0, padded_size(g)...)                          # flat initial guess
ctx = (DI.Constant(data), DI.Constant(u.data), DI.Constant(g))
prep = DI.prepare_gradient(objective, backend, κ, ctx...)

# Trust, but verify: spot-check the AD gradient against central finite differences
# at a few entries before letting an optimizer rely on it.
let dκ = DI.gradient(objective, prep, backend, κ, ctx...), ε = 1e-6
    println("AD gradient vs central finite differences:")
    for idx in rand(rng, findall(!iszero, dκ), 4)
        κp = copy(κ); κp[idx] += ε
        κm = copy(κ); κm[idx] -= ε
        fd = (objective(κp, data, u.data, g) - objective(κm, data, u.data, g)) / (2ε)
        @printf "  κ[%3d,%3d]   AD %+.6e   FD %+.6e\n" idx[1] idx[2] dκ[idx] fd
    end
end

# Gradient descent with backtracking: the problem is scaled like h⁻², so a fixed
# step is hopeless and a two-line line search is the honest minimum.
J = objective(κ, data, u.data, g)
@printf "\ninitial objective  %.4e\n" J
step = 1.0
for iter in 1:300
    global J, step
    _, dκ = DI.value_and_gradient(objective, prep, backend, κ, ctx...)
    accepted = false
    for _ in 1:40
        trial = κ .- step .* dκ
        Jt = objective(trial, data, u.data, g)
        if Jt < J
            κ .= trial
            J = Jt
            step *= 1.5              # grow while it keeps working
            accepted = true
            break
        end
        step /= 2
    end
    accepted || break                # step underflowed: converged as far as it goes
    iter % 100 == 0 && @printf "  iter %3d   objective %.4e\n" iter J
end
@printf "final objective    %.4e\n" J

Δ = spacing(g)
xs = range(0.5Δ[1], 1 - 0.5Δ[1]; length=n)
ys = range(0.5Δ[2], 1 - 0.5Δ[2]; length=n)
recovered = collect(interior(Field(κ, g)))
truth = collect(interior(κ★))
lims = extrema(vcat(vec(truth), vec(recovered)))

# The blob comes back in the right place at roughly the right amplitude. The
# speckle around it is not an optimizer failure — it is the null space: one
# excitation carries no information about κ where ∇u ≈ 0, which for
# u = sin(πx)sin(πy) is the domain centre and the diagonals, exactly where the
# artifacts sit. More excitations, or a stronger prior, shrink it.
fig = Figure(size=(760, 330))
for (col, (title, field)) in enumerate(("true κ" => truth, "recovered κ" => recovered))
    ax = Axis(fig[1, col]; xlabel="x", ylabel="y", title=title, aspect=DataAspect())
    heatmap!(ax, xs, ys, field; colorrange=lims)
end
Colorbar(fig[1, 3]; colorrange=lims)
save(joinpath(@__DIR__, "inverse_diffusion.png"), fig)
println("wrote inverse_diffusion.png")
