# Inverse problem: recover the diffusion coefficient field κ(x,y) of
# ∇·(κ∇u) from noisy observations of the response, by gradient descent.
#
# This is the example the whole design exists for. The gradient is with respect to
# an *operator parameter* — the coefficient field carried by the diffusion leaf —
# not with respect to the solution field, and nobody wrote an adjoint rule for it.
# Operator bodies are array-level broadcasts, so Enzyme differentiates straight
# through the leaf, boundary conditions and all. DifferentiationInterface is the
# frontend: one backend object, no annotation vocabulary at the call site.
#
# The discretization is what makes this work from a single excitation.
# `diffusion(g, κ)` is the compact flux form: fluxes κ∇u live on cell faces, with
# κ averaged to each face, and the cell balance differences neighbouring face
# fluxes. Every equation therefore couples κ at *adjacent* cells. The algebraic
# spelling `divergence(g) * scaling(κ) * gradient(g)` does not: chaining two
# centered differences samples the flux only at cells I±e, so no equation ever ties
# κ at neighbouring pixels together, the even and odd checkerboard sublattices are
# fit to disjoint halves of the noisy data, and the reconstruction speckles. That
# is issue #48, and it is a property of the stencil, not of the optimizer — a
# stronger smoothness prior barely dents it.
#
# Run with: julia --project=examples examples/inverse_diffusion.jl

using Pkg; Pkg.activate(@__DIR__)
using MatrixFreeOperators, CairoMakie, LinearAlgebra, Printf, Random
import DifferentiationInterface as DI
import Enzyme

n = 48
g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (n, n))       # homogeneous Dirichlet

# The truth we are trying to recover: a smooth background with a blob of high
# conductivity off-centre.
κ_true(x) = 1 + 1.5 * exp(-60 * ((x[1] - 0.65)^2 + (x[2] - 0.4)^2)) + 0.3 * x[1]

# The observation operator: apply ∇·(κ∇·) to a fixed excitation u. Working from the
# operator action rather than a solve keeps the example about the gradient, not
# about differentiating through a Krylov loop.
#
# Everything the loss needs is an explicit argument — Enzyme wants typed arguments,
# not captured non-const globals, and at the DI layer the fixed ones become
# `Constant` contexts. `check=false` keeps the coefficient-positivity validation
# out of the differentiated region; it would rescan κ on every objective call.
function response(κdata, udata, gg)
    D = diffusion(gg, Field(κdata, gg); check=false)
    return interior(apply(D, Field(copy(udata), gg)))
end

roughness(κi) =
    sum(abs2, diff(κi; dims=1)) / length(κi) + sum(abs2, diff(κi; dims=2)) / length(κi)

# A token smoothness penalty, kept only so the knob is visible. It no longer does
# any work: sweeping λ from 0 to 5e-4 moves the recovery error by less than 0.01
# percentage points, because the data now determine κ pixel by pixel. Under the
# wide composition the same sweep was the difference between a usable answer and
# speckle, and even a 20× stronger prior could not rescue it.
const λ = 5e-4

function objective(κdata, obs, udata, gg)
    J = sum(abs2, response(κdata, udata, gg) .- obs) / length(obs)
    return J + λ * roughness(interior(Field(κdata, gg)))
end

# One drive is enough. The data are sensitive to κ only through the flux κ∇u, so a
# single excitation still sees least where its gradient is smallest. With the
# compact stencil that shows up as faint streaks along the directions where ∇u is
# small — a resolution limit of this one drive — rather than as pixel-scale speckle.
u = set!(scalar_field(g), x -> sinpi(x[1]) * sinpi(x[2]))

rng = MersenneTwister(20260731)
κ★ = set!(scalar_field(g), κ_true)
clean = collect(response(κ★.data, u.data, g))
obs = clean .+ 0.01 * maximum(abs, clean) .* randn(rng, size(clean))

backend = DI.AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))

# Trust, but verify: spot-check the AD gradient against central finite differences
# at a few entries before letting an optimizer rely on it.
let κ = fill(1.0, padded_size(g)...), ε = 1e-6
    dκ = DI.gradient(objective, backend, κ, DI.Constant(obs), DI.Constant(u.data), DI.Constant(g))
    println("AD gradient vs central finite differences:")
    for idx in rand(rng, findall(!iszero, dκ), 4)
        κp = copy(κ); κp[idx] += ε
        κm = copy(κ); κm[idx] -= ε
        fd = (objective(κp, obs, u.data, g) - objective(κm, obs, u.data, g)) / (2ε)
        @printf "  κ[%3d,%3d]   AD %+.6e   FD %+.6e\n" idx[1] idx[2] dκ[idx] fd
    end
end

# Gradient descent with backtracking: the problem is scaled like h⁻², so a fixed
# step is hopeless and a two-line line search is the honest minimum.
function recover(obs, udata, gg, backend)
    κ = fill(1.0, padded_size(gg)...)                     # flat initial guess
    ctx = (DI.Constant(obs), DI.Constant(udata), DI.Constant(gg))
    prep = DI.prepare_gradient(objective, backend, κ, ctx...)
    J = objective(κ, obs, udata, gg)
    @printf "  initial objective  %.4e\n" J
    step = 1.0
    for iter in 1:300
        _, dκ = DI.value_and_gradient(objective, prep, backend, κ, ctx...)
        accepted = false
        for _ in 1:40
            trial = κ .- step .* dκ
            Jt = objective(trial, obs, udata, gg)
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
    @printf "  final objective    %.4e\n" J
    return κ
end

println("\nrecovering κ from 1 excitation:")
κ̂ = recover(obs, u.data, g, backend)

truth = collect(interior(κ★))
recovered = collect(interior(Field(κ̂, g)))

# Report the two numbers the reconstruction lives or dies by, rather than asserting
# them in prose. `jaggedness` is the RMS neighbour-to-neighbour variation: it is the
# checkerboard detector, since parity-decoupled sublattices disagree pixel by pixel
# and inflate it while leaving the smooth error largely unchanged.
jaggedness(κi) = sqrt(roughness(κi))
@printf "\n  relative error   %5.2f %%\n" 100 * norm(recovered .- truth) / norm(truth)
@printf "  jaggedness       %.4f   (truth %.4f)\n" jaggedness(recovered) jaggedness(truth)

Δ = spacing(g)
xs = range(0.5Δ[1], 1 - 0.5Δ[1]; length=n)
ys = range(0.5Δ[2], 1 - 0.5Δ[2]; length=n)
lims = extrema(vcat(vec(truth), vec(recovered)))

fig = Figure(size=(760, 330))
for (col, (title, field)) in
    enumerate(("true κ" => truth, "recovered, 1 excitation" => recovered))
    ax = Axis(fig[1, col]; xlabel="x", ylabel="y", title=title, aspect=DataAspect())
    heatmap!(ax, xs, ys, field; colorrange=lims)
end
Colorbar(fig[1, 3]; colorrange=lims)
save(joinpath(@__DIR__, "inverse_diffusion.png"), fig)
println("wrote inverse_diffusion.png")
