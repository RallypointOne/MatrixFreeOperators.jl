# Inverse problem: recover the diffusion coefficient field κ(x,y) of
# ∇·(κ∇u) from noisy observations of the response, by gradient descent.
#
# This is the example the whole design exists for. The gradient is with respect to
# an *operator parameter* — the coefficient field inside `scaling(κ)` — not with
# respect to the solution field, and nobody wrote an adjoint rule for it. Operator
# bodies are array-level broadcasts, so Enzyme differentiates straight through the
# composition `divergence(g) * scaling(κ) * gradient(g)`, boundary conditions and
# all. DifferentiationInterface is the frontend: one backend object, no annotation
# vocabulary at the call site.
#
# One drive cannot see everything: the data are sensitive to κ only through the
# flux κ∇u, so a single excitation is blind wherever its gradient vanishes. We
# therefore observe the response to three drive patterns (as in EIT, where several
# current patterns are injected for exactly this reason) — and reconstruct from the
# first drive alone as well, as the cautionary middle panel of the figure.
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

# The observation operator: apply ∇·(κ∇·) to a fixed excitation u. Working from the
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

# Even several drives leave κ weakly determined where every excitation has small
# ∇u, so a small smoothness penalty stays. λ is deliberately visible rather than
# tuned away — with three drives the result barely depends on it.
const λ = 5e-4

function objective(κdata, obs, udatas, gg)
    J = 0.0
    for (o, ud) in zip(obs, udatas)
        J += sum(abs2, response(κdata, ud, gg) .- o) / length(o)
    end
    κi = interior(Field(κdata, gg))
    rough =
        sum(abs2, diff(κi; dims=1)) / length(κi) + sum(abs2, diff(κi; dims=2)) / length(κi)
    return J + λ * rough
end

# The drive patterns. The first is blind at the domain centre (∇u = 0 there); the
# other two put gradients exactly where it has none.
us = map(
    f -> set!(scalar_field(g), f),
    (
        x -> sinpi(x[1]) * sinpi(x[2]),
        x -> sinpi(2x[1]) * sinpi(x[2]),
        x -> sinpi(x[1]) * sinpi(2x[2]),
    ),
)
udatas = map(u -> u.data, us)

rng = MersenneTwister(20260731)
κ★ = set!(scalar_field(g), κ_true)
obs = map(udatas) do ud
    d = collect(response(κ★.data, ud, g))
    d .+ 0.01 * maximum(abs, d) .* randn(rng, size(d))
end

backend = DI.AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))

# Trust, but verify: spot-check the AD gradient against central finite differences
# at a few entries before letting an optimizer rely on it.
let κ = fill(1.0, padded_size(g)...), ε = 1e-6
    dκ = DI.gradient(
        objective, backend, κ, DI.Constant(obs), DI.Constant(udatas), DI.Constant(g)
    )
    println("AD gradient vs central finite differences:")
    for idx in rand(rng, findall(!iszero, dκ), 4)
        κp = copy(κ); κp[idx] += ε
        κm = copy(κ); κm[idx] -= ε
        fd = (objective(κp, obs, udatas, g) - objective(κm, obs, udatas, g)) / (2ε)
        @printf "  κ[%3d,%3d]   AD %+.6e   FD %+.6e\n" idx[1] idx[2] dκ[idx] fd
    end
end

# Gradient descent with backtracking: the problem is scaled like h⁻², so a fixed
# step is hopeless and a two-line line search is the honest minimum.
function recover(obs, udatas, gg, backend)
    κ = fill(1.0, padded_size(gg)...)                     # flat initial guess
    ctx = (DI.Constant(obs), DI.Constant(udatas), DI.Constant(gg))
    prep = DI.prepare_gradient(objective, backend, κ, ctx...)
    J = objective(κ, obs, udatas, gg)
    @printf "  initial objective  %.4e\n" J
    step = 1.0
    for iter in 1:300
        _, dκ = DI.value_and_gradient(objective, prep, backend, κ, ctx...)
        accepted = false
        for _ in 1:40
            trial = κ .- step .* dκ
            Jt = objective(trial, obs, udatas, gg)
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

println("\nrecovering from 1 excitation:")
κ1 = recover(obs[1:1], udatas[1:1], g, backend)
println("recovering from 3 excitations:")
κ3 = recover(obs, udatas, g, backend)

Δ = spacing(g)
xs = range(0.5Δ[1], 1 - 0.5Δ[1]; length=n)
ys = range(0.5Δ[2], 1 - 0.5Δ[2]; length=n)
truth = collect(interior(κ★))
recovered1 = collect(interior(Field(κ1, g)))
recovered3 = collect(interior(Field(κ3, g)))
lims = extrema(vcat(vec(truth), vec(recovered1), vec(recovered3)))

# The middle panel is the lesson. Its speckle is not an optimizer failure and no
# stronger prior fixes it — the exact minimizer of the single-drive objective looks
# the same. Two mechanisms produce it. Centered differences sample the flux κ∇u
# only at neighbouring cells, so no equation ever couples κ at adjacent pixels: the
# even and odd checkerboard sublattices are fit to disjoint halves of the noisy
# data and disagree pixel by pixel, tied together only by the weak roughness prior.
# And sensitivity scales with ∇u, which for sin(πx)sin(πy) vanishes at the centre —
# the dark pixel sits there, with streaks along the characteristics of ∇u. Two more
# drives fill in what the first cannot see: the recovery error drops roughly
# eightfold and the answer stops depending on λ.
fig = Figure(size=(1080, 330))
for (col, (title, field)) in enumerate((
    "true κ" => truth,
    "recovered, 1 excitation" => recovered1,
    "recovered, 3 excitations" => recovered3,
))
    ax = Axis(fig[1, col]; xlabel="x", ylabel="y", title=title, aspect=DataAspect())
    heatmap!(ax, xs, ys, field; colorrange=lims)
end
Colorbar(fig[1, 4]; colorrange=lims)
save(joinpath(@__DIR__, "inverse_diffusion.png"), fig)
println("wrote inverse_diffusion.png")
