# 2D heat equation ∂u/∂t = α∇²u on (0,1)², homogeneous Dirichlet, solved by
# unconditionally-stable backward Euler. Each step is a direct fast-diagonalization
# solve of (I − αΔt·Δ) — no CFL limit, no iteration — contrasted with the explicit
# Euler stability bound the matrix-free `mul!` is otherwise stuck behind.
#
# Run with: julia --project=examples examples/heat_equation_implicit.jl

using Pkg; Pkg.activate(@__DIR__)
using MatrixFreeOperators, LinearAlgebra, Printf

n, α = 128, 0.01
tspan = (0.0, 0.2)
g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (n, n))

# Smooth initial state; its evolution stays well-resolved so backward Euler is accurate.
u0 = flatten(set!(scalar_field(g), x -> sin(π * x[1]) * sin(π * x[2])))

dt_explicit = minimum(spacing(g))^2 / (8α)            # explicit Euler stability ceiling
dt = (tspan[2] - tspan[1]) / 40                       # 40 implicit steps, dt ≫ dt_explicit
S = fast_diag_solver(g; α=1.0, β=-α * dt)             # (I − αΔt·Δ), reused every step

u = copy(u0)
for _ in 1:40
    ldiv!(u, S, u)                                    # backward Euler: uⁿ⁺¹ = (I − αΔt·Δ)⁻¹ uⁿ
end

# Reference: refined explicit Euler over the same horizon (≈ exact discrete evolution).
P = prepare(laplacian(g))
uref = copy(u0); du = similar(uref)
m = ceil(Int, (tspan[2] - tspan[1]) / dt_explicit) + 1
sdt = (tspan[2] - tspan[1]) / m
for _ in 1:m
    mul!(du, P, uref)
    uref .+= (α * sdt) .* du
end

@printf "explicit dt ceiling: %.3e\n" dt_explicit
@printf "implicit dt used:    %.3e  (%.0f× larger, 40 steps vs %d)\n" dt dt / dt_explicit m
@printf "‖implicit − refined explicit‖∞: %.3e\n" maximum(abs, u .- uref)
