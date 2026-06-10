# Hand-rolled RK4: the explicit time-stepping interop is just an RHS closure over
# the prepared operator's mul! — the same closure shape OrdinaryDiffEq's
# ODEProblem takes (see the `prepare` docstring).
function rk4!(f!, u, dt, nsteps)
    k1 = similar(u)
    k2 = similar(u)
    k3 = similar(u)
    k4 = similar(u)
    utmp = similar(u)
    for _ in 1:nsteps
        f!(k1, u)
        utmp .= u .+ (dt / 2) .* k1
        f!(k2, utmp)
        utmp .= u .+ (dt / 2) .* k2
        f!(k3, utmp)
        utmp .= u .+ dt .* k3
        f!(k4, utmp)
        u .+= (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end
    return u
end

@testset "Explicit time stepping through the RHS closure (heat equation)" begin
    g = CartesianGrid(((0.0, 2π),), (64,); bc=((Periodic(), Periodic()),))
    P = prepare(laplacian(g))
    f!(du, u) = mul!(du, P, u)

    u = flatten(set!(scalar_field(g), x -> sin(x[1])))
    dt = 0.005
    nsteps = 200
    rk4!(f!, u, dt, nsteps)

    t = dt * nsteps
    uref = exp(-t) .* flatten(set!(scalar_field(g), x -> sin(x[1])))
    @test maximum(abs, u .- uref) < 1e-3
end
