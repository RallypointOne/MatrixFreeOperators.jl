# Included by reactant.jl only when MFO_TEST_REACTANT=true and Reactant.jl is available.
using Reactant

Reactant.set_default_backend("cpu")

@testset "Reactant parity vs default leaf" begin
    @testset "prepared mul! parity (2D, mixed BCs)" begin
        g = CartesianGrid(
            ((0.0, 2π), (0.0, 1.0)), (32, 24);
            bc=((Periodic(), Periodic()), (Dirichlet(), Neumann())),
        )
        P = prepare(laplacian(g), scalar_field(g))
        x = flatten(set!(scalar_field(g), x -> sin(x[1]) * x[2]^2))
        y = zero(x)
        mul!(y, P, x)

        Pr = Reactant.to_rarray(P)
        xr = Reactant.to_rarray(copy(x))
        yr = Reactant.to_rarray(zero(x))
        mul_c = @compile mul!(yr, Pr, xr)
        mul_c(yr, Pr, xr)
        @test Array(yr) ≈ y rtol = 1e-13
    end

    @testset "prepared mul! parity (3D)" begin
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), (8, 9, 10))
        P = prepare(laplacian(g), scalar_field(g))
        x = flatten(set!(scalar_field(g), x -> x[1]^2 + sinpi(x[2]) * x[3]))
        y = zero(x)
        mul!(y, P, x)

        Pr = Reactant.to_rarray(P)
        xr = Reactant.to_rarray(copy(x))
        yr = Reactant.to_rarray(zero(x))
        mul_c = @compile mul!(yr, Pr, xr)
        mul_c(yr, Pr, xr)
        @test Array(yr) ≈ y rtol = 1e-13
    end

    @testset "apply! α/β accumulation parity" begin
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16))
        L = laplacian(g)
        u = set!(scalar_field(g), x -> sinpi(x[1]) * x[2])
        w = set!(scalar_field(g), x -> x[1] + x[2])
        y = apply!(deepcopy(w), L, deepcopy(u), g, 2.0, 0.5)

        ur = Reactant.to_rarray(deepcopy(u))
        wr = Reactant.to_rarray(deepcopy(w))
        apply_c = @compile apply!(wr, L, ur, g, 2.0, 0.5)
        apply_c(wr, L, ur, g, 2.0, 0.5)
        @test Array(wr.data)[interior(g)] ≈ collect(interior(y)) rtol = 1e-13
    end
end
