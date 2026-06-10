@testset "Boundaries" begin
    @testset "1-D homogeneous ghost fills" begin
        interior_vals = [1.0, 2.0, 3.0, 4.0]
        padded(g) = (x = zeros(6); x[2:5] .= interior_vals; apply_bc!(x, g); x)

        gp = CartesianGrid(((0.0, 1.0),), (4,); bc=((Periodic(), Periodic()),))
        xp = padded(gp)
        @test xp[1] == 4.0 && xp[6] == 1.0

        gd = CartesianGrid(((0.0, 1.0),), (4,); bc=((Dirichlet(), Dirichlet()),))
        xd = padded(gd)
        @test xd[1] == -1.0 && xd[6] == -4.0

        gn = CartesianGrid(((0.0, 1.0),), (4,); bc=((Neumann(), Neumann()),))
        xn = padded(gn)
        @test xn[1] == 1.0 && xn[6] == 4.0

        for x in (xp, xd, xn)
            @test x[2:5] == interior_vals
        end
    end

    @testset "halo width 2 mirror layers" begin
        g = CartesianGrid(((0.0, 1.0),), (4,); halo=(2,), bc=((Dirichlet(), Neumann()),))
        x = zeros(8)
        x[3:6] .= [1.0, 2.0, 3.0, 4.0]
        apply_bc!(x, g)
        @test x[2] == -1.0 && x[1] == -2.0
        @test x[7] == 4.0 && x[8] == 3.0
    end

    @testset "2-D corner consistency (ghost of ghost)" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (3, 3); bc=((Periodic(), Periodic()), (Periodic(), Periodic()))
        )
        x = zeros(5, 5)
        x[2:4, 2:4] .= reshape(1.0:9.0, 3, 3)
        apply_bc!(x, g)
        @test x[1, 1] == x[4, 4]
        @test x[5, 5] == x[2, 2]
        @test x[1, 5] == x[4, 2]
    end

    @testset "zero_ghosts!" begin
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (3, 3))
        x = rand(5, 5)
        xi = copy(x[2:4, 2:4])
        MatrixFreeOperators.zero_ghosts!(x, g)
        @test x[2:4, 2:4] == xi
        @test sum(abs, x) ≈ sum(abs, xi)
    end

    @testset "fill/fold adjointness ⟨Px,y⟩ = ⟨x,Pᵀy⟩" begin
        rng = Random.MersenneTwister(7)
        cases = [
            CartesianGrid(((0.0, 1.0),), (5,); bc=((Periodic(), Periodic()),)),
            CartesianGrid(((0.0, 1.0),), (5,); bc=((Dirichlet(), Dirichlet()),)),
            CartesianGrid(((0.0, 1.0),), (5,); bc=((Neumann(), Neumann()),)),
            CartesianGrid(((0.0, 1.0),), (5,); halo=(2,), bc=((Dirichlet(), Neumann()),)),
            CartesianGrid(
                ((0.0, 1.0), (0.0, 1.0)),
                (4, 3);
                bc=((Periodic(), Periodic()), (Dirichlet(), Neumann())),
            ),
            CartesianGrid(
                ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                (3, 4, 2);
                halo=(2, 1, 1),
                bc=(
                    (Neumann(), Dirichlet()),
                    (Periodic(), Periodic()),
                    (Dirichlet(), Dirichlet()),
                ),
            ),
        ]
        for g in cases
            x = rand(rng, padded_size(g)...)
            y = rand(rng, padded_size(g)...)
            Px = apply_bc!(copy(x), g)
            Pty = fold_bc!(copy(y), g)
            @test dot(Px, y) ≈ dot(x, Pty)
            ghosts_zeroed = copy(Pty)
            MatrixFreeOperators.zero_ghosts!(ghosts_zeroed, g)
            @test ghosts_zeroed == Pty
        end
    end

    @testset "fill/fold adjointness for SVector eltype" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (4, 4); bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann()))
        )
        rng = Random.MersenneTwister(11)
        x = [SVector{2,Float64}(rand(rng), rand(rng)) for _ in 1:6, _ in 1:6]
        y = [SVector{2,Float64}(rand(rng), rand(rng)) for _ in 1:6, _ in 1:6]
        Px = apply_bc!(copy(x), g)
        Pty = fold_bc!(copy(y), g)
        @test sum(dot.(Px, y)) ≈ sum(dot.(x, Pty))
    end
end
