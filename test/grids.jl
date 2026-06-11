@testset "Grids" begin
    @testset "construction and derived quantities" begin
        g = CartesianGrid(((0.0, 1.0), (0.0, 2.0)), (4, 8))
        @test dimension(g) == 2
        @test spacing(g) == (0.25, 0.25)
        @test local_size(g) == (4, 8)
        @test halo_width(g) == (1, 1)
        @test padded_size(g) == (6, 10)
        @test interior(g) == CartesianIndices((2:5, 2:9))
        @test all(bc -> bc isa Tuple{Dirichlet{Int},Dirichlet{Int}}, boundary_conditions(g))
        @test KernelAbstractions.get_backend(g) == KernelAbstractions.CPU()

        g32 = CartesianGrid(((0.0f0, 1.0f0),), (10,))
        @test spacing(g32) === (0.1f0,)
    end

    @testset "validation" begin
        @test_throws ArgumentError CartesianGrid(((0.0, 1.0),), (0,))
        @test_throws ArgumentError CartesianGrid(((0.0, 1.0),), (4,); halo=(0,))
        @test_throws ArgumentError CartesianGrid(((1.0, 0.0),), (4,))
        @test_throws ArgumentError CartesianGrid(
            ((0.0, 1.0),), (4,); bc=((Periodic(), Dirichlet()),)
        )
        @test_throws ArgumentError CartesianGrid(((0.0, 1.0),), (4,); bc=((Periodic(),),))
    end

    @testset "cell centers" begin
        g = CartesianGrid(((0.0, 1.0),), (4,))
        @test cell_center(g, CartesianIndex(2)) ≈ [0.125]
        @test cell_center(g, CartesianIndex(5)) ≈ [0.875]
        @test cell_center(g, CartesianIndex(1)) ≈ [-0.125]

        g2 = CartesianGrid(((0.0, 1.0), (2.0, 4.0)), (4, 4))
        @test cell_center(g2, CartesianIndex(2, 2)) ≈ [0.125, 2.25]
        @test eltype(cell_center(g2, CartesianIndex(2, 2))) === Float64
    end

    @testset "isbits (GPU-capture invariant)" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)),
            (4, 4);
            bc=((Periodic(), Periodic()), (Dirichlet(), Neumann())),
        )
        @test isbits(g)
    end

    @testset "halo_update! no-op seam" begin
        g = CartesianGrid(((0.0, 1.0),), (4,))
        x = rand(6)
        x_before = copy(x)
        @test halo_update!(x, g) === x
        @test x == x_before
    end
end
