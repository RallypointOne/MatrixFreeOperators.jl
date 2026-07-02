@testset "BlockField" begin
    MFO = MatrixFreeOperators

    base = CartesianGrid(
        ((0.0, 1.0), (0.0, 1.0)), (8, 8);
        bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
    )

    @testset "allocation" begin
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=3)   # nroot (2, 2)
        f = scalar_field(bf)
        @test length(f.blocks) == MFO.nleaves(bf) == 4
        @test all(b -> size(b) == (4 + 2, 4 + 2), f.blocks)    # blocksize + 2·halo
        @test all(b -> all(iszero, b), f.blocks)
        @test eltype(f) == Float64
        @test ncomponents(f) == 1

        vf = vector_field(bf)
        @test eltype(vf) == SVector{2,Float64}
        @test ncomponents(vf) == 2
    end

    @testset "set! matches per-leaf standalone field" begin
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=3)
        f = scalar_field(bf)
        set!(f, x -> x[1] + 2x[2])
        for i in 1:MFO.nleaves(bf)
            ref = set!(scalar_field(MFO.leaf_grid(bf, i)), x -> x[1] + 2x[2])
            @test collect(interior(MFO.block(f, i))) ≈ collect(interior(ref))
        end
    end

    @testset "flat ↔ interior round-trip (uniform)" begin
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=3)
        f = scalar_field(bf)
        set!(f, x -> sin(3x[1]) * cos(2x[2]))
        v = flatten(f)
        @test length(v) == MFO.flat_length(f) == 4 * 16   # nleaves·prod(blocksize)
        g = scalar_field(bf)
        flat_to_interior!(g, v)
        @test flatten(g) == v                              # exact round-trip
    end

    @testset "flat ↔ interior round-trip (non-uniform forest)" begin
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=4)
        refine!(bf, x -> x[1] < 0.4 && x[2] < 0.4)         # refine a corner region
        refine!(bf, x -> x[1] < 0.2 && x[2] < 0.2)
        f = scalar_field(bf)
        set!(f, x -> x[1]^2 - x[2])
        v = flatten(f)
        @test length(v) == MFO.flat_length(f) == MFO.nleaves(bf) * 16
        g = scalar_field(bf)
        flat_to_interior!(g, v)
        @test flatten(g) == v
    end

    @testset "interior_to_flat! axpby" begin
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=3)
        f = scalar_field(bf)
        set!(f, x -> x[1] - x[2])
        v = flatten(f)
        v2 = copy(v)
        interior_to_flat!(v2, f, 2.0, 3.0)                 # 2·interior + 3·v2 = 5v
        @test v2 ≈ 5 .* v
    end

    @testset "vector field round-trip" begin
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=3)
        vf = vector_field(bf)
        set!(vf, x -> SVector(x[1], x[2]))
        w = flatten(vf)
        @test length(w) == MFO.flat_length(vf) == 4 * 16 * 2
        g = vector_field(bf)
        flat_to_interior!(g, w)
        @test flatten(g) == w
    end
end
