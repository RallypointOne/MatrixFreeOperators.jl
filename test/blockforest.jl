@testset "BlockForest grid" begin
    MFO = MatrixFreeOperators
    Interface = MFO.Interface
    LeafKey = MFO.LeafKey

    @testset "construction + divisibility" begin
        base = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (16, 16);
            bc=((Dirichlet(), Dirichlet()), (Dirichlet(), Dirichlet())),
        )
        bf = BlockForest(base; blocksize=(8, 8), maxlevel=4)
        @test MFO.nleaves(bf) == 4                 # nroot = (2, 2)
        @test dimension(bf) == 2
        @test_throws ArgumentError BlockForest(base; blocksize=(5, 8), maxlevel=2)
    end

    @testset "leaf geometry tiles the base at level 0" begin
        base = CartesianGrid(
            ((0.0, 2.0), (0.0, 1.0)), (8, 8);          # spacing (0.25, 0.125)
            bc=((Neumann(), Neumann()), (Dirichlet(), Dirichlet())),
        )
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=3)   # nroot (2, 2)
        grids = [g for (_, g) in leaves(bf)]
        @test length(grids) == 4
        for g in grids
            @test spacing(g) == base.spacing            # level 0 ⇒ same spacing
            @test local_size(g) == (4, 4)
        end
        xlos = sort(unique(g.extent[1][1] for g in grids))
        @test xlos ≈ [0.0, 1.0]                          # block width 4·0.25 = 1.0
    end

    @testset "leaf boundary conditions (Interface on internal faces)" begin
        base = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (16, 16);
            bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
        )
        bf = BlockForest(base; blocksize=(8, 8), maxlevel=2)   # nroot (2, 2)
        bc00 = MFO.leaf_bc(bf, LeafKey(0, (0, 0)))
        @test bc00[1] == (Dirichlet(), Interface())            # x: low domain, high internal
        @test bc00[2] == (Neumann(), Interface())
        bc11 = MFO.leaf_bc(bf, LeafKey(0, (1, 1)))
        @test bc11[1] == (Interface(), Dirichlet())
        @test bc11[2] == (Interface(), Neumann())
    end

    @testset "periodic ⇒ every face is Interface" begin
        base = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (8, 8);
            bc=((Periodic(), Periodic()), (Periodic(), Periodic())),
        )
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=2)
        for (k, _) in leaves(bf)
            bc = MFO.leaf_bc(bf, k)
            @test all(face -> face[1] isa Interface && face[2] isa Interface, bc)
        end
    end

    @testset "uniform refine halves spacing, multiplies leaves" begin
        base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8))
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=3)   # nroot (2, 2)
        refine!(bf, _ -> true)
        @test MFO.nleaves(bf) == 4 * 4                          # ×2ᴺ
        for (k, g) in leaves(bf)
            @test all(spacing(g) .≈ base.spacing ./ 2)
            @test k.level == 1
        end
    end

    @testset "leaf_center" begin
        base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (4, 4))
        bf = BlockForest(base; blocksize=(2, 2), maxlevel=2)   # block width 0.5
        @test MFO.leaf_center(bf, LeafKey(0, (0, 0))) ≈ [0.25, 0.25]
        @test MFO.leaf_center(bf, LeafKey(0, (1, 1))) ≈ [0.75, 0.75]
    end

    @testset "Adapt to Array preserves type" begin
        base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8))
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=2)
        bf2 = Adapt.adapt(Array, bf)
        @test bf2 isa BlockForest
        @test MFO.nleaves(bf2) == MFO.nleaves(bf)
    end
end
