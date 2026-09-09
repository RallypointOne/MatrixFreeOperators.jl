# Top level (not in a @testset): struct definitions need global scope.
struct UnsupportedBC <: MatrixFreeOperators.AbstractBC end

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
        # physical BCs without a face-pass implementation are rejected at construction
        for bad in (
            ((UnsupportedBC(), UnsupportedBC()), (Dirichlet(), Dirichlet())),
            ((Dirichlet(), Dirichlet()), (Interface(), Interface())),
        )
            badbase = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=bad)
            @test_throws ArgumentError BlockForest(badbase; blocksize=(8, 8), maxlevel=2)
        end
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

    @testset "leaf grids are all-Interface (one concrete type)" begin
        # Physical BCs live on bf.bc, applied by the forest-level face pass; every
        # leaf grid is the same concrete all-Interface type, so leaf_grid is
        # type-stable.
        base = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (16, 16);
            bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
        )
        bf = BlockForest(base; blocksize=(8, 8), maxlevel=2)   # nroot (2, 2)
        g1 = @inferred MFO.leaf_grid(bf, 1)
        for (_, g) in leaves(bf)
            @test typeof(g) === typeof(g1)
            @test all(f -> f[1] isa Interface && f[2] isa Interface, boundary_conditions(g))
        end
        @test bf.bc === base.bc
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
        # the schedule Ref is shared and stays concretely typed across adaptation
        @test bf2.schedule === bf.schedule
        @test eltype(bf2.schedule) === eltype(bf.schedule) === MFO.ExchangeSchedule{2,Float64}
        @test isconcretetype(eltype(bf2.schedule))
    end

    @testset "operators apply across refinement levels (coarse–fine phase)" begin
        bcd = ((Dirichlet(), Dirichlet()), (Dirichlet(), Dirichlet()))
        base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8); bc=bcd)
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=2)
        refine!(bf, x -> x[1] < 0.5 && x[2] < 0.5)     # one corner root → mixed levels
        @test !bf.forest.uniform[]
        uf = set!(scalar_field(bf), x -> x[1]^2 + x[2]^2)   # Δu = 4 exactly
        Lu = laplacian(bf) * uf
        # Every stencil fed only by interior/interface ghosts must be exact; skip
        # the one-cell layer whose stencil reads a homogeneous physical-BC ghost.
        for (i, key) in enumerate(bf.forest.leaves)
            nblocks = ntuple(d -> bf.forest.nroot[d] << key.level, 2)
            vals = collect(interior(MFO.block(Lu, i)))
            for I in CartesianIndices(vals)
                skip = any(
                    d ->
                        (key.coords[d] == 0 && I[d] == 1) ||
                        (key.coords[d] == nblocks[d] - 1 && I[d] == bf.blocksize[d]),
                    1:2,
                )
                skip || @test vals[I] ≈ 4 atol = 1e-10
            end
        end
    end

    @testset "fields allocated before a regrid are rejected" begin
        base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8))
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=2)
        uf = scalar_field(bf)
        refine!(bf, _ -> false)                         # no-op regrid keeps fields valid
        @test set!(uf, x -> x[1]) isa BlockField
        refine!(bf, _ -> true)                          # uniform level 1: new leaf set
        @test_throws ArgumentError laplacian(bf) * uf
        @test_throws ArgumentError flatten(uf)
        @test_throws ArgumentError set!(uf, x -> x[1])
        # copies and similars of a stale field are equally stale
        @test_throws ArgumentError set!(copy(uf), x -> x[1])
        uf2 = scalar_field(bf)                          # fresh allocation works
        @test flatten(set!(uf2, x -> x[1])) isa Vector
        # coarsening back to the original leaf count is still a different generation
        coarsen!(bf, _ -> true)
        @test MFO.nleaves(bf) == 4
        @test_throws ArgumentError set!(uf, x -> x[1])
        @test_throws ArgumentError set!(uf2, x -> x[1])
    end
end
