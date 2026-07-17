@testset "Forest topology" begin
    MFO = MatrixFreeOperators
    LeafKey = MFO.LeafKey

    # A face-neighbor 2:1-balance checker independent of balance!'s own logic.
    function is_balanced(forest)
        N = length(forest.nroot)
        for F in forest.leaves, dim in 1:N, side in (-1, 1)
            nbr = MFO.face_neighbor(forest, F, dim, side)
            nbr === nothing && continue
            cover = MFO.leaf_covering(forest, nbr)
            cover === nothing && continue          # neighbor is finer; checked from its side
            abs(F.level - cover.level) <= 1 || return false
        end
        return true
    end

    @testset "parent_key / children" begin
        @test MFO.parent_key(LeafKey(2, (2, 3))) == LeafKey(1, (1, 1))
        @test Set(MFO.children(LeafKey(0, (1,)))) == Set([LeafKey(1, (2,)), LeafKey(1, (3,))])
        kids = MFO.children(LeafKey(1, (1, 1)))
        @test length(kids) == 4
        @test Set(kids) == Set([LeafKey(2, c) for c in ((2, 2), (3, 2), (2, 3), (3, 3))])
        @test all(MFO.parent_key(c) == LeafKey(1, (1, 1)) for c in kids)
    end

    @testset "construction" begin
        forest = MFO.Forest((2, 2), (false, false), 3)
        @test MFO.nleaves(forest) == 4
        @test Set(forest.leaves) == Set([LeafKey(0, c) for c in ((0, 0), (1, 0), (0, 1), (1, 1))])
        @test all(k -> k.level == 0, forest.leaves)
        @test sort(collect(values(forest.index))) == collect(1:4)   # unique 1:nleaves
    end

    @testset "face_neighbor: interior / boundary / periodic" begin
        forest = MFO.Forest((2, 2), (false, false), 3)
        @test MFO.face_neighbor(forest, LeafKey(0, (0, 0)), 1, 1) == LeafKey(0, (1, 0))
        @test MFO.face_neighbor(forest, LeafKey(0, (0, 0)), 2, 1) == LeafKey(0, (0, 1))
        @test MFO.face_neighbor(forest, LeafKey(0, (0, 0)), 1, -1) === nothing   # domain boundary
        @test MFO.face_neighbor(forest, LeafKey(0, (1, 1)), 2, 1) === nothing

        per = MFO.Forest((2, 2), (true, true), 3)
        @test MFO.face_neighbor(per, LeafKey(0, (0, 0)), 1, -1) == LeafKey(0, (1, 0))  # wraps
        @test MFO.face_neighbor(per, LeafKey(0, (1, 1)), 2, 1) == LeafKey(0, (1, 0))
        # finer level wraps over nroot·2^level blocks
        @test MFO.face_neighbor(per, LeafKey(1, (0, 0)), 1, -1) == LeafKey(1, (3, 0))
    end

    @testset "uniform refine / coarsen round-trip" begin
        forest = MFO.Forest((2, 2), (false, false), 3)
        MFO.refine!(forest, _ -> true)
        @test MFO.nleaves(forest) == 16
        @test all(k -> k.level == 1, forest.leaves)
        @test is_balanced(forest)
        MFO.coarsen!(forest, _ -> true)
        @test MFO.nleaves(forest) == 4
        @test all(k -> k.level == 0, forest.leaves)
    end

    @testset "localized refine stays 2:1 balanced" begin
        forest = MFO.Forest((4, 4), (false, false), 4)
        for _ in 1:4
            MFO.refine!(forest, k -> k.coords == (0, 0))   # cascade-refine the origin corner
        end
        @test is_balanced(forest)
        @test maximum(k -> k.level, forest.leaves) == 4
        @test minimum(k -> k.level, forest.leaves) < maximum(k -> k.level, forest.leaves)  # graded
        @test MFO.nleaves(forest) > 16
    end

    @testset "refine respects maxlevel" begin
        forest = MFO.Forest((1, 1), (false, false), 2)
        for _ in 1:5
            MFO.refine!(forest, _ -> true)
        end
        @test maximum(k -> k.level, forest.leaves) == 2     # capped at maxlevel
        @test MFO.nleaves(forest) == 4^2                    # 2^(N·maxlevel) leaves
    end
end
