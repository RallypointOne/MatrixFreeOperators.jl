@testset "PackedBlockField" begin
    MFO = MatrixFreeOperators
    fun = x -> sinpi(x[1]) * cospi(2x[2]) + 0.3 * x[1]
    vfun = x -> SVector(sinpi(x[1]) + 0.2 * x[2], cospi(x[2]) - x[1])

    bc = ((Dirichlet(), Dirichlet()), (Neumann(), Neumann()))
    g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=bc)
    bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)   # 16 leaves
    uf = set!(scalar_field(bf), fun)

    @testset "pack/unpack round trip is bit-exact" begin
        p = pack(uf)
        @test p isa PackedBlockField
        @test size(p.data) == (6, 6, MFO.nleaves(bf))   # (blocksize .+ 2halo ..., nleaves)
        @test eltype(p) === Float64
        @test collect(p.levels) == [k.level for k in bf.forest.leaves]
        for i in 1:MFO.nleaves(bf)
            @test MFO._block_array(p, i) == uf.blocks[i]
        end
        up = unpack(p)
        @test up isa BlockField
        for i in 1:MFO.nleaves(bf)
            @test up.blocks[i] == uf.blocks[i]
        end

        w = set!(vector_field(bf), vfun)                # SVector eltype packs too
        pw = pack(w)
        @test eltype(pw) === SVector{2,Float64}
        for i in 1:MFO.nleaves(bf)
            @test MFO._block_array(pw, i) == w.blocks[i]
        end
    end

    @testset "set! on packed matches pack ∘ set!" begin
        p = set!(MFO._zero_all!(pack(scalar_field(bf))), fun)
        @test p.data == pack(uf).data
    end

    @testset "flat boundary parity with BlockField" begin
        p = pack(uf)
        v = flatten(p)
        @test v == flatten(uf)
        r = MFO._zero_all!(similar(p))
        flat_to_interior!(r, v)
        for i in 1:MFO.nleaves(bf)
            @test interior(MFO.block(r, i)) == interior(MFO.block(uf, i))
        end
        out = 3.0 .* one.(v)
        ref = copy(out)
        interior_to_flat!(out, p, 2.0, 0.5)
        interior_to_flat!(ref, uf, 2.0, 0.5)
        @test out == ref
    end

    @testset "similar / copy / zero_ghosts!" begin
        p = pack(uf)
        s = similar(p)
        @test s isa PackedBlockField && size(s.data) == size(p.data)
        @test s.levels === p.levels                     # geometry SoA is shared
        sv = similar(p, SVector{2,Float64})             # rank-changer allocation path
        @test eltype(sv) === SVector{2,Float64} && size(sv.data) == size(p.data)
        c = copy(p)
        @test c.data == p.data
        c.data[1] += 1.0
        @test c.data[1] != p.data[1]                    # storage is independent
        z = MFO.zero_ghosts!(copy(p))
        for i in 1:MFO.nleaves(bf)
            @test interior(MFO.block(z, i)) == interior(MFO.block(p, i))
        end
        @test sum(z.data) ≈ sum(map(i -> sum(interior(MFO.block(p, i))), 1:MFO.nleaves(bf)))
    end

    @testset "regrid guards" begin
        bf2 = BlockForest(
            CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8)); blocksize=(4, 4), maxlevel=2
        )
        u2 = set!(scalar_field(bf2), fun)
        p2 = pack(u2)
        @test_throws ArgumentError regrid!(p2; refine=Returns(true))         # packed never regrids
        @test_throws ArgumentError regrid!(u2, p2; refine=Returns(true))     # nor mixed in varargs
        refine!(bf2, _ -> true)
        @test_throws ArgumentError MFO.block(p2, 1)     # stale after regrid
        @test_throws ArgumentError pack(u2)             # stale source refuses to pack
        @test_throws ArgumentError unpack(p2)
    end
end
