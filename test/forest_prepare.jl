@testset "Prepared forest (PreparedForest)" begin
    MFO = MatrixFreeOperators
    fun = x -> sinpi(x[1]) * cospi(2x[2]) + 0.3 * x[1]

    # A 4×4 tiling has interior, edge, and corner blocks; physical BCs live on the
    # forest and are applied by the face passes, so every leaf grid is one concrete
    # all-Interface type and the prepared sweep is type-stable by construction.
    bc = ((Dirichlet(), Dirichlet()), (Neumann(), Neumann()))
    g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=bc)
    bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)   # 16 leaves
    uf = set!(scalar_field(bf), fun)
    v = flatten(uf)
    out = similar(v)

    # Measure behind a function barrier so operands are typed arguments — @allocated at
    # global scope would charge type-unstable global access to the call.
    function alloc_mul(P, out, v)
        mul!(out, P, v)
        mul!(out, P, v)
        return @allocated mul!(out, P, v)
    end

    # Leaf grids are isbits and type-stable, so the sweep rebuilds them for free; the
    # residual is the per-leaf stencil apply, which is allocation-free only when
    # inlined into one mul! (tracked as the alloc-free-kernel follow-up). This loose
    # bound guards against a type-instability regressing; the exact type-stability
    # guard is the @inferred test below.
    alloc_bound(nl) = 1000 * nl

    @testset "forward mul! is correct" begin
        A = prepare(laplacian(bf))
        @test A isa PreparedForest
        @test size(A) == (256, 256)                 # nleaves·prod(blocksize) = 16·16
        @test eltype(A) === Float64
        mul!(out, A, v)
        @test out == flatten(laplacian(bf) * uf)    # matches the un-prepared path
        @test alloc_mul(A, out, v) ≤ alloc_bound(MFO.nleaves(bf))
    end

    @testset "adjoint / accumulating / combinator mul!" begin
        Df = derivative(bf, 1; order=1)             # order 1 ⇒ adjoint is a PreparedAdjoint
        At = prepare(adjoint(Df))
        mul!(out, At, v)
        @test out ≈ flatten(adjoint(Df) * copy(uf))
        @test alloc_mul(At, out, v) ≤ alloc_bound(MFO.nleaves(bf))

        out2 = copy(v)
        A = materialize(prepare(Df))
        mul!(out2, At, v, 2.0, 3.0)                 # accumulating adjoint uses adjscratch
        @test out2 ≈ 2.0 .* (A' * v) .+ 3.0 .* v

        Sf = laplacian(bf) + adjoint(Df)            # Added(leaf, PreparedAdjoint)
        As = prepare(Sf)
        mul!(out, As, v)
        @test out ≈ flatten(Sf * copy(uf))
        @test alloc_mul(As, out, v) ≤ alloc_bound(MFO.nleaves(bf))
    end

    # The forest walk has the same gap as the grid one: no cached adjoint action for
    # a PreparedComposed, and its tmp is shaped for the forward pass. prepare
    # normalizes AdjointOp(A*B) to Composed(Bᵀ, Aᵀ) first, so the nested form runs
    # through the ordinary per-node recursion (and its inter-block exchange).
    @testset "prepare normalizes AdjointOp over a composite" begin
        inner = derivative(bf, 1; order=1) * laplacian(bf)
        At = prepare(MFO.AdjointOp(inner))
        @test At.op isa MFO.PreparedComposed
        mul!(out, At, v)
        @test out ≈ materialize(prepare(inner))' * v
        @test alloc_mul(At, out, v) ≤ alloc_bound(MFO.nleaves(bf))

        κ = set!(scalar_field(bf), x -> 1 + x[1] * x[2])
        K = divergence(bf) * scaling(κ)                          # vector → scalar
        Kt = prepare(MFO.AdjointOp(K), scalar_field(bf))
        B = materialize(prepare(K, vector_field(bf)))
        @test size(Kt) == (2 * 256, 256)
        @test materialize(Kt) ≈ B'
    end

    @testset "cached apply is type-stable (the allocation guard)" begin
        A = prepare(laplacian(bf))
        infer(P) = @inferred MFO._forest_capply!(P.ypad, P.op, P.xpad, P, true, false)
        @test infer(A) === A.ypad
    end

    @testset "regrid invalidates a prepared forest (generation guard)" begin
        bf3 = BlockForest(
            CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8)); blocksize=(4, 4), maxlevel=2
        )
        A3 = prepare(laplacian(bf3))
        x3 = rand(size(A3, 2))
        y3 = similar(x3)
        @test mul!(y3, A3, x3) === y3               # valid before regrid
        refine!(bf3, _ -> true)                     # bumps the forest generation
        @test_throws ArgumentError mul!(y3, A3, x3) # stale prepared operator now errors
    end
    @testset "Added shares one exchange in the prepared hot path (issue #85)" begin
        aniso = 0.13 * laplacian(bf) + 0.7 * derivative(bf, 1; order=2)
        @test shares_exchange(aniso)
        # the exchange count on the cached path, with the prepared scratch wrapped
        function prepared_exchanges(A, v)
            c = CountingBlockField(A.xpad)
            flat_to_interior!(A.xpad, v)
            MFO._forest_capply!(A.ypad, A.op, c, A, true, false)
            return c.exchanges[], c.bcfills[]
        end
        # the un-shared reference, one full forest action per operand
        function per_operand_flat(L::Added, x)
            y = apply!(similar(x), L.a, x, x.grid)
            apply!(y, L.b, x, x.grid, true, true)
            return flatten(y)
        end

        A = prepare(aniso)
        @test prepared_exchanges(A, v) == (1, 1)
        @test prepared_exchanges(prepare(laplacian(bf)), v) == (1, 1)   # the baseline
        mul!(out, A, v)
        @test out == per_operand_flat(aniso, uf)             # bit-identical
        @test out == flatten(aniso * copy(uf))               # and to the pure path
        @test alloc_mul(A, out, v) ≤ alloc_bound(MFO.nleaves(bf))
        # accumulating mul! blends on top of the shared sweep
        out2 = copy(v)
        mul!(out2, A, v, 2.0, 3.0)
        @test out2 ≈ 2.0 .* per_operand_flat(aniso, uf) .+ 3.0 .* v
        # nested sums and a diagonal operand share too
        κ = set!(scalar_field(bf), x -> 1 + x[1] * x[2])
        three = (aniso + scaling(κ)) + identity_op()
        A3 = prepare(three)
        @test prepared_exchanges(A3, v) == (1, 1)
        mul!(out, A3, v)
        @test out == flatten(three * copy(uf))
        # packed scratch: the counter wraps A.xpad, so this count sees the packed
        # layout only through storage/`block` forwarding — dispatch lands on the
        # AbstractBlockField sweep (also where the PackedBlockField overrides fall
        # back to on CPU). The unwrapped mul! below is what reaches those
        # overrides, and must agree bit-for-bit with the per-operand reference.
        Ap = prepare(aniso, pack(uf))
        @test Ap.xpad isa PackedBlockField
        @test prepared_exchanges(Ap, v) == (1, 1)
        mul!(out, Ap, v)
        @test out == per_operand_flat(aniso, uf)
        # prepared nodes never share: a PreparedAdjoint or PreparedComposed operand
        # keeps the per-operand actions (one exchange on x per non-adjoint operand)
        Aadj = prepare(laplacian(bf) + derivative(bf, 1; order=1)')
        @test Aadj.op isa Added && Aadj.op.b isa MFO.PreparedAdjoint
        @test !shares_exchange(Aadj.op)
        @test prepared_exchanges(Aadj, v) == (1, 1)
        Acomp = prepare(laplacian(bf) + derivative(bf, 1; order=1) * laplacian(bf))
        @test Acomp.op.b isa MFO.PreparedComposed
        @test !shares_exchange(Acomp.op)
        @test prepared_exchanges(Acomp, v) == (2, 2)
        # self-adjoint sum: the prepared adjoint takes the shared forward path and
        # the dense matrix is symmetric; a non-self-adjoint sum keeps its transpose
        At = prepare(adjoint(aniso))
        @test prepared_exchanges(At, v) == (1, 1)
        M = materialize(A)
        @test M ≈ M'
        @test materialize(At) ≈ M'
        skew = 0.13 * laplacian(bf) + 0.7 * derivative(bf, 1; order=1)
        @test materialize(prepare(adjoint(skew))) ≈ materialize(prepare(skew))'
        # an adapted forest: the sum is no longer self-adjoint, the forward still
        # shares, and the prepared transpose is exact
        bfr = BlockForest(g; blocksize=(4, 4), maxlevel=3)
        refine!(bfr, x -> x[1] < 0.5 && x[2] < 0.5)
        balance!(bfr)
        ur = set!(scalar_field(bfr), fun)
        vr = flatten(ur)
        anisor = 0.13 * laplacian(bfr) + 0.7 * derivative(bfr, 1; order=2)
        @test !isselfadjoint(anisor)
        Ar = prepare(anisor)
        @test prepared_exchanges(Ar, vr) == (1, 1)
        outr = similar(vr)
        mul!(outr, Ar, vr)
        @test outr == per_operand_flat(anisor, ur)
        @test materialize(prepare(adjoint(anisor))) ≈ materialize(Ar)'
    end
end
