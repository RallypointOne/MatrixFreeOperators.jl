# Adaptor that doubles every Array it reaches — proves an Adapt rule recursed into a
# wrapper's contents (the _AsScalar broadcast scalar of the fused ghost gather).
struct DoubleAdaptor end
Adapt.adapt_storage(::DoubleAdaptor, x::Array) = 2 .* x

# Block storage that counts what a sweep does to it: every element write, and every
# broadcast materialized into a view of it. This is the structural guard on the
# fused gather — a per-term loop over a K-wide row issues K broadcasts and writes
# each dst cell K times, the fused gather issues one and writes each cell once, and
# no numerical test can tell them apart.
struct ProbeArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::A
    writes::Base.RefValue{Int}
    bcasts::Base.RefValue{Int}
end
ProbeArray(a::AbstractArray, w::Base.RefValue{Int}, b::Base.RefValue{Int}) =
    ProbeArray{eltype(a),ndims(a),typeof(a)}(a, w, b)
Base.size(p::ProbeArray) = size(p.parent)
Base.IndexStyle(::Type{<:ProbeArray}) = IndexCartesian()
Base.@propagate_inbounds Base.getindex(p::ProbeArray{T,N}, I::Vararg{Int,N}) where {T,N} =
    p.parent[I...]
Base.@propagate_inbounds function Base.setindex!(
    p::ProbeArray{T,N}, v, I::Vararg{Int,N}
) where {T,N}
    p.writes[] += 1
    return p.parent[I...] = v
end
const ProbeView{T,N} = SubArray{T,N,<:ProbeArray}
function Base.copyto!(dst::ProbeView, bc::Base.Broadcast.Broadcasted{Nothing})
    parent(dst).bcasts[] += 1
    return invoke(copyto!, Tuple{AbstractArray,Base.Broadcast.Broadcasted{Nothing}}, dst, bc)
end

@testset "Exchange schedule" begin
    MFO = MatrixFreeOperators

    # 2×2 root tiling (nroot = base ncells ÷ blocksize): 4 level-0 leaves, each with
    # interior, edge, and corner faces — the smallest forest exercising every face case.
    make_bf(bc) = BlockForest(
        CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8); bc=bc); blocksize=(4, 4), maxlevel=2
    )
    periodic = ((Periodic(), Periodic()), (Periodic(), Periodic()))
    dirichlet = ((Dirichlet(), Dirichlet()), (Dirichlet(), Dirichlet()))
    mixed = ((Periodic(), Periodic()), (Dirichlet(), Dirichlet()))   # periodic-x, Dirichlet-y

    # Independent recount of same-level faces straight from the topology (not the schedule).
    function count_faces(bf)
        n = 0
        for K in bf.forest.leaves, d in 1:2, s in (-1, 1)
            nbr = MFO.face_neighbor(bf.forest, K, d, s)
            nbr !== nothing && MFO.is_leaf(bf.forest, nbr) && (n += 1)
        end
        return n
    end

    @testset "build correctness: counts and index bands" begin
        h, n = (1, 1), (4, 4)
        full = ntuple(t -> 1:(n[t] + 2h[t]), 2)           # (1:6, 1:6)
        for (bc, expected) in ((dirichlet, 8), (periodic, 16), (mixed, 12))
            bf = make_bf(bc)
            sched = MFO._exchange_schedule(bf)
            @test length(sched.copies) == count_faces(bf) == expected
            for c in sched.copies
                # exactly one normal dim; transverse dims span the full padded extent
                normal = findall(d -> c.dst_ranges[d] != full[d], 1:2)
                @test length(normal) == 1
                d = normal[1]
                @test all(t -> t == d || c.dst_ranges[t] == full[t], 1:2)
                @test all(t -> t == d || c.src_ranges[t] == full[t], 1:2)
                # dst is a ghost band, src the matching boundary-interior band
                low = (c.dst_ranges[d] == 1:h[d])
                high = (c.dst_ranges[d] == (h[d] + n[d] + 1):(2h[d] + n[d]))
                @test low ⊻ high
                expected_src = low ? ((n[d] + 1):(n[d] + h[d])) : ((h[d] + 1):(2h[d]))
                @test c.src_ranges[d] == expected_src
            end
        end
    end

    @testset "touch-exactly-once (per ghost box)" begin
        # Each ghost slab is the dst of exactly one descriptor; the adjoint fold-then-zero
        # relies on this. (Face boxes span the full transverse extent, so distinct boxes
        # still share corner cells — box uniqueness, the invariant Part 2 inherits.)
        for bc in (dirichlet, periodic, mixed)
            sched = MFO._exchange_schedule(make_bf(bc))
            keys = [(c.dst, c.dst_ranges) for c in sched.copies]
            @test allunique(keys)
        end
    end

    @testset "cfflux: coarse–fine face-flux descriptors" begin
        # Weight-free topology records for the Diffusion coarse-ghost rewrite: one
        # per (coarse face × abutting fine child), riding the same _emit_restrict!
        # child walk — so exactly one per restriction descriptor, dst-identical to
        # its restriction twin, and none on a uniform forest.
        for bc in (dirichlet, periodic, mixed)
            @test isempty(MFO._exchange_schedule(make_bf(bc)).cfflux)
            bfr = make_bf(bc)
            refine!(bfr, x -> x[1] < 0.5 && x[2] < 0.5)
            sched = MFO._exchange_schedule(bfr)
            @test !isempty(sched.cfflux)
            @test length(sched.cfflux) == length(sched.restrict)
            @test all(
                r.coarse == f.dst_block && r.gC == f.dst_ranges for
                (r, f) in zip(sched.cfflux, sched.restrict)
            )
            @test allunique([(r.coarse, r.gC) for r in sched.cfflux])
            # fully isbits — the shape that keeps a per-apply descriptor loop out of
            # Enzyme's issue-#26 territory
            @test isbitstype(eltype(sched.cfflux))
        end
    end

    @testset "coarse–fine fills: isbits CSR descriptors with dimension-fixed rows" begin
        # Every fill keeps its terms as a [tfirst, tlast] row into the phase's flat
        # term buffer — the same CSR layout the device schedule uploads — so both
        # descriptor vectors have isbits eltypes and the schedule type is concrete
        # from (N, T) alone. The row width is a function of N alone (interp
        # 1 + 2·3^(N-1), restrict 1 + 2^N): the emitters loop over a tensor product
        # of per-tangential-dim stencils, never over topology. Checked on 1D/2D/3D
        # forests on interior, boundary-adjacent and cascaded (multi-level) faces.
        for N in (1, 2, 3), T in (Float64,)
            KI, KR = 1 + 2 * 3^(N - 1), 1 + 2^N
            @test MFO._ninterp_terms(Val(N)) == KI
            @test MFO._nrestrict_terms(Val(N)) == KR
            @test isbitstype(MFO.GhostFill{N})
            @test isbitstype(MFO.SlabTerm{N,T})
            @test isconcretetype(MFO.ExchangeSchedule{N,T})
            ext = ntuple(_ -> (zero(T), one(T)), N)
            for bc in (
                ntuple(_ -> (Periodic(), Periodic()), N),
                ntuple(d -> d == 1 ? (Dirichlet(), Neumann()) : (Periodic(), Periodic()), N),
            )
                g = CartesianGrid(ext, ntuple(_ -> 16, N); bc=bc)
                bf = BlockForest(g; blocksize=ntuple(_ -> 4, N), maxlevel=3)
                @test eltype(bf.schedule) === MFO.ExchangeSchedule{N,T}
                refine!(bf, x -> x[1] < 0.5)                  # one interface plane
                refine!(bf, x -> x[1] < 0.25)                 # balance cascade ⇒ 3 levels
                sched = @inferred MFO._exchange_schedule(bf)
                @test typeof(sched) === MFO.ExchangeSchedule{N,T}
                @test bf.schedule[] === sched
                @test !isempty(sched.interp) && !isempty(sched.restrict)
                for (fills, terms, K) in (
                    (sched.interp, sched.interp_terms, KI),
                    (sched.restrict, sched.restrict_terms, KR),
                )
                    @test eltype(fills) === MFO.GhostFill{N}
                    @test eltype(terms) === MFO.SlabTerm{N,T}
                    @test isbitstype(eltype(fills)) && isbitstype(eltype(terms))
                    # rows are K wide, contiguous, in fill order, and partition the buffer
                    @test length(terms) == K * length(fills)
                    @test all(
                        f.tfirst == (i - 1) * K + 1 && f.tlast == i * K for
                        (i, f) in enumerate(fills)
                    )
                    @test all(f -> length(MFO._fill_terms(terms, f)) == K, fills)
                    @test all(t -> t.weight isa T, terms)
                end
                # the fill count itself still follows topology: every fine face
                # under a coarse neighbor is tiled by 4^(N-1) interp fills (one
                # per tangential-class combination — 4 classes per tangential
                # dim: two one-sided extreme columns and two centered parities),
                # every coarse face over fine children by 2^(N-1) restriction
                # fills (one per abutting child)
                ninterp_faces = nrestrict_faces = 0
                for K in bf.forest.leaves, d in 1:N, sd in (-1, 1)
                    nbr = MFO.face_neighbor(bf.forest, K, d, sd)
                    (nbr === nothing || MFO.is_leaf(bf.forest, nbr)) && continue
                    if MFO.leaf_covering(bf.forest, nbr) !== nothing
                        ninterp_faces += 1
                    else
                        nrestrict_faces += 1
                    end
                end
                @test length(sched.interp) == ninterp_faces * 4^(N - 1)
                @test length(sched.restrict) == nrestrict_faces * 2^(N - 1)
                # weights are the ones the emitters declare: interpolation
                # partitions unity (a constant field is reproduced exactly), the
                # restriction's own-cell term is 1 and its fine terms ±2/2^(N-1)
                iterms = [MFO._fill_terms(sched.interp_terms, f) for f in sched.interp]
                rterms = [MFO._fill_terms(sched.restrict_terms, f) for f in sched.restrict]
                @test all(ts -> sum(t.weight for t in ts) ≈ 1, iterms)
                @test all(
                    ts[1].block == f.dst_block for (ts, f) in zip(iterms, sched.interp)
                )   # own cell
                wf = T(2) / 2^(N - 1)
                @test all(
                    ts[1].weight == 1 && ts[1].block == f.dst_block for
                    (ts, f) in zip(rterms, sched.restrict)
                )
                @test all(ts -> all(t -> abs(t.weight) == wf, ts[2:end]), rterms)
                @test all(ts -> sum(t.weight for t in ts) ≈ 1, rterms)
            end
        end
        # `_close_fill` asserts the three row invariants the sweeps index by: width,
        # every term window shaped like the dst slab, and every term window disjoint
        # from it. All three are emitter bugs — a ragged row, an out-of-bounds read
        # in the fused gather, and a fill that reads the cells it writes (which the
        # gather could not see, its term views riding a non-array broadcast scalar).
        box = (1:1:1, 2:1:3)                                    # 1×2 dst on block 1
        good(b, r) = MFO.SlabTerm{2,Float64}(b, r, 0.5)
        row = [good(2, (4:1:4, 2:1:3)), good(2, (5:1:5, 2:1:3))]
        @test MFO._close_fill(row, 1, box, 1, 2) === MFO.GhostFill{2}(1, box, 1, 2)
        @test_throws AssertionError MFO._close_fill(row, 1, box, 1, 3)   # width
        @test_throws AssertionError MFO._close_fill(                     # width
            [row; good(2, (6:1:6, 2:1:3))], 1, box, 1, 2
        )
        @test_throws AssertionError MFO._close_fill(                     # shape
            [row[1], good(2, (5:1:5, 2:1:4))], 1, box, 1, 2
        )
        @test_throws AssertionError MFO._close_fill(                     # reads its dst
            [row[1], good(1, (1:1:1, 2:1:3))], 1, box, 1, 2
        )
        # a same-block term that misses the dst cells is fine (step-2 windows
        # interleave all over the coarse–fine descriptors)
        @test MFO._close_fill([row[1], good(1, (1:1:1, 4:1:5))], 1, box, 1, 2) isa
            MFO.GhostFill{2}
        # and both invariants hold of every row a real emitter produces
        for N in (2, 3)
            ext = ntuple(_ -> (0.0, 1.0), N)
            g = CartesianGrid(ext, ntuple(_ -> 16, N); bc=ntuple(_ -> (Dirichlet(), Neumann()), N))
            bfe = BlockForest(g; blocksize=ntuple(_ -> 4, N), maxlevel=3)
            refine!(bfe, x -> x[1] < 0.5)
            refine!(bfe, x -> x[1] < 0.25)
            sched = MFO._exchange_schedule(bfe)
            for (fills, terms) in
                ((sched.interp, sched.interp_terms), (sched.restrict, sched.restrict_terms))
                @test !isempty(fills)
                for f in fills, t in MFO._fill_terms(terms, f)
                    @test length.(t.ranges) == length.(f.dst_ranges)
                    @test !MFO._boxes_overlap(f.dst_block, f.dst_ranges, t.block, t.ranges)
                end
            end
        end
    end

    @testset "bcfaces: physical-face lists per (dim, side)" begin
        # 2×2 tiling: 2 leaves per non-periodic domain side; periodic dims stay empty.
        for (bc, counts) in
            ((dirichlet, (2, 2, 2, 2)), (periodic, (0, 0, 0, 0)), (mixed, (0, 0, 2, 2)))
            bf = make_bf(bc)
            sched = MFO._exchange_schedule(bf)
            idx = 0
            for d in 1:2, (s, list) in zip((-1, 1), sched.bcfaces[d])
                idx += 1
                want = [
                    i for (i, K) in enumerate(bf.forest.leaves) if
                    MFO.face_neighbor(bf.forest, K, d, s) === nothing
                ]
                @test list == want
                @test length(list) == counts[idx]
            end
        end
        # regeneration after refine!: lists follow the new leaf set
        bf = make_bf(dirichlet)
        refine!(bf, _ -> true)                          # 4×4 leaves at level 1
        sched = MFO._exchange_schedule(bf)
        for d in 1:2, (s, list) in zip((-1, 1), sched.bcfaces[d])
            want = [
                i for (i, K) in enumerate(bf.forest.leaves) if
                MFO.face_neighbor(bf.forest, K, d, s) === nothing
            ]
            @test list == want
            @test length(list) == 4
        end
    end

    @testset "halo_update! fills ghosts from the physical neighbor" begin
        # Independent of the schedule: compare each block's face-ghost interior band to
        # the same-level neighbor's boundary-interior band (topology recomputed here).
        h, n = (1, 1), (4, 4)
        interior_t = (h[1] + 1):(h[1] + n[1])             # transverse-interior band 2:5
        for bc in (periodic, mixed)
            bf = make_bf(bc)
            f = set!(scalar_field(bf), x -> sinpi(x[1]) * cospi(2x[2]) + 0.5x[1])
            halo_update!(f, bf)
            for (i, K) in enumerate(bf.forest.leaves), d in 1:2, s in (-1, 1)
                nbr = MFO.face_neighbor(bf.forest, K, d, s)
                (nbr === nothing && continue)
                MFO.is_leaf(bf.forest, nbr) || continue
                j = MFO.leaf_index(bf.forest, nbr)
                gidx = s == -1 ? 1 : h[d] + n[d] + 1       # this block's ghost line
                sidx = s == -1 ? n[d] + 1 : h[d] + 1       # neighbor's boundary interior
                gbox = ntuple(t -> t == d ? (gidx:gidx) : interior_t, 2)
                sbox = ntuple(t -> t == d ? (sidx:sidx) : interior_t, 2)
                @test f.blocks[i][gbox...] == f.blocks[j][sbox...]
            end
        end
    end

    @testset "halo_update_adjoint! is the exact transpose (⟨Hx,w⟩ = ⟨x,Hᵀw⟩)" begin
        # Holds over full padded storage when w's corner ghosts are zero — the documented
        # exactness precondition (axis-aligned stencil adjoints leave corner ghosts zero).
        h, n = (1, 1), (4, 4)
        is_corner(I) = count(d -> I[d] <= h[d] || I[d] > h[d] + n[d], 1:2) >= 2
        full_dot(a, b) = sum(i -> dot(vec(a.blocks[i]), vec(b.blocks[i])), 1:MFO.nleaves(a.grid))
        rng = Random.MersenneTwister(11)
        for bc in (dirichlet, periodic, mixed)
            bf = make_bf(bc)
            x = scalar_field(bf)
            w = scalar_field(bf)
            for i in 1:MFO.nleaves(bf)
                rand!(rng, x.blocks[i])
                rand!(rng, w.blocks[i])
                for I in CartesianIndices(w.blocks[i])
                    is_corner(I) && (w.blocks[i][I] = 0)
                end
            end
            Hx = halo_update!(copy(x), bf)
            Htw = MFO.halo_update_adjoint!(copy(w), bf)
            @test full_dot(Hx, w) ≈ full_dot(x, Htw)
        end
    end

    @testset "coarse–fine fills: adjoint identity and constant reproduction" begin
        # The fills' adjoint has no device twin, so it is covered here directly:
        # ⟨Hx,w⟩ = ⟨x,Hᵀw⟩ over full padded storage on refined 2D and 3D forests
        # (interior + boundary-adjacent + cascaded interfaces), w's corner ghosts
        # zeroed per the documented exactness precondition. And since every
        # interpolation stencil partitions unity and the restriction is
        # flux-matching, a constant field must come back constant on every
        # coarse–fine ghost the fills own — an independent check on the weights
        # frozen into each CSR row. Both element types: the weights are formed as
        # T inside the emitters, so Float32 is the check that no term weight is
        # computed in Float64 and narrowed (which would still pass a Float64-only
        # run) — tolerances scale with eps(T).
        rng = Random.MersenneTwister(23)
        for N in (2, 3), T in (Float64, Float32)
            ext = ntuple(_ -> (zero(T), one(T)), N)
            bc = ntuple(d -> d == 1 ? (Dirichlet(), Neumann()) : (Periodic(), Periodic()), N)
            g = CartesianGrid(ext, ntuple(_ -> 16, N); bc=bc)
            bf = BlockForest(g; blocksize=ntuple(_ -> 4, N), maxlevel=3)
            refine!(bf, x -> x[1] < 0.5)
            refine!(bf, x -> x[1] < 0.25 && x[2] < 0.5)
            sched = MFO._exchange_schedule(bf)
            @test !isempty(sched.interp) && !isempty(sched.restrict)
            @test eltype(sched.interp_terms) === MFO.SlabTerm{N,T}
            h, n = bf.halo, bf.blocksize
            is_corner(I) = count(d -> I[d] <= h[d] || I[d] > h[d] + n[d], 1:N) >= 2
            full_dot(a, b) =
                sum(i -> dot(vec(a.blocks[i]), vec(b.blocks[i])), 1:MFO.nleaves(a.grid))
            x = scalar_field(bf)
            w = scalar_field(bf)
            @test eltype(x.blocks[1]) === T
            for i in 1:MFO.nleaves(bf)
                rand!(rng, x.blocks[i])
                rand!(rng, w.blocks[i])
                for I in CartesianIndices(w.blocks[i])
                    is_corner(I) && (w.blocks[i][I] = 0)
                end
            end
            Hx = halo_update!(copy(x), bf)
            Htw = MFO.halo_update_adjoint!(copy(w), bf)
            lhs, rhs = full_dot(Hx, w), full_dot(x, Htw)
            @test lhs ≈ rhs rtol = 20 * eps(T)
            lhs ≈ rhs || @info "adjoint identity" N T lhs rhs
            # constant reproduction (to roundoff of the weighted sum) on every
            # fill-owned ghost cell
            c = set!(scalar_field(bf), _ -> T(0.75))
            halo_update!(c, bf)
            tol = 50 * eps(T)
            for (phase, fills) in (("interp", sched.interp), ("restrict", sched.restrict))
                worst = maximum(fills) do f
                    maximum(abs, c.blocks[f.dst_block][f.dst_ranges...] .- T(0.75))
                end
                @test worst < tol
                worst < tol || @info "constant not reproduced" N T phase worst
            end
        end
    end

    @testset "fused gather: one pass per fill, and what it matches bit for bit" begin
        # `_run_fills!` evaluates dst[I] = Σₖ wₖ·srcₖ[I] in ONE broadcast per fill,
        # over the CSR row `tfirst:tlast` of its phase. Three claims, each checked by
        # something that fails when the fusion is reverted or reassociated:
        #   (1) same term and accumulation order as the retired per-term loop ⇒
        #       bit-identical to it whenever the field eltype IS the weight type,
        #   (2) it really is one pass — a structural count, because (1) passes either
        #       way (its reference is a re-written copy of the loop it replaces) and
        #       so does the zero-allocation testset,
        #   (3) in mixed precision it is bit-identical to the device CSR kernel, which
        #       the retired loop was not.
        MFOL = MatrixFreeOperators
        # (1)'s reference: the retired per-term loop, over the same CSR buffers.
        function ref_fills!(store, lay, fills, terms)
            for f in fills
                dst = MFOL._leaf_view(store, lay, f.dst_block, f.dst_ranges)
                t1 = terms[f.tfirst]
                dst .= t1.weight .* MFOL._leaf_view(store, lay, t1.block, t1.ranges)
                for k in (f.tfirst + 1):f.tlast
                    tk = terms[k]
                    dst .+= tk.weight .* MFOL._leaf_view(store, lay, tk.block, tk.ranges)
                end
            end
            return nothing
        end
        function ref_fills_adjoint!(store, lay, fills, terms)
            for f in Iterators.reverse(fills)
                dst = MFOL._leaf_view(store, lay, f.dst_block, f.dst_ranges)
                for k in f.tfirst:f.tlast
                    tk = terms[k]
                    MFOL._leaf_view(store, lay, tk.block, tk.ranges) .+= tk.weight .* dst
                end
                fill!(dst, zero(eltype(dst)))
            end
            return nothing
        end
        function ref_exchange!(x, sched)
            store, lay = MFOL._storage(x), MFOL._layout(x)
            MFOL._run_copies!(store, lay, sched.copies)
            ref_fills!(store, lay, sched.interp, sched.interp_terms)
            ref_fills!(store, lay, sched.restrict, sched.restrict_terms)
            return x
        end
        function ref_exchange_adjoint!(x, sched)
            store, lay = MFOL._storage(x), MFOL._layout(x)
            ref_fills_adjoint!(store, lay, sched.restrict, sched.restrict_terms)
            ref_fills_adjoint!(store, lay, sched.interp, sched.interp_terms)
            MFOL._run_copies_adjoint!(store, lay, sched.copies)
            return x
        end
        # exact per-cell comparison over the full padded storage, with a diagnostic
        # on mismatch (how many cells, how far apart, on which leaf)
        function bit_equal(a, b, what)
            for i in 1:MFO.nleaves(a.grid)
                A, B = MFO._block_array(a, i), MFO._block_array(b, i)
                A == B && continue
                bad = count(!iszero, A .- B)
                @info "fused gather mismatch" what leaf = i ncells = bad worst =
                    maximum(maximum.(abs.(A .- B)))
                return false
            end
            return true
        end
        # dimension-fixed row widths reach the sweep as compile-time values
        @test @inferred(MFO._interp_width(Val(2))) === Val(7)
        @test @inferred(MFO._interp_width(Val(3))) === Val(19)
        @test @inferred(MFO._restrict_width(Val(2))) === Val(5)
        @test @inferred(MFO._restrict_width(Val(3))) === Val(9)

        rng = Random.MersenneTwister(31)
        for T in (Float64, Float32), N in (2, 3)
            ext = ntuple(_ -> (T(0), T(1)), N)
            bc = ntuple(d -> d == 1 ? (Dirichlet(), Neumann()) : (Periodic(), Periodic()), N)
            g = CartesianGrid(ext, ntuple(_ -> 16, N); bc=bc)
            bf = BlockForest(g; blocksize=ntuple(_ -> 4, N), maxlevel=3)
            refine!(bf, x -> x[1] < 0.5)
            refine!(bf, x -> x[1] < 0.25 && x[2] < 0.5)
            sched = MFO._exchange_schedule(bf)
            @test eltype(sched.interp_terms) === MFO.SlabTerm{N,T}
            @test !isempty(sched.interp) && !isempty(sched.restrict)
            x = scalar_field(bf)
            @test eltype(x) === T
            for i in 1:MFO.nleaves(bf)
                rand!(rng, x.blocks[i])
            end
            what = (T=T, N=N)
            # (1) forward, BlockField and PackedBlockField (both run the host gather)
            @test bit_equal(halo_update!(copy(x), bf), ref_exchange!(copy(x), sched), (what..., :fwd))
            @test bit_equal(
                halo_update!(pack(copy(x)), bf), ref_exchange!(pack(copy(x)), sched),
                (what..., :fwd_packed),
            )
            # adjoint: the transposed per-term scatter, unchanged by the fusion, but
            # its fill/term order is part of the same contract
            @test bit_equal(
                MFO.halo_update_adjoint!(copy(x), bf), ref_exchange_adjoint!(copy(x), sched),
                (what..., :adj),
            )
            @test bit_equal(
                MFO.halo_update_adjoint!(pack(copy(x)), bf),
                ref_exchange_adjoint!(pack(copy(x)), sched), (what..., :adj_packed),
            )
            # SVector elements ride the same fills (the weight is a scalar T)
            v = vector_field(bf)
            for i in 1:MFO.nleaves(bf)
                v.blocks[i] .= SVector{N,T}.(ntuple(_ -> rand(rng, T, size(v.blocks[i])), N)...)
            end
            @test bit_equal(halo_update!(copy(v), bf), ref_exchange!(copy(v), sched), (what..., :fwd_vec))
            @test bit_equal(
                MFO.halo_update_adjoint!(copy(v), bf), ref_exchange_adjoint!(copy(v), sched),
                (what..., :adj_vec),
            )

            # (2) one broadcast per fill, one write per dst cell. A per-term loop over
            # a K-wide row issues K broadcasts and writes every dst cell K times.
            for (fills, terms, K) in (
                (sched.interp, sched.interp_terms, MFO._ninterp_terms(Val(N))),
                (sched.restrict, sched.restrict_terms, MFO._nrestrict_terms(Val(N))),
            )
                nw, nb = Ref(0), Ref(0)
                store = [ProbeArray(copy(b), nw, nb) for b in x.blocks]
                MFO._run_fills!(store, MFO.BlocksLayout(), fills, terms, Val(K))
                @test nb[] == length(fills)
                @test nw[] == sum(f -> prod(length.(f.dst_ranges)), fills)
                # and the counted run really did the work: same cells as the reference
                ref = [copy(b) for b in x.blocks]
                ref_fills!(ref, MFO.BlocksLayout(), fills, terms)
                @test all(i -> store[i].parent == ref[i], eachindex(ref))
                # a row that is not K wide is an emitter bug, not an out-of-bounds read
                @test_throws AssertionError MFO._run_fills!(
                    store, MFO.BlocksLayout(), fills, terms, Val(K + 1)
                )
            end
        end

        # (3) mixed precision — a Float32 field on a Float64 forest, which the API
        # allows (`scalar_field(bf, Float32)`). The retired loop rounded its partial
        # sum into the Float32 slab K−1 times; the fused gather keeps the promoted
        # accumulator and rounds once, which is what the device CSR kernel
        # `_fill_kernel!` has always done. So the fusion does not merely differ from
        # the retired loop here — it removes a host/device divergence. Compared with
        # the host copy phase run on both sides, to isolate the fills from the
        # documented corner divergence of the batched copy kernel.
        cpu = KernelAbstractions.CPU()
        for N in (2, 3)
            ext = ntuple(_ -> (0.0, 1.0), N)
            bc = ntuple(d -> d == 1 ? (Dirichlet(), Neumann()) : (Periodic(), Periodic()), N)
            g = CartesianGrid(ext, ntuple(_ -> 16, N); bc=bc)
            bf = BlockForest(g; blocksize=ntuple(_ -> 4, N), maxlevel=3)
            refine!(bf, x -> x[1] < 0.5)
            refine!(bf, x -> x[1] < 0.25 && x[2] < 0.5)
            sched = MFO._exchange_schedule(bf)
            @test eltype(sched.interp_terms) === MFO.SlabTerm{N,Float64}
            ds = MFO._flatten_schedule(sched, bf, cpu)
            rng32 = Random.MersenneTwister(17)
            x32 = scalar_field(bf, Float32)
            @test eltype(x32) === Float32
            for i in 1:MFO.nleaves(bf)
                rand!(rng32, x32.blocks[i])
            end
            copies!(y) = MFO._run_copies!(MFO._storage(y), MFO._layout(y), sched.copies)
            fused = pack(copy(x32))
            copies!(fused)
            MFO._run_fills!(
                MFO._storage(fused), MFO._layout(fused), sched.interp, sched.interp_terms,
                MFO._interp_width(Val(N)),
            )
            MFO._run_fills!(
                MFO._storage(fused), MFO._layout(fused), sched.restrict,
                sched.restrict_terms, MFO._restrict_width(Val(N)),
            )
            retired = pack(copy(x32))
            copies!(retired)
            ref_fills!(MFO._storage(retired), MFO._layout(retired), sched.interp, sched.interp_terms)
            ref_fills!(
                MFO._storage(retired), MFO._layout(retired), sched.restrict, sched.restrict_terms
            )
            device = pack(copy(x32))
            copies!(device)
            MFO._run_fills_device!(
                device.data, ds.interp, ds.interp_terms, ds.interp_maxcells, cpu
            )
            MFO._run_fills_device!(
                device.data, ds.restrict, ds.restrict_terms, ds.restrict_maxcells, cpu
            )
            @test fused.data == device.data                      # fused ≡ device kernel
            ndiff = count(!iszero, retired.data .- device.data)  # retired loop was not
            @test ndiff > 0
            @test maximum(abs, retired.data .- device.data) <= 4 * eps(Float32)
            @info "mixed precision (Float32 field, Float64 weights)" N ndiff
        end

        # The pieces: term views alias storage cell-for-cell from the CSR row, the
        # weights come out in term order, and the per-cell body is the LEFT-associated
        # sum — checked with values whose reassociation is visible in Float32:
        # (1 + 1e8) + (−1e8) = 0 but 1 + (1e8 − 1e8) = 1.
        ws = (1.0f0, 1.0f0, 1.0f0)
        blocks = [fill(1.0f0, 2, 2), fill(1.0f8, 2, 2), fill(-1.0f8, 2, 2)]
        row = MFO.SlabTerm{2,Float32}[
            MFO.SlabTerm{2,Float32}(k, (1:1:2, 2:1:2), ws[k]) for k in 1:3
        ]
        srcs = MFO._term_views(blocks, MFO.BlocksLayout(), row, 1, Val(3))
        @test srcs === ntuple(k -> view(blocks[k], 1:1:2, 2:1:2), 3)
        @test MFO._term_weights(row, 1, Val(3)) === ws
        @test MFO._term_weights([row; row], 4, Val(3)) === ws      # rows start anywhere
        I = CartesianIndex(2, 1)
        @test (ws[1] * srcs[1][I] + ws[2] * srcs[2][I]) + ws[3] * srcs[3][I] === 0.0f0
        @test ws[1] * srcs[1][I] + (ws[2] * srcs[2][I] + ws[3] * srcs[3][I]) === 1.0f0
        @test MFO._gather_at(I, srcs, ws) === 0.0f0
        packed = cat(blocks...; dims=3)
        psrcs = MFO._term_views(packed, MFO.PackedLayout(), row, 1, Val(3))
        @test all(k -> psrcs[k] == srcs[k], 1:3)
        @test MFO._gather_at(I, psrcs, ws) === 0.0f0
        # and a mis-weighted probe: the seed really is term 1 (not a zero accumulator,
        # which would also make (0 + 1) + 1e8 − 1e8 = 0)
        @test MFO._gather_at(I, srcs, (2.0f0, 1.0f0, 1.0f0)) === 0.0f0
        @test MFO._gather_at(I, srcs, (1.0f0, 1.0f0, 0.0f0)) === 1.0f8

        # _AsScalar: an immutable Ref that broadcasts as a scalar (a mutable Ref of
        # the term views escapes into copyto! and heap-allocates per fill), whose
        # contents Adapt still converts for a device broadcast.
        sc = MFO._AsScalar((1, 2))
        @test sc isa Ref
        @test isbitstype(typeof(sc))
        @test sc[] === (1, 2)
        @test ((i, t) -> i + t[1] * t[2]).(1:3, sc) == [3, 4, 5]
        @test ((i, t) -> i + t[1] * t[2]).(1:3, MFO._AsScalar((1, 2))) == [3, 4, 5]
        a = Float32[1 2; 3 4]
        adapted = Adapt.adapt(DoubleAdaptor(), MFO._AsScalar((a, view(a, 1:1:1, 1:1:2))))
        @test adapted isa MFO._AsScalar
        @test adapted[] == (2 .* a, view(2 .* a, 1:1:1, 1:1:2))
    end

    @testset "forest apply_bc! matches the per-leaf physical fill" begin
        # Reference: per-leaf single-grid apply_bc! on hand-built leaf grids carrying
        # the physical/Interface mix leaf grids themselves no longer encode.
        rng = Random.MersenneTwister(3)
        bf = make_bf(((Dirichlet(), Dirichlet()), (Neumann(), Neumann())))
        refine!(bf, _ -> true)                          # boundary leaves above level 0
        x = scalar_field(bf)
        for i in 1:MFO.nleaves(bf)
            rand!(rng, x.blocks[i])
        end
        ref = copy(x)
        for (i, K) in enumerate(bf.forest.leaves)
            lg = MFO.leaf_grid(bf, i)
            mixbc = ntuple(2) do d
                nblocks = bf.forest.nroot[d] << K.level
                lo = K.coords[d] == 0 ? bf.bc[d][1] : MFO.Interface()
                hi = K.coords[d] == nblocks - 1 ? bf.bc[d][2] : MFO.Interface()
                (lo, hi)
            end
            mg = CartesianGrid{2,Float64,typeof(mixbc),typeof(bf.device),Nothing}(
                lg.extent, lg.spacing, lg.size, lg.halo, mixbc, bf.device,
                lg.local_range, nothing,
            )
            MFO.apply_bc!(ref.blocks[i], mg)
        end
        got = apply_bc!(copy(x), bf)
        @test all(got.blocks[i] == ref.blocks[i] for i in 1:MFO.nleaves(bf))
    end

    @testset "forest apply_bc!/fold_bc! duality (⟨Bx,w⟩ = ⟨x,Bᵀw⟩)" begin
        full_dot(a, b) = sum(i -> dot(vec(a.blocks[i]), vec(b.blocks[i])), 1:MFO.nleaves(a.grid))
        rng = Random.MersenneTwister(7)
        bf = make_bf(((Dirichlet(), Dirichlet()), (Neumann(), Neumann())))
        x = scalar_field(bf)
        w = scalar_field(bf)
        for i in 1:MFO.nleaves(bf)
            rand!(rng, x.blocks[i])
            rand!(rng, w.blocks[i])
        end
        Bx = apply_bc!(copy(x), bf)
        Btw = MFO.fold_bc!(copy(w), bf)
        @test full_dot(Bx, w) ≈ full_dot(x, Btw)
    end

    @testset "per-generation cache: reuse and rebuild" begin
        bf = make_bf(dirichlet)
        s1 = MFO._exchange_schedule(bf)
        @test MFO._exchange_schedule(bf) === s1        # cached — same object, no rebuild
        @test s1.generation == bf.forest.generation[]
        refine!(bf, _ -> true)                          # uniform level 1: 16 leaves, new gen
        s2 = MFO._exchange_schedule(bf)
        @test s2 !== s1
        @test s2.generation == bf.forest.generation[]
        @test length(s2.copies) == count_faces(bf)      # 4× the leaves ⇒ more descriptors
        f = set!(scalar_field(bf), x -> x[1] - 2x[2])   # fresh field exchanges on the new forest
        @test halo_update!(f, bf) === f
    end

    @testset "inference and zero-allocation" begin
        bf = make_bf(((Dirichlet(), Dirichlet()), (Neumann(), Neumann())))
        @inferred MFO._exchange_schedule(bf)
        uf = set!(scalar_field(bf), x -> sinpi(x[1]) * x[2])
        @inferred apply_bc!(uf, bf)
        @inferred MFO.fold_bc!(uf, bf)
        function alloc_halo(f, g)
            halo_update!(f, g)
            halo_update!(f, g)
            return @allocated halo_update!(f, g)
        end
        # (qualified through the module constant: `MFO` is a testset-local binding,
        # and a captured local makes the call a dynamic lookup that boxes its
        # arguments — an allocation of the scaffolding, not of the sweep)
        function alloc_halo_adjoint(f, g)
            MatrixFreeOperators.halo_update_adjoint!(f, g)
            MatrixFreeOperators.halo_update_adjoint!(f, g)
            return @allocated MatrixFreeOperators.halo_update_adjoint!(f, g)
        end
        @test alloc_halo(uf, bf) == 0
        # Refined forests run the coarse–fine fills — 7 terms per interpolation fill
        # in 2D, 19 in 3D. The fused gather's term views ride an immutable broadcast
        # scalar precisely so this stays at zero: a mutable `Ref` escapes the
        # un-inlined gather body and heap-allocates the whole tuple of views on every
        # fill (~2 KB per 3D fill), inside the `_exchange_storage!` rule seam that
        # must not allocate.
        for N in (2, 3)
            ext = ntuple(_ -> (0.0, 1.0), N)
            bc = ntuple(d -> d == 1 ? (Dirichlet(), Neumann()) : (Periodic(), Periodic()), N)
            g = CartesianGrid(ext, ntuple(_ -> 16, N); bc=bc)
            bfr = BlockForest(g; blocksize=ntuple(_ -> 4, N), maxlevel=3)
            refine!(bfr, x -> x[1] < 0.5)
            sched = MFO._exchange_schedule(bfr)
            @test !isempty(sched.interp) && !isempty(sched.restrict)
            for xr in (scalar_field(bfr), pack(scalar_field(bfr)))
                @inferred halo_update!(xr, bfr)
                @inferred MFO.halo_update_adjoint!(xr, bfr)
                @test alloc_halo(xr, bfr) == 0
                @test alloc_halo_adjoint(xr, bfr) == 0
            end
        end
    end
end
