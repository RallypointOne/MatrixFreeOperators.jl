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

    @testset "GhostFill is isbits with a dimension-fixed term count" begin
        # The term count of every coarse–fine fill is a function of N alone
        # (interp 1 + 2·3^(N-1), restrict 1 + 2^N): the emitters loop over a
        # tensor product of per-tangential-dim stencils, never over topology. That
        # is what lets `terms` be an NTuple, GhostFill isbits, and the schedule
        # type derivable from (N, T) — checked here on 1D/2D/3D forests, on
        # interior, boundary-adjacent and cascaded (multi-level) faces, with the
        # descriptor vectors' eltypes and the forest's cached Ref all concrete.
        for N in (1, 2, 3)
            KI, KR = 1 + 2 * 3^(N - 1), 1 + 2^N
            @test MFO._ninterp_terms(Val(N)) == KI
            @test MFO._nrestrict_terms(Val(N)) == KR
            @test isbitstype(MFO.GhostFill{N,Float64,KI})
            @test isbitstype(MFO.GhostFill{N,Float64,KR})
            @test MFO._schedule_type(Val(N), Float64) ===
                MFO.ExchangeSchedule{N,Float64,KI,KR}
            @test isconcretetype(MFO._schedule_type(Val(N), Float32))
            ext = ntuple(_ -> (0.0, 1.0), N)
            for bc in (
                ntuple(_ -> (Periodic(), Periodic()), N),
                ntuple(d -> d == 1 ? (Dirichlet(), Neumann()) : (Periodic(), Periodic()), N),
            )
                g = CartesianGrid(ext, ntuple(_ -> 16, N); bc=bc)
                bf = BlockForest(g; blocksize=ntuple(_ -> 4, N), maxlevel=3)
                @test isconcretetype(eltype(bf.schedule))
                refine!(bf, x -> x[1] < 0.5)                  # one interface plane
                refine!(bf, x -> x[1] < 0.25)                 # balance cascade ⇒ 3 levels
                sched = @inferred MFO._exchange_schedule(bf)
                @test typeof(sched) === MFO._schedule_type(Val(N), Float64)
                @test bf.schedule[] === sched
                @test !isempty(sched.interp) && !isempty(sched.restrict)
                @test eltype(sched.interp) === MFO.GhostFill{N,Float64,KI}
                @test eltype(sched.restrict) === MFO.GhostFill{N,Float64,KR}
                @test isbitstype(eltype(sched.interp)) && isbitstype(eltype(sched.restrict))
                @test all(f -> length(f.terms) == KI, sched.interp)
                @test all(f -> length(f.terms) == KR, sched.restrict)
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
                @test all(f -> sum(t.weight for t in f.terms) ≈ 1, sched.interp)
                @test all(f -> f.terms[1].block == f.dst_block, sched.interp)   # own cell
                wf = 2.0 / 2^(N - 1)
                @test all(
                    f -> f.terms[1].weight == 1 && f.terms[1].block == f.dst_block,
                    sched.restrict,
                )
                @test all(f -> all(t -> abs(t.weight) == wf, f.terms[2:end]), sched.restrict)
                @test all(f -> sum(t.weight for t in f.terms) ≈ 1, sched.restrict)
            end
        end
        # A term list of the wrong width is an emitter bug and must throw, never pad.
        t = MFO.SlabTerm{2,Float64}(1, (1:1:1, 2:1:3), 0.5)
        @test_throws AssertionError MFO._term_tuple(Val(3), [t, t])
        @test MFO._term_tuple(Val(2), [t, t]) === (t, t)
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
        # frozen into each NTuple.
        rng = Random.MersenneTwister(23)
        for N in (2, 3)
            ext = ntuple(_ -> (0.0, 1.0), N)
            bc = ntuple(d -> d == 1 ? (Dirichlet(), Neumann()) : (Periodic(), Periodic()), N)
            g = CartesianGrid(ext, ntuple(_ -> 16, N); bc=bc)
            bf = BlockForest(g; blocksize=ntuple(_ -> 4, N), maxlevel=3)
            refine!(bf, x -> x[1] < 0.5)
            refine!(bf, x -> x[1] < 0.25 && x[2] < 0.5)
            sched = MFO._exchange_schedule(bf)
            @test !isempty(sched.interp) && !isempty(sched.restrict)
            h, n = bf.halo, bf.blocksize
            is_corner(I) = count(d -> I[d] <= h[d] || I[d] > h[d] + n[d], 1:N) >= 2
            full_dot(a, b) =
                sum(i -> dot(vec(a.blocks[i]), vec(b.blocks[i])), 1:MFO.nleaves(a.grid))
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
            # constant reproduction (to roundoff of the weighted sum) on every
            # fill-owned ghost cell
            c = set!(scalar_field(bf), _ -> 0.75)
            halo_update!(c, bf)
            for (phase, fills) in (("interp", sched.interp), ("restrict", sched.restrict))
                worst = maximum(fills) do f
                    maximum(abs, c.blocks[f.dst_block][f.dst_ranges...] .- 0.75)
                end
                @test worst < 1e-14
                worst < 1e-14 || @info "constant not reproduced" N phase worst
            end
        end
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
        @test alloc_halo(uf, bf) == 0
    end
end
