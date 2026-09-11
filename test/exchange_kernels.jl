@testset "Device exchange/BC kernels (flattened schedule)" begin
    MFO = MatrixFreeOperators
    cpu = KernelAbstractions.CPU()
    fun = x -> sinpi(x[1]) * cospi(2x[2]) + 0.3 * x[1]
    vfun = x -> SVector(sinpi(x[1]) + 0.2 * x[2], cospi(x[2]) - x[1])

    bcs = [
        ((Periodic(), Periodic()), (Periodic(), Periodic())),
        ((Dirichlet(), Dirichlet()), (Dirichlet(), Dirichlet())),
        ((Neumann(), Neumann()), (Neumann(), Neumann())),
        ((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
    ]

    # Non-corner comparison: batched dim-grouped copies leave different CORNER
    # ghost values than the interleaved host order (documented divergence —
    # cells no axis-aligned stencil reads); every other padded cell must be
    # bit-identical.
    function noncorner_parity(a, b)
        h, n = a.grid.halo, a.grid.blocksize
        N = length(n)
        for l in 1:MFO.nleaves(a.grid)
            A, B = MFO._block_array(a, l), MFO._block_array(b, l)
            for I in CartesianIndices(A)
                out = count(d -> !(h[d] < I[d] <= h[d] + n[d]), 1:N)
                out >= 2 && continue
                A[I] == B[I] || return false
            end
        end
        return true
    end

    # Run the device-path launches directly on the CPU backend (the public API
    # gates them to GPU) — same coverage MFO_TEST_GPU gets through the API.
    function run_device!(x, bf, ds)
        MFO._run_copies_device!(x.data, ds, bf, cpu)
        MFO._run_fills_device!(x.data, ds.interp, ds.interp_terms, ds.interp_maxcells, cpu)
        MFO._run_fills_device!(
            x.data, ds.restrict, ds.restrict_terms, ds.restrict_maxcells, cpu
        )
        MFO._bc_faces_launch!(
            MFO._bc_face_kernel!(cpu), x.data, bf.bc, ds.bcfaces, bf.halo, bf.blocksize,
            bf.blocksize .+ 2 .* bf.halo, Val(1),
        )
        return x
    end

    @testset "flattening round-trip" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (16, 16);
            bc=((Periodic(), Periodic()), (Dirichlet(), Dirichlet())),
        )
        bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
        refine!(bf, x -> x[1] < 0.5)
        sched = MFO._exchange_schedule(bf)
        ds = MFO._flatten_schedule(sched, bf, cpu)
        psize = bf.blocksize .+ 2 .* bf.halo

        # copies: per-dim buckets cover every descriptor, host order within a dim
        @test ds.copy_offsets[end] == length(sched.copies)
        recon = MFO._DevCopy{2}[]
        for d in 1:2, c in sched.copies
            MFO._copy_normal_dim(c, psize) == d || continue
            push!(
                recon,
                MFO._DevCopy{2}(
                    Int32(c.src), Int32(c.dst),
                    Int32.(first.(c.src_ranges)), Int32.(first.(c.dst_ranges)),
                ),
            )
        end
        @test collect(ds.copies) == recon

        # fills: per-fill records + CSR rows partition the concatenated terms, and
        # the device rows are the host rows verbatim (the host schedule is CSR too)
        for (fills, hterms, devfills, terms) in (
            (sched.interp, sched.interp_terms, ds.interp, ds.interp_terms),
            (sched.restrict, sched.restrict_terms, ds.restrict, ds.restrict_terms),
        )
            @test length(devfills) == length(fills)
            @test length(terms) == length(hterms)
            next = 1
            for (i, f) in enumerate(fills)
                df = devfills[i]
                fterms = MFO._fill_terms(hterms, f)
                @test Int(df.dst) == f.dst_block
                @test Int.(df.first) == first.(f.dst_ranges)
                @test Int.(df.step) == step.(f.dst_ranges)
                @test Int.(df.len) == length.(f.dst_ranges)
                @test Int(df.tfirst) == next == f.tfirst
                @test Int(df.tlast) == next + length(fterms) - 1 == f.tlast
                for (k, t) in enumerate(fterms)
                    dt = terms[next + k - 1]
                    @test Int(dt.block) == t.block
                    @test Int.(dt.first) == first.(t.ranges)
                    @test Int.(dt.step) == step.(t.ranges)
                    @test dt.weight == t.weight
                end
                next += length(fterms)
            end
            @test next - 1 == length(terms)
        end

        # bcfaces device lists match; periodic dims stay empty
        for d in 1:2, s in 1:2
            @test collect(ds.bcfaces[d][s]) == Int32.(sched.bcfaces[d][s])
        end
        @test isempty(ds.bcfaces[1][1]) && isempty(ds.bcfaces[1][2])
    end

    @testset "kernel bit-parity vs host loops (uniform + refined, all BC combos)" begin
        for bc in bcs, refined in (false, true)
            g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=bc)
            bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
            refined && refine!(bf, x -> x[1] < 0.5)
            sched = MFO._exchange_schedule(bf)
            ds = MFO._flatten_schedule(sched, bf, cpu)
            xr = pack(set!(scalar_field(bf), fun))
            MFO._run_exchange_host!(xr, sched)
            MFO._run_bc_host!(xr, bf, sched)
            xk = run_device!(pack(set!(scalar_field(bf), fun)), bf, ds)
            @test noncorner_parity(xk, xr)
            # operator action reads no corners ⇒ exact equality through the sweep
            L = laplacian(bf)
            yk = MFO._forest_sweep_leaves!(similar(xk), L, xk, bf, 1.0, false)
            yr = MFO._forest_sweep_leaves!(similar(xr), L, xr, bf, 1.0, false)
            @test all(
                i -> interior(MFO.block(yk, i)) == interior(MFO.block(yr, i)),
                1:MFO.nleaves(bf),
            )
            # SVector fields ride the same descriptors
            wr = pack(set!(vector_field(bf), vfun))
            MFO._run_exchange_host!(wr, sched)
            MFO._run_bc_host!(wr, bf, sched)
            wk = run_device!(pack(set!(vector_field(bf), vfun)), bf, ds)
            @test noncorner_parity(wk, wr)
        end
    end

    @testset "flat transfers: broadcast body vs per-leaf loops" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (16, 16);
            bc=((Dirichlet(), Dirichlet()), (Dirichlet(), Dirichlet())),
        )
        bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
        refine!(bf, x -> x[2] < 0.5)
        dims = (bf.blocksize..., MFO.nleaves(bf))
        for mk in (scalar_field, vector_field)
            f = pack(set!(mk(bf), mk === scalar_field ? fun : vfun))
            # gather (interior_to_flat!): broadcast body vs loops, α/β combos
            vref = flatten(f)
            for (α, β) in ((1.0, 0.0), (2.0, 0.5))
                v1 = copy(vref)
                MFO._interior_to_flat_leaves!(v1, f, α, β)
                v2 = copy(vref)
                vi = MFO._as_eltype(eltype(f), v2, dims)
                if iszero(β)
                    vi .= α .* MFO._interior_view(f)
                else
                    vi .= α .* MFO._interior_view(f) .+ β .* vi
                end
                @test v1 == v2
            end
            # scatter (flat_to_interior!): broadcast body vs loops
            f1 = MFO._zero_all!(similar(f))
            MFO._flat_to_interior_leaves!(f1, vref)
            f2 = MFO._zero_all!(similar(f))
            MFO._interior_view(f2) .= MFO._as_eltype(eltype(f), vref, dims)
            @test f1.data == f2.data
        end
    end

    @testset "device-schedule cache: identity, generation invalidation" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (16, 16);
            bc=((Dirichlet(), Dirichlet()), (Dirichlet(), Dirichlet())),
        )
        bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
        ds1 = MFO._device_schedule(bf, MFO._exchange_schedule(bf), cpu)
        @test MFO._device_schedule(bf, MFO._exchange_schedule(bf), cpu) === ds1
        refine!(bf, x -> x[1] < 0.5)
        sched2 = MFO._exchange_schedule(bf)
        ds2 = MFO._device_schedule(bf, sched2, cpu)
        @test ds2 !== ds1
        @test ds2.generation == sched2.generation == bf.forest.generation[]
        @test MFO._device_schedule(bf, sched2, cpu) === ds2
    end

    @testset "device launches: allocations are forest-size independent" begin
        bc = ((Dirichlet(), Dirichlet()), (Neumann(), Neumann()))
        function alloc_device(x, bf, ds)
            run_device!(x, bf, ds)
            run_device!(x, bf, ds)
            a = @allocated run_device!(x, bf, ds)
            return a, sum(sum(interior(MFO.block(x, i))) for i in 1:MFO.nleaves(bf))
        end
        allocs = map((16, 32)) do n
            gn = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (n, n); bc=bc)
            bfn = BlockForest(gn; blocksize=(4, 4), maxlevel=2)
            refine!(bfn, x -> x[1] < 0.5)
            sched = MFO._exchange_schedule(bfn)
            ds = MFO._flatten_schedule(sched, bfn, cpu)
            x = pack(set!(scalar_field(bfn), fun))
            a, s = alloc_device(x, bfn, ds)
            @test isfinite(s)
            (a, MFO.nleaves(bfn))
        end
        # The KA CPU runtime allocates per WORKGROUP (~30 B/leaf here: the copy
        # ndrange spans multiple workgroups at larger forests); GPU launches
        # don't heap-allocate. Direct CPU launches are test scaffolding behind
        # the backend gate, so assert a loose linear bound, not flatness
        # (cf. the masked-gather note in forest_packed.jl).
        @test allocs[1][1] <= 200 * allocs[1][2]
        @test allocs[2][1] <= 200 * allocs[2][2]
    end
end
