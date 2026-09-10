#--------------------------------------------------------------------------------# Diffusion on a BlockForest

@testset "Diffusion on a BlockForest" begin
    MFO = MatrixFreeOperators

    FOREST_DIFF_AVGS = (ArithmeticMean(), HarmonicMean())

    # Positive everywhere (HarmonicMean is singular otherwise) and genuinely varying
    # across every refinement interface, so a κ-weighting inconsistency cannot hide.
    κ_varying(x) = 1.5 + 0.5 * sin(x[1]) * cos(x[2]) + 0.1 * x[1]

    # Volume-weighted conservation defect Σ V·(Lu) and its scale Σ V·|Lu| — cells at
    # level ℓ have volume prod(spacing0)/2^(N·ℓ), so the sum must be level-weighted.
    function conservation_defect(y, bf)
        total = 0.0
        scale = 0.0
        for i in 1:MFO.nleaves(bf)
            V = prod(MFO._leaf_spacing(bf, bf.forest.leaves[i].level))
            yi = collect(interior(MFO.block(y, i)))
            total += V * sum(yi)
            scale += V * sum(abs, yi)
        end
        return total, scale
    end

    function block_rand!(f, bf, rng)
        for i in 1:MFO.nleaves(bf)
            interior(MFO.block(f, i)) .= rand(rng, bf.blocksize...)
        end
        return f
    end

    @testset "conservation on a refined forest: Σ V·(Lu) ≈ 0" begin
        # Interior face fluxes telescope pairwise; periodic wrap and homogeneous
        # no-flux walls contribute nothing; and at each coarse–fine face the
        # flux-matching restriction (src/transfer.jl) defines the coarse ghost so the
        # coarse stencil's face flux equals the area-weighted mean of the fine fluxes
        # — for UNWEIGHTED differences. The Laplacian case validates the harness and
        # must pass; constant κ scales every flux identically and must pass; variable
        # κ weights each side of a coarse–fine face by an independently-formed face κ
        # (coarse: avg with the 2⁻ᴺ volume average, fine: avg with the injected
        # coarse value), which is exactly issue #58's open question 2. A random u is
        # the strongest probe: telescoping is exact, no smoothness is assumed.
        function refined_cases()
            # periodic band: a CF interface at x ≈ 1.6 and a wrapped one at x = 0
            bcp = ((Periodic(), Periodic()), (Periodic(), Periodic()))
            basep = CartesianGrid(((0.0, 2π), (0.0, 2π)), (16, 16); bc=bcp)
            bfp = BlockForest(basep; blocksize=(4, 4), maxlevel=2)
            refine!(bfp, x -> x[1] < 1.6)
            # no-flux box: three levels, 2:1 graded, refined region touching walls
            bcn = ((Neumann(), Neumann()), (Neumann(), Neumann()))
            basen = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=bcn)
            bfn = BlockForest(basen; blocksize=(4, 4), maxlevel=3)
            refine!(bfn, x -> x[1] < 0.5 && x[2] < 0.5)
            refine!(bfn, x -> x[1] < 0.2 && x[2] < 0.2)
            return (("periodic", bfp), ("no-flux", bfn))
        end

        for (name, bf) in refined_cases()
            @test !bf.forest.uniform[]
            rng = Random.MersenneTwister(29)
            u = block_rand!(scalar_field(bf), bf, rng)

            @testset "$name: Laplacian (harness baseline)" begin
                defect, scale = conservation_defect(laplacian(bf) * u, bf)
                @info "conservation Σ V·(Lu)" case = "$name Laplacian" defect scale
                @test abs(defect) ≤ 1e3 * eps() * scale
            end

            @testset "$name: Diffusion, κ ≡ const, $(nameof(typeof(avg)))" for avg in
                                                                               FOREST_DIFF_AVGS
                κc = scalar_field(bf)
                for i in 1:MFO.nleaves(bf)
                    interior(MFO.block(κc, i)) .= 2.3
                end
                y = diffusion(bf, κc; averaging=avg) * u
                defect, scale = conservation_defect(y, bf)
                @info "conservation Σ V·(Lu)" case = "$name const-κ $(nameof(typeof(avg)))" defect scale
                @test abs(defect) ≤ 1e3 * eps() * scale
            end

            @testset "$name: Diffusion, varying κ, $(nameof(typeof(avg)))" for avg in
                                                                               FOREST_DIFF_AVGS
                κv = set!(scalar_field(bf), κ_varying)
                y = diffusion(bf, κv; averaging=avg) * u
                defect, scale = conservation_defect(y, bf)
                @info "conservation Σ V·(Lu)" case = "$name varying-κ $(nameof(typeof(avg)))" defect scale
                # The verdict on issue #58's question 2 (recorded in the commit that
                # introduced this file): the operator-independent exchange does NOT
                # deliver one authoritative κ-weighted flux — the defect was ~1e-2
                # against a ~1e3 scale. The Diffusion coarse-ghost rewrite
                # (`_cf_flux_rewrite!`) is what makes this hold to roundoff.
                @test abs(defect) ≤ 1e3 * eps() * scale
            end
        end
    end

    # Reconstruct a uniform-level forest scalar field into the equivalent dense
    # array (the forest_parity.jl idiom, local so this file runs standalone).
    function reconstruct(f, dims)
        bff = f.grid
        b = bff.blocksize
        full = zeros(eltype(f), dims)
        for i in 1:MFO.nleaves(bff)
            key = bff.forest.leaves[i]
            idx = ntuple(d -> (key.coords[d] * b[d]) .+ (1:b[d]), length(dims))
            full[idx...] .= collect(interior(MFO.block(f, i)))
        end
        return full
    end

    # The small refined forest the dense/materialize tests use: 7 leaves, one
    # coarse–fine ring, n = 7·16 = 112 flat unknowns.
    function small_refined(bc)
        base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8); bc=bc)
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=2)
        refine!(bf, x -> x[1] < 0.5 && x[2] < 0.5)
        return bf
    end

    κ_fun(x) = 1.2 + 0.8 * x[1]^2 + 0.5 * x[2] + 0.3 * x[1] * x[2]
    u_fun(x) = sinpi(x[1]) * cospi(2 * x[2]) + 0.3 * x[1]
    PARITY_BCS = (
        ((Periodic(), Periodic()), (Periodic(), Periodic())),
        ((Dirichlet(), Dirichlet()), (Dirichlet(), Dirichlet())),
        ((Neumann(), Neumann()), (Neumann(), Neumann())),
        ((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
    )

    @testset "level-0 parity vs equal-resolution single grid" begin
        for bc in PARITY_BCS, avg in FOREST_DIFF_AVGS
            g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8); bc=bc)
            bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)   # level 0 ⇒ same (8, 8)
            Dg = diffusion(g, set!(scalar_field(g), κ_fun); averaging=avg)
            Df = diffusion(bf, set!(scalar_field(bf), κ_fun); averaging=avg)
            ref = collect(interior(Dg * set!(scalar_field(g), u_fun)))
            rec = reconstruct(Df * set!(scalar_field(bf), u_fun), (8, 8))
            @test rec == ref                                    # bit-identical
        end
    end

    @testset "uniformly-refined parity vs double-resolution grid" begin
        for bc in PARITY_BCS
            g16 = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=bc)
            base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8); bc=bc)
            bf = BlockForest(base; blocksize=(4, 4), maxlevel=3)
            refine!(bf, _ -> true)                              # uniform level 1 = 16×16
            @test all(k -> k.level == 1, bf.forest.leaves)
            Dg = diffusion(g16, set!(scalar_field(g16), κ_fun))
            Df = diffusion(bf, set!(scalar_field(bf), κ_fun))
            ref = collect(interior(Dg * set!(scalar_field(g16), u_fun)))
            rec = reconstruct(Df * set!(scalar_field(bf), u_fun), (16, 16))
            @test rec == ref
        end
    end

    @testset "boundary_rhs parity with inhomogeneous data (uniform forest)" begin
        bci = ((Dirichlet(2.0), Dirichlet(-1.0)), (Neumann(0.5), Dirichlet(3.0)))
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8); bc=bci)
        bf = BlockForest(g; blocksize=(4, 4), maxlevel=2)
        Dg = diffusion(g, set!(scalar_field(g), κ_fun))
        Df = diffusion(bf, set!(scalar_field(bf), κ_fun))
        bfor = boundary_rhs(Df, scalar_field(bf))
        @test bfor isa BlockField
        @test reconstruct(bfor, (8, 8)) == collect(interior(boundary_rhs(Dg, scalar_field(g))))
    end

    @testset "refined constant κ reduces to c·laplacian" begin
        bf = small_refined(((Dirichlet(), Dirichlet()), (Neumann(), Neumann())))
        c = 3.7
        κc = scalar_field(bf)
        for i in 1:MFO.nleaves(bf)
            interior(MFO.block(κc, i)) .= c
        end
        rng = Random.MersenneTwister(41)
        u = block_rand!(scalar_field(bf), bf, rng)
        yD = diffusion(bf, κc) * u
        yL = laplacian(bf) * u
        # Algebraically identical, never bitwise: the flux form multiplies by κ per
        # face before summing, and the rewrite divides by the face κ it then
        # multiplies back — each a rounding the Laplacian path does not perform.
        for i in 1:MFO.nleaves(bf)
            @test collect(interior(MFO.block(yD, i))) ≈
                c .* collect(interior(MFO.block(yL, i))) rtol = 1e-12
        end
    end

    @testset "adjoint on a refined forest: identity, declared transpose, α/β" begin
        rng = Random.MersenneTwister(43)
        for bc in (
                ((Periodic(), Periodic()), (Periodic(), Periodic())),
                ((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
            ),
            avg in FOREST_DIFF_AVGS

            bf = small_refined(bc)
            D = diffusion(bf, set!(scalar_field(bf), κ_fun); averaging=avg)
            @test islinear(D) && isconstant(D) && !isdiagonal(D)
            @test !isselfadjoint(D)                  # CF coupling breaks the symmetry
            @test adjoint(D) isa AdjointOp           # must not fold to D

            x = block_rand!(scalar_field(bf), bf, rng)
            y = block_rand!(scalar_field(bf), bf, rng)
            Dx = apply(D, copy(x))
            Dty = apply_adjoint!(scalar_field(bf), D, copy(y), bf)
            @test all(i -> all(isfinite, MFO._block_array(Dty, i)), 1:MFO.nleaves(bf))
            ip1 = sum(
                i -> dot(
                    collect(interior(MFO.block(Dx, i))), collect(interior(MFO.block(y, i)))
                ),
                1:MFO.nleaves(bf),
            )
            ip2 = sum(
                i -> dot(
                    collect(interior(MFO.block(x, i))), collect(interior(MFO.block(Dty, i)))
                ),
                1:MFO.nleaves(bf),
            )
            @test ip1 ≈ ip2

            # dense declared transpose, through the prepared path (the seam rides
            # _forest_sweep!/_forest_adjoint_sweep!, so the dense matrices include it)
            A = materialize(prepare(D))
            At = materialize(prepare(adjoint(D)))
            @test At ≈ A'
            @test !(A ≈ A')

            # accumulating adjoint (β ≠ 0, the scratch-and-blend branch) vs dense
            α, β = 1.3, -0.7
            x̄0 = block_rand!(scalar_field(bf), bf, rng)
            x̄ = copy(x̄0)
            apply_adjoint!(x̄, D, copy(y), bf, α, β)
            @test flatten(x̄) ≈ α .* (A' * flatten(y)) .+ β .* flatten(x̄0)

            # prepared round-trips, forward and adjoint
            P = prepare(D)
            v = flatten(x)
            out = similar(v)
            mul!(out, P, v)
            @test out == flatten(D * copy(x))
            Pt = prepare(adjoint(D))
            mul!(out, Pt, v)
            @test out ≈ A' * v
        end
    end

    @testset "uniform forest keeps the self-adjoint shortcut" begin
        bf = BlockForest(
            CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8)); blocksize=(4, 4), maxlevel=2
        )
        D = diffusion(bf, set!(scalar_field(bf), κ_fun))
        @test isselfadjoint(D)
        @test adjoint(D) === D
        A = materialize(prepare(D))
        @test A ≈ A'
    end

    @testset "corner ghosts stay exactly zero through the adjoint sweep" begin
        # The halo_update_adjoint! precondition (src/transfer.jl): per-leaf stencil
        # adjoints — and the seam's transpose — must leave corner ghosts exactly
        # zero, or a cotangent would leak into a diagonal neighbor's interior.
        # Ones-poisoned x̄ also proves the gather overwrites corners rather than
        # skipping them.
        rng = Random.MersenneTwister(47)
        bf = small_refined(((Dirichlet(), Dirichlet()), (Neumann(), Neumann())))
        D = diffusion(bf, set!(scalar_field(bf), κ_fun))
        ȳ = block_rand!(scalar_field(bf), bf, rng)
        x̄ = scalar_field(bf)
        for i in 1:MFO.nleaves(bf)
            MFO._block_array(x̄, i) .= 1.0
        end
        MFO._forest_adjoint_sweep!(x̄, D, copy(ȳ), bf, true)
        n = bf.blocksize
        is_corner(I) = count(d -> I[d] == 1 || I[d] == n[d] + 2, 1:2) >= 2
        for i in 1:MFO.nleaves(bf)
            a = MFO._block_array(x̄, i)
            for I in CartesianIndices(a)
                is_corner(I) && @test a[I] === 0.0
            end
        end
    end

    @testset "affine lift on a refined forest" begin
        # D_full(u) = D(u) + b with FULL inhomogeneous ghosts. The rewrite C is
        # linear and C(z) = z on the lift input (zero interiors and Interface
        # ghosts), so the identity must survive the seam — this is the test that
        # locks boundary_rhs needing no change.
        bci = ((Dirichlet(1.3), Dirichlet(-0.7)), (Neumann(0.9), Neumann(2.1)))
        base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8); bc=bci)
        bf = BlockForest(base; blocksize=(4, 4), maxlevel=2)
        refine!(bf, x -> x[1] < 0.5 && x[2] < 0.5)
        D = diffusion(bf, set!(scalar_field(bf), κ_fun))
        rng = Random.MersenneTwister(53)
        u = block_rand!(scalar_field(bf), bf, rng)

        full = copy(u)
        halo_update!(full, bf)
        apply_bc!(full, bf)                          # homogeneous ghosts
        z = MFO._zero_all!(similar(u))
        MFO.fill_bc_inhomogeneous!(z, bf)            # inhomogeneous offsets alone
        for i in 1:MFO.nleaves(bf)
            MFO._block_array(full, i) .+= MFO._block_array(z, i)
        end
        MFO._cf_flux_rewrite!(                       # the full application's seam pass
            MFO._storage(full), MFO._layout(full), MFO._storage(D.κ), MFO._layout(D.κ),
            D.avg, MFO._exchange_schedule(bf).cfflux, bf.blocksize,
        )
        yfull = scalar_field(bf)
        for i in 1:MFO.nleaves(bf)
            lg = MFO.leaf_grid(bf, i)
            MFO._apply_raw!(
                MFO.block(yfull, i, lg), MFO._leaf_op(D, i, lg), MFO.block(full, i, lg),
                lg, true, false,
            )
        end
        Du = apply(D, copy(u))
        b = boundary_rhs(D, u)
        for i in 1:MFO.nleaves(bf)
            @test collect(interior(MFO.block(yfull, i))) ≈
                collect(interior(MFO.block(Du, i))) .+ collect(interior(MFO.block(b, i)))
        end

        # the lazy adjoint's lift is identically zero (the adjoint action is built
        # homogeneous) — meaningful only here, where adjoint(D) does not fold to D
        @test adjoint(D) isa AdjointOp
        badj = boundary_rhs(adjoint(D), scalar_field(bf))
        @test all(i -> all(iszero, MFO._block_array(badj, i)), 1:MFO.nleaves(bf))
    end

    @testset "coefficient exchange: exact policies for a linear κ" begin
        # Linear κ makes every fill policy exact: a same-level copy reproduces
        # κ(ghost center); the 2⁻ᴺ fine average equals κ at the fine cells'
        # centroid, which IS the coarse ghost center; injection reproduces κ at
        # the covering coarse cell's center; the even wall mirror reproduces κ at
        # the reflected point.
        κlin(x) = 0.7 + 2.0 * x[1] - 1.3 * x[2]
        bf = small_refined(((Dirichlet(), Dirichlet()), (Neumann(), Neumann())))
        κx = MFO.fill_coefficient_ghosts!(copy(set!(scalar_field(bf), κlin)), bf)
        n = bf.blocksize
        lo = ntuple(k -> bf.extent[k][1], 2)
        counts = Dict("mirror" => 0, "copy" => 0, "inject" => 0, "average" => 0)
        for (i, K) in enumerate(bf.forest.leaves)
            ext = MFO._leaf_extent(bf, K)
            sp = MFO._leaf_spacing(bf, K.level)
            for d in 1:2, side in (-1, 1)
                nbr = MFO.face_neighbor(bf.forest, K, d, side)
                gp = side == -1 ? 1 : n[d] + 2
                t = d == 1 ? 2 : 1
                for jt in 2:(n[t] + 1)
                    idx = d == 1 ? (gp, jt) : (jt, gp)
                    center = ntuple(k -> ext[k][1] + (idx[k] - 1.5) * sp[k], 2)
                    val = MFO._block_array(κx, i)[idx...]
                    if nbr === nothing
                        face = side == -1 ? ext[d][1] : ext[d][2]
                        mirrored = ntuple(k -> k == d ? 2 * face - center[k] : center[k], 2)
                        @test val ≈ κlin(mirrored) atol = 1e-12
                        counts["mirror"] += 1
                    elseif MFO.is_leaf(bf.forest, nbr)
                        @test val ≈ κlin(center) atol = 1e-12
                        counts["copy"] += 1
                    elseif MFO.leaf_covering(bf.forest, nbr) !== nothing
                        # this leaf is finer: injected covering-coarse-cell value
                        H = ntuple(k -> 2 * sp[k], 2)
                        cc = ntuple(
                            k -> lo[k] + (floor((center[k] - lo[k]) / H[k]) + 0.5) * H[k], 2
                        )
                        @test val ≈ κlin(cc) atol = 1e-12
                        counts["inject"] += 1
                    else
                        # this leaf is coarser: 2⁻ᴺ average = centroid value
                        @test val ≈ κlin(center) atol = 1e-12
                        counts["average"] += 1
                    end
                end
            end
        end
        @test all(>(0), values(counts))
    end

    @testset "the operator reads its exchanged κ ghosts (negative control)" begin
        rng = Random.MersenneTwister(59)
        bf = small_refined(((Periodic(), Periodic()), (Periodic(), Periodic())))
        D = diffusion(bf, set!(scalar_field(bf), κ_fun))
        u = block_rand!(scalar_field(bf), bf, rng)
        clean = D * copy(u)
        κz = copy(D.κ)
        for i in 1:MFO.nleaves(bf)
            keep = collect(interior(MFO.block(κz, i)))
            MFO._block_array(κz, i) .= 0.0
            interior(MFO.block(κz, i)) .= keep
        end
        Dz = MFO.Diffusion(bf, κz, D.avg)            # inner ctor: ghosts as given
        dirty = Dz * copy(u)
        @test any(
            i -> collect(interior(MFO.block(dirty, i))) !=
                 collect(interior(MFO.block(clean, i))),
            1:MFO.nleaves(bf),
        )
    end

    @testset "mutating κ after construction is inert" begin
        bf = small_refined(((Dirichlet(), Dirichlet()), (Dirichlet(), Dirichlet())))
        κ = set!(scalar_field(bf), κ_fun)
        D = diffusion(bf, κ)
        rng = Random.MersenneTwister(61)
        u = block_rand!(scalar_field(bf), bf, rng)
        y1 = D * copy(u)
        for i in 1:MFO.nleaves(bf)
            interior(MFO.block(κ, i)) .*= 2
        end
        y2 = D * copy(u)
        for i in 1:MFO.nleaves(bf)
            @test collect(interior(MFO.block(y1, i))) == collect(interior(MFO.block(y2, i)))
        end
    end

    @testset "constructor validation and staleness" begin
        bf = small_refined(((Dirichlet(), Dirichlet()), (Neumann(), Neumann())))
        @test_throws ArgumentError diffusion(bf, vector_field(bf))         # rank
        other = BlockForest(
            CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8)); blocksize=(4, 4), maxlevel=2
        )
        @test_throws ArgumentError diffusion(bf, scalar_field(other))      # wrong forest

        neg = scalar_field(bf)
        for i in 1:MFO.nleaves(bf)
            interior(MFO.block(neg, i)) .= -1.0
        end
        @test_throws ArgumentError diffusion(bf, neg; averaging=HarmonicMean())
        @test diffusion(bf, neg; averaging=HarmonicMean(), check=false) isa Diffusion

        # sign-changing κ that averages to exactly zero on a coarse–fine face:
        # +1 on refined blocks, −1 on coarse ones ⇒ avg(−1, mean(+1)) = 0
        κpm = scalar_field(bf)
        for (i, K) in enumerate(bf.forest.leaves)
            interior(MFO.block(κpm, i)) .= K.level == 0 ? -1.0 : 1.0
        end
        @test_throws ArgumentError diffusion(bf, κpm)                      # divisor guard
        @test diffusion(bf, κpm; check=false) isa Diffusion

        # operator_diagonal stays unavailable on forests, matching Laplacian
        D = diffusion(bf, set!(scalar_field(bf), κ_fun))
        @test_throws ArgumentError operator_diagonal(D)

        # a regrid invalidates the operator through its κ generation stamp
        κold = set!(scalar_field(bf), κ_fun)
        Dold = diffusion(bf, κold)
        nl0 = MFO.nleaves(bf)
        refine!(bf, x -> x[1] > 0.7 && x[2] > 0.7)   # matches the top-right leaf CENTER
        @test MFO.nleaves(bf) > nl0                  # the topology actually moved
        @test_throws ArgumentError diffusion(bf, κold)                     # stale κ
        u = scalar_field(bf)                                               # current field
        @test_throws ArgumentError Dold * u                                # stale operator
    end

    @testset "steady-state allocations of prepared mul! (seam included)" begin
        # Pkg.test runs under --check-bounds=yes, where every per-leaf forest sweep
        # allocates a small constant per leaf (72 B/leaf for Diffusion, 48 for the
        # Laplacian on Julia 1.12; exactly 0 B under default flags — the
        # forest_packed.jl precedent), so an absolute bound measures the flags, not
        # the seam. The guard is relative instead: on the refined forest the rewrite
        # runs on every coarse–fine face, and it must cost no more per leaf than the
        # uniform forest, where `cfflux` is empty and the seam is a no-op. A boxed
        # capture or a dynamic ntuple in the rewrite would add ~180 B per coarse–fine
        # face while producing the right numbers — only this catches it.
        #
        # For adj = true the denominator is NOT the adjoint: a uniform forest has no
        # coarse–fine coupling, so `isselfadjoint` holds and `_forest_capply_adjoint!`
        # short-circuits to the forward sweep. The budget therefore reads "the
        # adjoint must cost no more per leaf than the forward sweep", which is the
        # right thing to enforce and is what caught the ghost-plane gather in #91
        # (136 B/leaf against the forward path's 72). Do not mistake it for a
        # per-leaf-cost cancellation and loosen it.
        function alloc_mul(P, out, v)
            mul!(out, P, v)
            mul!(out, P, v)
            a = @allocated mul!(out, P, v)
            return a, sum(out)                          # DCE-proof: consume the output
        end
        function forest_alloc(bf, adj)
            L = diffusion(bf, set!(scalar_field(bf), κ_fun))
            P = prepare(adj ? adjoint(L) : L)
            v = rand(Random.MersenneTwister(3), size(P, 2))
            a, s = alloc_mul(P, similar(v), v)
            @test !iszero(s)
            return a
        end
        bcs = ((Dirichlet(), Dirichlet()), (Neumann(), Neumann()))
        base = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=bcs)
        uniform = BlockForest(base; blocksize=(4, 4), maxlevel=3)   # 16 leaves, no CF faces
        refined = BlockForest(base; blocksize=(4, 4), maxlevel=3)
        refine!(refined, x -> x[1] < 0.5 && x[2] < 0.5)
        refine!(refined, x -> x[1] < 0.2 && x[2] < 0.2)             # 40 leaves, three levels
        @test isempty(MFO._exchange_schedule(uniform).cfflux)
        @test length(MFO._exchange_schedule(refined).cfflux) == 16
        for adj in (false, true)
            per_leaf = forest_alloc(uniform, adj) / MFO.nleaves(uniform)
            @test forest_alloc(refined, adj) ≤ per_leaf * MFO.nleaves(refined) + 256
        end
    end

    @testset "packed coefficient path (prepared prototype packs κ)" begin
        bf = small_refined(((Dirichlet(), Dirichlet()), (Neumann(), Neumann())))
        D = diffusion(bf, set!(scalar_field(bf), κ_fun))
        uf = set!(scalar_field(bf), u_fun)
        P = prepare(D, pack(uf))
        @test P.xpad isa PackedBlockField            # packed prototype ⇒ packed scratch
        @test P.op.κ isa PackedBlockField            # _prepare_tree packed the coefficient
        v = flatten(uf)
        out = similar(v)
        mul!(out, P, v)
        @test out ≈ flatten(D * copy(uf))
        Pt = prepare(adjoint(D), pack(uf))
        # the AdjointOp _prepare_tree recursion packs the wrapped leaf's κ too, so
        # the adjoint hot path dispatches to the kernel sweep instead of the fallback
        @test Pt.op.op.κ isa PackedBlockField
        mul!(out, Pt, v)
        @test out ≈ materialize(prepare(D))' * v
    end

    @testset "packed sweeps: public parity and direct kernel launch (CPU backend)" begin
        # The public API routes non-GPU backends to the per-leaf fallback (with the
        # coarse–fine rewrite still applied), so the kernel bodies are exercised by
        # direct launch on the KA CPU backend — the same coverage MFO_TEST_GPU gets
        # through the API on CUDA (the forest_packed.jl idiom).
        cpu = KernelAbstractions.CPU()
        for refined in (false, true), avg in FOREST_DIFF_AVGS
            base = CartesianGrid(
                ((0.0, 1.0), (0.0, 1.0)), (8, 8);
                bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
            )
            bf = BlockForest(base; blocksize=(4, 4), maxlevel=2)
            refined && refine!(bf, x -> x[1] < 0.5)
            uf = set!(scalar_field(bf), u_fun)
            D = diffusion(bf, set!(scalar_field(bf), κ_fun); averaging=avg)
            Dp = MFO.Diffusion(bf, pack(D.κ), D.avg)  # inner ctor: ghosts ride pack

            # public packed path: packed κ on packed x hits the new override; a
            # BlockField κ on packed x is the documented mismatch fallback — both
            # must reproduce the reference-layout action exactly
            yref = D * copy(uf)
            for L in (Dp, D)
                yp = L * pack(copy(uf))
                @test all(
                    i -> collect(interior(MFO.block(yp, i))) ==
                         collect(interior(MFO.block(yref, i))),
                    1:MFO.nleaves(bf),
                )
            end

            # direct forward launch: bit-parity with the per-leaf reference on the
            # SAME rewritten input, overwrite and accumulate blends
            x = pack(copy(uf))
            MFO.halo_update!(x, bf)
            MFO.apply_bc!(x, bf)
            MFO._cf_flux_rewrite!(
                MFO._storage(x), MFO._layout(x), MFO._storage(Dp.κ), MFO._layout(Dp.κ),
                Dp.avg, MFO._exchange_schedule(bf).cfflux, bf.blocksize,
            )
            nd = (bf.blocksize..., MFO.nleaves(bf))
            y = MFO._zero_all!(MFO.allocate_output(Dp, x))
            MFO._diff_forest_kernel!(cpu)(
                y.data, x.data, Dp.κ.data, x.levels, bf.spacing0, bf.halo, Dp.avg,
                2.0, false; ndrange=nd,
            )
            ref = MFO._forest_sweep_leaves!(
                MFO._zero_all!(MFO.allocate_output(Dp, x)), Dp, x, bf, 2.0, false
            )
            @test y.data == ref.data
            y2 = copy(y)
            ref2 = copy(ref)
            MFO._diff_forest_kernel!(cpu)(
                y2.data, x.data, Dp.κ.data, x.levels, bf.spacing0, bf.halo, Dp.avg,
                2.0, 3.0; ndrange=nd,
            )
            MFO._forest_sweep_leaves!(ref2, Dp, x, bf, 2.0, 3.0)
            @test y2.data == ref2.data

            # direct adjoint launch: full padded equality pre-fold (the ghost
            # cotangents are the point of the padded ndrange)
            ndp = (bf.blocksize .+ 2 .* bf.halo..., MFO.nleaves(bf))
            ȳ = pack(set!(scalar_field(bf), x -> cospi(x[1]) + x[2]^2))
            ȳk = copy(ȳ)
            ȳr = copy(ȳ)
            x̄k = MFO.allocate_input(Dp, ȳk)
            x̄r = MFO.allocate_input(Dp, ȳr)
            MFO.zero_ghosts!(ȳk)    # the seam zeroes before launching
            MFO._diff_adjoint_forest_kernel!(cpu)(
                x̄k.data, ȳk.data, Dp.κ.data, ȳk.levels, bf.spacing0, Dp.avg, 2.0;
                ndrange=ndp,
            )
            MFO._forest_adjoint_sweep_leaves!(x̄r, Dp, ȳr, bf, 2.0)
            @test x̄k.data == x̄r.data

            # full adjoint through the public packed path (kernel-fallback + seam
            # transpose + folds) against the reference layout
            x̄p = apply_adjoint!(similar(ȳ), Dp, copy(ȳ), bf)
            x̄b = apply_adjoint!(scalar_field(bf), D, unpack(copy(ȳ)), bf)
            @test all(
                i -> collect(interior(MFO.block(x̄p, i))) ==
                     collect(interior(MFO.block(x̄b, i))),
                1:MFO.nleaves(bf),
            )
        end
    end
end
