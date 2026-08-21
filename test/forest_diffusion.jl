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
end
