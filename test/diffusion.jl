#--------------------------------------------------------------------------------# Compact flux-form diffusion

# The BC sweep every structural claim is checked over. Degenerate cell counts are
# deliberate: a periodic dimension of a single cell wraps onto the cell itself, which is
# the one case a naive "periodic faces need no diagonal correction" rule gets wrong.
const DIFF_BCS = (
    ("DD/NN", ((Dirichlet(), Dirichlet()), (Neumann(), Neumann()))),
    ("PP/DD", ((Periodic(), Periodic()), (Dirichlet(), Dirichlet()))),
    ("PP/PP", ((Periodic(), Periodic()), (Periodic(), Periodic()))),
    ("DN/PP", ((Dirichlet(), Neumann()), (Periodic(), Periodic()))),
    ("NN/NN", ((Neumann(), Neumann()), (Neumann(), Neumann()))),
)
const DIFF_SIZES = ((5, 4), (3, 3), (2, 2), (1, 4), (4, 1))
const DIFF_AVGS = (ArithmeticMean(), HarmonicMean())

# Positive κ everywhere: HarmonicMean is singular otherwise.
function diff_kappa(g, seed::Int=7)
    rng = Random.MersenneTwister(seed)
    κ = scalar_field(g)
    interior(κ) .= 1 .+ rand(rng, local_size(g)...)
    return κ
end

@testset "Diffusion (compact flux form)" begin
    @testset "analytic action and convergence order" begin
        # Manufactured on a periodic square, so no boundary term pollutes the order.
        κf(x) = 2 + sin(2π * x[1]) * cos(2π * x[2])
        uf(x) = sin(2π * x[1]) * sin(2π * x[2])
        # ∇·(κ∇u) = κΔu + ∇κ·∇u  — the continuous identity, valid as the *reference*
        # even though it is not the discrete one (see issue #53).
        function exactf(x)
            s1, c1 = sinpi(2 * x[1]), cospi(2 * x[1])
            s2, c2 = sinpi(2 * x[2]), cospi(2 * x[2])
            lap = -2 * (2π)^2 * s1 * s2
            return κf(x) * lap + (2π * c1 * c2) * (2π * c1 * s2) +
                   (-2π * s1 * s2) * (2π * s1 * c2)
        end
        function diff_error(n, avg)
            g = CartesianGrid(
                ((0.0, 1.0), (0.0, 1.0)), (n, n);
                bc=((Periodic(), Periodic()), (Periodic(), Periodic())),
            )
            y = diffusion(g, set!(scalar_field(g), κf); averaging=avg) *
                set!(scalar_field(g), uf)
            return maximum(
                abs, collect(interior(y)) .- collect(interior(set!(scalar_field(g), exactf)))
            )
        end
        @testset "$(nameof(typeof(avg)))" for avg in DIFF_AVGS
            e_coarse = diff_error(32, avg)
            e_fine = diff_error(64, avg)
            @test e_fine < e_coarse
            @test log2(e_coarse / e_fine) ≥ 1.9
        end
    end

    @testset "1-D and 3-D action" begin
        # κ(x) = 1 + x, u(x) = x²  ⇒  ∇·(κ∇u) = (κu')' = (2x + 2x²)' = 2 + 4x
        g1 = CartesianGrid(((0.0, 1.0),), (128,); bc=((Neumann(), Neumann()),))
        y1 = diffusion(g1, set!(scalar_field(g1), x -> 1 + x[1])) *
             set!(scalar_field(g1), x -> x[1]^2)
        # Interior only: the wall rows carry the homogeneous-Neumann flux, not (κu')'.
        yi = collect(interior(y1))[2:(end - 1)]
        xi = [cell_center(g1, I)[1] for I in interior(g1)][2:(end - 1)]
        @test maximum(abs, yi .- (2 .+ 4 .* xi)) < 0.05

        # 3-D: constant κ reduces to c·Δ on a smooth field.
        g3 = CartesianGrid(
            ntuple(_ -> (0.0, 2π), 3), ntuple(_ -> 16, 3);
            bc=ntuple(_ -> (Periodic(), Periodic()), 3),
        )
        κ3 = scalar_field(g3)
        interior(κ3) .= 2.0
        u3 = set!(scalar_field(g3), x -> prod(sin, x))
        @test collect(interior(diffusion(g3, κ3) * u3)) ≈
            2 .* collect(interior(laplacian(g3) * u3))
    end

    @testset "dense symmetry: $name n=$n $(nameof(typeof(avg)))" for (name, bc) in DIFF_BCS,
        n in DIFF_SIZES, avg in DIFF_AVGS

        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), n; bc=bc)
        A = materialize(prepare(diffusion(g, diff_kappa(g); averaging=avg), scalar_field(g)))
        # Exactly symmetric, not merely to tolerance: each shared face contributes the
        # same κ_f to A[I,J] and A[J,I], and a wall face contributes to A[I,I] alone.
        # (The composed div∘κ∘grad form is NOT symmetric — see test/algebra.jl.)
        @test A == A'
    end

    @testset "declared adjoint and dot-product identity" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (5, 4);
            bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
        )
        D = diffusion(g, diff_kappa(g))
        @test islinear(D) && isconstant(D) && isselfadjoint(D) && !isdiagonal(D)
        @test adjoint(D) === D
        @test MatrixFreeOperators.operator_grid(D) === g

        rng = Random.MersenneTwister(3)
        x = scalar_field(g)
        y = scalar_field(g)
        interior(x) .= rand(rng, local_size(g)...)
        interior(y) .= rand(rng, local_size(g)...)
        Dx = apply(D, copy(x))
        Dty = apply_adjoint!(scalar_field(g), D, copy(y), g)
        @test dot(collect(interior(Dx)), collect(interior(y))) ≈
            dot(collect(interior(x)), collect(interior(Dty)))

        # Complex κ: symmetric but not Hermitian, so the adjoint is the conjugated leaf.
        κc = Field(ComplexF64.(diff_kappa(g).data, 0.5 .* diff_kappa(g, 9).data), g)
        Dc = diffusion(g, κc)
        @test !isselfadjoint(Dc)
        @test adjoint(Dc) isa Diffusion
        M = materialize(prepare(Dc, scalar_field(g, ComplexF64)))
        Mt = materialize(prepare(adjoint(Dc), scalar_field(g, ComplexF64)))
        @test Mt ≈ M'
    end

    @testset "adjoint gather on Interface faces" begin
        # The public constructor refuses Interface faces because κ's cross-block ghosts
        # are an external input it has nothing to fill them from; the inner constructor
        # is the seam that supplies them, and it is what `_slab_op` builds through on a
        # partition slab (`test/partitioning.jl`). Here they are filled by hand, to
        # exercise the declared transpose the forest path (issue #58) will also rely on.
        # x's Interface ghosts are zero, so the identity holds over interiors.
        gi = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (8, 6);
            bc=((MatrixFreeOperators.Interface(), Dirichlet()),
                (Neumann(), MatrixFreeOperators.Interface())),
        )
        @test_throws ArgumentError diffusion(gi, diff_kappa(gi))

        rng = Random.MersenneTwister(21)
        κ = scalar_field(gi)
        κ.data .= 1 .+ rand(rng, padded_size(gi)...)      # ghosts included
        D = MatrixFreeOperators.Diffusion(gi, κ, ArithmeticMean())
        @test !MatrixFreeOperators.isselfadjoint(D) ||
            MatrixFreeOperators._has_interface(gi)         # gather path, not the shortcut

        x = scalar_field(gi)
        y = scalar_field(gi)
        interior(x) .= rand(rng, local_size(gi)...)
        interior(y) .= rand(rng, local_size(gi)...)
        Dx = apply(D, copy(x))
        Dty = apply_adjoint!(scalar_field(gi), D, copy(y), gi)
        @test dot(collect(interior(Dx)), collect(interior(y))) ≈
            dot(collect(interior(x)), collect(interior(Dty)))
    end

    @testset "operator_diagonal: $name n=$n $(nameof(typeof(avg)))" for (name, bc) in
                                                                       DIFF_BCS,
        n in DIFF_SIZES, avg in DIFF_AVGS

        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), n; bc=bc)
        D = diffusion(g, diff_kappa(g); averaging=avg)
        @test flatten(operator_diagonal(D)) ≈ diag(materialize(prepare(D, scalar_field(g))))
    end

    @testset "constant κ reduces to c·laplacian" begin
        @testset "$name c=$c" for (name, bc) in DIFF_BCS, c in (1.0, 3.7)
            g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (5, 4); bc=bc)
            κ = scalar_field(g)
            interior(κ) .= c
            A = materialize(prepare(diffusion(g, κ), scalar_field(g)))
            B = materialize(prepare(c * laplacian(g), scalar_field(g)))
            # Algebraically identical, not bitwise: the flux form multiplies by κ per
            # face before summing, and (u₊−u_c)−(u_c−u₋) rounds differently from
            # u₋−2u_c+u₊. Deliberately NOT compared against div∘scaling∘grad, whose
            # stencil is wider by design — that difference is the point of issue #48.
            @test A ≈ B rtol = 1e-13
            @test A[1, :] ≈ B[1, :] rtol = 1e-13        # boundary rows explicitly
            @test A[end, :] ≈ B[end, :] rtol = 1e-13
        end
    end

    @testset "compact stencil couples adjacent cells, the composition does not" begin
        # Both forms are 5-point; the difference is *which* cells they reach. The wide
        # composition samples the flux at I±e, so its row touches I±2e and skips the
        # immediate neighbours entirely — which is why its parity sublattices decouple.
        n = 6
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (n, n); bc=ntuple(_ -> (Periodic(), Periodic()), 2))
        κ = diff_kappa(g)
        A = materialize(prepare(diffusion(g, κ), scalar_field(g)))
        W = materialize(
            prepare(divergence(g) * scaling(κ) * MatrixFreeOperators.gradient(g), scalar_field(g))
        )
        flat(i, j) = i + n * (j - 1)                     # column-major interior ordering
        row, near, far = flat(3, 3), flat(4, 3), flat(5, 3)

        @test A[row, near] != 0 && A[row, flat(2, 3)] != 0
        @test A[row, far] == 0
        @test W[row, near] == 0 && W[row, flat(2, 3)] == 0
        @test W[row, far] != 0
        @test all(count(!iszero, A[i, :]) ≤ 5 for i in axes(A, 1))
    end

    @testset "compact κ-sensitivity includes the central cell (issue #48)" begin
        # A strictly interior row of the wide composition samples κ only at I±e,
        # so ∂(Lu)_I/∂κ_I ≡ 0. The compact form includes κ_I. This local
        # sensitivity does not prove that the full κ-Jacobian has no null space.
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (7, 7))
        u = set!(scalar_field(g), x -> sinpi(x[1]) * sinpi(2 * x[2]) + x[1] * x[2])
        I0 = CartesianIndex(4, 4)                        # a strictly interior cell

        function dLu_dkappa(build, κdata, I)
            ε = 1e-6
            κp = copy(κdata)
            κp[I] += ε
            κm = copy(κdata)
            κm[I] -= ε
            yp = collect(interior(apply(build(Field(κp, g)), copy(u))))
            ym = collect(interior(apply(build(Field(κm, g)), copy(u))))
            return (yp .- ym) ./ (2ε)
        end

        κdata = 1 .+ 0.3 .* rand(Random.MersenneTwister(5), padded_size(g)...)
        # index of I0 within the interior array
        i0 = CartesianIndex(Tuple(I0) .- halo_width(g))

        compact = dLu_dkappa(κ -> diffusion(g, κ), κdata, I0)
        wide = dLu_dkappa(
            κ -> divergence(g) * scaling(κ) * MatrixFreeOperators.gradient(g), κdata, I0
        )
        @test abs(wide[i0]) < 1e-8                       # structurally zero
        @test abs(compact[i0]) > 1e-3                    # genuinely coupled
    end

    @testset "arithmetic averaging retains a κ checkerboard null mode" begin
        @testset "$name n=$n" for (name, n, bc) in (
            ("periodic", (6, 4), ntuple(_ -> (Periodic(), Periodic()), 2)),
            ("homogeneous Neumann", (5, 4), ntuple(_ -> (Neumann(), Neumann()), 2)),
        )
            g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), n; bc=bc)
            κ = scalar_field(g)
            # Binary-exact, varying coefficients and perturbations let us check
            # equality exactly, without a tolerance hiding weak sensitivity.
            interior(κ) .= [2 + I[1] / 8 + I[2] / 16 for I in CartesianIndices(n)]
            checkerboard = [isodd(sum(Tuple(I))) ? -1 : 1 for I in CartesianIndices(n)]
            A = materialize(prepare(diffusion(g, κ), scalar_field(g)))

            @testset "checkerboard amplitude ε=$ε" for ε in (-1 / 4, 1 / 4)
                shifted = copy(κ)
                interior(shifted) .+= ε .* checkerboard
                @test all(>(0), interior(shifted))
                @test interior(shifted) != interior(κ)
                A_shifted = materialize(prepare(diffusion(g, shifted), scalar_field(g)))
                @test A_shifted == A  # identical response for EVERY excitation
            end

            # Controls: a local coefficient change is observable, and the same
            # checkerboard used as a solution is not in the operator's null space.
            local_change = copy(κ)
            interior(local_change)[2, 2] += 1 / 4
            A_local = materialize(prepare(diffusion(g, local_change), scalar_field(g)))
            @test A_local != A
            @test norm(A * vec(checkerboard)) > 1

            # Dirichlet walls use κ_I directly and break the face-average
            # cancellation, so the periodic/no-flux conclusion cannot carry over.
            gd = CartesianGrid(g.extent, n)
            κd = Field(copy(κ.data), gd)
            shifted_d = copy(κd)
            interior(shifted_d) .+= checkerboard ./ 4
            Ad = materialize(prepare(diffusion(gd, κd), scalar_field(gd)))
            Ad_shifted = materialize(prepare(diffusion(gd, shifted_d), scalar_field(gd)))
            @test Ad_shifted != Ad
        end
    end

    @testset "conservation" begin
        # Σ(Lu) telescopes to the net boundary flux, which is zero for periodic and for
        # homogeneous Neumann (no-flux) walls. Uniform grid ⇒ equal cell volumes.
        @testset "$name" for (name, bc) in (
            ("periodic", ntuple(_ -> (Periodic(), Periodic()), 2)),
            ("no-flux", ntuple(_ -> (Neumann(), Neumann()), 2)),
        )
            g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16); bc=bc)
            rng = Random.MersenneTwister(11)
            u = scalar_field(g)
            interior(u) .= rand(rng, 16, 16)
            @testset "$(nameof(typeof(avg)))" for avg in DIFF_AVGS
                y = diffusion(g, diff_kappa(g); averaging=avg) * u
                @test abs(sum(collect(interior(y)))) < 1e-9
            end
        end
    end

    @testset "boundary_rhs affine split with varying κ" begin
        # L_full(u) == L(u) + b, where L_full uses the *full* inhomogeneous ghost fill.
        # Varying κ cannot break it: the ghost is affine in u and the stencil carries the
        # same per-face κ weight in both halves.
        @testset "$name" for (name, bc) in (
            ("Dirichlet + Neumann", ((Dirichlet(1.3), Dirichlet(-0.7)), (Neumann(0.9), Neumann(2.1)))),
            ("Periodic + Dirichlet", ((Periodic(), Periodic()), (Dirichlet(2.0), Dirichlet(0.5)))),
        )
            g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (6, 5); bc=bc)
            D = diffusion(g, diff_kappa(g))
            rng = Random.MersenneTwister(13)
            u = scalar_field(g)
            interior(u) .= rand(rng, local_size(g)...)

            hom = copy(u)
            apply_bc!(hom)
            off = zero(u.data)
            MatrixFreeOperators.fill_bc_inhomogeneous!(off, g)
            full = Field(hom.data .+ off, g)

            yfull = scalar_field(g)
            MatrixFreeOperators._apply_raw!(yfull, D, full, g, true, false)
            lifted = collect(interior(apply(D, copy(u)))) .+
                     collect(interior(boundary_rhs(D, u)))
            @test collect(interior(yfull)) ≈ lifted
        end
    end

    @testset "steady-state allocations of prepared mul!" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (32, 32);
            bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())),
        )
        # A boxed neighbour index in the coefficient accessor allocates a fresh Core.Box
        # per cell while still producing the right numbers — only this catches it.
        @testset "$(nameof(typeof(avg)))" for avg in DIFF_AVGS
            P = prepare(diffusion(g, diff_kappa(g); averaging=avg), scalar_field(g))
            x = rand(32 * 32)
            y = similar(x)
            mul!(y, P, x)
            mul!(y, P, x)
            @test (@allocated mul!(y, P, x)) ≤ 512
            @test !iszero(sum(y))                        # DCE-proof
        end
    end

    @testset "element-type genericity and Adapt" begin
        g = CartesianGrid(((0.0f0, 1.0f0), (0.0f0, 1.0f0)), (8, 8))
        κ = scalar_field(g, Float32)
        interior(κ) .= 1.0f0
        D = diffusion(g, κ)
        y = D * set!(scalar_field(g, Float32), x -> sinpi(x[1]))
        @test eltype(y.data) === Float32
        @test collect(interior(y)) ≈ collect(interior(laplacian(g) * set!(scalar_field(g, Float32), x -> sinpi(x[1])))) rtol =
            1.0f-5
        @test Adapt.adapt(Array, D) isa Diffusion
    end

    @testset "constructor validation" begin
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (4, 4))
        κ = diff_kappa(g)
        @test diffusion(g, κ).avg isa ArithmeticMean
        @test diffusion(g, κ; averaging=HarmonicMean()).avg isa HarmonicMean

        @test_throws ArgumentError diffusion(g, vector_field(g))          # rank
        @test_throws ArgumentError diffusion(
            g, scalar_field(CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (8, 8)))
        )                                                                  # size mismatch

        neg = scalar_field(g)
        interior(neg) .= -1.0
        @test_throws ArgumentError diffusion(g, neg; averaging=HarmonicMean())
        @test diffusion(g, neg; averaging=HarmonicMean(), check=false) isa Diffusion
        @test diffusion(g, neg) isa Diffusion                              # arithmetic: no sign restriction
    end

    @testset "coefficient grid compatibility" begin
        g = CartesianGrid(((0.0, 1.0),), (6,))
        equivalent = CartesianGrid(((0.0, 1.0),), (6,))
        κ = set!(scalar_field(equivalent), _ -> 2)
        u = set!(scalar_field(g), x -> sinpi(x[1]))
        @testset "$(nameof(typeof(avg))) check=$check" for avg in DIFF_AVGS,
            check in (true, false)

            D = diffusion(g, κ; averaging=avg, check=check)
            @test collect(interior(D * copy(u))) ≈
                2 .* collect(interior(laplacian(g) * copy(u)))

            @testset "$name" for (name, other) in (
                ("equal padding, different interior and halo",
                    CartesianGrid(((0.0, 1.0),), (4,); halo=(2,))),
                ("shifted domain, same spacing",
                    CartesianGrid(((1.0, 2.0),), (6,))),
                ("different spacing",
                    CartesianGrid(((0.0, 2.0),), (6,))),
                ("different boundary conditions",
                    CartesianGrid(((0.0, 1.0),), (6,); bc=((Periodic(), Periodic()),))),
            )
                @test padded_size(other) == padded_size(g)
                # Positive interiors pass the harmonic check on `other`, while
                # its zero ghosts must never become interior coefficients on g.
                foreign = set!(scalar_field(other), _ -> 2)
                @test all(>(0), interior(foreign))
                @test_throws ArgumentError diffusion(
                    g, foreign; averaging=avg, check=check
                )
            end
        end
    end

    @testset "composition and trait propagation" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)), (6, 5);
            bc=((Periodic(), Periodic()), (Dirichlet(), Dirichlet())),
        )
        D = diffusion(g, diff_kappa(g))
        L = laplacian(g)
        u = set!(scalar_field(g), x -> sinpi(2 * x[1]) * x[2])
        ints(f) = collect(interior(f))

        @test ints((D + L) * copy(u)) ≈ ints(D * copy(u)) .+ ints(L * copy(u))
        @test ints((D - L) * copy(u)) ≈ ints(D * copy(u)) .- ints(L * copy(u))
        @test ints((2.5 * D) * copy(u)) ≈ 2.5 .* ints(D * copy(u))
        @test ints((-D) * copy(u)) ≈ -ints(D * copy(u))
        @test ints((D * identity_op()) * copy(u)) ≈ ints(D * copy(u))
        @test islinear(D + L) && isconstant(D + L)
        @test !isdiagonal(D + L)
        @test adjoint(D * L) isa MatrixFreeOperators.Composed
    end

    @testset "multigrid rediscretization" begin
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16))
        κ = set!(scalar_field(g), x -> 1 + x[1]^2 + x[2])
        D = diffusion(g, κ; averaging=HarmonicMean())
        gc = coarsen(g)
        Dc = MatrixFreeOperators._rediscretize(D, gc)
        @test Dc isa Diffusion
        @test Dc.grid === gc
        @test Dc.avg isa HarmonicMean
        # child mean, not Restriction: R would fold Dirichlet's -1 mirror into a wall κ
        @test collect(interior(Dc.κ)) ≈
            collect(interior(MatrixFreeOperators._average_to_coarse(κ, gc)))
        @test flatten(operator_diagonal(Dc)) ≈
            diag(materialize(prepare(Dc, scalar_field(gc))))
    end
end
