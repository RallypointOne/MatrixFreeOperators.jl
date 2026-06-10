function lap_periodic_error(n::Int, ::Val{D}) where {D}
    g = CartesianGrid(
        ntuple(_ -> (0.0, 2π), Val(D)),
        ntuple(_ -> n, Val(D));
        bc=ntuple(_ -> (Periodic(), Periodic()), Val(D)),
    )
    u = set!(scalar_field(g), x -> prod(sin, x))
    y = laplacian(g) * u
    return maximum(abs, collect(interior(y)) .+ D .* collect(interior(u)))
end

@testset "Laplacian" begin
    @testset "analytic action and convergence order (periodic, $(D)-D)" for D in 1:3
        n = D == 3 ? 16 : 32
        e_coarse = lap_periodic_error(n, Val(D))
        e_fine = lap_periodic_error(2 * n, Val(D))
        order = log2(e_coarse / e_fine)
        @test e_fine < e_coarse
        @test order ≥ 1.9
    end

    @testset "dense symmetry under $(nameof(typeof(bc))) BCs" for bc in
                                                                  (Periodic(), Dirichlet(), Neumann())
        g = CartesianGrid(((0.0, 1.0),), (5,); bc=((bc, bc),))
        A = materialize(prepare(laplacian(g)))
        @test A ≈ A'

        g2 = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (4, 3); bc=((bc, bc), (bc, bc)))
        A2 = materialize(prepare(laplacian(g2)))
        @test A2 ≈ A2'
    end

    @testset "dense symmetry under mixed BCs" begin
        g = CartesianGrid(
            ((0.0, 1.0), (0.0, 1.0)),
            (4, 4);
            bc=((Dirichlet(), Neumann()), (Periodic(), Periodic())),
        )
        A = materialize(prepare(laplacian(g)))
        @test A ≈ A'
    end

    @testset "declared adjoint and adjoint identity" begin
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (5, 6); bc=((Dirichlet(), Dirichlet()), (Neumann(), Neumann())))
        L = laplacian(g)
        @test adjoint(L) === L
        @test isselfadjoint(L) && islinear(L) && isconstant(L)

        rng = Random.MersenneTwister(3)
        x = scalar_field(g)
        y = scalar_field(g)
        interior(x) .= rand(rng, local_size(g)...)
        interior(y) .= rand(rng, local_size(g)...)
        Lx = apply(L, copy(x))
        Lty = apply_adjoint!(scalar_field(g), L, copy(y), g)
        @test dot(collect(interior(Lx)), collect(interior(y))) ≈
            dot(collect(interior(x)), collect(interior(Lty)))
    end

    @testset "element-type genericity: SVector field ≡ componentwise scalar" begin
        g = CartesianGrid(
            ((0.0, 2π), (0.0, 2π)), (16, 16);
            bc=((Periodic(), Periodic()), (Periodic(), Periodic())),
        )
        v = set!(vector_field(g), x -> SVector(sin(x[1]) * sin(x[2]), cos(x[1])))
        u1 = set!(scalar_field(g), x -> sin(x[1]) * sin(x[2]))
        u2 = set!(scalar_field(g), x -> cos(x[1]))
        L = laplacian(g)
        Lv = L * v
        @test getindex.(collect(interior(Lv)), 1) ≈ collect(interior(L * u1))
        @test getindex.(collect(interior(Lv)), 2) ≈ collect(interior(L * u2))
    end

    @testset "stencil primitive returns center value and Laplacian" begin
        g = CartesianGrid(((0.0, 1.0),), (4,); bc=((Neumann(), Neumann()),))
        u = set!(scalar_field(g), x -> x[1]^2)
        apply_bc!(u)
        inv_h2 = inv.(spacing(g) .^ 2)
        uc, lap = laplacian_stencil(u.data, CartesianIndex(3), inv_h2)
        @test uc == u.data[3]
        @test lap ≈ (u.data[2] - 2 * u.data[3] + u.data[4]) * inv_h2[1]
    end
end
