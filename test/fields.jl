@testset "Fields" begin
    @testset "construction and allocation" begin
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (4, 6))
        @test_throws DimensionMismatch Field(zeros(4, 6), g)

        u = scalar_field(g)
        @test size(u.data) == (6, 8)
        @test eltype(u) === Float64
        @test all(iszero, u.data)

        u32 = scalar_field(g, Float32)
        @test eltype(u32) === Float32

        v = vector_field(g)
        @test eltype(v) === SVector{2,Float64}
        @test ncomponents(v) == 2
        @test ncomponents(u) == 1
    end

    @testset "set! at cell centers" begin
        g = CartesianGrid(((0.0, 1.0),), (4,))
        u = set!(x -> 2 * x[1], scalar_field(g))
        @test vec(collect(interior(u))) ≈ [0.25, 0.75, 1.25, 1.75]
        @test u.data[1] == 0.0 && u.data[end] == 0.0

        g32 = CartesianGrid(((0.0f0, 1.0f0),), (4,))
        u32 = set!(x -> x[1]^2, scalar_field(g32))
        @test eltype(u32) === Float32
    end

    @testset "op! sees the same cell as set!" begin
        # Non-square 2D grid: a padded/interior index mix-up shifts or scrambles cells.
        g = CartesianGrid(((0.0, 1.0), (0.0, 2.0)), (3, 5))
        f(x) = x[1] + 10 * x[2]
        u = set!(f, scalar_field(g))
        op!((x, a) -> a - f(x), u)
        @test maximum(abs, collect(interior(u))) < 1e-12
    end

    @testset "component extraction" begin
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (3, 3))
        v = set!(x -> SVector(x[1], -x[2]), vector_field(g))
        vx = component(v, 1)
        vy = component(v, 2)
        @test eltype(vx) === Float64
        @test collect(interior(vx)) ≈ getindex.(collect(interior(v)), 1)
        @test collect(interior(vy)) ≈ getindex.(collect(interior(v)), 2)
        @test_throws ArgumentError component(v, 3)

        u = set!(x -> x[1], scalar_field(g))
        @test collect(interior(component(u, 1))) == collect(interior(u))
        @test_throws ArgumentError component(u, 2)
    end

    @testset "flat round trip (scalar)" begin
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (3, 4))
        u = set!(x -> x[1] + 10 * x[2], scalar_field(g))
        flat = flatten(u)
        @test length(flat) == 12
        w = scalar_field(g)
        flat_to_interior!(w, flat)
        @test collect(interior(w)) == collect(interior(u))
        @test all(iszero, w.data[1, :])
    end

    @testset "flat round trip (SVector)" begin
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (3, 4))
        v = set!(x -> SVector(x[1], -x[2]), vector_field(g))
        flat = flatten(v)
        @test length(flat) == 24
        @test eltype(flat) === Float64
        w = vector_field(g)
        flat_to_interior!(w, flat)
        @test collect(interior(w)) == collect(interior(v))
    end

    @testset "interior_to_flat! axpby fusion" begin
        g = CartesianGrid(((0.0, 1.0),), (5,))
        u = set!(x -> x[1], scalar_field(g))
        v = ones(5)
        interior_to_flat!(v, u, 2.0, 3.0)
        @test v ≈ 2 .* vec(collect(interior(u))) .+ 3
        interior_to_flat!(v, u)
        @test v ≈ vec(collect(interior(u)))

        vf = set!(x -> SVector(x[1]), vector_field(CartesianGrid(((0.0, 1.0),), (3,))))
        fv = fill(0.5, 3)
        interior_to_flat!(fv, vf, 1.0, -1.0)
        @test fv ≈ getindex.(vec(collect(interior(vf))), 1) .- 0.5
    end

    @testset "field-level BC and copy/similar" begin
        g = CartesianGrid(((0.0, 1.0),), (4,); bc=((Neumann(), Neumann()),))
        u = set!(x -> x[1], scalar_field(g))
        apply_bc!(u)
        @test u.data[1] == u.data[2]
        @test u.data[end] == u.data[end - 1]

        u2 = copy(u)
        @test u2.data == u.data && u2.data !== u.data
        u3 = similar(u)
        @test size(u3.data) == size(u.data) && eltype(u3) === Float64
    end
end
