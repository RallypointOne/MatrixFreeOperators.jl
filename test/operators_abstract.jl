struct DummyNonlinearOp <: MatrixFreeOperators.AbstractOperator end

struct DummyDoubleOp{G} <: MatrixFreeOperators.AbstractOperator
    grid::G
end
MatrixFreeOperators.islinear(::DummyDoubleOp) = true
MatrixFreeOperators.isselfadjoint(::DummyDoubleOp) = true
MatrixFreeOperators.operator_grid(L::DummyDoubleOp) = L.grid
function MatrixFreeOperators.apply!(
    y::Field, ::DummyDoubleOp, x::Field, g::AbstractGrid, α, β
)
    if iszero(β)
        interior(y) .= α .* 2 .* interior(x)
    else
        interior(y) .= α .* 2 .* interior(x) .+ β .* interior(y)
    end
    return y
end
function MatrixFreeOperators.apply_adjoint!(
    x̄::Field, L::DummyDoubleOp, ȳ::Field, g::AbstractGrid, α, β
)
    return MatrixFreeOperators.apply!(x̄, L, ȳ, g, α, β)
end

@testset "Operator abstraction" begin
    g = CartesianGrid(((0.0, 1.0),), (4,))
    u = set!(scalar_field(g), x -> x[1])

    @testset "trait defaults make the weak claim" begin
        L = DummyNonlinearOp()
        @test !MatrixFreeOperators.islinear(L)
        @test !MatrixFreeOperators.isconstant(L)
        @test !MatrixFreeOperators.isselfadjoint(L)
        @test !MatrixFreeOperators.isdiagonal(L)
    end

    @testset "adjoint gated on islinear" begin
        err = try
            adjoint(DummyNonlinearOp())
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("linearize", err.msg)
    end

    @testset "apply matches apply! and applies α, β" begin
        L = DummyDoubleOp(g)
        y = apply(L, u)
        @test collect(interior(y)) ≈ 2 .* collect(interior(u))
        @test all(iszero, y.data[[1, end]])
        @test collect(interior(L(u))) == collect(interior(y))
        @test collect(interior(L * u)) == collect(interior(y))

        z = set!(scalar_field(g), x -> 1.0)
        MatrixFreeOperators.apply!(z, L, u, g, 3.0, 2.0)
        @test collect(interior(z)) ≈ 6 .* collect(interior(u)) .+ 2
    end

    @testset "AdjointOp wrapper" begin
        L = DummyDoubleOp(g)
        A = MatrixFreeOperators.AdjointOp(L)
        @test MatrixFreeOperators.islinear(A)
        @test MatrixFreeOperators.adjoint_operator(A) === L
        y = apply(A, u)
        @test collect(interior(y)) ≈ 2 .* collect(interior(u))
        @test sprint(show, A) == "adjoint(DummyDoubleOp)"
    end

    @testset "size and eltype" begin
        L = DummyDoubleOp(g)
        @test size(L) == (4, 4)
        @test eltype(L) === Float64
        @test_throws ArgumentError size(DummyNonlinearOp())
    end
end
