using MatrixFreeOperators
using Test
using LinearAlgebra
using Random
using StaticArrays
import KernelAbstractions

@testset "MatrixFreeOperators.jl" begin
    include("grids.jl")
    include("boundaries.jl")
    include("fields.jl")
    include("operators_abstract.jl")
end
