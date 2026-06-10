using MatrixFreeOperators
using Test
using LinearAlgebra
using Random
using StaticArrays
import KernelAbstractions
import Krylov

include("test_utils.jl")

@testset "MatrixFreeOperators.jl" begin
    include("grids.jl")
    include("boundaries.jl")
    include("fields.jl")
    include("operators_abstract.jl")
    include("laplacian.jl")
    include("derivative.jl")
    include("gradient_divergence.jl")
    include("scaling_identity_advection.jl")
    include("algebra.jl")
    include("prepare_linalg.jl")
end
