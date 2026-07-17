using MatrixFreeOperators
using Test
using LinearAlgebra
using Random
using StaticArrays
import Adapt
import Enzyme
import KernelAbstractions
import Krylov
import Mooncake

include("test_utils.jl")

@testset "MatrixFreeOperators.jl" begin
    include("grids.jl")
    include("boundaries.jl")
    include("fields.jl")
    include("topology.jl")
    include("blockforest.jl")
    include("blockfield.jl")
    include("exchange_schedule.jl")
    include("operators_abstract.jl")
    include("laplacian.jl")
    include("derivative.jl")
    include("forest_parity.jl")
    include("forest_prepare.jl")
    include("forest_amr.jl")
    include("gradient_divergence.jl")
    include("scaling_identity_advection.jl")
    include("algebra.jl")
    include("prepare_linalg.jl")
    include("linearize.jl")
    include("device.jl")
    include("reactant.jl")
    include("autodiff.jl")
    include("ode_rhs.jl")
end
