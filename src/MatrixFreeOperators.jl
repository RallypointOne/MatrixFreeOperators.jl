module MatrixFreeOperators

using Adapt: Adapt
using LinearAlgebra: LinearAlgebra, dot, mul!, norm
using StaticArrays: SVector
import KernelAbstractions

export AbstractGrid, CartesianGrid
export dimension, spacing, local_size, halo_width, boundary_conditions
export interior, padded_size, cell_center, coarsen, halo_update!
export AbstractBC, Periodic, Dirichlet, Neumann, apply_bc!, fold_bc!
export BlockForest, BlockField, refine!, coarsen!, balance!, leaves
export AbstractField, Field, Center, scalar_field, vector_field, set!, ncomponents, component
export flatten, flat_to_interior!, interior_to_flat!
export AbstractOperator, apply, apply!, apply_adjoint!, AdjointOp
export islinear, isconstant, isselfadjoint, isdiagonal
export Laplacian, laplacian, laplacian_stencil, laplacian_7pt_noflux
export Derivative, derivative, derivative_stencil
export Gradient, gradient, Divergence, divergence
export ScalingOp, scaling, IdentityOp, identity_op
export Advection, advection, SelfAdvection
export Scaled, Added, Composed
export LinearizedOp, linearize, linearize!
export PreparedOperator, PreparedForest, prepare, boundary_rhs

include("Grids.jl")
include("boundaries.jl")
include("topology.jl")
include("schedule.jl")
include("BlockForest.jl")
include("Fields.jl")
include("blockfield.jl")
include("transfer.jl")
include("operators/abstract.jl")
include("operators/algebra.jl")
include("operators/laplacian.jl")
include("operators/derivative.jl")
include("operators/gradient.jl")
include("operators/divergence.jl")
include("operators/scaling.jl")
include("operators/advection.jl")
include("operators/forest.jl")
include("operators/linearize.jl")
include("linalg.jl")

end # module
