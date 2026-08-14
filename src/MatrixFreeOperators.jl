module MatrixFreeOperators

using Adapt: Adapt
using LinearAlgebra: LinearAlgebra, diag, dot, issuccess, ldiv!, lu, mul!, norm
using StaticArrays: SVector
import KernelAbstractions
using KernelAbstractions: @Const, @index, @kernel

export AbstractGrid, CartesianGrid
export dimension, spacing, local_size, halo_width, boundary_conditions
export interior, padded_size, cell_center, coarsen, halo_update!, partition_grid
export AbstractBC, Periodic, Dirichlet, Neumann, apply_bc!, fold_bc!
export BlockForest, BlockField, PackedBlockField, pack, unpack
export refine!, coarsen!, balance!, leaves, regrid!
export AbstractField, Field, Center, scalar_field, vector_field, set!, ncomponents, component
export flatten, flat_to_interior!, interior_to_flat!
export AbstractOperator, apply, apply!, apply_adjoint!, AdjointOp
export islinear, isconstant, isselfadjoint, isdiagonal, operator_diagonal
export Laplacian, laplacian, laplacian_stencil, laplacian_7pt_noflux
export Derivative, derivative, derivative_stencil
export Gradient, gradient, Divergence, divergence
export ScalingOp, scaling, IdentityOp, identity_op
export Diffusion, diffusion, diffusion_stencil, fill_coefficient_ghosts!
export ArithmeticMean, HarmonicMean
export Advection, advection, SelfAdvection
export Restriction, restriction, Prolongation, prolongation
export Scaled, Added, Composed
export LinearizedOp, linearize, linearize!
export AbstractJVPBackend, FiniteDifferenceJVP, EnzymeJVP
export PreparedOperator, PreparedForest, prepare, prepare_distributed, boundary_rhs
export assemble_rhs, local_grids
export MultigridPreconditioner, MultigridSolver, solve, Jacobi, Chebyshev

include("Grids.jl")
include("boundaries.jl")
include("topology.jl")
include("schedule.jl")
include("BlockForest.jl")
include("Fields.jl")
include("partitioning.jl")
include("blockfield.jl")
include("packedfield.jl")
include("transfer.jl")
include("transfer_kernels.jl")
include("operators/abstract.jl")
include("operators/algebra.jl")
include("operators/laplacian.jl")
include("operators/derivative.jl")
include("operators/gradient.jl")
include("operators/divergence.jl")
include("operators/scaling.jl")
include("operators/diffusion.jl")
include("operators/advection.jl")
include("operators/prolongation.jl")
include("operators/restriction.jl")
include("operators/diagonal.jl")
include("operators/forest.jl")
include("operators/forest_packed.jl")
include("operators/linearize.jl")
include("linalg.jl")
include("distributed.jl")
include("amr.jl")
include("multigrid.jl")

end # module
