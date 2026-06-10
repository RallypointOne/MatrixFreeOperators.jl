module MatrixFreeOperators

using Adapt: Adapt
using LinearAlgebra: LinearAlgebra, dot, mul!
using StaticArrays: SVector
import KernelAbstractions

export AbstractGrid, CartesianGrid
export dimension, spacing, local_size, halo_width, boundary_conditions
export interior, padded_size, cell_center, halo_update!
export AbstractBC, Periodic, Dirichlet, Neumann, apply_bc!, fold_bc!
export Field, Center, scalar_field, vector_field, set!, ncomponents, component
export flatten, flat_to_interior!, interior_to_flat!

include("Grids.jl")
include("boundaries.jl")
include("Fields.jl")

end # module
