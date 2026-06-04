[![CI](https://github.com/RallypointOne/MatrixFreeOperators.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/RallypointOne/MatrixFreeOperators.jl/actions/workflows/CI.yml)
[![Docs Build](https://github.com/RallypointOne/MatrixFreeOperators.jl/actions/workflows/Docs.yml/badge.svg)](https://github.com/RallypointOne/MatrixFreeOperators.jl/actions/workflows/Docs.yml)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue)](https://RallypointOne.github.io/MatrixFreeOperators.jl/stable/)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue)](https://RallypointOne.github.io/MatrixFreeOperators.jl/dev/)

# MatrixFreeOperators.jl

Matrix-free linear and nonlinear operators for solving PDEs on structured grids,
built for device-agnostic execution (CPU/GPU) and efficient forward- and
reverse-mode automatic differentiation — including gradients with respect to
operator parameters for inverse problems and PDE-constrained optimization. The
package exposes a composable operator algebra (`L1 * L2`, `L1 + L2`,
`adjoint(L)`) and targets the Krylov.jl + OrdinaryDiffEq.jl solver stack.

> Status: early development — the public API is not yet stable.
