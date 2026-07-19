# AirspeedVelocity entry point: `benchpkg` runs this file on both revisions of a PR,
# so every benchmark must use API that exists on the PR's base branch too.
using BenchmarkTools
using LinearAlgebra
using StaticArrays
using MatrixFreeOperators

const SUITE = BenchmarkGroup()

#--------------------------------------------------------------------------------# single grid

per2 = ((Periodic(), Periodic()), (Periodic(), Periodic()))
g2 = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (256, 256); bc=per2)
g3 = CartesianGrid(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), (64, 64, 64))

for (dim, g) in (("2D 256²", g2), ("3D 64³", g3))
    L = laplacian(g)
    P = prepare(L, scalar_field(g))
    x = flatten(set!(scalar_field(g), p -> sin(4p[1]) + cos(3p[end])))
    y = similar(x)
    SUITE["grid"][dim]["laplacian prepare"] = @benchmarkable prepare($L, $(scalar_field(g)))
    SUITE["grid"][dim]["laplacian mul!"] = @benchmarkable mul!($y, $P, $x)
end

xs2 = flatten(set!(scalar_field(g2), p -> sin(4p[1]) + cos(3p[2])))
ys2 = similar(xs2)
xv2 = flatten(set!(vector_field(g2), p -> SVector(sin(p[2]), cos(p[1]))))
yv2 = similar(xv2)

PG = prepare(gradient(g2), scalar_field(g2))
SUITE["grid"]["2D 256²"]["gradient mul!"] = @benchmarkable mul!($yv2, $PG, $xs2)

PD = prepare(divergence(g2), vector_field(g2))
SUITE["grid"]["2D 256²"]["divergence mul!"] = @benchmarkable mul!($ys2, $PD, $xv2)

vel = set!(vector_field(g2), p -> SVector(sin(p[2]), cos(p[1])))
PA = prepare(advection(g2, vel), scalar_field(g2))
SUITE["grid"]["2D 256²"]["advection mul!"] = @benchmarkable mul!($ys2, $PA, $xs2)

κ = set!(scalar_field(g2), p -> 1 + 0.5 * sin(p[1]))
PS = prepare(2.0 * laplacian(g2) + scaling(κ), scalar_field(g2))
SUITE["grid"]["2D 256²"]["2λ + κ·I mul!"] = @benchmarkable mul!($ys2, $PS, $xs2)

K = divergence(g2) * scaling(κ) * gradient(g2)
PK = prepare(K, scalar_field(g2))
SUITE["grid"]["2D 256²"]["∇·(κ∇u) prepare"] = @benchmarkable prepare($K, $(scalar_field(g2)))
SUITE["grid"]["2D 256²"]["∇·(κ∇u) mul!"] = @benchmarkable mul!($ys2, $PK, $xs2)

#--------------------------------------------------------------------------------# block forest

# 8×8 root tiling of 32² blocks (64 uniform leaves) — same DOFs as the 2D grid above,
# so the forest overhead (halo exchange + per-leaf dispatch) is directly comparable.
bf = BlockForest(g2; blocksize=(32, 32), maxlevel=2)
xb = set!(scalar_field(bf), p -> sin(4p[1]) + cos(3p[2]))
Lf = laplacian(bf)
Pf = prepare(Lf, scalar_field(bf))
xf = flatten(xb)
yf = similar(xf)

SUITE["forest"]["2D 64×32²"]["halo_update!"] = @benchmarkable halo_update!($xb, $bf)
SUITE["forest"]["2D 64×32²"]["prepare"] = @benchmarkable prepare($Lf, $(scalar_field(bf)))
SUITE["forest"]["2D 64×32²"]["laplacian mul!"] = @benchmarkable mul!($yf, $Pf, $xf)

# Packed single-launch sweep at the same DOFs — the third leg of the single-grid /
# per-leaf-forest / packed-forest comparison. Guarded: base revisions predate pack.
if isdefined(MatrixFreeOperators, :pack)
    Pp = prepare(Lf, pack(scalar_field(bf)))
    yp = similar(yf)
    SUITE["forest"]["2D 64×32²"]["laplacian mul! (packed)"] = @benchmarkable mul!($yp, $Pp, $xf)
end
