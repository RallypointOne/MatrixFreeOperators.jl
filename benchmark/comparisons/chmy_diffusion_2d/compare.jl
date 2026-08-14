# Congruency check: both packages must produce the same field after NT steps from
# the same seeded IC. The discretizations are algebraically identical (constant-χ
# staggered flux form ≡ compact 5-point Laplacian on a uniform grid), so any
# difference is floating-point op-order drift — machine-eps scale, far below 1e-11.
#
# Run after chmy.jl and mfo.jl: julia --project=. compare.jl [128,512,...]

using Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "common.jl"))

const TOL = 1e-11

fail = false
for n in parse_sizes()
    a, b = load_field("chmy", n), load_field("mfo", n)
    size(a) == size(b) || error("size mismatch at n=$n: $(size(a)) vs $(size(b))")
    maxabs = maximum(abs.(a .- b))
    rel = maxabs / maximum(abs, a)
    ok = rel <= TOL
    global fail |= !ok
    @printf("n=%4d  interior=%4d²  max|Δ|=%.3e  rel=%.3e  %s\n",
        n, n - 2, maxabs, rel, ok ? "PASS" : "FAIL")
end
exit(fail ? 1 : 0)
