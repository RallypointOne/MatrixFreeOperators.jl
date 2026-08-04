# The Niederer et al. (2011) N-version cardiac tissue benchmark on an adaptive
# BlockForest.  Phil. Trans. R. Soc. A 369:4331-4351.
#
# Monodomain electrophysiology on a 20 x 7 x 3 mm slab of ventricular tissue with
# fibres along x, ten Tusscher-Panfilov 2006 epicardial kinetics, stimulated in a
# 1.5 mm cube at one corner:
#
#   ∂V/∂t = ∇·(D∇V) − I_ion(V, s) + I_stim/(βCm),      ∂s/∂t = f(V, s)
#
# with D = diag(D_L, D_T, D_T).  The benchmark metric is *activation time* — the
# first upward crossing of 0 mV — at nine fixed points, and it is the standard
# quantitative check for a monodomain solver.
#
# It is also close to the best case AMR has.  The depolarization wavefront is about
# a millimetre thick and sweeps a 420 mm³ slab, so at any instant almost none of
# the domain needs fine resolution; `regrid!` refines on |∇V| and coarsens behind
# the front.  In 3D each level costs 8x rather than 4x, so the savings are much
# larger than in the 2D examples/monodomain_amr.jl sheet.
#
# What this resolves and what it does not: activation time is the benchmark metric
# and is checked two ways (below).  The plateau and repolarization behind the front
# are deliberately *not* resolved — coarsening there is exactly what makes AMR pay,
# and activation time is insensitive to it.  Do not read the late-time field as a
# converged action potential.
#
# Both the adaptive run and a uniform control at the same finest spacing are timed,
# because cells saved is not the same as time saved.  The run reports the wall clock
# split by phase, and the split is the interesting part: the reaction term scales
# with the cell count as you would hope (5.7x on 5.3x fewer cells) and regridding
# costs under 6%, but diffusion runs *slower* on the adaptive forest.  Essentially
# all of the adaptivity overhead is the coarse-fine halo exchange — 92% of an apply
# on an adapted forest, and paid twice here because the anisotropic operator is an
# `Added` whose operands each exchange independently.
#
# Run with:      julia -t auto --project=examples examples/niederer_benchmark.jl
# Fast check:    NIEDERER_SMOKE=true        ...  (seconds; exercises the 3D forest paths)
# Adaptive only: NIEDERER_SKIP_UNIFORM=true ...  (skips the uniform control)

using Pkg; Pkg.activate(@__DIR__)
using MatrixFreeOperators, CairoMakie, LinearAlgebra, Printf, StaticArrays

const MFO = MatrixFreeOperators
include(joinpath(@__DIR__, "ten_tusscher_2006.jl"))

#--------------------------------------------------------------------------------# Physiology (Niederer 2011 Table 3)

const LX, LY, LZ = 20.0, 7.0, 3.0    # slab dimensions (mm); fibres along x

# Intra- and extracellular conductivities in mS/mm.  σ in S/m is numerically equal
# to mS/mm, so the Table 3 values carry over with no conversion factor.
const σ_iL, σ_iT = 0.17, 0.019
const σ_eL, σ_eT = 0.62, 0.24
const σ_L = σ_iL * σ_eL / (σ_iL + σ_eL)   # monodomain harmonic mean, 0.133418 mS/mm
const σ_T = σ_iT * σ_eT / (σ_iT + σ_eT)   #                           0.017606 mS/mm

const BETA = 140.0                   # surface-to-volume ratio (1/mm)
const CM = 0.01                      # membrane capacitance (µF/mm²)
const D_L = σ_L / (BETA * CM)        # 0.0952984 mm²/ms along the fibre
const D_T = σ_T / (BETA * CM)        # 0.0125758 mm²/ms across it

# 50 000 µA/cm³ == 50 µA/mm³ for 2 ms in a 1.5 mm corner cube.  Dividing by
# βCm = 1.4 µF/mm³ puts it in mV/ms, matching the pA/pF ionic currents.
const STIM_DVDT = 50.0 / (BETA * CM)      # 35.714 mV/ms
const STIM_DUR = 2.0                      # ms; applied for t < STIM_DUR
const STIM_SIZE = 1.5                     # mm
const V_ACTIVATION = 0.0                  # mV

#--------------------------------------------------------------------------------# Discretization

const SMOKE = get(ENV, "NIEDERER_SMOKE", "false") == "true"
const SKIP_UNIFORM = get(ENV, "NIEDERER_SKIP_UNIFORM", "false") == "true"

const BS = (8, 8, 8)                 # cells per block; even and >= 4 for coarse-fine
const DT = 0.01                      # ms

# The base tiling is chosen so that the finest spacing lands on Niederer's 0.1 mm
# reference resolution while every dimension stays divisible by the blocksize:
# 48x16x8 base cells over 20x7x3 mm gives (0.417, 0.438, 0.375) mm at level 0 and
# (0.104, 0.109, 0.094) mm at level 2.  Cells are slightly non-cubic; the operator
# carries per-axis spacing, so that costs nothing.
const NBASE = (48, 16, 8)
const MAXLEV = SMOKE ? 1 : 2

# The uniform control is a *forest with no refinement* at the finest spacing rather
# than a plain CartesianGrid.  That holds the forest machinery fixed and isolates
# exactly what adaptivity adds: coarse-fine interfaces and the regrid transfer.
const NUNIFORM = NBASE .* 2^(SMOKE ? 1 : 2)
const UNIFORM_CELLS = prod(NUNIFORM)      # 393 216 at MAXLEV = 2

const TEND = SMOKE ? 12.0 : 60.0     # ms; the last reference point activates near 44 ms
const REGRID_EVERY = 50              # steps == 0.5 ms; the front advances ~0.3 mm
const SAMPLE_EVERY = 10              # steps == 0.1 ms, matching the reference's quantization

# Absolute |∇V| thresholds in mV/mm.  V is in physical millivolts here, so the
# upstroke gradient is unambiguous — the front runs about 200 mV/mm (105 mV across a
# ~0.5 mm upstroke), against ~0 both at rest and in the plateau.  The 2D example
# normalizes by the running peak, which needs a zero guard at t = 0 and degenerates
# once the whole slab has depolarized; absolute thresholds have neither failure mode.
#
# The answer is insensitive to these: sweeping TAU_REFINE over 20-100 and TAU_COARSEN
# over 5-40 leaves all nine activation times bit-identical, because refinement is
# block-granular and any threshold in that range flags the same blocks.  What changes
# is only how fast the wake coarsens.
const TAU_REFINE = 60.0
const TAU_COARSEN = 20.0

const SNAP_TIMES = SMOKE ? [4.0, 8.0, 11.0] : [5.0, 15.0, 30.0, 45.0]
const Z_SLICE = 1.5                  # mid-plane for the figures (mm)

# Smoke runs write under their own names so a quick check cannot overwrite the
# committed full-resolution figures.
outpath(name) = joinpath(@__DIR__, SMOKE ? "niederer_smoke_$name" : "niederer_$name")

#--------------------------------------------------------------------------------# Reference points

# Niederer 2011 Figure 1b: the eight slab corners plus the centre.  P1 is the
# stimulated corner and P8 the far corner, the first and last to activate.
const P_NAMES = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9")
const P_REF = (
    (0.0, 0.0, 0.0), (0.0, 7.0, 0.0), (20.0, 7.0, 0.0), (20.0, 0.0, 0.0),
    (0.0, 0.0, 3.0), (0.0, 7.0, 3.0), (20.0, 0.0, 3.0), (20.0, 7.0, 3.0),
    (10.0, 3.5, 1.5),
)

# Cross-code reference: a structured 7-point finite-difference solve of the same
# problem at dx = 0.1 mm, dt = 0.01 ms (RadialBasisFunctions.jl/Macchiato).  An
# independent discretization, so agreement here validates the physics rather than
# just the grid.  Note that implementation's calcium handling differs (see the
# header of ten_tusscher_2006.jl); activation time is driven by I_Na and diffusion
# and is insensitive to that.
const FD_REFERENCE = (1.3, 30.1, 42.9, 31.9, 9.1, 31.5, 33.0, 43.8, 19.9)

const NDIAG = 50                     # samples along the P1 -> P8 diagonal

#--------------------------------------------------------------------------------# Fields and operators

"""
∇·(D∇V) for a constant diagonal D = diag(D_L, D_T, D_T) with fibres along x.
Because D is axis-aligned this is exactly D_T∇² + (D_L − D_T)∂ₓₓ — no tensor
coefficient is needed, and writing it as a Laplacian plus a correction costs one
inter-block halo exchange per step less than three separate second derivatives.

Zero-flux boundaries are `Neumann()` on all six faces: for axis-aligned diagonal D,
n·D∇V = 0 reduces to ∂V/∂n = 0, which is what the mirror ghost fill applies.  Zero
flux also means `boundary_rhs` vanishes, so there is nothing to fold into a source.
"""
diffusion_operator(g) = D_T * laplacian(g) + (D_L - D_T) * derivative(g, 1; order=2)

"""
A `BlockField` holding all 18 gating and concentration states per cell in one
`SVector`.  `scalar_field` only accepts `T<:Number` and `vector_field` fixes the
component count at the spatial dimension, so this uses the raw constructor.  The
regrid transfer is element-type generic — interpolation and averaging are linear
combinations — so these ride through `regrid!` alongside V.
"""
function state_field(bf)
    psize = bf.blocksize .+ 2 .* bf.halo
    blocks = [zeros(SVector{NSTATES,Float64}, psize...) for _ in 1:MFO.nleaves(bf)]
    return BlockField(blocks, bf)
end

"Cell-centre coordinate of interior index `I` in a leaf with extent `ext`, spacing `sp`."
@inline cellcenter(ext, sp, I) = ntuple(d -> ext[d][1] + (I[d] - 0.5) * sp[d], 3)

#--------------------------------------------------------------------------------# Time stepping

"""
Diffusion half of the Godunov split: explicit Euler on ∇·(D∇V).

Stable with room to spare — at the finest spacing the operator's largest eigenvalue
is 4(D_L/hx² + D_T/hy² + D_T/hz²) ≈ 45, so the bound is dt < 0.044 ms.  The TT06
upstroke, not diffusion, is what pins dt at 0.01.  That is fortunate: an implicit
solve on an adapted forest has no preconditioner available (`operator_diagonal`
rejects Interface faces and multigrid rejects a BlockForest outright), so it would
mean unpreconditioned GMRES every step.
"""
function diffuse!(V, LV, Lop, bf)
    apply!(LV, Lop, V, bf)
    Threads.@threads for i in 1:MFO.nleaves(bf)
        lg = MFO.leaf_grid(bf, i)
        v = interior(MFO.block(V, i, lg))
        lv = interior(MFO.block(LV, i, lg))
        @. v += DT * lv
    end
    return nothing
end

"Reaction half: pointwise TT06 forward Euler, plus the stimulus while it is on."
function react!(V, S, bf, t)
    stim_on = t < STIM_DUR
    Threads.@threads for i in 1:MFO.nleaves(bf)
        lg = MFO.leaf_grid(bf, i)
        ext, sp = lg.extent, spacing(lg)
        v = interior(MFO.block(V, i, lg))
        s = interior(MFO.block(S, i, lg))
        # Only leaves overlapping the corner cube can be stimulated; skip the
        # coordinate test everywhere else.
        maybe_stim = stim_on && all(d -> ext[d][1] < STIM_SIZE, 1:3)
        @inbounds for I in CartesianIndices(v)
            stim = 0.0
            if maybe_stim
                x = cellcenter(ext, sp, Tuple(I))
                all(d -> x[d] <= STIM_SIZE, 1:3) && (stim = STIM_DVDT)
            end
            v[I], s[I] = step_cell(v[I], s[I], DT, stim)
        end
    end
    return nothing
end

#--------------------------------------------------------------------------------# Probing

"""
Sample `V` at each of `points`, in one pass over the leaves.

Probes must be re-located every time: the leaf covering a point changes refinement
level as the forest adapts, so a cached (leaf, index) pair silently reads the wrong
cell after a regrid.
"""
function probe_all(V, bf, points)
    out = fill(NaN, length(points))
    for (i, (_, lg)) in enumerate(leaves(bf))
        ext, sp = lg.extent, spacing(lg)
        vi = interior(MFO.block(V, i, lg))
        for (k, x) in enumerate(points)
            isnan(out[k]) || continue
            all(d -> ext[d][1] <= x[d] <= ext[d][2], 1:3) || continue
            idx = ntuple(d -> clamp(ceil(Int, (x[d] - ext[d][1]) / sp[d]), 1, bf.blocksize[d]), 3)
            out[k] = vi[idx...]
        end
    end
    return out
end

"Points evenly spaced along the P1 -> P8 diagonal, and their arc lengths."
function diagonal_points()
    p1, p8 = SVector(P_REF[1]), SVector(P_REF[8])
    total = norm(p8 - p1)
    ss = range(0, 1; length=NDIAG)
    return [Tuple(p1 + s * (p8 - p1)) for s in ss], [s * total for s in ss]
end

#--------------------------------------------------------------------------------# Figures

const LEVELCOLORS = Makie.wong_colors()

"Leaf geometry and the z = Z_SLICE plane of V — enough to redraw after the forest moves on."
function snapshot(V, bf, t)
    data = []
    for (i, (key, lg)) in enumerate(leaves(bf))
        ext, sp = lg.extent, spacing(lg)
        # Half-open in z so a leaf boundary landing exactly on the slice is picked
        # by one leaf, not two overlapping ones.
        ext[3][1] <= Z_SLICE < ext[3][2] || continue
        k = clamp(ceil(Int, (Z_SLICE - ext[3][1]) / sp[3]), 1, bf.blocksize[3])
        push!(data, (extent=(ext[1], ext[2]), level=key.level,
                     vals=collect(view(interior(MFO.block(V, i, lg)), :, :, k))))
    end
    return (t=t, leaves=data, nleaf=MFO.nleaves(bf), ncells=MFO.nleaves(bf) * prod(BS))
end

function draw_slice!(ax, leafdata)
    hm = nothing
    for lf in leafdata
        n = size(lf.vals)
        sp = ntuple(d -> (lf.extent[d][2] - lf.extent[d][1]) / n[d], 2)
        xs, ys = ntuple(d -> range(lf.extent[d][1] + 0.5sp[d],
                                   lf.extent[d][2] - 0.5sp[d]; length=n[d]), 2)
        hm = heatmap!(ax, xs, ys, lf.vals; colorrange=(-90, 40), colormap=:inferno)
    end
    for lf in leafdata
        e = lf.extent
        lines!(ax, [e[1][1], e[1][2], e[1][2], e[1][1], e[1][1]],
                   [e[2][1], e[2][1], e[2][2], e[2][2], e[2][1]];
               color=LEVELCOLORS[lf.level + 1], linewidth=max(0.4, 1.6 / 1.6^lf.level))
    end
    return hm
end

function slices_figure(snaps, path)
    fig = Figure(size=(1180, 240 * length(snaps) + 190), fontsize=17)
    hm = nothing
    for (j, s) in enumerate(snaps)
        ax = Axis(fig[j, 1]; aspect=DataAspect(), ylabel="y (mm)",
                  xlabel=(j == length(snaps) ? "x (mm)" : ""),
                  title=@sprintf("t = %.0f ms — %d leaves, %.1f%% of uniform",
                                 s.t, s.nleaf, 100 * s.ncells / UNIFORM_CELLS))
        hm = draw_slice!(ax, s.leaves)
        j < length(snaps) && hidexdecorations!(ax; grid=false)
        limits!(ax, 0, LX, 0, LY)
    end
    Colorbar(fig[1:length(snaps), 2], hm; label="transmembrane potential V (mV)")
    Label(fig[0, 1:2],
          "Niederer benchmark on an adaptive block forest — z = $(Z_SLICE) mm mid-plane";
          fontsize=23, font=:bold)
    maxlev = maximum(lf -> lf.level, snaps[end].leaves)
    fig[length(snaps) + 1, 1:2] = Legend(
        fig, [LineElement(color=LEVELCOLORS[l + 1], linewidth=3) for l in 0:maxlev],
        ["level $l" for l in 0:maxlev], "leaf refinement";
        orientation=:horizontal, framevisible=false, titleposition=:left)
    Label(fig[length(snaps) + 2, 1:2],
          "ten Tusscher-Panfilov 2006 epicardial kinetics; D = diag($(round(D_L; digits=4)), " *
          "$(round(D_T; digits=4)), $(round(D_T; digits=4))) mm²/ms, fibres along x.";
          fontsize=14, color=:gray25)
    rowgap!(fig.layout, 6)
    save(path, fig)
    return path
end

function activation_figure(dist, at_amr, at_uniform, path)
    fig = Figure(size=(820, 480), fontsize=17)
    ax = Axis(fig[1, 1]; xlabel="distance along the P1 → P8 diagonal (mm)",
              ylabel="activation time (ms)",
              title="Activation along the slab diagonal")
    lines!(ax, dist, at_amr; color=LEVELCOLORS[1], linewidth=3, label="adaptive forest")
    if at_uniform !== nothing
        lines!(ax, dist, at_uniform; color=:black, linewidth=1.8, linestyle=:dash,
               label="uniform forest, finest spacing")
    end
    # Only the reference points that actually lie on this diagonal belong here —
    # P1, the centre P9, and the far corner P8. The other six are corners off the
    # line; they are in the printed table, not on this plot.
    p1, p8 = SVector(P_REF[1]), SVector(P_REF[8])
    û = (p8 - p1) / norm(p8 - p1)
    on_diag = [k for k in eachindex(P_REF) if
               norm((SVector(P_REF[k]) - p1) - ((SVector(P_REF[k]) - p1) ⋅ û) * û) < 1e-9]
    scatter!(ax, [norm(SVector(P_REF[k]) - p1) for k in on_diag],
             [FD_REFERENCE[k] for k in on_diag];
             color=:firebrick, markersize=14, marker=:diamond,
             label="finite-difference reference, dx = 0.1 mm")
    axislegend(ax; position=:lt, framevisible=false)
    save(path, fig)
    return path
end

function cost_figure(amr, uniform, path)
    ts = [h[1] for h in amr.hist]
    pct = [100 * h[2] / UNIFORM_CELLS for h in amr.hist]
    mean_pct = sum(pct) / length(pct)

    fig = Figure(size=(1150, 470), fontsize=16)

    ax1 = Axis(fig[1, 1]; xlabel="time (ms)", ylabel="% of uniform-grid cells",
               title=@sprintf("Cells carried: mean %.1f%% — %.1f× fewer",
                              mean_pct, 100 / mean_pct),
               subtitle=@sprintf("uniform: %d × %d × %d = %d cells",
                                 NUNIFORM[1], NUNIFORM[2], NUNIFORM[3], UNIFORM_CELLS))
    lines!(ax1, ts, pct; color=LEVELCOLORS[1], linewidth=3)
    band!(ax1, ts, zeros(length(ts)), pct; color=(LEVELCOLORS[1], 0.18))
    hlines!(ax1, [mean_pct]; color=:gray45, linestyle=:dash, linewidth=1.5)
    ylims!(ax1, 0, min(100, 1.3 * maximum(pct)))
    hidespines!(ax1, :t, :r)

    runs = uniform === nothing ? [("adaptive", amr)] :
           [("adaptive", amr), ("uniform", uniform)]
    xs, ys, grp = Int[], Float64[], Int[]
    for (ri, (_, r)) in enumerate(runs)
        tt = phase_times(r)
        for i in eachindex(PHASES)
            tt[i] > 1e-6 || continue
            push!(xs, ri); push!(ys, tt[i]); push!(grp, i)
        end
    end
    ax2 = Axis(fig[1, 2]; ylabel="wall time (s)",
               title=(uniform === nothing ? "Wall time by phase" :
                      @sprintf("Wall time: %.1f× faster overall", uniform.wall / amr.wall)),
               subtitle="cells saved ≠ time saved — the gap is the adaptivity overhead",
               xticks=(1:length(runs), [r[1] for r in runs]))
    barplot!(ax2, xs, ys; stack=grp, color=LEVELCOLORS[grp], width=0.5)
    for (ri, (_, r)) in enumerate(runs)
        text!(ax2, ri, r.wall; text=@sprintf("%.0f s", r.wall),
              align=(:center, :bottom), offset=(0, 6), fontsize=16)
    end
    ylims!(ax2, 0, 1.2 * maximum(r[2].wall for r in runs))
    xlims!(ax2, 0.4, length(runs) + 0.6)
    hidespines!(ax2, :t, :r)

    used = sort(unique(grp))
    Legend(fig[1, 3], [PolyElement(color=LEVELCOLORS[i]) for i in used],
           [PHASES[i] for i in used], "phase"; framevisible=false)
    save(path, fig)
    return path
end

#--------------------------------------------------------------------------------# Solver

function simulate(; adaptive::Bool)
    ncells = adaptive ? NBASE : NUNIFORM
    maxlev = adaptive ? MAXLEV : 0
    base = CartesianGrid(((0.0, LX), (0.0, LY), (0.0, LZ)), ncells;
                         bc=ntuple(_ -> (Neumann(), Neumann()), 3))
    bf = BlockForest(base; blocksize=BS, maxlevel=maxlev)

    # At t = 0 the slab is uniformly at rest, so |∇V| ≡ 0 and a gradient indicator
    # cannot bootstrap. Pre-refine the stimulus corner geometrically instead.
    if adaptive
        for _ in 1:maxlev
            refine!(bf, x -> all(d -> x[d] < STIM_SIZE + 0.5, 1:3))
        end
        balance!(bf)
    end

    V = set!(scalar_field(bf), _ -> initial_voltage())
    S = set!(state_field(bf), _ -> initial_state())
    LV = scalar_field(bf)
    Lop = diffusion_operator(bf)

    diagpts, diagdist = diagonal_points()
    probes = vcat(collect(P_REF), diagpts)
    act = fill(NaN, length(probes))

    # Warm the hot paths before the timers start.  Whichever run goes first would
    # otherwise be charged for compiling kernels the second run inherits for free,
    # which is precisely the comparison being measured.  The fields are rebuilt
    # afterwards, so the warmup leaves no trace in the solution.
    diffuse!(V, LV, Lop, bf)
    react!(V, S, bf, 0.0)
    probe_all(V, bf, probes)
    V = set!(scalar_field(bf), _ -> initial_voltage())
    S = set!(state_field(bf), _ -> initial_state())
    LV = scalar_field(bf)

    snaps, hist = Any[], Tuple{Float64,Int}[]
    t, nstep, next_snap = 0.0, 0, 1
    t_diff, t_react, t_regrid, t_probe = 0.0, 0.0, 0.0, 0.0
    cellsteps = 0
    tstart = time_ns()

    while t < TEND - 1e-9
        c = time_ns()                     # Godunov: diffusion first, then reaction
        diffuse!(V, LV, Lop, bf)
        t_diff += (time_ns() - c) * 1e-9
        c = time_ns()
        react!(V, S, bf, t)
        t_react += (time_ns() - c) * 1e-9
        t += DT
        nstep += 1
        cellsteps += MFO.nleaves(bf) * prod(BS)

        if nstep % SAMPLE_EVERY == 0
            c = time_ns()
            vals = probe_all(V, bf, probes)
            @inbounds for k in eachindex(vals)
                isnan(act[k]) && vals[k] > V_ACTIVATION && (act[k] = t)
            end
            t_probe += (time_ns() - c) * 1e-9
            push!(hist, (t, MFO.nleaves(bf) * prod(BS)))
        end

        if adaptive && nstep % REGRID_EVERY == 0
            c = time_ns()
            η = MFO.gradient(bf) * V
            η, V, S = regrid!(η, V, S;
                              refine=b -> maximum(norm, interior(b)) > TAU_REFINE,
                              coarsen=b -> maximum(norm, interior(b)) < TAU_COARSEN)
            LV = scalar_field(bf)         # the old one belongs to the previous generation
            Lop = diffusion_operator(bf)
            t_regrid += (time_ns() - c) * 1e-9
        end

        if next_snap <= length(SNAP_TIMES) && t >= SNAP_TIMES[next_snap]
            push!(snaps, snapshot(V, bf, t))
            lo, hi = extrema(flatten(V))
            @printf "  t = %5.1f ms   %5d leaves   %6.2f%% of uniform   V ∈ [%7.2f, %6.2f]\n" t MFO.nleaves(bf) (100 * MFO.nleaves(bf) * prod(BS) / UNIFORM_CELLS) lo hi
            next_snap += 1
        end
    end

    wall = (time_ns() - tstart) * 1e-9
    @printf "  %d steps in %.1f s (%.2f ms/step, %.1f ns per cell-update)\n" nstep wall (1e3 * wall / nstep) (1e9 * wall / cellsteps)
    return (; act_ref=act[1:9], act_diag=act[10:end], diagdist, snaps, hist, wall,
            t_diff, t_react, t_regrid, t_probe, cellsteps, nleaf=MFO.nleaves(bf))
end

# Phase buckets, in the order they are reported.  "other" is whatever the timers did
# not attribute — snapshots and loop overhead — so the buckets always sum to `wall`.
const PHASES = ("diffusion", "reaction", "regrid", "probing", "other")

phase_times(r) = (r.t_diff, r.t_react, r.t_regrid, r.t_probe,
                  max(0.0, r.wall - r.t_diff - r.t_react - r.t_regrid - r.t_probe))

#--------------------------------------------------------------------------------# Reporting

function report(act_ref, label)
    @printf "\n  %-4s %-22s %10s %12s %9s\n" "" "target (mm)" "$label" "FD ref" "Δ"
    for k in 1:9
        p = P_REF[k]
        a = act_ref[k]
        astr = isnan(a) ? "  not activ." : @sprintf("%9.1f ms", a)
        dstr = isnan(a) ? "     —" : @sprintf("%+6.1f", a - FD_REFERENCE[k])
        @printf "  %-4s (%5.1f,%4.1f,%4.1f) %s %9.1f ms %s\n" P_NAMES[k] p[1] p[2] p[3] astr FD_REFERENCE[k] dstr
    end
    good = [k for k in 1:9 if !isnan(act_ref[k])]
    if !isempty(good)
        err = [abs(act_ref[k] - FD_REFERENCE[k]) for k in good]
        @printf "\n  %d/9 activated;  mean |Δ| vs FD %.2f ms,  max %.2f ms\n" length(good) (sum(err) / length(err)) maximum(err)
    end
    return nothing
end

"""
Where the wall time actually went. The headline is the decomposition of the overall
speedup: cells saved divided by the extra cost per cell.
"""
function report_cost(amr, uniform)
    ta = phase_times(amr)
    println("\ncomputational cost")
    if uniform === nothing
        @printf "\n  %-11s %10s %8s\n" "phase" "adaptive" "share"
        for i in eachindex(PHASES)
            @printf "  %-11s %8.1f s %7.1f%%\n" PHASES[i] ta[i] (100 * ta[i] / amr.wall)
        end
        @printf "  %-11s %8.1f s\n" "TOTAL" amr.wall
        @printf "\n  %.3g cell-updates at %.1f ns each\n" Float64(amr.cellsteps) (1e9 * amr.wall / amr.cellsteps)
        return nothing
    end

    tu = phase_times(uniform)
    @printf "\n  %-11s %10s %10s %9s\n" "phase" "adaptive" "uniform" "speedup"
    for i in eachindex(PHASES)
        # A ratio only means something when both runs actually spend time here;
        # `regrid` is adaptive-only by construction.
        sp = (ta[i] > 1e-3 && tu[i] > 1e-3) ? @sprintf("%8.2f×", tu[i] / ta[i]) : "        —"
        @printf "  %-11s %8.1f s %8.1f s %s\n" PHASES[i] ta[i] tu[i] sp
    end
    @printf "  %-11s %8.1f s %8.1f s %8.2f×\n" "TOTAL" amr.wall uniform.wall (uniform.wall / amr.wall)

    cellratio = uniform.cellsteps / amr.cellsteps
    nsa = 1e9 * amr.wall / amr.cellsteps
    nsu = 1e9 * uniform.wall / uniform.cellsteps
    @printf "\n  %-11s %8.3g   %8.3g   %8.2f× fewer\n" "cell-updates" Float64(amr.cellsteps) Float64(uniform.cellsteps) cellratio
    @printf "  %-11s %8.1f   %8.1f   %8.2f× dearer\n" "ns/update" nsa nsu (nsa / nsu)
    @printf "\n  %.1f× fewer cell-updates ÷ %.1f× higher cost per update = %.1f× faster overall.\n" cellratio (nsa / nsu) (uniform.wall / amr.wall)
    @printf "  Adaptivity overhead is real: %.0f%% of the cell saving is given back at the\n" (100 * (1 - (uniform.wall / amr.wall) / cellratio))
    @printf "  coarse-fine interfaces, in regridding, and in the rebuilt exchange schedule.\n"
    return nothing
end

function write_diagonal_csv(dist, at, path)
    pts, _ = diagonal_points()
    open(path, "w") do io
        println(io, "distance_mm,x_mm,y_mm,z_mm,activation_time_ms")
        for i in eachindex(dist)
            a = isnan(at[i]) ? "NaN" : @sprintf("%.2f", at[i])
            @printf io "%.4f,%.4f,%.4f,%.4f,%s\n" dist[i] pts[i][1] pts[i][2] pts[i][3] a
        end
    end
    return path
end

#--------------------------------------------------------------------------------

@printf "Niederer 2011 benchmark — %s\n" (SMOKE ? "SMOKE mode (reduced levels and duration)" : "full run")
@printf "  slab %.0f × %.0f × %.0f mm, D = (%.5f, %.5f, %.5f) mm²/ms, fibres along x\n" LX LY LZ D_L D_T D_T
@printf "  base %d×%d×%d, blocksize %d³, maxlevel %d → finest spacing (%.3f, %.3f, %.3f) mm\n" NBASE[1] NBASE[2] NBASE[3] BS[1] MAXLEV (LX / NUNIFORM[1]) (LY / NUNIFORM[2]) (LZ / NUNIFORM[3])
@printf "  dt = %.3f ms to t = %.0f ms, %d threads\n\n" DT TEND Threads.nthreads()

println("adaptive forest:")
amr = simulate(; adaptive=true)
report(amr.act_ref, "adaptive")

uniform = nothing
if !SKIP_UNIFORM
    @printf "\nuniform forest at the finest spacing (%d cells):\n" UNIFORM_CELLS
    uniform = simulate(; adaptive=false)
    report(uniform.act_ref, "uniform")
    d = [abs(a - u) for (a, u) in zip(amr.act_ref, uniform.act_ref) if !isnan(a) && !isnan(u)]
    if !isempty(d)
        @printf "\n  adaptive vs uniform activation: mean |Δ| %.2f ms, max %.2f ms\n" (sum(d) / length(d)) maximum(d)
    end
end

report_cost(amr, uniform)

p1 = slices_figure(amr.snaps, outpath("benchmark.png"))
@printf "\nfigure: %s\n" p1
p2 = activation_figure(amr.diagdist, amr.act_diag,
                       uniform === nothing ? nothing : uniform.act_diag,
                       outpath("activation.png"))
@printf "figure: %s\n" p2
p3 = cost_figure(amr, uniform, outpath("cost.png"))
@printf "figure: %s\n" p3
p4 = write_diagonal_csv(amr.diagdist, amr.act_diag, outpath("diagonal.csv"))
@printf "data:   %s\n" p4
