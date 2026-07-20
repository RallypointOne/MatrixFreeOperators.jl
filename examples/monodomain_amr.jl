# Monodomain cardiac electrophysiology on an adaptive BlockForest.
#
# Aliev-Panfilov kinetics (dimensionless):
#   ∂V/∂t = D∇²V − kV(V−a)(V−1) − VW
#   ∂W/∂t = (ε₀ + μ₁W/(μ₂+V))(−W − kV(V−a−1))
#
# An S1-S2 cross-field protocol breaks a planar wave against partially-refractory
# tissue, producing a reentrant spiral — the mechanism behind reentrant
# arrhythmia. `regrid!` refines on |∇V|, so resolution tracks the depolarization
# wavefront and coarsens behind it: the whole point of AMR for tissue EP, where
# the upstroke is razor-thin compared with the tissue it crosses.
#
# Run with:   julia --project=examples examples/monodomain_amr.jl
# Animation:  EP_GIF=true julia --project=examples examples/monodomain_amr.jl
#             (writes monodomain_amr.mp4; the sim reruns, so it takes ~15 min)

using Pkg; Pkg.activate(@__DIR__)
using MatrixFreeOperators, CairoMakie, LinearAlgebra, Printf

const MFO = MatrixFreeOperators

#--------------------------------------------------------------------------------
# Model and discretization

const K_AP = 8.0
const A_AP = 0.15
const EPS0 = 0.002
const MU1  = 0.2
const MU2  = 0.3
const DIFF = 1.0

const L      = 80.0            # square tissue sheet, side L (space units)
const NBASE  = 64              # base grid: the level-0 root tiling
const BS     = 8               # cells per block edge
const MAXLEV = 3
const DT     = 0.006           # explicit Euler; stable at the finest spacing
const TEND   = 220.0
const T_S2   = 50.0
const REGRID_EVERY = 20

# Refine/coarsen thresholds as fractions of the peak |∇V|. The band only has to
# stay ahead of the front between regrids — at REGRID_EVERY steps the front
# advances ≪ one fine block — so it can hug the upstroke tightly.
const TAU_REFINE  = 0.25
const TAU_COARSEN = 0.08

const UNIFORM_CELLS = (NBASE * 2^MAXLEV)^2   # equal-resolution uniform grid

#--------------------------------------------------------------------------------
# Simulation

# Cell-center coordinates of a leaf, from its extent — `cell_center` takes a
# padded index, but block interiors are indexed from 1.
function leaf_axes(bf, lg)
    sp, ext = spacing(lg), lg.extent
    return ntuple(d -> range(ext[d][1] + 0.5sp[d], ext[d][2] - 0.5sp[d];
                             length=bf.blocksize[d]), 2)
end

function step!(V, W, bf, dv, dw)
    LV = laplacian(bf) * V
    for i in 1:MFO.nleaves(bf)
        v  = interior(MFO.block(V, i))
        w  = interior(MFO.block(W, i))
        lv = interior(MFO.block(LV, i))
        @. dv = DIFF * lv - K_AP * v * (v - A_AP) * (v - 1) - v * w
        @. dw = (EPS0 + MU1 * w / (MU2 + v)) * (-w - K_AP * v * (v - A_AP - 1))
        @. v += DT * dv
        @. w += DT * dw
    end
    return nothing
end

"Depolarize every cell whose center satisfies `inregion`."
function stimulate!(V, bf, inregion)
    for (i, (_, lg)) in enumerate(leaves(bf))
        xs, ys = leaf_axes(bf, lg)
        v = interior(MFO.block(V, i))
        for (jy, y) in enumerate(ys), (jx, x) in enumerate(xs)
            inregion(x, y) && (v[jx, jy] = 1.0)
        end
    end
end

"Leaf geometry plus values — enough to redraw a frame after the forest has moved on."
function snapshot(V, bf, t)
    data = [(extent=lg.extent, level=key.level,
             vals=collect(interior(MFO.block(V, i))))
            for (i, (key, lg)) in enumerate(leaves(bf))]
    return (t=t, leaves=data, nleaf=MFO.nleaves(bf))
end

"""
Run the S1-S2 protocol. Returns snapshots at `snap_times`; if `on_frame` is
given it is called as `on_frame(V, bf, t)` every `frame_every` steps instead of
collecting, so the animation reuses this exact solver.
"""
function simulate(; snap_times=Float64[], on_frame=nothing, frame_every=250)
    base = CartesianGrid(((0.0, L), (0.0, L)), (NBASE, NBASE);
                         bc=ntuple(_ -> (Neumann(), Neumann()), 2))
    bf = BlockForest(base; blocksize=(BS, BS), maxlevel=MAXLEV)

    V = set!(scalar_field(bf), x -> x[1] < 4.0 ? 1.0 : 0.0)   # S1 along the left edge
    W = scalar_field(bf)
    dv, dw = zeros(BS, BS), zeros(BS, BS)

    snaps = Any[]
    t, nstep, did_s2, next_snap = 0.0, 0, false, 1
    tstart = time()

    while t < TEND
        step!(V, W, bf, dv, dw)
        t += DT
        nstep += 1

        # S2 into the lower-left quadrant: its left part has recovered from S1
        # while its right part is still refractory, so the new wave breaks.
        if !did_s2 && t >= T_S2
            stimulate!(V, bf, (x, y) -> x < 0.5L && y < 0.5L)
            did_s2 = true
        end

        if nstep % REGRID_EVERY == 0
            η = MFO.gradient(bf) * V
            gmax = maximum(i -> maximum(norm, interior(MFO.block(η, i))),
                           1:MFO.nleaves(bf))
            η, V, W = regrid!(η, V, W;
                              refine  = b -> maximum(norm, interior(b)) > TAU_REFINE * gmax,
                              coarsen = b -> maximum(norm, interior(b)) < TAU_COARSEN * gmax)
        end

        if next_snap <= length(snap_times) && t >= snap_times[next_snap]
            push!(snaps, snapshot(V, bf, t))
            @printf "  t=%6.1f  %5d leaves  %6.1f%% of uniform\n" t MFO.nleaves(bf) (
                100 * MFO.nleaves(bf) * BS^2 / UNIFORM_CELLS)
            next_snap += 1
        end

        on_frame !== nothing && nstep % frame_every == 0 && on_frame(V, bf, t)
    end
    @printf "  %d steps, %.1fs wall\n" nstep (time() - tstart)
    return snaps
end

#--------------------------------------------------------------------------------
# Figures

const LEVELCOLORS = Makie.wong_colors()

"Heatmap of V per leaf plus each leaf's outline, colored by refinement level."
function draw_state!(ax, leafdata; scale=1.0)
    hm = nothing
    for lf in leafdata
        n = size(lf.vals)
        sp = ntuple(d -> (lf.extent[d][2] - lf.extent[d][1]) / n[d], 2)
        xs, ys = ntuple(d -> range(lf.extent[d][1] + 0.5sp[d],
                                   lf.extent[d][2] - 0.5sp[d]; length=n[d]), 2)
        hm = heatmap!(ax, xs, ys, lf.vals; colorrange=(0, 1), colormap=:viridis)
    end
    # Thin the outlines with depth: fine leaves are numerous and would otherwise
    # mat into a solid haze over the field they are meant to resolve.
    for lf in leafdata
        e = lf.extent
        lines!(ax, [e[1][1], e[1][2], e[1][2], e[1][1], e[1][1]],
                   [e[2][1], e[2][1], e[2][2], e[2][2], e[2][1]];
               color=LEVELCOLORS[lf.level + 1],
               linewidth=scale * max(0.45, 1.7 / 1.6^lf.level))
    end
    return hm
end

function level_legend(fig, maxlevel)
    elems = [LineElement(color=LEVELCOLORS[l + 1], linewidth=3) for l in 0:maxlevel]
    labels = ["level $l" for l in 0:maxlevel]
    return Legend(fig, elems, labels, "leaf refinement";
                  orientation=:horizontal, framevisible=false, titleposition=:left)
end

function panel_figure(snaps, captions, path)
    fig = Figure(size=(1500, 530), fontsize=17)
    hm = nothing
    for (j, s) in enumerate(snaps)
        ax = Axis(fig[1, j]; aspect=DataAspect(), title=@sprintf("t = %.0f — %s", s.t, captions[j]),
                  xlabel="x", ylabel=(j == 1 ? "y" : ""))
        hm = draw_state!(ax, s.leaves)
        j > 1 && hideydecorations!(ax; grid=false)
        limits!(ax, 0, L, 0, L)
    end
    ncol = length(snaps) + 1
    Colorbar(fig[1, ncol], hm; label="normalized transmembrane potential V")
    # Span the title/caption only once every column exists — `:` on an empty
    # layout resolves to column 1 and stretches it.
    Label(fig[0, 1:ncol],
          "Monodomain cardiac EP on an adaptive block forest — resolution follows the wavefront";
          fontsize=24, font=:bold, padding=(0, 0, 2, 0))
    fig[2, 1:ncol] = level_legend(fig, MAXLEV)
    Label(fig[3, 1:ncol],
          "Aliev–Panfilov kinetics, dimensionless time and space; S1 along the left edge, " *
          "S2 into the lower-left quadrant at t = $(Int(T_S2)).";
          fontsize=15, color=:gray25)
    rowgap!(fig.layout, 6)
    save(path, fig)
    return path
end

function savings_figure(snaps, captions, path)
    pct = [100 * s.nleaf * BS^2 / UNIFORM_CELLS for s in snaps]
    labels = [@sprintf("t = %.0f\n%s", s.t, captions[j]) for (j, s) in enumerate(snaps)]

    fig = Figure(size=(760, 430), fontsize=17)
    ax = Axis(fig[1, 1];
              title="Cells carried vs. an equal-resolution uniform grid",
              subtitle=@sprintf("uniform grid at the finest spacing: %d × %d cells",
                                NBASE * 2^MAXLEV, NBASE * 2^MAXLEV),
              ylabel="% of uniform-grid cells", xticks=(1:length(snaps), labels))
    ylims!(ax, 0, 100)
    hidespines!(ax, :t, :r)
    barplot!(ax, 1:length(snaps), pct; color=LEVELCOLORS[1], width=0.6)
    text!(ax, 1:length(snaps), pct; text=[@sprintf("%.0f%%\n(%.1f× fewer)", p, 100 / p) for p in pct],
          align=(:center, :bottom), offset=(0, 6), fontsize=15)
    save(path, fig)
    return (path=path, pct=pct)
end

#--------------------------------------------------------------------------------

const SNAP_TIMES = [40.0, 72.0, 110.0, 210.0]
const CAPTIONS   = ["planar S1 wave", "S2 wave break", "wavefront curls", "reentrant spiral"]

if get(ENV, "EP_GIF", "false") == "true"
    println("recording animation")
    fig = Figure(size=(620, 640), fontsize=17)
    ax = Axis(fig[1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y")
    Colorbar(fig[1, 2], colorrange=(0, 1), colormap=:viridis, label="V")
    fig[2, 1:2] = level_legend(fig, MAXLEV)
    # mp4, not gif: at this frame count and figure size a gif lands around 60 MB,
    # which no slide deck wants. Derive a gif from it with ffmpeg if one is needed.
    vidpath = joinpath(@__DIR__, "monodomain_amr.mp4")
    record(fig, vidpath; framerate=20) do io
        simulate(; frame_every=120, on_frame=(V, bf, t) -> begin
                     empty!(ax)
                     draw_state!(ax, snapshot(V, bf, t).leaves; scale=0.8)
                     limits!(ax, 0, L, 0, L)
                     ax.title = @sprintf("adaptive monodomain — t = %.0f, %d leaves",
                                         t, MFO.nleaves(bf))
                     recordframe!(io)
                 end)
    end
    @printf "animation: %s\n" vidpath
else
    println("simulating monodomain S1-S2 on an adaptive forest")
    snaps = simulate(; snap_times=SNAP_TIMES)

    p1 = panel_figure(snaps, CAPTIONS, joinpath(@__DIR__, "monodomain_amr.png"))
    @printf "figure: %s\n" p1
    p2 = savings_figure(snaps, CAPTIONS, joinpath(@__DIR__, "monodomain_amr_savings.png"))
    @printf "figure: %s\n" p2.path
end
