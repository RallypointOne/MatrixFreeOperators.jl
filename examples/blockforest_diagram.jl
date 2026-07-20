# Concept diagram: what a BlockForest is — the spatial tiling beside the tree it
# encodes. Vector redraw of the box-drawing figure in docs/pages/amr.qmd, using
# the same refinement-level colors as the data figures so a level means the same
# thing on the concept slide and the results slide.
#
# Run with: julia --project=examples examples/blockforest_diagram.jl

using Pkg; Pkg.activate(@__DIR__)
using CairoMakie, Printf

const LEVELCOLORS = Makie.wong_colors()
const CELLS = 4          # cells per block edge, as drawn
const INK   = :gray20

#--------------------------------------------------------------------------------
# Spatial view

"A leaf block: its outline in the level color, with its CELLS×CELLS cells faint inside."
function block!(ax, x0, y0, w; level, label=nothing)
    c = LEVELCOLORS[level + 1]
    poly!(ax, Rect2f(x0, y0, w, w); color=(c, 0.10), strokecolor=c, strokewidth=2.6)
    h = w / CELLS
    for i in 1:(CELLS - 1)
        lines!(ax, [x0 + i * h, x0 + i * h], [y0, y0 + w]; color=(c, 0.45), linewidth=0.7)
        lines!(ax, [x0, x0 + w], [y0 + i * h, y0 + i * h]; color=(c, 0.45), linewidth=0.7)
    end
    label !== nothing && text!(ax, x0 + w / 2, y0 + w / 2; text=label, align=(:center, :center),
                               fontsize=19, font=:bold, color=INK)
    return nothing
end

function spatial_panel!(ax)
    # r0 (top-left) has been refined into four half-size children; r1..r3 are still leaves.
    block!(ax, 1, 1, 1; level=0, label="r₁")
    block!(ax, 0, 0, 1; level=0, label="r₂")
    block!(ax, 1, 0, 1; level=0, label="r₃")
    # r₀ occupies the top-left quadrant [0,1]×[1,2]; its children subdivide exactly that.
    for (dx, dy, lab) in ((0.0, 1.0, "c₀"), (0.5, 1.0, "c₁"), (0.0, 1.5, "c₂"), (0.5, 1.5, "c₃"))
        block!(ax, dx, dy, 0.5; level=1, label=lab)
    end
    text!(ax, 0.5, 2.16; text="r₀ refined → 4 children", align=(:center, :bottom),
          fontsize=16, color=LEVELCOLORS[2])
    lines!(ax, [0.5, 0.5], [2.14, 2.03]; color=LEVELCOLORS[2], linewidth=1.5)
    limits!(ax, -0.32, 2.32, -0.06, 2.46)
end

#--------------------------------------------------------------------------------
# Tree view

function node!(ax, x, y; level, label, leaf)
    c = LEVELCOLORS[level + 1]
    w, h = 0.46, 0.30
    poly!(ax, Rect2f(x - w / 2, y - h / 2, w, h);
          color=leaf ? (c, 0.18) : (:white, 0.0), strokecolor=c,
          strokewidth=2.6, linestyle=leaf ? :solid : :dash)
    text!(ax, x, y; text=label, align=(:center, :center), fontsize=17,
          font=:bold, color=INK)
    return nothing
end

function tree_panel!(ax)
    roots = [(0.75, "r₀", false), (1.85, "r₁", true), (2.75, "r₂", true), (3.65, "r₃", true)]
    for (x, lab, leaf) in roots
        node!(ax, x, 1.6; level=0, label=lab, leaf=leaf)
    end
    kids = [(0.0, "c₀"), (0.5, "c₁"), (1.0, "c₂"), (1.5, "c₃")]
    for (x, lab) in kids
        lines!(ax, [0.75, x], [1.45, 0.75]; color=LEVELCOLORS[2], linewidth=1.6)
        node!(ax, x, 0.6; level=1, label=lab, leaf=true)
    end
    text!(ax, -0.34, 1.6; text="level 0", align=(:right, :center), fontsize=15, color=INK)
    text!(ax, -0.34, 0.6; text="level 1", align=(:right, :center), fontsize=15, color=INK)
    limits!(ax, -1.45, 4.0, 0.25, 2.1)
end

#--------------------------------------------------------------------------------

fig = Figure(size=(1180, 560), fontsize=18)

Label(fig[1, 1:2], "A BlockForest: fixed-size blocks, refined as a tree";
      fontsize=25, font=:bold, padding=(0, 0, 4, 0))

axl = Axis(fig[2, 1]; aspect=DataAspect(), title="Spatial view — the leaves tile the domain")
axr = Axis(fig[2, 2]; title="Tree view — the forest that encodes it")
for ax in (axl, axr)
    hidedecorations!(ax); hidespines!(ax)
end
spatial_panel!(axl)
tree_panel!(axr)

Label(fig[3, 1:2],
      "Every block holds the same $(CELLS)×$(CELLS) cells — only the spacing halves with depth, " *
      "so each leaf stays a plain dense array and every operator runs on it unchanged.";
      fontsize=16, color=INK, padding=(0, 0, 6, 0))

elems = [PolyElement(color=(LEVELCOLORS[1], 0.18), strokecolor=LEVELCOLORS[1], strokewidth=2.6),
         PolyElement(color=(LEVELCOLORS[2], 0.18), strokecolor=LEVELCOLORS[2], strokewidth=2.6),
         LineElement(color=LEVELCOLORS[1], linestyle=:dash, linewidth=2.6)]
fig[4, 1:2] = Legend(fig, elems, ["level-0 leaf", "level-1 leaf", "refined (not a leaf)"];
                     orientation=:horizontal, framevisible=false)

rowsize!(fig.layout, 2, Relative(0.74))
for p in (joinpath(@__DIR__, "blockforest_diagram.svg"), joinpath(@__DIR__, "blockforest_diagram.png"))
    save(p, fig)
    @printf "figure: %s\n" p
end
