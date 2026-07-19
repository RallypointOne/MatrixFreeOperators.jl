#--------------------------------------------------------------------------------# BlockForest grid

"""
    BlockForest(base::CartesianGrid; blocksize, maxlevel) -> BlockForest

Block-structured adaptive grid: a [`Forest`](@ref) of fixed-size leaf-blocks laid
over the physical domain of `base`. Each leaf is an ordinary [`CartesianGrid`](@ref)
of `blocksize` cells with the same halo as `base`; refining a block replaces it
with `2ᴺ` children at half the spacing. Leaf grids carry the
[`Interface`](@ref) boundary on every face; inter-block ghosts are filled by
[`halo_update!`](@ref) and physical domain faces by the forest-level
`apply_bc!` face pass from `base`'s boundary conditions (kept on the forest).

`base` ncells must be divisible by `blocksize` (the quotient is the root tiling).
Operators built on a `BlockForest` run the existing per-`CartesianGrid` stencil
code unchanged on every leaf.

# Keyword Arguments
- `blocksize::NTuple{N,Int}`: cells per block per dimension
- `maxlevel::Int`: maximum refinement level (root blocks are level 0)

### Examples

```julia
base   = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (16, 16);
                       bc=((Dirichlet(), Dirichlet()), (Dirichlet(), Dirichlet())))
forest = BlockForest(base; blocksize=(8, 8), maxlevel=4)
refine!(forest, x -> sum(abs2, x .- 0.5) < 0.05)   # refine a disc near the center
```

See also: [`refine!`](@ref), [`coarsen!`](@ref), [`leaves`](@ref).
"""
struct BlockForest{N,T,BC<:Tuple,Dev} <: AbstractGrid{N}
    forest::Forest{N}
    extent::NTuple{N,Tuple{T,T}}
    spacing0::NTuple{N,T}          # root-level (coarsest) spacing
    blocksize::NTuple{N,Int}
    halo::NTuple{N,Int}
    bc::BC                         # physical domain boundary conditions
    device::Dev
    schedule::Base.RefValue{ExchangeSchedule{N,T}}   # per-generation halo-exchange cache
end

function BlockForest(
    base::CartesianGrid{N,T}; blocksize::NTuple{N,Int}, maxlevel::Int
) where {N,T}
    for d in 1:N
        blocksize[d] >= 1 || throw(ArgumentError("blocksize must be ≥ 1, got $blocksize"))
        base.size[d] % blocksize[d] == 0 || throw(
            ArgumentError(
                "base ncells $(base.size) must be divisible by blocksize $blocksize " *
                "(dimension $d)",
            ),
        )
        if !(base.bc[d][1] isa Periodic)
            _require_face_bc(base.bc[d][1])
            _require_face_bc(base.bc[d][2])
        end
    end
    maxlevel >= 0 || throw(ArgumentError("maxlevel must be ≥ 0, got $maxlevel"))
    nroot = ntuple(d -> base.size[d] ÷ blocksize[d], Val(N))
    periodic = ntuple(d -> base.bc[d][1] isa Periodic, Val(N))
    forest = Forest(nroot, periodic, maxlevel)
    return BlockForest{N,T,typeof(base.bc),typeof(base.device)}(
        forest, base.extent, base.spacing, blocksize, base.halo, base.bc, base.device,
        Ref(_empty_schedule(Val(N), T)),
    )
end

KernelAbstractions.get_backend(bf::BlockForest) = bf.device
nleaves(bf::BlockForest) = nleaves(bf.forest)

function Base.show(io::IO, bf::BlockForest{N}) where {N}
    print(io, "BlockForest{$N}(blocksize=$(bf.blocksize), ", bf.forest, ")")
end

#--------------------------------------------------------------------------------# Per-leaf geometry (recomputed from the key)

# Leaf spacing at level ℓ is the root spacing halved ℓ times.
_leaf_spacing(bf::BlockForest{N,T}, ℓ::Int) where {N,T} =
    ntuple(d -> bf.spacing0[d] / (1 << ℓ), Val(N))

# Physical extent of one block: its width is blocksize·spacing; it starts at the
# domain minimum offset by coords whole blocks.
function _leaf_extent(bf::BlockForest{N,T}, key::LeafKey{N}) where {N,T}
    sp = _leaf_spacing(bf, key.level)
    return ntuple(Val(N)) do d
        w = bf.blocksize[d] * sp[d]
        lo = bf.extent[d][1] + key.coords[d] * w
        (lo, lo + w)
    end
end

"""
    leaf_center(bf::BlockForest, key::LeafKey) -> SVector

Physical center of the block identified by `key`.
"""
function leaf_center(bf::BlockForest{N,T}, key::LeafKey{N}) where {N,T}
    sp = _leaf_spacing(bf, key.level)
    return SVector(
        ntuple(d -> bf.extent[d][1] + (key.coords[d] + T(0.5)) * bf.blocksize[d] * sp[d], Val(N))
    )
end

"""
    leaf_grid(bf::BlockForest, key) -> CartesianGrid
    leaf_grid(bf::BlockForest, i::Integer) -> CartesianGrid

The ordinary `CartesianGrid` for one leaf block — what per-block operators run on.
Every face is [`Interface`](@ref): inter-block ghosts are filled by `halo_update!`
and physical-boundary ghosts by the forest-level `apply_bc!` face pass, so every
leaf shares one concrete grid type — type-stable, isbits, and free to recompute
inside apply loops. Physical BCs live on `bf.bc`. Built via the unchecked inner
constructor (the forest guarantees validity).
"""
function leaf_grid(bf::BlockForest{N,T}, key::LeafKey{N}) where {N,T}
    ext = _leaf_extent(bf, key)
    sp = _leaf_spacing(bf, key.level)
    bc = ntuple(_ -> (Interface(), Interface()), Val(N))
    lr = ntuple(d -> 1:bf.blocksize[d], Val(N))
    return CartesianGrid{N,T,typeof(bc),typeof(bf.device),Nothing}(
        ext, sp, bf.blocksize, bf.halo, bc, bf.device, lr, nothing
    )
end
leaf_grid(bf::BlockForest, i::Integer) = leaf_grid(bf, bf.forest.leaves[i])

"""
    leaves(bf::BlockForest)

Iterator over `(key, leaf_grid)` pairs for every leaf, in Morton (storage) order.
Block storage in a [`BlockField`](@ref) over `bf` is indexed in the same order.
Leaf grids are all-[`Interface`](@ref); physical BCs live on `bf.bc`.
"""
leaves(bf::BlockForest) = ((key, leaf_grid(bf, key)) for key in bf.forest.leaves)

#--------------------------------------------------------------------------------# Adaptivity

"""
    refine!(bf::BlockForest, predicate) -> bf

Refine every leaf whose center satisfies `predicate(center::SVector)` (and is below
`maxlevel`), then re-establish 2:1 balance. Fields must be (re)allocated after a
regrid — block storage is tied to the leaf set at allocation time.
"""
function refine!(bf::BlockForest, predicate)
    refine!(bf.forest, key -> predicate(leaf_center(bf, key)))
    return bf
end

"""
    coarsen!(bf::BlockForest, predicate) -> bf

Coarsen each complete family of `2ᴺ` sibling leaves whose centers all satisfy
`predicate(center::SVector)`, then re-establish 2:1 balance.
"""
function coarsen!(bf::BlockForest, predicate)
    coarsen!(bf.forest, key -> predicate(leaf_center(bf, key)))
    return bf
end

"""
    balance!(bf::BlockForest) -> bf

Re-establish the 2:1 balance invariant across leaf faces.
"""
balance!(bf::BlockForest) = (balance!(bf.forest); bf)

function Adapt.adapt_structure(to, bf::BlockForest{N,T}) where {N,T}
    device = KernelAbstractions.get_backend(Adapt.adapt(to, similar(Vector{Bool}, 0)))
    # The schedule Ref is shared deliberately, like `forest`: descriptors are
    # device-independent index data, and sharing keeps the cache warm across
    # adaptation.
    return BlockForest{N,T,typeof(bf.bc),typeof(device)}(
        bf.forest, bf.extent, bf.spacing0, bf.blocksize, bf.halo, bf.bc, device,
        bf.schedule,
    )
end
