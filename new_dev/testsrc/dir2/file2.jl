module Sub2

# 1) Include file1.jl so that Sub1 gets defined under this module’s parent
include(joinpath(@__DIR__,"..","dir1","file1.jl"))

# 2) Bring MPS into File2’s namespace
using .Sub1: MPS

# now you can write code that refers to MPS directly:
function make_chain(::Type{T}, N::Int) where T
    # e.g. build a length-N MPS of zeros
    MPS{T}( [ zeros(T,2,2,2) for i in 1:N ] )
end

export make_chain

end
