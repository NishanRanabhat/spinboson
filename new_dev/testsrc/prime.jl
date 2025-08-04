module Prime

include(joinpath(@__DIR__, "dir1", "file1.jl"))
using .Sub1: MPS

include(joinpath(@__DIR__, "dir2", "file2.jl"))
using .Sub2: make_chain
export MPS, make_chain

end
