using LinearAlgebra
using Kronecker
using BlockDiagonals
using TensorOperations
using Random

num_states = 1
dx = 2
path = num_states+1:num_states+dx
println(path[1],",",1)
for p in 2:length(path)
    println(path[p],",",path[p-1])
end
println(4,",",path[end])