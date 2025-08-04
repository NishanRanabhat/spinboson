# Add src to LOAD_PATH and load the module

using Revise
using Test

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# This both includes MyDMRG.jl and tells Revise to watch every file it pulls in
includet(joinpath(@__DIR__, "..", "src", "TNCodebase.jl"))

using .TNCodebase

mps = MPS{Float64}([rand(2,2,2) for _ in 1:3])
ops1 = spin_ops(2)
ops2 = boson_ops(3)

println(AxisSpinSite(:Z,2).up) 
println(AxisSpinSite(:Z,2).down) 

println(AxisSpinSite(:X,2).up) 
println(AxisSpinSite(:X,2).down) 
println(BosonSite(3,1))

