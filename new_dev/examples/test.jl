# Add src to LOAD_PATH and load the module
using Revise
using Test

push!(LOAD_PATH, joinpath(@__DIR__, "..", "testsrc"))

# This both includes MyDMRG.jl and tells Revise to watch every file it pulls in
includet(joinpath(@__DIR__, "..", "testsrc", "TNCodebase.jl"))

using .TNCodebase

site1 = SpinSite(1/2,T=ComplexF64)
site0 = BosonSite(4,T=ComplexF64)

spin = fill(site1, 10)
spinboson = vcat(site0,spin)
labels = vcat(2,fill((:Z,1),10))

mps1 = product_state(spinboson,labels)
mps2 = random_state(spinboson,5)

for i in 1:11
    println(size(mps2.tensors[i]))
end
#bt = state_tensor(boson1,0)

#println(size(bt))

#st = state_tensor(spin1,(:X,1))
#println(st[1,:,1])

#sites = [BosonSite(4,2),AxisSpinSite(:Z,2),AxisSpinSite(:Z,2),AxisSpinSite(:Z,2)]
#label = [:up,:up,:up]

#mps = product_state(sites,label).tensors

#println(size(mps[1]))

#println(mps[1])

