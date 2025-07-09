# Add src to LOAD_PATH and load the module

using Revise
# This both includes MyDMRG.jl and tells Revise to watch every file it pulls in
includet(joinpath(@__DIR__, "..", "src", "MyDMRG.jl"))

using Test
using .MyDMRG.Types
using .MyDMRG.States

mps = MPS{Float64}([rand(2,2,2) for _ in 1:3])

println(boson_excitation(4,3,Float64).tensors[1])
