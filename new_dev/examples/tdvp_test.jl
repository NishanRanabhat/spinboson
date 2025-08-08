# Add src to LOAD_PATH and load the module
using Revise
using Test

push!(LOAD_PATH, joinpath(@__DIR__, "..", "newsrc"))

# This both includes MyDMRG.jl and tells Revise to watch every file it pulls in
includet(joinpath(@__DIR__, "..", "newsrc", "TNCodebase.jl"))

using .TNCodebase

"""
Build the system a site at a time
"""

nmax = 6
spin_site = SpinSite(1/2,T=ComplexF64) #define a spin 1/2 particle 
boson_site = BosonSite(nmax,T=ComplexF64) #define a Boson awith nmax=4 

#build the system by putting the boson at site 0 and spins at rest of the sites
Ns = 10 #total spins
spinboson = vcat(boson_site,fill(spin_site,Ns))

"""
define a state
"""

labels = vcat(2,fill((:X,1),Ns))
psi = product_state(spinboson,labels)


"""
define a Hamiltonian, in this case we will illustrate LR Ising Dickie
"""

#Ising part
J = -1.0; #coupling strength
alpha = 1.5; #range of interaction
n = 4; #number of exponential to approximate the 1/r^a interaction
h = 2.0;

#define Spin channels
spinchannel1 = [PowerLawCoupling(:X,:X,J,alpha,n,Ns),Field(:Z,h)]
spinchannel2 = [Field(:X,1.0)]

#Boson part
g = 0.2; #spin boson interaction strength
w = 1.0 #boson energy

#spin boson channel
channel = [SpinBosonInteraction(spinchannel1,:Ib,1.0),SpinBosonInteraction(spinchannel2,:a,g),SpinBosonInteraction(spinchannel2,:adag,g),BosonOnly(:Bn,w)]

#build Finite State Machine based on the channels
fsm = build_FSM(channel)

#build mpo from the Finite State Machine
Ham = build_mpo(fsm,N=Ns+1,d=2,nmax=nmax).tensors

#initialize the environments
M,new_psi,Env = Initialize(Ns+1,psi,Ham,"MPS")
