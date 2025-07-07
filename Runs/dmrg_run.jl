using Revise
using MKL
using LinearAlgebra
using PyCall
using Printf
using JLD2
using BenchmarkTools
using TensorOperations

# Import NumPy via PyCall
np = pyimport("numpy")

"""
This is a file that tests our codes. We will show how to run a TDVP code with pure states for transverse field Ising Hamiltonian.

Setup: In this simplest example we will initialize our state as an fully polarized (in longitudinal direction) MPS and evolve it with
       two site TDVP algorithm. 
"""
#include the path to the module
includet("/home/nishan/PD_UMBC/Research/spin_boson/Module/TenMB.jl")
#CALL module
using .TenMB


function run_dmrg_sweeps(alpha,h,w,g,M,psi,Env,Ham,N,num_sweep_DMRG,krydim_DMRG,maxit,d,chi_max,ctf_val)

       a = TenMB.boson_annihilator(32)
       sX,sY,sZ,sI = TenMB.spin_ops(d)

       folderName = "DMRG_data"
       mkpath(folderName)
       
       for i in 1:num_sweep_DMRG
   
           state = Array{Any,1}(undef,N)
   
           M,psi,Env = TenMB.right_sweep_DMRG_two_site(M,psi,Env,Ham,N,krydim_DMRG,maxit,d,chi_max,ctf_val);
           M,psi,Env = TenMB.left_sweep_DMRG_two_site(M,psi,Env,Ham,N,krydim_DMRG,maxit,d,chi_max,ctf_val)
   
           state[1] = M
   
           for j in 2:N
               state[j] = psi[j]
           end

           if i == num_sweep_DMRG
              filepath = @sprintf("%s/DMRG_MPS_N=%d_chi_max=%d_a=%1.2f_h=%1.2f_w=%1.2f_g=%1.2f.jld2",folderName,N,chi_max,alpha,h,w,g)
              save(filepath, "data",state)
              count = 0
           end

           println(i)
           println("boson_number=",TenMB.expect_single_site_updated(1,state,a'*a,"MPS"))
           println("central_magnetization=",TenMB.expect_single_site_updated(9,state,sZ,"MPS"))
   
           state = nothing
       end
   end

"""
Build Hamiltonian as MPO
"""

function build_sb_ham(Ns,nmax,d,J,alpha,h,w,g,spins,longdir,kac,n)

       HLRIsing_bulk = TenMB.hamiltonian_lr_longitudinal_ising(alpha,J,h,Ns,d,n,kac,longdir)
       HLRIsing_0 = zeros(1,n+2,nmax+1,nmax+1)
       HLRIsing_0[1,n+2,:,:] =  TenMB.boson_identity(nmax)

       HDickie_0,HDickie_bulk = TenMB.hamiltonian_dickie(Ns,nmax,d,w,g,longdir)

       vR_LRI = reshape([1.0; zeros(n+1)],(n+2,1))
       vR_I = transpose([1.0 0.0 0.0])
       vR_D = transpose([1.0 0.0])
       
       return TenMB.cavity_mpo(Ns,HDickie_0,HDickie_bulk,HLRIsing_0,HLRIsing_bulk,vR_D,vR_LRI)
end

Ns = 16
nmax = 32
d = 2
J = -1.0
alpha = 1.4
h = -0.1
w = 1.0
g_list = [1.6,1.7,1.8,1.9,2.0]
spins = "Pauli"
longdir = "Z"
transdir = "X"

kac = "false"
n = min(Int(floor(Ns/2)-1.0),14)

#DMRG sweeps
krydim_DMRG = 4; #krylov dimension for DMRG sweeps
maxit = 14; #iteration for eigensolver 
num_sweep_DMRG = 50; #number of DMRG sweeps
chi_max = 256; #upper limit in auxilliary bond dimension
ctf_val = 0.00000001 #cutoff for SVD truncation

"""
Initialize a random state with  given bond dimension
"""
chi = 5;
psi = Array{Any,1}(undef,Ns+1)
psi[1] = rand(1,nmax+1,chi)
psi[Ns+1] = rand(chi,d,1)

@inbounds for i in 2:Ns
    psi[i] = rand(chi,d,chi)
end

g = 2.4
Ham = build_sb_ham(Ns,nmax,d,J,alpha,h,w,g,spins,longdir,kac,n)
M,psi_new,Env = TenMB.Initialize(Ns+1,psi,Ham,"MPS")
run_dmrg_sweeps(alpha,h,w,g,M,psi_new,Env,Ham,Ns+1,num_sweep_DMRG,krydim_DMRG, maxit,d,chi_max,ctf_val)

#function all_g(Ns,nmax,d,J,alpha,h,w,g_list,spins,longdir,kac,n,psi,num_sweep_DMRG,krydim_DMRG,maxit,chi_max,ctf_val)

#       for g in g_list
#              Ham = build_sb_ham(Ns,nmax,d,J,alpha,h,w,g,spins,longdir,kac,n)

              #initialize before DMRG sweeps
#              M,psi_new,Env = TenMB.Initialize(Ns+1,psi,Ham,"MPS");

#              run_dmrg_sweeps(alpha,h,w,g,M,psi_new,Env,Ham,Ns+1,num_sweep_DMRG,krydim_DMRG, maxit,d,chi_max,ctf_val)
#       end 
#end

#all_g(Ns,nmax,d,J,alpha,h,w,g_list,spins,longdir,kac,n,psi,num_sweep_DMRG,krydim_DMRG,maxit,chi_max,ctf_val)