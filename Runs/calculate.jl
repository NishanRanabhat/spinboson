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

function read_and_compute(folderName,N,d,chi_max,alpha,h,w,g_list)

    a = TenMB.boson_annihilator(32)
    sX,sY,sZ,sI = TenMB.spin_ops(d)

    b_num = Array{Any,1}(undef,length(g_list))
    mag = Array{Any,1}(undef,length(g_list))

    count = 0

    for g in g_list
        count = count + 1
        filepath = joinpath(folderName,@sprintf("DMRG_MPS_N=%d_chi_max=%d_a=%1.2f_h=%1.2f_w=%1.2f_g=%1.2f.jld2",N,chi_max,alpha,h,w,g))
        state = load(filepath)["data"]

        println("boson_number=",TenMB.expect_single_site_updated(1,state,a'*a,"MPS"))
        println("central_magnetization=",TenMB.expect_single_site_updated(9,state,sZ,"MPS"))

        b_num[count] = TenMB.expect_single_site_updated(1,state,a'*a,"MPS")
        mag[count] = TenMB.expect_single_site_updated(9,state,sZ,"MPS")
    end 

    filepath1 = @sprintf("%s/boson_number_N=%d_chi_max=%d_a=%1.2f_h=%1.2f_w=%1.2f.jld2",folderName,N,chi_max,alpha,h,w)
    save(filepath1, "data",b_num)

    filepath2 = @sprintf("%s/mag_N=%d_chi_max=%d_a=%1.2f_h=%1.2f_w=%1.2f.jld2",folderName,N,chi_max,alpha,h,w)
    save(filepath2, "data",mag)

end

folderName = "DMRG_data"
Ns = 16;
d = 2;
chi_max = 256
alpha = 1.40
h = -0.10
w = 1.00
g_list = np.arange(0.0,2.5,0.1)

read_and_compute(folderName,Ns+1,d,chi_max,alpha,h,w,g_list)