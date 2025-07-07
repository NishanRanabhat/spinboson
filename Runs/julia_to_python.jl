using LinearAlgebra
using PyCall
using Printf
using JLD2
using TensorOperations


# Import NumPy via PyCall
np = pyimport("numpy")

folderName = "DMRG_data"

filename1 = "boson_number_N=17_chi_max=256_a=1.40_h=-0.10_w=1.00.jld2"
filepath1 = joinpath(folderName,filename1)
boson_num = load(filepath1)["data"]

filename2 = "mag_N=17_chi_max=256_a=1.40_h=-0.10_w=1.00.jld2"
filepath2 = joinpath(folderName,filename2)
mag = load(filepath2)["data"]

np.save("boson_number_N=17_chi_max=256_a=1.40_h=-0.10_w=1.00.npy",np.array(boson_num))
np.save("mag_N=17_chi_max=256_a=1.40_h=-0.10_w=1.00.npy",np.array(mag))