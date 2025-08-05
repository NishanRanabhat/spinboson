export MPS, MPO, DMRGEnv, DMRGOptions

"""
Abstract supertype for tensor networks (MPS, MPO, MPDO, etc.)
"""
abstract type TensorNetwork{T} end

# A "pure virtual" type: it cannot be instantiated itself,
# but MPS, MPO, and MPDO are subtypes.
# Functions can be written to accept any TensorNetwork,
# so you can dispatch on e.g. f(x::TensorNetwork) for generic behavior.

"""
An MPS is a sequence of rank-3 tensors of element-type `T`.
"""
struct MPS{T} <: TensorNetwork{T}
  tensors::Vector{Array{T,3}}
end

"""
An MPO is a sequence of rank-4 tensors of element-type `T` representing operators.
"""
struct MPO{T} <: TensorNetwork{T}
  tensors::Vector{Array{T,4}}
end

"""
Environment holds left and right boundary tensors for DMRG sweeps.
"""
struct DMRGEnv{T}
  left::Vector{Array{T,3}}        # length N+1
  right::Vector{Array{T,3}}
end

"""
Options for DMRG sweeps (e.g. krylov dim, ctf, chi_max).
"""
struct DMRGOptions
  krylov_dim::Int
  max_iter::Int
  chi_max::Int
  ctf::Float64
  local_dim::Int
end