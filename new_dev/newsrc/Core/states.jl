# Core/states.jl
using LinearAlgebra
using TensorOperations

export MPSState

# ============= State Types =============
"""
    MPSState{T}

Holds the complete state for MPS-based algorithms (DMRG, TDVP, etc.)
Contains the MPS, MPO, environment tensors, and orthogonality center information.
"""
mutable struct MPSState{T}
    mps::MPS{T}
    mpo::MPO{T}
    environment::Env{T}
    center::Int              # Current orthogonality center
    direction::Symbol        # Sweep direction (:left or :right)
end

# ============= Unified Constructors =============
function MPSState(mps::MPS{T}, mpo::MPO{T}; center=1, direction=:right) where T
    mps_copy = deepcopy(mps)
    canonicalize(mps_copy, center, direction)
    env = build_environment(mps_copy, mpo, center)
    return MPSState(mps_copy, mpo, env, center, direction)
end


