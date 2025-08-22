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
    environment::Environment{T}
    center::Int              # Current orthogonality center
    #direction::Symbol        # Sweep direction (:left or :right), i don't think direction belongs here since same MPSState will be updated during left and right sweep
end

# ============= Unified Constructors =============
#function MPSState(mps::MPS, mpo::MPO; center=1)
#    mps_copy = deepcopy(mps)
#    canonicalize(mps_copy, center)
#    env = build_environment(mps_copy, mpo, center)
#    return MPSState(mps_copy, mpo, env, center)
#end

# Add these convert methods for MPS and MPO
Base.convert(::Type{MPS{T}}, m::MPS) where T = 
    MPS{T}([convert(Array{T,3}, tensor) for tensor in m.tensors])

Base.convert(::Type{MPO{T}}, m::MPO) where T = 
    MPO{T}([convert(Array{T,4}, tensor) for tensor in m.tensors])

Base.convert(::Type{Environment{T}}, e::Environment) where T = 
    Environment{T}([isnothing(tensor) ? nothing : convert(Array{T,3}, tensor) 
                    for tensor in e.tensors])

# Also need convert for same type (no-op)
Base.convert(::Type{MPS{T}}, m::MPS{T}) where T = m
Base.convert(::Type{MPO{T}}, m::MPO{T}) where T = m
Base.convert(::Type{Environment{T}}, e::Environment{T}) where T = e

function MPSState(mps::MPS{T1}, mpo::MPO{T2}; center=1) where {T1,T2}
    T = promote_type(T1, T2)
    
    mps_promoted = convert(MPS{T}, mps)
    mpo_promoted = convert(MPO{T}, mpo)
    mps_copy = deepcopy(mps_promoted)
    canonicalize(mps_copy, center)
    env = build_environment(mps_copy, mpo_promoted, center)
    
    return MPSState{T}(mps_copy, mpo_promoted, env, center)
end